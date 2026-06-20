"""交易所官方仓单数据 fetcher（核心层 public 模块）。

架构位置：core/data/receipt_fetcher.py

三层架构：
  - 接口层：core/data/_receipt_adapters.py （4 个交易所适配器）
  - 缓存层：core/data/_receipt_cache.py   （parquet + meta + TTL）
  - 逻辑层：本文件                            （编排 + 信号计算）

职责（本文件）：
  1. ReceiptFetcher 编排器：组合接口层 + 缓存层
  2. get_receipt_change_signal：从仓单序列计算变化率信号 [-1, 1]
  3. SYMBOL_TO_AK_NAME：品种代码 → 交易所中文名映射
  4. load_receipt_cache：兼容旧 API（委托给缓存层）

配置（config.yaml receipt 段）：
  receipt:
    cache_dir: "data/receipt_cache"
    enable_online: false           # 沙盒默认 false，生产环境 true
    request_interval_min: 1.0
    request_interval_max: 4.0
    retry_times: 3
    cache_ttl_days: 7

设计要点（规则 17/21）：
  - 反爬虫机制委托给 _receipt_adapters（不重复造轮子）
  - 缓存管理委托给 _receipt_cache（不重复造轮子）
  - 第三方依赖（pandas / requests）顶部 import（规则 21.2）
  - enable_online=False 时不发起任何外部请求（验收标准）
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.data._receipt_adapters import (
    ADAPTER_FN,
    build_session,
    polite_sleep,
)
from core.data._receipt_cache import ReceiptCache, load_receipt_cache

logger = logging.getLogger(__name__)

# ── 品种代码 → 交易所中文名 映射 ─────────────────────────────
SYMBOL_TO_AK_NAME: Dict[str, str] = {
    "SHFE.RB": "螺纹钢",
    "SHFE.HC": "热轧卷板",
    "SHFE.CU": "铜",
    "SHFE.AL": "铝",
    "SHFE.RU": "天然橡胶",
    "SHFE.AU": "黄金",
    "SHFE.AG": "白银",
    "SHFE.ZN": "锌",
    "SHFE.NI": "镍",
    "DCE.M": "豆粕",
    "DCE.I": "铁矿石",
    "DCE.J": "焦炭",
    "DCE.JM": "焦煤",
    "DCE.PP": "聚丙烯",
    "DCE.L": "聚乙烯",
    "DCE.V": "聚氯乙烯",
    "DCE.Y": "豆油",
    "DCE.P": "棕榈油",
    "DCE.C": "玉米",
    "CZCE.CF": "棉花",
    "CZCE.FG": "玻璃",
    "CZCE.SR": "白糖",
    "CZCE.OI": "菜籽油",
    "CZCE.MA": "甲醇",
    "CZCE.TA": "PTA",
    "CZCE.RM": "菜粕",
    "CZCE.SA": "纯碱",
}


# ═══════════════════════════════════════════════════════════════
# 逻辑层：变化率信号计算
# ═══════════════════════════════════════════════════════════════


def get_receipt_change_signal(
    receipt_series: pd.Series,
    window: int = 20,
) -> pd.Series:
    """计算仓单变化率信号 [-1, 1]。

    公式：
      change[t] = (receipt[t] - receipt[t-1]) / max(|receipt[t-1]|, 1.0)
      std_change = rolling_std(change, window)
      raw = change / std_change
      signal = -clip(raw, -1, 1)   # 仓单↑ → 做空；仓单↓ → 做多

    Args:
        receipt_series: 日度仓单序列
        window: 滚动标准差窗口（默认 20）

    Returns:
        signal_series（与 receipt_series 同长度，前 warmup 段为 0）
    """
    if receipt_series is None or receipt_series.empty:
        return pd.Series(dtype=float)
    s = receipt_series.astype(float).copy()

    # 日变化率（分母用绝对值兜底，避免零值附近放大失真）
    prev = s.shift(1)
    denom = prev.abs().where(prev.abs() > 1.0, 1.0)
    change = (s - prev) / denom
    change = change.fillna(0.0)

    # 异常值截断：变化率超过 10 倍中位数时 clip
    median = float(change.median()) if change.notna().any() else 0.0
    if abs(median) > 1e-6:
        upper = median * 10.0
        lower = median * 0.1
        change = change.clip(lower, upper)

    # 滚动 std（min_periods=window 保证稳定）
    std_change = change.rolling(window=window, min_periods=window).std()
    median_std = float(std_change.median()) if std_change.notna().any() else 1e-3
    if not np.isfinite(median_std) or median_std <= 0:
        median_std = 1e-3
    std_safe = std_change.replace(0, median_std).fillna(median_std)

    # 归一化 + 取负（仓单↑ → 做空）
    raw = change / std_safe
    signal = raw.clip(-1.0, 1.0) * -1.0

    # warmup 段置零
    signal.iloc[:window] = 0.0
    return signal.fillna(0.0)


# ═══════════════════════════════════════════════════════════════
# 编排器：ReceiptFetcher
# ═══════════════════════════════════════════════════════════════


class ReceiptFetcher:
    """仓单数据 fetcher，支持缓存 + 在线拉取 + 反爬虫。

    Usage::

        fetcher = ReceiptFetcher(config={
            "cache_dir": "data/receipt_cache",
            "enable_online": False,
            "cache_ttl_days": 7,
        })
        df = fetcher.fetch_range(
            symbols=["SHFE.RB", "DCE.M"],
            start_date="2024-01-01",
            end_date="2024-12-31",
        )
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        """初始化 fetcher。

        Args:
            config: 配置字典，键值见模块顶部
        """
        cfg = config or {}
        self.cache = ReceiptCache(
            cache_dir=cfg.get("cache_dir", "data/receipt_cache"),
            ttl_days=int(cfg.get("cache_ttl_days", 7)),
        )
        self.enable_online: bool = bool(cfg.get("enable_online", False))
        self.interval_min: float = float(cfg.get("request_interval_min", 1.0))
        self.interval_max: float = float(cfg.get("request_interval_max", 4.0))
        self.retry_times: int = int(cfg.get("retry_times", 3))

        # Session 懒加载（仅在 enable_online=True 时构造）
        self._session = None

    def _get_session(self) -> Optional["requests.Session"]:
        """懒加载 requests Session（仅在线模式）。

        Returns:
            配置好的 Session；若 enable_online=False 则返回 None（不构造）
        """
        if not self.enable_online:
            return None
        if self._session is None:
            self._session = build_session(max_retries=self.retry_times)
        return self._session

    # ── 主入口 ────────────────────────────────────────────────

    def fetch_range(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """拉取指定品种在区间内的仓单数据，优先读缓存。

        Args:
            symbols: 品种代码列表（如 ["SHFE.RB", "DCE.M"]）
            start_date: 起始日期 YYYY-MM-DD
            end_date: 结束日期 YYYY-MM-DD
            force_refresh: 强制重新拉取（忽略缓存）

        Returns:
            DataFrame(columns=['date', 'symbol', 'receipt'])，按 date 升序。
        """
        # 1. 缓存优先
        if not force_refresh:
            cached = self.cache.load(start_date, end_date)
            if cached is not None:
                # 过滤请求的品种
                return self._filter_symbols(cached, symbols)

        # 2. 在线拉取（仅 enable_online=True）
        if not self.enable_online:
            logger.info(
                "ReceiptFetcher: enable_online=False，仅使用缓存。"
                " 区间 %s~%s, 品种 %s",
                start_date, end_date, symbols,
            )
            return self._empty_result()

        # 3. 逐品种逐日拉取
        result = self._fetch_online(symbols, start_date, end_date)

        # 4. 写缓存（2026-06-19 优化：在线拉取完成一律记 success，
        #    即便 result 为空也写空 parquet，标记"已确认无仓单数据"，
        #    避免下一周期反复重拉）
        self.cache.save(result, start_date, end_date, status="success")
        return result

    # ── 在线拉取（反爬虫集成） ────────────────────────────────

    def _fetch_online(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """逐品种调用对应交易所适配器。"""
        session = self._get_session()
        if session is None:
            # 防御：enable_online=False 时根本不应调用此方法
            return self._empty_result()
        # 2026-06-19 优化：freq='B' 仅拉取工作日，跳过周末
        date_range = pd.bdate_range(start_date, end_date)
        all_dfs: List[pd.DataFrame] = []

        for sym in symbols:
            exchange = sym.split(".")[0]
            cn_name = SYMBOL_TO_AK_NAME.get(sym)
            if cn_name is None:
                logger.debug("品种 %s 无映射，跳过", sym)
                continue
            adapter = ADAPTER_FN.get(exchange)
            if adapter is None:
                logger.debug("交易所 %s 无适配器，跳过 %s", exchange, sym)
                continue

            rows: List[pd.DataFrame] = []
            for dt in date_range:
                df_day = adapter(session, sym, cn_name, dt)
                if df_day is not None and not df_day.empty:
                    rows.append(df_day)
                # 反爬虫：每次请求后随机间隔
                polite_sleep(self.interval_min, self.interval_max)

            if rows:
                all_dfs.append(pd.concat(rows, ignore_index=True))

        if not all_dfs:
            logger.warning("ReceiptFetcher: 所有品种均无数据，区间 %s~%s", start_date, end_date)
            return self._empty_result()
        result = pd.concat(all_dfs, ignore_index=True)
        result["date"] = pd.to_datetime(result["date"])
        return result.sort_values(["symbol", "date"]).reset_index(drop=True)

    # ── 工具方法 ──────────────────────────────────────────────

    @staticmethod
    def _filter_symbols(
        df: pd.DataFrame,
        symbols: List[str],
    ) -> pd.DataFrame:
        """过滤指定品种。"""
        if df is None or df.empty or not symbols:
            return df
        return df[df["symbol"].isin(symbols)].reset_index(drop=True)

    @staticmethod
    def _empty_result() -> pd.DataFrame:
        """返回空 DataFrame（标准列）。"""
        return pd.DataFrame(columns=["date", "symbol", "receipt"])

    # ── 兼容旧 API ────────────────────────────────────────────

    @staticmethod
    def get_receipt_series(
        symbol: str,
        receipt_df: pd.DataFrame,
    ) -> pd.Series:
        """从合并的 receipt_df 中提取单品种的仓单序列。

        Args:
            symbol: 品种代码
            receipt_df: fetch_range() 返回的 DataFrame

        Returns:
            pd.Series(index=DatetimeIndex, values=receipt float)
        """
        if receipt_df is None or receipt_df.empty:
            return pd.Series(dtype=float)
        sub = receipt_df[receipt_df["symbol"] == symbol].copy()
        if sub.empty:
            return pd.Series(dtype=float)
        sub["date"] = pd.to_datetime(sub["date"])
        sub = sub.set_index("date")["receipt"].sort_index()
        return sub[~sub.index.duplicated(keep="last")]

    def has_receipt_data(self, symbol: str) -> bool:
        """检查品种是否有中文名映射（决定是否能拉取数据）。"""
        return symbol in SYMBOL_TO_AK_NAME


__all__ = [
    "ReceiptFetcher",
    "SYMBOL_TO_AK_NAME",
    "get_receipt_change_signal",
    "load_receipt_cache",
]
