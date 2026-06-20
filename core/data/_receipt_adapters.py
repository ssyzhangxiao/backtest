"""交易所官方仓单接口适配器（核心层 private 模块）。

架构位置：core/data/_receipt_adapters.py

职责：
  1. 封装 4 个交易所仓单日报接口（SHFE / DCE / CZCE / GFEX）
  2. 统一返回标准 DataFrame：列 `[date, symbol, receipt]`
  3. 集成反爬虫机制（UA 轮换 / 随机间隔 / 指数退避 / Session 复用）
  4. 单一职责：只负责"调用接口 → 标准化 DataFrame"，不写缓存

设计要点（规则 21.2）：
  - 第三方依赖（akshare / requests）顶部 import
  - 反爬虫配置走参数注入，不耦合到 fetcher 业务逻辑
  - 4 个适配器签名一致：fetch(exchange, symbol, cn_name, start_date, end_date, session) -> DataFrame | None
  - 列名映射通过 RECEIPT_COLUMN_MAP 集中维护，便于适配 AKShare 版本变更

约束：
  - 沙盒默认不调用本模块（由 fetcher 通过 enable_online 控制）
  - 适配器内部不抛异常，失败统一返回 None + logger.warning
"""

from __future__ import annotations

import logging
import random
import time
from io import BytesIO, StringIO
from typing import Callable, Dict, List, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# ── 主流 UA 池（5 个，轮换使用） ────────────────────────────
USER_AGENTS: List[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) "
    "Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
]

# ── 4 个交易所的仓单接口名（AKShare 公开 API） ──────────────
EXCHANGE_RECEIPT_FN: Dict[str, str] = {
    "SHFE": "futures_shfe_warehouse_receipt",
    "DCE": "futures_dce_warehouse_receipt",
    "CZCE": "futures_czce_warehouse_receipt",
    "GFEX": "futures_gfex_warehouse_receipt",
}

# ── 列名映射（按交易所 × 数据源） ───────────────────────────
# 不同交易所/不同 AKShare 版本可能使用不同列名，候选列表按优先级排序。
# 解析时依次尝试，找到即用，找不到返回 None。
RECEIPT_COLUMN_MAP: Dict[str, Dict[str, List[str]]] = {
    "SHFE": {
        "date": ["date", "日期", "DATE"],
        "symbol": ["VARNAME", "variety", "品种", "symbol"],
        "receipt": ["RECEIPT", "WARRANT", "receipt", "仓单量", "数量"],
    },
    "DCE": {
        "date": ["date", "日期", "DATE"],
        "symbol": ["品种", "symbol", "variety"],
        "receipt": ["仓单量", "数量", "receipt", "今日仓单"],
    },
    "CZCE": {
        "date": ["date", "日期", "DATE"],
        "symbol": ["symbol", "品种", "variety"],
        "receipt": ["仓单数量", "数量", "receipt", "仓单量"],
    },
    "GFEX": {
        "date": ["date", "日期", "gen_date", "DATE"],
        "symbol": ["品种", "symbol", "variety"],
        "receipt": ["今日仓单量", "wbillQty", "receipt"],
    },
}


# ═══════════════════════════════════════════════════════════════
# Session 管理（反爬虫基础设施）
# ═══════════════════════════════════════════════════════════════


def build_session(
    timeout: tuple = (10, 30),
    max_retries: int = 3,
) -> requests.Session:
    """构造带连接池适配器 + 指数退避的 requests Session。

    Args:
        timeout: (连接超时, 读取超时) 秒
        max_retries: 429/5xx 状态码的最大重试次数

    Returns:
        配置好的 requests.Session
    """
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })
    session._timeout = timeout  # type: ignore[attr-defined]
    return session


def polite_sleep(
    interval_min: float = 1.0,
    interval_max: float = 4.0,
) -> None:
    """随机间隔（反爬虫核心）。"""
    delay = random.uniform(interval_min, interval_max)
    time.sleep(delay)


# ═══════════════════════════════════════════════════════════════
# 列名解析工具
# ═══════════════════════════════════════════════════════════════


def _resolve_column(
    df: pd.DataFrame,
    candidates: List[str],
) -> Optional[str]:
    """从候选列名列表中找到实际存在的列。"""
    for col in candidates:
        if col in df.columns:
            return col
    # 模糊匹配（处理空格/特殊字符）
    for col in df.columns:
        col_str = str(col).strip()
        for cand in candidates:
            if col_str == cand or col_str.lower() == cand.lower():
                return col
    return None


def _normalize_dataframe(
    raw_df: pd.DataFrame,
    exchange: str,
    target_symbol: str,
    cn_name: str,
    fetch_date: pd.Timestamp,
) -> Optional[pd.DataFrame]:
    """把任意结构的原始 DataFrame 标准化为 [date, symbol, receipt]。

    品种匹配策略（2026-06-19 优化，优先 symbol 精确匹配）：
      1) 先按 sym_col 精确匹配 target_symbol 短码（如 "RB"）
      2) fallback 到 cn_name 模糊匹配
      3) 再 fallback 到 short code 大小写不敏感匹配

    Args:
        raw_df: 接口返回的原始 DataFrame
        exchange: 交易所代码（用于选列名映射表）
        target_symbol: 目标品种代码（如 SHFE.RB）
        cn_name: 品种中文名（如 "螺纹钢"）
        fetch_date: 拉取日期（用于填充 date 列）

    Returns:
        标准 DataFrame，列 [date, symbol, receipt]，或 None
    """
    if raw_df is None or raw_df.empty:
        return None
    col_map = RECEIPT_COLUMN_MAP.get(exchange, {})
    sym_col = _resolve_column(raw_df, col_map.get("symbol", []))
    rec_col = _resolve_column(raw_df, col_map.get("receipt", []))
    if sym_col is None or rec_col is None:
        logger.debug(
            "[%s] 列名未匹配: 期望 %s / %s, 实际 %s",
            exchange, col_map.get("symbol"), col_map.get("receipt"),
            raw_df.columns.tolist(),
        )
        return None

    sym_series = raw_df[sym_col].astype(str).str.strip()
    short = target_symbol.split(".")[-1].upper()

    # 1) 优先：symbol 列精确匹配（短码全大写 / 全小写 / 原样）
    sub = sym_series[sym_series.str.upper() == short]
    if sub.empty:
        sub = sym_series[sym_series == short]
    if sub.empty:
        sub = sym_series[sym_series == short.lower()]

    # 2) fallback：cn_name 精确匹配（避免子串误匹配，如"螺纹钢"误匹配"螺纹钢线材"）
    if sub.empty and cn_name:
        sub = sym_series[sym_series == cn_name]

    # 3) fallback：短码 contains（最后一手，SHFE 字段 "铜$cu" 已 split 完）
    if sub.empty:
        sub = sym_series[sym_series.str.upper().str.contains(short, na=False, regex=False)]

    if sub.empty:
        return None

    # 数值列清洗
    sub_df = raw_df.loc[sub.index]
    receipt_vals = pd.to_numeric(sub_df[rec_col], errors="coerce").dropna()
    if receipt_vals.empty:
        return None
    receipt_total = float(receipt_vals.sum())

    return pd.DataFrame([{
        "date": pd.Timestamp(fetch_date).normalize(),
        "symbol": target_symbol,
        "receipt": receipt_total,
    }])


# ═══════════════════════════════════════════════════════════════
# 4 个交易所适配器
# ═══════════════════════════════════════════════════════════════


def _fetch_from_shfe(
    session: requests.Session,
    symbol: str,
    cn_name: str,
    fetch_date: pd.Timestamp,
) -> Optional[pd.DataFrame]:
    """上期所 SHFE 仓单日报（JSON API）。"""
    date_str = fetch_date.strftime("%Y%m%d")
    url = f"https://tsite.shfe.com.cn/data/dailydata/{date_str}dailystock.dat"
    headers = {
        "Host": "tsite.shfe.com.cn",
        "Referer": "https://tsite.shfe.com.cn/statements/dataview.html?paramid=dailystock",
    }
    try:
        r = session.get(url, headers=headers, timeout=session._timeout)  # type: ignore[attr-defined]
        r.raise_for_status()
        data_json = r.json()
        raw_df = pd.DataFrame(data_json["o_cursor"])
        # 拆分品种名（SHFE 字段是 "铜$cu" 格式）
        for col in ("VARNAME", "REGNAME", "WHABBRNAME"):
            if col in raw_df.columns:
                raw_df[col] = raw_df[col].str.split(r"$", expand=True).iloc[:, 0]
        return _normalize_dataframe(raw_df, "SHFE", symbol, cn_name, fetch_date)
    except Exception as e:  # noqa: BLE001
        logger.warning("SHFE@%s 失败: %s", date_str, e)
        return None


def _fetch_from_dce(
    session: requests.Session,
    symbol: str,
    cn_name: str,
    fetch_date: pd.Timestamp,
) -> Optional[pd.DataFrame]:
    """大商所 DCE 仓单日报（HTML 表格）。"""
    date_str = fetch_date.strftime("%Y%m%d")
    url = "http://www.dce.com.cn/publicweb/quotesdata/wbillWeeklyQuotes.html"
    params = {
        "wbillWeeklyQuotes.variety": "all",
        "year": date_str[:4],
        "month": str(int(date_str[4:6]) - 1),
        "day": date_str[6:],
    }
    headers = {"Referer": "http://www.dce.com.cn/dalianshangpin/xqsj/tjsj26/rtj/cdrb/"}
    try:
        r = session.get(url, params=params, headers=headers, timeout=session._timeout)  # type: ignore[attr-defined]
        r.raise_for_status()
        tables = pd.read_html(StringIO(r.text))
        if not tables:
            return None
        raw_df = tables[0]
        return _normalize_dataframe(raw_df, "DCE", symbol, cn_name, fetch_date)
    except Exception as e:  # noqa: BLE001
        logger.debug("DCE@%s 失败: %s", date_str, e)
        return None


def _fetch_from_czce(
    session: requests.Session,
    symbol: str,
    cn_name: str,
    fetch_date: pd.Timestamp,
) -> Optional[pd.DataFrame]:
    """郑商所 CZCE 仓单日报（XLS 多 sheet 文件）。

    2026-06-19 优化：CZCE 的 XLS 包含多个 sheet（每个品种一个），
    之前只读第一个 sheet 会遗漏其他品种。改为 sheet_name=None
    一次性读取所有 sheet 并拼接。
    """
    date_str = fetch_date.strftime("%Y%m%d")
    url = f"http://www.czce.com.cn/cn/DFSStaticFiles/Future/{date_str[:4]}/{date_str}/FutureDataWhsheet.xls"
    headers = {
        "Host": "www.czce.com.cn",
        "Referer": "http://www.czce.com.cn/cn/jysj/cdrb/H770310index_1.htm",
    }
    try:
        r = session.get(url, headers=headers, timeout=session._timeout)  # type: ignore[attr-defined]
        r.raise_for_status()

        # 2026-06-19 优化：sheet_name=None 读取所有 sheet
        try:
            all_sheets = pd.read_excel(BytesIO(r.content), sheet_name=None)
        except Exception as e:  # noqa: BLE001
            logger.warning("CZCE@%s 解析 XLS 失败: %s", date_str, e)
            return None

        if not all_sheets:
            return None

        # 拼接所有 sheet，并附上 sheet 名前缀到首列（便于区分品种来源）
        frames = []
        for sheet_name, sub_df in all_sheets.items():
            if sub_df is None or sub_df.empty:
                continue
            tagged = sub_df.copy()
            # 用 sheet 名填充缺失的"品种"列（XLS 多 sheet 命名 = 品种名）
            tagged["__sheet__"] = str(sheet_name)
            frames.append(tagged)

        if not frames:
            return None
        raw_df = pd.concat(frames, ignore_index=True)

        # 用 _normalize_dataframe 标准化（已支持 symbol 优先匹配）
        return _normalize_dataframe(raw_df, "CZCE", symbol, cn_name, fetch_date)
    except Exception as e:  # noqa: BLE001
        logger.warning("CZCE@%s 失败: %s", date_str, e)
        return None


def _fetch_from_gfex(
    session: requests.Session,
    symbol: str,
    cn_name: str,
    fetch_date: pd.Timestamp,
) -> Optional[pd.DataFrame]:
    """广期所 GFEX 仓单日报（POST 接口返回 JSON）。"""
    date_str = fetch_date.strftime("%Y%m%d")
    url = "http://www.gfex.com.cn/u/interfacesWebTdWbillWeeklyQuotes/loadList"
    headers = {
        "Host": "www.gfex.com.cn",
        "Referer": "https://www.gfex.com.cn/gfex/cdrb/hqsj_tjsj.shtml",
    }
    payload = {"gen_date": date_str}
    try:
        r = session.post(url, data=payload, headers=headers, timeout=session._timeout)  # type: ignore[attr-defined]
        r.raise_for_status()
        data_json = r.json()
        if "data" not in data_json or not data_json["data"]:
            return None
        raw_df = pd.DataFrame(data_json["data"])
        # GFEX 字段映射
        rename_map = {
            "variety": "品种",
            "whAbbr": "仓库/分库",
            "lastWbillQty": "昨日仓单量",
            "wbillQty": "今日仓单量",
        }
        raw_df = raw_df.rename(columns=rename_map)
        return _normalize_dataframe(raw_df, "GFEX", symbol, cn_name, fetch_date)
    except Exception as e:  # noqa: BLE001
        logger.debug("GFEX@%s 失败: %s", date_str, e)
        return None


# ── 适配器注册表（按交易所代码路由） ─────────────────────────
ADAPTER_FN: Dict[str, Callable[..., Optional[pd.DataFrame]]] = {
    "SHFE": _fetch_from_shfe,
    "DCE": _fetch_from_dce,
    "CZCE": _fetch_from_czce,
    "GFEX": _fetch_from_gfex,
}


__all__ = [
    "USER_AGENTS",
    "EXCHANGE_RECEIPT_FN",
    "RECEIPT_COLUMN_MAP",
    "ADAPTER_FN",
    "build_session",
    "polite_sleep",
]
