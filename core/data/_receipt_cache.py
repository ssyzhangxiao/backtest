"""仓单数据缓存层（核心层 private 模块）。

架构位置：core/data/_receipt_cache.py

职责：
  1. 读写本地 parquet 缓存文件
  2. 维护 .meta 元数据文件（status / rows / timestamp / date range）
  3. 实现 TTL 过期检查（默认 7 天）
  4. 区分"空数据"（status=success, rows=0）与"拉取失败"（status=failed）

设计要点（规则 7）：
  - 单一职责：只管缓存 IO，不做接口调用
  - 失败安全：meta 文件缺失/损坏 → 视为缓存失效，重拉
  - TTL 行为：仅当 status=success 时检查 TTL；failed 缓存立即失效
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Meta 文件 IO
# ═══════════════════════════════════════════════════════════════


def _meta_path(cache_path: Path) -> Path:
    """根据 parquet 路径推导 meta 路径。"""
    return cache_path.with_suffix(cache_path.suffix + ".meta")


def _read_meta(meta_file: Path) -> Optional[dict]:
    """读取 meta 文件，失败返回 None。"""
    if not meta_file.exists():
        return None
    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        logger.warning("读取 meta 失败 %s: %s", meta_file, e)
        return None


def _write_meta(meta_file: Path, payload: dict) -> None:
    """写入 meta 文件，失败仅警告不抛异常。"""
    try:
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:  # noqa: BLE001
        logger.warning("写入 meta 失败 %s: %s", meta_file, e)


# ═══════════════════════════════════════════════════════════════
# 缓存类
# ═══════════════════════════════════════════════════════════════


class ReceiptCache:
    """仓单缓存管理器（parquet + meta 双文件）。"""

    def __init__(
        self,
        cache_dir: str = "data/receipt_cache",
        ttl_days: int = 7,
    ) -> None:
        """初始化缓存管理器。

        Args:
            cache_dir: 缓存目录路径
            ttl_days: 缓存有效期（天），超过则视为过期
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_days = ttl_days

    def _cache_path(self, start_date: str, end_date: str) -> Path:
        """生成缓存文件路径。"""
        return self.cache_dir / f"receipt_{start_date}_{end_date}.parquet"

    # ── 读 ────────────────────────────────────────────────────

    def load(
        self,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """从缓存加载数据。

        三态返回（2026-06-19 优化）：
          - parquet 缺失 / meta 缺失 / meta 损坏 → None（未命中，需重拉）
          - status=failed → None（拉取失败，需重拉）
          - status=success + TTL 过期 → None（缓存过期，需重拉）
          - status=success + TTL 内 → DataFrame（可能为空，已成功拉取无数据）

        Returns:
            缓存命中的 DataFrame（可能为空）；未命中 / 失败 / 过期 → None
        """
        cache_path = self._cache_path(start_date, end_date)
        meta_path = _meta_path(cache_path)
        if not cache_path.exists() or not meta_path.exists():
            return None

        meta = _read_meta(meta_path)
        if meta is None:
            return None

        # 状态校验：仅 success 命中；failed 一律视为未命中
        if meta.get("status") != "success":
            logger.debug("缓存状态非 success: %s", meta.get("status"))
            return None

        # TTL 检查
        ts_str = meta.get("timestamp")
        if ts_str and not self._is_fresh(ts_str):
            logger.info("缓存已过期 (TTL=%d天): %s", self.ttl_days, cache_path)
            return None

        # 读 parquet（status=success 必有 parquet 文件，即便 rows=0）
        try:
            df = pd.read_parquet(cache_path)
        except Exception as e:  # noqa: BLE001
            logger.warning("读取缓存失败 %s: %s", cache_path, e)
            return None

        # df 为空 DataFrame 也算命中（status=success, rows=0 表示已确认无数据）
        rows = len(df) if df is not None else 0
        logger.info("命中缓存: %s (%d 行)", cache_path, rows)
        return df

    def _is_fresh(self, timestamp_str: str) -> bool:
        """检查 timestamp 是否在 TTL 窗口内。"""
        try:
            ts = datetime.fromisoformat(timestamp_str)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - ts).total_seconds() / 86400
            return age_days <= self.ttl_days
        except Exception:  # noqa: BLE001
            return False

    # ── 写 ────────────────────────────────────────────────────

    def save(
        self,
        df: pd.DataFrame,
        start_date: str,
        end_date: str,
        status: str = "success",
    ) -> None:
        """写缓存 + meta。

        Args:
            df: 数据 DataFrame
            start_date: 拉取起始日期
            end_date: 拉取结束日期
            status: 'success' / 'failed'

        写入策略（2026-06-19 优化）：
          - status=success → 一律写 parquet（空数据也写空 df，标记"已确认无数据"）
          - status=failed → 只写 meta（不写 parquet，下一周期需重拉）
        """
        cache_path = self._cache_path(start_date, end_date)
        meta_path = _meta_path(cache_path)

        df_empty = df is None or df.empty
        meta_payload = {
            "status": status,
            "rows": int(len(df)) if df is not None else 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "start_date": start_date,
            "end_date": end_date,
            "schema": list(df.columns) if not df_empty else [],
        }

        if status == "success":
            # 成功（含空数据）：写 parquet（空 df 也写）+ meta
            try:
                if df_empty:
                    # 显式写空 schema 的 parquet，作为"已确认无数据"的标记
                    pd.DataFrame(columns=["date", "symbol", "receipt"]).to_parquet(cache_path)
                else:
                    df.to_parquet(cache_path)
                _write_meta(meta_path, meta_payload)
                logger.info("已缓存: %s (%d 行, status=%s)", cache_path, meta_payload["rows"], status)
            except Exception as e:  # noqa: BLE001
                logger.warning("写缓存失败 %s: %s", cache_path, e)
        else:
            # 失败：只写 meta（不写 parquet），避免下次误判为已确认无数据
            _write_meta(meta_path, meta_payload)
            logger.info("记录失败状态: %s (status=%s)", cache_path, status)

    def clear(self, start_date: str, end_date: str) -> None:
        """清除指定日期范围的缓存。"""
        cache_path = self._cache_path(start_date, end_date)
        meta_path = _meta_path(cache_path)
        for p in (cache_path, meta_path):
            if p.exists():
                try:
                    p.unlink()
                except Exception as e:  # noqa: BLE001
                    logger.warning("删除缓存失败 %s: %s", p, e)


# ═══════════════════════════════════════════════════════════════
# 全量加载（无网络）
# ═══════════════════════════════════════════════════════════════


def load_receipt_cache(
    cache_dir: Path,
    symbols: Optional[list] = None,
) -> dict:
    """从缓存目录加载所有 parquet 切片，按品种合并为 {symbol: Series}。

    无 AKShare 网络依赖，纯本地 IO。供回测/实验在已下载数据后快速加载。

    Args:
        cache_dir: 缓存目录路径
        symbols: 品种过滤列表，None 表示加载所有品种

    Returns:
        {symbol: pd.Series(按日期索引)} 字典，缺失品种返回空 Series
    """
    out: dict = {}
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return out

    parquet_files = sorted(cache_dir.glob("receipt_*.parquet"))
    if not parquet_files:
        return out

    all_dfs = []
    for p in parquet_files:
        try:
            df = pd.read_parquet(p)
            if df is not None and not df.empty:
                all_dfs.append(df)
        except Exception as e:  # noqa: BLE001
            logger.warning("读取缓存文件失败 %s: %s", p, e)
            continue

    if not all_dfs:
        return out

    merged = pd.concat(all_dfs, ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"])

    target_syms = set(symbols) if symbols else None
    for sym, sub in merged.groupby("symbol"):
        if target_syms is not None and sym not in target_syms:
            continue
        sub = sub.sort_values("date").drop_duplicates("date", keep="last")
        out[str(sym)] = pd.Series(
            sub["receipt"].astype(float).to_numpy(),
            index=pd.to_datetime(sub["date"]).to_numpy(),
            name=str(sym),
        )
    return out


__all__ = [
    "ReceiptCache",
    "load_receipt_cache",
]
