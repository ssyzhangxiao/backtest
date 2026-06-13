"""
CTA 批处理脚本 — 全品种 × 多策略 × 参数扫描。

用法:
    # 全品种基准跑分
    python scripts/run_cta_batch.py --mode benchmark

    # 参数扫描（momentum_ma 的 entry_threshold 和 fast_ma/slow_ma）
    python scripts/run_cta_batch.py --mode sweep --strategy momentum_ma

    # 参数扫描（donchian_breakout 的 entry_lookback）
    python scripts/run_cta_batch.py --mode sweep --strategy donchian_breakout

    # 打印汇总
    python scripts/run_cta_batch.py --mode summary

    # 仅跑特定品种
    python scripts/run_cta_batch.py --mode benchmark --symbols SHFE.AU DCE.I INE.SC
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 确保项目根目录在 path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_cta_batch")

# ── 可用品种 ──
# 按大类和流动性排序
_PREFERRED_SYMBOLS = [
    "SHFE.AU", "SHFE.AG",
    "SHFE.CU", "SHFE.AL", "SHFE.ZN", "SHFE.NI",
    "DCE.I", "DCE.J", "DCE.JM", "DCE.JD",
    "INE.SC", "CZCE.TA", "CZCE.MA", "DCE.EG", "DCE.PP",
    "DCE.L", "DCE.V", "CZCE.FG", "SHFE.BU", "SHFE.RB",
    "SHFE.HC", "CZCE.ZC",
    "DCE.M", "DCE.Y", "DCE.P", "CZCE.OI", "CZCE.CF",
    "CZCE.SR", "CZCE.RM", "DCE.C", "DCE.CS",
    "CFFEX.IF", "CFFEX.IC", "CFFEX.IH",
    "SHFE.RU",
    "INE.LU", "INE.NR",
]

# ── 参数扫描网格 ──
_PARAM_GRID = {
    "momentum_ma": {
        "entry_threshold": [0.02, 0.05, 0.10],
        "params": [
            {"fast_ma": 5, "slow_ma": 20},
            {"fast_ma": 10, "slow_ma": 30},
            {"fast_ma": 20, "slow_ma": 60},
        ],
    },
    "donchian_breakout": {
        "entry_threshold": [0.02, 0.05],
        "params": [
            {"entry_lookback": 10, "atr_entry_mult": 0.5, "use_adx_filter": True},
            {"entry_lookback": 20, "atr_entry_mult": 0.5, "use_adx_filter": True},
            {"entry_lookback": 40, "atr_entry_mult": 0.3, "use_adx_filter": True},
        ],
    },
    "vol_mean_reversion": {
        "entry_threshold": [0.02, 0.05],
        "params": [
            {"vol_window": 10, "entry_z": 1.2, "vol_percentile": 0.7},
            {"vol_window": 20, "entry_z": 1.2, "vol_percentile": 0.7},
            {"vol_window": 20, "entry_z": 1.5, "vol_percentile": 0.8},
        ],
    },
}

# 退出参数与回测配置（可通过 CLI 覆盖）
_DEFAULT_EXIT_PARAMS = {
    "max_holding_days": 30,
    "atr_stop_multiple": 1.5,
    "atr_window": 14,
    "stop_loss_pct": 0.02,
    "global_risk_pct": 0.05,
    "risk_per_trade": 0.02,
    "target_vol": 0.15,          # sigma 风险平价（启用）
}

# 策略专属退出参数（覆盖 _DEFAULT_EXIT_PARAMS）
_STRATEGY_EXIT_PARAMS = {
    "donchian_breakout":    {"max_holding_days": 45, "atr_stop_multiple": 2.0, "target_vol": 0.0, "risk_per_trade": 0.015},
    "momentum_ma":          {"max_holding_days": 30, "atr_stop_multiple": 1.5},
    "vol_mean_reversion":   {"max_holding_days": 20, "atr_stop_multiple": 1.2},
    "tsi_garch":            {"max_holding_days": 25, "atr_stop_multiple": 1.5, "target_vol": 0.15},
    "carry":                {"max_holding_days": 20, "atr_stop_multiple": 1.2},
    "pair_trading":         {"max_holding_days": 20, "atr_stop_multiple": 1.2},
}
_WARMUP = 30  # 可通过 CLI --warmup 覆盖
_FULL_START = "2020-01-01"  # 回测起始日期
_TEST_START = "2023-01-01"  # 回测截止日期

# TqSdk 预加载的数据缓存（由 main() 在 --tqsdk 模式下填充）
_TQ_DFS: Dict[str, pd.DataFrame] = {}


_EXCHANGE_MAP = {
    ".SHF": "SHFE",
    ".CZC": "CZCE",
    ".DCE": "DCE",
    ".CFE": "CFFEX",
    ".INE": "INE",
}

# 反向映射：交易所缩写 → 后缀（用于 _normalize_symbol_for_pybroker 判断格式）
_REVERSE_EXCHANGE_MAP = {exch: suffix for suffix, exch in _EXCHANGE_MAP.items()}


def _to_tqsdk_symbol(symbol: str) -> str:
    """转换内部格式 "RB.SHF" → TqSDK 格式 "SHFE.RB"。"""
    parts = symbol.split(".")
    if len(parts) != 2:
        return symbol
    prod, exch_suffix = parts[0], "." + parts[1]
    exch = _EXCHANGE_MAP.get(exch_suffix, exch_suffix.lstrip("."))
    return f"{exch}.{prod.upper()}"


def _normalize_symbol_for_pybroker(symbol: str) -> str:
    """标准化 symbol 为 PyBroker 格式 "EXCHANGE.PRODUCT"。

    输入 "RB.SHF" → "SHFE.RB"；输入 "SHFE.RB" → "SHFE.RB"。
    """
    if symbol.count(".") == 1:
        left, right = symbol.split(".")
        # 如果左边是交易所缩写（3-4大写字母）→ 已是标准格式
        if left in _REVERSE_EXCHANGE_MAP:
            return symbol.upper()
        # 否则是 "RB.SHF" 格式 → 转为 "SHFE.RB"
        return _to_tqsdk_symbol(symbol)
    return symbol.upper()


def _resolve_cache_file(symbol: str) -> Optional[Path]:
    """找到品种最近修改的缓存文件。

    支持两种 symbol 格式：
      - "RB.SHF"     (PRODUCT.EXCHANGE) → 搜索 SHFE_RB_*.pkl
      - "SHFE.RB"    (EXCHANGE.PRODUCT) → 搜索 SHFE_RB_*.pkl
    """
    cache_dir = _PROJECT_ROOT / "data_cache"
    prefix = _symbol_to_cache_prefix(symbol)
    candidates = sorted(
        cache_dir.glob(f"{prefix}_*.pkl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    fallback = cache_dir / f"{prefix}.pkl"
    return fallback if fallback.exists() else None


def _symbol_to_cache_prefix(symbol: str) -> str:
    """将 symbol 转为缓存文件名前缀。

    输入 "RB.SHF" 或 "SHFE.RB" → 统一输出 "SHFE_RB"。
    """
    parts = symbol.split(".")
    if len(parts) != 2:
        return symbol.replace(".", "_").replace("-", "_")

    left, right = parts[0], parts[1]

    # 情况 1："RB.SHF" → 通过 _EXCHANGE_MAP 查交易所全名
    exch_suffix = "." + right  # ".SHF"
    if exch_suffix in _EXCHANGE_MAP:
        return f"{_EXCHANGE_MAP[exch_suffix]}_{left.upper()}"

    # 情况 2："SHFE.RB" → 左半是交易所全名
    exch_full = left.upper()
    prod = right.upper()
    # 验证左半是否为已知交易所
    known_exchanges = {v.lower() for v in _EXCHANGE_MAP.values()}
    if left.lower() in known_exchanges:
        return f"{exch_full}_{prod}"

    # 未知格式：直接拼接
    return f"{left.upper()}_{right.upper()}"


def _load_symbol_data(symbol: str) -> Optional[pd.DataFrame]:
    """加载单个品种的缓存数据。

    若缓存数据不存在或日期范围不覆盖回测区间，自动通过 TqSDK 下载。
    """
    cache_file = _resolve_cache_file(symbol)
    df = None
    if cache_file:
        try:
            with open(cache_file, "rb") as f:
                df = pickle.load(f)
            df["product_code"] = symbol
            dominant = df.loc[
                df.groupby(df["date"].dt.date)["open_interest"].idxmax()
            ].copy()
            dominant["symbol"] = symbol
            # 检查必要列是否包含过多 NaN（超过 10% 则丢弃）
            for col in ["close", "high", "low", "open_interest"]:
                if col in dominant.columns and dominant[col].isna().sum() > len(dominant) * 0.1:
                    logger.warning("%s: %s 列 NaN 占比过高，跳过", symbol, col)
                    dominant = None
                    break
            if dominant is not None and len(dominant) > 100:
                df = dominant
            else:
                df = None
        except Exception as exc:
            logger.warning("%s: 加载失败 %s", symbol, exc)
            df = None

    if df is not None:
        return df

    # 缓存不可用：尝试 TqSDK 下载
    return _ensure_tqsdk_data(symbol)


# 需要 spread 数据的策略名列表
_SPREAD_DEPENDENT_STRATEGIES = {"carry", "pair_trading", "carry_zscore"}


def _has_spread_columns(df: pd.DataFrame) -> bool:
    """检查 DataFrame 是否包含 spread 或 far_close 列。"""
    return "spread" in df.columns or "far_close" in df.columns


def _ensure_tqsdk_spread_data(symbol: str) -> Optional[pd.DataFrame]:
    """通过 TqSDK 下载含 spread/far_close 的真实数据，并缓存。

    流程（规则30）：
        1. 检查 data_cache 中是否有已保存的 _spread.pkl 缓存
        2. 若无，通过 DataLoader(tqsdk) 下载 → build_spread_pairs()
        3. 缓存结果到 data_cache/{EXCHANGE}_{PRODUCT}_spread.pkl
        4. 返回 PyBroker 格式 DataFrame（含 spread/far_close/close列）

    Returns:
        PyBroker 格式 DataFrame，失败返回 None
    """
    # 1. 检查已有 spread 缓存
    cache_dir = _PROJECT_ROOT / "data_cache"
    prefix = _symbol_to_cache_prefix(symbol)

    spread_cache = cache_dir / f"{prefix}_spread.pkl"
    if spread_cache.exists():
        logger.info("%s: 使用 spread 缓存", symbol)
        try:
            with open(spread_cache, "rb") as f:
                df = pickle.load(f)
            if _has_spread_columns(df) and len(df) > 100:
                return df
        except Exception:
            logger.warning("%s: spread 缓存损坏，重新下载", symbol)

    # 2. TqSDK 下载（复用 _ensure_tqsdk_data 的基础下载，再补 build_spread_pairs）
    logger.info("%s: 缓存无 spread/far_close，通过 TqSDK 下载...", symbol)
    try:
        from dotenv import load_dotenv
        load_dotenv()
        from core.data.data_loader import DataLoader

        # 加载 TqSDK 凭证
        loader = _create_tqsdk_loader(symbol)
        if loader is None:
            return None

        loader.load_from_tqsdk(show_progress=False)
        loader.identify_dominant_contracts()
        loader.build_continuous_series()
        loader.build_spread_pairs()
        df = loader.get_pybroker_df()

        if df is None or df.empty:
            logger.warning("%s: TqSDK 下载结果为空", symbol)
            return None
        if not _has_spread_columns(df):
            logger.warning("%s: TqSDK 数据仍无 spread 列（可能无远月合约数据）", symbol)
        if len(df) < 100:
            logger.warning("%s: TqSDK 数据不足 100 行", symbol)
            return None

        # 3. 缓存
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(spread_cache, "wb") as f:
            pickle.dump(df, f)
        logger.info("%s: spread 数据已缓存到 %s", symbol, spread_cache.name)
        return df

    except Exception as exc:
        logger.warning("%s: TqSDK spread 下载失败: %s", symbol, exc)
        return None


def _load_tqsdk_credentials() -> tuple:
    """从环境变量或 config.yaml 加载 TqSDK 凭证。"""
    import os
    phone = os.environ.get("TQSDK_PHONE")
    password = os.environ.get("TQSDK_PASSWORD")
    if not phone or not password:
        try:
            import yaml as _yaml
            _cfg_path = _PROJECT_ROOT / "config.yaml"
            if _cfg_path.exists():
                with open(_cfg_path, "r") as _f:
                    _cfg = _yaml.safe_load(_f)
                _data_cfg = _cfg.get("data", {})
                phone = phone or _data_cfg.get("tqsdk_phone")
                password = password or _data_cfg.get("tqsdk_password")
        except Exception:
            pass
    return phone, password


def _create_tqsdk_loader(symbol: str) -> Optional[object]:
    """创建 TqSDK DataLoader，返回 None 表示凭证不可用。"""
    phone, password = _load_tqsdk_credentials()
    if not phone or not password:
        logger.warning("%s: TqSDK 凭证未配置，无法下载数据", symbol)
        return None
    from core.data.data_loader import DataLoader
    tqsdk_sym = _normalize_symbol_for_pybroker(symbol)
    return DataLoader(
        data_source="tqsdk",
        phone=phone,
        password=password,
        symbols=[tqsdk_sym],
        data_length=2000,
    )


def _ensure_tqsdk_data(symbol: str) -> Optional[pd.DataFrame]:
    """通过 TqSDK 下载基础 OHLCV 数据（不含 spread），并缓存。

    用于非 spread 依赖策略（momentum_ma/donchian/vol_mr/tsi_garch）
    在缓存数据过时或缺失时自动补数据。

    Returns:
        PyBroker 格式 DataFrame，失败返回 None
    """
    cache_dir = _PROJECT_ROOT / "data_cache"
    prefix = _symbol_to_cache_prefix(symbol)
    basic_cache = cache_dir / f"{prefix}.pkl"

    logger.info("%s: 缓存数据不足，通过 TqSDK 下载...", symbol)
    try:
        from dotenv import load_dotenv
        load_dotenv()

        loader = _create_tqsdk_loader(symbol)
        if loader is None:
            return None

        loader.load_from_tqsdk(show_progress=False)
        loader.identify_dominant_contracts()
        loader.build_continuous_series()
        df = loader.get_pybroker_df()

        if df is None or df.empty or len(df) < 100:
            logger.warning("%s: TqSDK 基础数据不足", symbol)
            return None

        # 缓存
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(basic_cache, "wb") as f:
            pickle.dump(df, f)
        logger.info("%s: 基础数据已缓存到 %s", symbol, basic_cache.name)
        return df

    except Exception as exc:
        logger.warning("%s: TqSDK 基础数据下载失败: %s", symbol, exc)
        return None


def _run_single(
    symbol: str,
    strategy_name: str,
    strategy_params: Dict[str, Any],
    entry_threshold: float,
    full_start: Optional[str] = None,
    test_start: Optional[str] = None,
    initial_cash: float = 1_000_000,
    exit_params: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """运行单个品种×单个策略×单组参数的回测。"""
    try:
        import pybroker
        from pybroker import StrategyConfig
    except ImportError:
        return None

    from core.engine.cta_executor_builder import CTAExecutorBuilder
    from core.strategies.cta.registry import get_cta_strategy

    # 使用模块级默认日期
    full_start = full_start or _FULL_START
    test_start = test_start or _TEST_START

    # 优先使用 TqSdk 预加载数据，否则从本地缓存加载
    df_pb = _TQ_DFS.get(symbol)
    if df_pb is None:
        df = _load_symbol_data(symbol)
        if df is None:
            return None
        from core.engine.pybroker_data_source import PyBrokerDataSource
        ds = PyBrokerDataSource(df)
        try:
            symbol_data = ds.for_symbol(symbol)
        except ValueError:
            return None
        df_pb = symbol_data.to_pybroker_df()

    if len(df_pb) < 100:
        return None

    # 检查数据是否覆盖回测区间，若不足则自动 TqSDK 补数据
    if test_start is not None and "date" in df_pb.columns:
        max_date = pd.to_datetime(df_pb["date"]).max()
        test_start_dt = pd.to_datetime(test_start)
        if max_date < test_start_dt - pd.Timedelta(days=30):
            logger.info(
                "%s: 缓存数据截止 %s，回测需覆盖 %s，自动通过 TqSDK 下载最新数据",
                symbol, max_date.date(), test_start,
            )
            replacement = _ensure_tqsdk_data(symbol)
            if replacement is not None:
                df_pb = replacement

    try:
        cta = get_cta_strategy(strategy_name, strategy_params)
    except ValueError:
        return None

    # 注入 spread 数据（规则30：优先从列读取，无则 TqSDK 自动下载）
    has_spread = _has_spread_columns(df_pb)
    needs_spread = strategy_name.lower() in _SPREAD_DEPENDENT_STRATEGIES
    if needs_spread and not has_spread:
        logger.info(
            "%s %s: 缓存无 spread/far_close，自动通过 TqSDK 下载真实数据",
            symbol, strategy_name,
        )
        spread_df = _ensure_tqsdk_spread_data(symbol)
        if spread_df is not None and len(spread_df) >= 100:
            # TqSDK 的 symbol 列是 "EXCHANGE.PRODUCT" 格式（如 "CZCE.MA"），
            # 但 _run_single 期望的 symbol 参数是 "PRODUCT.EXCHANGE" 格式（如 "MA.CZC"）。
            # 把 symbol 列统一为输入格式，使 PyBroker 能正确匹配。
            spread_df["symbol"] = symbol
            df_pb = spread_df
            has_spread = True

    if "spread" in df_pb.columns:
        spread_arr = df_pb["spread"].to_numpy(dtype=float, copy=True)
        cta.set_state(symbol, "_spread", spread_arr)
    elif "far_close" in df_pb.columns:
        far = df_pb["far_close"].to_numpy(dtype=float, copy=True)
        close_val = df_pb["close"].to_numpy(dtype=float, copy=True)
        spread_synth = np.where(np.isfinite(far) & np.isfinite(close_val) & (close_val > 0), (far - close_val) / close_val * 100, np.nan)
        cta.set_state(symbol, "_spread", spread_synth)

    # 注入 far_close 到策略状态（供 pair_trading 使用）
    if "far_close" in df_pb.columns:
        far_close_arr = df_pb["far_close"].to_numpy(dtype=float, copy=True)
        cta.set_state(symbol, "_far_price", far_close_arr)

    # ── Spread 数据质量检查 ──
    needs_spread = strategy_name.lower() in _SPREAD_DEPENDENT_STRATEGIES
    if needs_spread:
        spread_check = cta.get_state(symbol, "_spread", None)
        if spread_check is not None:
            valid = spread_check[np.isfinite(spread_check)]
            valid_pct = len(valid) / max(len(spread_check), 1) * 100
            logger.debug(
                "%s %s: spread 有效 %.0f%% (%d/%d)",
                symbol, strategy_name, valid_pct,
                len(valid), len(spread_check),
            )
            if valid_pct < 10:
                logger.warning(
                    "%s %s: spread 有效仅 %.0f%%，信号可能不可靠",
                    symbol, strategy_name, valid_pct,
                )

    ep = {**(exit_params or _DEFAULT_EXIT_PARAMS)}
    # 策略专属退出参数覆盖
    if strategy_name in _STRATEGY_EXIT_PARAMS:
        for k, v in _STRATEGY_EXIT_PARAMS[strategy_name].items():
            ep[k] = v
    builder = CTAExecutorBuilder(
        cta_strategy=cta,
        entry_threshold=entry_threshold,
        max_position_pct=0.3,
        max_holding_days=ep["max_holding_days"],
        atr_stop_multiple=ep["atr_stop_multiple"],
        atr_window=ep["atr_window"],
        stop_loss_pct=ep["stop_loss_pct"],
        global_risk_pct=ep["global_risk_pct"],
        risk_per_trade=ep.get("risk_per_trade", 0.02),
        target_vol=ep.get("target_vol", 0.15),
    )
    executor_fn = builder.build()

    pb_config = StrategyConfig(initial_cash=initial_cash)
    strategy = pybroker.Strategy(df_pb, full_start, test_start, config=pb_config)
    strategy.add_execution(executor_fn, symbols=[symbol])

    try:
        result = strategy.backtest(warmup=_WARMUP)
    except ValueError as e:
        logger.warning("%s %s 回测失败: %s", symbol, strategy_name, e)
        return None

    # 提取指标
    metrics: Dict[str, Any] = {
        "symbol": symbol,
        "strategy": strategy_name,
        "entry_threshold": entry_threshold,
        "full_start": full_start,
        "test_start": test_start,
        "total_return_pct": 0.0,
        "annual_return_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "sharpe_ratio": 0.0,
        "calmar_ratio": 0.0,
        "total_trades": 0,
        "win_rate_pct": 0.0,
    }
    # 保存策略参数到结果（便于分析最优参数组合）
    for k, v in strategy_params.items():
        metrics[k] = v

    if hasattr(result, "metrics_df") and result.metrics_df is not None:
        try:
            m = result.metrics_df.set_index("name")["value"] if "name" in result.metrics_df.columns else {}
        except Exception:
            m = {}
        metrics["total_return_pct"] = float(m.get("total_return_pct", 0) or 0)
        metrics["max_drawdown_pct"] = float(m.get("max_drawdown_pct", 0) or 0)
        metrics["sharpe_ratio"] = float(m.get("sharpe", 0) or 0)
        trade_count = m.get("trade_count")
        if trade_count is not None:
            try:
                metrics["total_trades"] = int(trade_count)
            except (ValueError, TypeError):
                pass
        ret = metrics["total_return_pct"]
        dd = abs(metrics["max_drawdown_pct"])
        metrics["calmar_ratio"] = round(ret / dd, 2) if dd > 0.01 else 0.0

    # 计算年化收益 = total_return 按实际交易日年化
    if metrics["total_return_pct"] != 0.0 and len(df_pb) > 1:
        import pandas as _pd
        dates = _pd.to_datetime(df_pb["date"])
        trading_days = (dates.max() - dates.min()).days
        if trading_days > 60:
            ann_factor = 365.0 / trading_days
            metrics["annual_return_pct"] = round(
                (1 + metrics["total_return_pct"] / 100) ** ann_factor - 1, 4
            ) * 100
            # 用实际年化收益重算 calmar
            dd2 = abs(metrics["max_drawdown_pct"])
            if dd2 > 0.01:
                metrics["calmar_ratio"] = round(metrics["annual_return_pct"] / dd2, 2)

    if result.trades is not None and not result.trades.empty:
        if "pnl" in result.trades.columns:
            pnl = result.trades["pnl"]
            win = (pnl > 0).sum()
            metrics["total_trades"] = len(pnl)
            metrics["win_rate_pct"] = round(float(win) / len(pnl) * 100, 1)

    return metrics


# ══════════════════════════════════════════════════════════════════
# MODES
# ══════════════════════════════════════════════════════════════════


def _load_results() -> pd.DataFrame:
    """加载已保存的结果。"""
    results_file = _PROJECT_ROOT / "output" / "cta_batch_results.parquet"
    if results_file.exists():
        return pd.read_parquet(results_file)
    return pd.DataFrame()


def _save_results(df: pd.DataFrame) -> None:
    """保存结果。"""
    output_dir = _PROJECT_ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    df.to_parquet(output_dir / "cta_batch_results.parquet")
    df.to_csv(output_dir / "cta_batch_results.csv", index=False)


def _run_strategy_sweep(
    strategy_name: str,
    symbols: List[str],
    grid: Dict,
) -> pd.DataFrame:
    """运行策略的参数扫描。"""
    results = []
    params_list = grid.get("params", [{}])
    thresholds = grid.get("entry_threshold", [0.05])
    total = len(symbols) * len(params_list) * len(thresholds)
    count = 0

    for symbol in symbols:
        for sp in params_list:
            for et in thresholds:
                count += 1
                print(f"  [{count}/{total}] {strategy_name} {symbol} et={et} params={sp}", end="", flush=True)
                t0 = time.time()
                r = _run_single(symbol, strategy_name, sp, et)
                dt = time.time() - t0
                if r:
                    results.append(r)
                    ret = r.get("total_return_pct", 0)
                    cal = r.get("calmar_ratio", 0)
                    print(f"  ret={ret:+.1f}% calmar={cal:.2f} ({dt:.1f}s)")
                else:
                    print(f"  SKIP ({dt:.1f}s)")

    return pd.DataFrame(results)


def mode_benchmark(symbols: List[str], period: str = "both") -> pd.DataFrame:
    """全品种 × 默认参数的基准跑分。"""
    strategies = [
        ("momentum_ma", {"rsi_window": 14}),
        ("donchian_breakout", {"entry_lookback": 20, "atr_window": 14, "atr_entry_mult": 0.3, "trend_filter_ma": 60, "momentum_factor": 0.2}),
        ("vol_mean_reversion", {"vol_window": 20, "lookback": 252, "entry_z": 1.0, "vol_percentile": 0.6}),
        ("tsi_garch", {"reg_window": 60, "min_obs": 30, "t_stat_threshold": 2.0, "cache_update_freq": 5}),
        ("carry", {"lookback": 60, "entry_z": 1.5, "direction": "long_only", "ema_alpha": 0.3, "use_slope": True}),
        ("pair_trading", {"lookback": 60, "entry_z": 2.0, "ols_window": 90, "adf_interval": 20}),
    ]

    # 定义时段
    periods = []
    if period in ("is", "both"):
        periods.append(("IS", _FULL_START, _TEST_START))
    if period in ("oos", "both"):
        periods.append(("OOS", _TEST_START, "2024-12-31"))

    all_results = []
    total = len(symbols) * len(strategies) * len(periods)
    count = 0

    for period_name, full_start, test_end in periods:
        for sname, sp in strategies:
            for symbol in symbols:
                count += 1
                print(f"  [{count}/{total}] {period_name:3s} {sname:25s} {symbol:12s}  ", end="", flush=True)
                t0 = time.time()
                r = _run_single(symbol, sname, sp, entry_threshold=0.005,
                                full_start=full_start, test_start=test_end)
                dt = time.time() - t0
                if r:
                    r["period"] = period_name
                    all_results.append(r)
                    ret = r.get("total_return_pct", 0)
                    print(f"ret={ret:+.1f}%  trades={r.get('total_trades', 0)}  ({dt:.1f}s)")
                else:
                    print(f"SKIP  ({dt:.1f}s)")

    df = pd.DataFrame(all_results)
    _save_results(df)
    return df


def mode_sweep(strategy_name: str, symbols: List[str]) -> pd.DataFrame:
    """策略参数扫描。"""
    grid = _PARAM_GRID.get(strategy_name)
    if not grid:
        print(f"未知策略: {strategy_name}，可选: {list(_PARAM_GRID.keys())}")
        return pd.DataFrame()

    print(f"\n参数扫描: {strategy_name}")
    print(f"  品种数: {len(symbols)}")
    print(f"  参数组合: {len(grid['params'])} × entry_threshold {grid['entry_threshold']}")
    print()

    new_results = _run_strategy_sweep(strategy_name, symbols, grid)

    # 合并已有结果
    existing = _load_results()
    if not existing.empty:
        existing = existing[existing["strategy"] != strategy_name]
    combined = pd.concat([existing, new_results], ignore_index=True)
    # 去重：同策略+同品种+同参数只保留最后一条
    dedup_cols = [c for c in ["symbol", "strategy", "entry_threshold"] + list(grid.get("params", [{}])[0].keys()) if c in combined.columns]
    if dedup_cols:
        combined = combined.drop_duplicates(subset=dedup_cols, keep="last")
    _save_results(combined)
    return combined


def mode_summary() -> None:
    """汇总结果。"""
    df = _load_results()
    if df.empty:
        print("暂无结果，先运行 --mode benchmark 或 --mode sweep")
        return

    print("\n" + "=" * 100)
    print("  CTA 全品种回测结果汇总")
    # 显示回测日期区间
    if not df.empty and "full_start" in df.columns:
        fs = df["full_start"].iloc[0]
        ts = df["test_start"].iloc[0]
        print(f"  回测区间: {fs} ~ {ts}")
    print("=" * 100)

    for sname in sorted(df["strategy"].unique()):
        sdf = df[df["strategy"] == sname].copy()
        print(f"\n▶ {sname} ({len(sdf)} 个品种×参数组合)")

        best = sdf.sort_values("calmar_ratio", ascending=False).head(10)
        print(f"  {'symbol':12s} {'et':5s} {'ret%':>6s} {'dd%':>6s} {'sharpe':>7s} {'calmar':>7s} {'trades':>6s} {'win%':>5s}")
        print(f"  {'-'*12} {'-'*5} {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*6} {'-'*5}")
        for _, r in best.iterrows():
            print(f"  {r['symbol']:12s} {r.get('entry_threshold',0):.2f}  {r['total_return_pct']:+.1f}% {r['max_drawdown_pct']:.1f}% {r['sharpe_ratio']:.2f}  {r['calmar_ratio']:.2f}   {int(r['total_trades']):5d}  {r['win_rate_pct']:.1f}%")

        win_count = (sdf["total_return_pct"] > 0).sum()
        print(f"  盈利: {win_count}/{len(sdf)}  平均收益: {sdf['total_return_pct'].mean():+.1f}%  平均卡玛: {sdf['calmar_ratio'].mean():.2f}")

    print("\n" + "-" * 100)
    print("  跨策略 TOP 10")
    print("-" * 100)
    all_best = df.sort_values("calmar_ratio", ascending=False).head(10)
    print(f"  {'strategy':25s} {'symbol':12s} {'et':5s} {'ret%':>6s} {'dd%':>6s} {'sharpe':>7s} {'calmar':>7s} {'trades':>5s} {'win%':>5s}")
    print(f"  {'-'*25} {'-'*12} {'-'*5} {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*5} {'-'*5}")
    for _, r in all_best.iterrows():
        print(f"  {r['strategy']:25s} {r['symbol']:12s} {r.get('entry_threshold',0):.2f}  {r['total_return_pct']:+.1f}% {r['max_drawdown_pct']:.1f}% {r['sharpe_ratio']:.2f}  {r['calmar_ratio']:.2f}   {int(r['total_trades']):4d}  {r['win_rate_pct']:.1f}%")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="CTA 批处理脚本")
    parser.add_argument("--mode", choices=["benchmark", "sweep", "summary"], default="summary")
    parser.add_argument("--strategy", default=None, help="策略名（sweep 模式必需）")
    parser.add_argument("--symbols", nargs="+", default=None, help="品种列表，默认全品种")
    # 退出参数（可选覆盖默认值）
    parser.add_argument("--max_holding_days", type=int, default=None, help="最大持仓天数")
    parser.add_argument("--atr_stop_multiple", type=float, default=None, help="ATR 止损倍数")
    parser.add_argument("--global_risk_pct", type=float, default=None, help="日组合亏损熔断比例")
    parser.add_argument("--warmup", type=int, default=30, help="回测 warmup bar 数")
    parser.add_argument("--period", choices=["is", "oos", "both"], default="both", help="回测时段: is(2020-2023), oos(2023-2024), both(默认)")
    parser.add_argument("--tqsdk", action="store_true", help="使用 TqSdk 实时数据源（替代本地缓存）")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径（tqsdk 模式需要日期区间）")
    args = parser.parse_args()

    symbols = args.symbols or _PREFERRED_SYMBOLS

    # ── TqSdk 模式：预加载数据 ──
    if args.tqsdk:
        from dotenv import load_dotenv
        load_dotenv()
        from core.engine.pybroker_data_source import create_hybrid_data_source
        from core.config import BacktestConfig

        config = BacktestConfig.from_yaml(args.config)
        logger.info("TqSdk 模式: 加载 %d 个品种数据 (日期 %s ~ %s)", len(symbols), config.full_start, config.test_start)
        ds = create_hybrid_data_source(symbols=symbols)
        for sym in symbols:
            try:
                sd = ds.for_symbol(sym)
                df = sd.to_pybroker_df()
                if len(df) >= 100:
                    _TQ_DFS[sym] = df
                    logger.info("  %s: %d 行", sym, len(df))
                else:
                    logger.warning("  %s: 数据不足 %d 行，跳过", sym, len(df))
            except Exception as exc:
                logger.warning("  %s: 加载失败 %s", sym, exc)
        logger.info("TqSdk 加载完成: %d/%d 品种可用", len(_TQ_DFS), len(symbols))
        symbols = [s for s in symbols if s in _TQ_DFS]
        if not symbols:
            logger.error("无可用品种，退出")
            sys.exit(1)
        # 把配置日期注入模块级变量，供 _run_single 使用
        global _FULL_START, _TEST_START
        _FULL_START = config.full_start
        _TEST_START = config.test_start

    # 构建退出参数字典（只传非 None 的）
    exit_params_override = {}
    for k in ["max_holding_days", "atr_stop_multiple", "global_risk_pct"]:
        v = getattr(args, k, None)
        if v is not None:
            exit_params_override[k] = v

    # 注入 CLI 参数覆盖
    global _WARMUP
    if exit_params_override:
        _DEFAULT_EXIT_PARAMS.update(exit_params_override)
    _WARMUP = args.warmup

    if args.mode == "benchmark":
        print(f"基准跑分: {len(symbols)} 品种 × 6 策略 × 时段={args.period}")
        df = mode_benchmark(symbols, period=args.period)
        print(f"\n完成 {len(df)} 条记录，保存至 output/cta_batch_results.parquet")
        # 自动展示汇总分析
        mode_summary()

    elif args.mode == "sweep":
        if not args.strategy:
            print("sweep 模式需要 --strategy")
            sys.exit(1)
        df = mode_sweep(args.strategy, symbols)
        print(f"\n完成 {len(df)} 条记录")
        mode_summary()

    elif args.mode == "summary":
        mode_summary()


if __name__ == "__main__":
    main()
