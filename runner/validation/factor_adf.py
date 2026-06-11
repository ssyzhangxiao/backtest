"""
因子 ADF 平稳性检验。

对所有已注册因子值进行 ADF（Augmented Dickey-Fuller）单位根检验，
判断因子是否平稳。平稳性是因子可建模、可比较的前提：
- p-value < 0.05：拒绝单位根，因子平稳
- p-value >= 0.05：未拒绝单位根，因子非平稳，需做差分或对数变换

委托：
- `core.factors.alpha_futures.factor_engine.FactorEngine`：计算因子值
- `statsmodels.tsa.stattools.adfuller`：ADF 检验（轻量依赖）

通过标准（规则 28 阶段 A 新增）：
- p-value < 0.05：平稳
- p-value >= 0.05：非平稳，建议 diff(1) 或 log 变换后重测

输出：output/validate/factor_adf.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

from core.config import BacktestConfig
from core.factors import AlphaFuturesConfig
from core.factors.alpha_futures.factor_engine import FactorEngine
from runner.common.utils import save_csv

# ADF 显著性阈值（拒绝单位根的标准）
ADF_PVALUE_THRESHOLD = 0.05

# 最大滞后阶数（避免样本过短时报错）
ADF_MAXLAG = 10

# 输出文件名
OUTPUT_FILENAME = "factor_adf.csv"


def _run_adf_single(series: pd.Series) -> Dict[str, float]:
    """
    对单条因子序列运行 ADF 检验。

    Args:
        series: 因子值序列（已 dropna）

    Returns:
        {adf_stat, p_value, lags_used, n_obs, is_stationary} 字典
    """
    from statsmodels.tsa.stattools import adfuller

    clean = series.dropna()
    if len(clean) < 30:
        return {
            "adf_stat": np.nan,
            "p_value": np.nan,
            "lags_used": 0,
            "n_obs": len(clean),
            "is_stationary": False,
        }
    try:
        # autolag="AIC" 自动选择滞后阶数；regression="c" 仅截距
        result = adfuller(clean.values, maxlag=ADF_MAXLAG, autolag="AIC", regression="c")
        adf_stat, p_value, lags_used, n_obs = result[0], result[1], result[2], result[3]
        return {
            "adf_stat": float(adf_stat),
            "p_value": float(p_value),
            "lags_used": int(lags_used),
            "n_obs": int(n_obs),
            "is_stationary": bool(p_value < ADF_PVALUE_THRESHOLD),
        }
    except Exception as e:  # 数值不稳定时 adfuller 抛 LinAlgError
        logger.warning(f"    ADF 检验失败: {e}")
        return {
            "adf_stat": np.nan,
            "p_value": np.nan,
            "lags_used": 0,
            "n_obs": len(clean),
            "is_stationary": False,
        }


def factor_adf_validation(
    data_source,
    config: BacktestConfig,
    lib=None,
    output_dir: Path = Path("output/validate"),
    best_params: Optional[Dict[str, Dict[str, Any]]] = None,
    cross_sectional: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    因子 ADF 平稳性检验（按品种 × 因子输出）。

    对 config.symbols 内每个品种、每个已注册因子跑 ADF 检验，
    汇总成 (symbol, factor, adf_stat, p_value, is_stationary) 表。

    Args:
        data_source: PyBrokerDataSource
        config: 回测配置
        lib: 策略库（未使用，保持接口一致）
        output_dir: 输出目录
        best_params: 最优参数（未使用）
        cross_sectional: 是否横截面（未使用）
        **kwargs: forward_period=5（未使用，ADF 不需要前瞻收益）

    Returns:
        {
            "output_path": Path,
            "n_factors": int,
            "n_symbols": int,
            "stationary_rate": float,
            "results": {symbol: DataFrame}
        }
    """
    logger.info("=" * 60)
    logger.info(f"因子 ADF 平稳性检验（p < {ADF_PVALUE_THRESHOLD} 视为平稳）")
    logger.info("=" * 60)

    af_cfg = AlphaFuturesConfig()
    engine = FactorEngine(af_cfg)
    symbols = config.symbols

    all_rows: list[Dict[str, Any]] = []
    per_symbol: Dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        try:
            ohlcv = data_source.query(
                config.train_start, config.test_end, symbols=[symbol]
            )
            if ohlcv is None or len(ohlcv) < 100:
                logger.warning(f"  {symbol}: 数据不足，跳过")
                continue
            ohlcv = ohlcv.sort_values("date").reset_index(drop=True)
            raw_data = {
                "close": ohlcv["close"].values.astype(float),
                "open_price": ohlcv["open"].values.astype(float),
                "high": ohlcv["high"].values.astype(float),
                "low": ohlcv["low"].values.astype(float),
                "open_interest": (
                    ohlcv["open_interest"].values.astype(float)
                    if "open_interest" in ohlcv.columns
                    else np.zeros(len(ohlcv))
                ),
                "volume": ohlcv["volume"].values.astype(float) if "volume" in ohlcv.columns else None,
            }
            factor_scores = engine.compute_all(raw_data)
        except Exception as e:
            logger.warning(f"  {symbol}: 因子计算失败: {e}")
            continue

        rows: list[Dict[str, Any]] = []
        for fname, fvalues in factor_scores.items():
            series = pd.Series(fvalues)
            adf_res = _run_adf_single(series)
            adf_res["symbol"] = symbol
            adf_res["factor"] = fname
            rows.append(adf_res)
        df = pd.DataFrame(rows)
        per_symbol[symbol] = df
        all_rows.extend(rows)

        n_pass = int(df["is_stationary"].sum())
        n_total = len(df)
        logger.info(f"  {symbol}: 平稳 {n_pass}/{n_total} ({100*n_pass/max(n_total,1):.1f}%)")

    if not all_rows:
        logger.warning("  无有效 ADF 结果")
        return {
            "output_path": None,
            "n_factors": 0,
            "n_symbols": 0,
            "stationary_rate": 0.0,
            "results": {},
        }

    full_df = pd.DataFrame(all_rows)[
        ["symbol", "factor", "adf_stat", "p_value", "lags_used", "n_obs", "is_stationary"]
    ]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / OUTPUT_FILENAME
    save_csv(full_df, out_path)
    logger.info(f"  ADF 结果已保存: {out_path}")

    n_total = len(full_df)
    n_pass = int(full_df["is_stationary"].sum())
    return {
        "output_path": out_path,
        "n_factors": int(full_df["factor"].nunique()),
        "n_symbols": int(full_df["symbol"].nunique()),
        "stationary_rate": n_pass / max(n_total, 1),
        "results": per_symbol,
    }


__all__ = ["factor_adf_validation", "ADF_PVALUE_THRESHOLD", "OUTPUT_FILENAME"]
