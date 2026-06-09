#!/usr/bin/env python3
"""
因子统计显著性测试脚本。

对 AlphaFutures24 全部24个因子逐一计算 IC、IR、Sharpe、胜率等指标，
输出排序后的统计汇总表，验证因子是否具备显著的统计优势。

规则9要求：IC > 0.03 且 IR > 0.5 的因子方可保留。

用法:
  python scripts/test_factors.py                    # 全部品种测试
  python scripts/test_factors.py --symbol SHFE.RB   # 单品种测试
  python scripts/test_factors.py --skip-winsorize   # 跳过后处理缩尾
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.factors import AlphaFutures24, AlphaFuturesConfig
from core.factors.operators import winsorize
from core.engine.pybroker_data_source import PyBrokerDataSource, create_hybrid_data_source
from runner.data.loader import DataLoader, load_raw_config, get_tqsdk_credentials
from runner.common.utils import is_valid_number, save_csv


# ── 因子元信息（类别 + 中文描述） ──
FACTOR_META: Dict[str, Dict[str, str]] = {
    "T_01": {"category": "趋势", "desc": "6日动量与日增仓乘积"},
    "T_02": {"category": "趋势", "desc": "12日动量与总持仓乘积"},
    "T_03": {"category": "趋势", "desc": "日度收益率与日增仓乘积"},
    "T_04": {"category": "趋势", "desc": "期限结构与增仓共振"},
    "T_05": {"category": "趋势", "desc": "6日条件增仓累积"},
    "R_01": {"category": "回归", "desc": "平滑日内涨跌与增仓率背离"},
    "R_02": {"category": "回归", "desc": "最高价与增仓率相关性"},
    "R_03": {"category": "回归", "desc": "收益率变化与平滑开盘增仓"},
    "R_04": {"category": "回归", "desc": "期限结构均值回复"},
    "R_05": {"category": "回归", "desc": "负相对持仓量"},
    "V_01": {"category": "波动率", "desc": "5日持仓量变化率"},
    "V_02": {"category": "波动率", "desc": "平滑日内波动率与5日增仓"},
    "V_03": {"category": "波动率", "desc": "日内振幅与增仓幅度"},
    "V_04": {"category": "波动率", "desc": "持仓量均线差率"},
    "M_01": {"category": "资金流", "desc": "6日日内多空力量与增仓"},
    "M_02": {"category": "资金流", "desc": "20日日内多空力量与增仓"},
    "M_03": {"category": "资金流", "desc": "20日条件增仓累积"},
    "M_04": {"category": "资金流", "desc": "期限结构驱动的资金流"},
    "M_05": {"category": "资金流", "desc": "持仓量MACD指标"},
    "H_01": {"category": "高阶复合", "desc": "条件性结构动量"},
    "H_02": {"category": "高阶复合", "desc": "7日价格变化与持仓衰减"},
    "H_03": {"category": "高阶复合", "desc": "相对持仓与反转时序"},
    "H_04": {"category": "高阶复合", "desc": "价格加速度与持仓排名"},
    "H_05": {"category": "高阶复合", "desc": "三重共振因子"},
}


def _compute_long_short_returns(
    factor_values: np.ndarray,
    close: np.ndarray,
    forward_period: int = 5,
) -> np.ndarray:
    """
    基于因子方向构建多空组合收益。

    因子值 > 0 → 做多；因子值 < 0 → 做空。
    使用前瞻收益作为实际收益。

    Args:
        factor_values: 因子值序列
        close: 收盘价序列
        forward_period: 前瞻周期

    Returns:
        多空组合日收益序列
    """
    forward_ret = (close - np.roll(close, forward_period)) / np.roll(close, forward_period)
    # 去掉NaN
    valid = ~(np.isnan(factor_values) | np.isnan(forward_ret))
    if valid.sum() < 30:
        return np.full_like(factor_values, np.nan)

    # 因子方向：正值做多，负值做空
    position = np.where(factor_values > 0, 1.0, -1.0)
    return position * forward_ret


def _compute_ic(
    factor_values: np.ndarray,
    forward_returns: np.ndarray,
) -> Dict[str, float]:
    """
    计算因子IC（Information Coefficient）及其统计量。

    IC = corr(factor, forward_return)，使用Pearson相关系数。

    Args:
        factor_values: 因子值序列
        forward_returns: 前瞻收益序列

    Returns:
        {mean_ic, std_ic, ir, ic_positive_pct, ic_series_length}
    """
    valid = ~(np.isnan(factor_values) | np.isnan(forward_returns))
    if valid.sum() < 30:
        return {"mean_ic": np.nan, "std_ic": np.nan, "ir": np.nan,
                "ic_positive_pct": np.nan, "ic_series_length": 0}

    fv = factor_values[valid]
    fr = forward_returns[valid]

    # 整体IC
    ic = np.corrcoef(fv, fr)[0, 1] if len(fv) > 2 else np.nan

    # 滚动IC（60天窗口）
    rolling_ic = pd.Series(fv, dtype=float).rolling(60, min_periods=30).corr(
        pd.Series(fr, dtype=float)
    ).dropna().values

    if len(rolling_ic) < 10:
        return {
            "mean_ic": round(ic, 6) if not np.isnan(ic) else np.nan,
            "std_ic": np.nan, "ir": np.nan,
            "ic_positive_pct": np.nan,
            "ic_series_length": len(rolling_ic),
        }

    mean_ic = np.nanmean(rolling_ic)
    std_ic = np.nanstd(rolling_ic)
    ir = mean_ic / std_ic if std_ic > 0 else 0.0
    ic_positive_pct = np.nanmean(rolling_ic > 0)

    return {
        "mean_ic": round(float(mean_ic), 6),
        "std_ic": round(float(std_ic), 6),
        "ir": round(float(ir), 4),
        "ic_positive_pct": round(float(ic_positive_pct), 4),
        "ic_series_length": len(rolling_ic),
    }


def _compute_sharpe_metrics(returns: np.ndarray) -> Dict[str, float]:
    """
    计算Sharpe比率、最大回撤、胜率等交易指标。

    Args:
        returns: 日收益序列

    Returns:
        {annual_ret, annual_vol, sharpe, max_drawdown, win_rate, avg_win, avg_loss}
    """
    valid = returns[~np.isnan(returns)]
    if len(valid) < 30:
        return {
            "annual_ret": np.nan, "annual_vol": np.nan,
            "sharpe": np.nan, "max_drawdown": np.nan,
            "win_rate": np.nan, "avg_win": np.nan, "avg_loss": np.nan,
        }

    ann_factor = 252 ** 0.5
    mean_ret = np.nanmean(valid)
    std_ret = np.nanstd(valid)
    annual_ret = mean_ret * 252
    annual_vol = std_ret * ann_factor
    sharpe = annual_ret / annual_vol if annual_vol > 0 else 0.0

    # 最大回撤
    cum_ret = np.cumprod(1 + valid)
    running_max = np.maximum.accumulate(cum_ret)
    drawdown = (cum_ret - running_max) / running_max
    max_dd = np.nanmin(drawdown)

    # 胜率
    wins = valid[valid > 0]
    losses = valid[valid < 0]
    win_rate = len(wins) / len(valid) if len(valid) > 0 else np.nan
    avg_win = np.nanmean(wins) if len(wins) > 0 else np.nan
    avg_loss = np.nanmean(losses) if len(losses) > 0 else np.nan

    return {
        "annual_ret": round(float(annual_ret), 4),
        "annual_vol": round(float(annual_vol), 4),
        "sharpe": round(float(sharpe), 4),
        "max_drawdown": round(float(max_dd), 4),
        "win_rate": round(float(win_rate), 4),
        "avg_win": round(float(avg_win), 6),
        "avg_loss": round(float(avg_loss), 6),
    }


def test_single_factor(
    factor_name: str,
    factor_values: np.ndarray,
    close: np.ndarray,
    forward_period: int = 5,
) -> Dict[str, Any]:
    """
    对单个因子进行完整统计测试。

    Args:
        factor_name: 因子名称
        factor_values: 因子值序列
        close: 收盘价序列
        forward_period: 前瞻周期

    Returns:
        统计指标字典
    """
    # 前瞻收益
    forward_ret = np.full_like(close, np.nan, dtype=float)
    forward_ret[:-forward_period] = (
        close[forward_period:] - close[:-forward_period]
    ) / close[:-forward_period]

    # IC统计
    ic_stats = _compute_ic(factor_values, forward_ret)

    # 多空组合收益
    ls_returns = _compute_long_short_returns(factor_values, close, forward_period)

    # Sharpe等指标
    perf = _compute_sharpe_metrics(ls_returns)

    meta = FACTOR_META.get(factor_name, {"category": "未知", "desc": ""})

    return {
        "factor": factor_name,
        "category": meta["category"],
        "desc": meta["desc"],
        **ic_stats,
        **perf,
        # 通过规则9标准的标记
        "pass_ic": (
            ic_stats.get("mean_ic", 0) is not None
            and not np.isnan(ic_stats.get("mean_ic", np.nan))
            and abs(ic_stats.get("mean_ic", 0)) > 0.03
        ),
        "pass_ir": (
            ic_stats.get("ir", 0) is not None
            and not np.isnan(ic_stats.get("ir", np.nan))
            and abs(ic_stats.get("ir", 0)) > 0.5
        ),
    }


def test_all_factors(
    ds: PyBrokerDataSource,
    symbols: List[str],
    start_date: str,
    end_date: str,
    do_winsorize: bool = True,
) -> pd.DataFrame:
    """
    对所有因子进行统计测试。

    Args:
        ds: 数据源
        symbols: 品种列表
        start_date: 开始日期
        end_date: 结束日期
        do_winsorize: 是否对因子值做缩尾

    Returns:
        排序后的统计结果 DataFrame
    """
    calc = AlphaFutures24(AlphaFuturesConfig())
    all_rows: List[Dict[str, Any]] = []

    for symbol in symbols:
        logger.info(f"测试品种: {symbol}")

        try:
            # 获取OHLCV数据
            ohlcv = ds.query(start_date, end_date, symbols=[symbol])
            if ohlcv is None or len(ohlcv) < 100:
                logger.warning(f"  {symbol}: 数据不足，跳过")
                continue

            ohlcv = ohlcv.sort_values("date").reset_index(drop=True)
            close = ohlcv["close"].values.astype(float)
            high = ohlcv["high"].values.astype(float)
            low = ohlcv["low"].values.astype(float)
            open_price = ohlcv["open"].values.astype(float)
            oi = ohlcv["open_interest"].values.astype(float) if "open_interest" in ohlcv.columns else None

            if oi is None:
                logger.warning(f"  {symbol}: 无持仓量数据，跳过")
                continue

            # 计算24个因子（无近远月价格，Carry因子置零）
            factors = calc.compute_all(
                close=close,
                open_price=open_price,
                high=high,
                low=low,
                open_interest=oi,
            )

            # 后处理
            if do_winsorize:
                factors = calc.post_process(factors, do_winsorize=True)

            # 逐个测试
            for fname, fvalues in factors.items():
                result = test_single_factor(fname, fvalues, close)
                result["symbol"] = symbol
                all_rows.append(result)

        except Exception as e:
            logger.error(f"  {symbol}: 因子计算失败 - {e}")

    if not all_rows:
        logger.error("无有效测试结果")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # 按 abs(mean_ic) 降序排序
    df["abs_ic"] = df["mean_ic"].abs()
    df = df.sort_values("abs_ic", ascending=False).reset_index(drop=True)
    df = df.drop(columns=["abs_ic"])

    return df


def print_summary(df: pd.DataFrame) -> None:
    """打印测试结果摘要。"""
    print("\n" + "=" * 100)
    print("  因子统计显著性测试结果")
    print("=" * 100)

    # 按因子汇总（多品种取平均）
    factor_summary = df.groupby("factor").agg({
        "mean_ic": "mean",
        "ir": "mean",
        "sharpe": "mean",
        "max_drawdown": "mean",
        "win_rate": "mean",
        "pass_ic": "mean",
        "pass_ir": "mean",
    }).reset_index()

    # 按 abs(mean_ic) 排序
    factor_summary["abs_ic"] = factor_summary["mean_ic"].abs()
    factor_summary = factor_summary.sort_values("abs_ic", ascending=False)

    # 打印表头
    print(f"\n{'因子':<8} {'类别':<8} {'mean_IC':>10} {'IR':>8} {'Sharpe':>8} "
          f"{'MaxDD':>8} {'胜率':>8} {'IC达标':>8} {'IR达标':>8}")
    print("-" * 100)

    for _, row in factor_summary.iterrows():
        meta = FACTOR_META.get(row["factor"], {"category": "未知"})
        ic_pass = "✓" if row["pass_ic"] > 0.5 else "✗"
        ir_pass = "✓" if row["pass_ir"] > 0.5 else "✗"
        print(
            f"{row['factor']:<8} {meta['category']:<8} "
            f"{row['mean_ic']:>10.4f} {row['ir']:>8.2f} {row['sharpe']:>8.2f} "
            f"{row['max_drawdown']:>8.2%} {row['win_rate']:>8.2%} "
            f"{ic_pass:>8} {ir_pass:>8}"
        )

    print("-" * 100)
    total = len(factor_summary)
    ic_pass_count = (factor_summary["pass_ic"] > 0.5).sum()
    ir_pass_count = (factor_summary["pass_ir"] > 0.5).sum()
    print(f"\n总计: {total} 个因子")
    print(f"IC达标(>0.03): {ic_pass_count}/{total}")
    print(f"IR达标(>0.5): {ir_pass_count}/{total}")
    print(f"IC+IR双达标: {((factor_summary['pass_ic'] > 0.5) & (factor_summary['pass_ir'] > 0.5)).sum()}/{total}")
    print("=" * 100)


def main() -> None:
    """主入口：加载数据 → 计算因子 → 统计测试 → 输出结果。"""
    parser = argparse.ArgumentParser(description="AlphaFutures24 因子统计测试")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--symbol", default=None, help="单品种测试（如 SHFE.RB）")
    parser.add_argument("--skip-winsorize", action="store_true", help="跳过缩尾后处理")
    parser.add_argument("--output", default="output_validation", help="输出目录")
    args = parser.parse_args()

    print("=" * 100)
    print("  AlphaFutures24 因子统计显著性测试")
    print("=" * 100)

    try:
        # 加载配置和数据
        raw_config = load_raw_config(args.config)
        phone, password = get_tqsdk_credentials(raw_config)

        ds = create_hybrid_data_source(
            symbols=raw_config.get("symbols", []),
            tqsdk_phone=phone,
            tqsdk_password=password,
            start_date=raw_config.get("backtest", {}).get("full_start_date", "2016-01-01"),
            end_date=raw_config.get("backtest", {}).get("full_end_date", "2025-12-31"),
        )
        ds.load()

        symbols = [args.symbol] if args.symbol else raw_config.get("symbols", [])
        start_date = raw_config.get("backtest", {}).get("full_start_date", "2016-01-01")
        end_date = raw_config.get("backtest", {}).get("full_end_date", "2025-12-31")

        logger.info(f"测试品种: {symbols}")
        logger.info(f"日期范围: {start_date} ~ {end_date}")
        logger.info(f"后处理缩尾: {'否' if args.skip_winsorize else '是'}")

        # 执行测试
        df = test_all_factors(
            ds, symbols, start_date, end_date,
            do_winsorize=not args.skip_winsorize,
        )

        if df.empty:
            logger.error("无有效测试结果")
            sys.exit(1)

        # 输出结果
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "factor_test_results.csv"
        save_csv(df, csv_path)
        logger.info(f"详细结果已保存到: {csv_path}")

        # 打印摘要
        print_summary(df)

    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()