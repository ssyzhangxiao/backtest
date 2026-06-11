"""参数优化 — Walk-Forward 优化与稳健性评分。"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from core.optimizer._grid import generate_param_combinations

logger = logging.getLogger(__name__)


def compute_robustness(
    param_grid: Dict[str, List],
    results_df: pd.DataFrame,
    metric: str,
) -> Dict[int, float]:
    """计算每个参数组合的稳健性评分。

    robust_score = 0.7 * own_sharpe + 0.3 * neighborhood_avg_sharpe
    """
    param_cols = list(param_grid.keys())
    scores: Dict[int, float] = {}

    if metric not in results_df.columns:
        return {i: 0.0 for i in results_df.index}

    metric_vals = results_df[metric].values

    for idx in results_df.index:
        own_sharpe = metric_vals[idx] if not np.isnan(metric_vals[idx]) else 0.0
        neighbor_sharpes = []

        row = results_df.loc[idx]
        for col in param_cols:
            if col not in results_df.columns:
                continue
            current_val = row[col]
            for direction in [-1, 1]:
                param_values = param_grid.get(col, [])
                if not param_values:
                    continue
                try:
                    current_idx_pos = param_values.index(current_val)
                except (ValueError, AttributeError):
                    continue
                neighbor_idx = current_idx_pos + direction
                if 0 <= neighbor_idx < len(param_values):
                    neighbor_val = param_values[neighbor_idx]
                    neighbor_rows = results_df[results_df[col] == neighbor_val]
                    if not neighbor_rows.empty:
                        neighbor_mean = neighbor_rows[metric].mean()
                        if not np.isnan(neighbor_mean):
                            neighbor_sharpes.append(neighbor_mean)

        neighbor_avg = np.mean(neighbor_sharpes) if neighbor_sharpes else own_sharpe
        robust = 0.7 * own_sharpe + 0.3 * neighbor_avg
        scores[idx] = robust

    return scores


def walkforward_optimize(
    param_grid: Dict[str, List],
    metric: str,
    maximize: bool,
    data: pd.DataFrame,
    symbols: List[str],
    runner_factory: Callable,
    train_years: int = 2,
    test_years: int = 1,
) -> pd.DataFrame:
    """Walk-Forward 参数优化。

    将数据分为 N 个连续窗口（train_years 年训练 + test_years 年验证），
    每个窗口内做网格搜索，选择最优参数，在下一个窗口验证。
    """
    df = data.copy()
    df["date"] = pd.to_datetime(df["date"])
    all_dates = sorted(df["date"].unique())

    if len(all_dates) < 2:
        logger.warning("数据不足，无法执行 Walk-Forward 优化")
        return pd.DataFrame()

    start_date = all_dates[0]
    end_date = all_dates[-1]
    wf_results: List[Dict] = []
    current_start = start_date

    while current_start < end_date:
        train_end = current_start + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(years=test_years)

        if train_end > end_date:
            break

        train_data = df[(df["date"] >= current_start) & (df["date"] < train_end)]
        test_data = df[(df["date"] >= train_end) & (df["date"] < test_end)]

        if train_data.empty or test_data.empty:
            current_start = train_end
            continue

        combos = generate_param_combinations(param_grid)
        train_results: List[Dict] = []

        for params in combos:
            try:
                runner = runner_factory(list(param_grid.keys()))
                train_start_str = str(current_start.date())
                train_end_str = str(train_end.date())
                param_dict = {list(param_grid.keys())[0]: params}
                result = runner.run(
                    start_date=train_start_str,
                    end_date=train_end_str,
                    custom_params=param_dict,
                )
                m = result.metrics
                m = {
                    k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
                    for k, v in m.items()
                }
                m.update(params)
                train_results.append(m)
            except Exception as e:
                logger.debug("参数组合 %s 回测失败: %s", params, e)
                continue

        if not train_results:
            current_start = train_end
            continue

        train_df = pd.DataFrame(train_results)
        if metric not in train_df.columns:
            current_start = train_end
            continue

        robust_score = compute_robustness(param_grid, train_df, metric)
        train_df["robust_score"] = train_df.index.map(robust_score)
        best_idx = train_df["robust_score"].idxmax()
        best_params = train_df.loc[best_idx]
        param_cols = list(param_grid.keys())
        best_param_dict = {k: best_params[k] for k in param_cols if k in best_params.index}

        try:
            runner = runner_factory(list(param_grid.keys()))
            test_start_str = str(train_end.date())
            test_end_str = str(test_end.date())
            param_dict = {list(param_grid.keys())[0]: best_param_dict}
            test_result = runner.run(
                start_date=test_start_str,
                end_date=test_end_str,
                custom_params=param_dict,
            )
            test_m = test_result.metrics
            test_m = {
                k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
                for k, v in test_m.items()
            }
            record = {
                "train_start": str(current_start.date()),
                "train_end": str(train_end.date()),
                "test_start": str(train_end.date()),
                "test_end": str(test_end.date()),
                **best_param_dict,
                "train_sharpe": best_params.get(metric, 0),
                "train_robust_score": best_params.get("robust_score", 0),
                **test_m,
            }
            wf_results.append(record)
        except Exception as e:
            logger.debug("滚动窗口回测失败: %s", e)

        current_start = train_end

    if not wf_results:
        return pd.DataFrame()

    return pd.DataFrame(wf_results)


def select_robust_params(wf_df: pd.DataFrame, param_cols: List[str]) -> Dict:
    """从 Walk-Forward 结果中选择稳健参数（投票法）。"""
    if wf_df.empty:
        return {}

    param_combos = wf_df[param_cols].apply(lambda row: tuple(row.values), axis=1)
    combo_counts = param_combos.value_counts()

    if combo_counts.iloc[0] > 1:
        best_combo = combo_counts.index[0]
        return dict(zip(param_cols, best_combo))

    if "train_robust_score" in wf_df.columns:
        best_idx = wf_df["train_robust_score"].idxmax()
    elif "train_sharpe" in wf_df.columns:
        best_idx = wf_df["train_sharpe"].idxmax()
    else:
        best_idx = 0

    return {col: wf_df.loc[best_idx, col] for col in param_cols}
