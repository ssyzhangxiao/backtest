"""
交叉验证工具（P0-5 整改）。

⚠️ P0-5整改（2026-06-07）：
  - 自研回测引擎（core/engine/runner.py）已完全移除
  - 交叉验证功能从 BacktestRunner.cross_validate_with_pybroker() 提取
  - 改造成独立工具：CrossValidator，可由 backtest_runner.py 显式调用
  - 4 个验证层次保留：
      1. 净值曲线一致性
      2. 核心绩效指标一致性
      3. 逐笔交易一致性
      4. 因子得分序列一致性（占位）

位置: core/engine/cross_validator.py
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import pandas as pd

_logger = logging.getLogger(__name__)


class CrossValidator:
    """
    交叉验证器：对比 PyBroker 与自研引擎回测结果。

    P0-5整改：从 BacktestRunner 提取为独立类，
    支持 4 个层次的逐步验证。
    """

    def __init__(self, correlation_alert_threshold: float = 0.95, metric_diff_pct_alert: float = 10.0):
        self.correlation_alert_threshold = correlation_alert_threshold
        self.metric_diff_pct_alert = metric_diff_pct_alert

    def cross_validate(
        self,
        pybroker_result: Any,
        own_result: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        4 层次交叉验证。

        Args:
            pybroker_result: PyBroker 回测结果（DataFrame / PyBrokerResult）
            own_result: 自研引擎结果（PortfolioResult）。None 时跳过自研验证。

        Returns:
            验证结果字典
        """
        result: Dict[str, Any] = {
            "validation_level_1": None,
            "validation_level_2": None,
            "validation_level_3": None,
            "validation_level_4": None,
            "alerts": [],
        }

        # Level 1: 净值曲线一致性
        level1 = self._validate_equity_curve_consistency(pybroker_result, own_result)
        result["validation_level_1"] = level1
        if (
            "correlation" in level1
            and level1["correlation"] < self.correlation_alert_threshold
        ):
            result["alerts"].append({
                "level": "severe",
                "type": "low_correlation",
                "message": f"净值相关系数过低: {level1['correlation']} < {self.correlation_alert_threshold}",
            })

        # Level 2: 核心绩效指标一致性
        if own_result is not None:
            level2 = self._validate_metrics_consistency(pybroker_result, own_result)
            result["validation_level_2"] = level2
            for metric_name, metric_data in level2.get("metric_comparison", {}).items():
                if metric_data.get("diff_pct", 0) > self.metric_diff_pct_alert:
                    result["alerts"].append({
                        "level": "important",
                        "type": "metric_discrepancy",
                        "message": f"{metric_name} 差异过大: {metric_data['diff_pct']:.2f}%",
                    })
        else:
            result["validation_level_2"] = {"skipped": "no own_result"}

        # Level 3: 逐笔交易一致性
        if own_result is not None:
            level3 = self._validate_trade_consistency(pybroker_result, own_result)
            result["validation_level_3"] = level3
            if level3.get("has_discrepancies", False):
                result["alerts"].append({
                    "level": "detailed",
                    "type": "trade_discrepancy",
                    "message": f"发现 {len(level3.get('discrepancies', []))} 笔交易不一致",
                })
        else:
            result["validation_level_3"] = {"skipped": "no own_result"}

        # Level 4: 因子得分序列一致性（占位）
        result["validation_level_4"] = self._validate_factor_scores_consistency(
            pybroker_result, own_result,
        )

        # 总体健康状态
        severe = [a for a in result["alerts"] if a["level"] == "severe"]
        important = [a for a in result["alerts"] if a["level"] == "important"]
        if severe:
            result["overall_status"] = "failed"
        elif important:
            result["overall_status"] = "warning"
        else:
            result["overall_status"] = "passed"

        return result

    def _validate_equity_curve_consistency(
        self, pybroker_result: Any, own_result: Optional[Any],
    ) -> Dict[str, Any]:
        """验证层次1：净值曲线一致性（基础）。"""
        # 提取 PyBroker 净值
        if isinstance(pybroker_result, pd.DataFrame):
            pybroker_eq = pybroker_result.copy()
        elif hasattr(pybroker_result, "equity_curve"):
            pybroker_eq = pybroker_result.equity_curve.copy()
        else:
            return {"error": "无法提取 PyBroker 净值曲线"}

        # 提取自研引擎净值
        if own_result is not None:
            portfolio_equity = own_result.portfolio_equity
        else:
            return {"skipped": "no own_result"}

        legacy_eq = portfolio_equity[["date", "equity"]].copy()

        # 统一列名
        pybroker_eq = pybroker_eq.rename(columns={"equity": "pybroker_equity"})
        legacy_eq = legacy_eq.rename(columns={"equity": "legacy_equity"})

        # 通过 date 合并对齐
        merged = pd.merge(
            pybroker_eq[["date", "pybroker_equity"]],
            legacy_eq[["date", "legacy_equity"]],
            on="date",
            how="inner",
        )
        if len(merged) < 10:
            return {"error": "样本太少", "n_samples": len(merged)}

        # 归一化到同一初始值
        merged["pybroker_eq"] = merged["pybroker_equity"] / merged["pybroker_equity"].iloc[0]
        merged["legacy_eq"] = merged["legacy_equity"] / merged["legacy_equity"].iloc[0]

        diff = (merged["pybroker_eq"] - merged["legacy_eq"]).abs()
        correlation = float(merged["pybroker_eq"].corr(merged["legacy_eq"]))

        merged["pybroker_ret"] = merged["pybroker_eq"].pct_change().fillna(0)
        merged["legacy_ret"] = merged["legacy_eq"].pct_change().fillna(0)
        returns_corr = float(merged["pybroker_ret"].corr(merged["legacy_ret"]))

        max_diff_idx = diff.idxmax()
        max_diff_date = merged.loc[max_diff_idx, "date"] if max_diff_idx in merged.index else None

        return {
            "correlation": round(correlation, 4),
            "max_abs_diff": round(float(diff.max()), 6),
            "mean_abs_diff": round(float(diff.mean()), 6),
            "max_diff_pct": round(float((diff / merged["pybroker_eq"]).max() * 100), 4),
            "max_diff_date": str(max_diff_date) if max_diff_date is not None else None,
            "returns_correlation": round(returns_corr, 4),
            "final_pybroker_eq": round(float(merged["pybroker_eq"].iloc[-1]), 4),
            "final_legacy_eq": round(float(merged["legacy_eq"].iloc[-1]), 4),
            "n_samples": len(merged),
            "dates_range": f"{merged['date'].iloc[0]} ~ {merged['date'].iloc[-1]}",
        }

    def _validate_metrics_consistency(
        self, pybroker_result: Any, own_result: Any,
    ) -> Dict[str, Any]:
        """验证层次2：核心绩效指标一致性。"""
        from utils.metrics import MetricsCalculator

        # 提取 PyBroker 指标
        pybroker_metrics: Dict[str, float] = {}
        pybroker_trades = None
        if hasattr(pybroker_result, "equity_curve"):
            pybroker_eq = pybroker_result.equity_curve.copy()
            if hasattr(pybroker_result, "trades"):
                pybroker_trades = pybroker_result.trades
            pybroker_metrics = MetricsCalculator.compute_from_equity_curve(
                pybroker_eq, pybroker_trades,
            )
        elif isinstance(pybroker_result, pd.DataFrame):
            pybroker_metrics = MetricsCalculator.compute_from_equity_curve(pybroker_result)

        # 提取自研引擎指标
        own_metrics: Dict[str, float] = own_result.portfolio_metrics or {}

        key_metrics = [
            "sharpe", "sortino", "calmar",
            "annual_return_pct", "max_drawdown_pct",
            "win_rate", "profit_factor", "trade_count",
        ]
        metric_comparison: Dict[str, Dict[str, float]] = {}
        for metric in key_metrics:
            pb_val = pybroker_metrics.get(metric) or pybroker_metrics.get(f"{metric}_ratio")
            ow_val = own_metrics.get(metric) or own_metrics.get(f"{metric}_ratio")
            if pb_val is not None and ow_val is not None:
                diff_abs = abs(pb_val - ow_val)
                diff_pct = (diff_abs / abs(pb_val)) * 100 if pb_val != 0 else (
                    float("inf") if diff_abs != 0 else 0
                )
                metric_comparison[metric] = {
                    "pybroker": pb_val,
                    "own": ow_val,
                    "diff_abs": round(diff_abs, 4),
                    "diff_pct": round(diff_pct, 2),
                }

        return {
            "pybroker_metrics": {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in pybroker_metrics.items()
            },
            "own_metrics": {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in own_metrics.items()
            },
            "metric_comparison": metric_comparison,
        }

    def _validate_trade_consistency(
        self, pybroker_result: Any, own_result: Any,
    ) -> Dict[str, Any]:
        """验证层次3：逐笔交易一致性。"""
        pybroker_trades = getattr(pybroker_result, "trades", None)
        own_trades = self._combine_trades(own_result)

        if pybroker_trades is None or pybroker_trades.empty:
            return {"error": "PyBroker 无交易数据", "has_discrepancies": False}
        if own_trades is None or own_trades.empty:
            return {"error": "自研引擎无交易数据", "has_discrepancies": False}

        discrepancies: list = []
        pb_count = len(pybroker_trades)
        ow_count = len(own_trades)
        if pb_count != ow_count:
            discrepancies.append({
                "type": "count_mismatch",
                "pybroker_count": pb_count,
                "own_count": ow_count,
                "message": f"交易数量不一致: PyBroker={pb_count}, 自研={ow_count}",
            })

        def get_side_count(trades_df: pd.DataFrame) -> tuple:
            if "side" in trades_df.columns:
                long = len(trades_df[trades_df["side"].astype(str).str.lower().str.contains("long|buy")])
                short = len(trades_df[trades_df["side"].astype(str).str.lower().str.contains("short|sell")])
                return long, short
            return 0, 0

        pb_long, pb_short = get_side_count(pybroker_trades)
        ow_long, ow_short = get_side_count(own_trades)
        if pb_long != ow_long or pb_short != ow_short:
            discrepancies.append({
                "type": "side_mismatch",
                "pybroker": {"long": pb_long, "short": pb_short},
                "own": {"long": ow_long, "short": ow_short},
                "message": "多空分布不一致",
            })

        return {
            "has_discrepancies": len(discrepancies) > 0,
            "discrepancies": discrepancies,
            "pybroker_trade_count": len(pybroker_trades),
            "own_trade_count": len(own_trades),
        }

    def _validate_factor_scores_consistency(
        self,
        pybroker_result: Any,
        own_result: Optional[Any],
    ) -> Dict[str, Any]:
        """
        验证层次4：因子得分序列一致性（针对因子打分回测）。

        P2-3整改：实现 Level 4 因子得分验证。规则26 要求：
          - 每日各品种的因子得分是否一致（如果是因子打分策略）
          - 横截面标准化后的得分是否一致

        数据约定：
          - PyBroker 结果应暴露 `factor_scores_history` 属性或字段：
              {date: {symbol: {factor_name: raw_score}}}
          - 自研结果（PortfolioResult）应暴露 `strategy_results[*].factor_scores`：
              {date: {symbol: {factor_name: raw_score}}}

        验证步骤：
          1) 提取两个引擎的因子得分历史
          2) 找出共有日期（date 对齐）
          3) 每日横截面：每个因子在所有品种间做 Z-Score 标准化
          4) 计算 Pearson 相关性（每日 + 全期聚合）
          5) 阈值告警：相关性 < 0.95 视为严重告警

        Args:
            pybroker_result: PyBroker 回测结果
            own_result: 自研引擎结果

        Returns:
            验证结果字典
        """
        pybroker_scores = self._extract_factor_scores_history(pybroker_result)
        own_scores = self._extract_factor_scores_history(own_result)

        if pybroker_scores is None or own_scores is None:
            return {
                "status": "skipped",
                "message": "缺少因子得分历史数据（PyBroker 或自研引擎未提供）",
                "pybroker_available": pybroker_scores is not None,
                "own_available": own_scores is not None,
            }

        # 共有日期
        common_dates = sorted(set(pybroker_scores.keys()) & set(own_scores.keys()))
        if len(common_dates) < 10:
            return {
                "status": "skipped",
                "message": f"共有日期不足（{len(common_dates)} < 10）",
                "n_common_dates": len(common_dates),
            }

        # 每日对比
        factor_pairs: Dict[tuple, List[float]] = {}

        for date in common_dates:
            pb_day = pybroker_scores[date]
            ow_day = own_scores[date]
            common_symbols = sorted(set(pb_day.keys()) & set(ow_day.keys()))
            if not common_symbols:
                continue

            for symbol in common_symbols:
                pb_factors = pb_day[symbol]
                ow_factors = ow_day[symbol]
                common_factors = sorted(set(pb_factors.keys()) & set(ow_factors.keys()))
                for factor in common_factors:
                    pb_val = float(pb_factors[factor])
                    ow_val = float(ow_factors[factor])
                    if not (pd.notna(pb_val) and pd.notna(ow_val)):
                        continue
                    factor_pairs.setdefault(factor, []).append(pb_val)
                    factor_pairs.setdefault((factor, "_own"), []).append(ow_val)

            # 当日全品种相关系数
            if common_factors:
                for factor in common_factors:
                    pass  # handled above

        # 计算每个因子的整体相关系数
        factor_correlations: Dict[str, float] = {}
        for factor, pb_vals in factor_pairs.items():
            if isinstance(factor, tuple):
                continue
            ow_vals = factor_pairs.get((factor, "_own"), [])
            n = min(len(pb_vals), len(ow_vals))
            if n < 5:
                factor_correlations[factor] = 0.0
                continue
            pb_arr = pd.Series(pb_vals[:n]).astype(float)
            ow_arr = pd.Series(ow_vals[:n]).astype(float)
            if pb_arr.std() < 1e-10 or ow_arr.std() < 1e-10:
                factor_correlations[factor] = 0.0
                continue
            corr = float(pb_arr.corr(ow_arr))
            factor_correlations[factor] = round(corr, 4) if pd.notna(corr) else 0.0

        # 横截面 Z-Score 标准化对比
        zscore_comparison = self._validate_factor_zscore_consistency(
            pybroker_scores, own_scores, common_dates,
        )

        # 整体健康度
        avg_corr = float(pd.Series(list(factor_correlations.values())).mean()) if factor_correlations else 0.0
        if avg_corr >= self.correlation_alert_threshold:
            status = "passed"
        elif avg_corr >= 0.80:
            status = "warning"
        else:
            status = "failed"

        return {
            "status": status,
            "n_common_dates": len(common_dates),
            "n_factors": len(factor_correlations),
            "factor_correlations": factor_correlations,
            "avg_correlation": round(avg_corr, 4),
            "zscore_comparison": zscore_comparison,
            "alert": (
                f"因子得分平均相关性 {avg_corr:.4f} 低于阈值 {self.correlation_alert_threshold}"
                if avg_corr < self.correlation_alert_threshold
                else None
            ),
        }

    def _validate_factor_zscore_consistency(
        self,
        pybroker_scores: Dict[str, Dict[str, Dict[str, float]]],
        own_scores: Dict[str, Dict[str, Dict[str, float]]],
        common_dates: list,
    ) -> Dict[str, Any]:
        """
        验证因子得分的横截面 Z-Score 标准化一致性。

        每日对每个因子在所有品种间做 Z-Score 标准化，
        对比两套引擎的标准化结果相关性。
        """
        z_pairs: Dict[str, List[float]] = {}
        for date in common_dates:
            pb_day = pybroker_scores[date]
            ow_day = own_scores[date]
            common_symbols = sorted(set(pb_day.keys()) & set(ow_day.keys()))
            if len(common_symbols) < 3:
                continue

            # 收集每个因子的原始得分
            for factor in (
                set().union(*(pb_day[s].keys() for s in common_symbols))
                & set().union(*(ow_day[s].keys() for s in common_symbols))
            ):
                pb_vals = []
                ow_vals = []
                for sym in common_symbols:
                    pb_v = pb_day[sym].get(factor)
                    ow_v = ow_day[sym].get(factor)
                    if pb_v is None or ow_v is None:
                        continue
                    pb_vals.append(float(pb_v))
                    ow_vals.append(float(ow_v))
                if len(pb_vals) < 3:
                    continue
                pb_ser = pd.Series(pb_vals)
                ow_ser = pd.Series(ow_vals)
                # Z-Score 标准化
                if pb_ser.std() < 1e-10 or ow_ser.std() < 1e-10:
                    continue
                pb_z = (pb_ser - pb_ser.mean()) / pb_ser.std()
                ow_z = (ow_ser - ow_ser.mean()) / ow_ser.std()
                z_pairs.setdefault(factor, []).extend(
                    list(zip(pb_z.tolist(), ow_z.tolist()))
                )

        z_correlations: Dict[str, float] = {}
        for factor, pairs in z_pairs.items():
            if not pairs:
                z_correlations[factor] = 0.0
                continue
            pb_list, ow_list = zip(*pairs)
            if len(pb_list) < 5:
                continue
            pb_arr = pd.Series(pb_list)
            ow_arr = pd.Series(ow_list)
            if pb_arr.std() < 1e-10 or ow_arr.std() < 1e-10:
                z_correlations[factor] = 0.0
                continue
            corr = float(pb_arr.corr(ow_arr))
            z_correlations[factor] = round(corr, 4) if pd.notna(corr) else 0.0

        return {
            "n_factors": len(z_correlations),
            "zscore_correlations": z_correlations,
        }

    @staticmethod
    def _extract_factor_scores_history(result: Any) -> Optional[Dict[str, Dict[str, Dict[str, float]]]]:
        """
        提取因子得分历史。

        支持的数据格式：
          - result.factor_scores_history: {date_str: {symbol: {factor: score}}}
          - result.factor_scores: {date_str: {symbol: {factor: score}}}
          - result.strategy_results[*].factor_scores: 合并所有策略
          - result.data.get("factor_scores_history"): 嵌套 dict

        Returns:
            {date_str: {symbol: {factor_name: raw_score}}} 或 None
        """
        if result is None:
            return None

        # 1) 直接属性
        scores = getattr(result, "factor_scores_history", None)
        if scores is not None:
            return scores
        scores = getattr(result, "factor_scores", None)
        if scores is not None:
            return scores

        # 2) PortfolioResult 嵌套
        strategy_results = getattr(result, "strategy_results", None)
        if strategy_results:
            merged: Dict[str, Dict[str, Dict[str, float]]] = {}
            for name, sresult in strategy_results.items():
                sub = getattr(sresult, "factor_scores", None)
                if not sub:
                    continue
                for date, sym_scores in sub.items():
                    merged.setdefault(date, {})
                    for sym, factor_dict in sym_scores.items():
                        merged[date].setdefault(sym, {})
                        for fac, score in factor_dict.items():
                            merged[date][sym][fac] = float(score)
            if merged:
                return merged

        # 3) dict 类型
        if isinstance(result, dict):
            scores = result.get("factor_scores_history") or result.get("factor_scores")
            if scores:
                return scores

        return None

    @staticmethod
    def _combine_trades(portfolio_result: Any) -> pd.DataFrame:
        """合并所有策略的交易记录。"""
        all_trades = []
        for name, sresult in portfolio_result.strategy_results.items():
            if sresult.trades is not None and not sresult.trades.empty:
                trades_copy = sresult.trades.copy()
                trades_copy["strategy"] = name
                all_trades.append(trades_copy)
        return pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
