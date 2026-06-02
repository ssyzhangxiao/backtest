"""
品种选择模块配置（规则14）。

品种选择是杠杆效应最大的改进，选对品种比优化参数更重要。
"""

from dataclasses import dataclass


@dataclass
class InstrumentModuleConfig:
    """品种选择模块配置。"""

    enabled: bool = False
    evaluator_volume_min: float = 10000
    evaluator_turnover_min: float = 1e8
    evaluator_hv_ideal_low: float = 0.10
    evaluator_hv_ideal_high: float = 0.30
    evaluator_adx_trend_threshold: float = 25.0
    evaluator_cost_max_pct: float = 0.002
    fitness_sharpe_weight: float = 0.3
    fitness_drawdown_weight: float = 0.25
    fitness_win_rate_weight: float = 0.2
    fitness_plr_weight: float = 0.15
    fitness_calmar_weight: float = 0.1
    fitness_min_trades: int = 10

    @staticmethod
    def from_yaml(raw: dict) -> "InstrumentModuleConfig":
        """从YAML原始字典解析品种选择模块配置。"""
        i = raw.get("instrument", {})
        ev = i.get("evaluator", {})
        fs = i.get("fitness_scorer", {})
        return InstrumentModuleConfig(
            enabled=bool(i.get("enabled", False)),
            evaluator_volume_min=float(ev.get("volume_min", 10000)),
            evaluator_turnover_min=float(ev.get("turnover_min", 1e8)),
            evaluator_hv_ideal_low=float(ev.get("hv_ideal_low", 0.10)),
            evaluator_hv_ideal_high=float(ev.get("hv_ideal_high", 0.30)),
            evaluator_adx_trend_threshold=float(ev.get("adx_trend_threshold", 25.0)),
            evaluator_cost_max_pct=float(ev.get("cost_max_pct", 0.002)),
            fitness_sharpe_weight=float(fs.get("sharpe_weight", 0.3)),
            fitness_drawdown_weight=float(fs.get("drawdown_weight", 0.25)),
            fitness_win_rate_weight=float(fs.get("win_rate_weight", 0.2)),
            fitness_plr_weight=float(fs.get("plr_weight", 0.15)),
            fitness_calmar_weight=float(fs.get("calmar_weight", 0.1)),
            fitness_min_trades=int(fs.get("min_trades", 10)),
        )
