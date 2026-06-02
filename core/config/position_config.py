"""
动态仓位模块配置（规则12）。

仓位调整单次幅度必须≤20%，否则引入新的不稳定性。
"""

from dataclasses import dataclass


@dataclass
class PositionModuleConfig:
    """动态仓位模块配置。"""

    enabled: bool = False
    rolling_sharpe_window: str = "3M"
    rolling_sharpe_risk_free_rate: float = 0.0
    dynamic_weight_max_adjustment: float = 0.20
    dynamic_weight_min_weight: float = 0.0
    dynamic_weight_max_weight: float = 1.0
    guard_sharpe_warning_threshold: float = 0.0
    guard_sharpe_danger_threshold: float = -0.5
    guard_consecutive_warning_days: int = 20
    guard_consecutive_danger_days: int = 10
    guard_drawdown_suspension_ratio: float = 1.5

    @staticmethod
    def from_yaml(raw: dict) -> "PositionModuleConfig":
        """从YAML原始字典解析动态仓位模块配置。"""
        p = raw.get("position", {})
        rs = p.get("rolling_sharpe", {})
        dw = p.get("dynamic_weight", {})
        sg = p.get("strategy_guard", {})
        return PositionModuleConfig(
            enabled=bool(p.get("enabled", False)),
            rolling_sharpe_window=str(rs.get("window", "3M")),
            rolling_sharpe_risk_free_rate=float(rs.get("risk_free_rate", 0.0)),
            dynamic_weight_max_adjustment=float(dw.get("max_adjustment", 0.20)),
            dynamic_weight_min_weight=float(dw.get("min_weight", 0.0)),
            dynamic_weight_max_weight=float(dw.get("max_weight", 1.0)),
            guard_sharpe_warning_threshold=float(
                sg.get("sharpe_warning_threshold", 0.0)
            ),
            guard_sharpe_danger_threshold=float(
                sg.get("sharpe_danger_threshold", -0.5)
            ),
            guard_consecutive_warning_days=int(sg.get("consecutive_warning_days", 20)),
            guard_consecutive_danger_days=int(sg.get("consecutive_danger_days", 10)),
            guard_drawdown_suspension_ratio=float(
                sg.get("drawdown_suspension_ratio", 1.5)
            ),
        )
