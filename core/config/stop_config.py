"""
止损优化模块配置（规则13）。

止损优化应先验证追踪止损，再叠加时间止损，最后考虑复合止损。
"""

from dataclasses import dataclass


@dataclass
class StopOptimizationConfig:
    """止损优化模块配置。"""

    enabled: bool = False
    trailing_mode: str = "pct"
    trailing_pct: float = 0.03
    trailing_atr_multiplier: float = 2.0
    time_stop_max_holding_days: int = 10
    time_stop_target_return: float = 0.01
    composite_fixed_stop_pct: float = 0.05

    @staticmethod
    def from_yaml(raw: dict) -> "StopOptimizationConfig":
        """从YAML原始字典解析止损优化模块配置。"""
        r = raw.get("risk", {})
        ts = r.get("trailing_stop", {})
        tms = r.get("time_stop", {})
        cs = r.get("composite_stop", {})
        return StopOptimizationConfig(
            enabled=bool(r.get("stop_optimization_enabled", False)),
            trailing_mode=str(ts.get("mode", "pct")),
            trailing_pct=float(ts.get("trail_pct", 0.03)),
            trailing_atr_multiplier=float(ts.get("atr_multiplier", 2.0)),
            time_stop_max_holding_days=int(tms.get("max_holding_days", 10)),
            time_stop_target_return=float(tms.get("target_return", 0.01)),
            composite_fixed_stop_pct=float(cs.get("fixed_stop_pct", 0.05)),
        )
