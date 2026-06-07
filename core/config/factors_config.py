"""
因子模块配置（规则9）。

因子必须通过IC检验才能进入策略组合，无效因子不入库。
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class FactorModuleConfig:
    """因子模块配置。"""

    enabled: bool = False
    ic_threshold: float = 0.03
    ir_threshold: float = 0.5
    max_correlation: float = 0.7
    target_avg_ic: float = 0.04
    target_max_correlation: float = 0.6
    capital_flow_enabled: bool = False
    capital_flow_oi_change_window: int = 10
    capital_flow_flow_window: int = 20
    capital_flow_divergence_window: int = 20
    term_structure_enabled: bool = False
    term_structure_basis_window: int = 20
    term_structure_roll_yield_smooth_window: int = 5  # 期限结构信号平滑窗口

    @staticmethod
    def from_yaml(raw: dict) -> "FactorModuleConfig":
        """从YAML原始字典解析因子模块配置。"""
        f = raw.get("factors", {})
        cf = f.get("capital_flow", {})
        ts = f.get("term_structure", {})
        return FactorModuleConfig(
            enabled=bool(f.get("enabled", False)),
            ic_threshold=float(f.get("ic_threshold", 0.03)),
            ir_threshold=float(f.get("ir_threshold", 0.5)),
            max_correlation=float(f.get("max_correlation", 0.7)),
            target_avg_ic=float(f.get("target_avg_ic", 0.04)),
            target_max_correlation=float(f.get("target_max_correlation", 0.6)),
            capital_flow_enabled=bool(cf.get("enabled", False)),
            capital_flow_oi_change_window=int(cf.get("oi_change_window", 10)),
            capital_flow_flow_window=int(cf.get("flow_window", 20)),
            capital_flow_divergence_window=int(cf.get("divergence_window", 20)),
            term_structure_enabled=bool(ts.get("enabled", False)),
            term_structure_basis_window=int(ts.get("basis_window", 20)),
            term_structure_roll_yield_smooth_window=int(
                ts.get("roll_yield_smooth_window", 5)
            ),
        )
