"""
回测验证模块配置（规则15）。

所有新功能必须通过样本外验证，Sharpe不得低于旧版本90%。
"""

from dataclasses import dataclass


@dataclass
class ValidationModuleConfig:
    """回测验证模块配置。"""

    enabled: bool = False
    monte_carlo_n_simulations: int = 1000
    monte_carlo_random_seed: int = 42
    sensitivity_perturbation: float = 0.20
    sensitivity_high_threshold: float = 0.30

    @staticmethod
    def from_yaml(raw: dict) -> "ValidationModuleConfig":
        """从YAML原始字典解析回测验证模块配置。"""
        v = raw.get("validation", {})
        mc = v.get("monte_carlo", {})
        sa = v.get("sensitivity", {})
        return ValidationModuleConfig(
            enabled=bool(v.get("enabled", False)),
            monte_carlo_n_simulations=int(mc.get("n_simulations", 1000)),
            monte_carlo_random_seed=int(mc.get("random_seed", 42)),
            sensitivity_perturbation=float(sa.get("perturbation", 0.20)),
            sensitivity_high_threshold=float(
                sa.get("high_sensitivity_threshold", 0.30)
            ),
        )
