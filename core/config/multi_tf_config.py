"""
多时间框架模块配置（规则11）。

多时间框架的核心价值是过滤逆势交易，而非增加交易机会。
"""

from dataclasses import dataclass


@dataclass
class MultiTFModuleConfig:
    """多时间框架模块配置。"""

    enabled: bool = False
    weekly_weight: float = 0.6
    daily_weight: float = 0.4
    conflict_position_scale: float = 0.33
    adx_period: int = 14
    adx_threshold: float = 25.0
    ma_short: int = 5
    ma_medium: int = 20
    ma_long: int = 60

    @staticmethod
    def from_yaml(raw: dict) -> "MultiTFModuleConfig":
        """从YAML原始字典解析多时间框架模块配置。"""
        m = raw.get("multi_tf", {})
        tf = m.get("trend_filter", {})
        return MultiTFModuleConfig(
            enabled=bool(m.get("enabled", False)),
            weekly_weight=float(m.get("weekly_weight", 0.6)),
            daily_weight=float(m.get("daily_weight", 0.4)),
            conflict_position_scale=float(m.get("conflict_position_scale", 0.33)),
            adx_period=int(tf.get("adx_period", 14)),
            adx_threshold=float(tf.get("adx_threshold", 25.0)),
            ma_short=int(tf.get("ma_short", 5)),
            ma_medium=int(tf.get("ma_medium", 20)),
            ma_long=int(tf.get("ma_long", 60)),
        )
