"""
自适应参数模块配置（规则10）。

自适应参数必须有切换频率上限和回退机制，避免追逐近期噪音。
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class AdaptiveModuleConfig:
    """自适应参数模块配置。"""

    enabled: bool = False
    vol_monitor_hv_windows: Tuple = (20, 60, 120)
    vol_monitor_atr_window: int = 14
    vol_monitor_low_percentile: float = 0.25
    vol_monitor_high_percentile: float = 0.75
    vol_monitor_min_switch_interval_days: int = 21
    ema_base_window: int = 5
    ema_low_vol_window: int = 10
    ema_medium_vol_window: int = 5
    ema_high_vol_window: int = 3
    ema_max_step: int = 2
    atr_base_mult: float = 1.5
    atr_low_vol_mult: float = 0.5
    atr_medium_vol_mult: float = 1.5
    atr_high_vol_mult: float = 3.0
    param_logger_path: str = "logs/param_changes.jsonl"

    @staticmethod
    def from_yaml(raw: dict) -> "AdaptiveModuleConfig":
        """从YAML原始字典解析自适应参数模块配置。"""
        a = raw.get("adaptive", {})
        vm = a.get("vol_monitor", {})
        ema = a.get("ema_adapter", {})
        atr = a.get("atr_adapter", {})
        pl = a.get("param_logger", {})
        return AdaptiveModuleConfig(
            enabled=bool(a.get("enabled", False)),
            vol_monitor_hv_windows=tuple(vm.get("hv_windows", [20, 60, 120])),
            vol_monitor_atr_window=int(vm.get("atr_window", 14)),
            vol_monitor_low_percentile=float(vm.get("low_percentile", 0.25)),
            vol_monitor_high_percentile=float(vm.get("high_percentile", 0.75)),
            vol_monitor_min_switch_interval_days=int(
                vm.get("min_switch_interval_days", 21)
            ),
            ema_base_window=int(ema.get("base_window", 5)),
            ema_low_vol_window=int(ema.get("low_vol_window", 10)),
            ema_medium_vol_window=int(ema.get("medium_vol_window", 5)),
            ema_high_vol_window=int(ema.get("high_vol_window", 3)),
            ema_max_step=int(ema.get("max_step", 2)),
            atr_base_mult=float(atr.get("base_mult", 1.5)),
            atr_low_vol_mult=float(atr.get("low_vol_mult", 0.5)),
            atr_medium_vol_mult=float(atr.get("medium_vol_mult", 1.5)),
            atr_high_vol_mult=float(atr.get("high_vol_mult", 3.0)),
            param_logger_path=str(pl.get("log_path", "logs/param_changes.jsonl")),
        )
