"""
策略表现预警系统。

当策略表现低于阈值时自动触发降权或暂停。

规则12要求：
  - 滚动Sharpe < 0 连续20日 → 降权50%
  - 滚动Sharpe < -0.5 连续10日 → 暂停策略
  - 最大回撤超过历史最大回撤1.5倍 → 暂停策略
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """预警级别。"""

    NORMAL = "normal"
    WARNING = "warning"
    DANGER = "danger"
    SUSPENDED = "suspended"


@dataclass
class StrategyAlert:
    """策略预警信息。"""

    strategy_name: str
    level: AlertLevel = AlertLevel.NORMAL
    rolling_sharpe: float = 0.0
    consecutive_bad_days: int = 0
    current_drawdown: float = 0.0
    max_historical_drawdown: float = 0.0
    message: str = ""


class StrategyGuard:
    """
    策略表现预警器。

    监控各策略的滚动Sharpe和最大回撤，
    根据规则12触发降权或暂停。

    用法:
        guard = StrategyGuard()
        alert = guard.check("ts_momentum", rolling_sharpe=-0.1, consecutive_bad=25)
        if alert.level == AlertLevel.DANGER:
            # 降权或暂停
    """

    def __init__(
        self,
        sharpe_warning_threshold: float = 0.0,
        sharpe_danger_threshold: float = -0.5,
        consecutive_warning_days: int = 20,
        consecutive_danger_days: int = 10,
        drawdown_suspension_ratio: float = 1.5,
    ):
        """
        初始化策略预警器。

        Args:
            sharpe_warning_threshold: Sharpe预警阈值（默认0）
            sharpe_danger_threshold: Sharpe危险阈值（默认-0.5）
            consecutive_warning_days: 连续低Sharpe预警天数（默认20）
            consecutive_danger_days: 连续低Sharpe危险天数（默认10）
            drawdown_suspension_ratio: 回撤暂停倍数（默认1.5倍历史最大回撤）
        """
        self.sharpe_warning_threshold = sharpe_warning_threshold
        self.sharpe_danger_threshold = sharpe_danger_threshold
        self.consecutive_warning_days = consecutive_warning_days
        self.consecutive_danger_days = consecutive_danger_days
        self.drawdown_suspension_ratio = drawdown_suspension_ratio

        # 策略状态追踪
        self._consecutive_bad: Dict[str, int] = {}
        self._max_drawdown: Dict[str, float] = {}
        self._suspended: Dict[str, bool] = {}

    def check(
        self,
        strategy_name: str,
        rolling_sharpe: float,
        current_drawdown: float = 0.0,
    ) -> StrategyAlert:
        """
        检查策略表现并返回预警。

        Args:
            strategy_name: 策略名称
            rolling_sharpe: 当前滚动Sharpe
            current_drawdown: 当前回撤（正数，如0.15表示15%）

        Returns:
            StrategyAlert 预警信息
        """
        # 已暂停的策略
        if self._suspended.get(strategy_name, False):
            return StrategyAlert(
                strategy_name=strategy_name,
                level=AlertLevel.SUSPENDED,
                rolling_sharpe=rolling_sharpe,
                current_drawdown=current_drawdown,
                max_historical_drawdown=self._max_drawdown.get(strategy_name, 0.0),
                message="策略已暂停",
            )

        # 更新连续低Sharpe天数
        if rolling_sharpe < self.sharpe_warning_threshold:
            self._consecutive_bad[strategy_name] = (
                self._consecutive_bad.get(strategy_name, 0) + 1
            )
        else:
            self._consecutive_bad[strategy_name] = 0

        consecutive = self._consecutive_bad.get(strategy_name, 0)

        # 更新历史最大回撤
        if current_drawdown > self._max_drawdown.get(strategy_name, 0.0):
            self._max_drawdown[strategy_name] = current_drawdown

        max_dd = self._max_drawdown.get(strategy_name, 0.0)

        # 判定预警级别
        level = AlertLevel.NORMAL
        message = ""

        # 回撤暂停检查
        if max_dd > 0 and current_drawdown > max_dd * self.drawdown_suspension_ratio:
            level = AlertLevel.SUSPENDED
            message = (
                f"回撤{current_drawdown:.1%}超过历史最大{max_dd:.1%}的"
                f"{self.drawdown_suspension_ratio}倍，暂停策略"
            )
            self._suspended[strategy_name] = True

        # Sharpe危险检查
        elif (
            rolling_sharpe < self.sharpe_danger_threshold
            and consecutive >= self.consecutive_danger_days
        ):
            level = AlertLevel.DANGER
            message = (
                f"Sharpe={rolling_sharpe:.4f}<{self.sharpe_danger_threshold} "
                f"连续{consecutive}天>={self.consecutive_danger_days}天，建议暂停"
            )

        # Sharpe预警检查
        elif consecutive >= self.consecutive_warning_days:
            level = AlertLevel.WARNING
            message = (
                f"Sharpe={rolling_sharpe:.4f}<{self.sharpe_warning_threshold} "
                f"连续{consecutive}天>={self.consecutive_warning_days}天，建议降权50%"
            )

        if level != AlertLevel.NORMAL:
            logger.warning(f"策略预警 [{strategy_name}]: {message}")

        return StrategyAlert(
            strategy_name=strategy_name,
            level=level,
            rolling_sharpe=rolling_sharpe,
            consecutive_bad_days=consecutive,
            current_drawdown=current_drawdown,
            max_historical_drawdown=max_dd,
            message=message,
        )

    def get_position_scale(self, strategy_name: str) -> float:
        """
        根据预警级别获取仓位比例。

        NORMAL=1.0, WARNING=0.5, DANGER=0.0, SUSPENDED=0.0

        Args:
            strategy_name: 策略名称

        Returns:
            仓位比例（0~1）
        """
        if self._suspended.get(strategy_name, False):
            return 0.0

        consecutive = self._consecutive_bad.get(strategy_name, 0)
        if consecutive >= self.consecutive_danger_days:
            return 0.0
        elif consecutive >= self.consecutive_warning_days:
            return 0.5
        else:
            return 1.0

    def resume(self, strategy_name: str) -> None:
        """
        恢复暂停的策略。

        Args:
            strategy_name: 策略名称
        """
        self._suspended[strategy_name] = False
        self._consecutive_bad[strategy_name] = 0
        logger.info(f"策略 [{strategy_name}] 已恢复")

    def reset(self):
        """重置所有策略状态。"""
        self._consecutive_bad.clear()
        self._max_drawdown.clear()
        self._suspended.clear()
