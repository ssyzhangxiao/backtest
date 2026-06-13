"""
策略执行器 — 已废弃，请使用 core.execution.pybroker_executor 或 core.risk_controller。

⚠️ 此文件从 core/engine/strategy_executor.py 移入（2026-06-13）。
  新代码请使用:
    - from core.execution.pybroker_executor import PyBrokerExecutorBuilder
    - from core.risk_controller import RiskController
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from core.risk_controller import RiskController, RiskConfig
from core.risk.composite_stop import CompositeStopResult

logger = logging.getLogger(__name__)

try:
    from pybroker import ExecContext
    PYBROKER_AVAILABLE = True
except ImportError:
    PYBROKER_AVAILABLE = False
    ExecContext = Any  # type: ignore


class RiskManagerAdapter:
    """
    PyBroker 风控适配层兼容层。

    ⚠️ P0/P1/P2 整改（2026-06-07 → 2026-06-13）：
      - 所有风控决策委托给 RiskController.check_composite_stop
      - 本类仅保留薄壳兼容
      - 新代码请直接使用 core.risk_controller.RiskController
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        controller: Optional[RiskController] = None,
    ) -> None:
        self.controller = controller or RiskController(
            config=RiskConfig(stop_loss_pct=getattr(config, "stop_loss_pct", 0.05))
        )

    def check_stop_loss(self, ctx: ExecContext) -> bool:
        """检查止损（委托 RiskController）。"""
        symbol = ctx.symbol
        close = getattr(ctx, "close", None)
        if close is None or len(close) == 0:
            return False
        current_price = float(close[-1])

        result = self.controller.check_composite_stop(
            symbol=symbol,
            direction="long",
            entry_price=current_price * 0.95,
            current_price=current_price,
            highest_since_entry=current_price,
            lowest_since_entry=current_price,
            entry_day=0,
            current_day=0,
        )
        return result.triggered
