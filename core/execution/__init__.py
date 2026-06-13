"""执行模块（execution/）— 回测执行器与 PyBroker 集成。

从 core/engine/ 迁移而来（2026-06-12）：
  - backtest_runner.py  ← 原 core/engine/backtest_runner.py
  - pybroker_executor.py ← 原 core/engine/pybroker_executor.py
  - _bootstrap.py       ← 原 core/engine/_bootstrap.py
  - _result_types.py    ← 原 core/engine/_result_types.py
  - _walkforward.py     ← 原 core/engine/_walkforward.py

公共接口：
  - PyBrokerBacktestRunner: PyBroker 主回测运行器
  - PyBrokerExecutorBuilder: PyBroker 执行器构建器（蓝图模式 + CTA 退出策略注入）
  - CTAExitPolicy / CTAExitConfig: CTA 四层退出策略（可注入 PyBrokerExecutorBuilder）
  - PyBrokerResult: 回测结果封装
  - WalkforwardResult: Walkforward 结果封装
"""

from core.execution.backtest_runner import PyBrokerBacktestRunner
from core.execution.pybroker_executor import PyBrokerExecutorBuilder
from core.execution.cta_exit_policy import CTAExitPolicy, CTAExitConfig
from core.execution._result_types import PyBrokerResult, WalkforwardResult

__all__ = [
    "PyBrokerBacktestRunner",
    "PyBrokerExecutorBuilder",
    "CTAExitPolicy",
    "CTAExitConfig",
    "PyBrokerResult",
    "WalkforwardResult",
]
