"""执行模块（execution/）— 回测执行器与 PyBroker 集成。

从 core/engine/ 迁移而来（2026-06-12）：
  - backtest_runner.py  ← 原 core/engine/backtest_runner.py
  - pybroker_executor.py ← 原 core/engine/pybroker_executor.py
  - _bootstrap.py       ← 原 core/engine/_bootstrap.py
  - _result_types.py    ← 原 core/engine/_result_types.py
  - _walkforward.py     ← 原 core/engine/_walkforward.py

新增（2026-06-13）：
  - factor_pool.py      — 统一因子池（24α + 6CTA 信号统一入口）
  - signal_abstraction.py — 信号抽象层（横截面/CTA/混合三种模式提取）

架构示意：
  OHLCV → UnifiedFactorPool → DataFrame(11+1 列)
                             → SignalAbstractionLayer
                                ├── get_cross_sectional_signals()  ← 5 子策略
                                ├── get_cta_signals()              ← 6 CTA 策略
                                ├── get_cta_composite_signal()     ← 加权合成 1 值
                                └── get_hybrid_signal()            ← 横截面 × CTA 混合

公共接口：
  - PyBrokerBacktestRunner: PyBroker 主回测运行器
  - PyBrokerExecutorBuilder: PyBroker 执行器构建器（蓝图模式 + CTA 退出策略注入）
  - CTAExitPolicy / CTAExitConfig: CTA 四层退出策略（可注入 PyBrokerExecutorBuilder）
  - UnifiedFactorPool: 统一因子池（所有信号源单入口）
  - SignalAbstractionLayer: 信号抽象层（三种模式提取）
  - SignalMode: 信号模式枚举
  - PyBrokerResult: 回测结果封装
  - WalkforwardResult: Walkforward 结果封装
"""

from core.execution.backtest_runner import PyBrokerBacktestRunner
from core.execution.pybroker_executor import PyBrokerExecutorBuilder
from core.execution.cta_exit_policy import CTAExitPolicy, CTAExitConfig
from core.execution.factor_pool import UnifiedFactorPool, ALL_SIGNAL_NAMES, CTA_SIGNAL_NAMES
from core.execution.signal_abstraction import (
    SignalAbstractionLayer,
    SignalMode,
    DEFAULT_CTA_WEIGHTS,
)
from core.execution._result_types import PyBrokerResult, WalkforwardResult

__all__ = [
    "PyBrokerBacktestRunner",
    "PyBrokerExecutorBuilder",
    "CTAExitPolicy",
    "CTAExitConfig",
    "UnifiedFactorPool",
    "SignalAbstractionLayer",
    "SignalMode",
    "ALL_SIGNAL_NAMES",
    "CTA_SIGNAL_NAMES",
    "DEFAULT_CTA_WEIGHTS",
    "PyBrokerResult",
    "WalkforwardResult",
]
