# 规则1：引擎回退禁止，必须并行验证

**核心原则**：自研回测引擎不做 PyBroker 的回退方案，两套引擎并行运行用于交叉验证。

**具体规则**：
- PyBroker 执行失败必须直接抛出异常，不允许静默回退到自研引擎
- 自研引擎（runner.py）与 PyBroker 引擎同时运行，结果对比作为验证
- 并行运行后对比核心指标（Sharpe、最大回撤、年化收益），差异超 10% 发出警告
- 自研引擎仅用于交叉验证和边缘场景测试，不做主回测引擎

**涉及代码**：
- `core/engine/backtest_runner.py:run()`：PyBroker 不可用时直接抛 RuntimeError，不回退
- `core/engine/runner.py`：`BacktestRunner.cross_validate_with_pybroker()` 方法实现并行验证
- `core/config/backtest_config.py`：`cross_validate` 开关控制是否执行交叉验证
