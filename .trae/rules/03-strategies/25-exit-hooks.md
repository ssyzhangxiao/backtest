# 规则25：策略退出钩子注册表 — 解耦退出逻辑与执行器

**核心原则**：各策略特定的退出条件通过注册表钩子机制实现，消除 `strategy_executor.py` 中的策略特定硬编码。

**具体规则**：
- 策略退出逻辑通过 `StrategyExitHookRegistry.register()` 注册，每个钩子包含：策略名、检查函数、退出原因
- `strategy_executor.py` 通过 `StrategyExitHookRegistry.check_exit()` 统一检查退出条件
- 收集所有已注册策略的指标值，传递给退出钩子做检查
- 新增策略的退出逻辑只需注册钩子，无需修改执行器核心代码
- 钩子检查异常时返回 False，不阻塞交易

**涉及代码**：
- `core/engine/strategy_indicators.py`：`StrategyExitHookRegistry` 类，管理退出钩子注册与检查
- `core/engine/strategy_executor.py`：调用 `check_exit()` 替代硬编码退出逻辑
