# 规则24：策略指标注册表 — 解耦指标计算与回测引擎

**核心原则**：策略指标通过注册表机制集中管理，消除 `backtest_runner.py` 中的硬编码指标构建逻辑。

**具体规则**：
- 所有策略指标必须通过 `StrategyIndicatorRegistry.register()` 注册
- 注册内容包括：指标构建函数、指标名列表、指标名→因子名映射
- `backtest_runner.py` 通过 `StrategyIndicatorRegistry.build_all(sub_params)` 动态构建指标，不硬编码任何指标计算逻辑
- `switch_engine.py` 通过 `StrategyIndicatorRegistry.get_indicator_to_factor_map()` 动态获取映射关系，不硬编码 `indicator_map`
- 新增因子只需注册指标构建函数，无需修改回测引擎和打分引擎

**涉及代码**：
- `core/engine/strategy_indicators.py`：`StrategyIndicatorRegistry` 类，管理指标注册与构建
- `core/engine/backtest_runner.py`：调用 `build_all()` 替代硬编码
- `core/engine/switch_engine.py`：调用 `get_indicator_to_factor_map()` 替代硬编码
