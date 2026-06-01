# 量化回测系统开发规范

## 规则1：引擎回退禁止，必须并行验证

**核心原则**：自研回测引擎不做 PyBroker 的回退方案，两套引擎并行运行用于交叉验证。

**具体规则**：
- PyBroker 执行失败必须直接抛出异常，不允许静默回退到自研引擎
- 自研引擎（runner.py）与 PyBroker 引擎同时运行，结果对比作为验证
- 并行运行后对比核心指标（Sharpe、最大回撤、年化收益），差异超 10% 发出警告
- 自研引擎仅用于交叉验证和边缘场景测试，不做主回测引擎

**涉及代码**：
- `core/engine/backtest_runner.py:run()`：移除 try/except 回退逻辑
- 新增 `scripts/cross_validate.py`：并行运行两套引擎对比结果

## 规则2：配置管理 — config.yaml 是单一数据源

**核心原则**：config.yaml 是一切配置的最终来源，BacktestConfig 必须与 yaml 完全同步。

**具体规则**：
- 删除 config.yaml 中的废弃字段（fusion_mode、regime_filter_enabled、strategy_switching 等）
- BacktestConfig 字段命名与 config.yaml 保持一致
- 新增配置项先在 yaml 定义，再在 BacktestConfig 中映射
- 运行 `BacktestConfig.from_yaml()` 后做字段完整性校验

## 规则3：废弃代码必须彻底清理

**核心原则**：已废弃的模块、字段、兼容层一律删除，不留后患。

**具体规则**：
- 废弃模块直接删除，不做 `@deprecated` 兼容别名
- 废弃字段从 config.yaml 和代码中同步删除
- 每轮重构完成后，全量 grep 检查废弃引用
- 根目录兼容层（如 config.py）在确认无引用后立即删除

## 规则4：风控类统一 — 一个系统只有一个风控

**核心原则**：不允许两个风控类并存，新代码用 RiskController，旧代码迁移。

**具体规则**：
- 核心风控逻辑在 RiskController 中实现
- RiskManager 作为 PyBroker 适配层，委托给 RiskController
- 新功能只加在 RiskController，RiskManager 不再扩展

## 规则5：策略注册统一

**核心原则**：只有一个策略注册入口，删除冗余注册机制。

**具体规则**：
- 统一使用 `core/strategy_registry.py` 的 StrategyRegistry
- 删除 `core/strategies/registry.py` 的轻量注册表
- 策略发现、参数获取、性能档案全部走统一入口

## 规则6：测试覆盖 — 关键路径必须有测试

**核心原则**：因子计算、调仓决策、风控触发必须有自动化测试覆盖。

**具体规则**：
- 新增策略必须有对应的因子计算正确性测试
- 新增风控规则必须有触发条件测试
- 调仓逻辑修改必须有调仓日判断测试
- 修改核心逻辑前先补测试，再重构

## 规则7：文件行数限制

**核心原则**：单文件不超过 500 行，超过必须拆分。

**具体规则**：
- 超过 500 行的文件在下次修改时强制拆分
- 拆分原则：按职责单一，每个模块只做一件事
- 当前需拆分的文件：暂无（broker_adapter.py 已拆分）

## 规则8：命名必须与功能一致

**核心原则**：类名、函数名必须准确反映当前功能，不允许名不副实。

**具体规则**：
- 类名变更后同步更新所有引用和文档
- 策略名称与策略文件一一对应，不允许别名
- 变量名包含计量单位（如 `_days`、`_pct`、`_bars`）

---

*最后更新：2026-06-01*
*相关知识文档：.trae/knowledges/20260601_003_lesson_engine-fallback-rule.md*