# 规则4：风控类统一 — 一个系统只有一个风控

**核心原则**：不允许两个风控类并存，新代码用 RiskController，旧代码迁移。

**具体规则**：
- 核心风控逻辑在 RiskController 中实现
- RiskManager 作为 PyBroker 适配层，委托给 RiskController
- 新功能只加在 RiskController，RiskManager 不再扩展
