# 规则19：依赖方向检查 — 禁止反向依赖

**核心原则**：确保架构分层清晰，禁止跨层反向依赖。

**依赖约束**：
- `runner/validation/` 不得依赖 `runner/optimization/`
- `runner/report/` 不得依赖 `runner/backtest/` 或 `runner/optimization/`
- `runner/strategy/` 不得依赖 `runner/optimization/`
- 所有 `runner/` 模块仅依赖 `core/`、`utils/` 或同层内其他模块

**具体规则**：
- CI 中加入 `pylint` 或自定义脚本检查依赖方向
- 发现反向依赖立即重构，确保分层清晰
- 模块间仅通过公共接口交互
