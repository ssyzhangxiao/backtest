# 规则18：Pipeline 编排器 — 声明式调用

**核心原则**：使用 `runner.pipeline.Pipeline` 类组合回测流程，实现链式调用。

**具体规则**：
- 根目录脚本（`run_*.py`）仅解析参数并调用 Pipeline
- 使用 `with_config(**overrides)` 进行配置热更新
- 通过 `load_data().run_backtest().optimize().validate().report()` 链式组合流程
- 新增实验/优化/验证方法只需在对应目录添加文件并注册到 Pipeline
- 使用 `is_healthy()` 检查状态健康度
