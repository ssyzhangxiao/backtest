# 规则2：配置管理 — config.yaml 是单一数据源

**核心原则**：config.yaml 是一切配置的最终来源，BacktestConfig 必须与 yaml 完全同步。

**具体规则**：
- 删除 config.yaml 中的废弃字段（fusion_mode、regime_filter_enabled、strategy_switching 等）
- BacktestConfig 字段命名与 config.yaml 保持一致
- 新增配置项先在 yaml 定义，再在 BacktestConfig 中映射
- 运行 `BacktestConfig.from_yaml()` 后做字段完整性校验
