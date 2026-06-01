# 架构审查与修改建议

## 摘要

基于 README.md 和代码审核，对 CTA + Alpha101 多因子回测系统提出架构修改建议。当前系统已完成从"环境感知策略切换"到"因子打分调仓"的范式转换，但存在模块耦合、废弃代码残留、文档不一致等问题。

## 背景

系统从 v2（市场环境判断→策略切换）演进到 v3（因子打分→调仓），核心逻辑已变更但部分旧代码和文档未同步清理。

## 当前架构问题

### 1. 废弃模块残留

| 模块 | 状态 | 建议 |
|------|------|------|
| `core/environment.py` | 已标记废弃但仍被 `utils/session_state.py` 引用 | 短期保留兼容，长期迁移 session_state 到 MarketRegimeDetector |
| `core/market_regime/v3_regime.py` | V3RegimeParamManager/V3RegimeAwareRunner 仍被 run_validation.py 延迟导入 | 保留为辅助分析工具，但应从主流程入口脚本中剥离 |
| `BacktestConfig.fusion_mode` | 兼容旧接口，默认 True 但无实际作用 | 下个版本移除 |
| `BacktestConfig.regime_filter_enabled` | 兼容旧接口，默认 False | 下个版本移除 |

### 2. 引擎层职责不清

| 文件 | 当前职责 | 问题 | 建议 |
|------|---------|------|------|
| `switch_engine.py` | 因子打分调仓引擎 | 类名 StrategySwitchEngine 与实际功能不符 | 重命名为 ScoringEngine 或 FactorScoringEngine |
| `broker_adapter.py` | PyBroker 执行适配器 | 包含过多业务逻辑（因子得分计算、调仓决策） | 将业务逻辑下沉到 ScoringEngine，adapter 仅做执行层 |
| `runner.py` | 自研回测引擎 | 与 broker_adapter 职责重叠 | 明确分工：runner 负责编排，adapter 负责执行 |

### 3. 配置分散

- `config.yaml` 和 `BacktestConfig` 存在字段不一致
- `DEFAULT_FACTOR_WEIGHTS` 在 config.py 和 portfolio.py 中各有一份
- 调仓周期在 config.yaml 中是字符串 `"3d"`，在 BacktestConfig 中是 int `3`

**建议**：统一为 config.yaml 单一数据源，BacktestConfig 从 yaml 自动构建。

### 4. 策略注册机制冗余

- `registry.py` 维护 STRATEGY_REGISTRY 字典
- `strategy_library.py` 维护 StrategyLibrary + StrategyProfile
- 两者功能重叠，registry 更轻量但 library 更完整

**建议**：合并为统一的 StrategyRegistry，支持策略发现、元数据、权重管理。

### 5. 数据层耦合

- `DataLoader` 直接依赖 TqSdk（import 在方法内部）
- 缓存使用 pickle（安全风险已修复，但格式不够透明）
- 无数据版本管理

**建议**：
- 引入 DataProvider 抽象层，支持多数据源
- 缓存改用 Parquet 格式（跨平台、可读、压缩率高）
- 增加数据校验（缺失值比例、时间连续性）

### 6. 测试覆盖不足

- 无单元测试目录
- 验证脚本（run_validation.py）承担了部分集成测试职责
- 关键路径（因子计算、调仓决策、风控触发）无自动化测试

**建议**：
- 新增 `tests/` 目录，按模块组织
- 优先覆盖：因子计算正确性、调仓日判断、风控触发条件
- 使用 pytest + fixture 管理测试数据

## 推荐的架构演进路线

### Phase 1: 清理（当前阶段）
- ✅ 移除废弃字段和冗余代码
- ✅ 修复安全问题（pickle、print→logger）
- ✅ 统一配置格式
- 🔄 更新 README 与代码一致

### Phase 2: 解耦
- 重命名 StrategySwitchEngine → FactorScoringEngine
- 抽取 DataProvider 接口
- 合并策略注册机制
- 缓存格式迁移（pickle → parquet）

### Phase 3: 质量保障
- 建立单元测试体系
- CI/CD 集成（lint + typecheck + test）
- 性能基准测试

### Phase 4: 扩展
- 滚动IC加权替代固定权重
- 因子衰减监控
- 实盘信号生成接口

## 要点

1. 当前最紧迫的是 Phase 2 的引擎重命名和业务逻辑下沉，避免新开发者被旧类名误导
2. 配置分散是 bug 温床，应尽早统一为 yaml 单一数据源
3. 测试覆盖不足是最大技术债，任何重构都应先补测试
4. environment.py 的废弃迁移需要同步更新 utils/session_state.py

## 相关文件

- [config.py](file:///Users/luojiutian/Documents/backtest/core/config.py)
- [switch_engine.py](file:///Users/luojiutian/Documents/backtest/core/engine/switch_engine.py)
- [broker_adapter.py](file:///Users/luojiutian/Documents/backtest/core/engine/broker_adapter.py)
- [portfolio.py](file:///Users/luojiutian/Documents/backtest/core/portfolio.py)
- [README.md](file:///Users/luojiutian/Documents/backtest/README.md)
