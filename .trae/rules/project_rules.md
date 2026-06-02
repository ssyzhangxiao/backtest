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

## 规则9：因子开发规范 — IC 驱动，先验证后集成

**核心原则**：因子必须通过 IC 检验才能进入策略组合，无效因子不入库。

**具体规则**：
- 新因子必须实现 `FactorEvaluator` 接口，输出 IC、IR、多周期稳定性
- IC > 0.03 且 IR > 0.5 的因子方可保留，否则标记为待优化
- 因子间相关性 > 0.7 视为冗余，保留 IC 更高的因子
- 因子变换（对数/指数/幂函数/交叉项）必须对比原始因子的 IC 变化
- 最终因子集平均 IC > 0.04，最大互相关 < 0.6

**涉及代码**：
- `core/factors/factor_evaluator.py`：因子评估框架
- `core/factors/factor_selector.py`：因子筛选与去冗余
- `core/factors/factor_transformer.py`：因子变换与交叉项

## 规则10：自适应参数 — 切换频率受限，防止过拟合

**核心原则**：自适应参数必须有切换频率上限和回退机制，避免追逐近期噪音。

**具体规则**：
- 波动率 regime 切换频率不得超过每月 1 次
- 参数调整触发条件必须明确（波动率变化阈值、IC 衰减程度）
- EMA 窗口自适应范围：3~20 日，单次调整步长 ≤ 2 日
- ATR 倍数动态范围：0.5~3.0 倍，按波动率分位数分档
- 所有参数变更必须记录日志（时间、参数名、旧值→新值、触发原因）
- 自适应参数在样本外 3 个月内 Sharpe 不得劣于固定参数

**涉及代码**：
- `core/adaptive/vol_monitor.py`：波动率监测
- `core/adaptive/param_optimizer.py`：滚动窗口参数优化
- `core/adaptive/ema_adapter.py`：EMA 窗口自适应
- `core/adaptive/atr_adapter.py`：ATR 倍数动态调整
- `core/adaptive/param_logger.py`：参数变更日志

## 规则11：多时间框架 — 过滤优先，降逆势交易

**核心原则**：多时间框架的核心价值是过滤逆势交易，而非增加交易机会。

**具体规则**：
- 日频信号与周频趋势方向一致时才执行交易，不一致时跳过或减仓
- 周频权重 60%，日频权重 40%，不得随意调整
- 冲突场景占比应 < 40%，冲突时盈亏比 > 1.0
- 周频信号缓存，日频信号实时计算，周五对齐同步
- 过滤后交易次数减少 > 30% 且胜率提升 > 5% 方为有效

**涉及代码**：
- `core/multi_tf/trend_filter.py`：周频/月频趋势判断
- `core/multi_tf/signal_filter.py`：时间框架过滤规则
- `core/multi_tf/signal_sync.py`：信号延迟处理与同步

## 规则12：动态仓位 — 调幅受限，预警触发

**核心原则**：仓位调整单次幅度必须 ≤ 20%，否则引入新的不稳定性。

**具体规则**：
- 策略权重基于滚动 Sharpe 动态调整，归一化后权重之和 = 1
- 单次权重调整幅度 ≤ 20%，避免频繁大幅调整
- 风险预算分配考虑策略波动率和相关性
- 预警条件：滚动 Sharpe < 0 连续 20 日 → 降权 50%；< -0.5 连续 10 日 → 暂停
- 最大回撤超过历史最大回撤 1.5 倍 → 暂停策略

**涉及代码**：
- `core/position/rolling_sharpe.py`：滚动 Sharpe 计算
- `core/position/dynamic_weight.py`：策略权重动态调整
- `core/position/risk_budget.py`：风险预算分配
- `core/position/strategy_guard.py`：策略表现预警

## 规则13：止损策略 — 分层叠加，效果可量化

**核心原则**：止损优化应先验证追踪止损，再叠加时间止损，最后考虑复合止损。

**具体规则**：
- 追踪止损支持固定点数和 ATR 倍数两种模式
- 时间止损：持仓 N 个交易日（5~15 可配置）未达目标则强制平仓
- 复合止损优先级：价格止损 > 时间止损 > 波动率止损
- 波动率止损：ATR 突然放大 3 倍以上时触发紧急止损
- 止损效果必须量化：触发频率、平均盈亏、最大回撤改善、对 Sharpe 的影响

**涉及代码**：
- `core/risk/trailing_stop.py`：追踪止损
- `core/risk/time_stop.py`：时间止损
- `core/risk/composite_stop.py`：复合止损管理
- `core/risk/stop_analyzer.py`：止损效果分析

## 规则14：品种选择 — 适配评分，动态进出

**核心原则**：品种选择是杠杆效应最大的改进，选对品种比优化参数更重要。

**具体规则**：
- 品种特征评估包含波动率（HV/RV）、流动性（成交量/持仓量）、趋势性（ADX/持续周期）
- 每个策略有品种适配分数，不同策略权重不同
- 品种池每月初重新评估，准入/退出条件明确
- 组合内品种平均相关系数 < 0.5，超过时移除最高相关品种
- 品种准入条件：fitness_score > 阈值 且 流动性 > 最低标准

**涉及代码**：
- `core/instrument/instrument_evaluator.py`：品种特征评估
- `core/instrument/fitness_scorer.py`：品种适配性评分
- `core/instrument/pool_manager.py`：品种池动态调整
- `core/instrument/diversifier.py`：品种间风险分散

## 规则15：回测验证 — 多阶段验证，鲁棒性优先

**核心原则**：所有新功能必须通过样本外验证，Sharpe 不得低于旧版本 90%。

**具体规则**：
- 多阶段回测：样本内（2018-2020）→ 样本外（2021-2022）→ 实时模拟（2023 至今）
- 样本外 Sharpe 衰减 < 30%，最大回撤不超过样本内的 1.5 倍
- 蒙特卡洛模拟 1000 次 Bootstrap，Sharpe 95% 置信区间不含 0
- 参数敏感性分析：关键参数 ±20% 扰动，Sharpe 变化 < 15%
- 灰度发布：新功能通过 config.yaml 开关控制，默认关闭
- 每个里程碑完成后打 git tag，出问题时回滚到上一个 tag

**涉及代码**：
- `core/validation/monte_carlo.py`：蒙特卡洛模拟
- `core/validation/sensitivity.py`：参数敏感性分析
- `config.yaml`：灰度开关配置

## 规则16：模块目录结构 — 职责单一，接口清晰

**核心原则**：每个模块目录只做一件事，模块间通过明确接口交互。

**目录结构**：
```
core/
├── factors/           # 因子模块（评估+变换+筛选）
├── adaptive/          # 自适应参数模块（波动率+参数优化+适配器）
├── multi_tf/          # 多时间框架模块（趋势过滤+信号同步）
├── position/          # 动态仓位模块（权重+风险预算+预警）
├── risk/              # 止损优化模块（追踪+时间+复合止损）
├── instrument/        # 品种选择模块（评估+评分+池管理）
├── validation/        # 回测验证模块（蒙特卡洛+敏感性分析）
├── engine/            # 回测引擎（PyBroker+自研）
├── strategies/        # 策略实现
├── monitor/           # 策略监控（异常检测+绩效归因）
└── market_regime/     # 市场环境检测（辅助分析）

runner/                # 编排层（仅调用 core/ 和 utils/）
├── common/            # 通用工具
├── data/              # 数据加载与预处理
├── strategy/          # 策略选择与权重
├── backtest/          # 回测执行与实验
├── optimization/      # 参数优化
├── validation/        # 验证流程
└── report/            # 报告生成
```

**具体规则**：
- 新增模块必须在上述目录结构中，不得在 core/ 根目录新建文件
- runner/ 是编排层，不实现核心逻辑，仅调用 core/ 和 utils/
- 模块间依赖方向：strategies → factors → adaptive → engine，不得反向依赖
- 每个模块的 `__init__.py` 必须导出公共接口，隐藏内部实现
- 跨模块调用必须通过接口，不得直接访问其他模块的内部变量

## 规则17：不重复造轮子 — 优先调用公共系统

**核心原则**：runner/ 层仅做编排，核心逻辑必须委托给已有公共系统。

**公共系统清单（必须直接调用）**：
- **配置管理**：`core/config/` - 使用 `BacktestConfig.from_yaml()` 加载配置
- **数据加载**：`core/engine/pybroker_data_source.py` - 使用 `create_hybrid_data_source()`
- **指标计算**：`utils/indicators.py` - 使用 `compute_true_range()`, `compute_adx()` 等
- **绩效指标**：`utils/metrics.py` - 使用 `MetricsCalculator`
- **报告生成**：`core/report_builder.py` - 使用 `generate_report()`
- **绘图**：`utils/plots.py` - 使用 `PlotManager`
- **策略注册**：`core/strategy_registry.py` - 使用 `StrategyLibrary`

**具体规则**：
- 禁止在 run_ 脚本或 runner/ 模块中重复实现已有功能
- 所有工具函数优先检查 `utils/` 和 `core/` 中是否已存在
- 公共函数提取统一到 `runner/common/utils.py`
- 发现重复实现必须先提取再使用

## 规则18：Pipeline 编排器 — 声明式调用

**核心原则**：使用 `runner.pipeline.Pipeline` 类组合回测流程，实现链式调用。

**具体规则**：
- 根目录脚本（`run_*.py`）仅解析参数并调用 Pipeline
- 使用 `with_config(**overrides)` 进行配置热更新
- 通过 `load_data().run_backtest().optimize().validate().report()` 链式组合流程
- 新增实验/优化/验证方法只需在对应目录添加文件并注册到 Pipeline
- 使用 `is_healthy()` 检查状态健康度

## 规则19：依赖方向检查 — 禁止反向依赖

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

---

*最后更新：2026-06-02*
*相关知识文档：.trae/knowledges/20260602_001_workflow_strategy-enhancement-roadmap.md*
*相关知识文档：.trae/knowledges/20260602_002_workflow_runner-scripts-refactor-plan.md*
