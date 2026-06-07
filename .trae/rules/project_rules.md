# 量化回测系统开发规范

---

> **注意**：本文件由 `merge-rules.py` 自动生成，请勿直接编辑。
> 如需修改规则，请编辑对应分类目录下的规则文件，然后重新运行合并脚本。

---

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


---

# 规则2：配置管理 — config.yaml 是单一数据源

**核心原则**：config.yaml 是一切配置的最终来源，BacktestConfig 必须与 yaml 完全同步。

**具体规则**：
- 删除 config.yaml 中的废弃字段（fusion_mode、regime_filter_enabled、strategy_switching 等）
- BacktestConfig 字段命名与 config.yaml 保持一致
- 新增配置项先在 yaml 定义，再在 BacktestConfig 中映射
- 运行 `BacktestConfig.from_yaml()` 后做字段完整性校验


---

# 规则3：废弃代码必须彻底清理

**核心原则**：已废弃的模块、字段、兼容层一律删除，不留后患。

**具体规则**：
- 废弃模块直接删除，不做 `@deprecated` 兼容别名
- 废弃字段从 config.yaml 和代码中同步删除
- 每轮重构完成后，全量 grep 检查废弃引用
- 根目录兼容层（如 config.py）在确认无引用后立即删除


---

# 规则4：风控类统一 — 一个系统只有一个风控

**核心原则**：不允许两个风控类并存，新代码用 RiskController，旧代码迁移。

**具体规则**：
- 核心风控逻辑在 RiskController 中实现
- RiskManager 作为 PyBroker 适配层，委托给 RiskController
- 新功能只加在 RiskController，RiskManager 不再扩展


---

# 规则5：策略注册统一 — 多策略横截面打分

**核心原则**：彻底移除单策略绑定机制，所有策略通过横截面打分进行动态仓位分配，统一使用 `CrossSectionalStrategy` 管理多策略组合。

**具体规则**：
- 统一使用 `core/strategy_registry.py` 的 StrategyLibrary 管理策略档案
- 所有策略类必须实现 `compute_score` 方法，返回归一化到 [-1, 1] 的因子得分
- 不再使用单策略 `execute` 方法做多/空二元决策，改为因子得分输出
- 多策略组合通过 `CrossSectionalStrategy` 进行横截面标准化 + 排名叠加
- 策略发现、参数获取、性能档案全部走统一入口

**涉及代码**：
- `core/strategies/cross_sectional.py`：多策略横截面打分引擎
- `core/strategy_registry.py`：策略库与档案管理
- `core/strategies/strategy_*.py`：各策略的 `compute_score` 方法


---

# 规则6：测试覆盖 — 关键路径必须有测试

**核心原则**：因子计算、调仓决策、风控触发必须有自动化测试覆盖。

**具体规则**：
- 新增策略必须有对应的因子计算正确性测试
- 新增风控规则必须有触发条件测试
- 调仓逻辑修改必须有调仓日判断测试
- 修改核心逻辑前先补测试，再重构


---

# 规则7：文件行数限制

**核心原则**：单文件不超过 500 行，超过必须拆分。

**具体规则**：
- 超过 500 行的文件在下次修改时强制拆分
- 拆分原则：按职责单一，每个模块只做一件事
- 当前需拆分的文件：暂无（broker_adapter.py 已拆分）


---

# 规则8：命名必须与功能一致

**核心原则**：类名、函数名必须准确反映当前功能，不允许名不副实。

**具体规则**：
- 类名变更后同步更新所有引用和文档
- 策略名称与策略文件一一对应，不允许别名
- 变量名包含计量单位（如 `_days`、`_pct`、`_bars`）


---

# 规则9：因子开发规范 — 24因子体系，IC 驱动，先验证后集成

**核心原则**：因子必须通过 IC 检验才能进入策略组合，无效因子不入库。基于 24 个因子（5 大类：趋势 T_01~T_05、回归 R_01~R_05、波动率 V_01~V_04、资金流 M_01~M_05、高阶复合 H_01~H_05）构建因子体系。

**因子准入标准**：
- 新因子必须实现 `FactorEvaluator` 接口，输出 IC、IR、多周期稳定性
- IC > 0.03 且 IR > 0.5 的因子方可保留，否则标记为待优化
- 因子间相关性 > 0.7 视为冗余，保留 IC 更高的因子
- 因子变换（对数/指数/幂函数/交叉项）必须对比原始因子的 IC 变化
- 最终因子集平均 IC > 0.04，最大互相关 < 0.6

**因子复核清单（6 项必须通过）**：
1. **数据存活率**：因子有效值占比 ≥ 85%，缺失率 > 15% 说明适用面过窄，需剔除或降级
2. **缺失值占比**：每个因子缺失率 ≤ 15%，超过则标记为待优化
3. **异常值抵抗**：极值处理前后 IC 对比，若极值导致 IC 翻转，说明因子抗噪极差，需增加 Winsorize 截尾
4. **参数敏感性**：关键参数微调（如跳空修复权重 0.3/0.7），若 IC 大幅衰减，说明过拟合，不具稳健性
5. **因子正交性**：与传统 Barra 风格因子（动量、波动率）相关性 ≤ 0.5，正交化后 t 值 ≥ 1.96
6. **时序稳定性**：滚动 1 年期 ICIR 时间方差，若某段年份极好另一段极差（甚至变号），说明逻辑非普适

**涉及代码**：
- `core/factors/factor_evaluator.py`：因子评估框架
- `core/factors/factor_selector.py`：因子筛选与去冗余
- `core/factors/factor_transformer.py`：因子变换与交叉项
- `core/factors/factor_review.py`：因子复核模块（6项检查）


---

# 规则10：（已移除）自适应参数模块已删除，功能由子策略体系覆盖

本规则已移除，相关功能由规则 21 的子策略体系覆盖。


---

# 规则11：多时间框架（规划中）

**核心原则**：多时间框架的核心价值是过滤逆势交易，而非增加交易机会。

**具体规则**：
- 日频信号与周频趋势方向一致时才执行交易，不一致时跳过或减仓
- 周频权重 60%，日频权重 40%，不得随意调整
- 冲突场景占比应 < 40%，冲突时盈亏比 > 1.0
- 周频信号缓存，日频信号实时计算，周五对齐同步
- 过滤后交易次数减少 > 30% 且胜率提升 > 5% 方为有效

**涉及代码**（规划中，模块尚未实现）：
- `core/multi_tf/trend_filter.py`：周频/月频趋势判断
- `core/multi_tf/signal_filter.py`：时间框架过滤规则
- `core/multi_tf/signal_sync.py`：信号延迟处理与同步


---

# 规则12：（已移除）动态仓位模块已删除，功能由子策略体系覆盖

本规则已移除，相关功能由规则 21 的子策略体系覆盖。


---

# 规则13：止损策略 — 分层叠加，效果可量化

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


---

# 规则14：（已移除）品种选择模块已删除，功能由子策略体系覆盖

本规则已移除，相关功能由规则 21 的子策略体系覆盖。


---

# 规则15：回测验证 — 多阶段验证，鲁棒性优先

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


---

# 规则16：模块目录结构 — 职责单一，接口清晰

**核心原则**：每个模块目录只做一件事，模块间通过明确接口交互。

**目录结构**（当前实际状态，标注"规划中"的模块尚未实现）：
```
core/
├── config/            # 配置管理（BacktestConfig + 因子/止损/验证配置）
├── factors/           # 因子模块（24因子体系 + 评估 + 变换 + 筛选 + 复核 + 清洗）
│   └── alpha_futures/ # 新因子库（基于抽象基类的独立因子类 + 注册表 + 引擎调度）
├── multi_tf/          # 多时间框架模块（规划中）
├── risk/              # 止损优化模块（追踪+时间+复合止损）
├── validation/        # 回测验证模块（蒙特卡洛+敏感性）
├── engine/            # 回测引擎（PyBroker+自研+策略集成）
│   ├── backtest_runner.py    # PyBroker 主回测运行器
│   ├── runner.py             # 自研验证引擎
│   ├── switch_engine.py      # 因子打分引擎（5子策略信号动态加载）
│   ├── strategy_executor.py  # 策略执行器工厂
│   ├── strategy_indicators.py# 策略指标注册表 + 退出钩子注册表（解耦核心）
│   ├── sub_strategy_adapter.py# 子策略适配器（连接因子库与子策略体系）
│   ├── top_level_integrator.py# 顶层策略集成器（信号合并）
│   ├── rolling_ic.py         # 滚动IC动态权重引擎
│   ├── factor_decay.py       # 因子衰减监控器
│   └── pybroker_data_source.py# PyBroker 数据源封装
├── strategies/        # 策略实现（5子策略 + 基类 + 横截面打分）
│   └── sub_strategies/# 5子策略：趋势/期限结构/均值回归/波动率突破/复合共振
├── performance/       # 绩效评估
└── monitor/           # 策略监控（规划中）

# 以下模块已移除，功能由子策略体系覆盖：
# ├── adaptive/        # 已移除（规则10）
# ├── position/        # 已移除（规则12）
# ├── instrument/      # 已移除（规则14）
# └── market_regime/   # 已移除（兼容性桩保留在 core/engine/runner.py 和 core/__init__.py）

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
- 模块间依赖方向：strategies → factors → engine，不得反向依赖
- 每个模块的 `__init__.py` 必须导出公共接口，隐藏内部实现
- 跨模块调用必须通过接口，不得直接访问其他模块的内部变量


---

# 规则17：不重复造轮子 — 优先调用公共系统

**核心原则**：runner/ 层仅做编排，核心逻辑必须委托给已有公共系统。

**公共系统清单（必须直接调用）**：
- **配置管理**：`core/config/` - 使用 `BacktestConfig.from_yaml()` 加载配置
- **数据提供者接口**：`core/data_provider.py` - 使用 `DataProvider` 抽象接口，实现数据源解耦
- **统一数据加载器**：`core/data_loader.py` - 使用 `DataLoader` 支持 TqSdk/CSV 数据源，主力合约识别，展期处理
- **PyBroker数据源封装**：`core/engine/pybroker_data_source.py` - 使用 `create_hybrid_data_source()`
- **因子计算引擎**：`core/factors/alpha_futures/factor_engine.py` - 使用 `FactorEngine` 统一调度因子计算
- **因子注册表**：`core/factors/alpha_futures/factor_registry.py` - 使用 `@register_factor` 装饰器注册因子类
- **因子基类**：`core/factors/alpha_futures/base_factor.py` - 继承 `BaseFactor` 实现自定义因子
- **数据清洗算子**：`core/factors/futures_data_cleaners.py` - 使用 `compute_open_adj()`/`compute_carry()`/`compute_oi_safe()` 等
- **基础算子库**：`core/factors/operators.py` - 使用 `delay()`/`delta()`/`sma()`/`std()`/`roll_ic()` 等通用函数
- **因子评估框架**：`core/factors/factor_evaluator.py` - 使用 `FactorEvaluator` 做 IC/IR/稳定性评估
- **策略指标注册表**：`core/engine/strategy_indicators.py` - 使用 `StrategyIndicatorRegistry` 注册策略指标
- **策略退出钩子注册表**：`core/engine/strategy_indicators.py` - 使用 `StrategyExitHookRegistry` 注册退出钩子
- **PyBroker回测引擎**：`core/engine/backtest_runner.py` - 使用 `PyBrokerBacktestRunner` 执行主回测
- **自研验证引擎**：`core/engine/runner.py` - 使用 `BacktestRunner` 执行验证，含 `cross_validate_with_pybroker()` 交叉验证
- **参数优化器**：`core/optimizer.py` - 使用 `ParameterOptimizer` 做网格搜索/滚动优化/Walk-Forward优化
- **多策略组合管理**：`core/portfolio.py` - 使用 `PortfolioManager` 管理多策略组合、权重分配
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


---

# 规则18：Pipeline 编排器 — 声明式调用

**核心原则**：使用 `runner.pipeline.Pipeline` 类组合回测流程，实现链式调用。

**具体规则**：
- 根目录脚本（`run_*.py`）仅解析参数并调用 Pipeline
- 使用 `with_config(**overrides)` 进行配置热更新
- 通过 `load_data().run_backtest().optimize().validate().report()` 链式组合流程
- 新增实验/优化/验证方法只需在对应目录添加文件并注册到 Pipeline
- 使用 `is_healthy()` 检查状态健康度


---

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


---

# 规则20：因子数据清洗与工程化 — 换月/交割/涨跌停处理

**核心原则**：因子计算前必须进行数据清洗，确保无前瞻性偏差和脏数据污染。

**1. 主力合约换月处理**：
- 使用复权价格（后复权）构建连续价格序列，消除换月跳空
- 换月日前后 3 个交易日的 `OI` 及 `DELTA(OI)` 强制设为 `NaN`
- 滚动窗口函数（`SUM`, `MEAN`）遇到 `NaN` 自动跳过，不前向填充

**2. 交割月数据剔除**：
- 进入交割月前 N 个交易日（可配置，默认 5 天）的全部数据剔除
- 持仓量使用全市场该品种所有合约的总持仓量，而非单合约持仓

**3. 涨跌停板过滤**：
- 若 `|(open - prev_close) / prev_close| > threshold`（默认 0.06），则当日：
  - `INTRADAY_RET` 直接置 0
  - 所有依赖 `high-low`、`(C-L)-(H-C)` 等日内结构的因子值置为 `NaN`

**4. 跳空缺口修复（全局）**：
- 基础公式：`OPEN_ADJ = OPEN * w + DELAY(CLOSE,1) * (1-w)`
- 自适应权重：w 根据该品种历史跳空延续率动态计算，范围 [0.2, 0.8]
- `INTRADAY_RET = (CLOSE - OPEN_ADJ) / OPEN_ADJ`，作为所有日内收益替代量

**5. 无前瞻性标准化（强制）**：
- 禁止使用全序列 `mean/std` 的 `ZSCORE`
- 强制使用滚动窗口标准化：`ZSCORE(x, window)`，仅用过去 window 天数据
- 或使用扩张窗口标准化：`ZSCORE_expanding(x)`，从第一根 K 线到当前 t
- 所有 `CORR`、`RANK` 也必须基于滚动窗口或扩张窗口

**6. 统一后处理**：
- 缩尾（Winsorize）：每个因子计算完成后，按 1% 和 99% 分位数截断
- 缺失值填充：默认不填充（保留 NaN），策略层自行决定前向填充或剔除
- 横截面标准化（多品种）：`factor = (factor - mean) / std`，按日期计算
- 时序标准化（单品种）：`factor = (factor - rolling_mean(60)) / rolling_std(60)`

**涉及代码**：
- `core/factors/factor_review.py`：因子复核与数据质量检查
- `core/factors/data_cleaner.py`：换月/交割/涨跌停处理
- `core/factors/gap_fixer.py`：跳空缺口自适应修复
- `core/factors/normalizer.py`：无前瞻性滚动标准化


---

# 规则21：多策略子策略划分与集成 — 5 子策略体系

**核心原则**：基于因子逻辑类别，构建 5 个独立子策略，通过集成方法形成最终信号，实现稳健绝对收益。

**子策略划分方案**：

| 子策略名称 | 使用的因子 | 逻辑核心 | 信号方向 |
|---------|---------|---------|---------|
| 趋势策略 | T_01, T_02, T_03, T_05, V_02, M_03 | 趋势确认 + 资金流确认 | 顺势交易 |
| 期限结构策略 | T_04, R_04, M_04, H_05 | Carry + 增仓/资金流共振 | Back做多，Contango做空 |
| 均值回归策略 | R_01, R_02, R_03, R_05, H_03 | 增仓背离、持仓萎缩反转 | 逆势交易 |
| 波动率突破策略 | V_01, V_03, V_04, H_04 | 持仓异动 + 价格加速度 | 突破跟进 |
| 复合共振策略 | H_01, H_02, M_01, M_02, M_05 | 多维度高阶统计共振 | 综合打分 |

**阶段二实施规则（子策略合成与集成）**：

### 21.1 子策略基类设计
- 所有子策略继承 `SubStrategyBase` 抽象基类
- 必须实现 `compute_signal(ctx, factor_data)` 方法，返回该子策略的信号
- 必须定义 `factor_list` 属性（该子策略使用的因子列表）
- 可选实现 `post_process(signal)` 做子策略特定后处理
- 通过 `self.config` 访问全局配置

### 21.2 单个子策略信号生成
- **因子标准化**：对子策略内每个因子，每天计算横截面 Z 分数（多品种）
- **方向调整**：若因子方向为反向（如 R_05），需乘 -1 调整方向
- **因子加权合成**：
  - 默认使用等权法：`sub_signal = mean(factor1_z, factor2_z, ...)`
  - 可选滚动 IC 动态权重：IC 越高权重越大
- **信号裁剪**：`position = np.clip(sub_signal, -1, 1)`

### 21.3 子策略级风控
- **波动率目标**：调整仓位使子策略预期波动率等于目标值（默认 15%）
- **最大回撤止损**：子策略净值回撤超过 8% 时，该子策略清仓并暂停 3 天
- **持仓限制**：单品种单边仓位不超过总资金的 10%

### 21.4 多策略集成（顶层模型）
- **信号合并方法**：
  - **等权叠加**（默认）：`final_signal = (signal1 + ... + signal5) / 5`，再裁剪到 [-1, 1]
  - **波动率倒数加权**：`weight_i = 1 / vol_i`，动态调整，降低高波动子策略权重
  - **基于收益率的自适应权重**：使用卡尔曼滤波或滚动优化最大化综合 Sharpe 比
  - **多数投票**：将连续信号转为方向（+1 / -1 / 0），取多数方向作为最终方向
- **顶层风控**：
  - 总杠杆限制：所有子策略叠加后的总名义仓位不超过 2 倍
  - 品种集中度：同一品种上的净持仓不超过总资金的 15%
  - 市场状态过滤：全市场波动率处于历史 80% 分位数以上时，整体仓位减半

### 21.5 因子准入标准
- 每个因子进入子策略前，需先通过 IC 检验（IC > 0.03, IR > 0.5）筛选
- 因子间相关性 > 0.7 视为冗余，保留 IC 更高的因子
- 缺失率 > 15% 的因子排除

**涉及代码**：
- `core/strategies/sub_strategies/base.py`：子策略基类
- `core/strategies/sub_strategies/trend.py`：趋势策略
- `core/strategies/sub_strategies/term_structure.py`：期限结构策略
- `core/strategies/sub_strategies/mean_reversion.py`：均值回归策略
- `core/strategies/sub_strategies/vol_breakout.py`：波动率突破策略
- `core/strategies/sub_strategies/composite.py`：复合共振策略
- `core/engine/top_level_integrator.py`：顶层策略集成器（新增）
- `core/engine/sub_strategy_adapter.py`：子策略适配器（新增）
- `core/engine/backtest_runner.py`：集成子策略体系
- `core/config/backtest_config.py`：`signal_merge_method` 配置项
- `config.yaml`：信号合并方法配置

**使用方式**：
1. 在 `config.yaml` 中设置信号合并方法：
   ```yaml
   backtest:
     signal_merge_method: equal_weight  # 可选: equal_weight/volatility_inverse/adaptive/majority_vote
   ```
2. 运行回测：`python run_backtest.py`


---

# 规则22：回测验证 — 滚动窗口 + 样本外验证

**核心原则**：所有新功能必须通过样本外验证，Sharpe 不得低于旧版本 90%。

**验证流程**：
- 滚动窗口测试：使用 3 年训练，1 年测试，滚动优化子策略权重
- 绩效指标：年化收益率、Sharpe 比（目标 > 1.5）、最大回撤（< 15%）、卡玛比（> 2）
- 稳定性检验：分年度绩效、不同品种分组绩效、参数敏感性测试

**实施顺序**：
1. 先完成因子工程化改造，进行 IC/IR 验证，剔除无效因子
2. 构建子策略，对每个子策略进行独立回测，优化内部因子权重
3. 集成测试，比较不同集成方法的效果，选择最优
4. 样本外验证（至少 1 年），确认策略稳健性
5. 实盘模拟，再进行实盘

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
- `core/validation/rolling_window.py`：滚动窗口测试
- `config.yaml`：灰度开关配置


---

# 规则23：因子库工程化重构 — 基于抽象基类的独立因子类体系

**核心原则**：将函数式+编排类的因子计算结构，重构为**基于抽象基类的独立因子类 + 注册表 + 引擎调度** 的架构，实现易扩展、易测试、易维护。

**新架构设计**：
```
core/factors/
├── alpha_futures/                    # 新因子库目录
│   ├── __init__.py
│   ├── base_factor.py                 # 因子抽象基类 BaseFactor
│   ├── factor_registry.py             # 因子注册表（装饰器注册）
│   ├── factor_engine.py               # 因子计算引擎（数据清洗+调度）
│   ├── factors/                      # 独立因子类目录
│   │   ├── __init__.py
│   │   ├── t_01.py, t_02.py, ...    # 24个独立因子类
│   ├── operators.py                 # 保持不变（基础算子）
│   └── futures_data_cleaners.py     # 保持不变（数据清洗）
├── alpha_futures_23.py -> alpha_futures_24.py  # 保持不变，内部委托给新引擎
```

**具体规则**：

1. **因子基类（`base_factor.py`）**：
   - 每个因子继承 `BaseFactor` 抽象基类
   - 必须定义 `name`、`category`、`formula`、`dependencies` 类属性
   - 实现 `compute` 纯计算方法，仅依赖 kwargs 提供的字段
   - 可选实现 `post_process` 做因子特定后处理
   - 通过 `self.config` 访问全局配置

2. **因子注册表（`factor_registry.py`）**：
   - 使用 `@register_factor` 装饰器自动注册因子类
   - 提供 `get_factor`、`list_available_factors` 等查询接口
   - 注册表是全局单例，因子导入时自动注册

3. **因子引擎（`factor_engine.py`）**：
   - `FactorEngine` 负责：数据清洗 → 公共数据准备 → 因子调度 → 结果汇总
   - 在 `_prepare_public_data` 中集中计算所有因子需要的中间量并缓存（如 `oi_mean_20`、`delta_oi_1`、`carry_orth`）
   - 统一处理所有因子的公共依赖，避免重复计算
   - 检查因子依赖是否已准备，缺失则报错

4. **独立因子类（`factors/t_01.py` 等）**：
   - 每个因子一个独立文件，类名与因子编号对应（如 `class T_01(BaseFactor)`）
   - 明确声明 `dependencies` 列表（如 `["close", "oi_safe"]`）
   - `compute` 方法纯计算，无副作用
   - 复用 `operators.py` 中的基础算子

5. **向后兼容**：
   - 保留原 `AlphaFutures24` 类作为外观类，内部委托给新 `FactorEngine`
   - 保持原有 `compute_all` 接口签名完全一致，外部调用无需修改

**迁移指南**：
- 从 `alpha_futures_trend.py` 等模块中提取单个因子计算逻辑
- 封装为独立类，用 `@register_factor` 装饰
- 将原函数中的全局配置引用改为 `self.config`
- 在 `FactorEngine._prepare_public_data` 中计算公共依赖字段
- 编写独立单元测试验证每个因子

**涉及代码**：
- `core/factors/alpha_futures/base_factor.py`：因子抽象基类
- `core/factors/alpha_futures/factor_registry.py`：因子注册表
- `core/factors/alpha_futures/factor_engine.py`：因子计算引擎
- `core/factors/alpha_futures/factors/`：24个独立因子类文件
- `core/factors/alpha_futures_24.py`：向后兼容外观类


---

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


---

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


---

# 规则26：交叉验证机制 — 自研引擎与 PyBroker 并行验证

**核心原则**：自研引擎不做 PyBroker 的回退方案，两套引擎并行运行用于交叉验证。

**并行验证流程**：
1. PyBroker 引擎执行主回测（`PyBrokerBacktestRunner.run()`），PyBroker 不可用时直接抛异常，不静默回退
2. 自研引擎执行独立回测（`BacktestRunner.run()`），产生 `PortfolioResult`
3. 调用 `BacktestRunner.cross_validate_with_pybroker(pybroker_result, own_result)` 进行交叉验证
4. 验证分为 4 个层次，从粗到细逐步收敛问题：

**验证层次1：净值曲线一致性（基础）**
- 归一化净值 Pearson 相关系数（评估整体趋势一致性）
- 日收益率相关系数（评估每日波动方向一致性）
- 最大绝对差异、平均绝对差异、最大百分比差异
- 最大偏离日期（定位具体交易日差异来源）

**验证层次2：核心绩效指标一致性（重要）**
- Sharpe 比、Calmar 比、Sortino 比
- 年化收益、最大回撤幅度、最大回撤发生日期
- 胜率、盈亏比、总交易次数、多空占比

**验证层次3：逐笔交易一致性（深度）**
- 开仓时间、标的、方向、价格、持仓量是否匹配
- 平仓时间、价格、盈亏是否匹配
- 持仓时间分布对比（平均持仓天数、最长/最短持仓、持仓周期直方图）

**验证层次4：因子得分序列一致性（针对因子打分回测）**
- 每日各品种的因子得分是否一致（如果是因子打分策略）
- 横截面标准化后的得分是否一致

**告警规则**：
- 净值相关系数 < 0.95 → 严重告警
- 核心绩效指标差异 > 10% → 重要告警
- 逐笔交易不一致 → 详细告警，打印前 N 笔差异交易
- 差异超过 10% 发出警告（由规则1覆盖）

**实现细节**：
- `BacktestRunner.cross_validate_with_pybroker()` 通过 date 对齐两条净值曲线，归一化到同一初始值
- 计算 Pearson 相关系数评估整体趋势一致性
- 计算日收益率相关系数评估每日波动方向一致性
- 定位最大偏离日期，便于排查具体交易日差异来源
- `BacktestConfig.cross_validate` 开关控制是否执行交叉验证

**涉及代码**：
- `core/engine/runner.py`：`BacktestRunner.cross_validate_with_pybroker()` 方法
- `core/engine/backtest_runner.py`：`PyBrokerBacktestRunner.run()` 主回测
- `core/config/backtest_config.py`：`cross_validate` 配置开关


---

# 规则27：策略基类设计 — 可配置化与可扩展性

**核心原则**：`BaseStrategy` 抽象基类提供公共展期、止损、持仓管理，参数可配置化，子类无需重复实现。

**基类职责**：
- `_check_rollover()`：展期检查与自动平仓
- `_init_position_session()` / `_register_*_entry()` / `_clear_position()`：持仓会话管理
- `_check_trailing_stop_long/short()`：百分比跟踪止损
- `_check_time_stop()`：时间止损（持仓超过 N 天强制平仓）
- `_compute_oi_change()`：持仓量变化率计算
- `_compute_oi_divergence()`：价格与持仓量背离检测

**可配置化要求**：
- 阈值参数（如 `oi_change_threshold=0.03`、`price_change_threshold=0.005`）通过 `__init__` 或 config 字典传入
- 止损参数（如 `stop_pct`、`time_stop_days`）支持动态配置，允许子类为每个标的单独设置
- 所有方法添加准确类型注解（`numpy.typing.ArrayLike` 等）

**容错机制**：
- `_check_rollover` 先判断 `hasattr(ctx, 'is_dominant')` 再取值，避免过度 try-catch
- 所有数值计算包裹 try/except，返回安全默认值（0.0 或 False）
- 持仓状态检查失败时，不阻塞交易执行

**涉及代码**：
- `core/strategies/base.py`：`BaseStrategy` 抽象基类
- `core/strategies/sub_strategies/base.py`：子策略基类 `SubStrategyBase`


---

---

*最后更新：2026-06-06*
*参考指南：商品期货量化模型改造指南.docx*
*参考指南：商品期货 Alpha 因子库工程化重构提示词.docx*
*相关知识文档：../knowledges/20260602_001_workflow_strategy-enhancement-roadmap.md*
*相关知识文档：../knowledges/20260602_002_workflow_runner-scripts-refactor-plan.md*
