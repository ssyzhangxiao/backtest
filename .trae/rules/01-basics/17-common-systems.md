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

**官方入口脚本（必须使用，禁止重复实现）**：
- **回测**：`run_backtest.py` - 执行回测流程，支持多策略横截面打分模式
- **优化**：`run_optimize.py` - 执行参数优化，支持网格搜索、窗口搜索、OOS选择
- **验证**：`run_validate.py` - 执行策略验证，支持蒙特卡洛、交叉验证、因子IC稳定性等

**具体规则**：
- 禁止在 run_ 脚本或 runner/ 模块中重复实现已有功能
- 所有工具函数优先检查 `utils/` 和 `core/` 中是否已存在
- 公共函数提取统一到 `runner/common/utils.py`
- 发现重复实现必须先提取再使用
- 回测、优化、验证任务必须使用官方入口脚本，禁止重写回测脚本
