# 规则17：不重复造轮子 — 优先调用公共系统 + 根目录脚本收敛

**核心原则**：runner/ 层仅做编排，核心逻辑必须委托给已有公共系统；根目录脚本（`run_*.py`）只允许 3 个官方入口，其他工作流必须通过 Pipeline 编排器调用 `runner/` 模块。

**生效日期**：2026-06-10（吸收规则 20）

---

## 17.1 公共系统清单（必须直接调用）

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
- **统一因子池**：`core/execution/factor_pool.py` - 使用 `UnifiedFactorPool` 单入口计算所有信号（24 Alpha + 6 CTA）
- **信号抽象层**：`core/execution/signal_abstraction.py` - 使用 `SignalAbstractionLayer` 按模式提取信号（横截面/CTA/混合）

---

## 17.2 官方入口脚本（吸收自规则 20）

根目录脚本（`run_*.py`）**只允许 3 个官方入口**，其他工作流必须通过 Pipeline 编排器调用 `runner/` 模块。

| 脚本 | 委托方法 | 用途 |
|------|----------|------|
| `run_backtest.py` | `pipe.run_backtest()` / `pipe.optimize()` / `pipe.validate()` / `pipe.report()` | 单实验 / 优化 / 验证 / 报告 |
| `run_optimize.py` | `pipe.optimize()` | 仅参数优化 |
| `run_validate.py` | `pipe.validate()` | 仅验证 |

> 这 3 个入口已记录于 README.md 和 docs/strategy_validation_plan.md，共 50+ 处引用，不可删除。

### 规则 17.2.1：禁止新增自定义根目录脚本

任何非官方入口的 `run_*.py` 都不允许新增。统一在 `runner/` 下新增模块 + 在 Pipeline 注册方法。

### 规则 17.2.2：禁止在根目录脚本自实现核心逻辑

若必须修改 3 个官方入口，主体逻辑（数据加载、回测执行、验证、报告）必须委托 `runner/` 模块或 `core/` 模块，不得在 `run_*.py` 内直接调用 `PyBrokerBacktestRunner.run()` 等底层 API。

### 规则 17.2.3：删除自定义工作流脚本须同时迁移到 Pipeline

删除根目录工作流脚本时，必须同步完成：
1. 提取其核心逻辑到 `runner/` 下对应模块（`backtest/` / `validation/` / `optimization/` / `report/`）
2. 在 `Pipeline` 类中新增对应方法
3. 更新 README.md 和 docs/ 中所有引用
4. 在本规则"已删除脚本"表格中记录迁移版本号

### 规则 17.2.4：Pipeline 方法命名规范

- 单一实验 → `pipe.run_backtest(name: str)`
- 批量实验 → `pipe.run_experiments(names: List[str])`
- 多窗口 OOS → `pipe.multi_oos(...)`
- 全量验证 → `pipe.full_validation(...)`
- 参数优化 → `pipe.optimize(...)`
- 验证方法 → `pipe.validate(method: str)`
- 报告生成 → `pipe.report(fmt: str)`

方法名使用动词或动名词短语，不使用缩写（除 `mc` / `oos` 等通用术语外）。

### 规则 17.2.5：模块导出与 hidden internal

- 每个新模块必须通过 `__all__` 显式导出公共接口
- 内部辅助函数使用下划线前缀（如 `_phase1_optimize` / `_phase2_ew_backtest`）
- Pipeline 内部入口方法（`run_*` / `multi_*` / `full_*`）必须为 `self` 返回类型（链式调用）

### 已删除脚本（迁移历史）

| 已删除 | 原行数 | 替换入口 | 迁移版本 |
|--------|--------|----------|----------|
| `run_full_experiments.py` | 78 | `pipe.run_experiments(experiments: List[str])` | a42e5fa (2026-06-10) |
| `run_full_validation.py` | 314 | `pipe.full_validation(in_sample_start, in_sample_end, oos_start, oos_end, ...)` | a42e5fa (2026-06-10) |
| `run_multi_oos.py` | 123 | `pipe.multi_oos(windows, strategies, best_params, ...)` | a42e5fa (2026-06-10) |

---

## 17.3 具体规则

- 禁止在 run_ 脚本或 runner/ 模块中重复实现已有功能
- 所有工具函数优先检查 `utils/` 和 `core/` 中是否已存在
- 公共函数提取统一到 `runner/common/utils.py`
- 发现重复实现必须先提取再使用
- 回测、优化、验证任务必须使用官方入口脚本，禁止重写回测脚本

---

## 17.4 维护检查清单

新增/删除/修改根目录脚本或 Pipeline 方法时，必须确认：

- [ ] 根目录仅有 `run_backtest.py` / `run_optimize.py` / `run_validate.py` 3 个
- [ ] 新方法/新模块已在 17.2.5 命名规范中登记
- [ ] `runner/` 下模块文件不超过 500 行（规则 7）
- [ ] `__all__` 显式导出公共接口
- [ ] README.md / docs/ 中引用保持同步
- [ ] git commit 信息标注 `chore(cleanup)` 或 `refactor(pipeline)`

---

## 与其他规则的关系

| 关联规则 | 关系 |
|----------|------|
| 规则 18（Pipeline 编排器） | 17.2 是 18 的入口约束：仅 3 个根脚本调用 Pipeline |
| 规则 16（目录结构） | `runner/` 子模块归属见 16 的目录结构图 |
