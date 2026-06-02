---
type: workflow
title: 调参/回测脚本拆分架构方案（含公共系统调用清单）
date: 2026-06-02
context: quant_system / runner架构重构
related: run_full_backtest.py, run_parameter_optimization.py, run_validation.py, core/engine/, utils/
---

# 调参/回测脚本拆分架构方案（含公共系统调用清单）

## 摘要

将4045行的3个根目录脚本拆分为runner/模块化架构。**核心原则：不重复造轮子，优先调用已有公共系统。** 经盘点发现7处重复实现，拆分时必须消除。

## 背景

### 文件规模
- run_full_backtest.py: 1732行，34个函数
- run_parameter_optimization.py: 925行，12个函数
- run_validation.py: 1388行，19个函数

### 核心问题
1. **循环依赖**：run_validation.py 从另外两个脚本导入函数
2. **职责混杂**：数据加载、回测执行、绘图、报告生成混在同一文件
3. **重复造轮子**：7处函数与已有公共模块重复（详见下文）
4. **违反规则7**：单文件超过500行限制

---

## 公共系统调用清单（必须直接调用，禁止重复实现）

### 1. 配置管理 → `core/config/`

| 已有公共接口 | 位置 | run_脚本中的重复 | 处理方式 |
|---|---|---|---|
| `BacktestConfig` | `core/config/backtest_config.py` | `_build_backtest_config()` | **直接用 `BacktestConfig.from_yaml()`** |
| `DEFAULT_FACTOR_WEIGHTS` | `core/config/constants.py` | 硬编码权重字典 | **直接导入常量** |
| `PYBROKER_EXTRA_COLUMNS` | `core/config/constants.py` | 重复定义列名 | **直接导入常量** |
| `INITIAL_CASH` | `core/config/constants.py` | 硬编码初始资金 | **直接导入常量** |

### 2. 数据加载 → `core/data_loader.py` + `core/engine/pybroker_data_source.py`

| 已有公共接口 | 位置 | run_脚本中的重复 | 处理方式 |
|---|---|---|---|
| `DataLoader` | `core/data_loader.py` | `load_config()` 中的yaml读取 | **用 `DataLoader` 加载数据** |
| `create_hybrid_data_source()` | `core/engine/pybroker_data_source.py` | 各脚本独立创建数据源 | **统一调用此函数** |
| `load_data_cached()` | `utils/session_state.py` | 无缓存的数据加载 | **Streamlit场景用缓存版本** |
| `register_pybroker_columns()` | `utils/session_state.py` | 重复注册列 | **直接调用** |

### 3. 指标计算 → `utils/indicators.py`

| 已有公共接口 | 位置 | run_脚本中的重复 | 处理方式 |
|---|---|---|---|
| `compute_true_range()` | `utils/indicators.py` | `_compute_atr()` 中的TR计算 | **ATR基于此函数实现** |
| `compute_adx()` | `utils/indicators.py` | backtest_runner中的ADX | **统一调用** |
| `compute_adx_series()` | `utils/indicators.py` | trend_filter中的ADX序列 | **统一调用** |

**关键发现**：`_compute_atr` 在3处重复实现：
- `run_full_backtest.py:1194` → 简单rolling mean
- `core/engine/backtest_runner.py:165` → Wilder平滑
- `utils/indicators.py` → 基于 `compute_true_range` 的标准实现

**统一方案**：ATR计算统一到 `utils/indicators.py`，新增 `compute_atr()` 函数，其他位置全部替换调用。

### 4. 绩效指标 → `utils/metrics.py` + `core/performance/`

| 已有公共接口 | 位置 | run_脚本中的重复 | 处理方式 |
|---|---|---|---|
| `MetricsCalculator` | `utils/metrics.py` | `format_metrics()` | **用 `MetricsCalculator.extract_from_pybroker_result()`** |
| `PerformanceEvaluator` | `core/performance/` | 重复计算Sharpe/回撤 | **直接调用** |
| `PerformanceMonitor` | `core/performance/` | 无监控的指标计算 | **接入监控** |

### 5. 报告生成 → `core/report_builder.py`

| 已有公共接口 | 位置 | run_脚本中的重复 | 处理方式 |
|---|---|---|---|
| `_safe_float()` | `core/report_builder.py:67` | `run_full_backtest.py:81` | **提取到 `runner/common/utils.py`，两处统一调用** |
| `_annualized_return()` | `core/report_builder.py:75` | run_脚本中重复计算 | **直接调用** |
| `_compute_drawdown()` | `core/report_builder.py:91` | 重复实现 | **直接调用** |
| `_compute_sharpe()` | `core/report_builder.py:124` | 重复实现 | **直接调用** |
| `_rolling_sharpe()` | `core/report_builder.py:136` | 重复实现 | **直接调用** |
| `generate_report()` | `core/report_builder.py:576` | `run_e10_html_report()` | **统一用 `generate_report()`** |
| `collect_from_directory()` | `core/report_builder.py:318` | 重复的结果收集逻辑 | **直接调用** |
| `collect_from_validation()` | `core/report_builder.py:427` | run_validation中的收集逻辑 | **直接调用** |

### 6. 绘图 → `utils/plots.py`

| 已有公共接口 | 位置 | run_脚本中的重复 | 处理方式 |
|---|---|---|---|
| `PlotManager` | `utils/plots.py` | 10个 `_plot_*` 函数 | **逐步迁移到 `PlotManager`，matplotlib→plotly** |

**注意**：`utils/plots.py` 使用 Plotly（面向Streamlit），而 run_脚本使用 matplotlib（面向静态图）。拆分时两种后端都保留，通过 `PlotManager` 统一接口，内部根据 `backend` 参数分发。

### 7. 日期处理 → `utils/date_utils.py`

| 已有公共接口 | 位置 | run_脚本中的重复 | 处理方式 |
|---|---|---|---|
| `safe_to_timestamp()` | `utils/date_utils.py` | 各脚本中的日期转换 | **直接调用** |
| `apply_date_filter()` | `utils/date_utils.py` | 重复的日期筛选逻辑 | **直接调用** |

### 8. 风控 → `core/risk/` + `core/risk_controller.py`

| 已有公共接口 | 位置 | run_脚本中的重复 | 处理方式 |
|---|---|---|---|
| `RiskController` | `core/risk_controller.py` | 无（run_脚本未使用） | **回测流程应接入风控** |
| `CompositeStopManager` | `core/risk/composite_stop.py` | 无 | **止损逻辑统一用此类** |
| `TrailingStop` | `core/risk/trailing_stop.py` | 无 | **追踪止损统一用此类** |
| `TimeStop` | `core/risk/time_stop.py` | 无 | **时间止损统一用此类** |

### 9. 优化 → `core/optimizer.py` + `core/validation/sensitivity.py`

| 已有公共接口 | 位置 | run_脚本中的重复 | 处理方式 |
|---|---|---|---|
| `ParameterOptimizer` | `core/optimizer.py` | `grid_search_single_strategy()` | **评估是否可复用或合并** |
| `SensitivityAnalyzer` | `core/validation/sensitivity.py` | `_param_stability_test()` | **直接调用** |

### 10. 策略注册 → `core/strategy_registry.py`

| 已有公共接口 | 位置 | run_脚本中的重复 | 处理方式 |
|---|---|---|---|
| `StrategyLibrary` | `core/strategy_registry.py` | `_get_strategy_names()` | **用 `StrategyLibrary.list_names()`** |
| `StrategyProfile` | `core/strategy_registry.py` | `_get_param_spaces()` | **用 `profile.param_ranges`** |

---

## 重复实现汇总（7处必须消除）

| # | 重复函数 | 出现位置 | 应调用的公共接口 |
|---|---|---|---|
| 1 | `_safe_float()` | `run_full_backtest.py:81` + `core/report_builder.py:67` | 提取到 `runner/common/utils.py` |
| 2 | `_compute_atr()` | `run_full_backtest.py:1194` + `backtest_runner.py:165` | 统一到 `utils/indicators.compute_atr()` |
| 3 | `format_metrics()` | `run_full_backtest.py:140` | `utils/metrics.MetricsCalculator` |
| 4 | `_get_strategy_names()` | `run_full_backtest.py:174` | `core/strategy_registry.StrategyLibrary` |
| 5 | `_get_param_spaces()` | `run_parameter_optimization.py:58` | `StrategyLibrary.get_profile().param_ranges` |
| 6 | `_compute_sharpe/_compute_drawdown` | `core/report_builder.py` 内部 | 拆分时直接调用，不重新实现 |
| 7 | `load_config()` | `run_full_backtest.py:108` | `core/config.BacktestConfig.from_yaml()` |

---

## 目标架构（整合用户修改建议）

```
runner/
├── __init__.py              # 导出Pipeline、快捷函数
├── pipeline.py              # Pipeline编排器（核心入口）
├── common/                  # 通用工具（不重复造轮子）
│   ├── __init__.py
│   ├── utils.py             # _safe_float, save_csv, 日期处理（委托utils/date_utils.py）
│   └── errors.py            # PipelineError, ConfigError
├── data/
│   ├── __init__.py
│   ├── loader.py            # 委托 core/data_loader.py + core/engine/pybroker_data_source.py
│   └── preprocessor.py      # 因子得分计算（委托 core/engine/switch_engine.py）
├── strategy/
│   ├── __init__.py
│   ├── selector.py          # 委托 core/strategy_registry.py
│   └── weights.py           # 委托 core/engine/rolling_ic.py + core/position/dynamic_weight.py
├── backtest/
│   ├── __init__.py
│   ├── runner.py            # 委托 core/engine/backtest_runner.py
│   ├── experiments/
│   │   ├── e1_e5.py         # 实验1-5
│   │   └── e6_e11.py        # 实验6-11
│   └── walkforward.py       # WalkForward（若E6逻辑独立可保留）
├── optimization/
│   ├── __init__.py
│   ├── grid_search.py       # 网格搜索
│   ├── window_search.py     # 窗口搜索
│   ├── oos_selector.py      # 样本外优先选择
│   └── sensitivity.py       # 委托 core/validation/sensitivity.py
├── validation/
│   ├── __init__.py
│   ├── train_test.py        # 训练/测试分割
│   ├── monte_carlo.py       # 委托 core/validation/monte_carlo.py
│   ├── bootstrap.py         # Bootstrap置信区间
│   └── factor_stability.py  # 因子IC稳定性
└── report/
    ├── __init__.py
    ├── plots.py             # 委托 utils/plots.py（PlotManager）+ matplotlib后端
    ├── html_report.py       # 委托 core/report_builder.py
    └── exporters.py         # 多格式导出（CSV/PDF）
```

**关键原则**：runner/ 各模块是**编排层**，核心逻辑委托给 `core/` 和 `utils/`，不重新实现。

---

## Pipeline编排器设计

```python
from core.config import BacktestConfig
from core.data_loader import DataLoader
from core.engine.backtest_runner import PyBrokerBacktestRunner
from core.strategy_registry import StrategyLibrary
from utils.metrics import MetricsCalculator
from utils.date_utils import safe_to_timestamp, apply_date_filter

class Pipeline:
    """回测流水线编排器，支持声明式组合实验步骤。"""

    def __init__(self, config_path: str = "config.yaml"):
        # 直接调用公共配置系统，不重新解析yaml
        self.config = BacktestConfig.from_yaml(config_path)
        self._data = None
        self._results = {}

    def load_data(self) -> "Pipeline":
        """加载数据（委托 core/engine/pybroker_data_source）。"""
        from core.engine.pybroker_data_source import create_hybrid_data_source
        self._data = create_hybrid_data_source(self.config)
        return self

    def run_backtest(self, experiment: str) -> "Pipeline":
        """执行指定实验（委托 core/engine/backtest_runner）。"""
        from runner.backtest.experiments import get_experiment_runner
        runner = get_experiment_runner(experiment)
        self._results[experiment] = runner(self.config, self._data)
        return self

    def optimize(self, strategy: str) -> "Pipeline":
        """参数优化（委托 runner/optimization/）。"""
        from runner.optimization.grid_search import grid_search_single_strategy
        self._results["optimization"] = grid_search_single_strategy(
            self.config, strategy, self._data
        )
        return self

    def validate(self, method: str = "walkforward") -> "Pipeline":
        """验证（委托 runner/validation/ + core/validation/）。"""
        from runner.validation import get_validator
        validator = get_validator(method)
        self._results["validation"] = validator(self.config, self._data, self._results)
        return self

    def report(self, fmt: str = "html") -> "Pipeline":
        """生成报告（委托 core/report_builder + utils/plots）。"""
        from runner.report import generate
        generate(fmt, self._results, self.config)
        return self

    def with_config(self, **overrides) -> "Pipeline":
        """配置热更新，返回新实例。"""
        new_config = self.config.copy(update=overrides)
        new_pipe = Pipeline.__new__(Pipeline)
        new_pipe.config = new_config
        new_pipe._data = self._data
        new_pipe._results = dict(self._results)
        return new_pipe

    def is_healthy(self) -> bool:
        """状态检查。"""
        return self._data is not None
```

---

## 迁移步骤（每步完成后运行测试）

### 第1步：创建目录骨架与通用模块
- 创建 `runner/` 及所有子目录，添加 `__init__.py`
- 创建 `runner/common/utils.py`，**从 `core/report_builder.py` 和 `run_full_backtest.py` 提取 `_safe_float`（消除重复#1）**
- 创建 `runner/common/errors.py`，定义 `PipelineError`, `ConfigError`
- **同步更新 `core/report_builder.py` 中的 `_safe_float` 改为从 `runner/common/utils.py` 导入**

### 第2步：提取数据层
- `runner/data/loader.py`：**直接调用 `BacktestConfig.from_yaml()`（消除重复#7），调用 `create_hybrid_data_source()`**
- `runner/data/preprocessor.py`：迁移 `_compute_factor_scores_from_ohlcv`
- **新增 `utils/indicators.compute_atr()`（消除重复#2），替换 `run_full_backtest.py` 和 `backtest_runner.py` 中的实现**

### 第3步：提取策略层
- `runner/strategy/selector.py`：**直接调用 `StrategyLibrary`（消除重复#4、#5）**
- `runner/strategy/weights.py`：**委托 `RollingICWeightEngine` 和 `DynamicWeightAdjuster`**

### 第4步：提取回测核心
- `runner/backtest/runner.py`：**委托 `PyBrokerBacktestRunner`**
- `runner/backtest/experiments/e1_e5.py`：迁移 E1~E5
- `runner/backtest/experiments/e6_e11.py`：迁移 E6~E11
- **`format_metrics` 改为调用 `MetricsCalculator`（消除重复#3）**

### 第5步：提取优化层
- `runner/optimization/grid_search.py`：迁移网格搜索
- `runner/optimization/window_search.py`：迁移窗口搜索
- `runner/optimization/oos_selector.py`：迁移OOS优先选择
- `runner/optimization/sensitivity.py`：**委托 `core/validation/sensitivity.py`**

### 第6步：提取验证层
- `runner/validation/train_test.py`：迁移训练/测试分割
- `runner/validation/monte_carlo.py`：**委托 `core/validation/monte_carlo.py`**
- `runner/validation/bootstrap.py`：迁移Bootstrap
- `runner/validation/factor_stability.py`：迁移IC稳定性

### 第7步：提取报告层
- `runner/report/plots.py`：**委托 `utils/plots.py`（PlotManager）+ 保留matplotlib后端**
- `runner/report/html_report.py`：**委托 `core/report_builder.generate_report()`**
- `runner/report/exporters.py`：多格式导出

### 第8步：实现Pipeline编排器
- `runner/pipeline.py`：Pipeline类（如上设计）
- `runner/__init__.py`：导出Pipeline及主要模块

### 第9步：根目录脚本改为薄壳
- `run_full_backtest.py` → `Pipeline("config.yaml").load_data().run_backtest("e1").report()`
- `run_parameter_optimization.py` → `Pipeline(...).optimize("ts_momentum")`
- `run_validation.py` → `Pipeline(...).validate("walkforward")`

### 第10步：测试与清理
- 运行 `pytest tests/` 确保功能一致
- 对比拆分前后所有实验输出（总收益率、夏普、最大回撤），误差 < 1e-8
- 删除7处重复实现
- CI中加入依赖方向检查

---

## 强制约束

- 每个 `.py` 文件不超过 500 行
- **禁止重复造轮子**：runner/ 是编排层，核心逻辑必须委托给 `core/` 和 `utils/`
- 禁止跨层循环依赖
- 所有数值计算函数必须保留原有浮点精度
- 每完成一步必须运行现有回测脚本确保功能不变

## 新功能快速整合方式

1. **新增实验**：在 `runner/backtest/experiments/` 添加文件，注册到Pipeline
2. **新增优化方法**：在 `runner/optimization/` 添加文件，Pipeline.optimize("bayesian")即可
3. **新增验证方法**：在 `runner/validation/` 添加文件，委托 `core/validation/` 核心逻辑
4. **新增报告格式**：在 `runner/report/` 添加文件，Pipeline.report("pdf")即可

## 要点

1. **不重复造轮子**：7处重复实现必须消除，统一调用公共系统
2. runner/ 是编排层，core/ 是实现层，utils/ 是工具层
3. Pipeline编排器实现声明式链式调用
4. 迁移分10步渐进执行，每步可独立验证
5. 根目录脚本最终变为薄壳（<50行）
