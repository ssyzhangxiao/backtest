# 策略价值验证 To-Do 计划

> **依据**：[规则 28 — 策略价值验证：5 阶段硬性验证](../../.trae/rules/05-validation/28-strategy-value-validation.md)
> **目标**：在投入工程化（滑点模型、实时风控、模拟盘）前，端到端确认当前因子/策略体系具有可交易价值
> **执行原则**：任何阶段未达阈值即停，回到因子/策略研发
> **更新日期**：2026-06-08

---

## 0. 全局硬性阈值速查

| 阶段 | 关键阈值 | 硬性标准 |
|---|---|---|
| **A** | 因子通过率 | ≥ 50%（≥ 12/24） |
| **A** | 单因子 abs(IC 均值) | > 0.03 且 IR > 0.5 |
| **A** | 通过因子集合平均 abs(IC) | > 0.04 |
| **A** | 因子计算参数敏感性（±20% 扰动） | IC 衰减 < 30% |
| **B** | 组合 abs(IC 均值) | > 0.03 |
| **B** | 组合 ICIR | > 0.5 |
| **B** | 品种覆盖 | 至少 60% 品种上为正 |
| **C** | 组合 Sharpe | > 0.5 |
| **C** | 最大回撤 | < 20% |
| **C** | 胜率 | > 45% |
| **D** | Walk-Forward 样本外 Sharpe | 均值 > 0.3，衰减 < 30% |
| **D** | E7 样本外 Sharpe | > 0，最大回撤 ≤ 样本内 1.5× |
| **D** | 交易执行参数敏感性（±20% 扰动） | Sharpe 变化 < 30% |
| **D** | Bootstrap/MC 95% 置信区间 | 不含 0 |
| **D** | 双引擎交叉验证 | 净值相关系数 > 0.99，收益偏差 < 1% |
| **E** | 2×/5×/10× 成本下收益 | 5× 仍为正 |
| **E** | 环境切片 | ≥ 3/4 环境 Sharpe > 0，**且**任意单一环境 Sharpe > -1.0 |

---

## 1. 阶段 A — 因子价值验证

**Go 标准**：A1+A2+A3 全部通过才能进入 B 阶段。

### 1.1 24 因子逐个 IC/IR 筛选（factor_alpha24）
- [ ] 运行 `python run_validate.py --method factor_alpha24`
- [ ] 确认通过率 ≥ 50%（≥ 12/24 个因子）
- [ ] 逐个核对：abs(IC 均值) > 0.03 且 IR > 0.5
- [ ] 验证通过因子集合平均 abs(IC) > 0.04
- [ ] 验证通过因子间最大互相关 < 0.6
- [ ] 反向因子（负 IC）必须额外确认 ICIR > 0.5
- [ ] 产物：`output/validate/factor_alpha24_screening.csv`

### 1.2 滚动 IC 时序稳定性（factor_ic）
- [ ] 运行 `python run_validate.py --method factor_ic`
- [ ] 检查滚动 60 天 IC 方差 < 0.05
- [ ] 任何通过因子**无大段年份变号**（如 2024 年全为负、2025 年全为正）
- [ ] 排除不稳定因子（时序上 alpha 失效）
- [ ] 产物：`output/validate/factor_ic_stability/*.csv` + 滚动 IC 时序图

### 1.3 6 项因子复核（factor_review）
- [ ] 运行 `python run_validate.py --method factor_review`
- [ ] 数据存活率 ≥ 85%
- [ ] 缺失值占比 ≤ 15%
- [ ] 异常值抵抗：Winsorize 前后 IC 不翻转
- [ ] **因子计算参数敏感性**：回看周期（`lookback`/`window`/`half_life`）±20% 扰动后，IC 衰减 < 30%
- [ ] 因子正交性：与 Barra 风格因子相关性 ≤ 0.5
- [ ] 时序稳定性：1 年期 ICIR 方差可控
- [ ] 产物：`output/validate/factor_review_report.csv`

### 1.4 失败行动
- 通过率 < 50%：**停止所有回测优化**，回归 `core/factors/alpha_futures/` 研发新因子
- 单个因子不达标：使用 `FactorPipeline` 做非线性变换、交叉项构造
- 数据存活率不达标：检查 `core/factors/futures_data_cleaners.py` 复权/换月/交割月清洗逻辑
- 滚动 IC 变号：剔除不稳定因子

---

## 2. 阶段 B — 多因子组合 IC

**Go 标准**：组合 abs(IC) > 0.03 + ICIR > 0.5 + 60% 品种覆盖。

### 2.1 新增模块 `runner/validation/factor_combo_ic.py`
- [ ] 实现 `factor_combo_ic_validation(data_source, config, lib, output_dir, **kwargs) -> Dict`
- [ ] 委托 `core/factors/alpha_futures/factor_engine.py` 计算各因子值
- [ ] 委托 `core/engine/rolling_ic.py` 计算组合的滚动 IC 与 ICIR
- [ ] 支持等权 + IC 加权两种合成方式
- [ ] 跨品种横截面标准化（Z 分数）

### 2.2 注册到 `VALIDATOR_MAP`
- [ ] 在 `runner/validation/__init__.py` 中：
  ```python
  # 公共导出接口（模块级常量，不带下划线前缀，便于跨包引用）
  VALIDATOR_MAP: Dict[str, Callable] = {}
  from runner.validation.factor_combo_ic import factor_combo_ic_validation
  VALIDATOR_MAP["factor_combo_ic"] = factor_combo_ic_validation
  ```
- [ ] 确认 `run_validate.py --method factor_combo_ic` 可正常调用

### 2.3 通过标准校验
- [ ] 运行 `python run_validate.py --method factor_combo_ic`
- [ ] 等权组合 abs(IC 均值) > 0.03
- [ ] 等权组合 ICIR > 0.5
- [ ] 至少 60% 品种上为正
- [ ] 产物：`output/validate/factor_combo_ic.csv`

### 2.4 失败行动
- 组合 IC 衰减严重：调整因子权重（ICIR 加权）、引入非线性变换
- 品种覆盖不足：剔除小众品种、调整横截面标准化方法
- 仍不达标：进入新因子研发或机器学习合成

---

## 3. 阶段 C — 策略回测验证

**Go 标准**：组合 Sharpe > 0.5 + 回撤 < 20% + 胜率 > 45%。

### 3.1 E1 单策略多品种基线
- [ ] 运行 `python run_backtest.py --experiment e1`
- [ ] 检查每个策略 × 每个品种的 Sharpe 矩阵
- [ ] 至少 70% 品种的 Sharpe > 0
- [ ] 各策略×品种矩阵平均 Sharpe > 0.3
- [ ] 胜率 > 45%
- [ ] 产物：`output/backtest/e1_baseline_metrics.csv`

### 3.2 E2/E3 多策略组合
- [ ] 运行 `python run_backtest.py --experiment e2`（等权组合）
- [ ] 运行 `python run_backtest.py --experiment e3`（动态权 / rolling IC 加权）
- [ ] 组合 Sharpe > 0.5
- [ ] 最大回撤 < 20%
- [ ] 胜率 > 45%
- [ ] 产物：`output/backtest/e2_e3_*.csv`

### 3.3 横截面多策略（CrossSectional）
- [ ] 运行 `python run_backtest.py --cross-sectional`
- [ ] 委托 `core/strategies/cross_sectional.py` 做多策略横截面打分
- [ ] 委托 `core/strategy_registry.py` 的 `StrategyLibrary` 获取策略
- [ ] 组合 Sharpe > 0.5
- [ ] 最大回撤 < 20%
- [ ] 胜率 > 45%
- [ ] 年化收益 > 8%
- [ ] 与单品种策略相关性 < 0.5
- [ ] 产物：`output/backtest/cross_sectional_*`

### 3.4 失败行动
- 大多数品种 Sharpe < 0：策略设计有问题，回到因子/子策略研发
- 组合 Sharpe < 0.5：重新分组子策略、`PortfolioManager` 动态权重
- 回撤 > 20%：降低仓位上限、加强风控（ATR 止损等）

---

## 4. 阶段 D — 稳健性验证（防过拟合）

**Go 标准**：样本外 Sharpe 衰减 < 30% + 参数扰动 Sharpe 变化 < 30% + 95% 置信区间不含 0。

### 4.1 Walk-Forward（E6）
- [ ] 运行 `python run_backtest.py --experiment e6`
- [ ] 训练 252 bars，测试 63 bars，步进 21 bars（`config.yaml:walk_forward`）
- [ ] 测试期 Sharpe 均值 > 0.3
- [ ] 较训练期衰减 < 30%
- [ ] 产物：`output/backtest/e6_walkforward_metrics.csv`

### 4.2 样本外验证（E7）
- [ ] 运行 `python run_backtest.py --experiment e7`
- [ ] 样本内截止 `in_sample_end_date: '2023-01-01'`
- [ ] 样本外 **2023-01-01 至 2026-06-07**（动态滚动：默认过去 3 年样本内 + 最近 1 年样本外，随数据更新自动平移；右界取昨日）
- [ ] 样本外 Sharpe > 0
- [ ] 最大回撤不超过样本内 1.5 倍
- [ ] 产物：`output/backtest/e7_out_of_sample_metrics.csv`

### 4.3 交易执行参数敏感性
- [ ] 运行 `python run_optimize.py --method sensitivity`
- [ ] 测试范围：`rebalance_freq`/`stop_loss_pct`/`entry_threshold`/`position_cap`
- [ ] ±20% 扰动后，组合 Sharpe 变化 < 30%
- [ ] 产物：`output/optimize/sensitivity_*.csv`
- [ ] **注意**：本项专测交易执行参数，因子计算参数敏感性在 A 阶段（2.3.4）执行

### 4.5 Bootstrap / Monte Carlo（E8/E9）
- [ ] 运行 `python run_backtest.py --experiment e8`（Bootstrap 5000 次）
- [ ] 运行 `python run_backtest.py --experiment e9`（Monte Carlo 1000 次）
- [ ] Sharpe 95% 置信区间不含 0
- [ ] 产物：`output/backtest/e8_bootstrap_*.csv` / `e9_monte_carlo_*.csv`

### 4.6 双引擎交叉验证（规则 1 / 26）
- [ ] 启用 `BacktestConfig.cross_validate = True`（开关控制是否执行）
- [ ] 同一参数下分别调用：
  - `core/engine/backtest_runner.py::PyBrokerBacktestRunner.run()`（主回测）
  - `core/engine/runner.py::BacktestRunner.run()`（自研引擎，验证用）
- [ ] 委托 `BacktestRunner.cross_validate_with_pybroker(pybroker_result, own_result)` 一次性比对
- [ ] **通过标准**：
  - 归一化净值 Pearson 相关系数 > **0.99**
  - 日收益率相关系数 > 0.95
  - 总收益偏差 < **1%**
- [ ] 若任一指标 < 阈值：定位最大偏离日期并打印前 N 笔差异交易
- [ ] **注意**：PyBroker 不可用时**禁止**静默回退到自研引擎，必须直接抛 `RuntimeError`（规则 1）
- [ ] 产物：`output/backtest/dual_engine_diff_report.csv`

### 4.7 失败行动
- 样本外 Sharpe 衰减 > 30%：**过拟合严重**，简化模型/减少参数
- 参数扰动 Sharpe 变化 > 30%：锁定最优参数可能过拟合，引入正则化
- Bootstrap 区间含 0：统计显著性不足，扩充样本或降低策略复杂度
- 双引擎相关系数 < 0.99 或收益偏差 > 1%：定位偏离日期，复核逐笔交易（规则 26）

---

## 5. 阶段 E — 成本与环境抗噪

**Go 标准**：5× 成本仍正收益 + 3/4 环境 Sharpe > 0 + 任意单一环境 Sharpe > -1.0。

### 5.1 交易成本敏感度
- [ ] 创建 `config.cost_2x.yaml`、`config.cost_5x.yaml`、`config.cost_10x.yaml`（独立配置文件）
- [ ] 确认 `run_backtest.py` 支持 `--config <yaml_path>` 参数
- [ ] **强制使用 `--config`** 跑批量成本测试（不再使用 `--override` 避免污染）：
  ```bash
  python run_backtest.py --config config.cost_2x.yaml --experiment e1
  python run_backtest.py --config config.cost_5x.yaml --experiment e1
  python run_backtest.py --config config.cost_10x.yaml --experiment e1
  ```
- [ ] 归档 2×/5×/10× 三档成本结果
- [ ] 5× 成本下组合仍为正收益
- [ ] Sharpe 衰减 < 50%
- [ ] 产物一一对应：
  - `output/backtest/cost_2x_*.csv`
  - `output/backtest/cost_5x_*.csv`
  - `output/backtest/cost_10x_*.csv`
- [ ] **禁止**直接修改 `config.yaml`（配置文件隔离即天然防污染）

### 5.2 市场环境切片
- [ ] 新增模块 `runner/validation/market_regime_slice.py`
- [ ] 切片参数**外置到 `config.yaml` 的 `regime_slice` 节点**（配置与代码分离），同时在加载时做**只读校验**防止过拟合篡改：

  ```yaml
  # config.yaml
  regime_slice:
    ADX_WINDOW: 20
    TREND_THRESHOLD: 25
    SMA_WINDOW: 60
    TREND_DEV_PCT: 0.05
    RANGE_DEV_PCT: 0.02
    VOL_HIGH_QUANTILE: 0.80
    VOL_LOW_QUANTILE: 0.20
  ```

- [ ] 加载期只读校验（运行时若被篡改则抛异常，防止优化器扫描）：

  ```python
  # runner/validation/market_regime_slice.py（伪代码）
  REGIME_CONFIG = load_config("config.yaml")["regime_slice"]
  EXPECTED = {"ADX_WINDOW": 20, "TREND_THRESHOLD": 25, "SMA_WINDOW": 60,
              "TREND_DEV_PCT": 0.05, "RANGE_DEV_PCT": 0.02,
              "VOL_HIGH_QUANTILE": 0.80, "VOL_LOW_QUANTILE": 0.20}
  for key, val in EXPECTED.items():
      assert REGIME_CONFIG[key] == val, (
          f"防过拟合：regime_slice.{key} 严禁修改（{val}）！"
      )
  ```

- [ ] 切片参数对照表：

  | 参数 | 标准值 | 含义 |
  |---|---|---|
  | `ADX_WINDOW` | 20 | ADX 回看周期 |
  | `TREND_THRESHOLD` | 25 | ADX 趋势/震荡分界 |
  | `SMA_WINDOW` | 60 | 中长期均线回看周期 |
  | `TREND_DEV_PCT` | 0.05 | 趋势市偏离阈值 |
  | `RANGE_DEV_PCT` | 0.02 | 震荡市偏离阈值 |
  | `VOL_HIGH_QUANTILE` | 0.80 | 高波动分位 |
  | `VOL_LOW_QUANTILE` | 0.20 | 低波动分位 |

- [ ] 切片规则（基于 `utils/indicators.py`）：
  - 趋势市：`ADX(20) > 25` AND `|close/SMA(60) - 1| > 5%`
  - 震荡市：`ADX(20) < 25` AND `|close/SMA(60) - 1| < 2%`
  - 高波动：`ATR(20) / close` 处于历史 80 分位以上
  - 低波动：`ATR(20) / close` 处于历史 20 分位以下
- [ ] 至少在 **3 种**环境下 Sharpe > 0
- [ ] **任意单一环境 Sharpe > -1.0**（禁止脆断）
- [ ] 产物：`output/validate/regime_slice_*.csv`

### 5.3 失败行动
- 5× 成本下负收益：降低交易频率、扩大 `entry_threshold`、合并信号
- 环境覆盖 < 3/4：识别策略的"友好环境"，增加反向信号或环境过滤
- 单环境 Sharpe < -1.0：脆断风险，禁止上线

---

## 6. 决策树（Go / No-Go）检查表

按顺序逐项打勾，任何一项"否"即停：

- [ ] **A** — 因子价值通过率 ≥ 50%？
  - 否 → **停止：回归因子研发**
  - 是 ↓
- [ ] **B** — 组合 abs(IC) > 0.03 且 ICIR > 0.5？
  - 否 → **调整：因子权重 / 非线性变换**
  - 是 ↓
- [ ] **C** — 组合 Sharpe > 0.5 且回撤 < 20%？
  - 否 → **调整：重新分组 / 动态权重**
  - 是 ↓
- [ ] **D** — 样本外 Sharpe 衰减 < 30%？
  - 否 → **调整：简化参数，降频**
  - 是 ↓
- [ ] **E** — 5× 成本仍正收益 且 3/4 环境 Sharpe > 0 且无脆断？
  - 否 → **调整：降频 / 合并信号**
  - 是 → **✅ 进入工程化阶段**

---

## 7. 验证产物清单

| 阶段 | 产物路径 | 内容 |
|---|---|---|
| A1 | `output/validate/factor_alpha24_screening.csv` | 24 因子 IC/IR/Pass 标记 |
| A2 | `output/validate/factor_ic_stability/*.csv` | 滚动 IC 时序 |
| A3 | `output/validate/factor_review_report.csv` | 6 项检查明细 |
| B | `output/validate/factor_combo_ic.csv` | 组合 IC / ICIR / 品种覆盖 |
| C1 | `output/backtest/e1_baseline_metrics.csv` | 单策略×多品种 Sharpe 矩阵 |
| C2 | `output/backtest/e2_e3_*.csv` | 多策略组合净值 |
| C3 | `output/backtest/cross_sectional_*` | 横截面组合净值 |
| D1 | `output/backtest/e6_walkforward_metrics.csv` | 滚动窗口 Sharpe |
| D2 | `output/backtest/e7_out_of_sample_metrics.csv` | 样本内/外对比 |
| D3 | `output/optimize/sensitivity_*.csv` | 参数扰动 Sharpe 变化 |
| D4 | `output/backtest/e8_bootstrap_*.csv` / `e9_monte_carlo_*.csv` | 置信区间 |
| E1 | `output/backtest/cost_5x_*.csv` | 5× 成本下净值 |
| E2 | `output/validate/regime_slice_*.csv` | 各环境下 Sharpe |

---

## 8. 配置文件准备清单

- [ ] `config.yaml`（基础配置，不修改）
- [ ] `config.cost_2x.yaml`（2× 成本独立配置）
- [ ] `config.cost_5x.yaml`（5× 成本独立配置）
- [ ] `config.cost_10x.yaml`（10× 成本独立配置）
- [ ] 确认入口脚本支持：
  - `python run_backtest.py --config <yaml_path>`
  - `python run_backtest.py --override key=value`
- [ ] 入口脚本需支持 `--override` 的文件：`run_backtest.py`、`run_optimize.py`、`run_validate.py`

---

## 9. 失败时的全局行动

> **不要继续优化回测系统**：再精良的引擎也无法让负期望策略盈利。

**回到因子研究**：
- [ ] 使用 `core/factors/alpha_futures/factor_pipeline.py::FactorPipeline` 对现有因子做非线性变换
- [ ] 尝试新数据源（订单流、期限结构精细化）
- [ ] 引入机器学习模型（滚动回归、XGBoost）合成因子

**重新设计子策略**：
- [ ] 基于有效因子重新分组（而非固定 5 类）
- [ ] 使用 `core/portfolio.py::PortfolioManager` 动态分配因子权重

**简化模型**：
- [ ] 减少参数数量（去除冗余 `lookback`/`threshold`）
- [ ] 降低交易频率（更长 `rebalance_freq`）
- [ ] 放宽 `entry_threshold`，减少过度交易

---

## 10. 涉及代码模块

| 模块 | 路径 | 备注 |
|---|---|---|
| 24 因子筛选 | `runner/validation/factor_alpha24.py` | A 阶段 |
| 滚动 IC 稳定性 | `runner/validation/factor_stability.py` | A 阶段 |
| 6 项因子复核 | `runner/validation/factor_review.py` | A 阶段 |
| 组合 IC（新增） | `runner/validation/factor_combo_ic.py` | B 阶段 |
| 市场环境切片（新增） | `runner/validation/market_regime_slice.py` | E 阶段 |
| 实验 E1-E5 | `runner/backtest/experiments/e1_e5.py` | C 阶段 |
| 实验 E6-E11 | `runner/backtest/experiments/e6_e11.py` | D 阶段 |
| 网格搜索 | `runner/optimization/grid_search.py` | D 阶段 |
| 窗口搜索 | `runner/optimization/window_search.py` | D 阶段 |
| 参数敏感性 | `runner/optimization/sensitivity.py` | D 阶段 |
| OOS 选择 | `runner/optimization/oos_selector.py` | D 阶段 |
| 公共指标 | `utils/indicators.py`、`utils/metrics.py` | 全阶段 |
| 核心因子 | `core/factors/alpha_futures/factor_engine.py`、`factor_registry.py` | A/B 阶段 |
| 因子管线 | `core/factors/alpha_futures/factor_pipeline.py` | 失败回退 |
| 组合管理 | `core/portfolio.py::PortfolioManager` | C 阶段 / 失败回退 |
| 策略库 | `core/strategy_registry.py::StrategyLibrary` | C 阶段 |
| 横截面策略 | `core/strategies/cross_sectional.py` | C 阶段 |
| 自研引擎 | `core/engine/runner.py` | 交叉验证 |
| PyBroker 引擎 | `core/engine/backtest_runner.py` | 主回测 |

---

## 11. 与其他规则的关联

- **规则 1 / 26**：双引擎交叉验证在 D 阶段执行
- **规则 9**：阶段 A 的 6 项复核清单
- **规则 15**：本计划是规则 15 在策略价值层面的扩展（5 阶段硬性验证）
- **规则 22**：阶段 D 的滚动窗口基准

---

*本文档与规则 28 保持同步；规则更新时同步刷新本文档。*
