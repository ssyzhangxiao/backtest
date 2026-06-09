# 策略价值验证 To-Do 执行清单

> **配套文档**：[strategy_validation_plan.md](./strategy_validation_plan.md)（含详细阈值、命令、产物路径）
> **规则依据**：[规则 28](../../.trae/rules/05-validation/28-strategy-value-validation.md)
> **创建日期**：2026-06-08
> **执行模式**：必须按阶段顺序执行，前一阶段未通过禁止进入下一阶段

## 状态图例

- ⬜ **Pending**：待开始
- 🔄 **In Progress**：进行中
- ✅ **Done**：已完成
- ❌ **Blocked**：未通过，需回退

---

## 阶段 A — 因子价值验证（前置门）

> **Go 阈值**：通过率 ≥ 50%，集合平均 abs(IC) > 0.04，单因子 abs(IC) > 0.03 且 IR > 0.5
> **失败回退**：回到 `core/factors/alpha_futures/` 研发新因子

| # | 任务 | 命令 | 产物 | 状态 |
|---|---|---|---|---|
| A-01 | 运行 24 因子 IC/IR 筛选 | `python run_validate.py --method factor_alpha24` | `output/validate/factor_alpha24_screening.csv` | ⬜ |
| A-02 | 核对通过率 ≥ 12/24 | — | — | ⬜ |
| A-03 | 核对单因子 abs(IC) > 0.03 且 IR > 0.5 | — | — | ⬜ |
| A-04 | 核对集合平均 abs(IC) > 0.04 | — | — | ⬜ |
| A-05 | 核对因子间最大互相关 < 0.6 | — | — | ⬜ |
| A-06 | 反向因子额外确认 ICIR > 0.5 | — | — | ⬜ |
| A-07 | 运行滚动 IC 时序稳定性 | `python run_validate.py --method factor_ic` | `output/validate/factor_ic_stability/*.csv` | ⬜ |
| A-08 | 核对滚动 60 天 IC 方差 < 0.05 | — | — | ⬜ |
| A-09 | 确认无大段年份变号 | — | — | ⬜ |
| A-10 | 运行 6 项因子复核 | `python run_validate.py --method factor_review` | `output/validate/factor_review_report.csv` | ⬜ |
| A-11 | 核对数据存活率 ≥ 85%、缺失值 ≤ 15% | — | — | ⬜ |
| A-12 | 核对因子计算参数敏感性 ±20% 后 IC 衰减 < 30% | — | — | ⬜ |
| A-13 | 核对 Barra 风格因子相关性 ≤ 0.5 | — | — | ⬜ |

**🚦 阶段 A 决策**：
- ✅ A-01 ~ A-13 全部通过 → 进入阶段 B
- ❌ 任意关键项未通过 → 停止，回归因子研发

---

## 阶段 B — 多因子组合 IC

> **Go 阈值**：组合 abs(IC) > 0.03，ICIR > 0.5，至少 60% 品种为正
> **依赖**：阶段 A 通过

| # | 任务 | 文件/命令 | 状态 |
|---|---|---|---|
| B-01 | 新建 `runner/validation/factor_combo_ic.py` | — | ⬜ |
| B-02 | 实现 `factor_combo_ic_validation(data_source, config, lib, output_dir, **kwargs)` | — | ⬜ |
| B-03 | 委托 `core/factors/alpha_futures/factor_engine.py` 计算因子值 | — | ⬜ |
| B-04 | 委托 `core/engine/rolling_ic.py` 计算组合 IC/ICIR | — | ⬜ |
| B-05 | 实现等权 + IC 加权两种合成方式 | — | ⬜ |
| B-06 | 实现跨品种横截面 Z 分数标准化 | — | ⬜ |
| B-07 | 在 `runner/validation/__init__.py` 注册 `VALIDATOR_MAP["factor_combo_ic"]` | — | ⬜ |
| B-08 | 运行组合 IC 验证 | `python run_validate.py --method factor_combo_ic` | ⬜ |
| B-09 | 核对组合 abs(IC) > 0.03 | — | ⬜ |
| B-10 | 核对组合 ICIR > 0.5 | — | ⬜ |
| B-11 | 核对品种覆盖 ≥ 60% | — | ⬜ |
| B-12 | 归档产物 | `output/validate/factor_combo_ic.csv` | ⬜ |

**🚦 阶段 B 决策**：
- ✅ B-01 ~ B-12 全部通过 → 进入阶段 C
- ❌ 任意关键项未通过 → 调整因子权重/非线性变换/进入新因子研发

---

## 阶段 C — 策略回测验证

> **Go 阈值**：组合 Sharpe > 0.5，回撤 < 20%，胜率 > 45%
> **依赖**：阶段 B 通过

| # | 任务 | 命令 | 产物 | 状态 |
|---|---|---|---|---|
| C-01 | 运行 E1 单策略多品种基线 | `python run_backtest.py --experiment e1` | `output/backtest/e1_baseline_metrics.csv` | ⬜ |
| C-02 | 核对 70% 品种 Sharpe > 0 | — | — | ⬜ |
| C-03 | 核对策略×品种矩阵平均 Sharpe > 0.3 | — | — | ⬜ |
| C-04 | 核对胜率 > 45% | — | — | ⬜ |
| C-05 | 运行 E2 等权组合 | `python run_backtest.py --experiment e2` | `output/backtest/e2_e3_*.csv` | ⬜ |
| C-06 | 运行 E3 动态权组合（rolling IC 加权） | `python run_backtest.py --experiment e3` | — | ⬜ |
| C-07 | 核对组合 Sharpe > 0.5 | — | — | ⬜ |
| C-08 | 核对最大回撤 < 20% | — | — | ⬜ |
| C-09 | 核对胜率 > 45% | — | — | ⬜ |
| C-10 | 运行横截面多策略 | `python run_backtest.py --cross-sectional` | `output/backtest/cross_sectional_*` | ⬜ |
| C-11 | 核对组合 Sharpe > 0.5、最大回撤 < 20%、胜率 > 45% | — | — | ⬜ |
| C-12 | 核对年化收益 > 8% | — | — | ⬜ |
| C-13 | 核对与单品种策略相关性 < 0.5 | — | — | ⬜ |

**🚦 阶段 C 决策**：
- ✅ C-01 ~ C-13 全部通过 → 进入阶段 D
- ❌ 任意关键项未通过 → 重新分组子策略 / `PortfolioManager` 动态权重 / 加强风控

---

## 阶段 D — 稳健性验证（防过拟合）

> **Go 阈值**：样本外 Sharpe 衰减 < 30%，参数扰动 Sharpe 变化 < 30%，Bootstrap/MC 95% 区间不含 0，双引擎相关 > 0.99
> **依赖**：阶段 C 通过

| # | 任务 | 命令 | 产物 | 状态 |
|---|---|---|---|---|
| D-01 | 运行 E6 Walk-Forward | `python run_backtest.py --experiment e6` | `output/backtest/e6_walkforward_metrics.csv` | ⬜ |
| D-02 | 核对测试期 Sharpe 均值 > 0.3、衰减 < 30% | — | — | ⬜ |
| D-03 | 运行 E7 样本外验证 | `python run_backtest.py --experiment e7` | `output/backtest/e7_out_of_sample_metrics.csv` | ⬜ |
| D-04 | 核对样本外 2023-01-01 至 2026-06-07 | — | — | ⬜ |
| D-05 | 核对样本外 Sharpe > 0、回撤 ≤ 样本内 1.5× | — | — | ⬜ |
| D-06 | 运行交易执行参数敏感性 | `python run_optimize.py --method sensitivity` | `output/optimize/sensitivity_*.csv` | ⬜ |
| D-07 | 核对 `rebalance_freq`/`stop_loss_pct`/`entry_threshold`/`position_cap` ±20% 扰动后 Sharpe 变化 < 30% | — | — | ⬜ |
| D-08 | 运行 E8 Bootstrap 5000 次 | `python run_backtest.py --experiment e8` | `output/backtest/e8_bootstrap_*.csv` | ⬜ |
| D-09 | 运行 E9 Monte Carlo 1000 次 | `python run_backtest.py --experiment e9` | `output/backtest/e9_monte_carlo_*.csv` | ⬜ |
| D-10 | 核对 Sharpe 95% 置信区间不含 0 | — | — | ⬜ |
| D-11 | 启用 `BacktestConfig.cross_validate = True` | — | — | ⬜ |
| D-12 | 运行双引擎交叉验证 | 同时调用 `PyBrokerBacktestRunner.run()` + `BacktestRunner.run()` | `output/backtest/dual_engine_diff_report.csv` | ⬜ |
| D-13 | 核对归一化净值 Pearson 相关 > 0.99、收益偏差 < 1% | — | — | ⬜ |
| D-14 | 若偏离超阈值，定位最大偏离日期 + 打印前 N 笔差异交易 | — | — | ⬜ |

**🚦 阶段 D 决策**：
- ✅ D-01 ~ D-14 全部通过 → 进入阶段 E
- ❌ 过拟合 → 简化参数、降频
- ❌ 参数扰动 > 30% → 引入正则化
- ❌ 显著性不足 → 扩充样本或降低复杂度
- ❌ 双引擎偏离 → 复核逐笔交易（规则 26）

---

## 阶段 E — 成本与环境抗噪

> **Go 阈值**：5× 成本仍正收益，3/4 环境 Sharpe > 0，任意单一环境 Sharpe > -1.0
> **依赖**：阶段 D 通过

### E.1 交易成本敏感度

| # | 任务 | 命令 | 状态 |
|---|---|---|---|
| E-01 | 创建 `config.cost_2x.yaml` | — | ⬜ |
| E-02 | 创建 `config.cost_5x.yaml` | — | ⬜ |
| E-03 | 创建 `config.cost_10x.yaml` | — | ⬜ |
| E-04 | 确认 `run_backtest.py` 支持 `--config` 参数 | — | ⬜ |
| E-05 | 运行 2× 成本回测 | `python run_backtest.py --config config.cost_2x.yaml --experiment e1` | ⬜ |
| E-06 | 运行 5× 成本回测 | `python run_backtest.py --config config.cost_5x.yaml --experiment e1` | ⬜ |
| E-07 | 运行 10× 成本回测 | `python run_backtest.py --config config.cost_10x.yaml --experiment e1` | ⬜ |
| E-08 | 归档 `cost_2x_*.csv` / `cost_5x_*.csv` / `cost_10x_*.csv` | — | ⬜ |
| E-09 | 核对 5× 成本下仍为正收益、Sharpe 衰减 < 50% | — | ⬜ |

### E.2 市场环境切片

| # | 任务 | 命令/位置 | 状态 |
|---|---|---|---|
| E-10 | 新建 `runner/validation/market_regime_slice.py` | — | ⬜ |
| E-11 | 在 `config.yaml` 添加 `regime_slice` 节点 | `config.yaml` | ⬜ |
| E-12 | 实现加载期只读 assert 校验（防过拟合篡改抛异常） | `market_regime_slice.py` | ⬜ |
| E-13 | 实现趋势市切片（ADX > 25 + |偏离| > 5%） | `market_regime_slice.py` | ⬜ |
| E-14 | 实现震荡市切片（ADX < 25 + |偏离| < 2%） | `market_regime_slice.py` | ⬜ |
| E-15 | 实现高/低波动切片（ATR/close 80/20 分位） | `market_regime_slice.py` | ⬜ |
| E-16 | 运行市场环境切片验证 | `python run_validate.py --method market_regime_slice` | ⬜ |
| E-17 | 核对 ≥ 3/4 环境 Sharpe > 0 | — | ⬜ |
| E-18 | 核对任意单一环境 Sharpe > -1.0 | — | ⬜ |
| E-19 | 归档 `regime_slice_*.csv` | — | ⬜ |

**🚦 阶段 E 决策**：
- ✅ E-01 ~ E-19 全部通过 → **进入工程化阶段**（滑点模型、实时风控、模拟盘）
- ❌ 5× 成本负收益 → 降频/合并信号
- ❌ 环境覆盖 < 3/4 → 增加反向信号/环境过滤
- ❌ 单环境脆断 → 禁止上线

---

## 全局准备任务（任何阶段执行前必须完成）

| # | 任务 | 状态 |
|---|---|---|
| P-01 | 确认 `core/config/backtest_config.py::BacktestConfig.from_yaml()` 可正常加载 | ⬜ |
| P-02 | 创建 `config.cost_2x.yaml` / `config.cost_5x.yaml` / `config.cost_10x.yaml` | ⬜ |
| P-03 | 在 `config.yaml` 添加 `regime_slice` 节点（7 项常量） | ⬜ |
| P-04 | 确认 `run_backtest.py`/`run_optimize.py`/`run_validate.py` 支持 `--config` 参数 | ⬜ |
| P-05 | 确认 `BacktestConfig.cross_validate` 开关可用 | ⬜ |
| P-06 | 创建 `output/validate/` 和 `output/backtest/` 目录 | ⬜ |
| P-07 | 确认 `core/factors/alpha_futures/factor_engine.py` 可加载 24 因子 | ⬜ |
| P-08 | 确认 `core/portfolio.py::PortfolioManager` 可调用 | ⬜ |

---

## 进度看板

| 阶段 | 总任务 | 已完成 | 完成率 | 状态 |
|---|---|---|---|---|
| 全局准备 (P) | 8 | 0 | 0% | ⬜ |
| 阶段 A | 13 | 0 | 0% | ⬜ |
| 阶段 B | 12 | 0 | 0% | ⬜ |
| 阶段 C | 13 | 0 | 0% | ⬜ |
| 阶段 D | 14 | 0 | 0% | ⬜ |
| 阶段 E | 19 | 0 | 0% | ⬜ |
| **合计** | **79** | **0** | **0%** | ⬜ |

---

## 执行顺序（推荐）

1. **P-01 ~ P-08**：环境准备（一次性，约 1 天）
2. **A-01 ~ A-13**：阶段 A（最关键，决定因子库价值）
3. **B-01 ~ B-12**：阶段 B（含新模块开发，约 1-2 天）
4. **C-01 ~ C-13**：阶段 C（端到端回测）
5. **D-01 ~ D-14**：阶段 D（防过拟合验证）
6. **E-01 ~ E-19**：阶段 E（成本与环境抗噪）
7. **决策树检查表**：最终 Go/No-Go 决策

---

## 关键依赖关系

```
P-01~P-08 (准备)
    ↓
阶段 A (因子)
    ↓ (Go)
阶段 B (组合IC)
    ↓ (Go)
阶段 C (策略回测)
    ↓ (Go)
阶段 D (稳健性)
    ↓ (Go)
阶段 E (抗噪)
    ↓ (Go)
✅ 进入工程化阶段
```

**任何阶段未通过 → 回归前一阶段研发，禁止跳级**

---

## 更新日志

| 日期 | 变更 | 操作人 |
|---|---|---|
| 2026-06-08 | 初版生成（79 项任务） | — |
