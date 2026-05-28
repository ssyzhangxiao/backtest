# 量化回测系统 v2.0 - 使用说明

## 🚀 概述

按照8项要求全面优化后的回测系统，主要特性：

1. ✅ **安全配置**：TqSdk凭证从环境变量读取，所有参数外置到`config.yaml`
2. ✅ **缺失模块补齐**：重写`create_hybrid_data_source`（TqSdk→CSV兜底），实现`PyBrokerBacktestRunner`
3. ✅ **策略切换**：重写`MarketRegimeDetector`，实现3状态判断（趋势/震荡/高波），策略映射确保切换次数>0
4. ✅ **风控加固**：固定止损-5%，最大回撤-25%清盘
5. ✅ **WalkForward并行**：使用`concurrent.futures`并行执行各窗口
6. ✅ **Bootstrap增强**：5000样本，绘制Sharpe比率分布图
7. ✅ **HTML报告**：生成带图表（净值、回撤、柱状图）的可视化报告
8. ✅ **错误处理**：`loguru`日志记录，单个品种失败不崩溃

---

## 📦 快速开始

### 1. 安装依赖

```bash
pip install pandas numpy matplotlib pybroker pyyaml loguru
```

### 2. 设置TqSdk凭证（可选）

```bash
# macOS/Linux
export TQSDK_PHONE="your_phone"
export TQSDK_PASSWORD="your_password"

# Windows
set TQSDK_PHONE=your_phone
set TQSDK_PASSWORD=your_password
```

如果未设置，系统会自动使用本地CSV数据。

### 3. 准备数据

在`./data/`目录下放置CSV文件，格式：
- 文件名：`交易所.品种.csv`，如`SHFE.RB.csv`
- 列名：`date/datetime, open, high, low, close, volume`

### 4. 配置参数

编辑`config.yaml`，根据需要调整回测参数：

```yaml
backtest:
  initial_cash: 1000000          # 初始资金
  full_start_date: "2016-01-01"
  full_end_date: "2026-05-01"

strategy_switching:
  cool_down_days: 20             # 切换冷却期
  regime_map:
    trend: "dual_ma"
    range: "rsi"
    high_vol: "vol_breakout"

risk_management:
  stop_loss_pct: 0.05            # 单笔止损-5%
  max_drawdown_pct: 0.25         # 最大回撤-25%清盘

bootstrap:
  n_samples: 5000                # Bootstrap样本数
```

### 5. 运行回测

```bash
python run_pybroker_full_backtest_v2.py
```

---

## 📁 文件说明

### 新增文件

| 文件 | 说明 |
|------|------|
| `config.yaml` | 所有参数配置文件 |
| `run_pybroker_full_backtest_v2.py` | 优化版主回测文件 |
| `V2_README.md` | 本文档 |

### 核心模块（v2文件内）

| 模块 | 类/函数 | 说明 |
|------|---------|------|
| 配置加载 | `load_config()` | 从YAML加载配置 |
| 数据源 | `HybridDataSource` | TqSdk优先，CSV兜底 |
| 市场环境 | `MarketRegimeDetector` | 3状态分类：trend/range/high_vol |
| 策略 | `create_strategy_functions()` | dual_ma/rsi/vol_breakout |
| 风控 | `RiskManager` | 固定止损+最大回撤清盘 |
| 回测运行器 | `PyBrokerBacktestRunner` | 单策略/融合/切换 |
| WalkForward | `run_walkforward_parallel()` | 并行窗口优化 |
| Bootstrap | `run_bootstrap()` | 5000样本+Sharpe分布图 |
| 报告 | `generate_html_report()` | 可视化HTML报告 |

---

## 📊 输出文件

运行完成后，`./output_backtest_pybroker/`目录下：

| 文件 | 说明 |
|------|------|
| `data_summary.csv` | 数据摘要 |
| `e1_equity_*.csv` | 单策略净值曲线 |
| `e1_trades_*.csv` | 单策略交易记录 |
| `e2_equity_fusion.csv` | 信号融合净值 |
| `e4_equity_switching.csv` | 策略切换净值 |
| `e4_switch_log.csv` | 策略切换日志 |
| `e5_walkforward.csv` | Walkforward窗口结果 |
| `e8_bootstrap_samples.csv` | Bootstrap样本 |
| `bootstrap_sharpe_distribution.png` | Sharpe分布图 |
| `experiment_comparison.png` | 实验对比图 |
| `equity_curves.png` | 净值曲线对比 |
| `all_metrics.csv` | 所有实验指标汇总 |
| `backtest_report.html` | **可视化HTML报告** |

---

## 🎯 使用示例

### 示例1：调整参数后回测

```yaml
# config.yaml
backtest:
  initial_cash: 2000000

strategies:
  - name: "dual_ma"
    params:
      short_ma: 5
      long_ma: 20
```

```bash
python run_pybroker_full_backtest_v2.py
```

### 示例2：查看HTML报告

用浏览器打开：
```
./output_backtest_pybroker/backtest_report.html
```

报告包含三个标签页：
- 📌 概览：配置说明+核心改进
- 📈 图表：实验对比+净值曲线
- 📋 指标：绩效指标表格

### 示例3：查看日志

```bash
# 主日志
tail -f ./logs/backtest.log

# 错误日志
tail -f ./logs/error.log
```

---

## 🔧 关键配置项说明

### 市场环境分类

```yaml
market_regime:
  volatility_window: 20        # 波动率窗口
  adx_period: 14               # ADX周期
  trend_threshold: 25          # 趋势阈值（ADX>25认为有趋势）
  vol_high_percentile: 0.7     # 高波阈值（70%分位数）
  vol_low_percentile: 0.3      # 低波阈值（30%分位数）
```

### 策略切换

```yaml
strategy_switching:
  enabled: true
  cool_down_days: 20           # 冷却期：防止频繁切换
  regime_map:
    trend: "dual_ma"           # 趋势市用双均线
    range: "rsi"               # 震荡市用RSI
    high_vol: "vol_breakout"   # 高波市用波动率突破
```

### WalkForward并行

```yaml
walk_forward:
  train_years: 2
  test_years: 1
  step_years: 1
  parallel: true               # 开启并行
  max_workers: 4               # 最大进程数
```

---

## 📈 回测流程

```
Phase 1: 数据加载
    └─> TqSdk优先 -> 失败则CSV兜底

Phase 2: 单策略基线
    └─> dual_ma / rsi / vol_breakout 独立回测

Phase 3: 信号融合
    └─> 多策略等权信号融合

Phase 4: 策略切换
    └─> 3状态分类 -> 策略映射 -> 冷却期控制

Phase 5: WalkForward并行
    └─> 多进程并行执行时间窗口

Phase 6: 样本内外验证
    └─> in_sample / out_sample 分开验证

Phase 7: Bootstrap
    └─> 5000样本 -> Sharpe分布直方图

Phase 8: HTML报告
    └─> 图表+指标完整报告
```

---

## 🛡️ 风控机制

1. **单笔止损**：任意单品种亏损-5%强制平仓
2. **最大回撤清盘**：组合最大回撤超-25%全部清仓，停止交易
3. **仓位控制**：配置文件中可设置`position_limit_pct`

---

## 📝 修改记录

### v2.0 (2026-05-28)

| 项 | 说明 |
|----|------|
| 安全配置 | TqSdk凭证从环境变量读取，config.yaml外置参数 |
| 数据加载 | HybridDataSource：TqSdk优先，CSV兜底 |
| 策略切换 | MarketRegimeDetector 3状态 + 策略映射 |
| 风控 | RiskManager：-5%止损，-25%清盘 |
| WalkForward | concurrent.futures并行执行 |
| Bootstrap | 5000样本 + Sharpe分布图 |
| 报告 | 可视化HTML，带图表 |
| 日志 | loguru记录，错误不崩溃 |

---

## ❓ 常见问题

### Q: TqSdk连接失败怎么办？

A: 系统会自动fallback到本地CSV，无需担心。只要`./data/`目录下有数据即可运行。

### Q: 单个品种报错会导致整个回测停止吗？

A: 不会。`try-except`包裹了每个品种，单个失败会记录到`./logs/error.log`，其他继续。

### Q: 如何调整策略参数？

A: 编辑`config.yaml`中的`strategies.params`即可，无需改代码。

### Q: Bootstrap样本数能改吗？

A: 可以，在`config.yaml`的`bootstrap.n_samples`调整（建议至少1000）。

---

## 📄 License

MIT License
