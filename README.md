# 量化回测系统 v3.0 — CTA + Alpha101 多因子策略

## 概述

基于 CTA 因子 + Alpha101 因子组合的多因子期货量化回测系统。

核心特性：
- **4因子策略**：时间序列动量、展期收益、Alpha#019、Alpha#032
- **信号融合**：多策略加权信号合成，纯因子打分调仓
- **环境模块降级**：市场环境检测保留为辅助分析工具，主流程不再依赖环境判断
- **完整回测链**：WalkForward、Bootstrap、蒙特卡洛、样本外验证

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置参数

编辑 `config.yaml`，核心配置：

```yaml
backtest:
  initial_cash: 1000000
  rebalance_freq: 3               # 每3个交易日调仓
  commission: 0.0003
  slippage: 0.0002
  stop_loss_pct: 0.05

factor_weights:
  ts_momentum: 0.25
  roll_yield: 0.25
  alpha019: 0.25
  alpha032: 0.25

symbols:
  - "SHFE.RB"   # 螺纹钢
  - "DCE.M"     # 豆粕
  - "DCE.MA"    # 甲醇
  - "CZCE.TA"   # PTA
```

### 3. 运行回测

```bash
# 完整回测（10个实验）
python run_full_backtest.py

# 参数优化
python run_parameter_optimization.py

# 验证（WalkForward + 样本外 + 蒙特卡洛）
python run_validation.py

# Streamlit 界面
streamlit run app.py
```

---

## 项目结构

```
backtest/
├── config.yaml                  # 统一配置文件
├── run_full_backtest.py         # 完整回测脚本（E1-E10实验）
├── run_parameter_optimization.py # 参数优化脚本
├── run_validation.py            # 验证脚本（WalkForward/样本外/蒙特卡洛）
├── app.py                       # Streamlit Web 界面
├── core/
│   ├── __init__.py              # 核心模块导出
│   ├── config.py                # BacktestConfig 配置类
│   ├── data_loader.py           # 数据加载与展期处理
│   ├── environment.py           # 环境适配器
│   ├── optimizer.py             # 参数优化器
│   ├── portfolio.py             # 组合管理
│   ├── report_builder.py        # 报告生成
│   ├── risk_controller.py       # 风控逻辑（纯逻辑类）
│   ├── risk_manager.py          # 风控兼容层（→ RiskManagerAdapter）
│   ├── strategy_registry.py     # 策略注册表 + 策略库
│   ├── rollover.py              # 展期管理
│   ├── market_regime/           # 市场环境检测（辅助分析工具）
│   ├── strategies/              # 策略实现
│   │   ├── base.py              # 策略基类
│   │   ├── ts_momentum.py       # 时间序列动量策略
│   │   ├── roll_yield.py        # 展期收益策略
│   │   ├── alpha019.py          # Alpha#019 策略
│   │   └── alpha032.py          # Alpha#032 策略
│   ├── engine/                  # 回测引擎
│   │   ├── broker_adapter.py    # PyBroker 适配器（聚合导入层）
│   │   ├── pybroker_data_source.py  # PyBroker 数据源
│   │   ├── regime_indicator.py  # 环境指标
│   │   ├── strategy_executor.py # 策略执行器 + 风控适配
│   │   ├── backtest_runner.py   # PyBroker 回测运行器
│   │   ├── runner.py            # 自研回测引擎
│   │   └── switch_engine.py     # 因子打分调仓引擎
│   └── performance/             # 绩效评估
├── components/                  # Streamlit 组件
├── pages/                       # Streamlit 页面
├── utils/                       # 工具函数
├── examples/                    # 示例脚本
└── data/                        # 数据目录
```

---

## 策略说明

### CTA 侧（50%权重）

| 策略 | 因子 | 信号逻辑 | 默认参数 |
|------|------|---------|---------|
| ts_momentum | 时间序列动量 | N日累计收益率>0做多，<0做空 | window=20 |
| roll_yield | 展期收益 | 价差偏离均线超阈值反向开仓 | lookback=20, entry=2.0% |

### Alpha101 侧（50%权重）

| 策略 | 因子 | 信号逻辑 | 默认参数 |
|------|------|---------|---------|
| alpha019 | Alpha#019 | 短期价格变化符号×长期累计收益排名 | short=7, long=250 |
| alpha032 | Alpha#032 | 均线偏离+VWAP相关性 | ma=7, corr=230 |

### 权重分配

平衡型：CTA侧 50%（各25%）+ Alpha101侧 50%（各25%），固定权重。

---

## 风控机制

| 规则 | 参数 | 说明 |
|------|------|------|
| 单品种止损 | 5% | 亏损达5%强制平仓，当日不再开仓 |
| 手续费过滤 | 0.1% | 预期收益低于双边成本0.1%则跳过 |
| 涨跌停保护 | 开启 | 触及涨跌停不开新仓 |
| 最大持仓数 | 6 | 多+空合计不超过6个品种 |
| 保证金占比 | 10% | 每品种分配总资金10%作为保证金 |
| 最大回撤清盘 | 25% | 组合回撤超25%全部清仓 |

---

## 回测流程

```
数据加载（TqSdk优先 → CSV兜底）
    ↓
策略实例化（4因子策略注册指标）
    ↓
信号融合（加权合成 → 纯因子打分调仓）
    ↓
风控过滤（止损/手续费/涨跌停/持仓限制）
    ↓
绩效评估（Sharpe/最大回撤/胜率/盈亏比）
    ↓
稳健性验证（WalkForward/Bootstrap/蒙特卡洛）
```

---

## 修改记录

### v3.0 (2026-05-31)

| 项 | 说明 |
|----|------|
| 策略重构 | 拆分为4个独立策略：ts_momentum/roll_yield/alpha019/alpha032 |
| 信号融合 | 默认启用融合模式，纯因子打分调仓 |
| 环境降级 | 环境检测保留为辅助工具，主流程不再依赖环境判断 |
| 品种池 | RB/M/MA/TA（流动性>5万手，保证金<5000元） |
| 调仓周期 | 每3个交易日 |
| 风控增强 | 新增手续费过滤、涨跌停保护、最大持仓数限制 |
| 代码清理 | 移除旧策略（dual_ma/rsi/vol_breakout/spread），清理冗余引用 |

### v3.1 (2026-06-01)

| 项 | 说明 |
|----|------|
| 配置清理 | 删除 fusion_mode、regime_filter_enabled、strategy_switching 等废弃字段 |
| 策略注册合并 | 删除 core/strategies/registry.py 和 core/strategy_library/，合并为 core/strategy_registry.py |
| 风控统一 | 删除 RiskManager，保留 RiskController（纯逻辑）+ RiskManagerAdapter（PyBroker适配） |
| 引擎拆分 | broker_adapter.py 拆分为 pybroker_data_source/regime_indicator/strategy_executor/backtest_runner |
| 测试更新 | 删除过时测试（rebalance_frequency），更新配置测试 |
