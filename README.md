# 量化回测系统 v3.2 — 5子策略体系 + 24因子库

## 概述

基于 24 因子（5 大类）+ 5 子策略体系的多因子期货量化回测系统。

核心特性：
- **5子策略体系**：趋势、期限结构、均值回归、波动率突破、复合共振
- **24因子库**：趋势 T_01~T_05、回归 R_01~R_05、波动率 V_01~V_04、资金流 M_01~M_05、高阶复合 H_01~H_05
- **多策略集成**：支持4种信号合并方法（等权、波动率倒数、自适应、多数投票）
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
  initial_capital: 1000000
  rebalance_freq: 3               # 每3个交易日调仓
  commission: 0.0003
  slippage: 0.0002
  stop_loss_pct: 0.05
  signal_merge_method: equal_weight  # 信号合并方法：equal_weight/volatility_inverse/adaptive/majority_vote

symbols:
  - "SHFE.RB"   # 螺纹钢
  - "DCE.M"     # 豆粕
  - "DCE.MA"    # 甲醇
  - "CZCE.TA"   # PTA
```

### 3. 运行回测

```bash
# 完整回测
python run_backtest.py

# 参数优化
python run_optimize.py

# 验证
python run_validate.py
```

---

## 项目结构

```
backtest/
├── config.yaml                  # 统一配置文件
├── run_backtest.py              # 回测入口脚本
├── run_optimize.py              # 参数优化脚本
├── run_validate.py              # 验证脚本
├── core/
│   ├── config/                  # 配置模块
│   │   └── backtest_config.py   # BacktestConfig 配置类
│   ├── strategy_registry.py     # 策略注册表 + 策略库
│   ├── strategies/              # 策略实现
│   │   ├── base.py              # 策略基类
│   │   ├── cross_sectional.py   # 横截面打分引擎
│   │   └── sub_strategies/     # 5子策略目录
│   │       ├── base.py         # 子策略基类
│   │       ├── trend.py        # 趋势策略
│   │       ├── term_structure.py  # 期限结构策略
│   │       ├── mean_reversion.py  # 均值回归策略
│   │       ├── vol_breakout.py    # 波动率突破策略
│   │       └── composite.py     # 复合共振策略
│   ├── engine/                  # 回测引擎
│   │   ├── pybroker_data_source.py  # PyBroker 数据源
│   │   ├── strategy_executor.py # 策略执行器
│   │   ├── backtest_runner.py   # PyBroker 回测运行器
│   │   ├── switch_engine.py     # 因子打分调仓引擎
│   │   ├── top_level_integrator.py  # 顶层策略集成器
│   │   └── sub_strategy_adapter.py  # 子策略适配器
│   └── factors/                 # 因子模块
│       ├── alpha_futures/      # 24因子库
│       └── alpha_futures_24.py
├── runner/                      # 编排层
│   ├── pipeline.py             # Pipeline 编排器
│   ├── data/
│   ├── optimization/
│   ├── validation/
│   └── report/
├── utils/                       # 工具函数
└── data/                        # 数据目录
```

---

## 策略说明

### 5子策略体系

| 子策略名称 | 使用的因子 | 逻辑核心 | 信号方向 |
|------------|------------|----------|-----------|
| 趋势策略 | T_01, T_02, T_03, T_05, V_02, M_03 | 趋势确认 + 资金流向确认 | 顺势交易 |
| 期限结构策略 | T_04, R_04, M_04, H_05 | Carry + 增仓/资金流共振 | Back做多，Contango做空 |
| 均值回归策略 | R_01, R_02, R_03, R_05, H_03 | 增仓背离、持仓萎缩反转 | 逆势交易 |
| 波动率突破策略 | V_01, V_03, V_04, H_04 | 持仓异动 + 价格加速度 | 突破跟进 |
| 复合共振策略 | H_01, H_02, M_01, M_02, M_05 | 多维度高阶统计共振 | 综合打分 |

### 信号合并方法

- **等权叠加**（默认）：`final_signal = (signal1 + ... + signal5) / 5`
- **波动率倒数加权**：降低高波动子策略权重
- **基于收益率的自适应权重**：使用滚动优化最大化综合 Sharpe 比
- **多数投票**：取多数方向作为最终方向

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
因子计算（24因子库）
    ↓
子策略信号生成（5子策略）
    ↓
多策略集成（信号合并）
    ↓
风控过滤
    ↓
绩效评估
    ↓
稳健性验证
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
| 策略档案迁移 | 删除 core/strategy_registry.py，迁移至 core/config/strategy_profiles.py（更准确反映"元数据"本质） |
| 风控统一 | 删除 RiskManager，保留 RiskController（纯逻辑）+ RiskManagerAdapter（PyBroker适配） |
| 引擎拆分 | broker_adapter.py 拆分为 pybroker_data_source/regime_indicator/strategy_executor/backtest_runner |
| 测试更新 | 删除过时测试（rebalance_frequency），更新配置测试 |

### v3.2 (2026-06-06)

| 项 | 说明 |
|----|------|
| 子策略体系 | 5子策略体系（趋势/期限结构/均值回归/波动率突破/复合共振） |
| 顶层集成器 | `core/engine/top_level_integrator.py`，支持4种信号合并方法 |
| 子策略适配器 | `core/engine/sub_strategy_adapter.py`，连接因子库和子策略 |
| 旧体系移除 | 移除旧4因子体系（ts_momentum/roll_yield/alpha019/alpha032） |
| 配置升级 | 新增 `signal_merge_method` 配置项，移除旧灰度开关 |
