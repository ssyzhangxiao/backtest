# L1 实盘模拟 — 离线日级回放（2026-06-19）

**创建时间**：2026-06-19
**目标**：低成本验证实盘撮合流程（按次日开盘价成交），对比 e12 回测，估算滑点。

## 实施情况

### 已完成
- ✅ `runner/live/daily_replay.py` (222 行)：日级回放核心逻辑
- ✅ `runner/live/__init__.py` (115 行)：DailyReplaySimulator 入口类
- ✅ `scripts/run_l1_daily_replay.py` (76 行)：9 品种批量验证脚本
- ✅ 跑通 1 个月（2024-12-01 ~ 2025-01-15，约 32 个交易日）

### L1 简化模型的根本限制（关键发现）

L1 用**反推 position** 模拟实盘撮合，但**反推精度差**：

| 指标 | 真实回测 | L1 模拟 | 偏差倍数 |
|------|---------|--------|---------|
| 总 PnL | +11,977 | -441,496 | **37 倍** |
| 平均方向胜率 | — | 43.4% | （比 50% 随机还低） |

**根本原因**：
1. K（资金敞口换算）用 `initial_capital / initial_price`，但 e12 实际仓位受 `max_position_pct=0.15` 限制
2. e12 内部 PnL 计算含 commission、stop loss、signal 缩放，不是简单 `pos × price_diff`
3. 5 天调仓周期内 PnL 累加 = pos × (close[end] - close[start]) × K，但 K 不是常数
4. 简化模型**无法重建** e12 的精细撮合逻辑

### L1 能做的定性分析

虽然 PnL 偏差大，但 L1 仍能提供有价值的**定性信息**：

| 指标 | 含义 | 用途 |
|------|------|------|
| `long_pct` / `short_pct` | 调仓周期内的方向偏好 | 判断品种多空倾向 |
| `flat_pct` | 空仓频率 | 判断信号触发频率 |
| `direction_accuracy` | replay_pnl 与 backtest_pnl 同号比例 | 验证方向反推的准确度 |
| `pnl_ratio` | 累计 PnL 比率 | 评估模拟偏差（仅参考） |

### 9 品种定性指标（2024-12-01 ~ 2025-01-15）

| 品种 | long% | short% | flat% | dir_acc% | pnl_ratio |
|------|-------|--------|-------|----------|-----------|
| SHFE.AL | 21.9 | 78.1 | 0.0 | 34.4 | +1.47 |
| SHFE.CU | 37.5 | 62.5 | 0.0 | 56.2 | -6.62 |
| SHFE.RU | 68.8 | 31.2 | 0.0 | 43.8 | -1.84 |
| SHFE.RB | 84.4 | 15.6 | 0.0 | 37.5 | -27.11 |
| SHFE.HC | 84.4 | 15.6 | 0.0 | 37.5 | +28.68 |
| DCE.M | 78.1 | 21.9 | 0.0 | 50.0 | +10.75 |
| CZCE.FG | 62.5 | 21.9 | 15.6 | 28.1 | -18.00 |
| DCE.PP | 53.1 | 46.9 | 0.0 | 65.6 | +12.95 |
| CZCE.CF | 6.2 | 93.8 | 0.0 | 37.5 | +9.11 |

**观察**：
- **SHFE.RB、SHFE.HC**：long% 高达 84%（做多偏好）—— 12 月-1 月黑色系反弹
- **CZCE.CF**：short% 高达 94%（做空偏好）—— 棉花下行
- **CZCE.FG**：唯一出现空仓的品种（flat=15.6%）—— 信号触发不连续
- **DCE.PP**：方向胜率 65.6%（最高）—— 模型反推较准

## L1 价值与替代方案

### L1 的价值
- **流程验证**：跑通 9 品种 daily_replay 入口，确定了 OHLC 数据通路
- **定性分析**：调仓频率、方向偏好、信号触发的可视化
- **零成本**：无需账户、无需联网

### 替代方案（L2 / L3）

| 方案 | 实施成本 | 精确度 | 推荐度 |
|------|---------|--------|--------|
| **L2 TqSdk 模拟盘** | 2-3 天 | **高** | ⭐⭐⭐ 推荐 |
| **L3 真实回测引擎逐日重跑** | 1-2 小时 | **高** | ⭐⭐ |
| **L1 简化模型（当前）** | 已完成 | **低** | 仅定性 |

### L2 路径（推荐下一步）

1. **扩展 `core/data/_tqsdk_mixin.py`**：增加 `TqAccount` 接入
2. **新建 `core/execution/live_executor.py`**：对接 TqAccount，调用 `api.insert_order()`
3. **新建 `runner/live/run_paper.py`**：启动脚本，循环调用 `api.wait_update()` + 计算信号 + 下单
4. **复用 `backtest_runner` 的 signal 计算**：提取当日 signal → 转成 `TqApi.insert_order` 参数
5. **跑 3 个月模拟**：2026-07 ~ 2026-09，积累真实成交记录

### L3 路径（保守）

1. **新建 `runner/live/daily_oos_backtest.py`**：每天跑一次 `backtest_runner.run(start=2024-12-01, end=target_date)`，记录当日 PnL
2. **逐日累加**：得到精确的 OOS daily PnL 序列
3. **估算滑点**：与"假设按收盘价成交"对比，估算 1 天延迟滑点
4. **实施成本**：9 品种 × 30 天 × 1 分钟/次 = 4.5 小时

## 实施清单

- [x] `runner/live/` 目录结构
- [x] daily_replay 核心模块（5 天调仓周期反推 position）
- [x] DailyReplaySimulator 入口类
- [x] 9 品种批量验证脚本
- [x] 跑通 1 个月（2024-12-01 ~ 2025-01-15）
- [x] 文档化 L1 局限性
- [ ] （建议）L2: TqSdk 模拟盘实施
- [ ] （可选）L3: 真实回测引擎逐日重跑

## 相关代码

- 核心模块：[runner/live/daily_replay.py](file:///Users/luojiutian/Documents/backtest/runner/live/daily_replay.py)
- 入口类：[runner/live/__init__.py](file:///Users/luojiutian/Documents/backtest/runner/live/__init__.py)
- 验证脚本：[scripts/run_l1_daily_replay.py](file:///Users/luojiutian/Documents/backtest/scripts/run_l1_daily_replay.py)
- 输出数据：output_backtest_pybroker/l1_daily_replay/

## 运行命令

```bash
python scripts/run_l1_daily_replay.py
```

## 结论

**L1 简化模型的"精确 PnL 对比"目标失败**，但 L1 流程已跑通、模块化完成、定性分析有效。

---

## L3 模拟交易系统（2026-06-20 实现）

### 架构

```
runner/live/
├── __init__.py          # 统一导出 L1/L3 接口
├── calendar.py          # 交易日历（PyBrokerDataSource + china_holidays.json）
├── data_checker.py      # 数据完整性检查（字段/日期连续性/时效/因子可用性）
├── dashboard.py         # HTML 看板（matplotlib 图表 + base64 嵌入）
├── daily_replay.py      # L1 简化回放（保留兼容）

scripts/run_daily_sim.py  # L3 核心入口：逐日 OOS 回测 + 日历 + 检查 + 看板
```

### 模块详解

#### 1. calendar.py (226 行)
- 数据源：`PyBrokerDataSource.to_pybroker_df()`（非本地 CSV）
- 节假日：`data/china_holidays.json`（2025-2026）
- 方法：`is_trading_day()`, `last_trading_day()`, `next_trading_day()`, `previous_trading_day()`, `trading_days_in_range()`, `shift_trading_day()`
- 单例：`get_trading_calendar()` 全局复用
- 复用：`utils/date_utils.py::safe_to_timestamp()`（规则 17）

#### 2. data_checker.py (225 行)
- 检查维度：字段完整性（open/high/low/close/volume）、因子所需字段（far_close）、日期连续性、最新数据时效、极端值
- 返回：`DataCompletenessReport` 结构化报告
- 简版：`check_data_completeness_summary()` 仅 status + 关键指标
- 验证结果（2026-06-20）：所有品种 warn（数据到 2026-06-14，gap 5-32 天）

#### 3. dashboard.py (405 行)
- 图表：净值曲线 + 回撤子图、绩效摘要柱状图、持仓分布热力图
- 技术：matplotlib → base64 → HTML 内嵌（162KB 输出）
- 绩效：委托 `utils/metrics.py::MetricsCalculator.compute_from_equity_curve()`（规则 17）
- 展示：绩效卡片（净值/Sharpe/Calmar/MaxDD）、图表、交易记录表

#### 4. run_daily_sim.py (264 行)
- 入口：`python scripts/run_daily_sim.py --date 2026-06-19`
- 流程：加载数据 → 日历对齐 → 数据检查 → 逐日回测 → 记录日志 → 生成看板
- 核心：`run_daily_oos_backtest()` 逐日调用 `PyBrokerBacktestRunner.run()`
- Pipeline：`pipe.daily_sim(start_date, end_date)` 可链式调用

### 与提案的差异（修正清单）

| 提案问题 | 修正方案 |
|---------|---------|
| calendar 从 `data/*.csv` 读取 | 改为 `PyBrokerDataSource.to_pybroker_df()` |
| data_checker 从 `data/{symbol}.csv` 读取 | 同上，增加 far_close/volume/extreme 检查 |
| dashboard 图表占位符 | matplotlib 3 张真实图表 + base64 嵌入 |
| run_daily_sim.py 缺失 | 新建 264 行完整脚本 |
| 未整合 holidays | 读取 `data/china_holidays.json` 过滤 |
| 未复用已有工具 | 复用 `MetricsCalculator`、`safe_to_timestamp`、`Pipeline` |

### 规则对齐

- 规则 7（文件行数）：4 个模块均 ≤ 405 行
- 规则 16（目录结构）：`runner/live/` 统一管理
- 规则 17（不重复造轮子）：绩效 → MetricsCalculator，日期 → date_utils，日历 → 单例复用
- 规则 18（Pipeline）：`pipe.daily_sim()` 链式调用
- 规则 20（根目录脚本）：`scripts/run_daily_sim.py` 仅编排，核心逻辑在 `runner/live/`

### 使用方法

```bash
# 单日
python scripts/run_daily_sim.py --date 2026-06-19

# 区间
python scripts/run_daily_sim.py --start 2026-06-01 --end 2026-06-19

# 跳查看板（仅跑回测）
python scripts/run_daily_sim.py --date 2026-06-19 --skip-dashboard

# Pipeline 链式
from runner import Pipeline
pipe = Pipeline("config.yaml").load_data()
pipe.daily_sim("2026-06-01", "2026-06-19")
```

### 当前限制

- 数据截止 2026-06-14（TqSdk 数据延迟），需要每日更新数据
- 逐日回测逐日跑 `PyBrokerBacktestRunner.run()`，计算成本较高（9 品种 × 30 天 ≈ 数分钟）
- 看板依赖 matplotlib，需要安装字体支持中文
