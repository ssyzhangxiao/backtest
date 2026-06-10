# E1 多品种 v2 回归：blend=0.1+SMA 斜率整体略差

## 摘要

承接 A 单品种 v2（IC IR +25.7%，Sharpe 损失从 -7.1% 缩到 -1.0%）的成功改进，
跑 E1 多品种 × 多策略端到端回归验证（PyBroker 完整回测，含交易/滑点/手续费）。
**结果反向**：E1 v2 avg Sharpe 0.0127 比 v1 0.0139 下降 8.19%，3/40 提升、10/40 下降。
**结论**：blend 0.3 → 0.1 + 动量 → SMA 斜率这两项改动在端到端回测中**整体不利**，
A 单品种 pure alpha 改善无法转化为多品种 Sharpe 提升。

## 实验设置

| 项 | 值 |
|---|---|
| 数据范围 | 2019-09-16 ~ 2022-04-14（Test split） |
| 品种 × 策略 | 8 × 5 = 40 单元 |
| 数据源 | TqSdk |
| 引擎 | PyBroker 完整回测（含交易、滑点、手续费） |
| v1 | blend=0.3，算法=原始动量（close/close.shift(w)-1） |
| v2 | blend=0.1，算法=SMA 斜率（(close-sma(w))/sma(w)） |

## 实验结果

### 整体

| 指标 | v1 | v2 | Δ |
|---|---|---|---|
| **avg Sharpe** | **+0.0139** | **+0.0127** | **-8.19%** ⚠️ |
| avg total_return_pct | +2.36 | +1.98 | -16.0% |
| avg max_drawdown_pct | -5.56 | -5.66 | -1.8% |
| win_rate | 46.75% | 46.81% | +0.1% |
| trade_count | 62.1 | 62.2 | +0.1% |

### 按策略

| 策略 | v1 | v2 | Δ |
|---|---|---|---|
| trend | +0.0182 | +0.0167 | **-8.2%** |
| term_structure | +0.0133 | +0.0095 | **-28%** ⚠️ |
| mean_reversion | 0.0000 | 0.0000 | 0% |
| vol_breakout | +0.0188 | +0.0184 | -2.7% |
| composite_resonance | +0.0190 | +0.0190 | 0% |

### 按品种

| 品种 | v1 | v2 | Δ |
|---|---|---|---|
| SHFE.AL | +0.0424 | +0.0422 | -0.05% |
| SHFE.CU | +0.0310 | +0.0312 | +0.6% |
| CZCE.FG | +0.0319 | +0.0272 | -15% |
| SHFE.RU | +0.0126 | +0.0121 | -4.0% |
| DCE.PP | +0.0166 | +0.0164 | -1.2% |
| CZCE.CF | +0.0160 | +0.0138 | -13.8% |
| DCE.J | -0.0305 | -0.0311 | -2.0% |
| SHFE.RB | -0.0092 | -0.0100 | -8.7% |

### 单元分布

- 提升: 3/40 (7.5%)
- 下降: 10/40 (25.0%)
- 持平: 27/40 (67.5%)
- **净下降 17.5pp**

## 根因分析

### A v2 vs E1 v2 方向相反的根因

A 单品种实验用 signal-to-PnL 直接计算（pure alpha）：
- v2 best Sharpe 1.15 比 v1 best 1.08 高 6.6%
- v2 把信号扰动从 0.32 降到 0.10（-67%）
- v2 Signal std 0.77→0.91，保留主信号强度

E1 多品种用 PyBroker 完整回测：
- v2 把 blend 从 0.3 降到 0.1，**降低了 best_params 边际贡献**
- term_structure 受影响最大（-28%）——其 best_params 主要是 lookback，缩权重后失去边际
- trend 也下降 -8.2%——窗口动量本应是 trend 增强器，权重降低后变成噪声

### 改进 1（blend 0.3→0.1）的反效果

| 角度 | A 看 | E1 看 |
|---|---|---|
| Signal 扰动幅度 | 大幅改善（-67%） | 不重要（PyBroker 内部容错） |
| 主信号保留 | 改善（0.77→0.91） | 反而让 best_params 失效 |
| Pure alpha 评价 | +6.6% | — |
| 端到端回测 | — | **-8.2%** |

**结论**：A 测试的"信号扰动减小"在 E1 端到端没意义。**真正重要的是 best_params 提供的边际 alpha**。
blend 越小，best_params 越接近"无操作"——直接降低了 best_params 设计的初衷。

### 改进 2（动量→SMA 斜率）的影响

A v2 best 的 SMA 斜率（window=5 = close 偏离 5 日 SMA）噪声低于 v1 动量。
但 E1 上**没有 best_params 的对照**（A 没有 default_SMA 斜率基线），
无法隔离"算法升级"和"权重降低"两个因素。

## 建议

### 选项 A：回滚到 v1（blend=0.3 + 动量）
- E1 v1 是当前 baseline best
- 放弃改进 1+2，回到原方案
- **风险**：放弃了已验证的"信号扰动减小"和"算法鲁棒性"

### 选项 B：部分回滚（blend=0.3 + SMA 斜率）
- 仅保留算法升级（动量→SMA 斜率），恢复原 blend=0.3
- A 单品种需要重做：验证 v3（blend=0.3 + SMA 斜率）的 pure alpha
- 推测：E1 端到端 v3 ≈ v1（±2%），但算法鲁棒性更好

### 选项 C：保留 v2 + 加改进 3（条件性混合）
- 保持 v2（blend=0.1 + SMA 斜率）
- 增加条件性混合：仅当窗口信号 IC > 阈值时叠加
- 解决"无条件叠加 = 噪声"问题
- **实现成本高**：需要在每根 bar 估 IC，OOS 上不可行，需用冷启动预热

### 选项 D：彻底移除窗口动量通道
- 把 `_apply_param_window` 改成 no-op（blend=0 默认值或写 0）
- best_params 退化为只用于 `_register_default_indicators` 中的策略元数据
- 简化代码，去掉"修复 best_params 注入"后的伪改进
- **风险**：放弃 best_params OOS 提升的可能

## 推荐

**选项 B（blend=0.3 + SMA 斜率）**——平衡两个改进：
- 保留算法升级（SMA 斜率更鲁棒）
- 恢复 blend=0.3（保留 best_params 边际贡献）
- A 重做 → E1 回归（端到端验证）

**次选选项 D**——若选项 B 重做后仍无显著提升，建议彻底移除窗口动量通道。
best_params 走 StrategyIndicatorRegistry 已有 metadata 路径即可，无需叠加信号。

## Takeaways

1. **A 单品种 + E1 多品种必须双验证**——pure alpha 改善不能等同于端到端提升。
2. **PyBroker 完整回测 = 真实业务**——不要用 signal-to-PnL 当唯一评价标准。
3. **改动 1 项 + 验证 1 项 = 基本研究单位**——同时改 2 项无法归因。
4. **blend 是个旋钮**——不是越大越好或越小越好，依赖 best_params 质量。

## 相关

- `core/factors/alpha_futures/sub_strategy_aggregator.py:_apply_param_window`
- `runner/experiments/a_single_strategy_compare.py`
- `runner/backtest/experiments/e1_e5.py:run_e1_single_strategy_baselines`
- `output_backtest_pybroker/e1_baseline_metrics.blend03.csv`（v1 备份）
- `output_backtest_pybroker/e1_baseline_metrics.csv`（v2 当前）
