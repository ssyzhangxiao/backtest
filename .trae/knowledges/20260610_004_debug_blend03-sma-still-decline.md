# E1 多品种 v3 回归：blend=0.3+SMA 斜率也未超 v1

## 摘要

选项 B（blend=0.3 恢复 + SMA 斜率算法升级）端到端验证：**v3 avg Sharpe 0.0130 比 v1 0.0139 下降 6.03%**。
**A 单品种 v3 比 v1 改善 1.9pp**（pure alpha 视角），但 E1 多品种**反向**——交易摩擦下 SMA 斜率反而不如原始动量。
**最终结论：v1（blend=0.3 + 动量）仍是当前最优方案**。改进 1+2 整体失败，blend 不动是合理选择。

## 实验设置

| 项 | 值 |
|---|---|
| 数据范围 | 2019-09-16 ~ 2022-04-14（Test split） |
| 品种 × 策略 | 8 × 5 = 40 单元 |
| 引擎 | PyBroker 完整回测 |
| v1 | blend=0.3，算法=原始动量（修复 best_params 注入后 baseline） |
| v2 | blend=0.1，算法=SMA 斜率（已废，降权方案） |
| v3 | blend=0.3，算法=SMA 斜率（选项 B，恢复权重 + 算法升级） |

## 实验结果

### 整体对比

| 指标 | v1 | v2 | v3 | v3-v1 | v3-v2 |
|---|---|---|---|---|---|
| **avg Sharpe** | **+0.0139** | +0.0127 | **+0.0130** | **-6.03%** | **+2.36%** |
| avg_return | +2.36 | +1.98 | +2.05 | -0.30 | +0.07 |
| avg_mdd | -5.56 | -5.66 | -5.65 | -0.09 | +0.01 |

### 按策略细分

| 策略 | v1 | v2 | v3 | v3 vs v1 | v3 vs v2 |
|---|---|---|---|---|---|
| trend | +0.0182 | +0.0167 | +0.0174 | -0.0008 | +0.0007 |
| term_structure | +0.0133 | +0.0095 | +0.0104 | -0.0029 | +0.0009 |
| vol_breakout | +0.0188 | +0.0184 | +0.0184 | -0.0005 | 0 |
| composite_resonance | +0.0190 | +0.0190 | +0.0190 | 0 | 0 |
| mean_reversion | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |

### 按品种（v1 vs v3）

| 品种 | v1 | v3 | Δ |
|---|---|---|---|
| SHFE.AL | +0.0424 | +0.0406 | -0.0018 ↓ |
| SHFE.CU | +0.0310 | +0.0315 | +0.0005 ↑ |
| CZCE.FG | +0.0319 | +0.0286 | -0.0033 ↓ |
| SHFE.RU | +0.0126 | +0.0121 | -0.0005 ↓ |
| DCE.PP | +0.0166 | +0.0164 | -0.0002 ↓ |
| CZCE.CF | +0.0160 | +0.0135 | -0.0025 ↓ |
| DCE.J | -0.0305 | -0.0283 | +0.0022 ↑ |
| SHFE.RB | -0.0092 | -0.0103 | -0.0011 ↓ |

### 单元分布（v3 vs v1）

- 提升: 2/40 (5.0%)
- 下降: 10/40 (25.0%)
- 持平: 28/40 (70.0%)

## A v3 vs E1 v3 矛盾分析

### A 单品种 v3（pure alpha）

| 指标 | v1 | v3 |
|---|---|---|
| best Sharpe gap vs default | -7.1% | **-5.2%** |

A 看：v3 比 v1 改善 1.9pp（SMA 斜率比原始动量鲁棒）。

### E1 多品种 v3（端到端）

| 指标 | v1 | v3 |
|---|---|---|
| avg Sharpe | +0.0139 | +0.0130 (-6.03%) |

E1 看：v3 比 v1 下降 6%。

### 矛盾根因

| 因素 | A 看 | E1 看 |
|---|---|---|
| 评价方式 | signal-to-PnL 直接转换 | PyBroker 完整回测 |
| 交易摩擦 | 无 | 含手续费、滑点 |
| 主信号 | SMA 斜率噪声小 | SMA 斜率对"快速反转"响应慢 |
| 算法影响 | 算法升级改善 pure alpha | 算法升级在交易摩擦下反向 |

**SMA 斜率的本质问题**：
- SMA 平滑了价格 → 滞后于反转
- 原始动量（close/close.shift(w)-1）**更敏感**于价格反转
- 在 OOS + 交易摩擦场景下，**敏感度 > 平滑度**
- 短窗口（w=5）下 SMA 斜率是"被 SMA 拖慢的动量"——既无原始动量的敏感度，也无长窗口的稳定性

## 根因汇总

1. **算法升级（SMA 斜率）在交易摩擦下反向**——SMA 平滑的"鲁棒性"在 OOS 反而是累赘。
2. **blend 0.3 保留 best_params 30% 边际贡献是必要**——v3 vs v2 改善 +2.36% 证实。
3. **A 单品种 + E1 多品种必须双验证**——pure alpha 改善不能等同于端到端提升。
4. **小窗口 (w=5) 的 SMA 斜率是"鸡肋"**——既无原始动量的敏感度，也无长窗口的稳定性。

## 最终建议

### 选项 A：完全回滚到 v1（blend=0.3 + 动量）
- E1 v1 仍是当前 baseline best
- 放弃改进 1+2，回到原方案
- **风险**：放弃了 A 单品种已验证的"信号扰动减小"

### 选项 D：彻底移除窗口动量通道
- 把 `_apply_param_window` 改成 no-op
- best_params 退化为只用于 StrategyIndicatorRegistry metadata
- 简化代码，去掉"修复 best_params 注入"后的伪改进
- **风险**：放弃 best_params OOS 提升的可能（但 A/E1 都未观测到显著提升）

### 推荐：**选项 A**（回滚到 v1）

**理由**：
- v1 是 E1 已知 baseline best（+0.0139）
- v3 端到端损失 6% 是"算法升级"的代价
- 改进 3（条件性混合）作为未来方向保留
- **回归最简方案**：blend 0.3 + 动量已被验证 OK

## Takeaways

1. **算法升级必须在端到端验证**——A 单品种 pure alpha 改善不能等同于 E1 提升。
2. **小窗口 SMA 斜率 = 平滑的噪声**——既无原始动量敏感度，也无长窗口稳定性。
3. **交易摩擦放大信号扰动影响**——任何"鲁棒"算法都要在交易成本下重新评估。
4. **回滚不可耻**——端到端验证显示 v3 不如 v1，**回滚是正确决策**。
5. **单元分布揭示真相**——v3 的 2/40 提升 vs 10/40 下降 = 不对称恶化。

## 相关

- `core/factors/alpha_futures/sub_strategy_aggregator.py:_apply_param_window`
- `.trae/knowledges/20260610_001_debug_sub-strategy-aggregator-params-injection.md`（v1 baseline）
- `.trae/knowledges/20260610_002_debug_single-symbol-best-params-oos-decline.md`（A 单品种）
- `.trae/knowledges/20260610_003_debug_e1-multi-symbol-blend01-decline.md`（v2 E1）
- `output_backtest_pybroker/e1_baseline_metrics.blend03.csv`（v1）
- `output_backtest_pybroker/e1_baseline_metrics.blend03sma.csv`（v3）
