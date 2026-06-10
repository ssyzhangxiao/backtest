# 单品种 best_params OOS 反向：trend.window=5 在 SHFE.RB 上比 default 差

## 摘要

在修复 best_params 注入 bug 后（见 [20260610_001](20260610_001_debug_sub-strategy-aggregator-params-injection.md)），用 SHFE.RB + trend 策略做 A 单品种 deep dive：发现 best_params（window=5）相对 default（window=20）在 OOS 上 **全面下降**：IC IR -21.5%、PnL Sharpe -7.1%、Annual Return -26.5%。这与用户问的"OOS Sharpe 没提升"完全吻合，根因有三层：(1) window=5 引入高频动量噪声，(2) blend=0.3 把 trend std 从 1.0 压到 0.77 削弱信号强度，(3) IC IR 0.78→0.61 在 623 样本上统计上可能不显著。

## 实验方法

| 项 | 值 |
|---|---|
| 品种 | SHFE.RB（流动性最好） |
| 策略 | trend |
| 数据范围 | 2019-09-16 ~ 2022-04-14，623 根日线 |
| 数据源 | TqSdk (`data_length=4000`) |
| 对比组 | (A) default: `custom_params={}` → window=20；(B) best: `custom_params={"trend": {"window": 5}}` |
| 信号提取 | 直接调 `compute_sub_strategy_scores_from_ohlcv` 拿 trend 列 |
| 评估 | (1) IC overall + 滚动 60 样本 IC IR；(2) signal-to-PnL 引擎：pnl_t = signal_t * ret_{t+1} * 0.1 |

## 实验结果

| 指标 | default (w=20) | best (w=5) | Δ |
|---|---|---|---|
| IC overall | +0.0741 | +0.0677 | **-8.6%** |
| IC rolling mean | +0.0633 | +0.0448 | **-29.2%** |
| **IC IR** | **+0.7793** | +0.6119 | **-21.5%** |
| **PnL Sharpe** | **+1.1608** | +1.0786 | **-7.1%** |
| PnL Annual Return | +2.91% | +2.14% | -26.5% |
| PnL Max DD | -2.79% | -2.27% | 风险更低 |
| PnL Win Rate | 51.36% | 51.36% | 持平 |
| Signal mean | -0.018 | +0.036 | **反向** |
| Signal std | 1.00 | 0.77 | **-23%** |

## 根因分析（三层）

### 根因 1：窗口动量是高频动量，单品种噪声大

`window=5` 算的是 `close / close.shift(5) - 1`：
- 5 日动量反映**短期价格漂移**
- 单品种 SHFE.RB 短期价格受**日内波动、流动性事件、宏观新闻**主导
- 与 24 因子集成的低频动量（6/12 日）形成"高低频对冲"，互相抵消

### 根因 2：blend=0.3 削弱了趋势主信号

```
blended = 0.7 * ser_24factor + 0.3 * ser_window_momentum
```

观察：
- `Signal std: 1.00 → 0.77`（**-23%**）—— 趋势主信号被压扁
- `Signal mean: -0.018 → +0.036`（**反向**）—— 5 日动量把信号翻多了，**不是平移而是反转**
- IC IR 与 Sharpe 同步下降 → **窗口动量是噪声，不是 alpha**

### 根因 3：623 样本的 IR 差异统计上不显著

| | default | best |
|---|---|---|
| IC rolling std | 0.0812 | 0.0732 |
| IC rolling mean | 0.0633 | 0.0448 |

差异 = (0.0633 - 0.0448) / √(0.0812²/60 + 0.0732²/60) ≈ 0.0185 / 0.0145 ≈ **1.28σ**

**1.28σ 不构成 95% 显著**（需 > 1.96σ）——这两个 IR 在统计上**没有显著差异**。

## 与 E1 多品种回测的对照

| 实验 | 范围 | 修复后 avg Sharpe | Δ vs default |
|---|---|---|---|
| E1 | 8 品种 × 5 策略 | 0.0139 | **-6.1%** |
| A | SHFE.RB × trend | 1.0786 (单) | **-7.1%** |

E1 与 A **方向一致**（都略降），幅度相当（~6-7%）。说明 best_params 在 in-sample + 全期窗口上**系统性轻微过拟合**。

## 结论

1. **best_params 注入修复彻底成功**（Signal std -23%、mean 反向、绝对值变化巨大）
2. **但 best_params 在 OOS 表现不增反降**——A 实验直接验证
3. **问题不在修复，而在优化目标 + blend 方案设计**：
   - 优化器用 Sharpe，但 5 日动量在单品种 OOS 上 IC 反而下降
   - blend=0.3 让低频趋势被高频动量"稀释"
4. **下一步方向**：
   - 改 blend=0.1 或 0（仅在 IC 显著时叠加窗口动量）
   - 窗口动量算法改进：用 RSI/SMA 斜率代替原始动量
   - 多品种横截面 + OOS 时间窗上验证（多品种分散化能减轻单品种噪声）
   - 优化器目标改为 IR（更鲁棒）而非 Sharpe

## Takeaways

1. **修复"参数不生效"≠"参数有效"**——OOS 表现独立于注入链路是否通。
2. **高频动量参数在单品种 OOS 上需要谨慎**——单品种噪声主导，5 日动量通常弱于 20 日。
3. **小样本 IC IR 差异必须做显著性检验**——1σ 内的差异可能是随机。
4. **A/B 实验必须用同一个 signal-to-PnL 引擎**——避免 PyBroker 开仓逻辑把信号对比"二次污染"。

## 相关

- `core/factors/alpha_futures/sub_strategy_aggregator.py:_apply_param_window`
- `runner/experiments/a_single_strategy_compare.py`
- `output_backtest_pybroker/a_single_strategy_compare.csv`
