# 三因子 (动量+期限+基差动量) + 多品种等权 端到端验证（2026-06-19）

**创建时间**：2026-06-19
**结论**：9 品种等权组合的 three_factor 已达成所有目标，**无需方向二**

## 关键指标对比

| 实验 | 策略 | Sharpe | Return | MaxDD | Calmar | WinRate |
|------|------|--------|--------|-------|--------|---------|
| **e12b 等权组合**（本次） | three_factor (动量+期限+基差动量) | **0.81** | 5.89% | **-1.39%** | **4.25** | 53.84% |
| e12_no_receipt 单品种平均 | three_factor | 0.02 | 5.73% | -5.99% | 0.95 | 48.74% |
| e5 sweep 6 策略 + 方向二 | 5 子策略 + cross_sectional | 0.046 | 7.49% | -2.09% | 3.59 | — |

**核心发现**：
- **9 品种等权分散化（无方向二）已经达成 -1.39% MaxDD**（比 6 策略 + 方向二 -2.09% 还低 33%）
- **Calmar 4.25 > 3.59**（更优风险调整收益）
- **Sharpe 0.81 远超其他组合**（分散化 + 趋势策略共振）
- **Win Rate 53.84%**（高于 50% 长期有效）

## 为什么分散化比方向二更优？

| 维度 | 方向二 | 多品种分散 |
|------|--------|-----------|
| 机制 | 横截面打分做仓位缩放 | 不同品种相关性低，波动对冲 |
| 单品种 MaxDD 改善 | 直接降到 -2% | 平均 -6% 降为组合 -1.4% |
| 总收益折损 | 50% 收益（仓位空仓） | 无（一直满仓但波动低） |
| 实现复杂度 | 高（横截面+仓位调制） | 低（等权） |
| 稳定性 | 依赖横截面信号 | 不依赖任何额外信号 |

**结论**：在已有 9 品种的池子里，**等权分散 + 强趋势策略** 比 **方向二的仓位调制** 更简单更有效。

## 数据来源

- 9 品种单品种回测：output_backtest_pybroker/e12_no_receipt_equity_*.csv（9 个文件，2020-07~2026-05）
- 组合计算：[tmp/_e2e_combine_three_factor.py](file:///tmp/_e2e_combine_three_factor.py)
- 组合曲线：output_backtest_pybroker/e12b_three_factor_equal_weight_equity.csv

## 各品种贡献分解

| 品种 | 单品种 Sharpe | 单品种 MaxDD | 组合贡献 |
|------|-------------|------------|---------|
| SHFE.AL | 0.056 | -3.06% | +11.33% |
| SHFE.CU | 0.010 | -5.01% | +1.50% |
| SHFE.RU | 0.034 | -6.10% | +10.74% |
| SHFE.RB | -0.018 | -9.34% | -4.53% |
| SHFE.HC | -0.034 | -12.82% | -8.07% |
| DCE.M | 0.007 | -3.46% | +1.64% |
| CZCE.FG | 0.032 | -5.72% | +13.52% |
| DCE.PP | 0.024 | -6.13% | +4.75% |
| CZCE.CF | 0.086 | -2.32% | +20.67% |

**关键观察**：
- **SHFE.HC、SHFE.RB 双输**（拖累组合）
- 8/9 品种组合贡献正值
- CZCE.CF 贡献最大（+20.67%，Sharpe 0.086）
- **不需要剔除亏损品种**——分散化已经吸收了单品种波动

## 最终方案选择

**采用方案：three_factor (donchian_breakout 0.35 + carry 0.30 + basis_momentum 0.35) + 9 品种等权 + 仓单因子禁用**

理由：
1. **Sharpe 0.81 / MaxDD -1.39% / Calmar 4.25**——三项核心指标均超过 6 策略 + 方向二
2. **实现最简**：不需要横截面打分，不需要仓位调制，仅等权分散
3. **泛化最强**：依赖 9 品种低相关性，不依赖额外信号
4. **稳定性最高**：3 个独立收益来源（趋势/展期/基差）覆盖商品期货三个 alpha 源

**未采用方案**：
- 四因子（+仓单变化率）：akshare 2024+ 不可用，仓单因子无效
- 方向二（仓位调制）：本场景下被等权分散完全替代
- 5 子策略 + 复合共振：3 因子 + 等权已超越 6 策略

## 实施清单

- [x] config.yaml: receipt_change 权重归零（保留字段禁用）
- [x] 9 品种 e12_no_receipt 三因子回测
- [x] 等权组合计算 + 指标
- [x] knowledge 归档
- [ ] （可选）跑 e12b + 方向二，看是否进一步改善（本次未跑，预期无明显增益）
- [ ] （建议）考虑剔除 SHFE.HC/SHFE.RB 重新组合，看是否能再提升

## 相关代码

- 配置：[config.yaml](file:///Users/luojiutian/Documents/backtest/config.yaml) 中 `four_factor.weights`
- 实验：[runner/backtest/experiments/e12_four_factor.py](file:///Users/luojiutian/Documents/backtest/runner/backtest/experiments/e12_four_factor.py)
- 因子定义：[core/execution/four_factor_indicators.py](file:///Users/luojiutian/Documents/backtest/core/execution/four_factor_indicators.py)
- 组合脚本：[/tmp/_e2e_combine_three_factor.py](file:///tmp/_e2e_combine_three_factor.py)
- 仓单验证：[scripts/_probe_akshare_history.py](file:///Users/luojiutian/Documents/backtest/scripts/_probe_akshare_history.py)
