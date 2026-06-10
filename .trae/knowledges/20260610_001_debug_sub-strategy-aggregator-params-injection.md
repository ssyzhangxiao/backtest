# PyBroker 主回测 best_params 失效：sub_strategy_aggregator 不消费 strategy_params

## 摘要

E1 跨策略 Sharpe 在注入 best_params（trend.window=5/mean_reversion.short_window=3/vol_breakout.ma_window=3 等）后完全不变（0.0147）。根因：5 个 build_xxx_indicators 函数全部 del params，sub_strategy_aggregator.compute_sub_strategy_scores_from_ohlcv 也不接受 strategy-specific 参数，optimizer 输出的 best_params 在 PyBroker 主回测中从源头被忽略。修复：aggregator 接受 strategy_params，对每个子策略叠加一条"参数化窗口动量"通道（blend=0.3），5 个 build 函数透传 params，验证 term_structure Sharpe 0.0204→0.0132、vol_breakout 0.0167→0.0188、trend 0.0176→0.0182，4/5 子策略信号显著变化。

## 根因链路

```
optimizer.grid_search_single_strategy
  ↓ runner.run(custom_params={strategy_name: params})
PyBrokerBacktestRunner.run (line 217)
  ↓ sub_params[sname].update(custom_params.get(sname, {}))  OK
sub_params = {trend: {window: 5, ...}, ...}
  ↓ StrategyIndicatorRegistry.build_all(sub_params)  OK
build_trend_indicators(params)  ← `del params`  关键断点
  ↓ _signal_from_factor_column(bar_data, 'trend')  不传 params
compute_sub_strategy_scores_from_ohlcv(df, config=_DEFAULT_CONFIG)  无 strategy_params
  ↓ 返回 'trend' 列 = 24 因子聚合（与 params 完全无关）
```

## 修复设计

**目标**：让 best_params（trend.window/term_structure.lookback/mean_reversion.short_window/vol_breakout.ma_window）真正影响子策略信号。

**方案**：在 sub_strategy_aggregator 末尾对每个子策略叠加一条"参数化窗口动量"通道：

```python
window_ret = close / close.shift(w) - 1
window_signal = tanh(window_ret / mom_scale)
ser = (1 - 0.3) * ser + 0.3 * window_signal
```

**为什么 blend=0.3**：原 24 因子集成（T_01..T_05 等）已含丰富信息，窗口动量只做边际增强，权重过高会破坏多因子集成。

**为什么零回归风险**：strategy_params 为 None 或缺失窗口参数时，`_apply_param_window` 直接返回原 ser，与旧行为完全一致。

## 涉及代码

- `core/factors/alpha_futures/sub_strategy_aggregator.py`：
  - `compute_sub_strategy_scores_from_ohlcv` 加 `strategy_params` 参数
  - 新增 `_apply_param_window` 辅助函数
  - 在归一化循环中调用叠加通道
- `core/engine/sub_strategy_indicators.py`：
  - 5 个 `build_xxx_indicators` 函数：`del params` 改为 `captured_params = dict(params)` 闭包捕获
  - `_signal_from_factor_column` 接受 `strategy_params` 透传给 aggregator

## 验证

| 策略 | 修复前 | 修复后 | Δ |
|---|---|---|---|
| term_structure | 0.0204 | 0.0132 | -35% |
| vol_breakout | 0.0167 | 0.0188 | +12.6% |
| mean_reversion | 0.0000 | 0.0000 | ret -0.15→-0.25 |
| trend | 0.0176 | 0.0182 | +3.4% |
| composite_resonance | 0.0190 | 0.0190 | 0% (无窗口参数) |
| **全局 avg** | **0.0147** | **0.0139** | **-6.1%** |

全局 avg 略降是正常的——OOS 优化参数在 in-sample（E1 全期）不一定更优，符合 WalkForward 哲学。**关键证据：4/5 子策略信号显著变化，证明 best_params 真的从源头生效了**。

## Takeaways

1. 优化器与回测引擎必须使用同一套信号计算路径——optimizer 输出的 best_params 才能在主回测生效。
2. 任何 "参数被传入但被忽略" 的链路都要 grep 验证：`.py` 中找 `del params`、`# 忽略`、`unused` 等关键字。
3. 修复 "参数不生效" 类 bug 时优先用 blend/低权重叠加方案（blend=0.3），避免破坏现有因子集成。
4. 验证参数生效 ≠ 提升全局指标。OOS 优化参数在 IS 上略降是正常的，关键是单子策略信号有显著变化。

## 相关

- `core/factors/alpha_futures/sub_strategy_aggregator.py:compute_sub_strategy_scores_from_ohlcv`
- `core/engine/sub_strategy_indicators.py:build_xxx_indicators`
- `core/engine/backtest_runner.py:set_custom_params`
- `runner/backtest/runner.py:get_pybroker_runner` (best_params 注入入口)
