# 规则32：统一因子池架构 — 所有信号单入口，抽象层按模式提取

**生效日期**：2026-06-13
**核心原则**：所有因子（24 Alpha + 6 CTA + 未来新增）必须通过 `UnifiedFactorPool` 统一计算，再经 `SignalAbstractionLayer` 按模式提取，不得绕过。

---

## 1. 三层架构

```
OHLCV DataFrame
      │
      ▼
┌────────────────────────────────────────────────────────┐
│              UnifiedFactorPool (factor_pool.py)         │
│                                                        │
│  compute_all(ohlcv, symbol) → DataFrame(11+1 列)       │
│                                                        │
│  ├── compute_sub_strategy_scores_from_ohlcv()  ← 5子策略│
│  │     (24 Alpha 因子: T_01..T_05, TS_01..TS_03,       │
│  │      M_01..M_05, V_01..V_04, H_01..H_05,            │
│  │      R_01..R_05, CF_01..CF_03)                      │
│  │                                                      │
│  └── _CTABatchWrapper.compute_all()          ← 6 CTA    │
│        (carry, vol_mean_reversion, donchian_breakout,   │
│         momentum_ma, tsi_garch, pair_trading)           │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│          SignalAbstractionLayer (signal_abstraction.py) │
│                                                        │
│  3 种模式标准化输出：                                    │
│  ├── get_cross_sectional_signals()  → 5 子策略得分     │
│  │    喂给 FactorScoringEngine 做横截面排名              │
│  ├── get_cta_signals()              → 6 策略信号        │
│  │    get_cta_composite_signal()    → 加权合成 1 值     │
│  └── get_hybrid_signal()            → 横截面×CTA 混合   │
│       blend = (1-w)*cross_section_z + w*cta_composite   │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│            PyBrokerExecutorBuilder (pybroker_executor.py)│
│                                                        │
│  注入 signal_abstraction 后自动走统一因子池路径          │
│  未注入时保持原 cta_mode / 蓝图模式向后兼容              │
└────────────────────────────────────────────────────────┘
```

## 2. 新增因子的流程

所有新因子（无论是新的 Alpha 因子还是新的 CTA 策略）必须按以下流程添加：

### 新增 Alpha 因子（WorldQuant 风格）

1. 在 `core/ext/factors/alpha_futures/factors/` 下创建因子类，继承 `BaseFactor`
2. 使用 `@register_factor` 装饰器注册
3. 将因子编号加入 `factor_registry.py` 的 `SUB_STRATEGY_FACTOR_GROUPS` 对应子策略组
4. 无需修改 `UnifiedFactorPool` —— 它自动通过 `FactorEngine.compute_all()` 获取新因子

### 新增 CTA 策略

1. 在 `core/strategies/cta/` 下创建策略类，继承 `CTABaseStrategy`
2. 实现 `compute_signal(symbol, close, high, low, volume, ctx) → float`
3. 使用 `@register_cta_strategy` 装饰器注册
4. 在 `_CTABatchWrapper._CTA_PRIMARY_NAMES` 中添加策略规范名
5. 在 `signal_abstraction.py` 的 `DEFAULT_CTA_WEIGHTS` 中添加默认权重（可选）

### 新增信号模式

1. 在 `signal_abstraction.py` 的 `SignalMode` 枚举中新增值
2. 在 `SignalAbstractionLayer` 中实现对应的 `get_xxx_signals()` 方法
3. 输出必须 clip 到 [-1, 1]

## 3. 关键约束

### 3.1 禁止绕过

- 禁止新模块直接调用 `FactorEngine.compute_all()` 或 `compute_sub_strategy_scores_from_ohlcv()`
- 禁止新模块直接实例化 CTA 策略并调用 `compute_signal()`
- 所有信号必须通过 `UnifiedFactorPool.compute_all()` → `SignalAbstractionLayer` 路径获取

### 3.2 性能优化

- `UnifiedFactorPool` 内部缓存 `{symbol: DataFrame}`，同一品种多次调用避免重复计算
- `compute_signals_for_bar()` 从缓存读取最新 bar，首次调用时自动触发全量计算
- 如需清除缓存调用 `pool.clear_cache(symbol)` 或 `pool.clear_cache()`

### 3.3 向后兼容

- `PyBrokerExecutorBuilder.__init__` 新增 `signal_abstraction` 参数，默认 None
- 不注入时 `build()` 走原分支（cta_mode / 蓝图模式），零行为变化
- 旧的 `compute_sub_strategy_scores_from_ohlcv()` 和 `CTA_STRATEGY_REGISTRY` 保持可用

### 3.4 CTA 策略状态管理

- `_CTABatchWrapper` 内部维护策略实例，每个品种跨 bar 保持状态
- spread 依赖策略（carry, pair_trading）在 batch 计算前自动注入 spread/far_close
- tsi_garch 的 GARCH sigma 缓存由策略实例自身维护

## 4. 已删除/废弃的文件

| 文件 | 状态 | 替代方案 |
|------|------|----------|
| `core/engine/cta_executor_builder.py` | 已删除 | `PyBrokerExecutorBuilder` + `SignalAbstractionLayer` |
| `scripts/run_cta_batch.py` | 已废弃 | `UnifiedFactorPool` + `runner/common/single_backtest.py` |
| `scripts/run_cta_backtest.py` | 已废弃 | `UnifiedFactorPool` + `runner/common/single_backtest.py` |
| `runner/validation/signal_fusion.py` | 已重构 | `UnifiedFactorPool.compute_all()` 替代旧 CTAExecutorBuilder |

## 5. 涉及代码

- `core/execution/factor_pool.py`：`UnifiedFactorPool` + `_CTABatchWrapper`
- `core/execution/signal_abstraction.py`：`SignalAbstractionLayer` + `SignalMode`
- `core/execution/pybroker_executor.py`：`PyBrokerExecutorBuilder` 注入 `signal_abstraction`
- `core/execution/backtest_runner.py`：创建 `UnifiedFactorPool` + `SignalAbstractionLayer`
- `runner/common/single_backtest.py`：基于 UnifiedFactorPool 的回测辅助函数
