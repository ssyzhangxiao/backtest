# 规则30：数据质量 — 禁止降级合成，必须使用真实数据源

**核心原则**：回测数据必须来自真实数据源（TqSDK），禁止在代码中合成/代理缺失列作为降级方案。

## 具体规则

### 1. spread（期限结构）数据
- **必须**使用 TqSDK 提供的真实 `spread` 列（`tqsdk` 数据源天然包含远近月价差）
- **禁止**用 `close / SMA(close, N)` 或任何价格变换合成 spread
- **允许**用 `far_close` 列计算 spread（仅当 `far_close` 来自真实合约数据，非合成）
- 若无真实 spread 数据，carry/pair_trading 策略应返回 0 信号（空仓），不允许合成
- **数据补足**：若缓存数据缺失 spread/far_close，运行器（如 `_run_single`）应自动通过 `DataLoader(data_source="tqsdk") + build_spread_pairs()` 下载真实数据并缓存

### 2. TqSDK 自动补数据流程
- 当 `_run_single()` 检测到策略需要 spread/far_close 但缓存数据中无此列时：
  1. 通过 `DataLoader(data_source="tqsdk")` 加载该品种数据
  2. 调用 `identify_dominant_contracts()` → `build_continuous_series()` → `build_spread_pairs()`
  3. 调用 `get_pybroker_df()` 获取含 spread/far_close 列的完整数据
  4. 将结果写入 `data_cache/{EXCHANGE}_{PRODUCT}_spread.pkl` 供后续复用
  5. 若 TqSDK 下载失败，直接返回 None（策略得 0 信号）

### 3. 其他数据列
- 任何策略所需的自定义列（如 volatility surface、order flow 等）必须由数据源原生提供
- 不允许在策略代码或执行器代码中通过 close/high/low 反向推导并构造缺失列

### 4. 降级约束
- `_run_single()` 等运行函数发现数据缺失时，应直接使依赖该数据的策略返回 0 信号，不允许在运行层合成补丁
- 若策略因数据缺失无法交易，该结果应在汇总表中明确标注（如 trades=0 且 period 有效），以便区分"策略信号无效"和"数据源问题"

## 原因

- 合成数据引入人为假设（如 `close/SMA` 假设价格偏离均线 = 期限结构），这些假设在 OOS 中无保证
- 合成数据会掩盖数据源的真实问题（如 TqSdk 配置错误、数据未预加载）
- 真实 spread 才能反映期货展期收益的实际可交易性
- TqSDK 自动补数据避免用户每次手动 `--tqsdk`，降低摩擦

## 涉及代码
- `scripts/run_cta_batch.py:_run_single()` — 注入 spread 到策略状态，只从 `spread` 或 `far_close` 列取真实数据；缺失时自动 TqSDK 加载
- `scripts/run_cta_batch.py:_ensure_tqsdk_spread_data()` — TqSDK 下载 + build_spread_pairs + 缓存
- `core/strategies/cta/carry_strategy.py` — 依赖真实 spread，无则返回 0
- `core/strategies/cta/pair_trading.py` — 依赖真实 spread，无则返回 0
