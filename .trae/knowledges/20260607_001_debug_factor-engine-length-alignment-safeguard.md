# FactorEngine length-alignment safeguard: TS_01/02/03 hardcoded length=100 fix

> **Type:** debug
> **Date:** 2026-06-07
> **Context:** core/factors/alpha_futures/factor_engine.py compute_all

## Summary

compute_all 末尾新增 NaN right-align 兜底，消除 'Length of values (100) does not match length of index (X)' 阻断 train_test / monte_carlo / e1 端到端的根因。

## Background

Pandas length mismatch 阻断 train_test / monte_carlo / e1 端到端验证

## Details

## 根因

TS_01/02/03 三个期限结构因子内部有 hardcoded fallback:

```python
def compute(self, near_price=None, far_price=None, close=None, **kwargs):
    if near_price is None or far_price is None:
        length = len(close) if close is not None else 100  # magic number
        return np.full(length, np.nan, dtype=float)
```

当 close 上下文未通过 needs 注入时（TS 因子 dependencies 不含 close），
fallback 返回 100 元素数组，但下游 `sub_strategy_adapter.compute_factors()`
的 `df[factor_name] = factor_values` 要求长度与 `df.index` 一致，于是抛 ValueError。

## 症状

- 阻断 `_run_period_backtest`：`Length of values (100) does not match length of index (X)`
  （X = 183 / 599 / 624 / 996 / 1180 / 1204 / 1334 等）
- `train_test` summary / `monte_carlo` 逐策略 / `e1` 跨策略回测输出 0 行
- `verify_chain` 仍 11/11 ready（验证层未发现），但 `run_validate --method all` 端到端崩

## 修复

在 `core/factors/alpha_futures/factor_engine.py:compute_all` 末尾增加边界保护：

```python
target_len = len(public_data["close"])
for _name in list(results.keys()):
    _vals = results[_name]
    if len(_vals) != target_len:
        logger.warning("因子 %s 长度 %d != 期望 %d, 自动 NaN right-align", ...)
        _aligned = np.full(target_len, np.nan, dtype=float)
        _n = min(len(_vals), target_len)
        if _n > 0:
            _aligned[-_n:] = _vals[-_n:]
        results[_name] = _aligned
```

## 取舍

- 不下沉到每个因子（破坏规则 8 最少变更；TS_01/02/03 是 hardcoded magic number 而非逻辑问题）
- 不修改 TS_01/02/03 fallback（修一个 magic number 不解决契约问题，下个因子还会踩坑）
- 边界保护在系统集成处，符合"系统集成处不信任任何内部"原则

## 验证

- 新增单测 `test_compute_all_aligns_misaligned_factor_length`：注入 _ShortFactor (返回 100) 验证强制对齐
- 端到端：verify_chain 11/11、train_test 5 DataFrame 全部含真实数据、monte_carlo 6 策略 × 9 列
- 236/236 单元测试通过（235 原有 + 1 新增）


## Key Takeaways

- 1. 系统集成边界必须做契约保护：compute_all 是 30 因子唯一汇聚点，长度契约在该处强制最稳妥
- 2. 不要信任因子内部 fallback：magic number（如 hardcoded 100）早晚与 df 长度冲突
- 3. right-align with NaN pad 是时间序列因子的安全兜底（前置 NaN 表示因子尚未有数据）
- 4. 验证层（verify_chain）≠ 业务流层（end-to-end），必须两者都跑

## Related

core/factors/alpha_futures/factor_engine.py:141-165, core/factors/alpha_futures/factors/ts_01.py:34, core/engine/sub_strategy_adapter.py:203
