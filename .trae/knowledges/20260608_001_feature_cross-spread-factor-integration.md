# Cross-Spread Factor 集成至 factor_combo_ic 验证 Pipeline

> **Type:** refactor + feature
> **Date:** 2026-06-08
> **Context:** runner/validation/factor_alpha24.py:factor_combo_ic_validation

## Summary

将跨品种价差因子（CU-NI/CU-ZN/RB-I/RB-J/AU-AG/FU-BU 等强 IC 配对）加入 factor_combo_ic 验证 pipeline。
实现路径：按品种聚合为单一 `XSPR_FACTOR`，避免单配对 2 腿 IC 样本不足问题。

## Background

- 用户指定强 IC 配对（CU-NI、CU-ZN、RB-J、RB-I、AU-AG、FU-BU）需作为独立候选因子
- 原 `factor_combo_ic_validation` 仅评估 24 alpha 因子，无跨品种信号
- 跨品种价差因子 `compute_pair_spread_factor` 已存在于 `core/factors/alpha_futures/cross_spread.py`

## Design 决策

**为什么不直接用每对配对作为一个独立因子？**
- 单配对 (A, B) 跨截面只有 2 个品种 → 横截面 IC 计算需 len(g) >= 3
- 2 腿 IC 计算无意义（rank corr 恒为 ±1）

**为什么按品种聚合为 XSPR_FACTOR？**
- 同一品种可参与多个配对（CU 在 XCU_NI/XCU_ZN/XAL_CU）
- 按 (date, symbol) 取所有配对信号均值 → 单一聚合因子
- 横截面样本 8 品种（参与配对的）→ 满足 n>=3 约束
- 因子数稳定 +1（不随配对数膨胀），便于回测复用

## 涉及代码

- `core/factors/alpha_futures/cross_spread.py:CHAIN_PAIRS` 扩展（6 → 13 配对）
- `core/factors/alpha_futures/cross_spread.py:STRONG_IC_PAIRS` 新增
- `runner/validation/factor_alpha24.py:_compute_pair_signal` 新增（按日期对齐 A/B 收盘价）
- `runner/validation/factor_alpha24.py:_build_cross_spread_panel` 新增（聚合为 XSPR_FACTOR）
- `runner/validation/factor_alpha24.py:factor_combo_ic_validation` 集成（注入面板）

## 关键 Bug 修复

1. **日期类型错位**（注入 0 行）
   - 原因：主面板 date 来自 ohlcv 原值，cross_spread_panel date 已转 Timestamp
   - 修复：ret_lookup 键统一用 `pd.to_datetime()` 转换

2. **样本不足过滤**（因子不出现在 single.csv）
   - 原因：单配对 2 腿导致 groupby 跳过（len(g) < 3）
   - 修复：按品种聚合后因子 XSPR_FACTOR 覆盖 8 品种

## 验证结果

```
跨品种价差面板 (XSPR_FACTOR): 4932 行, 8 品种, 配对数 6
跨品种价差因子注入: 4815 行 (未匹配 ret 的 30 行)
面板规模: 757785 行, 1972 交易日, 32 因子
候选因子 (abs IC>=0.03): 7/32 — ['TS_03', 'TS_composite', 'TS_01', 'TS_02', 'M_05', 'V_02', 'H_01']
XSPR_FACTOR: raw_mean_ic=-0.0243, abs=0.0243, sign=-1, 不达标但量级合理
```

## 后续优化

- XSPR_FACTOR 当前 |IC| = 0.0243，未过 0.03 阈值；可考虑加权聚合（按配对历史 IC 加权）替代简单均值
- 后续接入回测时，XSPR_FACTOR 需先 winsorize（同 24 alpha 因子）
