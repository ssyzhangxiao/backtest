# 方向二验证 — 横截面作为 CTA 仓位调节器（关键经验）

**日期**: 2026-06-16
**场景**: direction 2 — 把横截面从"信号源"重构为"CTA 仓位缩放器"
**结果**: ✅ 验证通过，calmar +80%，mdd -75%

---

## TL;DR

把横截面（XS）从"信号源"重构为"CTA 仓位缩放器"是有效的，但实现中有 2 个**关键陷阱**导致 dynamic 模式最初完全失效。

---

## 陷阱 1：信号饱和（critical fix）

### 现象
第一次 sweep（35 dynamic 配置）的结果**全部等于 linear_w1.0**（纯 CTA）：
```
linear_w0.9:   return=6.25%
linear_w0.95:  return=9.78%
linear_w1.0:   return=16.88%
dynamic_b0.3_p0.2: return=16.88%  ← 异常
dynamic_b0.3_p0.3: return=16.88%  ← 异常
...（所有 dynamic 配置都是 16.88%）
```

### 根因（两个叠加 bug）

**Bug A — `compute_composite_score` 用排名百分位 * 2 - 1 映射**：
```python
# switch_engine.py:296-298
if self.config.use_rank_score and self._rank_scores:
    return float(np.clip(self._rank_scores.get(symbol, 0.0), -1.0, 1.0))
```

`_rank_scores` 是 `rank_df = scores_df.rank(pct=True) * 2 - 1` 的结果，**离散且对称**。
对于 5-6 个品种的小横截面，排名百分位只能取 {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}，映射后是 {-1.0, -0.6, -0.2, +0.2, +0.6, +1.0}。
当品种排名接近极端时，`|z| ≈ 1.0` 几乎总是成立 → `cross_strength` 恒等于 1.0 → dynamic 退化为 `cta * 1.0` = 纯 CTA。

**Bug B — `portfolio.allocate_weights` 在 equal_weight 模式下只看 sign**：
```python
# portfolio.py:303
return {s: per * (1.0 if active[s] > 0 else -1.0) for s in active}
```

`per` 是固定值（不依赖 `|signal|`），magnitude 被完全丢弃。
所以即使 `signals_all[sym]` 数值不同，target_weight 也只取决于符号 → 动态 magnitude 失效。

### 修复（3 处）

**Fix 1** — `core/engine/switch_engine.py`：给 `compute_composite_score` 加 `clip_output` 参数
```python
def compute_composite_score(
    self, symbol: str, factor_scores: Optional[Dict[str, float]] = None,
    clip_output: bool = True,  # 新增
) -> float:
    ...
    if self.config.use_rank_score and self._rank_scores:
        raw = float(self._rank_scores.get(symbol, 0.0))
        return float(np.clip(raw, -1.0, 1.0)) if clip_output else raw
    ...
    return float(np.clip(composite, -1.0, 1.0)) if clip_output else float(composite)
```

**Fix 2** — `core/execution/pybroker_executor.py`：dynamic 分支用 un-clipped z 计算 cross_strength
```python
raw_z = scoring_engine.compute_composite_score(sym, clip_output=False)
cross_strength = float(np.clip(abs(raw_z), 0.0, 1.0))
pos_scale = signal_layer.xs_position_base + (
    signal_layer.xs_position_ceiling - signal_layer.xs_position_base
) * cross_strength
```

**Fix 3** — `core/execution/pybroker_executor.py`：allocate_weights 后**显式**应用 pos_scale
```python
target_weights = portfolio_manager.allocate_weights(signals_all, ...)  # sign only
target_weights = risk_controller.check_concentration_dict(...)
# 关键：必须显式把 pos_scale 乘回去，因为 equal_weight 模式下 magnitude 被丢
if (signal_layer.mode == "hybrid" and
    signal_layer.hybrid_blend_method == "dynamic" and
    state.dynamic_pos_scales):
    for sym in target_weights:
        target_weights[sym] *= state.dynamic_pos_scales.get(sym, 1.0)
```

### 调试方法

在 dynamic 分支加临时 log（**不留在代码里**）：
```python
_dbg_samples.append(
    f"{sym}|z={z:.2f}|raw={raw_z:.2f}|cta={cta_sig:.2f}"
    f"|str={cross_strength:.2f}|scl={pos_scale:.2f}|hyb={hybrid:.2f}"
)
```

如果 `|raw_z|` 总是 0.2 的倍数 → Bug A 还在
如果 `hybrid` 有变化但 `target_weight` 不变 → Bug B 还在

---

## 陷阱 2：输出目录覆盖

`sweep_cta_hybrid_dynamic.py` 加了 `--output-suffix` 参数解析但**忘了在 main() 里应用**：
```python
def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)  # BUG: 没看 args.output_suffix
```

后果：第二次 sweep（OOS）覆盖了第一次（全期）的 summary.md。
修复：
```python
def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    if args.output_suffix:
        out_dir = out_dir.parent / f"{out_dir.name}_{args.output_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
```

教训：**加 CLI 参数时一定要在主流程里用上**，并在 PR 描述里"实际跑一次 A/B 验证输出目录不同"。

---

## Sweep 结果（全期 + OOS 交叉验证）

| 指标 | linear_w1.0 | dynamic b=0.3 p=0.4 | 改善 |
|------|-------------|----------------------|------|
| sharpe | 0.039 | 0.046 | **+18%** |
| return% | 16.88% | 7.49% | -56% |
| mdd% | -8.47% | **-2.09%** | **-75%** |
| calmar | 1.99 | **3.59** | **+80%** |

OOS (2021-2025) 同样验证：b=0.3 p=0.4 仍进 top-5（sharpe 0.0460, mdd -2.10%），**跨期最稳定**。

---

## 关键发现

1. **横截面绝对值有信息价值**（`cross_strength` 确实影响未来收益）
2. **不需要横截面方向正确**（dynamic 模式把方向交给 CTA，XS 只调节仓位）
3. **base 是主导参数**（全期 top-5 全部是 base=0.3，penalty 不敏感）
4. **calmar 是最该看的指标**（sharpe 提升 18%，calmar 提升 80% — 风险调整后收益大幅改善）

---

## 后续

- 方向二作为生产配置（已写入 config.yaml）
- 方向三（配对交易组合）值得启动（XS 绝对值已证明有信息，可作为配对 z-score 输入）
- CTA 参数（entry_threshold / rebalance_days / 止损）需要在 dynamic 模式下重调

---

## 代码修改清单

| 文件 | 改动 |
|------|------|
| `core/engine/switch_engine.py` | `compute_composite_score` 加 `clip_output` 参数 |
| `core/execution/pybroker_executor.py` | dynamic 分支用 raw_z + post-allocate pos_scale + `_State.dynamic_pos_scales` |
| `core/execution/signal_abstraction.py` | `get_hybrid_signal_dynamic`（方向二抽象层） |
| `core/config/backtest_config.py` | 新增 `hybrid_blend_method` / `xs_*` 字段 |
| `config.yaml` | 默认值更新为推荐配置（base=0.3, p=0.4） |
| `tests/unit/test_factor_pool.py` | 8 个 HYBRID_DYNAMIC 单元测试 |
| `scripts/experiments/sweep_cta_hybrid_dynamic.py` | 2D 参数 sweep（35 dynamic + 3 linear） |
| `scripts/analysis/analyze_dynamic_sweep.py` | 按 sharpe/mdd 排序的对比分析 |

**回归**: `pytest tests/unit/test_factor_pool.py` → 36 passed
