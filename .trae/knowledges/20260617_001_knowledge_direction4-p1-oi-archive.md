# 方向四 P1：持仓量衍生信号 OOS 验收（2026-06-17）

## 结论
**P1 不达标，已归档为研究分支。**

## OOS 实测（2021-01-01 ~ 2024-12-31，6 品种）

| 实验 | ret% | sharpe | mdd% |
|------|------|--------|------|
| baseline_6cta | 45.00% | 0.1645 | -2.99% |
| pure_oi | -0.28% | 0.0008 | -9.96% |
| fusion_oi@0.10 | 13.66% | 0.0387 | -5.82% |

## 验收项
- ✗ pure_oi Sharpe=0.0008 < 0.02（OI 信号本身无独立盈利能力）
- ✗ fusion_oi 提升 = -76.5% < 10%（融合反而拉低组合 76.5%）
- ⚠ |corr|=0 不可信（pure_oi 无持仓，OI 信号与 CTA 组合的收益方差为 0）

## 不达标原因分析
1. OI 信号在 OOS 期间波动剧烈（std=0.4~0.7），但缺乏动量持续性
2. entry_threshold=0.2 下 pure_oi 几乎不触发交易（导致 ret=0、sharpe≈0）
3. 0.10 权重混入组合后，整体仓位被 OI 噪声稀释，反而拉低夏普

## 实施过程中修复的 2 个关键 Bug

### Bug 1：OI 策略未注册到 factor pool
- **现象**：`compute_all()` 返回的 oi_signal 列全为 NaN
- **原因**：`core/strategies/cta/__init__.py` 缺少 `from core.strategies.cta.oi_strategy import OISignalStrategy`
- **修复**：添加 import 行，触发 `@register_cta_strategy` 装饰器

### Bug 2：CTA 合成权重被忽略
- **现象**：`cta_composite_weights={"oi_signal": 1.0}` 不生效，3 个实验结果完全一致
- **原因**：`core/execution/pybroker_executor.py::_precompute_signals` 中 CTA 合成用等权 (1/N)，绕过了 `signal_layer.cta_composite_weights`
- **修复**：executor 改为优先读取 `signal_abstraction.cta_composite_weights`，否则回退到 `DEFAULT_CTA_WEIGHTS` 归一化

### Bug 3（次要）：装饰器误用
- **现象**：`@register_cta_strategy("oi_signal", OISignalStrategy)` 报 `TypeError: 'NoneType' object is not callable`
- **原因**：registry 是函数 `register_cta_strategy(name, cls) -> None`，不支持装饰器用法
- **修复**：改为普通函数调用 `register_cta_strategy("oi_signal", OISignalStrategy)`

## 涉及代码
- `core/factors/oi_signal.py`（新增）
- `core/strategies/cta/oi_strategy.py`（新增）
- `core/strategies/cta/__init__.py`（添加 OISignalStrategy 导入）
- `core/execution/factor_pool.py`（已含 oi_signal 注入逻辑）
- `core/execution/pybroker_executor.py`（修复 CTA 合成权重读取）
- `core/config/backtest_config.py`（添加 cta_composite_weights 字段 + overrides 读取）
- `core/execution/signal_abstraction.py`（支持 cta_composite_weights）
- `scripts/exp_oi_signal.py`（实验脚本）

## 后续
方向四 P2/P3/P4 依赖 P1 达标，**全部跳过**。
可考虑 OI 信号的变体（如 OI 趋势 + ATR 止损），或转向其他正交源（如波动率自适应仓位）。
