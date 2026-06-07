# 规则27：策略基类设计 — 可配置化与可扩展性

**核心原则**：`BaseStrategy` 抽象基类提供公共展期、止损、持仓管理，参数可配置化，子类无需重复实现。

**基类职责**：
- `_check_rollover()`：展期检查与自动平仓
- `_init_position_session()` / `_register_*_entry()` / `_clear_position()`：持仓会话管理
- `_check_trailing_stop_long/short()`：百分比跟踪止损
- `_check_time_stop()`：时间止损（持仓超过 N 天强制平仓）
- `_compute_oi_change()`：持仓量变化率计算
- `_compute_oi_divergence()`：价格与持仓量背离检测

**可配置化要求**：
- 阈值参数（如 `oi_change_threshold=0.03`、`price_change_threshold=0.005`）通过 `__init__` 或 config 字典传入
- 止损参数（如 `stop_pct`、`time_stop_days`）支持动态配置，允许子类为每个标的单独设置
- 所有方法添加准确类型注解（`numpy.typing.ArrayLike` 等）

**容错机制**：
- `_check_rollover` 先判断 `hasattr(ctx, 'is_dominant')` 再取值，避免过度 try-catch
- 所有数值计算包裹 try/except，返回安全默认值（0.0 或 False）
- 持仓状态检查失败时，不阻塞交易执行

**涉及代码**：
- `core/strategies/base.py`：`BaseStrategy` 抽象基类
- `core/strategies/sub_strategies/base.py`：子策略基类 `SubStrategyBase`
