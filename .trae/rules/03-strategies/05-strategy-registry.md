# 规则5：策略注册统一 — 多策略横截面打分

**核心原则**：彻底移除单策略绑定机制，所有策略通过横截面打分进行动态仓位分配，统一使用 `CrossSectionalStrategy` 管理多策略组合。

**具体规则**：
- 统一使用 `core/strategy_registry.py` 的 StrategyLibrary 管理策略档案
- 所有策略类必须实现 `compute_score` 方法，返回归一化到 [-1, 1] 的因子得分
- 不再使用单策略 `execute` 方法做多/空二元决策，改为因子得分输出
- 多策略组合通过 `CrossSectionalStrategy` 进行横截面标准化 + 排名叠加
- 策略发现、参数获取、性能档案全部走统一入口

**涉及代码**：
- `core/strategies/cross_sectional.py`：多策略横截面打分引擎
- `core/strategy_registry.py`：策略库与档案管理
- `core/strategies/strategy_*.py`：各策略的 `compute_score` 方法
