# 架构评判与修改意见（第二轮深度审查）

## 摘要

基于 Phase 1-5 重构后的深度架构评判。识别出6类遗留问题：双风控类并存、策略注册机制冗余、config.yaml 与 BacktestConfig 不同步、session_state 仍引用废弃 environment、broker_adapter 职责过重、测试覆盖不足。提出具体修改方案和优先级。

## 背景

系统从 v2（市场环境判断→策略切换）演进到 v3（因子打分→调仓），Phase 1-5 重构已完成核心改造，但深度审查发现6类遗留问题。

## 已完成的改进（Phase 1-5）

| 改进项 | 状态 |
|--------|------|
| StrategySwitchEngine → FactorScoringEngine | ✅ 完成，保留兼容别名 |
| BacktestConfig 移除 fusion_mode/regime_filter_enabled | ✅ 完成 |
| V3RegimeParamManager 移至 core/param_manager.py | ✅ 完成 |
| V3RegimeAwareRunner 移至 scripts/analysis_runner.py | ✅ 完成 |
| RiskController 独立风控类 | ✅ 完成 |
| DataProvider 抽象接口 | ✅ 完成 |
| BacktestConfig.from_yaml() | ✅ 完成 |
| DEFAULT_FACTOR_WEIGHTS 统一 | ✅ 完成 |
| 单元测试 44 个 | ✅ 通过 |

## 仍存在的架构问题

### 问题1：双风控类并存（RiskManager vs RiskController）

**现状**：
- `core/risk_manager.py`（RiskManager）：旧类，被 `pages/backtest.py` 和 `core/__init__.py` 引用
- `core/risk_controller.py`（RiskController）：新类，仅被测试引用

**问题**：
- 两个类功能高度重叠（止损、仓位限制），但接口不同
- RiskManager 依赖 PyBroker ExecContext，RiskController 是纯逻辑类
- 新代码应使用 RiskController，但旧代码仍依赖 RiskManager

**建议**：
1. RiskManager 保留为 PyBroker 适配层（包装 RiskController）
2. RiskManager 的核心逻辑委托给 RiskController
3. 逐步迁移调用方

### 问题2：策略注册机制冗余（registry.py vs strategy_library/）

**现状**：
- `core/strategies/registry.py`：轻量注册表，STRATEGY_REGISTRY 字典 + get_strategy_class()
- `core/strategy_library/__init__.py`：重量级管理器，StrategyProfile + StrategyLibrary

**问题**：
- registry.py 只做类名映射，strategy_library 做环境映射+性能档案
- 两者职责不同但命名混淆，容易误用
- FactorScoringEngine 使用 StrategyLibrary，但实际只用到了策略参数

**建议**：
1. 合并为 `core/strategy_registry.py`
2. StrategyRegistry 类同时支持：类映射、参数获取、性能档案
3. 删除 `core/strategies/registry.py` 和 `core/strategy_library/` 目录

### 问题3：config.yaml 与 BacktestConfig 不同步

**现状**：
- config.yaml 仍有 `fusion_mode: true` 和 `regime_filter_enabled: false`（已从 BacktestConfig 移除）
- config.yaml 用 `strategy_weights`，BacktestConfig 用 `factor_weights`
- config.yaml 用 `rebalance_frequency: "3d"`，BacktestConfig 用 `rebalance_days: 3`
- config.yaml 有 `strategy_switching` 整节（已废弃）

**建议**：
1. 清理 config.yaml 中的废弃字段
2. 统一命名：`strategy_weights` → `factor_weights`
3. `rebalance_frequency: "3d"` → `rebalance_freq: 3`
4. 删除 `strategy_switching` 节

### 问题4：session_state 仍引用废弃 environment

**现状**：
- `utils/session_state.py:43` 仍 `from core.environment import EnvironmentAdapter`
- `core/environment.py` 已被删除（或标记废弃）

**建议**：
1. 将 `compute_env_cached()` 改为使用 `MarketRegimeDetector`
2. 或直接删除该函数（如果 Streamlit 页面不再使用环境计算）

### 问题5：broker_adapter.py 职责过重

**现状**：
- 文件 2000+ 行，包含 7 个类
- PyBrokerDataSource、RegimeIndicator、StrategyExecutorFactory、PyBrokerBacktestRunner 等全在一个文件

**建议**：
1. 拆分为独立模块：
   - `core/engine/data_source.py` → PyBrokerDataSource
   - `core/engine/executor_factory.py` → StrategyExecutorFactory
   - `core/engine/backtest_runner.py` → PyBrokerBacktestRunner
   - `core/engine/regime_indicator.py` → RegimeIndicator
2. broker_adapter.py 仅保留适配逻辑

### 问题6：根目录 config.py 兼容层

**现状**：
- `/config.py` 是向后兼容层，发出 DeprecationWarning
- 无任何文件从根目录 config.py 导入

**建议**：
- 可以安全删除，但用户选择保留
- 若保留，应定期提醒清理

## 修改优先级

| 优先级 | 问题 | 风险 | 工作量 |
|--------|------|------|--------|
| P0 | config.yaml 同步清理 | 配置不一致导致运行时错误 | 小 |
| P1 | session_state 引用废弃模块 | 导入失败 | 小 |
| P1 | 双风控类统一 | 逻辑不一致 | 中 |
| P2 | 策略注册机制合并 | 维护成本 | 中 |
| P2 | broker_adapter 拆分 | 可读性 | 大 |
| P3 | 根目录 config.py 清理 | 无实际风险 | 小 |

## 要点

1. 双风控类（RiskManager/RiskController）必须统一，否则新代码用 RiskController 旧代码用 RiskManager 导致行为不一致
2. config.yaml 是单一数据源，必须与 BacktestConfig 字段完全同步，否则 from_yaml() 加载的配置与代码预期不符
3. session_state.py 引用已删除的 core.environment 会导致 Streamlit 应用启动失败
4. broker_adapter.py 2000+ 行是最大的可维护性瓶颈，应拆分为5个独立模块
5. 策略注册机制（registry.py + strategy_library/）功能重叠应合并

## 相关文件

- [core/risk_manager.py](file:///Users/luojiutian/Documents/backtest/core/risk_manager.py)
- [core/risk_controller.py](file:///Users/luojiutian/Documents/backtest/core/risk_controller.py)
- [core/strategies/registry.py](file:///Users/luojiutian/Documents/backtest/core/strategies/registry.py)
- [core/strategy_library/__init__.py](file:///Users/luojiutian/Documents/backtest/core/strategy_library/__init__.py)
- [config.yaml](file:///Users/luojiutian/Documents/backtest/config.yaml)
- [utils/session_state.py](file:///Users/luojiutian/Documents/backtest/utils/session_state.py)
- [core/engine/broker_adapter.py](file:///Users/luojiutian/Documents/backtest/core/engine/broker_adapter.py)
