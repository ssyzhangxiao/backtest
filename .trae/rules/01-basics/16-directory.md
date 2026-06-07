# 规则16：模块目录结构 — 职责单一，接口清晰

**核心原则**：每个模块目录只做一件事，模块间通过明确接口交互。

**目录结构**（当前实际状态，标注"规划中"的模块尚未实现）：
```
core/
├── config/            # 配置管理（BacktestConfig + 因子/止损/验证配置）
├── factors/           # 因子模块（24因子体系 + 评估 + 变换 + 筛选 + 复核 + 清洗）
│   └── alpha_futures/ # 新因子库（基于抽象基类的独立因子类 + 注册表 + 引擎调度）
├── multi_tf/          # 多时间框架模块（规划中）
├── risk/              # 止损优化模块（追踪+时间+复合止损）
├── validation/        # 回测验证模块（蒙特卡洛+敏感性）
├── engine/            # 回测引擎（PyBroker+自研+策略集成）
│   ├── backtest_runner.py    # PyBroker 主回测运行器
│   ├── runner.py             # 自研验证引擎
│   ├── switch_engine.py      # 因子打分引擎（5子策略信号动态加载）
│   ├── strategy_executor.py  # 策略执行器工厂
│   ├── strategy_indicators.py# 策略指标注册表 + 退出钩子注册表（解耦核心）
│   ├── sub_strategy_adapter.py# 子策略适配器（连接因子库与子策略体系）
│   ├── top_level_integrator.py# 顶层策略集成器（信号合并）
│   ├── rolling_ic.py         # 滚动IC动态权重引擎
│   ├── factor_decay.py       # 因子衰减监控器
│   └── pybroker_data_source.py# PyBroker 数据源封装
├── strategies/        # 策略实现（5子策略 + 基类 + 横截面打分）
│   └── sub_strategies/# 5子策略：趋势/期限结构/均值回归/波动率突破/复合共振
├── performance/       # 绩效评估
└── monitor/           # 策略监控（规划中）

# 以下模块已移除，功能由子策略体系覆盖：
# ├── adaptive/        # 已移除（规则10）
# ├── position/        # 已移除（规则12）
# ├── instrument/      # 已移除（规则14）
# └── market_regime/   # 已移除（兼容性桩保留在 core/engine/runner.py 和 core/__init__.py）

runner/                # 编排层（仅调用 core/ 和 utils/）
├── common/            # 通用工具
├── data/              # 数据加载与预处理
├── strategy/          # 策略选择与权重
├── backtest/          # 回测执行与实验
├── optimization/      # 参数优化
├── validation/        # 验证流程
└── report/            # 报告生成
```

**具体规则**：
- 新增模块必须在上述目录结构中，不得在 core/ 根目录新建文件
- runner/ 是编排层，不实现核心逻辑，仅调用 core/ 和 utils/
- 模块间依赖方向：strategies → factors → engine，不得反向依赖
- 每个模块的 `__init__.py` 必须导出公共接口，隐藏内部实现
- 跨模块调用必须通过接口，不得直接访问其他模块的内部变量
