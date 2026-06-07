# 规则26：交叉验证机制 — 自研引擎与 PyBroker 并行验证

**核心原则**：自研引擎不做 PyBroker 的回退方案，两套引擎并行运行用于交叉验证。

**并行验证流程**：
1. PyBroker 引擎执行主回测（`PyBrokerBacktestRunner.run()`），PyBroker 不可用时直接抛异常，不静默回退
2. 自研引擎执行独立回测（`BacktestRunner.run()`），产生 `PortfolioResult`
3. 调用 `BacktestRunner.cross_validate_with_pybroker(pybroker_result, own_result)` 进行交叉验证
4. 验证分为 4 个层次，从粗到细逐步收敛问题：

**验证层次1：净值曲线一致性（基础）**
- 归一化净值 Pearson 相关系数（评估整体趋势一致性）
- 日收益率相关系数（评估每日波动方向一致性）
- 最大绝对差异、平均绝对差异、最大百分比差异
- 最大偏离日期（定位具体交易日差异来源）

**验证层次2：核心绩效指标一致性（重要）**
- Sharpe 比、Calmar 比、Sortino 比
- 年化收益、最大回撤幅度、最大回撤发生日期
- 胜率、盈亏比、总交易次数、多空占比

**验证层次3：逐笔交易一致性（深度）**
- 开仓时间、标的、方向、价格、持仓量是否匹配
- 平仓时间、价格、盈亏是否匹配
- 持仓时间分布对比（平均持仓天数、最长/最短持仓、持仓周期直方图）

**验证层次4：因子得分序列一致性（针对因子打分回测）**
- 每日各品种的因子得分是否一致（如果是因子打分策略）
- 横截面标准化后的得分是否一致

**告警规则**：
- 净值相关系数 < 0.95 → 严重告警
- 核心绩效指标差异 > 10% → 重要告警
- 逐笔交易不一致 → 详细告警，打印前 N 笔差异交易
- 差异超过 10% 发出警告（由规则1覆盖）

**实现细节**：
- `BacktestRunner.cross_validate_with_pybroker()` 通过 date 对齐两条净值曲线，归一化到同一初始值
- 计算 Pearson 相关系数评估整体趋势一致性
- 计算日收益率相关系数评估每日波动方向一致性
- 定位最大偏离日期，便于排查具体交易日差异来源
- `BacktestConfig.cross_validate` 开关控制是否执行交叉验证

**涉及代码**：
- `core/engine/runner.py`：`BacktestRunner.cross_validate_with_pybroker()` 方法
- `core/engine/backtest_runner.py`：`PyBrokerBacktestRunner.run()` 主回测
- `core/config/backtest_config.py`：`cross_validate` 配置开关
