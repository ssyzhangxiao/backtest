# 规则21：多策略子策略划分与集成 — 5 子策略体系

**核心原则**：基于因子逻辑类别，构建 5 个独立子策略，通过集成方法形成最终信号，实现稳健绝对收益。

**子策略划分方案**：

| 子策略名称 | 使用的因子 | 逻辑核心 | 信号方向 |
|---------|---------|---------|---------|
| 趋势策略 | T_01, T_02, T_03, T_05, V_02, M_03 | 趋势确认 + 资金流确认 | 顺势交易 |
| 期限结构策略 | T_04, R_04, M_04, H_05 | Carry + 增仓/资金流共振 | Back做多，Contango做空 |
| 均值回归策略 | R_01, R_02, R_03, R_05, H_03 | 增仓背离、持仓萎缩反转 | 逆势交易 |
| 波动率突破策略 | V_01, V_03, V_04, H_04 | 持仓异动 + 价格加速度 | 突破跟进 |
| 复合共振策略 | H_01, H_02, M_01, M_02, M_05 | 多维度高阶统计共振 | 综合打分 |

**阶段二实施规则（子策略合成与集成）**：

### 21.1 子策略基类设计
- 所有子策略继承 `SubStrategyBase` 抽象基类
- 必须实现 `compute_signal(ctx, factor_data)` 方法，返回该子策略的信号
- 必须定义 `factor_list` 属性（该子策略使用的因子列表）
- 可选实现 `post_process(signal)` 做子策略特定后处理
- 通过 `self.config` 访问全局配置

### 21.2 单个子策略信号生成
- **因子标准化**：对子策略内每个因子，每天计算横截面 Z 分数（多品种）
- **方向调整**：若因子方向为反向（如 R_05），需乘 -1 调整方向
- **因子加权合成**：
  - 默认使用等权法：`sub_signal = mean(factor1_z, factor2_z, ...)`
  - 可选滚动 IC 动态权重：IC 越高权重越大
- **信号裁剪**：`position = np.clip(sub_signal, -1, 1)`

### 21.3 子策略级风控
- **波动率目标**：调整仓位使子策略预期波动率等于目标值（默认 15%）
- **最大回撤止损**：子策略净值回撤超过 8% 时，该子策略清仓并暂停 3 天
- **持仓限制**：单品种单边仓位不超过总资金的 10%

### 21.4 多策略集成（顶层模型）
- **信号合并方法**：
  - **等权叠加**（默认）：`final_signal = (signal1 + ... + signal5) / 5`，再裁剪到 [-1, 1]
  - **波动率倒数加权**：`weight_i = 1 / vol_i`，动态调整，降低高波动子策略权重
  - **基于收益率的自适应权重**：使用卡尔曼滤波或滚动优化最大化综合 Sharpe 比
  - **多数投票**：将连续信号转为方向（+1 / -1 / 0），取多数方向作为最终方向
- **顶层风控**：
  - 总杠杆限制：所有子策略叠加后的总名义仓位不超过 2 倍
  - 品种集中度：同一品种上的净持仓不超过总资金的 15%
  - 市场状态过滤：全市场波动率处于历史 80% 分位数以上时，整体仓位减半

### 21.5 因子准入标准
- 每个因子进入子策略前，需先通过 IC 检验（IC > 0.03, IR > 0.5）筛选
- 因子间相关性 > 0.7 视为冗余，保留 IC 更高的因子
- 缺失率 > 15% 的因子排除

**涉及代码**：
- `core/strategies/sub_strategies/base.py`：子策略基类
- `core/strategies/sub_strategies/trend.py`：趋势策略
- `core/strategies/sub_strategies/term_structure.py`：期限结构策略
- `core/strategies/sub_strategies/mean_reversion.py`：均值回归策略
- `core/strategies/sub_strategies/vol_breakout.py`：波动率突破策略
- `core/strategies/sub_strategies/composite.py`：复合共振策略
- `core/engine/top_level_integrator.py`：顶层策略集成器（新增）
- `core/engine/sub_strategy_adapter.py`：子策略适配器（新增）
- `core/engine/backtest_runner.py`：集成子策略体系
- `core/config/backtest_config.py`：`signal_merge_method` 配置项
- `config.yaml`：信号合并方法配置

**使用方式**：
1. 在 `config.yaml` 中设置信号合并方法：
   ```yaml
   backtest:
     signal_merge_method: equal_weight  # 可选: equal_weight/volatility_inverse/adaptive/majority_vote
   ```
2. 运行回测：`python run_backtest.py`
