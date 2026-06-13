# 规则31：CTA 策略优化管线（7 步标准流程）

**核心原则**：CTA/因子策略的优化不是"找更好指标"，而是按以下步骤依次推进。

---

## 七步标准管线

### Step A：信号量纲

**规则**：
- 信号必须是可加总的：方向（-1/0/+1）或 score，统一量纲
- **禁止**：`raw / (atr / slow_ma) / 3` 这类量纲混乱的表达式
- **应使用**：`np.tanh((fast_ma - slow_ma) / price * scaling)` 或 `(RSI-50)/30`
- 阈值必须可读：0.2–1.0 范围

### Step B：Regime 开关

**规则**：
- 趋势策略配标准 ADX（+DM/-DM/TR），ADX>20 且 +DI>-DI 才允许多头
- 均值回复只在高波动（>70% 分位）+ ER 低时开仓
- 震荡市策略在趋势 regime 归零信号

### Step C：退出/止损 — 执行器统一管理

**规则**：
- **禁止**在策略内部实现退出逻辑
- 退出栈优先级：信号归零 > 固定止损 > 移动止损 > 时间止损 > 信号逆转 > 熔断
- 策略专属退出参数：趋势策略(donchian) max_holding=45/mult=2.0，震荡策略(carry/vol) max_holding=20/mult=1.2

### Step D：头寸规模 = 风险预算

**规则**：
- `position_pct = min(risk_pct * |signal| / (stop_distance/price), max_pos)`
- 趋势市场：`stop_distance = ATR_mult × ATR`
- 震荡市场：`stop_distance = stop_loss_pct × price`
- 若策略提供 sigma，叠加波动率平价：`position_pct *= target_vol / (sigma × √252)`

### Step E：参数稳定性 — 平原寻优

**规则**：
- 必须做 ±20% 扰动测试，保留变化 <15% 的区域
- 子样本（牛/熊/震荡/高波）Sharpe 不异号

### Step F：Walk‑Forward 滚动验证

**规则**：
- 每年滚动训练/测试，替代单次 IS/OOS 切分
- 仅跨多段 OOS 都活用的参数进候选集

### Step G：实盘过渡

**规则**：
- Paper 1–3 个月，对比实盘滑点手续费
- 监控换手率、回撤、成交异常、相关性漂移

---

## 策略 — 执行器职责分离架构

### 策略职责

1. `compute_signal()` → signal [-1, 1]
2. `set_state("market_state", "trend"/"oscillation")`
3. `set_state("sigma", value)` — 供执行器做波动率平价
4. `set_state("adx", value)` — 供执行器参考

### 执行器职责

1. 读取信号 + market_state + sigma
2. 四层退出栈 + 策略专属参数
3. 风险预算 + 波动率平价仓位

---

## 6 策略当前实现

| 策略 | 核心信号 | 增强功能 | 专属退出参数 |
|------|----------|----------|-------------|
| **donchian_breakout** | 通道突破强度 × 动量因子(1+0.2×ret/ATR) | 标准ADX(+DM/-DM/TR) + 动态ATR(低波0.3/高波0.7) | max_holding=45, atr_mult=2.0 |
| **momentum_ma** | (RSI-50)/30 偏离 + 多周期(5/20/60)至少两个同向 | ADX方向限制(+DI/-DI) + RSI与动量加权融合 | 默认 |
| **vol_mean_reversion** | 滚动波动率z-score × 效率比(ER)动态方向窗口 | 增量波动率 + ER动态窗口 + 二次确认(连续2根同向) | max_holding=20, atr_mult=1.2 |
| **tsi_garch** | 滚动t统计量(OLS斜率/标准误) | GARCH sigma降级为风险平价调节器 + 更新频率5bar | max_holding=25, target_vol=0.15 |
| **carry** | EMA平滑spread z-score × bootstrap置信区间 × 期限斜率 | EMA替代卷积(消除尾部偏差) + 斜率增强 | max_holding=20, atr_mult=1.2 |
| **pair_trading** | 动态β(rolling OLS 90天) spread z-score → 连续信号 | ADF协整检验(每月, p<0.05) + 无死区 | max_holding=20, atr_mult=1.2 |

---

## 评价标准

| 阶段 | 可接受 | 好 | 优秀 |
|------|--------|----|------|
| OOS 盈利比例 | > 50% | > 60% | > 70% |
| OOS/IS 收益比 | > 0.3 | > 0.5 | > 0.8 |
| OOS Sharpe | > 0 | > 0.5 | > 1.0 |
| OOS Calmar | > 0.1 | > 0.5 | > 1.0 |
