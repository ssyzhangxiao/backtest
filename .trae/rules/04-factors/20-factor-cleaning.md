# 规则20：因子数据清洗与工程化 — 换月/交割/涨跌停处理

**核心原则**：因子计算前必须进行数据清洗，确保无前瞻性偏差和脏数据污染。

**1. 主力合约换月处理**：
- 使用复权价格（后复权）构建连续价格序列，消除换月跳空
- 换月日前后 3 个交易日的 `OI` 及 `DELTA(OI)` 强制设为 `NaN`
- 滚动窗口函数（`SUM`, `MEAN`）遇到 `NaN` 自动跳过，不前向填充

**2. 交割月数据剔除**：
- 进入交割月前 N 个交易日（可配置，默认 5 天）的全部数据剔除
- 持仓量使用全市场该品种所有合约的总持仓量，而非单合约持仓

**3. 涨跌停板过滤**：
- 若 `|(open - prev_close) / prev_close| > threshold`（默认 0.06），则当日：
  - `INTRADAY_RET` 直接置 0
  - 所有依赖 `high-low`、`(C-L)-(H-C)` 等日内结构的因子值置为 `NaN`

**4. 跳空缺口修复（全局）**：
- 基础公式：`OPEN_ADJ = OPEN * w + DELAY(CLOSE,1) * (1-w)`
- 自适应权重：w 根据该品种历史跳空延续率动态计算，范围 [0.2, 0.8]
- `INTRADAY_RET = (CLOSE - OPEN_ADJ) / OPEN_ADJ`，作为所有日内收益替代量

**5. 无前瞻性标准化（强制）**：
- 禁止使用全序列 `mean/std` 的 `ZSCORE`
- 强制使用滚动窗口标准化：`ZSCORE(x, window)`，仅用过去 window 天数据
- 或使用扩张窗口标准化：`ZSCORE_expanding(x)`，从第一根 K 线到当前 t
- 所有 `CORR`、`RANK` 也必须基于滚动窗口或扩张窗口

**6. 统一后处理**：
- 缩尾（Winsorize）：每个因子计算完成后，按 1% 和 99% 分位数截断
- 缺失值填充：默认不填充（保留 NaN），策略层自行决定前向填充或剔除
- 横截面标准化（多品种）：`factor = (factor - mean) / std`，按日期计算
- 时序标准化（单品种）：`factor = (factor - rolling_mean(60)) / rolling_std(60)`

**涉及代码**：
- `core/factors/factor_review.py`：因子复核与数据质量检查
- `core/factors/data_cleaner.py`：换月/交割/涨跌停处理
- `core/factors/gap_fixer.py`：跳空缺口自适应修复
- `core/factors/normalizer.py`：无前瞻性滚动标准化
