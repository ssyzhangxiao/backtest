# 规则33：OOS截止日期 + 年化折算

**核心原则**：回测全量截止日期 = 当前时间上个月底。样本外（OOS）= 倒退24个完整月。所有绩效指标必须按年化折算。

---

## 33.1 日期三要素

### 核心公式

```
full_end_date    = 当前时间上个月底
out_sample_start = full_end_date 倒退 24 个月 + 1 天（即 OOS = 24 个完整月）
in_sample_end    = out_sample_start - 1 天
```

### 当前值（2026-06-14）

```yaml
backtest:
  full_start_date: '2020-01-01'     # 固定
  full_end_date: '2026-05-31'       # 上个月底，每月更新
  in_sample_end_date: '2024-05-31'  # 自动计算：OOS起点-1天
  out_sample_start_date: '2024-06-01' # 自动计算：2026-05-31倒退24个月
```

### 定期更新

每月1日执行：

| 当前月 | full_end_date | OOS 区间 |
|--------|--------------|----------|
| 2026-06 | 2026-05-31 | 2024-06-01 ~ 2026-05-31 |
| 2026-07 | 2026-06-30 | 2024-07-01 ~ 2026-06-30 |
| 2026-08 | 2026-07-31 | 2024-08-01 ~ 2026-07-31 |

### 更新方式

```bash
# 每月初执行
./scripts/update_oos_dates.sh
```

或手动：

```yaml
backtest:
  full_end_date: '2026-05-31'       # ← 每月改这里
  in_sample_end_date: '2024-05-31'  # ← 倒退24个月 + 1天
  out_sample_start_date: '2024-06-01' # ← 倒退24个月
```

---

## 33.2 绩效指标年化折算

### 规则

所有回测报告中的收益类指标必须按**年化**展示，禁止裸展示合计收益。

| 指标 | 公式 | 备注 |
|------|------|------|
| **年化收益率** | `(1 + total_return)^(252 / n_bars) - 1` 或 `(1 + total_return)^(1 / n_years) - 1` | `n_years = (end_date - start_date).days / 365.25` |
| **夏普比** | PyBroker 已年化，直接使用 | 无需额外计算 |
| **卡玛比** | `年化收益率 / 最大回撤` | |
| **最大回撤** | 绝对值展示，不需要年化 | |

### 当前回测期

```
full:  2020-01-01 ~ 2026-05-31  →  6.41 年
in:   2020-01-01 ~ 2024-05-31  →  4.42 年
OOS:  2024-06-01 ~ 2026-05-31  →  2.00 年（24个月整）
```

### 代码实现

```python
from datetime import datetime

def _get_years(raw_config):
    bt = raw_config["backtest"]
    start = datetime.strptime(bt["full_start_date"], "%Y-%m-%d")
    end = datetime.strptime(bt["full_end_date"], "%Y-%m-%d")
    return (end - start).days / 365.25

def annualize(total_return_pct, years):
    return (1 + total_return_pct / 100) ** (1.0 / years) - 1
```

---

## 33.3 样本外验证标准

| 指标 | 标准 |
|------|------|
| OOS 夏普 | > 0（正收益） |
| OOS 回撤 | ≤ 样本内 1.5 倍 |
| OOS vs 样本内夏普衰减 | < 30% |

---

## 33.4 相关文件

| 文件 | 维护项 |
|------|--------|
| `config.yaml` | `full_end_date` / `in_sample_end_date` / `out_sample_start_date`，每月更新 |
| `.trae/rules/01-basics/33-oos-cutoff-annualized.md` | 本规则 |
| `.trae/rules/project_rules.md` | 规则33引用段 |
