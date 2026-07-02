# 规则2：配置管理 — config.yaml 是单一数据源 + 分层配置 + OOS 截止与年化

**核心原则**：config.yaml 是一切配置的最终来源；按优先级 `defaults < YAML < env vars < runtime overrides` 分层叠加；OOS 截止日期固定为"上个月底 + 倒退 24 个月"；所有绩效指标必须年化。

**生效日期**：2026-06-11（吸收规则 23 + 规则 33）

---

## 2.1 yaml 与 BacktestConfig 同步

- 删除 config.yaml 中的废弃字段（fusion_mode、regime_filter_enabled、strategy_switching 等）
- BacktestConfig 字段命名与 config.yaml 保持一致
- 新增配置项先在 yaml 定义，再在 BacktestConfig 中映射
- 运行 `BacktestConfig.from_yaml()` 后做字段完整性校验

---

## 2.2 分层配置（吸收自规则 23）

**优先级（低 → 高）**：
```
1. dataclass 默认值     （代码内置）
2. YAML 文件            （config.yaml — 单一数据源）
3. 环境变量 QUANT_*     （部署/容器/CI 注入）
4. 运行时 overrides     （Pipeline / 脚本 / 测试）
```

**违反本规则的典型表现**：
- 在脚本顶部 `os.environ["X"] = ...` 然后再 `from core.config import ...` —— 散落难追踪
- 在多个 yaml 文件里维护同一字段（dev.yaml / prod.yaml）—— 重复且易错
- 测试用例里直接修改 yaml 文件 —— 污染用户配置

**正确做法**：一份 yaml + 一套环境变量约定 + 一份代码默认，三层自动合并。

### 规则 2.2.1：env 变量必须带 `QUANT_` 前缀

```bash
# ✅ 合规
export QUANT_BACKTEST__REBALANCE_FREQ=7
export QUANT_OUTPUT__OUTPUT_DIR=/tmp/run1

# ❌ 违规（无前缀会污染环境）
export BACKTEST_REBALANCE_FREQ=7
```

### 规则 2.2.2：env 变量名约定 `QUANT_<SECTION>__<FIELD>`

- `<SECTION>` 对应 yaml 顶层段名（小写）
- 双下划线 `__` 分隔段与字段
- `<FIELD>` 对应 yaml 字段名（小写）

| env 变量 | yaml 路径 |
|----------|----------|
| `QUANT_BACKTEST__REBALANCE_FREQ` | `backtest.rebalance_freq` |
| `QUANT_BACKTEST__STOP_LOSS_PCT` | `backtest.stop_loss_pct` |
| `QUANT_OUTPUT__OUTPUT_DIR` | `output.output_dir` |

### 规则 2.2.3：overrides dict 的 key 格式

- **顶层字段**（yaml 顶层键）：`"symbols"` / `"factor_weights"`
- **段路径**：`"backtest__rebalance_freq"`（推荐）
- **嵌套 dict**：`{"backtest": {"rebalance_freq": 7}}`（推荐用于覆盖多个字段）

**禁止**用 dataclass 字段名作为 override key：yaml 字段是 `rebalance_freq`，dataclass 字段是 `rebalance_days`，名字不一致时会静默失效。

### 规则 2.2.4：overrides 类型与 yaml 字段一致

- yaml 字段是 `int` → override 传 `int`（不要传 `"10"` 字符串）
- yaml 字段是 `bool` → override 传 `bool`
- env 变量全是字符串，由 `_coerce_env_value` 自动启发式转换
- runtime override 不做类型转换（保证精度），由调用方负责

### 规则 2.2.5：加载器接口必须保留向后兼容

`BacktestConfig.from_yaml(path)` 必须保持原有签名（不传 overrides 行为不变）：

```python
# 老代码：仍能跑
cfg = BacktestConfig.from_yaml("config.yaml")

# 新代码：可选 overrides
cfg = BacktestConfig.from_yaml("config.yaml", overrides={"backtest__rebalance_freq": 7})
```

**禁止**：把 `overrides` 改成必传位置参数，破坏调用方。

### 规则 2.2.6：环境变量只在加载时读取一次

`load_env_overrides()` 在 `from_yaml()` 内部调用一次，结果冻结到 BacktestConfig 实例。后续修改 `os.environ` 不影响已加载的实例（避免 race condition）。

### 规则 2.2.7：secret 不应进 yaml（用 `${VAR}` 占位符）

API key、密码、token 等敏感值：
- 在 config.yaml 中用 `${VAR_NAME}` 占位（如 `tqsdk_phone: ${TQSDK_PHONE}`）
- 实际值存 `.env` 文件（gitignored，不会泄露）
- 由 `yaml_utils.load_yaml()` 内 `load_dotenv()` + 正则替换自动展开
- **禁止**直接把明文密码写进 yaml

**历史教训**（2026-06-14）：原 `pybroker_data_source.py` 中手工读 config.yaml 拿到的 `${TQSDK_PHONE}` 是字面量，因为 YAML 本身不展开 `${VAR}`。必须在 `load_yaml()` 内先 `load_dotenv()`，再做正则替换。仅靠调用方侧 `load_dotenv()` 不可靠（模块导入顺序影响）。

---

## 2.3 OOS 截止日期（吸收自规则 33）

### 核心公式

```
full_end_date         = 当前时间上个月底
out_sample_start_date = full_end_date 倒退 24 个月 + 1 天
in_sample_end_date    = out_sample_start_date - 1 天
```

**OOS = 24 个完整月**（不可缩短、不可延长；缩短会导致样本不足，延长会引入未来数据风险）。

### 当前值（2026-06-18）

```yaml
backtest:
  full_start_date: '2020-01-01'          # 固定
  full_end_date: '2026-05-31'            # 上个月底
  in_sample_end_date: '2024-05-31'       # OOS 起点 - 1 天
  out_sample_start_date: '2024-06-01'    # 倒退 24 个月
```

### 定期更新

每月 1 日执行 `./scripts/update_oos_dates.sh`：

| 当前月 | full_end_date | OOS 区间 |
|--------|--------------|----------|
| 2026-06 | 2026-05-31 | 2024-06-01 ~ 2026-05-31 |
| 2026-07 | 2026-06-30 | 2024-07-01 ~ 2026-06-30 |
| 2026-08 | 2026-07-31 | 2024-08-01 ~ 2026-07-31 |

### OOS 验证标准

| 指标 | 标准 |
|------|------|
| OOS 夏普 | > 0（正收益） |
| OOS 回撤 | ≤ 样本内 1.5 倍 |
| OOS vs 样本内夏普衰减 | < 30% |

---

## 2.4 绩效指标年化折算（吸收自规则 33）

所有回测报告中的收益类指标**必须**按年化展示，禁止裸展示合计收益。

| 指标 | 公式 | 备注 |
|------|------|------|
| **年化收益率** | `(1 + total_return)^(1 / n_years) - 1` | `n_years = (end - start).days / 365.25` |
| **夏普比** | PyBroker 已年化，直接使用 | 无需额外计算 |
| **卡玛比** | `年化收益率 / 最大回撤` | |
| **最大回撤** | 绝对值展示，不需要年化 | |

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

## 涉及代码

| 文件 | 职责 |
|------|------|
| `core/config/layered_config.py` | `LayeredConfigLoader` / `load_env_overrides` / `merge_overrides` |
| `core/config/backtest_config.py` | `BacktestConfig.from_yaml(overrides=...)` 集成 |
| `core/config/__init__.py` | 公共 API 导出 |
| `config.yaml` | `full_end_date` / `in_sample_end_date` / `out_sample_start_date`，每月更新 |
| `scripts/update_oos_dates.sh` | 每月初自动更新 OOS 日期 |

---

## 维护检查清单

### 配置加载（2.2）
- [ ] `BacktestConfig.from_yaml()` 保留 `path` 必传 + `overrides` 可选
- [ ] 优先级顺序：defaults < yaml < env < runtime 保持不变
- [ ] 新增 env 变量前先在 `ENV_SECTION_ALIAS` 注册段名（若非标准）
- [ ] secret 不进 yaml（走 env / overrides）
- [ ] 测试覆盖：基础合并 / 嵌套合并 / 优先级 / 字段名兼容

### OOS 日期（2.3）
- [ ] `full_end_date` 每月 1 日已更新为"上个月底"
- [ ] `in_sample_end_date` / `out_sample_start_date` 与 `full_end_date` 同步计算
- [ ] OOS 区间 = 24 个完整月（不可缩短）
- [ ] OOS 验证结果符合标准（Sharpe>0 / MDD≤1.5x / 衰减<30%）

### 绩效展示（2.4）
- [ ] 所有收益类指标按年化展示
- [ ] 报告/日志中无裸合计收益
- [ ] 卡玛比计算使用年化收益

---

## 与其他规则的关系

| 关联规则 | 关系 |
|----------|------|
| 规则 17（不重复造轮子） | env/runtime override 必须走 `from_yaml(overrides=...)`，禁止重复实现 |
| 规则 18（Pipeline 编排器） | `pipe.with_config(overrides={...})` 实现本规则的运行时层 |
