# 规则23：分层配置 — YAML → 环境变量 → 运行时覆盖

**核心原则**：配置按优先级分层叠加，避免"散落在 N 个文件 / N 个环境变量 / N 个 shell 参数里"的混乱。

**优先级（低 → 高）**：
```
1. dataclass 默认值     （代码内置）
2. YAML 文件            （config.yaml — 单一数据源，规则 2）
3. 环境变量 QUANT_*     （部署/容器/CI 注入）
4. 运行时 overrides     （Pipeline / 脚本 / 测试）
```

**违反本规则的典型表现**：
- 在脚本顶部 `os.environ["X"] = ...` 然后再 `from core.config import ...` —— 散落难追踪
- 在多个 yaml 文件里维护同一字段（dev.yaml / prod.yaml）—— 重复且易错
- 测试用例里直接修改 yaml 文件 —— 污染用户配置

**正确做法**：
- 一份 yaml + 一套环境变量约定 + 一份代码默认
- 三层自动合并，无需手动 if/else

---

## 具体规则

### 规则 23.1：env 变量必须带 `QUANT_` 前缀

```bash
# ✅ 合规
export QUANT_BACKTEST__REBALANCE_FREQ=7
export QUANT_OUTPUT__OUTPUT_DIR=/tmp/run1

# ❌ 违规（无前缀，会污染环境）
export BACKTEST_REBALANCE_FREQ=7
export REBALANCE_FREQ=7
```

**为什么**：避免与系统其他变量 / 用户终端变量冲突，namespace 隔离。

### 规则 23.2：env 变量名约定 `QUANT_<SECTION>__<FIELD>`

- `<SECTION>` 对应 yaml 顶层段名（小写）
- 双下划线 `__` 分隔段与字段（不与字段内的下划线冲突）
- `<FIELD>` 对应 yaml 字段名（小写）

**示例**：
| env 变量 | yaml 路径 |
|----------|----------|
| `QUANT_BACKTEST__REBALANCE_FREQ` | `backtest.rebalance_freq` |
| `QUANT_BACKTEST__STOP_LOSS_PCT` | `backtest.stop_loss_pct` |
| `QUANT_OUTPUT__OUTPUT_DIR` | `output.output_dir` |
| `QUANT_FACTOR_WEIGHTS__TREND` | `factor_weights.trend` |

### 规则 23.3：overrides dict 的 key 格式

- **顶层字段**（yaml 顶层键）：`"symbols"` / `"factor_weights"`
- **段路径**：`"backtest__rebalance_freq"`（推荐）
- **嵌套 dict**：`{"backtest": {"rebalance_freq": 7}}`（推荐用于覆盖多个字段）

**禁止**用 dataclass 字段名作为 override key（如 `rebalance_days`）：
- yaml 字段是 `rebalance_freq`，dataclass 字段是 `rebalance_days`，名字不一致
- 改 dataclass 字段名 → override 失效（静默）
- 统一用 yaml 字段名（`backtest__rebalance_freq`）最稳

### 规则 23.4：overrides 类型与 yaml 字段一致

- yaml 字段是 `int` → override 传 `int`（不要传 `"10"` 字符串）
- yaml 字段是 `bool` → override 传 `bool`（不要传 `"true"` 字符串）
- env 变量全是字符串，由 `_coerce_env_value` 自动启发式转换
- runtime override 不做类型转换（保证精度），由调用方负责

### 规则 23.5：加载器接口必须保留向后兼容

`BacktestConfig.from_yaml(path)` 必须保持原有签名（不传 overrides 行为不变）：

```python
# 老代码：仍能跑
cfg = BacktestConfig.from_yaml("config.yaml")

# 新代码：可选 overrides
cfg = BacktestConfig.from_yaml("config.yaml", overrides={"backtest__rebalance_freq": 7})
```

**禁止**：把 `overrides` 改成必传位置参数，破坏调用方。

### 规则 23.6：环境变量只在加载时读取一次

`load_env_overrides()` 在 `from_yaml()` 内部调用一次，结果冻结到 BacktestConfig 实例。
后续修改 `os.environ` 不影响已加载的实例（避免 race condition）。

### 规则 23.7：secret 不应进 yaml（用 `${VAR}` 占位符）

API key、密码、token 等敏感值：
- ✅ 在 config.yaml 中用 `${VAR_NAME}` 占位（如 `tqsdk_phone: ${TQSDK_PHONE}`）
- ✅ 实际值存 `.env` 文件（gitignored，不会泄露）
- ✅ 由 `yaml_utils.load_yaml()` 内 `load_dotenv()` + 正则替换自动展开
- ❌ 直接把明文密码写进 yaml

**历史教训**（2026-06-14）：
原 `pybroker_data_source.py` 中手工读 config.yaml 拿到的 `${TQSDK_PHONE}` 是字面量，
因为 YAML 本身不展开 `${VAR}`。必须在 `load_yaml()` 内先 `load_dotenv()`，再做正则
`re.sub(r"\$\{(\w+)\}", ...)` 替换。仅靠调用方侧 `load_dotenv()` 不可靠（模块导入
顺序影响）。

---

## 涉及代码

| 文件 | 职责 |
|------|------|
| `core/config/layered_config.py` | `LayeredConfigLoader` / `load_env_overrides` / `merge_overrides` |
| `core/config/backtest_config.py` | `BacktestConfig.from_yaml(overrides=...)` 集成 |
| `core/config/__init__.py` | 公共 API 导出 |

---

## 维护检查清单

新增 / 修改配置加载时，确认：

- [ ] `BacktestConfig.from_yaml()` 保留 `path` 必传 + `overrides` 可选
- [ ] 优先级顺序：defaults < yaml < env < runtime 保持不变
- [ ] 新增 env 变量前先在 `ENV_SECTION_ALIAS` 注册段名（若非标准）
- [ ] secret 不进 yaml（走 env / overrides）
- [ ] 测试覆盖：基础合并 / 嵌套合并 / 优先级 / 字段名兼容

---

## 与其他规则的关系

| 关联规则 | 关系 |
|----------|------|
| 规则 2（config.yaml 单一数据源） | 兼容：yaml 仍是默认，env/runtime 是覆盖 |
| 规则 18（Pipeline 编排器） | `pipe.with_config(overrides={...})` 实现本规则的运行时层 |
| 规则 17（不重复造轮子） | 已有 env 处理分散在 5+ 处，本规则统一收口 |
