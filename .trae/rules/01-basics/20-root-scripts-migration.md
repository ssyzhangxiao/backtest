# 规则20：根目录脚本迁移 — run_*.py 必须收敛到 Pipeline

**核心原则**：根目录脚本（`run_*.py`）只允许 3 个官方入口（`run_backtest.py` / `run_optimize.py` / `run_validate.py`），其他工作流必须通过 Pipeline 编排器调用 `runner/` 模块。

**生效日期**：2026-06-10
**适用范围**：所有根目录脚本的新增、修改、删除

---

## 背景

项目早期按"每个工作流一个根目录脚本"的模式运行，导致：

1. 根目录脚本膨胀到 6 个，违反规则 18（根目录脚本仅解析参数并调用 Pipeline）
2. 后 3 个自定义脚本内部直接调用 `PyBrokerBacktestRunner`、自实现 phase1/2/3 循环，违反规则 17（不重复造轮子）
3. 用户入口混乱：`python run_full_validation.py` vs `python run_validate.py --method monte_carlo` 行为差异不明显

**结论**：将自定义工作流统一收敛到 Pipeline 编排器，根目录脚本仅保留 3 个官方入口。

---

## 官方入口脚本（保留）

| 脚本 | 委托方法 | 用途 |
|------|----------|------|
| `run_backtest.py` | `pipe.run_backtest()` / `pipe.optimize()` / `pipe.validate()` / `pipe.report()` | 单实验 / 优化 / 验证 / 报告 |
| `run_optimize.py` | `pipe.optimize()` | 仅参数优化 |
| `run_validate.py` | `pipe.validate()` | 仅验证 |

> 这 3 个入口已记录于 README.md 和 docs/strategy_validation_plan.md，共 50+ 处引用，不可删除。

---

## 已删除脚本（迁移历史）

| 已删除 | 原行数 | 替换入口 | 迁移版本 |
|--------|--------|----------|----------|
| `run_full_experiments.py` | 78 | `pipe.run_experiments(experiments: List[str])` | a42e5fa (2026-06-10) |
| `run_full_validation.py` | 314 | `pipe.full_validation(in_sample_start, in_sample_end, oos_start, oos_end, ...)` | a42e5fa (2026-06-10) |
| `run_multi_oos.py` | 123 | `pipe.multi_oos(windows, strategies, best_params, ...)` | a42e5fa (2026-06-10) |

---

## 新增 Pipeline 方法（与新模块一一对应）

| Pipeline 方法 | 委托模块 | 输入 | 输出 |
|---------------|----------|------|------|
| `pipe.run_experiments(experiments)` | `runner/backtest/experiments/run_experiment` | 实验名列表 | 各实验 DataFrame 存于 `pipe._results[exp_name]` |
| `pipe.multi_oos(windows, ...)` | `runner/validation/multi_oos.py::run_multi_oos` | 窗口列表 + 策略列表 | 嵌套字典存于 `pipe._results["multi_oos"]` |
| `pipe.full_validation(...)` | `runner/validation/full_validation.py::run_full_validation` | 时间区间 + 策略列表 | 3 阶段汇总存于 `pipe._results["full_validation"]` |

---

## 新增模块（实现层）

| 模块 | 职责 | 关键导出 |
|------|------|----------|
| `runner/validation/multi_oos.py` | 多窗口 OOS 子策略回测 | `run_multi_oos()` / `DEFAULT_STRATEGIES` / `DEFAULT_WINDOWS` |
| `runner/validation/full_validation.py` | 3 阶段全量验证 | `run_full_validation()` / `DEFAULT_SYMBOLS_6` / `DEFAULT_STRATEGIES_5` |

**模块设计约束**：
- 单一职责：每个模块只做一件事
- 委托核心：核心逻辑委托 `core/engine/backtest_runner.py` / `core/validation/monte_carlo.py` 等公共系统，不重新实现
- 公共工具：使用 `runner/common/utils.py::safe_float` 等通用工具，不重复造轮子
- 文件行数：每个模块不超过 500 行（规则 7）

---

## 调用范式

### 范式 1：Pipeline 链式调用（推荐）

```python
from runner import Pipeline

# 批量实验（原 run_full_experiments.py）
pipe = Pipeline("config.yaml").load_data()
pipe.run_experiments(["e1", "e2", "e11", "e9"])

# 多窗口 OOS（原 run_multi_oos.py）
pipe = Pipeline("config.yaml").load_data()
pipe.optimize()  # 先调参（可选）
opt = pipe._results.get("optimization", {})
pipe.multi_oos(best_params=opt.get("best_params"))

# 全量 3 阶段验证（原 run_full_validation.py）
pipe = Pipeline("config.yaml").load_data()
pipe.full_validation(
    in_sample_start="2020-01-01",
    in_sample_end="2023-01-01",
    oos_start="2023-01-01",
    oos_end="2024-12-31",
)
```

### 范式 2：根目录官方入口（向后兼容）

```bash
# 单实验
python run_backtest.py --experiment e1

# 仅优化
python run_optimize.py --strategy trend

# 仅验证
python run_validate.py --method monte_carlo
```

### 范式 3：模块直接调用（高级用法）

```python
from runner.validation.multi_oos import run_multi_oos, DEFAULT_WINDOWS
from core.engine.pybroker_data_source import create_hybrid_data_source
from core.config import BacktestConfig

config = BacktestConfig.from_yaml("config.yaml")
ds = create_hybrid_data_source(config)
result = run_multi_oos(
    data_source=ds,
    config=config,
    output_dir=Path("output/multi_oos"),
    windows=DEFAULT_WINDOWS,  # 2022/2023/2024
)
```

---

## 具体规则

### 规则 20.1：禁止新增自定义根目录脚本

任何非官方入口的 `run_*.py` 都不允许新增。统一在 `runner/` 下新增模块 + 在 Pipeline 注册方法。

### 规则 20.2：禁止在根目录脚本自实现核心逻辑

若必须修改 3 个官方入口，主体逻辑（数据加载、回测执行、验证、报告）必须委托 `runner/` 模块或 `core/` 模块，不得在 `run_*.py` 内直接调用 `PyBrokerBacktestRunner.run()` 等底层 API。

### 规则 20.3：删除自定义工作流脚本须同时迁移到 Pipeline

删除根目录工作流脚本时，必须同步完成：
1. 提取其核心逻辑到 `runner/` 下对应模块（`backtest/` / `validation/` / `optimization/` / `report/`）
2. 在 `Pipeline` 类中新增对应方法
3. 更新 README.md 和 docs/ 中所有引用（如果存在）
4. 在本规则"已删除脚本"表格中记录迁移版本号

### 规则 20.4：Pipeline 方法命名规范

- 单一实验 → `pipe.run_backtest(name: str)`
- 批量实验 → `pipe.run_experiments(names: List[str])`
- 多窗口 OOS → `pipe.multi_oos(...)`
- 全量验证 → `pipe.full_validation(...)`
- 参数优化 → `pipe.optimize(...)`
- 验证方法 → `pipe.validate(method: str)`
- 报告生成 → `pipe.report(fmt: str)`

方法名使用动词或动名词短语，不使用缩写（除 `mc` / `oos` 等通用术语外）。

### 规则 20.5：模块导出与 hidden internal

- 每个新模块必须通过 `__all__` 显式导出公共接口
- 内部辅助函数使用下划线前缀（如 `_phase1_optimize` / `_phase2_ew_backtest`）
- Pipeline 内部入口方法（`run_*` / `multi_*` / `full_*`）必须为 `self` 返回类型（链式调用）

---

## 涉及代码

- `runner/pipeline.py`：`Pipeline` 类，新增 `run_experiments` / `multi_oos` / `full_validation` 方法
- `runner/validation/multi_oos.py`：新模块，多窗口 OOS 验证
- `runner/validation/full_validation.py`：新模块，3 阶段全量验证
- `runner/validation/__init__.py`：未注册 multi_oos / full_validation 到 `_VALIDATOR_MAP`（它们是独立流程，不是验证方法之一）
- 根目录 `run_full_experiments.py` / `run_full_validation.py` / `run_multi_oos.py`：已删除

---

## 维护检查清单

新增/删除/修改根目录脚本或 Pipeline 方法时，必须确认：

- [ ] 根目录仅有 `run_backtest.py` / `run_optimize.py` / `run_validate.py` 3 个
- [ ] 新方法/新模块已在本规则表格中登记
- [ ] `runner/` 下模块文件不超过 500 行
- [ ] `__all__` 显式导出公共接口
- [ ] README.md / docs/ 中引用保持同步
- [ ] git commit 信息标注 `chore(cleanup)` 或 `refactor(pipeline)`
