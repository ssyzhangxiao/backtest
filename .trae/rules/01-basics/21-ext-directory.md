# 规则21：扩展目录 ext/ — 借鉴 QuantML-Qlib 的"按需加载 + 工厂注册"模式 + 目录迁移

**核心原则**：所有"可选/实验/第三方依赖"扩展能力集中在 `core/ext/`，通过 `extras_require` 按需安装，通过工厂注册表动态加载，不污染核心架构；建立新目录后必须归类整理旧代码，禁止"新建一份 + 旧文件保留"的伪迁移。

**生效日期**：2026-06-11
**借鉴来源**：QuantML 微信文章 + `qlib.ext` 重构（详见 `.trae/knowledges/20260611_001_knowledge_quantml-qlib-ext-borrow.md`）
**吸收规则**：22（目录迁移，2026-06-18）

---

## 背景

当前项目存在 3 个核心痛点：
1. **依赖膨胀**：`requirements.txt` 一锅端，安装即全量
2. **扩展散落**：`cross_spread.py` 等新功能散落在 `core/factors/alpha_futures/`，未来 GP/LLM/AlphaGPT 因子挖掘会进一步污染核心
3. **数据源硬编码**：`create_hybrid_data_source()` 内置 TqSdk/CSV 两种，新增 AKShare/RQData 需改核心

QuantML-Qlib 通过 `qlib.ext` + `extras_require` + 工厂模式解决。本项目采纳其思想但不全面照搬——只做 4 项高/中 ROI 改造。

---

## 21.1 扩展目录结构

`core/ext/` 必须在如下结构内（**新增模块必须登记到对应子目录**）：

```
core/ext/
├── adapters/                  # 数据源适配器（数据层扩展）
│   ├── base.py                # DataSourceAdapter 抽象基类
│   ├── tqsdk_adapter.py       # 从 core/data_loader 抽出（可选）
│   ├── csv_adapter.py         # 从 core/data_loader 抽出（可选）
│   └── factory.py             # create_data_source(name, **kwargs) + @register_adapter
├── factors/                   # 因子挖掘扩展（因子层扩展）
│   ├── generation/            # 因子生成（GP/LLM/AlphaGPT）
│   │   ├── gplearn.py
│   │   ├── llm_generator.py
│   │   └── alphagpt.py
│   ├── pool/                  # 因子池（互斥 IC + 权重 + 衰减）
│   │   ├── manager.py
│   │   └── decay.py           # 复用 core/engine/factor_decay.py
│   └── operators/             # 算子扩展（TA-Lib 等）
│       └── talib_ops.py
├── models/                    # 预测模型扩展（评估层扩展）
│   ├── base.py                # BasePredictor 抽象
│   ├── lgbm.py
│   ├── mlp.py
│   └── configs/               # 模型 YAML
├── handlers/                  # 多频/高频数据处理器
└── utils/                     # 工具函数扩展
```

**建立新目录后必须遵守 21.8 目录迁移流程**：识别旧代码候选、选 A（物理迁移）或 B（委托弃用）、删除旧位置或加 `@deprecated`、调用方重写、等价性测试。

---

## 21.2 按需安装（extras_require）

`pyproject.toml` 必须定义 4 个 extras（**新增 extras 需更新此处 + README**）：

| extras | 依赖 | 用途 |
|--------|------|------|
| `core` | pybroker, tqsdk | 核心回测（默认） |
| `data-sources` | tqsdk, akshare, tushare | 数据源适配器 |
| `factors` | gplearn, deap | GP 因子挖掘 |
| `llm` | openai>=1.0, anthropic | LLM 因子生成 |
| `models` | lightgbm, xgboost, torch | 预测模型 |
| `dashboard` | streamlit, plotly | UI |
| `all` | 以上全部 | 全量（不推荐） |

**安装命令**：
```bash
pip install -e .[core]              # 最小安装（开发必备）
pip install -e .[factors]           # 加上遗传规划
pip install -e .[llm,models]        # 加上 AI 模型
pip install -e .[all]               # 全量
```

**禁止**：在 `dependencies = []` 中放入可选依赖（如 torch、gplearn），必须放 `[project.optional-dependencies]`。

### 21.2.1 根目录组织原则

`requirements-{name}.txt` 文件**保持在项目根目录**（Python 生态惯例 + QuantML 借鉴）：

- `requirements.txt` 核心依赖
- `requirements-data-sources.txt` / `requirements-factors.txt` / `requirements-llm.txt` / `requirements-models.txt` / `requirements-all.txt`
- `REQUIREMENTS.md` 索引文件（描述各文件用途 + 同步要求）

**禁止**：建 `requirements/` 子目录。pip 命令会变长（`pip install -r requirements/xxx.txt`），无任何收益。

**同步约束**（规则 21.5）：两份清单必须保持一致——`pyproject.toml::[project.optional-dependencies]` 与 `requirements-{name}.txt` 互相同步。

---

## 21.3 工厂注册模式（数据源/算子/模型）

`core/ext/` 下的所有可扩展对象（数据源、算子、模型）必须通过工厂注册表暴露：

```python
# core/ext/adapters/factory.py
_DATA_SOURCE_REGISTRY: Dict[str, Type["DataSourceAdapter"]] = {}

def register_adapter(name: str):
    """装饰器：注册数据源到工厂。"""
    def deco(cls):
        _DATA_SOURCE_REGISTRY[name] = cls
        return cls
    return deco

def create_data_source(name: str, **kwargs) -> "DataSourceAdapter":
    if name not in _DATA_SOURCE_REGISTRY:
        raise KeyError(f"未知数据源 {name}，已注册: {list(_DATA_SOURCE_REGISTRY)}")
    return _DATA_SOURCE_REGISTRY[name](**kwargs)
```

**禁止**：
- 在 `create_*` 函数中硬编码 `if/elif name == "x"` 链式判断
- 直接 `import` 具体适配器（必须通过 `create_data_source(name)` 工厂调用）

**适配器实现示例**：
```python
# core/ext/adapters/akshare_adapter.py
from core.ext.adapters.base import DataSourceAdapter
from core.ext.adapters.factory import register_adapter

@register_adapter("akshare")
class AKShareAdapter(DataSourceAdapter):
    def load(self, symbols, start, end): ...
```

---

## 21.4 复用约束（不重复造轮子）

`core/ext/` 子模块必须**复用**核心系统：

| ext 子模块 | 复用的核心模块 |
|-----------|---------------|
| `adapters/` | `core/engine/pybroker_data_source.py`（PyBrokerDataSource 基类） |
| `factors/generation/` | `core/factors/alpha_futures/factor_engine.py`（BaseFactor + register_factor） |
| `factors/pool/` | `core/engine/factor_decay.py`（衰减监控） + `core/factors/factor_evaluator.py`（IC 评估） |
| `factors/operators/` | `core/factors/operators.py`（基础算子） |
| `models/` | `core/factors/factor_pipeline.py`（pipeline 编排） |
| `handlers/` | `core/data_loader.py`（数据加载） |

**禁止**：
- 在 `core/ext/factors/generation/` 中重写 `BaseFactor` 基类
- 在 `core/ext/adapters/` 中绕过 `PyBrokerDataSource` 直接读 CSV（必须走适配器接口）

---

## 21.5 依赖方向

```
runner/ → core/ → core/ext/  (允许)
core/ext/ → core/  (允许，复用核心)
core/ext/ → runner/  (禁止)
core/ext/ 内部 cross-deps  (允许，但需通过 __init__.py 显式导出)
```

---

## 21.6 文件行数与导出

- 单文件 ≤ 500 行（规则 7）
- `core/ext/__init__.py` 必须显式 `__all__` 导出公共接口
- 子目录的 `__init__.py` 同样要求

---

## 21.7 阶段路线（按 ROI 排序）

**第一阶段（高 ROI，3.5h）**：
1. `core/ext/` 目录骨架 + `__init__.py` 公共导出
2. `core/ext/adapters/base.py + factory.py + 迁移 TqSdk/CSV`
3. `pyproject.toml` 的 `extras_require`（4 个 extras）

**第二阶段（中 ROI，3h）**：
4. `core/ext/factors/operators/talib_ops.py`
5. `cli.py` 统一入口（保留 3 个 run_*.py）

**第三阶段（探索，长期）**：
6. `core/ext/factors/generation/gplearn.py`
7. `core/ext/factors/generation/llm_generator.py`
8. `core/ext/models/lgbm.py + mlp.py`
9. 分层配置（YAML → 环境变量 → 运行时）

---

## 21.8 目录迁移流程（吸收自规则 22）

建立新的文件目录/模块结构后，**必须**识别并迁移属于该目录的旧代码，禁止"新建一份 + 旧文件保留"的伪迁移。

### 反例代价（`core/ext/` 第一阶段教训）

| 做法 | 是否符合本规则 |
|------|---------------|
| 在 `core/ext/adapters/` 下新建 `csv_adapter.py` | ✅ |
| `csv_adapter.py` 内部直接 `from core.data_loader import DataLoader` 复用 | ❌ **伪迁移**（应为移动而非委托） |
| `core/data_loader.py` 内硬编码的 TqSdk/CSV 分支未删除 | ❌ **违反规则 3 废弃清理** |
| `cli.py` 统一入口，**保留** 3 个 `run_*.py` | ✅（这是规则 17.2 显式允许的） |

### 规则 21.8.1：建立新目录前先做"旧代码审计"

新建 `core/ext/{adapters,factors,models,...}/` 前，必须先回答：

1. **哪些旧模块/函数属于这个新目录？**（在 `git grep` 中搜索关键词）
2. **旧代码的调用方清单**（被谁 import / 被谁调用）
3. **迁移策略**：物理迁移 vs 委托弃用（21.8.2）
4. **新目录的 `__init__.py` 是否需要做"重定向导出"以保持向后兼容？**

```bash
# 1. 找出所有候选旧代码
git grep -l "TqSdk\|tqsdk" -- "core/" "utils/" "runner/"

# 2. 找出调用方
git grep -l "from core.data_loader import DataLoader" -- "*.py"

# 3. 写审计报告 .trae/notes/migration-audit-{new_dir}.md
```

### 规则 21.8.2：迁移策略二选一（禁止同时存在）

| 策略 | 适用场景 | 操作 |
|------|---------|------|
| **A. 物理迁移** | 旧代码与新目录职责完全一致 | 移动文件 + 删除旧位置 + 改 import |
| **B. 委托 + 弃用** | 旧代码有大量调用方，一时迁移成本高 | 新文件委托旧文件 + 旧文件加 `@deprecated` + 计划废弃日期 |

**禁止**：A 和 B 同时存在——选一种就贯彻到底。**推荐**：A 物理迁移（符合规则 3 废弃清理原则）。

### 规则 21.8.3：迁移后必须做"调用方重写"

旧位置删除后，所有调用方必须同步更新：

```python
# 旧（迁移前）
from core.data_loader import DataLoader
loader = DataLoader(data_source="tqsdk", ...)

# 新（迁移后）
from core.ext.adapters import create_data_source
ds = create_data_source("tqsdk", ...)
```

**禁止**：在调用方 `try: from core.ext import ...; except: from core.data_loader import ...` 做双轨兼容。

### 规则 21.8.4：回归测试必须覆盖迁移前后等价性

```python
def test_migration_equivalence():
    old_result = DataLoader(data_source="csv", data_dir=...).get_bars(...)
    new_result = create_data_source("csv", data_dir=...).get_bars(...)
    pd.testing.assert_frame_equal(old_result, new_result)
```

### 规则 21.8.5：迁移完成必须更新规则 16 目录结构表

`01-basics/16-directory.md` 中的目录结构图必须**反映迁移后的真实状态**。

### 规则 21.8.6：委托弃用必须经二次审计

策略 B 落地后，**至少经历一个 release 周期 + 一次二次审计**，才可进入物理移除阶段。二次审计清单：

```bash
# 1. 旧 API 引用清零（除迁移目标文件和迁移工具外）
git grep "DataLoader(data_source" -- "*.py" | grep -v "core/data_loader.py" | grep -v "core/ext/adapters/"
# 期望：0 行

# 2. @deprecated 警告无新增调用（CI 跑测试，统计 DeprecationWarning 触发次数变化）
# 期望：触发次数随 release 递减

# 3. 文档已更新
grep -r "DataLoader(data_source" docs/ README.md REQUIREMENTS.md
# 期望：0 行

# 4. 等价性测试覆盖
pytest tests/test_migration_equivalence.py -v
# 期望：全部通过
```

**禁止**：跳过二次审计直接删除旧代码。**二次审计未通过 → 延期删除 → 重新发 v0.2.x 计划**。

### 规则 21.8.7：空文件/空目录必须物理移除

整个文件 + 所在目录一并物理删除（包括 `__init__.py` 空壳）。判断标准（满足任一即物理删除）：

| 条件 | 检查命令 |
|------|---------|
| 旧文件无任何 `def` / `class` | `grep -c "def \|class " old.py` = 0 |
| 旧文件无 import | `wc -l old.py` < 5 |
| 旧目录只剩 `__init__.py` 空壳 | `ls dir/` 仅 `__init__.py`，且文件 < 5 行 |
| 旧目录无任何 `.py` 文件 | `find dir/ -name "*.py"` 为空 |

**禁止**：
- 保留空 `__init__.py` 做"占位"
- 保留空文件做"兼容垫片"
- 保留空目录做"未来扩展点"（违反规则 21.1）

---

## 具体规则

### 规则 21.1：禁止在 core/ 根目录新建"扩展"模块
所有"可选/实验/第三方依赖"扩展能力必须放 `core/ext/`，不得在 `core/` 根目录或 `core/factors/alpha_futures/` 新建 GP/LLM/AlphaGPT 因子类。

### 规则 21.2：禁止硬编码依赖
任何 `import torch` / `import gplearn` / `import openai` 必须放在 `core/ext/` 内，且必须在 `pyproject.toml` 的对应 extras 中声明。`try: import` 的兜底不允许。

### 规则 21.3：工厂注册必须使用装饰器
所有"按名字创建"的工厂必须提供 `@register_xxx(name)` 装饰器，禁止 if/elif 链。

### 规则 21.4：扩展模块必须先复用地基
GP/LLM 因子挖掘必须继承 `BaseFactor`、使用 `register_factor` 注册，禁止另起一套因子体系。

### 规则 21.5：新增 extras 必须更新规则 21.2
每新增 1 个 extras（如 `quant` / `viz`），必须：
1. 在本规则"按需安装"表格中登记
2. 在 `pyproject.toml` 的 `[project.optional-dependencies]` 中定义
3. 在 README.md 安装说明中追加命令

---

## 涉及代码

- `core/ext/__init__.py`：公共接口导出
- `core/ext/adapters/factory.py`：数据源工厂
- `pyproject.toml` 或 `setup.py`：extras_require
- `.trae/knowledges/20260611_001_knowledge_quantml-qlib-ext-borrow.md`：借鉴评估
- `core/data_loader.py`：候选迁移源（TqSdk/CSV 分支）
- `core/ext/adapters/*.py`：迁移目标位置
- `runner/data/*.py`、`core/engine/*.py`：调用方需同步
- `.trae/notes/migration-audit-*.md`：迁移审计报告

---

## 维护检查清单

### 新增 `core/ext/` 子模块

- [ ] 模块位于 `core/ext/{adapters,factors,models,handlers,utils}/` 之一
- [ ] 文件 ≤ 500 行（规则 7）
- [ ] `__init__.py` 显式 `__all__` 导出
- [ ] 第三方依赖已加入对应 extras（规则 21.2）
- [ ] 复用核心系统（规则 21.4）
- [ ] 工厂类使用装饰器注册（规则 21.3）
- [ ] git commit 信息标注 `feat(ext)` 或 `refactor(ext)`

### 新建 `core/ext/{xxx}/` 目录（21.8）

- [ ] 已写 `.trae/notes/migration-audit-{xxx}.md` 审计报告
- [ ] 已选 A（物理迁移）或 B（委托弃用）并贯彻
- [ ] 旧位置已删除（策略 A）或加 `@deprecated`（策略 B）
- [ ] 所有调用方 import 已更新
- [ ] 等价性回归测试通过
- [ ] 规则 16 目录结构图已更新
- [ ] git commit 信息标注 `refactor(migration)`

### 策略 B 拆除时（21.8.6 + 21.8.7）

- [ ] 二次审计清单 4 项全过（git grep / warning 计数 / 文档 / 等价性测试）
- [ ] `git grep "旧 API 签名"` 在生产代码（非测试）= 0
- [ ] 旧文件已无 `def` / `class`（或整个文件删除）
- [ ] 旧目录无 `.py` 文件（已物理删除）
- [ ] 旧目录无空 `__init__.py` 占位
- [ ] git commit 信息标注 `refactor(remove-deprecated)`

---

## 与其他规则的关系

| 关联规则 | 关系 |
|----------|------|
| 规则 3（废弃代码清理） | 21.8 是 3 在"目录建立"场景下的具体化 |
| 规则 16（目录结构） | 21.8.5 推动 16 目录结构图与代码同步 |
| 规则 17（不重复造轮子） | 21.8 避免"新文件 + 旧文件并存"的双套维护 |
