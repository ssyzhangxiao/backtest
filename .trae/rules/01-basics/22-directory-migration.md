# 规则22：目录迁移 — 建立新目录后必须归类整理旧代码

**核心原则**：建立新的文件目录/模块结构后，**必须**识别并迁移属于该目录的旧代码，禁止"新建一份 + 旧文件保留"的伪迁移。

**生效日期**：2026-06-11
**适用范围**：所有新增 `core/`、`core/ext/`、`runner/` 子目录或大粒度重构

---

## 背景

回顾 `core/ext/` 第一阶段落地：

| 做法 | 是否符合本规则 |
|------|---------------|
| 在 `core/ext/adapters/` 下新建 `csv_adapter.py` | ✅ |
| `csv_adapter.py` 内部直接 `from core.data_loader import DataLoader` 复用 | ❌ **伪迁移**（应为移动而非委托） |
| `core/data_loader.py` 内硬编码的 TqSdk/CSV 分支未删除 | ❌ **违反规则 3 废弃清理** |
| `cli.py` 统一入口，**保留** 3 个 `run_*.py` | ✅（这是规则 20 显式允许的） |

**反例代价**：
1. 维护两套：旧 `DataLoader` 逻辑 和 新 `CsvAdapter` 适配器
2. 用户困惑：`create_data_source("csv")` 还是 `DataLoader(data_source="csv")`？
3. 规则 21.4（复用约束）沦为文档摆设

---

## 具体规则

### 规则 22.1：建立新目录前先做"旧代码审计"

新建 `core/ext/{adapters,factors,models,...}/` 前，必须先回答：

1. **哪些旧模块/函数属于这个新目录？**（在 `git grep` 中搜索关键词）
2. **旧代码的调用方清单**（被谁 import / 被谁调用）
3. **迁移策略**：移动 vs 委托（规则 22.2）
4. **新目录的 `__init__.py` 是否需要做"重定向导出"以保持向后兼容？**

**审计模板**：

```bash
# 1. 找出所有候选旧代码
git grep -l "TqSdk\|tqsdk" -- "core/" "utils/" "runner/"

# 2. 找出调用方
git grep -l "from core.data_loader import DataLoader" -- "*.py"

# 3. 写审计报告
cat > .trae/notes/migration-audit-{new_dir}.md <<EOF
## 旧代码候选
- core/data_loader.py (TqSdk/CSV 分支)
- utils/legacy_source.py (硬编码 AKShare)

## 调用方
- runner/data/loader.py
- core/engine/backtest_runner.py

## 迁移策略
- 移动：TqSdk → core/ext/adapters/tqsdk_adapter.py（物理搬迁）
- 委托：CSV 暂保留 DataLoader，标记 @deprecated
- 删除：utils/legacy_source.py (无调用方)
EOF
```

### 规则 22.2：迁移策略二选一（禁止同时存在）

| 策略 | 适用场景 | 操作 |
|------|---------|------|
| **A. 物理迁移** | 旧代码与新目录职责完全一致 | 移动文件 + 删除旧位置 + 改 import |
| **B. 委托 + 弃用** | 旧代码有大量调用方，一时迁移成本高 | 新文件委托旧文件 + 旧文件加 `@deprecated` + 计划废弃日期 |

**禁止**：A 和 B 同时存在——选一种就贯彻到底。

**推荐**：A 物理迁移（符合规则 3 废弃清理原则）。

### 规则 22.3：迁移后必须做"调用方重写"

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

### 规则 22.4：回归测试必须覆盖迁移前后等价性

迁移后必须做对比测试：

```python
def test_migration_equivalence():
    """迁移前后的行为必须等价。"""
    # 旧接口
    old_result = DataLoader(data_source="csv", data_dir=...).get_bars(...)
    # 新接口
    new_result = create_data_source("csv", data_dir=...).get_bars(...)
    # 必须等价
    pd.testing.assert_frame_equal(old_result, new_result)
```

### 规则 22.5：迁移完成必须更新规则 16 目录结构表

`01-basics/16-directory.md` 中的目录结构图必须**反映迁移后的真实状态**。

### 规则 22.6：委托弃用必须经二次审计

策略 B（委托 + 弃用）落地后，**至少经历一个 release 周期 + 一次二次审计**，才可进入物理移除阶段。

**二次审计清单**：

```bash
# 1. 旧 API 引用清零
git grep "DataLoader(data_source" -- "*.py" | grep -v "core/data_loader.py" | grep -v "core/ext/adapters/"
# 期望：0 行（除迁移目标文件和迁移工具外）

# 2. @deprecated 警告无新增调用
# 在 CI 中跑测试，统计 DeprecationWarning 触发次数变化
# 期望：触发次数随 release 递减

# 3. 文档已更新
grep -r "DataLoader(data_source" docs/ README.md REQUIREMENTS.md
# 期望：0 行

# 4. 等价性测试覆盖
pytest tests/test_migration_equivalence.py -v
# 期望：全部通过
```

**禁止**：跳过二次审计直接删除旧代码（包括 v0.2.0 release 计划删除的代码）。**二次审计未通过 → 延期删除 → 重新发 v0.2.x 计划**。

### 规则 22.7：空文件/空目录必须物理移除

当一个文件全部内容迁移完成、剩余 @deprecated 也删除后，**整个文件 + 所在目录一并物理删除**（包括 `__init__.py` 空壳）。

**反例**：
```
core/
├── data_loader.py            # 1071 行已全部分散到 core/ext/adapters/
│   └── (空)                  # 留个空文件做占位
└── data_loader/
    └── __init__.py           # 留个空 __init__.py 做占位
```

**正例**：
```
core/
└── (data_loader.py 已删除)
```

**判断标准**（满足任一即物理删除）：

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

**与规则 22.6 的关系**：
```
22.6 二次审计通过
    ↓
22.7 物理移除（一次性完成，不留尾巴）
```


---

## 涉及代码

- `core/data_loader.py`：候选迁移源（TqSdk/CSV 分支）
- `core/ext/adapters/*.py`：目标位置
- `runner/data/*.py`、`core/engine/*.py`：调用方需同步
- `.trae/notes/migration-audit-*.md`：审计报告

---

## 维护检查清单

新建 `core/ext/{xxx}/` 目录时，确认：

- [ ] 已写 `.trae/notes/migration-audit-{xxx}.md` 审计报告
- [ ] 已选 A（物理迁移）或 B（委托弃用）并贯彻
- [ ] 旧位置已删除（策略 A）或加 `@deprecated`（策略 B）
- [ ] 所有调用方 import 已更新
- [ ] 等价性回归测试通过
- [ ] 规则 16 目录结构图已更新
- [ ] git commit 信息标注 `refactor(migration)`

**策略 B 拆除时（22.6 + 22.7）**：

- [ ] 二次审计清单 4 项全过（git grep / warning 计数 / 文档 / 等价性测试）
- [ ] `git grep "旧 API 签名"` 在生产代码（非测试）= 0
- [ ] 旧文件已无 `def` / `class`（或整个文件删除）
- [ ] 旧目录无 `.py` 文件（已物理删除）
- [ ] 旧目录无空 `__init__.py` 占位
- [ ] git commit 信息标注 `refactor(remove-deprecated)`

---

## 与其他规则的关系

| 规则 | 关系 |
|------|------|
| 规则 3（废弃代码清理） | 22 是 3 在"目录建立"场景下的具体化 |
| 规则 16（目录结构） | 22 推动 16 目录结构图与代码同步 |
| 规则 17（不重复造轮子） | 22 避免"新文件 + 旧文件并存"的双套维护 |
| 规则 21（ext 目录） | 22 是 21 落地流程的强制补充 |
