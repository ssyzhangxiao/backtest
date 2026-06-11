# 依赖文件索引（规则21.2）

> **本目录决策**：所有 requirements 文件**保持在项目根目录**（Python 生态惯例 + QuantML 借鉴）。
> 本 README 是根目录的索引文件，方便用户查找各文件用途。

---

## 文件清单（全部在根目录）

| 文件 | 用途 | 等价的 pyproject.toml extras |
|------|------|------------------------------|
| [requirements.txt](./requirements.txt) | 核心依赖（13 个包） | `dependencies = []` |
| [requirements-data-sources.txt](./requirements-data-sources.txt) | 数据源适配器 | `[data-sources]` |
| [requirements-factors.txt](./requirements-factors.txt) | 因子挖掘（GP/DEAP） | `[factors]` |
| [requirements-llm.txt](./requirements-llm.txt) | LLM 因子生成 | `[llm]` |
| [requirements-models.txt](./requirements-models.txt) | 预测模型 | `[models]` |
| [requirements-all.txt](./requirements-all.txt) | 全量（开发自测） | `[all]` |

---

## 为什么根目录？

1. **pip 命令简洁**：`pip install -r requirements-factors.txt` vs `pip install -r requirements/requirements-factors.txt`
2. **生态惯例**：绝大多数 Python 项目把 requirements 放根目录
3. **QuantML 借鉴**：`qlib.ext` 的可选依赖文件也在仓库根目录
4. **降低摩擦**：CI / Docker / 教程无需修改路径

---

## 安装命令对照

```bash
# pip install -r 方式（项目当前主流）
pip install -r requirements.txt                    # 核心
pip install -r requirements-data-sources.txt       # 数据源
pip install -r requirements-factors.txt            # 因子挖掘
pip install -r requirements-llm.txt                # LLM
pip install -r requirements-models.txt             # 预测模型
pip install -r requirements-all.txt                # 全量

# pip install -e . 方式（pyproject.toml extras）
pip install -e .[core]                             # 等价 requirements.txt
pip install -e .[data-sources]                     # 等价 requirements-data-sources.txt
pip install -e .[factors]                          # 等价 requirements-factors.txt
pip install -e .[llm,models]                       # 组合
pip install -e .[all]                              # 等价 requirements-all.txt
```

---

## 同步要求（规则 21.5）

**两份清单必须保持同步**：
- 修改 `pyproject.toml::[project.optional-dependencies]` 时必须同步对应 `requirements-{name}.txt`
- 修改 `requirements-{name}.txt` 时必须同步 `pyproject.toml::[project.optional-dependencies]`
- 核心 `requirements.txt` 必须与 `pyproject.toml::dependencies` 一致

---

## 第三方依赖来源声明（规则 21.4）

| 第三方库 | 使用模块 | extras |
|---------|---------|--------|
| gplearn | `core/ext/factors/generation/gplearn.py`（规划中） | `[factors]` |
| openai  | `core/ext/factors/generation/llm_generator.py`（规划中） | `[llm]` |
| lightgbm | `core/ext/models/lgbm.py`（规划中） | `[models]` |
| torch   | `core/ext/models/mlp.py`（规划中） | `[models]` |
| TA-Lib  | `core/ext/factors/operators/talib_ops.py` | `[factors]` |
| tqsdk   | `core/ext/adapters/tqsdk_adapter.py` | `[data-sources]` |
| akshare / tushare / rqdatac | 各 xxx_adapter.py（规划中） | `[data-sources]` |

---

*最后更新：2026-06-11（规则21 阶段 2 落地）*
