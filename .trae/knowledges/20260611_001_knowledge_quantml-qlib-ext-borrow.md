# QuantML-Qlib ext 重构借鉴评估

> **资料来源**：
> - 本地文档：`/Users/luojiutian/Downloads/QuantML系统架构借鉴.docx`（2026-06-11）
> - 微信文章：`https://mp.weixin.qq.com/s/Ewq8GVATCVLFA29xI-P21w`（docx 已包含全文）
>
> **核心内容**：QuantML 团队对 Qlib 三年积累后的系统性重构——把散落的扩展功能集中到 `qlib.ext/`，实现"按需安装 + 工厂模式 + 统一 CLI + 分层配置"。

---

## 1. QuantML 借鉴的核心思想

### 1.1 重构前的痛点（与本项目对照）
- **代码散落**：examples 变第二个源码目录
- **依赖膨胀**：只想用 LightGBM 却被带入 torch + Dashboard
- **维护困难**：pandas/numpy/torch/TA-Lib 版本冲突

### 1.2 核心机制：扩展目录聚合
```
qlib/ext/
├── benchmarks/   # 模型 + 配置 + 训练流程
├── models/       # 25+ 预测模型
├── configs/      # 每个 model 一个 YAML
├── data_formatters/  # Informer/DLinear/PatchTST
├── handlers/     # LightGBM 多频 / 高频处理器
├── factors/
│   ├── evaluator.py         # RankIC / ICIR
│   ├── generation.py        # DeepSeek LLM 因子生成
│   ├── alphagpt_framework.py# 迭代优化
│   ├── gplearn/             # 遗传规划
│   ├── alphakan/            # KAN 网络
│   └── pool.py              # 因子池（互斥 IC + 权重）
├── adapters/     # 5 个数据源适配器
├── factory.py    # 工厂模式按名字创建
├── config.py     # 分层配置（YAML → 环境变量）
└── cli.py        # qlib-data-source CLI
```

**关键价值**：
1. 代码与 Qlib 核心**解耦**（不修改源码即可热插拔）
2. **按需安装**（`pip install -e .[factors]`）
3. 领域边界**清晰**（实现 / 配置 / 示例彻底分离）

---

## 2. 与本项目现状对比

### 2.1 本项目已具备的优势（不要重写）
| 层级 | 模块 | 状态 |
|------|------|------|
| 数据层 | `DataLoader`, `PyBrokerDataSource` | ✅ 成熟 |
| 因子层 | `FactorEngine`, 31 因子 | ✅ 成熟 |
| 评估层 | `FactorEvaluator`, `FactorPipeline` | ✅ 成熟 |
| 组合层 | `PortfolioManager`, `RiskController` | ✅ 成熟 |
| 回测层 | `PyBrokerBacktestRunner` | ✅ 成熟 |
| 编排层 | `Pipeline` | ✅ 成熟 |
| 入口层 | `run_backtest.py` / `run_optimize.py` / `run_validate.py` | ✅ 收敛完毕（规则 20） |
| 配置层 | `BacktestConfig.from_yaml` + `with_config()` | ✅ 单一数据源（规则 2） |

### 2.2 与 QuantML 类似的现存问题
| 问题 | 本项目现状 |
|------|-----------|
| **依赖边界不清** | `requirements.txt` 全量，无 `extras_require` |
| **第三方扩展散落** | `cross_spread.py` 散落在 `core/factors/alpha_futures/`，未来更多 |
| **数据源工厂未沉淀** | `create_hybrid_data_source()` 在 `core/engine/` 里硬编码 TqSdk/CSV 两种 |
| **因子挖掘空白** | 评估强，挖掘空白（无 GP/LLM 因子生成） |
| **CLI 未统一** | 3 个 `run_*.py` 各有 argparse，无顶层 `quant-system` CLI |

---

## 3. 借鉴方案（按 ROI 排序）

### 3.1 高 ROI：建立 `core/ext/` 扩展目录（4 子模块）

**目标**：未来新增的"可选能力"全部进 `core/ext/`，不影响核心架构。

```
core/ext/
├── adapters/                  # 数据源适配器（数据层扩展）
│   ├── base.py                # DataSourceAdapter 抽象基类
│   ├── tqsdk_adapter.py       # 从 core/data_loader 抽出
│   ├── csv_adapter.py         # 从 core/data_loader 抽出
│   └── factory.py             # create_data_source(name, **kwargs)
├── factors/                   # 因子挖掘扩展（因子层扩展）
│   ├── generation/
│   │   ├── gplearn.py         # 遗传规划（gplearn 库）
│   │   ├── llm_generator.py   # LLM 因子生成（DeepSeek/OpenAI）
│   │   └── alphagpt.py        # 迭代优化编排
│   ├── pool/                  # 因子池（互斥 IC + 权重）
│   │   ├── manager.py
│   │   └── decay.py           # 复用 core/engine/factor_decay.py
│   └── operators/             # 算子扩展（TA-Lib）
│       └── talib_ops.py
├── models/                    # 预测模型扩展（评估层扩展）
│   ├── base.py                # BasePredictor 抽象
│   ├── lgbm.py
│   ├── mlp.py
│   └── configs/               # 模型 YAML
├── handlers/                  # 多频/高频数据处理器
└── utils/                     # 工具函数扩展
```

**约束**：
- 复用 `core/factors/alpha_futures/factor_engine.py` 的 `BaseFactor` 和 `register_factor`，不重复造轮子
- 复用 `core/engine/pybroker_data_source.py` 的 `PyBrokerDataSource`，不重复造数据源
- 通过 `core/ext/adapters/factory.py::create_data_source(name, **kwargs)` 暴露统一接口

### 3.2 高 ROI：按需安装（extras_require）

**当前**：`requirements.txt` 100+ 行一锅端。

**改造**：建立 `pyproject.toml` 或 `setup.py`：

```toml
[project]
name = "backtest"
dependencies = [  # 核心：最小依赖
    "numpy>=1.24",
    "pandas>=2.0",
    "pyyaml>=6.0",
    "loguru>=0.7",
    "pybroker>=0.4",
]

[project.optional-dependencies]
core = ["pybroker", "tqsdk"]                              # 核心回测
data-sources = ["tqsdk", "akshare", "tushare"]            # 数据源
factors = ["gplearn", "deap"]                             # 因子挖掘（GP）
llm = ["openai>=1.0", "anthropic"]                        # LLM 因子生成
models = ["lightgbm", "xgboost", "torch"]                 # 预测模型
dashboard = ["streamlit", "plotly"]                       # UI
all = ["backtest[core,data-sources,factors,llm,models]"]  # 全量
```

**安装命令**：
```bash
pip install -e .[core]              # 最小安装（开发必备）
pip install -e .[factors]           # 加上遗传规划
pip install -e .[llm,models]        # 加上 AI 模型
pip install -e .[all]               # 全量（不推荐）
```

### 3.3 中 ROI：数据源工厂模式

**当前**：
```python
# core/engine/pybroker_data_source.py
def create_hybrid_data_source(phone=None, password=None, symbols=None, ...):
    """混合数据源工厂：TqSdk 在线 + 本地 CSV"""
    if online_ok:
        return TqSdkDataSource(...)
    raise RuntimeError("禁止回退到全量本地 CSV")  # 规则 1
```

**改造**（先抽出抽象，再注册到工厂）：

```python
# core/ext/adapters/base.py
class DataSourceAdapter(ABC):
    @abstractmethod
    def load(self, symbols: List[str], start: str, end: str) -> pd.DataFrame: ...

# core/ext/adapters/factory.py
_DATA_SOURCE_REGISTRY: Dict[str, Type[DataSourceAdapter]] = {}

def register_adapter(name: str):
    def deco(cls): _DATA_SOURCE_REGISTRY[name] = cls; return cls
    return deco

def create_data_source(name: str, **kwargs) -> DataSourceAdapter:
    if name not in _DATA_SOURCE_REGISTRY:
        raise KeyError(f"未知数据源 {name}，已注册: {list(_DATA_SOURCE_REGISTRY)}")
    return _DATA_SOURCE_REGISTRY[name](**kwargs)
```

效果：新增 AKShare/RQData/Binance 只需写 `akshare_adapter.py` + `@register_adapter("akshare")`，不动 core。

### 3.4 中 ROI：统一 CLI（可选优化）

**当前**：
```bash
python run_backtest.py --experiment e1
python run_optimize.py --strategy trend
python run_validate.py --method monte_carlo
```

**改造**（保留 3 个入口，新增聚合 CLI）：
```bash
quant-system backtest --experiment e1
quant-system optimize --strategy trend
quant-system validate --method monte-carlo
quant-system factor mine --method gplearn    # 后续接 core/ext/factors/generation/
```

实现：
```python
# cli.py
import click
from runner import Pipeline

@click.group()
def cli(): pass

@cli.command()
@click.option('--experiment', default='e1')
def backtest(experiment): Pipeline('config.yaml').run_backtest(experiment)

@cli.command()
@click.option('--strategy', required=True)
def optimize(strategy): Pipeline('config.yaml').optimize(tasks=['grid'], strategy=strategy)

# ... 注册到 pyproject.toml 的 [project.scripts]
```

**注意**：不删除 3 个 `run_*.py`（规则 20：保留官方入口，50+ 处引用），`quant-system` 是增强入口。

### 3.5 长期：分层配置 + 因子挖掘

- **分层配置**：YAML + 环境变量 + 运行时覆盖三段；当前 `Pipeline.with_config()` 已支持运行时覆盖，但 YAML→环境变量尚未引入。**优先级低**，因为项目主要在单环境跑。
- **因子挖掘（GP/LLM/AlphaGPT）**：等核心架构稳定后再上。**优先级最低**。

---

## 4. 不必借鉴的部分

| 借鉴项 | 不借鉴理由 |
|--------|-----------|
| `qlib.ext` 的 `data_formatters/` | 本项目数据格式单一（OHLCV + OI），无多模型多格式需求 |
| `qlib.ext` 的 `talib_ops.py` | operators.py 已自建，talib 是 C 依赖反而增加维护成本 |
| 完整的 `RD-Agent` 框架 | 太重，违反"避免过度工程化"原则 |
| 全部 25+ 预测模型 | 本项目因子驱动，非模型驱动；ML 预测是另一个方向 |

---

## 5. 落地的具体 TODO（按 ROI 排序）

| # | 任务 | 依赖 | 工作量 | 优先级 |
|---|------|------|--------|--------|
| 1 | 创建 `core/ext/` 目录骨架 + `__init__.py` 公共导出 | 无 | 0.5h | 高 |
| 2 | `core/ext/adapters/base.py` + `factory.py` + 迁移 TqSdk/CSV | 1 | 2h | 高 |
| 3 | `pyproject.toml` 的 `extras_require`（4 个 extras） | 1 | 1h | 高 |
| 4 | `core/ext/factors/operators/talib_ops.py`（可选，依赖 `pip install -e .[factors]`） | 1, 3 | 2h | 中 |
| 5 | `cli.py` 统一入口（不删 3 个 run_*.py） | 1 | 1h | 中 |
| 6 | `core/ext/factors/generation/gplearn.py`（GP 因子挖掘） | 1, 3 | 4h | 中 |
| 7 | `core/ext/factors/generation/llm_generator.py`（LLM 因子生成） | 1, 3 | 4h | 低 |
| 8 | `core/ext/models/lgbm.py` + `mlp.py`（预测模型） | 1, 3 | 4h | 低 |
| 9 | 分层配置（YAML → 环境变量 → 运行时） | 无 | 3h | 低 |

**第一阶段（高 ROI）**：任务 1+2+3，预计 3.5h，建立扩展边界 + 按需安装 + 数据源工厂。
**第二阶段（中 ROI）**：任务 4+5，预计 3h，TA-Lib 算子 + 统一 CLI。
**第三阶段（探索）**：任务 6+7+8+9，长期演进。

---

## 6. 决策与下一步

- **借鉴决策**：采纳 QuantML 的"ext/ 聚合 + 按需安装 + 工厂模式"三大思想
- **保留本项目特色**：规则 2（BacktestConfig 单一来源）+ 规则 20（3 个官方入口不删）+ 规则 17（不重复造轮子）
- **避免过度工程化**：仅做 4 项高/中 ROI 改造，不全面照搬 Qlib 架构
- **下一步**：执行第一阶段 3 个高 ROI 任务（建立 core/ext/ 骨架 + 数据源工厂 + 按需安装）

---

## 7. 参考资料

- [QuantML 微信文章] https://mp.weixin.qq.com/s/Ewq8GVATCVLFA29xI-P21w
- [Qlib 官方文档] https://qlib.readthedocs.io/
- [Qlib 数据提供者结构演变] 0.4.0 changelog
- 本地 docx：`/Users/luojiutian/Downloads/QuantML系统架构借鉴.docx`
