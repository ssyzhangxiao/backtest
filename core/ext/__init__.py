"""core/ext - 可选扩展目录（规则21）。

借鉴 QuantML-Qlib `qlib.ext` 重构思想：

- **按需安装**：第三方依赖通过 requirements-{name}.txt 拆分，不进入核心 requirements.txt
- **工厂注册**：数据源/算子/模型使用装饰器注册，禁止 if/elif 链
- **复用核心**：所有扩展必须复用 core/factors/alpha_futures、core/engine、core/data_loader 等核心系统
- **依赖方向**：core/ext → core 允许；core/ext → runner 禁止

扩展模块（按 ROI 排序）：

| 子目录 | 状态 | 说明 |
|--------|------|------|
| adapters/ | 规划中 | 数据源适配器（TqSdk/CSV/AKShare/RQData） |
| factors/generation/ | 规划中 | 因子挖掘（GP/LLM/AlphaGPT） |
| factors/pool/ | 规划中 | 因子池（互斥 IC + 权重 + 衰减） |
| factors/operators/ | 规划中 | 算子扩展（TA-Lib 等） |
| models/ | 规划中 | 预测模型（LGBM/MLP） |
| handlers/ | 规划中 | 多频/高频数据处理器 |
| utils/ | 规划中 | 工具函数扩展 |

按需安装：

    pip install -r requirements.txt                # 核心
    pip install -r requirements-data-sources.txt    # 数据源适配
    pip install -r requirements-factors.txt         # GP 因子挖掘
    pip install -r requirements-llm.txt             # LLM 因子生成
    pip install -r requirements-models.txt          # 预测模型
    pip install -r requirements-all.txt             # 全量

参考：
    .trae/rules/01-basics/21-ext-directory.md
    .trae/knowledges/20260611_001_knowledge_quantml-qlib-ext-borrow.md
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__: list[str] = []
