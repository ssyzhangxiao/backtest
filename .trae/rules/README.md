# 量化回测系统开发规范

本目录包含项目开发规范的模块化文件，便于维护和查找。

## 目录结构

```
.trae/rules/
├── README.md              # 本文件：规范索引
├── project_rules.md       # 原文件（已归档）
├── 01-basics/             # 基础架构与规范
│   ├── 02-config.md
│   ├── 03-deprecated.md
│   ├── 06-testing.md
│   ├── 07-file-limit.md
│   ├── 08-naming.md
│   ├── 16-directory.md
│   ├── 17-common-systems.md
│   ├── 18-pipeline.md
│   ├── 19-dependencies.md
│   └── 21-ext-directory.md
├── 02-engine/             # 回测引擎
│   ├── 01-no-fallback.md
│   └── 26-cross-validation.md
├── 03-strategies/         # 策略开发
│   ├── 04-risk-control.md
│   ├── 05-strategy-registry.md
│   ├── 11-multi-tf.md
│   ├── 13-stop-loss.md
│   ├── 21-sub-strategies.md
│   ├── 24-indicator-registry.md
│   ├── 25-exit-hooks.md
│   └── 27-strategy-base.md
├── 04-factors/            # 因子开发
│   ├── 09-factor-dev.md
│   ├── 20-factor-cleaning.md
│   └── 23-factor-refactor.md
└── 05-validation/         # 验证与优化
    ├── 10-removed.md
    ├── 12-removed.md
    ├── 14-removed.md
    ├── 15-backtest-validation.md
    ├── 22-rolling-window.md
    └── 28-strategy-value-validation.md
```

## 规范索引

### 基础架构与规范

| 规则编号 | 规则名称 | 文件 |
|---------|---------|------|
| 2 | 配置管理 — config.yaml 是单一数据源 | [01-basics/02-config.md](./01-basics/02-config.md) |
| 3 | 废弃代码必须彻底清理 | [01-basics/03-deprecated.md](./01-basics/03-deprecated.md) |
| 6 | 测试覆盖 — 关键路径必须有测试 | [01-basics/06-testing.md](./01-basics/06-testing.md) |
| 7 | 文件行数限制 | [01-basics/07-file-limit.md](./01-basics/07-file-limit.md) |
| 8 | 命名必须与功能一致 | [01-basics/08-naming.md](./01-basics/08-naming.md) |
| 16 | 模块目录结构 — 职责单一，接口清晰 | [01-basics/16-directory.md](./01-basics/16-directory.md) |
| 17 | 不重复造轮子 — 优先调用公共系统 | [01-basics/17-common-systems.md](./01-basics/17-common-systems.md) |
| 18 | Pipeline 编排器 — 声明式调用 | [01-basics/18-pipeline.md](./01-basics/18-pipeline.md) |
| 19 | 依赖方向检查 — 禁止反向依赖 | [01-basics/19-dependencies.md](./01-basics/19-dependencies.md) |
| 20 | 根目录脚本迁移 — run_*.py 必须收敛到 Pipeline | [01-basics/20-root-scripts-migration.md](./01-basics/20-root-scripts-migration.md) |
| 21 | 扩展目录 ext/ — 借鉴 QuantML-Qlib 按需加载 + 工厂注册 | [01-basics/21-ext-directory.md](./01-basics/21-ext-directory.md) |

### 回测引擎

| 规则编号 | 规则名称 | 文件 |
|---------|---------|------|
| 1 | 引擎回退禁止，必须并行验证 | [02-engine/01-no-fallback.md](./02-engine/01-no-fallback.md) |
| 26 | 交叉验证机制 — 自研引擎与 PyBroker 并行验证 | [02-engine/26-cross-validation.md](./02-engine/26-cross-validation.md) |

### 策略开发

| 规则编号 | 规则名称 | 文件 |
|---------|---------|------|
| 4 | 风控类统一 — 一个系统只有一个风控 | [03-strategies/04-risk-control.md](./03-strategies/04-risk-control.md) |
| 5 | 策略注册统一 — 多策略横截面打分 | [03-strategies/05-strategy-registry.md](./03-strategies/05-strategy-registry.md) |
| 11 | 多时间框架（规划中） | [03-strategies/11-multi-tf.md](./03-strategies/11-multi-tf.md) |
| 13 | 止损策略 — 分层叠加，效果可量化 | [03-strategies/13-stop-loss.md](./03-strategies/13-stop-loss.md) |
| 21 | 多策略子策略划分与集成 — 5 子策略体系 | [03-strategies/21-sub-strategies.md](./03-strategies/21-sub-strategies.md) |
| 24 | 策略指标注册表 — 解耦指标计算与回测引擎 | [03-strategies/24-indicator-registry.md](./03-strategies/24-indicator-registry.md) |
| 25 | 策略退出钩子注册表 — 解耦退出逻辑与执行器 | [03-strategies/25-exit-hooks.md](./03-strategies/25-exit-hooks.md) |
| 27 | 策略基类设计 — 可配置化与可扩展性 | [03-strategies/27-strategy-base.md](./03-strategies/27-strategy-base.md) |

### 因子开发

| 规则编号 | 规则名称 | 文件 |
|---------|---------|------|
| 9 | 因子开发规范 — 24因子体系，IC 驱动，先验证后集成 | [04-factors/09-factor-dev.md](./04-factors/09-factor-dev.md) |
| 20 | 因子数据清洗与工程化 — 换月/交割/涨跌停处理 | [04-factors/20-factor-cleaning.md](./04-factors/20-factor-cleaning.md) |
| 23 | 因子库工程化重构 — 基于抽象基类的独立因子类体系 | [04-factors/23-factor-refactor.md](./04-factors/23-factor-refactor.md) |

### 验证与优化

| 规则编号 | 规则名称 | 文件 |
|---------|---------|------|
| 10 | （已移除）自适应参数模块已删除，功能由子策略体系覆盖 | [05-validation/10-removed.md](./05-validation/10-removed.md) |
| 12 | （已移除）动态仓位模块已删除，功能由子策略体系覆盖 | [05-validation/12-removed.md](./05-validation/12-removed.md) |
| 14 | （已移除）品种选择模块已删除，功能由子策略体系覆盖 | [05-validation/14-removed.md](./05-validation/15-backtest-validation.md) |
| 15 | 回测验证 — 多阶段验证，鲁棒性优先 | [05-validation/15-backtest-validation.md](./05-validation/15-backtest-validation.md) |
| 22 | 回测验证 — 滚动窗口 + 样本外验证 | [05-validation/22-rolling-window.md](./05-validation/22-rolling-window.md) |
| 28 | 策略价值验证 — 5 阶段硬性验证（A因子→B组合IC→C回测→D稳健→E抗噪） | [05-validation/28-strategy-value-validation.md](./05-validation/28-strategy-value-validation.md) |

## 维护指南

### 新增规则

1. 确定规则所属分类（基础/引擎/策略/因子/验证）
2. 在对应分类目录下创建新文件，命名规则：`XX-规则描述.md`（XX 为规则编号）
3. 在文件中编写规则内容
4. 更新本 README.md 的对应分类表格，添加新规则索引

### 修改规则

1. 找到对应规则的文件
2. 直接修改文件内容
3. 如需变更规则编号或分类，同步更新本 README.md 的索引

### 删除规则

1. 将规则文件移动或标记为已移除（如规则 10/12/14）
2. 更新本 README.md 的索引，标记为已移除或删除对应条目

### 生成单一汇总文件（可选）

如需生成类似原 `project_rules.md` 的单一汇总文件，可以使用脚本将所有规则文件合并。

---

*最后更新：2026-06-06*
*参考指南：商品期货量化模型改造指南.docx*
*参考指南：商品期货 Alpha 因子库工程化重构提示词.docx*
*相关知识文档：../knowledges/20260602_001_workflow_strategy-enhancement-roadmap.md*
*相关知识文档：../knowledges/20260602_002_workflow_runner-scripts-refactor-plan.md*
