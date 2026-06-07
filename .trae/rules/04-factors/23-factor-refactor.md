# 规则23：因子库工程化重构 — 基于抽象基类的独立因子类体系

**核心原则**：将函数式+编排类的因子计算结构，重构为**基于抽象基类的独立因子类 + 注册表 + 引擎调度** 的架构，实现易扩展、易测试、易维护。

**新架构设计**：
```
core/factors/
├── alpha_futures/                    # 新因子库目录
│   ├── __init__.py
│   ├── base_factor.py                 # 因子抽象基类 BaseFactor
│   ├── factor_registry.py             # 因子注册表（装饰器注册）
│   ├── factor_engine.py               # 因子计算引擎（数据清洗+调度）
│   ├── factors/                      # 独立因子类目录
│   │   ├── __init__.py
│   │   ├── t_01.py, t_02.py, ...    # 24个独立因子类
│   ├── operators.py                 # 保持不变（基础算子）
│   └── futures_data_cleaners.py     # 保持不变（数据清洗）
├── alpha_futures_23.py -> alpha_futures_24.py  # 保持不变，内部委托给新引擎
```

**具体规则**：

1. **因子基类（`base_factor.py`）**：
   - 每个因子继承 `BaseFactor` 抽象基类
   - 必须定义 `name`、`category`、`formula`、`dependencies` 类属性
   - 实现 `compute` 纯计算方法，仅依赖 kwargs 提供的字段
   - 可选实现 `post_process` 做因子特定后处理
   - 通过 `self.config` 访问全局配置

2. **因子注册表（`factor_registry.py`）**：
   - 使用 `@register_factor` 装饰器自动注册因子类
   - 提供 `get_factor`、`list_available_factors` 等查询接口
   - 注册表是全局单例，因子导入时自动注册

3. **因子引擎（`factor_engine.py`）**：
   - `FactorEngine` 负责：数据清洗 → 公共数据准备 → 因子调度 → 结果汇总
   - 在 `_prepare_public_data` 中集中计算所有因子需要的中间量并缓存（如 `oi_mean_20`、`delta_oi_1`、`carry_orth`）
   - 统一处理所有因子的公共依赖，避免重复计算
   - 检查因子依赖是否已准备，缺失则报错

4. **独立因子类（`factors/t_01.py` 等）**：
   - 每个因子一个独立文件，类名与因子编号对应（如 `class T_01(BaseFactor)`）
   - 明确声明 `dependencies` 列表（如 `["close", "oi_safe"]`）
   - `compute` 方法纯计算，无副作用
   - 复用 `operators.py` 中的基础算子

5. **向后兼容**：
   - 保留原 `AlphaFutures24` 类作为外观类，内部委托给新 `FactorEngine`
   - 保持原有 `compute_all` 接口签名完全一致，外部调用无需修改

**迁移指南**：
- 从 `alpha_futures_trend.py` 等模块中提取单个因子计算逻辑
- 封装为独立类，用 `@register_factor` 装饰
- 将原函数中的全局配置引用改为 `self.config`
- 在 `FactorEngine._prepare_public_data` 中计算公共依赖字段
- 编写独立单元测试验证每个因子

**涉及代码**：
- `core/factors/alpha_futures/base_factor.py`：因子抽象基类
- `core/factors/alpha_futures/factor_registry.py`：因子注册表
- `core/factors/alpha_futures/factor_engine.py`：因子计算引擎
- `core/factors/alpha_futures/factors/`：24个独立因子类文件
- `core/factors/alpha_futures_24.py`：向后兼容外观类
