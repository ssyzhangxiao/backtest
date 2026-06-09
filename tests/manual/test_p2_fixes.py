"""验证 P2 三项整改"""
import inspect

# 1. 验证 grid_search.py 中的可配置参数
from runner.optimization.grid_search import (
    param_stability_test,
    DEFAULT_FAILURE_STABILITY_SCORE,
    DEFAULT_NEUTRAL_STABILITY_SCORE,
)
print(f"[1] DEFAULT_FAILURE_STABILITY_SCORE = {DEFAULT_FAILURE_STABILITY_SCORE}")
print(f"[1] DEFAULT_NEUTRAL_STABILITY_SCORE = {DEFAULT_NEUTRAL_STABILITY_SCORE}")

sig = inspect.signature(param_stability_test)
print(f"[1] param_stability_test 参数: {list(sig.parameters.keys())}")
assert "failure_stability_score" in sig.parameters
assert "neutral_stability_score" in sig.parameters
assert sig.parameters["failure_stability_score"].default == 0.3
assert sig.parameters["neutral_stability_score"].default == 0.5
print("[1] OK: param_stability_test 包含可配置回退分数参数")

# 2. 验证 copy_config 文档包含示例
from runner.optimization import copy_config
doc = copy_config.__doc__
assert "Examples:" in doc, "copy_config 文档缺少 Examples 章节"
assert "dataclasses.replace" in doc, "copy_config 文档缺少 dataclasses.replace 关系说明"
assert "Basic usage" in doc or "基本用法" in doc
print("[2] OK: copy_config 文档包含 Examples + dataclasses.replace 关系说明")

# 3. 验证 pipeline.py 不再检查 factor_weights（注释中提到历史不算）
from runner.pipeline import Pipeline
src = inspect.getsource(Pipeline.verify_chain)
# 取出实际可执行代码（去掉注释行）来判断
exec_lines = "\n".join(
    line for line in src.splitlines() if not line.strip().startswith("#")
)
assert "backtest_config_has_factor_weights" not in exec_lines, (
    f"实际代码中仍含 factor_weights 检查:\n{exec_lines}"
)
assert "bool(self._config.factor_weights)" not in exec_lines, (
    "实际代码中仍含 bool(factor_weights) 检查"
)
print("[3] OK: pipeline.verify_chain 已删除 factor_weights 检查（注释中提及历史除外）")

# 4. 验证 Pipeline 仍能正常导入
print("[4] OK: Pipeline 可正常导入")

# 5. 验证 copy_config 函数行为
from dataclasses import dataclass

@dataclass
class MockCfg:
    x: int = 1
    y: int = 2

m = MockCfg()
m2 = copy_config(m, x=99)
assert m2.x == 99
assert m2.y == 2
assert m.x == 1  # 原对象未变
print("[5] OK: copy_config 正确复制并覆盖字段，原对象未被修改")

print()
print("ALL P2 CHECKS PASSED")
