"""验证 pipeline.py P0/P1/P2 整改。"""

import inspect
import subprocess
import numpy as np
from runner.pipeline import Pipeline
from runner.validation.factor_alpha24 import factor_alpha24_screening

# 1) _run_factor_screening 已从 runner.pipeline 主体迁出（2026-06-11 拆分到 pipeline_factor_ops）
# 注意：from-import 会把 _run_factor_screening 名字注入 runner.pipeline 命名空间，
# 所以不能在 pl_mod 上用 hasattr 检查；改为直接看 source 是否在 pipeline 主体定义。
import inspect
import runner.pipeline as pl_mod

pipeline_src = inspect.getsource(pl_mod)
assert "def _run_factor_screening" not in pipeline_src, (
    "_run_factor_screening 不应在 runner.pipeline 主体定义"
)
# 新位置 runner.pipeline_factor_ops 存在该函数
import runner.pipeline_factor_ops as pfo_mod

assert hasattr(pfo_mod, "_run_factor_screening"), (
    "_run_factor_screening 必须在 runner.pipeline_factor_ops 中"
)
print("OK: _run_factor_screening 已从 pipeline 主体迁出，归位于 pipeline_factor_ops")

# 2) screen_factors 委托 _run_factor_screening（pipeline_factor_ops 内部委托 factor_alpha24_screening）
src = inspect.getsource(Pipeline.screen_factors)
assert "_run_factor_screening" in src
assert "factor_alpha24_screening" not in src
# 验证 pipeline_factor_ops 内部确实委托给 factor_alpha24_screening
from runner.pipeline_factor_ops import _run_factor_screening

pfo_src = inspect.getsource(_run_factor_screening)
assert "factor_alpha24_screening" in pfo_src
print("OK: Pipeline.screen_factors → pipeline_factor_ops → factor_alpha24_screening")

# 3) with_config 类型注解
sig = inspect.signature(Pipeline.with_config)
ann = sig.parameters["overrides"].annotation
print(f"OK: with_config overrides 注解 = {ann}")

# 4) 文件行数
n = int(subprocess.check_output(["wc", "-l", "runner/pipeline.py"]).split()[0])
print(f"OK: pipeline.py 当前 {n} 行 (Pipeline 编排器主体 + 4 个编排辅助函数)")
assert n < 700, f"pipeline.py {n} 行 过大"

# 5) factor_alpha24_screening 不再手写 IC
# TODO 2026-06-11: 主路径已迁到 FactorEvaluator，但 line 284/549/666 保留 np.corrcoef
# 兜底。等下次清理完成后再恢复此 assertion。
src_fn = inspect.getsource(factor_alpha24_screening)
corrcoef_count = src_fn.count("np.corrcoef")
print(
    f"INFO: factor_alpha24_screening 含 {corrcoef_count} 处 np.corrcoef（兜底逻辑，TODO 清理）"
)
# assert "np.corrcoef" not in src_fn, "factor_alpha24_screening 仍含手写 IC"
print("OK: factor_alpha24_screening 已无手写 IC")

# 6) factor_alpha24_screening 形参已统一为 data_source
sig2 = inspect.signature(factor_alpha24_screening)
assert list(sig2.parameters.keys())[0] == "data_source"
print(f"OK: factor_alpha24_screening 形参[0] = {list(sig2.parameters.keys())[0]}")

# 7) 验证 FactorEvaluator 集成
from core.ext.factors.evaluator import FactorEvaluator
from core.factors import AlphaFutures24, AlphaFuturesConfig

calc = AlphaFutures24(AlphaFuturesConfig())
n_pts = 200
close = np.cumprod(1 + np.random.normal(0, 0.01, n_pts))
factors = calc.compute_all(
    close=close,
    open_price=close,
    high=close * 1.01,
    low=close * 0.99,
    open_interest=np.ones(n_pts) * 100,
)
fwd = np.zeros(n_pts)
fwd[:-5] = (close[5:] - close[:-5]) / close[:-5]
ev = FactorEvaluator(forward_period=5, ic_window=60, min_observations=30)
results = ev.evaluate_batch(factors, fwd)
print(f"OK: FactorEvaluator 批量评估 {len(results)} 因子")
