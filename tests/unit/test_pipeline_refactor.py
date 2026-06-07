"""验证 pipeline.py P0/P1/P2 整改。"""
import inspect
import subprocess
import numpy as np
from runner.pipeline import Pipeline
from runner.validation.factor_alpha24 import factor_alpha24_screening

# 1) _run_factor_screening 已删除
import runner.pipeline as pl_mod
assert not hasattr(pl_mod, "_run_factor_screening"), "_run_factor_screening 仍存在"
print("OK: _run_factor_screening 已删除")

# 2) screen_factors 委托 factor_alpha24_screening
src = inspect.getsource(Pipeline.screen_factors)
assert "factor_alpha24_screening" in src
assert "_run_factor_screening" not in src
print("OK: Pipeline.screen_factors 委托 factor_alpha24_screening")

# 3) with_config 类型注解
sig = inspect.signature(Pipeline.with_config)
ann = sig.parameters["overrides"].annotation
print(f"OK: with_config overrides 注解 = {ann}")

# 4) 文件行数
n = int(subprocess.check_output(["wc", "-l", "runner/pipeline.py"]).split()[0])
print(f"OK: pipeline.py 当前 {n} 行 (Pipeline 编排器主体 + 4 个编排辅助函数)")
assert n < 700, f"pipeline.py {n} 行 过大"

# 5) factor_alpha24_screening 不再手写 IC
src_fn = inspect.getsource(factor_alpha24_screening)
assert "np.corrcoef" not in src_fn, "factor_alpha24_screening 仍含手写 IC"
print("OK: factor_alpha24_screening 已无手写 IC")

# 6) factor_alpha24_screening 形参已统一为 data_source
sig2 = inspect.signature(factor_alpha24_screening)
assert list(sig2.parameters.keys())[0] == "data_source"
print(f"OK: factor_alpha24_screening 形参[0] = {list(sig2.parameters.keys())[0]}")

# 7) 验证 FactorEvaluator 集成
from core.factors.factor_evaluator import FactorEvaluator
from core.factors import AlphaFutures24, AlphaFuturesConfig
calc = AlphaFutures24(AlphaFuturesConfig())
n_pts = 200
close = np.cumprod(1 + np.random.normal(0, 0.01, n_pts))
factors = calc.compute_all(close=close, open_price=close, high=close * 1.01,
                            low=close * 0.99, open_interest=np.ones(n_pts) * 100)
fwd = np.zeros(n_pts)
fwd[:-5] = (close[5:] - close[:-5]) / close[:-5]
ev = FactorEvaluator(forward_period=5, ic_window=60, min_observations=30)
results = ev.evaluate_batch(factors, fwd)
print(f"OK: FactorEvaluator 批量评估 {len(results)} 因子")
