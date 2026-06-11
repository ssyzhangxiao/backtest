"""
测试：验证 FactorPipeline 全流程 + 30个因子（24原有 + 6新增）。
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.factors.alpha_futures_24 import AlphaFutures24, AlphaFuturesConfig
from core.ext.factors.alpha_futures.factor_pipeline import FactorPipeline
from core.ext.factors.alpha_futures.factor_registry import list_available_factors

np.random.seed(42)
n = 500

# 生成模拟数据
close = np.cumsum(np.random.randn(n) * 0.01) + 100
open_price = close + np.random.randn(n) * 0.005
high = np.maximum(close, open_price) + np.abs(np.random.randn(n) * 0.01)
low = np.minimum(close, open_price) - np.abs(np.random.randn(n) * 0.01)
open_interest = np.cumsum(np.random.randn(n) * 0.5) + 10000
volume = np.abs(np.random.randn(n) * 1000) + 5000
near_price = close * 0.99
far_price = close * 1.01

# 前瞻收益（5日）
forward_ret = np.zeros(n)
forward_ret[:-5] = (close[5:] - close[:-5]) / close[:-5]

print("=== 1. 检查因子注册 ===")
factors = list_available_factors()
print(f"已注册因子: {len(factors)} 个")
print(f"因子列表: {sorted(factors)}")

print("\n=== 2. AlphaFutures24 计算全部因子 ===")
config = AlphaFuturesConfig()
af = AlphaFutures24(config)
result = af.compute_all(
    close=close, open_price=open_price, high=high, low=low,
    open_interest=open_interest, near_price=near_price, far_price=far_price,
    volume=volume,
)
print(f"计算完成: {len(result)} 个因子")
for name in sorted(result.keys()):
    arr = result[name]
    print(f"  {name}: shape={arr.shape}, NaN占比={np.isnan(arr).mean():.1%}, mean={np.nanmean(arr):.4f}")

print("\n=== 3. FactorPipeline 全流程测试 ===")
raw_data = {
    "close": close, "open_price": open_price, "high": high, "low": low,
    "open_interest": open_interest, "near_price": near_price, "far_price": far_price,
    "volume": volume,
}

pipeline = FactorPipeline(config)
pipeline_result = pipeline.run(raw_data, forward_ret)

print("\n=== 4. Pipeline 结果摘要 ===")
print(pipeline.report())

print("\n=== 5. 分项结果 ===")
print(f"有效因子: {pipeline.get_valid_factors()}")
print(f"精选因子: {pipeline.get_selected_factors()}")
print(f"保留因子: {pipeline.get_retain_factors()}")
print(f"有效变换: {len(pipeline.get_effective_transforms())} 个")

print("\n=== 测试完成 ===")