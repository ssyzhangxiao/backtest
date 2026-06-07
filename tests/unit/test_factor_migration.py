"""测试：验证所有24个因子在新引擎下计算正常。"""
import numpy as np
from core.factors.alpha_futures_24 import AlphaFutures24, AlphaFuturesConfig

np.random.seed(42)
n = 500
close = np.cumsum(np.random.randn(n) * 0.01) + 100
open_price = close + np.random.randn(n) * 0.005
high = np.maximum(close, open_price) + np.abs(np.random.randn(n) * 0.01)
low = np.minimum(close, open_price) - np.abs(np.random.randn(n) * 0.01)
open_interest = np.cumsum(np.random.randn(n) * 0.5) + 10000
near_price = close * 0.99
far_price = close * 1.01

config = AlphaFuturesConfig()
af = AlphaFutures24(config)

result = af.compute_all(
    close=close, open_price=open_price, high=high, low=low,
    open_interest=open_interest, near_price=near_price, far_price=far_price,
)

print(f'计算完成，共 {len(result)} 个因子:')
for name in sorted(result.keys()):
    arr = result[name]
    nan_ratio = np.isnan(arr).mean()
    print(f'  {name}: shape={arr.shape}, NaN占比={nan_ratio:.2%}, mean={np.nanmean(arr):.4f}')