import numpy as np
import pandas as pd
from runner.common.portfolio_utils import fuse_equities_by_weights

# 测试 1: 等权融合两条递增曲线
np.random.seed(0)
n = 100
dates = pd.date_range('2024-01-01', periods=n)
eq1 = pd.Series(np.cumprod(1 + np.random.RandomState(0).normal(0, 0.01, n)), index=dates, name='A')
eq2 = pd.Series(np.cumprod(1 + np.random.RandomState(1).normal(0, 0.01, n)), index=dates, name='B')
df = pd.DataFrame({'A': eq1, 'B': eq2})

fused_eq = fuse_equities_by_weights(df, {'A': 0.5, 'B': 0.5})
assert fused_eq.iloc[0] == 1.0
assert len(fused_eq) == n
assert np.all(np.isfinite(fused_eq))
print('Test 1 OK: 等权融合 len=', len(fused_eq), 'final=', round(fused_eq.iloc[-1], 4))

# 测试 2: 空权重 → 全部为 1.0
fused_zero = fuse_equities_by_weights(df, {})
assert (fused_zero == 1.0).all()
print('Test 2 OK: 空权重 → 全 1.0')

# 测试 3: 单策略（fuse 返回的首值强制为 1.0，并按日收益率重新累积）
# 预期：每日日收益率等于原策略从第 2 行起的日收益率
fused_one = fuse_equities_by_weights(df, {'A': 1.0})
assert fused_one.iloc[0] == 1.0
expected_daily = (df['A'] / df['A'].shift(1) - 1.0).fillna(0.0)
expected_equity = (1.0 + expected_daily).cumprod()
expected_equity.iloc[0] = 1.0
np.testing.assert_allclose(fused_one.values, expected_equity.values)
print('Test 3 OK: 单策略融合保持日收益率结构')

# 测试 4: 缺失策略（weights 含不存在的 key）— 与单策略等价
fused_missing = fuse_equities_by_weights(df, {'A': 1.0, 'Z': 0.0})
np.testing.assert_allclose(fused_missing.values, fused_one.values)
print('Test 4 OK: 缺失策略不干扰')

# 测试 5: 权重自动归一化
fused_unnorm = fuse_equities_by_weights(df, {'A': 1.0, 'B': 1.0})
np.testing.assert_allclose(fused_unnorm.values, fuse_equities_by_weights(df, {'A': 0.5, 'B': 0.5}).values)
print('Test 5 OK: 权重自动归一化')

# 测试 6: 空 DataFrame
empty_fused = fuse_equities_by_weights(pd.DataFrame(), {'A': 1.0})
assert empty_fused.empty
print('Test 6 OK: 空 DataFrame')

# 测试 7: 净值为 0 或负的策略
df_zero = df.copy()
df_zero.loc[df_zero.index[50], 'B'] = 0.0
fused_zero_eq = fuse_equities_by_weights(df_zero, {'A': 0.5, 'B': 0.5})
assert np.all(np.isfinite(fused_zero_eq))
print('Test 7 OK: 净值为 0 的策略不引发除零错误')

# 测试 8: 全零权重
fused_all_zero = fuse_equities_by_weights(df, {'A': 0.0, 'B': 0.0})
assert (fused_all_zero == 1.0).all()
print('Test 8 OK: 全零权重 → 全 1.0')

print('ALL TESTS PASSED')
