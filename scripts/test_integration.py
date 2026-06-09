"""
整合验证脚本。

验证:
1. 因子函数导入正常
2. 策略类导入正常（新命名和旧命名都正常）
3. 策略注册表工作正常
4. 因子计算功能正常
"""
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("=" * 80)
print("开始整合验证")
print("=" * 80)

print("\n[1] 验证 core.factors 导入...")
try:
    from core.factors import (
        compute_ts_momentum,
        compute_roll_yield,
        compute_alpha019,
        compute_alpha032,
        compute_factor_scores_from_ohlcv,
    )
    print("  ✓ 因子计算函数导入成功")
except Exception as e:
    print(f"  ✗ 导入失败: {e}")
    sys.exit(1)

print("\n[2] 验证 core.strategies 导入...")
try:
    # 新命名导入
    from core.strategies.strategy_ts_momentum import TSMomentumStrategy
    from core.strategies.strategy_roll_yield import RollYieldStrategy
    from core.strategies.strategy_alpha019 import Alpha019Strategy
    from core.strategies.strategy_alpha032 import Alpha032Strategy
    print("  ✓ 新命名策略导入成功")
    
    # 旧命名导入（向后兼容）
    from core.strategies.ts_momentum import TSMomentumStrategy as TSMomentumStrategy_old
    from core.strategies.roll_yield import RollYieldStrategy as RollYieldStrategy_old
    from core.strategies.alpha019 import Alpha019Strategy as Alpha019Strategy_old
    from core.strategies.alpha032 import Alpha032Strategy as Alpha032Strategy_old
    print("  ✓ 旧命名策略导入成功（向后兼容）")
    
    # 直接从包导入
    from core.strategies import (
        TSMomentumStrategy as TSMomentumStrategy_pkg,
        RollYieldStrategy as RollYieldStrategy_pkg,
        Alpha019Strategy as Alpha019Strategy_pkg,
        Alpha032Strategy as Alpha032Strategy_pkg,
    )
    print("  ✓ 从包导入成功")
    
except Exception as e:
    print(f"  ✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[3] 验证策略注册表...")
try:
    from core.strategy_registry import (
        get_strategy_class,
        create_strategy,
        STRATEGY_REGISTRY,
        StrategyLibrary,
    )
    
    print(f"  ✓ 注册表导入成功，注册策略数: {len(STRATEGY_REGISTRY)}")
    
    # 测试创建策略
    for name in ["ts_momentum", "roll_yield", "alpha019", "alpha032"]:
        strategy = create_strategy(name)
        print(f"  ✓ 策略 '{name}' 创建成功")
    
    # 测试 StrategyLibrary
    library = StrategyLibrary()
    print(f"  ✓ StrategyLibrary 初始化成功，策略档案数: {len(library.list_all())}")
    
except Exception as e:
    print(f"  ✗ 注册表验证失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[4] 验证因子计算...")
try:
    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=300, freq='D')
    close = 100 + np.cumsum(np.random.randn(300))
    high = close + np.abs(np.random.randn(300)) * 2
    low = close - np.abs(np.random.randn(300)) * 2
    volume = np.random.randint(1000, 10000, size=300)
    
    ohlcv = pd.DataFrame({
        'date': dates,
        'close': close,
        'high': high,
        'low': low,
        'volume': volume,
    }).set_index('date', drop=False)
    
    # 测试单个因子计算
    close_series = pd.Series(close, index=dates)
    ts_mom = compute_ts_momentum(close_series, window=20)
    print(f"  ✓ ts_momentum 计算成功，值范围: [{np.nanmin(ts_mom):.4f}, {np.nanmax(ts_mom):.4f}]")
    
    roll_yld = compute_roll_yield(close_series, lookback=20)
    print(f"  ✓ roll_yield 计算成功，值范围: [{np.nanmin(roll_yld):.4f}, {np.nanmax(roll_yld):.4f}]")
    
    alpha019_val = compute_alpha019(close_series, 7, 250)
    print(f"  ✓ alpha019 计算成功，值范围: [{np.nanmin(alpha019_val):.4f}, {np.nanmax(alpha019_val):.4f}]")
    
    alpha032_val = compute_alpha032(
        close_series,
        pd.Series(high, index=dates),
        pd.Series(low, index=dates),
        pd.Series(volume, index=dates),
        7, 230
    )
    print(f"  ✓ alpha032 计算成功，值范围: [{np.nanmin(alpha032_val):.4f}, {np.nanmax(alpha032_val):.4f}]")
    
    # 测试 compute_factor_scores_from_ohlcv
    scores = compute_factor_scores_from_ohlcv(ohlcv)
    print(f"  ✓ compute_factor_scores_from_ohlcv 计算成功，列: {list(scores.columns)}")
    
except Exception as e:
    print(f"  ✗ 因子计算验证失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✓ 整合验证全部通过！")
print("=" * 80)
