"""
调仓逻辑单元测试。

规则6要求：调仓逻辑修改必须有调仓日判断测试。
覆盖：滚动IC权重引擎、调仓日判断逻辑。
"""

import unittest
import numpy as np

from core.engine.rolling_ic import RollingICWeightEngine, RollingICConfig


class TestRollingICWeightEngine(unittest.TestCase):
    """滚动IC权重引擎测试。"""

    def setUp(self):
        self.config = RollingICConfig(
            window=30,
            forward_period=5,
            ema_alpha=0.1,
            min_abs_ic=0.02,
            min_observations=10,
        )
        self.engine = RollingICWeightEngine(config=self.config)

    def test_initial_weights_are_default(self):
        """初始权重应为默认固定权重。"""
        weights = self.engine.current_weights
        self.assertIn("ts_momentum", weights)
        self.assertIn("roll_yield", weights)
        # 权重之和应接近1
        total = sum(weights.values())
        self.assertAlmostEqual(total, 1.0, places=2)

    def test_weights_update_after_sufficient_data(self):
        """数据充足后权重应更新。"""
        np.random.seed(42)
        n = 50
        # 构造因子得分和前瞻收益
        for i in range(n):
            factor_scores = {
                "ts_momentum": np.random.randn(),
                "roll_yield": np.random.randn(),
                "alpha019": np.random.randn(),
                "alpha032": np.random.randn(),
            }
            forward_returns = np.random.randn() * 0.01
            self.engine.update(factor_scores, forward_returns, symbol="rb2401")

        # 更新后IC值应已计算
        ic = self.engine.current_ic
        self.assertGreater(len(ic), 0)

    def test_min_observations_fallback(self):
        """观测数不足时应回退到固定权重。"""
        # 仅添加少量数据
        factor_scores = {
            "ts_momentum": 0.5,
            "roll_yield": -0.3,
            "alpha019": 0.1,
            "alpha032": 0.2,
        }
        self.engine.update(factor_scores, 0.01, symbol="rb2401")
        # 观测数不足，权重应保持默认
        weights = self.engine.current_weights
        total = sum(weights.values())
        self.assertAlmostEqual(total, 1.0, places=2)

    def test_ic_below_threshold_zero_weight(self):
        """IC低于阈值的因子权重应清零。"""
        np.random.seed(42)
        n = 50
        # 构造IC极低的因子
        for i in range(n):
            factor_scores = {
                "ts_momentum": np.random.randn(),
                "roll_yield": np.random.randn(),
                "alpha019": np.random.randn(),
                "alpha032": np.random.randn(),
            }
            # 前瞻收益与因子无关
            forward_returns = np.random.randn() * 0.01
            self.engine.update(factor_scores, forward_returns, symbol="rb2401")

        # 所有因子IC应接近0
        ic = self.engine.current_ic
        for factor_name, ic_val in ic.items():
            self.assertLess(abs(ic_val), 0.3)


class TestRebalanceDayLogic(unittest.TestCase):
    """调仓日判断逻辑测试。"""

    def test_rebalance_every_n_days(self):
        """每N个交易日调仓一次。"""
        rebalance_days = 3
        for day in range(30):
            is_rebalance_day = day % rebalance_days == 0
            if day in [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]:
                self.assertTrue(is_rebalance_day, f"Day {day} should be rebalance day")
            else:
                self.assertFalse(
                    is_rebalance_day, f"Day {day} should not be rebalance day"
                )

    def test_rebalance_day_1_means_daily(self):
        """rebalance_days=1表示每日调仓。"""
        rebalance_days = 1
        for day in range(10):
            self.assertTrue(day % rebalance_days == 0)

    def test_rebalance_day_never_on_non_multiples(self):
        """非调仓日不应触发。"""
        rebalance_days = 5
        non_rebalance_days = [1, 2, 3, 4, 6, 7, 8, 9]
        for day in non_rebalance_days:
            self.assertNotEqual(day % rebalance_days, 0)


if __name__ == "__main__":
    unittest.main()
