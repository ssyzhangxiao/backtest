"""
测试 e1_e5.py 模块（2026-06-11 整改：适配 P2 公共系统迁移）。

WeightedSignalFusion / _calculate_rolling_volatility / _calculate_risk_parity_weights
已被提取到 runner/common/portfolio_utils，本测试改为测：
  - e1_e5._run_weighted_fusion：E2/E3 通用加权融合
  - e1_e5.PortfolioResult：TypedDict 字段约束
  - runner.common.portfolio_utils.fuse_equities_by_weights / calculate_risk_parity_fusion
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ══════════════════════════════════════════════════════════════════════════════
# PortfolioResult TypedDict 字段约束
# ══════════════════════════════════════════════════════════════════════════════
class TestPortfolioResult:
    """PortfolioResult 必含 metrics + equity 字段。"""

    def test_required_keys(self):
        from runner.backtest.experiments.e1_e5 import PortfolioResult

        # TypedDict 在 runtime 是普通 dict，验证构造后键可访问
        result: PortfolioResult = {
            "metrics": {"sharpe": 0.5},
            "equity": pd.DataFrame(
                {"date": pd.date_range("2020-01-01", periods=5), "equity": [1.0] * 5}
            ),
        }
        assert "sharpe" in result["metrics"]
        assert len(result["equity"]) == 5


# ══════════════════════════════════════════════════════════════════════════════
# 公共系统 fuse_equities_by_weights（WeightedSignalFusion 替代）
# 接口契约（2026-06-11）：输入策略净值 DataFrame（首值=1.0），输出融合净值 Series（首值=1.0）
# ══════════════════════════════════════════════════════════════════════════════
class TestFuseEquitiesByWeights:
    """fuse_equities_by_weights：公共系统替代 WeightedSignalFusion。"""

    def test_equal_weights(self):
        from runner.common.portfolio_utils import fuse_equities_by_weights

        # 输入策略净值（首值=1.0），不是 raw score
        equity = pd.DataFrame(
            {
                "a": [1.0, 1.10, 1.21],  # 收益 0.10, 0.10
                "b": [1.0, 1.20, 1.32],  # 收益 0.20, 0.10
            }
        )
        out = fuse_equities_by_weights(equity, {"a": 0.5, "b": 0.5})
        # 融合净值首值固定=1.0（接口契约）
        assert out.iloc[0] == 1.0
        # 长度对齐
        assert len(out) == 3
        # 等权融合日收益：d1=(0.10+0.20)/2=0.15, d2=(0.10+0.10)/2=0.10
        # 净值: 1.0 → 1.15 → 1.265
        assert abs(out.iloc[1] - 1.15) < 1e-6
        assert abs(out.iloc[2] - 1.265) < 1e-6

    def test_unequal_weights(self):
        from runner.common.portfolio_utils import fuse_equities_by_weights

        equity = pd.DataFrame(
            {
                "a": [1.0, 1.10, 1.21],
                "b": [1.0, 1.20, 1.32],
            }
        )
        out = fuse_equities_by_weights(equity, {"a": 0.3, "b": 0.7})
        # 首值=1.0
        assert out.iloc[0] == 1.0
        # 加权日收益 = 0.3*0.10 + 0.7*0.20 = 0.17
        # 第 1 日净值 = 1.17
        assert abs(out.iloc[1] - 1.17) < 1e-6

    def test_weights_sum_not_one_normalized(self):
        """权重 sum != 1 时应归一化。"""
        from runner.common.portfolio_utils import fuse_equities_by_weights

        equity = pd.DataFrame(
            {
                "a": [1.0, 1.10],
                "b": [1.0, 1.20],
            }
        )
        out = fuse_equities_by_weights(
            equity, {"a": 1.0, "b": 1.0}
        )  # sum=2 → 归一为 0.5/0.5
        # 归一化后等权，结果同 test_equal_weights
        assert abs(out.iloc[1] - 1.15) < 1e-6

    def test_all_zero_weights_returns_ones(self):
        """全部权重为 0 → 返回全 1.0 净值。"""
        from runner.common.portfolio_utils import fuse_equities_by_weights

        equity = pd.DataFrame(
            {
                "a": [1.0, 1.10, 1.21],
                "b": [1.0, 1.20, 1.32],
            }
        )
        out = fuse_equities_by_weights(equity, {"a": 0.0, "b": 0.0})
        # 全部 1.0
        assert (out == 1.0).all()


# ══════════════════════════════════════════════════════════════════════════════
# 公共系统 calculate_risk_parity_fusion（_calculate_risk_parity_weights 替代）
# 接口契约（2026-06-11）：输入策略收益率时序 dict，输出 {策略: 权重} 字典
# ══════════════════════════════════════════════════════════════════════════════
class TestCalculateRiskParityFusion:
    """calculate_risk_parity_fusion：公共系统替代 _calculate_risk_parity_weights。"""

    def test_equal_volatility_inverse_weights(self):
        from runner.common.portfolio_utils import calculate_risk_parity_fusion

        # 输入是收益率时序，不是净值
        import numpy as np

        returns = {
            "a": pd.Series(np.full(120, 0.001)),  # 恒定收益
            "b": pd.Series(np.full(120, 0.001)),  # 恒定收益
        }
        weights = calculate_risk_parity_fusion(returns, window=60)
        # 等波动率时权重应近似相等（a:b ≈ 0.5:0.5）
        assert abs(weights["a"] - 0.5) < 0.05
        assert abs(weights["b"] - 0.5) < 0.05
        # 权重和=1
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_high_vol_strategy_gets_lower_weight(self):
        """高波动率策略应获得更低权重。"""
        from runner.common.portfolio_utils import calculate_risk_parity_fusion
        import numpy as np

        np.random.seed(0)
        returns = {
            "stable": pd.Series(np.random.normal(0, 0.005, 200)),
            "volatile": pd.Series(np.random.normal(0, 0.05, 200)),
        }
        weights = calculate_risk_parity_fusion(returns, window=60)
        assert weights["stable"] > weights["volatile"]
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_empty_returns_returns_empty_dict(self):
        """空输入应返回空字典。"""
        from runner.common.portfolio_utils import calculate_risk_parity_fusion

        assert calculate_risk_parity_fusion({}) == {}
