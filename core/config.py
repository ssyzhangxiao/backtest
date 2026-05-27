"""
回测系统统一配置。

从 core/engine/runner.py 中提取并扩展，供 PyBroker 主引擎和自研验证引擎共用。
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional


# ── 路径常量 ──
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)

# ── PyBroker 附加列 ──
PYBROKER_EXTRA_COLUMNS = (
    "open_interest",
    "is_dominant",
    "dominant_symbol",
    "prev_dominant_symbol",
    "rollover_flag",
    "rollover_signal",
    "rollover_from",
    "rollover_to",
    "rollover_cost",
    "product",
    "env_atr",
    "env_adx",
    "env_plus_di",
    "env_minus_di",
    "env_market_regime",
    "env_trend_score",
    "env_compression_score",
    "env_momentum_score",
    "env_liquidity_score",
    "env_bearish_exhaustion",
    "env_bullish_exhaustion",
    "env_weight_trend",
    "env_weight_reversal",
    "env_weight_spread",
)

# 向后兼容别名
_PYBROKER_COLUMNS = PYBROKER_EXTRA_COLUMNS

# ── 初始资金 ──
INITIAL_CASH = 1_000_000


def get_default_stress_events() -> list:
    """获取默认压力测试事件列表。"""
    return [
        {"name": "2020新冠疫情", "start": "2020-02-15", "end": "2020-03-31"},
        {"name": "2022俄乌冲突", "start": "2022-02-24", "end": "2022-04-30"},
        {"name": "2023硅谷银行", "start": "2023-03-08", "end": "2023-03-31"},
        {"name": "2024红海危机", "start": "2024-01-15", "end": "2024-03-15"},
    ]


@dataclass
class BacktestConfig:
    """回测配置（PyBroker 主引擎 + 自研验证引擎共用）。"""

    # ── 基础参数 ──
    initial_cash: float = INITIAL_CASH
    commission_rate: float = 0.0003
    slippage_rate: float = 0.0002

    # ── 样本内/外分割 ──
    in_sample_end: Optional[str] = None

    # ── 策略权重 ──
    strategy_weights: Dict[str, float] = field(default_factory=dict)

    # ── 风控 ──
    stop_loss_pct: float = 0.05
    max_position_pct: float = 0.2
    max_total_position_pct: float = 0.6

    # ── PyBroker 相关（新增） ──
    # Bootstrap 评估：对绩效指标进行重采样，计算置信区间
    pybroker_bootstrap_samples: int = 10000

    # 买入/卖出延迟：BarNumber 延迟，模拟次日成交
    pybroker_buy_delay: int = 1
    pybroker_sell_delay: int = 1

    # ── Walkforward 向前滚动分析（新增） ──
    # 训练集占比（用于参数调优）
    wf_train_ratio: float = 0.6
    # 每次前进步长占比
    wf_step_ratio: float = 0.1

    # ── 交叉验证（新增） ──
    # 是否在 PyBroker 运行后自动用自研引擎交叉验证
    cross_validate: bool = False

    # ── 组合再平衡（新增） ──
    # none: 无再平衡，权重随各策略净值变化自然漂移
    # daily: 每日收盘再平衡
    # weekly: 每周最后一个交易日再平衡（使用 Friday，或周内最后一天）
    # monthly: 每月最后一个交易日再平衡
    rebalance_frequency: str = "none"

    # ── 多策略模式（新增） ──
    # True: 信号融合模式（多策略加权信号，不切换）
    # False: 策略切换模式（市场环境→策略匹配切换）
    fusion_mode: bool = False
