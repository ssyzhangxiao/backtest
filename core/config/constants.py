"""
回测系统常量定义。

集中管理全局常量，避免硬编码散落各处。
"""

import os
from typing import Dict

# 数据目录
DATA_DIR: str = os.environ.get("BACKTEST_DATA_DIR", "data")

# PyBroker 额外数据列
PYBROKER_EXTRA_COLUMNS: list = [
    "open_interest",
    "settle",
    "prev_settle",
    "prev_close",
]

# 初始资金
INITIAL_CASH: float = 1_000_000.0

# 5子策略默认因子权重（等权）
DEFAULT_FACTOR_WEIGHTS: Dict[str, float] = {
    "trend": 0.20,
    "term_structure": 0.20,
    "mean_reversion": 0.20,
    "vol_breakout": 0.20,
    "composite_resonance": 0.20,
}


def get_default_stress_events() -> list:
    """获取默认压力测试事件列表。"""
    return [
        {"name": "2020_oil_crash", "start": "2020-01-06", "end": "2020-04-21"},
        {"name": "covid_recovery", "start": "2020-03-23", "end": "2020-06-08"},
        {"name": "2022_rate_hike", "start": "2022-01-03", "end": "2022-10-12"},
    ]
