"""
回测系统常量定义。

供 PyBroker 主引擎和自研验证引擎共用。
"""

import os

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data"
)

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
)

INITIAL_CASH = 1_000_000

DEFAULT_FACTOR_WEIGHTS = {
    "ts_momentum": 0.25,
    "roll_yield": 0.25,
    "alpha019": 0.25,
    "alpha032": 0.25,
}


def get_default_stress_events() -> list:
    """获取默认压力测试事件列表。"""
    return [
        {"name": "2020新冠疫情", "start": "2020-02-15", "end": "2020-03-31"},
        {"name": "2022俄乌冲突", "start": "2022-02-24", "end": "2022-04-30"},
        {"name": "2023硅谷银行", "start": "2023-03-08", "end": "2023-03-31"},
        {"name": "2024红海危机", "start": "2024-01-15", "end": "2024-03-15"},
    ]
