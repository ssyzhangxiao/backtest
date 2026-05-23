import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

_PYBROKER_COLUMNS = (
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


def get_default_stress_events() -> list:
    return [
        {"name": "2020新冠疫情", "start": "2020-02-15", "end": "2020-03-31"},
        {"name": "2022俄乌冲突", "start": "2022-02-24", "end": "2022-04-30"},
        {"name": "2023硅谷银行", "start": "2023-03-08", "end": "2023-03-31"},
        {"name": "2024红海危机", "start": "2024-01-15", "end": "2024-03-15"},
    ]