"""
回测系统统一配置。

供 PyBroker 主引擎和自研验证引擎共用。
因子打分调仓模式：多因子综合得分决定持仓方向和仓位。
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import yaml


DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
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

DEFAULT_FACTOR_WEIGHTS: Dict[str, float] = {
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


@dataclass
class BacktestConfig:
    """回测配置（PyBroker 主引擎 + 自研验证引擎共用）。"""

    # ── 基础参数 ──
    initial_cash: float = INITIAL_CASH
    commission_rate: float = 0.0001
    slippage_rate: float = 0.0001

    # ── 样本内/外分割 ──
    in_sample_end: Optional[str] = None

    # ── 因子打分调仓 ──
    rebalance_days: int = 3
    factor_weights: Dict[str, float] = field(default_factory=lambda: DEFAULT_FACTOR_WEIGHTS.copy())
    entry_threshold: float = 0.05
    score_scale: float = 1.0
    stop_loss_cooldown: int = 1

    # ── 风控 ──
    stop_loss_pct: float = 0.03
    max_position_pct: float = 0.15
    max_total_position_pct: float = 0.6
    min_position_pct: float = 0.0

    # ── 横截面标准化与排名叠加 ──
    use_cross_section: bool = True
    use_rank_score: bool = True
    use_rolling_ic: bool = True
    top_n_symbols: int = 5

    # ── PyBroker 相关 ──
    pybroker_bootstrap_samples: int = 10000
    pybroker_buy_delay: int = 1
    pybroker_sell_delay: int = 1

    # ── Walkforward 向前滚动分析 ──
    wf_train_bars: int = 252
    wf_test_bars: int = 63
    wf_step_bars: int = 21
    wf_train_ratio: float = 0.6
    wf_step_ratio: float = 0.1

    # ── 交叉验证 ──
    cross_validate: bool = False

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "BacktestConfig":
        """
        从 YAML 文件加载配置。

        YAML 结构示例:
          backtest:
            start_date: "2020-01-01"
            end_date: "2023-12-31"
            initial_capital: 1000000
            rebalance_freq: 3
            commission: 0.0002
            slippage: 0.001
          factor_weights:
            ts_momentum: 0.25
            roll_yield: 0.25
            alpha019: 0.25
            alpha032: 0.25

        Args:
            path: YAML 文件路径

        Returns:
            BacktestConfig 实例
        """
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        bt = raw.get("backtest", {})
        fw = raw.get("factor_weights", {})

        return cls(
            initial_cash=float(bt.get("initial_capital", INITIAL_CASH)),
            commission_rate=float(bt.get("commission", 0.0001)),
            slippage_rate=float(bt.get("slippage", 0.0001)),
            rebalance_days=int(bt.get("rebalance_freq", 3)),
            factor_weights=fw if fw else DEFAULT_FACTOR_WEIGHTS.copy(),
            entry_threshold=float(bt.get("entry_threshold", 0.05)),
            stop_loss_pct=float(bt.get("stop_loss_pct", 0.03)),
            max_position_pct=float(bt.get("max_position_pct", 0.15)),
            max_total_position_pct=float(bt.get("max_total_position_pct", 0.6)),
            min_position_pct=float(bt.get("min_position_pct", 0.0)),
            use_cross_section=bool(bt.get("use_cross_section", True)),
            use_rank_score=bool(bt.get("use_rank_score", True)),
            use_rolling_ic=bool(bt.get("use_rolling_ic", True)),
            top_n_symbols=int(bt.get("top_n_symbols", 5)),
        )
