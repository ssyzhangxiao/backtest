"""conftest smoke test"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_conftest_fixtures_exist(synth_config, synth_ds, synth_lib, synth_all_df):
    """conftest fixture 全部可用。"""
    assert synth_config.factor_weights != {}, "factor_weights 必须非空"
    assert len(synth_config.symbols) == 6
    assert synth_all_df["symbol"].nunique() == 6
    assert synth_all_df["date"].nunique() >= 250
    assert synth_ds is not None
    assert synth_lib is not None


def test_conftest_synth_config_default_rebalance_days(synth_config):
    """synth_config.rebalance_days 必须 == 5。"""
    assert synth_config.rebalance_days == 5
    assert synth_config.stop_loss_pct == 0.05
    assert synth_config.max_total_position_pct == 0.8
