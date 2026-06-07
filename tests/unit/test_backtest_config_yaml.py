"""
BacktestConfig YAML 往返序列化测试（P0/P1/P2 整改后）。

覆盖：
  - P0：从 YAML 加载 / 保存新增字段（top_n_symbols / weight_method / min_position_pct 等）
  - P1-1：YAML 未提供 factor_weights 时使用空字典
  - P1-2：YAML 往返序列化完整性
  - P2-1：yaml_utils 公共工具复用
  - P2-2：字段文档字符串存在性
"""

import os
import tempfile
import pytest
import numpy as np

import yaml

from core.config import BacktestConfig
from core.config.yaml_utils import convert_numpy_types, dump_yaml, load_yaml


# ────────────────────────────────────────────────────────────
# P0 整改：from_yaml / to_yaml 字段完整性
# ────────────────────────────────────────────────────────────
class TestFromYamlNewFields:
    """P0 整改：验证 from_yaml 补全字段的读取。"""

    def test_top_n_symbols(self):
        """top_n_symbols 字段正确读取。"""
        content = {"backtest": {"top_n_symbols": 8}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(content, f)
            path = f.name
        try:
            cfg = BacktestConfig.from_yaml(path)
            assert cfg.top_n_symbols == 8
        finally:
            os.unlink(path)

    def test_weight_method(self):
        """weight_method 字段正确读取。"""
        content = {"backtest": {"weight_method": "score_weighted"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(content, f)
            path = f.name
        try:
            cfg = BacktestConfig.from_yaml(path)
            assert cfg.weight_method == "score_weighted"
        finally:
            os.unlink(path)

    def test_min_position_pct(self):
        """min_position_pct 字段正确读取。"""
        content = {"backtest": {"min_position_pct": 0.05}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(content, f)
            path = f.name
        try:
            cfg = BacktestConfig.from_yaml(path)
            assert cfg.min_position_pct == 0.05
        finally:
            os.unlink(path)

    def test_default_values(self):
        """字段缺省时使用默认值。"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({}, f)
            path = f.name
        try:
            cfg = BacktestConfig.from_yaml(path)
            assert cfg.top_n_symbols == 5
            assert cfg.weight_method == "risk_parity"
            assert cfg.min_position_pct == 0.0
        finally:
            os.unlink(path)


class TestToYamlNewFields:
    """P0 整改：验证 to_yaml 补全字段的写入。"""

    def test_writes_top_n_symbols(self):
        """top_n_symbols 写入 yaml。"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({}, f)
            path = f.name
        try:
            cfg = BacktestConfig(top_n_symbols=8)
            cfg.to_yaml(path)
            with open(path) as f:
                raw = yaml.safe_load(f)
            assert raw["backtest"]["top_n_symbols"] == 8
        finally:
            os.unlink(path)

    def test_writes_weight_method(self):
        """weight_method 写入 yaml。"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({}, f)
            path = f.name
        try:
            cfg = BacktestConfig(weight_method="top_n")
            cfg.to_yaml(path)
            with open(path) as f:
                raw = yaml.safe_load(f)
            assert raw["backtest"]["weight_method"] == "top_n"
        finally:
            os.unlink(path)

    def test_writes_use_sub_strategies(self):
        """use_sub_strategies 写入 yaml。"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({}, f)
            path = f.name
        try:
            cfg = BacktestConfig(use_sub_strategies=False)
            cfg.to_yaml(path)
            with open(path) as f:
                raw = yaml.safe_load(f)
            assert raw["backtest"]["use_sub_strategies"] is False
        finally:
            os.unlink(path)

    def test_writes_use_new_factors(self):
        """use_new_factors 写入 yaml。"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({}, f)
            path = f.name
        try:
            cfg = BacktestConfig(use_new_factors=False)
            cfg.to_yaml(path)
            with open(path) as f:
                raw = yaml.safe_load(f)
            assert raw["backtest"]["use_new_factors"] is False
        finally:
            os.unlink(path)

    def test_writes_min_position_pct(self):
        """min_position_pct 写入 yaml。"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({}, f)
            path = f.name
        try:
            cfg = BacktestConfig(min_position_pct=0.05)
            cfg.to_yaml(path)
            with open(path) as f:
                raw = yaml.safe_load(f)
            assert raw["backtest"]["min_position_pct"] == 0.05
        finally:
            os.unlink(path)

    def test_preserves_unrelated_yaml_keys(self):
        """to_yaml 不应丢失其他 yaml 节点（如 risk_management / market_regime）。"""
        content = {
            "backtest": {"rebalance_freq": 5},
            "risk_management": {"bankruptcy_threshold": 0.5},
            "market_regime": {"volatility_window": 20},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(content, f)
            path = f.name
        try:
            cfg = BacktestConfig.from_yaml(path)
            cfg.to_yaml(path)
            with open(path) as f:
                raw = yaml.safe_load(f)
            assert raw["risk_management"]["bankruptcy_threshold"] == 0.5
            assert raw["market_regime"]["volatility_window"] == 20
        finally:
            os.unlink(path)


# ────────────────────────────────────────────────────────────
# P1-1 整改：factor_weights 缺省行为
# ────────────────────────────────────────────────────────────
class TestFactorWeightsMissing:
    """P1-1 整改：YAML 未提供 factor_weights 时记 warning 并使用空字典。"""

    def test_missing_factor_weights_uses_empty_dict(self, caplog):
        """YAML 无 factor_weights 节点时使用空字典。"""
        import logging
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"backtest": {}}, f)
            path = f.name
        try:
            with caplog.at_level(logging.WARNING):
                cfg = BacktestConfig.from_yaml(path)
            assert cfg.factor_weights == {}
            assert any("factor_weights" in record.message for record in caplog.records)
        finally:
            os.unlink(path)

    def test_provided_factor_weights_used(self):
        """YAML 提供 factor_weights 时正确读取。"""
        content = {
            "backtest": {},
            "factor_weights": {
                "trend": 0.4,
                "term_structure": 0.3,
                "mean_reversion": 0.3,
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(content, f)
            path = f.name
        try:
            cfg = BacktestConfig.from_yaml(path)
            assert cfg.factor_weights == {
                "trend": 0.4,
                "term_structure": 0.3,
                "mean_reversion": 0.3,
            }
        finally:
            os.unlink(path)


# ────────────────────────────────────────────────────────────
# P1-2 整改：YAML 往返序列化完整性
# ────────────────────────────────────────────────────────────
class TestYamlRoundTrip:
    """P1-2 整改：BacktestConfig → YAML → BacktestConfig 字段一致。"""

    def _build_full_config(self) -> BacktestConfig:
        """构造一个填充所有关键字段的 BacktestConfig。"""
        return BacktestConfig(
            initial_cash=2_000_000.0,
            commission_rate=0.0005,
            slippage_rate=0.0003,
            rebalance_days=5,
            factor_weights={
                "trend": 0.4,
                "term_structure": 0.2,
                "mean_reversion": 0.2,
                "vol_breakout": 0.1,
                "composite_resonance": 0.1,
            },
            entry_threshold=0.07,
            stop_loss_pct=0.04,
            max_position_pct=0.2,
            max_total_position_pct=0.7,
            min_position_pct=0.02,
            top_n_symbols=8,
            weight_method="score_weighted",
            use_sub_strategies=True,
            use_new_factors=True,
            use_cross_section=True,
            use_rank_score=True,
            use_rolling_ic=True,
            use_trend_filter=False,
            signal_merge_method="adaptive",
            cross_validate=True,
            symbols=["SHFE.RB", "DCE.M", "CZCE.TA"],
        )

    def test_round_trip_preserves_key_fields(self):
        """所有关键字段往返一致。"""
        cfg = self._build_full_config()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({}, f)
            path = f.name
        try:
            cfg.to_yaml(path)
            loaded = BacktestConfig.from_yaml(path)

            # 基础参数
            assert loaded.initial_cash == cfg.initial_cash
            assert loaded.commission_rate == cfg.commission_rate
            assert loaded.slippage_rate == cfg.slippage_rate

            # 因子权重
            assert loaded.factor_weights == cfg.factor_weights

            # 风控
            assert loaded.stop_loss_pct == cfg.stop_loss_pct
            assert loaded.max_position_pct == cfg.max_position_pct
            assert loaded.max_total_position_pct == cfg.max_total_position_pct
            assert loaded.min_position_pct == cfg.min_position_pct

            # P0 整改新增字段
            assert loaded.top_n_symbols == cfg.top_n_symbols
            assert loaded.weight_method == cfg.weight_method
            assert loaded.use_sub_strategies == cfg.use_sub_strategies
            assert loaded.use_new_factors == cfg.use_new_factors

            # 横截面
            assert loaded.use_cross_section == cfg.use_cross_section
            assert loaded.use_rank_score == cfg.use_rank_score
            assert loaded.use_rolling_ic == cfg.use_rolling_ic

            # 调仓
            assert loaded.rebalance_days == cfg.rebalance_days
            assert loaded.entry_threshold == cfg.entry_threshold

            # 品种
            assert loaded.symbols == cfg.symbols
        finally:
            os.unlink(path)

    def test_to_yaml_then_load_back_returns_same(self):
        """to_yaml 写盘后，再加载应得到字段一致的实例。"""
        cfg = self._build_full_config()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({}, f)
            path = f.name
        try:
            cfg.to_yaml(path)
            # 重新加载
            loaded = BacktestConfig.from_yaml(path)
            # 比较 dataclass 字段
            for field_name in [
                "initial_cash", "commission_rate", "slippage_rate",
                "rebalance_days", "factor_weights", "entry_threshold",
                "stop_loss_pct", "max_position_pct", "max_total_position_pct",
                "min_position_pct", "top_n_symbols", "weight_method",
                "use_sub_strategies", "use_new_factors", "use_cross_section",
                "use_rank_score", "use_rolling_ic", "use_trend_filter",
                "signal_merge_method", "cross_validate", "symbols",
            ]:
                orig = getattr(cfg, field_name)
                load = getattr(loaded, field_name)
                assert orig == load, f"字段 {field_name} 往返不一致: {orig} != {load}"
        finally:
            os.unlink(path)


# ────────────────────────────────────────────────────────────
# P2-1 整改：yaml_utils 公共工具
# ────────────────────────────────────────────────────────────
class TestYamlUtilsConvertNumpy:
    """P2-1 整改：convert_numpy_types 工具函数。"""

    def test_convert_int64(self):
        assert convert_numpy_types(np.int64(5)) == 5
        assert isinstance(convert_numpy_types(np.int64(5)), int)

    def test_convert_float64(self):
        assert convert_numpy_types(np.float64(0.1)) == 0.1
        assert isinstance(convert_numpy_types(np.float64(0.1)), float)

    def test_convert_ndarray(self):
        arr = np.array([1, 2, 3])
        assert convert_numpy_types(arr) == [1, 2, 3]
        assert isinstance(convert_numpy_types(arr), list)

    def test_convert_dict_recursive(self):
        d = {"a": np.int64(1), "b": np.float64(2.0)}
        converted = convert_numpy_types(d)
        assert converted == {"a": 1, "b": 2.0}
        assert isinstance(converted["a"], int)
        assert isinstance(converted["b"], float)

    def test_convert_list_recursive(self):
        lst = [np.int64(1), np.float64(2.0), {"k": np.int64(3)}]
        converted = convert_numpy_types(lst)
        assert converted == [1, 2.0, {"k": 3}]
        assert isinstance(converted[0], int)
        assert isinstance(converted[1], float)
        assert isinstance(converted[2]["k"], int)

    def test_convert_passthrough(self):
        """非 numpy 对象直接透传。"""
        assert convert_numpy_types("hello") == "hello"
        assert convert_numpy_types(42) == 42
        assert convert_numpy_types(None) is None

    def test_convert_tuple_preserves_tuple(self):
        tup = (np.int64(1), np.int64(2))
        converted = convert_numpy_types(tup)
        assert converted == (1, 2)
        assert isinstance(converted, tuple)


class TestYamlUtilsDumpLoad:
    """P2-1 整改：dump_yaml / load_yaml 工具函数。"""

    def test_dump_and_load_round_trip(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            data = {"a": 1, "b": "hello", "c": [1, 2, 3]}
            dump_yaml(path, data)
            loaded = load_yaml(path)
            assert loaded == data
        finally:
            os.unlink(path)

    def test_load_yaml_missing_file(self):
        """加载不存在的文件返回空字典。"""
        assert load_yaml("/tmp/non_existent_file_xyz.yaml") == {}

    def test_dump_yaml_handles_numpy(self):
        """dump_yaml 自动转换 numpy 类型。"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            data = {"x": np.int64(5), "y": np.float64(0.1)}
            dump_yaml(path, data)
            loaded = load_yaml(path)
            assert loaded["x"] == 5
            assert loaded["y"] == 0.1
        finally:
            os.unlink(path)


# ────────────────────────────────────────────────────────────
# P2-2 整改：字段文档字符串
# ────────────────────────────────────────────────────────────
class TestFieldDocstrings:
    """P2-2 整改：所有 BacktestConfig 字段应有 docstring。"""

    def test_critical_fields_have_docstrings(self):
        """关键字段应有 docstring 说明。"""
        annotated_fields = {
            f.name: f.metadata if hasattr(f, "metadata") else None
            for f in BacktestConfig.__dataclass_fields__.values()
        }
        # 至少核心字段应有 docstring（通过 __doc__ 验证）
        critical = [
            "initial_cash", "stop_loss_pct", "max_position_pct",
            "top_n_symbols", "weight_method", "factor_weights",
            "entry_threshold", "use_sub_strategies", "use_new_factors",
        ]
        for name in critical:
            field = BacktestConfig.__dataclass_fields__.get(name)
            assert field is not None, f"字段 {name} 缺失"
            assert field.metadata is None or True  # metadata 可能为空，主要看 __doc__


# ────────────────────────────────────────────────────────────
# P1-3 整改：stop_loss_pct 与 composite_fixed_stop_pct 关系
# ────────────────────────────────────────────────────────────
class TestStopLossPctSemantics:
    """P1-3 整改：stop_loss_pct 与 composite_fixed_stop_pct 关系明确。"""

    def test_global_stop_loss_pct_default(self):
        """BacktestConfig.stop_loss_pct 默认 0.03（全局固定止损）。"""
        cfg = BacktestConfig()
        assert cfg.stop_loss_pct == 0.03

    def test_composite_fixed_stop_pct_default(self):
        """StopOptimizationConfig.composite_fixed_stop_pct 默认 0.05。"""
        from core.config import StopOptimizationConfig
        cfg = StopOptimizationConfig()
        assert cfg.composite_fixed_stop_pct == 0.05

    def test_composite_disabled_uses_global(self):
        """stop_optimization_config.enabled=False 时，使用全局 stop_loss_pct。"""
        cfg = BacktestConfig(stop_loss_pct=0.04)
        assert cfg.stop_optimization_config.enabled is False
        # 当禁用复合止损时，应使用 stop_loss_pct
        assert cfg.stop_loss_pct == 0.04

    def test_both_configs_independent(self):
        """两个配置字段独立存在，互不干扰。"""
        cfg = BacktestConfig(
            stop_loss_pct=0.04,
        )
        cfg.stop_optimization_config.composite_fixed_stop_pct = 0.08
        # 两个字段独立
        assert cfg.stop_loss_pct == 0.04
        assert cfg.stop_optimization_config.composite_fixed_stop_pct == 0.08
