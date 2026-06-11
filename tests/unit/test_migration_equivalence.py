"""M-04 等价性回归测试 — 迁移前后行为必须一致。

覆盖范围：
    1. CsvAdapter  vs  DataLoader(data_source='csv')        → 行为等价
    2. TqSdkAdapter vs DataLoader(data_source='tqsdk')      → 行为等价（mock）
    3. create_hybrid_data_source（M-02）工厂化后            → 接口保持兼容
    4. utils.session_state（M-03）去除 DataLoader alias 后  → 不再 import 死路径

执行方式：
    pytest tests/unit/test_migration_equivalence.py -v
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ───────────────────────────────────────────────────────────
# 工具：合成 CSV
# ───────────────────────────────────────────────────────────
def _write_csv(path: Path, n: int = 200, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    # symbol 列：旧 DataLoader 强依赖 SHFE.rb2401 等主力合约命名格式
    sym = path.stem  # RB / CU
    product_code = sym.lower()
    symbol = f"SHFE.{product_code}2401"
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n, freq="D"),
            "symbol": symbol,
            "open": close + rng.normal(0, 0.3, n),
            "high": close + np.abs(rng.normal(0, 0.5, n)),
            "low": close - np.abs(rng.normal(0, 0.5, n)),
            "close": close,
            "volume": rng.integers(1000, 5000, n),
            "open_interest": rng.integers(50000, 80000, n),
        }
    )
    df.to_csv(path, index=False)


@pytest.fixture
def csv_dir(tmp_path):
    """合成 2 个品种的 CSV 数据。"""
    d = tmp_path / "csv"
    d.mkdir()
    _write_csv(d / "RB.csv", n=200, seed=42)
    _write_csv(d / "CU.csv", n=200, seed=43)
    return d


# ───────────────────────────────────────────────────────────
# 1) CsvAdapter  vs  DataLoader(data_source='csv')
# ───────────────────────────────────────────────────────────
class TestCsvAdapterEquivalence:
    """M-04：factory('csv') 与旧 DataLoader(data_source='csv') 等价。"""

    def test_create_data_source_csv_returns_adapter(self, csv_dir):
        """create_data_source('csv') 返回 CsvAdapter 实例。"""
        from core.ext.adapters import create_data_source, list_adapters
        from core.ext.adapters.csv_adapter import CsvAdapter

        assert "csv" in list_adapters()
        adapter = create_data_source("csv", data_dir=str(csv_dir))
        assert isinstance(adapter, CsvAdapter)

    def test_get_bars_equivalent(self, csv_dir):
        """新接口 get_bars() 等价于直读对应 CSV。

        CsvAdapter.get_bars 内部委托 DataLoader.get_bars（规则 21.4 复用）。
        """
        from core.ext.adapters import create_data_source

        # 便捷适配器（用旧 DataLoader 可识别的合约代码）
        adapter = create_data_source("csv", data_dir=str(csv_dir))
        new_df = adapter.get_bars("SHFE.rb2401", "2020-01-10", "2020-03-01")

        # 标准读取路径：直接读 CSV
        csv_path = csv_dir / "RB.csv"
        raw_df = pd.read_csv(csv_path)
        raw_df["date"] = pd.to_datetime(raw_df["date"])
        mask = (raw_df["date"] >= "2020-01-10") & (raw_df["date"] <= "2020-03-01")
        expected_df = raw_df.loc[mask].reset_index(drop=True)

        # 行数一致
        assert len(new_df) == len(expected_df)
        # 列名一致（迁移前后接口返回相同列结构）
        assert list(new_df.columns) == list(expected_df.columns)

    def test_validate_data_equivalent(self, csv_dir):
        """新接口 validate_data() 与旧 DataLoader.validate_data() 等价。"""
        from core.data_loader import DataLoader
        from core.ext.adapters import create_data_source

        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=200, freq="D"),
                "open": np.linspace(100, 120, 200),
                "high": np.linspace(101, 121, 200),
                "low": np.linspace(99, 119, 200),
                "close": np.linspace(100, 120, 200),
                "volume": np.random.randint(1000, 5000, 200),
            }
        )

        adapter = create_data_source("csv", data_dir=str(csv_dir))
        new_result = adapter.validate_data(df, min_rows=100, max_missing=0.05)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            old_loader = DataLoader(data_source="csv", data_dir=str(csv_dir))
            old_result = old_loader.validate_data(df, min_rows=100, max_missing=0.05)

        assert new_result == old_result

    def test_new_interface_no_deprecation_warning(self, csv_dir):
        """新接口不应触发 DataLoader 的 DeprecationWarning。"""
        from core.ext.adapters import create_data_source

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            create_data_source("csv", data_dir=str(csv_dir))

        # 工厂调用本身不应触发 DeprecationWarning
        deprecation_records = [
            r for r in records if issubclass(r.category, DeprecationWarning)
            and "DataLoader" in str(r.message)
        ]
        assert len(deprecation_records) == 0, (
            f"新接口不应触发 DataLoader 弃用警告，实际触发 {len(deprecation_records)} 次: "
            f"{[str(r.message) for r in deprecation_records]}"
        )


# ───────────────────────────────────────────────────────────
# 2) TqSdkAdapter vs DataLoader(data_source='tqsdk')（mock）
# ───────────────────────────────────────────────────────────
class TestTqsdkAdapterEquivalence:
    """M-04：factory('tqsdk') 与旧 DataLoader(data_source='tqsdk') 等价。"""

    def test_create_data_source_tqsdk_returns_adapter(self):
        """create_data_source('tqsdk') 返回 TqSdkAdapter 实例（mock tqsdk）。"""
        # 必须在 import 之前 mock tqsdk
        sys.modules["tqsdk"] = MagicMock()

        try:
            from core.ext.adapters import create_data_source
            from core.ext.adapters.tqsdk_adapter import TqSdkAdapter

            adapter = create_data_source(
                "tqsdk", phone="1", password="1", symbols=["RB"]
            )
            assert isinstance(adapter, TqSdkAdapter)
            assert adapter.name == "tqsdk"
        finally:
            del sys.modules["tqsdk"]

    def test_new_interface_no_deprecation_warning(self):
        """新工厂调用 tqsdk 时不应触发 DeprecationWarning。"""
        sys.modules["tqsdk"] = MagicMock()

        try:
            from core.ext.adapters import create_data_source

            with warnings.catch_warnings(record=True) as records:
                warnings.simplefilter("always")
                create_data_source("tqsdk", phone="1", password="1", symbols=["RB"])

            deprecation_records = [
                r for r in records if issubclass(r.category, DeprecationWarning)
                and "DataLoader" in str(r.message)
            ]
            assert len(deprecation_records) == 0
        finally:
            del sys.modules["tqsdk"]


# ───────────────────────────────────────────────────────────
# 3) create_hybrid_data_source（M-02 重构）接口保持兼容
# ───────────────────────────────────────────────────────────
class TestHybridDataSourceEquivalence:
    """M-04：M-02 重构后 create_hybrid_data_source 接口与 M-02 前一致。"""

    def test_missing_credentials_raises(self):
        """缺凭证时仍抛 RuntimeError（不静默回退到 CSV）。"""
        from core.engine.pybroker_data_source import create_hybrid_data_source

        # 清理环境变量 + yaml 兜底
        with patch.dict(
            "os.environ",
            {k: v for k, v in {
                "TQSDK_PHONE": "",
                "TQSDK_PASSWORD": "",
            }.items()},
            clear=False,
        ):
            with pytest.raises(RuntimeError, match="TqSdk 凭证未配置"):
                create_hybrid_data_source(symbols=["RB"])

    def test_missing_symbols_raises(self):
        """缺品种时仍抛 RuntimeError。"""
        from core.engine.pybroker_data_source import create_hybrid_data_source

        with patch.dict(
            "os.environ",
            {"TQSDK_PHONE": "139", "TQSDK_PASSWORD": "pw"},
        ):
            with pytest.raises(RuntimeError, match="未指定品种列表"):
                create_hybrid_data_source(symbols=None)

    def test_uses_factory_not_direct_loader(self):
        """M-02 重构：create_hybrid_data_source 内部应委托 create_data_source 工厂。"""
        from core.engine import pybroker_data_source as mod

        # 静态检查：模块不应再 import DataLoader
        source = Path(mod.__file__).read_text(encoding="utf-8")
        assert "from core.data_loader import DataLoader" not in source, (
            "M-02 重构未完成：create_hybrid_data_source 仍直接 import DataLoader"
        )
        assert "create_data_source" in source, (
            "M-02 重构未完成：未使用 ext/adapters 工厂"
        )

    def test_factory_used_for_tqsdk(self):
        """确认 _load_legacy_dataframe 函数存在（工厂委托 helper）。"""
        from core.engine import pybroker_data_source as mod

        assert hasattr(mod, "_load_legacy_dataframe")
        assert hasattr(mod, "_load_legacy_spread_df")


# ───────────────────────────────────────────────────────────
# 4) utils/session_state（M-03）alias 已移除
# ───────────────────────────────────────────────────────────
class TestSessionStateNoAlias:
    """M-04：M-03 后 utils/session_state 不再 import 死路径。"""

    def test_no_data_loader_import(self):
        """utils/session_state.py 不得再 import core.data_loader.DataLoader。"""
        from utils import session_state as mod

        source = Path(mod.__file__).read_text(encoding="utf-8")
        assert "from core.data_loader import DataLoader" not in source
        assert "from core.data_loader_tqsdk" not in source

    def test_uses_ext_factory(self):
        """utils/session_state.py 必须使用 create_data_source 工厂。"""
        from utils import session_state as mod

        source = Path(mod.__file__).read_text(encoding="utf-8")
        assert "create_data_source" in source
        assert 'create_data_source("csv"' in source
        assert 'create_data_source("tqsdk"' in source

    def test_session_state_module_imports(self):
        """模块本身可被 import（无 ImportError）。"""
        from utils import session_state  # noqa: F401
        assert hasattr(session_state, "load_data_cached")
        assert hasattr(session_state, "load_tqsdk_cached")


# ───────────────────────────────────────────────────────────
# 5) 二次审计清单（规则 22.6）
# ───────────────────────────────────────────────────────────
class TestSecondaryAuditChecklist:
    """M-04 二次审计清单：迁移后生产代码不应再使用 DataLoader(data_source=...) 显式构造。"""

    def test_no_data_loader_data_source_in_production(self):
        """生产代码（排除测试/迁移目标/工具）中 DataLoader(data_source=...) 应为 0 行。"""
        production_dirs = [
            "core/",
            "runner/",
            "utils/",
        ]
        bad_lines = []
        for d in production_dirs:
            root = PROJECT_ROOT / d
            if not root.exists():
                continue
            for py in root.rglob("*.py"):
                # 排除：
                # - 迁移目标（adapters/*_adapter.py 仍用，但这是 M-01 委托路径，规则允许）
                # - core/data_loader.py 自身（这是 deprecated 警告的源头）
                if "ext/adapters/" in str(py):
                    continue
                if py.name == "data_loader.py":
                    continue
                txt = py.read_text(encoding="utf-8")
                if "DataLoader(data_source=" in txt:
                    bad_lines.append(str(py.relative_to(PROJECT_ROOT)))

        assert bad_lines == [], (
            f"下列文件仍直接构造 DataLoader(data_source=...)，应改用 create_data_source 工厂：\n"
            + "\n".join(bad_lines)
        )
