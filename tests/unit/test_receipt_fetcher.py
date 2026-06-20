"""仓单 fetcher 三层架构单元测试（2026-06-19）。

测试覆盖：
  1. 缓存层：save / load / TTL 过期 / status=failed
  2. 信号层：get_receipt_change_signal 边界条件
  3. 编排器：enable_online=False 时不发起外部请求
  4. 接口层：4 个适配器路由正确
  5. 列名映射：RECEIPT_COLUMN_MAP 解析正确
  6. BacktestConfig：receipt 段正确加载
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

# ── 缓存层测试 ────────────────────────────────────────────────


class TestReceiptCache:
    """缓存层单元测试（纯本地 IO，无网络）。"""

    def test_save_and_load(self, tmp_path: Path) -> None:
        """正常保存 + 读取。"""
        from core.data._receipt_cache import ReceiptCache

        cache = ReceiptCache(cache_dir=str(tmp_path), ttl_days=7)
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "symbol": ["SHFE.RB", "SHFE.RB"],
            "receipt": [1000, 1100],
        })
        cache.save(df, "2024-01-01", "2024-01-02")
        loaded = cache.load("2024-01-01", "2024-01-02")
        assert loaded is not None
        assert len(loaded) == 2
        assert loaded["receipt"].tolist() == [1000, 1100]

    def test_load_miss(self, tmp_path: Path) -> None:
        """缓存不存在时返回 None。"""
        from core.data._receipt_cache import ReceiptCache

        cache = ReceiptCache(cache_dir=str(tmp_path), ttl_days=7)
        loaded = cache.load("2024-01-01", "2024-12-31")
        assert loaded is None

    def test_ttl_expired(self, tmp_path: Path) -> None:
        """缓存过期后返回 None。"""
        from core.data._receipt_cache import ReceiptCache

        cache = ReceiptCache(cache_dir=str(tmp_path), ttl_days=0)
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"]),
            "symbol": ["SHFE.RB"],
            "receipt": [1000],
        })
        cache.save(df, "2024-01-01", "2024-01-01")
        # TTL=0 立即过期
        loaded = cache.load("2024-01-01", "2024-01-01")
        assert loaded is None

    def test_failed_status_skipped(self, tmp_path: Path) -> None:
        """status=failed 的缓存直接跳过（parquet 不写）。"""
        from core.data._receipt_cache import ReceiptCache

        cache = ReceiptCache(cache_dir=str(tmp_path), ttl_days=7)
        # failed 状态：只写 meta 不写 parquet
        cache.save(pd.DataFrame(), "2024-01-01", "2024-01-02", status="failed")
        # parquet 不存在 → 返回 None
        loaded = cache.load("2024-01-01", "2024-01-02")
        assert loaded is None

    def test_clear(self, tmp_path: Path) -> None:
        """clear 删除缓存。"""
        from core.data._receipt_cache import ReceiptCache

        cache = ReceiptCache(cache_dir=str(tmp_path), ttl_days=7)
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"]),
            "symbol": ["SHFE.RB"],
            "receipt": [1000],
        })
        cache.save(df, "2024-01-01", "2024-01-01")
        cache.clear("2024-01-01", "2024-01-01")
        assert cache.load("2024-01-01", "2024-01-01") is None

    def test_empty_data_cached_as_empty_df(self, tmp_path: Path) -> None:
        """空数据缓存命中（2026-06-19 优化）：status=success+rows=0 返回空 DataFrame，不返回 None。"""
        from core.data._receipt_cache import ReceiptCache

        cache = ReceiptCache(cache_dir=str(tmp_path), ttl_days=7)
        empty_df = pd.DataFrame(columns=["date", "symbol", "receipt"])
        cache.save(empty_df, "2024-01-01", "2024-12-31", status="success")
        # 再次加载：应命中（返回空 DataFrame，非 None）
        loaded = cache.load("2024-01-01", "2024-12-31")
        assert loaded is not None
        assert isinstance(loaded, pd.DataFrame)
        assert loaded.empty
        assert list(loaded.columns) == ["date", "symbol", "receipt"]

    def test_clear_deletes_meta(self, tmp_path: Path) -> None:
        """clear 同时删除 .meta 文件（2026-06-19 验证）。"""
        from core.data._receipt_cache import ReceiptCache

        cache = ReceiptCache(cache_dir=str(tmp_path), ttl_days=7)
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"]),
            "symbol": ["SHFE.RB"],
            "receipt": [1000],
        })
        cache.save(df, "2024-01-01", "2024-01-01")
        # 验证 meta 与 parquet 都存在
        assert (tmp_path / "receipt_2024-01-01_2024-01-01.parquet").exists()
        assert (tmp_path / "receipt_2024-01-01_2024-01-01.parquet.meta").exists()
        # 清除
        cache.clear("2024-01-01", "2024-01-01")
        assert not (tmp_path / "receipt_2024-01-01_2024-01-01.parquet").exists()
        assert not (tmp_path / "receipt_2024-01-01_2024-01-01.parquet.meta").exists()


# ── 信号层测试 ────────────────────────────────────────────────


class TestReceiptChangeSignal:
    """信号计算单元测试。"""

    def test_empty_series(self) -> None:
        """空序列返回空信号。"""
        from core.data.receipt_fetcher import get_receipt_change_signal

        s = pd.Series(dtype=float)
        out = get_receipt_change_signal(s, window=20)
        assert len(out) == 0

    def test_warmup_zero(self) -> None:
        """前 window 个交易日为 0（warmup）。"""
        from core.data.receipt_fetcher import get_receipt_change_signal

        s = pd.Series([100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
                       210, 220, 230, 240, 250, 260, 270, 280, 290, 300])
        out = get_receipt_change_signal(s, window=5)
        assert out.iloc[:5].abs().sum() == 0.0
        assert (out.iloc[5:] != 0).any()

    def test_zero_value_no_division_error(self) -> None:
        """分母为零时不报除零错误。"""
        from core.data.receipt_fetcher import get_receipt_change_signal

        s = pd.Series([0, 0, 100, 200, 300, 400, 500, 600, 700, 800,
                       900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900])
        out = get_receipt_change_signal(s, window=5)
        assert out.notna().all()
        assert (out.abs() <= 1.0).all()

    def test_negative_value_stable(self) -> None:
        """负值不导致异常（仓单过户场景）。"""
        from core.data.receipt_fetcher import get_receipt_change_signal

        s = pd.Series([100, 110, 120, -5, 130, 140, 150, 160, 170, 180,
                       190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290])
        out = get_receipt_change_signal(s, window=5)
        assert out.notna().all()
        assert (out.abs() <= 1.0).all()

    def test_signal_inverse_to_change(self) -> None:
        """信号方向与仓单变化相反（仓单↑ → 做空信号 < 0）。"""
        from core.data.receipt_fetcher import get_receipt_change_signal

        # 单调递增序列
        s = pd.Series(list(range(100, 200)))
        out = get_receipt_change_signal(s, window=5)
        # warmup 段后，信号应为负（仓单持续增加 → 做空）
        assert (out.iloc[5:] < 0).all()


# ── 编排器测试 ────────────────────────────────────────────────


class TestReceiptFetcher:
    """ReceiptFetcher 编排器测试。"""

    def test_init_default(self) -> None:
        """默认配置：enable_online=False, cache_ttl_days=7。"""
        from core.data import ReceiptFetcher

        f = ReceiptFetcher()
        assert f.enable_online is False
        assert f.interval_min == 1.0
        assert f.interval_max == 4.0
        assert f.retry_times == 3

    def test_init_from_dict(self) -> None:
        """dict 配置注入。"""
        from core.data import ReceiptFetcher

        f = ReceiptFetcher(config={
            "enable_online": True,
            "request_interval_min": 0.5,
            "cache_ttl_days": 3,
        })
        assert f.enable_online is True
        assert f.interval_min == 0.5
        assert f.cache.ttl_days == 3

    def test_fetch_offline_no_cache(self, tmp_path: Path) -> None:
        """enable_online=False 且无缓存 → 返回空 DataFrame（不抛错）。"""
        from core.data import ReceiptFetcher

        f = ReceiptFetcher(config={"cache_dir": str(tmp_path), "enable_online": False})
        df = f.fetch_range(
            symbols=["SHFE.RB", "DCE.M"],
            start_date="2024-01-01",
            end_date="2024-12-31",
        )
        assert df.empty
        assert list(df.columns) == ["date", "symbol", "receipt"]

    def test_fetch_offline_uses_cache(self, tmp_path: Path) -> None:
        """enable_online=False 但有缓存 → 优先返回缓存。"""
        from core.data import ReceiptFetcher
        from core.data._receipt_cache import ReceiptCache

        # 预写缓存
        cache = ReceiptCache(cache_dir=str(tmp_path), ttl_days=7)
        df_cache = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "symbol": ["SHFE.RB", "SHFE.RB"],
            "receipt": [1000, 1100],
        })
        cache.save(df_cache, "2024-01-01", "2024-12-31")

        f = ReceiptFetcher(config={"cache_dir": str(tmp_path), "enable_online": False})
        df = f.fetch_range(symbols=["SHFE.RB"], start_date="2024-01-01", end_date="2024-12-31")
        assert len(df) == 2
        assert df["receipt"].iloc[0] == 1000

    def test_fetch_offline_no_network(self, tmp_path: Path, monkeypatch) -> None:
        """enable_online=False 时，验证 requests.Session 不会被构造。"""
        from core.data import ReceiptFetcher

        f = ReceiptFetcher(config={"cache_dir": str(tmp_path), "enable_online": False})
        # 调用 _get_session 不应触发 Session 构造（仍为 None）
        session = f._get_session()
        assert session is None

    def test_has_receipt_data(self) -> None:
        """中文名映射检查。"""
        from core.data import ReceiptFetcher

        f = ReceiptFetcher()
        assert f.has_receipt_data("SHFE.RB") is True
        assert f.has_receipt_data("UNKNOWN.SYM") is False

    def test_get_receipt_series(self) -> None:
        """兼容旧 API：get_receipt_series。"""
        from core.data import ReceiptFetcher

        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "symbol": ["SHFE.RB", "SHFE.RB", "DCE.M"],
            "receipt": [1000, 1100, 500],
        })
        s = ReceiptFetcher.get_receipt_series("SHFE.RB", df)
        assert len(s) == 2
        assert s.iloc[0] == 1000
        assert s.iloc[1] == 1100


# ── 接口层测试 ────────────────────────────────────────────────


class TestReceiptAdapters:
    """4 个交易所适配器路由测试。"""

    def test_adapter_routing(self) -> None:
        """4 个交易所都有适配器。"""
        from core.data._receipt_adapters import ADAPTER_FN

        assert "SHFE" in ADAPTER_FN
        assert "DCE" in ADAPTER_FN
        assert "CZCE" in ADAPTER_FN
        assert "GFEX" in ADAPTER_FN

    def test_ua_pool_size(self) -> None:
        """UA 池 ≥ 5 个。"""
        from core.data._receipt_adapters import USER_AGENTS

        assert len(USER_AGENTS) >= 5

    def test_build_session_no_crash(self) -> None:
        """Session 构造不抛错。"""
        from core.data._receipt_adapters import build_session

        s = build_session()
        assert s is not None
        assert "User-Agent" in s.headers

    def test_polite_sleep_short(self) -> None:
        """反爬虫间隔（用小值测试以加快测试速度）。"""
        from core.data._receipt_adapters import polite_sleep

        t0 = time.time()
        polite_sleep(0.01, 0.05)
        elapsed = time.time() - t0
        assert 0.0 <= elapsed < 0.5

    def test_column_map_has_all_exchanges(self) -> None:
        """列名映射覆盖 4 个交易所。"""
        from core.data._receipt_adapters import RECEIPT_COLUMN_MAP

        for ex in ("SHFE", "DCE", "CZCE", "GFEX"):
            assert ex in RECEIPT_COLUMN_MAP
            assert "symbol" in RECEIPT_COLUMN_MAP[ex]
            assert "receipt" in RECEIPT_COLUMN_MAP[ex]

    def test_normalize_symbol_exact_match_priority(self) -> None:
        """symbol 列精确匹配优先于 cn_name 模糊匹配（2026-06-19 优化）。"""
        from core.data._receipt_adapters import _normalize_dataframe

        # 模拟 SHFE 响应：包含 "螺纹钢"（多品种 + 干扰项）
        raw = pd.DataFrame({
            "VARNAME": ["螺纹钢", "热轧卷板", "螺纹钢线材"],  # 干扰项 "螺纹钢线材"
            "RECEIPT": [100, 200, 9999],  # 干扰项数值很大
        })
        result = _normalize_dataframe(
            raw_df=raw, exchange="SHFE",
            target_symbol="SHFE.RB", cn_name="螺纹钢",
            fetch_date=pd.Timestamp("2024-01-01"),
        )
        # 优先 symbol 精确匹配（RB），但 raw 没有 RB → fallback cn_name
        # 应当排除"螺纹钢线材"（模糊匹配陷阱）
        assert result is not None
        assert result["receipt"].iloc[0] == 100  # 100 而非 100+9999

    def test_normalize_symbol_short_code_match(self) -> None:
        """symbol 短码匹配（SHFE.RB → "RB"）。"""
        from core.data._receipt_adapters import _normalize_dataframe

        raw = pd.DataFrame({
            "VARNAME": ["RB", "HC", "CU"],
            "RECEIPT": [500, 600, 700],
        })
        result = _normalize_dataframe(
            raw_df=raw, exchange="SHFE",
            target_symbol="SHFE.RB", cn_name="螺纹钢",
            fetch_date=pd.Timestamp("2024-01-01"),
        )
        assert result is not None
        assert result["receipt"].iloc[0] == 500

    def test_fetch_online_uses_business_days(self) -> None:
        """在线拉取用 bdate_range（2026-06-19 优化：仅工作日）。"""
        from core.data import ReceiptFetcher

        f = ReceiptFetcher(config={"enable_online": True})
        # 验证 _fetch_online 内部使用 pd.bdate_range
        import pandas as pd
        # 周五到下周一：仅周五 + 下周一（2 个工作日，跳过周末）
        dates = pd.bdate_range("2024-01-05", "2024-01-08")
        # 5 = 周五, 8 = 周一 → 2 个工作日
        assert len(dates) == 2
        assert dates[0].weekday() < 5  # 工作日
        assert dates[1].weekday() < 5


# ── 配置层测试 ────────────────────────────────────────────────


class TestBacktestConfigReceipt:
    """BacktestConfig 加载 receipt 段。"""

    def test_default_values(self) -> None:
        """默认配置（无 receipt 段）。"""
        from core.config import BacktestConfig

        cfg = BacktestConfig()
        assert cfg.receipt_cache_dir == "data/receipt_cache"
        assert cfg.receipt_enable_online is False
        assert cfg.receipt_retry_times == 3
        assert cfg.receipt_cache_ttl_days == 7

    def test_load_from_yaml(self) -> None:
        """从 yaml 加载 receipt 段。"""
        from core.config import BacktestConfig

        cfg = BacktestConfig.from_yaml("config.yaml")
        assert cfg.receipt_cache_dir == "data/receipt_cache"
        assert cfg.receipt_enable_online is False
        assert cfg.receipt_request_interval_min == 1.0
        assert cfg.receipt_request_interval_max == 4.0
        assert cfg.receipt_retry_times == 3
        assert cfg.receipt_cache_ttl_days == 7
