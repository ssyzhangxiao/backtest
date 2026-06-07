"""
因子计算引擎。

负责：数据清洗 → 公共数据准备 → 因子调度 → 结果汇总。
"""
import logging
from typing import Dict, List, Optional, Any

import numpy as np

from .base_factor import BaseFactor
from .factor_registry import get_factor, list_available_factors
from ..futures_data_cleaners import (
    adjust_price_for_roll,
    compute_adaptive_gap_weight,
    compute_oi_safe,
    compute_open_adj,
    compute_intraday_ret,
    compute_carry,
)
from ..operators import (
    delay,
    delta,
    sma,
    safe_div,
    std,
)

logger = logging.getLogger(__name__)


def _auto_generate_roll_map(is_dominant: np.ndarray) -> np.ndarray:
    """
    根据主力标记生成简易换月映射（1=无换月，-1=换月日）。

    当 `is_dominant[i] != is_dominant[i-1]` 时认为当日为换月日。
    此映射与 `adjust_price_for_roll` 配合使用：将换月日 close
    按主力切换比例（隐含在 close 自身）做后复权处理，确保因子
    计算不会因合约切换产生跳变。

    Args:
        is_dominant: 是否主力合约的布尔序列

    Returns:
        roll_map: 整型数组，1 表示该日无换月，-1 表示换月日
    """
    is_dom = np.asarray(is_dominant, dtype=bool)
    roll_map = np.ones_like(is_dom, dtype=int)
    if len(is_dom) < 2:
        return roll_map
    # 异或：true 表示切换日
    switched = is_dom[1:] != is_dom[:-1]
    roll_map[1:] = np.where(switched, -1, 1)
    return roll_map


class FactorEngine:
    """
    因子计算引擎。

    统一管理数据清洗、公共数据准备、因子计算和结果汇总。
    """

    def __init__(
        self,
        config: Any,
        factor_names: Optional[List[str]] = None,
    ):
        """
        初始化引擎。

        Args:
            config: 全局配置对象（AlphaFuturesConfig）
            factor_names: 要计算的因子列表，None 表示全部已注册因子
        """
        self.config = config
        self.factor_names = factor_names or list_available_factors()
        self.factors: List[BaseFactor] = [
            get_factor(name, config) for name in self.factor_names
        ]
        self._cache: Dict[str, np.ndarray] = {}

    def clear_cache(self) -> None:
        """清除缓存。"""
        self._cache.clear()

    def compute_all(
        self,
        raw_data: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        计算所有因子。

        Args:
            raw_data: 原始数据字典，包含：
                - close: 收盘价
                - open_price: 开盘价
                - high: 最高价
                - low: 最低价
                - open_interest: 持仓量
                - near_price: 近月价格（可选）
                - far_price: 远月价格（可选）
                - far_oi: 远月持仓量（可选）
                - is_dominant: 是否主力合约（可选）
                - delivery_exclude: 交割月剔除（可选）
                - gap_weight: 跳空权重（可选）

        Returns:
            {因子编号: 因子值序列}
        """
        self.clear_cache()

        # 1. 数据清洗，构建公共数据字典
        public_data = self._prepare_public_data(raw_data)

        # 2. 计算每个因子
        results: Dict[str, np.ndarray] = {}
        # 异常隔离模板：全 NaN，与公共数据等长
        nan_template = np.full(len(public_data["close"]), np.nan, dtype=float)
        for factor in self.factors:
            # 提取因子需要的字段
            needs = {k: public_data[k] for k in factor.get_needs() if k in public_data}
            # 传递配置参数
            needs["zscore_window"] = self._get_zscore_window()
            needs["oi_mean_20"] = public_data.get("oi_mean_20")
            # 计算因子（异常隔离：单个因子失败不影响整体）
            try:
                values = factor.compute(**needs)
            except Exception as e:  # noqa: BLE001
                logger.error(
                    "因子 %s compute() 失败: %s；返回全 NaN 兜底",
                    factor.name, e,
                    exc_info=logger.isEnabledFor(logging.DEBUG),
                )
                values = nan_template.copy()
            # 后处理
            try:
                values = factor.post_process(values)
            except Exception as e:  # noqa: BLE001
                logger.error(
                    "因子 %s post_process() 失败: %s；返回原始值",
                    factor.name, e,
                    exc_info=logger.isEnabledFor(logging.DEBUG),
                )
            # 保存结果
            results[factor.name] = values

        # ── 边界保护：强制等长契约 ──
        # 任何因子 compute() 内部若因数据缺失使用了硬编码 fallback（如 ts_01.py 的 length=100），
        # 会导致返回值长度与 public_data["close"] 不一致，下游 df[factor_name] = values 会抛
        # "Length of values (X) does not match length of index (Y)"。此处统一做 NaN right-align。
        target_len = len(public_data["close"])
        for _name in list(results.keys()):
            _vals = results[_name]
            if len(_vals) != target_len:
                logger.warning(
                    "因子 %s 返回长度 %d 与期望 %d 不一致，自动 NaN right-align 兜底",
                    _name, len(_vals), target_len,
                )
                _aligned = np.full(target_len, np.nan, dtype=float)
                _n = min(len(_vals), target_len)
                if _n > 0:
                    _aligned[-_n:] = _vals[-_n:]
                results[_name] = _aligned

        return results

    def _prepare_public_data(
        self,
        raw: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        执行所有清洗步骤，生成公共字段。

        Args:
            raw: 原始数据字典

        Returns:
            公共数据字典
        """
        cfg = self.config
        public: Dict[str, np.ndarray] = {}

        # 基础价格数据
        close = np.asarray(raw["close"], dtype=float)
        open_price = np.asarray(raw["open_price"], dtype=float)
        high = np.asarray(raw["high"], dtype=float)
        low = np.asarray(raw["low"], dtype=float)
        open_interest = np.asarray(raw["open_interest"], dtype=float)

        public["close"] = close
        public["open_price"] = open_price
        public["high"] = high
        public["low"] = low
        public["open_interest"] = open_interest

        # 成交量（CF_02 等资金流因子需要）
        if "volume" in raw:
            public["volume"] = np.asarray(raw["volume"], dtype=float)
        else:
            public["volume"] = np.zeros_like(close)

        # 近月/远月价格（TS_01/TS_02/TS_03 期限结构因子需要）
        if "near_price" in raw:
            public["near_price"] = np.asarray(raw["near_price"], dtype=float)
        if "far_price" in raw:
            public["far_price"] = np.asarray(raw["far_price"], dtype=float)

        # 可选数据
        near_price = raw.get("near_price")
        far_price = raw.get("far_price")
        far_oi = raw.get("far_oi")
        is_dominant = raw.get("is_dominant")
        delivery_exclude = raw.get("delivery_exclude")
        gap_weight = raw.get("gap_weight")

        # 可选复权调整（is_dominant 在前文赋值后才能使用）
        roll_map = raw.get("roll_map")
        if roll_map is None and is_dominant is not None:
            # is_dominant 转换日 = 换月日：据此自动生成简易 roll_map
            roll_map = _auto_generate_roll_map(np.asarray(is_dominant, dtype=bool))
        if roll_map is not None:
            close = adjust_price_for_roll(close, roll_map)
            open_price = adjust_price_for_roll(open_price, roll_map)
            high = adjust_price_for_roll(high, roll_map)
            low = adjust_price_for_roll(low, roll_map)

        # OI 清洗
        oi_safe = compute_oi_safe(
            open_interest,
            is_dominant=is_dominant,
            delivery_exclude=delivery_exclude,
        )
        public["oi_safe"] = oi_safe

        # 跳空修复：若未提供外部 gap_weight，则基于历史跳空延续率自适应计算
        if gap_weight is None:
            gap_weight = compute_adaptive_gap_weight(
                open_price,
                close,
                window=getattr(cfg, "gap_weight_window", 20),
                default=cfg.gap_weight,
            )
        open_adj = compute_open_adj(
            open_price,
            close,
            gap_weight=gap_weight,
            default_weight=cfg.gap_weight,
        )
        public["open_adj"] = open_adj

        # 日内收益率
        intraday_ret = compute_intraday_ret(
            close,
            open_adj,
            limit_move_threshold=cfg.limit_move_threshold,
            open_price=open_price,
        )
        public["intraday_ret"] = intraday_ret

        # Carry 计算
        if near_price is not None and far_price is not None:
            carry = compute_carry(
                near_price=np.asarray(near_price, dtype=float),
                far_price=np.asarray(far_price, dtype=float),
                far_oi=np.asarray(far_oi, dtype=float) if far_oi is not None else None,
                oi_threshold=cfg.carry_oi_threshold,
                symbol=cfg.symbol,
                momentum_window=cfg.momentum_orth_window,
                close_price=close,
            )
            public["carry"] = carry
        else:
            public["carry"] = np.zeros_like(close)

        # 常用滚动统计（缓存避免重复计算）
        self._prepare_common_rolling(public, close, oi_safe)

        return public

    def _prepare_common_rolling(
        self,
        public: Dict[str, np.ndarray],
        close: np.ndarray,
        oi_safe: np.ndarray,
    ) -> None:
        """
        准备常用的滚动统计量，避免不同因子重复计算。

        Args:
            public: 公共数据字典（会更新）
            close: 收盘价
            oi_safe: 安全持仓量
        """
        # 简单收益率（使用 safe_div 避免 delay(close,1)=0/NaN 时产生 inf）
        ret = safe_div(close - delay(close, 1), delay(close, 1))
        public["ret"] = ret

        # OI 差分
        public["delta_oi_1"] = delta(oi_safe, 1)
        public["delta_oi_5"] = delta(oi_safe, 5)

        # OI 均值
        public["oi_mean_5"] = sma(oi_safe, 5)
        public["oi_mean_20"] = sma(oi_safe, 20)

        # 波动率
        public["vol_20"] = std(ret, 20)

        # 价格差分
        public["delta_close_1"] = delta(close, 1)
        public["delta_close_5"] = delta(close, 5)

    def _get_zscore_window(self) -> Optional[int]:
        """获取zscore窗口：0=None（扩张窗口），>0=滚动窗口"""
        w = getattr(self.config, "zscore_window", 0)
        return None if w == 0 else w
