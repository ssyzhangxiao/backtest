"""
商品期货 Alpha 因子库 — 五大类24因子。

基于国泰君安191短周期价量因子，经三大核心改造适配商品期货：
  1. VOLUME → OI（持仓量）：持仓量比成交量更能反映资金博弈主力意图
  2. 引入 Carry 期限结构：近远月价差是商品独有的Alpha源
  3. 修复跳空缺口：OPEN_ADJ = OPEN*w + DELAY(CLOSE,1)*(1-w)，消除隔夜跳空污染

适用性工程改造（关键修正）：
  - 主力合约换月陷阱：使用全品种总持仓量，换月日前后剔除
  - 交割月强制平仓：剔除进入交割月前N个交易日的数据
  - Carry 流动性过滤：远月持仓量<阈值时 Carry 因子置零（支持按品种动态阈值）
  - Carry 与动量正交化：滚动回归剥离动量效应后 Carry 才是真实适用
  - 跳空权重自适应：权重根据品种历史跳空延续率分配，非固定0.5
  - 涨跌停板过滤：开盘触及涨跌停（涨停+跌停）的交易日INTRADAY_RET置零

因子分类：
  - 趋势类(T)：5个 — T_01~T_05
  - 回归类(R)：5个 — R_01~R_05
  - 波动率类(V)：4个 — V_01~V_04
  - 资金流类(M)：5个 — M_01~M_05
  - 高阶复合类(H)：5个 — H_01~H_05

价格序列要求：
  - 所有价格序列（close, open_price, high, low）必须传入**向后复权**或**比例复权**
    后的价格序列，以消除合约换月带来的价格跳空。
  - 若无法提供复权价格，可通过 compute_all 的 roll_map 参数启用库内复权逻辑。

规则9要求：新因子必须通过IC检验才能入库，IC>0.03且IR>0.5。

模块拆分：
  - futures_config.py: 配置类 AlphaFuturesConfig
  - operators.py: 基础算子（safe_div, delay, zscore 等）
  - futures_data_cleaners.py: 数据清洗算子（compute_open_adj, compute_carry 等）
  - alpha_futures_trend.py: 趋势类因子
  - alpha_futures_reversal.py: 回归类因子
  - alpha_futures_volatility.py: 波动率类因子
  - alpha_futures_money_flow.py: 资金流类因子
  - alpha_futures_high_order.py: 高阶复合类因子
  - 本文件: 编排入口 AlphaFutures24 类
"""

from typing import Callable, Dict, Optional
import logging

import numpy as np

# 配置类
from .futures_config import AlphaFuturesConfig, OIThresholdType

# 数据清洗算子
from .futures_data_cleaners import (
    adjust_price_for_roll,
    compute_carry,
    compute_intraday_ret,
    compute_oi_safe,
    compute_open_adj,
    generate_delivery_exclude,
)

# 各类因子计算函数
from .alpha_futures_trend import compute_trend_factors
from .alpha_futures_reversal import compute_reversal_factors
from .alpha_futures_volatility import compute_volatility_factors
from .alpha_futures_money_flow import compute_money_flow_factors
from .alpha_futures_high_order import compute_high_order_factors

# 后处理算子
from .operators import winsorize, clipping

logger = logging.getLogger(__name__)


class AlphaFutures24:
    """
    商品期货Alpha因子库计算器（编排入口）。

    五大类24因子，基于国泰君安191因子改造适配商品期货。
    所有因子均经过适用性工程改造，解决换月陷阱、交割月干扰、
    Carry流动性枯竭、跳空权重自适应等关键问题。

    本类仅负责编排：数据清洗 → Carry计算 → 各类因子计算 → 汇总。
    具体因子计算逻辑委托给各分类模块。

    用法:
        calc = AlphaFutures24()
        factors = calc.compute_all(
            close=close_arr,
            open_price=open_arr,
            high=high_arr,
            low=low_arr,
            open_interest=oi_arr,
            near_price=near_arr,
            far_price=far_arr,
        )
    """

    def __init__(self, config: Optional[AlphaFuturesConfig] = None):
        self.config = config or AlphaFuturesConfig()
        # 滚动计算缓存，避免重复计算
        self._cache: Dict[str, np.ndarray] = {}

    def _get_zscore_window(self) -> Optional[int]:
        """获取zscore窗口：0=None（扩张窗口），>0=滚动窗口"""
        w = self.config.zscore_window
        return None if w == 0 else w

    def _cached(self, key: str, fn: Callable[[], np.ndarray]) -> np.ndarray:
        """缓存滚动计算结果，避免同一序列重复计算"""
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def clear_cache(self) -> None:
        """清除缓存（每次compute_all开始时自动调用）"""
        self._cache.clear()

    def compute_all(
        self,
        close: np.ndarray,
        open_price: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        open_interest: np.ndarray,
        near_price: Optional[np.ndarray] = None,
        far_price: Optional[np.ndarray] = None,
        far_oi: Optional[np.ndarray] = None,
        is_dominant: Optional[np.ndarray] = None,
        delivery_exclude: Optional[np.ndarray] = None,
        gap_weight: Optional[np.ndarray] = None,
        roll_map: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        计算全部24个因子。

        适用性工程改造自动执行：
          1. OI清洗（换月/交割月剔除）
          2. 跳空缺口修复（自适应权重）
          3. Carry流动性过滤 + 动量正交化
          4. 涨跌停板过滤（涨停+跌停）

        Args:
            close: 收盘价序列（需向后复权或比例复权）
            open_price: 开盘价序列（需向后复权或比例复权）
            high: 最高价序列（需向后复权或比例复权）
            low: 最低价序列（需向后复权或比例复权）
            open_interest: 持仓量序列
            near_price: 近月合约价格序列（Carry因子需要）
            far_price: 远月合约价格序列（Carry因子需要）
            far_oi: 远月合约持仓量序列（Carry流动性过滤）
            is_dominant: 是否主力合约标记序列
            delivery_exclude: 交割月剔除标记序列
            gap_weight: 品种自适应跳空权重序列
            roll_map: 换月映射表（1=无换月，-1=换月日），启用库内复权

        Returns:
            {因子编号: 因子值序列}，共24个因子
        """
        cfg = self.config
        self.clear_cache()

        # ── 输入校验 ──
        n = len(close)
        for name, arr in [
            ("open_price", open_price),
            ("high", high),
            ("low", low),
            ("open_interest", open_interest),
        ]:
            assert len(arr) == n, f"{name}长度({len(arr)})与close({n})不一致"

        # ── 可选复权调整 ──
        if roll_map is not None:
            close = adjust_price_for_roll(close, roll_map)
            open_price = adjust_price_for_roll(open_price, roll_map)
            high = adjust_price_for_roll(high, roll_map)
            low = adjust_price_for_roll(low, roll_map)

        # ── Step 1: OI安全清洗 ──
        oi_safe = compute_oi_safe(
            open_interest,
            is_dominant=is_dominant,
            delivery_exclude=delivery_exclude,
        )

        # ── Step 2: 跳空缺口修复 ──
        open_adj = compute_open_adj(
            open_price,
            close,
            gap_weight=gap_weight,
            default_weight=cfg.gap_weight,
        )
        intraday_ret = compute_intraday_ret(
            close,
            open_adj,
            limit_move_threshold=cfg.limit_move_threshold,
            open_price=open_price,
        )

        # ── Step 3: Carry计算（含流动性过滤+动量正交化）──
        if near_price is not None and far_price is not None:
            carry = compute_carry(
                near_price,
                far_price,
                far_oi=far_oi,
                oi_threshold=cfg.carry_oi_threshold,
                symbol=cfg.symbol,
                momentum_window=cfg.momentum_orth_window,
                close_price=close,
            )
        else:
            carry = np.zeros_like(close)
            logger.warning("无近远月价格数据，Carry因子置零")

        # ── Step 4: 计算五大类因子 ──
        zw = self._get_zscore_window()
        all_factors: Dict[str, np.ndarray] = {}

        # 趋势类
        all_factors.update(compute_trend_factors(close, oi_safe, carry))

        # 回归类
        all_factors.update(
            compute_reversal_factors(
                close, high, oi_safe, intraday_ret, open_adj, carry, zw
            )
        )

        # 波动率类
        all_factors.update(compute_volatility_factors(high, low, oi_safe, intraday_ret))

        # 资金流类
        all_factors.update(
            compute_money_flow_factors(close, high, low, oi_safe, carry, self._cached)
        )

        # 高阶复合类
        all_factors.update(
            compute_high_order_factors(close, oi_safe, carry, zw, self._cached)
        )

        logger.info(f"商品期货Alpha因子库计算完成：{len(all_factors)}个因子")
        return all_factors

    @staticmethod
    def post_process(
        factors: Dict[str, np.ndarray],
        do_winsorize: bool = True,
        do_clip_v01: bool = True,
        v01_clip_range: float = 50.0,
    ) -> Dict[str, np.ndarray]:
        """
        因子后处理：缩尾、截断、NaN传播。

        规则9要求：每个因子计算完成后应进行缩尾处理去除极端值。

        Args:
            factors: compute_all() 返回的原始因子字典
            do_winsorize: 是否进行1%/99%缩尾
            do_clip_v01: 是否对V_01进行±50%截断
            v01_clip_range: V_01截断范围

        Returns:
            后处理后的因子字典（新字典，不修改输入）
        """
        result = {}
        for name, values in factors.items():
            arr = values.copy()
            # V_01特殊处理：截断±50%，防止OI变化率异常放大
            if do_clip_v01 and name == "V_01":
                arr = clipping(arr, -v01_clip_range, v01_clip_range)
            # 全局缩尾：1%/99%去除极端值
            if do_winsorize:
                arr = winsorize(arr, 0.01, 0.99)
            result[name] = arr
        return result

    @staticmethod
    def get_factor_info() -> Dict[str, Dict[str, str]]:
        """
        获取所有因子的元信息。

        Returns:
            {因子编号: {formula, category, desc}}
        """
        return {
            # 趋势类
            "T_01": {
                "formula": "(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6) * DELTA(OI,1)",
                "category": "趋势",
                "desc": "6日动量与日增仓乘积，价格突破伴随增仓是最确凿的趋势确认",
            },
            "T_02": {
                "formula": "(CLOSE-DELAY(CLOSE,12))/DELAY(CLOSE,12) * OI",
                "category": "趋势",
                "desc": "12日动量与总持仓乘积，过滤低持仓伪突破",
            },
            "T_03": {
                "formula": "(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1) * DELTA(OI,1)",
                "category": "趋势",
                "desc": "日度收益率与日增仓乘积，极短期资金入场方向确认",
            },
            "T_04": {
                "formula": "CARRY_ORTH * DELTA(OI,1)",
                "category": "趋势",
                "desc": "期限结构与增仓共振（Carry已正交化），商品顶级Alpha因子",
            },
            "T_05": {
                "formula": "SUM(CLOSE>DELAY(CLOSE,1)?DELTA(OI,1):(CLOSE<DELAY(CLOSE,1)?-DELTA(OI,1):0),6)",
                "category": "趋势",
                "desc": "6日条件增仓累积（OBV-OI变形），确认趋势资金厚度",
            },
            # 回归类
            "R_01": {
                "formula": "(-1*CORR(ZSCORE(DELTA(LOG(OI),1),w),ZSCORE(INTRADAY_RET,w),6))",
                "category": "回归",
                "desc": "平滑日内涨跌与增仓率背离，捕捉增仓滞涨顶部反转",
            },
            "R_02": {
                "formula": "(-1*CORR(HIGH,ZSCORE(DELTA(OI,1),20),5))",
                "category": "回归",
                "desc": "最高价与增仓率滚动标准化5日相关性，创新高+持仓流失→反转空头",
            },
            "R_03": {
                "formula": "((-1*ZSCORE(DELTA(RET,3),w))*CORR(OPEN_ADJ,DELTA(OI,1),10))",
                "category": "回归",
                "desc": "收益率变化与平滑开盘增仓相关性乘积，判断多空翻转点",
            },
            "R_04": {
                "formula": "(-1*ZSCORE(CARRY_ORTH,w))",
                "category": "回归",
                "desc": "期限结构均值回复（Carry已正交化），极端Back/Contango不可持续",
            },
            "R_05": {
                "formula": "(-1*OI/MEAN(OI,20))",
                "category": "回归",
                "desc": "负相对持仓量，持仓极度萎缩→蓄势反转节点",
            },
            # 波动率类
            "V_01": {
                "formula": "(OI-DELAY(OI,5))/DELAY(OI,5)*100",
                "category": "波动率",
                "desc": "5日持仓量变化率，持仓异动是波动率扩张先兆",
            },
            "V_02": {
                "formula": "STD(INTRADAY_RET,20)*DELTA(OI,5)",
                "category": "波动率",
                "desc": "平滑日内波动率与5日增仓乘积，确认趋势行情",
            },
            "V_03": {
                "formula": "ZSCORE(HIGH-LOW,20)*ZSCORE(DELTA(OI,1),20)",
                "category": "波动率",
                "desc": "日内振幅与增仓幅度双滚动标准化乘积，多空分歧极大",
            },
            "V_04": {
                "formula": "(MEAN(OI,9)-MEAN(OI,26))/MEAN(OI,12)*100",
                "category": "波动率",
                "desc": "持仓量均线差率（OI-MACD柱），资金面预期分歧加大",
            },
            # 资金流类
            "M_01": {
                "formula": "SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*DELTA(OI,1),6)",
                "category": "资金流",
                "desc": "6日日内多空力量与增仓累积，量化多空哪方在主动加仓",
            },
            "M_02": {
                "formula": "SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*DELTA(OI,1),20)",
                "category": "资金流",
                "desc": "20日日内多空力量与增仓累积，M_01中期波段版本",
            },
            "M_03": {
                "formula": "SUM(CLOSE>DELAY(CLOSE,1)?DELTA(OI,1):(CLOSE<DELAY(CLOSE,1)?-DELTA(OI,1):0),20)",
                "category": "资金流",
                "desc": "20日条件增仓累积，中期资金流向能量潮指标",
            },
            "M_04": {
                "formula": "SUM(CARRY_ORTH>0?DELTA(OI,1):-DELTA(OI,1),10)",
                "category": "资金流",
                "desc": "期限结构驱动的资金流（Carry已正交化），贴近商品产业链逻辑",
            },
            "M_05": {
                "formula": "SMA(OI,13,2)-SMA(OI,27,2)-SMA(SMA(OI,13,2)-SMA(OI,27,2),10,2)",
                "category": "资金流",
                "desc": "持仓量MACD指标，金叉/死叉提示资金面拐点",
            },
            # 高阶复合类
            "H_01": {
                "formula": "(MEAN(OI,20)<OI)?(CARRY_ORTH*TSRANK(ABS(DELTA(CLOSE,7)),60)):(-1*OI)",
                "category": "高阶复合",
                "desc": "条件性结构动量（Carry已正交化），增仓交易共振/缩仓退守防守",
            },
            "H_02": {
                "formula": "(-1*ZSCORE(DELTA(CLOSE,7)*(1-ZSCORE(DECAYLINEAR(OI/MEAN(OI,20),9),w)),w))*(1+ZSCORE(SUM(RET,250),w))",
                "category": "高阶复合",
                "desc": "7日价格变化与持仓衰减线性排名复合因子",
            },
            "H_03": {
                "formula": "TSRANK(OI/MEAN(OI,20),20)*TSRANK(-1*DELTA(CLOSE,7),8)",
                "category": "高阶复合",
                "desc": "相对持仓时序排名与反转时序排名乘积，捕捉反弹拐点",
            },
            "H_04": {
                "formula": "(-1*ZSCORE(TSRANK(CLOSE,10),w))*ZSCORE(DELTA(DELTA(CLOSE,1),1),w)*ZSCORE(TSRANK(OI/MEAN(OI,20),5),w)",
                "category": "高阶复合",
                "desc": "价格加速度与相对持仓排名复合，捕捉趋势启动极初期",
            },
            "H_05": {
                "formula": "ZSCORE(CARRY_ORTH,w)*ZSCORE(DELTA(OI,5),w)*SIGN(DELTA(CLOSE,5))",
                "category": "高阶复合",
                "desc": "三重共振因子（Carry已正交化），期限结构+持仓流入+价格突破同向共振",
            },
        }


# 向后兼容别名
AlphaFutures23 = AlphaFutures24
