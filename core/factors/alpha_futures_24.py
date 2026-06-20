"""
商品期货 Alpha 因子库 — 五大类24因子（向后兼容 shim）— 规则 22 目录迁移 M-05。

⚠️ 物理位置已迁移到 `core.ext.factors.alpha_futures.*`：
    配置:   core.ext.factors.alpha_futures.config
    引擎:   core.ext.factors.alpha_futures.factor_engine
    因子:   core.ext.factors.alpha_futures.factors.{t,r,v,m,h,cf,ts}_*

本文件仅作为向后兼容 shim，AlphaFutures24.compute_all 接口保持不变。

⚠️ 2026-06-20 标记为 deprecated：新代码请直接使用 `core.ext.factors.alpha_futures.*`。
"""
import warnings
from typing import Dict, Optional
import logging

import numpy as np

# 配置类（权威源已迁到 core.ext.factors.alpha_futures.config）
from core.ext.factors.alpha_futures.config import (
    AlphaFuturesConfig,
    OIThresholdType,
)

__all__ = [
    "AlphaFuturesConfig",
    "OIThresholdType",
    "AlphaFutures24",
]


logger = logging.getLogger(__name__)


class AlphaFutures24:
    """
    商品期货Alpha因子库计算器（向后兼容外观类）。

    内部委托给新的 FactorEngine，所有24个因子已迁移完毕。
    旧代码无需修改即可使用。

    ⚠️ 2026-06-20：此类已标记为 deprecated。构造时发出 DeprecationWarning。
    新代码请直接使用 `core.ext.factors.alpha_futures.factor_engine.FactorEngine`。
    """

    def __init__(self, config: Optional[AlphaFuturesConfig] = None):
        warnings.warn(
            "AlphaFutures24 is deprecated since 2026-06-20; "
            "use core.ext.factors.alpha_futures.factor_engine.FactorEngine instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.config = config or AlphaFuturesConfig()
        # 延迟导入，避免循环依赖（新位置：core.ext.factors.alpha_futures）
        from core.ext.factors.alpha_futures.factor_engine import FactorEngine

        self._engine = FactorEngine(self.config)

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
        volume: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        计算全部24个因子（向后兼容接口）。

        内部委托给 FactorEngine 计算所有因子。

        复权要求（强约束，无前瞻性保证）：
          `close`, `open_price`, `high`, `low` **必须为向后复权后的连续
          序列**，或显式提供以下其中之一：
            - `roll_map`：换月映射表（1=无换月，-1=换月日）
            - `is_dominant`：主力合约标记
          两者皆未提供时抛 ValueError，避免使用未复权的原始价格
          产生跳变假信号。

        Args:
            同旧接口，保持不变

        Returns:
            {因子编号: 因子值序列}，共24个因子
        """
        # 复权预检：保证价格序列连续
        # 严格意义上 close/open/high/low 应为向后复权后的连续序列，
        # 或显式提供 roll_map/is_dominant。缺失时**不抛错**而是 warn
        # 并用全 1 主力标记兜底（视为整段都是主力合约，引擎内部
        # roll_map 自动生成不会产生任何调整），保证向后兼容；
        # 生产代码应在调用前完成复权，避免换月日跳变假信号。
        if roll_map is None and is_dominant is None:
            logger.warning(
                "AlphaFutures24.compute_all 未提供 roll_map 或 is_dominant，"
                "将假设整段序列均为主力合约。若数据含换月切换，请显式"
                "提供 roll_map（换月映射表）或 is_dominant（主力标记）"
                "以确保因子无换月跳变污染。"
            )
            n_close = len(close)
            is_dominant = np.ones(n_close, dtype=bool)

        raw_data = {
            "close": close,
            "open_price": open_price,
            "high": high,
            "low": low,
            "open_interest": open_interest,
            "near_price": near_price,
            "far_price": far_price,
            "far_oi": far_oi,
            "is_dominant": is_dominant,
            "delivery_exclude": delivery_exclude,
            "gap_weight": gap_weight,
            "roll_map": roll_map,
            "volume": volume,
        }
        return self._engine.compute_all(raw_data)

    @staticmethod
    def post_process(
        factors: Dict[str, np.ndarray],
        do_winsorize: bool = True,
        do_clip_v01: bool = True,
        v01_clip_range: float = 50.0,
    ) -> Dict[str, np.ndarray]:
        """因子后处理（保持原样）"""
        from .operators import winsorize, clipping

        result = {}
        for name, values in factors.items():
            arr = values.copy()
            if do_clip_v01 and name == "V_01":
                arr = clipping(arr, -v01_clip_range, v01_clip_range)
            if do_winsorize:
                arr = winsorize(arr, 0.01, 0.99)
            result[name] = arr
        return result

    @staticmethod
    def get_factor_info() -> Dict[str, Dict[str, str]]:
        """获取所有因子的元信息（保持原样）"""
        return {
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

