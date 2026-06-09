"""
跨品种联动因子：成对价差（pair spread）。

输入：两个同产业链品种（A, B），用各自的 close 序列做：
  spread = zscore(close_A) - zscore(close_B)
再 EMA 平滑。

逻辑依据（产业链联动）：
  1. 黑色金属：RB (螺纹钢) - I (铁矿石) → 钢厂利润代理
  2. 能源化工：TA (PTA) - MA (甲醇) → 聚酯链强弱
  3. 油脂链：Y (豆油) - P (棕榈油) → 替代品价差
  4. 贵金属：AU (黄金) - AG (白银) → 金银比反向

如果 spread 走阔（A 强 B 弱）→ 预测 spread 回归（短期反转）：
  factor = -zscore(spread_t)  // 反向信号
如果 spread 走阔延续：
  factor = +zscore(spread_t)  // 趋势信号

默认：反转信号。回归窗口 60 日。
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ..operators import ema, zscore


# 产业链对定义（基于期货品种经验分类 + 历史 IC 强配对）
CHAIN_PAIRS: Dict[str, Tuple[str, str]] = {
    # ── 黑色金属（钢厂利润代理） ──
    "XPRB_I": ("SHFE.RB", "DCE.I"),  # 螺纹-铁矿
    "XPRB_J": ("SHFE.RB", "DCE.J"),  # 螺纹-焦炭
    "XJ_I": ("DCE.J", "DCE.I"),  # 焦炭-铁矿
    # ── 有色金属（强弱分化） ──
    "XCU_ZN": ("SHFE.CU", "SHFE.ZN"),  # 铜-锌
    "XCU_NI": ("SHFE.CU", "SHFE.NI"),  # 铜-镍
    "XAL_CU": ("SHFE.AL", "SHFE.CU"),  # 铝-铜
    "XAU_AG": ("SHFE.AU", "SHFE.AG"),  # 黄金-白银
    # ── 能源化工 ──
    "XPTA_MA": ("CZCE.TA", "CZCE.MA"),  # PTA-甲醇
    "XFU_BU": ("SHFE.FU", "SHFE.BU"),  # 燃料油-沥青
    "XBU_RU": ("SHFE.BU", "SHFE.RU"),  # 沥青-橡胶
    # ── 油脂链 ──
    "XY_P": ("DCE.Y", "DCE.P"),  # 豆油-棕榈油
    "XY_M": ("DCE.M", "DCE.Y"),  # 豆粕-豆油
    # ── 玉米链 ──
    "XCS_C": ("DCE.CS", "DCE.C"),  # 玉米淀粉-玉米
    "XJD_C": ("DCE.JD", "DCE.C"),  # 鸡蛋-玉米
}


# 强 IC 配对（用于 factor_combo_ic 候选池）。
# 来源：scratch_cross_spread.py 历史时序 IC 扫描结果。
# 注意：这里只列出通过 |IC| > 0.02 阈值的稳定配对。
STRONG_IC_PAIRS: Tuple[str, ...] = (
    "XPRB_I",  # 螺纹-铁矿（钢厂利润代理）
    "XPRB_J",  # 螺纹-焦炭
    "XCU_ZN",  # 铜-锌
    "XCU_NI",  # 铜-镍
    "XAU_AG",  # 金银比
    "XFU_BU",  # 燃料油-沥青
)


def compute_pair_spread_factor(
    close_a: np.ndarray,
    close_b: np.ndarray,
    spread_window: int = 60,
    smoothing_window: int = 3,
    direction: str = "revert",  # "revert"=反转信号, "trend"=趋势信号
) -> np.ndarray:
    """
    计算跨品种价差因子（配对品种A vs B）。

    步骤：
      1. 对齐 A 和 B（取交集）
      2. 计算 zscore(close_A) - zscore(close_B)
      3. 对价差做时序 zscore（rolling_window=spread_window）
      4. EMA 平滑（默认 3 日）
      5. 反转信号：取负号

    Args:
        close_a: 品种 A 的 close 序列
        close_b: 品种 B 的 close 序列
        spread_window: 价差标准化窗口
        smoothing_window: EMA 平滑窗口
        direction: "revert"→反转信号，价差偏离 → 预测回归；
                   "trend"→趋势信号，价差走阔 → 预测延续

    Returns:
        与 close_a 等长的因子值序列
    """
    a = np.asarray(close_a, dtype=float)
    b = np.asarray(close_b, dtype=float)
    n = min(len(a), len(b))
    a = a[-n:]
    b = b[-n:]

    # 1. 横截面 zscore（截面=品种间）
    a_z = zscore(a) if a.std() > 1e-8 else np.zeros_like(a)
    b_z = zscore(b) if b.std() > 1e-8 else np.zeros_like(b)
    spread = a_z - b_z

    # 2. 时序 zscore（滚动）
    spread_t = pd.Series(spread)
    rolling_mean = spread_t.rolling(spread_window, min_periods=10).mean()
    rolling_std = spread_t.rolling(spread_window, min_periods=10).std()
    safe_std = rolling_std.replace(0, np.nan)
    spread_norm = ((spread_t - rolling_mean) / safe_std).to_numpy()

    # 3. EMA 平滑
    smoothed = ema(spread_norm, window=smoothing_window)

    # 4. 反转 / 趋势
    if direction == "revert":
        return -smoothed
    return smoothed


def list_available_pairs() -> list:
    """列出所有预定义产业链对。"""
    return list(CHAIN_PAIRS.keys())
