"""数据加载器 — 常量与品种映射。"""

from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# 缓存目录配置
# ---------------------------------------------------------------------------
CACHE_DIR = Path(__file__).parent.parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 品种 → (TqSdk 交易所, TqSdk 品种代码) 完整映射
# ---------------------------------------------------------------------------
PRODUCT_EXCHANGE_MAP: Dict[str, Tuple[str, str]] = {
    # 上期所 SHFE
    "SHFE.RB": ("SHFE", "rb"),
    "SHFE.AG": ("SHFE", "ag"),
    "SHFE.AU": ("SHFE", "au"),
    "SHFE.AL": ("SHFE", "al"),
    "SHFE.ZN": ("SHFE", "zn"),
    "SHFE.CU": ("SHFE", "cu"),
    "SHFE.NI": ("SHFE", "ni"),
    "SHFE.SN": ("SHFE", "sn"),
    "SHFE.PB": ("SHFE", "pb"),
    "SHFE.HC": ("SHFE", "hc"),
    "SHFE.BU": ("SHFE", "bu"),
    "SHFE.RU": ("SHFE", "ru"),
    "SHFE.SS": ("SHFE", "ss"),
    "SHFE.SP": ("SHFE", "sp"),
    "SHFE.BR": ("SHFE", "br"),
    "SHFE.AO": ("SHFE", "ao"),
    # 大商所 DCE
    "DCE.M": ("DCE", "m"),
    "DCE.I": ("DCE", "i"),
    "DCE.J": ("DCE", "j"),
    "DCE.JM": ("DCE", "jm"),
    "DCE.C": ("DCE", "c"),
    "DCE.CS": ("DCE", "cs"),
    "DCE.A": ("DCE", "a"),
    "DCE.B": ("DCE", "b"),
    "DCE.P": ("DCE", "p"),
    "DCE.Y": ("DCE", "y"),
    "DCE.L": ("DCE", "l"),
    "DCE.PP": ("DCE", "pp"),
    "DCE.V": ("DCE", "v"),
    "DCE.EB": ("DCE", "eb"),
    "DCE.EG": ("DCE", "eg"),
    "DCE.PG": ("DCE", "pg"),
    "DCE.JD": ("DCE", "jd"),
    "DCE.LH": ("DCE", "lh"),
    # 郑商所 CZCE
    "CZCE.TA": ("CZCE", "TA"),
    "CZCE.MA": ("CZCE", "MA"),
    "CZCE.FG": ("CZCE", "FG"),
    "CZCE.SA": ("CZCE", "SA"),
    "CZCE.SF": ("CZCE", "SF"),
    "CZCE.SM": ("CZCE", "SM"),
    "CZCE.CF": ("CZCE", "CF"),
    "CZCE.SR": ("CZCE", "SR"),
    "CZCE.OI": ("CZCE", "OI"),
    "CZCE.RM": ("CZCE", "RM"),
    "CZCE.PF": ("CZCE", "PF"),
    "CZCE.PX": ("CZCE", "PX"),
    "CZCE.SH": ("CZCE", "SH"),
    "CZCE.UR": ("CZCE", "UR"),
    "CZCE.ZC": ("CZCE", "ZC"),
    "CZCE.AP": ("CZCE", "AP"),
    "CZCE.CY": ("CZCE", "CY"),
    "CZCE.PK": ("CZCE", "PK"),
    # 中金所 CFFEX
    "CFFEX.IF": ("CFFEX", "IF"),
    "CFFEX.IC": ("CFFEX", "IC"),
    "CFFEX.IH": ("CFFEX", "IH"),
    "CFFEX.IM": ("CFFEX", "IM"),
    "CFFEX.T": ("CFFEX", "T"),
    "CFFEX.TF": ("CFFEX", "TF"),
    "CFFEX.TS": ("CFFEX", "TS"),
    # 能源中心 INE
    "INE.SC": ("INE", "sc"),
    "INE.NR": ("INE", "nr"),
    "INE.BC": ("INE", "bc"),
    "INE.LU": ("INE", "lu"),
    "INE.EC": ("INE", "ec"),
    # 广期所 GFEX
    "GFEX.LC": ("GFEX", "LC"),
    "GFEX.SI": ("GFEX", "SI"),
}


# ---------------------------------------------------------------------------
# 默认加载的核心品种
# ---------------------------------------------------------------------------
DEFAULT_SYMBOLS: List[str] = [
    "SHFE.RB",
    "SHFE.HC",
    "SHFE.AU",
    "SHFE.AG",
    "SHFE.CU",
    "DCE.M",
    "DCE.I",
    "DCE.J",
    "DCE.JM",
    "DCE.C",
    "DCE.P",
    "DCE.Y",
    "DCE.EG",
    "DCE.PP",
    "DCE.L",
    "CZCE.TA",
    "CZCE.MA",
    "CZCE.FG",
    "CZCE.SA",
    "CZCE.CF",
    "CZCE.OI",
    "CZCE.RM",
    "CZCE.SR",
    "CZCE.ZC",
    "CFFEX.IF",
    "CFFEX.IC",
    "CFFEX.IH",
    "INE.SC",
    "INE.NR",
]


# ---------------------------------------------------------------------------
# 常量配置
# ---------------------------------------------------------------------------
DAILY_SECONDS = 86400  # 86400 秒 = 1 天，用于 TqSdk K 线日线周期
# 60 个合约 ≈ 5 年覆盖：
#   - SHFE/DCE（每月合约）: 60 × 1 month = 5 年
#   - CZCE（双月合约）: 60 × 2 months = 10 年
# 20 个合约只覆盖 20 个月，无法支持 3 年 OOS 验证。
MAX_CONTRACTS_PER_PRODUCT = 60  # 每个品种最多下载的合约数
