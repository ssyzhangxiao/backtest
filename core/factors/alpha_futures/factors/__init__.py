"""
独立因子类集合。

所有24个因子类在此模块中导入并自动注册。
"""
# 趋势类因子 T_01~T_05
from .t_01 import T_01
from .t_02 import T_02
from .t_03 import T_03
from .t_04 import T_04
from .t_05 import T_05

# 回归类因子 R_01~R_05
from .r_01 import R_01
from .r_02 import R_02
from .r_03 import R_03
from .r_04 import R_04
from .r_05 import R_05

# 波动率类因子 V_01~V_04
from .v_01 import V_01
from .v_02 import V_02
from .v_03 import V_03
from .v_04 import V_04

# 资金流类因子 M_01~M_05
from .m_01 import M_01
from .m_02 import M_02
from .m_03 import M_03
from .m_04 import M_04
from .m_05 import M_05

# 高阶复合类因子 H_01~H_05
from .h_01 import H_01
from .h_02 import H_02
from .h_03 import H_03
from .h_04 import H_04
from .h_05 import H_05

# 资金流扩展因子 CF_01~CF_03（源自 capital_flow.py）
from .cf_01 import CF_01
from .cf_02 import CF_02
from .cf_03 import CF_03

# 期限结构扩展因子 TS_01~TS_03（源自 term_structure.py）
from .ts_01 import TS_01
from .ts_02 import TS_02
from .ts_03 import TS_03

__all__ = [
    "T_01", "T_02", "T_03", "T_04", "T_05",
    "R_01", "R_02", "R_03", "R_04", "R_05",
    "V_01", "V_02", "V_03", "V_04",
    "M_01", "M_02", "M_03", "M_04", "M_05",
    "H_01", "H_02", "H_03", "H_04", "H_05",
    "CF_01", "CF_02", "CF_03",
    "TS_01", "TS_02", "TS_03",
]