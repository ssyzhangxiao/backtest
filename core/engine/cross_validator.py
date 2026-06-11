"""兼容层 — CrossValidator 已迁移到 core/validation/cross_validator.py。"""

import warnings
from core.validation.cross_validator import CrossValidator  # noqa: F401

warnings.warn(
    "请使用 from core.validation.cross_validator import CrossValidator",
    DeprecationWarning,
    stacklevel=2,
)
