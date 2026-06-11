"""兼容层 — _bootstrap 已迁移到 core/execution/_bootstrap.py。"""

from core.execution._bootstrap import (  # noqa: F401
    bootstrap_metrics,
    generate_simple_signal,
    compute_simple_metrics,
)
