"""
自定义异常类。

Pipeline 流程中使用的异常层次，便于区分错误来源。
"""


class PipelineError(Exception):
    """Pipeline 流程基础异常。"""
    pass


class ConfigError(PipelineError):
    """配置相关异常（yaml 解析、字段缺失、类型不匹配等）。"""
    pass


class DataError(PipelineError):
    """数据加载相关异常。"""
    pass


class BacktestError(PipelineError):
    """回测执行相关异常。"""
    pass


class OptimizationError(PipelineError):
    """参数优化相关异常。"""
    pass


class ValidationError(PipelineError):
    """验证流程相关异常。"""
    pass
