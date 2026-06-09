"""
测试 utils.py 模块
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from runner.common.utils import (
    safe_float,
    is_valid_number,
    safe_div,
    sanitize_filename,
    handle_backtest_errors,
)


class TestUtils:
    """工具函数测试"""
    
    def test_safe_float_valid(self):
        """测试有效 float 转换"""
        assert safe_float("123.45") == 123.45
        assert safe_float(678) == 678.0
        assert safe_float(9.10) == 9.10
    
    def test_safe_float_invalid(self):
        """测试无效 float 转换"""
        assert safe_float("not a number") == 0.0
        assert safe_float(None) == 0.0
        assert safe_float([]) == 0.0
    
    def test_is_valid_number(self):
        """测试有效数字检查"""
        assert is_valid_number(123) is True
        assert is_valid_number(123.45) is True
        assert is_valid_number("123.45") is True
        
        assert is_valid_number(np.nan) is False
        assert is_valid_number(np.inf) is False
        assert is_valid_number(-np.inf) is False
        assert is_valid_number("not a number") is False
        assert is_valid_number(None) is False
    
    def test_safe_div(self):
        """测试安全除法"""
        assert safe_div(10, 2) == 5.0
        assert safe_div(10, 0) == 0.0
        assert safe_div(0, 10) == 0.0
        assert safe_div(10, 0.0000001) == 100000000.0
    
    def test_sanitize_filename(self):
        """测试文件名清理"""
        assert sanitize_filename("test/file.txt") == "test_file.txt"
        assert sanitize_filename('test"file.txt') == "test_file.txt"
        assert sanitize_filename("test file.txt") == "test file.txt"
        assert sanitize_filename("..") == "_"
        assert sanitize_filename("") == "unnamed"
        assert sanitize_filename("test\nfile") == "testfile"


class TestHandleBacktestErrors:
    """测试错误处理装饰器"""
    
    def test_handle_backtest_errors_normal_execution(self):
        """测试正常执行"""
        
        @handle_backtest_errors(return_value="default")
        def test_func(x):
            return x * 2
        
        assert test_func(5) == 10
    
    def test_handle_backtest_errors_catch_exception(self):
        """测试捕获异常"""
        
        @handle_backtest_errors(return_value="error_value")
        def test_func():
            raise ValueError("Something went wrong")
        
        assert test_func() == "error_value"
    
    def test_handle_backtest_errors_reraise_keyboard_interrupt(self):
        """测试重新抛出 KeyboardInterrupt"""
        
        @handle_backtest_errors(return_value="default")
        def test_func():
            raise KeyboardInterrupt()
        
        with pytest.raises(KeyboardInterrupt):
            test_func()
    
    def test_handle_backtest_errors_reraise_system_exit(self):
        """测试重新抛出 SystemExit"""
        
        @handle_backtest_errors(return_value="default")
        def test_func():
            raise SystemExit()
        
        with pytest.raises(SystemExit):
            test_func()
    
    def test_handle_backtest_errors_custom_reraise_types(self):
        """测试自定义重新抛出的异常类型"""
        
        @handle_backtest_errors(return_value="default", reraise_types=(ValueError,))
        def test_func(raise_val_error):
            if raise_val_error:
                raise ValueError()
            raise TypeError()
        
        with pytest.raises(ValueError):
            test_func(True)
        
        assert test_func(False) == "default"
    
    def test_handle_backtest_errors_log_error(self):
        """测试记录错误日志"""
        with patch("runner.common.utils.logger") as mock_logger:
            
            @handle_backtest_errors(return_value="default", log_error=True)
            def test_func():
                raise ValueError("Test error")
            
            test_func()
            mock_logger.error.assert_called_once()
