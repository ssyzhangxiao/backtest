"""
数据加载模块。

委托 core/config/、core/engine/pybroker_data_source.py 等公共系统，
不重复实现数据加载逻辑。
"""

import os
from typing import Any, Dict, Optional, Tuple

import yaml
from loguru import logger
from dotenv import load_dotenv

# 自动加载 .env 文件
load_dotenv()

from core.engine.pybroker_data_source import (
    PyBrokerDataSource,
    create_hybrid_data_source,
)
from runner.common.errors import DataError, ConfigError


class DataLoader:
    """
    数据加载器，统一封装配置加载和数据源创建。

    消除重复#7：直接调用 BacktestConfig.from_yaml()，
    不重新解析 yaml。
    """

    def __init__(self, config_path: str = "config.yaml"):
        self._config_path = config_path
        self._raw_config: Optional[Dict[str, Any]] = None

    @property
    def raw_config(self) -> Optional[Dict[str, Any]]:
        return self._raw_config

    def load(self) -> PyBrokerDataSource:
        """
        加载配置并创建数据源。

        Returns:
            PyBrokerDataSource 实例

        Raises:
            DataError: 数据加载失败
        """
        self._raw_config = load_raw_config(self._config_path)
        phone, password = get_tqsdk_credentials(self._raw_config)
        data_cfg = self._raw_config.get("data", {})
        symbols = self._raw_config.get("symbols")

        try:
            ds = create_hybrid_data_source(
                phone=phone,
                password=password,
                symbols=symbols,
                data_dir=data_cfg.get("csv_data_dir", "data"),
                data_length=data_cfg.get("tqsdk_data_length", 4000),
            )
            pybroker_df = ds.to_pybroker_df()
            if pybroker_df is not None and not pybroker_df.empty:
                if "date" in pybroker_df.columns:
                    data_min = pybroker_df["date"].min()
                    data_max = pybroker_df["date"].max()
                    logger.info(
                        f"  数据日期范围: {data_min} ~ {data_max}, "
                        f"{len(pybroker_df)} 行, "
                        f"{pybroker_df['symbol'].nunique()} 品种"
                    )
            return ds
        except Exception as e:
            raise DataError(f"数据加载失败: {e}") from e


def load_raw_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    加载 YAML 配置文件为原始字典。

    注意：推荐使用 BacktestConfig.from_yaml() 获取结构化配置。
    此函数保留用于需要原始字典的场景。

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典

    Raises:
        ConfigError: 配置文件加载失败
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config: Dict[str, Any] = yaml.safe_load(f)
        return config
    except Exception as e:
        raise ConfigError(f"配置文件加载失败 {config_path}: {e}") from e


def get_tqsdk_credentials(
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    获取天勤 SDK 凭证，优先环境变量，回退 config.yaml。

    Args:
        config: 可选配置字典，未提供时自动加载

    Returns:
        (phone, password) 元组
    """
    phone: Optional[str] = os.getenv("TQSDK_PHONE")
    password: Optional[str] = os.getenv("TQSDK_PASSWORD")
    if not phone or not password:
        try:
            cfg = config or load_raw_config()
            data_cfg = cfg.get("data", {})
            phone = phone or data_cfg.get("tqsdk_phone")
            password = password or data_cfg.get("tqsdk_password")
        except Exception:
            pass
    if not phone or not password:
        logger.warning("TqSdk凭证未设置，将仅使用CSV数据")
    return phone, password
