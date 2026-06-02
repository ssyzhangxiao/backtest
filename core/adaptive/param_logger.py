"""
参数变更日志。

记录每次参数变更的时间、原因及市场环境特征。
输出格式：JSON Lines，便于回测复现和审计。

规则10要求：所有参数变更必须记录日志。
"""

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ParamChangeRecord:
    """参数变更记录。"""

    timestamp: str
    param_name: str
    old_value: Any
    new_value: Any
    trigger_reason: str
    market_regime: str = ""
    vol_percentile: float = 0.0
    day_index: int = 0


class ParamChangeLogger:
    """
    参数变更日志记录器。

    以JSON Lines格式记录参数变更，每行一条记录。
    便于回测复现和参数调整审计。

    用法:
        logger = ParamChangeLogger("logs/param_changes.jsonl")
        logger.log("ema_window", 5, 3, "regime切换→HIGH", "high", 0.85)
    """

    def __init__(self, log_path: str = "logs/param_changes.jsonl"):
        """
        初始化参数变更日志。

        Args:
            log_path: 日志文件路径
        """
        self.log_path = log_path
        self._records: List[ParamChangeRecord] = []

        # 确保目录存在
        log_dir = os.path.dirname(log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

    def log(
        self,
        param_name: str,
        old_value: Any,
        new_value: Any,
        trigger_reason: str,
        market_regime: str = "",
        vol_percentile: float = 0.0,
        day_index: int = 0,
    ) -> None:
        """
        记录一次参数变更。

        Args:
            param_name: 参数名称
            old_value: 旧值
            new_value: 新值
            trigger_reason: 触发原因
            market_regime: 市场波动率regime
            vol_percentile: 波动率分位数
            day_index: 交易日索引
        """
        record = ParamChangeRecord(
            timestamp=datetime.now().isoformat(),
            param_name=param_name,
            old_value=old_value,
            new_value=new_value,
            trigger_reason=trigger_reason,
            market_regime=market_regime,
            vol_percentile=vol_percentile,
            day_index=day_index,
        )

        self._records.append(record)
        self._write_record(record)

        logger.debug(
            f"参数变更：{param_name} {old_value}→{new_value} "
            f"原因={trigger_reason} regime={market_regime}"
        )

    def _write_record(self, record: ParamChangeRecord) -> None:
        """将单条记录追加写入文件。"""
        try:
            line = json.dumps(asdict(record), ensure_ascii=False)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except (IOError, OSError) as e:
            logger.warning(f"参数变更日志写入失败：{e}")

    def get_records(self, param_name: Optional[str] = None) -> List[ParamChangeRecord]:
        """
        获取变更记录。

        Args:
            param_name: 可选，按参数名过滤

        Returns:
            变更记录列表
        """
        if param_name is None:
            return list(self._records)
        return [r for r in self._records if r.param_name == param_name]

    def load_from_file(self) -> List[ParamChangeRecord]:
        """
        从文件加载历史记录。

        Returns:
            变更记录列表
        """
        records: List[ParamChangeRecord] = []
        if not os.path.exists(self.log_path):
            return records

        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    records.append(ParamChangeRecord(**data))
        except (IOError, json.JSONDecodeError) as e:
            logger.warning(f"参数变更日志读取失败：{e}")

        return records

    def clear(self) -> None:
        """清空内存中的记录（不删除文件）。"""
        self._records.clear()
