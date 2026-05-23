"""
统一公共回测脚本。

通过调用现有系统中已实现的功能模块（严禁重写现有代码），
提供标准化的回测入口。支持：

- 多策略配置与选择
- 多种数据源和格式
- 自定义参数化回测
- 绩效指标计算与统计分析
- 标准化结果可视化与报告生成

使用方式：
    python unified_backtest.py                          # 交互式菜单
    python unified_backtest.py --config config.json     # 使用配置文件
    python unified_backtest.py --strategy dual_ma       # 指定策略并交互配置

新增回测要求时的扩展流程：
    1. 在 STRATEGY_SCENARIOS 中注册策略场景（或通过 --config 传入）
    2. 确保策略类已在 core/strategies/__init__.py 中注册
    3. 运行脚本，系统自动完成数据加载、回测、分析和可视化
"""

import sys
import os
import json
import logging
import argparse
import traceback
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import pybroker
from pybroker import Strategy, StrategyConfig, FeeMode

warnings.filterwarnings('ignore')

from core.data_loader import DataLoader
from core.environment import EnvironmentAdapter
from core.strategies import (
    STRATEGY_REGISTRY,
    get_strategy_class,
    create_strategy,
    BaseStrategy,
)
from utils.metrics import MetricsCalculator


# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None):
    logger = logging.getLogger("UnifiedBacktest")
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = setup_logging()


# ---------------------------------------------------------------------------
# 配置数据结构
# ---------------------------------------------------------------------------

@dataclass
class BacktestGlobalConfig:
    """回测全局配置。"""
    initial_cash: float = 1_000_000
    commission: float = 0.0003
    slippage: float = 0.0002
    max_long_positions: float = 0.6
    max_short_positions: float = 0.6
    data_dir: str = "./data"
    file_pattern: str = "*.csv"


@dataclass
class DataFilterConfig:
    """数据筛选配置。"""
    symbols: Optional[List[str]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    compute_environment: bool = True
    trend_threshold: float = 30.0


@dataclass
class PeriodConfig:
    """回测时段配置。"""
    label: str = "全量"
    start_date: Optional[str] = None
    end_date: Optional[str] = None


@dataclass
class StrategyScenario:
    """策略场景配置。

    一个场景 = 一个策略 + 一组参数，可跨多个时段回测。
    """
    strategy_id: str
    strategy_name: str = ""
    description: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    periods: Optional[List[PeriodConfig]] = None


@dataclass
class BacktestJobConfig:
    """一次回测任务的完整配置。"""
    global_config: BacktestGlobalConfig = field(default_factory=BacktestGlobalConfig)
    data_filter: DataFilterConfig = field(default_factory=DataFilterConfig)
    scenarios: List[StrategyScenario] = field(default_factory=list)
    output_dir: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "global_config": asdict(self.global_config),
            "data_filter": asdict(self.data_filter),
            "scenarios": [
                {
                    "strategy_id": s.strategy_id,
                    "strategy_name": s.strategy_name,
                    "description": s.description,
                    "params": s.params,
                    "periods": [asdict(p) for p in (s.periods or [])],
                }
                for s in self.scenarios
            ],
            "output_dir": self.output_dir,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "BacktestJobConfig":
        return cls(
            global_config=BacktestGlobalConfig(**data.get("global_config", {})),
            data_filter=DataFilterConfig(**data.get("data_filter", {})),
            scenarios=[
                StrategyScenario(
                    strategy_id=s["strategy_id"],
                    strategy_name=s.get("strategy_name", s["strategy_id"]),
                    description=s.get("description", ""),
                    params=s.get("params", {}),
                    periods=[PeriodConfig(**p) for p in s.get("periods", [])]
                    if s.get("periods") else None,
                )
                for s in data.get("scenarios", [])
            ],
            output_dir=data.get("output_dir"),
        )


# ---------------------------------------------------------------------------
# 预定义策略场景
# ---------------------------------------------------------------------------

STRATEGY_SCENARIOS: Dict[str, StrategyScenario] = {
    "dual_ma_original": StrategyScenario(
        strategy_id="dual_ma",
        strategy_name="双均线-原策略",
        description="均线交叉出场，ADX趋势过滤",
        params={
            "short_ma": 5, "long_ma": 20, "adx_threshold": 30.0,
            "position_size": 0.3, "trailing_stop_pct": None, "time_stop_days": None,
        },
    ),
    "dual_ma_trailing": StrategyScenario(
        strategy_id="dual_ma",
        strategy_name="双均线-追踪止损",
        description="5%追踪止损 + 20天时间止损",
        params={
            "short_ma": 5, "long_ma": 20, "adx_threshold": 30.0,
            "position_size": 0.3, "trailing_stop_pct": 0.05, "time_stop_days": 20,
        },
    ),
    "term_structure": StrategyScenario(
        strategy_id="term_structure",
        strategy_name="期限结构套利",
        description="基于价格偏离长期均值的均值回归策略+5%跟踪止损",
        params={
            "lookback": 60, "entry_threshold": 5.0, "exit_threshold": 1.0,
            "position_size": 0.2, "trailing_stop_pct": 0.05, "time_stop_days": None,
        },
    ),
    "vol_breakout": StrategyScenario(
        strategy_id="vol_breakout",
        strategy_name="波动率突破",
        description="基于ATR通道的波动率突破策略+3*ATR跟踪止损",
        params={
            "atr_period": 20, "band_period": 20, "atr_multiplier": 2.0,
            "position_size": 0.2, "trailing_stop_atr_mult": 3.0, "time_stop_days": None,
        },
    ),
}

PREBUILT_SCENARIOS: Dict[str, List[str]] = {
    "dual_ma_comparison": ["dual_ma_original", "dual_ma_trailing"],
    "new_strategies": ["term_structure", "vol_breakout"],
    "all": list(STRATEGY_SCENARIOS.keys()),
}


# ---------------------------------------------------------------------------
# 数据管理器
# ---------------------------------------------------------------------------

class DataManager:
    """
    数据管理模块。

    封装 DataLoader 和 EnvironmentAdapter 的调用，
    提供统一的数据加载和预处理接口。
    """

    def __init__(self, config: BacktestGlobalConfig, data_filter: DataFilterConfig):
        self._config = config
        self._filter = data_filter
        self._df: Optional[pd.DataFrame] = None
        self._data_info: Dict[str, Any] = {}

    def load_and_prepare(self) -> Tuple[pd.DataFrame, Dict]:
        """
        加载并准备回测数据。

        Returns:
            (pybroker_df, data_info):
                pybroker_df - 回测用DataFrame
                data_info   - 数据信息字典
        """
        logger.info("=" * 60)
        logger.info("数据加载与预处理")
        logger.info("=" * 60)

        loader = DataLoader(self._config.data_dir)
        loader.load_csv_files(file_pattern=self._config.file_pattern)
        loader.identify_dominant_contracts()
        loader.build_continuous_series()

        df = loader.get_pybroker_df()

        if self._filter.symbols:
            df = df[df["symbol"].isin(self._filter.symbols)].copy()
            logger.info(f"品种筛选: {self._filter.symbols}")

        df["date"] = pd.to_datetime(df["date"])

        if self._filter.start_date:
            start = pd.to_datetime(self._filter.start_date)
            df = df[df["date"] >= start]
        if self._filter.end_date:
            end = pd.to_datetime(self._filter.end_date)
            df = df[df["date"] <= end]

        df = df.reset_index(drop=True)

        if self._filter.compute_environment:
            env_adapter = EnvironmentAdapter(
                trend_threshold=self._filter.trend_threshold
            )
            df = env_adapter.compute_for_pybroker(df)
            pybroker.register_columns(
                "open_interest", "is_dominant", "env_atr", "env_adx",
                "env_plus_di", "env_minus_di", "env_market_regime",
                "env_trend_score", "env_weight_trend", "env_weight_reversal",
            )
            logger.info("环境指标计算完成")

        symbols = df["symbol"].unique().tolist()
        date_min = df["date"].min()
        date_max = df["date"].max()

        self._data_info = {
            "symbols": symbols,
            "date_range": (str(date_min)[:10], str(date_max)[:10]),
            "total_rows": len(df),
        }
        self._df = df

        logger.info(f"品种: {symbols}")
        logger.info(f"时间范围: {self._data_info['date_range'][0]} ~ {self._data_info['date_range'][1]}")
        logger.info(f"数据行数: {self._data_info['total_rows']:,}")

        return df, self._data_info

    def get_subset(self, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        """获取数据子集（按日期范围筛选）。"""
        if self._df is None:
            raise RuntimeError("数据尚未加载，请先调用 load_and_prepare()")
        df = self._df
        if start:
            df = df[df["date"] >= pd.to_datetime(start)]
        if end:
            df = df[df["date"] <= pd.to_datetime(end)]
        return df.copy()

    @property
    def data_info(self) -> Dict:
        return self._data_info


# ---------------------------------------------------------------------------
# 策略选择器
# ---------------------------------------------------------------------------

class StrategySelector:
    """
    策略选择模块。

    封装 STRATEGY_REGISTRY 和 create_strategy 的调用，
    提供策略发现、验证和实例化功能。
    """

    @staticmethod
    def list_available() -> List[str]:
        """列出所有可用策略ID。"""
        return sorted(STRATEGY_REGISTRY.keys())

    @staticmethod
    def resolve_scenario(scenario_key: str) -> List[StrategyScenario]:
        """
        解析策略场景。

        支持：
        - 预置场景组名（如 'dual_ma_comparison', 'all'）
        - 单个策略ID（如 'dual_ma'）
        - 预置场景名（如 'term_structure'）

        Args:
            scenario_key: 场景键名

        Returns:
            策略场景列表
        """
        if scenario_key in PREBUILT_SCENARIOS:
            keys = PREBUILT_SCENARIOS[scenario_key]
            return [STRATEGY_SCENARIOS[k] for k in keys if k in STRATEGY_SCENARIOS]

        if scenario_key in STRATEGY_SCENARIOS:
            return [STRATEGY_SCENARIOS[scenario_key]]

        if scenario_key in STRATEGY_REGISTRY:
            return [StrategyScenario(
                strategy_id=scenario_key,
                strategy_name=scenario_key,
            )]

        raise ValueError(
            f"未知场景 '{scenario_key}'。"
            f"可用预置组: {list(PREBUILT_SCENARIOS.keys())}。"
            f"可用策略: {list(STRATEGY_REGISTRY.keys())}。"
        )

    @staticmethod
    def create_instance(strategy_id: str, params: Optional[Dict] = None) -> BaseStrategy:
        """
        根据策略ID和参数创建策略实例。

        Args:
            strategy_id: 策略标识
            params: 策略参数字典

        Returns:
            策略实例
        """
        kwargs = params or {}
        logger.info(f"创建策略实例: {strategy_id}, 参数: {kwargs}")
        return create_strategy(strategy_id, **kwargs)

    @staticmethod
    def get_strategy_info(strategy_id: str) -> Dict:
        """
        获取策略信息。

        Args:
            strategy_id: 策略标识

        Returns:
            策略信息字典
        """
        cls = get_strategy_class(strategy_id)
        info = {
            "id": strategy_id,
            "class": cls.__name__,
            "module": cls.__module__,
        }
        if cls.__doc__:
            info["doc"] = cls.__doc__.strip().split("\n")[0]
        return info


# ---------------------------------------------------------------------------
# 回测执行器
# ---------------------------------------------------------------------------

class BacktestRunner:
    """
    回测执行模块。

    整合 PyBroker 回测引擎，支持参数化配置。
    可执行单次回测或批量回测。
    """

    def __init__(self, config: BacktestGlobalConfig):
        self._config = config

    def run_single(
        self,
        df: pd.DataFrame,
        strategy_id: str,
        params: Dict,
        period_label: str = "",
    ) -> Dict:
        """
        运行单次回测。

        Args:
            df: 回测数据
            strategy_id: 策略标识
            params: 策略参数
            period_label: 时段标签

        Returns:
            回测结果字典 {
                'strategy_id': str,
                'strategy_name': str,
                'period': str,
                'params': dict,
                'result': PyBroker TestResult,
                'metrics': dict,
                'portfolio': DataFrame,
                'trades': DataFrame,
            }
        """
        strat_name = f"{strategy_id}"
        if period_label:
            strat_name += f" [{period_label}]"
        logger.info(f"开始回测: {strat_name}")

        strategy_config = StrategyConfig(
            initial_cash=self._config.initial_cash,
            fee_mode=FeeMode.ORDER_PERCENT,
            fee_amount=self._config.commission + self._config.slippage,
            max_long_positions=self._config.max_long_positions,
            max_short_positions=self._config.max_short_positions,
        )

        strategy = Strategy(
            df,
            str(df["date"].min().date()),
            str(df["date"].max().date()),
            strategy_config,
        )

        strat_instance = StrategySelector.create_instance(strategy_id, params)
        indicators = strat_instance.register_indicators()
        symbols = df["symbol"].unique().tolist()
        strategy.add_execution(
            fn=strat_instance.execute, symbols=symbols, indicators=indicators
        )

        result = strategy.backtest()

        metrics_calc = MetricsCalculator()
        pybroker_metrics = metrics_calc.extract_from_pybroker_result(result)
        additional_metrics = metrics_calc.calculate_additional_metrics(
            portfolio_df=result.portfolio, trades_df=result.trades,
        )
        all_metrics = {**pybroker_metrics, **additional_metrics}

        portfolio_df = result.portfolio.copy() if result.portfolio is not None else None
        trades_df = result.trades.copy() if result.trades is not None else None

        logger.info(f"回测完成: {strat_name} | "
                     f"总收益={all_metrics.get('total_return_pct', all_metrics.get('annual_return_pct', 0)):.2f}% | "
                     f"Sharpe={all_metrics.get('sharpe_ratio', all_metrics.get('sharpe', 0)):.4f}")

        return {
            "strategy_id": strategy_id,
            "strategy_name": strat_name,
            "period": period_label,
            "params": params,
            "result": result,
            "metrics": all_metrics,
            "portfolio": portfolio_df,
            "trades": trades_df,
        }

    def run_scenario(
        self,
        df: pd.DataFrame,
        scenario: StrategyScenario,
    ) -> List[Dict]:
        """
        运行一个策略场景的所有时段回测。

        Args:
            df: 全量数据
            scenario: 策略场景配置

        Returns:
            回测结果列表
        """
        results = []
        periods = scenario.periods

        if not periods:
            result = self.run_single(df, scenario.strategy_id, scenario.params)
            result["strategy_name"] = scenario.strategy_name or result["strategy_name"]
            results.append(result)
        else:
            for period in periods:
                subset = df
                if period.start_date:
                    subset = subset[subset["date"] >= pd.to_datetime(period.start_date)]
                if period.end_date:
                    subset = subset[subset["date"] <= pd.to_datetime(period.end_date)]

                if len(subset) == 0:
                    logger.warning(f"时段 '{period.label}' 无数据，跳过")
                    continue

                result = self.run_single(
                    subset, scenario.strategy_id, scenario.params, period.label,
                )
                result["strategy_name"] = scenario.strategy_name or result["strategy_name"]
                results.append(result)

        return results

    def run_all(
        self,
        df: pd.DataFrame,
        scenarios: List[StrategyScenario],
    ) -> List[Dict]:
        """
        运行所有策略场景的回测。

        Args:
            df: 全量数据
            scenarios: 策略场景列表

        Returns:
            所有回测结果的展平列表
        """
        all_results = []
        total = sum(len(s.periods) if s.periods else 1 for s in scenarios)
        logger.info(f"共 {len(scenarios)} 个场景，预计 {total} 次回测")

        for i, scenario in enumerate(scenarios):
            logger.info(f"[{i+1}/{len(scenarios)}] 场景: {scenario.strategy_name}")
            try:
                results = self.run_scenario(df, scenario)
                all_results.extend(results)
            except Exception as e:
                logger.error(f"场景 '{scenario.strategy_name}' 回测失败: {e}")
                logger.debug(traceback.format_exc())

        logger.info(f"回测全部完成，成功: {len(all_results)}/{total}")
        return all_results


# ---------------------------------------------------------------------------
# 指标分析器
# ---------------------------------------------------------------------------

class MetricsAnalyzer:
    """
    结果分析模块。

    封装 MetricsCalculator，提供对比分析、逐年收益、
    市场环境分析等功能。
    """

    @staticmethod
    def safe_get(metrics: Dict, *keys) -> float:
        """安全获取指标值，依次尝试多个键名。"""
        for k in keys:
            if k in metrics and metrics[k] is not None:
                try:
                    return float(metrics[k])
                except (ValueError, TypeError):
                    return 0.0
        return 0.0

    @staticmethod
    def build_comparison_table(all_results: List[Dict]) -> pd.DataFrame:
        """
        构建策略对比表。

        Args:
            all_results: 所有回测结果列表

        Returns:
            对比DataFrame，行=策略×时期，列=指标
        """
        _g = MetricsAnalyzer.safe_get
        rows = []
        for r in all_results:
            m = r["metrics"]
            row = {
                "策略": r.get("strategy_name", r["strategy_id"]),
                "时段": r.get("period", "-"),
                "总收益率(%)": round(_g(m, "total_return_pct", "annual_return_pct"), 2),
                "年化收益率(%)": round(_g(m, "annual_return_pct"), 2),
                "最大回撤(%)": round(_g(m, "max_drawdown_pct"), 2),
                "年化波动率(%)": round(_g(m, "annual_volatility_pct"), 2),
                "Sharpe比率": round(_g(m, "sharpe_ratio", "sharpe"), 4),
                "Sortino比率": round(_g(m, "sortino_ratio", "sortino"), 4),
                "Calmar比率": round(_g(m, "calmar_ratio"), 4),
                "胜率(%)": round(_g(m, "win_rate", "daily_win_rate"), 2),
                "盈亏比": round(_g(m, "profit_factor"), 4),
                "交易次数": int(_g(m, "trade_count")),
                "回撤持续天数": int(_g(m, "max_drawdown_duration_days")),
                "期望收益": round(_g(m, "expectancy"), 4),
            }
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def compute_yearly_returns(
        portfolios: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        计算各策略逐年收益率。

        Args:
            portfolios: {策略名: portfolio DataFrame}

        Returns:
            逐年收益率 DataFrame，列为策略名，索引为年份
        """
        yearly_data = {}
        for name, df in portfolios.items():
            if df is None or df.empty:
                continue
            pdf = df.copy()
            if "date" not in pdf.columns:
                if pdf.index.name == "date":
                    pdf = pdf.reset_index()
                else:
                    continue
            pdf["date"] = pd.to_datetime(pdf["date"])
            pdf["year"] = pdf["date"].dt.year

            equity_col = None
            for col in ["equity", "market_value"]:
                if col in pdf.columns:
                    equity_col = col
                    break
            if equity_col is None:
                continue

            yearly_eq = pdf.groupby("year")[equity_col].agg(["first", "last"])
            yearly_eq["return_pct"] = (yearly_eq["last"] / yearly_eq["first"] - 1) * 100
            yearly_data[name] = yearly_eq["return_pct"]

        if not yearly_data:
            return pd.DataFrame()
        result = pd.DataFrame(yearly_data)
        result.index.name = "年份"
        return result


# ---------------------------------------------------------------------------
# 结果持久化
# ---------------------------------------------------------------------------

class ResultSaver:
    """回测结果保存模块。"""

    @staticmethod
    def save(
        all_results: List[Dict],
        comparison_df: pd.DataFrame,
        data_info: Dict,
        config: BacktestJobConfig,
        output_dir: Optional[str] = None,
    ) -> str:
        """
        保存回测结果到文件。

        Args:
            all_results: 所有回测结果
            comparison_df: 对比表
            data_info: 数据信息
            config: 任务配置
            output_dir: 输出目录（None则自动生成）

        Returns:
            输出目录路径
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"./unified_backtest_results_{timestamp}"

        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"保存结果到: {output_dir}")

        comparison_df.to_csv(f"{output_dir}/comparison_table.csv", encoding="utf-8-sig")

        portfolio_dict = {}
        for r in all_results:
            label = f"{r['strategy_id']}"
            if r.get("period"):
                label += f"_{r['period']}"

            if r["portfolio"] is not None:
                pdf = r["portfolio"].copy()
                if "date" not in pdf.columns and pdf.index.name == "date":
                    pdf = pdf.reset_index()
                pdf.to_csv(f"{output_dir}/portfolio_{label}.csv", encoding="utf-8-sig", index=False)
                portfolio_dict[label] = pdf

            if r["trades"] is not None:
                r["trades"].to_csv(f"{output_dir}/trades_{label}.csv", encoding="utf-8-sig", index=False)

        combined_list = []
        for r in all_results:
            if r["portfolio"] is not None:
                pdf = r["portfolio"].copy()
                if "date" not in pdf.columns and pdf.index.name == "date":
                    pdf = pdf.reset_index()
                pdf["strategy"] = r.get("strategy_name", r["strategy_id"])
                combined_list.append(pdf)

        if combined_list:
            combined = pd.concat(combined_list, ignore_index=True)
            combined.to_csv(f"{output_dir}/combined_portfolio.csv", encoding="utf-8-sig", index=False)

        config_data = config.to_dict()
        config_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        config_data["data_info"] = data_info
        with open(f"{output_dir}/config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)

        with open(f"{output_dir}/summary.txt", "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("统一回测分析报告\n")
            f.write("=" * 80 + "\n\n")
            f.write("数据信息\n")
            f.write("-" * 40 + "\n")
            f.write(f"品种: {data_info.get('symbols', [])}\n")
            f.write(f"时间范围: {data_info.get('date_range', ('N/A', 'N/A'))[0]} ~ {data_info.get('date_range', ('N/A', 'N/A'))[1]}\n")
            f.write(f"数据行数: {data_info.get('total_rows', 0):,}\n\n")
            f.write("策略对比表\n")
            f.write("-" * 40 + "\n")
            f.write(comparison_df.to_string())
            f.write("\n")

        logger.info(f"结果保存完成: {output_dir}")
        return output_dir


# ---------------------------------------------------------------------------
# 可视化与报告
# ---------------------------------------------------------------------------

class VisualizationManager:
    """
    可视化与报告模块。

    封装 backtest_visualization.py 和 run_new_strategies.py 中的
    可视化函数，提供统一的图表生成和HTML报告接口。
    """

    @staticmethod
    def generate_all(result_dir: str):
        """
        调用 backtest_visualization.py 生成所有可视化图表。

        Args:
            result_dir: 回测结果目录
        """
        logger.info("=" * 60)
        logger.info("生成可视化图表")
        logger.info("=" * 60)

        try:
            import backtest_visualization as viz
            results = viz.load_results(result_dir)
        except Exception as e:
            logger.warning(f"使用 backtest_visualization.load_results 加载结果失败: {e}")
            results = VisualizationManager._load_results_universal(result_dir)

        try:
            viz = __import__("backtest_visualization")
        except ImportError:
            logger.warning("无法导入 backtest_visualization，将使用本地可视化")
            viz = None

        if viz:
            try:
                viz.plot_equity_curves(results, result_dir)
            except Exception as e:
                logger.warning(f"净值曲线绘制失败: {e}")
            try:
                viz.plot_drawdown_curves(results, result_dir)
            except Exception as e:
                logger.warning(f"回撤曲线绘制失败: {e}")
            try:
                viz.plot_performance_radar(results, result_dir)
            except Exception as e:
                logger.warning(f"雷达图绘制失败: {e}")
            try:
                viz.plot_trade_analysis(results, result_dir)
            except Exception as e:
                logger.warning(f"交易分析绘制失败: {e}")
            try:
                viz.plot_returns_distribution(results, result_dir)
            except Exception as e:
                logger.warning(f"收益分布绘制失败: {e}")
            try:
                viz.plot_yearly_returns(results, result_dir)
            except Exception as e:
                logger.warning(f"逐年收益绘制失败: {e}")
            try:
                viz.plot_yearly_returns_comparison(results, result_dir)
            except Exception as e:
                logger.warning(f"逐年收益对比绘制失败: {e}")
        else:
            VisualizationManager._generate_fallback_viz(results, result_dir)

        VisualizationManager._generate_universal_html_report(
            results, result_dir,
        )
        logger.info("可视化与报告生成完成")

    @staticmethod
    def _load_results_universal(result_dir: str) -> Dict:
        """通用结果加载（不依赖 backtest_visualization.load_results）。"""
        results = {}
        comparison_path = f"{result_dir}/comparison_table.csv"
        if os.path.exists(comparison_path):
            results["comparison"] = pd.read_csv(comparison_path, index_col=0)

        config_path = f"{result_dir}/config.json"
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                results["config"] = json.load(f)

        combined_path = f"{result_dir}/combined_portfolio.csv"
        if os.path.exists(combined_path):
            results["combined_portfolio"] = pd.read_csv(combined_path)

        results["portfolios"] = {}
        results["trades"] = {}
        for filename in os.listdir(result_dir):
            if filename.startswith("portfolio_") and filename.endswith(".csv"):
                key = filename.replace("portfolio_", "").replace(".csv", "")
                results["portfolios"][key] = pd.read_csv(f"{result_dir}/{filename}")
            elif filename.startswith("trades_") and filename.endswith(".csv"):
                key = filename.replace("trades_", "").replace(".csv", "")
                results["trades"][key] = pd.read_csv(f"{result_dir}/{filename}")

        return results

    @staticmethod
    def _generate_fallback_viz(results: Dict, output_dir: str):
        """备用可视化（当 backtest_visualization 不可用时）。"""
        import plotly.graph_objects as go
        import plotly.express as px

        if "combined_portfolio" in results:
            df = results["combined_portfolio"].copy()
            df["date"] = pd.to_datetime(df["date"])

            fig = go.Figure()
            for i, strategy in enumerate(df["strategy"].unique()):
                strat_data = df[df["strategy"] == strategy].sort_values("date")
                equity_col = "equity" if "equity" in strat_data.columns else "market_value"
                if equity_col not in strat_data.columns:
                    continue
                eq = strat_data[equity_col] / strat_data[equity_col].iloc[0]
                fig.add_trace(go.Scatter(
                    x=strat_data["date"], y=eq, name=strategy,
                    mode="lines", line=dict(width=2),
                ))
            fig.update_layout(
                title="策略净值曲线对比",
                template="plotly_white", height=600,
            )
            fig.write_html(f"{output_dir}/equity_curves.html")

    @staticmethod
    def _generate_universal_html_report(results: Dict, output_dir: str):
        """生成通用HTML综合报告。"""
        config = results.get("config", {})
        comparison = results.get("comparison")

        reports = [
            f'<li><a href="{name}.html" target="_blank">{title}</a></li>'
            for name, title in [
                ("equity_curves", "净值曲线对比"),
                ("drawdown_curves", "回撤曲线对比"),
                ("performance_radar", "绩效雷达图"),
                ("trade_analysis", "交易分析"),
                ("returns_distribution", "收益率分布"),
                ("yearly_returns", "逐年收益率"),
                ("yearly_returns_comparison", "逐年收益率对比"),
            ]
            if os.path.exists(f"{output_dir}/{name}.html")
        ]

        comparison_html = ""
        if comparison is not None:
            comparison_html = comparison.to_html(classes="data-table")

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>统一回测分析报告</title>
<style>
body {{ font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif;
       max-width: 1280px; margin: 0 auto; padding: 20px; background: #f5f7fa; }}
h1 {{ color: #1a1a2e; border-bottom: 3px solid #1f77b4; padding-bottom: 10px; }}
h2 {{ color: #2c3e50; margin-top: 30px; }}
.section {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px;
           box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
.data-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
.data-table th {{ background: #1f77b4; color: white; padding: 10px; }}
.data-table td {{ padding: 8px; border-bottom: 1px solid #eee; text-align: center; }}
.data-table tr:hover {{ background: #f0f4ff; }}
iframe {{ width: 100%; height: 650px; border: none; border-radius: 8px; }}
footer {{ text-align: center; color: #888; margin-top: 40px; font-size: 12px; }}
</style>
</head>
<body>
<h1>统一回测分析报告</h1>
<div class="section">
    <h2>回测配置</h2>
    <pre>{json.dumps(config.get('global_config', {}), indent=2, ensure_ascii=False)}</pre>
</div>
<div class="section">
    <h2>策略绩效对比</h2>
    {comparison_html}
</div>
<div class="section">
    <h2>可视化分析</h2>
    <ul>{''.join(reports)}</ul>
</div>
<footer>报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</footer>
</body>
</html>"""

        with open(f"{output_dir}/report.html", "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"HTML报告: {output_dir}/report.html")


# ---------------------------------------------------------------------------
# 统一回测编排器
# ---------------------------------------------------------------------------

class UnifiedBacktest:
    """
    统一回测编排器。

    整合所有模块，提供标准化的回测流程入口。

    使用方式：
        # 方式1: 编程接口
        ub = UnifiedBacktest()
        ub.setup_scenario("dual_ma_comparison")
        ub.run()

        # 方式2: 自定义配置
        config = BacktestJobConfig(...)
        ub = UnifiedBacktest(config)
        ub.run()

        # 方式3: 从配置文件
        ub = UnifiedBacktest.from_config_file("config.json")
        ub.run()

    新增回测要求的扩展流程：
        1. 准备策略类（若为新策略，在 core/strategies/ 中实现并注册）
        2. 创建配置：可直接构建 BacktestJobConfig，或使用 --config 传入JSON
        3. 调用 run() 执行

    示例 - 新增一个策略回测：
        config = BacktestJobConfig(
            global_config=BacktestGlobalConfig(),
            data_filter=DataFilterConfig(
                symbols=["SHFE.RB", "DCE.M"],
                start_date="2020-01-01",
                end_date="2024-12-31",
            ),
            scenarios=[
                StrategyScenario(
                    strategy_id="dual_ma",
                    strategy_name="我的双均线策略",
                    params={"short_ma": 10, "long_ma": 30, "adx_threshold": 25.0},
                    periods=[
                        PeriodConfig("样本内", "2020-01-01", "2022-12-31"),
                        PeriodConfig("样本外", "2023-01-01", "2024-12-31"),
                    ],
                ),
            ],
        )
        ub = UnifiedBacktest(config)
        ub.run()
    """

    def __init__(self, config: Optional[BacktestJobConfig] = None):
        self.config = config or BacktestJobConfig()
        self._data_manager: Optional[DataManager] = None
        self._runner: Optional[BacktestRunner] = None
        self._all_results: List[Dict] = []
        self._comparison_df: Optional[pd.DataFrame] = None
        self._output_dir: Optional[str] = None

    @classmethod
    def from_config_file(cls, filepath: str) -> "UnifiedBacktest":
        """从JSON配置文件创建实例。"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        config = BacktestJobConfig.from_dict(data)
        return cls(config)

    def setup_scenario(self, scenario_key: str) -> "UnifiedBacktest":
        """
        便捷方法：使用预置场景配置。

        Args:
            scenario_key: 场景键名，如 'dual_ma_comparison', 'new_strategies', 'all'
        """
        self.config.scenarios = StrategySelector.resolve_scenario(scenario_key)
        logger.info(f"已加载场景 '{scenario_key}': {[s.strategy_name for s in self.config.scenarios]}")
        return self

    def run(self) -> Dict:
        """
        执行完整回测流程。

        流程：
        1. 数据加载与预处理
        2. 策略回测执行
        3. 绩效指标对比分析
        4. 结果保存
        5. 可视化与报告生成

        Returns:
            包含所有结果的字典
        """
        logger.info("=" * 80)
        logger.info("统一回测系统启动")
        logger.info("=" * 80)

        if not self.config.scenarios:
            raise ValueError("未配置任何策略场景，请先调用 setup_scenario() 或设置 config.scenarios")

        self._data_manager = DataManager(self.config.global_config, self.config.data_filter)
        df, data_info = self._data_manager.load_and_prepare()

        self._runner = BacktestRunner(self.config.global_config)
        self._all_results = self._runner.run_all(df, self.config.scenarios)

        if not self._all_results:
            logger.error("没有成功的回测结果！")
            return {"status": "failed", "error": "no results"}

        self._comparison_df = MetricsAnalyzer.build_comparison_table(self._all_results)
        logger.info("\n" + self._comparison_df.to_string())

        self._output_dir = ResultSaver.save(
            self._all_results, self._comparison_df, data_info,
            self.config, self.config.output_dir,
        )

        VisualizationManager.generate_all(self._output_dir)

        logger.info("=" * 80)
        logger.info(f"回测完成！结果目录: {self._output_dir}")
        logger.info("=" * 80)

        return {
            "status": "success",
            "output_dir": self._output_dir,
            "results": self._all_results,
            "comparison": self._comparison_df,
            "data_info": data_info,
        }

    @property
    def results(self) -> List[Dict]:
        return self._all_results

    @property
    def comparison(self) -> Optional[pd.DataFrame]:
        return self._comparison_df

    @property
    def output_dir(self) -> Optional[str]:
        return self._output_dir


# ---------------------------------------------------------------------------
# 交互式菜单
# ---------------------------------------------------------------------------

def interactive_menu():
    """交互式菜单模式。"""
    print("\n" + "=" * 70)
    print("  统一回测系统 - 交互式菜单")
    print("=" * 70)

    print("\n可用的预置场景:")
    print("  1. dual_ma_comparison  - 双均线策略变体对比")
    print("  2. new_strategies       - 新策略对比（期限结构+波动率突破）")
    print("  3. all                  - 所有策略")
    print("  4. 自定义配置")

    choice = input("\n请选择场景 [1-4]: ").strip()
    scenario_map = {
        "1": "dual_ma_comparison",
        "2": "new_strategies",
        "3": "all",
    }

    if choice in scenario_map:
        scenario = scenario_map[choice]
    elif choice == "4":
        return custom_config_flow()
    else:
        print("无效选择，使用默认: dual_ma_comparison")
        scenario = "dual_ma_comparison"

    symbols = input("品种筛选 (回车=全部, 如 SHFE.RB,DCE.M): ").strip()
    symbols_list = [s.strip() for s in symbols.split(",") if s.strip()] if symbols else None

    start = input("开始日期 (回车=全部, 如 2020-01-01): ").strip() or None
    end = input("结束日期 (回车=全部, 如 2024-12-31): ").strip() or None

    config = BacktestJobConfig(
        global_config=BacktestGlobalConfig(),
        data_filter=DataFilterConfig(
            symbols=symbols_list,
            start_date=start,
            end_date=end,
        ),
    )

    ub = UnifiedBacktest(config)
    ub.setup_scenario(scenario)
    return ub.run()


def custom_config_flow():
    """自定义配置流程。"""
    print("\n--- 自定义配置 ---")

    symbols = input("品种 (逗号分隔, 如 SHFE.RB,DCE.M): ").strip()
    symbols_list = [s.strip() for s in symbols.split(",") if s.strip()]

    start = input("开始日期 (如 2020-01-01): ").strip()
    end = input("结束日期 (如 2024-12-31): ").strip()

    print("\n可用策略:", ", ".join(StrategySelector.list_available()))
    strategy_id = input("策略ID: ").strip()

    params_str = input("参数 (JSON, 如 {\"short_ma\":5,\"long_ma\":20}): ").strip()
    params = json.loads(params_str) if params_str else {}

    use_periods = input("是否分时段回测? (y/n): ").strip().lower() == "y"
    periods = None
    if use_periods:
        periods = []
        while True:
            label = input("时段名称 (回车结束): ").strip()
            if not label:
                break
            ps = input("  开始日期: ").strip()
            pe = input("  结束日期: ").strip()
            periods.append(PeriodConfig(label, ps, pe))

    config = BacktestJobConfig(
        global_config=BacktestGlobalConfig(),
        data_filter=DataFilterConfig(
            symbols=symbols_list,
            start_date=start,
            end_date=end,
        ),
        scenarios=[
            StrategyScenario(
                strategy_id=strategy_id,
                strategy_name=f"{strategy_id}-自定义",
                params=params,
                periods=periods,
            ),
        ],
    )

    ub = UnifiedBacktest(config)
    return ub.run()


# ---------------------------------------------------------------------------
# CLI入口
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="统一回测脚本 - 多策略回测对比分析平台",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
    python unified_backtest.py                                        # 交互式菜单
    python unified_backtest.py --scenario dual_ma_comparison          # 预置场景
    python unified_backtest.py --config config.json                   # 配置文件
    python unified_backtest.py --strategy dual_ma                      # 单策略交互
    python unified_backtest.py --list-strategies                      # 列出可用策略
        """,
    )

    parser.add_argument(
        "--config", "-c", type=str,
        help="JSON配置文件路径",
    )
    parser.add_argument(
        "--scenario", "-s", type=str,
        help=f"预置场景名: {', '.join(PREBUILT_SCENARIOS.keys())}",
    )
    parser.add_argument(
        "--strategy", type=str,
        help="单个策略ID（进入交互式参数配置）",
    )
    parser.add_argument(
        "--list-strategies", action="store_true",
        help="列出所有可用策略",
    )
    parser.add_argument(
        "--list-scenarios", action="store_true",
        help="列出所有预置场景",
    )
    parser.add_argument(
        "--symbols", type=str,
        help="品种筛选，逗号分隔 (如 SHFE.RB,DCE.M)",
    )
    parser.add_argument(
        "--start", type=str,
        help="开始日期 (如 2020-01-01)",
    )
    parser.add_argument(
        "--end", type=str,
        help="结束日期 (如 2024-12-31)",
    )
    parser.add_argument(
        "--output-dir", "-o", type=str,
        help="输出目录",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="静默模式，减少输出",
    )
    parser.add_argument(
        "--log-file", type=str,
        help="日志文件路径",
    )

    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    if args.quiet:
        setup_logging(logging.WARNING, args.log_file)
    elif args.log_file:
        setup_logging(logging.INFO, args.log_file)

    if args.list_strategies:
        print("\n可用策略:")
        for sid in StrategySelector.list_available():
            info = StrategySelector.get_strategy_info(sid)
            print(f"  {sid:25s} {info['class']:30s} {info.get('doc', '')}")
        return

    if args.list_scenarios:
        print("\n预置场景:")
        for group, keys in PREBUILT_SCENARIOS.items():
            print(f"\n  [{group}]")
            for k in keys:
                s = STRATEGY_SCENARIOS.get(k)
                if s:
                    print(f"    {k:30s} {s.strategy_name} - {s.description}")
        return

    symbols_list = None
    if args.symbols:
        symbols_list = [s.strip() for s in args.symbols.split(",") if s.strip()]

    if args.config:
        logger.info(f"从配置文件加载: {args.config}")
        ub = UnifiedBacktest.from_config_file(args.config)
        if symbols_list or args.start or args.end or args.output_dir:
            if symbols_list:
                ub.config.data_filter.symbols = symbols_list
            if args.start:
                ub.config.data_filter.start_date = args.start
            if args.end:
                ub.config.data_filter.end_date = args.end
            if args.output_dir:
                ub.config.output_dir = args.output_dir
        return ub.run()

    if args.scenario:
        config = BacktestJobConfig(
            global_config=BacktestGlobalConfig(),
            data_filter=DataFilterConfig(
                symbols=symbols_list,
                start_date=args.start,
                end_date=args.end,
            ),
            output_dir=args.output_dir,
        )
        ub = UnifiedBacktest(config)
        ub.setup_scenario(args.scenario)
        return ub.run()

    if args.strategy:
        strategy_id = args.strategy
        if strategy_id not in STRATEGY_REGISTRY:
            logger.error(f"未知策略 '{strategy_id}'，可用: {list(STRATEGY_REGISTRY.keys())}")
            return

        scenario = STRATEGY_SCENARIOS.get(strategy_id)
        if scenario:
            config = BacktestJobConfig(
                global_config=BacktestGlobalConfig(),
                data_filter=DataFilterConfig(
                    symbols=symbols_list,
                    start_date=args.start,
                    end_date=args.end,
                ),
                scenarios=[scenario],
                output_dir=args.output_dir,
            )
            ub = UnifiedBacktest(config)
        else:
            ub = UnifiedBacktest()
            ub.setup_scenario(strategy_id)
        return ub.run()

    # 无参数则进入交互式菜单
    try:
        result = interactive_menu()
        if result and result.get("status") == "failure":
            logger.error(f"回测失败: {result.get('error')}")
    except KeyboardInterrupt:
        print("\n\n用户中断。")
    except Exception as e:
        logger.error(f"运行错误: {e}")
        logger.debug(traceback.format_exc())


if __name__ == "__main__":
    main()