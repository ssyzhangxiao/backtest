"""
蒙特卡洛模拟模块。

对策略收益序列进行Bootstrap重采样，生成置信区间：
  - 1000次重采样
  - 输出Sharpe/最大回撤/年化收益的5%/50%/95%分位数
  - Sharpe的5%分位数>0视为稳健

规则15要求：蒙特卡洛模拟1000次，输出置信区间。

v2: 向量化优化，使用NumPy矩阵运算替代Python for循环，
    1000次模拟从~2s降至~50ms，内存占用可控。
P2 整改（2026-06-07）：
    - trading_days_per_year 参数化（默认 252，支持加密货币 365 等）
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import logging
import time

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_N_SIMULATIONS = 1000
DEFAULT_TRADING_DAYS_PER_YEAR = 252

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]


@dataclass
class MonteCarloResult:
    """蒙特卡洛模拟结果。"""

    n_simulations: int = 0
    sharpe_quantiles: Dict[float, float] = field(default_factory=dict)
    max_drawdown_quantiles: Dict[float, float] = field(default_factory=dict)
    annual_return_quantiles: Dict[float, float] = field(default_factory=dict)
    is_robust: bool = False
    # 性能数据（向量化优化后新增）
    elapsed_seconds: float = 0.0
    # P2 整改：记录交易日参数，便于不同市场对比
    trading_days_per_year: int = DEFAULT_TRADING_DAYS_PER_YEAR
    # P0 整改：可选保存完整模拟路径 (n_simulations, n_days+1)
    #   索引 0 是初始值 1.0；用于保存详细结果/绘制路径分布图
    paths: Optional[np.ndarray] = None

    def summary(self) -> str:
        """返回模拟摘要。"""
        sharpe_5 = self.sharpe_quantiles.get(0.05, 0.0)
        sharpe_50 = self.sharpe_quantiles.get(0.50, 0.0)
        sharpe_95 = self.sharpe_quantiles.get(0.95, 0.0)
        dd_95 = self.max_drawdown_quantiles.get(0.95, 0.0)
        ret_5 = self.annual_return_quantiles.get(0.05, 0.0)
        ret_95 = self.annual_return_quantiles.get(0.95, 0.0)

        robust_str = "✅稳健" if self.is_robust else "⚠️不稳健"
        return (
            f"蒙特卡洛{self.n_simulations}次({self.elapsed_seconds:.3f}s, "
            f"年化基数={self.trading_days_per_year}) | "
            f"Sharpe: [{sharpe_5:.2f}, {sharpe_50:.2f}, {sharpe_95:.2f}] | "
            f"MDD@95%: {dd_95:.1%} | "
            f"Return: [{ret_5:.1%}, {ret_95:.1%}] | "
            f"{robust_str}"
        )


class MonteCarloSimulator:
    """
    蒙特卡洛模拟器（向量化版本）。

    对策略日收益率进行Bootstrap重采样，
    生成Sharpe/最大回撤/年化收益的置信区间。

    优化策略：
      - 一次性生成 (n_simulations, n_days) 的重采样矩阵
      - 沿axis=1向量化计算Sharpe/最大回撤/年化收益
      - 避免Python for循环，利用NumPy底层C加速

    用法:
        mc = MonteCarloSimulator(n_simulations=1000, trading_days_per_year=365)
        result = mc.simulate(daily_returns)
        if result.is_robust:
            print("策略稳健")
    """

    def __init__(
        self,
        n_simulations: int = DEFAULT_N_SIMULATIONS,
        random_seed: Optional[int] = None,
        trading_days_per_year: int = DEFAULT_TRADING_DAYS_PER_YEAR,
    ):
        if trading_days_per_year <= 0:
            raise ValueError(
                f"trading_days_per_year 必须为正数，实际: {trading_days_per_year}"
            )
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        self.trading_days_per_year = int(trading_days_per_year)

    @staticmethod
    def _vectorized_sharpe(
        samples: np.ndarray, trading_days_per_year: int
    ) -> np.ndarray:
        """
        向量化计算年化Sharpe（P2 整改：年化基数参数化）。

        Args:
            samples: (n_sim, n_days) 重采样矩阵
            trading_days_per_year: 年化交易日数（252=A股, 365=加密货币）

        Returns:
            (n_sim,) Sharpe数组
        """
        mean_r = np.mean(samples, axis=1)
        std_r = np.std(samples, axis=1)
        # 避免除零：std<1e-10时Sharpe=0
        valid = std_r > 1e-10
        sharpes = np.zeros(samples.shape[0])
        sharpes[valid] = mean_r[valid] / std_r[valid] * np.sqrt(trading_days_per_year)
        return sharpes

    @staticmethod
    def _vectorized_max_drawdown(samples: np.ndarray) -> np.ndarray:
        """
        向量化计算最大回撤。

        Args:
            samples: (n_sim, n_days) 重采样矩阵

        Returns:
            (n_sim,) 最大回撤数组
        """
        # 净值曲线：(n_sim, n_days)
        equity = np.cumprod(1 + samples, axis=1)
        # 滚动峰值：沿axis=1逐元素取历史最大值
        peak = np.maximum.accumulate(equity, axis=1)
        # 回撤序列
        drawdown = (equity - peak) / peak
        # 最大回撤（取绝对值）
        max_dd = -np.min(drawdown, axis=1)
        return max_dd

    @staticmethod
    def _vectorized_annual_return(
        samples: np.ndarray, trading_days_per_year: int
    ) -> np.ndarray:
        """
        向量化计算年化收益（P2 整改：年化基数参数化）。

        Args:
            samples: (n_sim, n_days) 重采样矩阵
            trading_days_per_year: 年化交易日数

        Returns:
            (n_sim,) 年化收益数组
        """
        # 累积收益
        total = np.prod(1 + samples, axis=1)
        n_years = samples.shape[1] / trading_days_per_year
        # 避免除零和负数开方
        annual_ret = np.where(
            n_years > 1e-6,
            np.power(np.maximum(total, 0), 1 / n_years) - 1,
            0.0,
        )
        return annual_ret

    def simulate(
        self, daily_returns: np.ndarray, return_paths: bool = False
    ) -> MonteCarloResult:
        """
        执行蒙特卡洛模拟（向量化版本）。

        Args:
            daily_returns: 日收益率序列
            return_paths: 是否在结果中保存完整模拟路径 (n_sim, n_days+1)。
                True 时额外保存（占内存：n_sim × n_days × 8 字节），
                用于下游保存详细结果/绘制路径分布图。

        Returns:
            MonteCarloResult 模拟结果（is_robust 字段供调用方判定稳健性）
        """
        ret = np.asarray(daily_returns, dtype=float)
        n = len(ret)

        if n < 20:
            logger.warning("收益率序列过短，蒙特卡洛模拟结果不可靠")
            return MonteCarloResult(
                n_simulations=0,
                is_robust=False,
                trading_days_per_year=self.trading_days_per_year,
            )

        t0 = time.perf_counter()

        rng = np.random.default_rng(self.random_seed)

        # 一次性生成重采样矩阵：(n_simulations, n_days)
        # 使用rng.choice的axis参数实现批量Bootstrap
        indices = rng.integers(0, n, size=(self.n_simulations, n))
        samples = ret[indices]

        # 向量化计算三个指标（P2 整改：传入年化基数）
        sharpes = self._vectorized_sharpe(samples, self.trading_days_per_year)
        max_dds = self._vectorized_max_drawdown(samples)
        annual_rets = self._vectorized_annual_return(
            samples, self.trading_days_per_year
        )

        # P0 整改：可选保存完整模拟路径
        # 路径 = 累积净值 = cumprod(1 + samples)，前缀 1.0
        # 仅在 return_paths=True 时构造，节省内存
        paths: Optional[np.ndarray] = None
        if return_paths:
            equity_paths = np.cumprod(1.0 + samples, axis=1)
            paths = np.empty((self.n_simulations, n + 1), dtype=float)
            paths[:, 0] = 1.0
            paths[:, 1:] = equity_paths

        elapsed = time.perf_counter() - t0

        # 计算分位数
        sharpe_q = {q: float(np.quantile(sharpes, q)) for q in QUANTILES}
        dd_q = {q: float(np.quantile(max_dds, q)) for q in QUANTILES}
        ret_q = {q: float(np.quantile(annual_rets, q)) for q in QUANTILES}

        # 稳健性判定：Sharpe的5%分位数>0
        is_robust = sharpe_q.get(0.05, 0.0) > 0

        result = MonteCarloResult(
            n_simulations=self.n_simulations,
            sharpe_quantiles=sharpe_q,
            max_drawdown_quantiles=dd_q,
            annual_return_quantiles=ret_q,
            is_robust=is_robust,
            elapsed_seconds=elapsed,
            trading_days_per_year=self.trading_days_per_year,
            paths=paths,
        )

        logger.info(result.summary())
        return result

    def simulate_loop(self, daily_returns: np.ndarray) -> MonteCarloResult:
        """
        旧版循环实现（保留用于性能对比和结果验证）。

        Args:
            daily_returns: 日收益率序列

        Returns:
            MonteCarloResult 模拟结果
        """
        ret = np.asarray(daily_returns, dtype=float)
        n = len(ret)

        if n < 20:
            return MonteCarloResult(
                n_simulations=0,
                is_robust=False,
                trading_days_per_year=self.trading_days_per_year,
            )

        t0 = time.perf_counter()
        rng = np.random.default_rng(self.random_seed)
        tdp = self.trading_days_per_year  # P2 整改

        sharpes = np.zeros(self.n_simulations)
        max_dds = np.zeros(self.n_simulations)
        annual_rets = np.zeros(self.n_simulations)

        for i in range(self.n_simulations):
            sample = rng.choice(ret, size=n, replace=True)
            # Sharpe
            mean_r = np.mean(sample)
            std_r = np.std(sample)
            if std_r > 1e-10:
                sharpes[i] = mean_r / std_r * np.sqrt(tdp)
            # 最大回撤
            equity = np.cumprod(1 + sample)
            peak = np.maximum.accumulate(equity)
            dd = (equity - peak) / peak
            max_dds[i] = -np.min(dd)
            # 年化收益
            total = np.prod(1 + sample)
            n_years = n / tdp
            if n_years > 1e-6:
                annual_rets[i] = total ** (1 / n_years) - 1

        elapsed = time.perf_counter() - t0

        sharpe_q = {q: float(np.quantile(sharpes, q)) for q in QUANTILES}
        dd_q = {q: float(np.quantile(max_dds, q)) for q in QUANTILES}
        ret_q = {q: float(np.quantile(annual_rets, q)) for q in QUANTILES}

        return MonteCarloResult(
            n_simulations=self.n_simulations,
            sharpe_quantiles=sharpe_q,
            max_drawdown_quantiles=dd_q,
            annual_return_quantiles=ret_q,
            is_robust=sharpe_q.get(0.05, 0.0) > 0,
            elapsed_seconds=elapsed,
            trading_days_per_year=tdp,
        )
