"""统一 CLI 入口（规则21）。

提供 `quant-system` 命令聚合 3 个官方入口的功能（不删除原 run_*.py，规则20）：

    quant-system backtest  --experiment e1
    quant-system optimize  --strategy trend
    quant-system validate  --method monte_carlo
    quant-system report    --fmt html
    quant-system adaptors  # 列出已注册数据源（规则21.3 工厂）

核心约束：
    - 不重新实现 Pipeline 逻辑，主体委托 runner.Pipeline
    - 3 个 run_*.py 保留不动，CLI 是"增强入口"
    - 单文件 ≤ 500 行（规则7）

依赖：click（核心 requirements 已包含）
    安装：pip install -e .[core]  # click 来自 streamlit 传递依赖
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click

from runner.pipeline import Pipeline


__all__ = ["cli", "main"]


# ---------------------------------------------------------------------------
# 公共选项
# ---------------------------------------------------------------------------
def _common_options(func):
    """公共选项：--config / --verbose。"""
    func = click.option(
        "--config",
        "-c",
        default="config.yaml",
        type=click.Path(exists=False),
        help="配置文件路径（默认 config.yaml）",
    )(func)
    func = click.option(
        "--verbose",
        "-v",
        is_flag=True,
        default=False,
        help="启用 DEBUG 日志",
    )(func)
    return func


def _build_pipeline(config_path: str, verbose: bool) -> Pipeline:
    """构造 Pipeline（委托核心编排器，规则17）。"""
    if verbose:
        from loguru import logger
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    return Pipeline(config_path).load_data()


# ---------------------------------------------------------------------------
# CLI 主入口
# ---------------------------------------------------------------------------
@click.group()
@click.version_option(version="0.1.0", prog_name="quant-system")
def cli() -> None:
    """5子策略量化回测系统 - 统一 CLI 入口。"""
    pass


# ---------------------------------------------------------------------------
# backtest 子命令
# ---------------------------------------------------------------------------
@cli.command()
@_common_options
@click.option("--experiment", "-e", default="all", help="实验名（e1/e2/...），默认全部")
@click.option("--cross-sectional/--no-cross-sectional", default=True, help="多策略横截面打分模式")
@click.option("--strategy", "-s", default=None, help="指定策略名（与 --cross-sectional 互斥）")
def backtest(config: str, verbose: bool, experiment: str, cross_sectional: bool, strategy: Optional[str]) -> None:
    """执行回测（委托 pipe.run_backtest）。"""
    pipe = _build_pipeline(config, verbose)
    pipe.run_backtest(
        experiment=experiment,
        cross_sectional=cross_sectional,
        strategy=strategy,
    )
    click.echo(f"[OK] 回测完成 experiment={experiment}")


# ---------------------------------------------------------------------------
# optimize 子命令
# ---------------------------------------------------------------------------
@cli.command()
@_common_options
@click.option("--strategy", "-s", default=None, help="指定策略名")
@click.option(
    "--tasks", "-t",
    multiple=True,
    type=click.Choice(["grid", "window", "oos"], case_sensitive=False),
    help="优化任务类型（可多选）",
)
@click.option("--symbol", default=None, help="指定品种")
@click.option("--save/--no-save", "save_to_config", default=True, help="是否写回 config.yaml")
def optimize(
    config: str,
    verbose: bool,
    strategy: Optional[str],
    tasks: tuple,
    symbol: Optional[str],
    save_to_config: bool,
) -> None:
    """执行参数优化（委托 pipe.optimize）。"""
    pipe = _build_pipeline(config, verbose)
    task_list = list(tasks) if tasks else None
    pipe.optimize(
        strategy=strategy,
        tasks=task_list,
        symbol=symbol,
        save_to_config=save_to_config,
    )
    click.echo(f"[OK] 优化完成 strategy={strategy} tasks={task_list}")


# ---------------------------------------------------------------------------
# validate 子命令
# ---------------------------------------------------------------------------
@cli.command()
@_common_options
@click.option(
    "--method", "-m",
    default="train_test",
    help="验证方法: train_test/monte_carlo/cross_validate/factor_adf/factor_prf/event_study/standard_report",
)
@click.option("--cross-sectional/--no-cross-sectional", default=False, help="多策略横截面打分模式")
def validate(config: str, verbose: bool, method: str, cross_sectional: bool) -> None:
    """执行验证（委托 pipe.validate）。"""
    pipe = _build_pipeline(config, verbose)
    pipe.validate(method=method, cross_sectional=cross_sectional)
    click.echo(f"[OK] 验证完成 method={method}")


# ---------------------------------------------------------------------------
# report 子命令
# ---------------------------------------------------------------------------
@cli.command()
@_common_options
@click.option(
    "--fmt", "-f",
    default="html",
    type=click.Choice(["html", "csv", "json"], case_sensitive=False),
    help="报告格式",
)
def report(config: str, verbose: bool, fmt: str) -> None:
    """生成报告（委托 pipe.report）。"""
    pipe = _build_pipeline(config, verbose)
    pipe.report(fmt=fmt)
    click.echo(f"[OK] 报告生成 fmt={fmt}")


# ---------------------------------------------------------------------------
# adaptors 子命令（规则21.3 工厂自省）
# ---------------------------------------------------------------------------
@cli.command("adaptors")
def adaptors() -> None:
    """列出已注册的数据源适配器（规则21 工厂）。"""
    from core.ext.adapters import list_adapters
    names = list_adapters()
    click.echo(f"已注册 {len(names)} 个数据源适配器：")
    for name in names:
        click.echo(f"  - {name}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main() -> None:
    """console_scripts 入口。"""
    cli()


if __name__ == "__main__":
    main()
