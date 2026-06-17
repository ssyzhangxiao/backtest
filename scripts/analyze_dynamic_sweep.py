"""
分析 sweep 结果：按 sharpe / calmar / mdd 综合排序，输出 top-N。

输入: output_backtest_pybroker/cta_dynamic_sweep_{full,oos}/summaries.json
输出: stdout 表格 + 标记最佳配置
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def load_summaries(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def analyze(period_name: str, summaries: List[Dict[str, Any]]) -> None:
    lin = [s for s in summaries if s.get("blend_method") == "linear" and s.get("status") == "ok"]
    dyn = [s for s in summaries if s.get("blend_method") == "dynamic" and s.get("status") == "ok"]

    # 线性基线
    if lin:
        best_lin = max(lin, key=lambda s: s.get("sharpe") or -1e9)
        best_lin_ret = max(lin, key=lambda s: s.get("total_return_pct") or -1e9)
        print(f"\n=== {period_name} 线性基线（按 sharpe） ===")
        print(
            f"  {best_lin['tag']:<14}  "
            f"sharpe={best_lin['sharpe']:.4f}  "
            f"return={best_lin['total_return_pct']:6.2f}%  "
            f"mdd={best_lin['max_drawdown_pct']:6.2f}%  "
            f"calmar={_calmar(best_lin):.2f}"
        )
        print(f"  (按收益: {best_lin_ret['tag']} 收益={best_lin_ret['total_return_pct']:.2f}%)")

    # 动态 top5 by sharpe
    print(f"\n=== {period_name} dynamic top-5 (按 sharpe) ===")
    top5 = sorted(dyn, key=lambda s: s.get("sharpe") or -1e9, reverse=True)[:5]
    print(f"  {'tag':<24}  {'sharpe':>8}  {'return%':>8}  {'mdd%':>7}  {'calmar':>7}")
    for s in top5:
        print(
            f"  d_b{s['xs_position_base']}_p{s['xs_opposite_penalty']:<8.1f}  "
            f"{s['sharpe']:>8.4f}  "
            f"{s['total_return_pct']:>7.2f}%  "
            f"{s['max_drawdown_pct']:>6.2f}%  "
            f"{_calmar(s):>7.2f}"
        )

    # 动态 top5 by mdd（最低回撤）
    print(f"\n=== {period_name} dynamic top-5 (按最低 mdd) ===")
    top5_mdd = sorted(dyn, key=lambda s: s.get("max_drawdown_pct") or 0)[:5]
    print(f"  {'tag':<24}  {'mdd%':>7}  {'sharpe':>8}  {'return%':>8}  {'calmar':>7}")
    for s in top5_mdd:
        print(
            f"  d_b{s['xs_position_base']}_p{s['xs_opposite_penalty']:<8.1f}  "
            f"{s['max_drawdown_pct']:>6.2f}%  "
            f"{s['sharpe']:>8.4f}  "
            f"{s['total_return_pct']:>7.2f}%  "
            f"{_calmar(s):>7.2f}"
        )

    # 收益/回撤改善
    if lin and dyn:
        lin_ref = max(lin, key=lambda s: s.get("sharpe") or -1e9)
        best_dyn = sorted(dyn, key=lambda s: s.get("sharpe") or -1e9, reverse=True)[0]
        print(f"\n=== {period_name} 最佳 dynamic vs 最佳 linear ===")
        print(f"  dynamic  : b={best_dyn['xs_position_base']}, p={best_dyn['xs_opposite_penalty']}")
        print(f"             sharpe={best_dyn['sharpe']:.4f}  "
              f"return={best_dyn['total_return_pct']:6.2f}%  "
              f"mdd={best_dyn['max_drawdown_pct']:6.2f}%  "
              f"calmar={_calmar(best_dyn):.2f}")
        print(f"  linear   : {lin_ref['tag']}")
        print(f"             sharpe={lin_ref['sharpe']:.4f}  "
              f"return={lin_ref['total_return_pct']:6.2f}%  "
              f"mdd={lin_ref['max_drawdown_pct']:6.2f}%  "
              f"calmar={_calmar(lin_ref):.2f}")
        ret_drop = best_dyn["total_return_pct"] - lin_ref["total_return_pct"]
        mdd_improve = best_dyn["max_drawdown_pct"] - lin_ref["max_drawdown_pct"]  # 负数 = 改善
        sharpe_diff = best_dyn["sharpe"] - lin_ref["sharpe"]
        print(f"  → 收益差: {ret_drop:+.2f}pp, "
              f"回撤差: {mdd_improve:+.2f}pp (负=减少), "
              f"sharpe差: {sharpe_diff:+.4f}")


def _calmar(s: Dict[str, Any]) -> float:
    mdd = abs(s.get("max_drawdown_pct") or 0)
    if mdd < 1e-6:
        return 0.0
    return (s.get("total_return_pct") or 0) / mdd


def main() -> int:
    base = Path("output_backtest_pybroker")
    for name, sub in [("全期 (2016-2025)", "cta_dynamic_sweep_full"),
                       ("OOS (2021-2025)", "cta_dynamic_sweep_oos")]:
        path = base / sub / "summaries.json"
        if not path.exists():
            print(f"[skip] {path} 不存在")
            continue
        analyze(name, load_summaries(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
