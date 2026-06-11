"""
因子标准化验证报告生成器（5 段式）。

聚合 ADF / IC / PRF / 事件研究 / 冗余 5 段，输出统一 CSV。
任一段缺失 → 因子标记为"未完整验证"，禁止进入策略组合。

委托：
- `runner.validation.factor_adf.factor_adf_validation`
- `runner.validation.factor_alpha24.factor_alpha24_screening` (IC)
- `runner.validation.factor_prf.factor_prf_validation`
- `runner.validation.event_study.factor_event_study_validation`
- `runner.validation.factor_review.factor_review_validation` (含 Spearman 冗余)

通过标准（规则 28 阶段 A 完整 5 段）：
- ADF：p_value < 0.05（平稳）
- IC：abs(IC) > 0.03
- PRF：Precision > 0.55 AND Lift > 0
- Event Study：T+5 ~ T+10 p_value < 0.01
- 冗余：与高 IC 因子 Spearman ρ < 0.7

完整验证（fully_validated）= 5 段全部存在 AND 至少 4/5 段通过
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from core.config import BacktestConfig
from runner.common.utils import save_csv
from runner.validation.event_study import factor_event_study_validation
from runner.validation.factor_adf import factor_adf_validation
from runner.validation.factor_prf import factor_prf_validation
from runner.validation.factor_review import factor_review_validation

# 通过阈值（与各模块保持一致）
THRESHOLDS = {
    "adf": {"p_value_max": 0.05},
    "ic": {"abs_ic_min": 0.03},
    "prf": {"precision_min": 0.55, "lift_min": 0.0},
    "event_study": {"p_value_max": 0.01},
    "redundancy": {"abs_corr_max": 0.7},
}

# 输出文件名
OUTPUT_FILENAME = "factor_standard_report.csv"
SUMMARY_FILENAME = "factor_standard_summary.csv"

# 完整验证通过的最少段数
MIN_PASS_SECTIONS = 4
TOTAL_SECTIONS = 5


def _normalize_section_result(
    df: Optional[pd.DataFrame],
    section: str,
    symbol_col: str = "symbol",
    factor_col: str = "factor",
) -> pd.DataFrame:
    """
    将各模块输出规整为 (symbol, factor, pass_{section}) 三列。

    各模块输出列名不同，需要在 `_SECTION_SCHEMA` 中统一。
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=[symbol_col, factor_col, f"pass_{section}"])
    schema = _SECTION_SCHEMA.get(section, {})
    if not schema:
        return df
    out = df[[symbol_col, factor_col]].copy()
    out[f"pass_{section}"] = df[schema["pass_col"]].astype(bool) if schema["pass_col"] in df.columns else False
    out[f"value_{section}"] = df[schema["value_col"]] if schema["value_col"] in df.columns else np.nan
    return out


# 各模块的 (pass 列, 主指标列) 映射
_SECTION_SCHEMA = {
    "adf": {"pass_col": "is_stationary", "value_col": "p_value"},
    "ic": {"pass_col": "is_pass", "value_col": "ic_mean"},
    "prf": {"pass_col": "is_pass", "value_col": "precision"},
    "event_study": {"pass_col": "is_pass", "value_col": "t5_pvalue"},
    "redundancy": {"pass_col": "is_pass", "value_col": "max_abs_corr"},
}


def factor_standard_report_validation(
    data_source,
    config: BacktestConfig,
    lib=None,
    output_dir: Path = Path("output/validate"),
    best_params: Optional[Dict[str, Dict[str, Any]]] = None,
    cross_sectional: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    因子标准化验证报告（5 段式聚合）。

    Args:
        data_source: PyBrokerDataSource
        config: 回测配置
        lib: 策略库
        output_dir: 输出目录
        best_params: 最优参数
        cross_sectional: 是否横截面
        **kwargs:
            skip_sections: 跳过的段列表，如 ["redundancy"] 加速

    Returns:
        {
            "output_path": Path,
            "summary_path": Path,
            "n_factors": int,
            "n_fully_validated": int,
            "fully_validated_rate": float,
        }
    """
    skip: List[str] = list(kwargs.get("skip_sections", []))

    logger.info("=" * 60)
    logger.info(f"因子标准化报告（5 段式，{MIN_PASS_SECTIONS}/{TOTAL_SECTIONS} 段通过）")
    logger.info("=" * 60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) ADF 平稳性
    adf_res: Dict[str, pd.DataFrame] = {}
    if "adf" not in skip:
        adf_out = factor_adf_validation(
            data_source, config, lib, output_dir, best_params, cross_sectional
        )
        adf_res = adf_out.get("results", {})

    # 2) IC（复用现有 factor_alpha24 验证的 IC 评估部分）
    ic_res: Dict[str, pd.DataFrame] = {}
    if "ic" not in skip:
        try:
            from runner.validation.factor_alpha24 import factor_alpha24_screening
            ic_out = factor_alpha24_screening(
                data_source, config, lib, output_dir, best_params, cross_sectional
            )
            # factor_alpha24_screening 输出 dict，结构与 factor_adf 略有不同
            # 解析为统一格式
            for sym, val in ic_out.items():
                if isinstance(val, pd.DataFrame):
                    if "is_pass" in val.columns and "ic_mean" in val.columns:
                        ic_res[sym] = val
                    elif "is_pass" in val.columns:
                        val = val.copy()
                        if "ic_mean" not in val.columns:
                            val["ic_mean"] = np.nan
                        ic_res[sym] = val
        except Exception as e:
            logger.warning(f"  IC 验证失败: {e}")

    # 3) PRF
    prf_res: Dict[str, pd.DataFrame] = {}
    if "prf" not in skip:
        prf_out = factor_prf_validation(
            data_source, config, lib, output_dir, best_params, cross_sectional
        )
        prf_res = prf_out.get("results", {})

    # 4) 事件研究
    es_res: Dict[str, pd.DataFrame] = {}
    if "event_study" not in skip:
        es_out = factor_event_study_validation(
            data_source, config, lib, output_dir, best_params, cross_sectional
        )
        es_res = es_out.get("results", {})

    # 5) 冗余（复用 factor_review 的 6 项复核，其中 _check_orthogonality 给出最大互相关）
    red_res: Dict[str, pd.DataFrame] = {}
    if "redundancy" not in skip:
        try:
            red_out = factor_review_validation(
                data_source, config, lib, output_dir, best_params, cross_sectional
            )
            # 解析 review_report DataFrame，提取正交性指标
            for sym, val in red_out.items():
                if isinstance(val, pd.DataFrame):
                    red_res[sym] = val
        except Exception as e:
            logger.warning(f"  冗余验证失败: {e}")

    # 6) 聚合 5 段结果
    all_symbols = set(adf_res) | set(ic_res) | set(prf_res) | set(es_res) | set(red_res)
    if not all_symbols:
        logger.warning("  无任何段输出，无法生成报告")
        return {
            "output_path": None,
            "summary_path": None,
            "n_factors": 0,
            "n_fully_validated": 0,
            "fully_validated_rate": 0.0,
        }

    report_rows: List[Dict[str, Any]] = []
    for sym in sorted(all_symbols):
        sections: Dict[str, pd.DataFrame] = {
            "adf": adf_res.get(sym, pd.DataFrame()),
            "ic": ic_res.get(sym, pd.DataFrame()),
            "prf": prf_res.get(sym, pd.DataFrame()),
            "event_study": es_res.get(sym, pd.DataFrame()),
            "redundancy": red_res.get(sym, pd.DataFrame()),
        }
        # 收集所有因子
        all_factors: set[str] = set()
        for sec_df in sections.values():
            if not sec_df.empty and "factor" in sec_df.columns:
                all_factors.update(sec_df["factor"].astype(str).tolist())
        # 按因子聚合
        for fname in sorted(all_factors):
            row: Dict[str, Any] = {"symbol": sym, "factor": fname}
            n_pass_sections = 0
            n_present_sections = 0
            for sec, sec_df in sections.items():
                if sec_df.empty or "factor" not in sec_df.columns:
                    row[f"pass_{sec}"] = False
                    row[f"value_{sec}"] = np.nan
                    row[f"present_{sec}"] = False
                    continue
                rec = sec_df[sec_df["factor"].astype(str) == fname]
                if len(rec) == 0:
                    row[f"pass_{sec}"] = False
                    row[f"value_{sec}"] = np.nan
                    row[f"present_{sec}"] = False
                    continue
                n_present_sections += 1
                schema = _SECTION_SCHEMA[sec]
                pass_val = bool(rec[schema["pass_col"]].iloc[0]) if schema["pass_col"] in rec.columns else False
                value_val = rec[schema["value_col"]].iloc[0] if schema["value_col"] in rec.columns else np.nan
                row[f"pass_{sec}"] = pass_val
                row[f"value_{sec}"] = value_val
                row[f"present_{sec}"] = True
                if pass_val:
                    n_pass_sections += 1
            row["n_pass_sections"] = n_pass_sections
            row["n_present_sections"] = n_present_sections
            # 完整验证 = 5 段全部存在 AND ≥4 段通过
            row["fully_validated"] = bool(
                n_present_sections == TOTAL_SECTIONS and n_pass_sections >= MIN_PASS_SECTIONS
            )
            # 任一段缺失即标记"未完整验证"
            row["is_complete"] = bool(n_present_sections == TOTAL_SECTIONS)
            if not row["is_complete"]:
                row["fully_validated"] = False
            report_rows.append(row)

    full_df = pd.DataFrame(report_rows)
    if full_df.empty:
        logger.warning("  标准化报告为空")
        return {
            "output_path": None,
            "summary_path": None,
            "n_factors": 0,
            "n_fully_validated": 0,
            "fully_validated_rate": 0.0,
        }

    out_path = output_dir / OUTPUT_FILENAME
    save_csv(full_df, out_path)
    logger.info(f"  标准化报告已保存: {out_path}")

    # 摘要：按因子汇总
    summary = (
        full_df.groupby("factor")
        .agg(
            n_symbols=("symbol", "nunique"),
            n_pass_adf=("pass_adf", "sum"),
            n_pass_ic=("pass_ic", "sum"),
            n_pass_prf=("pass_prf", "sum"),
            n_pass_event_study=("pass_event_study", "sum"),
            n_pass_redundancy=("pass_redundancy", "sum"),
            n_fully_validated=("fully_validated", "sum"),
        )
        .reset_index()
    )
    summary_path = output_dir / SUMMARY_FILENAME
    save_csv(summary, summary_path)
    logger.info(f"  摘要已保存: {summary_path}")

    n_factors = int(full_df["factor"].nunique())
    n_fully = int(full_df["fully_validated"].sum())
    n_total = len(full_df)
    return {
        "output_path": out_path,
        "summary_path": summary_path,
        "n_factors": n_factors,
        "n_total_records": n_total,
        "n_fully_validated": n_fully,
        "fully_validated_rate": n_fully / max(n_total, 1),
    }


__all__ = [
    "factor_standard_report_validation",
    "THRESHOLDS",
    "MIN_PASS_SECTIONS",
    "TOTAL_SECTIONS",
    "OUTPUT_FILENAME",
    "SUMMARY_FILENAME",
]
