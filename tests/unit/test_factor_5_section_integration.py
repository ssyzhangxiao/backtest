"""
5 段式因子验证报告集成测试（2026-06-12）。

覆盖差距 2 / 3：
  - html_report.build_factor_5_section_html / _df_to_html_table
  - html_report._maybe_build_factor_5_section / _inject_html_anchor
  - plots.plot_factor_prf / plot_event_study_returns / plot_factor_redundancy_heatmap
  - E10._plot_factor_5_section_pngs 集成路径
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from runner.report.html_report import (
    FACTOR_5_SECTION_FILENAME,
    _df_to_html_table,
    _inject_html_anchor,
    _maybe_build_factor_5_section,
    build_factor_5_section_html,
)


# ──────────────────────────────────────────────────────────────────
# 1. 纯函数：build_factor_5_section_html
# ──────────────────────────────────────────────────────────────────


class TestBuildFactor5SectionHtml:
    """5 段式 HTML 片段渲染测试。"""

    def test_minimal_input(self):
        """空 DataFrame + 空 meta：渲染空表 + 占位符。"""
        df = pd.DataFrame({
            "symbol": [],
            "factor": [],
            "fully_validated": [],
        })
        html = build_factor_5_section_html(df, summary_df=None, meta=None)
        assert isinstance(html, str)
        assert "5 段式因子验证报告" in html
        assert "通过条件" in html
        # 摘要卡片渲染占位符
        assert "已验证因子数" in html

    def test_with_summary(self):
        """summary_df 存在时渲染「按因子汇总」表。"""
        full = pd.DataFrame({
            "symbol": ["A", "B"],
            "factor": ["f1", "f1"],
            "n_pass_sections": [5, 4],
            "n_present_sections": [5, 5],
            "fully_validated": [True, False],
            "is_complete": [True, True],
            "pass_adf": [True, False],
            "value_adf": [0.01, 0.5],
        })
        summary = pd.DataFrame({
            "factor": ["f1", "f2"],
            "n_symbols": [2, 1],
            "n_fully_validated": [1, 0],
        })
        html = build_factor_5_section_html(full, summary_df=summary)
        assert "按因子汇总" in html
        assert "每品种 × 每因子 5 段明细" in html

    def test_truncation_large_df(self):
        """> 200 行明细表应截断 + 显示提示。"""
        n = 250
        full = pd.DataFrame({
            "symbol": [f"S{i}" for i in range(n)],
            "factor": [f"f{i % 5}" for i in range(n)],
            "fully_validated": [True] * n,
        })
        html = build_factor_5_section_html(full)
        assert "前 200 行" in html
        assert "factor_standard_report.csv" in html

    def test_meta_fully_validated_rate_format(self):
        """fully_validated_rate 应格式化为百分比。"""
        df = pd.DataFrame({"factor": ["f1"]})
        html = build_factor_5_section_html(
            df, meta={"n_factors": 1, "n_fully_validated": 1, "fully_validated_rate": 0.85}
        )
        assert "85.0%" in html

    def test_threshold_section_present(self):
        """5 段阈值说明区段应出现。"""
        df = pd.DataFrame({"factor": ["f1"]})
        html = build_factor_5_section_html(df)
        for label in ["ADF 平稳性", "IC 绝对值", "PRF 离散信号", "事件研究", "Spearman 冗余", "完整验证"]:
            assert label in html, f"缺少阈值说明：{label}"


# ──────────────────────────────────────────────────────────────────
# 2. _df_to_html_table 布尔与 NaN 处理
# ──────────────────────────────────────────────────────────────────


class TestDfToHtmlTable:
    def test_bool_to_emoji(self):
        df = pd.DataFrame({"pass": [True, False, True]})
        html = _df_to_html_table(df)
        assert "✅" in html
        assert "❌" in html

    def test_nan_to_dash(self):
        df = pd.DataFrame({"v": [1.0, np.nan, 2.0]})
        html = _df_to_html_table(df)
        assert "—" in html

    def test_empty_returns_placeholder(self):
        html = _df_to_html_table(pd.DataFrame())
        assert "空" in html

    def test_table_id_applied(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        html = _df_to_html_table(df, table_id="t1")
        assert 'id="t1"' in html


# ──────────────────────────────────────────────────────────────────
# 3. _inject_html_anchor
# ──────────────────────────────────────────────────────────────────


class TestInjectHtmlAnchor:
    def test_inject_before_body(self, tmp_path: Path):
        f = tmp_path / "report.html"
        f.write_text("<html><body><h1>x</h1></body></html>", encoding="utf-8")
        ok = _inject_html_anchor(f, '<div id="anchor">hi</div>')
        assert ok is True
        text = f.read_text(encoding="utf-8")
        assert text.index('<div id="anchor">hi</div>') < text.index("</body>")

    def test_inject_case_insensitive(self, tmp_path: Path):
        """大写 </BODY> 也应被识别。"""
        f = tmp_path / "report.html"
        f.write_text("<HTML><BODY>x</BODY></HTML>", encoding="utf-8")
        ok = _inject_html_anchor(f, '<div>hi</div>')
        assert ok is True
        text = f.read_text(encoding="utf-8")
        assert '<div>hi</div>' in text
        assert text.find('<div>hi</div>') < text.upper().find("</BODY>")

    def test_no_body_tag_appends(self, tmp_path: Path):
        f = tmp_path / "report.html"
        f.write_text("<p>no body</p>", encoding="utf-8")
        ok = _inject_html_anchor(f, '<div>hi</div>')
        assert ok is True
        assert '<div>hi</div>' in f.read_text(encoding="utf-8")

    def test_nonexistent_file_returns_false(self, tmp_path: Path):
        ok = _inject_html_anchor(tmp_path / "missing.html", "<div></div>")
        assert ok is False


# ──────────────────────────────────────────────────────────────────
# 4. _maybe_build_factor_5_section
# ──────────────────────────────────────────────────────────────────


class TestMaybeBuildFactor5Section:
    def _write_standard_csv(self, tmp_path: Path) -> Path:
        df = pd.DataFrame({
            "symbol": ["A", "A", "B"],
            "factor": ["f1", "f2", "f1"],
            "n_pass_sections": [5, 3, 4],
            "n_present_sections": [5, 5, 5],
            "fully_validated": [True, False, False],
            "is_complete": [True, True, True],
            "pass_adf": [True, False, True],
            "value_adf": [0.01, 0.3, 0.02],
            "pass_ic": [True, False, True],
            "value_ic": [0.05, 0.01, 0.04],
        })
        csv_path = tmp_path / "factor_standard_report.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        return csv_path

    def test_returns_none_when_no_validation(self, tmp_path: Path):
        assert _maybe_build_factor_5_section({}, tmp_path) is None

    def test_returns_none_when_no_standard_report(self, tmp_path: Path):
        results = {"validation": {"train_test": {}}}
        assert _maybe_build_factor_5_section(results, tmp_path) is None

    def test_generates_html_when_csv_exists(self, tmp_path: Path):
        csv = self._write_standard_csv(tmp_path)
        results = {
            "validation": {
                "standard_report": {
                    "output_path": csv,
                    "summary_path": None,
                    "n_factors": 2,
                    "n_fully_validated": 1,
                    "fully_validated_rate": 0.5,
                }
            }
        }
        html_path = _maybe_build_factor_5_section(results, tmp_path)
        assert html_path is not None
        assert html_path.name == FACTOR_5_SECTION_FILENAME
        assert html_path.exists()
        text = html_path.read_text(encoding="utf-8")
        assert "5 段式因子验证报告" in text
        # 摘要卡片
        assert "已验证因子数" in text
        # 因子数据
        assert "f1" in text or "f2" in text

    def test_compat_old_format_with_results_dict(self, tmp_path: Path):
        """旧格式：standard_report 直接是 {results: {sym: DataFrame}}。"""
        df = pd.DataFrame({
            "factor": ["f1", "f2"],
            "fully_validated": [True, False],
        })
        results = {
            "validation": {
                "standard_report": {"results": {"A": df}}
            }
        }
        html_path = _maybe_build_factor_5_section(results, tmp_path)
        assert html_path is not None
        assert html_path.exists()


# ──────────────────────────────────────────────────────────────────
# 5. plots 3 个新图：仅验证不抛异常 + 生成 PNG
# ──────────────────────────────────────────────────────────────────


class TestFactor5SectionPlots:
    """新加的 3 个绘图函数：smoke 测试（不验证具体像素值）。"""

    def test_plot_factor_prf_runs(self, tmp_path: Path):
        from runner.report.plots import plot_factor_prf
        df = pd.DataFrame({
            "factor": ["f1", "f2", "f3"],
            "precision": [0.6, 0.5, 0.7],
            "recall": [0.4, 0.6, 0.5],
            "f1": [0.48, 0.55, 0.58],
            "lift": [0.1, -0.05, 0.2],
        })
        out = tmp_path / "prf.png"
        plot_factor_prf(df, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_plot_factor_prf_skips_on_empty(self, tmp_path: Path):
        from runner.report.plots import plot_factor_prf
        out = tmp_path / "prf_empty.png"
        plot_factor_prf(pd.DataFrame(), out)
        assert not out.exists()

    def test_plot_event_study_returns_runs(self, tmp_path: Path):
        from runner.report.plots import plot_event_study_returns
        df = pd.DataFrame({
            "factor": ["f1"] * 4 + ["f2"] * 4,
            "window": ["T+1", "T+3", "T+5", "T+10"] * 2,
            "mean_return": [0.01, 0.02, 0.025, 0.03, -0.005, 0.01, 0.015, 0.02],
            "p_value": [0.1, 0.05, 0.01, 0.001, 0.8, 0.2, 0.1, 0.05],
        })
        out = tmp_path / "es.png"
        plot_event_study_returns(df, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_plot_event_study_no_pvalue(self, tmp_path: Path):
        """无 p_value 列时仍能渲染（左图有，右图占位）。"""
        from runner.report.plots import plot_event_study_returns
        df = pd.DataFrame({
            "factor": ["f1", "f2"],
            "window": ["T+1", "T+1"],
            "mean_return": [0.01, 0.02],
        })
        out = tmp_path / "es_no_p.png"
        plot_event_study_returns(df, out)
        assert out.exists()

    def test_plot_redundancy_heatmap_runs(self, tmp_path: Path):
        from runner.report.plots import plot_factor_redundancy_heatmap
        corr = pd.DataFrame(
            [[1.0, 0.8, 0.3], [0.8, 1.0, 0.5], [0.3, 0.5, 1.0]],
            index=["f1", "f2", "f3"],
            columns=["f1", "f2", "f3"],
        )
        out = tmp_path / "red.png"
        plot_factor_redundancy_heatmap(corr, out, threshold=0.7)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_plot_redundancy_heatmap_empty(self, tmp_path: Path):
        from runner.report.plots import plot_factor_redundancy_heatmap
        out = tmp_path / "red_empty.png"
        plot_factor_redundancy_heatmap(pd.DataFrame(), out)
        assert not out.exists()


# ──────────────────────────────────────────────────────────────────
# 6. E10 _plot_factor_5_section_pngs 集成
# ──────────────────────────────────────────────────────────────────


class TestE10PlotFactor5SectionPngs:
    """E10 内部辅助函数：standard_report 跑过时生成 PNG。"""

    def test_no_validation_returns_silently(self, tmp_path: Path):
        from runner.backtest.experiments.e10_e11_reporting import (
            _plot_factor_5_section_pngs,
        )
        # 不抛异常 + 不生成 png
        _plot_factor_5_section_pngs({}, tmp_path)
        _plot_factor_5_section_pngs({"validation": {}}, tmp_path)
        assert not (tmp_path / "factor_prf.png").exists()

    def test_runs_pngs_when_csvs_exist(self, tmp_path: Path):
        """模拟 validate/ 下有 PRF/EventStudy csv 时应生成 PNG。"""
        # 构造 input/output 目录结构
        out_dir = tmp_path / "report"
        validate_dir = tmp_path / "validate"
        validate_dir.mkdir()
        out_dir.mkdir()

        # PRF csv
        pd.DataFrame({
            "factor": ["f1", "f2"],
            "precision": [0.6, 0.5],
            "recall": [0.4, 0.6],
            "f1": [0.48, 0.55],
            "lift": [0.1, -0.05],
        }).to_csv(validate_dir / "factor_prf.csv", index=False)

        # EventStudy csv
        pd.DataFrame({
            "factor": ["f1", "f1", "f2", "f2"],
            "window": ["T+1", "T+5", "T+1", "T+5"],
            "mean_return": [0.01, 0.02, 0.005, 0.015],
            "p_value": [0.1, 0.01, 0.5, 0.05],
        }).to_csv(validate_dir / "event_study.csv", index=False)

        # 冗余 csv（边表）
        pd.DataFrame({
            "factor_1": ["f1", "f2"],
            "factor_2": ["f2", "f3"],
            "spearman_rho": [0.8, 0.3],
        }).to_csv(validate_dir / "factor_review_summary.csv", index=False)

        # output_dir.parent = tmp_path，tmp_path/validate 存在 → 走 validate 路径
        results = {"validation": {"standard_report": {"output_path": "x"}}}

        from runner.backtest.experiments.e10_e11_reporting import (
            _plot_factor_5_section_pngs,
        )
        _plot_factor_5_section_pngs(results, out_dir)

        # 三个 PNG 至少生成 PRF 和 EventStudy（冗余图在边表转矩阵后生成）
        assert (out_dir / "factor_prf.png").exists()
        assert (out_dir / "event_study.png").exists()
        # 冗余图在 factor_1/2 边表路径下应生成
        assert (out_dir / "factor_redundancy_heatmap.png").exists()
