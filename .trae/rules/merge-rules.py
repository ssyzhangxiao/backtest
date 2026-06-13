#!/usr/bin/env python3
"""
合并规则脚本：将所有模块化规则文件合并成单一汇总文件
使用方法：python merge-rules.py
输出：生成 project_rules.md

自动发现所有规则文件并按规则编号排序，无需手动维护 RULE_ORDER。
"""

import os
import re
from pathlib import Path


def _extract_rule_number(filepath: Path) -> int:
    """从文件名提取规则编号（如 '02-config.md' → 2, '30-data-quality.md' → 30）。"""
    m = re.match(r"(\d+)", filepath.stem)
    return int(m.group(1)) if m else 999


def _sort_key(filepath: Path) -> tuple:
    """排序键：(规则编号, 分类权重)"""
    num = _extract_rule_number(filepath)
    # 分类权重：同一编号下，按目录字母序排列
    dir_weight = filepath.parent.name
    return (num, dir_weight, filepath.name)


def main():
    script_dir = Path(__file__).parent
    output_file = script_dir / "project_rules.md"

    # 自动发现所有规则 md 文件（排除自身和 README/archive）
    exclude_names = {"project_rules.md", "project_rules.archived.md", "README.md", "merge-rules.py"}
    rule_files = sorted(
        [f for f in script_dir.rglob("*.md") if f.name not in exclude_names],
        key=_sort_key,
    )

    print(f"发现 {len(rule_files)} 个规则文件:")
    for f in rule_files:
        rel = f.relative_to(script_dir)
        print(f"  {rel}")

    # 开始写入
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# 量化回测系统开发规范\n\n")
        f.write("---\n\n")
        f.write("> **注意**：本文件由 `merge-rules.py` 自动生成，请勿直接编辑。\n")
        f.write("> 如需修改规则，请编辑对应分类目录下的规则文件，然后重新运行合并脚本。\n\n")
        f.write("---\n\n")

        # 按排序合并所有规则文件
        for rule_path in rule_files:
            try:
                with open(rule_path, "r", encoding="utf-8") as rule_f:
                    content = rule_f.read()
                    f.write(content)
                    f.write("\n\n---\n\n")
            except Exception as e:
                print(f"  错误：无法读取 {rule_path}: {e}")

        # 添加末尾信息
        f.write("---\n\n")
        f.write("*最后更新：2026-06-13*\n")
        f.write("*参考指南：商品期货量化模型改造指南.docx*\n")
        f.write("*参考指南：商品期货 Alpha 因子库工程化重构提示词.docx*\n")
        f.write("*相关知识文档：../knowledges/20260602_001_workflow_strategy-enhancement-roadmap.md*\n")
        f.write("*相关知识文档：../knowledges/20260602_002_workflow_runner-scripts-refactor-plan.md*\n")

    file_size = output_file.stat().st_size
    print(f"\n✓ 规则合并完成！输出文件：{output_file}")
    print(f"✓ 共合并 {len(rule_files)} 个规则文件 ({file_size} bytes)")


if __name__ == "__main__":
    main()
