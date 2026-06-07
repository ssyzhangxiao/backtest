#!/usr/bin/env python3
"""
合并规则脚本：将所有模块化规则文件合并成单一汇总文件
使用方法：python merge-rules.py
输出：生成 project_rules.md
"""

import os
from pathlib import Path

# 规则文件的顺序（按规则编号）
RULE_ORDER = [
    "02-engine/01-no-fallback.md",
    "01-basics/02-config.md",
    "01-basics/03-deprecated.md",
    "03-strategies/04-risk-control.md",
    "03-strategies/05-strategy-registry.md",
    "01-basics/06-testing.md",
    "01-basics/07-file-limit.md",
    "01-basics/08-naming.md",
    "04-factors/09-factor-dev.md",
    "05-validation/10-removed.md",
    "03-strategies/11-multi-tf.md",
    "05-validation/12-removed.md",
    "03-strategies/13-stop-loss.md",
    "05-validation/14-removed.md",
    "05-validation/15-backtest-validation.md",
    "01-basics/16-directory.md",
    "01-basics/17-common-systems.md",
    "01-basics/18-pipeline.md",
    "01-basics/19-dependencies.md",
    "04-factors/20-factor-cleaning.md",
    "03-strategies/21-sub-strategies.md",
    "05-validation/22-rolling-window.md",
    "04-factors/23-factor-refactor.md",
    "03-strategies/24-indicator-registry.md",
    "03-strategies/25-exit-hooks.md",
    "02-engine/26-cross-validation.md",
    "03-strategies/27-strategy-base.md",
]

def main():
    script_dir = Path(__file__).parent
    output_file = script_dir / "project_rules.md"
    
    # 开始写入
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# 量化回测系统开发规范\n\n")
        f.write("---\n\n")
        f.write("> **注意**：本文件由 `merge-rules.py` 自动生成，请勿直接编辑。\n")
        f.write("> 如需修改规则，请编辑对应分类目录下的规则文件，然后重新运行合并脚本。\n\n")
        f.write("---\n\n")
        
        # 按顺序合并所有规则文件
        for rule_path in RULE_ORDER:
            full_path = script_dir / rule_path
            if full_path.exists():
                with open(full_path, "r", encoding="utf-8") as rule_f:
                    content = rule_f.read()
                    f.write(content)
                    f.write("\n\n---\n\n")
            else:
                print(f"警告：规则文件不存在 - {rule_path}")
        
        # 添加末尾信息
        f.write("---\n\n")
        f.write("*最后更新：2026-06-06*\n")
        f.write("*参考指南：商品期货量化模型改造指南.docx*\n")
        f.write("*参考指南：商品期货 Alpha 因子库工程化重构提示词.docx*\n")
        f.write("*相关知识文档：../knowledges/20260602_001_workflow_strategy-enhancement-roadmap.md*\n")
        f.write("*相关知识文档：../knowledges/20260602_002_workflow_runner-scripts-refactor-plan.md*\n")
    
    print(f"✓ 规则合并完成！输出文件：{output_file}")
    print(f"✓ 共合并 {len(RULE_ORDER)} 个规则文件")

if __name__ == "__main__":
    main()
