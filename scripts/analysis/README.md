# scripts/analysis/

**作用**：对回测结果做后置分析（对比 / 排序 / 可视化），不参与回测流程本身。

**与 experiments/ 的区别**：
- `experiments/`：产生回测数据（exp / sweep）
- `analysis/`：消费回测数据（analyze / compare / visualize）

**当前清单**：

| 文件 | 用途 |
|------|------|
| `analyze_direction2_deep.py` | 方向二 — 深度诊断（信号分布 / 调仓命中率 / 归因） |
| `analyze_dynamic_sweep.py` | 2D sweep 结果按 sharpe/mdd 排序的对比分析 |
| `compare_sweep_runs.py` | 多组 sweep 结果对比（--old / --new） |

**调用范式**：

```bash
# 对 output_backtest_pybroker/<run_id>/ 目录做分析
python scripts/analysis/analyze_direction2_deep.py output_backtest_pybroker/run_xxx

# 对比两次 sweep
python scripts/analysis/compare_sweep_runs.py \
    --old output_backtest_pybroker/sweep_v1 \
    --new output_backtest_pybroker/sweep_v2
```

**维护规则**：
- 不允许在 `scripts/` 根目录新增 `analyze_*.py` / `compare_*.py`
- 分析脚本应该是"纯函数 + argparse"，不修改回测输出
- 报告/图表输出到 `output_backtest_pybroker/<script_name>/`
