# scripts/experiments/

**作用**：方向验证 / 单策略诊断 / 参数扫描的实验性脚本。

**与生产入口的区别**：
- 生产入口（`scripts/run_*.py`）：已收敛到 Pipeline 编排器（规则 17.2）
- 实验脚本：单次性研究 / 调参，**不进入 Pipeline**，仅供回溯

**当前清单**：

| 文件 | 用途 | 状态 |
|------|------|------|
| `exp_pair_trading.py` | 方向三 — 配对交易横截面信号验证 | 已归档（结论：协整不足） |
| `exp_oi_signal.py` | 方向四 P1 — 持仓量衍生信号验证 | 已归档（结论：不达标） |
| `sweep_direction2_fine.py` | 方向二 — 精细参数扫描 | 历史调参 |
| `sweep_cta_hybrid_weight.py` | CTA hybrid 权重 1D 扫描 | 历史调参 |
| `sweep_cta_hybrid_dynamic.py` | CTA hybrid dynamic 2D 扫描 | 历史调参 |

**调用范式**：

```bash
# 单策略诊断（直接执行）
python scripts/experiments/exp_pair_trading.py

# 参数扫描
python scripts/experiments/sweep_cta_hybrid_dynamic.py --help
```

**维护规则**：
- 新增实验脚本必须加 docstring 注明"目的 / 结论 / 状态"
- 失败的实验归档到 `scripts/experiments/_archive/` 子目录（不是 root archive/）
- 不允许在 `scripts/` 根目录新增 `exp_*.py` / `sweep_*.py`
- 实验脚本的引用必须更新对应 docstring（保持 self-documenting）
