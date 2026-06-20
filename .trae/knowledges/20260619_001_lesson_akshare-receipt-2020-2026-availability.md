# AKShare 仓单数据 2020-2026 完整性验证

**创建时间**：2026-06-19
**akshare 版本**：1.17.7
**网络环境**：本机 macOS

## 结论

akshare 1.17.7 在当前网络环境下，**不能** 提供 2020-2026 完整仓单历史数据。
探查脚本：[scripts/_probe_akshare_history.py](file:///Users/luojiutian/Documents/backtest/scripts/_probe_akshare_history.py)

## 可用性矩阵

| 交易所 | 2020-01-15 | 2022-06-15 | 2024-06-15 | 2026-05-15 | 备注 |
|--------|-----------|-----------|-----------|-----------|------|
| SHFE   | ❌ ConnectionError | ❌ ConnectionError | ❌ ConnectionError | ❌ ConnectionError | 域名 tsite.shfe.com.cn DNS 失败 |
| DCE    | ❌ ValueError | ❌ ValueError | ❌ ValueError | ❌ ValueError | HTTP 412 反爬虫阻断 |
| CZCE   | ✅ 20 sheets/1058 rows | ✅ 23 sheets/835 rows | ❌ ValueError | ❌ ValueError | 2024+ XLS 解析失败 |
| GFEX   | ⊘ EMPTY | ⊘ EMPTY | ⊘ EMPTY | ⊘ EMPTY | 多数品种本身无仓单 |

## 详细分析

### SHFE（上期所）
- 域名 `tsite.shfe.com.cn` 在本机 DNS 解析失败（`gaierror: nodename nor servname provided`）
- 备用域名 `www.shfe.com.cn`（220.248.39.134）可达，但不是仓单 API
- **`ak.futures_shfe_warehouse_receipt` 完全无法使用**
- 我们的 `_fetch_from_shfe` 同样依赖此域名 → 也会失败

### DCE（大商所）
- 域名 `www.dce.com.cn` 可达（218.25.154.72）
- **但仓单接口返回 HTTP 412（Precondition Failed）反爬虫阻断**
- 即使用 requests 直连（绕过 akshare 解析层），HTML 内也无 `<table>` 标签
- **`ak.futures_dce_warehouse_receipt` 完全无法使用**
- 我们的 `_fetch_from_dce` 用相同 URL+params → 也会失败

### CZCE（郑商所）
- 域名 `www.czce.com.cn` 可达（222.88.29.179）
- **2020-2022 期间 XLS 文件可正常解析**（20-23 个品种 sheet，800-1000 行）
- **2024 年起 XLS 文件格式变化**（可能是 XLSX），`pd.read_excel` 报 "Excel file format cannot be determined"
- **`ak.futures_czce_warehouse_receipt` 仅支持 2020-2023 早期 XLS**
- 我们的 `_fetch_from_czce` 用 `pd.read_excel(BytesIO(r.content), sheet_name=None)` → 同样会在 2024+ 失败

### GFEX（广期所）
- 所有抽样日均返回 `{}` 空 dict
- **GFEX 多数品种本身无仓单概念**（工业硅、碳酸锂等已上市品种无标准仓单）
- **预期行为，非 akshare 问题**

## 对四因子 CTA 回测的影响

e12 实验 9 个品种的 receipt 数据可用性：

| 品种 | 交易所 | akshare 真实数据可用性 | e2e 表现 |
|------|--------|---------------------|---------|
| SHFE.AL/CU/RU/RB/HC | SHFE | ❌ DNS 失败 | 依赖 mock |
| DCE.M/PP | DCE | ❌ HTTP 412 | 依赖 mock |
| CZCE.FG/CF | CZCE | ⚠️ 2020-2023 OK，2024+ 失败 | mock 可补 |
| GFEX | — | ⊘ 无仓单 | N/A |

**当前 e2e 验证的 receipt 因子有效性完全依赖 mock 数据**——不能用真实 akshare 数据回放。

## 建议方案

### 短期（不依赖 akshare）
- 使用现有 `core/data/_receipt_adapters.py`（4 交易所直接调官方 JSON/HTML/XLS）
- 现状：DCE 和 SHFE 在本机网络不可用 → e2e 只能 mock
- 解决：换有公网出口的环境运行（生产环境/服务器）

### 中期（akshare 升级 + 修 SHFE）
- 升级 akshare 到最新版本（1.18+），看是否修复 CZCE XLS 解析
- 排查 SHFE 域名切换（akshare 是否已支持 `query.shfe.com.cn` 备用域名）

### 长期（专业数据源）
- 考虑 `tushare`（需 token）或 `rqdatac`（需付费）作为生产环境主数据源
- 已在 `pyproject.toml` 列为规划中依赖

## 相关代码位置

- [core/data/_receipt_adapters.py](file:///Users/luojiutian/Documents/backtest/core/data/_receipt_adapters.py)：4 交易所适配器（不依赖 akshare）
- [core/data/receipt_fetcher.py](file:///Users/luojiutian/Documents/backtest/core/data/receipt_fetcher.py)：仓单 fetcher 业务逻辑
- [runner/backtest/experiments/e12_four_factor.py](file:///Users/luojiutian/Documents/backtest/runner/backtest/experiments/e12_four_factor.py)：四因子 CTA 回测
- [scripts/_probe_akshare_history.py](file:///Users/luojiutian/Documents/backtest/scripts/_probe_akshare_history.py)：本探查脚本
