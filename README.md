# 商品期货多策略量化回测系统

基于 PyBroker 回测引擎的商品期货多策略量化回测平台，支持展期法连续合约、多策略组合、动态权重调整、参数优化与全面的可视化分析。

## 功能概览

| 模块 | 功能 |
|------|------|
| **数据加载** | 自动识别合约/品种两种 CSV 格式，主力合约识别，展期法连续序列构建 |
| **数据筛选** | 三层数据架构（全量/筛选/消费），支持品种多选和日期范围，侧边栏实时状态显示 |
| **策略引擎** | 双均线趋势、RSI 反转、跨期套利、期限结构套利、波动率突破五种内置策略，支持自定义扩展 |
| **展期管理** | 时间触发 / 流动性触发 / 价差触发三种展期模式 |
| **组合管理** | 多策略组合，基于市场状态的动态权重调整 |
| **风控系统** | 单笔止损、仓位控制、日亏损限制、展期成本容忍度 |
| **参数优化** | 网格搜索 + 滚动窗口优化，支持 6 种优化指标 |
| **市场状态** | 市场环境分类引擎（v3，无前视偏差）：8种市场环境、9个量化指标、滚动IC权重、滚动百分位阈值 |
| **统一回测** | 命令行驱动的一键批量回测，多策略并行对比，样本内/外分割，自动生成 HTML 报告 |
| **可视化** | 37 种 Plotly 图表，覆盖数据概览、策略绩效、风险归因、交易执行、参数优化、市场状态六大模块 |

## 数据源说明

系统支持两种数据源，可互相配合使用：

| 数据源 | 模块 | 类型 | 历史长度 | 加载速度 | 网络依赖 | 展期支持 |
|--------|------|------|----------|----------|----------|----------|
| **本地CSV** | `data_loader.py` | 品种连续/合约模式 | 最长21年（2005年起） | 0.2秒 | 无 | 品种模式无，合约模式有 |
| **TqSdk实时** | `data_loader_tqsdk.py` | 独立合约 + 展期 | 最长10.4年（2016年起） | 2~8秒 | 需要登录 | 真实展期支持 |

### 数据源对比（2016-2026 对齐窗口回测，3品种: 螺纹钢+豆粕+PTA）

| 策略 | 指标 | 本地CSV | TqSdk独立+展期 |
|------|------|---------|----------------|
| dual_ma | 总收益率 | +6.45% | -2.25% |
| dual_ma | 最大回撤 | -18.77% | -10.14% |
| dual_ma | 交易笔数 | 385 | **87** |
| rsi | 总收益率 | -10.16% | **+0.52%** |
| rsi | 最大回撤 | -14.46% | **-5.83%** |
| rsi | 交易笔数 | 603 | **128** |

> PyBroker 的 `_pct` 指标值直接是百分比（如 6.45 即 6.45%），无需额外乘 100。

**关键结论：**

1. **本地CSV 与 TqSdk 回测结果可互相印证**，趋势方向一致
2. **TqSdk 独立合约 + 展期** 大幅减少交易噪音（交易量降低 78%），回撤改善显著（~8-9pp），是更真实的回测方式
3. **本地CSV 优势**：数据更长（21年 vs 10.4年）、零延迟、无网络依赖，适合长期回测
4. **TqSdk 优势**：实时更新、真实合约展期，适合实时策略验证

### 使用建议

- **长期历史回测、参数优化** → 使用本地CSV（数据范围 2005-2026）
- **展期策略验证、真实交易模拟** → 使用 TqSdk 独立合约模式

## 项目结构

```
backtest/
├── app.py                  # Streamlit 前端入口（已模块化）
├── config.py               # 全局配置与常量
├── unified_backtest.py     # 统一回测脚本（一键批量回测）
├── backtest_visualization.py  # 可视化工具（供统一回测脚本调用）
├── requirements.txt        # Python 依赖
├── .streamlit/
│   └── config.toml         # Streamlit 配置
├── core/
│   ├── data_loader.py         # 数据加载器（本地CSV，合约/品种双模式）
│   ├── data_loader_tqsdk.py   # TqSdk 数据源（独立合约 + 展期）
│   ├── environment.py         # 自适应市场状态引擎（旧版）
│   ├── market_regime/        # 市场环境分类引擎（v3，无前视偏差）
│   ├── strategies/         # 策略模块目录
│   │   ├── __init__.py
│   │   ├── base.py         # 策略基类
│   │   ├── registry.py     # 策略注册表
│   │   ├── dual_ma.py      # 双均线策略
│   │   ├── rsi.py          # RSI 反转策略
│   │   ├── spread.py       # 跨期套利策略
│   │   ├── term_structure.py  # 期限结构套利策略
│   │   └── vol_breakout.py     # 波动率突破策略
│   ├── rollover.py         # 展期管理器
│   ├── risk_manager.py     # 风控管理器
│   ├── portfolio.py        # 组合管理器
│   └── optimizer.py        # 参数优化器
├── pages/                  # Streamlit 页面
│   ├── data_import.py      # 数据导入页面
│   ├── data_analysis.py    # 数据分析页面
│   ├── backtest.py         # 回测运行页面
│   └── optimization.py     # 参数优化页面
├── components/             # Streamlit 组件
│   ├── sidebar.py          # 侧边栏配置组件
│   └── results.py          # 回测结果渲染组件
├── utils/                  # 工具模块
│   ├── session_state.py    # 会话状态管理
│   ├── pybroker_helpers.py # PyBroker 辅助工具
│   ├── date_utils.py       # 日期处理工具
│   ├── metrics.py          # 绩效指标计算器
│   └── plots.py            # PlotManager 图表模块（37 种图表）
└── data/                   # 期货数据目录
    ├── SHFE.RB.csv         # 示例：螺纹钢品种指数
    ├── DCE.I.csv           # 示例：铁矿石品种指数
    └── ...
```

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repo-url>
cd backtest

# 创建虚拟环境（推荐 Python 3.10+）
python -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 准备数据

将期货 CSV 数据放入 `data/` 目录。系统支持两种格式：

**格式一：品种指数模式**（推荐入门使用）

文件命名：`交易所.品种.csv`，如 `SHFE.RB.csv`、`DCE.I.csv`

| 列名 | 说明 |
|------|------|
| datetime | 日期时间 |
| open | 开盘价 |
| high | 最高价 |
| low | 最低价 |
| close | 收盘价 |
| volume | 成交量 |
| position | 持仓量 |

**格式二：合约模式**（支持展期）

文件命名：`交易所.品种.csv`，但数据包含多个合约

| 列名 | 说明 |
|------|------|
| date | 日期 |
| symbol | 合约代码（如 RB2310） |
| open | 开盘价 |
| high | 最高价 |
| low | 最低价 |
| close | 收盘价 |
| volume | 成交量 |
| open_interest | 持仓量 |

> 系统会自动检测数据格式，品种模式下不启用展期功能。

### 3. 启动应用

```bash
streamlit run app.py
```

浏览器打开 `http://localhost:8501` 即可使用。

### 4. 一键批量回测（命令行）

使用 `unified_backtest.py` 快速进行批量回测对比：

```bash
# 运行双均线策略对比（4种变体）
python unified_backtest.py --scenario dual_ma_comparison

# 运行新策略（期限结构套利+波动率突破）样本内/外回测
python unified_backtest.py --scenario new_strategies

# 使用自定义配置文件
python unified_backtest.py --config backtest_config_example.json

# 查看帮助
python unified_backtest.py --help
```

运行后会在当前目录下生成 `unified_backtest_results_YYYYMMDD_HHMMSS/`，包含：
- `report.html` - 综合分析报告
- `equity_curves.html` - 净值曲线对比
- `drawdown_curves.html` - 回撤曲线对比
- `comparison_table.csv` - 绩效指标对比表
- `portfolio_*.csv` - 各策略净值数据
- `trades_*.csv` - 各策略交易记录

## 使用指南

### 第一步：数据导入

1. 在左侧导航选择 **📁 数据导入**
2. 确认数据目录路径（默认 `./data`）
3. 可设置文件匹配模式（如 `SHFE.*.csv` 仅加载上期所品种）
4. 点击 **加载数据**
5. 加载成功后可：
   - 查看品种列表和数据概览
   - 使用 **数据切片** 选择回测合约和日期范围
     - **选择回测合约**：默认全选，可多选
     - **回测开始/结束日期**：默认数据最早/最晚日期
   - 预览原始数据和展期信息
   - 预览数据会自动应用当前选择的合约和日期筛选
   - 查看左侧边栏 **📌 当前回测范围** 实时确认筛选是否生效

#### 数据切片与回测范围（三层数据架构）

| 配置项 | 说明 |
|--------|------|
| 选择回测合约 | 多选，默认全选。仅对选中合约进行回测 |
| 回测开始/结束日期 | 日期范围，默认数据最早/最晚 |
| 预览数据行数 | 筛选后自动更新，仅显示符合条件的数据前 20 行 |

**数据架构说明：**

```
pybroker_df_full  (全量数据, 加载后只读不修改)
       │
       ▼  用户筛选 (品种 + 日期)
pybroker_df       (筛选后数据, 所有模块唯一消费)
       │
       ├── run_backtest()         回测引擎
       ├── render_data_analysis() 数据分析
       └── render_optimization()  参数优化
```

- 所有下游模块只消费 `pybroker_df`，与全量数据隔离
- `run_backtest` 启动前会显式输出当前数据范围确认
- 侧边栏实时显示：品种数、日期范围、数据行数、筛选状态（✅ 已筛选 / ⚠️ 全量数据）

### 第二步：策略配置

在左侧边栏 **⚙️ 策略配置** 区域进行设置：

#### 选择策略

| 策略 | 说明 | 关键参数 |
|------|------|----------|
| 📈 双均线趋势 | EMA 交叉 + ADX 趋势过滤 | 短期/长期均线周期、ADX 阈值、仓位比例 |
| 📉 RSI 反转 | RSI 超买超卖反转交易（仅在震荡市） | RSI 周期、超买/超卖阈值、ADX 阈值、仓位比例 |
| 🔄 跨期套利 | 近远月价差均值回归 | 价差均线周期、入场阈值（标准差）、近/远月合约代码 |
| 📊 期限结构套利 | 跨品种期限结构相对变化套利 | 展期收益率差阈值、配对品种选择、仓位比例 |
| ⚡ 波动率突破 | 基于 ATR 的通道突破策略 | ATR 周期、突破倍数、仓位比例 |

#### 组合管理

- **动态调整策略权重**：根据市场状态（趋势/震荡）自动调整各策略资金分配
- **总资金利用率上限**：控制组合整体仓位

#### 展期设置

| 模式 | 说明 | 参数 |
|------|------|------|
| 流动性触发 | 新合约持仓量超过旧合约时展期 | 持仓量比值阈值 |
| 时间触发 | 合约到期前 N 天展期 | 到期前展期天数 |
| 价差触发 | 新旧合约价差低于阈值时展期 | 价差阈值、最大延迟天数 |

#### 风控设置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 单笔止损 | 2% | 单笔交易最大亏损比例 |
| 单合约最大仓位 | 20% | 单合约占总权益最大比例 |
| 总仓位上限 | 40% | 组合总仓位占总权益最大比例 |
| 展期成本容忍度 | 50 元 | 超过此成本的展期将被拒绝 |
| 日内最大亏损上限 | 3% | 当日亏损超过此比例停止交易 |

#### 交易成本

- **手续费率**：如 0.0001 表示万分之一
- **滑点**：如 0.0002 表示万分之二

### 第三步：运行回测

1. 在左侧导航选择 **🚀 运行回测**
2. 确认当前配置信息（包括数据范围）
3. 点击 **▶️ 开始回测**
4. 回测完成后查看结果，包含 9 个标签页：

| 标签页 | 内容 |
|--------|------|
| 📈 资金曲线 | 净值走势图 |
| 📉 回撤分析 | 回撤曲线 |
| 📊 策略绩效 | 对数收益率、滚动夏普、滚动最大回撤、Q-Q 图、盈亏分布 |
| 🛡️ 风险归因 | 滚动 VaR、相关性热力图、上下捕获比率、风险贡献饼图、持仓集中度、杠杆率、压力测试 |
| 🔄 展期统计 | 展期时间线、展期成本累积曲线（仅合约模式） |
| 📋 交易记录 | 交易明细表，支持 CSV 下载 |
| 💹 交易执行 | VWAP 散点图、滑点分析、持仓时长分布、每日交易次数、品种盈亏箱线图 |
| 📊 月度收益 | 月度收益率柱状图、月度收益热力图 |
| 🌐 市场状态 | 状态背景叠加、状态转移矩阵、各状态绩效、买卖信号标记、滚动相关性动画 |

### 第四步：数据分析

在左侧导航选择 **📉 数据分析**，包含 5 个标签页：

| 标签页 | 内容 |
|--------|------|
| 📊 K线与信号 | K线+成交量图、买卖信号叠加 |
| 📈 持仓量与成交量 | 多合约持仓量与成交量曲线 |
| 🔄 展期与价差 | 展期时间线、价差热力图、跨期价差走势 |
| 🌡️ 数据缺失热力图 | 各合约时间覆盖情况 |
| 🌐 市场状态 | 市场状态背景叠加、状态转移矩阵、滚动相关性动画 |

### 第五步：参数优化

1. 在左侧边栏勾选 **启用参数优化**
2. 在左侧导航选择 **🔍 参数优化**
3. 选择待优化策略和参数搜索空间
4. 选择优化模式：
   - **网格搜索**：遍历所有参数组合
   - **滚动优化**：按时间窗口滚动训练/测试，避免过拟合
5. 设置训练/测试窗口（交易日）
6. 点击 **开始优化**
7. 查看优化结果，包含：
   - 二维参数热力图
   - 3D 参数曲面图
   - 平行坐标图
   - 一维参数扫描线图
   - 参数重要性条形图
   - 等高线图
   - 参数稳定性折线图（滚动优化时）
8. 点击 **📌 应用到回测** 将最佳参数应用到回测配置

## 图表清单

系统共提供 37 种 Plotly 交互式图表：

### 数据概览（4 种）

| 图表 | 方法 | 说明 |
|------|------|------|
| K线+成交量 | `plot_price_with_volume` | 蜡烛图 + 成交量柱状图 |
| 展期时间线 | `plot_rollover_timeline` | 展期事件标记 + 可选价格叠加 |
| 持仓量与成交量 | `plot_open_interest_volume` | 多合约持仓量/成交量曲线 |
| 价差热力图 | `plot_spread_heatmap` | 合约间价差矩阵 |
| 数据缺失热力图 | `plot_missing_data_heatmap` | 数据时间覆盖情况 |

### 策略绩效（7 种）

| 图表 | 方法 | 说明 |
|------|------|------|
| 净值曲线 | `plot_equity_curve` | 策略净值走势 |
| 对数收益率 | `plot_log_returns` | 对数收益率 + 波动率条带 |
| 回撤曲线 | `plot_drawdown` | 策略回撤时间序列 |
| 月度热力图 | `plot_monthly_heatmap` | 月度收益热力矩阵 |
| 滚动夏普 | `plot_rolling_sharpe` | 滚动夏普比率曲线 |
| 滚动最大回撤 | `plot_rolling_max_drawdown` | 滚动最大回撤曲线 |
| 盈亏分布 | `plot_pnl_distribution` | 单笔盈亏直方图 + 核密度 |
| Q-Q 图 | `plot_qq_plot` | 收益率正态分位数图 |

### 风险与归因（8 种）

| 图表 | 方法 | 说明 |
|------|------|------|
| 风险贡献饼图 | `plot_risk_pie` | 品种/策略风险贡献占比 |
| 持仓集中度 | `plot_concentration_curve` | 单合约市值占权益比例 |
| 杠杆率曲线 | `plot_leverage_ratio` | 杠杆率 + 安全阈值 |
| 压力测试瀑布图 | `plot_stress_test` | 历史极端行情区间收益 |
| 相关性热力图 | `plot_correlation_heatmap` | 多品种日收益率相关性 |
| 上下捕获比率 | `plot_up_down_capture` | 牛市/熊市捕获能力 |
| 滚动 VaR | `plot_rolling_var` | 滚动历史模拟 VaR |
| 月度收益率柱状图 | `plot_monthly_returns` | 月度收益柱状图 |

### 交易执行（6 种）

| 图表 | 方法 | 说明 |
|------|------|------|
| VWAP 散点图 | `plot_vwap_scatter` | 成交价 vs VWAP 对比 |
| 持仓时长分布 | `plot_holding_histogram` | 持仓天数/bars 直方图 |
| 每日交易次数 | `plot_daily_trades_count` | 买卖分离的日交易频次 |
| 品种盈亏箱线图 | `plot_pnl_by_symbol` | 按品种分组的盈亏分布 |
| 滑点时间序列 | `plot_slippage_time` | 滑点走势 + 累积模式 |
| 展期成本曲线 | `plot_rollover_cost_curve` | 展期成本累积 + 单次成本 |

### 参数优化（6 种）

| 图表 | 方法 | 说明 |
|------|------|------|
| 参数热力图 | `plot_param_heatmap` | 二维参数网格热力图 |
| 平行坐标图 | `plot_parallel_coordinate` | 多参数组合可视化 |
| 参数扫描 | `plot_param_scan` | 一维参数 vs 指标线图 |
| 参数稳定性 | `plot_param_stability` | 滚动优化参数演变 |
| 3D 曲面图 | `plot_surface_3d` | 三维参数曲面 |
| 参数重要性 | `plot_param_importance` | 参数与指标相关性排序 |

### 市场状态（6 种）

| 图表 | 方法 | 说明 |
|------|------|------|
| 市场状态叠加 | `plot_regime_overlay` | 价格 + 趋势/震荡背景色 |
| 买卖信号标记 | `plot_price_with_signals` | 价格 + 买卖信号标记 |
| 状态转移矩阵 | `plot_regime_transition_matrix` | 状态转移概率热力图 |
| 状态绩效 | `plot_regime_performance` | 各市场状态下年化收益 |
| 滚动相关性动画 | `animate_rolling_correlation` | 可播放的滚动相关性热力图 |
| 月度收益率柱状图 | `plot_monthly_returns` | 月度收益柱状图 |

## 核心模块详解

### 统一回测脚本

使用 `unified_backtest.py` 进行批量回测：

```python
from unified_backtest import (
    BacktestJobConfig,
    BacktestRunner,
    MetricsAnalyzer,
    ResultSaver,
    VisualizationManager,
    STRATEGY_SCENARIOS,
)

# 使用预定义场景
config = STRATEGY_SCENARIOS["dual_ma_comparison"]
results = BacktestRunner.run_all(config)

# 或自定义配置
config = BacktestJobConfig(
    name="custom_backtest",
    data_dir="./data",
    symbols=["SHFE.RB"],
    start_date="2015-01-01",
    end_date="2024-12-31",
    strategies=[
        ("dual_ma_5_20", {"short_ma": 5, "long_ma": 20}),
        ("dual_ma_10_30", {"short_ma": 10, "long_ma": 30}),
    ],
)
```

### 日期筛选辅助函数

系统提供两个健壮的日期筛选辅助函数：

```python
from utils.date_utils import safe_to_timestamp, apply_date_filter

# 安全转换日期
ts = safe_to_timestamp("2020-01-01", label="start_date")
# 失败返回 None，不会抛出异常

# 应用日期筛选
filtered_df = apply_date_filter(
    df=pybroker_df,
    bt_start=date(2015, 1, 1),
    bt_end=date(2020, 12, 31),
    date_col="date"
)
```

### DataLoader — 本地CSV数据加载器

自动检测 CSV 格式（合约模式 vs 品种模式），识别主力合约，构建展期法连续序列。

```python
from core.data_loader import DataLoader

loader = DataLoader("./data")
loader.load_csv_files(file_pattern="*.csv")
loader.identify_dominant_contracts()
loader.build_continuous_series()

pybroker_df = loader.get_pybroker_df()   # PyBroker 兼容格式
rollover_dates = loader.get_rollover_dates()  # 展期日期表
```

### TqSdkDataSource — TqSdk 实时数据源

独立合约 + 展期模式，从 TqSdk 下载真实合约数据，识别每日主力合约，构建连续序列。

```python
from core.data_loader_tqsdk import TqSdkDataSource

loader = TqSdkDataSource(
    phone="手机号", password="密码",
    symbols=["SHFE.RB", "DCE.M", "CZCE.TA"],
    data_length=2000,           # 每个合约 K 线数，2000 ≈ 6年
)
loader.load_csv_files()
loader.identify_dominant_contracts()
loader.build_continuous_series()
df = loader.get_pybroker_df()   # 含 is_dominant, rollover_flag, dominant_symbol 等列

# 查看数据统计
print(loader.get_data_summary())
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `phone` | str | — | TqSdk 手机号 |
| `password` | str | — | TqSdk 密码 |
| `symbols` | list | 核心品种 | 品种列表，如 `["SHFE.RB", "DCE.M"]` |
| `data_length` | int | 2000 | 每个合约下载的 K 线数量（日线），2000 ≈ 6年 |

### MarketRegimeDetector — 市场环境分类引擎（v3）

基于多维度量化指标的市场环境分类系统，**无前视偏差**，支持8种典型市场环境类型，输出连续环境分数和动态权重。

```python
from core.market_regime import MarketRegimeDetector, RegimeConfig

# 检测市场环境（兼容旧接口）
detector = MarketRegimeDetector()
result_df = detector.detect(df)

# fit/transform模式（推荐）
detector.fit(df_train)  # 训练参数
result_df = detector.transform(df_test)  # 样本外预测

# 样本外验证
validation = detector.validate(df)
print(f"KL散度: {validation.distribution_stability:.4f}")
```

#### 核心特性

| 特性 | 说明 |
|------|------|
| **8种市场环境** | 趋势上涨/下跌、区间震荡、高/低波动、突破、牛市/熊市衰竭 |
| **9个量化指标** | ADX、趋势方向、波动率水平、波动率压缩、动量、成交量强度、持仓量变化、布林带位置、背离强度 |
| **滚动IC动态权重** | 基于信息系数自动调整指标权重，支持定期重算 |
| **滚动百分位阈值** | 动态阈值避免固定值的过拟合问题 |
| **连续环境分数** | 输出0~1连续分数，比离散标签更灵活 |
| **确认窗口防抖动** | 状态机实现，新状态连续出现N次才切换 |
| **fit/transform分离** | 样本外验证时不使用未来数据 |
| **无shift(-k)** | 源码验证：无任何前视偏差代码 |

#### 核心改进（v3）

| 改进项 | 说明 |
|-------|------|
| **消除前视偏差** | 删除`future_return`，改为滚动IC权重（仅用历史数据） |
| **fit/transform模式** | 样本内训练参数，样本外固定使用，无未来数据泄露 |
| **背离检测修正** | 价格创新高/新低且RSI未同步创新高/新低（加`close != close.shift(1)`） |
| **确认窗口状态机** | 不再后视检查未来天数，维护当前状态+计数器 |
| **波动率压缩简化** | 直接使用`atr_short / atr_long` |
| **背离纳入IC** | 新增`divergence_strength`指标，计算与未来收益的IC |
| **阈值裁剪** | `vol_high/vol_low`、`bb_upper/bb_lower`裁剪到[0,1] |
| **缺失列处理** | 检查必需列，volume/open_interest缺失时警告并设默认值 |
| **类型注解** | 所有关键方法添加类型注解和文档字符串 |

### EnvironmentAdapter — 市场状态引擎（旧版）

多指标融合计算市场环境状态，输出连续趋势分数（0~1）和动态策略权重。

```python
from core.environment import EnvironmentAdapter

env = EnvironmentAdapter()
env_df = env.compute_for_pybroker(pybroker_df)
# 输出列: env_atr, env_adx, env_market_regime, env_trend_score,
#         env_weight_trend, env_weight_reversal, env_weight_spread, ...
```

#### 核心改进

| 改进项 | 说明 |
|-------|------|
| **滚动归一化无前瞻偏差** | `rolling_min/max` 使用 `shift(1)`，避免未来数据泄露 |
| **重写衰竭检测** | 价格创新高/新低且RSI未同步创新高/新低 |
| **EWMA min_periods** | EMA计算等待完整周期数据，避免早期异常值 |
| **权重总和防御** | 使用 `clip(lower=1e-8)` 防止除零错误 |
| **ATR早期缺失修复** | 首日TR设为`high-low`，滚动窗口等待完整数据 |
| **输入验证** | 检查必需列，空DataFrame安全返回 |
| **列冲突处理** | `compute_for_pybroker` 自动去重列名 |
| **流动性指标裁剪** | 限制上限为10.0，防止极端值影响归一化 |
| **文档提示** | 建议缓存结果提升性能 |

### Strategy — 策略基类

所有策略继承 `BaseStrategy`，实现 `execute(ctx)` 方法：

```python
from core.strategies import create_strategy, STRATEGY_REGISTRY

strat = create_strategy("dual_ma", short_ma=5, long_ma=20, adx_threshold=25, position_size=0.3)
indicators = strat.register_indicators()  # 注册 PyBroker 指标
strat.execute(ctx)  # 每个 bar 调用
```

### RolloverManager — 展期管理器

```python
from core.rollover import RolloverManager, RolloverMode

rollover = RolloverManager(mode=RolloverMode.LIQUIDITY, liquidity_ratio=1.5)
wrapped_fn = rollover.create_rollover_exec_fn(strategy.execute)
strategy.add_execution(fn=wrapped_fn, symbols=symbols, indicators=indicators)
```

### RiskManager — 风控管理器

```python
from core.risk_manager import RiskManager

risk = RiskManager(stop_loss_pct=0.02, max_position_pct=0.2, daily_loss_limit=0.03)
safe_fn = risk.wrap_with_risk_control(strategy.execute)
```

### PortfolioManager — 组合管理器

```python
from core.portfolio import PortfolioManager

pm = PortfolioManager(total_allocation=0.8)
pm.add_strategy("dual_ma", dual_ma_instance)
pm.add_strategy("rsi", rsi_instance)
pm.register_all_to_pybroker(pybroker_strategy=strategy, symbols=symbols)
```

### ParameterOptimizer — 参数优化器

```python
from core.optimizer import ParameterOptimizer

opt = ParameterOptimizer(param_grid={"short_ma": [5, 10], "long_ma": [20, 30]}, metric="sharpe")
results = opt.grid_search(strategy_class=DualMAStrategy, data=df, symbols=symbols)
best = opt.get_best_params()
```

## 自定义策略

继承 `BaseStrategy` 并实现 `execute` 方法：

```python
from core.strategies import BaseStrategy
from pybroker import ExecContext
import pybroker

class MyStrategy(BaseStrategy):
    def __init__(self, period=14, threshold=0.5):
        self.period = period
        self.threshold = threshold

    def register_indicators(self):
        return [pybroker.indicator("my_ind", lambda df: df["close"].rolling(self.period).mean())]

    def execute(self, ctx: ExecContext):
        if self._check_rollover(ctx):
            return
        ma = ctx.indicator("my_ind")
        if ma[-1] > self.threshold and not ctx.long_pos():
            ctx.buy_shares = ctx.calc_target_shares(0.2)
        elif ma[-1] < self.threshold and ctx.long_pos():
            ctx.sell_all_shares()
```

注册到系统：

```python
from core.strategies import STRATEGY_REGISTRY
STRATEGY_REGISTRY["my_strategy"] = MyStrategy
```

## 数据格式说明

### 品种指数 CSV 示例

```csv
datetime,open,high,low,close,volume,position
2024-01-02,3850,3880,3840,3870,1200000,1500000
2024-01-03,3870,3900,3860,3890,1100000,1480000
```

### 合约模式 CSV 示例

```csv
date,symbol,open,high,low,close,volume,open_interest
2024-01-02,RB2401,3850,3880,3840,3870,500000,800000
2024-01-02,RB2405,3780,3810,3770,3800,700000,1200000
2024-01-03,RB2401,3870,3890,3860,3880,300000,600000
2024-01-03,RB2405,3800,3830,3790,3820,800000,1300000
```

> 合约模式下，系统自动根据持仓量识别主力合约（持仓量最大者），并构建展期法连续序列。

## 常见问题

### Q: 数据加载失败？

- 检查 CSV 文件编码（推荐 UTF-8）
- 确认列名与上述格式一致
- 查看终端错误日志

### Q: 回测结果为空？

- 确认数据日期范围与策略参数匹配
- 检查是否选择了合约（数据切片中）
- 查看侧边栏 **📌 当前回测范围** 是否显示预期的品种数和日期
- 确认策略参数合理（如均线周期不能大于数据长度）

### Q: 数据筛选不生效？

- 加载数据后在数据切片中选择品种和日期
- 筛选仅在 `render_data_import` 页面中更新
- 查看侧边栏 `📌 当前回测范围` 确认筛选是否应用
- 查看 Info 栏提示确认回测前使用的数据范围
- 数据架构保证全量数据与筛选数据隔离，下游模块只会消费 `pybroker_df`

### Q: 展期功能不可用？

- 展期功能仅在合约模式下可用
- 品种指数模式（如 `SHFE.RB.csv`）无需展期

### Q: 参数优化耗时过长？

- 减少参数搜索空间（少选几个候选值）
- 使用滚动优化替代网格搜索
- 缩小训练窗口
- 缩小回测数据范围（使用数据切片）

### Q: 如何查看系统日志？

系统使用 Python `logging` 模块，日志输出到终端：
- **INFO**：日期筛选操作、数据行数变化、回测范围确认
- **WARNING**：无效日期格式、日期转换结果为 NaT
- **ERROR**：开始日期>结束日期（自动交换）、日期转换失败

### Q: 日期筛选支持哪些输入格式？

`_safe_to_timestamp` 支持以下格式：
- `datetime.date` 或 `datetime.datetime`
- `pd.Timestamp`
- ISO 格式字符串（`"2020-01-01"`、`"2020-01-01 14:30"`）
- `None`（表示不限制）

### Q: 开始日期大于结束日期怎么办？

系统会自动检测并交换两者，记录 ERROR 日志后继续执行。

### Q: 筛选状态显示什么？

侧边栏 `📌 当前回测范围` 显示：
- **品种数**：当前筛选后的品种数量
- **日期范围**：最早和最晚日期
- **数据行数**：总数据行数
- **筛选状态**：`✅ 已筛选` 或 `⚠️ 全量数据`

### Q: 调整 ADX 阈值后回测结果没有变化？

检查你修改的是哪个位置：
- **修改了 `EnvironmentAdapter` 中的 `trend_threshold`** → 这个只影响 `market_regime`（趋势市/震荡市标记），不影响策略交易决策
- **正确调整方式** → 在界面左侧栏的策略配置中调整：
  - 「双均线策略参数」下方 → `ADX趋势阈值`（仅在 ADX > 阈值时开仓）
  - 「RSI参数」下方 → `ADX震荡市阈值`（仅在 ADX < 阈值时开仓）

### Q: 两个 ADX 阈值有什么区别？

| 位置 | 参数名称 | 用途 |
|------|----------|------|
| `DualMAStrategy` | `adx_threshold` | ADX > 阈值 → 允许开仓（只在趋势市交易） |
| `RSIStrategy` | `adx_threshold` | ADX < 阈值 → 允许开仓（只在震荡市交易） |
| `EnvironmentAdapter` | `trend_threshold` | 仅用于 `market_regime` 标记（趋势市/震荡市） |

## 技术栈

| 组件 | 技术 |
|------|------|
| 回测引擎 | [PyBroker](https://github.com/edtechre/pybroker) |
| 前端框架 | [Streamlit](https://streamlit.io/) |
| 可视化 | [Plotly](https://plotly.com/python/) |
| 数据处理 | Pandas / NumPy |
| 机器学习 | scikit-learn |

## 更新日志

### 2026-05-24

#### MarketRegimeDetector v3 重大升级

**消除前视偏差**
- 删除 `_detect_single` 中全局 `future_return`（`close.pct_change(5).shift(-5)`）
- 实现 `compute_ic_weights_rolling`：每个时间点的权重基于 `[t-window, t-1]` 历史数据
- `validate` 方法改用 `fit/transform` 模式：样本内训练参数，样本外固定使用，无未来数据泄露
- 源码验证：无任何 `shift(-k)` 前视偏差代码

**修复逻辑错误**
- 背离检测：添加 `close != close.shift(1)` 条件，确保价格创新高/新低
- 确认窗口：重写为状态机实现，维护当前状态+连续计数器，不再后视检查未来天数
- 波动率压缩：简化为直接使用 `atr_short / atr_long`

**高优先级改进**
- 纳入 `divergence_strength` 到 IC 动态权重（顶背离-1，底背离+1）
- 动态阈值裁剪：`vol_high/vol_low`、`bb_upper/bb_lower` 裁剪到 [0,1]
- 缺失列处理：`compute_indicators` 开头检查必需列 `{close, high, low}`，volume/open_interest 缺失时警告并设默认值
- `validate` 收益对齐：使用 `pd.merge` 按日期对齐

**推荐改进**
- `_normalize` 新增 `lag` 参数（默认 False）
- `detect` 新增 `verbose` 参数
- 所有关键方法添加类型注解和文档字符串
- 新增 `fit_transform` 方法（回测场景用）
- 新增 `fit`、`transform` 方法

### 2026-05-23

#### 项目结构重构
- 将 `app.py` 拆分为 `pages/`、`components/`、`utils/` 目录，提高代码可维护性
- 新增 `config.py` 统一管理全局常量
- 将所有策略从 `core/strategies.py` 拆分为独立模块 `core/strategies/`
- 新增 `components/sidebar.py` 和 `components/results.py` 组件

#### 新增策略
- 期限结构套利策略：跨品种期限结构相对变化套利
- 波动率突破策略：基于 ATR 的通道突破策略

#### 统一回测脚本
- 新增 `unified_backtest.py` 提供命令行批量回测
- 支持多策略并行对比、样本内/外分割、自动生成 HTML 报告
- 预定义场景：`dual_ma_comparison`（双均线变体对比）、`new_strategies`（新策略回测）

#### 移除的文件
- `run_comparison.py` → 使用 `unified_backtest.py --scenario dual_ma_comparison` 替代
- `run_new_strategies.py` → 使用 `unified_backtest.py --scenario new_strategies` 替代
- `backtest_comparison.py` → 功能已整合到 `unified_backtest.py`
- `backtest_new_strategies.py` → 功能已整合到 `unified_backtest.py`

## License

MIT
