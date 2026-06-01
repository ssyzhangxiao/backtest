import sys, traceback
sys.path.insert(0, '.')
from core.config import BacktestConfig
from core.engine.backtest_runner import PyBrokerBacktestRunner
from core.engine.pybroker_data_source import create_hybrid_data_source

symbols = ['SHFE.RB', 'SHFE.AU', 'SHFE.CU', 'DCE.I']
ds = create_hybrid_data_source(phone=None, password=None, symbols=symbols, data_dir='data', data_length=4000)
bt_cfg = BacktestConfig.from_yaml()
runner = PyBrokerBacktestRunner(ds, bt_cfg, target_symbols=symbols)
runner.register_strategies(['ts_momentum', 'roll_yield', 'alpha019', 'alpha032'])
try:
    result = runner.run(start_date='2020-01-01', end_date='2023-01-01')
    if result:
        print(f"\nSharpe: {result.metrics.get('sharpe', 'N/A'):.4f}")
        print(f"年化收益: {result.metrics.get('annual_return', 'N/A')}")
        print(f"最大回撤: {result.metrics.get('max_drawdown', 'N/A')}")
except Exception as e:
    traceback.print_exc()