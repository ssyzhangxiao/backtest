#!/usr/bin/env python3
"""Inspect SHFE.AL cache detail."""
import pickle
import pandas as pd

for sym in ["SHFE_AL", "CZCE_CF"]:
    p = f"data_cache/{sym}_4000.pkl"
    with open(p, "rb") as f:
        df = pickle.load(f)
    print(f"\n=== {sym} ===")
    print(f"Shape: {df.shape}")
    print(f"Cols: {list(df.columns)}")
    print(f"Date min/max: {df['date'].min()} ~ {df['date'].max()}")
    print(f"Unique contracts: {df['symbol'].nunique()}")
    print(f"Contract list: {sorted(df['symbol'].unique())[:25]}")
    print(f"Per-contract row count:")
    print(df.groupby('symbol').agg(
        n=('date', 'count'),
        date_min=('date', 'min'),
        date_max=('date', 'max')
    ).sort_values('date_max', ascending=False).head(25))
