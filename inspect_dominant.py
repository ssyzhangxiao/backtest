#!/usr/bin/env python3
"""Inspect is_dominant distribution in loaded data."""
from runner import Pipeline
import pandas as pd

pipe = Pipeline("config.yaml").load_data()
ds = pipe._data
df = ds.to_pybroker_df()
print("Columns:", list(df.columns))
print("Total rows:", len(df))
print("Date range:", df["date"].min(), "~", df["date"].max())
print()
print("Symbol | T | F | total | products")
for sym in sorted(df.symbol.unique()):
    sub = df[df.symbol == sym]
    n_true = sub.is_dominant.sum() if "is_dominant" in sub.columns else 0
    n_false = len(sub) - n_true
    products = sub.get("product", pd.Series()).unique() if "product" in sub.columns else []
    print(f"{sym:<14}  T={n_true:<5} F={n_false:<5} total={len(sub):<5} products={list(products)[:3]}")

print()
print("=== 2023-2024 is_dominant 分布 ===")
for sym in sorted(df.symbol.unique()):
    sub = df[(df.symbol == sym) & (df.date >= "2023-01-01") & (df.date <= "2024-12-31")]
    n_true = sub.is_dominant.sum() if "is_dominant" in sub.columns else 0
    print(f"{sym:<14}  2023-2024 T={n_true:<5} total={len(sub):<5}")

# Per-symbol per-year
print()
print("=== Per-symbol per-year dominant=True count ===")
for sym in sorted(df.symbol.unique()):
    parts = []
    for yr in ["2022", "2023", "2024"]:
        sub = df[(df.symbol == sym) & (df.date >= f"{yr}-01-01") & (df.date <= f"{yr}-12-31")]
        n_true = sub.is_dominant.sum() if "is_dominant" in sub.columns else 0
        parts.append(f"{yr}={n_true}")
    print(f"{sym:<14}  {' '.join(parts)}")
