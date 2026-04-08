import os
import pandas as pd
from pathlib import Path
import numpy as np

data_dir = Path("/Users/bytedance/Desktop/my_notes/量化交易/hw2/data/MktEqudAfGet")
files = list(data_dir.glob("*.csv"))
files.sort()
print(f"Found {len(files)} files.")

dfs = []
for i, f in enumerate(files):
    if i % 500 == 0:
        print(f"Processing {i}/{len(files)}")
    try:
        df = pd.read_csv(f, usecols=["tradeDate", "ticker", "secShortName", "openPrice", "closePrice", "preClosePrice", "marketValue", "isOpen", "highestPrice", "lowestPrice", "turnoverVol", "vwap"])
        dfs.append(df)
    except Exception as e:
        print(f"Error reading {f}: {e}")

all_df = pd.concat(dfs, ignore_index=True)
all_df['tradeDate'] = pd.to_datetime(all_df['tradeDate'])
all_df = all_df.sort_values(["ticker", "tradeDate"]).reset_index(drop=True)
all_df.to_parquet("all_price.parquet")
print("Saved to all_price.parquet")
