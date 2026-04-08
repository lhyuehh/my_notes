import pandas as pd
import zipfile
import io

zips = [
    "FdmtBSGet资产负债表.zip",
    "FdmtCFGet现金流量表.zip",
    "FdmtISGet利润表.zip",
    "ST股票.zip",
    "成分股权重.zip",
    "股票名称-上市状态.zip",
    "交易日历.zip",
    "行情-后复权.zip",
    "指数行情.zip"
]

base_dir = "/Users/bytedance/Desktop/my_notes/量化交易/hw2/股票基本面数据"

for z in zips:
    try:
        with zipfile.ZipFile(f"{base_dir}/{z}", 'r') as zf:
            files = [f for f in zf.namelist() if f.endswith('.parquet') or f.endswith('.csv')]
            if not files: continue
            sample_file = files[-1]
            with zf.open(sample_file) as f:
                if sample_file.endswith('.parquet'):
                    df = pd.read_parquet(f)
                else:
                    df = pd.read_csv(f)
                print(f"--- {z} ({sample_file}) ---")
                print("Columns:", list(df.columns))
                print("Sample:", df.iloc[0].to_dict() if len(df) > 0 else "Empty")
                print()
    except Exception as e:
        print(f"Error reading {z}: {e}")
