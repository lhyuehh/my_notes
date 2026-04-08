import pandas as pd
import zipfile
import os

base_dir = "/Users/bytedance/Desktop/my_notes/量化交易/hw2/股票基本面数据"

def peek_zip_csv(zip_name):
    zip_path = os.path.join(base_dir, zip_name)
    print(f"\n--- Peeking into {zip_name} ---")
    with zipfile.ZipFile(zip_path, 'r') as z:
        # get the first csv file
        csv_files = [f for f in z.namelist() if f.endswith('.csv')]
        if csv_files:
            csv_file = csv_files[0]
            print(f"Reading {csv_file}...")
            with z.open(csv_file) as f:
                df = pd.read_csv(f, nrows=5)
                print("Columns:", df.columns.tolist())
                print(df.head(2).to_string())

peek_zip_csv("FdmtISGet利润表.zip")
peek_zip_csv("FdmtBSGet资产负债表.zip")

