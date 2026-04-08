import pandas as pd
import zipfile
import os
import glob

base_dir = "/Users/bytedance/Desktop/my_notes/量化交易/hw2/股票基本面数据"
temp_dir = "/Users/bytedance/Desktop/my_notes/量化交易/hw2/temp_fdmt"
os.makedirs(temp_dir, exist_ok=True)

print("1. Extracting Zip files...")
with zipfile.ZipFile(os.path.join(base_dir, "FdmtISGet利润表.zip"), 'r') as z:
    z.extractall(temp_dir)
with zipfile.ZipFile(os.path.join(base_dir, "FdmtBSGet资产负债表.zip"), 'r') as z:
    z.extractall(temp_dir)

print("2. Reading Income Statement (IS)...")
is_files = glob.glob(os.path.join(temp_dir, "FdmtISGet", "*.parquet"))
is_dfs = []
for f in is_files:
    df = pd.read_parquet(f)
    if not df.empty:
        # Keep consolidated statements (mergedFlag == '1')
        if 'mergedFlag' in df.columns:
            df = df[df['mergedFlag'] == '1']
        is_dfs.append(df[['ticker', 'publishDate', 'endDate', 'NIncomeAttrP']])
is_df = pd.concat(is_dfs, ignore_index=True)
is_df['publishDate'] = pd.to_datetime(is_df['publishDate'])
is_df['ticker'] = is_df['ticker'].astype(str).str.zfill(6)
is_df = is_df.sort_values(['ticker', 'publishDate']).drop_duplicates(subset=['ticker', 'publishDate'], keep='last')

print("3. Reading Balance Sheet (BS)...")
bs_files = glob.glob(os.path.join(temp_dir, "FdmtBSGet", "*.parquet"))
bs_dfs = []
for f in bs_files:
    df = pd.read_parquet(f)
    if not df.empty:
        if 'mergedFlag' in df.columns:
            df = df[df['mergedFlag'] == '1']
        bs_dfs.append(df[['ticker', 'publishDate', 'endDate', 'TEquityAttrP']])
bs_df = pd.concat(bs_dfs, ignore_index=True)
bs_df['publishDate'] = pd.to_datetime(bs_df['publishDate'])
bs_df['ticker'] = bs_df['ticker'].astype(str).str.zfill(6)
bs_df = bs_df.sort_values(['ticker', 'publishDate']).drop_duplicates(subset=['ticker', 'publishDate'], keep='last')

print("4. Merging IS and BS...")
# merge_asof on publishDate
is_df = is_df.sort_values('publishDate')
bs_df = bs_df.sort_values('publishDate')

fdmt_df = pd.merge_asof(
    is_df, 
    bs_df[['ticker', 'publishDate', 'TEquityAttrP']], 
    on='publishDate', 
    by='ticker', 
    direction='backward'
)
fdmt_df.to_parquet("fundamental_data.parquet")
print("Saved to fundamental_data.parquet")

