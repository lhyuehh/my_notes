
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_single_symbol(symbol):
    """加载单个品种的数据"""
    data_path = f'/Users/bytedance/Desktop/my_notes/量化交易/main&smain_market_data_01min/{symbol}_main.feather'
    df = pd.read_feather(data_path)
    df['datetime'] = pd.to_datetime(df['tradeDate'] + ' ' + df['barTime'])
    df = df.set_index('datetime')
    return df

def plot_price_and_volume(df, symbol, save_path):
    """绘制价格和成交量"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    
    plot_df = df.iloc[-10000:].copy()
    
    ax1.plot(plot_df.index, plot_df['hfq_closePrice'], label='后复权收盘价', linewidth=0.8)
    ax1.plot(plot_df.index, plot_df['hfq_twap'], label='后复权TWAP', linewidth=0.8, alpha=0.7)
    ax1.set_title(f'{symbol} - 价格走势', fontsize=14, fontweight='bold')
    ax1.set_ylabel('价格', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    ax2.bar(plot_df.index, plot_df['turnoverVol'], label='成交量', alpha=0.6, width=0.0005)
    ax2.set_title(f'{symbol} - 成交量', fontsize=14, fontweight='bold')
    ax2.set_xlabel('时间', fontsize=12)
    ax2.set_ylabel('成交量', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    date_formatter = DateFormatter('%Y-%m-%d')
    ax2.xaxis.set_major_formatter(date_formatter)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"价格和成交量图已保存到: {save_path}")

def plot_ohlc(df, symbol, save_path):
    """绘制OHLC图（简化版）"""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    plot_df = df.iloc[-500:].copy()
    
    for i in range(len(plot_df)):
        idx = plot_df.index[i]
        row = plot_df.iloc[i]
        
        ax.plot([idx, idx], [row['hfq_lowPrice'], row['hfq_highPrice']], 
                color='black', linewidth=0.8)
        
        if row['hfq_closePrice'] >= row['hfq_openPrice']:
            color = 'red'
        else:
            color = 'green'
        ax.plot([idx, idx], [row['hfq_openPrice'], row['hfq_closePrice']], 
                color=color, linewidth=2)
    
    ax.set_title(f'{symbol} - K线图（最近500根bar）', fontsize=14, fontweight='bold')
    ax.set_ylabel('价格', fontsize=12)
    ax.set_xlabel('时间', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    date_formatter = DateFormatter('%Y-%m-%d %H:%M')
    ax.xaxis.set_major_formatter(date_formatter)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"K线图已保存到: {save_path}")

def plot_open_interest(df, symbol, save_path):
    """绘制持仓量"""
    fig, ax = plt.subplots(figsize=(16, 6))
    
    plot_df = df.iloc[-10000:].copy()
    
    ax.plot(plot_df.index, plot_df['openInterest'], label='持仓量', 
            linewidth=0.8, color='purple')
    ax.set_title(f'{symbol} - 持仓量走势', fontsize=14, fontweight='bold')
    ax.set_ylabel('持仓量', fontsize=12)
    ax.set_xlabel('时间', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    date_formatter = DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(date_formatter)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"持仓量图已保存到: {save_path}")

def plot_multiple_symbols(symbols, save_path):
    """对比多个品种的价格走势"""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, symbol in enumerate(symbols):
        try:
            df = load_single_symbol(symbol)
            plot_df = df.iloc[-20000:].copy()
            normalized_price = plot_df['hfq_closePrice'] / plot_df['hfq_closePrice'].iloc[0]
            ax.plot(plot_df.index, normalized_price, label=symbol, 
                    linewidth=0.8, color=colors[i % len(colors)])
        except Exception as e:
            print(f"加载 {symbol} 失败: {e}")
    
    ax.set_title('多品种价格走势对比（标准化）', fontsize=14, fontweight='bold')
    ax.set_ylabel('标准化价格', fontsize=12)
    ax.set_xlabel('时间', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='black', linestyle='--', linewidth=0.8)
    
    date_formatter = DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(date_formatter)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"多品种对比图已保存到: {save_path}")

def print_data_summary(df, symbol):
    """打印数据摘要"""
    print(f"\n{'='*60}")
    print(f"{symbol} 数据摘要")
    print(f"{'='*60}")
    print(f"数据时间范围: {df.index[0]} 至 {df.index[-1]}")
    print(f"数据条数: {len(df)}")
    print(f"\n数据列:")
    for col in df.columns:
        print(f"  - {col}")
    
    print(f"\n价格统计（后复权）:")
    print(f"  开盘价:  {df['hfq_openPrice'].iloc[0]:.2f} → {df['hfq_openPrice'].iloc[-1]:.2f}")
    print(f"  收盘价:  {df['hfq_closePrice'].iloc[0]:.2f} → {df['hfq_closePrice'].iloc[-1]:.2f}")
    print(f"  最高价:  {df['hfq_highPrice'].max():.2f}")
    print(f"  最低价:  {df['hfq_lowPrice'].min():.2f}")
    
    print(f"\n成交量统计:")
    print(f"  平均成交量: {df['turnoverVol'].mean():.0f}")
    print(f"  最大成交量: {df['turnoverVol'].max():.0f}")
    
    print(f"\n持仓量统计:")
    print(f"  平均持仓量: {df['openInterest'].mean():.0f}")
    print(f"  最大持仓量: {df['openInterest'].max():.0f}")

def main():
    print("="*60)
    print("数据探索与可视化")
    print("="*60)
    
    output_dir = '/Users/bytedance/Desktop/my_notes/量化交易/output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    demo_symbols = ['AU', 'RB', 'CU', 'SC', 'AG']
    
    for symbol in demo_symbols:
        try:
            print(f"\n处理品种: {symbol}")
            df = load_single_symbol(symbol)
            
            print_data_summary(df, symbol)
            
            plot_price_and_volume(df, symbol, 
                os.path.join(output_dir, f'{symbol}_price_volume.png'))
            plot_ohlc(df, symbol, 
                os.path.join(output_dir, f'{symbol}_ohlc.png'))
            plot_open_interest(df, symbol, 
                os.path.join(output_dir, f'{symbol}_open_interest.png'))
            
        except Exception as e:
            print(f"处理 {symbol} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n生成多品种对比图...")
    plot_multiple_symbols(demo_symbols, 
        os.path.join(output_dir, 'multiple_symbols_comparison.png'))
    
    print(f"\n{'='*60}")
    print("数据探索完成！所有图表已保存到 output/ 目录")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
