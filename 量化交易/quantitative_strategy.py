
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（根据系统调整）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 策略品种列表
SYMBOLS = ['SP', 'P', 'BU', 'CU', 'V', 'FG', 'CF', 'SC', 'AU', 'AG', 
           'MA', 'PG', 'RM', 'RU', 'TA', 'EB', 'SR', 'FU', 'I', 'SN', 
           'OI', 'ZN', 'Y', 'SA', 'M', 'RB', 'EG', 'PP', 'AL', 'NI', 'HC']

class RSIReversionStrategy:
    """
    RSI均值回归策略：
    在1分钟这种高频数据上，价格往往呈现均值回归特性而不是趋势特性。
    - 当RSI极度超卖（<20）时开多仓，认为短期跌过头了
    - 当RSI极度超买（>80）时开空仓，认为短期涨过头了
    - 当RSI回归中性（50附近）时平仓
    """
    def __init__(self, rsi_window=14, oversold=20, overbought=80):
        self.rsi_window = rsi_window
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, df):
        df = df.copy()
        
        # 1. 计算RSI
        delta = df['hfq_closePrice'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_window).mean()
        
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 2. 生成信号
        df['signal'] = 0
        
        # 超卖开多
        df.loc[(df['rsi'] < self.oversold) & (df['rsi'].shift(1) >= self.oversold), 'signal'] = 1
        
        # 超买开空
        df.loc[(df['rsi'] > self.overbought) & (df['rsi'].shift(1) <= self.overbought), 'signal'] = -1
        
        # 回归中性平仓
        # 多头平仓：RSI回到50以上
        df.loc[df['rsi'] > 50, 'close_long'] = True
        # 空头平仓：RSI回到50以下
        df.loc[df['rsi'] < 50, 'close_short'] = True
        
        # 为了画图兼容性，添加布林带字段（画图函数还在用）
        df['upper_band'] = df['hfq_closePrice'].rolling(20).mean() + 2*df['hfq_closePrice'].rolling(20).std()
        df['lower_band'] = df['hfq_closePrice'].rolling(20).mean() - 2*df['hfq_closePrice'].rolling(20).std()
        df['middle_band'] = df['hfq_closePrice'].rolling(20).mean()
        
        return df

class BacktestEngine:
    def __init__(self, strategy, initial_capital=1000000):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.results = {}
    
    def backtest_symbol(self, symbol, df):
        df = df.sort_index().copy()
        df = self.strategy.generate_signals(df)
        
        position = 0
        entry_price = 0
        entry_time = None
        trades = []
        daily_equity = {}
        
        current_capital = self.initial_capital
        
        for i in range(len(df)):
            current_bar = df.iloc[i]
            current_date = current_bar.name.date()
            
            if current_date not in daily_equity:
                if position != 0:
                    pnl = (current_bar['hfq_closePrice'] - entry_price) * position
                    daily_equity[current_date] = current_capital + pnl
                else:
                    daily_equity[current_date] = current_capital
            
            signal = current_bar['signal']
            
            # 检查平仓条件
            should_close = False
            if position == 1 and current_bar.get('close_long', False):
                should_close = True
            elif position == -1 and current_bar.get('close_short', False):
                should_close = True
            
            # 平仓
            if should_close and position != 0:
                if i + 1 < len(df):
                    next_bar = df.iloc[i + 1]
                    exit_price = next_bar['hfq_twap']
                    exit_time = next_bar.name
                    
                    pnl = (exit_price - entry_price) * position
                    current_capital += pnl
                    
                    trades.append({
                        'symbol': symbol,
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'direction': 'long' if position == 1 else 'short',
                        'pnl': pnl,
                        'holding_period': (exit_time - entry_time).total_seconds() / 60
                    })
                    
                    position = 0
            
            # 开仓
            elif signal != 0 and position == 0:
                if i + 1 < len(df):
                    next_bar = df.iloc[i + 1]
                    entry_price = next_bar['hfq_twap']
                    entry_time = next_bar.name
                    position = signal
            
            # 反向开仓 (如果当前持仓，且出现反向强烈信号)
            elif signal != 0 and position != 0 and signal != position:
                if i + 1 < len(df):
                    next_bar = df.iloc[i + 1]
                    exit_price = next_bar['hfq_twap']
                    exit_time = next_bar.name
                    
                    pnl = (exit_price - entry_price) * position
                    current_capital += pnl
                    
                    trades.append({
                        'symbol': symbol,
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'direction': 'long' if position == 1 else 'short',
                        'pnl': pnl,
                        'holding_period': (exit_time - entry_time).total_seconds() / 60
                    })
                    
                    entry_price = exit_price
                    entry_time = exit_time
                    position = signal
        
        if position != 0:
            last_bar = df.iloc[-1]
            exit_price = last_bar['hfq_closePrice']
            exit_time = last_bar.name
            pnl = (exit_price - entry_price) * position
            current_capital += pnl
            trades.append({
                'symbol': symbol,
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'direction': 'long' if position == 1 else 'short',
                'pnl': pnl,
                'holding_period': (exit_time - entry_time).total_seconds() / 60
            })
        
        if daily_equity:
            equity_df = pd.DataFrame.from_dict(daily_equity, orient='index', columns=['equity'])
            equity_df.index = pd.to_datetime(equity_df.index)
            equity_df = equity_df.sort_index()
            equity_df['net_value'] = equity_df['equity'] / self.initial_capital
        else:
            equity_df = pd.DataFrame()
        
        self.results[symbol] = {
            'trades': trades,
            'equity': equity_df,
            'final_capital': current_capital,
            'data_with_signals': df
        }
        
        return self.results[symbol]
    
    def calculate_metrics(self, combined_equity):
        equity = combined_equity['net_value'].values
        returns = np.diff(equity) / equity[:-1]
        
        if len(returns) == 0:
            return {}
        
        total_return = equity[-1] - 1
        n_days = len(equity)
        annualized_return = (1 + total_return) ** (252 / n_days) - 1
        
        sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        cum_returns = np.maximum.accumulate(equity)
        drawdown = (equity - cum_returns) / cum_returns
        max_drawdown = np.min(drawdown)
        
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        all_trades = []
        for symbol in self.results:
            all_trades.extend(self.results[symbol]['trades'])
        
        if len(all_trades) == 0:
            return {
                'sharpe_ratio': sharpe_ratio,
                'annualized_return': annualized_return,
                'max_drawdown': max_drawdown,
                'calmar_ratio': calmar_ratio,
                'win_rate': 0,
                'win_loss_ratio': 0,
                'num_transactions': 0,
                'avg_holding_time': 0,
                'avg_profit_per_trade': 0,
                'avg_loss_per_trade': 0
            }
        
        winning_trades = [t for t in all_trades if t['pnl'] > 0]
        losing_trades = [t for t in all_trades if t['pnl'] <= 0]
        
        win_rate = len(winning_trades) / len(all_trades) if len(all_trades) > 0 else 0
        
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        avg_holding_time = np.mean([t['holding_period'] for t in all_trades])
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'win_loss_ratio': win_loss_ratio,
            'num_transactions': len(all_trades),
            'avg_holding_time': avg_holding_time,
            'avg_profit_per_trade': avg_win,
            'avg_loss_per_trade': avg_loss
        }

def load_data(data_dir):
    data = {}
    
    # 检查多个可能的数据目录
    possible_dirs = [
        data_dir,
        os.path.join(data_dir, 'main&smain_market_data_01min'),
        '/Users/bytedance/Desktop/my_notes/量化交易'
    ]
    
    data_files = []
    for check_dir in possible_dirs:
        if os.path.exists(check_dir):
            for root, dirs, files in os.walk(check_dir):
                for file in files:
                    if file.endswith('.feather') or file.endswith('.csv'):
                        file_path = os.path.join(root, file)
                        if os.path.getsize(file_path) > 0:  # 只加载非空文件
                            data_files.append(file_path)
    
    if len(data_files) == 0:
        print("未找到有效数据文件")
        return data
    
    print(f"找到 {len(data_files)} 个数据文件")
    
    # 我们需要的品种列表
    target_symbols = ['SP', 'P', 'BU', 'CU', 'V', 'FG', 'CF', 'SC', 'AU', 'AG', 
                      'MA', 'PG', 'RM', 'RU', 'TA', 'EB', 'SR', 'FU', 'I', 'SN', 
                      'OI', 'ZN', 'Y', 'SA', 'M', 'RB', 'EG', 'PP', 'AL', 'NI', 'HC']
    
    for data_file in data_files:
        filename = os.path.basename(data_file)
        # 从文件名中提取品种代码（如 AG_main.feather -> AG）
        symbol = filename.split('_')[0]
        
        # 只加载目标品种
        if symbol not in target_symbols:
            continue
        
        try:
            if data_file.endswith('.feather'):
                df = pd.read_feather(data_file)
            else:
                df = pd.read_csv(data_file)
            
            # 合并tradeDate和barTime创建datetime索引
            if 'tradeDate' in df.columns and 'barTime' in df.columns:
                df['datetime'] = pd.to_datetime(df['tradeDate'] + ' ' + df['barTime'])
                df = df.set_index('datetime')
            
            data[symbol] = df
            print(f"加载 {symbol}: {len(df)} 条数据")
        except Exception as e:
            print(f"加载 {data_file} 失败: {e}")
    
    return data

def plot_kline_with_signals(df, symbol, save_path=None):
    fig, ax = plt.subplots(figsize=(15, 8))
    
    plot_df = df.iloc[-500:].copy()
    
    ax.plot(plot_df.index, plot_df['hfq_closePrice'], label='收盘价', linewidth=1)
    ax.plot(plot_df.index, plot_df['upper_band'], label='上轨', linewidth=1.5, color='orange', alpha=0.7)
    ax.plot(plot_df.index, plot_df['middle_band'], label='中轨', linewidth=1.5, color='gray', alpha=0.7)
    ax.plot(plot_df.index, plot_df['lower_band'], label='下轨', linewidth=1.5, color='orange', alpha=0.7)
    
    # 填充布林带区域
    ax.fill_between(plot_df.index, plot_df['lower_band'], plot_df['upper_band'], color='orange', alpha=0.1)
    
    long_entries = plot_df[plot_df['signal'] == 1]
    short_entries = plot_df[plot_df['signal'] == -1]
    
    ax.scatter(long_entries.index, long_entries['hfq_closePrice'], 
               marker='^', color='red', s=100, label='开多仓')
    ax.scatter(short_entries.index, short_entries['hfq_closePrice'], 
               marker='v', color='green', s=100, label='开空仓')
    
    ax.set_title(f'{symbol} K线图与交易信号')
    ax.legend()
    ax.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"K线图已保存到: {save_path}")
    
    plt.close()

def plot_equity_curves(results, save_path=None):
    fig, ax = plt.subplots(figsize=(15, 8))
    
    for symbol in results:
        equity_df = results[symbol]['equity']
        if len(equity_df) > 0:
            ax.plot(equity_df.index, equity_df['net_value'], label=symbol, linewidth=1, alpha=0.7)
    
    ax.axhline(y=1, color='black', linestyle='--', linewidth=0.8)
    ax.set_title('各品种净值曲线')
    ax.set_xlabel('日期')
    ax.set_ylabel('净值')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"净值曲线已保存到: {save_path}")
    
    plt.close()

def plot_combined_equity(combined_equity, save_path=None):
    fig, ax = plt.subplots(figsize=(15, 8))
    
    ax.plot(combined_equity.index, combined_equity['net_value'], 
            label='组合净值', linewidth=2, color='blue')
    ax.axhline(y=1, color='black', linestyle='--', linewidth=0.8)
    
    ax.set_title('组合净值曲线')
    ax.set_xlabel('日期')
    ax.set_ylabel('净值')
    ax.legend()
    ax.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"组合净值曲线已保存到: {save_path}")
    
    plt.close()

def main():
    print("=" * 60)
    print("量化策略回测系统")
    print("=" * 60)
    
    output_dir = '/Users/bytedance/Desktop/my_notes/量化交易/output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    strategy = RSIReversionStrategy(rsi_window=14, oversold=20, overbought=80)
    print(f"\n策略: RSI均值回归策略 (RSI窗口={strategy.rsi_window}, 超卖线={strategy.oversold}, 超买线={strategy.overbought})")
    
    engine = BacktestEngine(strategy, initial_capital=1000000)
    
    data_dir = '/Users/bytedance/Desktop/my_notes/量化交易/main&smain_market_data_01min'
    data = load_data(data_dir)
    
    if len(data) == 0:
        print("\n未找到真实数据，请检查数据目录。回测终止。")
        return
    
    print("\n开始回测...")
    all_equities = []
    
    for symbol in data:
        print(f"回测品种: {symbol}")
        try:
            result = engine.backtest_symbol(symbol, data[symbol])
            if len(result['equity']) > 0:
                equity_series = result['equity']['net_value']
                equity_series.name = symbol
                all_equities.append(equity_series)
            
            if symbol == list(data.keys())[0]:
                kline_path = os.path.join(output_dir, f'{symbol}_kline.png')
                plot_kline_with_signals(result['data_with_signals'], symbol, kline_path)
        except Exception as e:
            print(f"回测 {symbol} 失败: {e}")
    
    if len(all_equities) > 0:
        combined_df = pd.concat(all_equities, axis=1)
        combined_df = combined_df.fillna(method='ffill').fillna(1)
        combined_equity = pd.DataFrame({
            'net_value': combined_df.mean(axis=1)
        }, index=combined_df.index)
        
        print("\n生成图表...")
        equity_path = os.path.join(output_dir, 'equity_curves.png')
        plot_equity_curves(engine.results, equity_path)
        
        combined_path = os.path.join(output_dir, 'combined_equity.png')
        plot_combined_equity(combined_equity, combined_path)
        
        print("\n" + "=" * 60)
        print("综合回测指标")
        print("=" * 60)
        metrics = engine.calculate_metrics(combined_equity)
        
        for key, value in metrics.items():
            if isinstance(value, float):
                if key in ['sharpe_ratio', 'calmar_ratio', 'win_loss_ratio']:
                    print(f"{key:25s}: {value:.4f}")
                elif key in ['annualized_return', 'max_drawdown', 'win_rate']:
                    print(f"{key:25s}: {value:.2%}")
                elif key == 'avg_holding_time':
                    print(f"{key:25s}: {value:.2f} 分钟")
                elif 'profit' in key or 'loss' in key:
                    print(f"{key:25s}: {value:.2f}")
                else:
                    print(f"{key:25s}: {value:.4f}")
            else:
                print(f"{key:25s}: {value}")
        
        metrics_path = os.path.join(output_dir, 'metrics.txt')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            f.write("综合回测指标\n")
            f.write("=" * 60 + "\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
                
        # 打印各品种详细表现并保存
        print("\n" + "=" * 60)
        print("各品种表现排行 (按总收益率排序)")
        print("=" * 60)
        
        symbol_metrics = []
        for symbol, res in engine.results.items():
            if len(res['equity']) > 0:
                total_return = res['equity']['net_value'].iloc[-1] - 1
                trades = res['trades']
                win_trades = [t for t in trades if t['pnl'] > 0]
                win_rate = len(win_trades) / len(trades) if len(trades) > 0 else 0
                num_trades = len(trades)
                
                symbol_metrics.append({
                    'symbol': symbol,
                    'return': total_return,
                    'win_rate': win_rate,
                    'num_trades': num_trades
                })
        
        # 按收益率降序排序
        symbol_metrics.sort(key=lambda x: x['return'], reverse=True)
        
        print(f"{'品种':<8} | {'总收益率':<12} | {'胜率':<10} | {'交易次数':<10}")
        print("-" * 50)
        for m in symbol_metrics:
            print(f"{m['symbol']:<10} | {m['return']:>10.2%} | {m['win_rate']:>8.2%} | {m['num_trades']:>8}")
            
        # 将品种表现也写入metrics.txt
        with open(metrics_path, 'a', encoding='utf-8') as f:
            f.write("\n\n各品种表现排行 (按总收益率排序)\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'品种':<8} | {'总收益率':<12} | {'胜率':<10} | {'交易次数':<10}\n")
            for m in symbol_metrics:
                f.write(f"{m['symbol']:<10} | {m['return']:>10.2%} | {m['win_rate']:>8.2%} | {m['num_trades']:>8}\n")
        
        print(f"\n所有结果已保存到: {output_dir}")
    
    print("\n回测完成！")

if __name__ == '__main__':
    main()
