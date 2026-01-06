"""
Gap-Fill Backtest Using Daily Bars
==================================
Uses 2 years of daily bar data to backtest gap-fill strategy.

This is a simulation since we don't have minute bars, but:
- Gap calculation is accurate (prev close → open)
- Simulates 120-minute hold with assumptions about intraday movement
- Provides much larger sample size for statistical validation

Assumptions:
- Entry at open
- If gap fills (price reaches prev close), we exit at prev close
- If gap doesn't fill by EOD, we exit at close (simulating 120-min exit)
- Stop loss checked against day's low/high
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict
import logging

import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.cached_data_manager import CachedDataManager
from config import DIRS

logger = logging.getLogger(__name__)


class GapFillDailyBacktest:
    """
    Backtest gap-fill using daily bars.
    
    Conservative assumptions:
    - Entry at open price
    - Check if gap filled using intraday high/low
    - Exit at prev_close if filled, else at close
    - Stop loss at 0.5% from entry
    """
    
    # Universe
    UNIVERSE = ['SPY', 'QQQ']
    
    # Parameters (matching intraday strategy)
    MIN_GAP_PCT = 0.15
    MAX_GAP_PCT = 0.60
    STOP_LOSS_PCT = 0.50
    
    def __init__(self):
        self.data_mgr = CachedDataManager()
        
    def load_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load daily data for a symbol."""
        return self.data_mgr.get_bars(symbol)
    
    def calculate_gap(self, row: pd.Series, prev_close: float) -> dict:
        """Calculate gap from previous close to current open."""
        gap_dollars = row['open'] - prev_close
        gap_percent = (gap_dollars / prev_close) * 100
        
        return {
            'prev_close': prev_close,
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'gap_dollars': gap_dollars,
            'gap_percent': gap_percent
        }
    
    def check_3day_range(
        self, 
        df: pd.DataFrame, 
        idx: int, 
        open_price: float
    ) -> bool:
        """Check if open is outside previous 3-day range."""
        if idx < 3:
            return False
        
        prev_3 = df.iloc[idx-3:idx]
        range_low = prev_3['low'].min()
        range_high = prev_3['high'].max()
        
        return open_price < range_low or open_price > range_high
    
    def simulate_trade(self, gap_info: dict, is_long: bool) -> dict:
        """
        Simulate gap-fill trade using daily OHLC.
        
        For long trades (gap down):
        - Entry at open
        - Target = prev_close (gap fill)
        - Check if high reached prev_close (gap filled)
        - Check if low hit stop
        - Exit at close if neither
        """
        entry = gap_info['open']
        target = gap_info['prev_close']
        high = gap_info['high']
        low = gap_info['low']
        close = gap_info['close']
        
        if is_long:
            stop = entry * (1 - self.STOP_LOSS_PCT / 100)
            
            # Check stop first (conservative - assume it happened first if both triggered)
            if low <= stop:
                return {
                    'exit_price': stop,
                    'exit_reason': 'stop_loss',
                    'gap_filled': False,
                    'pnl_percent': -self.STOP_LOSS_PCT
                }
            
            # Check if gap filled
            if high >= target:
                pnl = (target / entry - 1) * 100
                return {
                    'exit_price': target,
                    'exit_reason': 'target',
                    'gap_filled': True,
                    'pnl_percent': pnl
                }
            
            # Time exit (simulated as close)
            pnl = (close / entry - 1) * 100
            return {
                'exit_price': close,
                'exit_reason': 'time_exit',
                'gap_filled': False,
                'pnl_percent': pnl
            }
        
        else:  # Short
            stop = entry * (1 + self.STOP_LOSS_PCT / 100)
            
            if high >= stop:
                return {
                    'exit_price': stop,
                    'exit_reason': 'stop_loss',
                    'gap_filled': False,
                    'pnl_percent': -self.STOP_LOSS_PCT
                }
            
            if low <= target:
                pnl = (entry / target - 1) * 100
                return {
                    'exit_price': target,
                    'exit_reason': 'target',
                    'gap_filled': True,
                    'pnl_percent': pnl
                }
            
            pnl = (entry / close - 1) * 100
            return {
                'exit_price': close,
                'exit_reason': 'time_exit',
                'gap_filled': False,
                'pnl_percent': pnl
            }
    
    def backtest_symbol(
        self, 
        symbol: str,
        start_date: datetime = None,
        end_date: datetime = None,
        long_only: bool = True,
        require_outside_range: bool = False
    ) -> pd.DataFrame:
        """
        Backtest gap-fill on a single symbol.
        
        Args:
            symbol: Stock symbol
            start_date: Start date (default: 1 year ago)
            end_date: End date (default: today)
            long_only: Only trade gap-downs (long)
            require_outside_range: Only trade if outside 3-day range
        """
        df = self.load_data(symbol)
        if df is None or len(df) < 10:
            logger.warning(f"Insufficient data for {symbol}")
            return pd.DataFrame()
        
        # Ensure we have a datetime index or column
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        elif 'date' in df.columns:
            df = df.set_index('date')
        
        # Convert index to datetime if needed
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                logger.warning(f"Cannot convert index to datetime for {symbol}: {e}")
                return pd.DataFrame()
        
        # Filter date range
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]
        
        results = []
        
        for i in range(1, len(df)):
            prev_close = df.iloc[i-1]['close']
            row = df.iloc[i]
            date = df.index[i]
            
            # Calculate gap
            gap = self.calculate_gap(row, prev_close)
            gap_pct = abs(gap['gap_percent'])
            
            # Filter by gap size
            if gap_pct < self.MIN_GAP_PCT or gap_pct > self.MAX_GAP_PCT:
                continue
            
            # Check 3-day range filter
            outside_range = self.check_3day_range(df, i, gap['open'])
            if require_outside_range and not outside_range:
                continue
            
            # Determine direction
            is_gap_down = gap['gap_percent'] < 0
            
            if long_only and not is_gap_down:
                continue
            
            # Simulate trade
            trade_result = self.simulate_trade(gap, is_long=is_gap_down)
            
            results.append({
                'date': date.date() if hasattr(date, 'date') else date,
                'symbol': symbol,
                'direction': 'long' if is_gap_down else 'short',
                'gap_percent': gap['gap_percent'],
                'entry_price': gap['open'],
                'exit_price': trade_result['exit_price'],
                'exit_reason': trade_result['exit_reason'],
                'pnl_percent': trade_result['pnl_percent'],
                'gap_filled': trade_result['gap_filled'],
                'outside_range': outside_range,
                'win': trade_result['pnl_percent'] > 0
            })
        
        return pd.DataFrame(results)
    
    def backtest_all(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
        long_only: bool = True,
        require_outside_range: bool = False
    ) -> pd.DataFrame:
        """Backtest across all symbols in universe."""
        all_results = []
        
        for symbol in self.UNIVERSE:
            logger.info(f"Backtesting {symbol}...")
            results = self.backtest_symbol(
                symbol, start_date, end_date, 
                long_only, require_outside_range
            )
            if len(results) > 0:
                all_results.append(results)
        
        if not all_results:
            return pd.DataFrame()
        
        return pd.concat(all_results, ignore_index=True)
    
    def analyze_results(self, results: pd.DataFrame) -> dict:
        """Calculate strategy statistics."""
        if len(results) == 0:
            return {}
        
        wins = results['win'].sum()
        total = len(results)
        
        winners = results[results['win']]
        losers = results[~results['win']]
        
        stats = {
            'total_trades': total,
            'wins': wins,
            'losses': total - wins,
            'win_rate': wins / total * 100 if total > 0 else 0,
            'avg_win': winners['pnl_percent'].mean() if len(winners) > 0 else 0,
            'avg_loss': losers['pnl_percent'].mean() if len(losers) > 0 else 0,
            'total_pnl': results['pnl_percent'].sum(),
            'avg_pnl': results['pnl_percent'].mean(),
            'gap_fill_rate': results['gap_filled'].mean() * 100,
            'by_exit_reason': results['exit_reason'].value_counts().to_dict(),
            'by_symbol': results.groupby('symbol')['pnl_percent'].sum().to_dict()
        }
        
        # Calculate profit factor
        gross_profit = winners['pnl_percent'].sum() if len(winners) > 0 else 0
        gross_loss = abs(losers['pnl_percent'].sum()) if len(losers) > 0 else 1
        # Clamp profit_factor to avoid infinity propagation
        stats['profit_factor'] = min(10.0, gross_profit / gross_loss) if gross_loss > 0 else 10.0
        
        # Sharpe approximation (daily)
        if len(results) > 1:
            daily_returns = results.groupby('date')['pnl_percent'].sum()
            if len(daily_returns) > 1 and daily_returns.std() > 0:
                stats['sharpe_approx'] = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
            else:
                stats['sharpe_approx'] = 0
        else:
            stats['sharpe_approx'] = 0
        
        return stats
    
    def print_results(self, results: pd.DataFrame, title: str = ""):
        """Pretty print backtest results."""
        if len(results) == 0:
            print(f"\n{title}")
            print("No trades found")
            return
        
        stats = self.analyze_results(results)
        
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")
        print(f"Total Trades: {stats['total_trades']}")
        print(f"Win Rate: {stats['win_rate']:.1f}%")
        print(f"Gap Fill Rate: {stats['gap_fill_rate']:.1f}%")
        print(f"Avg Win: {stats['avg_win']:.2f}%")
        print(f"Avg Loss: {stats['avg_loss']:.2f}%")
        print(f"Total P&L: {stats['total_pnl']:.2f}%")
        print(f"Profit Factor: {stats['profit_factor']:.2f}")
        print(f"Sharpe (approx): {stats['sharpe_approx']:.2f}")
        print(f"\nBy Exit Reason: {stats['by_exit_reason']}")
        print(f"By Symbol: {stats['by_symbol']}")


def run_daily_backtest():
    """Run comprehensive gap-fill backtest using daily bars."""
    bt = GapFillDailyBacktest()
    
    # 2 year backtest
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    
    print("="*60)
    print("GAP-FILL BACKTEST (Using Daily Bars)")
    print("="*60)
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Universe: {bt.UNIVERSE}")
    print(f"Gap Range: {bt.MIN_GAP_PCT}% - {bt.MAX_GAP_PCT}%")
    
    # Run all trades
    results_all = bt.backtest_all(start_date, end_date, require_outside_range=False)
    bt.print_results(results_all, "ALL GAPS (0.15% - 0.60%)")
    
    # Run high confidence only
    results_hc = bt.backtest_all(start_date, end_date, require_outside_range=True)
    bt.print_results(results_hc, "HIGH CONFIDENCE (outside 3-day range)")
    
    # Year-over-year comparison
    if len(results_all) > 0:
        print(f"\n{'='*60}")
        print("YEAR-OVER-YEAR COMPARISON")
        print(f"{'='*60}")
        
        results_all['year'] = pd.to_datetime(results_all['date']).dt.year
        for year in sorted(results_all['year'].unique()):
            year_results = results_all[results_all['year'] == year]
            wins = year_results['win'].sum()
            total = len(year_results)
            pnl = year_results['pnl_percent'].sum()
            print(f"{year}: {total} trades, {wins/total*100:.1f}% win rate, {pnl:.2f}% total P&L")
    
    # Monthly distribution
    if len(results_all) > 0:
        print(f"\n{'='*60}")
        print("MONTHLY WIN RATE")
        print(f"{'='*60}")
        
        results_all['month'] = pd.to_datetime(results_all['date']).dt.month
        monthly = results_all.groupby('month').agg({
            'win': ['sum', 'count'],
            'pnl_percent': 'sum'
        })
        monthly.columns = ['wins', 'total', 'pnl']
        monthly['win_rate'] = monthly['wins'] / monthly['total'] * 100
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for m in range(1, 13):
            if m in monthly.index:
                row = monthly.loc[m]
                print(f"{months[m-1]}: {int(row['total']):3d} trades, {row['win_rate']:5.1f}% win, {row['pnl']:.2f}% P&L")
    
    # Show recent trades
    if len(results_all) > 0:
        print(f"\n{'='*60}")
        print("RECENT TRADES (Last 20)")
        print(f"{'='*60}")
        recent = results_all.tail(20)
        print(recent[['date', 'symbol', 'gap_percent', 'exit_reason', 'pnl_percent', 'win']].to_string())
    
    return results_all


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_daily_backtest()
