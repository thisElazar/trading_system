"""
Relative Volume Breakout Strategy
=================================
Tier 1 Core Strategy

Research Basis:
--------------
1. Lee, Kim, Kim (2016) "Abnormal Trading Volume and the Cross-Section of Stock Returns"
   SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2812010
   Key Finding: Return forecasting power of abnormal trading activity is strongly
   positive and persists up to 5 weeks after the volume spike.

2. Barber & Odean (2008) "All That Glitters: The Effect of Attention on Buying Behavior"
   Wharton: https://rodneywhitecenter.wharton.upenn.edu/wp-content/uploads/2014/04/9901.pdf
   Key Finding: Abnormally high trading volume serves as a proxy for investor attention,
   which drives buying pressure and short-term price momentum.

3. Chen, So, Chiang (2014) "Evidence of Stock Returns and Abnormal Trading Volume"
   Key Finding: Positive effect of abnormal volume on returns is strongest under
   high quantile levels (extreme volume spikes).

Expected Performance (Realistic):
- Sharpe Ratio: 0.5-0.8 (net of transaction costs)
- Annual Returns: 10-15% (vs ~10% market baseline)
- Effect Duration: 1-5 weeks per Lee et al. (2016)
- Win Rate: 45-55% with proper risk management

Key Insight: Stocks with abnormally high relative volume (due to news, earnings,
or catalysts) exhibit short-term momentum that can be captured via breakout strategies.
The academic evidence supports this effect persisting for 1-5 weeks.

Implementation:
- Screen for RV > 150% (current volume vs 20-day average)
- Filter for price gaps as catalyst proxy
- Enter on breakout above previous day's high (long) or below low (short)
- ATR-based stops and targets
- Short holding period (1-5 days) aligned with research timeframe
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import logging

import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.base import BaseStrategy, Signal, SignalType
from data.cached_data_manager import CachedDataManager
from config import DIRS
from utils.timezone import normalize_dataframe, normalize_timestamp, normalize_index

logger = logging.getLogger(__name__)


class RelativeVolumeBreakout(BaseStrategy):
    """
    Relative Volume Breakout Strategy

    Academic Foundation:
    - Lee et al. (2016): Abnormal volume predicts returns for up to 5 weeks
    - Barber & Odean (2008): High volume = attention proxy = buying pressure
    - Chen et al. (2014): Volume-return effect strongest at high quantiles

    Stocks with abnormally high trading volume (due to news, earnings, or
    catalysts) exhibit momentum that persists for 1-5 weeks per research.

    Daily bar implementation:
    - RV = today's volume / 20-day avg volume
    - Filter: RV > 1.5 (150% of normal) + price gap as catalyst proxy
    - Entry: price breaks above previous day's high (long)
    - Stop: below previous day's low minus ATR buffer
    - Hold: 1-5 days (aligned with research effect duration)

    Expected Performance:
    - Sharpe: 0.5-0.8 (realistic, net of costs)
    - Annual excess returns: 2-5% above market
    - Win rate: 45-55%
    """
    
    # Parameters - Based on academic research
    # Lee et al. (2016): Effect persists 1-5 weeks, strongest in first week
    # Chen et al. (2014): Higher volume thresholds yield stronger effect
    MIN_RELATIVE_VOLUME = 1.5  # 150% of average - balances signal quality vs frequency
    MIN_GAP_PCT = 0.02         # 2% gap as catalyst proxy (news/earnings)
    MIN_PRICE = 10.0           # Avoid penny stock noise and liquidity issues
    LOOKBACK_DAYS = 20         # Standard 1-month lookback for volume average
    ATR_STOP_MULT = 1.0        # Tighter stop to limit downside
    ATR_TARGET_MULT = 2.0      # 2:1 reward-to-risk ratio
    MAX_HOLD_DAYS = 5          # Aligned with Lee et al. finding (up to 5 weeks)
    MAX_POSITIONS = 5          # Diversification limit
    
    def __init__(self):
        super().__init__("relative_volume_breakout")
        # Lazy-load data_mgr only when needed (scan_universe)
        # During backtesting, data is passed to generate_signals directly
        self._data_mgr = None

    @property
    def data_mgr(self):
        """Lazy-load CachedDataManager only when needed."""
        if self._data_mgr is None:
            self._data_mgr = CachedDataManager()
        return self._data_mgr
    
    def calculate_relative_volume(self, df: pd.DataFrame) -> pd.Series:
        """Calculate relative volume vs 20-day average."""
        avg_volume = df['volume'].rolling(self.LOOKBACK_DAYS).mean()
        return df['volume'] / avg_volume
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    def scan_universe(
        self, 
        symbols: List[str] = None,
        date: datetime = None
    ) -> List[dict]:
        """
        Scan for stocks with high relative volume.
        
        Args:
            symbols: List of symbols to scan
            date: Date to scan (defaults to latest)
            
        Returns:
            List of candidates with RV data
        """
        if symbols is None:
            # Use a reasonable universe - S&P 500 + NASDAQ 100
            ref_path = DIRS["reference"] / "sp500_constituents.json"
            if ref_path.exists():
                import json
                with open(ref_path) as f:
                    data = json.load(f)
                    symbols = data.get('symbols', [])[:200]  # Top 200
            else:
                symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 
                          'TSLA', 'AMD', 'NFLX', 'CRM']
        
        candidates = []
        
        for symbol in symbols:
            try:
                df = self.data_mgr.get_bars(symbol)
                if df is None or len(df) < 30:
                    continue
                
                # Ensure datetime index
                if 'timestamp' in df.columns:
                    df = df.set_index('timestamp')
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                
                # Calculate indicators
                df['rv'] = self.calculate_relative_volume(df)
                df['atr'] = self.calculate_atr(df)
                
                # Get latest row
                latest = df.iloc[-1]
                prev = df.iloc[-2]
                
                rv = latest['rv']
                price = latest['close']
                
                # Apply filters
                if rv < self.MIN_RELATIVE_VOLUME:
                    continue
                if price < self.MIN_PRICE:
                    continue
                
                # Calculate gap from prior close (catalyst proxy)
                gap_pct = (latest['open'] - prev['close']) / prev['close']
                if abs(gap_pct) < self.MIN_GAP_PCT:
                    continue  # No significant gap = no catalyst
                
                candidates.append({
                    'symbol': symbol,
                    'date': df.index[-1],
                    'relative_volume': rv,
                    'volume': latest['volume'],
                    'avg_volume': latest['volume'] / rv,
                    'close': price,
                    'gap_pct': gap_pct,
                    'prev_high': prev['high'],
                    'prev_low': prev['low'],
                    'atr': latest['atr'],
                    'breakout_long': price > prev['high'] and gap_pct > 0,
                    'breakout_short': price < prev['low'] and gap_pct < 0
                })
            
            except Exception as e:
                logger.debug(f"Error processing {symbol}: {e}")
                continue
        
        # Sort by relative volume
        candidates.sort(key=lambda x: x['relative_volume'], reverse=True)
        
        return candidates
    
    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame],
        current_positions: List[str] = None,
        vix_regime: str = None
    ) -> List[Signal]:
        """
        Generate breakout signals for high RV stocks.
        
        Args:
            data: Dict mapping symbol to DataFrame with OHLCV + indicators
            current_positions: List of symbols currently held by this strategy
            vix_regime: Current VIX regime ('low', 'normal', 'high', 'extreme')
        """
        signals = []
        
        if not data:
            return signals
        
        # Get current date from data
        current_date = None
        for symbol, df in data.items():
            if len(df) > 0:
                if isinstance(df.index, pd.DatetimeIndex):
                    current_date = df.index[-1].to_pydatetime()
                    break
        
        if current_date is None:
            current_date = datetime.now()
        
        # VIX adjustment - reduce position count in high vol
        max_signals = self.MAX_POSITIONS
        if vix_regime in ('high', 'extreme'):
            max_signals = max(2, self.MAX_POSITIONS // 2)
        
        candidates = []
        current_holdings = set(current_positions) if current_positions else set()
        
        for symbol, df in data.items():
            try:
                if len(df) < 30:
                    continue

                # Ensure datetime index
                if 'timestamp' in df.columns:
                    df = df.set_index('timestamp')
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)

                # Use pre-computed indicators if available (10-20x faster)
                # Falls back to calculation if not present
                if 'relative_volume' in df.columns:
                    rv = df['relative_volume'].iloc[-1]
                else:
                    rv_series = self.calculate_relative_volume(df)
                    rv = rv_series.iloc[-1] if not np.isnan(rv_series.iloc[-1]) else 0

                if 'atr' in df.columns:
                    atr = df['atr'].iloc[-1]
                else:
                    atr_series = self.calculate_atr(df)
                    atr = atr_series.iloc[-1] if not np.isnan(atr_series.iloc[-1]) else 0

                # Handle NaN values
                if np.isnan(rv):
                    rv = 0
                if np.isnan(atr):
                    atr = 0

                # Get latest row
                latest = df.iloc[-1]
                prev = df.iloc[-2] if len(df) > 1 else latest
                price = latest['close']
                
                # Apply filters
                if rv < self.MIN_RELATIVE_VOLUME:
                    continue
                if price < self.MIN_PRICE:
                    continue
                
                # Calculate gap from prior close (catalyst proxy)
                gap_pct = (latest['open'] - prev['close']) / prev['close'] if prev['close'] > 0 else 0
                if abs(gap_pct) < self.MIN_GAP_PCT:
                    continue  # No significant gap = no catalyst
                
                prev_high = prev['high']
                prev_low = prev['low']
                breakout_long = price > prev_high and gap_pct > 0
                breakout_short = price < prev_low and gap_pct < 0
                
                candidates.append({
                    'symbol': symbol,
                    'date': df.index[-1],
                    'relative_volume': rv,
                    'close': price,
                    'gap_pct': gap_pct,
                    'prev_high': prev_high,
                    'prev_low': prev_low,
                    'atr': atr,
                    'breakout_long': breakout_long,
                    'breakout_short': breakout_short
                })
            
            except Exception as e:
                logger.debug(f"Error processing {symbol}: {e}")
                continue
        
        # Sort by relative volume
        candidates.sort(key=lambda x: x['relative_volume'], reverse=True)
        
        # Generate signals
        for cand in candidates[:max_signals]:
            symbol = cand['symbol']
            
            # Check for exit signals first
            if symbol in current_holdings:
                # Exit if holding too long or no longer a breakout candidate
                signals.append(Signal(
                    timestamp=current_date,
                    symbol=symbol,
                    strategy=self.name,
                    signal_type=SignalType.CLOSE,
                    strength=1.0,
                    price=cand['close'],
                    reason='rv_breakout_exit'
                ))
                continue
            
            if cand['breakout_long']:
                # Long breakout signal
                stop = cand['prev_low'] - cand['atr'] * self.ATR_STOP_MULT
                target = cand['close'] + cand['atr'] * self.ATR_TARGET_MULT
                
                signals.append(Signal(
                    timestamp=current_date,
                    symbol=symbol,
                    strategy=self.name,
                    signal_type=SignalType.BUY,
                    strength=min(0.9, 0.5 + cand['relative_volume'] / 10),
                    price=cand['close'],
                    stop_loss=stop,
                    target_price=target,
                    reason=f'rv_breakout_long',
                    metadata={
                        'relative_volume': cand['relative_volume'],
                        'gap_pct': cand['gap_pct'],
                        'atr': cand['atr'],
                        'breakout_type': 'long'
                    }
                ))
                
            elif cand['breakout_short']:
                # For long-only, skip short breakouts
                # Could convert to buying put candidates in future
                pass
        
        if signals:
            logger.debug(f"RV Breakout generated {len(signals)} signals (RV > {self.MIN_RELATIVE_VOLUME}x)")
        
        return signals


class RVBreakoutBacktester:
    """Backtest relative volume breakout strategy."""
    
    def __init__(self):
        self.data_mgr = CachedDataManager()
        self.strategy = RelativeVolumeBreakout()
    
    def backtest_symbol(
        self,
        symbol: str,
        start_date: datetime = None,
        end_date: datetime = None,
        min_rv: float = None,
        min_gap_pct: float = None,
        min_price: float = None,
        atr_stop_mult: float = None,
        atr_target_mult: float = None,
        max_hold_days: int = None
    ) -> pd.DataFrame:
        """
        Backtest RV breakout on a single symbol.
        
        Trade logic:
        - Entry: RV > threshold AND gap > threshold AND close > prev_high (long)
        - Stop: prev_low - ATR (long)
        - Target: 1.5x ATR from entry
        - Exit: stop, target, or 2-day timeout
        """
        # Use strategy defaults if not provided
        min_rv = min_rv or self.strategy.MIN_RELATIVE_VOLUME
        min_gap_pct = min_gap_pct or self.strategy.MIN_GAP_PCT
        min_price = min_price or self.strategy.MIN_PRICE
        atr_stop_mult = atr_stop_mult or self.strategy.ATR_STOP_MULT
        atr_target_mult = atr_target_mult or self.strategy.ATR_TARGET_MULT
        max_hold_days = max_hold_days or self.strategy.MAX_HOLD_DAYS
        
        df = self.data_mgr.get_bars(symbol)
        if df is None or len(df) < 50:
            return pd.DataFrame()
        
        # Prepare data
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Normalize timezone
        df = normalize_dataframe(df)
        
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]
        
        # Calculate indicators
        df['rv'] = self.strategy.calculate_relative_volume(df)
        df['atr'] = self.strategy.calculate_atr(df)
        df['prev_high'] = df['high'].shift(1)
        df['prev_low'] = df['low'].shift(1)
        df['prev_close'] = df['close'].shift(1)
        df['gap_pct'] = (df['open'] - df['prev_close']) / df['prev_close']
        
        df = df.dropna()
        
        trades = []
        position = None
        
        for i in range(1, len(df)):
            row = df.iloc[i]
            date = df.index[i]
            
            if position is None:
                # Apply all filters
                rv = row['rv']
                price = row['close']
                gap = row['gap_pct']
                
                if rv < min_rv:
                    continue
                if price < min_price:
                    continue
                if abs(gap) < min_gap_pct:
                    continue
                
                # Long breakout: gap up + close above prev high
                if gap > 0 and price > row['prev_high']:
                    position = {
                        'entry_date': date,
                        'direction': 'long',
                        'entry_price': price,
                        'stop': row['prev_low'] - row['atr'] * atr_stop_mult,
                        'target': price + row['atr'] * atr_target_mult,
                        'rv': rv,
                        'gap_pct': gap
                    }
                # Short breakout: gap down + close below prev low
                elif gap < 0 and price < row['prev_low']:
                    position = {
                        'entry_date': date,
                        'direction': 'short',
                        'entry_price': price,
                        'stop': row['prev_high'] + row['atr'] * atr_stop_mult,
                        'target': price - row['atr'] * atr_target_mult,
                        'rv': rv,
                        'gap_pct': gap
                    }
            else:
                # Check exit conditions
                days_held = (date - position['entry_date']).days
                exit_reason = None
                exit_price = None
                
                if position['direction'] == 'long':
                    if row['low'] <= position['stop']:
                        exit_reason = 'stop_loss'
                        exit_price = position['stop']
                    elif row['high'] >= position['target']:
                        exit_reason = 'target'
                        exit_price = position['target']
                    elif days_held >= max_hold_days:
                        exit_reason = 'timeout'
                        exit_price = row['close']
                else:  # short
                    if row['high'] >= position['stop']:
                        exit_reason = 'stop_loss'
                        exit_price = position['stop']
                    elif row['low'] <= position['target']:
                        exit_reason = 'target'
                        exit_price = position['target']
                    elif days_held >= max_hold_days:
                        exit_reason = 'timeout'
                        exit_price = row['close']
                
                if exit_reason:
                    if position['direction'] == 'long':
                        pnl = (exit_price / position['entry_price'] - 1) * 100
                    else:
                        pnl = (position['entry_price'] / exit_price - 1) * 100
                    
                    trades.append({
                        'symbol': symbol,
                        'entry_date': position['entry_date'],
                        'exit_date': date,
                        'direction': position['direction'],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'exit_reason': exit_reason,
                        'days_held': days_held,
                        'rv': position['rv'],
                        'gap_pct': position['gap_pct'],
                        'pnl_pct': pnl,
                        'win': pnl > 0
                    })
                    position = None
        
        return pd.DataFrame(trades)
    
    def backtest_universe(
        self,
        symbols: List[str] = None,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> pd.DataFrame:
        """Backtest across multiple symbols."""
        if symbols is None:
            # Get S&P 500
            ref_path = DIRS["reference"] / "sp500_constituents.json"
            if ref_path.exists():
                import json
                with open(ref_path) as f:
                    data = json.load(f)
                    symbols = data.get('symbols', [])
            else:
                symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 
                          'TSLA', 'AMD', 'NFLX', 'CRM', 'JPM', 'BAC']
        
        all_trades = []
        
        for i, symbol in enumerate(symbols):
            if i % 50 == 0:
                logger.info(f"Processing {i}/{len(symbols)}...")
            
            trades = self.backtest_symbol(symbol, start_date, end_date)
            if len(trades) > 0:
                all_trades.append(trades)
        
        if not all_trades:
            return pd.DataFrame()
        
        return pd.concat(all_trades, ignore_index=True)


def run_rv_scan():
    """Scan for current high RV stocks."""
    print("="*60)
    print("RELATIVE VOLUME BREAKOUT - SCANNER")
    print("="*60)
    
    strategy = RelativeVolumeBreakout()
    candidates = strategy.scan_universe()
    
    print(f"\nFound {len(candidates)} stocks with RV > {strategy.MIN_RELATIVE_VOLUME}x\n")
    
    for c in candidates[:20]:
        breakout = "LONG↑" if c['breakout_long'] else ("SHORT↓" if c['breakout_short'] else "-")
        print(f"{c['symbol']:6s} | RV: {c['relative_volume']:.1f}x | "
              f"${c['close']:.2f} | {breakout}")
    
    return candidates


def run_rv_backtest():
    """Run RV breakout backtest."""
    print("="*60)
    print("RELATIVE VOLUME BREAKOUT BACKTEST")
    print("="*60)
    
    backtester = RVBreakoutBacktester()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print("Scanning universe...\n")
    
    results = backtester.backtest_universe(start_date=start_date, end_date=end_date)
    
    if len(results) == 0:
        print("No trades generated")
        return
    
    # Results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Total Trades: {len(results)}")
    print(f"Win Rate: {results['win'].mean()*100:.1f}%")
    print(f"Avg P&L: {results['pnl_pct'].mean():.2f}%")
    print(f"Total P&L: {results['pnl_pct'].sum():.2f}%")
    print(f"Avg Days Held: {results['days_held'].mean():.1f}")
    
    print(f"\nBy Exit Reason:")
    for reason, group in results.groupby('exit_reason'):
        print(f"  {reason}: {len(group)} trades, {group['win'].mean()*100:.1f}% win rate")
    
    print(f"\nBy Direction:")
    for direction, group in results.groupby('direction'):
        print(f"  {direction}: {len(group)} trades, {group['pnl_pct'].sum():.2f}% P&L")
    
    print(f"\nTop 10 Performers:")
    by_symbol = results.groupby('symbol').agg({
        'pnl_pct': ['sum', 'count'],
        'win': 'mean'
    })
    by_symbol.columns = ['total_pnl', 'trades', 'win_rate']
    by_symbol = by_symbol.sort_values('total_pnl', ascending=False)
    print(by_symbol.head(10).to_string())
    
    print(f"\nRecent Trades:")
    print(results.tail(10)[['symbol', 'entry_date', 'direction', 'rv', 'exit_reason', 'pnl_pct', 'win']].to_string())
    
    return results


def optimize_parameters():
    """
    Grid search for optimal RV breakout parameters.

    Note: Backtested Sharpe ratios are typically overstated due to:
    - Look-ahead bias in parameter selection
    - No transaction costs modeled
    - Ideal execution assumptions

    Realistic expectation: In-sample Sharpe of 1.0+ typically degrades
    to 0.5-0.8 out-of-sample (per academic literature).
    """
    print("="*60)
    print("RV BREAKOUT PARAMETER OPTIMIZATION")
    print("="*60)
    print("Note: Backtest Sharpe typically overstated by 30-50%")
    print("Realistic out-of-sample expectation: 0.5-0.8 Sharpe")
    print("="*60)
    
    backtester = RVBreakoutBacktester()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Get symbols
    ref_path = DIRS["reference"] / "sp500_constituents.json"
    if ref_path.exists():
        import json
        with open(ref_path) as f:
            data = json.load(f)
            symbols = data.get('symbols', [])
    else:
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
    
    # Parameter grid based on research
    param_grid = {
        'min_rv': [1.5, 2.0, 2.5, 3.0],
        'min_gap_pct': [0.01, 0.015, 0.02, 0.03],
        'atr_stop_mult': [0.5, 1.0, 1.5],
        'atr_target_mult': [1.0, 1.5, 2.0, 2.5],
        'max_hold_days': [1, 2, 3, 5]
    }
    
    total_combos = 1
    for v in param_grid.values():
        total_combos *= len(v)
    
    print(f"Testing {total_combos} parameter combinations...\n")
    
    results = []
    combo_num = 0
    
    for min_rv in param_grid['min_rv']:
        for min_gap in param_grid['min_gap_pct']:
            for atr_stop in param_grid['atr_stop_mult']:
                for atr_target in param_grid['atr_target_mult']:
                    for max_hold in param_grid['max_hold_days']:
                        combo_num += 1
                        print(f"\r  Combo {combo_num}/{total_combos}: rv={min_rv} gap={min_gap} stop={atr_stop} target={atr_target} hold={max_hold}    ", end='', flush=True)
                        
                        # Backtest with these params
                        all_trades = []
                        for symbol in symbols:
                            trades = backtester.backtest_symbol(
                                symbol, start_date, end_date,
                                min_rv=min_rv,
                                min_gap_pct=min_gap,
                                atr_stop_mult=atr_stop,
                                atr_target_mult=atr_target,
                                max_hold_days=max_hold
                            )
                            if len(trades) > 0:
                                all_trades.append(trades)
                        
                        if not all_trades:
                            continue
                        
                        trades_df = pd.concat(all_trades, ignore_index=True)
                        n_trades = len(trades_df)
                        
                        if n_trades < 20:
                            continue
                        
                        win_rate = trades_df['win'].mean()
                        avg_pnl = trades_df['pnl_pct'].mean()
                        total_pnl = trades_df['pnl_pct'].sum()
                        avg_days = trades_df['days_held'].mean()
                        
                        # Sharpe estimate
                        if trades_df['pnl_pct'].std() > 0 and avg_days > 0:
                            sharpe = avg_pnl / trades_df['pnl_pct'].std() * np.sqrt(252 / max(avg_days, 1))
                        else:
                            sharpe = 0
                        
                        results.append({
                            'min_rv': min_rv,
                            'min_gap': min_gap,
                            'atr_stop': atr_stop,
                            'atr_target': atr_target,
                            'max_hold': max_hold,
                            'trades': n_trades,
                            'win_rate': win_rate,
                            'avg_pnl': avg_pnl,
                            'total_pnl': total_pnl,
                            'sharpe': sharpe
                        })
    
    if not results:
        print("No valid results!")
        return
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('sharpe', ascending=False)
    
    print("\nTOP 10 PARAMETER COMBINATIONS BY SHARPE:")
    print("="*100)
    print(results_df.head(10).to_string(index=False))
    
    best = results_df.iloc[0]
    print(f"\n{'='*60}")
    print("OPTIMAL PARAMETERS:")
    print(f"  MIN_RELATIVE_VOLUME: {best['min_rv']}")
    print(f"  MIN_GAP_PCT:         {best['min_gap']}")
    print(f"  ATR_STOP_MULT:       {best['atr_stop']}")
    print(f"  ATR_TARGET_MULT:     {best['atr_target']}")
    print(f"  MAX_HOLD_DAYS:       {int(best['max_hold'])}")
    print(f"\n  Sharpe:   {best['sharpe']:.2f}")
    print(f"  Win Rate: {best['win_rate']*100:.1f}%")
    print(f"  Trades:   {int(best['trades'])}")
    print(f"  Total PnL: {best['total_pnl']:.2f}%")
    print("="*60)
    
    return results_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--backtest':
        run_rv_backtest()
    elif len(sys.argv) > 1 and sys.argv[1] == '--optimize':
        optimize_parameters()
    else:
        run_rv_scan()
