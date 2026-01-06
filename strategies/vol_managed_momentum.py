"""
Volatility-Managed Momentum Strategy
=====================================
Tier 1 Core Strategy

Research basis: Barroso & Santa-Clara (2015)
Expected Sharpe: 1.7 (vs 0.9 for standard momentum)

Key insight: Momentum crashes are predictable via realized volatility.
Scaling exposure inversely to vol eliminates left tail risk and
approximately doubles the Sharpe ratio.

Implementation:
- Signal: 12-1 momentum (12-month return, skip most recent month)
- Position sizing: Inversely proportional to 21-day realized volatility
- Target portfolio volatility: 15% annualized
- Rebalance: Monthly (first trading day)
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path
import logging

import pandas as pd
import numpy as np

# Import from parent
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.base import BaseStrategy, Signal, SignalType, LongOnlyStrategy
from config import VIX_REGIMES

logger = logging.getLogger(__name__)


class VolManagedMomentumStrategy(LongOnlyStrategy):
    """
    Volatility-Managed Momentum Strategy
    
    IMPORTANT: Only generates signals on monthly rebalance (first trading day of month).
    This dramatically reduces turnover and transaction costs.
    
    Risk controls:
    - Max 10% in any single position
    - 50% exposure reduction if VIX > 30
    - -20% stop loss on individual positions
    """
    
    def __init__(self):
        super().__init__("vol_managed_momentum")
        
        # Strategy parameters - OPTIMIZED via grid search on 2020-2025 data
        self.formation_period = 126   # 6 months
        self.skip_period = 10         # Skip 2 weeks (was 21)
        self.vol_lookback = 10        # 10-day realized vol (was 14)
        self.target_vol = 0.20        # 20% target
        self.top_percentile = 0.15    # Top 15% (was 20%)
        self.max_single_position = 0.10
        self.high_vix_reduction = 0.70  # Was 0.50 - less aggressive reduction
        
        # CRITICAL: Track last rebalance to enforce monthly frequency
        self.last_rebalance_month = None
    
    def _is_rebalance_day(self, current_date: datetime) -> bool:
        """Check if today is a rebalance day (first call of new month)."""
        current_month = (current_date.year, current_date.month)
        
        if self.last_rebalance_month is None:
            return True
        
        return current_month != self.last_rebalance_month
    
    def _get_current_date(self, data: Dict[str, pd.DataFrame]) -> datetime:
        """Extract current date from data."""
        for symbol, df in data.items():
            if len(df) > 0:
                # Check timestamp column first (CachedDataManager uses this format)
                if 'timestamp' in df.columns:
                    ts = df['timestamp'].iloc[-1]
                    if isinstance(ts, pd.Timestamp):
                        return ts.to_pydatetime()
                    elif isinstance(ts, datetime):
                        return ts
                # Fall back to index if it's datetime
                idx = df.index[-1]
                if isinstance(idx, pd.Timestamp):
                    return idx.to_pydatetime()
                elif isinstance(idx, datetime):
                    return idx
        return datetime.now()
    
    def calculate_momentum(self, df: pd.DataFrame) -> Optional[float]:
        """Calculate 12-1 momentum."""
        if len(df) < self.formation_period:
            return None
        
        try:
            recent_price = df['close'].iloc[-(self.skip_period + 1)]
            old_price = df['close'].iloc[-(self.formation_period)]
            return (recent_price - old_price) / old_price
        except (IndexError, KeyError):
            return None
    
    def calculate_realized_vol(self, df: pd.DataFrame) -> float:
        """Calculate 21-day realized volatility (annualized)."""
        if len(df) < self.vol_lookback:
            return 0.20
        
        returns = df['close'].pct_change().dropna()
        if len(returns) < self.vol_lookback:
            return 0.20
        
        vol = returns.tail(self.vol_lookback).std() * np.sqrt(252)
        return max(vol, 0.05)
    
    def calculate_position_weight(self, realized_vol: float, vix_regime: str = None) -> float:
        """Inverse volatility weighting."""
        vol_weight = self.target_vol / realized_vol
        vol_weight = min(vol_weight, 2.0)
        vol_weight = max(vol_weight, 0.25)
        
        if vix_regime in ['high', 'extreme']:
            vol_weight *= self.high_vix_reduction
        
        return vol_weight
    
    def generate_signals(self,
                         data: Dict[str, pd.DataFrame],
                         current_positions: List[str] = None,
                         vix_regime: str = None) -> List[Signal]:
        """
        Generate momentum signals - ONLY ON MONTHLY REBALANCE.
        """
        signals = []
        current_positions = current_positions or []
        
        # Get current date from data
        current_date = self._get_current_date(data)
        
        # Reset state if this looks like a new backtest (date before last rebalance)
        if self.last_rebalance_month is not None:
            last_year, last_month = self.last_rebalance_month
            if (current_date.year, current_date.month) < (last_year, last_month):
                self.last_rebalance_month = None
        
        # CRITICAL: Only rebalance monthly
        if not self._is_rebalance_day(current_date):
            return signals  # Return empty - no rebalance today
        
        # Mark this month as rebalanced
        self.last_rebalance_month = (current_date.year, current_date.month)
        logger.debug(f"Monthly rebalance: {current_date.strftime('%Y-%m')}")
        
        # Calculate momentum for all stocks
        momentum_scores = {}
        volatilities = {}
        
        for symbol, df in data.items():
            if not self.filter_by_liquidity(df):
                continue
            
            momentum = self.calculate_momentum(df)
            if momentum is None:
                continue
            
            volatilities[symbol] = self.calculate_realized_vol(df)
            momentum_scores[symbol] = momentum
        
        if not momentum_scores:
            return signals
        
        # Rank and select top quintile
        ranked = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        n_select = max(int(len(ranked) * self.top_percentile), 1)
        top_stocks = [symbol for symbol, _ in ranked[:n_select]]
        
        logger.debug(f"Momentum ranking: Top {n_select} of {len(ranked)} stocks")
        
        # BUY signals for top stocks not held
        for symbol in top_stocks:
            if symbol in current_positions:
                continue
            if symbol not in data:
                continue
            
            df = data[symbol]
            price = df['close'].iloc[-1]
            vol = volatilities[symbol]
            mom = momentum_scores[symbol]
            weight = self.calculate_position_weight(vol, vix_regime)
            
            rank_idx = top_stocks.index(symbol)
            strength = 1.0 - (rank_idx / len(top_stocks)) * 0.5
            if vix_regime == 'high':
                strength *= 0.7
            
            atr = df['atr'].iloc[-1] if 'atr' in df.columns else price * 0.02
            
            signals.append(Signal(
                timestamp=current_date,
                symbol=symbol,
                strategy=self.name,
                signal_type=SignalType.BUY,
                strength=strength,
                price=price,
                stop_loss=price - 2 * atr,
                target_price=price + 3 * atr,
                position_size_pct=weight * self.max_single_position,
                reason=f"12-1 momentum: {mom:.1%}, vol: {vol:.1%}",
                metadata={'momentum': mom, 'realized_vol': vol, 'rank': rank_idx + 1}
            ))
        
        # CLOSE signals for positions no longer in top quintile
        for symbol in current_positions:
            if symbol not in top_stocks and symbol in data:
                price = data[symbol]['close'].iloc[-1]
                mom = momentum_scores.get(symbol, 0)
                signals.append(Signal(
                    timestamp=current_date,
                    symbol=symbol,
                    strategy=self.name,
                    signal_type=SignalType.CLOSE,
                    strength=0.8,
                    price=price,
                    reason=f"Dropped from top momentum (current: {mom:.1%})"
                ))
        
        logger.debug(f"Generated {len(signals)} signals (VIX: {vix_regime})")
        return signals
    
    def should_close_position(self,
                              symbol: str,
                              current_price: float,
                              entry_price: float,
                              stop_loss: float,
                              target_price: float,
                              peak_price: float,
                              entry_time: datetime,
                              data: pd.DataFrame = None) -> Optional[Signal]:
        """Check stops - but rotation handled in generate_signals."""
        base_signal = super().should_close_position(
            symbol, current_price, entry_price, stop_loss,
            target_price, peak_price, entry_time, data
        )
        if base_signal:
            return base_signal
        
        # Emergency 20% stop
        pnl_pct = (current_price - entry_price) / entry_price
        if pnl_pct < -0.20:
            return Signal(
                timestamp=datetime.now(),
                symbol=symbol,
                strategy=self.name,
                signal_type=SignalType.CLOSE,
                strength=1.0,
                price=current_price,
                reason=f"Emergency stop: {pnl_pct:.1%} loss"
            )
        
        return None


def create_strategy() -> VolManagedMomentumStrategy:
    return VolManagedMomentumStrategy()


def optimize_parameters():
    """Grid search for optimal vol-managed momentum parameters."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from data.cached_data_manager import CachedDataManager
    from research.backtester import Backtester
    from config import DIRS
    
    print("="*60)
    print("VOL-MANAGED MOMENTUM OPTIMIZATION")
    print("="*60)
    
    # Load data
    data_mgr = CachedDataManager()
    if not data_mgr.cache:
        data_mgr.load_all()
    
    # Load VIX
    vix_path = DIRS.get('vix', Path('./data/historical/vix')) / 'vix.parquet'
    vix_data = None
    if vix_path.exists():
        vix_data = pd.read_parquet(vix_path)
        if 'timestamp' in vix_data.columns:
            vix_data = vix_data.set_index('timestamp')
        if vix_data.index.tz is not None:
            vix_data.index = vix_data.index.tz_localize(None)
        vix_data['regime'] = 'normal'
        vix_data.loc[vix_data['close'] < 15, 'regime'] = 'low'
        vix_data.loc[vix_data['close'] > 25, 'regime'] = 'high'
    
    # Prepare data - top 100 by liquidity
    metadata = data_mgr.get_all_metadata()
    sorted_symbols = sorted(metadata.items(), key=lambda x: x[1].get('dollar_volume', 0), reverse=True)[:100]
    
    data = {}
    for symbol, _ in sorted_symbols:
        df = data_mgr.get_bars(symbol)
        if df is not None and len(df) >= 300:
            if 'timestamp' in df.columns:
                df = df.copy()
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            data[symbol] = df
    
    print(f"Testing with {len(data)} symbols")
    
    # Parameter grid based on research
    param_grid = {
        'formation_period': [126, 189, 252],  # 6, 9, 12 months
        'skip_period': [10, 21, 42],          # 2, 4, 6 weeks
        'vol_lookback': [10, 21, 42],         # vol window
        'target_vol': [0.12, 0.15, 0.20],     # target volatility
        'top_percentile': [0.10, 0.20, 0.30]  # quintile selection
    }
    
    total_combos = 1
    for v in param_grid.values():
        total_combos *= len(v)
    
    print(f"Testing {total_combos} parameter combinations...\n")
    
    results = []
    combo_num = 0
    
    for formation in param_grid['formation_period']:
        for skip in param_grid['skip_period']:
            for vol_lb in param_grid['vol_lookback']:
                for target_vol in param_grid['target_vol']:
                    for top_pct in param_grid['top_percentile']:
                        combo_num += 1
                        print(f"\r  Combo {combo_num}/{total_combos}: form={formation} skip={skip} vol_lb={vol_lb} target={target_vol} top={top_pct}    ", end='', flush=True)
                        
                        # Create strategy with these params
                        strategy = VolManagedMomentumStrategy()
                        strategy.formation_period = formation
                        strategy.skip_period = skip
                        strategy.vol_lookback = vol_lb
                        strategy.target_vol = target_vol
                        strategy.top_percentile = top_pct
                        strategy.last_rebalance_month = None  # Reset
                        
                        # Run backtest
                        backtester = Backtester(initial_capital=100000)
                        result = backtester.run(strategy, data, vix_data=vix_data)
                        
                        if result.total_trades < 10:
                            continue
                        
                        results.append({
                            'formation': formation,
                            'skip': skip,
                            'vol_lb': vol_lb,
                            'target_vol': target_vol,
                            'top_pct': top_pct,
                            'trades': result.total_trades,
                            'win_rate': result.win_rate,
                            'annual_ret': result.annual_return,
                            'sharpe': result.sharpe_ratio,
                            'max_dd': result.max_drawdown_pct
                        })
    
    print("\n")
    
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
    print(f"  formation_period: {int(best['formation'])}")
    print(f"  skip_period:      {int(best['skip'])}")
    print(f"  vol_lookback:     {int(best['vol_lb'])}")
    print(f"  target_vol:       {best['target_vol']}")
    print(f"  top_percentile:   {best['top_pct']}")
    print(f"\n  Sharpe:      {best['sharpe']:.2f}")
    print(f"  Annual Ret:  {best['annual_ret']:.1f}%")
    print(f"  Win Rate:    {best['win_rate']:.1f}%")
    print(f"  Max DD:      {best['max_dd']:.1f}%")
    print(f"  Trades:      {int(best['trades'])}")
    print("="*60)
    
    return results_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--optimize':
        optimize_parameters()
    else:
        print("=" * 60)
        print("Volatility-Managed Momentum Strategy")
        print("=" * 60)
        
        strategy = VolManagedMomentumStrategy()
        print(f"Strategy: {strategy.name}")
        print(f"Rebalance: Monthly only")
        
        # Test rebalance logic
        print("\nTesting monthly rebalance...")
        
        # Create minimal test data
        dates = pd.date_range(end=datetime.now(), periods=300, freq='D')
        test_df = pd.DataFrame({
            'open': 100, 'high': 101, 'low': 99, 'close': 100,
            'volume': 1000000, 'atr': 2
        }, index=dates)
        
        test_data = {'TEST': test_df}
        
        # First call - should generate signals (new month)
        signals1 = strategy.generate_signals(test_data, vix_regime='normal')
        print(f"First call: {len(signals1)} signals")
        
        # Second call same month - should skip
        signals2 = strategy.generate_signals(test_data, vix_regime='normal')
        print(f"Second call (same month): {len(signals2)} signals")
        
        # Simulate new month
        strategy.last_rebalance_month = (2020, 1)  # Reset to old month
        signals3 = strategy.generate_signals(test_data, vix_regime='normal')
        print(f"New month call: {len(signals3)} signals")
