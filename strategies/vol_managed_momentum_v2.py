"""
Volatility-Managed Momentum Strategy - RESEARCH-ALIGNED VERSION
================================================================
Tier 1 Core Strategy

Research basis: Barroso & Santa-Clara (2015) "Momentum has its moments"
Expected Sharpe: 1.7 (vs 0.9 for standard momentum)

Key insight from the paper:
"The STRATEGY's realized volatility, not individual stock volatility,
predicts momentum crashes. Scaling exposure inversely to the strategy's
recent volatility eliminates most of the left tail risk."

The critical distinction:
- WRONG: Weight each stock by inverse of its own volatility
- RIGHT: Scale the entire momentum portfolio by inverse of momentum's volatility

Implementation:
- Signal: 12-1 momentum (12-month return, skip most recent month)
- Portfolio sizing: Inversely proportional to STRATEGY's 21-day realized vol
- Target strategy volatility: 12% annualized (per paper)
- Rebalance: Monthly (first trading day)
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
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


class VolManagedMomentumV2(LongOnlyStrategy):
    """
    Volatility-Managed Momentum Strategy - Research-Aligned Version
    
    Key differences from V1:
    1. Uses 12-month formation (not 6)
    2. Scales PORTFOLIO exposure by strategy vol (not per-stock)
    3. Tracks strategy returns to calculate strategy volatility
    4. Target volatility of 12% (paper's recommendation)
    
    The intuition: Momentum crashes happen after momentum has been volatile.
    High recent strategy volatility → reduce exposure before the crash.
    """
    
    def __init__(self):
        super().__init__("vol_managed_momentum")  # Use same name as V1 for config compatibility

        # ==========================================================================
        # MOMENTUM SIGNAL (Research: Barroso & Santa-Clara 2015)
        # ==========================================================================
        # Standard academic momentum: 12-month return, skip most recent month
        # The skip period avoids short-term reversal effect (Jegadeesh 1990)
        self.formation_period = 252   # 12 months (research standard - NOT 6!)
        self.skip_period = 21         # Skip most recent month (avoid reversal)
        self.holding_period = 1       # Monthly rebalance (reduces turnover)

        # ==========================================================================
        # VOLATILITY MANAGEMENT - THE KEY INNOVATION
        # ==========================================================================
        # Critical insight from Barroso & Santa-Clara:
        # "Scale STRATEGY exposure by inverse of STRATEGY's recent volatility"
        # This is NOT the same as weighting stocks by their individual vol
        #
        # Why it works: Momentum crashes are preceded by high strategy vol
        # By scaling down when vol is high, we avoid the left tail
        #
        # Paper's parameters:
        # - 6-month vol lookback (126 days) - captures regime changes
        # - 12% target vol - standard institutional target
        # - Scale bounds: 0.1x to 2.0x - avoid extreme positions
        self.strategy_vol_lookback = 126  # 6 months of daily strategy returns
        self.target_strategy_vol = 0.12   # 12% target (paper recommendation)
        self.min_scale = 0.20             # Minimum 20% exposure (was 10%, too conservative)
        self.max_scale = 1.50             # Maximum 150% exposure (was 200%, reduced for safety)

        # ==========================================================================
        # PORTFOLIO CONSTRUCTION (Research-Optimized)
        # ==========================================================================
        # Academic standard: Top decile (10%) by momentum
        # Equal-weighting within winners outperforms cap-weighting (Asness 2014)
        self.top_percentile = 0.10    # Top decile (10%) winners (research standard)
        self.bottom_percentile = 0.10 # Bottom decile losers (for tracking, not shorting)
        self.equal_weight = True      # Equal-weight within winners (research standard)

        # ==========================================================================
        # RISK CONTROLS (Conservative for Real Trading)
        # ==========================================================================
        # Individual position limits prevent concentration risk
        # VIX reduction provides additional crash protection beyond vol scaling
        self.max_single_position = 0.10  # Max 10% in single stock (was 15%, too concentrated)
        self.vix_reduction = 0.50        # 50% reduction if VIX > 30

        # State tracking
        self.last_rebalance_month = None
        self.strategy_returns_history = []  # Track strategy returns for vol calc
        self.last_portfolio_value = None
    
    def _get_current_date(self, data: Dict[str, pd.DataFrame]) -> datetime:
        """Extract current date from data."""
        for symbol, df in data.items():
            if len(df) > 0:
                # Check attrs first (set by backtester)
                if hasattr(df, 'attrs') and 'backtest_date' in df.attrs:
                    bd = df.attrs['backtest_date']
                    if isinstance(bd, pd.Timestamp):
                        return bd.to_pydatetime()
                    elif isinstance(bd, datetime):
                        return bd
                
                # Check index
                idx = df.index[-1]
                if isinstance(idx, pd.Timestamp):
                    return idx.to_pydatetime()
                elif isinstance(idx, datetime):
                    return idx
                
                # Check timestamp column
                if 'timestamp' in df.columns:
                    ts = df['timestamp'].iloc[-1]
                    if isinstance(ts, pd.Timestamp):
                        return ts.to_pydatetime()
                    elif isinstance(ts, datetime):
                        return ts
        
        logger.warning("Could not determine date from data")
        return datetime.now()
    
    def _is_rebalance_day(self, current_date: datetime) -> bool:
        """Check if today is a rebalance day (first call of new month)."""
        current_month = (current_date.year, current_date.month)
        
        if self.last_rebalance_month is None:
            return True
        
        return current_month != self.last_rebalance_month
    
    def calculate_momentum(self, df: pd.DataFrame) -> Optional[float]:
        """
        Calculate 12-1 momentum (12-month return, skip most recent month).
        
        This is the standard momentum signal from the academic literature.
        The skip period avoids the short-term reversal effect.
        """
        required_bars = self.formation_period + self.skip_period
        if len(df) < required_bars:
            return None
        
        try:
            # Price at t - skip_period (end of momentum window)
            recent_price = df['close'].iloc[-(self.skip_period + 1)]
            # Price at t - formation_period - skip_period (start of momentum window)
            old_price = df['close'].iloc[-(required_bars)]
            
            if old_price <= 0:
                return None
            
            return (recent_price - old_price) / old_price
        except (IndexError, KeyError):
            return None
    
    def calculate_strategy_volatility(self) -> float:
        """
        Calculate the momentum STRATEGY's realized volatility.
        
        This is the key insight from Barroso & Santa-Clara:
        We scale by the strategy's vol, not individual stock vol.
        """
        if len(self.strategy_returns_history) < 21:
            # Not enough history - use default
            return self.target_strategy_vol
        
        # Use last 6 months of daily returns
        recent_returns = self.strategy_returns_history[-self.strategy_vol_lookback:]
        
        # Annualized volatility
        vol = np.std(recent_returns) * np.sqrt(252)
        
        # Floor at 5% to avoid division issues
        return max(vol, 0.05)
    
    def calculate_vol_scale(self, vix_regime: str = None) -> float:
        """
        Calculate portfolio scaling factor based on strategy volatility.
        
        scale = target_vol / realized_strategy_vol
        
        When strategy vol is high (before crashes): scale < 1 (reduce exposure)
        When strategy vol is low: scale > 1 (increase exposure)
        """
        strategy_vol = self.calculate_strategy_volatility()
        
        # Inverse volatility scaling
        scale = self.target_strategy_vol / strategy_vol
        
        # Apply bounds
        scale = max(self.min_scale, min(self.max_scale, scale))
        
        # Additional reduction in high VIX
        if vix_regime in ['high', 'extreme']:
            scale *= self.vix_reduction
            logger.debug(f"High VIX regime: reducing scale by {self.vix_reduction:.0%}")
        
        return scale
    
    def update_strategy_return(self, daily_return: float):
        """Track daily strategy returns for volatility calculation."""
        self.strategy_returns_history.append(daily_return)
        
        # Keep only what we need
        max_history = self.strategy_vol_lookback * 2
        if len(self.strategy_returns_history) > max_history:
            self.strategy_returns_history = self.strategy_returns_history[-max_history:]
    
    def generate_signals(self,
                         data: Dict[str, pd.DataFrame],
                         current_positions: List[str] = None,
                         vix_regime: str = None) -> List[Signal]:
        """
        Generate momentum signals with volatility-managed sizing.
        
        Process:
        1. Calculate 12-1 momentum for all stocks
        2. Select top decile (winners)
        3. Calculate strategy volatility scaling
        4. Equal-weight winners, scaled by vol factor
        """
        signals = []
        current_positions = current_positions or []
        
        current_date = self._get_current_date(data)
        
        # Reset state if backtest restarted
        if self.last_rebalance_month is not None:
            last_year, last_month = self.last_rebalance_month
            if (current_date.year, current_date.month) < (last_year, last_month):
                self.last_rebalance_month = None
                self.strategy_returns_history = []
        
        # Only rebalance monthly
        if not self._is_rebalance_day(current_date):
            return signals
        
        self.last_rebalance_month = (current_date.year, current_date.month)
        
        # Calculate momentum for all stocks
        momentum_scores = {}
        prices = {}
        
        for symbol, df in data.items():
            if not self.filter_by_liquidity(df):
                continue
            
            momentum = self.calculate_momentum(df)
            if momentum is None:
                continue
            
            momentum_scores[symbol] = momentum
            prices[symbol] = df['close'].iloc[-1]
        
        if len(momentum_scores) < 10:
            logger.warning(f"Only {len(momentum_scores)} stocks with valid momentum")
            return signals
        
        # Rank stocks by momentum
        ranked = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select winners (top decile)
        n_winners = max(int(len(ranked) * self.top_percentile), 5)
        winners = [symbol for symbol, _ in ranked[:n_winners]]
        
        # Calculate volatility scaling
        vol_scale = self.calculate_vol_scale(vix_regime)
        strategy_vol = self.calculate_strategy_volatility()
        
        logger.debug(f"Monthly rebalance: {current_date.strftime('%Y-%m')}")
        logger.info(f"  Universe: {len(momentum_scores)} stocks")
        logger.info(f"  Winners: {n_winners} stocks (top {self.top_percentile:.0%})")
        logger.info(f"  Strategy vol: {strategy_vol:.1%} → scale: {vol_scale:.2f}x")
        
        # Equal weight within winners, scaled by vol factor
        base_weight = 1.0 / n_winners
        scaled_weight = base_weight * vol_scale
        
        # Cap individual position size
        position_weight = min(scaled_weight, self.max_single_position)
        
        # Generate BUY signals for winners
        for symbol in winners:
            if symbol in current_positions:
                continue
            
            if symbol not in data:
                continue
            
            df = data[symbol]
            price = prices[symbol]
            mom = momentum_scores[symbol]
            
            # ATR for stops - momentum is a monthly strategy, use wider stops
            atr = df['atr'].iloc[-1] if 'atr' in df.columns else price * 0.02

            # Strength based on momentum rank (top-ranked = stronger)
            rank = winners.index(symbol)
            strength = 1.0 - (rank / n_winners) * 0.3

            # Stop loss and target optimized for monthly momentum:
            # - Wider stops (3x ATR) to avoid getting stopped out on daily noise
            # - Trailing stop implicitly via monthly rebalance (losers get sold)
            # - Target is 5x ATR (good risk/reward for momentum)
            # - Emergency 20% stop in should_close_position handles catastrophic losses
            signals.append(Signal(
                timestamp=current_date,
                symbol=symbol,
                strategy=self.name,
                signal_type=SignalType.BUY,
                strength=strength,
                price=price,
                stop_loss=price - 3.0 * atr,   # Wider stop (was 2.5x) - momentum needs room
                target_price=price + 5 * atr,  # Better R:R ratio (was 4x)
                position_size_pct=position_weight,
                reason=f"12-1 mom: {mom:.1%}, rank: {rank+1}/{n_winners}, scale: {vol_scale:.2f}x",
                metadata={
                    'momentum': mom,
                    'rank': rank + 1,
                    'vol_scale': vol_scale,
                    'strategy_vol': strategy_vol,
                    'atr_multiple_stop': 3.0,
                    'atr_multiple_target': 5.0
                }
            ))
        
        # Generate CLOSE signals for positions dropped from winners
        for symbol in current_positions:
            if symbol not in winners and symbol in data:
                price = prices.get(symbol, data[symbol]['close'].iloc[-1])
                mom = momentum_scores.get(symbol, 0)
                
                signals.append(Signal(
                    timestamp=current_date,
                    symbol=symbol,
                    strategy=self.name,
                    signal_type=SignalType.CLOSE,
                    strength=0.8,
                    price=price,
                    reason=f"Dropped from winners (momentum: {mom:.1%})"
                ))
        
        logger.debug(f"  Signals: {len([s for s in signals if s.signal_type == SignalType.BUY])} BUY, "
                   f"{len([s for s in signals if s.signal_type == SignalType.CLOSE])} CLOSE")
        
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
        """Emergency stop loss check."""
        
        # Base class stop/target check
        base_signal = super().should_close_position(
            symbol, current_price, entry_price, stop_loss,
            target_price, peak_price, entry_time, data
        )
        if base_signal:
            return base_signal
        
        # Emergency 25% stop (momentum can be volatile)
        pnl_pct = (current_price - entry_price) / entry_price
        if pnl_pct < -0.25:
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


def create_strategy() -> VolManagedMomentumV2:
    """Factory function for strategy creation."""
    return VolManagedMomentumV2()


def compare_v1_vs_v2():
    """Compare V1 (current) vs V2 (research-aligned) implementations."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.timezone import normalize_dataframe, normalize_timestamp, normalize_index
    
    from data.cached_data_manager import CachedDataManager
    from research.backtester import Backtester
    from strategies.vol_managed_momentum import VolManagedMomentumStrategy
    from config import DIRS
    
    print("="*70)
    print("COMPARING VOL-MANAGED MOMENTUM: V1 vs V2 (Research-Aligned)")
    print("="*70)
    
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
        vix_data = normalize_dataframe(vix_data)
        vix_data['regime'] = 'normal'
        vix_data.loc[vix_data['close'] < 15, 'regime'] = 'low'
        vix_data.loc[vix_data['close'] > 25, 'regime'] = 'high'
    
    # Prepare data - top 200 by liquidity for better momentum universe
    metadata = data_mgr.get_all_metadata()
    sorted_symbols = sorted(metadata.items(), key=lambda x: x[1].get('dollar_volume', 0), reverse=True)[:200]
    
    data = {}
    for symbol, _ in sorted_symbols:
        df = data_mgr.get_bars(symbol)
        if df is not None and len(df) >= 300:
            if 'timestamp' in df.columns:
                df = df.copy()
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            df = normalize_dataframe(df)
            data[symbol] = df
    
    print(f"\nUniverse: {len(data)} stocks")
    
    # Test V1
    print("\n" + "-"*50)
    print("V1: Current Implementation (6-month formation)")
    print("-"*50)
    
    v1_strategy = VolManagedMomentumStrategy()
    v1_strategy.last_rebalance_month = None
    
    backtester = Backtester(initial_capital=100000)
    v1_result = backtester.run(v1_strategy, data.copy(), vix_data=vix_data)
    
    print(f"  Period: {v1_result.start_date} to {v1_result.end_date}")
    print(f"  Trades: {v1_result.total_trades}")
    print(f"  Sharpe: {v1_result.sharpe_ratio:.2f}")
    print(f"  Annual Return: {v1_result.annual_return:.1f}%")
    print(f"  Max Drawdown: {v1_result.max_drawdown_pct:.1f}%")
    print(f"  Win Rate: {v1_result.win_rate:.1f}%")
    
    # Test V2
    print("\n" + "-"*50)
    print("V2: Research-Aligned (12-month formation, strategy vol scaling)")
    print("-"*50)
    
    v2_strategy = VolManagedMomentumV2()
    v2_strategy.last_rebalance_month = None
    
    backtester = Backtester(initial_capital=100000)
    v2_result = backtester.run(v2_strategy, data.copy(), vix_data=vix_data)
    
    print(f"  Period: {v2_result.start_date} to {v2_result.end_date}")
    print(f"  Trades: {v2_result.total_trades}")
    print(f"  Sharpe: {v2_result.sharpe_ratio:.2f}")
    print(f"  Annual Return: {v2_result.annual_return:.1f}%")
    print(f"  Max Drawdown: {v2_result.max_drawdown_pct:.1f}%")
    print(f"  Win Rate: {v2_result.win_rate:.1f}%")
    
    # Comparison
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Metric':<20} {'V1':>12} {'V2':>12} {'Diff':>12}")
    print("-"*56)
    print(f"{'Sharpe Ratio':<20} {v1_result.sharpe_ratio:>12.2f} {v2_result.sharpe_ratio:>12.2f} {v2_result.sharpe_ratio - v1_result.sharpe_ratio:>+12.2f}")
    print(f"{'Annual Return %':<20} {v1_result.annual_return:>12.1f} {v2_result.annual_return:>12.1f} {v2_result.annual_return - v1_result.annual_return:>+12.1f}")
    print(f"{'Max Drawdown %':<20} {v1_result.max_drawdown_pct:>12.1f} {v2_result.max_drawdown_pct:>12.1f} {v2_result.max_drawdown_pct - v1_result.max_drawdown_pct:>+12.1f}")
    print(f"{'Win Rate %':<20} {v1_result.win_rate:>12.1f} {v2_result.win_rate:>12.1f} {v2_result.win_rate - v1_result.win_rate:>+12.1f}")
    print(f"{'Total Trades':<20} {v1_result.total_trades:>12} {v2_result.total_trades:>12} {v2_result.total_trades - v1_result.total_trades:>+12}")
    print("="*70)
    
    print(f"\nResearch Target Sharpe: 1.7")
    print(f"V1 vs Target: {v1_result.sharpe_ratio / 1.7 * 100:.0f}%")
    print(f"V2 vs Target: {v2_result.sharpe_ratio / 1.7 * 100:.0f}%")
    
    return v1_result, v2_result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    compare_v1_vs_v2()
