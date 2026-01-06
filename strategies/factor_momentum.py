"""
Factor Momentum Strategy
========================
Tier 1 Core Strategy

Research basis: Ehsani & Linnainmaa (2022)
- Factor momentum explains most of stock momentum
- Annual Sharpe: 0.84 with t-stats > 5.55
- CRITICAL: Does NOT crash like price momentum

Key insight: Stock momentum is largely driven by persistence in factor returns,
not idiosyncratic performance. Factor exposures mean-revert more gradually than
individual stock prices during market reversals, avoiding catastrophic crashes.

Historical momentum crashes avoided:
- 1932: -91% (price momentum)
- 2009: -73% (price momentum)  
- Factor momentum: No comparable crashes

Implementation:
- Calculate momentum of factor portfolios (Value, Size, Quality, Low-Vol)
- Go long factors with positive 12-month momentum
- Inverse volatility weighting across factors
- Monthly rebalancing

Uses sector ETFs as factor proxies:
- Value: XLF (Financials), XLE (Energy) - high book-to-market sectors
- Growth: XLK (Tech), XLC (Comms) - low book-to-market sectors  
- Defensive: XLU (Utilities), XLP (Staples) - low beta
- Cyclical: XLI (Industrials), XLY (Consumer Disc) - high beta
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import logging

import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.base import BaseStrategy, Signal, SignalType, LongOnlyStrategy
from config import VIX_REGIMES

logger = logging.getLogger(__name__)


@dataclass
class FactorMomentum:
    """Momentum statistics for a factor."""
    name: str
    etf: str
    momentum_12m: float  # 12-month return
    momentum_6m: float   # 6-month return
    momentum_1m: float   # 1-month return (for timing)
    volatility: float    # Realized volatility
    sharpe: float        # Risk-adjusted momentum
    signal: float        # Final signal strength
    weight: float = 0.0  # Portfolio weight


# Factor definitions using sector ETFs
# Research shows sector ETFs capture ~70% of factor exposure
FACTOR_ETFS = {
    # Value factors (high book-to-market, low growth expectations)
    'value': {
        'etfs': ['XLF', 'XLE', 'XLB'],  # Financials, Energy, Materials
        'description': 'High book-to-market sectors'
    },
    # Growth factors (low book-to-market, high growth expectations)
    'growth': {
        'etfs': ['XLK', 'XLC'],  # Tech, Communications
        'description': 'Low book-to-market, high growth sectors'
    },
    # Defensive factors (low beta, stable earnings)
    'defensive': {
        'etfs': ['XLU', 'XLP', 'XLV'],  # Utilities, Staples, Healthcare
        'description': 'Low beta, stable cash flows'
    },
    # Cyclical factors (high beta, economic sensitivity)
    'cyclical': {
        'etfs': ['XLI', 'XLY'],  # Industrials, Consumer Discretionary
        'description': 'High beta, economically sensitive'
    },
    # Size factor (small vs large cap)
    'size': {
        'etfs': ['IWM'],  # Russell 2000 as small-cap proxy
        'description': 'Small-cap premium',
        'benchmark': 'SPY'  # Compare to large-cap
    },
    # Quality factor (profitability)
    'quality': {
        'etfs': ['XLV', 'XLP'],  # Healthcare, Staples (high margins)
        'description': 'High profitability sectors'
    },
}

# All unique ETFs needed
ALL_FACTOR_ETFS = list(set(
    etf 
    for factor in FACTOR_ETFS.values() 
    for etf in factor['etfs']
)) + ['SPY']  # Need SPY as benchmark


class FactorMomentumStrategy(LongOnlyStrategy):
    """
    Factor Momentum Strategy

    Instead of buying stocks with high momentum (which crashes),
    this strategy buys FACTORS with high momentum.

    Research basis: Ehsani & Linnainmaa (2022)
    - Factor momentum explains most of stock momentum
    - Annual Sharpe: 0.84 with t-stats > 5.55
    - CRITICAL: Does NOT crash like price momentum

    Why it works:
    1. Factor returns are persistent (winners keep winning)
    2. Factor mean-reversion is gradual (no sudden crashes)
    3. Diversification across factors reduces idiosyncratic risk

    Risk controls:
    - Inverse volatility weighting (lower vol = higher weight)
    - Max 35% in any single factor (conservative)
    - Skip factors with negative 12-month momentum
    - Reduce exposure when all factors show negative momentum
    """

    # ==========================================================================
    # MOMENTUM CALCULATION (Research: Ehsani & Linnainmaa 2022)
    # ==========================================================================
    # The paper uses 12-month momentum for factors (standard in literature)
    # Shorter periods (6-month, 1-month) are used for timing/confirmation
    # Skip period avoids microstructure noise from recent trading
    FORMATION_PERIOD_LONG = 252   # 12 months (primary signal)
    FORMATION_PERIOD_MED = 126    # 6 months (confirmation)
    FORMATION_PERIOD_SHORT = 21   # 1 month (timing - avoid selling into weakness)
    SKIP_PERIOD = 5               # 1 week skip (was 21 - reduced for faster response)
                                  # Factors are less noisy than stocks

    # ==========================================================================
    # VOLATILITY CALCULATION (Risk-Parity Style)
    # ==========================================================================
    # 3-month vol lookback captures regime changes without being too reactive
    VOL_LOOKBACK = 63  # 3 months for realized vol

    # ==========================================================================
    # POSITION LIMITS (Conservative for Factor Concentration)
    # ==========================================================================
    # Factors are already diversified, but we still limit concentration
    MAX_FACTOR_WEIGHT = 0.35  # Max 35% in one factor (was 40% - more conservative)
    MIN_FACTOR_WEIGHT = 0.08  # Min 8% position (was 5% - avoid tiny positions)

    # ==========================================================================
    # SIGNAL THRESHOLDS
    # ==========================================================================
    # Only go long factors with positive momentum
    # Research shows factor momentum is about AVOIDING losers, not just buying winners
    MIN_MOMENTUM_THRESHOLD = 0.0  # Only long positive momentum factors

    # ==========================================================================
    # VIX ADJUSTMENTS (Regime-Based Risk Management)
    # ==========================================================================
    # Factor momentum is more stable than stock momentum, but still reduce in crisis
    # Thresholds from central config (VIX_REGIMES)
    HIGH_VIX_THRESHOLD = VIX_REGIMES['normal']      # 25 - normal/high boundary
    EXTREME_VIX_THRESHOLD = VIX_REGIMES['high']     # 35 - high/extreme boundary
    HIGH_VIX_REDUCTION = 0.75     # 75% exposure if VIX > normal threshold (was 70%)
    EXTREME_VIX_REDUCTION = 0.50  # 50% exposure if VIX > high threshold (was 40%)
    
    def __init__(self):
        super().__init__("factor_momentum")
        
        # Track rebalancing
        self.last_rebalance_month = None
        
        # Factor weights
        self._factor_weights: Dict[str, float] = {}
        
        # Performance tracking
        self._factor_momentum_history: List[Dict] = []
    
    def _is_rebalance_day(self, current_date: datetime) -> bool:
        """Check if today is a rebalance day (first trading day of month)."""
        current_month = (current_date.year, current_date.month)
        
        if self.last_rebalance_month is None:
            return True
        
        return current_month != self.last_rebalance_month
    
    def _calculate_factor_momentum(
        self,
        factor_name: str,
        etf_data: Dict[str, pd.DataFrame]
    ) -> Optional[FactorMomentum]:
        """
        Calculate momentum for a factor using its ETF proxies.
        
        For factors with multiple ETFs, we equal-weight the momentum signals.
        """
        factor_config = FACTOR_ETFS.get(factor_name)
        if not factor_config:
            return None
        
        etfs = factor_config['etfs']
        
        momentums_12m = []
        momentums_6m = []
        momentums_1m = []
        volatilities = []
        
        for etf in etfs:
            if etf not in etf_data:
                continue
            
            df = etf_data[etf]
            if len(df) < self.FORMATION_PERIOD_LONG + self.SKIP_PERIOD:
                continue
            
            # Get close prices
            if 'close' in df.columns:
                close = df['close']
            elif 'Close' in df.columns:
                close = df['Close']
            else:
                continue
            
            try:
                # Skip most recent month (avoid microstructure effects)
                close_adj = close.iloc[:-self.SKIP_PERIOD] if self.SKIP_PERIOD > 0 else close
                
                if len(close_adj) < self.FORMATION_PERIOD_LONG:
                    continue
                
                # Calculate momentum (total return over period)
                mom_12m = (close_adj.iloc[-1] / close_adj.iloc[-self.FORMATION_PERIOD_LONG]) - 1
                mom_6m = (close_adj.iloc[-1] / close_adj.iloc[-self.FORMATION_PERIOD_MED]) - 1 if len(close_adj) >= self.FORMATION_PERIOD_MED else mom_12m
                mom_1m = (close_adj.iloc[-1] / close_adj.iloc[-self.FORMATION_PERIOD_SHORT]) - 1 if len(close_adj) >= self.FORMATION_PERIOD_SHORT else 0
                
                # Calculate realized volatility
                returns = close.pct_change().dropna()
                vol = returns.iloc[-self.VOL_LOOKBACK:].std() * np.sqrt(252) if len(returns) >= self.VOL_LOOKBACK else returns.std() * np.sqrt(252)
                
                momentums_12m.append(mom_12m)
                momentums_6m.append(mom_6m)
                momentums_1m.append(mom_1m)
                volatilities.append(vol)
                
            except Exception as e:
                logger.warning(f"Error calculating momentum for {etf}: {e}")
                continue
        
        if not momentums_12m:
            return None
        
        # Average across ETFs in factor
        avg_mom_12m = np.mean(momentums_12m)
        avg_mom_6m = np.mean(momentums_6m)
        avg_mom_1m = np.mean(momentums_1m)
        avg_vol = np.mean(volatilities)
        
        # Risk-adjusted momentum (Sharpe-like)
        sharpe = avg_mom_12m / avg_vol if avg_vol > 0 else 0

        # Composite signal: weight longer-term momentum more heavily
        # Research-optimized weights:
        # - 70% 12-month: Primary signal (most predictive for factors)
        # - 20% 6-month: Medium-term confirmation
        # - 10% 1-month: Short-term timing (avoid selling into strength)
        signal = 0.70 * avg_mom_12m + 0.20 * avg_mom_6m + 0.10 * avg_mom_1m
        
        return FactorMomentum(
            name=factor_name,
            etf=etfs[0],  # Primary ETF for trading
            momentum_12m=avg_mom_12m,
            momentum_6m=avg_mom_6m,
            momentum_1m=avg_mom_1m,
            volatility=avg_vol,
            sharpe=sharpe,
            signal=signal
        )
    
    def _calculate_factor_weights(
        self,
        factors: List[FactorMomentum]
    ) -> Dict[str, float]:
        """
        Calculate portfolio weights for each factor.
        
        Uses inverse volatility weighting with momentum filter:
        1. Filter to factors with positive momentum
        2. Weight inversely to volatility (safer factors get more weight)
        3. Apply position limits
        """
        # Filter to positive momentum factors
        positive_factors = [f for f in factors if f.signal > self.MIN_MOMENTUM_THRESHOLD]
        
        if not positive_factors:
            logger.warning("No factors with positive momentum - going to cash")
            return {}
        
        # Inverse volatility weights
        inv_vols = {f.name: 1.0 / f.volatility if f.volatility > 0 else 0 for f in positive_factors}
        total_inv_vol = sum(inv_vols.values())
        
        if total_inv_vol == 0:
            # Equal weight if vol calculation fails
            weight = 1.0 / len(positive_factors)
            return {f.name: weight for f in positive_factors}
        
        # Normalize weights
        weights = {name: inv_vol / total_inv_vol for name, inv_vol in inv_vols.items()}
        
        # Apply position limits
        capped_weights = {}
        excess = 0.0
        uncapped_count = 0
        
        for name, weight in weights.items():
            if weight > self.MAX_FACTOR_WEIGHT:
                capped_weights[name] = self.MAX_FACTOR_WEIGHT
                excess += weight - self.MAX_FACTOR_WEIGHT
            elif weight < self.MIN_FACTOR_WEIGHT:
                capped_weights[name] = 0.0  # Too small, skip
                excess += weight
            else:
                capped_weights[name] = weight
                uncapped_count += 1
        
        # Redistribute excess to uncapped factors
        if excess > 0 and uncapped_count > 0:
            redistribution = excess / uncapped_count
            for name in capped_weights:
                if self.MIN_FACTOR_WEIGHT <= capped_weights[name] < self.MAX_FACTOR_WEIGHT:
                    capped_weights[name] = min(
                        self.MAX_FACTOR_WEIGHT,
                        capped_weights[name] + redistribution
                    )
        
        # Remove zero weights
        final_weights = {k: v for k, v in capped_weights.items() if v > 0}
        
        # Normalize to sum to 1
        total = sum(final_weights.values())
        if total > 0:
            final_weights = {k: v / total for k, v in final_weights.items()}
        
        return final_weights
    
    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame],
        current_positions: List[str] = None,
        vix_regime: str = None
    ) -> List[Signal]:
        """
        Generate factor momentum signals.
        
        Trades ETFs representing factors with positive momentum,
        weighted by inverse volatility.
        """
        signals = []
        
        if not data:
            return signals
        
        # Get current date
        current_date = self._get_current_date(data)
        if current_date is None:
            return signals
        
        # Check if rebalance day
        if not self._is_rebalance_day(current_date):
            return signals
        
        logger.debug(f"Factor Momentum rebalance day: {current_date}")
        self.last_rebalance_month = (current_date.year, current_date.month)
        
        # VIX adjustment based on regime
        vix_multiplier = 1.0
        if vix_regime == 'extreme':
            vix_multiplier = self.EXTREME_VIX_REDUCTION
            logger.debug(f"Extreme VIX regime, reducing to {vix_multiplier*100:.0f}% exposure")
        elif vix_regime == 'high':
            vix_multiplier = self.HIGH_VIX_REDUCTION
            logger.debug(f"High VIX regime, reducing to {vix_multiplier*100:.0f}% exposure")
        
        # Calculate factor momentum
        factor_momentums = []
        for factor_name in FACTOR_ETFS.keys():
            fm = self._calculate_factor_momentum(factor_name, data)
            if fm:
                factor_momentums.append(fm)
                logger.debug(f"Factor {factor_name}: mom_12m={fm.momentum_12m:.2%}, vol={fm.volatility:.2%}, signal={fm.signal:.3f}")
        
        if not factor_momentums:
            logger.warning("Could not calculate momentum for any factors")
            return signals
        
        # Calculate target weights
        target_weights = self._calculate_factor_weights(factor_momentums)
        
        if not target_weights:
            logger.debug("No factors with positive momentum - generating close signals")
            # Sell all current positions
            if current_positions:
                for symbol in current_positions:
                    if symbol in data:
                        df = data[symbol]
                        price = df['close'].iloc[-1] if 'close' in df.columns else df['Close'].iloc[-1]
                        signals.append(Signal(
                            timestamp=current_date,
                            symbol=symbol,
                            strategy=self.name,
                            signal_type=SignalType.CLOSE,
                            strength=1.0,
                            price=price,
                            reason='no_positive_factors'
                        ))
            return signals
        
        # Apply VIX adjustment to weights
        adjusted_weights = {k: v * vix_multiplier for k, v in target_weights.items()}
        
        # Map factors to their primary ETFs
        factor_to_etf = {}
        for factor_name, config in FACTOR_ETFS.items():
            factor_to_etf[factor_name] = config['etfs'][0]
        
        # Current holdings (by ETF)
        current_holdings = set(current_positions) if current_positions else set()
        
        # Target ETFs and weights
        target_etfs = {}
        for factor_name, weight in adjusted_weights.items():
            etf = factor_to_etf.get(factor_name)
            if etf:
                # Accumulate weights if multiple factors use same ETF
                target_etfs[etf] = target_etfs.get(etf, 0) + weight
        
        # Generate close signals for ETFs no longer wanted
        for symbol in current_holdings:
            if symbol not in target_etfs and symbol in data:
                df = data[symbol]
                price = df['close'].iloc[-1] if 'close' in df.columns else df['Close'].iloc[-1]
                signals.append(Signal(
                    timestamp=current_date,
                    symbol=symbol,
                    strategy=self.name,
                    signal_type=SignalType.CLOSE,
                    strength=1.0,
                    price=price,
                    reason='factor_rotation'
                ))
        
        # Generate buy/adjust signals for target ETFs
        for etf, weight in target_etfs.items():
            if etf not in data:
                continue
            
            df = data[etf]
            price = df['close'].iloc[-1] if 'close' in df.columns else df['Close'].iloc[-1]
            
            # Find which factor(s) this ETF represents
            factor_names = [
                fn for fn, config in FACTOR_ETFS.items() 
                if etf in config['etfs'] and fn in adjusted_weights
            ]
            
            # Get momentum stats for metadata
            factor_stats = {
                fn: next((fm for fm in factor_momentums if fm.name == fn), None)
                for fn in factor_names
            }
            
            current_weight = 1.0 if etf in current_holdings else 0
            
            if etf not in current_holdings:
                # New position
                signals.append(Signal(
                    timestamp=current_date,
                    symbol=etf,
                    strategy=self.name,
                    signal_type=SignalType.BUY,
                    strength=weight,
                    price=price,
                    position_size_pct=weight,
                    reason='factor_momentum',
                    metadata={
                        'factors': factor_names,
                        'target_weight': weight,
                        'momentum_12m': np.mean([fs.momentum_12m for fs in factor_stats.values() if fs]),
                        'vix_adjustment': vix_multiplier
                    }
                ))
        
        # Store momentum history for analysis
        self._factor_momentum_history.append({
            'date': current_date,
            'factors': {fm.name: fm.signal for fm in factor_momentums},
            'weights': target_weights,
            'vix_regime': vix_regime
        })
        
        logger.info(
            f"Factor momentum signals: {len(signals)} "
            f"(targeting {len(target_etfs)} ETFs, "
            f"VIX mult={vix_multiplier:.0%})"
        )
        
        return signals
    
    def _get_current_date(self, data: Dict[str, pd.DataFrame]) -> Optional[datetime]:
        """Extract current date from data."""
        for symbol, df in data.items():
            if len(df) > 0:
                if 'timestamp' in df.columns:
                    return pd.to_datetime(df['timestamp'].iloc[-1])
                elif isinstance(df.index, pd.DatetimeIndex):
                    return df.index[-1].to_pydatetime()
        return None
    
    def get_parameters(self) -> Dict:
        """Return current strategy parameters for optimization."""
        return {
            'formation_period_long': self.FORMATION_PERIOD_LONG,
            'formation_period_med': self.FORMATION_PERIOD_MED,
            'skip_period': self.SKIP_PERIOD,
            'vol_lookback': self.VOL_LOOKBACK,
            'max_factor_weight': self.MAX_FACTOR_WEIGHT,
            'min_factor_weight': self.MIN_FACTOR_WEIGHT,
            'high_vix_threshold': self.HIGH_VIX_THRESHOLD,
            'high_vix_reduction': self.HIGH_VIX_REDUCTION,
        }
    
    def set_parameters(self, params: Dict) -> None:
        """Update strategy parameters (for genetic optimization)."""
        if 'formation_period_long' in params:
            self.FORMATION_PERIOD_LONG = int(params['formation_period_long'])
        if 'formation_period_med' in params:
            self.FORMATION_PERIOD_MED = int(params['formation_period_med'])
        if 'skip_period' in params:
            self.SKIP_PERIOD = int(params['skip_period'])
        if 'vol_lookback' in params:
            self.VOL_LOOKBACK = int(params['vol_lookback'])
        if 'max_factor_weight' in params:
            self.MAX_FACTOR_WEIGHT = params['max_factor_weight']
        if 'min_factor_weight' in params:
            self.MIN_FACTOR_WEIGHT = params['min_factor_weight']
        if 'high_vix_threshold' in params:
            self.HIGH_VIX_THRESHOLD = params['high_vix_threshold']
        if 'high_vix_reduction' in params:
            self.HIGH_VIX_REDUCTION = params['high_vix_reduction']
    
    def get_required_symbols(self) -> List[str]:
        """Return list of ETFs required for this strategy."""
        return ALL_FACTOR_ETFS


# Genetic algorithm parameter specs for optimization
# These ranges are based on academic research and practical constraints
OPTIMIZATION_PARAMS = [
    {'name': 'formation_period_long', 'min_val': 189, 'max_val': 252, 'step': 21},  # 9-12 months
    {'name': 'formation_period_med', 'min_val': 63, 'max_val': 126, 'step': 21},    # 3-6 months
    {'name': 'skip_period', 'min_val': 0, 'max_val': 21, 'step': 5},                # 0-1 month (factors need less)
    {'name': 'vol_lookback', 'min_val': 42, 'max_val': 126, 'step': 21},            # 2-6 months
    {'name': 'max_factor_weight', 'min_val': 0.25, 'max_val': 0.40, 'step': 0.05},  # 25-40% max
    {'name': 'high_vix_threshold', 'min_val': 20, 'max_val': 30, 'step': 5},        # VIX trigger
    {'name': 'high_vix_reduction', 'min_val': 0.50, 'max_val': 0.80, 'step': 0.10}, # 50-80% exposure
]
