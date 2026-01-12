"""
Sector Rotation Strategy
========================
Rotates between sectors based on VIX regime and momentum.

Research basis:
- Interest rate-based rotation: +2.61% annual outperformance (Europe)
- Sector momentum: 10% annual returns with 3-6 month formation
- Healthcare/Utilities defensive during high VIX
- Tech/Financials offensive during low VIX

Implementation:
- VIX < 18: Overweight cyclicals (XLK, XLF, XLI, XLY)
- VIX 18-25: Balanced allocation
- VIX > 25: Overweight defensives (XLV, XLU, XLP, XLRE)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.base import BaseStrategy, Signal, SignalType
from data.cached_data_manager import CachedDataManager
from config import DIRS, VIX_REGIMES
from utils.timezone import normalize_dataframe, normalize_timestamp, normalize_index

logger = logging.getLogger(__name__)


# Sector ETFs by type
SECTORS = {
    'cyclical': ['XLK', 'XLF', 'XLI', 'XLY', 'XLB', 'XLE'],
    'defensive': ['XLV', 'XLU', 'XLP', 'XLC'],  # Removed XLRE - not in data
}

ALL_SECTOR_ETFS = SECTORS['cyclical'] + SECTORS['defensive']

# Target allocations by regime
REGIME_ALLOCATIONS = {
    'low_vix': {  # VIX < 18: Risk-on
        'XLK': 0.20,  # Tech
        'XLF': 0.15,  # Financials
        'XLI': 0.15,  # Industrials
        'XLY': 0.15,  # Consumer Disc
        'XLV': 0.10,  # Healthcare
        'XLU': 0.05,  # Utilities
        'XLP': 0.05,  # Staples
        'XLE': 0.10,  # Energy
        'XLB': 0.05,  # Materials
    },
    'normal': {  # VIX 18-25: Balanced
        'XLK': 0.15,
        'XLF': 0.10,
        'XLI': 0.10,
        'XLY': 0.10,
        'XLV': 0.15,
        'XLU': 0.10,
        'XLP': 0.10,
        'XLE': 0.10,
        'XLB': 0.10,
    },
    'high_vix': {  # VIX > 25: Risk-off
        'XLK': 0.10,
        'XLF': 0.05,
        'XLI': 0.05,
        'XLY': 0.05,
        'XLV': 0.25,  # Healthcare - defensive
        'XLU': 0.20,  # Utilities - defensive
        'XLP': 0.15,  # Staples - defensive
        'XLE': 0.05,
        'XLB': 0.05,
        'XLC': 0.05,
    },
}


@dataclass
class SectorScore:
    """Momentum score for a sector."""
    symbol: str
    momentum_1m: float
    momentum_3m: float
    momentum_6m: float
    combined_score: float
    regime_weight: float


class SectorRotationStrategy(BaseStrategy):
    """
    Sector rotation based on VIX regime and momentum.

    Generates signals to rebalance sector allocation when:
    1. VIX regime changes
    2. Daily check: allocation drift exceeds threshold (5%)
    3. Minimum 5 trading days between rebalances (prevents over-trading)
    4. Sector momentum diverges significantly

    Enhanced to generate 50-100 trades/year vs the previous 5-9.
    """

    def __init__(
        self,
        low_vix_threshold: float = None,  # Uses VIX_REGIMES['low'] if None
        high_vix_threshold: float = None,  # Uses VIX_REGIMES['normal'] if None
        rebalance_threshold: float = 0.05,  # 5% drift triggers rebalance
        momentum_weight: float = 0.3,  # 30% momentum overlay
        min_days_between_rebalance: int = 5,  # Minimum days between rebalances
        # BUG-003: GA-optimized parameters (Sharpe -0.38 -> 1.08)
        momentum_period: int = 105,  # GA optimal: ~5 months lookback
        top_n_sectors: int = 2,      # GA optimal: concentrate in top 2 sectors
        rebalance_days: int = 28,    # GA optimal: monthly rebalancing
    ):
        super().__init__(name="sector_rotation")

        # Use central config thresholds by default
        self.low_vix = low_vix_threshold if low_vix_threshold is not None else VIX_REGIMES['low']
        self.high_vix = high_vix_threshold if high_vix_threshold is not None else VIX_REGIMES['normal']
        self.rebalance_threshold = rebalance_threshold
        self.momentum_weight = momentum_weight
        self.min_days_between_rebalance = int(min_days_between_rebalance)

        # BUG-003: GA-discovered optimal parameters
        # Cast to int to prevent "Cannot index by location with non-integer key" errors
        # GA optimizer can pass floats (e.g., 122.00) which fail with pandas iloc
        self.momentum_period = int(momentum_period)
        self.top_n_sectors = int(top_n_sectors)
        self.rebalance_days = int(rebalance_days)

        self.data_mgr = CachedDataManager()
        self.current_regime = None
        self.last_rebalance: Optional[datetime] = None
        # Target allocations (what we set after rebalance)
        self.target_allocations: Dict[str, float] = {}
        # Actual position weights (should be updated from portfolio tracker)
        self.actual_weights: Dict[str, float] = {}
    
    def get_vix_regime(self, vix_value: float) -> str:
        """Determine regime from VIX value."""
        if vix_value < self.low_vix:
            return 'low_vix'
        elif vix_value > self.high_vix:
            return 'high_vix'
        return 'normal'
    
    def calculate_momentum(self, symbol: str, data: pd.DataFrame = None) -> Optional[SectorScore]:
        """
        Calculate momentum scores for a sector ETF.

        BUG-003: Now uses configurable momentum_period (GA optimal: 105 days).
        """
        if data is None:
            data = self.data_mgr.get_bars(symbol)

        if data is None or len(data) < self.momentum_period:
            return None

        close = data['close']

        # BUG-003: Use single momentum period (GA discovered this beats multi-horizon)
        # Primary momentum: configurable period (default 105 days = ~5 months)
        mom_primary = (close.iloc[-1] / close.iloc[-self.momentum_period] - 1) if len(close) >= self.momentum_period else 0

        # Secondary horizons for reference (lower weights)
        mom_1m = (close.iloc[-1] / close.iloc[-21] - 1) if len(close) >= 21 else 0
        mom_3m = (close.iloc[-1] / close.iloc[-63] - 1) if len(close) >= 63 else 0

        # BUG-003: Primary period dominates (GA optimal)
        combined = 0.7 * mom_primary + 0.2 * mom_3m + 0.1 * mom_1m

        return SectorScore(
            symbol=symbol,
            momentum_1m=mom_1m,
            momentum_3m=mom_3m,
            momentum_6m=mom_primary,  # Use primary period for 6m slot
            combined_score=combined,
            regime_weight=0  # Set later
        )
    
    def get_target_allocations(self, regime: str, include_momentum: bool = True) -> Dict[str, float]:
        """
        Get target allocations for current regime.

        BUG-003: GA optimization discovered that concentrating in top N sectors
        by momentum dramatically outperforms diversified regime allocation.
        - Original approach: Spread across all sectors by regime
        - GA optimal: Select only top 2 sectors by momentum (Sharpe: -0.38 -> 1.08)
        """
        base_alloc = REGIME_ALLOCATIONS.get(regime, REGIME_ALLOCATIONS['normal']).copy()

        if not include_momentum:
            return base_alloc

        # Calculate momentum scores for all sectors
        scores = {}
        for symbol in ALL_SECTOR_ETFS:
            score = self.calculate_momentum(symbol)
            if score:
                scores[symbol] = score.combined_score

        if not scores:
            return base_alloc

        # BUG-003: Select only top N sectors by momentum (GA optimal: top 2)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_sectors = [symbol for symbol, _ in ranked[:self.top_n_sectors]]

        logger.debug(f"BUG-003: Top {self.top_n_sectors} sectors by momentum: {top_sectors}")

        # Equal weight among top sectors (GA discovered this beats regime weighting)
        equal_weight = 1.0 / self.top_n_sectors

        return {symbol: equal_weight for symbol in top_sectors}
    
    def check_drift(self, targets: Dict[str, float]) -> Tuple[bool, Dict[str, float]]:
        """
        Check if actual weights have drifted from targets.

        Args:
            targets: Target allocation weights

        Returns:
            Tuple of (needs_rebalance, drift_amounts)
        """
        drifts = {}
        max_drift = 0

        for symbol, target in targets.items():
            actual = self.actual_weights.get(symbol, 0)
            drift = abs(actual - target)
            drifts[symbol] = drift
            max_drift = max(max_drift, drift)

        # Also check for positions we have but shouldn't
        for symbol, actual in self.actual_weights.items():
            if symbol not in targets and actual > 0.01:
                drifts[symbol] = actual
                max_drift = max(max_drift, actual)

        needs_rebalance = max_drift > self.rebalance_threshold
        return needs_rebalance, drifts

    def can_rebalance(self, current_date: datetime) -> bool:
        """
        Check if enough time has passed since last rebalance.

        BUG-003: Uses GA-optimized rebalance_days (default 28 = monthly).
        """
        if self.last_rebalance is None:
            return True

        days_since = (current_date - self.last_rebalance).days
        # BUG-003: Use rebalance_days (GA optimal: 28 days = monthly)
        return days_since >= self.rebalance_days

    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame] = None,
        current_positions: List[str] = None,
        vix_regime: str = None
    ) -> List[Signal]:
        """
        Generate sector rotation signals.

        Now checks daily for:
        1. Regime changes (immediate rebalance after cooldown)
        2. Allocation drift > 5% from targets
        3. Position mismatches with target portfolio
        """
        signals = []

        # Map regime names from backtester to our naming convention
        regime_map = {
            'low': 'low_vix',
            'normal': 'normal',
            'high': 'high_vix',
            'extreme': 'high_vix',  # Treat extreme same as high
            'low_vix': 'low_vix',
            'high_vix': 'high_vix',
        }
        vix_regime = regime_map.get(vix_regime, 'normal')

        # Extract current date from data
        current_date = None
        if data:
            for symbol, df in data.items():
                if len(df) > 0 and isinstance(df.index, pd.DatetimeIndex):
                    current_date = df.index[-1].to_pydatetime()
                    break
        if current_date is None:
            current_date = datetime.now()

        # Get target allocations for current regime
        targets = self.get_target_allocations(vix_regime)

        # Check for regime change
        regime_changed = (self.current_regime != vix_regime)
        if regime_changed:
            logger.debug(f"Regime change: {self.current_regime} -> {vix_regime}")
            self.current_regime = vix_regime

        # Determine if we need to rebalance
        needs_rebalance = False
        rebalance_reason = ""

        # Reason 1: First run (no positions/allocations yet)
        if not self.target_allocations and not self.actual_weights:
            needs_rebalance = True
            rebalance_reason = "initialization"
            logger.info("First run - initializing sector allocations")

        # Reason 2: Regime changed
        elif regime_changed:
            if self.can_rebalance(current_date):
                needs_rebalance = True
                rebalance_reason = "regime_change"
            else:
                logger.debug("Regime changed but cooldown not expired")

        # Reason 3: Check for drift from targets (daily check)
        if not needs_rebalance and self.actual_weights:
            has_drift, drifts = self.check_drift(targets)
            if has_drift and self.can_rebalance(current_date):
                needs_rebalance = True
                rebalance_reason = "drift"
                max_drift_symbol = max(drifts.items(), key=lambda x: x[1])
                logger.debug(f"Drift detected: {max_drift_symbol[0]} drifted {max_drift_symbol[1]:.1%}")

        # Reason 4: Check if positions don't match target symbols at all
        if not needs_rebalance:
            target_symbols = set(targets.keys())
            current_symbols = set(current_positions) if current_positions else set()
            # Only check symbols with meaningful allocations
            meaningful_targets = {s for s, w in targets.items() if w > 0.02}
            if meaningful_targets != current_symbols:
                if self.can_rebalance(current_date):
                    needs_rebalance = True
                    rebalance_reason = "position_mismatch"
                    logger.debug(f"Position mismatch: have {current_symbols}, need {meaningful_targets}")

        if not needs_rebalance:
            return signals

        # Generate rebalance signals
        for symbol, target_weight in targets.items():
            # Use actual weights if available, otherwise use target_allocations
            if self.actual_weights:
                current_weight = self.actual_weights.get(symbol, 0)
            else:
                current_weight = self.target_allocations.get(symbol, 0)

            # Get price from data
            if data and symbol in data and len(data[symbol]) > 0:
                price = float(data[symbol]['close'].iloc[-1])
            else:
                # Try loading from cached data
                df = self.data_mgr.get_bars(symbol)
                if df is not None and len(df) > 0:
                    price = float(df['close'].iloc[-1])
                else:
                    logger.debug(f"Skipping {symbol}: no price data")
                    continue

            if price <= 0:
                continue

            weight_diff = target_weight - current_weight

            if weight_diff > 0.01:  # Buy/increase
                signals.append(Signal(
                    timestamp=current_date,
                    symbol=symbol,
                    strategy=self.name,
                    signal_type=SignalType.BUY,
                    strength=min(target_weight * 2, 1.0),
                    price=price,
                    position_size_pct=target_weight,
                    metadata={
                        'target_weight': target_weight,
                        'current_weight': current_weight,
                        'regime': vix_regime,
                        'rebalance_reason': rebalance_reason
                    }
                ))
            elif weight_diff < -0.01:  # Reduce/close
                if target_weight < 0.02:  # Close position entirely
                    signals.append(Signal(
                        timestamp=current_date,
                        symbol=symbol,
                        strategy=self.name,
                        signal_type=SignalType.CLOSE,
                        strength=1.0,
                        price=price,
                        metadata={
                            'reason': 'below_threshold',
                            'rebalance_reason': rebalance_reason
                        }
                    ))
                else:  # Reduce position
                    signals.append(Signal(
                        timestamp=current_date,
                        symbol=symbol,
                        strategy=self.name,
                        signal_type=SignalType.SELL,
                        strength=0.7,
                        price=price,
                        position_size_pct=current_weight - target_weight,
                        metadata={
                            'target_weight': target_weight,
                            'current_weight': current_weight,
                            'regime': vix_regime,
                            'rebalance_reason': rebalance_reason
                        }
                    ))

        # Update state after generating signals
        if signals:
            self.target_allocations = targets.copy()
            self.last_rebalance = current_date
            logger.debug(f"Sector rotation: {len(signals)} signals ({rebalance_reason}) for regime '{vix_regime}'")

        return signals

    def update_actual_weights(self, weights: Dict[str, float]):
        """
        Update actual position weights (call after portfolio update).

        Args:
            weights: Dict mapping symbol to current weight (0-1)
        """
        self.actual_weights = weights.copy()

    def calculate_weights_from_positions(
        self,
        data: Dict[str, pd.DataFrame],
        positions: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate current weights from share counts and prices.

        Args:
            data: Price data for symbols
            positions: Dict mapping symbol to share count

        Returns:
            Dict mapping symbol to weight (0-1)
        """
        if not positions:
            return {}

        total_value = 0
        values = {}

        for symbol, shares in positions.items():
            if symbol in data and len(data[symbol]) > 0:
                price = float(data[symbol]['close'].iloc[-1])
                value = shares * price
                values[symbol] = value
                total_value += value

        if total_value <= 0:
            return {}

        return {symbol: value / total_value for symbol, value in values.items()}

    def update_allocations(self, allocations: Dict[str, float]):
        """Update actual allocation weights after execution (legacy compatibility)."""
        self.actual_weights = allocations.copy()


class SectorRotationBacktester:
    """Backtest sector rotation strategy with realistic drift simulation."""

    def __init__(self):
        self.data_mgr = CachedDataManager()
        self.strategy = SectorRotationStrategy()

    def load_vix(self) -> pd.DataFrame:
        """Load VIX data."""
        vix_path = DIRS['vix'] / 'vix.parquet'
        if vix_path.exists():
            return pd.read_parquet(vix_path)
        return None

    def backtest(
        self,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> pd.DataFrame:
        """
        Run backtest of sector rotation with realistic drift simulation.

        Now properly tracks:
        - Actual position weights that drift due to price movements
        - Daily rebalance checks (not just monthly)
        - Signals generated through strategy's generate_signals method

        Returns:
            DataFrame with daily returns
        """
        # Load data for all sector ETFs
        sector_data = {}
        for symbol in ALL_SECTOR_ETFS:
            df = self.data_mgr.get_bars(symbol)
            if df is not None:
                # Ensure datetime index
                if 'timestamp' in df.columns:
                    df = df.set_index('timestamp')
                if hasattr(df.index, 'tz') and df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                sector_data[symbol] = df

        # Load VIX
        vix_df = self.load_vix()
        if vix_df is None:
            logger.error("No VIX data")
            return pd.DataFrame()

        vix_df = normalize_dataframe(vix_df)

        # Get common date range
        all_dates = set(vix_df.index)
        for df in sector_data.values():
            all_dates &= set(df.index)

        dates = sorted(all_dates)

        if start_date:
            dates = [d for d in dates if d >= pd.Timestamp(start_date)]
        if end_date:
            dates = [d for d in dates if d <= pd.Timestamp(end_date)]

        if len(dates) < 30:
            logger.warning("Insufficient data for backtest")
            return pd.DataFrame()

        # Initialize tracking variables
        results = []
        # Track actual dollar values (not just weights) to simulate drift
        position_values: Dict[str, float] = {}  # symbol -> dollar value
        target_weights: Dict[str, float] = {}
        rebalance_dates = []
        total_signals = 0
        initial_capital = 100000.0

        for i, date in enumerate(dates):
            if i < 21:  # Need 1 month warmup
                continue

            # Get VIX regime
            vix_value = vix_df.loc[date, 'close']
            regime = self.strategy.get_vix_regime(vix_value)

            # Calculate current weights from position values (simulates drift)
            total_value = sum(position_values.values()) if position_values else initial_capital
            if total_value > 0 and position_values:
                actual_weights = {s: v / total_value for s, v in position_values.items()}
            else:
                actual_weights = {}

            # Update strategy's actual weights for drift detection
            self.strategy.update_actual_weights(actual_weights)

            # Prepare data slice for strategy
            data_slice = {}
            for symbol, df in sector_data.items():
                if date in df.index:
                    # Get data up to current date
                    mask = df.index <= date
                    data_slice[symbol] = df.loc[mask].copy()

            # Get current position symbols
            current_positions = list(position_values.keys()) if position_values else []

            # Generate signals using the strategy
            signals = self.strategy.generate_signals(
                data=data_slice,
                current_positions=current_positions,
                vix_regime=regime
            )

            # Process signals (simulate execution)
            if signals:
                rebalance_dates.append(date)
                total_signals += len(signals)

                # Get new target allocations
                target_weights = self.strategy.target_allocations.copy()

                # Reset positions to target weights
                position_values = {}
                for symbol, weight in target_weights.items():
                    if symbol in sector_data and date in sector_data[symbol].index:
                        position_values[symbol] = total_value * weight

            # Calculate daily portfolio return and update position values
            portfolio_return = 0
            if position_values and i > 21:
                prev_date = dates[i - 1]
                new_position_values = {}

                for symbol, value in position_values.items():
                    if symbol in sector_data:
                        df = sector_data[symbol]
                        if date in df.index and prev_date in df.index:
                            daily_ret = df.loc[date, 'close'] / df.loc[prev_date, 'close'] - 1
                            # Update position value (this creates drift!)
                            new_value = value * (1 + daily_ret)
                            new_position_values[symbol] = new_value
                            # Weight-adjusted return
                            weight = value / total_value if total_value > 0 else 0
                            portfolio_return += weight * daily_ret
                        else:
                            new_position_values[symbol] = value

                position_values = new_position_values

            results.append({
                'date': date,
                'return': portfolio_return,
                'regime': regime,
                'vix': vix_value,
                'rebalanced': date in rebalance_dates,
                'num_positions': len(position_values)
            })

        results_df = pd.DataFrame(results)
        results_df.set_index('date', inplace=True)

        # Calculate metrics
        total_return = (1 + results_df['return']).prod() - 1
        num_years = len(results_df) / 252
        annual_return = (1 + total_return) ** (1 / num_years) - 1 if num_years > 0 else 0
        volatility = results_df['return'].std() * np.sqrt(252)
        sharpe = annual_return / volatility if volatility > 0 else 0

        logger.info(f"Sector Rotation Backtest:")
        logger.info(f"  Period: {dates[21]} to {dates[-1]}")
        logger.info(f"  Total Return: {total_return:.2%}")
        logger.info(f"  Annual Return: {annual_return:.2%}")
        logger.info(f"  Sharpe: {sharpe:.2f}")
        logger.info(f"  Rebalances: {len(rebalance_dates)}")
        logger.info(f"  Total Signals: {total_signals}")
        logger.info(f"  Signals/Year: {total_signals / num_years:.1f}" if num_years > 0 else "  Signals/Year: N/A")

        return results_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("SECTOR ROTATION STRATEGY")
    print("=" * 60)
    
    bt = SectorRotationBacktester()
    results = bt.backtest()
    
    if len(results) > 0:
        print(f"\nDaily returns sample:")
        print(results.tail(10))
        
        # Regime breakdown
        print("\nPerformance by regime:")
        for regime in ['low_vix', 'normal', 'high_vix']:
            regime_data = results[results['regime'] == regime]
            if len(regime_data) > 0:
                ret = (1 + regime_data['return']).prod() - 1
                print(f"  {regime}: {ret:.2%} ({len(regime_data)} days)")
