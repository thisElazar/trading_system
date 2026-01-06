"""
Industry Mean Reversion Strategy
=================================
Mean reversion within industry groups - stocks that underperform their
industry peers tend to revert.

Research basis (Federal Reserve):
- Within-industry mean reversion: +0.821%/month
- Cross-industry mean reversion: -0.295%/month
- Industry context is crucial for identifying true mispricing

Strategy logic:
1. Group stocks by industry (using sector/industry from stock characteristics)
2. Calculate industry average returns (rolling 20-day)
3. Compute z-score: (stock_return - industry_avg) / industry_std
4. Long signal: z-score < -2.0 (underperformed industry significantly)
5. Exit signal: |z-score| < 0.5 (reverted to industry mean)

Key features:
- Only trades within-industry, not cross-industry
- Uses 20-day rolling returns for stability
- Z-score threshold of 2.0 (2 standard deviations)
- Integrates with stock characteristics for industry grouping

Usage:
    from strategies.industry_mean_reversion import IndustryMeanReversionStrategy

    strategy = IndustryMeanReversionStrategy()
    signals = strategy.generate_signals(data, industry_map)
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

class SignalType(Enum):
    """Signal types for mean reversion."""
    LONG = "long"           # Stock underperformed, expect reversion up
    SHORT = "short"         # Stock outperformed, expect reversion down
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"
    HOLD = "hold"


@dataclass
class MeanReversionSignal:
    """Signal from industry mean reversion strategy."""
    symbol: str
    signal_type: SignalType
    timestamp: datetime

    # Z-score metrics
    z_score: float                    # Current z-score
    stock_return: float               # Stock's rolling return
    industry_avg_return: float        # Industry average return
    industry_std: float               # Industry return std dev

    # Industry context
    industry: str
    industry_peers: int               # Number of peers in industry

    # Entry/exit info
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_price: Optional[float] = None

    # Strength metrics
    signal_strength: float = 0.0      # 0-1, higher = stronger signal
    expected_return: float = 0.0      # Expected return to mean

    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'timestamp': self.timestamp.isoformat(),
            'z_score': self.z_score,
            'stock_return': self.stock_return,
            'industry_avg_return': self.industry_avg_return,
            'industry': self.industry,
            'industry_peers': self.industry_peers,
            'signal_strength': self.signal_strength,
            'expected_return': self.expected_return
        }


@dataclass
class IndustryMeanReversionConfig:
    """Configuration for industry mean reversion strategy."""
    # Lookback periods
    return_lookback: int = 20          # Days for calculating rolling returns
    zscore_lookback: int = 60          # Days for z-score calculation

    # Entry thresholds (in standard deviations)
    long_entry_zscore: float = -2.0    # Enter long when z < -2
    short_entry_zscore: float = 2.0    # Enter short when z > 2 (if enabled)

    # Exit thresholds
    exit_zscore: float = 0.5           # Exit when |z| < 0.5

    # Risk management
    stop_loss_pct: float = -0.08       # 8% stop loss
    take_profit_pct: float = 0.12      # 12% take profit
    max_holding_days: int = 20         # Maximum holding period

    # Industry constraints
    min_industry_peers: int = 3        # Minimum stocks in industry
    min_industry_volume: float = 5e6   # Min average industry dollar volume

    # Position sizing
    max_positions: int = 5             # Maximum concurrent positions
    position_size_pct: float = 0.05    # 5% per position

    # Filters
    min_dollar_volume: float = 5e6     # Minimum stock dollar volume
    max_spread_pct: float = 0.005      # Maximum 0.5% spread

    # Enable short selling
    allow_short: bool = False          # Disabled by default


@dataclass
class IndustryStats:
    """Statistics for an industry group."""
    industry: str
    symbols: List[str]
    avg_return: float
    std_return: float
    dollar_volume: float
    num_stocks: int


# =============================================================================
# INDUSTRY MEAN REVERSION STRATEGY
# =============================================================================

class IndustryMeanReversionStrategy:
    """
    Mean reversion strategy that trades stocks relative to their industry peers.

    Core insight: Stocks that underperform their industry tend to revert,
    while cross-industry comparisons are less predictive.
    """

    def __init__(self, config: IndustryMeanReversionConfig = None):
        """
        Initialize strategy.

        Args:
            config: Strategy configuration
        """
        self.config = config or IndustryMeanReversionConfig()
        self.name = "industry_mean_reversion"

        # State
        self._industry_map: Dict[str, str] = {}        # symbol -> industry
        self._industry_groups: Dict[str, List[str]] = {} # industry -> [symbols]
        self._industry_stats: Dict[str, IndustryStats] = {}

        # Position tracking
        self._positions: Dict[str, Dict] = {}  # symbol -> position info
        self._returns_cache: Dict[str, pd.Series] = {}

    # =========================================================================
    # INDUSTRY MAPPING
    # =========================================================================

    def load_industry_map(
        self,
        industry_map: Optional[Dict[str, str]] = None,
        stock_characteristics=None
    ):
        """
        Load industry mapping for symbols.

        Args:
            industry_map: Dict mapping symbol -> industry
            stock_characteristics: StockCharacterizer instance (alternative source)
        """
        if industry_map:
            self._industry_map = industry_map.copy()

        elif stock_characteristics:
            # Load from stock characteristics
            for symbol in stock_characteristics.get_all_symbols():
                char = stock_characteristics.get_characteristics(symbol)
                if char and char.industry:
                    self._industry_map[symbol] = char.industry

        # Build reverse mapping (industry -> symbols)
        self._industry_groups = {}
        for symbol, industry in self._industry_map.items():
            if industry not in self._industry_groups:
                self._industry_groups[industry] = []
            self._industry_groups[industry].append(symbol)

        # Filter industries with enough peers
        self._industry_groups = {
            ind: syms for ind, syms in self._industry_groups.items()
            if len(syms) >= self.config.min_industry_peers
        }

        logger.info(f"Loaded {len(self._industry_map)} symbols across "
                    f"{len(self._industry_groups)} qualifying industries")

    def get_industry(self, symbol: str) -> Optional[str]:
        """Get industry for a symbol."""
        return self._industry_map.get(symbol)

    def get_industry_peers(self, symbol: str) -> List[str]:
        """Get industry peers for a symbol (excluding the symbol itself)."""
        industry = self.get_industry(symbol)
        if not industry or industry not in self._industry_groups:
            return []
        return [s for s in self._industry_groups[industry] if s != symbol]

    # =========================================================================
    # CORE CALCULATIONS
    # =========================================================================

    def calculate_rolling_returns(
        self,
        data: Dict[str, pd.DataFrame],
        lookback: int = None
    ) -> Dict[str, pd.Series]:
        """
        Calculate rolling returns for all symbols.

        Args:
            data: Dict of symbol -> OHLCV DataFrame
            lookback: Number of days for rolling return

        Returns:
            Dict of symbol -> rolling returns series
        """
        lookback = lookback or self.config.return_lookback
        returns = {}

        for symbol, df in data.items():
            if 'close' not in df.columns:
                continue

            # Calculate rolling return over lookback period
            close = df['close']
            rolling_ret = close.pct_change(lookback)
            returns[symbol] = rolling_ret

        self._returns_cache = returns
        return returns

    def calculate_industry_stats(
        self,
        data: Dict[str, pd.DataFrame],
        returns: Dict[str, pd.Series] = None,
        as_of_date: Optional[datetime] = None
    ) -> Dict[str, IndustryStats]:
        """
        Calculate statistics for each industry.

        Args:
            data: Market data
            returns: Pre-calculated returns (optional)
            as_of_date: Date for calculation (defaults to latest)

        Returns:
            Dict of industry -> IndustryStats
        """
        if returns is None:
            returns = self.calculate_rolling_returns(data)

        stats = {}

        for industry, symbols in self._industry_groups.items():
            # Get returns for industry members
            industry_returns = []
            valid_symbols = []
            total_volume = 0

            for symbol in symbols:
                if symbol not in returns or symbol not in data:
                    continue

                ret_series = returns[symbol]
                df = data[symbol]

                # Get latest return
                if as_of_date:
                    ret = ret_series.loc[:as_of_date].iloc[-1] if len(ret_series) > 0 else np.nan
                else:
                    ret = ret_series.iloc[-1] if len(ret_series) > 0 else np.nan

                if not np.isnan(ret):
                    industry_returns.append(ret)
                    valid_symbols.append(symbol)

                    # Add volume
                    if 'volume' in df.columns and 'close' in df.columns:
                        avg_volume = df['volume'].tail(20).mean()
                        avg_price = df['close'].tail(20).mean()
                        total_volume += avg_volume * avg_price

            if len(industry_returns) < self.config.min_industry_peers:
                continue

            # Calculate stats
            avg_return = np.mean(industry_returns)
            std_return = np.std(industry_returns)

            if std_return < 0.001:  # Avoid division by zero
                std_return = 0.001

            stats[industry] = IndustryStats(
                industry=industry,
                symbols=valid_symbols,
                avg_return=avg_return,
                std_return=std_return,
                dollar_volume=total_volume,
                num_stocks=len(valid_symbols)
            )

        self._industry_stats = stats
        return stats

    def calculate_zscore(
        self,
        symbol: str,
        stock_return: float,
        industry_stats: IndustryStats
    ) -> float:
        """
        Calculate z-score for a stock relative to its industry.

        Args:
            symbol: Stock symbol
            stock_return: Stock's rolling return
            industry_stats: Industry statistics

        Returns:
            Z-score (negative = underperformed, positive = outperformed)
        """
        z = (stock_return - industry_stats.avg_return) / industry_stats.std_return
        return float(z)

    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================

    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame],
        industry_map: Optional[Dict[str, str]] = None,
        as_of_date: Optional[datetime] = None
    ) -> List[MeanReversionSignal]:
        """
        Generate mean reversion signals.

        Args:
            data: Dict of symbol -> OHLCV DataFrame
            industry_map: Symbol to industry mapping (optional if already loaded)
            as_of_date: Date for signal generation

        Returns:
            List of MeanReversionSignal objects
        """
        # Load industry map if provided
        if industry_map:
            self.load_industry_map(industry_map)

        if not self._industry_groups:
            logger.warning("No industry groups loaded")
            return []

        # Calculate returns and industry stats
        returns = self.calculate_rolling_returns(data)
        industry_stats = self.calculate_industry_stats(data, returns, as_of_date)

        signals = []

        # Check each symbol
        for symbol, ret_series in returns.items():
            industry = self.get_industry(symbol)
            if not industry or industry not in industry_stats:
                continue

            stats = industry_stats[industry]

            # Get current return
            if as_of_date:
                ret = ret_series.loc[:as_of_date].iloc[-1] if len(ret_series) > 0 else np.nan
            else:
                ret = ret_series.iloc[-1] if len(ret_series) > 0 else np.nan

            if np.isnan(ret):
                continue

            # Calculate z-score
            z_score = self.calculate_zscore(symbol, ret, stats)

            # Check for existing position
            has_position = symbol in self._positions

            # Generate signals based on z-score
            signal = self._evaluate_signal(
                symbol=symbol,
                z_score=z_score,
                stock_return=ret,
                stats=stats,
                data=data.get(symbol),
                has_position=has_position
            )

            if signal:
                signals.append(signal)

        # Sort by signal strength
        signals.sort(key=lambda s: abs(s.z_score), reverse=True)

        # Limit to max positions
        new_entry_signals = [s for s in signals
                            if s.signal_type in [SignalType.LONG, SignalType.SHORT]]
        if len(new_entry_signals) > self.config.max_positions:
            new_entry_signals = new_entry_signals[:self.config.max_positions]

        # Combine with exit signals
        exit_signals = [s for s in signals
                       if s.signal_type in [SignalType.EXIT_LONG, SignalType.EXIT_SHORT]]

        return exit_signals + new_entry_signals

    def _evaluate_signal(
        self,
        symbol: str,
        z_score: float,
        stock_return: float,
        stats: IndustryStats,
        data: Optional[pd.DataFrame],
        has_position: bool
    ) -> Optional[MeanReversionSignal]:
        """Evaluate and create signal for a symbol."""

        timestamp = datetime.now()
        current_price = data['close'].iloc[-1] if data is not None and 'close' in data.columns else None

        # Check exit conditions first
        if has_position:
            position = self._positions.get(symbol, {})
            position_type = position.get('type', 'long')

            # Exit if reverted to mean
            if abs(z_score) < self.config.exit_zscore:
                signal_type = SignalType.EXIT_LONG if position_type == 'long' else SignalType.EXIT_SHORT
                return MeanReversionSignal(
                    symbol=symbol,
                    signal_type=signal_type,
                    timestamp=timestamp,
                    z_score=z_score,
                    stock_return=stock_return,
                    industry_avg_return=stats.avg_return,
                    industry_std=stats.std_return,
                    industry=stats.industry,
                    industry_peers=stats.num_stocks,
                    signal_strength=1.0 - abs(z_score),  # Stronger signal when closer to mean
                    expected_return=0.0  # Already reverted
                )

            return None  # Hold position

        # Entry signals (use <= to include boundary)
        if z_score <= self.config.long_entry_zscore:
            # Stock underperformed industry - long opportunity
            signal_strength = min(1.0, abs(z_score) / 3.0)  # Normalize to 0-1
            expected_return = -z_score * stats.std_return  # Expected reversion

            # Calculate prices
            target_price = None
            stop_price = None
            if current_price:
                # Target: revert to industry mean
                target_price = current_price * (1 + expected_return)
                # Stop: below entry by stop_loss_pct
                stop_price = current_price * (1 + self.config.stop_loss_pct)

            return MeanReversionSignal(
                symbol=symbol,
                signal_type=SignalType.LONG,
                timestamp=timestamp,
                z_score=z_score,
                stock_return=stock_return,
                industry_avg_return=stats.avg_return,
                industry_std=stats.std_return,
                industry=stats.industry,
                industry_peers=stats.num_stocks,
                entry_price=current_price,
                target_price=target_price,
                stop_price=stop_price,
                signal_strength=signal_strength,
                expected_return=expected_return
            )

        elif z_score >= self.config.short_entry_zscore and self.config.allow_short:
            # Stock outperformed industry - short opportunity
            signal_strength = min(1.0, abs(z_score) / 3.0)
            expected_return = z_score * stats.std_return  # Negative = downside

            target_price = None
            stop_price = None
            if current_price:
                target_price = current_price * (1 - expected_return)
                stop_price = current_price * (1 - self.config.stop_loss_pct)

            return MeanReversionSignal(
                symbol=symbol,
                signal_type=SignalType.SHORT,
                timestamp=timestamp,
                z_score=z_score,
                stock_return=stock_return,
                industry_avg_return=stats.avg_return,
                industry_std=stats.std_return,
                industry=stats.industry,
                industry_peers=stats.num_stocks,
                entry_price=current_price,
                target_price=target_price,
                stop_price=stop_price,
                signal_strength=signal_strength,
                expected_return=-expected_return
            )

        return None

    # =========================================================================
    # POSITION MANAGEMENT
    # =========================================================================

    def open_position(
        self,
        symbol: str,
        signal: MeanReversionSignal,
        shares: int,
        entry_price: float
    ):
        """Record a new position."""
        self._positions[symbol] = {
            'type': 'long' if signal.signal_type == SignalType.LONG else 'short',
            'shares': shares,
            'entry_price': entry_price,
            'entry_date': signal.timestamp,
            'z_score_at_entry': signal.z_score,
            'industry': signal.industry,
            'target_price': signal.target_price,
            'stop_price': signal.stop_price
        }
        logger.debug(f"Opened {signal.signal_type.value} position: {symbol} @ {entry_price:.2f}")

    def close_position(self, symbol: str, exit_price: float) -> Optional[Dict]:
        """Close a position and return trade summary."""
        if symbol not in self._positions:
            return None

        position = self._positions.pop(symbol)
        entry_price = position['entry_price']
        shares = position['shares']

        if position['type'] == 'long':
            pnl = (exit_price - entry_price) * shares
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl = (entry_price - exit_price) * shares
            pnl_pct = (entry_price - exit_price) / entry_price

        trade = {
            'symbol': symbol,
            'type': position['type'],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'shares': shares,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'entry_date': position['entry_date'],
            'exit_date': datetime.now(),
            'holding_days': (datetime.now() - position['entry_date']).days,
            'industry': position['industry']
        }

        logger.debug(f"Closed {position['type']} position: {symbol} @ {exit_price:.2f} "
                    f"(PnL: {pnl_pct:.2%})")

        return trade

    def get_positions(self) -> Dict[str, Dict]:
        """Get current positions."""
        return self._positions.copy()

    def check_stops(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> List[MeanReversionSignal]:
        """Check stop losses and take profits for open positions."""
        signals = []

        for symbol, position in list(self._positions.items()):
            if symbol not in data:
                continue

            df = data[symbol]
            current_price = df['close'].iloc[-1]
            entry_price = position['entry_price']

            # Check holding period
            days_held = (datetime.now() - position['entry_date']).days
            if days_held >= self.config.max_holding_days:
                signal_type = SignalType.EXIT_LONG if position['type'] == 'long' else SignalType.EXIT_SHORT
                signals.append(MeanReversionSignal(
                    symbol=symbol,
                    signal_type=signal_type,
                    timestamp=datetime.now(),
                    z_score=0,
                    stock_return=0,
                    industry_avg_return=0,
                    industry_std=0,
                    industry=position['industry'],
                    industry_peers=0,
                    signal_strength=0.5
                ))
                continue

            # Check stop loss and take profit
            if position['type'] == 'long':
                pnl_pct = (current_price - entry_price) / entry_price

                if pnl_pct <= self.config.stop_loss_pct:
                    signals.append(MeanReversionSignal(
                        symbol=symbol,
                        signal_type=SignalType.EXIT_LONG,
                        timestamp=datetime.now(),
                        z_score=0,
                        stock_return=pnl_pct,
                        industry_avg_return=0,
                        industry_std=0,
                        industry=position['industry'],
                        industry_peers=0,
                        signal_strength=1.0
                    ))
                elif pnl_pct >= self.config.take_profit_pct:
                    signals.append(MeanReversionSignal(
                        symbol=symbol,
                        signal_type=SignalType.EXIT_LONG,
                        timestamp=datetime.now(),
                        z_score=0,
                        stock_return=pnl_pct,
                        industry_avg_return=0,
                        industry_std=0,
                        industry=position['industry'],
                        industry_peers=0,
                        signal_strength=1.0
                    ))

        return signals

    # =========================================================================
    # ANALYTICS
    # =========================================================================

    def get_industry_opportunities(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Get overview of mean reversion opportunities by industry.

        Returns DataFrame with industries ranked by opportunity.
        """
        returns = self.calculate_rolling_returns(data)
        stats = self.calculate_industry_stats(data, returns)

        opportunities = []

        for industry, ind_stats in stats.items():
            # Find stocks with extreme z-scores
            underperformers = 0
            outperformers = 0

            for symbol in ind_stats.symbols:
                if symbol not in returns:
                    continue
                ret = returns[symbol].iloc[-1]
                z = self.calculate_zscore(symbol, ret, ind_stats)

                if z < self.config.long_entry_zscore:
                    underperformers += 1
                elif z > self.config.short_entry_zscore:
                    outperformers += 1

            opportunities.append({
                'industry': industry,
                'num_stocks': ind_stats.num_stocks,
                'avg_return': ind_stats.avg_return,
                'std_return': ind_stats.std_return,
                'underperformers': underperformers,
                'outperformers': outperformers,
                'total_opportunities': underperformers + outperformers,
                'dollar_volume': ind_stats.dollar_volume
            })

        df = pd.DataFrame(opportunities)
        return df.sort_values('total_opportunities', ascending=False)


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate industry mean reversion strategy."""
    print("=" * 60)
    print("Industry Mean Reversion Strategy Demo")
    print("=" * 60)

    # Create strategy
    config = IndustryMeanReversionConfig(
        long_entry_zscore=-2.0,
        exit_zscore=0.5,
        min_industry_peers=3
    )
    strategy = IndustryMeanReversionStrategy(config)

    # Create sample data with industry structure
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=60, freq='D')

    # Simulate Tech industry - most stocks up, one lagging
    data = {}
    industry_map = {}

    # Tech stocks - 4 up ~5%, 1 down ~20%
    for i, symbol in enumerate(['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA']):
        base_price = 100 + i * 10

        if symbol == 'META':
            # META: -25% over the period (strong underperformance, z-score < -2)
            close = pd.Series([base_price * (1 - 0.25 * x/59) for x in range(60)], index=dates)
        else:
            # Others: +5% over the period
            close = pd.Series([base_price * (1 + 0.05 * x/59) for x in range(60)], index=dates)

        data[symbol] = pd.DataFrame({
            'open': close * 0.99,
            'high': close * 1.01,
            'low': close * 0.98,
            'close': close,
            'volume': np.random.randint(1000000, 5000000, 60)
        }, index=dates)
        industry_map[symbol] = 'Technology'

    # Healthcare stocks
    health_base = np.random.randn(60).cumsum() * 0.015
    for i, symbol in enumerate(['JNJ', 'PFE', 'UNH', 'MRK']):
        noise = np.random.randn(60) * 0.008
        close = 80 + health_base + noise + i * 3

        data[symbol] = pd.DataFrame({
            'open': close * 0.99,
            'high': close * 1.01,
            'low': close * 0.98,
            'close': close,
            'volume': np.random.randint(500000, 2000000, 60)
        }, index=dates)
        industry_map[symbol] = 'Healthcare'

    # Load industry map
    strategy.load_industry_map(industry_map)

    print(f"\nLoaded {len(industry_map)} stocks across {len(strategy._industry_groups)} industries")
    for ind, syms in strategy._industry_groups.items():
        print(f"  {ind}: {syms}")

    # Generate signals
    print("\n--- Generating Signals ---")
    signals = strategy.generate_signals(data, industry_map)

    if signals:
        print(f"\nFound {len(signals)} signals:")
        for signal in signals:
            print(f"\n  {signal.symbol} ({signal.industry}):")
            print(f"    Signal: {signal.signal_type.value}")
            print(f"    Z-Score: {signal.z_score:.2f}")
            print(f"    Stock Return: {signal.stock_return:.2%}")
            print(f"    Industry Avg: {signal.industry_avg_return:.2%}")
            print(f"    Signal Strength: {signal.signal_strength:.2f}")
            print(f"    Expected Return: {signal.expected_return:.2%}")
    else:
        print("\nNo signals generated")

    # Industry opportunities
    print("\n--- Industry Opportunities ---")
    opps = strategy.get_industry_opportunities(data)
    print(opps.to_string())

    print("\n" + "=" * 60)
    print("Demo Complete")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
