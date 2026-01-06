"""
VWAP Mean Reversion Configuration
=================================
Configuration for the VWAP mean reversion intraday strategy.

Research basis:
- Price tends to revert to VWAP during choppy/range-bound days
- Works best when price deviates 1-2% from VWAP
- Success rate drops in strong trending markets
"""

from dataclasses import dataclass, field
from typing import List, Optional
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config import TOTAL_CAPITAL


@dataclass
class VWAPReversionConfig:
    """
    Configuration for VWAP mean reversion strategy.

    Strategy Logic:
    1. Calculate VWAP throughout the session
    2. When price deviates significantly from VWAP, enter counter-trend
    3. Exit when price returns toward VWAP or hits stop/target

    Entry Conditions:
    - Price > VWAP + threshold% -> SHORT (expect reversion down)
    - Price < VWAP - threshold% -> LONG (expect reversion up)
    - Relative volume > 1.0 (confirms activity)
    - RSI confirms overextension

    Exit Conditions:
    - Price crosses VWAP (full reversion)
    - Partial target hit (75% of deviation closed)
    - Stop loss hit
    - Time limit exceeded
    - Market close approaching
    """

    # Universe
    symbols: List[str] = field(default_factory=lambda: ['SPY', 'QQQ'])

    # Entry thresholds (backtested optimal for liquid ETFs)
    min_vwap_deviation_pct: float = 0.3   # Min deviation from VWAP to enter
    max_vwap_deviation_pct: float = 1.5   # Max deviation (avoid trend days)

    # Confirmation filters
    min_relative_volume: float = 1.0      # Volume must be at least average
    rsi_overbought: float = 70.0          # RSI threshold for shorts
    rsi_oversold: float = 30.0            # RSI threshold for longs

    # Exit targets
    reversion_target_pct: float = 0.50    # Exit when 50% of deviation closed (backtested)
    vwap_cross_exit: bool = True          # Exit on VWAP cross (full reversion)

    # Risk management (backtested optimal)
    stop_loss_pct: float = 0.8            # 0.8% stop from entry
    max_hold_minutes: int = 60            # 1 hour max hold

    # Position sizing
    max_position_pct: float = 0.10        # 10% of portfolio per trade
    portfolio_value: Optional[float] = None  # Uses TOTAL_CAPITAL from config if None

    # Timing
    start_trading_minute: int = 30        # Wait 30 min for VWAP to stabilize
    stop_trading_minute: int = 360        # Stop new entries 30 min before close
    force_exit_minute: int = 385          # Force close 5 min before close

    # Direction preference
    long_only: bool = False               # Backtested: shorts outperform (72% vs 65% WR)

    def __post_init__(self):
        """Validate configuration and set defaults."""
        # Use TOTAL_CAPITAL from config if portfolio_value not specified
        if self.portfolio_value is None:
            self.portfolio_value = TOTAL_CAPITAL

        if self.min_vwap_deviation_pct <= 0:
            raise ValueError("min_vwap_deviation_pct must be positive")
        if self.max_vwap_deviation_pct <= self.min_vwap_deviation_pct:
            raise ValueError("max_vwap_deviation_pct must be > min")
        if not 0 < self.reversion_target_pct <= 1.0:
            raise ValueError("reversion_target_pct must be between 0 and 1")
        if self.stop_loss_pct <= 0:
            raise ValueError("stop_loss_pct must be positive")

    @property
    def max_position_value(self) -> float:
        """Maximum position value in dollars."""
        return self.portfolio_value * self.max_position_pct

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'symbols': self.symbols,
            'min_vwap_deviation_pct': self.min_vwap_deviation_pct,
            'max_vwap_deviation_pct': self.max_vwap_deviation_pct,
            'min_relative_volume': self.min_relative_volume,
            'rsi_overbought': self.rsi_overbought,
            'rsi_oversold': self.rsi_oversold,
            'reversion_target_pct': self.reversion_target_pct,
            'stop_loss_pct': self.stop_loss_pct,
            'max_hold_minutes': self.max_hold_minutes,
            'max_position_pct': self.max_position_pct,
            'start_trading_minute': self.start_trading_minute,
            'stop_trading_minute': self.stop_trading_minute,
            'long_only': self.long_only,
        }


# Default configuration
DEFAULT_CONFIG = VWAPReversionConfig()

# More aggressive (wider deviation tolerance)
AGGRESSIVE_CONFIG = VWAPReversionConfig(
    min_vwap_deviation_pct=0.75,
    max_vwap_deviation_pct=3.0,
    stop_loss_pct=2.0,
    max_hold_minutes=180,
)

# Conservative (wait for larger deviations)
CONSERVATIVE_CONFIG = VWAPReversionConfig(
    min_vwap_deviation_pct=1.5,
    max_vwap_deviation_pct=2.5,
    stop_loss_pct=1.0,
    max_hold_minutes=90,
    min_relative_volume=1.2,
)
