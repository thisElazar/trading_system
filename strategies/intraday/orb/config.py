"""
Opening Range Breakout Configuration
====================================
Configuration for the ORB intraday strategy.

Research basis:
- First 30 minutes often sets the day's high/low (40-50% of days)
- Breakouts from opening range with volume confirm trend
- False breakouts common - requires confirmation filters
- Best on trend days, poor on choppy days
"""

from dataclasses import dataclass, field
from typing import List, Optional
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config import TOTAL_CAPITAL


@dataclass
class ORBConfig:
    """
    Configuration for Opening Range Breakout strategy.

    Strategy Logic:
    1. Measure the high/low range of first N minutes
    2. Wait for price to break outside this range
    3. Enter on breakout with volume confirmation
    4. Target = range height, stop = opposite side of range

    Entry Conditions:
    - Price breaks above/below opening range
    - Volume > threshold (confirms breakout)
    - Price above/below VWAP (confirms direction)
    - Not a choppy/inside day (range must be significant)

    Exit Conditions:
    - Target reached (1-2x range height)
    - Stop loss hit (opposite side of range)
    - Time limit exceeded
    - End of day approaching
    """

    # Universe
    symbols: List[str] = field(default_factory=lambda: ['SPY', 'QQQ'])

    # Opening range parameters
    range_minutes: int = 30              # First 30 minutes define the range
    min_range_pct: float = 0.15          # Min range size (filter tiny ranges)
    max_range_pct: float = 1.5           # Max range size (avoid gap days)

    # Breakout confirmation
    breakout_buffer_pct: float = 0.02    # Price must break by this % beyond range
    min_relative_volume: float = 1.5     # Volume must be 150% of average
    require_vwap_alignment: bool = True  # Price above/below VWAP confirms trend

    # Exit targets
    target_multiple: float = 1.5         # Target = range_height * this multiple
    use_trailing_stop: bool = True       # Trail stop after profit target 1x
    trailing_stop_pct: float = 0.5       # Trail by 50% of range

    # Risk management
    stop_loss_buffer_pct: float = 0.1    # Stop below/above range by this %
    max_hold_minutes: int = 240          # 4 hours max hold

    # Position sizing
    max_position_pct: float = 0.10       # 10% of portfolio per trade
    portfolio_value: Optional[float] = None  # Uses TOTAL_CAPITAL from config if None

    # Timing
    entry_start_minute: int = 31         # Start looking after range forms
    entry_end_minute: int = 180          # Stop entering after 3 hours
    force_exit_minute: int = 385         # Force close 5 min before close

    # Direction preference
    long_only: bool = True               # Only trade breakouts to upside

    # Additional filters
    require_inside_range_bar: bool = False  # Wait for consolidation before break

    def __post_init__(self):
        """Validate configuration and set defaults."""
        # Use TOTAL_CAPITAL from config if portfolio_value not specified
        if self.portfolio_value is None:
            self.portfolio_value = TOTAL_CAPITAL

        if self.range_minutes < 5 or self.range_minutes > 60:
            raise ValueError("range_minutes should be between 5 and 60")
        if self.min_range_pct <= 0:
            raise ValueError("min_range_pct must be positive")
        if self.max_range_pct <= self.min_range_pct:
            raise ValueError("max_range_pct must be > min_range_pct")
        if self.target_multiple <= 0:
            raise ValueError("target_multiple must be positive")

    @property
    def max_position_value(self) -> float:
        """Maximum position value in dollars."""
        return self.portfolio_value * self.max_position_pct

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'symbols': self.symbols,
            'range_minutes': self.range_minutes,
            'min_range_pct': self.min_range_pct,
            'max_range_pct': self.max_range_pct,
            'breakout_buffer_pct': self.breakout_buffer_pct,
            'min_relative_volume': self.min_relative_volume,
            'require_vwap_alignment': self.require_vwap_alignment,
            'target_multiple': self.target_multiple,
            'stop_loss_buffer_pct': self.stop_loss_buffer_pct,
            'max_hold_minutes': self.max_hold_minutes,
            'max_position_pct': self.max_position_pct,
            'entry_start_minute': self.entry_start_minute,
            'entry_end_minute': self.entry_end_minute,
            'long_only': self.long_only,
        }


# Default configuration
DEFAULT_CONFIG = ORBConfig()

# 15-minute ORB (faster signals)
ORB_15MIN_CONFIG = ORBConfig(
    range_minutes=15,
    entry_start_minute=16,
    target_multiple=2.0,         # Smaller range needs larger target multiple
    max_hold_minutes=180,
)

# 5-minute ORB (scalping)
ORB_5MIN_CONFIG = ORBConfig(
    range_minutes=5,
    entry_start_minute=6,
    target_multiple=2.5,
    max_hold_minutes=60,
    min_relative_volume=2.0,     # Need stronger volume confirmation
    min_range_pct=0.10,
    max_range_pct=0.8,
)

# Conservative (longer range, tighter filters)
ORB_CONSERVATIVE_CONFIG = ORBConfig(
    range_minutes=30,
    min_range_pct=0.20,
    breakout_buffer_pct=0.05,
    min_relative_volume=2.0,
    target_multiple=1.0,         # More modest target
    max_hold_minutes=120,
)
