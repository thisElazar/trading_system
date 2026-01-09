"""
Gap Fill Strategy Configuration
===============================
Configuration parameters for the intraday gap-fill strategy.

Research basis:
- Gaps in the 0.15% - 0.60% range fill 59-61% of the time
- Most gap fills occur within the first 2 hours of trading
- Larger gaps (>0.60%) are often news-driven and less predictable
"""

from dataclasses import dataclass, field
from typing import List, Optional
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config import TOTAL_CAPITAL, INTRADAY_UNIVERSE, INTRADAY_SYMBOLS


@dataclass
class GapFillConfig:
    """
    Configuration for the intraday gap-fill strategy.

    This strategy fades (trades against) overnight gaps, expecting them
    to fill back towards the previous day's close.

    Attributes:
        symbols: List of symbols to trade (default: SPY, QQQ)
        min_gap_pct: Minimum gap size to trade (filter noise)
        max_gap_pct: Maximum gap size to trade (avoid news gaps)
        max_hold_minutes: Maximum time to hold position
        fill_threshold: Exit when gap has filled this much (0.0-1.0)
        stop_loss_pct: Stop loss as percentage of entry
        max_position_pct: Maximum position size as % of portfolio
        portfolio_value: Total portfolio value for position sizing

    Gap Detection Window:
        Gaps are detected in the first 5 minutes of market open (9:30-9:35 AM).
        This allows for the opening auction to settle.

    Exit Conditions:
        1. Gap fills to threshold (default 75%)
        2. Time limit exceeded (default 120 minutes)
        3. Stop loss hit (default 2%)
        4. Market close approaches

    Research Notes:
        - Gaps 0.15%-0.60% have 59-61% fill rate
        - Mean time to fill: ~45 minutes
        - Success rate decreases after 2 hours
    """

    # Universe - defaults to broad market ETFs from config
    # Can be expanded to full INTRADAY_SYMBOLS for more signals
    symbols: List[str] = field(default_factory=lambda: INTRADAY_UNIVERSE.get("broad_market", ["SPY", "QQQ", "IWM", "DIA"]))

    # Gap detection parameters
    min_gap_pct: float = 0.15  # Minimum gap to trade (filter noise)
    max_gap_pct: float = 0.60  # Maximum gap to trade (avoid news gaps)

    # Timing parameters
    gap_detection_start_minute: int = 0   # Minutes after open to start detection
    gap_detection_end_minute: int = 5     # Minutes after open to end detection
    max_hold_minutes: int = 120           # Maximum holding period

    # Exit parameters
    fill_threshold: float = 0.75          # Exit when 75% of gap has filled

    # Risk management
    stop_loss_pct: float = 2.0            # Stop loss percentage from entry
    max_position_pct: float = 0.05        # Max 5% of portfolio per position
    max_positions: int = 2                # Maximum concurrent positions

    # Portfolio
    portfolio_value: Optional[float] = None  # Uses TOTAL_CAPITAL from config if None

    # Trading restrictions
    long_only: bool = True                # Only fade gap-downs (go long) - backtested better

    def __post_init__(self):
        """Validate configuration and set defaults."""
        # Use TOTAL_CAPITAL from config if portfolio_value not specified
        if self.portfolio_value is None:
            self.portfolio_value = TOTAL_CAPITAL

        if self.min_gap_pct <= 0:
            raise ValueError("min_gap_pct must be positive")
        if self.max_gap_pct <= self.min_gap_pct:
            raise ValueError("max_gap_pct must be greater than min_gap_pct")
        if not 0 < self.fill_threshold <= 1.0:
            raise ValueError("fill_threshold must be between 0 and 1")
        if self.stop_loss_pct <= 0:
            raise ValueError("stop_loss_pct must be positive")
        if not 0 < self.max_position_pct <= 1.0:
            raise ValueError("max_position_pct must be between 0 and 1")

    @property
    def max_position_value(self) -> float:
        """Calculate maximum position value in dollars."""
        return self.portfolio_value * self.max_position_pct

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'symbols': self.symbols,
            'min_gap_pct': self.min_gap_pct,
            'max_gap_pct': self.max_gap_pct,
            'gap_detection_start_minute': self.gap_detection_start_minute,
            'gap_detection_end_minute': self.gap_detection_end_minute,
            'max_hold_minutes': self.max_hold_minutes,
            'fill_threshold': self.fill_threshold,
            'stop_loss_pct': self.stop_loss_pct,
            'max_position_pct': self.max_position_pct,
            'max_positions': self.max_positions,
            'portfolio_value': self.portfolio_value,
            'long_only': self.long_only,
            'max_position_value': self.max_position_value
        }


# Default configuration
DEFAULT_CONFIG = GapFillConfig()


if __name__ == "__main__":
    # Test configuration
    config = GapFillConfig()
    print("Gap Fill Strategy Configuration")
    print("=" * 40)
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
