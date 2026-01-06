"""
Gap Detection Module
====================
Detects and tracks overnight price gaps for the gap-fill strategy.

A gap occurs when a stock opens at a significantly different price
from its previous close. This module identifies tradeable gaps
within configured thresholds.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class Gap:
    """
    Represents a detected overnight gap.

    A gap is the difference between today's open and yesterday's close.
    This class tracks the gap details and provides methods to calculate
    how much of the gap has filled.

    Attributes:
        symbol: Stock symbol
        previous_close: Previous day's closing price
        open_price: Today's opening price
        gap_pct: Gap size as percentage (positive = gap up, negative = gap down)
        gap_direction: 'up' or 'down'
        detected_at: Timestamp when gap was detected
        metadata: Additional tracking data
    """
    symbol: str
    previous_close: float
    open_price: float
    gap_pct: float
    gap_direction: str  # 'up' or 'down'
    detected_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate gap data."""
        if self.previous_close <= 0:
            raise ValueError(f"previous_close must be positive: {self.previous_close}")
        if self.open_price <= 0:
            raise ValueError(f"open_price must be positive: {self.open_price}")
        if self.gap_direction not in ('up', 'down'):
            raise ValueError(f"gap_direction must be 'up' or 'down': {self.gap_direction}")

    @property
    def gap_dollars(self) -> float:
        """Gap size in dollars."""
        return self.open_price - self.previous_close

    @property
    def target_price(self) -> float:
        """
        Target price for a complete gap fill.

        This is the previous close - where the gap would be 100% filled.
        """
        return self.previous_close

    def fill_percentage(self, current_price: float) -> float:
        """
        Calculate percentage of gap that has filled.

        Returns a value from 0.0 (no fill) to 1.0+ (completely filled).
        Values > 1.0 indicate the price has moved past the previous close.

        For a gap UP:
            - Price at open = 0% filled
            - Price at previous_close = 100% filled
            - Price below previous_close = >100% filled

        For a gap DOWN:
            - Price at open = 0% filled
            - Price at previous_close = 100% filled
            - Price above previous_close = >100% filled

        Args:
            current_price: Current market price

        Returns:
            Fill percentage as decimal (0.0 to 1.0+)
        """
        if abs(self.gap_dollars) < 0.0001:  # Avoid division by zero
            return 1.0

        if self.gap_direction == 'up':
            # Gap up: price needs to come DOWN to fill
            # Fill = (open - current) / (open - prev_close)
            price_moved = self.open_price - current_price
            gap_size = self.open_price - self.previous_close
        else:
            # Gap down: price needs to come UP to fill
            # Fill = (current - open) / (prev_close - open)
            price_moved = current_price - self.open_price
            gap_size = self.previous_close - self.open_price

        return price_moved / gap_size if gap_size != 0 else 1.0

    def is_filled(self, current_price: float, threshold: float = 0.75) -> bool:
        """
        Check if gap has filled to the specified threshold.

        Args:
            current_price: Current market price
            threshold: Fill threshold (default 0.75 = 75%)

        Returns:
            True if gap has filled to threshold
        """
        return self.fill_percentage(current_price) >= threshold

    def get_fill_target_price(self, threshold: float = 0.75) -> float:
        """
        Calculate the price at which the gap will be considered filled.

        Args:
            threshold: Fill threshold (0.0 to 1.0)

        Returns:
            Target price for threshold fill
        """
        gap_size = abs(self.gap_dollars)
        fill_distance = gap_size * threshold

        if self.gap_direction == 'up':
            # Gap up: target is below open
            return self.open_price - fill_distance
        else:
            # Gap down: target is above open
            return self.open_price + fill_distance

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/storage."""
        return {
            'symbol': self.symbol,
            'previous_close': self.previous_close,
            'open_price': self.open_price,
            'gap_pct': self.gap_pct,
            'gap_dollars': self.gap_dollars,
            'gap_direction': self.gap_direction,
            'target_price': self.target_price,
            'detected_at': self.detected_at.isoformat(),
            'metadata': self.metadata
        }

    def __repr__(self) -> str:
        return (
            f"Gap({self.symbol}: {self.gap_direction} {abs(self.gap_pct):.2f}%, "
            f"${self.previous_close:.2f} -> ${self.open_price:.2f})"
        )


class GapDetector:
    """
    Detects tradeable gaps at market open.

    Filters gaps to only those within the configured size range,
    which have the highest probability of filling.

    Usage:
        detector = GapDetector(min_gap_pct=0.15, max_gap_pct=0.60)
        gap = detector.detect_gap('SPY', previous_close=450.00, open_price=451.00, timestamp)
        if gap:
            print(f"Detected: {gap}")
    """

    def __init__(
        self,
        min_gap_pct: float = 0.15,
        max_gap_pct: float = 0.60
    ):
        """
        Initialize gap detector.

        Args:
            min_gap_pct: Minimum gap size to consider (filters noise)
            max_gap_pct: Maximum gap size to consider (filters news gaps)
        """
        self.min_gap_pct = min_gap_pct
        self.max_gap_pct = max_gap_pct

        logger.info(
            f"GapDetector initialized: range {min_gap_pct:.2f}% - {max_gap_pct:.2f}%"
        )

    def detect_gap(
        self,
        symbol: str,
        previous_close: float,
        open_price: float,
        timestamp: datetime
    ) -> Optional[Gap]:
        """
        Detect if there's a tradeable gap.

        Checks if the gap between previous close and open price falls
        within the configured range for trading.

        Args:
            symbol: Stock symbol
            previous_close: Previous day's closing price
            open_price: Today's opening price
            timestamp: Detection timestamp

        Returns:
            Gap object if tradeable, None otherwise
        """
        if previous_close <= 0 or open_price <= 0:
            logger.warning(f"{symbol}: Invalid prices - prev: {previous_close}, open: {open_price}")
            return None

        # Calculate gap percentage
        gap_pct = ((open_price - previous_close) / previous_close) * 100
        abs_gap_pct = abs(gap_pct)

        # Determine direction
        gap_direction = 'up' if gap_pct > 0 else 'down'

        # Check if gap is within tradeable range
        if abs_gap_pct < self.min_gap_pct:
            logger.debug(
                f"{symbol}: Gap {gap_pct:+.2f}% too small (min: {self.min_gap_pct}%)"
            )
            return None

        if abs_gap_pct > self.max_gap_pct:
            logger.debug(
                f"{symbol}: Gap {gap_pct:+.2f}% too large (max: {self.max_gap_pct}%)"
            )
            return None

        # Create and return Gap object
        gap = Gap(
            symbol=symbol,
            previous_close=previous_close,
            open_price=open_price,
            gap_pct=gap_pct,
            gap_direction=gap_direction,
            detected_at=timestamp
        )

        logger.info(f"Detected tradeable gap: {gap}")
        return gap

    def is_gap_tradeable(self, gap_pct: float) -> bool:
        """
        Quick check if a gap percentage is within tradeable range.

        Args:
            gap_pct: Gap percentage (can be positive or negative)

        Returns:
            True if within range
        """
        abs_gap = abs(gap_pct)
        return self.min_gap_pct <= abs_gap <= self.max_gap_pct


if __name__ == "__main__":
    # Test gap detection
    logging.basicConfig(level=logging.DEBUG)

    detector = GapDetector(min_gap_pct=0.15, max_gap_pct=0.60)

    # Test cases
    test_cases = [
        ('SPY', 450.00, 451.00),  # 0.22% gap up - tradeable
        ('SPY', 450.00, 449.00),  # -0.22% gap down - tradeable
        ('QQQ', 380.00, 380.50),  # 0.13% gap - too small
        ('QQQ', 380.00, 385.00),  # 1.32% gap - too large
        ('SPY', 450.00, 451.35),  # 0.30% gap up - tradeable
    ]

    print("Gap Detection Tests")
    print("=" * 60)

    for symbol, prev_close, open_price in test_cases:
        gap = detector.detect_gap(
            symbol=symbol,
            previous_close=prev_close,
            open_price=open_price,
            timestamp=datetime.now()
        )

        gap_pct = ((open_price - prev_close) / prev_close) * 100

        if gap:
            print(f"\n{symbol}: ${prev_close:.2f} -> ${open_price:.2f} ({gap_pct:+.2f}%)")
            print(f"  Detected: {gap}")
            print(f"  75% fill target: ${gap.get_fill_target_price(0.75):.2f}")

            # Test fill calculation
            test_price = (prev_close + open_price) / 2  # Midpoint
            fill_pct = gap.fill_percentage(test_price)
            print(f"  At ${test_price:.2f}: {fill_pct:.1%} filled")
        else:
            print(f"\n{symbol}: ${prev_close:.2f} -> ${open_price:.2f} ({gap_pct:+.2f}%) - NOT TRADEABLE")
