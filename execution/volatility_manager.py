"""
Volatility Manager
==================
Dynamic position sizing and risk management based on realized volatility.

Key features:
- Realized volatility calculation (20-day rolling)
- Position scalar: target_vol / realized_vol, clamped [0.25, 2.0]
- Stop-loss adjustment based on volatility regime
- VIX-based regime overlay

Research basis:
- Volatility targeting improves risk-adjusted returns by 15-25%
- Dynamic stops reduce whipsaw in high-vol environments
- VIX > 25 signals regime shift requiring position reduction

Usage:
    from execution.volatility_manager import VolatilityManager

    vol_manager = VolatilityManager(target_volatility=0.15)

    # Get position scalar
    scalar = vol_manager.get_position_scalar(returns)
    adjusted_size = base_size * scalar

    # Adjust stop loss
    stop = vol_manager.adjust_stop_loss(base_stop=-0.02, returns=returns)
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class VolatilityRegime(Enum):
    """Volatility regime classifications."""
    LOW = "low"           # <15% annualized
    NORMAL = "normal"     # 15-25%
    ELEVATED = "elevated" # 25-35%
    HIGH = "high"         # 35-50%
    EXTREME = "extreme"   # >50%


class VIXRegime(Enum):
    """VIX-based market regime."""
    COMPLACENT = "complacent"   # VIX < 12
    CALM = "calm"               # VIX 12-18
    NORMAL = "normal"           # VIX 18-25
    FEARFUL = "fearful"         # VIX 25-35
    PANIC = "panic"             # VIX > 35


@dataclass
class VolatilityState:
    """Current volatility state for a symbol or portfolio."""
    symbol: str
    timestamp: datetime

    # Realized volatility metrics
    realized_vol_5d: float = 0.0      # 5-day realized vol (annualized)
    realized_vol_20d: float = 0.0     # 20-day realized vol (annualized)
    realized_vol_60d: float = 0.0     # 60-day realized vol (annualized)

    # Regime classification
    vol_regime: VolatilityRegime = VolatilityRegime.NORMAL
    vol_percentile: float = 0.5       # Where current vol sits in historical distribution
    vol_trend: str = "stable"         # "increasing", "decreasing", "stable"

    # Position sizing recommendations
    recommended_scalar: float = 1.0   # Position size multiplier
    max_position_pct: float = 0.10    # Maximum position size as % of portfolio

    # Stop loss adjustments
    stop_multiplier: float = 1.0      # Multiplier for base stop loss

    # VIX overlay (if available)
    vix_level: Optional[float] = None
    vix_regime: Optional[VIXRegime] = None

    def to_dict(self) -> dict:
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            else:
                result[key] = value
        return result


@dataclass
class VolatilityConfig:
    """Configuration for volatility manager."""
    # Target volatility (annualized)
    target_volatility: float = 0.15  # 15% target

    # Position scalar bounds
    min_scalar: float = 0.25         # Never go below 25% of base size
    max_scalar: float = 2.0          # Never exceed 200% of base size

    # Lookback periods
    short_lookback: int = 5          # Days for short-term vol
    medium_lookback: int = 20        # Days for medium-term vol
    long_lookback: int = 60          # Days for long-term vol

    # Regime thresholds (annualized vol)
    low_vol_threshold: float = 0.15
    normal_vol_threshold: float = 0.25
    elevated_vol_threshold: float = 0.35
    high_vol_threshold: float = 0.50

    # VIX thresholds
    vix_complacent: float = 12
    vix_calm: float = 18
    vix_normal: float = 25
    vix_fearful: float = 35

    # Stop loss adjustment
    min_stop_multiplier: float = 0.5   # Tighten stops in low vol
    max_stop_multiplier: float = 2.0   # Widen stops in high vol

    # Maximum position sizes by regime
    max_position_by_regime: Dict[str, float] = None

    def __post_init__(self):
        if self.max_position_by_regime is None:
            self.max_position_by_regime = {
                'low': 0.15,       # Can take larger positions in calm markets
                'normal': 0.10,
                'elevated': 0.07,
                'high': 0.05,
                'extreme': 0.03   # Minimal positions in extreme vol
            }


# =============================================================================
# VOLATILITY MANAGER
# =============================================================================

class VolatilityManager:
    """
    Manages position sizing and risk based on realized volatility.

    Core principle: Scale positions inversely with volatility to maintain
    consistent dollar risk across different market regimes.
    """

    def __init__(
        self,
        config: VolatilityConfig = None,
        db_path: Optional[Path] = None
    ):
        """
        Initialize volatility manager.

        Args:
            config: Volatility configuration
            db_path: Path to database for persistence
        """
        self.config = config or VolatilityConfig()
        from config import DATABASES
        self.db_path = db_path or DATABASES.get("volatility", Path(__file__).parent.parent / "db" / "volatility.db")

        # Cache for volatility states
        self._state_cache: Dict[str, VolatilityState] = {}
        self._historical_vol: Dict[str, pd.Series] = {}

        # Current VIX level (updated externally)
        self._current_vix: Optional[float] = None

        # Initialize database
        self._init_db()

    def _init_db(self):
        """Initialize database tables."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS volatility_history (
                    symbol TEXT,
                    date TEXT,
                    realized_vol_5d REAL,
                    realized_vol_20d REAL,
                    realized_vol_60d REAL,
                    vol_regime TEXT,
                    position_scalar REAL,
                    PRIMARY KEY (symbol, date)
                );

                CREATE TABLE IF NOT EXISTS regime_changes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    timestamp TEXT,
                    old_regime TEXT,
                    new_regime TEXT,
                    vol_level REAL
                );

                CREATE INDEX IF NOT EXISTS idx_vol_symbol
                ON volatility_history(symbol);

                CREATE INDEX IF NOT EXISTS idx_vol_date
                ON volatility_history(date);
            """)

    # =========================================================================
    # CORE VOLATILITY CALCULATIONS
    # =========================================================================

    def calculate_realized_vol(
        self,
        returns: pd.Series,
        lookback: int = 20,
        annualize: bool = True
    ) -> float:
        """
        Calculate realized volatility from returns.

        Args:
            returns: Daily returns series
            lookback: Number of days to look back
            annualize: Whether to annualize (multiply by sqrt(252))

        Returns:
            Realized volatility (annualized if specified)
        """
        if len(returns) < lookback:
            lookback = len(returns)

        if lookback < 2:
            return 0.0

        recent_returns = returns.tail(lookback)
        vol = recent_returns.std()

        if annualize:
            vol *= np.sqrt(252)

        return float(vol) if not np.isnan(vol) else 0.0

    def calculate_volatility_state(
        self,
        symbol: str,
        returns: pd.Series,
        vix_level: Optional[float] = None
    ) -> VolatilityState:
        """
        Calculate complete volatility state for a symbol.

        Args:
            symbol: Symbol identifier
            returns: Daily returns series
            vix_level: Current VIX level (optional)

        Returns:
            VolatilityState with all metrics
        """
        state = VolatilityState(
            symbol=symbol,
            timestamp=datetime.now()
        )

        # Calculate realized vol at different horizons
        state.realized_vol_5d = self.calculate_realized_vol(
            returns, self.config.short_lookback
        )
        state.realized_vol_20d = self.calculate_realized_vol(
            returns, self.config.medium_lookback
        )
        state.realized_vol_60d = self.calculate_realized_vol(
            returns, self.config.long_lookback
        )

        # Determine regime from 20-day vol
        state.vol_regime = self._classify_vol_regime(state.realized_vol_20d)

        # Calculate vol percentile (where does current vol sit historically?)
        state.vol_percentile = self._calculate_vol_percentile(
            symbol, state.realized_vol_20d, returns
        )

        # Determine vol trend
        state.vol_trend = self._determine_vol_trend(
            state.realized_vol_5d,
            state.realized_vol_20d,
            state.realized_vol_60d
        )

        # Calculate position scalar
        state.recommended_scalar = self.get_position_scalar(
            returns, use_cached=False
        )

        # Set max position based on regime
        state.max_position_pct = self.config.max_position_by_regime.get(
            state.vol_regime.value, 0.10
        )

        # Calculate stop multiplier
        state.stop_multiplier = self._calculate_stop_multiplier(state.vol_regime)

        # VIX overlay
        vix = vix_level or self._current_vix
        if vix is not None:
            state.vix_level = vix
            state.vix_regime = self._classify_vix_regime(vix)

            # Adjust scalar if VIX regime is worse than vol regime
            if state.vix_regime in [VIXRegime.FEARFUL, VIXRegime.PANIC]:
                vix_scalar = 0.5 if state.vix_regime == VIXRegime.PANIC else 0.7
                state.recommended_scalar = min(
                    state.recommended_scalar,
                    vix_scalar
                )

        # Cache the state
        self._state_cache[symbol] = state

        # Persist to database
        self._save_volatility_state(state)

        return state

    def _classify_vol_regime(self, vol: float) -> VolatilityRegime:
        """Classify volatility into regime."""
        if vol < self.config.low_vol_threshold:
            return VolatilityRegime.LOW
        elif vol < self.config.normal_vol_threshold:
            return VolatilityRegime.NORMAL
        elif vol < self.config.elevated_vol_threshold:
            return VolatilityRegime.ELEVATED
        elif vol < self.config.high_vol_threshold:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREME

    def _classify_vix_regime(self, vix: float) -> VIXRegime:
        """Classify VIX level into regime."""
        if vix < self.config.vix_complacent:
            return VIXRegime.COMPLACENT
        elif vix < self.config.vix_calm:
            return VIXRegime.CALM
        elif vix < self.config.vix_normal:
            return VIXRegime.NORMAL
        elif vix < self.config.vix_fearful:
            return VIXRegime.FEARFUL
        else:
            return VIXRegime.PANIC

    def _calculate_vol_percentile(
        self,
        symbol: str,
        current_vol: float,
        returns: pd.Series
    ) -> float:
        """Calculate where current vol sits in historical distribution."""
        if len(returns) < 60:
            return 0.5  # Not enough history

        # Calculate rolling 20-day vol for history
        rolling_vol = returns.rolling(20).std() * np.sqrt(252)
        rolling_vol = rolling_vol.dropna()

        if len(rolling_vol) < 20:
            return 0.5

        # Store for future reference
        self._historical_vol[symbol] = rolling_vol

        # Calculate percentile
        percentile = (rolling_vol < current_vol).mean()
        return float(percentile)

    def _determine_vol_trend(
        self,
        vol_5d: float,
        vol_20d: float,
        vol_60d: float
    ) -> str:
        """Determine if volatility is increasing, decreasing, or stable."""
        # Compare short-term to medium-term
        if vol_5d > vol_20d * 1.2:
            return "increasing"
        elif vol_5d < vol_20d * 0.8:
            return "decreasing"
        else:
            return "stable"

    def _calculate_stop_multiplier(self, regime: VolatilityRegime) -> float:
        """Calculate stop loss multiplier based on regime."""
        multipliers = {
            VolatilityRegime.LOW: 0.7,        # Tighter stops in calm markets
            VolatilityRegime.NORMAL: 1.0,
            VolatilityRegime.ELEVATED: 1.3,
            VolatilityRegime.HIGH: 1.6,
            VolatilityRegime.EXTREME: 2.0     # Wider stops to avoid whipsaw
        }
        return multipliers.get(regime, 1.0)

    # =========================================================================
    # POSITION SIZING
    # =========================================================================

    def get_position_scalar(
        self,
        returns: pd.Series,
        use_cached: bool = True,
        symbol: Optional[str] = None
    ) -> float:
        """
        Get position size scalar based on volatility targeting.

        Formula: scalar = target_vol / realized_vol
        Clamped to [min_scalar, max_scalar]

        Args:
            returns: Daily returns series
            use_cached: Whether to use cached state
            symbol: Symbol for cache lookup

        Returns:
            Position scalar (multiply base position by this)
        """
        # Check cache
        if use_cached and symbol and symbol in self._state_cache:
            cached = self._state_cache[symbol]
            # Use cache if less than 1 hour old
            if (datetime.now() - cached.timestamp).total_seconds() < 3600:
                return cached.recommended_scalar

        # Calculate realized vol
        realized_vol = self.calculate_realized_vol(
            returns, self.config.medium_lookback
        )

        if realized_vol <= 0:
            return 1.0

        # Calculate scalar
        scalar = self.config.target_volatility / realized_vol

        # Clamp to bounds
        scalar = max(self.config.min_scalar, min(self.config.max_scalar, scalar))

        return float(scalar)

    def adjust_position_size(
        self,
        base_size: float,
        returns: pd.Series,
        symbol: Optional[str] = None,
        max_position_value: Optional[float] = None
    ) -> Tuple[float, VolatilityState]:
        """
        Adjust position size based on volatility.

        Args:
            base_size: Base position size (shares or dollars)
            returns: Daily returns series
            symbol: Symbol identifier
            max_position_value: Maximum position value (optional cap)

        Returns:
            Tuple of (adjusted_size, volatility_state)
        """
        # Get full volatility state
        state = self.calculate_volatility_state(
            symbol or "UNKNOWN",
            returns
        )

        # Apply scalar
        adjusted_size = base_size * state.recommended_scalar

        # Apply max position cap if specified
        if max_position_value is not None:
            adjusted_size = min(adjusted_size, max_position_value)

        return adjusted_size, state

    # =========================================================================
    # STOP LOSS MANAGEMENT
    # =========================================================================

    def adjust_stop_loss(
        self,
        base_stop: float,
        returns: pd.Series,
        symbol: Optional[str] = None
    ) -> float:
        """
        Adjust stop loss based on volatility regime.

        In high volatility: widen stops to avoid whipsaw
        In low volatility: tighten stops for better risk control

        Args:
            base_stop: Base stop loss percentage (negative, e.g., -0.02 for -2%)
            returns: Daily returns series
            symbol: Symbol identifier

        Returns:
            Adjusted stop loss percentage
        """
        # Get or calculate volatility state
        if symbol and symbol in self._state_cache:
            state = self._state_cache[symbol]
        else:
            state = self.calculate_volatility_state(
                symbol or "UNKNOWN",
                returns
            )

        # Apply multiplier (base_stop is negative, so multiply directly)
        adjusted_stop = base_stop * state.stop_multiplier

        # Ensure stop isn't too tight or too wide
        min_stop = -0.01  # At least 1% stop
        max_stop = -0.15  # No more than 15% stop

        adjusted_stop = max(max_stop, min(min_stop, adjusted_stop))

        return adjusted_stop

    def calculate_atr_stop(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        atr_multiplier: float = 2.0,
        lookback: int = 14
    ) -> float:
        """
        Calculate ATR-based stop loss.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            atr_multiplier: Multiplier for ATR
            lookback: ATR lookback period

        Returns:
            Stop distance as percentage of current price
        """
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate ATR
        atr = true_range.rolling(lookback).mean().iloc[-1]

        # Convert to percentage
        current_price = close.iloc[-1]
        stop_pct = -(atr * atr_multiplier) / current_price

        return float(stop_pct)

    # =========================================================================
    # VIX INTEGRATION
    # =========================================================================

    def update_vix(self, vix_level: float):
        """Update current VIX level."""
        self._current_vix = vix_level
        logger.debug(f"VIX updated to {vix_level:.2f}")

    def get_vix_adjustment(self) -> float:
        """
        Get position adjustment factor based on VIX.

        Returns:
            Multiplier for position size (0.3 to 1.0)
        """
        if self._current_vix is None:
            return 1.0

        vix = self._current_vix

        if vix < 15:
            return 1.0      # Normal sizing
        elif vix < 20:
            return 0.9      # Slight reduction
        elif vix < 25:
            return 0.8      # Moderate reduction
        elif vix < 30:
            return 0.6      # Significant reduction
        elif vix < 40:
            return 0.4      # Major reduction
        else:
            return 0.3      # Minimal positions in panic

    def should_pause_trading(self) -> Tuple[bool, str]:
        """
        Check if trading should be paused based on volatility.

        Returns:
            Tuple of (should_pause, reason)
        """
        if self._current_vix is not None and self._current_vix > 40:
            return True, f"VIX at {self._current_vix:.1f} - extreme fear"

        # Check if any cached states show extreme vol
        for symbol, state in self._state_cache.items():
            if state.vol_regime == VolatilityRegime.EXTREME:
                return True, f"{symbol} in extreme volatility regime"

        return False, ""

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def _save_volatility_state(self, state: VolatilityState):
        """Save volatility state to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO volatility_history
                    (symbol, date, realized_vol_5d, realized_vol_20d,
                     realized_vol_60d, vol_regime, position_scalar)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    state.symbol,
                    state.timestamp.strftime('%Y-%m-%d'),
                    state.realized_vol_5d,
                    state.realized_vol_20d,
                    state.realized_vol_60d,
                    state.vol_regime.value,
                    state.recommended_scalar
                ))
        except Exception as e:
            logger.debug(f"Failed to save volatility state: {e}")

    def get_volatility_history(
        self,
        symbol: str,
        days: int = 30
    ) -> pd.DataFrame:
        """Get historical volatility data for a symbol."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query("""
                    SELECT * FROM volatility_history
                    WHERE symbol = ?
                    ORDER BY date DESC
                    LIMIT ?
                """, conn, params=(symbol, days))
                return df
        except Exception as e:
            logger.warning(f"Failed to get volatility history: {e}")
            return pd.DataFrame()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_volatility_manager(
    target_vol: float = 0.15,
    min_scalar: float = 0.25,
    max_scalar: float = 2.0
) -> VolatilityManager:
    """Create a volatility manager with custom settings."""
    config = VolatilityConfig(
        target_volatility=target_vol,
        min_scalar=min_scalar,
        max_scalar=max_scalar
    )
    return VolatilityManager(config=config)


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate volatility manager."""
    print("=" * 60)
    print("Volatility Manager Demo")
    print("=" * 60)

    # Create manager
    manager = VolatilityManager()

    # Simulate returns for different volatility regimes
    np.random.seed(42)

    # Low volatility period
    low_vol_returns = pd.Series(np.random.normal(0.001, 0.008, 60))  # ~12% annual

    # Normal volatility period
    normal_returns = pd.Series(np.random.normal(0.0005, 0.012, 60))  # ~19% annual

    # High volatility period
    high_vol_returns = pd.Series(np.random.normal(0, 0.025, 60))  # ~40% annual

    print("\n--- Low Volatility Scenario ---")
    state_low = manager.calculate_volatility_state("TEST_LOW", low_vol_returns)
    print(f"Realized Vol (20d): {state_low.realized_vol_20d:.1%}")
    print(f"Regime: {state_low.vol_regime.value}")
    print(f"Position Scalar: {state_low.recommended_scalar:.2f}x")
    print(f"Stop Multiplier: {state_low.stop_multiplier:.2f}x")
    print(f"Max Position: {state_low.max_position_pct:.1%}")

    print("\n--- Normal Volatility Scenario ---")
    state_normal = manager.calculate_volatility_state("TEST_NORMAL", normal_returns)
    print(f"Realized Vol (20d): {state_normal.realized_vol_20d:.1%}")
    print(f"Regime: {state_normal.vol_regime.value}")
    print(f"Position Scalar: {state_normal.recommended_scalar:.2f}x")
    print(f"Stop Multiplier: {state_normal.stop_multiplier:.2f}x")

    print("\n--- High Volatility Scenario ---")
    state_high = manager.calculate_volatility_state("TEST_HIGH", high_vol_returns)
    print(f"Realized Vol (20d): {state_high.realized_vol_20d:.1%}")
    print(f"Regime: {state_high.vol_regime.value}")
    print(f"Position Scalar: {state_high.recommended_scalar:.2f}x")
    print(f"Stop Multiplier: {state_high.stop_multiplier:.2f}x")
    print(f"Max Position: {state_high.max_position_pct:.1%}")

    print("\n--- VIX Integration ---")
    manager.update_vix(28.5)
    print(f"VIX Level: 28.5")
    print(f"VIX Adjustment: {manager.get_vix_adjustment():.1%}")

    pause, reason = manager.should_pause_trading()
    print(f"Should Pause: {pause} ({reason if pause else 'OK'})")

    print("\n--- Stop Loss Adjustment ---")
    base_stop = -0.02  # 2% stop
    for returns, name in [(low_vol_returns, "Low Vol"),
                           (normal_returns, "Normal"),
                           (high_vol_returns, "High Vol")]:
        adjusted = manager.adjust_stop_loss(base_stop, returns)
        print(f"{name}: Base {base_stop:.1%} -> Adjusted {adjusted:.1%}")

    print("\n" + "=" * 60)
    print("Demo Complete")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
