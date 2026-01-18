"""
Rapid Gain Scaler
=================
Automatically trims positions that gain quickly after entry.

Logic:
- Monitor positions opened within the last 24 hours
- If a position gains >= threshold (default 3%) within that window, trim a portion
- Each position can only be trimmed ONCE to avoid squashing runners
- After trim, remaining position follows normal exit rules (trailing stop, take profit)

Usage:
    scaler = RapidGainScaler(broker=alpaca_connector)

    # Check and execute any needed trims
    trimmed = scaler.check_and_trim_positions(positions)
"""

import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import math

logger = logging.getLogger(__name__)


@dataclass
class RapidGainConfig:
    """Configuration for rapid gain scaling."""

    # Gain threshold to trigger trim (as decimal, e.g., 0.03 = 3%)
    gain_threshold: float = 0.03

    # Time window from position open to monitor for rapid gains (hours)
    time_window_hours: int = 24

    # Fraction of position to trim when triggered (e.g., 0.33 = 1/3)
    trim_fraction: float = 0.33

    # Minimum shares to keep after trim (won't trim below this)
    min_shares_remaining: int = 1

    # Minimum shares required to consider trimming
    min_shares_to_trim: int = 3

    def __post_init__(self):
        if not 0 < self.gain_threshold < 1:
            raise ValueError("gain_threshold must be between 0 and 1")
        if not 0 < self.trim_fraction < 1:
            raise ValueError("trim_fraction must be between 0 and 1")


@dataclass
class TrimResult:
    """Result of a trim operation."""
    symbol: str
    shares_before: int
    shares_trimmed: int
    shares_remaining: int
    trim_price: float
    gain_pct: float
    profit_locked: float
    success: bool
    error: Optional[str] = None


class RapidGainScaler:
    """
    Monitors positions for rapid gains and automatically trims to lock in profits.

    Key design principle: Each position is only trimmed ONCE to avoid
    repeatedly cutting winners. After the initial trim, the remaining
    position follows normal trailing stop / take profit rules.
    """

    def __init__(
        self,
        broker,
        config: RapidGainConfig = None,
        db_path: str = None
    ):
        """
        Initialize the rapid gain scaler.

        Args:
            broker: AlpacaConnector instance for executing trades
            config: RapidGainConfig with thresholds and parameters
            db_path: Path to trades database for tracking scaled positions
        """
        self.broker = broker
        self.config = config or RapidGainConfig()

        if db_path is None:
            from config import DATABASES
            db_path = DATABASES.get('trades')
        self.db_path = db_path

        logger.info(
            f"RapidGainScaler initialized: "
            f"threshold={self.config.gain_threshold:.1%}, "
            f"window={self.config.time_window_hours}h, "
            f"trim={self.config.trim_fraction:.0%}"
        )

    def check_and_trim_positions(
        self,
        positions: List[Dict[str, Any]]
    ) -> List[TrimResult]:
        """
        Check all positions for rapid gains and trim if needed.

        Args:
            positions: List of position dicts with keys:
                - symbol, quantity, entry_price, current_price
                - opened_at (datetime or ISO string)
                - scaled_at (datetime/string or None if never scaled)

        Returns:
            List of TrimResult for any positions that were trimmed
        """
        results = []

        for pos in positions:
            result = self._check_position(pos)
            if result:
                results.append(result)

        return results

    def _check_position(self, pos: Dict[str, Any]) -> Optional[TrimResult]:
        """Check a single position and trim if criteria met."""
        symbol = pos.get('symbol')

        # Skip if already scaled (one-time only)
        if pos.get('scaled_at'):
            logger.debug(f"{symbol}: Already scaled, skipping")
            return None

        # Parse opened_at timestamp
        opened_at = pos.get('opened_at')
        if isinstance(opened_at, str):
            try:
                opened_at = datetime.fromisoformat(opened_at.replace('Z', '+00:00'))
                # Make naive for comparison
                if opened_at.tzinfo:
                    opened_at = opened_at.replace(tzinfo=None)
            except ValueError:
                logger.warning(f"{symbol}: Invalid opened_at timestamp")
                return None

        if not opened_at:
            logger.debug(f"{symbol}: No opened_at timestamp")
            return None

        # Check if within time window
        now = datetime.now()
        age_hours = (now - opened_at).total_seconds() / 3600

        if age_hours > self.config.time_window_hours:
            logger.debug(f"{symbol}: Outside time window ({age_hours:.1f}h > {self.config.time_window_hours}h)")
            return None

        # Calculate gain
        entry_price = float(pos.get('entry_price', 0))
        current_price = float(pos.get('current_price', 0))
        quantity = int(pos.get('quantity', 0))

        if entry_price <= 0 or current_price <= 0 or quantity <= 0:
            return None

        gain_pct = (current_price - entry_price) / entry_price

        # Check if gain threshold met
        if gain_pct < self.config.gain_threshold:
            logger.debug(f"{symbol}: Gain {gain_pct:.2%} below threshold {self.config.gain_threshold:.1%}")
            return None

        # Check minimum shares
        if quantity < self.config.min_shares_to_trim:
            logger.debug(f"{symbol}: Not enough shares to trim ({quantity} < {self.config.min_shares_to_trim})")
            return None

        # Calculate trim amount
        shares_to_trim = max(1, math.floor(quantity * self.config.trim_fraction))
        shares_remaining = quantity - shares_to_trim

        # Ensure minimum shares remaining
        if shares_remaining < self.config.min_shares_remaining:
            shares_to_trim = quantity - self.config.min_shares_remaining
            shares_remaining = self.config.min_shares_remaining

        if shares_to_trim <= 0:
            logger.debug(f"{symbol}: No shares to trim after constraints")
            return None

        # Execute the trim
        logger.info(
            f"RAPID GAIN TRIM: {symbol} gained {gain_pct:.2%} in {age_hours:.1f}h - "
            f"trimming {shares_to_trim} of {quantity} shares"
        )

        return self._execute_trim(
            symbol=symbol,
            shares_to_trim=shares_to_trim,
            shares_before=quantity,
            current_price=current_price,
            entry_price=entry_price,
            gain_pct=gain_pct
        )

    def _execute_trim(
        self,
        symbol: str,
        shares_to_trim: int,
        shares_before: int,
        current_price: float,
        entry_price: float,
        gain_pct: float
    ) -> TrimResult:
        """Execute the trim order via broker."""
        try:
            # Place market sell order
            order = self.broker.submit_market_order(
                symbol=symbol,
                qty=shares_to_trim,
                side='sell',
                time_in_force='day'
            )

            if order:
                profit_locked = shares_to_trim * (current_price - entry_price)

                # Get filled price if available, otherwise use current price
                filled_price = order.filled_avg_price if order.filled_avg_price else current_price

                # Record the trade in database
                self._record_trim_trade(
                    symbol=symbol,
                    shares_trimmed=shares_to_trim,
                    entry_price=entry_price,
                    exit_price=filled_price,
                    profit_locked=profit_locked,
                    gain_pct=gain_pct
                )

                # Update position quantity and mark as scaled
                self._update_position_after_trim(
                    symbol=symbol,
                    shares_remaining=shares_before - shares_to_trim
                )

                logger.info(
                    f"TRIM EXECUTED: {symbol} sold {shares_to_trim} shares @ ~${filled_price:.2f} "
                    f"(locked ${profit_locked:.2f} profit)"
                )

                return TrimResult(
                    symbol=symbol,
                    shares_before=shares_before,
                    shares_trimmed=shares_to_trim,
                    shares_remaining=shares_before - shares_to_trim,
                    trim_price=filled_price,
                    gain_pct=gain_pct,
                    profit_locked=profit_locked,
                    success=True
                )
            else:
                return TrimResult(
                    symbol=symbol,
                    shares_before=shares_before,
                    shares_trimmed=0,
                    shares_remaining=shares_before,
                    trim_price=current_price,
                    gain_pct=gain_pct,
                    profit_locked=0,
                    success=False,
                    error="Order rejected or failed"
                )

        except Exception as e:
            logger.error(f"Failed to execute trim for {symbol}: {e}")
            return TrimResult(
                symbol=symbol,
                shares_before=shares_before,
                shares_trimmed=0,
                shares_remaining=shares_before,
                trim_price=current_price,
                gain_pct=gain_pct,
                profit_locked=0,
                success=False,
                error=str(e)
            )

    def _mark_position_scaled(self, symbol: str) -> None:
        """Mark a position as scaled in the database."""
        try:
            import sqlite3

            if not self.db_path:
                return

            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Update the position with scaled_at timestamp
            cursor.execute("""
                UPDATE positions
                SET scaled_at = ?
                WHERE symbol = ? AND status = 'open'
            """, (datetime.now().isoformat(), symbol))

            conn.commit()
            conn.close()

            logger.debug(f"Marked {symbol} as scaled in database")

        except Exception as e:
            logger.warning(f"Failed to mark {symbol} as scaled: {e}")

    def _record_trim_trade(
        self,
        symbol: str,
        shares_trimmed: int,
        entry_price: float,
        exit_price: float,
        profit_locked: float,
        gain_pct: float
    ) -> None:
        """Record the trim as a trade in the trades table using original strategy."""
        try:
            import sqlite3

            if not self.db_path:
                return

            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            now = datetime.now().isoformat()
            pnl_pct = gain_pct * 100

            # Look up original strategy from positions table
            cursor.execute("""
                SELECT strategy_name FROM positions
                WHERE symbol = ? AND status = 'open'
                LIMIT 1
            """, (symbol,))
            row = cursor.fetchone()
            original_strategy = row[0] if row else 'unknown'

            cursor.execute("""
                INSERT INTO trades (
                    timestamp, symbol, strategy, side, quantity,
                    entry_price, exit_price, exit_timestamp,
                    pnl, pnl_pct, status, exit_reason,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now, symbol, original_strategy, 'SELL', shares_trimmed,
                entry_price, exit_price, now,
                profit_locked, pnl_pct, 'CLOSED', 'rapid_gain_trim',
                now, now
            ))

            conn.commit()
            conn.close()

            # Update strategy stats
            self._update_strategy_stats(original_strategy, profit_locked, gain_pct)

            logger.info(f"Recorded trim trade for {symbol} ({original_strategy}): {shares_trimmed} shares, P&L ${profit_locked:.2f}")

        except Exception as e:
            logger.warning(f"Failed to record trim trade for {symbol}: {e}")

    def _update_strategy_stats(
        self,
        strategy: str,
        pnl: float,
        pnl_pct: float
    ) -> None:
        """Update strategy performance stats after a trim."""
        try:
            import sqlite3
            from config import DATABASES

            perf_db = DATABASES.get('performance')
            if not perf_db:
                return

            is_win = pnl > 0
            conn = sqlite3.connect(str(perf_db))
            cursor = conn.cursor()
            now = datetime.now().isoformat()

            cursor.execute("""
                INSERT INTO strategy_stats (strategy, total_trades, winning_trades, losing_trades,
                                           total_pnl, last_trade_date, updated_at)
                VALUES (?, 1, ?, ?, ?, ?, ?)
                ON CONFLICT(strategy) DO UPDATE SET
                    total_trades = total_trades + 1,
                    winning_trades = winning_trades + ?,
                    losing_trades = losing_trades + ?,
                    total_pnl = total_pnl + ?,
                    avg_pnl = (total_pnl + ?) / (total_trades + 1),
                    win_rate = CAST(winning_trades + ? AS REAL) / (total_trades + 1),
                    best_trade = MAX(best_trade, ?),
                    worst_trade = MIN(worst_trade, ?),
                    last_trade_date = ?,
                    updated_at = ?
            """, (
                strategy,
                1 if is_win else 0, 0 if is_win else 1, pnl, now, now,
                1 if is_win else 0, 0 if is_win else 1, pnl, pnl,
                1 if is_win else 0,
                pnl if pnl > 0 else None,
                pnl if pnl < 0 else None,
                now, now
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.warning(f"Failed to update strategy stats: {e}")

    def _update_position_after_trim(
        self,
        symbol: str,
        shares_remaining: int
    ) -> None:
        """Update position quantity and mark as scaled after trim."""
        try:
            import sqlite3

            if not self.db_path:
                return

            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            now = datetime.now().isoformat()

            # Update quantity and mark as scaled
            cursor.execute("""
                UPDATE positions
                SET quantity = ?,
                    scaled_at = ?
                WHERE symbol = ? AND status = 'open'
            """, (shares_remaining, now, symbol))

            rows_updated = cursor.rowcount
            conn.commit()
            conn.close()

            if rows_updated > 0:
                logger.info(f"Updated {symbol} position: {shares_remaining} shares remaining, marked as scaled")
            else:
                logger.error(f"CRITICAL: Failed to update {symbol} position in DB - no rows matched! Trim may repeat.")

        except Exception as e:
            logger.error(f"CRITICAL: Database update failed for {symbol} after trim: {e} - Trim may repeat!")

    def get_eligible_positions(self) -> List[Dict[str, Any]]:
        """
        Get positions eligible for rapid gain checking from database.

        Returns:
            List of position dicts with required fields
        """
        try:
            import sqlite3

            if not self.db_path:
                return []

            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get open positions that haven't been scaled
            cursor.execute("""
                SELECT
                    symbol,
                    quantity,
                    entry_price,
                    current_price,
                    opened_at,
                    scaled_at
                FROM positions
                WHERE status = 'open'
                  AND scaled_at IS NULL
            """)

            positions = [dict(row) for row in cursor.fetchall()]
            conn.close()

            return positions

        except Exception as e:
            logger.error(f"Failed to get eligible positions: {e}")
            return []


def check_rapid_gains(broker, config: RapidGainConfig = None) -> List[TrimResult]:
    """
    Convenience function to check and trim rapid gains.

    Args:
        broker: AlpacaConnector instance
        config: Optional RapidGainConfig

    Returns:
        List of TrimResult for any executed trims
    """
    scaler = RapidGainScaler(broker=broker, config=config)
    positions = scaler.get_eligible_positions()

    if not positions:
        logger.debug("No eligible positions for rapid gain check")
        return []

    return scaler.check_and_trim_positions(positions)
