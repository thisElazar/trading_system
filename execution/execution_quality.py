"""
Execution Quality Tracking System
=================================
Track and analyze the quality of trade executions.

Features:
- Slippage tracking (expected price vs actual fill price)
- Market impact estimation
- Fill rate analysis by time of day, volatility, order size
- Execution quality reports
- Alerts for poor execution quality

Integration:
- Hook into OrderExecutor after fills
- Feed data to dashboard for visualization
"""

import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import json

import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATABASES, DIRS
from strategies.base import Signal, SignalType
from execution.order_executor import OrderResult, OrderStatus

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ExecutionRecord:
    """Record of a single execution with quality metrics."""
    id: Optional[int] = None
    order_id: str = ""
    symbol: str = ""
    side: str = ""  # 'buy' or 'sell'

    # Price points
    signal_price: float = 0.0
    limit_price: Optional[float] = None
    fill_price: float = 0.0

    # Slippage metrics
    slippage_bps: float = 0.0
    slippage_dollars: float = 0.0

    # Timing
    time_to_fill_seconds: float = 0.0

    # Market conditions
    market_volatility: float = 0.0  # ATR-based or VIX
    spread_at_signal: float = 0.0   # Bid-ask spread

    # Order size context
    order_size: int = 0
    daily_volume_pct: float = 0.0  # Order size as % of daily volume

    # Strategy context
    strategy: str = ""

    # Timestamp
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class ExecutionReport:
    """Summary report of execution quality metrics."""
    period_start: str
    period_end: str
    total_executions: int

    # Slippage statistics
    avg_slippage_bps: float
    median_slippage_bps: float
    max_slippage_bps: float
    total_slippage_dollars: float

    # Timing
    avg_time_to_fill_seconds: float

    # Fill rates
    fill_rate_pct: float
    partial_fill_rate_pct: float

    # Breakdown by side
    buy_slippage_bps: float
    sell_slippage_bps: float

    # Market conditions impact
    low_volatility_slippage_bps: float
    high_volatility_slippage_bps: float

    # Hourly breakdown (dict of hour -> avg slippage)
    hourly_slippage: Dict[int, float] = field(default_factory=dict)

    # Strategy breakdown
    strategy_slippage: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "EXECUTION QUALITY REPORT",
            "=" * 60,
            f"Period: {self.period_start[:10]} to {self.period_end[:10]}",
            f"Total Executions: {self.total_executions}",
            "",
            "SLIPPAGE METRICS",
            "-" * 40,
            f"  Average:  {self.avg_slippage_bps:.2f} bps",
            f"  Median:   {self.median_slippage_bps:.2f} bps",
            f"  Maximum:  {self.max_slippage_bps:.2f} bps",
            f"  Total $:  ${self.total_slippage_dollars:,.2f}",
            "",
            "BY SIDE",
            f"  Buy:      {self.buy_slippage_bps:.2f} bps",
            f"  Sell:     {self.sell_slippage_bps:.2f} bps",
            "",
            "BY VOLATILITY",
            f"  Low Vol:  {self.low_volatility_slippage_bps:.2f} bps",
            f"  High Vol: {self.high_volatility_slippage_bps:.2f} bps",
            "",
            "TIMING",
            f"  Avg Time to Fill: {self.avg_time_to_fill_seconds:.1f} seconds",
            f"  Fill Rate:        {self.fill_rate_pct:.1f}%",
            "",
        ]

        if self.strategy_slippage:
            lines.append("BY STRATEGY")
            for strategy, slippage in sorted(self.strategy_slippage.items()):
                lines.append(f"  {strategy}: {slippage:.2f} bps")

        return "\n".join(lines)


@dataclass
class SlippageStats:
    """Statistics for slippage analysis."""
    count: int = 0
    mean_bps: float = 0.0
    median_bps: float = 0.0
    std_bps: float = 0.0
    min_bps: float = 0.0
    max_bps: float = 0.0
    total_dollars: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AlertSeverity(Enum):
    """Severity levels for execution quality alerts."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ExecutionAlert:
    """Alert for poor execution quality."""
    severity: AlertSeverity
    message: str
    order_id: str
    symbol: str
    slippage_bps: float
    threshold_bps: float
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


# ============================================================================
# MAIN TRACKER CLASS
# ============================================================================

class ExecutionQualityTracker:
    """
    Track and analyze execution quality metrics.

    Usage:
        tracker = ExecutionQualityTracker()

        # After each fill
        tracker.record_execution(order_result, signal, market_data)

        # Generate reports
        report = tracker.get_execution_report(days=30)
        print(report.summary())

        # Check for poor executions
        if tracker.alert_if_poor_execution(threshold_bps=50):
            # Handle alert
            pass
    """

    # Volatility thresholds for bucketing
    LOW_VOLATILITY_VIX = 15.0
    HIGH_VOLATILITY_VIX = 25.0

    def __init__(self, db_path: Path = None):
        """
        Initialize the execution quality tracker.

        Args:
            db_path: Path to trades.db (default from config)
        """
        self.db_path = db_path or DATABASES.get('trades', DIRS['db'] / 'trades.db')
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_tables()

        # Recent alerts cache
        self._recent_alerts: List[ExecutionAlert] = []
        self._max_alerts = 100

        logger.info(f"ExecutionQualityTracker initialized with DB: {self.db_path}")

    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        """Create execution_quality table if it doesn't exist."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS execution_quality (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                signal_price REAL NOT NULL,
                limit_price REAL,
                fill_price REAL NOT NULL,
                slippage_bps REAL NOT NULL,
                slippage_dollars REAL NOT NULL,
                time_to_fill_seconds REAL DEFAULT 0,
                market_volatility REAL DEFAULT 0,
                spread_at_signal REAL DEFAULT 0,
                order_size INTEGER DEFAULT 0,
                daily_volume_pct REAL DEFAULT 0,
                strategy TEXT DEFAULT '',
                timestamp TEXT NOT NULL,
                UNIQUE(order_id)
            )
        """)

        # Create indexes for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_exec_quality_timestamp
            ON execution_quality(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_exec_quality_symbol
            ON execution_quality(symbol)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_exec_quality_strategy
            ON execution_quality(strategy)
        """)

        conn.commit()
        conn.close()

        logger.debug("execution_quality table initialized")

    # ========================================================================
    # CORE METHODS
    # ========================================================================

    def calculate_slippage(
        self,
        signal_price: float,
        fill_price: float,
        side: str,
        order_size: int = 1
    ) -> Tuple[float, float]:
        """
        Calculate slippage in basis points and dollars.

        Positive slippage = unfavorable (paid more / received less than expected)
        Negative slippage = favorable (got better price than expected)

        Args:
            signal_price: Expected price at signal generation
            fill_price: Actual fill price
            side: 'buy' or 'sell'
            order_size: Number of shares

        Returns:
            Tuple of (slippage_bps, slippage_dollars)
        """
        if signal_price <= 0:
            logger.warning(f"Invalid signal_price: {signal_price}")
            return 0.0, 0.0

        price_diff = fill_price - signal_price

        # For buys: positive price_diff = slippage (paid more)
        # For sells: negative price_diff = slippage (received less)
        if side.lower() == 'sell':
            price_diff = -price_diff

        slippage_bps = (price_diff / signal_price) * 10000  # Convert to basis points
        slippage_dollars = price_diff * order_size

        return round(slippage_bps, 2), round(slippage_dollars, 2)

    def record_execution(
        self,
        order_result: OrderResult,
        signal: Signal = None,
        market_data: Dict[str, Any] = None
    ) -> Optional[int]:
        """
        Record an execution with quality metrics.

        Args:
            order_result: Result from OrderExecutor
            signal: Original Signal that triggered the order
            market_data: Dict with market context:
                - 'volatility': VIX or ATR value
                - 'spread': Bid-ask spread at signal time
                - 'volume': Daily volume
                - 'signal_time': When signal was generated

        Returns:
            Execution record ID, or None if failed
        """
        if not order_result.success or order_result.filled_price <= 0:
            logger.debug(f"Skipping execution record for unsuccessful order: {order_result.order_id}")
            return None

        try:
            market_data = market_data or {}

            # Extract signal information
            signal_price = signal.price if signal else order_result.filled_price
            strategy = signal.strategy if signal else ""
            side = 'buy' if signal and signal.signal_type == SignalType.BUY else 'sell'
            order_size = signal.metadata.get('shares', 1) if signal else 1
            symbol = signal.symbol if signal else ""

            # Calculate slippage
            slippage_bps, slippage_dollars = self.calculate_slippage(
                signal_price=signal_price,
                fill_price=order_result.filled_price,
                side=side,
                order_size=order_size
            )

            # Calculate time to fill
            time_to_fill = 0.0
            if order_result.submitted_at and order_result.filled_at:
                time_to_fill = (order_result.filled_at - order_result.submitted_at).total_seconds()
            elif market_data.get('signal_time'):
                time_to_fill = (datetime.now() - market_data['signal_time']).total_seconds()

            # Market context
            volatility = market_data.get('volatility', 0.0)
            spread = market_data.get('spread', 0.0)
            daily_volume = market_data.get('volume', 0)
            daily_volume_pct = (order_size / daily_volume * 100) if daily_volume > 0 else 0.0

            # Create record
            record = ExecutionRecord(
                order_id=order_result.order_id or f"RECORD_{datetime.now().timestamp()}",
                symbol=symbol,
                side=side,
                signal_price=signal_price,
                limit_price=signal.metadata.get('limit_price') if signal else None,
                fill_price=order_result.filled_price,
                slippage_bps=slippage_bps,
                slippage_dollars=slippage_dollars,
                time_to_fill_seconds=time_to_fill,
                market_volatility=volatility,
                spread_at_signal=spread,
                order_size=order_size,
                daily_volume_pct=daily_volume_pct,
                strategy=strategy
            )

            # Save to database
            record_id = self._save_record(record)

            logger.info(
                f"Recorded execution: {symbol} {side} @ {order_result.filled_price:.2f} "
                f"(slippage: {slippage_bps:.1f} bps, ${slippage_dollars:.2f})"
            )

            return record_id

        except Exception as e:
            logger.error(f"Failed to record execution: {e}")
            return None

    def record_execution_direct(
        self,
        order_id: str,
        symbol: str,
        side: str,
        signal_price: float,
        fill_price: float,
        order_size: int = 1,
        strategy: str = "",
        limit_price: float = None,
        time_to_fill_seconds: float = 0.0,
        market_volatility: float = 0.0,
        spread_at_signal: float = 0.0,
        daily_volume: int = 0
    ) -> Optional[int]:
        """
        Record an execution directly with explicit parameters.

        Useful when not using OrderResult/Signal objects.

        Returns:
            Execution record ID, or None if failed
        """
        try:
            slippage_bps, slippage_dollars = self.calculate_slippage(
                signal_price=signal_price,
                fill_price=fill_price,
                side=side,
                order_size=order_size
            )

            daily_volume_pct = (order_size / daily_volume * 100) if daily_volume > 0 else 0.0

            record = ExecutionRecord(
                order_id=order_id,
                symbol=symbol,
                side=side,
                signal_price=signal_price,
                limit_price=limit_price,
                fill_price=fill_price,
                slippage_bps=slippage_bps,
                slippage_dollars=slippage_dollars,
                time_to_fill_seconds=time_to_fill_seconds,
                market_volatility=market_volatility,
                spread_at_signal=spread_at_signal,
                order_size=order_size,
                daily_volume_pct=daily_volume_pct,
                strategy=strategy
            )

            return self._save_record(record)

        except Exception as e:
            logger.error(f"Failed to record execution directly: {e}")
            return None

    def _save_record(self, record: ExecutionRecord) -> int:
        """Save execution record to database."""
        conn = self._get_conn()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO execution_quality (
                    order_id, symbol, side, signal_price, limit_price,
                    fill_price, slippage_bps, slippage_dollars,
                    time_to_fill_seconds, market_volatility, spread_at_signal,
                    order_size, daily_volume_pct, strategy, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.order_id, record.symbol, record.side,
                record.signal_price, record.limit_price, record.fill_price,
                record.slippage_bps, record.slippage_dollars,
                record.time_to_fill_seconds, record.market_volatility,
                record.spread_at_signal, record.order_size,
                record.daily_volume_pct, record.strategy, record.timestamp
            ))

            record_id = cursor.lastrowid
            conn.commit()
            return record_id

        finally:
            conn.close()

    # ========================================================================
    # REPORTING METHODS
    # ========================================================================

    def get_execution_report(self, days: int = 30) -> ExecutionReport:
        """
        Generate comprehensive execution quality report.

        Args:
            days: Number of days to include in report

        Returns:
            ExecutionReport with all metrics
        """
        conn = self._get_conn()

        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        try:
            # Get all records for the period
            df = pd.read_sql_query("""
                SELECT * FROM execution_quality
                WHERE timestamp >= ?
                ORDER BY timestamp
            """, conn, params=(cutoff,))

            if df.empty:
                return self._empty_report(days)

            # Calculate statistics
            total = len(df)

            # Slippage stats
            avg_slippage = df['slippage_bps'].mean()
            median_slippage = df['slippage_bps'].median()
            max_slippage = df['slippage_bps'].max()
            total_dollars = df['slippage_dollars'].sum()

            # Timing
            avg_time = df['time_to_fill_seconds'].mean()

            # By side
            buy_df = df[df['side'] == 'buy']
            sell_df = df[df['side'] == 'sell']
            buy_slippage = buy_df['slippage_bps'].mean() if len(buy_df) > 0 else 0.0
            sell_slippage = sell_df['slippage_bps'].mean() if len(sell_df) > 0 else 0.0

            # By volatility
            low_vol_df = df[df['market_volatility'] < self.LOW_VOLATILITY_VIX]
            high_vol_df = df[df['market_volatility'] >= self.HIGH_VOLATILITY_VIX]
            low_vol_slippage = low_vol_df['slippage_bps'].mean() if len(low_vol_df) > 0 else 0.0
            high_vol_slippage = high_vol_df['slippage_bps'].mean() if len(high_vol_df) > 0 else 0.0

            # Hourly breakdown
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            hourly = df.groupby('hour')['slippage_bps'].mean().to_dict()

            # Strategy breakdown
            strategy_slippage = df.groupby('strategy')['slippage_bps'].mean().to_dict()

            # Fill rates (assume 100% for now - would need order tracking for actual)
            fill_rate = 100.0
            partial_rate = 0.0

            return ExecutionReport(
                period_start=cutoff,
                period_end=datetime.now().isoformat(),
                total_executions=total,
                avg_slippage_bps=round(avg_slippage, 2),
                median_slippage_bps=round(median_slippage, 2),
                max_slippage_bps=round(max_slippage, 2),
                total_slippage_dollars=round(total_dollars, 2),
                avg_time_to_fill_seconds=round(avg_time, 2),
                fill_rate_pct=fill_rate,
                partial_fill_rate_pct=partial_rate,
                buy_slippage_bps=round(buy_slippage, 2),
                sell_slippage_bps=round(sell_slippage, 2),
                low_volatility_slippage_bps=round(low_vol_slippage, 2),
                high_volatility_slippage_bps=round(high_vol_slippage, 2),
                hourly_slippage={k: round(v, 2) for k, v in hourly.items()},
                strategy_slippage={k: round(v, 2) for k, v in strategy_slippage.items()}
            )

        finally:
            conn.close()

    def _empty_report(self, days: int) -> ExecutionReport:
        """Create empty report when no data available."""
        now = datetime.now()
        return ExecutionReport(
            period_start=(now - timedelta(days=days)).isoformat(),
            period_end=now.isoformat(),
            total_executions=0,
            avg_slippage_bps=0.0,
            median_slippage_bps=0.0,
            max_slippage_bps=0.0,
            total_slippage_dollars=0.0,
            avg_time_to_fill_seconds=0.0,
            fill_rate_pct=0.0,
            partial_fill_rate_pct=0.0,
            buy_slippage_bps=0.0,
            sell_slippage_bps=0.0,
            low_volatility_slippage_bps=0.0,
            high_volatility_slippage_bps=0.0,
            hourly_slippage={},
            strategy_slippage={}
        )

    def get_slippage_by_strategy(self, strategy: str, days: int = 30) -> SlippageStats:
        """
        Get slippage statistics for a specific strategy.

        Args:
            strategy: Strategy name
            days: Number of days to analyze

        Returns:
            SlippageStats for the strategy
        """
        conn = self._get_conn()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        try:
            df = pd.read_sql_query("""
                SELECT slippage_bps, slippage_dollars
                FROM execution_quality
                WHERE strategy = ? AND timestamp >= ?
            """, conn, params=(strategy, cutoff))

            if df.empty:
                return SlippageStats()

            return SlippageStats(
                count=len(df),
                mean_bps=round(df['slippage_bps'].mean(), 2),
                median_bps=round(df['slippage_bps'].median(), 2),
                std_bps=round(df['slippage_bps'].std(), 2),
                min_bps=round(df['slippage_bps'].min(), 2),
                max_bps=round(df['slippage_bps'].max(), 2),
                total_dollars=round(df['slippage_dollars'].sum(), 2)
            )

        finally:
            conn.close()

    def get_slippage_by_time_of_day(self, days: int = 30) -> Dict[int, SlippageStats]:
        """
        Get slippage breakdown by hour of day.

        Args:
            days: Number of days to analyze

        Returns:
            Dict mapping hour (0-23) to SlippageStats
        """
        conn = self._get_conn()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        try:
            df = pd.read_sql_query("""
                SELECT slippage_bps, slippage_dollars, timestamp
                FROM execution_quality
                WHERE timestamp >= ?
            """, conn, params=(cutoff,))

            if df.empty:
                return {}

            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour

            result = {}
            for hour, group in df.groupby('hour'):
                result[int(hour)] = SlippageStats(
                    count=len(group),
                    mean_bps=round(group['slippage_bps'].mean(), 2),
                    median_bps=round(group['slippage_bps'].median(), 2),
                    std_bps=round(group['slippage_bps'].std(), 2) if len(group) > 1 else 0.0,
                    min_bps=round(group['slippage_bps'].min(), 2),
                    max_bps=round(group['slippage_bps'].max(), 2),
                    total_dollars=round(group['slippage_dollars'].sum(), 2)
                )

            return result

        finally:
            conn.close()

    def get_slippage_by_symbol(self, days: int = 30) -> Dict[str, SlippageStats]:
        """
        Get slippage breakdown by symbol.

        Args:
            days: Number of days to analyze

        Returns:
            Dict mapping symbol to SlippageStats
        """
        conn = self._get_conn()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        try:
            df = pd.read_sql_query("""
                SELECT symbol, slippage_bps, slippage_dollars
                FROM execution_quality
                WHERE timestamp >= ?
            """, conn, params=(cutoff,))

            if df.empty:
                return {}

            result = {}
            for symbol, group in df.groupby('symbol'):
                result[symbol] = SlippageStats(
                    count=len(group),
                    mean_bps=round(group['slippage_bps'].mean(), 2),
                    median_bps=round(group['slippage_bps'].median(), 2),
                    std_bps=round(group['slippage_bps'].std(), 2) if len(group) > 1 else 0.0,
                    min_bps=round(group['slippage_bps'].min(), 2),
                    max_bps=round(group['slippage_bps'].max(), 2),
                    total_dollars=round(group['slippage_dollars'].sum(), 2)
                )

            return result

        finally:
            conn.close()

    def get_slippage_by_order_size(
        self,
        days: int = 30,
        size_buckets: List[int] = None
    ) -> Dict[str, SlippageStats]:
        """
        Get slippage breakdown by order size buckets.

        Args:
            days: Number of days to analyze
            size_buckets: List of bucket thresholds (default: [100, 500, 1000, 5000])

        Returns:
            Dict mapping size bucket label to SlippageStats
        """
        if size_buckets is None:
            size_buckets = [100, 500, 1000, 5000]

        conn = self._get_conn()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        try:
            df = pd.read_sql_query("""
                SELECT order_size, slippage_bps, slippage_dollars
                FROM execution_quality
                WHERE timestamp >= ?
            """, conn, params=(cutoff,))

            if df.empty:
                return {}

            # Create bucket labels
            def get_bucket(size):
                for i, threshold in enumerate(size_buckets):
                    if size < threshold:
                        if i == 0:
                            return f"<{threshold}"
                        else:
                            return f"{size_buckets[i-1]}-{threshold}"
                return f">{size_buckets[-1]}"

            df['bucket'] = df['order_size'].apply(get_bucket)

            result = {}
            for bucket, group in df.groupby('bucket'):
                result[bucket] = SlippageStats(
                    count=len(group),
                    mean_bps=round(group['slippage_bps'].mean(), 2),
                    median_bps=round(group['slippage_bps'].median(), 2),
                    std_bps=round(group['slippage_bps'].std(), 2) if len(group) > 1 else 0.0,
                    min_bps=round(group['slippage_bps'].min(), 2),
                    max_bps=round(group['slippage_bps'].max(), 2),
                    total_dollars=round(group['slippage_dollars'].sum(), 2)
                )

            return result

        finally:
            conn.close()

    # ========================================================================
    # ALERTING
    # ========================================================================

    def alert_if_poor_execution(
        self,
        threshold_bps: float = 50.0,
        order_id: str = None
    ) -> bool:
        """
        Check if recent execution(s) exceeded slippage threshold.

        Args:
            threshold_bps: Slippage threshold in basis points
            order_id: Specific order to check (or latest if None)

        Returns:
            True if alert was generated
        """
        conn = self._get_conn()

        try:
            if order_id:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT order_id, symbol, slippage_bps
                    FROM execution_quality
                    WHERE order_id = ?
                """, (order_id,))
                row = cursor.fetchone()

                if row and abs(row['slippage_bps']) > threshold_bps:
                    alert = self._create_alert(
                        order_id=row['order_id'],
                        symbol=row['symbol'],
                        slippage_bps=row['slippage_bps'],
                        threshold_bps=threshold_bps
                    )
                    self._process_alert(alert)
                    return True
            else:
                # Check most recent execution
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT order_id, symbol, slippage_bps
                    FROM execution_quality
                    ORDER BY timestamp DESC
                    LIMIT 1
                """)
                row = cursor.fetchone()

                if row and abs(row['slippage_bps']) > threshold_bps:
                    alert = self._create_alert(
                        order_id=row['order_id'],
                        symbol=row['symbol'],
                        slippage_bps=row['slippage_bps'],
                        threshold_bps=threshold_bps
                    )
                    self._process_alert(alert)
                    return True

            return False

        finally:
            conn.close()

    def check_recent_executions(
        self,
        threshold_bps: float = 50.0,
        lookback_minutes: int = 60
    ) -> List[ExecutionAlert]:
        """
        Check recent executions for poor quality.

        Args:
            threshold_bps: Slippage threshold in basis points
            lookback_minutes: How far back to check

        Returns:
            List of ExecutionAlert objects
        """
        conn = self._get_conn()
        cutoff = (datetime.now() - timedelta(minutes=lookback_minutes)).isoformat()

        alerts = []

        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT order_id, symbol, slippage_bps
                FROM execution_quality
                WHERE timestamp >= ? AND ABS(slippage_bps) > ?
                ORDER BY slippage_bps DESC
            """, (cutoff, threshold_bps))

            for row in cursor.fetchall():
                alert = self._create_alert(
                    order_id=row['order_id'],
                    symbol=row['symbol'],
                    slippage_bps=row['slippage_bps'],
                    threshold_bps=threshold_bps
                )
                alerts.append(alert)
                self._process_alert(alert)

            return alerts

        finally:
            conn.close()

    def _create_alert(
        self,
        order_id: str,
        symbol: str,
        slippage_bps: float,
        threshold_bps: float
    ) -> ExecutionAlert:
        """Create an execution alert."""
        # Determine severity based on slippage magnitude
        abs_slippage = abs(slippage_bps)

        if abs_slippage > threshold_bps * 3:
            severity = AlertSeverity.CRITICAL
        elif abs_slippage > threshold_bps * 2:
            severity = AlertSeverity.HIGH
        elif abs_slippage > threshold_bps * 1.5:
            severity = AlertSeverity.MEDIUM
        else:
            severity = AlertSeverity.LOW

        message = (
            f"Poor execution quality on {symbol}: "
            f"{slippage_bps:.1f} bps slippage (threshold: {threshold_bps:.0f} bps)"
        )

        return ExecutionAlert(
            severity=severity,
            message=message,
            order_id=order_id,
            symbol=symbol,
            slippage_bps=slippage_bps,
            threshold_bps=threshold_bps
        )

    def _process_alert(self, alert: ExecutionAlert):
        """Process and store an alert."""
        # Add to recent alerts
        self._recent_alerts.append(alert)
        if len(self._recent_alerts) > self._max_alerts:
            self._recent_alerts = self._recent_alerts[-self._max_alerts:]

        # Log based on severity
        if alert.severity == AlertSeverity.CRITICAL:
            logger.critical(alert.message)
        elif alert.severity == AlertSeverity.HIGH:
            logger.error(alert.message)
        elif alert.severity == AlertSeverity.MEDIUM:
            logger.warning(alert.message)
        else:
            logger.info(alert.message)

    def get_recent_alerts(self, count: int = 10) -> List[ExecutionAlert]:
        """Get recent execution quality alerts."""
        return self._recent_alerts[-count:]

    # ========================================================================
    # MARKET IMPACT ESTIMATION
    # ========================================================================

    def estimate_market_impact(
        self,
        symbol: str,
        order_size: int,
        daily_volume: int,
        volatility: float = 0.0
    ) -> float:
        """
        Estimate market impact for a given order.

        Uses simplified square-root model:
        Impact (bps) = C * sigma * sqrt(Q/V)

        Where:
        - C = scaling constant (empirically ~10)
        - sigma = daily volatility
        - Q = order quantity
        - V = daily volume

        Args:
            symbol: Stock symbol
            order_size: Number of shares
            daily_volume: Average daily volume
            volatility: Daily volatility (default uses 2% assumption)

        Returns:
            Estimated market impact in basis points
        """
        if daily_volume <= 0:
            logger.warning(f"Invalid daily_volume for {symbol}: {daily_volume}")
            return 100.0  # Conservative estimate for unknown liquidity

        # Default volatility assumption
        if volatility <= 0:
            volatility = 0.02  # 2% daily volatility

        # Participation rate
        participation = order_size / daily_volume

        # Square-root impact model
        C = 10.0  # Scaling constant
        impact_bps = C * volatility * 10000 * np.sqrt(participation)

        return round(impact_bps, 2)

    def get_historical_impact(self, symbol: str, days: int = 30) -> Dict[str, float]:
        """
        Get historical market impact data for a symbol.

        Returns:
            Dict with impact statistics
        """
        conn = self._get_conn()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        try:
            df = pd.read_sql_query("""
                SELECT slippage_bps, order_size, daily_volume_pct
                FROM execution_quality
                WHERE symbol = ? AND timestamp >= ?
            """, conn, params=(symbol, cutoff))

            if df.empty:
                return {
                    'count': 0,
                    'avg_slippage_bps': 0.0,
                    'avg_participation_pct': 0.0
                }

            return {
                'count': len(df),
                'avg_slippage_bps': round(df['slippage_bps'].mean(), 2),
                'avg_participation_pct': round(df['daily_volume_pct'].mean(), 4),
                'max_slippage_bps': round(df['slippage_bps'].max(), 2),
                'total_shares': int(df['order_size'].sum())
            }

        finally:
            conn.close()

    # ========================================================================
    # DASHBOARD DATA
    # ========================================================================

    def get_dashboard_data(self, days: int = 7) -> Dict[str, Any]:
        """
        Get execution quality data formatted for dashboard visualization.

        Args:
            days: Number of days to include

        Returns:
            Dict with data for various dashboard components
        """
        report = self.get_execution_report(days=days)
        hourly = self.get_slippage_by_time_of_day(days=days)
        strategy = {k: v.to_dict() for k, v in
                   self._get_strategy_stats(days).items()}

        conn = self._get_conn()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        try:
            # Get daily trend
            df = pd.read_sql_query("""
                SELECT DATE(timestamp) as date,
                       AVG(slippage_bps) as avg_slippage,
                       SUM(slippage_dollars) as total_cost,
                       COUNT(*) as executions
                FROM execution_quality
                WHERE timestamp >= ?
                GROUP BY DATE(timestamp)
                ORDER BY date
            """, conn, params=(cutoff,))

            daily_trend = df.to_dict('records') if not df.empty else []

        finally:
            conn.close()

        return {
            'summary': report.to_dict(),
            'daily_trend': daily_trend,
            'hourly_breakdown': {str(k): v.to_dict() for k, v in hourly.items()},
            'strategy_breakdown': strategy,
            'recent_alerts': [
                {
                    'severity': a.severity.value,
                    'message': a.message,
                    'symbol': a.symbol,
                    'slippage_bps': a.slippage_bps,
                    'timestamp': a.timestamp
                }
                for a in self.get_recent_alerts(10)
            ]
        }

    def _get_strategy_stats(self, days: int) -> Dict[str, SlippageStats]:
        """Get slippage stats for all strategies."""
        conn = self._get_conn()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT strategy FROM execution_quality
                WHERE timestamp >= ? AND strategy != ''
            """, (cutoff,))

            strategies = [row['strategy'] for row in cursor.fetchall()]

            return {
                strategy: self.get_slippage_by_strategy(strategy, days)
                for strategy in strategies
            }

        finally:
            conn.close()

    # ========================================================================
    # RAW DATA ACCESS
    # ========================================================================

    def get_raw_data(
        self,
        days: int = 30,
        symbol: str = None,
        strategy: str = None
    ) -> pd.DataFrame:
        """
        Get raw execution quality data as DataFrame.

        Args:
            days: Number of days to fetch
            symbol: Optional symbol filter
            strategy: Optional strategy filter

        Returns:
            DataFrame with execution records
        """
        conn = self._get_conn()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        try:
            query = "SELECT * FROM execution_quality WHERE timestamp >= ?"
            params = [cutoff]

            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)

            if strategy:
                query += " AND strategy = ?"
                params.append(strategy)

            query += " ORDER BY timestamp DESC"

            return pd.read_sql_query(query, conn, params=params)

        finally:
            conn.close()


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

_tracker: Optional[ExecutionQualityTracker] = None

def get_execution_tracker() -> ExecutionQualityTracker:
    """Get or create global execution quality tracker."""
    global _tracker
    if _tracker is None:
        _tracker = ExecutionQualityTracker()
    return _tracker


def create_tracker(db_path: Path = None) -> ExecutionQualityTracker:
    """Create a new execution quality tracker instance."""
    return ExecutionQualityTracker(db_path=db_path)


# ============================================================================
# DEMO / TEST
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("EXECUTION QUALITY TRACKER DEMO")
    print("=" * 60)

    # Create tracker
    tracker = ExecutionQualityTracker()

    # Record some test executions
    print("\nRecording test executions...")

    test_data = [
        ("AAPL", "buy", 175.00, 175.25, 100, "mean_reversion"),
        ("MSFT", "sell", 350.00, 349.50, 50, "pairs_trading"),
        ("GOOGL", "buy", 140.00, 140.75, 200, "vol_managed_momentum"),
        ("AMZN", "sell", 180.00, 179.00, 75, "mean_reversion"),
        ("NVDA", "buy", 500.00, 502.50, 30, "gap_fill"),
    ]

    for symbol, side, signal_price, fill_price, size, strategy in test_data:
        order_id = f"TEST_{symbol}_{datetime.now().timestamp()}"
        tracker.record_execution_direct(
            order_id=order_id,
            symbol=symbol,
            side=side,
            signal_price=signal_price,
            fill_price=fill_price,
            order_size=size,
            strategy=strategy,
            time_to_fill_seconds=1.5,
            market_volatility=20.0,
            daily_volume=1000000
        )

    # Generate report
    print("\n" + "=" * 60)
    report = tracker.get_execution_report(days=1)
    print(report.summary())

    # Check for alerts
    print("\n" + "-" * 40)
    print("Checking for poor executions (threshold: 25 bps)...")
    alerts = tracker.check_recent_executions(threshold_bps=25, lookback_minutes=60)
    if alerts:
        print(f"Found {len(alerts)} alerts:")
        for alert in alerts:
            print(f"  [{alert.severity.value}] {alert.message}")
    else:
        print("No alerts generated")

    # Test slippage calculation
    print("\n" + "-" * 40)
    print("Slippage calculation examples:")
    for signal, fill, side in [(100.0, 100.50, 'buy'), (100.0, 99.50, 'sell')]:
        bps, dollars = tracker.calculate_slippage(signal, fill, side, 100)
        print(f"  {side.upper()}: signal=${signal}, fill=${fill} -> {bps} bps, ${dollars}")

    # Market impact estimate
    print("\n" + "-" * 40)
    print("Market impact estimates:")
    for size in [100, 1000, 10000]:
        impact = tracker.estimate_market_impact("TEST", size, 1000000, 0.02)
        print(f"  {size:,} shares / 1M volume -> {impact:.1f} bps estimated impact")

    print("\n" + "=" * 60)
    print("Demo complete. Data saved to trades.db")
