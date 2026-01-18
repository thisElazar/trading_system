#!/usr/bin/env python3
"""
Signal Strength Scoring System
==============================
Score signals based on historical performance and signal characteristics.

Features:
- Track win/loss rates by signal attributes
- Score incoming signals based on historical patterns
- Adjust position sizing based on conviction
- Filter weak signals automatically
- Learn optimal signal thresholds over time

Usage:
    scorer = SignalScorer()

    # Record outcomes
    scorer.record_outcome(signal, pnl_pct=2.5, hold_days=3)

    # Score new signals
    score = scorer.score_signal(signal)
    if score.conviction >= 0.6:
        size_multiplier = score.size_multiplier
        execute_trade(signal, size=base_size * size_multiplier)
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
from core.types import Signal, Side

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

class SignalOutcome(Enum):
    """Outcome of a signal."""
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    PENDING = "pending"


@dataclass
class SignalRecord:
    """
    Historical record of a signal and its outcome.

    Uses old field names for database compatibility.
    New code should use canonical names via properties.
    """
    id: Optional[int] = None
    signal_id: str = ""
    symbol: str = ""
    strategy: str = ""              # DEPRECATED: Use strategy_id
    signal_type: str = ""           # DEPRECATED: Use side (buy/sell -> BUY/SELL)

    # Signal characteristics
    signal_strength: float = 0.0    # DEPRECATED: Use strength
    volatility_regime: str = ""     # low, normal, high
    trend_alignment: str = ""       # with_trend, counter_trend, neutral
    volume_confirmation: bool = False
    time_of_day: str = ""           # morning, midday, afternoon, close
    day_of_week: int = 0            # 0=Monday, 4=Friday

    # Market context at signal time
    vix_level: float = 0.0
    sector_momentum: float = 0.0
    market_trend: str = ""          # bull, bear, sideways

    # Outcome
    outcome: str = "pending"
    profit_pct: float = 0.0         # DEPRECATED: Use pnl_pct
    hold_days: int = 0
    max_drawdown_pct: float = 0.0
    max_profit_pct: float = 0.0

    # Timestamps
    signal_time: str = ""
    exit_time: str = ""

    def __post_init__(self):
        if not self.signal_time:
            self.signal_time = datetime.now().isoformat()

    # Canonical property aliases
    @property
    def strategy_id(self) -> str:
        return self.strategy

    @property
    def strength(self) -> float:
        return self.signal_strength

    @property
    def side(self) -> Side:
        sig_map = {'buy': 'BUY', 'sell': 'SELL'}
        return Side(sig_map.get(self.signal_type.lower(), 'BUY'))

    @property
    def pnl_pct(self) -> float:
        return self.profit_pct


@dataclass
class SignalScore:
    """
    Score for an incoming signal.

    Note: conviction is the primary score (0.0-1.0).
    """
    conviction: float                           # 0.0-1.0 overall score
    win_probability: float                      # Historical win rate
    expected_return: float                      # Average return for similar
    expected_risk: float                        # Average max drawdown
    risk_reward_ratio: float                    # expected_return / expected_risk

    # Sizing recommendation
    suggested_size_multiplier: float            # DEPRECATED: Use size_multiplier
    confidence_interval: Tuple[float, float]    # 95% CI for returns

    # Contributing factors
    factors: Dict[str, float] = field(default_factory=dict)
    sample_size: int = 0
    similar_signals_count: int = 0

    @property
    def size_multiplier(self) -> float:
        """Canonical name for suggested_size_multiplier."""
        return self.suggested_size_multiplier

    def __str__(self):
        return (
            f"SignalScore(conviction={self.conviction:.2f}, "
            f"win_prob={self.win_probability:.1%}, "
            f"exp_return={self.expected_return:.2%}, "
            f"size_mult={self.size_multiplier:.2f})"
        )

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['confidence_interval'] = list(self.confidence_interval)
        result['size_multiplier'] = self.size_multiplier
        return result


@dataclass
class ScoringConfig:
    """Configuration for signal scoring."""
    # Minimum samples needed for reliable scoring
    min_samples: int = 20

    # Win rate thresholds
    high_conviction_win_rate: float = 0.65
    low_conviction_win_rate: float = 0.45

    # Size multiplier bounds
    max_size_multiplier: float = 2.0
    min_size_multiplier: float = 0.25

    # Signal filtering
    min_conviction_to_trade: float = 0.4
    skip_low_conviction: bool = True

    # Learning rate for updating scores
    learning_rate: float = 0.1

    # Lookback period for historical analysis
    lookback_days: int = 180


# ============================================================================
# MAIN SCORER CLASS
# ============================================================================

class SignalScorer:
    """
    Score signals based on historical performance patterns.

    Learns from past signal outcomes to predict future performance
    and recommend position sizing.
    """

    # Volatility regime thresholds (VIX)
    LOW_VOL_VIX = 15.0
    HIGH_VOL_VIX = 25.0

    # Time of day buckets (Eastern time)
    TIME_BUCKETS = {
        (9, 10): "morning",
        (10, 12): "midday",
        (12, 14): "early_afternoon",
        (14, 16): "late_afternoon",
    }

    def __init__(self, config: ScoringConfig = None, db_path: Path = None):
        """
        Initialize the signal scorer.

        Args:
            config: Scoring configuration
            db_path: Path to database (default from config)
        """
        self.config = config or ScoringConfig()
        self.db_path = db_path or DATABASES.get('trades', DIRS['db'] / 'trades.db')
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_tables()

        # Cache for performance
        self._score_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_valid_until: datetime = datetime.min

        logger.info(f"SignalScorer initialized with DB: {self.db_path}")

    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        """Create signal_history table if it doesn't exist."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signal_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id TEXT UNIQUE NOT NULL,
                symbol TEXT NOT NULL,
                strategy_id TEXT NOT NULL,
                side TEXT NOT NULL,
                strength REAL DEFAULT 0,
                volatility_regime TEXT DEFAULT '',
                trend_alignment TEXT DEFAULT '',
                volume_confirmation INTEGER DEFAULT 0,
                time_of_day TEXT DEFAULT '',
                day_of_week INTEGER DEFAULT 0,
                vix_level REAL DEFAULT 0,
                sector_momentum REAL DEFAULT 0,
                market_trend TEXT DEFAULT '',
                outcome TEXT DEFAULT 'pending',
                profit_pct REAL DEFAULT 0,
                hold_days INTEGER DEFAULT 0,
                max_drawdown_pct REAL DEFAULT 0,
                max_profit_pct REAL DEFAULT 0,
                signal_time TEXT NOT NULL,
                exit_time TEXT DEFAULT ''
            )
        """)

        # Indexes for fast lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_signal_hist_strategy_id
            ON signal_history(strategy_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_signal_hist_outcome
            ON signal_history(outcome)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_signal_hist_time
            ON signal_history(signal_time)
        """)

        # Aggregated scores table for fast lookups
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signal_scores_cache (
                key TEXT PRIMARY KEY,
                win_rate REAL,
                avg_return REAL,
                avg_drawdown REAL,
                sample_count INTEGER,
                last_updated TEXT
            )
        """)

        conn.commit()
        conn.close()

        logger.debug("signal_history tables initialized")

    # ========================================================================
    # RECORDING OUTCOMES
    # ========================================================================

    def record_signal(
        self,
        signal_id: str,
        symbol: str,
        strategy: str,
        signal_type: str,
        signal_strength: float = 0.0,
        vix_level: float = 0.0,
        market_trend: str = "",
        volume_confirmed: bool = False,
        trend_alignment: str = "neutral",
        sector_momentum: float = 0.0,
    ) -> int:
        """
        Record a new signal (before outcome is known).

        Args:
            signal_id: Unique identifier for this signal
            symbol: Stock symbol
            strategy: Strategy that generated the signal
            signal_type: 'buy' or 'sell'
            signal_strength: Original signal strength (0-1)
            vix_level: Current VIX
            market_trend: 'bull', 'bear', or 'sideways'
            volume_confirmed: Whether volume confirms the signal
            trend_alignment: 'with_trend', 'counter_trend', or 'neutral'
            sector_momentum: Sector momentum (-1 to 1)

        Returns:
            Record ID
        """
        now = datetime.now()

        # Determine volatility regime
        if vix_level < self.LOW_VOL_VIX:
            vol_regime = "low"
        elif vix_level > self.HIGH_VOL_VIX:
            vol_regime = "high"
        else:
            vol_regime = "normal"

        # Determine time of day
        hour = now.hour
        time_of_day = "close"
        for (start, end), label in self.TIME_BUCKETS.items():
            if start <= hour < end:
                time_of_day = label
                break

        record = SignalRecord(
            signal_id=signal_id,
            symbol=symbol,
            strategy=strategy,
            signal_type=signal_type,
            signal_strength=signal_strength,
            volatility_regime=vol_regime,
            trend_alignment=trend_alignment,
            volume_confirmation=volume_confirmed,
            time_of_day=time_of_day,
            day_of_week=now.weekday(),
            vix_level=vix_level,
            sector_momentum=sector_momentum,
            market_trend=market_trend,
        )

        return self._save_record(record)

    def record_outcome(
        self,
        signal_id: str,
        outcome: str,
        profit_pct: float,
        hold_days: int = 0,
        max_drawdown_pct: float = 0.0,
        max_profit_pct: float = 0.0,
    ) -> bool:
        """
        Record the outcome of a previously recorded signal.

        Args:
            signal_id: The signal's unique identifier
            outcome: 'win', 'loss', or 'breakeven'
            profit_pct: Final profit/loss percentage
            hold_days: How long the position was held
            max_drawdown_pct: Maximum drawdown during the trade
            max_profit_pct: Maximum unrealized profit during the trade

        Returns:
            True if updated successfully
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                UPDATE signal_history SET
                    outcome = ?,
                    profit_pct = ?,
                    hold_days = ?,
                    max_drawdown_pct = ?,
                    max_profit_pct = ?,
                    exit_time = ?
                WHERE signal_id = ?
            """, (
                outcome, profit_pct, hold_days,
                max_drawdown_pct, max_profit_pct,
                datetime.now().isoformat(), signal_id
            ))

            conn.commit()
            updated = cursor.rowcount > 0

            if updated:
                # Invalidate cache
                self._cache_valid_until = datetime.min
                logger.info(f"Recorded outcome for {signal_id}: {outcome}, {profit_pct:.2f}%")

            return updated

        except Exception as e:
            logger.error(f"Failed to record outcome: {e}")
            return False

        finally:
            conn.close()

    def _save_record(self, record: SignalRecord) -> int:
        """Save signal record to database."""
        conn = self._get_conn()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO signal_history (
                    signal_id, symbol, strategy_id, side,
                    strength, volatility_regime, trend_alignment,
                    volume_confirmation, time_of_day, day_of_week,
                    vix_level, sector_momentum, market_trend,
                    outcome, profit_pct, hold_days, max_drawdown_pct,
                    max_profit_pct, signal_time, exit_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.signal_id, record.symbol, record.strategy,
                record.signal_type, record.signal_strength,
                record.volatility_regime, record.trend_alignment,
                1 if record.volume_confirmation else 0,
                record.time_of_day, record.day_of_week,
                record.vix_level, record.sector_momentum, record.market_trend,
                record.outcome, record.profit_pct, record.hold_days,
                record.max_drawdown_pct, record.max_profit_pct,
                record.signal_time, record.exit_time
            ))

            record_id = cursor.lastrowid
            conn.commit()
            return record_id

        finally:
            conn.close()

    # ========================================================================
    # SCORING SIGNALS
    # ========================================================================

    def score_signal(
        self,
        strategy: str,
        signal_type: str,
        vix_level: float = 0.0,
        trend_alignment: str = "neutral",
        volume_confirmed: bool = False,
        market_trend: str = "",
        signal_strength: float = 0.5,
    ) -> SignalScore:
        """
        Score an incoming signal based on historical patterns.

        Args:
            strategy: Strategy generating the signal
            signal_type: 'buy' or 'sell'
            vix_level: Current VIX level
            trend_alignment: 'with_trend', 'counter_trend', or 'neutral'
            volume_confirmed: Whether volume confirms the signal
            market_trend: Current market trend
            signal_strength: Original signal strength (0-1)

        Returns:
            SignalScore with conviction, sizing recommendations, etc.
        """
        # Determine volatility regime
        if vix_level < self.LOW_VOL_VIX:
            vol_regime = "low"
        elif vix_level > self.HIGH_VOL_VIX:
            vol_regime = "high"
        else:
            vol_regime = "normal"

        # Get historical performance for similar signals
        stats = self._get_historical_stats(
            strategy=strategy,
            signal_type=signal_type,
            volatility_regime=vol_regime,
            trend_alignment=trend_alignment,
            volume_confirmed=volume_confirmed,
            market_trend=market_trend,
        )

        # Calculate conviction score (0-1)
        factors = {}

        # Base: Historical win rate
        win_rate = stats.get('win_rate', 0.5)
        factors['historical_win_rate'] = win_rate

        # Factor: Volume confirmation bonus
        if volume_confirmed:
            factors['volume_confirmation'] = 0.1
        else:
            factors['volume_confirmation'] = 0.0

        # Factor: Trend alignment
        if trend_alignment == "with_trend":
            factors['trend_alignment'] = 0.1
        elif trend_alignment == "counter_trend":
            factors['trend_alignment'] = -0.1
        else:
            factors['trend_alignment'] = 0.0

        # Factor: Volatility regime
        if vol_regime == "normal":
            factors['volatility'] = 0.05
        elif vol_regime == "high":
            factors['volatility'] = -0.05
        else:
            factors['volatility'] = 0.0

        # Factor: Original signal strength
        factors['signal_strength'] = (signal_strength - 0.5) * 0.2

        # Calculate conviction
        base_conviction = win_rate
        adjustment = sum(v for k, v in factors.items() if k != 'historical_win_rate')
        conviction = max(0.0, min(1.0, base_conviction + adjustment))

        # Expected return and risk
        avg_return = stats.get('avg_return', 0.0)
        avg_drawdown = stats.get('avg_drawdown', 0.05)
        std_return = stats.get('std_return', 0.1)

        # Risk-reward ratio
        if avg_drawdown > 0:
            risk_reward = avg_return / avg_drawdown
        else:
            risk_reward = avg_return / 0.01  # Avoid division by zero

        # Calculate sizing multiplier
        size_mult = self._calculate_size_multiplier(
            conviction=conviction,
            win_rate=win_rate,
            risk_reward=risk_reward,
        )

        # Confidence interval (rough estimate)
        sample_size = stats.get('sample_count', 0)
        if sample_size > 10:
            margin = 1.96 * std_return / np.sqrt(sample_size)
        else:
            margin = std_return  # Wide margin for small samples

        confidence_interval = (avg_return - margin, avg_return + margin)

        return SignalScore(
            conviction=conviction,
            win_probability=win_rate,
            expected_return=avg_return,
            expected_risk=avg_drawdown,
            risk_reward_ratio=risk_reward,
            suggested_size_multiplier=size_mult,
            confidence_interval=confidence_interval,
            factors=factors,
            sample_size=sample_size,
            similar_signals_count=stats.get('similar_count', 0),
        )

    def _get_historical_stats(
        self,
        strategy: str,
        signal_type: str,
        volatility_regime: str = None,
        trend_alignment: str = None,
        volume_confirmed: bool = None,
        market_trend: str = None,
    ) -> Dict[str, float]:
        """
        Get historical statistics for similar signals.

        Returns progressively broader matches if specific match has few samples.
        """
        conn = self._get_conn()
        cutoff = (datetime.now() - timedelta(days=self.config.lookback_days)).isoformat()

        try:
            # Try exact match first
            conditions = [
                "strategy_id = ?",
                "side = ?",
                "outcome != 'pending'",
                "signal_time >= ?"
            ]
            params = [strategy, signal_type, cutoff]

            if volatility_regime:
                conditions.append("volatility_regime = ?")
                params.append(volatility_regime)

            if trend_alignment:
                conditions.append("trend_alignment = ?")
                params.append(trend_alignment)

            if volume_confirmed is not None:
                conditions.append("volume_confirmation = ?")
                params.append(1 if volume_confirmed else 0)

            if market_trend:
                conditions.append("market_trend = ?")
                params.append(market_trend)

            query = f"""
                SELECT outcome, profit_pct, max_drawdown_pct
                FROM signal_history
                WHERE {' AND '.join(conditions)}
            """

            df = pd.read_sql_query(query, conn, params=params)

            # If not enough samples, broaden the search
            if len(df) < self.config.min_samples:
                # Try strategy + side only
                df = pd.read_sql_query("""
                    SELECT outcome, profit_pct, max_drawdown_pct
                    FROM signal_history
                    WHERE strategy_id = ? AND side = ?
                    AND outcome != 'pending' AND signal_time >= ?
                """, conn, params=[strategy, signal_type, cutoff])

            if len(df) < 5:
                # Not enough data, return defaults
                return {
                    'win_rate': 0.5,
                    'avg_return': 0.0,
                    'avg_drawdown': 0.05,
                    'std_return': 0.1,
                    'sample_count': len(df),
                    'similar_count': len(df),
                }

            # Calculate statistics
            wins = (df['outcome'] == 'win').sum()
            total = len(df)
            win_rate = wins / total if total > 0 else 0.5

            avg_return = df['profit_pct'].mean() / 100
            std_return = df['profit_pct'].std() / 100 if len(df) > 1 else 0.1
            avg_drawdown = df['max_drawdown_pct'].mean() / 100

            return {
                'win_rate': win_rate,
                'avg_return': avg_return,
                'avg_drawdown': max(avg_drawdown, 0.01),
                'std_return': std_return,
                'sample_count': total,
                'similar_count': total,
            }

        finally:
            conn.close()

    def _calculate_size_multiplier(
        self,
        conviction: float,
        win_rate: float,
        risk_reward: float,
    ) -> float:
        """
        Calculate position size multiplier based on signal quality.

        Uses Kelly Criterion with fractional sizing for safety.
        """
        # Kelly fraction: f = (bp - q) / b
        # where b = win/loss ratio, p = win probability, q = loss probability
        if win_rate <= 0 or win_rate >= 1:
            kelly = 0.0
        else:
            q = 1 - win_rate
            b = max(risk_reward, 0.1)
            kelly = (b * win_rate - q) / b

        # Use fractional Kelly (25%) for safety
        fractional_kelly = kelly * 0.25

        # Combine with conviction
        base_mult = 0.5 + (conviction * 0.5)
        kelly_adj = max(0, fractional_kelly)

        # Final multiplier
        size_mult = base_mult + kelly_adj

        # Clamp to bounds
        size_mult = max(self.config.min_size_multiplier,
                       min(self.config.max_size_multiplier, size_mult))

        return round(size_mult, 2)

    # ========================================================================
    # ANALYSIS AND REPORTING
    # ========================================================================

    def get_strategy_stats(self, strategy: str, days: int = 90) -> Dict[str, Any]:
        """Get comprehensive signal statistics for a strategy."""
        conn = self._get_conn()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        try:
            df = pd.read_sql_query("""
                SELECT * FROM signal_history
                WHERE strategy_id = ? AND outcome != 'pending'
                AND signal_time >= ?
            """, conn, params=[strategy, cutoff])

            if df.empty:
                return {'error': 'No data', 'strategy': strategy}

            wins = (df['outcome'] == 'win').sum()
            losses = (df['outcome'] == 'loss').sum()
            total = len(df)

            return {
                'strategy': strategy,
                'total_signals': total,
                'wins': wins,
                'losses': losses,
                'win_rate': wins / total if total > 0 else 0,
                'avg_profit_pct': df['profit_pct'].mean(),
                'avg_hold_days': df['hold_days'].mean(),
                'avg_drawdown_pct': df['max_drawdown_pct'].mean(),
                'best_trade_pct': df['profit_pct'].max(),
                'worst_trade_pct': df['profit_pct'].min(),
                'profit_factor': (
                    df[df['profit_pct'] > 0]['profit_pct'].sum() /
                    abs(df[df['profit_pct'] < 0]['profit_pct'].sum())
                    if df[df['profit_pct'] < 0]['profit_pct'].sum() != 0 else float('inf')
                ),
                'by_volatility': df.groupby('volatility_regime').agg({
                    'profit_pct': 'mean',
                    'outcome': lambda x: (x == 'win').mean()
                }).to_dict('index'),
                'by_trend': df.groupby('trend_alignment').agg({
                    'profit_pct': 'mean',
                    'outcome': lambda x: (x == 'win').mean()
                }).to_dict('index') if 'trend_alignment' in df else {},
            }

        finally:
            conn.close()

    def get_optimal_conditions(self, strategy: str, min_samples: int = 10) -> Dict[str, Any]:
        """
        Find optimal trading conditions for a strategy.

        Returns the conditions (volatility, trend, etc.) with best win rates.
        """
        conn = self._get_conn()
        cutoff = (datetime.now() - timedelta(days=self.config.lookback_days)).isoformat()

        try:
            df = pd.read_sql_query("""
                SELECT volatility_regime, trend_alignment, time_of_day,
                       volume_confirmation, market_trend,
                       outcome, profit_pct
                FROM signal_history
                WHERE strategy_id = ? AND outcome != 'pending'
                AND signal_time >= ?
            """, conn, params=[strategy, cutoff])

            if len(df) < min_samples:
                return {'error': 'Insufficient data', 'sample_count': len(df)}

            results = {}

            # Best volatility regime
            vol_stats = df.groupby('volatility_regime').agg({
                'outcome': lambda x: ((x == 'win').sum(), len(x)),
                'profit_pct': 'mean'
            })
            if not vol_stats.empty:
                best_vol = vol_stats['profit_pct'].idxmax()
                results['best_volatility_regime'] = {
                    'regime': best_vol,
                    'win_rate': vol_stats.loc[best_vol, 'outcome'][0] / vol_stats.loc[best_vol, 'outcome'][1],
                    'avg_profit': vol_stats.loc[best_vol, 'profit_pct'],
                }

            # Best trend alignment
            trend_stats = df.groupby('trend_alignment').agg({
                'outcome': lambda x: ((x == 'win').sum(), len(x)),
                'profit_pct': 'mean'
            })
            if not trend_stats.empty:
                best_trend = trend_stats['profit_pct'].idxmax()
                results['best_trend_alignment'] = {
                    'alignment': best_trend,
                    'win_rate': trend_stats.loc[best_trend, 'outcome'][0] / trend_stats.loc[best_trend, 'outcome'][1],
                    'avg_profit': trend_stats.loc[best_trend, 'profit_pct'],
                }

            # Volume confirmation impact
            vol_conf = df.groupby('volume_confirmation')['profit_pct'].mean()
            if len(vol_conf) == 2:
                results['volume_confirmation_impact'] = {
                    'with_volume': vol_conf.get(1, 0),
                    'without_volume': vol_conf.get(0, 0),
                    'improvement': vol_conf.get(1, 0) - vol_conf.get(0, 0),
                }

            # Best time of day
            time_stats = df.groupby('time_of_day')['profit_pct'].mean()
            if not time_stats.empty:
                results['best_time_of_day'] = {
                    'time': time_stats.idxmax(),
                    'avg_profit': time_stats.max(),
                    'all_times': time_stats.to_dict(),
                }

            return results

        finally:
            conn.close()

    def should_trade(self, score: SignalScore) -> Tuple[bool, str]:
        """
        Determine if a signal should be traded based on its score.

        Args:
            score: SignalScore from score_signal()

        Returns:
            Tuple of (should_trade, reason)
        """
        if not self.config.skip_low_conviction:
            return True, "low_conviction_filter_disabled"

        if score.conviction < self.config.min_conviction_to_trade:
            return False, f"conviction_{score.conviction:.2f}_below_threshold_{self.config.min_conviction_to_trade}"

        if score.sample_size < self.config.min_samples // 2:
            return True, "insufficient_data_proceed_with_caution"

        if score.win_probability < self.config.low_conviction_win_rate:
            return False, f"win_rate_{score.win_probability:.1%}_below_threshold"

        if score.risk_reward_ratio < 0.5:
            return False, f"poor_risk_reward_{score.risk_reward_ratio:.2f}"

        return True, "passed_all_filters"

    def print_report(self, strategy: str = None):
        """Print a comprehensive scoring report."""
        conn = self._get_conn()

        try:
            if strategy:
                strategies = [strategy]
            else:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT strategy_id FROM signal_history WHERE outcome != 'pending'")
                strategies = [row['strategy_id'] for row in cursor.fetchall()]

            print("\n" + "=" * 70)
            print("SIGNAL SCORING REPORT")
            print("=" * 70)

            for strat in strategies:
                stats = self.get_strategy_stats(strat)
                if 'error' in stats:
                    continue

                print(f"\n{strat.upper()}")
                print("-" * 50)
                print(f"  Total Signals: {stats['total_signals']}")
                print(f"  Win Rate: {stats['win_rate']:.1%}")
                print(f"  Avg Profit: {stats['avg_profit_pct']:.2f}%")
                print(f"  Avg Hold: {stats['avg_hold_days']:.1f} days")
                print(f"  Profit Factor: {stats['profit_factor']:.2f}")

                # Optimal conditions
                optimal = self.get_optimal_conditions(strat)
                if 'error' not in optimal:
                    print(f"\n  Optimal Conditions:")
                    if 'best_volatility_regime' in optimal:
                        bv = optimal['best_volatility_regime']
                        print(f"    Volatility: {bv['regime']} ({bv['win_rate']:.1%} win rate)")
                    if 'best_trend_alignment' in optimal:
                        bt = optimal['best_trend_alignment']
                        print(f"    Trend: {bt['alignment']} ({bt['win_rate']:.1%} win rate)")
                    if 'best_time_of_day' in optimal:
                        print(f"    Time: {optimal['best_time_of_day']['time']}")

            print("\n" + "=" * 70)

        finally:
            conn.close()


# ============================================================================
# FACTORY
# ============================================================================

_scorer: Optional[SignalScorer] = None

def get_signal_scorer() -> SignalScorer:
    """Get or create global signal scorer."""
    global _scorer
    if _scorer is None:
        _scorer = SignalScorer()
    return _scorer


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("SIGNAL SCORING SYSTEM DEMO")
    print("=" * 60)

    scorer = SignalScorer()

    # Record some test signals with outcomes
    test_signals = [
        ("momentum", "buy", "low", "with_trend", True, 3.5, "win"),
        ("momentum", "buy", "low", "with_trend", True, 2.1, "win"),
        ("momentum", "buy", "normal", "with_trend", False, -1.5, "loss"),
        ("momentum", "buy", "high", "counter_trend", False, -4.2, "loss"),
        ("momentum", "sell", "normal", "with_trend", True, 1.8, "win"),
        ("mean_reversion", "buy", "high", "counter_trend", True, 2.8, "win"),
        ("mean_reversion", "buy", "normal", "neutral", False, -0.5, "loss"),
    ]

    print("\nRecording test signals...")
    for i, (strat, sig_type, vol, trend, vol_conf, profit, outcome) in enumerate(test_signals):
        signal_id = f"TEST_{i}_{datetime.now().timestamp()}"
        scorer.record_signal(
            signal_id=signal_id,
            symbol="TEST",
            strategy=strat,
            signal_type=sig_type,
            vix_level=15 if vol == "low" else (25 if vol == "high" else 20),
            trend_alignment=trend,
            volume_confirmed=vol_conf,
        )
        scorer.record_outcome(
            signal_id=signal_id,
            outcome=outcome,
            profit_pct=profit,
            hold_days=3,
            max_drawdown_pct=abs(profit) if profit < 0 else profit * 0.3,
        )

    # Score a new signal
    print("\nScoring new momentum signal...")
    score = scorer.score_signal(
        strategy="momentum",
        signal_type="buy",
        vix_level=14,
        trend_alignment="with_trend",
        volume_confirmed=True,
    )

    print(f"  {score}")
    print(f"  Factors: {score.factors}")

    should_trade, reason = scorer.should_trade(score)
    print(f"  Should trade: {should_trade} ({reason})")

    # Print report
    scorer.print_report()
