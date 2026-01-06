"""
Portfolio-Level Fitness Evaluation
===================================
Evaluates strategies based on their contribution to portfolio performance,
not just standalone metrics.

Key concepts:
- Marginal Sharpe Ratio: How much does adding this strategy improve portfolio Sharpe?
- Diversification Ratio: Does this strategy provide uncorrelated returns?
- Correlation Penalty: Reject highly correlated strategies
- Composite Fitness: Weighted combination of standalone and portfolio metrics

Research basis:
- Marginal contribution more predictive than standalone Sharpe
- Diversification ratio captures true portfolio benefit
- Correlation threshold of 0.7 prevents redundancy
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from research.backtester import BacktestResult

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PortfolioContribution:
    """
    Measures a strategy's contribution to portfolio performance.

    These metrics go beyond standalone performance to capture
    how a strategy interacts with the existing portfolio.
    """
    # Marginal contribution metrics
    marginal_sharpe: float = 0.0          # Sharpe improvement from adding strategy
    marginal_sortino: float = 0.0          # Sortino improvement
    marginal_return: float = 0.0           # Return contribution
    marginal_volatility: float = 0.0       # Volatility contribution

    # Correlation metrics
    max_correlation: float = 0.0           # Max correlation to any existing strategy
    avg_correlation: float = 0.0           # Average correlation to portfolio
    correlation_to_portfolio: float = 0.0  # Correlation to combined portfolio returns

    # Diversification metrics
    diversification_ratio: float = 1.0     # Portfolio vol / weighted avg vol
    optimal_allocation_pct: float = 0.0    # Suggested allocation (0-100)

    # Risk contribution
    marginal_var: float = 0.0              # Marginal Value at Risk contribution
    beta_to_portfolio: float = 1.0         # Beta relative to existing portfolio

    # Flags
    is_redundant: bool = False             # True if max_corr > threshold
    improves_sharpe: bool = False          # True if marginal_sharpe > 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CompositeScore:
    """
    Final composite fitness score combining standalone and portfolio metrics.
    """
    # Component scores (0-1 normalized)
    standalone_sortino_score: float = 0.0
    marginal_sharpe_score: float = 0.0
    diversification_score: float = 0.0
    max_drawdown_score: float = 0.0
    novelty_score: float = 0.0

    # Weighted composite
    composite_fitness: float = 0.0

    # Raw values (for reference)
    raw_sortino: float = 0.0
    raw_marginal_sharpe: float = 0.0
    raw_diversification_ratio: float = 1.0
    raw_max_drawdown: float = 0.0
    raw_novelty: float = 0.0

    # Rejection flags
    rejected: bool = False
    rejection_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FitnessWeights:
    """
    Weights for composite fitness calculation.

    Default weights based on research:
    - Marginal Sharpe most predictive of out-of-sample performance
    - Standalone Sortino captures individual strategy quality
    - Diversification ensures portfolio benefit
    - Max drawdown for risk management
    - Novelty for exploration
    """
    standalone_sortino: float = 0.25
    marginal_sharpe: float = 0.30
    diversification: float = 0.20
    max_drawdown: float = 0.15
    novelty: float = 0.10

    def __post_init__(self):
        total = (self.standalone_sortino + self.marginal_sharpe +
                 self.diversification + self.max_drawdown + self.novelty)
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Fitness weights sum to {total}, normalizing to 1.0")
            self.standalone_sortino /= total
            self.marginal_sharpe /= total
            self.diversification /= total
            self.max_drawdown /= total
            self.novelty /= total


# =============================================================================
# PORTFOLIO FITNESS EVALUATOR
# =============================================================================

class PortfolioFitnessEvaluator:
    """
    Evaluates strategy fitness in portfolio context.

    Maintains history of existing portfolio strategies and their returns
    to calculate marginal contributions for new candidates.

    Usage:
        evaluator = PortfolioFitnessEvaluator()
        evaluator.add_portfolio_strategy('momentum_1', returns_series)
        evaluator.add_portfolio_strategy('mean_rev_1', returns_series)

        # Evaluate new candidate
        contribution = evaluator.evaluate_contribution(candidate_returns)
        score = evaluator.calculate_composite_score(backtest_result, contribution)
    """

    # Thresholds
    MAX_CORRELATION_THRESHOLD = 0.70       # Reject if correlation exceeds this
    MIN_MARGINAL_SHARPE = -0.10            # Allow slightly negative (diversification benefit)
    MIN_TRADES = 30                         # Minimum trades for valid evaluation
    MIN_ALLOCATION = 0.02                   # 2% minimum allocation
    MAX_ALLOCATION = 0.25                   # 25% maximum per strategy

    def __init__(
        self,
        db_path: Optional[Path] = None,
        weights: FitnessWeights = None,
        risk_free_rate: float = 0.05
    ):
        """
        Initialize portfolio fitness evaluator.

        Args:
            db_path: Path to database for persistence
            weights: Custom fitness weights
            risk_free_rate: Annual risk-free rate for Sharpe calculations
        """
        self.db_path = db_path or Path(__file__).parent.parent.parent / "data" / "portfolio_fitness.db"
        self.weights = weights or FitnessWeights()
        self.risk_free_rate = risk_free_rate
        self.daily_rf = (1 + risk_free_rate) ** (1/252) - 1

        # In-memory portfolio state
        self._portfolio_returns: Dict[str, pd.Series] = {}
        self._portfolio_weights: Dict[str, float] = {}
        self._correlation_matrix: Optional[pd.DataFrame] = None
        self._combined_returns: Optional[pd.Series] = None

        # Initialize database
        self._init_db()
        self._load_portfolio_state()

    def _init_db(self):
        """Initialize database tables."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS portfolio_strategies (
                    strategy_name TEXT PRIMARY KEY,
                    weight REAL NOT NULL,
                    added_date TEXT NOT NULL,
                    status TEXT DEFAULT 'active',
                    sharpe REAL,
                    sortino REAL,
                    max_drawdown REAL
                );

                CREATE TABLE IF NOT EXISTS strategy_returns (
                    strategy_name TEXT,
                    date TEXT,
                    daily_return REAL,
                    PRIMARY KEY (strategy_name, date)
                );

                CREATE TABLE IF NOT EXISTS fitness_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT,
                    timestamp TEXT,
                    marginal_sharpe REAL,
                    max_correlation REAL,
                    diversification_ratio REAL,
                    composite_score REAL,
                    rejected INTEGER,
                    rejection_reason TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_returns_strategy
                ON strategy_returns(strategy_name);

                CREATE INDEX IF NOT EXISTS idx_returns_date
                ON strategy_returns(date);
            """)

    def _load_portfolio_state(self):
        """Load existing portfolio strategies from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Load active strategies
                cursor = conn.execute("""
                    SELECT strategy_name, weight
                    FROM portfolio_strategies
                    WHERE status = 'active'
                """)

                for row in cursor:
                    strategy_name, weight = row
                    self._portfolio_weights[strategy_name] = weight

                    # Load returns
                    returns_cursor = conn.execute("""
                        SELECT date, daily_return
                        FROM strategy_returns
                        WHERE strategy_name = ?
                        ORDER BY date
                    """, (strategy_name,))

                    dates = []
                    returns = []
                    for r in returns_cursor:
                        dates.append(pd.Timestamp(r[0]))
                        returns.append(r[1])

                    if dates:
                        self._portfolio_returns[strategy_name] = pd.Series(
                            returns, index=dates, name=strategy_name
                        )

                if self._portfolio_returns:
                    self._update_combined_returns()
                    logger.info(f"Loaded {len(self._portfolio_returns)} portfolio strategies")

        except Exception as e:
            logger.warning(f"Failed to load portfolio state: {e}")

    def _update_combined_returns(self):
        """Recalculate combined portfolio returns and correlation matrix."""
        if not self._portfolio_returns:
            self._combined_returns = None
            self._correlation_matrix = None
            return

        # Create returns DataFrame
        returns_df = pd.DataFrame(self._portfolio_returns)

        # Calculate correlation matrix
        self._correlation_matrix = returns_df.corr()

        # Calculate weighted portfolio returns
        weights = pd.Series(self._portfolio_weights)
        weights = weights / weights.sum()  # Normalize

        # Align weights with returns columns
        aligned_weights = weights.reindex(returns_df.columns).fillna(0)

        self._combined_returns = (returns_df * aligned_weights).sum(axis=1)
        self._combined_returns.name = 'portfolio'

    # =========================================================================
    # PORTFOLIO MANAGEMENT
    # =========================================================================

    def add_portfolio_strategy(
        self,
        strategy_name: str,
        returns: pd.Series,
        weight: float = 0.10,
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        Add a strategy to the existing portfolio.

        Args:
            strategy_name: Unique strategy identifier
            returns: Daily returns series with datetime index
            weight: Portfolio weight (0-1)
            metrics: Optional dict with sharpe, sortino, max_drawdown
        """
        # Store in memory
        self._portfolio_returns[strategy_name] = returns.copy()
        self._portfolio_weights[strategy_name] = weight

        # Update combined returns
        self._update_combined_returns()

        # Persist to database
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Insert strategy
                conn.execute("""
                    INSERT OR REPLACE INTO portfolio_strategies
                    (strategy_name, weight, added_date, status, sharpe, sortino, max_drawdown)
                    VALUES (?, ?, ?, 'active', ?, ?, ?)
                """, (
                    strategy_name,
                    weight,
                    datetime.now().isoformat(),
                    metrics.get('sharpe') if metrics else None,
                    metrics.get('sortino') if metrics else None,
                    metrics.get('max_drawdown') if metrics else None
                ))

                # Insert returns
                for date, ret in returns.items():
                    conn.execute("""
                        INSERT OR REPLACE INTO strategy_returns
                        (strategy_name, date, daily_return)
                        VALUES (?, ?, ?)
                    """, (strategy_name, str(date.date()), float(ret)))

        except Exception as e:
            logger.error(f"Failed to persist strategy {strategy_name}: {e}")

        logger.info(f"Added strategy '{strategy_name}' to portfolio (weight={weight:.1%})")

    def remove_portfolio_strategy(self, strategy_name: str):
        """Remove a strategy from the portfolio."""
        if strategy_name in self._portfolio_returns:
            del self._portfolio_returns[strategy_name]
        if strategy_name in self._portfolio_weights:
            del self._portfolio_weights[strategy_name]

        self._update_combined_returns()

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE portfolio_strategies
                    SET status = 'removed'
                    WHERE strategy_name = ?
                """, (strategy_name,))
        except Exception as e:
            logger.error(f"Failed to remove strategy: {e}")

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get summary of current portfolio composition."""
        if not self._portfolio_returns:
            return {
                'num_strategies': 0,
                'strategies': [],
                'total_weight': 0,
                'portfolio_sharpe': None,
                'portfolio_sortino': None
            }

        summary = {
            'num_strategies': len(self._portfolio_returns),
            'strategies': list(self._portfolio_returns.keys()),
            'weights': dict(self._portfolio_weights),
            'total_weight': sum(self._portfolio_weights.values()),
        }

        if self._combined_returns is not None and len(self._combined_returns) > 20:
            excess_returns = self._combined_returns - self.daily_rf
            sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)

            downside = excess_returns[excess_returns < 0]
            downside_std = downside.std() if len(downside) > 0 else 0.01
            sortino = excess_returns.mean() / downside_std * np.sqrt(252)

            summary['portfolio_sharpe'] = sharpe
            summary['portfolio_sortino'] = sortino

        return summary

    # =========================================================================
    # MARGINAL SHARPE CALCULATION
    # =========================================================================

    def calculate_marginal_sharpe(
        self,
        candidate_returns: pd.Series,
        candidate_weight: float = 0.10
    ) -> Tuple[float, float]:
        """
        Calculate marginal Sharpe ratio from adding a strategy.

        Uses the formula:
        MSR = (σ_s/σ_p) × SR_s - β × SR_p

        Where:
        - σ_s = strategy volatility
        - σ_p = portfolio volatility
        - SR_s = strategy Sharpe
        - β = strategy beta to portfolio
        - SR_p = portfolio Sharpe

        Args:
            candidate_returns: Daily returns of candidate strategy
            candidate_weight: Proposed weight for candidate

        Returns:
            Tuple of (marginal_sharpe, new_portfolio_sharpe)
        """
        if self._combined_returns is None or len(self._combined_returns) < 20:
            # No existing portfolio - marginal = standalone
            excess = candidate_returns - self.daily_rf
            if excess.std() > 0:
                return float(excess.mean() / excess.std() * np.sqrt(252)), 0.0
            return 0.0, 0.0

        # Align returns
        aligned = pd.DataFrame({
            'portfolio': self._combined_returns,
            'candidate': candidate_returns
        }).dropna()

        if len(aligned) < 20:
            return 0.0, 0.0

        # Current portfolio metrics
        port_excess = aligned['portfolio'] - self.daily_rf
        port_sharpe = port_excess.mean() / port_excess.std() * np.sqrt(252) if port_excess.std() > 0 else 0
        port_vol = port_excess.std() * np.sqrt(252)

        # Candidate metrics
        cand_excess = aligned['candidate'] - self.daily_rf
        cand_sharpe = cand_excess.mean() / cand_excess.std() * np.sqrt(252) if cand_excess.std() > 0 else 0
        cand_vol = cand_excess.std() * np.sqrt(252)

        # Calculate beta of candidate to portfolio
        covariance = aligned['candidate'].cov(aligned['portfolio'])
        port_variance = aligned['portfolio'].var()
        beta = covariance / port_variance if port_variance > 0 else 1.0

        # Marginal Sharpe formula
        # MSR = (σ_s/σ_p) × SR_s - β × SR_p
        if port_vol > 0:
            marginal_sharpe = (cand_vol / port_vol) * cand_sharpe - beta * port_sharpe
        else:
            marginal_sharpe = cand_sharpe

        # Calculate new portfolio Sharpe with candidate added
        # Normalize weights
        total_weight = sum(self._portfolio_weights.values()) + candidate_weight
        new_port_weight = (total_weight - candidate_weight) / total_weight
        new_cand_weight = candidate_weight / total_weight

        new_returns = (aligned['portfolio'] * new_port_weight +
                       aligned['candidate'] * new_cand_weight)
        new_excess = new_returns - self.daily_rf
        new_sharpe = new_excess.mean() / new_excess.std() * np.sqrt(252) if new_excess.std() > 0 else 0

        return float(marginal_sharpe), float(new_sharpe)

    # =========================================================================
    # CONTRIBUTION EVALUATION
    # =========================================================================

    def evaluate_contribution(
        self,
        candidate_returns: pd.Series,
        candidate_weight: float = 0.10
    ) -> PortfolioContribution:
        """
        Evaluate a candidate strategy's contribution to the portfolio.

        Args:
            candidate_returns: Daily returns series
            candidate_weight: Proposed portfolio weight

        Returns:
            PortfolioContribution with all metrics
        """
        contribution = PortfolioContribution()

        # If no existing portfolio, return standalone metrics
        if not self._portfolio_returns or self._combined_returns is None:
            excess = candidate_returns - self.daily_rf
            if len(excess) > 0 and excess.std() > 0:
                contribution.marginal_sharpe = excess.mean() / excess.std() * np.sqrt(252)
                contribution.diversification_ratio = 1.0
                contribution.optimal_allocation_pct = min(self.MAX_ALLOCATION * 100, 10)
                contribution.improves_sharpe = contribution.marginal_sharpe > 0
            return contribution

        # Calculate marginal Sharpe
        marginal_sharpe, new_portfolio_sharpe = self.calculate_marginal_sharpe(
            candidate_returns, candidate_weight
        )
        contribution.marginal_sharpe = marginal_sharpe

        # Align returns with existing strategies
        all_returns = pd.DataFrame(self._portfolio_returns)
        all_returns['candidate'] = candidate_returns
        aligned = all_returns.dropna()

        if len(aligned) < 20:
            contribution.is_redundant = True
            return contribution

        # Calculate correlations
        correlations = aligned.corr()['candidate'].drop('candidate')

        contribution.max_correlation = float(correlations.abs().max())
        contribution.avg_correlation = float(correlations.abs().mean())

        # Correlation to combined portfolio
        if self._combined_returns is not None:
            aligned_portfolio = pd.DataFrame({
                'portfolio': self._combined_returns,
                'candidate': candidate_returns
            }).dropna()

            if len(aligned_portfolio) > 10:
                contribution.correlation_to_portfolio = float(
                    aligned_portfolio['candidate'].corr(aligned_portfolio['portfolio'])
                )

        # Check redundancy
        contribution.is_redundant = contribution.max_correlation > self.MAX_CORRELATION_THRESHOLD

        # Calculate diversification ratio
        # DR = (weighted avg of individual vols) / portfolio vol
        individual_vols = aligned.std() * np.sqrt(252)
        weights = list(self._portfolio_weights.values()) + [candidate_weight]
        weights = np.array(weights) / np.sum(weights)

        weighted_avg_vol = np.sum(individual_vols.values * weights)

        # Calculate portfolio vol with new strategy
        cov_matrix = aligned.cov() * 252  # Annualize
        portfolio_vol = np.sqrt(weights @ cov_matrix.values @ weights)

        contribution.diversification_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1.0

        # Calculate optimal allocation using mean-variance
        contribution.optimal_allocation_pct = self._calculate_optimal_allocation(
            aligned, candidate_weight
        ) * 100

        # Beta to portfolio
        if self._combined_returns is not None and len(aligned_portfolio) > 10:
            covariance = aligned_portfolio['candidate'].cov(aligned_portfolio['portfolio'])
            portfolio_var = aligned_portfolio['portfolio'].var()
            contribution.beta_to_portfolio = covariance / portfolio_var if portfolio_var > 0 else 1.0

        # Marginal VaR (simplified)
        contribution.marginal_var = self._calculate_marginal_var(
            aligned['candidate'], candidate_weight
        )

        # Check if improves Sharpe
        contribution.improves_sharpe = marginal_sharpe > self.MIN_MARGINAL_SHARPE

        return contribution

    def _calculate_optimal_allocation(
        self,
        returns_df: pd.DataFrame,
        candidate_weight: float
    ) -> float:
        """
        Calculate optimal allocation using simplified mean-variance.

        Uses the formula:
        w* = (1/γ) × Σ^(-1) × μ

        Simplified to avoid matrix inversion issues.
        """
        try:
            # Get candidate excess returns
            candidate_excess = returns_df['candidate'] - self.daily_rf
            expected_return = candidate_excess.mean() * 252
            volatility = candidate_excess.std() * np.sqrt(252)

            # Simple allocation based on Sharpe contribution
            if volatility > 0:
                sharpe = expected_return / volatility

                # Scale by Sharpe ratio, with bounds
                # Higher Sharpe -> higher allocation, but capped
                raw_allocation = min(0.5, max(0.02, 0.05 + sharpe * 0.05))

                # Reduce if highly correlated
                max_corr = returns_df.corr()['candidate'].drop('candidate').abs().max()
                correlation_penalty = max(0.5, 1 - max_corr)

                optimal = raw_allocation * correlation_penalty

                return max(self.MIN_ALLOCATION, min(self.MAX_ALLOCATION, optimal))

            return self.MIN_ALLOCATION

        except Exception as e:
            logger.warning(f"Optimal allocation calculation failed: {e}")
            return candidate_weight

    def _calculate_marginal_var(
        self,
        candidate_returns: pd.Series,
        weight: float,
        confidence: float = 0.95
    ) -> float:
        """Calculate marginal Value at Risk contribution."""
        if self._combined_returns is None:
            return float(candidate_returns.quantile(1 - confidence))

        # VaR of portfolio with and without candidate
        aligned = pd.DataFrame({
            'portfolio': self._combined_returns,
            'candidate': candidate_returns
        }).dropna()

        if len(aligned) < 20:
            return 0.0

        # Current portfolio VaR
        current_var = aligned['portfolio'].quantile(1 - confidence)

        # New portfolio VaR
        total_weight = sum(self._portfolio_weights.values()) + weight
        port_weight = (total_weight - weight) / total_weight
        cand_weight = weight / total_weight

        new_portfolio = aligned['portfolio'] * port_weight + aligned['candidate'] * cand_weight
        new_var = new_portfolio.quantile(1 - confidence)

        # Marginal VaR
        return float(new_var - current_var)

    # =========================================================================
    # COMPOSITE SCORE CALCULATION
    # =========================================================================

    def calculate_composite_score(
        self,
        backtest_result: BacktestResult,
        contribution: PortfolioContribution,
        novelty_score: float = 0.0
    ) -> CompositeScore:
        """
        Calculate composite fitness score combining all metrics.

        Weights (default):
        - Standalone Sortino: 25%
        - Marginal Sharpe: 30%
        - Diversification: 20%
        - Max Drawdown: 15%
        - Novelty: 10%

        Args:
            backtest_result: Result from backtesting the strategy
            contribution: Portfolio contribution metrics
            novelty_score: Behavioral novelty score (0+)

        Returns:
            CompositeScore with final fitness
        """
        score = CompositeScore()

        # Check for rejection conditions
        if contribution.is_redundant:
            score.rejected = True
            score.rejection_reason = f"Max correlation {contribution.max_correlation:.2f} > {self.MAX_CORRELATION_THRESHOLD}"
            return score

        if backtest_result.trade_count < self.MIN_TRADES:
            score.rejected = True
            score.rejection_reason = f"Insufficient trades: {backtest_result.trade_count} < {self.MIN_TRADES}"
            return score

        if contribution.marginal_sharpe < self.MIN_MARGINAL_SHARPE:
            score.rejected = True
            score.rejection_reason = f"Negative marginal Sharpe: {contribution.marginal_sharpe:.2f}"
            return score

        # Store raw values
        score.raw_sortino = backtest_result.sortino_ratio or 0.0
        score.raw_marginal_sharpe = contribution.marginal_sharpe
        score.raw_diversification_ratio = contribution.diversification_ratio
        score.raw_max_drawdown = backtest_result.max_drawdown_pct or 0.0
        score.raw_novelty = novelty_score

        # Normalize to 0-1 scores

        # Sortino: 0 = 0, 3+ = 1
        score.standalone_sortino_score = min(1.0, max(0.0, score.raw_sortino / 3.0))

        # Marginal Sharpe: -0.5 = 0, 1+ = 1
        score.marginal_sharpe_score = min(1.0, max(0.0, (score.raw_marginal_sharpe + 0.5) / 1.5))

        # Diversification: 1.0 = 0, 1.5+ = 1 (higher is better)
        score.diversification_score = min(1.0, max(0.0, (score.raw_diversification_ratio - 1.0) / 0.5))

        # Max Drawdown: -50% = 0, 0% = 1 (less negative is better)
        score.max_drawdown_score = min(1.0, max(0.0, 1.0 + score.raw_max_drawdown / 50.0))

        # Novelty: 0 = 0, 100 = 1
        score.novelty_score = min(1.0, max(0.0, score.raw_novelty / 100.0))

        # Calculate weighted composite
        score.composite_fitness = (
            self.weights.standalone_sortino * score.standalone_sortino_score +
            self.weights.marginal_sharpe * score.marginal_sharpe_score +
            self.weights.diversification * score.diversification_score +
            self.weights.max_drawdown * score.max_drawdown_score +
            self.weights.novelty * score.novelty_score
        )

        return score

    # =========================================================================
    # INTEGRATION WITH EVOLUTION ENGINE
    # =========================================================================

    def evaluate_for_evolution(
        self,
        backtest_result: BacktestResult,
        novelty_score: float = 0.0,
        candidate_weight: float = 0.05
    ) -> Tuple[CompositeScore, PortfolioContribution]:
        """
        One-shot evaluation for evolution engine integration.

        Args:
            backtest_result: Backtest result with equity curve
            novelty_score: Novelty score from novelty search
            candidate_weight: Proposed portfolio weight

        Returns:
            Tuple of (CompositeScore, PortfolioContribution)
        """
        # Extract returns from equity curve
        if not backtest_result.equity_curve:
            contribution = PortfolioContribution()
            score = CompositeScore(rejected=True, rejection_reason="No equity curve")
            return score, contribution

        equity = pd.Series(backtest_result.equity_curve)
        returns = equity.pct_change().dropna()

        if len(returns) < 20:
            contribution = PortfolioContribution()
            score = CompositeScore(rejected=True, rejection_reason="Insufficient returns data")
            return score, contribution

        # Convert index to datetime if needed
        if not isinstance(returns.index, pd.DatetimeIndex):
            returns.index = pd.date_range(end=datetime.now(), periods=len(returns), freq='D')

        # Evaluate contribution
        contribution = self.evaluate_contribution(returns, candidate_weight)

        # Calculate composite score
        score = self.calculate_composite_score(backtest_result, contribution, novelty_score)

        # Log evaluation
        self._log_evaluation(backtest_result, score, contribution)

        return score, contribution

    def _log_evaluation(
        self,
        result: BacktestResult,
        score: CompositeScore,
        contribution: PortfolioContribution
    ):
        """Log evaluation to database."""
        try:
            strategy_name = getattr(result, 'strategy_name', 'unknown')

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO fitness_evaluations
                    (strategy_name, timestamp, marginal_sharpe, max_correlation,
                     diversification_ratio, composite_score, rejected, rejection_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    strategy_name,
                    datetime.now().isoformat(),
                    contribution.marginal_sharpe,
                    contribution.max_correlation,
                    contribution.diversification_ratio,
                    score.composite_fitness,
                    1 if score.rejected else 0,
                    score.rejection_reason
                ))
        except Exception as e:
            logger.debug(f"Failed to log evaluation: {e}")


# =============================================================================
# DEMO AND TESTING
# =============================================================================

def demo():
    """Demonstrate portfolio fitness evaluation."""
    print("=" * 60)
    print("Portfolio Fitness Evaluator Demo")
    print("=" * 60)

    # Create evaluator
    evaluator = PortfolioFitnessEvaluator()

    # Simulate existing portfolio strategies
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=252, freq='D')

    # Strategy 1: Momentum-like (positive autocorrelation)
    momentum_returns = pd.Series(
        np.random.normal(0.0005, 0.015, 252),  # ~12% annual, 24% vol
        index=dates,
        name='momentum_1'
    )
    # Add momentum effect
    momentum_returns = momentum_returns + momentum_returns.shift(1).fillna(0) * 0.1

    # Strategy 2: Mean reversion-like (negative autocorrelation)
    mean_rev_returns = pd.Series(
        np.random.normal(0.0003, 0.012, 252),  # ~8% annual, 19% vol
        index=dates,
        name='mean_rev_1'
    )
    mean_rev_returns = mean_rev_returns - mean_rev_returns.shift(1).fillna(0) * 0.05

    # Add to portfolio
    evaluator.add_portfolio_strategy('momentum_1', momentum_returns, weight=0.5)
    evaluator.add_portfolio_strategy('mean_rev_1', mean_rev_returns, weight=0.5)

    print("\nPortfolio Summary:")
    summary = evaluator.get_portfolio_summary()
    print(f"  Strategies: {summary['strategies']}")
    print(f"  Portfolio Sharpe: {summary.get('portfolio_sharpe', 'N/A'):.2f}")

    # Evaluate candidate strategies
    print("\n" + "-" * 60)
    print("Evaluating Candidate Strategies")
    print("-" * 60)

    # Candidate 1: Uncorrelated trend following
    candidate_1 = pd.Series(
        np.random.normal(0.0004, 0.018, 252),
        index=dates,
        name='trend_1'
    )

    contribution_1 = evaluator.evaluate_contribution(candidate_1)
    print(f"\nCandidate 1 (Uncorrelated Trend):")
    print(f"  Marginal Sharpe: {contribution_1.marginal_sharpe:.3f}")
    print(f"  Max Correlation: {contribution_1.max_correlation:.3f}")
    print(f"  Diversification Ratio: {contribution_1.diversification_ratio:.3f}")
    print(f"  Redundant: {contribution_1.is_redundant}")
    print(f"  Improves Sharpe: {contribution_1.improves_sharpe}")

    # Candidate 2: Highly correlated with momentum
    candidate_2 = momentum_returns * 0.9 + pd.Series(
        np.random.normal(0, 0.005, 252), index=dates
    )

    contribution_2 = evaluator.evaluate_contribution(candidate_2)
    print(f"\nCandidate 2 (Correlated with Momentum):")
    print(f"  Marginal Sharpe: {contribution_2.marginal_sharpe:.3f}")
    print(f"  Max Correlation: {contribution_2.max_correlation:.3f}")
    print(f"  Diversification Ratio: {contribution_2.diversification_ratio:.3f}")
    print(f"  Redundant: {contribution_2.is_redundant}")
    print(f"  Improves Sharpe: {contribution_2.improves_sharpe}")

    # Candidate 3: Negative correlated (crisis alpha)
    candidate_3 = -momentum_returns * 0.3 + pd.Series(
        np.random.normal(0.0002, 0.01, 252), index=dates
    )

    contribution_3 = evaluator.evaluate_contribution(candidate_3)
    print(f"\nCandidate 3 (Crisis Alpha - Negative Corr):")
    print(f"  Marginal Sharpe: {contribution_3.marginal_sharpe:.3f}")
    print(f"  Max Correlation: {contribution_3.max_correlation:.3f}")
    print(f"  Diversification Ratio: {contribution_3.diversification_ratio:.3f}")
    print(f"  Redundant: {contribution_3.is_redundant}")
    print(f"  Improves Sharpe: {contribution_3.improves_sharpe}")

    print("\n" + "=" * 60)
    print("Demo Complete")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
