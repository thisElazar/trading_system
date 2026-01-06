"""
Strategy Correlation Matrix
============================
Calculates and monitors correlations between strategy returns for
portfolio construction and risk management.

Features:
- Pairwise strategy return correlations
- Rolling correlation windows
- Regime-dependent correlation analysis
- Correlation-based position sizing recommendations
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    """Result of correlation analysis."""
    correlation_matrix: pd.DataFrame
    rolling_correlations: Dict[str, pd.DataFrame] = field(default_factory=dict)
    regime_correlations: Dict[str, pd.DataFrame] = field(default_factory=dict)
    eigenvalues: np.ndarray = None
    condition_number: float = 0.0
    effective_strategies: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    def get_pair_correlation(self, strategy1: str, strategy2: str) -> float:
        """Get correlation between two strategies."""
        if strategy1 in self.correlation_matrix.index and strategy2 in self.correlation_matrix.columns:
            return self.correlation_matrix.loc[strategy1, strategy2]
        return 0.0

    def get_highly_correlated_pairs(self, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """Get pairs of strategies with correlation above threshold."""
        pairs = []
        strategies = self.correlation_matrix.index.tolist()
        for i, s1 in enumerate(strategies):
            for s2 in strategies[i+1:]:
                corr = self.correlation_matrix.loc[s1, s2]
                if abs(corr) >= threshold:
                    pairs.append((s1, s2, corr))
        return sorted(pairs, key=lambda x: abs(x[2]), reverse=True)

    def get_diversification_score(self) -> float:
        """
        Calculate portfolio diversification score.

        Higher score = more diversification (lower correlations).
        Score of 1.0 = all strategies uncorrelated.
        Score approaching 0 = highly correlated strategies.
        """
        if self.correlation_matrix is None or len(self.correlation_matrix) < 2:
            return 0.0

        # Average absolute off-diagonal correlation
        n = len(self.correlation_matrix)
        corr_values = self.correlation_matrix.values

        # Get off-diagonal elements
        off_diag = []
        for i in range(n):
            for j in range(i+1, n):
                off_diag.append(abs(corr_values[i, j]))

        if not off_diag:
            return 1.0

        avg_corr = np.mean(off_diag)
        return 1.0 - avg_corr


class StrategyCorrelationAnalyzer:
    """
    Analyzes correlations between trading strategy returns.

    Usage:
        analyzer = StrategyCorrelationAnalyzer()

        # Add strategy returns
        for strategy_name, returns_series in strategy_returns.items():
            analyzer.add_strategy_returns(strategy_name, returns_series)

        # Calculate correlations
        result = analyzer.calculate_correlations()

        # Get position sizing recommendations
        weights = analyzer.get_correlation_adjusted_weights()
    """

    def __init__(self, min_periods: int = 30, rolling_window: int = 63):
        """
        Initialize correlation analyzer.

        Args:
            min_periods: Minimum periods required for correlation calculation
            rolling_window: Window size for rolling correlations (default: 63 = 3 months)
        """
        self.min_periods = min_periods
        self.rolling_window = rolling_window
        self.strategy_returns: Dict[str, pd.Series] = {}
        self.vix_data: Optional[pd.Series] = None

    def add_strategy_returns(self, strategy_name: str, returns: pd.Series) -> None:
        """
        Add or update returns for a strategy.

        Args:
            strategy_name: Name of the strategy
            returns: Daily returns series with DatetimeIndex
        """
        if not isinstance(returns.index, pd.DatetimeIndex):
            returns.index = pd.to_datetime(returns.index)

        # Ensure timezone-naive for consistency
        if returns.index.tz is not None:
            returns.index = returns.index.tz_localize(None)

        self.strategy_returns[strategy_name] = returns.sort_index()
        logger.debug(f"Added returns for {strategy_name}: {len(returns)} observations")

    def set_vix_data(self, vix_data: pd.Series) -> None:
        """Set VIX data for regime-based correlation analysis."""
        if not isinstance(vix_data.index, pd.DatetimeIndex):
            vix_data.index = pd.to_datetime(vix_data.index)
        if vix_data.index.tz is not None:
            vix_data.index = vix_data.index.tz_localize(None)
        self.vix_data = vix_data.sort_index()

    def _build_returns_matrix(self) -> pd.DataFrame:
        """Build aligned returns matrix from all strategies."""
        if not self.strategy_returns:
            return pd.DataFrame()

        # Combine all returns into a DataFrame
        returns_df = pd.DataFrame(self.strategy_returns)

        # Forward fill small gaps (up to 5 days)
        returns_df = returns_df.ffill(limit=5)

        # Drop rows where any strategy has no data
        returns_df = returns_df.dropna()

        return returns_df

    def calculate_correlations(self) -> CorrelationResult:
        """
        Calculate full correlation analysis.

        Returns:
            CorrelationResult with all correlation metrics
        """
        returns_df = self._build_returns_matrix()

        if len(returns_df) < self.min_periods:
            logger.warning(f"Insufficient data for correlation: {len(returns_df)} < {self.min_periods}")
            return CorrelationResult(
                correlation_matrix=pd.DataFrame(),
                effective_strategies=0
            )

        # Full-period correlation matrix
        corr_matrix = returns_df.corr()

        # Rolling correlations (between each pair)
        rolling_corrs = {}
        strategies = list(returns_df.columns)
        for i, s1 in enumerate(strategies):
            for s2 in strategies[i+1:]:
                pair_name = f"{s1}_vs_{s2}"
                rolling_corrs[pair_name] = returns_df[s1].rolling(
                    window=self.rolling_window,
                    min_periods=self.min_periods
                ).corr(returns_df[s2])

        # Regime-based correlations
        regime_corrs = self._calculate_regime_correlations(returns_df)

        # Calculate eigenvalues for condition number
        try:
            eigenvalues = np.linalg.eigvalsh(corr_matrix.values)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Descending
            condition_number = eigenvalues[0] / eigenvalues[-1] if eigenvalues[-1] > 1e-10 else float('inf')

            # Effective number of strategies (based on eigenvalue distribution)
            total_var = np.sum(eigenvalues)
            cumsum = np.cumsum(eigenvalues)
            effective_strategies = np.searchsorted(cumsum, total_var * 0.95) + 1
        except Exception as e:
            logger.warning(f"Eigenvalue calculation failed: {e}")
            eigenvalues = None
            condition_number = 0.0
            effective_strategies = len(strategies)

        return CorrelationResult(
            correlation_matrix=corr_matrix,
            rolling_correlations=rolling_corrs,
            regime_correlations=regime_corrs,
            eigenvalues=eigenvalues,
            condition_number=condition_number,
            effective_strategies=effective_strategies
        )

    def _calculate_regime_correlations(self, returns_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Calculate correlations for different VIX regimes."""
        if self.vix_data is None or len(self.vix_data) == 0:
            return {}

        # Align VIX with returns
        aligned_vix = self.vix_data.reindex(returns_df.index, method='ffill')

        # Define regimes
        regimes = {
            'low_vol': aligned_vix < 15,
            'normal': (aligned_vix >= 15) & (aligned_vix < 25),
            'high_vol': (aligned_vix >= 25) & (aligned_vix < 40),
            'extreme': aligned_vix >= 40
        }

        regime_corrs = {}
        for regime_name, mask in regimes.items():
            regime_returns = returns_df[mask]
            if len(regime_returns) >= self.min_periods:
                regime_corrs[regime_name] = regime_returns.corr()
            else:
                logger.debug(f"Insufficient data for {regime_name} regime: {len(regime_returns)}")

        return regime_corrs

    def get_correlation_adjusted_weights(self,
                                         base_weights: Dict[str, float] = None,
                                         target_correlation: float = 0.3) -> Dict[str, float]:
        """
        Calculate correlation-adjusted position weights.

        Reduces weights for highly correlated strategies to improve diversification.

        Args:
            base_weights: Initial equal or custom weights (default: equal weight)
            target_correlation: Target average correlation (reduce if above)

        Returns:
            Adjusted weights normalized to sum to 1.0
        """
        result = self.calculate_correlations()

        if result.correlation_matrix is None or len(result.correlation_matrix) == 0:
            # No correlation data, return equal weights
            n = len(self.strategy_returns)
            return {s: 1.0/n for s in self.strategy_returns}

        strategies = list(result.correlation_matrix.index)
        n = len(strategies)

        if base_weights is None:
            base_weights = {s: 1.0/n for s in strategies}

        # Calculate average correlation for each strategy
        avg_corrs = {}
        for s in strategies:
            other_corrs = [
                abs(result.correlation_matrix.loc[s, other])
                for other in strategies if other != s
            ]
            avg_corrs[s] = np.mean(other_corrs) if other_corrs else 0.0

        # Reduce weights for strategies with high average correlation
        adjusted = {}
        for s in strategies:
            base = base_weights.get(s, 1.0/n)
            if avg_corrs[s] > target_correlation:
                # Reduce weight proportionally to excess correlation
                reduction = (avg_corrs[s] - target_correlation) / (1.0 - target_correlation)
                adjusted[s] = base * (1.0 - reduction * 0.5)  # Max 50% reduction
            else:
                adjusted[s] = base

        # Normalize to sum to 1.0
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {s: w/total for s, w in adjusted.items()}

        return adjusted


def calculate_strategy_correlations(
    backtest_results: Dict[str, 'BacktestResult'],
    vix_data: pd.Series = None
) -> CorrelationResult:
    """
    Convenience function to calculate correlations from backtest results.

    Args:
        backtest_results: Dict mapping strategy names to BacktestResult objects
        vix_data: Optional VIX data for regime analysis

    Returns:
        CorrelationResult with all correlation metrics
    """
    analyzer = StrategyCorrelationAnalyzer()

    for name, result in backtest_results.items():
        if result.equity_curve and len(result.equity_curve) > 1:
            equity = pd.Series(result.equity_curve)
            returns = equity.pct_change().dropna()

            # Try to get dates from trades
            if result.trades and len(result.trades) > 0:
                try:
                    dates = pd.to_datetime([t.get('entry_date', t.get('exit_date'))
                                           for t in result.trades if t.get('entry_date') or t.get('exit_date')])
                    if len(dates) > 0:
                        # Create date index aligned with returns length
                        start_date = dates.min()
                        date_index = pd.date_range(start=start_date, periods=len(returns), freq='D')
                        returns.index = date_index
                except Exception:
                    # Fall back to integer index
                    pass

            analyzer.add_strategy_returns(name, returns)

    if vix_data is not None:
        analyzer.set_vix_data(vix_data)

    return analyzer.calculate_correlations()


if __name__ == "__main__":
    # Example usage
    import numpy as np

    logging.basicConfig(level=logging.INFO)

    # Generate sample returns for 4 strategies
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')

    # Create correlated returns
    base_returns = np.random.randn(500) * 0.01

    strategy_returns = {
        'momentum': pd.Series(base_returns + np.random.randn(500) * 0.005, index=dates),
        'value': pd.Series(-base_returns * 0.3 + np.random.randn(500) * 0.008, index=dates),
        'mean_reversion': pd.Series(base_returns * 0.5 + np.random.randn(500) * 0.006, index=dates),
        'pairs': pd.Series(np.random.randn(500) * 0.007, index=dates),  # Uncorrelated
    }

    # Analyze correlations
    analyzer = StrategyCorrelationAnalyzer()
    for name, returns in strategy_returns.items():
        analyzer.add_strategy_returns(name, returns)

    result = analyzer.calculate_correlations()

    print("\nCorrelation Matrix:")
    print(result.correlation_matrix.round(3))

    print(f"\nDiversification Score: {result.get_diversification_score():.3f}")
    print(f"Condition Number: {result.condition_number:.2f}")
    print(f"Effective Strategies: {result.effective_strategies}")

    print("\nHighly Correlated Pairs (>0.5):")
    for s1, s2, corr in result.get_highly_correlated_pairs(0.5):
        print(f"  {s1} vs {s2}: {corr:.3f}")

    print("\nCorrelation-Adjusted Weights:")
    weights = analyzer.get_correlation_adjusted_weights()
    for s, w in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {s}: {w:.3f}")
