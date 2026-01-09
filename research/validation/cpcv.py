"""
Combinatorial Purged Cross-Validation (CPCV)
=============================================
Implementation of CPCV for detecting backtest overfitting.

Based on: "The Probability of Backtest Overfitting" by Bailey et al. (2015)

Key concepts:
- Divides data into S subsets
- Tests all combinations of (S-1) train subsets vs 1 test subset
- Applies purging (gap) and embargo to prevent data leakage
- Calculates PBO (Probability of Backtest Overfitting) via rank correlation
"""

import logging
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import combinations
from math import comb
from typing import Callable, Dict, Generator, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CPCVConfig:
    """Configuration for CPCV validation."""

    n_subsets: int = 16              # S parameter (number of data subsets)
    purge_days: int = 5              # Gap between train and test (leakage prevention)
    embargo_pct: float = 0.01        # Skip this % of data after train end
    max_combinations: int = 1000     # Sample for Pi efficiency (full=12870 for S=16)
    pbo_reject_threshold: float = 0.05  # Reject if PBO > this
    min_train_days: int = 252        # Minimum training days (1 year)
    min_test_days: int = 63          # Minimum test days (~3 months)
    n_workers: int = 2               # Parallel workers for Pi

    def validate(self) -> List[str]:
        """Validate configuration, return list of errors."""
        errors = []
        if self.n_subsets < 4:
            errors.append("n_subsets must be at least 4")
        if self.purge_days < 0:
            errors.append("purge_days must be non-negative")
        if not 0 <= self.embargo_pct < 0.5:
            errors.append("embargo_pct must be between 0 and 0.5")
        if self.max_combinations < 10:
            errors.append("max_combinations must be at least 10")
        return errors


@dataclass
class CPCVSplit:
    """Represents a single CPCV train/test split."""

    split_id: int
    train_indices: np.ndarray        # Indices into original data
    test_indices: np.ndarray         # Indices into original data
    train_dates: Tuple[pd.Timestamp, pd.Timestamp]  # (start, end)
    test_dates: Tuple[pd.Timestamp, pd.Timestamp]   # (start, end)

    @property
    def train_days(self) -> int:
        return len(self.train_indices)

    @property
    def test_days(self) -> int:
        return len(self.test_indices)


@dataclass
class CPCVSplitResult:
    """Result from a single CPCV split evaluation."""

    split_id: int
    is_sharpe: float          # In-sample Sharpe
    oos_sharpe: float         # Out-of-sample Sharpe
    is_return: float          # In-sample total return
    oos_return: float         # Out-of-sample total return
    is_trades: int            # Number of trades in-sample
    oos_trades: int           # Number of trades out-of-sample
    sharpe_degradation: float  # is_sharpe - oos_sharpe

    @property
    def is_overfit(self) -> bool:
        """Strategy appears overfit if IS >> OOS."""
        return self.is_sharpe > 0 and self.oos_sharpe < 0


@dataclass
class CPCVResult:
    """Overall CPCV validation result."""

    pbo: float                       # Probability of Backtest Overfitting [0,1]
    pbo_ci_95: Tuple[float, float]   # 95% confidence interval
    mean_is_sharpe: float            # Mean in-sample Sharpe
    mean_oos_sharpe: float           # Mean out-of-sample Sharpe
    mean_sharpe_degradation: float   # Mean (IS - OOS) Sharpe
    std_oos_sharpe: float            # Std of OOS Sharpe (stability)
    n_splits_completed: int          # Number of splits evaluated
    n_splits_overfit: int            # Splits where IS > 0 and OOS < 0
    split_results: List[CPCVSplitResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Strategy passes validation if PBO <= threshold."""
        return self.pbo <= 0.05  # Default threshold

    def passes_threshold(self, threshold: float) -> bool:
        """Check against custom threshold."""
        return self.pbo <= threshold

    def summary(self) -> str:
        """Human-readable summary."""
        status = "PASS" if self.passed else "FAIL"
        return (
            f"CPCV Result: {status}\n"
            f"  PBO: {self.pbo:.1%} (95% CI: {self.pbo_ci_95[0]:.1%}-{self.pbo_ci_95[1]:.1%})\n"
            f"  Mean IS Sharpe: {self.mean_is_sharpe:.3f}\n"
            f"  Mean OOS Sharpe: {self.mean_oos_sharpe:.3f}\n"
            f"  Sharpe Degradation: {self.mean_sharpe_degradation:.3f}\n"
            f"  Splits: {self.n_splits_completed} (overfit: {self.n_splits_overfit})"
        )


def generate_cpcv_splits(
    dates: pd.DatetimeIndex,
    config: CPCVConfig,
    seed: int = 42
) -> Generator[CPCVSplit, None, None]:
    """
    Generate CPCV splits with purging and embargo.

    Creates combinatorial train/test splits where:
    - Data is divided into S contiguous subsets
    - Each split uses (S-1) subsets for training, 1 for testing
    - Purge gap applied between train and test
    - Embargo applied after training period

    Args:
        dates: DatetimeIndex of all trading days
        config: CPCV configuration
        seed: Random seed for sampling combinations

    Yields:
        CPCVSplit objects for each combination
    """
    n_days = len(dates)
    n_subsets = config.n_subsets

    # Calculate subset boundaries
    subset_size = n_days // n_subsets
    subset_boundaries = []
    for i in range(n_subsets):
        start = i * subset_size
        end = (i + 1) * subset_size if i < n_subsets - 1 else n_days
        subset_boundaries.append((start, end))

    # Generate all combinations of (S-1) train subsets
    # For S=16, this is C(16,15) = 16 combinations per test subset
    # Total = 16 * C(15, 7 or 8) but we use simpler approach

    # Each combination: pick which subset is test, rest are train
    all_combinations = []
    for test_idx in range(n_subsets):
        train_indices = [i for i in range(n_subsets) if i != test_idx]
        all_combinations.append((train_indices, test_idx))

    # Also generate more complex combinations (multiple test subsets)
    # Using k test subsets where k varies from 1 to n_subsets//2
    for k in range(2, min(n_subsets // 2 + 1, 5)):  # Limit to save computation
        for test_combo in combinations(range(n_subsets), k):
            train_indices = [i for i in range(n_subsets) if i not in test_combo]
            if len(train_indices) >= k:  # Ensure enough training data
                all_combinations.append((train_indices, test_combo))

    # Sample if too many combinations
    total_combinations = len(all_combinations)
    if total_combinations > config.max_combinations:
        random.seed(seed)
        all_combinations = random.sample(all_combinations, config.max_combinations)
        logger.info(f"Sampled {config.max_combinations} from {total_combinations} combinations")

    # Calculate embargo in days
    embargo_days = int(n_days * config.embargo_pct)

    # Generate splits
    for split_id, (train_subset_ids, test_subset_ids) in enumerate(all_combinations):
        # Handle single test subset
        if isinstance(test_subset_ids, int):
            test_subset_ids = (test_subset_ids,)

        # Collect train and test indices
        train_idx_list = []
        test_idx_list = []

        for subset_id in train_subset_ids:
            start, end = subset_boundaries[subset_id]
            train_idx_list.extend(range(start, end))

        for subset_id in test_subset_ids:
            start, end = subset_boundaries[subset_id]
            test_idx_list.extend(range(start, end))

        train_idx = np.array(sorted(train_idx_list))
        test_idx = np.array(sorted(test_idx_list))

        # Apply purge: remove train days too close to test start
        if len(test_idx) > 0:
            test_start = test_idx[0]
            test_end = test_idx[-1]

            # Purge: remove train days within purge_days of test
            purge_mask = np.ones(len(train_idx), dtype=bool)
            for i, idx in enumerate(train_idx):
                # Remove if within purge distance of any test day
                if (test_start - config.purge_days <= idx <= test_start or
                    test_end <= idx <= test_end + config.purge_days):
                    purge_mask[i] = False
            train_idx = train_idx[purge_mask]

            # Embargo: remove embargo_days after last train day before test
            if embargo_days > 0 and len(train_idx) > 0:
                train_before_test = train_idx[train_idx < test_start]
                if len(train_before_test) > 0:
                    last_train = train_before_test[-1]
                    # Remove test days in embargo period
                    embargo_mask = test_idx > last_train + embargo_days
                    test_idx = test_idx[embargo_mask]

        # Validate minimum days
        if len(train_idx) < config.min_train_days or len(test_idx) < config.min_test_days:
            continue

        # Create split
        train_dates_range = (dates[train_idx[0]], dates[train_idx[-1]])
        test_dates_range = (dates[test_idx[0]], dates[test_idx[-1]])

        yield CPCVSplit(
            split_id=split_id,
            train_indices=train_idx,
            test_indices=test_idx,
            train_dates=train_dates_range,
            test_dates=test_dates_range,
        )


def _evaluate_split(
    split: CPCVSplit,
    returns: pd.Series,
    strategy_signals: pd.Series,
) -> CPCVSplitResult:
    """
    Evaluate a single CPCV split.

    Args:
        split: The train/test split
        returns: Full return series
        strategy_signals: Signal series (1=long, 0=flat, -1=short)

    Returns:
        CPCVSplitResult with IS and OOS metrics
    """
    # Get in-sample data
    is_returns = returns.iloc[split.train_indices]
    is_signals = strategy_signals.iloc[split.train_indices]
    is_strategy_returns = is_returns * is_signals.shift(1).fillna(0)

    # Get out-of-sample data
    oos_returns = returns.iloc[split.test_indices]
    oos_signals = strategy_signals.iloc[split.test_indices]
    oos_strategy_returns = oos_returns * oos_signals.shift(1).fillna(0)

    # Calculate metrics
    is_sharpe = _calculate_sharpe(is_strategy_returns)
    oos_sharpe = _calculate_sharpe(oos_strategy_returns)

    is_return = (1 + is_strategy_returns).prod() - 1
    oos_return = (1 + oos_strategy_returns).prod() - 1

    # Count trades (signal changes)
    is_trades = int((is_signals.diff().abs() > 0).sum())
    oos_trades = int((oos_signals.diff().abs() > 0).sum())

    return CPCVSplitResult(
        split_id=split.split_id,
        is_sharpe=is_sharpe,
        oos_sharpe=oos_sharpe,
        is_return=float(is_return),
        oos_return=float(oos_return),
        is_trades=is_trades,
        oos_trades=oos_trades,
        sharpe_degradation=is_sharpe - oos_sharpe,
    )


def _calculate_sharpe(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate annualized Sharpe ratio."""
    if len(returns) < 20 or returns.std() == 0:
        return 0.0
    excess_returns = returns - risk_free_rate / 252
    return float(np.sqrt(252) * excess_returns.mean() / excess_returns.std())


def calculate_pbo(
    split_results: List[CPCVSplitResult],
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Tuple[float, Tuple[float, float]]:
    """
    Calculate Probability of Backtest Overfitting (PBO).

    PBO is the probability that the best in-sample strategy
    will underperform the median out-of-sample.

    Uses rank correlation between IS and OOS Sharpe ratios:
    - If IS rank correlates well with OOS rank: low overfitting
    - If IS rank anti-correlates with OOS rank: high overfitting

    Args:
        split_results: Results from all CPCV splits
        n_bootstrap: Number of bootstrap samples for CI
        seed: Random seed

    Returns:
        (pbo, (ci_lower, ci_upper)) - PBO and 95% confidence interval
    """
    if len(split_results) < 10:
        logger.warning(f"Only {len(split_results)} splits, PBO may be unreliable")
        if len(split_results) < 3:
            return 0.5, (0.0, 1.0)

    # Extract IS and OOS Sharpe ratios
    is_sharpes = np.array([r.is_sharpe for r in split_results])
    oos_sharpes = np.array([r.oos_sharpe for r in split_results])

    # Calculate PBO via logit method
    # PBO = probability that selecting strategy by IS Sharpe gives OOS < median(OOS)

    # Rank the strategies by IS Sharpe
    is_ranks = np.argsort(np.argsort(-is_sharpes))  # Higher IS Sharpe = lower rank (better)
    oos_ranks = np.argsort(np.argsort(-oos_sharpes))

    # Best IS strategy (rank 0)
    best_is_idx = np.argmin(is_ranks)
    best_is_oos = oos_sharpes[best_is_idx]
    median_oos = np.median(oos_sharpes)

    # Simple PBO: fraction of splits where best IS has OOS < median
    n_underperform = sum(1 for r in split_results
                         if r.is_sharpe == max(s.is_sharpe for s in split_results)
                         and r.oos_sharpe < median_oos)

    # More robust: use Spearman correlation
    from scipy.stats import spearmanr
    rho, _ = spearmanr(is_sharpes, oos_sharpes)

    # Convert correlation to PBO-like metric
    # rho=1 means perfect alignment (PBO~0), rho=-1 means anti-alignment (PBO~1)
    pbo_from_rho = (1 - rho) / 2

    # Also calculate via probability distribution
    # For each "trial", pick best IS and check if OOS < median
    np.random.seed(seed)
    pbo_samples = []

    for _ in range(n_bootstrap):
        # Bootstrap resample
        sample_idx = np.random.choice(len(split_results), len(split_results), replace=True)
        sample_is = is_sharpes[sample_idx]
        sample_oos = oos_sharpes[sample_idx]

        # Best IS in this sample
        best_idx = np.argmax(sample_is)
        best_oos = sample_oos[best_idx]
        sample_median = np.median(sample_oos)

        # Did best IS underperform median OOS?
        pbo_samples.append(1 if best_oos < sample_median else 0)

    pbo_bootstrap = np.mean(pbo_samples)

    # Combine estimates (weighted average)
    pbo = 0.5 * pbo_from_rho + 0.5 * pbo_bootstrap

    # Confidence interval from bootstrap
    ci_lower = np.percentile(pbo_samples, 2.5)
    ci_upper = np.percentile(pbo_samples, 97.5)

    # Adjust CI to account for rho estimate uncertainty
    ci_width = (ci_upper - ci_lower) / 2
    ci_lower = max(0.0, pbo - ci_width * 1.5)
    ci_upper = min(1.0, pbo + ci_width * 1.5)

    return float(pbo), (float(ci_lower), float(ci_upper))


def run_cpcv_validation(
    returns: pd.Series,
    strategy_signals: pd.Series,
    config: CPCVConfig = None,
    progress_callback: Callable[[int, int], None] = None,
) -> CPCVResult:
    """
    Run full CPCV validation on a strategy.

    Args:
        returns: Daily returns series (e.g., SPY or strategy universe)
        strategy_signals: Strategy signal series (1=long, 0=flat, -1=short)
        config: CPCV configuration (uses defaults if None)
        progress_callback: Optional callback(completed, total) for progress

    Returns:
        CPCVResult with PBO and detailed metrics
    """
    if config is None:
        config = CPCVConfig()

    errors = config.validate()
    if errors:
        raise ValueError(f"Invalid CPCV config: {errors}")

    # Ensure aligned indices
    common_idx = returns.index.intersection(strategy_signals.index)
    returns = returns.loc[common_idx]
    strategy_signals = strategy_signals.loc[common_idx]

    if len(returns) < config.min_train_days + config.min_test_days:
        raise ValueError(f"Insufficient data: {len(returns)} days, need at least "
                        f"{config.min_train_days + config.min_test_days}")

    # Generate splits
    splits = list(generate_cpcv_splits(returns.index, config))

    if len(splits) < 10:
        raise ValueError(f"Only {len(splits)} valid splits generated, need at least 10")

    logger.info(f"Running CPCV with {len(splits)} splits")

    # Evaluate splits
    split_results = []

    if config.n_workers > 1:
        # Parallel evaluation
        with ProcessPoolExecutor(max_workers=config.n_workers) as executor:
            futures = {
                executor.submit(_evaluate_split, split, returns, strategy_signals): split
                for split in splits
            }

            completed = 0
            for future in as_completed(futures):
                try:
                    result = future.result()
                    split_results.append(result)
                except Exception as e:
                    logger.warning(f"Split evaluation failed: {e}")

                completed += 1
                if progress_callback:
                    progress_callback(completed, len(splits))
    else:
        # Sequential evaluation
        for i, split in enumerate(splits):
            try:
                result = _evaluate_split(split, returns, strategy_signals)
                split_results.append(result)
            except Exception as e:
                logger.warning(f"Split {split.split_id} failed: {e}")

            if progress_callback:
                progress_callback(i + 1, len(splits))

    if len(split_results) < 10:
        raise ValueError(f"Only {len(split_results)} splits succeeded, need at least 10")

    # Calculate PBO
    pbo, pbo_ci = calculate_pbo(split_results)

    # Aggregate metrics
    is_sharpes = [r.is_sharpe for r in split_results]
    oos_sharpes = [r.oos_sharpe for r in split_results]
    degradations = [r.sharpe_degradation for r in split_results]

    n_overfit = sum(1 for r in split_results if r.is_overfit)

    return CPCVResult(
        pbo=pbo,
        pbo_ci_95=pbo_ci,
        mean_is_sharpe=float(np.mean(is_sharpes)),
        mean_oos_sharpe=float(np.mean(oos_sharpes)),
        mean_sharpe_degradation=float(np.mean(degradations)),
        std_oos_sharpe=float(np.std(oos_sharpes)),
        n_splits_completed=len(split_results),
        n_splits_overfit=n_overfit,
        split_results=split_results,
    )


def validate_strategy_with_cpcv(
    strategy_genome: Any,
    price_data: pd.DataFrame,
    config: CPCVConfig = None,
    progress_callback: Callable[[int, int], None] = None,
) -> CPCVResult:
    """
    Convenience function to validate a GP strategy genome with CPCV.

    Args:
        strategy_genome: StrategyGenome object with evaluate() method
        price_data: OHLCV DataFrame with strategy universe
        config: CPCV configuration
        progress_callback: Optional progress callback

    Returns:
        CPCVResult
    """
    # Generate signals from genome
    try:
        signals = strategy_genome.generate_signals(price_data)
        if isinstance(signals, pd.DataFrame):
            # Aggregate to single signal series
            signals = signals.mean(axis=1).apply(lambda x: 1 if x > 0.5 else (-1 if x < -0.5 else 0))
    except Exception as e:
        logger.error(f"Failed to generate signals: {e}")
        # Return failed result
        return CPCVResult(
            pbo=1.0,
            pbo_ci_95=(1.0, 1.0),
            mean_is_sharpe=0.0,
            mean_oos_sharpe=0.0,
            mean_sharpe_degradation=0.0,
            std_oos_sharpe=0.0,
            n_splits_completed=0,
            n_splits_overfit=0,
        )

    # Use SPY or first column as benchmark returns
    if 'SPY' in price_data.columns:
        returns = price_data['SPY'].pct_change().dropna()
    else:
        returns = price_data.iloc[:, 0].pct_change().dropna()

    return run_cpcv_validation(
        returns=returns,
        strategy_signals=signals,
        config=config,
        progress_callback=progress_callback,
    )
