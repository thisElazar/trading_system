"""
HMM Regime Detector (GP-010)
============================
Hidden Markov Model for probabilistic regime detection and transition forecasting.

Key features:
- Learns regime transitions from market data instead of hardcoded matrix
- Provides probabilistic regime assignment (not just classification)
- Forecasts regime transitions N days ahead
- Auto-maps learned states to interpretable regime names by volatility

Based on: "Regime Switches in Interest Rates" by Hamilton (1989)
"""

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Deque
from collections import deque

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional hmmlearn import with fallback
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.warning("hmmlearn not available - HMM regime detection disabled")


@dataclass
class HMMConfig:
    """Configuration for HMM regime detector."""

    n_states: int = 3                           # Number of hidden states (regimes)
    state_names: List[str] = field(default_factory=lambda: ["bull", "transition", "crisis"])
    covariance_type: str = "full"               # Covariance type: full, diag, spherical
    n_iter: int = 100                           # Max EM iterations
    lookback_days: int = 756                    # Training window (3 years)
    min_observations: int = 252                 # Minimum days for training
    convergence_tol: float = 1e-4               # EM convergence tolerance
    random_state: int = 42                      # For reproducibility
    model_path: Optional[Path] = None           # Path to save/load model

    # GP-015: Regime change confirmation lag
    regime_confirmation_days: int = 2           # Days of stable regime before confirming change
    
    def validate(self) -> List[str]:
        """Validate configuration."""
        errors = []
        if self.n_states < 2:
            errors.append("n_states must be at least 2")
        if self.n_states != len(self.state_names):
            errors.append(f"n_states ({self.n_states}) must match len(state_names) ({len(self.state_names)})")
        if self.covariance_type not in ["full", "diag", "spherical", "tied"]:
            errors.append("covariance_type must be: full, diag, spherical, or tied")
        if self.lookback_days < self.min_observations:
            errors.append("lookback_days must be >= min_observations")
        return errors


class HMMRegimeDetector:
    """
    HMM-based regime detector for market state identification.
    
    Learns regime transitions from historical returns and provides:
    - Current regime detection with probability
    - Transition matrix (learned, not hardcoded)
    - N-day regime forecasts
    
    Usage:
        detector = HMMRegimeDetector()
        detector.fit(spy_returns)
        
        current_regime, prob = detector.detect_regime(today_return)
        transition_probs = detector.get_transition_matrix()
        forecast = detector.predict_next_regime("bull", horizon_days=5)
    """
    
    def __init__(self, config: HMMConfig = None):
        """
        Initialize HMM regime detector.
        
        Args:
            config: HMM configuration (uses defaults if None)
        """
        self.config = config or HMMConfig()
        
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Invalid HMM config: {errors}")
        
        self.model: Optional[GaussianHMM] = None
        self.is_trained: bool = False
        self.state_mapping: Dict[int, str] = {}  # Learned state -> regime name
        self.last_fit_date: Optional[pd.Timestamp] = None
        
        # Training statistics
        self.state_means: Dict[str, float] = {}
        self.state_stds: Dict[str, float] = {}
        self.converged: bool = False

        # GP-015: Regime confirmation tracking
        self._regime_confirmation_buffer: deque = deque(maxlen=max(10, self.config.regime_confirmation_days + 1))
        self._confirmed_regime: Optional[str] = None
        self._pending_regime: Optional[str] = None
        self._pending_regime_count: int = 0
        
    def fit(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Fit HMM to historical returns.
        
        Args:
            returns: Daily returns series (log or simple)
            
        Returns:
            Dict with training results:
            - state_means: Mean return per state
            - state_stds: Volatility per state
            - state_mapping: Learned state -> regime name
            - log_likelihood: Model log likelihood
            - converged: Whether EM converged
        """
        if not HMM_AVAILABLE:
            raise ImportError("hmmlearn not installed. Run: pip install hmmlearn")
        
        # Validate input
        if len(returns) < self.config.min_observations:
            raise ValueError(f"Need at least {self.config.min_observations} observations, got {len(returns)}")
        
        # Use lookback window
        if len(returns) > self.config.lookback_days:
            returns = returns.iloc[-self.config.lookback_days:]
        
        # Prepare data (2D array for hmmlearn)
        X = returns.dropna().values.reshape(-1, 1)

        if len(X) < self.config.min_observations:
            raise ValueError(f"Insufficient non-NaN data: {len(X)} < {self.config.min_observations}")

        # Check for inf values and replace with clipped values
        X = np.clip(X, -0.5, 0.5)  # Clip extreme returns to +/- 50%

        # Try fitting with configured covariance type, fall back to 'diag' if numerical issues
        covariance_types = [self.config.covariance_type, 'diag', 'spherical']

        for cov_type in covariance_types:
            try:
                # Initialize HMM
                self.model = GaussianHMM(
                    n_components=self.config.n_states,
                    covariance_type=cov_type,
                    n_iter=self.config.n_iter,
                    tol=self.config.convergence_tol,
                    random_state=self.config.random_state,
                    init_params='stmc',  # Initialize start, trans, means, covars
                    params='stmc'
                )

                self.model.fit(X)
                self.converged = self.model.monitor_.converged

                if not self.converged:
                    logger.warning(f"HMM did not converge with {cov_type} - trying next")
                    if cov_type != covariance_types[-1]:
                        continue

                # Test that model works (score can fail with NaN covariance)
                _ = self.model.score(X)

                if cov_type != self.config.covariance_type:
                    logger.info(f"HMM fitted with fallback covariance type: {cov_type}")
                break

            except Exception as e:
                logger.warning(f"HMM fitting with {cov_type} failed: {e}")
                if cov_type == covariance_types[-1]:
                    raise RuntimeError(f"HMM fitting failed with all covariance types: {e}")

        # Auto-map states to regime names by volatility
        self._auto_map_states()

        self.is_trained = True
        self.last_fit_date = returns.index[-1] if hasattr(returns.index, '__len__') else None

        # Compute log likelihood (safe after successful score above)
        try:
            log_likelihood = self.model.score(X)
        except Exception:
            log_likelihood = float('-inf')
        
        results = {
            'state_means': self.state_means.copy(),
            'state_stds': self.state_stds.copy(),
            'state_mapping': self.state_mapping.copy(),
            'log_likelihood': float(log_likelihood),
            'converged': self.converged,
            'n_observations': len(X),
            'fit_date': str(self.last_fit_date) if self.last_fit_date else None
        }
        
        logger.info(f"HMM fitted: {results['n_observations']} obs, converged={results['converged']}")
        for state_name, mean in self.state_means.items():
            std = self.state_stds[state_name]
            logger.info(f"  {state_name}: mean={mean:.4%}, std={std:.4%}")
        
        return results
    
    def _auto_map_states(self):
        """
        Map learned HMM states to interpretable regime names.

        Strategy: Sort states by volatility (std of emissions)
        - Lowest volatility -> bull (stable uptrend)
        - Medium volatility -> transition (uncertain)
        - Highest volatility -> crisis (high fear)
        """
        if self.model is None:
            raise ValueError("Model not fitted")

        # Get means and covariances
        means = self.model.means_.flatten()

        # Get standard deviations (handle different covariance types robustly)
        try:
            covars = self.model.covars_
            n_states = self.config.n_states

            # Extract variance for each state (first feature)
            stds = []
            for i in range(n_states):
                try:
                    # Try different indexing patterns based on shape
                    if covars.ndim == 3:
                        # Shape (n_states, n_features, n_features) - full or diag with weird shape
                        var = covars[i][0, 0]
                    elif covars.ndim == 2:
                        # Shape (n_states, n_features) - typical diag
                        var = covars[i, 0]
                    elif covars.ndim == 1:
                        # Shape (n_states,) - spherical
                        var = covars[i]
                    else:
                        var = float(covars.flatten()[i])
                    stds.append(np.sqrt(max(1e-10, var)))
                except Exception:
                    stds.append(0.01)  # Default std

            stds = np.array(stds)
        except Exception as e:
            logger.warning(f"Error extracting stds: {e}, using means-based ordering")
            # Fallback: use means to order states (more negative = crisis)
            stds = -means  # Assume lower means = higher volatility regime
        
        # Ensure stds is array-like
        stds = np.atleast_1d(stds)

        # Sort states by volatility (low to high)
        state_order = np.argsort(stds)

        # Map to regime names
        self.state_mapping = {}
        self.state_means = {}
        self.state_stds = {}

        for i, state_idx in enumerate(state_order):
            if i >= len(self.config.state_names):
                break
            regime_name = self.config.state_names[i]
            self.state_mapping[int(state_idx)] = regime_name
            self.state_means[regime_name] = float(means[state_idx])
            self.state_stds[regime_name] = float(stds[state_idx])
    
    def detect_regime(
        self,
        returns: pd.Series,
        return_probabilities: bool = False
    ) -> Tuple[str, float]:
        """
        Detect current regime from recent returns.
        
        Args:
            returns: Recent returns series (uses last value for prediction)
            return_probabilities: If True, return all state probabilities
            
        Returns:
            Tuple of (regime_name, probability)
            If return_probabilities=True, returns (regime_name, prob, all_probs_dict)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Prepare data
        if isinstance(returns, pd.Series):
            X = returns.dropna().values.reshape(-1, 1)
        else:
            X = np.array(returns).reshape(-1, 1)
        
        if len(X) == 0:
            raise ValueError("No valid return data provided")
        
        # Get state probabilities for the sequence
        try:
            posteriors = self.model.predict_proba(X)
            last_posteriors = posteriors[-1]  # Probabilities for current state
        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            # Fallback to most likely state
            hidden_states = self.model.predict(X)
            last_state = hidden_states[-1]
            last_posteriors = np.zeros(self.config.n_states)
            last_posteriors[last_state] = 1.0
        
        # Find most likely regime
        best_state = int(np.argmax(last_posteriors))
        best_regime = self.state_mapping.get(best_state, f"state_{best_state}")
        best_prob = float(last_posteriors[best_state])
        
        if return_probabilities:
            all_probs = {
                self.state_mapping.get(i, f"state_{i}"): float(last_posteriors[i])
                for i in range(self.config.n_states)
            }
            return best_regime, best_prob, all_probs

        # GP-015: Update regime confirmation tracking
        self._update_regime_confirmation(best_regime)

        return best_regime, best_prob

    def _update_regime_confirmation(self, raw_regime: str):
        """
        GP-015: Update regime confirmation state machine.

        Requires stable regime classification for N consecutive days before
        confirming a regime change. Prevents whipsaws from noisy detection.
        """
        self._regime_confirmation_buffer.append(raw_regime)

        # Initialize confirmed regime if not set
        if self._confirmed_regime is None:
            self._confirmed_regime = raw_regime
            self._pending_regime = None
            self._pending_regime_count = 0
            return

        # If raw matches confirmed, reset pending
        if raw_regime == self._confirmed_regime:
            self._pending_regime = None
            self._pending_regime_count = 0
            return

        # If raw matches pending, increment count
        if raw_regime == self._pending_regime:
            self._pending_regime_count += 1
        else:
            # New pending regime
            self._pending_regime = raw_regime
            self._pending_regime_count = 1

        # Check if pending should be confirmed
        if self._pending_regime_count >= self.config.regime_confirmation_days:
            old_regime = self._confirmed_regime
            self._confirmed_regime = self._pending_regime
            self._pending_regime = None
            self._pending_regime_count = 0
            logger.info(
                f"HMM regime change confirmed: {old_regime} -> {self._confirmed_regime} "
                f"(after {self.config.regime_confirmation_days} consecutive days)"
            )

    def get_confirmed_regime(self, returns: pd.Series = None) -> Tuple[str, float, bool]:
        """
        GP-015: Get the confirmed regime (requires stable classification).

        Args:
            returns: Optional recent returns for fresh detection

        Returns:
            Tuple of (confirmed_regime, confidence, is_transitioning)
            - confirmed_regime: The stable, confirmed regime
            - confidence: Confidence level (lower if transitioning)
            - is_transitioning: True if a regime change is pending
        """
        # Optionally update with fresh detection
        if returns is not None and self.is_trained:
            try:
                self.detect_regime(returns)
            except Exception as e:
                logger.warning(f"Failed to update regime detection: {e}")

        # Get latest probability if available
        confidence = 0.7  # Default confidence
        if self._regime_confirmation_buffer and self.is_trained:
            # Use stored confidence from last detection
            confidence = 0.8

        # Check if we're in a transition state
        is_transitioning = (
            self._pending_regime is not None and
            self._pending_regime_count > 0
        )

        # Reduce confidence if transitioning
        if is_transitioning:
            progress = self._pending_regime_count / self.config.regime_confirmation_days
            confidence = confidence * (1 - 0.3 * progress)  # Up to 30% reduction

        return self._confirmed_regime or "transition", confidence, is_transitioning

    def get_transition_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Get learned transition probabilities.
        
        Returns:
            Nested dict: P(next_regime | current_regime)
            Example: {"bull": {"bull": 0.95, "transition": 0.04, "crisis": 0.01}}
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        trans_matrix = self.model.transmat_
        
        result = {}
        for from_state in range(self.config.n_states):
            from_regime = self.state_mapping.get(from_state, f"state_{from_state}")
            result[from_regime] = {}
            
            for to_state in range(self.config.n_states):
                to_regime = self.state_mapping.get(to_state, f"state_{to_state}")
                result[from_regime][to_regime] = float(trans_matrix[from_state, to_state])
        
        return result
    
    def get_regime_persistence(self) -> Dict[str, float]:
        """
        Get expected persistence (days) for each regime.
        
        Returns:
            Dict mapping regime name to expected days before transition
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        result = {}
        trans_matrix = self.model.transmat_
        
        for state in range(self.config.n_states):
            regime = self.state_mapping.get(state, f"state_{state}")
            # Expected duration = 1 / (1 - P(stay))
            p_stay = trans_matrix[state, state]
            if p_stay < 1.0:
                expected_duration = 1.0 / (1.0 - p_stay)
            else:
                expected_duration = float('inf')
            result[regime] = float(expected_duration)
        
        return result
    
    def predict_next_regime(
        self,
        current_regime: str,
        horizon_days: int = 5
    ) -> Dict[str, float]:
        """
        Forecast regime probabilities N days ahead.
        
        Uses transition matrix power: P^n gives n-step transition probs.
        
        Args:
            current_regime: Starting regime name
            horizon_days: Days ahead to forecast
            
        Returns:
            Dict of regime probabilities after horizon_days
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Find current state index
        current_state = None
        for state, regime in self.state_mapping.items():
            if regime == current_regime:
                current_state = state
                break
        
        if current_state is None:
            raise ValueError(f"Unknown regime: {current_regime}")
        
        # Compute n-step transition via matrix power
        trans_matrix = self.model.transmat_
        n_step_matrix = np.linalg.matrix_power(trans_matrix, horizon_days)
        
        # Get probabilities from current state
        probs = n_step_matrix[current_state, :]
        
        result = {}
        for state in range(self.config.n_states):
            regime = self.state_mapping.get(state, f"state_{state}")
            result[regime] = float(probs[state])
        
        return result
    
    def save(self, path: Path = None):
        """Save trained model to disk."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        save_path = path or self.config.model_path
        if save_path is None:
            raise ValueError("No save path specified")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'model': self.model,
            'config': self.config,
            'state_mapping': self.state_mapping,
            'state_means': self.state_means,
            'state_stds': self.state_stds,
            'last_fit_date': self.last_fit_date,
            'converged': self.converged
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"HMM model saved to {save_path}")
    
    def load(self, path: Path = None):
        """Load trained model from disk."""
        load_path = path or self.config.model_path
        if load_path is None:
            raise ValueError("No load path specified")
        
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        with open(load_path, 'rb') as f:
            state = pickle.load(f)
        
        self.model = state['model']
        self.config = state['config']
        self.state_mapping = state['state_mapping']
        self.state_means = state['state_means']
        self.state_stds = state['state_stds']
        self.last_fit_date = state['last_fit_date']
        self.converged = state['converged']
        self.is_trained = True
        
        logger.info(f"HMM model loaded from {load_path}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_detector_from_spy(
    lookback_years: int = 3,
    n_states: int = 3
) -> HMMRegimeDetector:
    """
    Create and train HMM detector using SPY returns.
    
    Args:
        lookback_years: Years of history to use
        n_states: Number of hidden states
        
    Returns:
        Trained HMMRegimeDetector
    """
    import yfinance as yf
    
    # Download SPY data
    lookback_days = lookback_years * 252
    spy = yf.download('SPY', period=f'{lookback_years + 1}y', progress=False)
    
    if len(spy) < lookback_days // 2:
        raise ValueError(f"Insufficient SPY data: {len(spy)} days")
    
    # Calculate returns
    returns = spy['Close'].pct_change().dropna()
    
    # Configure and fit
    config = HMMConfig(
        n_states=n_states,
        lookback_days=lookback_days
    )
    
    detector = HMMRegimeDetector(config)
    detector.fit(returns)
    
    return detector


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate HMM regime detector."""
    print("=" * 60)
    print("HMM Regime Detector Demo")
    print("=" * 60)
    
    if not HMM_AVAILABLE:
        print("hmmlearn not installed. Run: pip install hmmlearn")
        return
    
    # Generate synthetic market data
    np.random.seed(42)
    n_days = 1000
    
    # Simulate regime-switching returns
    regimes = []
    returns = []
    current_regime = 0  # Start in bull
    
    regime_params = [
        (0.0005, 0.01),   # Bull: slight positive drift, low vol
        (0.0, 0.02),      # Transition: no drift, medium vol  
        (-0.001, 0.035),  # Crisis: negative drift, high vol
    ]
    
    transition_probs = [
        [0.98, 0.015, 0.005],  # From bull
        [0.03, 0.94, 0.03],     # From transition
        [0.01, 0.05, 0.94],     # From crisis
    ]
    
    for _ in range(n_days):
        regimes.append(current_regime)
        mean, std = regime_params[current_regime]
        returns.append(np.random.normal(mean, std))
        
        # Transition
        current_regime = np.random.choice(3, p=transition_probs[current_regime])
    
    returns = pd.Series(returns, index=pd.date_range('2020-01-01', periods=n_days, freq='B'))
    
    print(f"\nGenerated {n_days} days of synthetic returns")
    print(f"True regime counts: Bull={regimes.count(0)}, Trans={regimes.count(1)}, Crisis={regimes.count(2)}")
    
    # Fit HMM
    print("\n--- Fitting HMM ---")
    config = HMMConfig(n_states=3, lookback_days=n_days)
    detector = HMMRegimeDetector(config)
    
    results = detector.fit(returns)
    print(f"\nFit results:")
    print(f"  Converged: {results['converged']}")
    print(f"  Log likelihood: {results['log_likelihood']:.2f}")
    print(f"\nLearned state characteristics:")
    for regime, mean in results['state_means'].items():
        std = results['state_stds'][regime]
        print(f"  {regime}: mean={mean:.4%}, std={std:.4%}")
    
    # Get transition matrix
    print("\n--- Transition Matrix ---")
    trans = detector.get_transition_matrix()
    for from_regime, probs in trans.items():
        prob_str = ", ".join(f"{to}={p:.1%}" for to, p in probs.items())
        print(f"  {from_regime} -> {prob_str}")
    
    # Regime persistence
    print("\n--- Expected Regime Duration ---")
    persistence = detector.get_regime_persistence()
    for regime, days in persistence.items():
        print(f"  {regime}: {days:.1f} days")
    
    # Current detection
    print("\n--- Current Regime Detection ---")
    regime, prob = detector.detect_regime(returns.iloc[-20:])
    print(f"  Current regime: {regime} (probability: {prob:.1%})")
    
    # Forecast
    print("\n--- 5-Day Forecast ---")
    forecast = detector.predict_next_regime(regime, horizon_days=5)
    for regime_name, prob in forecast.items():
        print(f"  P({regime_name}): {prob:.1%}")
    
    print("\n" + "=" * 60)
    print("Demo Complete")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
