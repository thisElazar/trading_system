#!/usr/bin/env python3
"""
ML-Based Regime Detection
=========================
Learn regime patterns from market data and strategy performance.

Features:
- Feature extraction from market data (VIX, breadth, momentum, etc.)
- Train regime classifier on historical performance
- Predict regime transitions before they fully develop
- Auto-tune strategy weights based on predicted regime
- Online learning from new data

Usage:
    detector = MLRegimeDetector()

    # Train on historical data
    detector.train(market_data, strategy_returns)

    # Predict current regime
    regime, confidence = detector.predict_regime(current_features)

    # Get optimal weights for predicted regime
    weights = detector.get_regime_weights(regime)
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
import pickle
import json

import pandas as pd
import numpy as np
from collections import deque

# ML imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, TimeSeriesSplit
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DIRS

logger = logging.getLogger(__name__)


# ============================================================================
# PICKLE HELPERS
# ============================================================================

class _ModuleRemappingUnpickler(pickle.Unpickler):
    """
    Custom unpickler that remaps classes saved from __main__ to this module.

    This fixes the common issue where a model is saved when running a script
    directly (classes are in __main__) but loaded when imported as a module.
    """

    # Classes that might have been saved from __main__
    _REMAP_CLASSES = {
        'RegimePerformance',
        'RegimePrediction',
        'FeatureSet',
        'MLRegimeConfig',
        'RegimeType',
    }

    def find_class(self, module: str, name: str):
        # If the class was saved from __main__, remap to this module
        if module == '__main__' and name in self._REMAP_CLASSES:
            module = 'research.ml_regime_detector'
        return super().find_class(module, name)


# ============================================================================
# DATA CLASSES
# ============================================================================

class RegimeType:
    """Regime type constants."""
    BULL = "bull"
    BEAR = "bear"
    HIGH_VOL = "high_volatility"
    LOW_VOL = "low_volatility"
    TRANSITION = "transition"
    CRISIS = "crisis"


@dataclass
class RegimePrediction:
    """Prediction result from the ML model."""
    regime: str
    confidence: float
    probabilities: Dict[str, float]
    features_used: Dict[str, float]
    transition_warning: bool
    predicted_duration_days: int
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class FeatureSet:
    """Feature set for regime detection."""
    # Volatility features
    vix_level: float = 0.0
    vix_percentile: float = 0.0  # Where current VIX is in historical distribution
    vix_change_5d: float = 0.0
    vix_change_20d: float = 0.0
    realized_vol_20d: float = 0.0

    # Trend features
    sp500_return_5d: float = 0.0
    sp500_return_20d: float = 0.0
    sp500_return_60d: float = 0.0
    sp500_vs_sma50: float = 0.0
    sp500_vs_sma200: float = 0.0

    # Breadth features
    pct_above_sma50: float = 0.0
    pct_above_sma200: float = 0.0
    advance_decline_ratio: float = 0.0
    new_highs_vs_lows: float = 0.0

    # Momentum features
    rsi_14d: float = 0.0
    macd_histogram: float = 0.0
    momentum_breadth: float = 0.0

    # Credit/Risk features
    credit_spread: float = 0.0  # High yield spread
    ted_spread: float = 0.0

    # Sector rotation
    sector_dispersion: float = 0.0
    defensive_vs_cyclical: float = 0.0

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML."""
        return np.array([
            self.vix_level, self.vix_percentile, self.vix_change_5d, self.vix_change_20d,
            self.realized_vol_20d, self.sp500_return_5d, self.sp500_return_20d,
            self.sp500_return_60d, self.sp500_vs_sma50, self.sp500_vs_sma200,
            self.pct_above_sma50, self.pct_above_sma200, self.advance_decline_ratio,
            self.new_highs_vs_lows, self.rsi_14d, self.macd_histogram,
            self.momentum_breadth, self.credit_spread, self.ted_spread,
            self.sector_dispersion, self.defensive_vs_cyclical
        ])

    @staticmethod
    def feature_names() -> List[str]:
        """Get feature names."""
        return [
            'vix_level', 'vix_percentile', 'vix_change_5d', 'vix_change_20d',
            'realized_vol_20d', 'sp500_return_5d', 'sp500_return_20d',
            'sp500_return_60d', 'sp500_vs_sma50', 'sp500_vs_sma200',
            'pct_above_sma50', 'pct_above_sma200', 'advance_decline_ratio',
            'new_highs_vs_lows', 'rsi_14d', 'macd_histogram',
            'momentum_breadth', 'credit_spread', 'ted_spread',
            'sector_dispersion', 'defensive_vs_cyclical'
        ]


@dataclass
class RegimePerformance:
    """Strategy performance in a regime."""
    regime: str
    strategy: str
    avg_return: float
    sharpe_ratio: float
    win_rate: float
    max_drawdown: float
    sample_count: int


@dataclass
class MLRegimeConfig:
    """Configuration for ML regime detection."""
    # Model settings
    model_type: str = "random_forest"  # random_forest, gradient_boosting
    n_estimators: int = 100
    max_depth: int = 10

    # Training settings
    min_samples_per_regime: int = 30
    test_size: float = 0.2
    cv_folds: int = 5

    # Prediction settings
    confidence_threshold: float = 0.6
    transition_lookback_days: int = 5
    prediction_horizon_days: int = 5

    # Online learning
    online_learning_rate: float = 0.1
    retrain_frequency_days: int = 7

    # Model persistence
    model_path: Path = field(default_factory=lambda: DIRS.get('models', Path('./models')) / 'regime_model.pkl')


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

class FeatureExtractor:
    """Extract features from market data for regime detection."""

    def __init__(self):
        self.vix_history: deque = deque(maxlen=252)  # 1 year of VIX
        self.sp500_history: deque = deque(maxlen=252)

    def extract_features(
        self,
        vix: float,
        sp500_price: float,
        market_data: pd.DataFrame = None,
    ) -> FeatureSet:
        """
        Extract features from current market state.

        Args:
            vix: Current VIX level
            sp500_price: Current S&P 500 price
            market_data: DataFrame with historical OHLCV data

        Returns:
            FeatureSet with all features populated
        """
        features = FeatureSet()

        # Store current values
        self.vix_history.append(vix)
        self.sp500_history.append(sp500_price)

        # VIX features
        features.vix_level = vix
        if len(self.vix_history) > 20:
            vix_array = np.array(list(self.vix_history))
            features.vix_percentile = (vix_array < vix).mean()
            features.vix_change_5d = (vix / vix_array[-5] - 1) if len(vix_array) >= 5 else 0
            features.vix_change_20d = (vix / vix_array[-20] - 1) if len(vix_array) >= 20 else 0

        # S&P 500 features
        if len(self.sp500_history) > 60:
            prices = np.array(list(self.sp500_history))
            returns = np.diff(np.log(prices))

            features.sp500_return_5d = (prices[-1] / prices[-5] - 1) if len(prices) >= 5 else 0
            features.sp500_return_20d = (prices[-1] / prices[-20] - 1) if len(prices) >= 20 else 0
            features.sp500_return_60d = (prices[-1] / prices[-60] - 1) if len(prices) >= 60 else 0

            # SMA comparisons
            sma50 = prices[-50:].mean() if len(prices) >= 50 else prices.mean()
            sma200 = prices.mean()
            features.sp500_vs_sma50 = (prices[-1] / sma50 - 1)
            features.sp500_vs_sma200 = (prices[-1] / sma200 - 1)

            # Realized volatility
            if len(returns) >= 20:
                features.realized_vol_20d = returns[-20:].std() * np.sqrt(252)

            # RSI
            features.rsi_14d = self._calculate_rsi(prices, 14)

        # Market data features (if provided)
        if market_data is not None and len(market_data) > 0:
            features = self._extract_from_dataframe(features, market_data)

        return features

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices)[-period:]
        gains = np.maximum(deltas, 0)
        losses = np.abs(np.minimum(deltas, 0))

        avg_gain = gains.mean()
        avg_loss = losses.mean()

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _extract_from_dataframe(self, features: FeatureSet, df: pd.DataFrame) -> FeatureSet:
        """Extract additional features from market DataFrame."""
        if 'close' not in df.columns:
            return features

        # Calculate breadth if we have multiple symbols
        if len(df.columns) > 5:  # Assuming multi-symbol data
            # Percent above SMAs
            for col in df.columns:
                if 'close' in col.lower():
                    prices = df[col].dropna()
                    if len(prices) >= 50:
                        sma50 = prices.rolling(50).mean()
                        features.pct_above_sma50 = (prices.iloc[-1] > sma50.iloc[-1]).mean()
                    if len(prices) >= 200:
                        sma200 = prices.rolling(200).mean()
                        features.pct_above_sma200 = (prices.iloc[-1] > sma200.iloc[-1]).mean()
                    break

        return features

    def extract_features_batch(
        self,
        vix_series: pd.Series,
        sp500_series: pd.Series,
    ) -> pd.DataFrame:
        """
        Extract features for a batch of historical data.

        Returns DataFrame with features for each date.
        """
        features_list = []
        dates = []

        for date in vix_series.index:
            if date not in sp500_series.index:
                continue

            vix = vix_series.loc[date]
            sp500 = sp500_series.loc[date]

            features = self.extract_features(vix, sp500)
            features_list.append(features.to_array())
            dates.append(date)

        return pd.DataFrame(
            features_list,
            index=dates,
            columns=FeatureSet.feature_names()
        )


# ============================================================================
# REGIME LABELER
# ============================================================================

class RegimeLabeler:
    """Label historical periods with regime types based on conditions."""

    @staticmethod
    def label_regimes(
        vix_series: pd.Series,
        sp500_returns: pd.Series,
        lookback: int = 20,
    ) -> pd.Series:
        """
        Label historical data with regime types.

        Args:
            vix_series: VIX time series
            sp500_returns: S&P 500 daily returns
            lookback: Period for calculating conditions

        Returns:
            Series with regime labels
        """
        labels = pd.Series(index=vix_series.index, dtype=str)

        # Calculate rolling metrics
        rolling_return = sp500_returns.rolling(lookback).sum()
        rolling_vol = sp500_returns.rolling(lookback).std() * np.sqrt(252)

        for date in vix_series.index:
            if date not in rolling_return.index:
                labels[date] = RegimeType.TRANSITION
                continue

            vix = vix_series.loc[date]
            ret = rolling_return.loc[date]
            vol = rolling_vol.loc[date]

            # Crisis: VIX > 35 or extreme negative return
            if vix > 35 or (ret < -0.15):
                labels[date] = RegimeType.CRISIS
            # High volatility: VIX > 25
            elif vix > 25:
                labels[date] = RegimeType.HIGH_VOL
            # Bull: Positive return and VIX < 20
            elif ret > 0.03 and vix < 20:
                labels[date] = RegimeType.BULL
            # Bear: Negative return
            elif ret < -0.05:
                labels[date] = RegimeType.BEAR
            # Low volatility: VIX < 15
            elif vix < 15:
                labels[date] = RegimeType.LOW_VOL
            else:
                labels[date] = RegimeType.TRANSITION

        return labels


# ============================================================================
# ML REGIME DETECTOR
# ============================================================================

class MLRegimeDetector:
    """
    Machine learning based regime detection.

    Trains on historical data to predict current market regime
    and optimal strategy allocations.
    """

    def __init__(self, config: MLRegimeConfig = None):
        """
        Initialize ML regime detector.

        Args:
            config: Model configuration
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for ML regime detection")

        self.config = config or MLRegimeConfig()

        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.feature_extractor = FeatureExtractor()
        self.labeler = RegimeLabeler()

        # Performance tracking per regime
        self.regime_performance: Dict[str, Dict[str, RegimePerformance]] = {}

        # Training state
        self.is_trained = False
        self.last_training: Optional[datetime] = None
        self.training_accuracy: float = 0.0
        self.feature_importance: Dict[str, float] = {}

        # Prediction history
        self.prediction_history: deque = deque(maxlen=100)

        # Try to load existing model
        self._load_model()

        logger.info(f"MLRegimeDetector initialized (trained={self.is_trained})")

    def _create_model(self):
        """Create the ML model based on config."""
        if self.config.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=42,
                n_jobs=-1,
            )
        elif self.config.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

    # ========================================================================
    # TRAINING
    # ========================================================================

    def train(
        self,
        vix_series: pd.Series,
        sp500_series: pd.Series,
        strategy_returns: Dict[str, pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Train the regime detection model.

        Args:
            vix_series: VIX time series
            sp500_series: S&P 500 price series
            strategy_returns: Optional dict of strategy return series

        Returns:
            Training results dict
        """
        logger.info("Training regime detection model...")

        # Calculate S&P 500 returns
        sp500_returns = sp500_series.pct_change().dropna()

        # Extract features
        features_df = self.feature_extractor.extract_features_batch(vix_series, sp500_series)

        # Label regimes
        regime_labels = self.labeler.label_regimes(vix_series, sp500_returns)

        # Align data
        common_idx = features_df.index.intersection(regime_labels.index)
        X = features_df.loc[common_idx].dropna()
        y = regime_labels.loc[X.index]

        # Check minimum samples
        regime_counts = y.value_counts()
        logger.info(f"Regime distribution:\n{regime_counts}")

        if regime_counts.min() < self.config.min_samples_per_regime:
            logger.warning(f"Insufficient samples for some regimes")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Create and train model
        self.model = self._create_model()

        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=tscv, scoring='accuracy')
        logger.info(f"CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

        # Final training
        self.model.fit(X_scaled, y)
        self.training_accuracy = self.model.score(X_scaled, y)

        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(
                FeatureSet.feature_names(),
                self.model.feature_importances_
            ))

        # Track strategy performance by regime if provided
        if strategy_returns:
            self._calculate_regime_performance(regime_labels, strategy_returns)

        # Save model
        self.is_trained = True
        self.last_training = datetime.now()
        self._save_model()

        results = {
            'accuracy': self.training_accuracy,
            'cv_accuracy': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'samples': len(X),
            'regime_counts': regime_counts.to_dict(),
            'feature_importance': dict(sorted(
                self.feature_importance.items(),
                key=lambda x: -x[1]
            )[:10])
        }

        logger.info(f"Training complete. Accuracy: {self.training_accuracy:.1%}")
        return results

    def _calculate_regime_performance(
        self,
        regime_labels: pd.Series,
        strategy_returns: Dict[str, pd.Series],
    ):
        """Calculate strategy performance per regime."""
        for regime in regime_labels.unique():
            self.regime_performance[regime] = {}
            regime_mask = regime_labels == regime

            for strategy, returns in strategy_returns.items():
                # Align returns with regime
                common_idx = returns.index.intersection(regime_labels[regime_mask].index)
                regime_returns = returns.loc[common_idx]

                if len(regime_returns) < 10:
                    continue

                # Calculate metrics
                avg_return = regime_returns.mean() * 252  # Annualized
                sharpe = regime_returns.mean() / regime_returns.std() * np.sqrt(252) if regime_returns.std() > 0 else 0
                win_rate = (regime_returns > 0).mean()
                max_dd = (regime_returns.cumsum() - regime_returns.cumsum().cummax()).min()

                self.regime_performance[regime][strategy] = RegimePerformance(
                    regime=regime,
                    strategy=strategy,
                    avg_return=avg_return,
                    sharpe_ratio=sharpe,
                    win_rate=win_rate,
                    max_drawdown=max_dd,
                    sample_count=len(regime_returns),
                )

    # ========================================================================
    # PREDICTION
    # ========================================================================

    def predict_regime(
        self,
        vix: float,
        sp500_price: float,
        market_data: pd.DataFrame = None,
    ) -> RegimePrediction:
        """
        Predict current market regime.

        Args:
            vix: Current VIX level
            sp500_price: Current S&P 500 price
            market_data: Optional additional market data

        Returns:
            RegimePrediction with regime and confidence
        """
        if not self.is_trained:
            # Fallback to heuristic
            return self._heuristic_prediction(vix, sp500_price)

        # Extract features
        features = self.feature_extractor.extract_features(vix, sp500_price, market_data)
        X = features.to_array().reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        # Predict
        regime = self.model.predict(X_scaled)[0]
        probabilities = dict(zip(
            self.model.classes_,
            self.model.predict_proba(X_scaled)[0]
        ))
        confidence = max(probabilities.values())

        # Check for transition warning
        transition_warning = self._check_transition(probabilities)

        # Estimate duration
        duration = self._estimate_duration(regime, probabilities)

        prediction = RegimePrediction(
            regime=regime,
            confidence=confidence,
            probabilities=probabilities,
            features_used={
                'vix_level': features.vix_level,
                'vix_percentile': features.vix_percentile,
                'sp500_return_20d': features.sp500_return_20d,
                'realized_vol_20d': features.realized_vol_20d,
            },
            transition_warning=transition_warning,
            predicted_duration_days=duration,
        )

        # Store in history
        self.prediction_history.append(prediction)

        return prediction

    def _heuristic_prediction(self, vix: float, sp500_price: float) -> RegimePrediction:
        """Fallback heuristic prediction when model not trained."""
        if vix > 35:
            regime = RegimeType.CRISIS
            confidence = 0.9
        elif vix > 25:
            regime = RegimeType.HIGH_VOL
            confidence = 0.7
        elif vix < 15:
            regime = RegimeType.LOW_VOL
            confidence = 0.7
        else:
            regime = RegimeType.TRANSITION
            confidence = 0.5

        return RegimePrediction(
            regime=regime,
            confidence=confidence,
            probabilities={regime: confidence},
            features_used={'vix_level': vix},
            transition_warning=False,
            predicted_duration_days=10,
        )

    def _check_transition(self, probabilities: Dict[str, float]) -> bool:
        """Check if regime transition is likely."""
        # If no regime has high confidence, transition may be happening
        max_prob = max(probabilities.values())
        second_prob = sorted(probabilities.values())[-2] if len(probabilities) > 1 else 0

        # Close competition between regimes suggests transition
        return (max_prob - second_prob) < 0.2

    def _estimate_duration(self, regime: str, probabilities: Dict[str, float]) -> int:
        """Estimate how long the regime might last."""
        # Base duration by regime type
        base_duration = {
            RegimeType.BULL: 60,
            RegimeType.BEAR: 30,
            RegimeType.HIGH_VOL: 20,
            RegimeType.LOW_VOL: 45,
            RegimeType.CRISIS: 10,
            RegimeType.TRANSITION: 5,
        }

        duration = base_duration.get(regime, 15)

        # Adjust by confidence
        confidence = probabilities.get(regime, 0.5)
        duration = int(duration * (0.5 + confidence))

        return max(1, min(90, duration))

    # ========================================================================
    # STRATEGY WEIGHTS
    # ========================================================================

    def get_regime_weights(
        self,
        regime: str,
        strategies: List[str] = None,
    ) -> Dict[str, float]:
        """
        Get optimal strategy weights for a given regime.

        Args:
            regime: The current regime
            strategies: List of strategies to weight (default: all known)

        Returns:
            Dict mapping strategy to recommended weight
        """
        if regime not in self.regime_performance:
            # Equal weights if no performance data
            if strategies:
                return {s: 1.0 / len(strategies) for s in strategies}
            return {}

        regime_perf = self.regime_performance[regime]

        if strategies is None:
            strategies = list(regime_perf.keys())

        # Weight by Sharpe ratio (with minimum floor)
        sharpes = {}
        for strategy in strategies:
            if strategy in regime_perf:
                sharpes[strategy] = max(regime_perf[strategy].sharpe_ratio, 0.1)
            else:
                sharpes[strategy] = 0.5  # Default for unknown

        total = sum(sharpes.values())
        return {s: sharpes[s] / total for s in strategies}

    def get_dynamic_weights(
        self,
        current_prediction: RegimePrediction,
        base_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Get dynamically adjusted weights based on prediction.

        Blends base weights with regime-optimal weights based on confidence.

        Args:
            current_prediction: Current regime prediction
            base_weights: Base strategy weights

        Returns:
            Adjusted weights dict
        """
        strategies = list(base_weights.keys())
        regime_weights = self.get_regime_weights(current_prediction.regime, strategies)

        # Blend based on confidence
        confidence = current_prediction.confidence
        adjusted = {}

        for strategy in strategies:
            base = base_weights.get(strategy, 0)
            regime = regime_weights.get(strategy, base)

            # Higher confidence = more regime-specific weighting
            adjusted[strategy] = base * (1 - confidence) + regime * confidence

        # Normalize
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {s: w / total for s, w in adjusted.items()}

        return adjusted

    # ========================================================================
    # MODEL PERSISTENCE
    # ========================================================================

    def _save_model(self):
        """Save model to disk."""
        if not self.is_trained:
            return

        self.config.model_path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance,
            'regime_performance': self.regime_performance,
            'training_accuracy': self.training_accuracy,
            'last_training': self.last_training.isoformat() if self.last_training else None,
        }

        with open(self.config.model_path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {self.config.model_path}")

    def _load_model(self):
        """Load model from disk."""
        if not self.config.model_path.exists():
            return

        try:
            # Use custom unpickler to handle __main__ module remapping
            # This fixes issues when model was saved from __main__ context
            with open(self.config.model_path, 'rb') as f:
                model_data = _ModuleRemappingUnpickler(f).load()

            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_importance = model_data.get('feature_importance', {})
            self.regime_performance = model_data.get('regime_performance', {})
            self.training_accuracy = model_data.get('training_accuracy', 0)

            if model_data.get('last_training'):
                self.last_training = datetime.fromisoformat(model_data['last_training'])

            self.is_trained = True
            logger.info(f"Model loaded from {self.config.model_path}")

        except Exception as e:
            logger.warning(f"Failed to load model: {e}")

    # ========================================================================
    # ONLINE LEARNING
    # ========================================================================

    def online_update(
        self,
        features: FeatureSet,
        actual_regime: str,
    ):
        """
        Update model with new observation (online learning).

        Args:
            features: Features from the observation
            actual_regime: The actual regime that occurred
        """
        if not self.is_trained:
            logger.warning("Model not trained, cannot do online update")
            return

        # For Random Forest, we can't do true online learning
        # Instead, we accumulate data and retrain periodically
        # This is a placeholder for more sophisticated online learning

        logger.debug(f"Online update: regime={actual_regime}")

    def should_retrain(self) -> bool:
        """Check if model should be retrained."""
        if not self.is_trained:
            return True

        if self.last_training is None:
            return True

        days_since = (datetime.now() - self.last_training).days
        return days_since >= self.config.retrain_frequency_days

    # ========================================================================
    # REPORTING
    # ========================================================================

    def print_status(self):
        """Print model status."""
        print("\n" + "=" * 70)
        print("ML REGIME DETECTOR STATUS")
        print("=" * 70)

        print(f"\nModel: {self.config.model_type}")
        print(f"Trained: {self.is_trained}")

        if self.is_trained:
            print(f"Training Accuracy: {self.training_accuracy:.1%}")
            print(f"Last Training: {self.last_training}")

            if self.feature_importance:
                print(f"\nTop Features:")
                for feat, imp in sorted(self.feature_importance.items(), key=lambda x: -x[1])[:5]:
                    print(f"  {feat}: {imp:.3f}")

            if self.regime_performance:
                print(f"\nStrategy Performance by Regime:")
                for regime, strategies in self.regime_performance.items():
                    print(f"\n  {regime.upper()}:")
                    for strategy, perf in sorted(strategies.items(), key=lambda x: -x[1].sharpe_ratio):
                        print(f"    {strategy}: Sharpe={perf.sharpe_ratio:.2f}, WR={perf.win_rate:.1%}")

        if self.prediction_history:
            recent = list(self.prediction_history)[-5:]
            print(f"\nRecent Predictions:")
            for pred in recent:
                print(f"  {pred.timestamp[:16]}: {pred.regime} (conf={pred.confidence:.1%})")

        print("\n" + "=" * 70)

    def get_status_dict(self) -> Dict[str, Any]:
        """Get status as dictionary."""
        return {
            'is_trained': self.is_trained,
            'model_type': self.config.model_type,
            'training_accuracy': self.training_accuracy,
            'last_training': self.last_training.isoformat() if self.last_training else None,
            'feature_importance': self.feature_importance,
            'should_retrain': self.should_retrain(),
        }


# ============================================================================
# FACTORY
# ============================================================================

_detector: Optional[MLRegimeDetector] = None

def get_ml_regime_detector() -> MLRegimeDetector:
    """Get or create global ML regime detector."""
    global _detector
    if _detector is None:
        _detector = MLRegimeDetector()
    return _detector


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    if not SKLEARN_AVAILABLE:
        print("scikit-learn not available. Install with: pip install scikit-learn")
        sys.exit(1)

    print("=" * 60)
    print("ML REGIME DETECTION DEMO")
    print("=" * 60)

    # Generate synthetic data for demo
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='B')

    # Simulate VIX
    vix = pd.Series(
        15 + 5 * np.sin(np.linspace(0, 8*np.pi, len(dates))) +
        np.random.randn(len(dates)) * 3,
        index=dates
    ).clip(10, 80)

    # Add crisis spikes
    vix.loc['2020-03-01':'2020-04-15'] += 30
    vix.loc['2022-01-01':'2022-03-01'] += 10

    # Simulate S&P 500
    sp500 = pd.Series(
        3000 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.01 + 0.0003)),
        index=dates
    )

    # Simulate strategy returns
    strategy_returns = {
        'momentum': pd.Series(np.random.randn(len(dates)) * 0.02 + 0.0005, index=dates),
        'mean_reversion': pd.Series(np.random.randn(len(dates)) * 0.015 + 0.0003, index=dates),
        'pairs_trading': pd.Series(np.random.randn(len(dates)) * 0.01 + 0.0002, index=dates),
    }

    # Create and train detector
    detector = MLRegimeDetector()

    print("\nTraining on synthetic data...")
    results = detector.train(vix, sp500, strategy_returns)

    print(f"\nTraining Results:")
    print(f"  Accuracy: {results['accuracy']:.1%}")
    print(f"  CV Accuracy: {results['cv_accuracy']:.1%} (+/- {results['cv_std']:.1%})")
    print(f"  Samples: {results['samples']}")
    print(f"\n  Regime Counts: {results['regime_counts']}")
    print(f"\n  Top Features: {list(results['feature_importance'].keys())[:5]}")

    # Make predictions
    print("\nMaking predictions...")
    prediction = detector.predict_regime(
        vix=vix.iloc[-1],
        sp500_price=sp500.iloc[-1],
    )

    print(f"\nCurrent Prediction:")
    print(f"  Regime: {prediction.regime}")
    print(f"  Confidence: {prediction.confidence:.1%}")
    print(f"  Transition Warning: {prediction.transition_warning}")
    print(f"  Predicted Duration: {prediction.predicted_duration_days} days")
    print(f"  Probabilities: {prediction.probabilities}")

    # Get weights
    weights = detector.get_regime_weights(prediction.regime, ['momentum', 'mean_reversion', 'pairs_trading'])
    print(f"\nRecommended Weights for {prediction.regime}:")
    for strat, weight in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {strat}: {weight:.1%}")

    # Print full status
    detector.print_status()
