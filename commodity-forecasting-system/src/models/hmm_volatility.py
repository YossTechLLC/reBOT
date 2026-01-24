"""
HMM Volatility Regime Detection
================================
Hidden Markov Model for detecting volatility regimes (not directional regimes).

Key Differences from Directional HMM:
- Target: intraday_range_pct (volatility) instead of returns
- Regimes: low_vol, normal_vol, high_vol (not bull/bear/neutral)
- Features: overnight_gap_abs, volatility metrics, VIX
- Output: Expected volatility per regime (for confidence scoring)

Regime Definitions:
- low_vol: <0.8% average intraday range (SKIP days)
- normal_vol: 0.8-1.5% average intraday range (TRADE small)
- high_vol: >1.5% average intraday range (TRADE full size)
"""

import logging
import pickle
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class VolatilityHMM:
    """
    Hidden Markov Model for volatility regime detection.

    Architecture:
    - 3 states (low_vol, normal_vol, high_vol)
    - Gaussian emissions (continuous observations)
    - Features: overnight_gap_abs, range_ma_5, vix_level, volume_ratio
    """

    def __init__(self, n_regimes: int = 3, random_state: int = 42):
        """
        Initialize HMM.

        Args:
            n_regimes: Number of volatility regimes (default: 3)
            random_state: Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        self.random_state = random_state

        # HMM model (will be trained)
        self.model = None

        # Feature scaler
        self.scaler = StandardScaler()

        # Regime metadata (learned during training)
        self.regime_labels = []  # ['low_vol', 'normal_vol', 'high_vol']
        self.regime_volatilities = {}  # {'low_vol': 0.006, 'normal_vol': 0.012, ...}
        self.regime_mappings = {}  # {0: 'low_vol', 1: 'normal_vol', 2: 'high_vol'}

        logger.info(f"Initialized VolatilityHMM with {n_regimes} regimes")

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.Series]:
        """
        Extract features for HMM training.

        Features:
        1. overnight_gap_abs - Morning volatility predictor
        2. range_ma_5 - Recent volatility trend
        3. vix_level - External fear gauge
        4. volume_ratio - Confirmation signal
        5. range_std_5 - Volatility of volatility

        Args:
            df: DataFrame with engineered features

        Returns:
            (feature_matrix, target_series)
        """
        required_cols = [
            'overnight_gap_abs',
            'range_ma_5',
            'vix_level',
            'volume_ratio',
            'range_std_5'
        ]

        # Check for missing feature columns
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")

        # Check for target column
        if 'intraday_range_pct' not in df.columns:
            raise ValueError(
                "Missing target column 'intraday_range_pct'. "
                "DataFrame must include both features AND target column."
            )

        # Extract features
        X = df[required_cols].values

        # Target: intraday volatility
        y = df['intraday_range_pct']

        logger.info(f"Prepared features: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y

    def train(
        self,
        df: pd.DataFrame,
        n_iter: int = 100,
        tol: float = 1e-4
    ) -> Dict:
        """
        Train HMM on historical data.

        Args:
            df: DataFrame with engineered features
            n_iter: Maximum training iterations
            tol: Convergence tolerance

        Returns:
            Training metrics dictionary
        """
        logger.info("Training HMM on volatility features...")

        # Prepare features
        X, y = self.prepare_features(df)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Initialize HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type='diag',
            n_iter=n_iter,
            tol=tol,
            random_state=self.random_state
        )

        # Fit model
        self.model.fit(X_scaled)

        # Predict regimes
        regimes = self.model.predict(X_scaled)

        # Calculate regime statistics
        self._learn_regime_mappings(regimes, y)

        # Calculate training metrics
        metrics = self._calculate_metrics(regimes, y)

        logger.info(f"Training complete. Converged: {self.model.monitor_.converged}")
        logger.info(f"Regime volatilities: {self.regime_volatilities}")

        return metrics

    def _learn_regime_mappings(self, regimes: np.ndarray, volatilities: pd.Series):
        """
        Learn which regime index corresponds to which volatility level.

        Args:
            regimes: Regime predictions (0, 1, 2)
            volatilities: Actual volatility values
        """
        # Calculate average volatility per regime
        regime_vols = {}
        for regime_idx in range(self.n_regimes):
            mask = regimes == regime_idx
            avg_vol = volatilities[mask].mean()
            regime_vols[regime_idx] = avg_vol

        # Sort regimes by volatility (low to high)
        sorted_regimes = sorted(regime_vols.items(), key=lambda x: x[1])

        # Map to labels
        labels = ['low_vol', 'normal_vol', 'high_vol'][:self.n_regimes]

        self.regime_mappings = {}
        self.regime_volatilities = {}

        for label, (regime_idx, avg_vol) in zip(labels, sorted_regimes):
            self.regime_mappings[regime_idx] = label
            self.regime_volatilities[label] = avg_vol

        self.regime_labels = labels

        logger.info(f"Regime mappings learned: {self.regime_mappings}")

    def _calculate_metrics(self, regimes: np.ndarray, volatilities: pd.Series) -> Dict:
        """
        Calculate training metrics.

        Returns:
            Dictionary with metrics
        """
        metrics = {
            'n_samples': len(regimes),
            'n_regimes': self.n_regimes,
            'converged': self.model.monitor_.converged,
            'log_likelihood': self.model.score(self.scaler.transform(self.prepare_features(
                pd.DataFrame({
                    'overnight_gap_abs': volatilities.index.map(lambda x: 0),
                    'range_ma_5': volatilities.values,
                    'vix_level': [15] * len(volatilities),
                    'volume_ratio': [1] * len(volatilities),
                    'range_std_5': volatilities.values,
                    'intraday_range_pct': volatilities.values
                })
            )[0]))
        }

        # Regime distribution
        regime_counts = pd.Series(regimes).value_counts()
        for regime_idx, count in regime_counts.items():
            label = self.regime_mappings.get(regime_idx, f'regime_{regime_idx}')
            metrics[f'{label}_count'] = count
            metrics[f'{label}_pct'] = count / len(regimes) * 100

        return metrics

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
        """
        Predict volatility regimes for new data.

        Args:
            df: DataFrame with engineered features

        Returns:
            (regime_indices, regime_labels, expected_volatilities)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Prepare features
        X, y = self.prepare_features(df)

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Predict regimes
        regime_indices = self.model.predict(X_scaled)

        # Map to labels
        regime_labels = np.array([self.regime_mappings[idx] for idx in regime_indices])

        # Get expected volatility per regime
        expected_vols = pd.Series([self.regime_volatilities[label] for label in regime_labels],
                                   index=df.index)

        return regime_indices, regime_labels, expected_vols

    def predict_latest(self, df: pd.DataFrame) -> Dict:
        """
        Predict regime for most recent observation.

        Args:
            df: DataFrame with engineered features

        Returns:
            Dictionary with prediction details
        """
        if len(df) == 0:
            raise ValueError("DataFrame is empty")

        # Predict all
        regime_indices, regime_labels, expected_vols = self.predict(df)

        # Get latest
        latest_idx = regime_indices[-1]
        latest_label = regime_labels[-1]
        latest_vol = expected_vols.iloc[-1]

        # Calculate regime probabilities
        X, _ = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)[-1]

        # Map probabilities to labels
        prob_dict = {self.regime_mappings[i]: probs[i] for i in range(self.n_regimes)}

        return {
            'regime_index': latest_idx,
            'regime_label': latest_label,
            'expected_volatility': latest_vol,
            'regime_probabilities': prob_dict,
            'confidence': probs[latest_idx]  # Probability of predicted regime
        }

    def save(self, filepath: str):
        """
        Save trained model to disk.

        Args:
            filepath: Path to save pickle file
        """
        if self.model is None:
            raise ValueError("Model not trained. Nothing to save.")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'regime_labels': self.regime_labels,
            'regime_volatilities': self.regime_volatilities,
            'regime_mappings': self.regime_mappings,
            'n_regimes': self.n_regimes
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """
        Load trained model from disk.

        Args:
            filepath: Path to pickle file
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.regime_labels = model_data['regime_labels']
        self.regime_volatilities = model_data['regime_volatilities']
        self.regime_mappings = model_data['regime_mappings']
        self.n_regimes = model_data['n_regimes']

        logger.info(f"Model loaded from {filepath}")
        logger.info(f"Regime volatilities: {self.regime_volatilities}")


def test_hmm_volatility():
    """Test HMM volatility detection."""
    print("Testing HMM Volatility Detection...")
    print("=" * 60)

    # Create sample data
    np.random.seed(42)
    n_samples = 100

    # Simulate 3 volatility regimes
    regimes = np.random.choice([0, 1, 2], size=n_samples, p=[0.5, 0.3, 0.2])

    # Generate features based on regimes
    overnight_gap_abs = np.where(regimes == 0, 0.003,
                                  np.where(regimes == 1, 0.008, 0.015)) + np.random.randn(n_samples) * 0.002
    range_ma_5 = np.where(regimes == 0, 0.005,
                          np.where(regimes == 1, 0.010, 0.018)) + np.random.randn(n_samples) * 0.002
    vix_level = np.where(regimes == 0, 12,
                         np.where(regimes == 1, 18, 28)) + np.random.randn(n_samples) * 3
    volume_ratio = np.where(regimes == 0, 0.9,
                            np.where(regimes == 1, 1.1, 1.5)) + np.random.randn(n_samples) * 0.2
    range_std_5 = np.where(regimes == 0, 0.002,
                           np.where(regimes == 1, 0.004, 0.008)) + np.random.randn(n_samples) * 0.001

    # Target: intraday volatility
    intraday_range_pct = np.where(regimes == 0, 0.006,
                                   np.where(regimes == 1, 0.012, 0.020)) + np.random.randn(n_samples) * 0.003

    df = pd.DataFrame({
        'overnight_gap_abs': np.abs(overnight_gap_abs),
        'range_ma_5': np.abs(range_ma_5),
        'vix_level': vix_level,
        'volume_ratio': np.abs(volume_ratio),
        'range_std_5': np.abs(range_std_5),
        'intraday_range_pct': np.abs(intraday_range_pct)
    })

    # Train HMM
    print("\n1. Training HMM...")
    print("-" * 60)
    hmm_model = VolatilityHMM(n_regimes=3)
    metrics = hmm_model.train(df)

    print(f"   Converged: {metrics['converged']}")
    print(f"   Samples: {metrics['n_samples']}")
    print(f"   Regime distribution:")
    for label in hmm_model.regime_labels:
        print(f"      {label}: {metrics.get(f'{label}_count', 0)} ({metrics.get(f'{label}_pct', 0):.1f}%)")

    # Test prediction
    print("\n2. Testing Prediction...")
    print("-" * 60)

    # Predict on last 10 samples
    test_df = df.tail(10)
    regime_indices, regime_labels, expected_vols = hmm_model.predict(test_df)

    print(f"   Predicted regimes (last 10):")
    for i, (label, vol) in enumerate(zip(regime_labels, expected_vols)):
        actual_vol = test_df['intraday_range_pct'].iloc[i]
        print(f"      {i+1}. {label}: expected={vol:.3f}, actual={actual_vol:.3f}")

    # Test latest prediction
    print("\n3. Testing Latest Prediction...")
    print("-" * 60)

    latest = hmm_model.predict_latest(df)
    print(f"   Regime: {latest['regime_label']}")
    print(f"   Expected volatility: {latest['expected_volatility']:.3f}")
    print(f"   Confidence: {latest['confidence']:.2%}")
    print(f"   Probabilities:")
    for label, prob in latest['regime_probabilities'].items():
        print(f"      {label}: {prob:.2%}")

    # Test save/load
    print("\n4. Testing Save/Load...")
    print("-" * 60)

    hmm_model.save('models/hmm_volatility_test.pkl')
    print("   ✅ Model saved")

    hmm_model2 = VolatilityHMM()
    hmm_model2.load('models/hmm_volatility_test.pkl')
    print("   ✅ Model loaded")

    latest2 = hmm_model2.predict_latest(df)
    assert latest2['regime_label'] == latest['regime_label'], "Load failed - predictions don't match"
    print("   ✅ Predictions match")

    print("\n" + "=" * 60)
    print("✅ All tests passed!")


if __name__ == "__main__":
    test_hmm_volatility()
