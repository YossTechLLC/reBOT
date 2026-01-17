"""
Hidden Markov Model Core Module
Wrapper for hmmlearn with commodity-specific enhancements.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class CommodityHMM:
    """
    Hidden Markov Model for commodity regime detection and forecasting.

    Features:
    - Multiple random initializations to avoid local optima
    - Intelligent regime labeling (bull/bear/neutral)
    - Spot price forecasting with uncertainty quantification
    - Regime analysis and transition dynamics
    - Model persistence

    Mathematical Framework:
    - Hidden states: S = {s₁, s₂, ..., sₙ} (typically n=3)
    - Observations: O (price returns, technical indicators, volatility)
    - Transition matrix: A[i,j] = P(sₜ = sⱼ | sₜ₋₁ = sᵢ)
    - Emission distributions: B ~ N(μ, Σ) (Gaussian)
    - Initial state: π[i] = P(s₁ = sᵢ)

    Training Algorithm: Baum-Welch (Expectation-Maximization)
    Inference Algorithm: Forward-backward, Viterbi
    """

    def __init__(self, config: Dict):
        """
        Initialize CommodityHMM.

        Args:
            config: Configuration dictionary with 'hmm' section
        """
        self.config = config
        self.hmm_config = config.get('hmm', {})

        # HMM parameters
        self.n_states = self.hmm_config.get('n_states', 3)
        self.covariance_type = self.hmm_config.get('covariance_type', 'diag')
        self.n_iter = self.hmm_config.get('n_iter', 1000)
        self.tol = self.hmm_config.get('tol', 1e-4)
        self.n_random_inits = self.hmm_config.get('n_random_inits', 10)
        self.random_seed = self.hmm_config.get('random_seed', 42)

        # Model state
        self.model = None
        self.scaler = None
        self.regime_stats = {}
        self.feature_names = []
        self.is_fitted = False

        # Training history
        self.training_history = {
            'scores': [],
            'convergence': [],
            'n_iter_used': []
        }

        logger.info(f"Initialized CommodityHMM with {self.n_states} states")
        logger.info(f"Covariance type: {self.covariance_type}")
        logger.info(f"Max iterations: {self.n_iter}, Tolerance: {self.tol}")

    def fit(
        self,
        features: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None
    ) -> 'CommodityHMM':
        """
        Fit HMM using single initialization (use fit_with_multiple_inits for robustness).

        Args:
            features: Feature matrix (n_samples, n_features)
            feature_names: Optional list of feature names

        Returns:
            self (fitted model)
        """
        # Convert to numpy if DataFrame
        if isinstance(features, pd.DataFrame):
            self.feature_names = list(features.columns)
            features = features.values
        elif feature_names is not None:
            self.feature_names = feature_names

        # Scale features
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)

        # Validate features
        if np.isnan(features_scaled).any():
            raise ValueError(
                "TFM2001 DATA: Features contain NaN values. "
                "Please handle missing data before fitting."
            )

        # Initialize and fit HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=self.random_seed,
            verbose=False
        )

        logger.info(f"Fitting HMM on {len(features)} samples with {features.shape[1]} features")

        try:
            self.model.fit(features_scaled)

            # Check convergence
            if self.model.monitor_.iter == self.n_iter:
                logger.warning(
                    f"TFM4001 INFERENCE: HMM did not converge within {self.n_iter} iterations. "
                    f"Consider increasing n_iter or adjusting tol."
                )

            # Calculate score
            score = self.model.score(features_scaled)
            logger.info(f"HMM fit complete. Log-likelihood: {score:.2f}")

            # Analyze regimes
            self._analyze_regimes(features_scaled, features)

            self.is_fitted = True
            return self

        except Exception as e:
            logger.error(f"TFM4001 INFERENCE: HMM fit failed: {e}")
            raise

    def fit_with_multiple_inits(
        self,
        features: Union[pd.DataFrame, np.ndarray],
        n_inits: Optional[int] = None,
        feature_names: Optional[List[str]] = None
    ) -> 'CommodityHMM':
        """
        Fit HMM with multiple random initializations to avoid local optima.

        This is the recommended fitting method for production use.

        Args:
            features: Feature matrix (n_samples, n_features)
            n_inits: Number of random initializations (default: from config)
            feature_names: Optional list of feature names

        Returns:
            self (fitted model with best initialization)
        """
        n_inits = n_inits or self.n_random_inits

        # Convert to numpy if DataFrame
        if isinstance(features, pd.DataFrame):
            self.feature_names = list(features.columns)
            features = features.values
        elif feature_names is not None:
            self.feature_names = feature_names

        # Scale features
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)

        # Validate features
        if np.isnan(features_scaled).any():
            raise ValueError(
                "TFM2001 DATA: Features contain NaN values. "
                "Please handle missing data before fitting."
            )

        logger.info(
            f"Fitting HMM with {n_inits} random initializations "
            f"on {len(features)} samples"
        )

        best_score = -np.inf
        best_model = None
        best_seed = None
        failed_inits = 0

        for seed_offset in range(n_inits):
            seed = self.random_seed + seed_offset

            try:
                # Initialize new model
                model = hmm.GaussianHMM(
                    n_components=self.n_states,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter,
                    tol=self.tol,
                    random_state=seed,
                    verbose=False
                )

                # Fit model
                model.fit(features_scaled)

                # Calculate score
                score = model.score(features_scaled)

                # Track training history
                self.training_history['scores'].append(score)
                self.training_history['convergence'].append(
                    model.monitor_.iter < self.n_iter
                )
                self.training_history['n_iter_used'].append(model.monitor_.iter)

                logger.debug(
                    f"Init {seed_offset + 1}/{n_inits}: "
                    f"score={score:.2f}, iter={model.monitor_.iter}"
                )

                # Update best model
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_seed = seed

            except Exception as e:
                failed_inits += 1
                logger.warning(
                    f"TFM4001 INFERENCE: Initialization {seed_offset + 1} failed: {e}"
                )
                continue

        # Check if any initialization succeeded
        if best_model is None:
            raise RuntimeError(
                f"TFM4001 INFERENCE: All {n_inits} HMM initializations failed. "
                f"Check data quality and HMM parameters."
            )

        if failed_inits > 0:
            logger.warning(
                f"TFM4001 INFERENCE: {failed_inits}/{n_inits} initializations failed"
            )

        # Set best model
        self.model = best_model
        logger.info(
            f"Best model: score={best_score:.2f}, seed={best_seed}, "
            f"converged={best_model.monitor_.iter < self.n_iter}"
        )

        # Analyze regimes
        self._analyze_regimes(features_scaled, features)

        self.is_fitted = True
        return self

    def _analyze_regimes(
        self,
        features_scaled: np.ndarray,
        features_raw: np.ndarray
    ) -> None:
        """
        Analyze regime characteristics and assign labels.

        Args:
            features_scaled: Scaled features used for training
            features_raw: Original unscaled features
        """
        logger.info("Analyzing regime characteristics...")

        # Predict states
        states = self.model.predict(features_scaled)

        # Analyze each regime
        for state in range(self.n_states):
            state_mask = (states == state)
            state_count = np.sum(state_mask)

            if state_count == 0:
                logger.warning(f"State {state} has no observations")
                continue

            # Extract returns (assuming first feature is returns or price-based)
            state_returns = features_raw[state_mask, 0]

            # Calculate statistics
            self.regime_stats[state] = {
                'mean_return': float(np.mean(state_returns)),
                'std_return': float(np.std(state_returns)),
                'volatility': float(np.std(state_returns)),
                'sharpe': float(np.mean(state_returns) / np.std(state_returns))
                if np.std(state_returns) > 0 else 0.0,
                'count': int(state_count),
                'frequency': float(state_count / len(states)),
                'persistence': float(self.model.transmat_[state, state]),
                'median_return': float(np.median(state_returns)),
                'skewness': float(pd.Series(state_returns).skew()),
                'kurtosis': float(pd.Series(state_returns).kurtosis())
            }

        # Label regimes intelligently
        self._label_regimes()

        # Log regime summary
        logger.info("Regime analysis complete:")
        for state, stats in self.regime_stats.items():
            label = stats['label']
            logger.info(
                f"  {label} (State {state}): "
                f"mean_return={stats['mean_return']:.4f}, "
                f"volatility={stats['volatility']:.4f}, "
                f"persistence={stats['persistence']:.2f}, "
                f"frequency={stats['frequency']:.1%}"
            )

    def _label_regimes(self) -> None:
        """
        Assign interpretable labels to regimes based on characteristics.

        Labels:
        - Bull: High positive returns
        - Bear: Negative returns
        - Neutral: Low/moderate returns with low volatility
        - High Volatility: High volatility regardless of direction
        """
        # Sort states by mean return
        returns = [stats['mean_return'] for stats in self.regime_stats.values()]
        sorted_states = sorted(range(self.n_states), key=lambda x: returns[x])

        labels = {}

        if self.n_states == 2:
            # Binary: Bull vs Bear
            labels[sorted_states[0]] = 'bear'
            labels[sorted_states[1]] = 'bull'

        elif self.n_states == 3:
            # Classic: Bull / Neutral / Bear
            labels[sorted_states[0]] = 'bear'
            labels[sorted_states[1]] = 'neutral'
            labels[sorted_states[2]] = 'bull'

        elif self.n_states == 4:
            # Extended: Strong Bear / Bear / Bull / Strong Bull
            labels[sorted_states[0]] = 'strong_bear'
            labels[sorted_states[1]] = 'bear'
            labels[sorted_states[2]] = 'bull'
            labels[sorted_states[3]] = 'strong_bull'

        else:
            # Generic numbering for n_states > 4
            for i, state in enumerate(sorted_states):
                labels[state] = f'regime_{i}'

        # Assign labels to regime stats
        for state in range(self.n_states):
            self.regime_stats[state]['label'] = labels.get(state, f'state_{state}')

    def predict_regime(
        self,
        features: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[int, str]:
        """
        Predict current regime for given features.

        Args:
            features: Feature matrix (n_samples, n_features)

        Returns:
            Tuple of (state_index, regime_label)
        """
        self._check_fitted()

        # Convert to numpy if needed
        if isinstance(features, pd.DataFrame):
            features = features.values

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Predict state
        state = self.model.predict(features_scaled)[-1]
        label = self.regime_stats[state]['label']

        return int(state), label

    def predict_proba(
        self,
        features: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Calculate posterior probabilities for each regime.

        Args:
            features: Feature matrix (n_samples, n_features)

        Returns:
            Array of shape (n_samples, n_states) with state probabilities
        """
        self._check_fitted()

        # Convert to numpy if needed
        if isinstance(features, pd.DataFrame):
            features = features.values

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Calculate posteriors
        posteriors = self.model.predict_proba(features_scaled)

        return posteriors

    def forecast_spot_price(
        self,
        current_features: Union[pd.DataFrame, np.ndarray],
        horizon: int = 1,
        n_simulations: int = 10000,
        current_price: Optional[float] = None
    ) -> Dict:
        """
        Forecast spot price using regime-based Monte Carlo simulation.

        Args:
            current_features: Current feature values
            horizon: Forecast horizon (days)
            n_simulations: Number of Monte Carlo paths
            current_price: Current spot price (if None, extracted from features)

        Returns:
            Dictionary with forecast statistics:
            - mean: Expected price
            - std: Standard deviation
            - quantiles: [5%, 25%, 50%, 75%, 95%]
            - regime: Current regime label
            - regime_persistence: Probability of staying in regime
        """
        self._check_fitted()

        # Get current regime
        current_state, regime_label = self.predict_regime(current_features[-1:])

        # Regime statistics
        regime_stats = self.regime_stats[current_state]
        expected_return = regime_stats['mean_return']
        expected_vol = regime_stats['volatility']
        persistence = regime_stats['persistence']

        # Extract current price
        if current_price is None:
            # Assume first feature contains price information
            if isinstance(current_features, pd.DataFrame):
                current_price = current_features.iloc[-1, 0]
            else:
                current_price = current_features[-1, 0]

        logger.info(
            f"Forecasting {horizon} days ahead from {regime_label} regime "
            f"(persistence={persistence:.2f})"
        )

        # Monte Carlo simulation
        simulated_prices = []

        for _ in range(n_simulations):
            price = current_price

            for day in range(horizon):
                # Geometric Brownian Motion with regime parameters
                dt = 1.0 / 252  # Daily time step
                drift = expected_return * dt
                diffusion = expected_vol * np.sqrt(dt) * np.random.randn()

                # Update price
                price = price * np.exp(drift + diffusion)

            simulated_prices.append(price)

        simulated_prices = np.array(simulated_prices)

        # Calculate statistics
        forecast_mean = np.mean(simulated_prices)
        forecast_std = np.std(simulated_prices)
        forecast_quantiles = np.percentile(
            simulated_prices,
            [5, 25, 50, 75, 95]
        )

        logger.info(
            f"Forecast: mean=${forecast_mean:.2f}, "
            f"std=${forecast_std:.2f}, "
            f"median=${forecast_quantiles[2]:.2f}"
        )

        return {
            'mean': float(forecast_mean),
            'std': float(forecast_std),
            'median': float(forecast_quantiles[2]),
            'quantiles': {
                '5%': float(forecast_quantiles[0]),
                '25%': float(forecast_quantiles[1]),
                '50%': float(forecast_quantiles[2]),
                '75%': float(forecast_quantiles[3]),
                '95%': float(forecast_quantiles[4])
            },
            'regime': regime_label,
            'regime_state': int(current_state),
            'regime_persistence': float(persistence),
            'horizon': horizon,
            'n_simulations': n_simulations,
            'current_price': float(current_price)
        }

    def get_transition_matrix(self) -> np.ndarray:
        """
        Get state transition matrix.

        Returns:
            Transition matrix A of shape (n_states, n_states)
            where A[i,j] = P(sₜ = sⱼ | sₜ₋₁ = sᵢ)
        """
        self._check_fitted()
        return self.model.transmat_.copy()

    def get_emission_params(self) -> Dict:
        """
        Get emission distribution parameters.

        Returns:
            Dictionary with 'means' and 'covars'
        """
        self._check_fitted()
        return {
            'means': self.model.means_.copy(),
            'covars': self.model.covars_.copy()
        }

    def get_regime_stats(self) -> Dict:
        """
        Get comprehensive regime statistics.

        Returns:
            Dictionary mapping state index to regime statistics
        """
        self._check_fitted()
        return self.regime_stats.copy()

    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        Save trained model to disk.

        Args:
            filepath: Path to save model
        """
        self._check_fitted()

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'regime_stats': self.regime_stats,
            'config': self.config,
            'feature_names': self.feature_names,
            'training_history': self.training_history
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: Union[str, Path]) -> 'CommodityHMM':
        """
        Load trained model from disk.

        Args:
            filepath: Path to saved model

        Returns:
            Loaded CommodityHMM instance
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(
                f"TFM3001 CHECKPOINT: Model file not found: {filepath}"
            )

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        # Reconstruct model
        instance = cls(model_data['config'])
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.regime_stats = model_data['regime_stats']
        instance.feature_names = model_data.get('feature_names', [])
        instance.training_history = model_data.get('training_history', {})
        instance.is_fitted = True

        logger.info(f"Model loaded from {filepath}")
        return instance

    def _check_fitted(self) -> None:
        """Check if model is fitted."""
        if not self.is_fitted:
            raise RuntimeError(
                "TFM4001 INFERENCE: Model not fitted. "
                "Call fit() or fit_with_multiple_inits() first."
            )

    def __repr__(self) -> str:
        """String representation."""
        if self.is_fitted:
            return (
                f"CommodityHMM(n_states={self.n_states}, "
                f"covariance='{self.covariance_type}', fitted=True)"
            )
        else:
            return (
                f"CommodityHMM(n_states={self.n_states}, "
                f"covariance='{self.covariance_type}', fitted=False)"
            )


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('../..')

    from src.config.loader import load_config
    from src.config.logging_config import setup_logging
    from src.data.acquisition import CommodityDataAcquisition
    from src.data.preprocessing import DataPreprocessor
    from src.data.features import FeatureEngineer

    # Load configuration
    config = load_config('../../config/parameters.toml')
    logger = setup_logging(config)

    # Fetch and preprocess data
    data_client = CommodityDataAcquisition(config)
    data = data_client.fetch_commodity_prices()

    preprocessor = DataPreprocessor(config)
    data, _ = preprocessor.preprocess(data)

    # Engineer features
    feature_engineer = FeatureEngineer(config)
    features = feature_engineer.engineer_features(data)

    # Select subset of features for HMM
    feature_cols = ['returns', 'volatility_5', 'rsi_14', 'macd']
    hmm_features = features[feature_cols].dropna()

    print(f"\nTraining HMM on {len(hmm_features)} samples with {len(feature_cols)} features")

    # Train HMM
    hmm_model = CommodityHMM(config)
    hmm_model.fit_with_multiple_inits(hmm_features, n_inits=5)

    # Display regime statistics
    print("\n" + "="*80)
    print("REGIME ANALYSIS")
    print("="*80)

    for state, stats in hmm_model.get_regime_stats().items():
        print(f"\n{stats['label'].upper()} (State {state}):")
        print(f"  Mean Return: {stats['mean_return']:.4f}")
        print(f"  Volatility: {stats['volatility']:.4f}")
        print(f"  Sharpe Ratio: {stats['sharpe']:.2f}")
        print(f"  Persistence: {stats['persistence']:.2f}")
        print(f"  Frequency: {stats['frequency']:.1%}")

    # Transition matrix
    print("\n" + "="*80)
    print("TRANSITION MATRIX")
    print("="*80)
    trans_matrix = hmm_model.get_transition_matrix()
    print(pd.DataFrame(
        trans_matrix,
        columns=[f"To {i}" for i in range(hmm_model.n_states)],
        index=[f"From {i}" for i in range(hmm_model.n_states)]
    ))

    # Forecast
    print("\n" + "="*80)
    print("SPOT PRICE FORECAST")
    print("="*80)

    current_features = hmm_features.iloc[-30:]  # Last 30 days
    current_price = data['close'].iloc[-1]

    forecast = hmm_model.forecast_spot_price(
        current_features,
        horizon=30,
        n_simulations=10000,
        current_price=current_price
    )

    print(f"\nCurrent Price: ${forecast['current_price']:.2f}")
    print(f"Current Regime: {forecast['regime']}")
    print(f"\n30-Day Forecast:")
    print(f"  Mean: ${forecast['mean']:.2f}")
    print(f"  Median: ${forecast['median']:.2f}")
    print(f"  Std Dev: ${forecast['std']:.2f}")
    print(f"\nQuantiles:")
    for q, val in forecast['quantiles'].items():
        print(f"  {q}: ${val:.2f}")
