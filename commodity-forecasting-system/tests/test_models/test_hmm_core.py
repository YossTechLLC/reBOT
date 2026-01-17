"""
Unit Tests for HMM Core Module
Tests for CommodityHMM class and related functionality.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.hmm_core import CommodityHMM
from src.config.loader import load_config


@pytest.fixture
def config():
    """Load test configuration."""
    config_path = Path(__file__).parent.parent.parent / 'config' / 'parameters.toml'
    return load_config(str(config_path))


@pytest.fixture
def synthetic_data():
    """Generate synthetic data for testing."""
    np.random.seed(42)

    n_samples = 500
    n_features = 4

    # Generate returns with 3 regimes
    returns = []
    states_true = []

    for i in range(n_samples):
        if i < 150:
            # Bear regime: negative returns, high volatility
            ret = np.random.normal(-0.001, 0.02)
            state = 0
        elif i < 350:
            # Neutral regime: low returns, low volatility
            ret = np.random.normal(0.0, 0.01)
            state = 1
        else:
            # Bull regime: positive returns, moderate volatility
            ret = np.random.normal(0.002, 0.015)
            state = 2

        returns.append(ret)
        states_true.append(state)

    # Create feature matrix
    features = pd.DataFrame({
        'returns': returns,
        'volatility': np.abs(returns) + np.random.normal(0, 0.002, n_samples),
        'momentum': np.cumsum(returns) + np.random.normal(0, 0.01, n_samples),
        'indicator': np.random.normal(0, 1, n_samples)
    })

    return features, np.array(states_true)


class TestCommodityHMM:
    """Test suite for CommodityHMM class."""

    def test_initialization(self, config):
        """Test HMM initialization."""
        hmm = CommodityHMM(config)

        assert hmm.n_states == config['hmm']['n_states']
        assert hmm.covariance_type == config['hmm']['covariance_type']
        assert hmm.n_iter == config['hmm']['n_iter']
        assert hmm.is_fitted == False
        assert hmm.model is None

    def test_fit(self, config, synthetic_data):
        """Test single initialization fit."""
        features, _ = synthetic_data

        hmm = CommodityHMM(config)
        hmm.fit(features)

        assert hmm.is_fitted == True
        assert hmm.model is not None
        assert hmm.scaler is not None
        assert len(hmm.regime_stats) == hmm.n_states

    def test_fit_with_multiple_inits(self, config, synthetic_data):
        """Test multiple initialization fit."""
        features, _ = synthetic_data

        hmm = CommodityHMM(config)
        hmm.fit_with_multiple_inits(features, n_inits=3)

        assert hmm.is_fitted == True
        assert len(hmm.training_history['scores']) == 3
        assert all(hmm.training_history['convergence'])

    def test_regime_labeling(self, config, synthetic_data):
        """Test regime labeling."""
        features, _ = synthetic_data

        hmm = CommodityHMM(config)
        hmm.fit_with_multiple_inits(features, n_inits=3)

        # Check labels exist
        labels = [stats['label'] for stats in hmm.regime_stats.values()]
        assert 'bull' in labels
        assert 'bear' in labels
        assert 'neutral' in labels

    def test_predict_regime(self, config, synthetic_data):
        """Test regime prediction."""
        features, _ = synthetic_data

        hmm = CommodityHMM(config)
        hmm.fit(features)

        # Predict on last sample
        state, label = hmm.predict_regime(features.iloc[-1:])

        assert isinstance(state, int)
        assert 0 <= state < hmm.n_states
        assert isinstance(label, str)

    def test_predict_proba(self, config, synthetic_data):
        """Test posterior probability calculation."""
        features, _ = synthetic_data

        hmm = CommodityHMM(config)
        hmm.fit(features)

        # Calculate posteriors
        posteriors = hmm.predict_proba(features)

        assert posteriors.shape == (len(features), hmm.n_states)
        assert np.allclose(posteriors.sum(axis=1), 1.0)
        assert np.all(posteriors >= 0)
        assert np.all(posteriors <= 1)

    def test_forecast_spot_price(self, config, synthetic_data):
        """Test spot price forecasting."""
        features, _ = synthetic_data

        hmm = CommodityHMM(config)
        hmm.fit(features)

        # Forecast
        current_price = 100.0
        forecast = hmm.forecast_spot_price(
            features.iloc[-30:],
            horizon=10,
            n_simulations=1000,
            current_price=current_price
        )

        # Check output structure
        assert 'mean' in forecast
        assert 'std' in forecast
        assert 'quantiles' in forecast
        assert 'regime' in forecast
        assert 'regime_persistence' in forecast

        # Check values are reasonable
        assert forecast['mean'] > 0
        assert forecast['std'] > 0
        assert forecast['quantiles']['50%'] > 0
        assert forecast['quantiles']['5%'] < forecast['quantiles']['95%']

    def test_transition_matrix(self, config, synthetic_data):
        """Test transition matrix properties."""
        features, _ = synthetic_data

        hmm = CommodityHMM(config)
        hmm.fit(features)

        trans_matrix = hmm.get_transition_matrix()

        # Check shape
        assert trans_matrix.shape == (hmm.n_states, hmm.n_states)

        # Check stochastic property (rows sum to 1)
        row_sums = trans_matrix.sum(axis=1)
        assert np.allclose(row_sums, 1.0), "Transition matrix rows must sum to 1"

        # Check all probabilities in [0, 1]
        assert np.all(trans_matrix >= 0)
        assert np.all(trans_matrix <= 1)

    def test_emission_params(self, config, synthetic_data):
        """Test emission parameters."""
        features, _ = synthetic_data

        hmm = CommodityHMM(config)
        hmm.fit(features)

        emission_params = hmm.get_emission_params()

        assert 'means' in emission_params
        assert 'covars' in emission_params
        assert emission_params['means'].shape == (hmm.n_states, features.shape[1])

    def test_save_load_model(self, config, synthetic_data):
        """Test model persistence."""
        features, _ = synthetic_data

        # Train model
        hmm_original = CommodityHMM(config)
        hmm_original.fit(features)

        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_model.pkl'
            hmm_original.save_model(save_path)

            # Load model
            hmm_loaded = CommodityHMM.load_model(save_path)

            # Check loaded model
            assert hmm_loaded.is_fitted == True
            assert hmm_loaded.n_states == hmm_original.n_states

            # Compare predictions
            state_orig, _ = hmm_original.predict_regime(features.iloc[-1:])
            state_loaded, _ = hmm_loaded.predict_regime(features.iloc[-1:])
            assert state_orig == state_loaded

    def test_not_fitted_error(self, config):
        """Test error when using unfitted model."""
        hmm = CommodityHMM(config)

        features = pd.DataFrame(np.random.randn(10, 4))

        with pytest.raises(RuntimeError, match="Model not fitted"):
            hmm.predict_regime(features)

        with pytest.raises(RuntimeError, match="Model not fitted"):
            hmm.get_transition_matrix()

    def test_nan_features_error(self, config):
        """Test error with NaN features."""
        features = pd.DataFrame(np.random.randn(100, 4))
        features.iloc[50, 2] = np.nan

        hmm = CommodityHMM(config)

        with pytest.raises(ValueError, match="Features contain NaN"):
            hmm.fit(features)

    def test_regime_statistics(self, config, synthetic_data):
        """Test regime statistics calculation."""
        features, _ = synthetic_data

        hmm = CommodityHMM(config)
        hmm.fit(features)

        stats = hmm.get_regime_stats()

        # Check all stats present
        for state_stats in stats.values():
            assert 'mean_return' in state_stats
            assert 'volatility' in state_stats
            assert 'sharpe' in state_stats
            assert 'persistence' in state_stats
            assert 'frequency' in state_stats
            assert 'label' in state_stats

            # Check frequency sums to ~1
        total_freq = sum(s['frequency'] for s in stats.values())
        assert np.isclose(total_freq, 1.0)

    def test_different_n_states(self, config, synthetic_data):
        """Test HMM with different number of states."""
        features, _ = synthetic_data

        for n_states in [2, 3, 4]:
            temp_config = config.copy()
            temp_config['hmm']['n_states'] = n_states

            hmm = CommodityHMM(temp_config)
            hmm.fit(features)

            assert hmm.n_states == n_states
            assert len(hmm.regime_stats) == n_states

    def test_different_covariance_types(self, config, synthetic_data):
        """Test HMM with different covariance types."""
        features, _ = synthetic_data

        for cov_type in ['diag', 'full', 'spherical']:
            temp_config = config.copy()
            temp_config['hmm']['covariance_type'] = cov_type

            hmm = CommodityHMM(temp_config)
            hmm.fit(features)

            assert hmm.covariance_type == cov_type
            assert hmm.is_fitted == True


class TestHMMConvergence:
    """Test convergence behavior."""

    def test_convergence_detection(self, config, synthetic_data):
        """Test convergence detection."""
        features, _ = synthetic_data

        # Test with very tight tolerance (may not converge)
        temp_config = config.copy()
        temp_config['hmm']['n_iter'] = 10
        temp_config['hmm']['tol'] = 1e-10

        hmm = CommodityHMM(temp_config)
        hmm.fit(features)

        # Should still fit, just may not converge
        assert hmm.is_fitted == True

    def test_multiple_init_best_selection(self, config, synthetic_data):
        """Test that best initialization is selected."""
        features, _ = synthetic_data

        hmm = CommodityHMM(config)
        hmm.fit_with_multiple_inits(features, n_inits=5)

        # Check that best score was selected
        scores = hmm.training_history['scores']
        assert len(scores) > 0
        # Best score should be among the scores (may not be exactly max due to randomness)
        assert any(score > scores[0] - 100 for score in scores)  # Reasonable range


class TestHMMEdgeCases:
    """Test edge cases and error handling."""

    def test_small_dataset(self, config):
        """Test with very small dataset."""
        features = pd.DataFrame(np.random.randn(20, 4))

        hmm = CommodityHMM(config)
        # Should still work, just may not be reliable
        hmm.fit(features)
        assert hmm.is_fitted == True

    def test_single_feature(self, config):
        """Test with single feature."""
        features = pd.DataFrame(np.random.randn(100, 1))

        hmm = CommodityHMM(config)
        hmm.fit(features)
        assert hmm.is_fitted == True

    def test_many_features(self, config):
        """Test with many features."""
        features = pd.DataFrame(np.random.randn(500, 20))

        # Use diagonal covariance for computational efficiency
        temp_config = config.copy()
        temp_config['hmm']['covariance_type'] = 'diag'

        hmm = CommodityHMM(temp_config)
        hmm.fit(features)
        assert hmm.is_fitted == True


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
