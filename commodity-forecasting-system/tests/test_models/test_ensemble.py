"""
Unit Tests for TimesFM-HMM Ensemble
Tests ensemble integration modes and forecast combination strategies.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.loader import load_config
from src.models.ensemble import TimesFMHMMEnsemble, EnsembleForecast
from src.models.timesfm_adapter import TimesFMAdapter
from src.models.hmm_core import CommodityHMM
from src.data.preprocessing import DataPreprocessor
from src.data.features import FeatureEngineer


@pytest.fixture
def config():
    """Load test configuration."""
    config_path = Path(__file__).parent.parent.parent / 'config' / 'parameters.toml'
    return load_config(str(config_path))


@pytest.fixture
def sample_data():
    """Generate sample data with features for ensemble."""
    np.random.seed(42)

    dates = pd.date_range('2023-01-01', periods=500, freq='D')
    close = 100 * np.exp(np.cumsum(np.random.randn(500) * 0.02))

    data = pd.DataFrame({
        'date': dates,
        'open': close * (1 + np.random.randn(500) * 0.01),
        'high': close * (1 + np.abs(np.random.randn(500)) * 0.015),
        'low': close * (1 - np.abs(np.random.randn(500)) * 0.015),
        'close': close,
        'volume': np.random.randint(1000000, 5000000, 500)
    }).set_index('date')

    return data


@pytest.fixture
def preprocessed_data(config, sample_data):
    """Preprocess data and engineer features."""
    preprocessor = DataPreprocessor(config)
    processed_data, _ = preprocessor.preprocess(sample_data)

    feature_engineer = FeatureEngineer(config)
    features = feature_engineer.engineer_features(processed_data)

    return features


class TestEnsembleInitialization:
    """Test ensemble initialization and configuration."""

    def test_ensemble_creation(self, config):
        """Test ensemble can be created."""
        try:
            ensemble = TimesFMHMMEnsemble(config, integration_mode='ensemble')

            assert ensemble.integration_mode == 'ensemble'
            assert ensemble.timesfm is not None
            assert ensemble.hmm is not None

        except (ImportError, RuntimeError):
            pytest.skip("TimesFM not available")

    def test_integration_mode_validation(self, config):
        """Test validation of integration mode."""
        # Valid modes
        for mode in ['ensemble', 'primary', 'regime_input']:
            try:
                ensemble = TimesFMHMMEnsemble(config, integration_mode=mode)
                assert ensemble.integration_mode == mode
            except (ImportError, RuntimeError):
                pytest.skip("TimesFM not available")

        # Invalid mode
        with pytest.raises(ValueError):
            TimesFMHMMEnsemble(config, integration_mode='invalid_mode')

    def test_ensemble_weights_initialization(self, config):
        """Test ensemble weights are initialized correctly."""
        try:
            ensemble = TimesFMHMMEnsemble(config)

            # Default weights should be 0.5/0.5
            assert ensemble.ensemble_weights['timesfm'] == 0.5
            assert ensemble.ensemble_weights['hmm'] == 0.5

        except (ImportError, RuntimeError):
            pytest.skip("TimesFM not available")

    def test_set_ensemble_weights(self, config):
        """Test setting custom ensemble weights."""
        try:
            ensemble = TimesFMHMMEnsemble(config)

            # Set custom weights
            ensemble.set_ensemble_weights(timesfm_weight=0.7, hmm_weight=0.3)

            assert ensemble.ensemble_weights['timesfm'] == 0.7
            assert ensemble.ensemble_weights['hmm'] == 0.3

            # Invalid weights (don't sum to 1)
            with pytest.raises(ValueError):
                ensemble.set_ensemble_weights(timesfm_weight=0.6, hmm_weight=0.5)

        except (ImportError, RuntimeError):
            pytest.skip("TimesFM not available")

    def test_ensemble_info(self, config):
        """Test get_ensemble_info returns correct structure."""
        try:
            ensemble = TimesFMHMMEnsemble(config)
            info = ensemble.get_ensemble_info()

            assert 'integration_mode' in info
            assert 'ensemble_weights' in info
            assert 'timesfm_enabled' in info
            assert 'hmm_fitted' in info
            assert 'timesfm_info' in info

        except (ImportError, RuntimeError):
            pytest.skip("TimesFM not available")


class TestEnsembleModeForecasting:
    """Test forecasting in ensemble mode."""

    @pytest.mark.skipif(
        not pytest.importorskip("timesfm", reason="timesfm not installed"),
        reason="Requires timesfm package"
    )
    def test_ensemble_mode_forecast(self, config, preprocessed_data):
        """Test ensemble mode: averaging TimesFM and HMM."""
        ensemble = TimesFMHMMEnsemble(config, integration_mode='ensemble')

        # Train HMM
        hmm_features = preprocessed_data[['returns', 'volatility_5']].dropna()
        ensemble.fit_hmm(hmm_features)

        # Generate forecast
        context = preprocessed_data.iloc[:400]
        horizon = 30

        result = ensemble.forecast(
            context=context,
            horizon=horizon,
            freq='D'
        )

        # Validate result
        assert isinstance(result, EnsembleForecast)
        assert len(result.point_forecast) == horizon
        assert result.ensemble_mode == 'ensemble'

        # Should have both TimesFM and HMM forecasts
        assert result.timesfm_forecast is not None
        assert result.hmm_forecast is not None

        # Ensemble should be weighted average
        expected_ensemble = (
            0.5 * result.timesfm_forecast + 0.5 * result.hmm_forecast
        )
        np.testing.assert_array_almost_equal(
            result.point_forecast,
            expected_ensemble,
            decimal=5
        )

        # Should have regime information
        assert result.regime_state is not None
        assert result.regime_label in ['bull', 'bear', 'neutral']
        assert result.regime_volatility is not None

    @pytest.mark.skipif(
        not pytest.importorskip("timesfm", reason="timesfm not installed"),
        reason="Requires timesfm package"
    )
    def test_primary_mode_forecast(self, config, preprocessed_data):
        """Test primary mode: TimesFM primary, HMM for regime only."""
        ensemble = TimesFMHMMEnsemble(config, integration_mode='primary')

        # Can work without HMM fitting in primary mode
        context = preprocessed_data.iloc[:400]
        horizon = 30

        result = ensemble.forecast(
            context=context,
            horizon=horizon,
            freq='D'
        )

        # Validate result
        assert isinstance(result, EnsembleForecast)
        assert result.ensemble_mode == 'primary'

        # Should use TimesFM forecast as primary
        assert result.timesfm_forecast is not None
        np.testing.assert_array_equal(
            result.point_forecast,
            result.timesfm_forecast
        )

        # HMM forecast not used
        assert result.hmm_forecast is None

        # Weights should reflect primary mode
        assert result.weights['timesfm'] == 1.0
        assert result.weights['hmm'] == 0.0

    @pytest.mark.skipif(
        not pytest.importorskip("timesfm", reason="timesfm not installed"),
        reason="Requires timesfm package"
    )
    def test_regime_input_mode_forecast(self, config, preprocessed_data):
        """Test regime_input mode: Feed HMM regime to TimesFM."""
        ensemble = TimesFMHMMEnsemble(config, integration_mode='regime_input')

        # Train HMM for regime detection
        hmm_features = preprocessed_data[['returns', 'volatility_5']].dropna()
        ensemble.fit_hmm(hmm_features)

        # Generate forecast
        context = preprocessed_data.iloc[:400]
        horizon = 30

        result = ensemble.forecast(
            context=context,
            horizon=horizon,
            freq='D'
        )

        # Validate result
        assert isinstance(result, EnsembleForecast)
        assert result.ensemble_mode == 'regime_input'

        # Should use regime-conditioned TimesFM
        assert result.timesfm_forecast is not None
        assert result.regime_state is not None

        # HMM forecast not directly used
        assert result.hmm_forecast is None


class TestEnsembleIntegration:
    """Test end-to-end ensemble integration."""

    @pytest.mark.skipif(
        not pytest.importorskip("timesfm", reason="timesfm not installed"),
        reason="Requires timesfm package"
    )
    def test_full_pipeline(self, config, sample_data):
        """Test full pipeline from raw data to ensemble forecast."""
        # 1. Preprocess
        preprocessor = DataPreprocessor(config)
        processed_data, _ = preprocessor.preprocess(sample_data)

        # 2. Feature engineering
        feature_engineer = FeatureEngineer(config)
        features = feature_engineer.engineer_features(processed_data)

        # 3. Create ensemble
        ensemble = TimesFMHMMEnsemble(config, integration_mode='ensemble')

        # 4. Train HMM
        hmm_features = features[['returns', 'volatility_5']].dropna()
        ensemble.fit_hmm(hmm_features)

        # 5. Generate forecast
        context = features.iloc[:400]
        result = ensemble.forecast(context=context, horizon=30, freq='D')

        # Validate complete forecast
        assert result.point_forecast is not None
        assert len(result.point_forecast) == 30
        assert result.regime_label in ['bull', 'bear', 'neutral']

    @pytest.mark.skipif(
        not pytest.importorskip("timesfm", reason="timesfm not installed"),
        reason="Requires timesfm package"
    )
    def test_ensemble_weight_impact(self, config, preprocessed_data):
        """Test that changing ensemble weights affects forecast."""
        ensemble = TimesFMHMMEnsemble(config, integration_mode='ensemble')

        # Train HMM
        hmm_features = preprocessed_data[['returns', 'volatility_5']].dropna()
        ensemble.fit_hmm(hmm_features)

        context = preprocessed_data.iloc[:400]

        # Forecast with default weights (0.5/0.5)
        result_balanced = ensemble.forecast(context=context, horizon=30, freq='D')

        # Change to TimesFM-heavy weights
        ensemble.set_ensemble_weights(timesfm_weight=0.9, hmm_weight=0.1)
        result_timesfm_heavy = ensemble.forecast(context=context, horizon=30, freq='D')

        # Forecasts should be different
        assert not np.allclose(
            result_balanced.point_forecast,
            result_timesfm_heavy.point_forecast
        )

        # TimesFM-heavy should be closer to TimesFM component
        distance_to_timesfm = np.mean(
            np.abs(result_timesfm_heavy.point_forecast - result_timesfm_heavy.timesfm_forecast)
        )
        distance_to_hmm = np.mean(
            np.abs(result_timesfm_heavy.point_forecast - result_timesfm_heavy.hmm_forecast)
        )

        assert distance_to_timesfm < distance_to_hmm

    def test_hmm_fitting_required(self, config, preprocessed_data):
        """Test that HMM must be fitted for ensemble mode."""
        try:
            ensemble = TimesFMHMMEnsemble(config, integration_mode='ensemble')

            context = preprocessed_data.iloc[:400]

            # Should raise error if HMM not fitted
            with pytest.raises(ValueError):
                ensemble.forecast(context=context, horizon=30, freq='D')

        except (ImportError, RuntimeError):
            pytest.skip("TimesFM not available")

    @pytest.mark.skipif(
        not pytest.importorskip("timesfm", reason="timesfm not installed"),
        reason="Requires timesfm package"
    )
    def test_frequency_inference(self, config, preprocessed_data):
        """Test that frequency is inferred correctly."""
        ensemble = TimesFMHMMEnsemble(config, integration_mode='primary')

        context = preprocessed_data.iloc[:400]

        # Forecast without specifying frequency
        result = ensemble.forecast(context=context, horizon=30, freq=None)

        assert result.point_forecast is not None

    @pytest.mark.skipif(
        not pytest.importorskip("timesfm", reason="timesfm not installed"),
        reason="Requires timesfm package"
    )
    def test_regime_detection(self, config, preprocessed_data):
        """Test regime detection in ensemble."""
        ensemble = TimesFMHMMEnsemble(config, integration_mode='ensemble')

        # Train HMM
        hmm_features = preprocessed_data[['returns', 'volatility_5']].dropna()
        ensemble.fit_hmm(hmm_features)

        # Generate multiple forecasts at different time points
        regimes = []
        for start_idx in [100, 200, 300, 400]:
            context = preprocessed_data.iloc[:start_idx]
            result = ensemble.forecast(context=context, horizon=10, freq='D')
            regimes.append(result.regime_label)

        # Should detect at least 2 different regimes in 500 days
        unique_regimes = set(regimes)
        assert len(unique_regimes) >= 2

        # All should be valid regime labels
        assert all(r in ['bull', 'bear', 'neutral'] for r in regimes)


class TestEnsembleRobustness:
    """Test ensemble robustness and edge cases."""

    def test_missing_close_column(self, config):
        """Test handling of missing 'close' column."""
        try:
            ensemble = TimesFMHMMEnsemble(config)

            # Create data without 'close' column
            bad_data = pd.DataFrame({
                'price': [1, 2, 3, 4, 5]
            })

            with pytest.raises(ValueError):
                ensemble.forecast(context=bad_data, horizon=5, freq='D')

        except (ImportError, RuntimeError):
            pytest.skip("TimesFM not available")

    @pytest.mark.skipif(
        not pytest.importorskip("timesfm", reason="timesfm not installed"),
        reason="Requires timesfm package"
    )
    def test_short_context(self, config):
        """Test handling of very short context."""
        ensemble = TimesFMHMMEnsemble(config, integration_mode='primary')

        # Very short context (10 days)
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        short_data = pd.DataFrame({
            'close': np.random.randn(10) + 100
        }, index=dates)

        # Should still work (may have reduced accuracy)
        result = ensemble.forecast(context=short_data, horizon=5, freq='D')

        assert result.point_forecast is not None
        assert len(result.point_forecast) == 5


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
