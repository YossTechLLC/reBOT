"""
Integration Tests for HMM Pipeline
End-to-end tests from data acquisition to forecasting.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.loader import load_config
from src.data.acquisition import CommodityDataAcquisition
from src.data.preprocessing import DataPreprocessor
from src.data.features import FeatureEngineer
from src.models.hmm_core import CommodityHMM
from src.models.hmm_selection import HMMModelSelector
from src.models.regime_analysis import RegimeAnalyzer


@pytest.fixture
def config():
    """Load test configuration."""
    config_path = Path(__file__).parent.parent.parent / 'config' / 'parameters.toml'
    return load_config(str(config_path))


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)

    dates = pd.date_range('2023-01-01', periods=300, freq='D')
    close = 100 * np.exp(np.cumsum(np.random.randn(300) * 0.02))

    data = pd.DataFrame({
        'date': dates,
        'open': close * (1 + np.random.randn(300) * 0.01),
        'high': close * (1 + np.abs(np.random.randn(300)) * 0.015),
        'low': close * (1 - np.abs(np.random.randn(300)) * 0.015),
        'close': close,
        'volume': np.random.randint(1000000, 5000000, 300)
    })

    return data


class TestHMMIntegration:
    """Integration tests for full HMM pipeline."""

    def test_preprocessing_to_hmm(self, config, sample_data):
        """Test preprocessing -> HMM training pipeline."""
        # Preprocess
        preprocessor = DataPreprocessor(config)
        processed_data, metadata = preprocessor.preprocess(sample_data)

        assert 'returns' in processed_data.columns
        assert len(processed_data) < len(sample_data)  # Some rows removed

        # Engineer features
        feature_engineer = FeatureEngineer(config)
        features = feature_engineer.engineer_features(processed_data)

        assert features.shape[1] > processed_data.shape[1]  # More features added

        # Train HMM
        feature_cols = ['returns', 'volatility_5']
        if 'rsi_14' in features.columns:
            feature_cols.append('rsi_14')

        hmm_features = features[feature_cols].dropna()
        assert len(hmm_features) > 0, "No features remaining after dropna"

        hmm = CommodityHMM(config)
        hmm.fit(hmm_features)

        assert hmm.is_fitted == True

    def test_full_forecast_pipeline(self, config, sample_data):
        """Test full pipeline from data to forecast."""
        # Preprocess
        preprocessor = DataPreprocessor(config)
        processed_data, _ = preprocessor.preprocess(sample_data)

        # Feature engineering
        feature_engineer = FeatureEngineer(config)
        features = feature_engineer.engineer_features(processed_data)

        # Select features
        feature_cols = ['returns', 'volatility_5']
        hmm_features = features[feature_cols].dropna()

        # Train HMM
        hmm = CommodityHMM(config)
        hmm.fit(hmm_features)

        # Forecast
        current_features = hmm_features.iloc[-30:]
        current_price = sample_data['close'].iloc[-1]

        forecast = hmm.forecast_spot_price(
            current_features,
            horizon=10,
            n_simulations=1000,
            current_price=current_price
        )

        # Validate forecast
        assert forecast['mean'] > 0
        assert forecast['regime'] in ['bull', 'bear', 'neutral']
        assert 0 <= forecast['regime_persistence'] <= 1

    def test_model_selection_pipeline(self, config, sample_data):
        """Test model selection integration."""
        # Preprocess
        preprocessor = DataPreprocessor(config)
        processed_data, _ = preprocessor.preprocess(sample_data)

        # Feature engineering
        feature_engineer = FeatureEngineer(config)
        features = feature_engineer.engineer_features(processed_data)

        # Select features
        feature_cols = ['returns', 'volatility_5']
        hmm_features = features[feature_cols].dropna()

        # Model selection
        selector = HMMModelSelector(config)
        best_model, best_n_states, results_df = selector.select_optimal_states(
            hmm_features,
            n_states_range=range(2, 4),  # Smaller range for speed
            n_inits=2,
            criterion='bic'
        )

        # Validate
        assert best_model.is_fitted == True
        assert 2 <= best_n_states <= 3
        assert len(results_df) == 2  # 2 models tested

    def test_regime_analysis_pipeline(self, config, sample_data):
        """Test regime analysis integration."""
        # Preprocess
        preprocessor = DataPreprocessor(config)
        processed_data, _ = preprocessor.preprocess(sample_data)

        # Feature engineering
        feature_engineer = FeatureEngineer(config)
        features = feature_engineer.engineer_features(processed_data)

        # Select features
        feature_cols = ['returns', 'volatility_5']
        hmm_features = features[feature_cols].dropna()

        # Train HMM
        hmm = CommodityHMM(config)
        hmm.fit(hmm_features)

        # Regime analysis
        analyzer = RegimeAnalyzer(hmm)

        # Test transition analysis
        trans_analysis = analyzer.analyze_transitions()
        assert 'persistence' in trans_analysis
        assert 'expected_duration' in trans_analysis
        assert 'steady_state' in trans_analysis

        # Test report generation
        report = analyzer.generate_report()
        assert len(report) > 0
        assert 'REGIME ANALYSIS REPORT' in report

    def test_save_load_retrain(self, config, sample_data):
        """Test save/load and retraining."""
        import tempfile

        # Preprocess
        preprocessor = DataPreprocessor(config)
        processed_data, _ = preprocessor.preprocess(sample_data)

        # Feature engineering
        feature_engineer = FeatureEngineer(config)
        features = feature_engineer.engineer_features(processed_data)

        # Select features
        feature_cols = ['returns', 'volatility_5']
        hmm_features = features[feature_cols].dropna()

        # Train original model
        hmm_original = CommodityHMM(config)
        hmm_original.fit(hmm_features)

        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_model.pkl'
            hmm_original.save_model(save_path)

            # Load
            hmm_loaded = CommodityHMM.load_model(save_path)

            # Compare predictions
            test_features = hmm_features.iloc[-10:]
            state_orig, label_orig = hmm_original.predict_regime(test_features)
            state_loaded, label_loaded = hmm_loaded.predict_regime(test_features)

            assert state_orig == state_loaded
            assert label_orig == label_loaded

    def test_regime_consistency(self, config, sample_data):
        """Test that regime predictions are consistent."""
        # Preprocess
        preprocessor = DataPreprocessor(config)
        processed_data, _ = preprocessor.preprocess(sample_data)

        # Feature engineering
        feature_engineer = FeatureEngineer(config)
        features = feature_engineer.engineer_features(processed_data)

        # Select features
        feature_cols = ['returns', 'volatility_5']
        hmm_features = features[feature_cols].dropna()

        # Train HMM
        hmm = CommodityHMM(config)
        hmm.fit(hmm_features)

        # Predict on same data twice
        state1, label1 = hmm.predict_regime(hmm_features.iloc[-1:])
        state2, label2 = hmm.predict_regime(hmm_features.iloc[-1:])

        # Should be identical
        assert state1 == state2
        assert label1 == label2

    def test_forecast_uncertainty_quantification(self, config, sample_data):
        """Test that forecast provides proper uncertainty quantification."""
        # Preprocess
        preprocessor = DataPreprocessor(config)
        processed_data, _ = preprocessor.preprocess(sample_data)

        # Feature engineering
        feature_engineer = FeatureEngineer(config)
        features = feature_engineer.engineer_features(processed_data)

        # Select features
        feature_cols = ['returns', 'volatility_5']
        hmm_features = features[feature_cols].dropna()

        # Train HMM
        hmm = CommodityHMM(config)
        hmm.fit(hmm_features)

        # Forecast
        current_features = hmm_features.iloc[-30:]
        current_price = sample_data['close'].iloc[-1]

        forecast = hmm.forecast_spot_price(
            current_features,
            horizon=30,
            n_simulations=10000,
            current_price=current_price
        )

        # Validate uncertainty quantification
        # Mean should be close to median for symmetric distributions
        mean_median_ratio = forecast['mean'] / forecast['median']
        assert 0.9 < mean_median_ratio < 1.1, "Mean and median should be close"

        # Quantiles should be ordered
        quantiles = forecast['quantiles']
        assert quantiles['5%'] < quantiles['25%']
        assert quantiles['25%'] < quantiles['50%']
        assert quantiles['50%'] < quantiles['75%']
        assert quantiles['75%'] < quantiles['95%']

        # Standard deviation should be positive and reasonable
        assert forecast['std'] > 0
        assert forecast['std'] < current_price  # Std shouldn't exceed price


class TestHMMRobustness:
    """Test robustness and edge cases."""

    def test_insufficient_data_warning(self, config):
        """Test behavior with very small dataset."""
        # Create minimal dataset
        data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=30, freq='D'),
            'open': np.random.randn(30) + 100,
            'high': np.random.randn(30) + 101,
            'low': np.random.randn(30) + 99,
            'close': np.random.randn(30) + 100,
            'volume': np.random.randint(1000, 5000, 30)
        })

        # Should still work but may not be reliable
        preprocessor = DataPreprocessor(config)
        processed_data, _ = preprocessor.preprocess(data)

        feature_engineer = FeatureEngineer(config)
        features = feature_engineer.engineer_features(processed_data)

        feature_cols = ['returns']
        hmm_features = features[feature_cols].dropna()

        if len(hmm_features) > 10:  # Only if enough data remains
            hmm = CommodityHMM(config)
            hmm.fit(hmm_features)
            assert hmm.is_fitted == True

    def test_high_volatility_regime(self, config):
        """Test with high volatility data."""
        # Generate high volatility data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        close = 100 * np.exp(np.cumsum(np.random.randn(200) * 0.05))  # High volatility

        data = pd.DataFrame({
            'date': dates,
            'open': close,
            'high': close * 1.02,
            'low': close * 0.98,
            'close': close,
            'volume': np.random.randint(1000000, 5000000, 200)
        })

        # Process and train
        preprocessor = DataPreprocessor(config)
        processed_data, _ = preprocessor.preprocess(data)

        feature_engineer = FeatureEngineer(config)
        features = feature_engineer.engineer_features(processed_data)

        feature_cols = ['returns', 'volatility_5']
        hmm_features = features[feature_cols].dropna()

        hmm = CommodityHMM(config)
        hmm.fit(hmm_features)

        # Check that high volatility is captured in regime stats
        max_vol = max(stats['volatility'] for stats in hmm.regime_stats.values())
        assert max_vol > 0.01  # Should detect high volatility


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
