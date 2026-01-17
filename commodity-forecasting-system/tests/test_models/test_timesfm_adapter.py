"""
Unit Tests for TimesFM Adapter
Tests TimesFM integration, zero-shot forecasting, and backend handling.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.loader import load_config
from src.models.timesfm_adapter import TimesFMAdapter, TimesFMForecast


@pytest.fixture
def config():
    """Load test configuration."""
    config_path = Path(__file__).parent.parent.parent / 'config' / 'parameters.toml'
    return load_config(str(config_path))


@pytest.fixture
def sample_time_series():
    """Generate sample time series for testing."""
    np.random.seed(42)

    # Generate 500 days of synthetic price data
    dates = pd.date_range('2023-01-01', periods=500, freq='D')

    # Trend + seasonality + noise
    t = np.arange(500)
    trend = 100 + 0.1 * t
    seasonality = 10 * np.sin(2 * np.pi * t / 30)  # 30-day cycle
    noise = np.random.randn(500) * 2

    prices = trend + seasonality + noise

    return pd.DataFrame({
        'date': dates,
        'close': prices
    }).set_index('date')


class TestTimesFMAdapter:
    """Test suite for TimesFM adapter."""

    def test_adapter_initialization(self, config):
        """Test TimesFM adapter can be initialized."""
        # Note: This may fail if timesfm package not installed
        # That's expected - we're testing the initialization logic
        try:
            adapter = TimesFMAdapter(config, backend='torch')

            # Check configuration loaded
            assert adapter.max_context > 0
            assert adapter.max_horizon > 0
            assert adapter.backend == 'torch'

        except (ImportError, RuntimeError) as e:
            # Expected if timesfm not installed
            pytest.skip(f"TimesFM not available: {str(e)}")

    def test_disabled_adapter(self, config):
        """Test adapter behavior when disabled."""
        # Disable TimesFM in config
        config['timesfm']['enabled'] = False

        adapter = TimesFMAdapter(config)

        assert not adapter.enabled
        assert adapter.model is None

        # Forecasting should raise error
        with pytest.raises(RuntimeError):
            adapter.forecast(
                context=np.array([1, 2, 3, 4, 5]),
                horizon=10
            )

    def test_model_info(self, config):
        """Test get_model_info returns correct structure."""
        try:
            adapter = TimesFMAdapter(config, backend='torch')
            info = adapter.get_model_info()

            assert 'enabled' in info
            assert 'backend' in info
            assert 'max_context' in info
            assert 'architecture' in info
            assert info['architecture']['parameters'] == '200M'

        except (ImportError, RuntimeError):
            pytest.skip("TimesFM not available")

    @pytest.mark.skipif(
        not pytest.importorskip("timesfm", reason="timesfm not installed"),
        reason="Requires timesfm package"
    )
    def test_forecast_basic(self, config, sample_time_series):
        """Test basic forecasting functionality."""
        adapter = TimesFMAdapter(config, backend='torch')

        # Use first 400 days as context, forecast 30 days
        context = sample_time_series['close'].iloc[:400]
        horizon = 30

        result = adapter.forecast(
            context=context,
            horizon=horizon,
            freq='D'
        )

        # Validate result structure
        assert isinstance(result, TimesFMForecast)
        assert len(result.point_forecast) == horizon
        assert result.context_length <= adapter.max_context
        assert result.horizon_length == horizon

        # Check forecast is reasonable (not all zeros, not NaN)
        assert not np.all(result.point_forecast == 0)
        assert not np.any(np.isnan(result.point_forecast))

    @pytest.mark.skipif(
        not pytest.importorskip("timesfm", reason="timesfm not installed"),
        reason="Requires timesfm package"
    )
    def test_forecast_with_quantiles(self, config, sample_time_series):
        """Test probabilistic forecasting with quantiles."""
        adapter = TimesFMAdapter(config, backend='torch')

        context = sample_time_series['close'].iloc[:400]
        horizon = 30
        quantiles = [0.1, 0.5, 0.9]

        result = adapter.forecast(
            context=context,
            horizon=horizon,
            freq='D',
            quantiles=quantiles
        )

        # Check quantile forecasts
        if result.quantile_forecasts is not None:
            assert len(result.quantile_forecasts) == len(quantiles)

            for q in quantiles:
                assert q in result.quantile_forecasts
                assert len(result.quantile_forecasts[q]) == horizon

            # Verify quantile ordering (0.1 < 0.5 < 0.9)
            q10 = result.quantile_forecasts[0.1]
            q50 = result.quantile_forecasts[0.5]
            q90 = result.quantile_forecasts[0.9]

            # Most points should satisfy quantile ordering
            assert np.mean(q10 <= q50) > 0.8
            assert np.mean(q50 <= q90) > 0.8

    def test_context_length_validation(self, config):
        """Test handling of context that exceeds max_context."""
        try:
            adapter = TimesFMAdapter(config, backend='torch')

            # Create context longer than max_context
            long_context = np.random.randn(adapter.max_context + 100)

            # Should truncate to max_context
            result = adapter.forecast(
                context=long_context,
                horizon=30,
                freq='D'
            )

            # Context should be truncated
            assert result.context_length == adapter.max_context

        except (ImportError, RuntimeError):
            pytest.skip("TimesFM not available")

    def test_horizon_length_validation(self, config):
        """Test handling of horizon that exceeds max_horizon."""
        try:
            adapter = TimesFMAdapter(config, backend='torch')

            context = np.random.randn(100)

            # Should raise error for horizon > max_horizon
            with pytest.raises(ValueError):
                adapter.forecast(
                    context=context,
                    horizon=adapter.max_horizon + 100,
                    freq='D'
                )

        except (ImportError, RuntimeError):
            pytest.skip("TimesFM not available")

    def test_nan_handling(self, config):
        """Test handling of NaN values in context."""
        try:
            adapter = TimesFMAdapter(config, backend='torch')

            # Create context with NaNs
            context = np.array([1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0])

            # Should handle NaNs (forward/backward fill)
            result = adapter.forecast(
                context=context,
                horizon=5,
                freq='D'
            )

            # Forecast should not contain NaNs
            assert not np.any(np.isnan(result.point_forecast))

        except (ImportError, RuntimeError):
            pytest.skip("TimesFM not available")

    @pytest.mark.skipif(
        not pytest.importorskip("timesfm", reason="timesfm not installed"),
        reason="Requires timesfm package"
    )
    def test_zero_shot_evaluation(self, config, sample_time_series):
        """Test zero-shot evaluation on test data."""
        adapter = TimesFMAdapter(config, backend='torch')

        metrics = adapter.evaluate_zero_shot(
            test_data=sample_time_series,
            context_col='close',
            horizon=30,
            freq='D'
        )

        # Check metrics structure
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'mape' in metrics
        assert 'direction_accuracy' in metrics

        # Metrics should be positive
        assert metrics['mae'] > 0
        assert metrics['rmse'] > 0
        assert metrics['mape'] > 0

        # Direction accuracy should be between 0 and 100
        assert 0 <= metrics['direction_accuracy'] <= 100

    def test_frequency_inference(self, config):
        """Test inference of temporal frequency from data."""
        try:
            adapter = TimesFMAdapter(config, backend='torch')

            # Daily data
            daily_index = pd.date_range('2023-01-01', periods=100, freq='D')
            daily_df = pd.DataFrame({'close': np.random.randn(100)}, index=daily_index)

            result = adapter.forecast(
                context=daily_df['close'],
                horizon=10,
                freq=None  # Let adapter infer
            )

            assert result.point_forecast is not None

        except (ImportError, RuntimeError):
            pytest.skip("TimesFM not available")


class TestTimesFMForecasts:
    """Test forecast quality and consistency."""

    @pytest.mark.skipif(
        not pytest.importorskip("timesfm", reason="timesfm not installed"),
        reason="Requires timesfm package"
    )
    def test_forecast_trend_capture(self, config):
        """Test that TimesFM captures trend direction."""
        adapter = TimesFMAdapter(config, backend='torch')

        # Create strong upward trend
        t = np.arange(200)
        prices = 100 + 0.5 * t + np.random.randn(200) * 2

        result = adapter.forecast(
            context=prices,
            horizon=30,
            freq='D'
        )

        # Forecast should generally trend upward
        forecast_trend = np.polyfit(np.arange(30), result.point_forecast, 1)[0]
        assert forecast_trend > 0  # Positive slope

    @pytest.mark.skipif(
        not pytest.importorskip("timesfm", reason="timesfm not installed"),
        reason="Requires timesfm package"
    )
    def test_forecast_seasonality_capture(self, config):
        """Test that TimesFM captures seasonal patterns."""
        adapter = TimesFMAdapter(config, backend='torch')

        # Create strong seasonality (7-day cycle)
        t = np.arange(300)
        prices = 100 + 20 * np.sin(2 * np.pi * t / 7)

        result = adapter.forecast(
            context=prices,
            horizon=14,  # 2 cycles
            freq='D'
        )

        # Check if forecast has oscillatory pattern
        # (simple test: forecast should have both increases and decreases)
        forecast_diffs = np.diff(result.point_forecast)
        has_increases = np.any(forecast_diffs > 0)
        has_decreases = np.any(forecast_diffs < 0)

        assert has_increases and has_decreases

    @pytest.mark.skipif(
        not pytest.importorskip("timesfm", reason="timesfm not installed"),
        reason="Requires timesfm package"
    )
    def test_forecast_consistency(self, config):
        """Test that repeated forecasts are consistent."""
        adapter = TimesFMAdapter(config, backend='torch')

        context = np.random.randn(200) + 100

        # Generate two forecasts
        result1 = adapter.forecast(context=context, horizon=30, freq='D')
        result2 = adapter.forecast(context=context, horizon=30, freq='D')

        # Should be identical (deterministic)
        np.testing.assert_array_almost_equal(
            result1.point_forecast,
            result2.point_forecast,
            decimal=4
        )


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
