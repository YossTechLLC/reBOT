"""
TimesFM Volatility Forecasting Module
======================================
Wrapper around TimesFMAdapter specialized for forecasting next-day intraday volatility.

Key Differences from Price Forecasting:
- Input: Historical intraday_range_pct (not prices)
- Output: Next-day expected volatility (single value, not 30-day forecast)
- Frequency: Daily granularity
- Context: 60-90 days of historical volatility

Usage:
    forecaster = TimesFMVolatilityForecaster()
    volatility_forecast = forecaster.predict_next_day(spy_features)
"""

import logging
import os
import numpy as np
import pandas as pd
from typing import Optional
from .timesfm_adapter import TimesFMAdapter

logger = logging.getLogger(__name__)


class TimesFMVolatilityForecaster:
    """
    TimesFM forecaster specialized for next-day volatility prediction.

    Uses TimesFM foundation model to forecast tomorrow's expected
    intraday range based on historical volatility patterns.
    """

    def __init__(
        self,
        checkpoint: str = 'google/timesfm-1.0-200m-pytorch',
        context_length: int = 60,
        device: str = 'cpu'
    ):
        """
        Initialize TimesFM volatility forecaster.

        Args:
            checkpoint: HuggingFace checkpoint (default: TimesFM 1.0 200M PyTorch)
                        Note: 1.0 version provides torch_model.ckpt format required by timesfm library.
                        The 2.5 version uses safetensors which isn't fully supported yet.
            context_length: Days of historical volatility to use as context (default: 60)
            device: Device for inference ('cpu', 'cuda', or 'auto')
        """
        self.context_length = context_length
        self.device = device

        # Use default checkpoint if None provided
        if checkpoint is None:
            checkpoint = 'google/timesfm-1.0-200m-pytorch'

        # Create configuration for TimesFMAdapter
        config = {
            'timesfm': {
                'enabled': True,
                'checkpoint': checkpoint,
                'max_context': 512,  # TimesFM max
                'max_horizon': 256,   # TimesFM max
                'normalize_inputs': True,
                'use_continuous_quantile_head': False,  # We only need point forecast
                'force_flip_invariance': False,  # Volatility is always positive
                'infer_is_positive': True,  # Volatility is positive
                'fix_quantile_crossing': False
            }
        }

        # Initialize TimesFM adapter
        try:
            self.adapter = TimesFMAdapter(
                config=config,
                backend='torch',
                checkpoint=checkpoint,
                device=device
            )
            logger.info("TimesFM volatility forecaster initialized")
        except Exception as e:
            logger.warning(f"Failed to load TimesFM: {str(e)}")
            logger.warning("TimesFM forecasting will be unavailable")
            self.adapter = None

    def predict_next_day(
        self,
        df: pd.DataFrame,
        volatility_col: str = 'intraday_range_pct',
        use_full_history: bool = False
    ) -> Optional[float]:
        """
        Predict next-day intraday volatility.

        Args:
            df: DataFrame with historical data (must have volatility_col)
            volatility_col: Column name for intraday volatility (default: 'intraday_range_pct')
            use_full_history: If True, use all available history (up to max_context)
                              If False, use last context_length days only

        Returns:
            Expected next-day volatility (e.g., 0.012 = 1.2%)
            Returns None if TimesFM not available or forecast fails
        """
        if self.adapter is None or not self.adapter.enabled:
            logger.warning("TimesFM not available - cannot generate forecast")
            return None

        if volatility_col not in df.columns:
            logger.error(f"Column '{volatility_col}' not found in DataFrame")
            return None

        # Extract volatility series
        volatility_series = df[volatility_col]

        # Determine context length
        if use_full_history:
            context = volatility_series.values
        else:
            context = volatility_series.tail(self.context_length).values

        if len(context) < 10:
            logger.warning(f"Insufficient context: only {len(context)} days available")
            return None

        try:
            # Forecast next 1 day
            forecast_result = self.adapter.forecast(
                context=context,
                horizon=1,  # Predict only 1 day ahead
                freq='D'     # Daily frequency
            )

            next_day_volatility = forecast_result.point_forecast[0]

            # Validate forecast (volatility should be between 0.1% and 10%)
            if next_day_volatility < 0.001 or next_day_volatility > 0.10:
                logger.warning(
                    f"TimesFM forecast outside reasonable range: {next_day_volatility:.3f} "
                    f"(expected 0.001-0.10). Using median fallback."
                )
                # Fallback: use recent median
                next_day_volatility = np.median(context[-10:])

            logger.info(f"TimesFM forecast: {next_day_volatility:.3f} ({next_day_volatility*100:.2f}%)")
            return float(next_day_volatility)

        except Exception as e:
            logger.error(f"TimesFM forecast failed: {str(e)}")
            return None

    def batch_forecast(
        self,
        df: pd.DataFrame,
        volatility_col: str = 'intraday_range_pct',
        min_context: int = 30
    ) -> pd.Series:
        """
        Generate forecasts for multiple days (walk-forward).

        Args:
            df: DataFrame with historical data
            volatility_col: Column name for intraday volatility
            min_context: Minimum days of context before starting forecasts

        Returns:
            Series of forecasts aligned with df index
        """
        if self.adapter is None or not self.adapter.enabled:
            logger.warning("TimesFM not available - cannot generate forecasts")
            return pd.Series(index=df.index, dtype=float)

        forecasts = []

        for i in range(min_context, len(df)):
            # Use data up to (but not including) day i to forecast day i
            context_df = df.iloc[:i]

            forecast = self.predict_next_day(
                context_df,
                volatility_col=volatility_col,
                use_full_history=False  # Use last 60 days only
            )

            forecasts.append(forecast if forecast is not None else np.nan)

        # Pad with NaN for days without forecasts
        forecasts = [np.nan] * min_context + forecasts

        return pd.Series(forecasts, index=df.index, name='timesfm_forecast')

    def is_available(self) -> bool:
        """Check if TimesFM is loaded and available."""
        return self.adapter is not None and self.adapter.enabled

    def get_info(self) -> dict:
        """Get information about the forecaster."""
        if self.adapter is None:
            return {'available': False, 'reason': 'Adapter not initialized'}

        info = self.adapter.get_model_info()
        info['context_length'] = self.context_length
        info['device'] = self.device
        return info


def test_timesfm_volatility():
    """Test TimesFM volatility forecaster."""
    print("Testing TimesFM Volatility Forecaster...")
    print("=" * 60)

    # Create sample volatility data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')

    # Simulate volatility with regime changes
    volatility = np.concatenate([
        np.random.normal(0.006, 0.002, 40),  # Low vol regime
        np.random.normal(0.012, 0.003, 40),  # Normal vol regime
        np.random.normal(0.020, 0.005, 20),  # High vol regime
    ])

    df = pd.DataFrame({
        'intraday_range_pct': np.abs(volatility)
    }, index=dates)

    # Initialize forecaster
    print("\n1. Initializing TimesFM...")
    print("-" * 60)

    try:
        forecaster = TimesFMVolatilityForecaster(device='cpu')

        if not forecaster.is_available():
            print("   ⚠️  TimesFM not available (likely not installed)")
            print("   Install with: pip install timesfm")
            return False

        print("   ✅ TimesFM loaded")

        # Test next-day prediction
        print("\n2. Testing Next-Day Prediction...")
        print("-" * 60)

        next_day_vol = forecaster.predict_next_day(df)
        actual_vol = df['intraday_range_pct'].iloc[-1]

        print(f"   Last actual volatility: {actual_vol:.3f} ({actual_vol*100:.2f}%)")
        print(f"   Next-day forecast: {next_day_vol:.3f} ({next_day_vol*100:.2f}%)")

        # Test batch forecasting
        print("\n3. Testing Batch Forecasting...")
        print("-" * 60)

        forecasts = forecaster.batch_forecast(df, min_context=30)
        valid_forecasts = forecasts.dropna()

        print(f"   Generated {len(valid_forecasts)} forecasts")
        print(f"   Average forecast: {valid_forecasts.mean():.3f}")
        print(f"   Average actual: {df['intraday_range_pct'].mean():.3f}")

        # Calculate error
        test_df = df.iloc[30:].copy()
        test_df['forecast'] = forecasts.iloc[30:]
        test_df['error'] = np.abs(test_df['forecast'] - test_df['intraday_range_pct'])

        mae = test_df['error'].mean()
        print(f"   Mean Absolute Error: {mae:.4f}")

        # Get model info
        print("\n4. Model Information...")
        print("-" * 60)

        info = forecaster.get_info()
        print(f"   Checkpoint: {info['checkpoint']}")
        print(f"   Context length: {info['context_length']}")
        print(f"   Max context: {info['max_context']}")
        print(f"   Architecture: {info['architecture']['type']}")
        print(f"   Parameters: {info['architecture']['parameters']}")

        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        print("\nNote: TimesFM requires the timesfm package:")
        print("  pip install timesfm")
        return False


if __name__ == "__main__":
    test_timesfm_volatility()
