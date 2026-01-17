#!/usr/bin/env python
"""
S&P 500 Forecast Test Script
Tests HMM + TimesFM ensemble forecasting on S&P 500 data
"""

import sys
import logging
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.loader import load_config
from src.models.hmm_core import CommodityHMM
from src.models.timesfm_adapter import TimesFMAdapter
from src.models.ensemble import TimesFMHMMEnsemble

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


def fetch_sp500_data(start_date='2015-01-01', end_date=None):
    """
    Fetch S&P 500 data from yfinance.

    Args:
        start_date: Start date for historical data
        end_date: End date (default: today)

    Returns:
        pd.DataFrame with OHLCV data
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    logger.info(f"📊 Fetching S&P 500 data from {start_date} to {end_date}")

    # Download data
    ticker = yf.Ticker("^GSPC")
    data = ticker.history(start=start_date, end=end_date)

    if data.empty:
        raise ValueError("No data retrieved from yfinance")

    # Rename columns to lowercase
    data.columns = [col.lower() for col in data.columns]

    logger.info(f"✅ Retrieved {len(data)} trading days of data")
    logger.info(f"📈 Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")

    return data


def prepare_features(data):
    """
    Prepare features for HMM training.

    Args:
        data: DataFrame with OHLCV data

    Returns:
        pd.DataFrame with engineered features
    """
    logger.info("🔧 Engineering features for HMM...")

    features = pd.DataFrame(index=data.index)

    # Returns
    features['returns'] = data['close'].pct_change()
    features['log_returns'] = np.log(data['close'] / data['close'].shift(1))

    # Volatility (rolling standard deviation)
    features['volatility_5'] = features['returns'].rolling(5).std()
    features['volatility_10'] = features['returns'].rolling(10).std()
    features['volatility_20'] = features['returns'].rolling(20).std()

    # Volume features
    features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()

    # Price momentum
    features['momentum_5'] = data['close'] / data['close'].shift(5) - 1
    features['momentum_10'] = data['close'] / data['close'].shift(10) - 1

    # Moving averages
    features['sma_20'] = data['close'].rolling(20).mean()
    features['sma_50'] = data['close'].rolling(50).mean()
    features['ma_ratio'] = features['sma_20'] / features['sma_50'] - 1

    # Range features (high-low normalized by close)
    features['range'] = (data['high'] - data['low']) / data['close']

    # Drop NaN values
    features_clean = features.dropna()

    logger.info(f"✅ Created {len(features_clean.columns)} features")
    logger.info(f"📊 Feature matrix shape: {features_clean.shape}")
    logger.info(f"🔢 Features: {list(features_clean.columns)}")

    return features_clean


def main():
    """Main execution function."""

    print("=" * 80)
    print("🚀 S&P 500 FORECAST TEST - HMM + TimesFM Ensemble")
    print("=" * 80)
    print()

    # 1. Load configuration
    logger.info("📋 Loading configuration...")
    config = load_config('config/parameters.toml')
    logger.info(f"✅ Configuration loaded")
    logger.info(f"   - TimesFM enabled: {config['timesfm']['enabled']}")
    logger.info(f"   - Integration mode: {config['timesfm']['integration_mode']}")
    logger.info(f"   - HMM states: {config['hmm']['n_states']}")
    print()

    # 2. Fetch S&P 500 data
    logger.info("📊 STEP 1: Fetching S&P 500 Data")
    logger.info("-" * 80)

    # Fetch maximum historical data (10 years)
    start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
    data = fetch_sp500_data(start_date=start_date)

    print()
    print(f"📈 S&P 500 Data Summary:")
    print(f"   - Period: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    print(f"   - Trading days: {len(data)}")
    print(f"   - Latest close: ${data['close'].iloc[-1]:.2f}")
    print(f"   - YTD return: {((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100:.2f}%")
    print()

    # 3. Prepare features
    logger.info("🔧 STEP 2: Feature Engineering")
    logger.info("-" * 80)
    features = prepare_features(data)
    print()

    # 4. Train HMM model
    logger.info("🤖 STEP 3: Training HMM Model")
    logger.info("-" * 80)

    # Use only features for HMM training
    hmm_features = features[['returns', 'volatility_5', 'volatility_10',
                              'momentum_5', 'range']].copy()

    logger.info(f"Training HMM on {len(hmm_features)} samples with {len(hmm_features.columns)} features...")

    hmm_model = CommodityHMM(config)
    hmm_model.fit_with_multiple_inits(hmm_features, n_inits=5)

    logger.info("✅ HMM model trained successfully!")

    # Display regime statistics
    print()
    print("📊 Detected Market Regimes:")
    print("-" * 80)
    regime_stats = hmm_model.get_regime_stats()
    for state_id, stats in regime_stats.items():
        print(f"\n🎯 Regime {state_id}: {stats['label'].upper()}")
        print(f"   - Mean return: {stats['mean_return']*100:.3f}% per day")
        print(f"   - Volatility: {stats['volatility']*100:.3f}%")
        print(f"   - Sharpe ratio: {stats['sharpe']:.3f}")
        print(f"   - Persistence: {stats['persistence']*100:.1f}%")
        print(f"   - Days in regime: {stats['count']}")

    # Show transition matrix
    print()
    print("🔄 Regime Transition Matrix:")
    print("-" * 80)
    trans_matrix = hmm_model.get_transition_matrix()
    trans_df = pd.DataFrame(
        trans_matrix,
        columns=[f"To {regime_stats[i]['label']}" for i in range(len(regime_stats))],
        index=[f"From {regime_stats[i]['label']}" for i in range(len(regime_stats))]
    )
    print(trans_df.round(3))
    print()

    # 5. Initialize TimesFM + HMM Ensemble
    logger.info("🌐 STEP 4: Initializing TimesFM + HMM Ensemble")
    logger.info("-" * 80)

    try:
        # Create ensemble with pre-trained HMM
        ensemble = TimesFMHMMEnsemble(
            config=config,
            hmm_model=hmm_model
        )

        logger.info("✅ Ensemble initialized successfully!")

        # Display ensemble info
        ensemble_info = ensemble.get_ensemble_info()
        print()
        print("🔗 Ensemble Configuration:")
        print(f"   - Integration mode: {ensemble_info['integration_mode']}")
        print(f"   - TimesFM enabled: {ensemble_info['timesfm_enabled']}")
        print(f"   - HMM fitted: {ensemble_info['hmm_fitted']}")
        print(f"   - HMM states: {ensemble_info['hmm_n_states']}")
        print(f"   - Ensemble weights: {ensemble_info['ensemble_weights']}")
        print()

    except Exception as e:
        logger.error(f"❌ Failed to initialize TimesFM: {str(e)}")
        logger.warning("⚠️  Falling back to HMM-only forecasting")
        ensemble = None

    # 6. Generate forecast
    logger.info("🔮 STEP 5: Generating 5-Day Forecast")
    logger.info("-" * 80)

    # Prepare context data (last 500 days)
    context_length = min(500, len(data))
    context_data = data.iloc[-context_length:].copy()

    # Add all HMM features that were used in training
    context_data['returns'] = context_data['close'].pct_change()
    context_data['log_returns'] = np.log(context_data['close'] / context_data['close'].shift(1))
    context_data['volatility_5'] = context_data['returns'].rolling(5).std()
    context_data['volatility_10'] = context_data['returns'].rolling(10).std()
    context_data['momentum_5'] = context_data['close'] / context_data['close'].shift(5) - 1
    context_data['range'] = (context_data['high'] - context_data['low']) / context_data['close']

    horizon = 5  # 5-day forecast

    print(f"📊 Forecast Configuration:")
    print(f"   - Context length: {context_length} trading days")
    print(f"   - Forecast horizon: {horizon} days")
    print(f"   - Current price: ${context_data['close'].iloc[-1]:.2f}")
    print()

    if ensemble is not None and ensemble.timesfm.enabled:
        # Use ensemble forecast
        logger.info("🌐 Using TimesFM + HMM ensemble forecast...")
        try:
            forecast_result = ensemble.forecast(
                context=context_data,
                horizon=horizon,
                freq='D'
            )

            print("✅ Ensemble Forecast Generated!")
            print()
            print("🎯 Current Market Regime:")
            print(f"   - Regime: {forecast_result.regime_label.upper()}")
            print(f"   - Regime volatility: {forecast_result.regime_volatility*100:.3f}%")
            print()

            print("📈 5-Day Price Forecast:")
            print("-" * 80)
            current_price = context_data['close'].iloc[-1]

            for day in range(horizon):
                forecast_price = forecast_result.point_forecast[day]
                change_pct = ((forecast_price / current_price) - 1) * 100

                print(f"Day {day+1}: ${forecast_price:,.2f} ({change_pct:+.2f}%)")

            print()
            print("📊 Forecast Components:")
            print(f"   - TimesFM forecast (Day 5): ${forecast_result.timesfm_forecast[-1]:,.2f}")
            if forecast_result.hmm_forecast is not None:
                print(f"   - HMM forecast (Day 5): ${forecast_result.hmm_forecast[-1]:,.2f}")
            print(f"   - Ensemble forecast (Day 5): ${forecast_result.point_forecast[-1]:,.2f}")

            # Show quantiles if available
            if forecast_result.quantile_forecasts is not None:
                print()
                print("📊 Uncertainty Quantiles (Day 5):")
                for q, values in forecast_result.quantile_forecasts.items():
                    print(f"   - {int(q*100)}th percentile: ${values[-1]:,.2f}")

        except Exception as e:
            logger.error(f"❌ Ensemble forecast failed: {str(e)}")
            logger.warning("⚠️  Falling back to HMM-only forecast")
            ensemble = None

    # Fallback to HMM-only if ensemble failed
    if ensemble is None:
        logger.info("🤖 Using HMM-only forecast...")

        # Get current regime
        last_features = hmm_features.iloc[-1:].copy()
        current_state, regime_label = hmm_model.predict_regime(last_features)

        print()
        print("🎯 Current Market Regime:")
        print(f"   - Regime: {regime_label.upper()}")
        print(f"   - State ID: {current_state}")

        # Simple forecast based on regime
        regime_stats = hmm_model.get_regime_stats()[current_state]
        current_price = context_data['close'].iloc[-1]

        print()
        print("📈 5-Day Price Forecast (HMM-based):")
        print("-" * 80)

        for day in range(horizon):
            # Simple random walk with regime drift
            expected_return = regime_stats['mean_return']
            forecast_price = current_price * (1 + expected_return * (day + 1))
            change_pct = ((forecast_price / current_price) - 1) * 100

            print(f"Day {day+1}: ${forecast_price:,.2f} ({change_pct:+.2f}%)")

    print()
    print("=" * 80)
    print("✅ FORECAST COMPLETE!")
    print("=" * 80)
    print()

    logger.info("🎉 Test forecast completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"❌ Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)
