"""
MVP: Daily Volatility Confidence Score Generator
=================================================
End-to-end pipeline for predicting next-day intraday volatility.

Combines:
1. Alpaca data download
2. Feature engineering
3. HMM regime detection
4. Confidence scoring

Usage:
    python notebooks/volatility_mvp.py

Output:
    Daily confidence score (0-100) with trading recommendation
"""

import sys
import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, 'src')

from data.alpaca_client import AlpacaDataClient
from data.volatility_features import VolatilityFeatureEngineer
from volatility.confidence_scorer import VolatilityConfidenceScorer
from models.hmm_volatility import VolatilityHMM
from models.timesfm_volatility import TimesFMVolatilityForecaster

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VolatilityMVP:
    """MVP pipeline for daily volatility prediction."""

    def __init__(self, alpaca_key: str = None, alpaca_secret: str = None, hmm_model_path: str = 'models/hmm_volatility.pkl'):
        """
        Initialize MVP pipeline.

        Args:
            alpaca_key: Alpaca API key (or set ALPACA_API_KEY env var)
            alpaca_secret: Alpaca secret key (or set ALPACA_SECRET_KEY env var)
            hmm_model_path: Path to trained HMM model (default: models/hmm_volatility.pkl)
        """
        # Initialize components
        self.alpaca_client = AlpacaDataClient(alpaca_key, alpaca_secret)
        self.feature_engineer = VolatilityFeatureEngineer()
        self.confidence_scorer = VolatilityConfidenceScorer()

        # Load HMM model
        self.hmm_model = VolatilityHMM()
        try:
            self.hmm_model.load(hmm_model_path)
            logger.info(f"Loaded HMM model from {hmm_model_path}")
        except FileNotFoundError:
            logger.warning(f"HMM model not found at {hmm_model_path}. Run scripts/train_hmm_volatility.py first.")
            logger.warning("Falling back to percentile-based regime detection.")
            self.hmm_model = None

        # Load TimesFM model (optional)
        try:
            self.timesfm_forecaster = TimesFMVolatilityForecaster(device=device)
            if self.timesfm_forecaster.is_available():
                logger.info("TimesFM forecaster loaded successfully")
            else:
                logger.info("TimesFM not available - using HMM-only mode")
                self.timesfm_forecaster = None
        except Exception as e:
            logger.info(f"TimesFM initialization failed: {str(e)} - using HMM-only mode")
            self.timesfm_forecaster = None

        logger.info("MVP pipeline initialized")

    def run_daily_prediction(self, days_history: int = 60) -> dict:
        """
        Run complete daily prediction pipeline.

        Args:
            days_history: Days of historical data to use

        Returns:
            Dictionary with prediction results
        """
        logger.info("=" * 60)
        logger.info("VOLATILITY MVP - DAILY PREDICTION")
        logger.info("=" * 60)

        # Step 1: Download data
        logger.info("\n📊 STEP 1: Downloading data...")
        spy_daily = self.alpaca_client.get_daily_bars('SPY', days=days_history)
        logger.info(f"   Downloaded {len(spy_daily)} SPY bars")

        # Download VIX using yfinance (Alpaca doesn't have VIX)
        import yfinance as yf
        vix_daily = yf.download('^VIX', period=f'{days_history}d', progress=False)
        if isinstance(vix_daily.columns, pd.MultiIndex):
            vix_daily.columns = [col[0].lower() if isinstance(col, tuple) else col.lower()
                                  for col in vix_daily.columns]
        else:
            vix_daily.columns = [col.lower() for col in vix_daily.columns]
        logger.info(f"   Downloaded {len(vix_daily)} VIX bars")

        # Step 2: Engineer features
        logger.info("\n🔧 STEP 2: Engineering features...")
        spy_features = self.feature_engineer.add_all_features(spy_daily, vix_daily)
        logger.info(f"   Created {len(spy_features.columns)} features")
        logger.info(f"   Usable rows: {len(spy_features)}")

        if len(spy_features) == 0:
            logger.error("   ❌ No data after feature engineering!")
            return None

        # Step 3: Regime detection (HMM-based)
        logger.info("\n🎯 STEP 3: Detecting volatility regime...")
        latest = spy_features.iloc[-1]

        if self.hmm_model is not None:
            # Use trained HMM for regime detection
            prediction = self.hmm_model.predict_latest(spy_features)
            regime_label = prediction['regime_label']
            regime_volatility = prediction['expected_volatility']
            confidence = prediction['confidence']

            logger.info(f"   Current regime: {regime_label}")
            logger.info(f"   Expected volatility: {regime_volatility:.2%}")
            logger.info(f"   HMM confidence: {confidence:.2%}")
            logger.info(f"   Regime probabilities:")
            for label, prob in prediction['regime_probabilities'].items():
                logger.info(f"      {label}: {prob:.2%}")
        else:
            # Fallback: percentile-based regime detection
            logger.warning("   Using fallback percentile-based regime detection")
            vol_percentile = spy_features['intraday_range_pct'].rank(pct=True).iloc[-1] * 100

            if vol_percentile > 66:
                regime_label = 'high_vol'
            elif vol_percentile > 33:
                regime_label = 'normal_vol'
            else:
                regime_label = 'low_vol'

            regime_volatility = latest['intraday_range_pct']

            logger.info(f"   Current regime: {regime_label}")
            logger.info(f"   Expected volatility: {regime_volatility:.2%}")
            logger.info(f"   Volatility percentile: {vol_percentile:.1f}%")

        # Step 3.5: TimesFM Volatility Forecast (optional)
        timesfm_forecast = None
        if self.timesfm_forecaster is not None and self.timesfm_forecaster.is_available():
            logger.info("\n🔮 STEP 3.5: Generating TimesFM forecast...")
            try:
                timesfm_forecast = self.timesfm_forecaster.predict_next_day(
                    spy_features,
                    volatility_col='intraday_range_pct'
                )
                if timesfm_forecast is not None:
                    logger.info(f"   TimesFM forecast: {timesfm_forecast:.2%}")
                else:
                    logger.warning("   TimesFM forecast returned None")
            except Exception as e:
                logger.warning(f"   TimesFM forecast failed: {str(e)}")
                timesfm_forecast = None

        # Step 4: Calculate confidence score
        logger.info("\n📈 STEP 4: Calculating confidence score...")

        feature_signals = {
            'overnight_gap_abs': latest['overnight_gap_abs'],
            'vix_change_1d': latest['vix_change_1d'],
            'vix_level': latest['vix_level'],
            'range_expansion': latest['range_expansion'],
            'volume_surge': latest['volume_surge'],
            'volume_ratio': latest['volume_ratio'],
            'high_range_days_5': latest['high_range_days_5']
        }

        score = self.confidence_scorer.calculate_score(
            regime_volatility=regime_volatility,
            regime_label=regime_label,
            timesfm_forecast=timesfm_forecast,  # Use TimesFM if available
            feature_signals=feature_signals
        )

        # Step 5: Display results
        logger.info("\n" + "=" * 60)
        print("\n" + "=" * 60)
        print(f"DAILY VOLATILITY FORECAST - {datetime.now().strftime('%Y-%m-%d')}")
        print("=" * 60)
        print(f"Latest Data: {spy_features.index[-1].strftime('%Y-%m-%d')}")
        print()
        print(score.explanation)
        print()
        print(f"DECISION: {score.recommendation}")
        print("=" * 60)
        print()

        # Return structured results
        return {
            'date': spy_features.index[-1],
            'confidence_score': score.total_score,
            'regime_label': regime_label,
            'regime_volatility': regime_volatility,
            'feature_signals': feature_signals,
            'recommendation': score.recommendation,
            'should_trade': score.total_score >= self.confidence_scorer.threshold
        }


def main():
    """Run MVP pipeline."""
    # You can pass credentials here or set environment variables
    mvp = VolatilityMVP(
        alpaca_key='PKDTSYSP4AYPZDNELOHNRPW2BR',
        alpaca_secret='Ae8HNurExREVLghBw5dw9D3Pinkc2Kv8kNcJykMB3XQE'
    )

    # Run daily prediction
    results = mvp.run_daily_prediction(days_history=60)

    if results:
        # Save results to file
        results_df = pd.DataFrame([results])
        results_df.to_csv('outputs/daily_forecast.csv', index=False)
        logger.info(f"✅ Results saved to: outputs/daily_forecast.csv")

        # Return exit code based on trading decision
        if results['should_trade']:
            logger.info("✅ TRADE signal - confidence threshold met")
            return 0  # Success
        else:
            logger.info("⏸️  SKIP signal - confidence below threshold")
            return 1  # Skip


if __name__ == "__main__":
    exit(main())
