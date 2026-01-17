"""
Train HMM on SPY Volatility Data
=================================
Script to train Hidden Markov Model for volatility regime detection.

Usage:
    python scripts/train_hmm_volatility.py

Output:
    - Trained model saved to: models/hmm_volatility.pkl
    - Training report printed to console
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, 'src')

from data.alpaca_client import AlpacaDataClient
from data.volatility_features import VolatilityFeatureEngineer
from models.hmm_volatility import VolatilityHMM

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_hmm(days_history: int = 365) -> VolatilityHMM:
    """
    Train HMM on historical SPY data.

    Args:
        days_history: Days of historical data to use (default: 365 = 1 year)

    Returns:
        Trained HMM model
    """
    logger.info("=" * 60)
    logger.info("TRAINING HMM ON SPY VOLATILITY DATA")
    logger.info("=" * 60)

    # Step 1: Download data
    logger.info("\n📊 STEP 1: Downloading data...")
    client = AlpacaDataClient(
        api_key='PKDTSYSP4AYPZDNELOHNRPW2BR',
        secret_key='Ae8HNurExREVLghBw5dw9D3Pinkc2Kv8kNcJykMB3XQE'
    )
    spy_daily = client.get_daily_bars('SPY', days=days_history)
    logger.info(f"   Downloaded {len(spy_daily)} SPY bars")

    # Download VIX
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
    engineer = VolatilityFeatureEngineer()
    spy_features = engineer.add_all_features(spy_daily, vix_daily)
    logger.info(f"   Created {len(spy_features.columns)} features")
    logger.info(f"   Usable rows: {len(spy_features)}")

    if len(spy_features) < 50:
        logger.warning(f"   ⚠️  Only {len(spy_features)} samples - may not be enough for robust training")
        logger.warning(f"   ⚠️  Recommend at least 100 samples (need ~120 days of history)")

    # Step 3: Train HMM
    logger.info("\n🎯 STEP 3: Training HMM...")
    hmm_model = VolatilityHMM(n_regimes=3)
    metrics = hmm_model.train(spy_features, n_iter=100)

    logger.info(f"   Converged: {metrics['converged']}")
    logger.info(f"   Training samples: {metrics['n_samples']}")

    # Step 4: Analyze regimes
    logger.info("\n📈 STEP 4: Regime Analysis...")
    logger.info(f"   Regime volatilities:")
    for label in hmm_model.regime_labels:
        vol = hmm_model.regime_volatilities[label]
        count = metrics.get(f'{label}_count', 0)
        pct = metrics.get(f'{label}_pct', 0)
        logger.info(f"      {label}: {vol:.3f} ({vol*100:.2f}%) - {count} days ({pct:.1f}%)")

    # Step 5: Validate regime separation
    logger.info("\n✅ STEP 5: Validation...")

    # Check if regimes are well-separated
    vols = [hmm_model.regime_volatilities[label] for label in hmm_model.regime_labels]
    if len(vols) >= 2:
        separation_ratio = vols[1] / vols[0] if vols[0] > 0 else 0
        logger.info(f"   Normal/Low separation: {separation_ratio:.2f}x")
        if separation_ratio < 1.3:
            logger.warning(f"   ⚠️  Regimes may not be well-separated (want >1.5x)")

    if len(vols) == 3:
        separation_ratio = vols[2] / vols[1] if vols[1] > 0 else 0
        logger.info(f"   High/Normal separation: {separation_ratio:.2f}x")
        if separation_ratio < 1.3:
            logger.warning(f"   ⚠️  Regimes may not be well-separated (want >1.5x)")

    # Check if distributions make sense
    if hmm_model.regime_volatilities.get('low_vol', 0) > 0.008:
        logger.warning(f"   ⚠️  Low vol regime seems high (>0.8%)")

    if hmm_model.regime_volatilities.get('high_vol', 0) < 0.015:
        logger.warning(f"   ⚠️  High vol regime seems low (<1.5%)")

    # Step 6: Test prediction on latest data
    logger.info("\n🔮 STEP 6: Test Prediction (Latest Day)...")
    latest_prediction = hmm_model.predict_latest(spy_features)

    logger.info(f"   Regime: {latest_prediction['regime_label']}")
    logger.info(f"   Expected volatility: {latest_prediction['expected_volatility']:.3f} ({latest_prediction['expected_volatility']*100:.2f}%)")
    logger.info(f"   Confidence: {latest_prediction['confidence']:.2%}")
    logger.info(f"   Regime probabilities:")
    for label, prob in latest_prediction['regime_probabilities'].items():
        logger.info(f"      {label}: {prob:.2%}")

    # Compare to actual
    actual_vol = spy_features['intraday_range_pct'].iloc[-1]
    logger.info(f"   Actual volatility: {actual_vol:.3f} ({actual_vol*100:.2f}%)")
    error = abs(latest_prediction['expected_volatility'] - actual_vol)
    logger.info(f"   Prediction error: {error:.3f} ({error*100:.2f}%)")

    # Step 7: Save model
    logger.info("\n💾 STEP 7: Saving model...")
    os.makedirs('models', exist_ok=True)
    model_path = 'models/hmm_volatility.pkl'
    hmm_model.save(model_path)
    logger.info(f"   ✅ Model saved to: {model_path}")

    # Save metadata
    metadata = {
        'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'training_samples': metrics['n_samples'],
        'regime_volatilities': hmm_model.regime_volatilities,
        'regime_distribution': {
            label: metrics.get(f'{label}_pct', 0)
            for label in hmm_model.regime_labels
        },
        'converged': metrics['converged']
    }

    import json
    metadata_path = 'models/hmm_volatility_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"   ✅ Metadata saved to: {metadata_path}")

    logger.info("\n" + "=" * 60)
    logger.info("✅ HMM TRAINING COMPLETE")
    logger.info("=" * 60)

    return hmm_model


if __name__ == "__main__":
    # Train on 1 year of data
    hmm_model = train_hmm(days_history=365)
