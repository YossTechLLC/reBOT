"""
Walk-Forward Validation for Volatility MVP
===========================================
Validates the volatility prediction system on historical data.

Methodology:
1. Train on first N days
2. Test on last M days (walk-forward)
3. Measure: Accuracy, Precision, Recall, Win Rate
4. Target: >50% accuracy on "will tomorrow exceed 1.2% range"

Usage:
    python scripts/validate_volatility_mvp.py

Output:
    - Validation metrics printed to console
    - Results saved to: outputs/validation_results.csv
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Add src to path
sys.path.insert(0, 'src')

from data.alpaca_client import AlpacaDataClient
from data.volatility_features import VolatilityFeatureEngineer
from models.hmm_volatility import VolatilityHMM
from volatility.confidence_scorer import VolatilityConfidenceScorer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_mvp(
    train_days: int = 180,
    test_days: int = 30,
    volatility_threshold: float = 0.012,  # 1.2% target
    n_regimes: int = 5,  # Match UI default (supports extreme_vol)
    use_timesfm: bool = True  # Use TimesFM for predictions
) -> dict:
    """
    Run walk-forward validation on volatility prediction system.

    Args:
        train_days: Days of data for training
        test_days: Days of data for testing
        volatility_threshold: Volatility threshold for profitable trades (default: 1.2%)
        n_regimes: Number of HMM regimes (default: 5 to support extreme_vol)
        use_timesfm: Whether to use TimesFM forecasting (default: True)

    Returns:
        Dictionary with validation metrics
    """
    logger.info("=" * 60)
    logger.info("WALK-FORWARD VALIDATION")
    logger.info("=" * 60)

    # Step 1: Download data
    logger.info("\n📊 STEP 1: Downloading data...")
    client = AlpacaDataClient(
        api_key='PKDTSYSP4AYPZDNELOHNRPW2BR',
        secret_key='Ae8HNurExREVLghBw5dw9D3Pinkc2Kv8kNcJykMB3XQE'
    )

    total_days = train_days + test_days
    spy_daily = client.get_daily_bars('SPY', days=total_days)
    logger.info(f"   Downloaded {len(spy_daily)} SPY bars")

    # Download VIX
    import yfinance as yf
    vix_daily = yf.download('^VIX', period=f'{total_days}d', progress=False)
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

    if len(spy_features) < train_days + test_days:
        logger.warning(f"   ⚠️  Only {len(spy_features)} usable rows (wanted {train_days + test_days})")

    # Step 3: Split data
    logger.info("\n✂️  STEP 3: Splitting data...")
    split_idx = len(spy_features) - test_days
    train_df = spy_features.iloc[:split_idx]
    test_df = spy_features.iloc[split_idx:]

    logger.info(f"   Train: {len(train_df)} days ({train_df.index[0]} to {train_df.index[-1]})")
    logger.info(f"   Test: {len(test_df)} days ({test_df.index[0]} to {test_df.index[-1]})")

    # Step 4: Train HMM
    logger.info(f"\n🎯 STEP 4: Training HMM on training data ({n_regimes} regimes)...")
    hmm_model = VolatilityHMM(n_regimes=n_regimes)
    metrics = hmm_model.train(train_df, n_iter=100)

    logger.info(f"   Converged: {metrics['converged']}")
    logger.info(f"   Regime volatilities:")
    for label in hmm_model.regime_labels:
        vol = hmm_model.regime_volatilities[label]
        logger.info(f"      {label}: {vol:.3f} ({vol*100:.2f}%)")

    # Step 4b: Load TimesFM (optional)
    timesfm_forecaster = None
    if use_timesfm:
        logger.info("\n🔮 STEP 4b: Loading TimesFM forecaster...")
        try:
            from models.timesfm_volatility import TimesFMVolatilityForecaster
            timesfm_forecaster = TimesFMVolatilityForecaster()
            if timesfm_forecaster.is_available():
                logger.info("   ✅ TimesFM loaded successfully")
            else:
                logger.warning("   ⚠️ TimesFM not available, proceeding without it")
                timesfm_forecaster = None
        except Exception as e:
            logger.warning(f"   ⚠️ Failed to load TimesFM: {e}")
            timesfm_forecaster = None

    # Step 5: Generate predictions on test data
    logger.info("\n🔮 STEP 5: Generating predictions on test data...")
    scorer = VolatilityConfidenceScorer()

    predictions = []
    actuals = []
    confidence_scores = []
    regime_labels = []

    for i in range(len(test_df)):
        # For day i, use all data up to day i (not including day i)
        context_df = spy_features.iloc[:split_idx + i]

        # Predict regime for day i
        prediction = hmm_model.predict_latest(context_df)

        regime_label = prediction['regime_label']
        regime_volatility = prediction['expected_volatility']

        # Get actual features for day i
        row = test_df.iloc[i]
        feature_signals = {
            'overnight_gap_abs': row['overnight_gap_abs'],
            'vix_change_1d': row['vix_change_1d'],
            'vix_level': row['vix_level'],
            'range_expansion': row['range_expansion'],
            'volume_surge': row['volume_surge'],
            'volume_ratio': row['volume_ratio'],
            'high_range_days_5': row['high_range_days_5']
        }

        # Get TimesFM forecast if available
        timesfm_forecast = None
        if timesfm_forecaster is not None:
            try:
                timesfm_forecast = timesfm_forecaster.predict_next_day(context_df)
            except Exception:
                pass  # Silently continue without TimesFM

        # Calculate confidence score
        score = scorer.calculate_score(
            regime_volatility=regime_volatility,
            regime_label=regime_label,
            timesfm_forecast=timesfm_forecast,
            feature_signals=feature_signals
        )

        # Determine prediction: will tomorrow exceed threshold?
        predicted_high_vol = score.total_score >= scorer.threshold  # Score >= 40 = TRADE

        # Actual: did today exceed threshold?
        actual_high_vol = row['intraday_range_pct'] >= volatility_threshold

        predictions.append(predicted_high_vol)
        actuals.append(actual_high_vol)
        confidence_scores.append(score.total_score)
        regime_labels.append(regime_label)

    logger.info(f"   Generated {len(predictions)} predictions")

    # Step 6: Calculate metrics
    logger.info("\n📊 STEP 6: Calculating metrics...")

    # Convert to arrays
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    confidence_scores = np.array(confidence_scores)

    # Accuracy: % of correct predictions
    accuracy = accuracy_score(actuals, predictions) * 100

    # Precision: Of predicted high-vol days, how many were actually high-vol?
    precision = precision_score(actuals, predictions, zero_division=0) * 100

    # Recall: Of actual high-vol days, how many did we predict?
    recall = recall_score(actuals, predictions, zero_division=0) * 100

    # F1 Score
    f1 = f1_score(actuals, predictions, zero_division=0)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(actuals, predictions).ravel()

    # Trade statistics
    total_trades = predictions.sum()  # Number of days we signaled TRADE
    actual_high_vol_days = actuals.sum()  # Number of actual high-vol days

    # Win rate: Of trades taken, how many were profitable?
    win_rate = (tp / total_trades * 100) if total_trades > 0 else 0

    # Miss rate: Of high-vol days, how many did we miss?
    miss_rate = (fn / actual_high_vol_days * 100) if actual_high_vol_days > 0 else 0

    # Display results
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 60)

    logger.info(f"\n📈 Classification Metrics:")
    logger.info(f"   Accuracy: {accuracy:.1f}%")
    logger.info(f"   Precision: {precision:.1f}%")
    logger.info(f"   Recall: {recall:.1f}%")
    logger.info(f"   F1 Score: {f1:.3f}")

    logger.info(f"\n📊 Confusion Matrix:")
    logger.info(f"   True Positives (TP): {tp} - Correctly predicted high-vol")
    logger.info(f"   True Negatives (TN): {tn} - Correctly predicted low-vol")
    logger.info(f"   False Positives (FP): {fp} - Predicted high-vol, was low-vol")
    logger.info(f"   False Negatives (FN): {fn} - Predicted low-vol, was high-vol")

    logger.info(f"\n💰 Trading Metrics:")
    logger.info(f"   Total Test Days: {len(test_df)}")
    logger.info(f"   Actual High-Vol Days: {actual_high_vol_days} ({actual_high_vol_days/len(test_df)*100:.1f}%)")
    logger.info(f"   Trade Signals: {total_trades} ({total_trades/len(test_df)*100:.1f}%)")
    logger.info(f"   Win Rate: {win_rate:.1f}% (of trades taken)")
    logger.info(f"   Miss Rate: {miss_rate:.1f}% (of high-vol days)")

    # Calculate expected value (simplified)
    avg_win = 150  # $150 per winning trade (from strategy docs)
    avg_loss = 80  # $80 per losing trade
    expected_value_per_trade = (win_rate/100) * avg_win - (1 - win_rate/100) * avg_loss

    logger.info(f"\n💵 Expected Value (Estimated):")
    logger.info(f"   Win Rate: {win_rate:.1f}%")
    logger.info(f"   Avg Win: ${avg_win}")
    logger.info(f"   Avg Loss: ${avg_loss}")
    logger.info(f"   Expected Value/Trade: ${expected_value_per_trade:.2f}")
    logger.info(f"   Expected Value/Month (20 trades): ${expected_value_per_trade * 20:.2f}")

    # Determine if validation passed
    logger.info(f"\n✅ VALIDATION STATUS:")
    if accuracy >= 50 and win_rate >= 40:
        logger.info(f"   ✅ PASS - Accuracy >= 50% and Win Rate >= 40%")
        logger.info(f"   Ready to proceed to extended validation (Week 3)")
        validation_status = "PASS"
    elif accuracy >= 50:
        logger.info(f"   ⚠️  PARTIAL PASS - Accuracy >= 50% but Win Rate < 40%")
        logger.info(f"   Consider adjusting confidence threshold")
        validation_status = "PARTIAL"
    else:
        logger.info(f"   ❌ FAIL - Accuracy < 50%")
        logger.info(f"   Need to reassess strategy or improve model")
        validation_status = "FAIL"

    # Step 7: Save results
    logger.info("\n💾 STEP 7: Saving results...")

    results_df = test_df.copy()
    results_df['predicted_high_vol'] = predictions
    results_df['trade_signal'] = predictions  # UI expects this column name
    results_df['actual_high_vol'] = actuals
    results_df['confidence_score'] = confidence_scores
    results_df['regime_label'] = regime_labels
    results_df['correct_prediction'] = predictions == actuals

    os.makedirs('outputs', exist_ok=True)
    results_df.to_csv('outputs/validation_results.csv')
    logger.info(f"   ✅ Results saved to: outputs/validation_results.csv")

    # Save summary metrics
    summary = {
        'validation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'train_days': len(train_df),
        'test_days': len(test_df),
        'volatility_threshold': volatility_threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'total_trades': int(total_trades),
        'actual_high_vol_days': int(actual_high_vol_days),
        'win_rate': win_rate,
        'miss_rate': miss_rate,
        'expected_value_per_trade': expected_value_per_trade,
        'validation_status': validation_status
    }

    import json
    with open('outputs/validation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"   ✅ Summary saved to: outputs/validation_summary.json")

    logger.info("\n" + "=" * 60)
    logger.info("✅ VALIDATION COMPLETE")
    logger.info("=" * 60)

    return summary


if __name__ == "__main__":
    # Run validation with default parameters
    summary = validate_mvp(train_days=180, test_days=30, volatility_threshold=0.012)
