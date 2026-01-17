#!/usr/bin/env python
"""
PHASE 0: VALIDATION SPRINT - Intraday Regime Shift Detection

Objective: Prove regime shifts are predictable and profitable BEFORE building infrastructure.

Success Criteria (Adjusted for 60-day data limit):
1. Regime shifts occur >8 days/60 days (>13% of trading days, ~40/year extrapolated)
2. Shift prediction AUC >0.65 (better than random)
3. Average move during shift >0.5% (enough for options profitability)
4. Simulated win rate >50%
5. Expectancy >$20/trade

NOTE: yfinance free tier limits 15-min data to 60 days. For production, use paid data (Polygon, Alpaca).

Timeline: 1 week
Effort: 15 hours
Cost: $0
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.loader import load_config
from src.models.hmm_core import CommodityHMM

# Configure
warnings.filterwarnings('ignore')
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (14, 8)

print("=" * 80)
print("🚀 PHASE 0: VALIDATION SPRINT - Intraday Regime Shift Detection")
print("=" * 80)
print()

# ============================================================================
# STEP 1: DATA COLLECTION
# ============================================================================

print("📊 STEP 1: Downloading 1 Year of SPY 15-Minute Data")
print("-" * 80)

# Download maximum available 15-minute bars (yfinance limit: 60 days)
print(f"⚠️  NOTE: yfinance limits 15-min data to last 60 days")
print(f"Fetching SPY 15-minute bars using period='60d'")

# Use period instead of specific dates (works better with yfinance)
spy_15min = yf.download(
    'SPY',
    period='60d',
    interval='15m',
    progress=False
)

if len(spy_15min) == 0:
    print("❌ ERROR: Could not fetch 15-minute data from yfinance")
    print("   This may be due to API limitations or market data availability")
    print("   For production, consider paid data sources (Polygon.io, Alpaca, etc.)")
    print()
    print("   WORKAROUND: Using daily data as fallback for demonstration")
    print()

    # Fallback: Use daily data at 5-minute granularity (simulated intraday)
    spy_daily = yf.download('SPY', period='1y', interval='1d', progress=False)

    if len(spy_daily) == 0:
        raise RuntimeError("Cannot fetch even daily data - yfinance may be down")

    # Create synthetic 15-min data from daily (for demo purposes only)
    # In production, you MUST use real intraday data
    print("   ⚠️  Using SYNTHETIC 15-minute data from daily bars")
    print("   ⚠️  This is for DEMONSTRATION ONLY - not suitable for real trading")
    print()

    synthetic_bars = []
    for date, row in spy_daily.iterrows():
        # Create 26 bars per day (9:30-16:00 = 6.5 hours = 26 × 15min)
        open_price = row['Open']
        close_price = row['Close']
        high_price = row['High']
        low_price = row['Low']
        daily_volume = row['Volume']

        # Simulate intraday price path (linear interpolation + noise)
        for i in range(26):
            timestamp = pd.Timestamp(date).replace(hour=9, minute=30) + timedelta(minutes=i*15)

            # Linear interpolation with noise
            progress = i / 25  # 0 to 1
            intraday_close = open_price + (close_price - open_price) * progress
            intraday_close += np.random.randn() * (high_price - low_price) * 0.1

            # Clamp to high/low
            intraday_close = np.clip(intraday_close, low_price, high_price)

            synthetic_bars.append({
                'timestamp': timestamp,
                'Open': intraday_close * (1 + np.random.randn() * 0.001),
                'High': intraday_close * (1 + abs(np.random.randn()) * 0.002),
                'Low': intraday_close * (1 - abs(np.random.randn()) * 0.002),
                'Close': intraday_close,
                'Volume': daily_volume / 26
            })

    spy_15min = pd.DataFrame(synthetic_bars).set_index('timestamp')
    print(f"   Created {len(spy_15min)} synthetic 15-min bars")

print(f"✅ Downloaded {len(spy_15min)} 15-minute bars")

print(f"✅ Downloaded {len(spy_15min)} 15-minute bars")
print(f"   Date range: {spy_15min.index[0]} to {spy_15min.index[-1]}")
print(f"   Columns: {list(spy_15min.columns)}")
print()

# Save data
data_dir = Path(__file__).parent.parent / 'data' / 'intraday'
data_dir.mkdir(parents=True, exist_ok=True)
spy_15min.to_csv(data_dir / 'spy_15min_validation.csv')
print(f"💾 Saved to: {data_dir / 'spy_15min_validation.csv'}")
print()

# Download VIX for regime context
print("Fetching VIX daily data...")
vix_daily = yf.download('^VIX', period='1y', interval='1d', progress=False)
print(f"✅ Downloaded {len(vix_daily)} VIX daily bars")
print()

# ============================================================================
# STEP 2: FEATURE ENGINEERING FOR INTRADAY
# ============================================================================

print("🔧 STEP 2: Engineering Intraday Features")
print("-" * 80)

# Flatten multi-level columns (yfinance returns tuples)
if isinstance(spy_15min.columns, pd.MultiIndex):
    spy_15min.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in spy_15min.columns]
else:
    spy_15min.columns = [col.lower() for col in spy_15min.columns]

# Basic features
spy_15min['returns'] = spy_15min['close'].pct_change()
spy_15min['log_returns'] = np.log(spy_15min['close'] / spy_15min['close'].shift(1))

# Volatility (rolling standard deviation)
spy_15min['volatility_5'] = spy_15min['returns'].rolling(5).std()
spy_15min['volatility_10'] = spy_15min['returns'].rolling(10).std()

# Range (high-low volatility estimator)
spy_15min['range'] = (spy_15min['high'] - spy_15min['low']) / spy_15min['close']

# Momentum
spy_15min['momentum_5'] = spy_15min['close'] / spy_15min['close'].shift(5) - 1
spy_15min['momentum_10'] = spy_15min['close'] / spy_15min['close'].shift(10) - 1

# Volume features
spy_15min['volume_ma20'] = spy_15min['volume'].rolling(20).mean()
spy_15min['volume_ratio'] = spy_15min['volume'] / spy_15min['volume_ma20']

# TIMEZONE FIX: Convert to ET (US/Eastern) for market hours filtering
# yfinance returns data in UTC, but NYSE hours are in ET
import pytz
et = pytz.timezone('US/Eastern')

# Convert index to ET timezone
if spy_15min.index.tz is None:
    # If naive, assume UTC
    spy_15min.index = spy_15min.index.tz_localize('UTC')

spy_15min.index = spy_15min.index.tz_convert(et)

# Time features (now in ET)
spy_15min['hour'] = spy_15min.index.hour
spy_15min['minute'] = spy_15min.index.minute
spy_15min['time_of_day'] = spy_15min['hour'] + spy_15min['minute'] / 60

# Market open/close indicators (9:30 AM ET and 12:00 PM ET)
spy_15min['is_open'] = (spy_15min['hour'] == 9) & (spy_15min['minute'] == 30)
spy_15min['is_noon'] = (spy_15min['hour'] == 12) & (spy_15min['minute'] == 0)

# Drop NaN from rolling calculations
spy_15min_clean = spy_15min.dropna()

print(f"✅ Created {len(spy_15min_clean.columns)} features")
print(f"   Feature matrix shape: {spy_15min_clean.shape}")
print(f"   Features: {list(spy_15min_clean.columns)}")
print()

# ============================================================================
# STEP 3: TRAIN 15-MINUTE HMM
# ============================================================================

print("🤖 STEP 3: Training 15-Minute HMM Regime Detector")
print("-" * 80)

# Load config
config = load_config('config/parameters.toml')

# Prepare HMM features (only select relevant ones)
hmm_features = spy_15min_clean[[
    'returns',
    'volatility_5',
    'volatility_10',
    'momentum_5',
    'range'
]].copy()

print(f"Training HMM on {len(hmm_features)} samples with {len(hmm_features.columns)} features")
print(f"Features: {list(hmm_features.columns)}")

# Initialize HMM with 3 states
hmm_15min = CommodityHMM(config)
hmm_15min.fit_with_multiple_inits(hmm_features, n_inits=5)

print("✅ HMM model trained successfully!")
print()

# Display regime statistics
print("📊 15-Minute Regime Statistics:")
print("-" * 80)
regime_stats = hmm_15min.get_regime_stats()
for state_id, stats in regime_stats.items():
    print(f"\n🎯 Regime {state_id}: {stats['label'].upper()}")
    print(f"   - Mean return: {stats['mean_return']*100:.4f}% per 15-min")
    print(f"   - Volatility: {stats['volatility']*100:.4f}%")
    print(f"   - Persistence: {stats['persistence']*100:.1f}%")
    print(f"   - Frequency: {stats['count']} bars ({stats['count']/len(hmm_features)*100:.1f}%)")

# Transition matrix
print()
print("🔄 Regime Transition Matrix:")
print("-" * 80)
trans_matrix = hmm_15min.get_transition_matrix()
trans_df = pd.DataFrame(
    trans_matrix,
    columns=[f"To {regime_stats[i]['label']}" for i in range(len(regime_stats))],
    index=[f"From {regime_stats[i]['label']}" for i in range(len(regime_stats))]
)
print(trans_df.round(3))
print()

# Predict regimes for all bars (use underlying hmmlearn model)
if hasattr(hmm_15min, 'scaler') and hmm_15min.scaler is not None:
    features_scaled = hmm_15min.scaler.transform(hmm_features)
    predicted_regimes = hmm_15min.model.predict(features_scaled)
else:
    predicted_regimes = hmm_15min.model.predict(hmm_features)

spy_15min_clean['regime'] = predicted_regimes
spy_15min_clean['regime_label'] = spy_15min_clean['regime'].map(
    {i: regime_stats[i]['label'] for i in regime_stats.keys()}
)

# ============================================================================
# STEP 4: IDENTIFY REGIME SHIFTS (9:30 AM → 12:00 PM)
# ============================================================================

print("🔍 STEP 4: Identifying Regime Shifts (9:30 AM → 12:00 PM)")
print("-" * 80)

# Extract 9:30 AM and 12:00 PM bars for each trading day
morning_bars = spy_15min_clean[spy_15min_clean['is_open'] == True].copy()
noon_bars = spy_15min_clean[spy_15min_clean['is_noon'] == True].copy()

# Match by date
morning_bars['date'] = morning_bars.index.date
noon_bars['date'] = noon_bars.index.date

# Merge
shift_analysis = morning_bars[['date', 'regime', 'regime_label', 'close']].merge(
    noon_bars[['date', 'regime', 'regime_label', 'close']],
    on='date',
    suffixes=('_open', '_noon')
)

# Identify shifts
shift_analysis['shift_occurred'] = (
    shift_analysis['regime_open'] != shift_analysis['regime_noon']
)
shift_analysis['shift_type'] = (
    shift_analysis['regime_label_open'] + ' → ' + shift_analysis['regime_label_noon']
)
shift_analysis['price_change'] = (
    shift_analysis['close_noon'] / shift_analysis['close_open'] - 1
)
shift_analysis['price_change_pct'] = shift_analysis['price_change'] * 100

# Statistics
n_days = len(shift_analysis)
n_shifts = shift_analysis['shift_occurred'].sum()
shift_rate = n_shifts / n_days * 100

print(f"📈 Regime Shift Analysis:")
print(f"   - Total trading days analyzed: {n_days}")
print(f"   - Days with regime shift: {n_shifts}")
print(f"   - Shift rate: {shift_rate:.1f}%")
print()

# Shift type breakdown
print("Shift Type Breakdown:")
shift_type_counts = shift_analysis[shift_analysis['shift_occurred']]['shift_type'].value_counts()
for shift_type, count in shift_type_counts.items():
    pct = count / n_shifts * 100
    print(f"   - {shift_type}: {count} ({pct:.1f}%)")
print()

# Price movement analysis
print("Price Movement Analysis:")
shift_days = shift_analysis[shift_analysis['shift_occurred']]
no_shift_days = shift_analysis[~shift_analysis['shift_occurred']]

avg_move_shift = shift_days['price_change_pct'].abs().mean()
avg_move_no_shift = no_shift_days['price_change_pct'].abs().mean()

print(f"   - Avg absolute move (shift days): {avg_move_shift:.3f}%")
print(f"   - Avg absolute move (no-shift days): {avg_move_no_shift:.3f}%")
print(f"   - Difference: {avg_move_shift - avg_move_no_shift:.3f}%")
print()

# Statistical significance (t-test)
from scipy.stats import ttest_ind
t_stat, p_value = ttest_ind(
    shift_days['price_change_pct'].abs(),
    no_shift_days['price_change_pct'].abs()
)
print(f"   - T-statistic: {t_stat:.3f}")
print(f"   - P-value: {p_value:.4f}")
if p_value < 0.05:
    print("   ✅ Statistically significant difference (p < 0.05)")
else:
    print("   ⚠️  Not statistically significant (p >= 0.05)")
print()

# ============================================================================
# STEP 5: PREDICTABILITY ANALYSIS
# ============================================================================

print("🎯 STEP 5: Regime Shift Predictability Analysis")
print("-" * 80)

# Create features at 9:30 AM to predict shift by noon
morning_bars_full = spy_15min_clean[spy_15min_clean['is_open'] == True].copy()
morning_bars_full['date'] = morning_bars_full.index.date

# Add overnight gap (current open vs previous close)
morning_bars_full['overnight_gap'] = morning_bars_full['close'].pct_change()

# Merge with shift labels
predictability_data = morning_bars_full.merge(
    shift_analysis[['date', 'shift_occurred', 'price_change_pct']],
    on='date',
    how='inner'
)

# Features for prediction
predictor_features = [
    'overnight_gap',
    'volatility_5',
    'volatility_10',
    'momentum_5',
    'volume_ratio'
]

X = predictability_data[predictor_features].dropna()
y = predictability_data.loc[X.index, 'shift_occurred'].astype(int)

print(f"Training predictive model:")
print(f"   - Samples: {len(X)}")
print(f"   - Positive class (shifts): {y.sum()} ({y.mean()*100:.1f}%)")
print(f"   - Features: {predictor_features}")
print()

# Train logistic regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_scaled, y)

# Predictions
y_pred = lr.predict(X_scaled)
y_pred_proba = lr.predict_proba(X_scaled)[:, 1]

# Metrics
auc = roc_auc_score(y, y_pred_proba)
print(f"📊 Predictive Performance:")
print(f"   - AUC (Area Under ROC): {auc:.3f}")

if auc > 0.65:
    print("   ✅ PASS: AUC > 0.65 (better than random)")
elif auc > 0.60:
    print("   🟡 MARGINAL: AUC > 0.60 (slight edge)")
else:
    print("   ❌ FAIL: AUC <= 0.60 (no better than random)")
print()

# Classification report
print("Classification Report:")
print(classification_report(y, y_pred, target_names=['No Shift', 'Shift']))
print()

# Feature importance
print("Feature Importance (Logistic Regression Coefficients):")
for feature, coef in zip(predictor_features, lr.coef_[0]):
    print(f"   - {feature}: {coef:+.4f}")
print()

# ============================================================================
# STEP 6: PROFITABILITY ESTIMATION
# ============================================================================

print("💰 STEP 6: Simulated Options Strategy Profitability")
print("-" * 80)

# Simple simulation: Buy ATM call spread when shift predicted
# Assume:
# - Spread width: 10 points ($1000 max profit)
# - Debit: $300 (30% of width, typical for ATM spread)
# - Exit at noon or +40% profit target

SPREAD_WIDTH = 10  # SPX points
MAX_PROFIT = SPREAD_WIDTH * 100  # $1000
DEBIT_COST = 300  # Typical ATM spread cost
PROFIT_TARGET = 0.40  # Exit at +40% of max profit
STOP_LOSS = 0.60  # Stop at -60% of capital at risk

# Add predictions to shift_analysis
shift_analysis_with_pred = shift_analysis.copy()
shift_analysis_with_pred['shift_predicted'] = False

# Only predict shift if probability > 0.50
shift_threshold = 0.50
for idx in X.index:
    date = predictability_data.loc[idx, 'date']
    prob = y_pred_proba[X.index.get_loc(idx)]
    if prob > shift_threshold:
        shift_analysis_with_pred.loc[
            shift_analysis_with_pred['date'] == date, 'shift_predicted'
        ] = True

# Simulate trades
trades = []
for _, row in shift_analysis_with_pred.iterrows():
    if row['shift_predicted']:
        # Trade entered
        price_move_pct = row['price_change_pct']
        price_move_points = row['close_open'] * (price_move_pct / 100)

        # Assume 1:1 delta for ATM spread initially
        # Profit scales linearly up to max profit
        spread_value_change = min(
            price_move_points / SPREAD_WIDTH * MAX_PROFIT,
            MAX_PROFIT
        )

        pnl = spread_value_change - DEBIT_COST

        # Apply profit target and stop loss
        if pnl > PROFIT_TARGET * MAX_PROFIT:
            pnl = PROFIT_TARGET * MAX_PROFIT
            exit_reason = 'profit_target'
        elif pnl < -STOP_LOSS * DEBIT_COST:
            pnl = -STOP_LOSS * DEBIT_COST
            exit_reason = 'stop_loss'
        else:
            exit_reason = 'time_stop'

        trades.append({
            'date': row['date'],
            'shift_actual': row['shift_occurred'],
            'price_change_pct': price_move_pct,
            'pnl': pnl,
            'exit_reason': exit_reason
        })

trades_df = pd.DataFrame(trades)

# Performance metrics
n_trades = len(trades_df)
wins = (trades_df['pnl'] > 0).sum()
losses = (trades_df['pnl'] <= 0).sum()
win_rate = wins / n_trades * 100 if n_trades > 0 else 0

avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losses > 0 else 0

total_pnl = trades_df['pnl'].sum()
avg_pnl = trades_df['pnl'].mean()

profit_factor = (
    trades_df[trades_df['pnl'] > 0]['pnl'].sum() /
    abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
    if losses > 0 else float('inf')
)

print(f"📊 Simulated Trading Performance:")
print(f"   - Total trades: {n_trades}")
print(f"   - Wins: {wins} | Losses: {losses}")
print(f"   - Win rate: {win_rate:.1f}%")
print(f"   - Avg win: ${avg_win:.2f}")
print(f"   - Avg loss: ${avg_loss:.2f}")
print(f"   - Profit factor: {profit_factor:.2f}")
print(f"   - Total P&L: ${total_pnl:.2f}")
print(f"   - Expectancy (avg P&L per trade): ${avg_pnl:.2f}")
print()

# Exit reason breakdown
print("Exit Reason Breakdown:")
exit_reasons = trades_df['exit_reason'].value_counts()
for reason, count in exit_reasons.items():
    pct = count / n_trades * 100
    print(f"   - {reason}: {count} ({pct:.1f}%)")
print()

# ============================================================================
# STEP 7: VALIDATION CRITERIA CHECK
# ============================================================================

print("=" * 80)
print("✅ PHASE 0 VALIDATION CRITERIA - RESULTS")
print("=" * 80)
print()

# Adjusted criteria for 60-day sample
expected_trading_days = n_days  # Actual days in sample
min_shifts_needed = int(expected_trading_days * 0.13)  # 13% = ~40/year extrapolated

criteria = {
    f'Regime shifts occur >{min_shifts_needed} days (~13% rate)': n_shifts > min_shifts_needed,
    'Shift prediction AUC >0.65': auc > 0.65,
    'Average move during shift >0.5%': avg_move_shift > 0.5,
    'Simulated win rate >50%': win_rate > 50,
    'Expectancy >$20/trade': avg_pnl > 20
}

all_passed = all(criteria.values())

for criterion, passed in criteria.items():
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status}: {criterion}")

print()
print("=" * 80)

if all_passed:
    print("🎉 ALL VALIDATION CRITERIA PASSED!")
    print("✅ RECOMMENDATION: PROCEED TO PHASE 1 (Data Foundation)")
    decision = "GO"
else:
    print("⚠️  SOME CRITERIA FAILED")
    print("❌ RECOMMENDATION: RECONSIDER APPROACH OR PIVOT STRATEGY")
    decision = "NO-GO"

print("=" * 80)
print()

# ============================================================================
# STEP 8: SAVE RESULTS
# ============================================================================

print("💾 Saving Results...")

outputs_dir = Path(__file__).parent.parent / 'outputs'
outputs_dir.mkdir(parents=True, exist_ok=True)

# Save shift analysis
shift_analysis.to_csv(outputs_dir / 'phase0_shift_analysis.csv', index=False)
print(f"✅ Saved: {outputs_dir / 'phase0_shift_analysis.csv'}")

# Save trades
if len(trades_df) > 0:
    trades_df.to_csv(outputs_dir / 'phase0_simulated_trades.csv', index=False)
    print(f"✅ Saved: {outputs_dir / 'phase0_simulated_trades.csv'}")

# Save model
models_dir = Path(__file__).parent.parent / 'models' / 'trained'
models_dir.mkdir(parents=True, exist_ok=True)
hmm_15min.save_model(models_dir / 'phase0_15min_hmm.pkl')
print(f"✅ Saved: {models_dir / 'phase0_15min_hmm.pkl'}")

# Generate report
report_path = Path(__file__).parent.parent / 'docs' / 'PHASE0_VALIDATION_REPORT.md'
report_path.parent.mkdir(parents=True, exist_ok=True)

with open(report_path, 'w') as f:
    f.write("# PHASE 0 VALIDATION REPORT - Intraday Regime Shift Detection\n\n")
    f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("---\n\n")

    f.write("## Executive Summary\n\n")
    f.write(f"**Decision:** {decision}\n\n")
    if decision == "GO":
        f.write("**Recommendation:** Proceed to Phase 1 (Data Foundation)\n\n")
    else:
        f.write("**Recommendation:** Reconsider approach or pivot strategy\n\n")

    f.write("---\n\n")

    f.write("## Data Summary\n\n")
    f.write(f"- **Period:** {spy_15min.index[0]} to {spy_15min.index[-1]}\n")
    f.write(f"- **15-minute bars:** {len(spy_15min)}\n")
    f.write(f"- **Trading days analyzed:** {n_days}\n")
    f.write(f"- **Features engineered:** {len(spy_15min_clean.columns)}\n\n")

    f.write("## HMM Regime Detection\n\n")
    f.write("**Regimes Identified:**\n\n")
    for state_id, stats in regime_stats.items():
        f.write(f"- **Regime {state_id} ({stats['label']}):**\n")
        f.write(f"  - Mean return: {stats['mean_return']*100:.4f}% per 15-min\n")
        f.write(f"  - Volatility: {stats['volatility']*100:.4f}%\n")
        f.write(f"  - Persistence: {stats['persistence']*100:.1f}%\n")
        f.write(f"  - Frequency: {stats['count']/len(hmm_features)*100:.1f}%\n\n")

    f.write("## Regime Shift Analysis\n\n")
    f.write(f"- **Days with shift (9:30 AM → 12:00 PM):** {n_shifts} / {n_days} ({shift_rate:.1f}%)\n")
    f.write(f"- **Avg move (shift days):** {avg_move_shift:.3f}%\n")
    f.write(f"- **Avg move (no-shift days):** {avg_move_no_shift:.3f}%\n")
    f.write(f"- **T-test p-value:** {p_value:.4f}\n\n")

    f.write("**Shift Type Breakdown:**\n\n")
    for shift_type, count in shift_type_counts.items():
        pct = count / n_shifts * 100
        f.write(f"- {shift_type}: {count} ({pct:.1f}%)\n")
    f.write("\n")

    f.write("## Predictability Analysis\n\n")
    f.write(f"- **AUC (ROC):** {auc:.3f}\n")
    f.write(f"- **Prediction threshold:** {shift_threshold:.2f}\n")
    f.write(f"- **Samples:** {len(X)}\n\n")

    f.write("**Feature Importance:**\n\n")
    for feature, coef in zip(predictor_features, lr.coef_[0]):
        f.write(f"- {feature}: {coef:+.4f}\n")
    f.write("\n")

    f.write("## Simulated Trading Performance\n\n")
    f.write(f"- **Total trades:** {n_trades}\n")
    f.write(f"- **Win rate:** {win_rate:.1f}%\n")
    f.write(f"- **Avg win:** ${avg_win:.2f}\n")
    f.write(f"- **Avg loss:** ${avg_loss:.2f}\n")
    f.write(f"- **Profit factor:** {profit_factor:.2f}\n")
    f.write(f"- **Total P&L:** ${total_pnl:.2f}\n")
    f.write(f"- **Expectancy:** ${avg_pnl:.2f}/trade\n\n")

    f.write("## Validation Criteria Results\n\n")
    for criterion, passed in criteria.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        f.write(f"- {status}: {criterion}\n")
    f.write("\n")

    f.write("---\n\n")

    f.write("## Conclusion\n\n")
    if all_passed:
        f.write("All validation criteria passed. The intraday regime shift strategy shows promise.\n\n")
        f.write("**Next Steps:**\n")
        f.write("1. Proceed to Phase 1: Data Foundation\n")
        f.write("2. Build robust intraday data pipeline\n")
        f.write("3. Develop hierarchical HMM architecture\n")
    else:
        f.write("Some validation criteria failed. Consider:\n\n")
        f.write("**Options:**\n")
        f.write("1. Refine HMM parameters (more/fewer states)\n")
        f.write("2. Try different features for prediction\n")
        f.write("3. Adjust shift definition (different time windows)\n")
        f.write("4. Pivot to daily swing trading instead of intraday\n")

print(f"✅ Saved: {report_path}")
print()

print("=" * 80)
print("🎉 PHASE 0 VALIDATION COMPLETE!")
print("=" * 80)
print()
print(f"📊 Review the full report: {report_path}")
print()
if decision == "GO":
    print("✅ APPROVED TO PROCEED TO PHASE 1")
else:
    print("⚠️  RECOMMEND FURTHER ANALYSIS BEFORE PROCEEDING")
print()
