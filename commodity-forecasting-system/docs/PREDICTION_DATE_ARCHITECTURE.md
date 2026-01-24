# Prediction Date Architecture Analysis
**Date:** 2026-01-24
**Author:** Claude Investigation Team
**Status:** Analysis Complete - Implementation Required

---

## Executive Summary

The current Volatility Prediction System is **hardcoded to always predict for "tomorrow"** using the latest available data. This creates a fundamental limitation: users cannot backtest how the model would have performed on any historical date.

**The Problem:**
- Data is always loaded from `(today - N days)` to `today`
- Training uses ALL loaded data (no temporal cutoff)
- Prediction always uses the last row (`features_df.iloc[-1]`)
- No mechanism to simulate "point-in-time" predictions

**The Solution:**
- Add a "Prediction Date" picker to the UI
- Load data UP TO (not including) the prediction date
- Train model ONLY on data before the prediction date
- Predict for the selected date, not always "latest"

This ensures **no data leakage** and enables proper historical backtesting.

---

## Current Architecture Analysis

### 1. Data Loading Flow

**Current Behavior:**
```
User sets history_days = 180
    ↓
load_complete_dataset(days=180)
    ↓
AlpacaDataClient.get_daily_bars('SPY', days=180, end_date=None)
    ↓
end_date defaults to datetime.now()  # Always TODAY
    ↓
Returns data from (today - 180 days) to today
```

**Key Files:**
| File | Lines | Current Issue |
|------|-------|---------------|
| `src/ui/data_manager.py` | 121-135 | `load_complete_dataset()` has no `end_date` parameter |
| `src/ui/data_manager.py` | 59-72 | `load_spy_data()` doesn't pass `end_date` to Alpaca |
| `src/ui/data_manager.py` | 75-98 | `load_vix_data()` uses yfinance `period` (always ends at now) |
| `src/data/alpaca_client.py` | 74-77 | `end_date` defaults to `datetime.now()` when None |

**The Fix:**
- Add `end_date` parameter to `load_complete_dataset()`
- Pass `end_date` through to `load_spy_data()` and `load_vix_data()`
- AlpacaDataClient already supports `end_date` - just need to use it

---

### 2. Model Training Flow

**Current Behavior:**
```
Train HMM clicked
    ↓
ModelController.train_hmm(df=features_df)
    ↓
Trains on ENTIRE features_df (all loaded data)
    ↓
No temporal cutoff - includes "future" data relative to any historical date
```

**Key Files:**
| File | Lines | Current Issue |
|------|-------|---------------|
| `src/ui/model_controller.py` | 61-99 | `train_hmm()` has no `prediction_date` parameter |
| `src/models/hmm_volatility.py` | 109-158 | `train()` uses full DataFrame without date validation |

**The Fix:**
- Add `prediction_date` parameter to `train_hmm()`
- Filter: `train_df = df[df.index < prediction_date]`
- This ensures the model only sees historical data

---

### 3. Prediction Flow

**Current Behavior:**
```
Run Prediction clicked
    ↓
ModelController.predict_latest(features_df)
    ↓
Line 237: latest = features_df.iloc[-1]  # ALWAYS last row
Line 259: 'date': features_df.index[-1]  # ALWAYS last date
    ↓
Returns prediction for "tomorrow" (implied)
```

**Key Files:**
| File | Lines | Current Issue |
|------|-------|---------------|
| `src/ui/model_controller.py` | 188-276 | `predict_latest()` hardcoded to use `iloc[-1]` |
| `src/models/hmm_volatility.py` | 255-290 | `predict_latest()` returns last row predictions |

**The Fix:**
- Rename to `predict_for_date(df, prediction_date=None)`
- If `prediction_date` is None, use current behavior (latest)
- Otherwise, locate specific row and predict for that date

---

### 4. UI Architecture

**Current Sidebar Structure:**
```
⚙️ Controls
├── 📊 Data Settings
│   ├── History Days Slider (30-365)
│   └── Load/Refresh Data Button
├── 🎯 HMM Parameters
│   ├── Number of Regimes (2-5)
│   ├── Training Iterations (50-500)
│   └── Train HMM Button
├── 🔮 TimesFM Parameters
│   └── [Enable, Load TimesFM]
├── ⚙️ Model Configuration
│   ├── Confidence Threshold
│   └── Advanced Weights
└── 🚀 Actions
    └── Run Prediction Button
```

**Required Addition (before History Days):**
```
📊 Data Settings
├── 🎯 Prediction Date   ← NEW DATE PICKER
├── History Days Slider
└── Load/Refresh Data Button
```

---

## Why This Matters: Preventing Data Leakage

### The Leakage Problem

Without a prediction date, users can unknowingly:
1. Train on data that includes the day they're "predicting"
2. Use future volatility patterns to predict past regimes
3. Get artificially high accuracy that won't hold in production

**Example of Leakage:**
```
Today: 2026-01-24
Load 180 days of data: 2025-07-28 to 2026-01-24

User wants to "predict" for 2026-01-15 (9 days ago)

Current behavior:
- Trains HMM on data including 2026-01-15 through 2026-01-24
- Model has SEEN the volatility regime from 2026-01-15 onward
- "Prediction" is just recalling what already happened

Correct behavior:
- Load data from 2025-07-09 to 2026-01-14 (day BEFORE prediction)
- Train HMM on 2025-07-09 to 2026-01-14 only
- Predict for 2026-01-15 using only prior information
- Compare to actual 2026-01-15 outcome
```

---

## Proposed Architecture

### New Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. USER SELECTS PREDICTION DATE                                │
│                                                                 │
│    st.date_input("Prediction Date")                            │
│    → prediction_date = 2026-01-15                              │
│                                                                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│ 2. DATA LOADING (Relative to Prediction Date)                  │
│                                                                 │
│    history_days = 180                                          │
│    end_date = prediction_date - 1 day = 2026-01-14            │
│    start_date = end_date - 180 days = 2025-07-18              │
│                                                                 │
│    load_complete_dataset(days=180, end_date=end_date)         │
│    → Returns data from 2025-07-18 to 2026-01-14               │
│    → NO DATA from prediction date onward                       │
│                                                                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│ 3. MODEL TRAINING (On Historical Data Only)                    │
│                                                                 │
│    train_hmm(df=features_df, prediction_date=prediction_date)  │
│    → Internally: train_df = df[df.index < prediction_date]    │
│    → Trains on 2025-07-18 to 2026-01-14                        │
│    → Stores training_date_range for validation                 │
│                                                                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│ 4. PREDICTION (For Specific Date)                              │
│                                                                 │
│    predict_for_date(features_df, prediction_date)              │
│    → Uses features from day BEFORE prediction date             │
│    → Predicts regime for prediction_date                       │
│    → Returns: "Prediction for 2026-01-15"                     │
│                                                                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│ 5. VALIDATION (Compare to Actual)                              │
│                                                                 │
│    If prediction_date is in the past:                          │
│    → Show actual outcome for 2026-01-15                        │
│    → Display: "Predicted HIGH_VOL, Actual: HIGH_VOL ✓"        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Session State & UI (app.py, utils.py)

**1.1 Add Session State Variables** (`src/ui/utils.py:26-48`)
```python
defaults = {
    # ... existing ...
    'prediction_date': None,           # Selected prediction date
    'use_latest_date': True,           # Flag for "predict for tomorrow" mode
    'available_date_range': None,      # (min_date, max_date) from loaded data
}
```

**1.2 Add Date Picker to Sidebar** (`src/ui/app.py:~137`)
```python
# Prediction Date Settings
st.sidebar.header("📊 Data Settings")

# Mode toggle
use_latest = st.sidebar.checkbox(
    "Predict for Tomorrow (Latest)",
    value=True,
    help="Uncheck to select a specific historical date"
)
st.session_state.use_latest_date = use_latest

if not use_latest:
    prediction_date = st.sidebar.date_input(
        "Prediction Date",
        value=datetime.now().date() - timedelta(days=1),
        max_value=datetime.now().date(),
        help="Select the date you want to predict volatility for"
    )
    st.session_state.prediction_date = pd.Timestamp(prediction_date)
else:
    st.session_state.prediction_date = None
```

---

### Phase 2: Data Loading Changes (data_manager.py)

**2.1 Update `load_complete_dataset()`**
```python
def load_complete_dataset(
    self,
    days: int = 180,
    end_date: Optional[datetime] = None  # NEW PARAMETER
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load complete dataset ending at specified date.

    Args:
        days: Number of historical days to load
        end_date: Load data UP TO this date (None = today)
    """
    spy_data = self.load_spy_data(days, end_date=end_date)
    vix_data = self.load_vix_data(days, end_date=end_date)
    features_df = self.engineer_features(spy_data, vix_data)
    return spy_data, vix_data, features_df
```

**2.2 Update `load_spy_data()`**
```python
def load_spy_data(self, days: int = 180, end_date: Optional[datetime] = None) -> pd.DataFrame:
    logger.info(f"Loading {days} days of SPY data ending at {end_date or 'now'}")
    spy_daily = self.alpaca_client.get_daily_bars('SPY', days=days, end_date=end_date)
    return spy_daily
```

**2.3 Update `load_vix_data()`**
```python
def load_vix_data(self, days: int = 180, end_date: Optional[datetime] = None) -> pd.DataFrame:
    import yfinance as yf

    if end_date is None:
        end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # Use explicit date range instead of period
    vix_daily = yf.download(
        '^VIX',
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        progress=False
    )
    return vix_daily
```

---

### Phase 3: Model Training Changes (model_controller.py)

**3.1 Update `train_hmm()`**
```python
def train_hmm(
    self,
    df: pd.DataFrame,
    prediction_date: Optional[pd.Timestamp] = None,  # NEW
    n_regimes: int = 3,
    n_iter: int = 100
) -> Tuple[VolatilityHMM, Dict]:
    """
    Train HMM on data BEFORE prediction_date.

    Args:
        df: Full features DataFrame
        prediction_date: Train only on data before this date (None = use all)
    """
    # Apply temporal cutoff if prediction_date specified
    if prediction_date is not None:
        train_df = df[df.index < prediction_date]
        logger.info(f"Training on {len(train_df)} rows before {prediction_date}")
        if len(train_df) < 50:
            raise ValueError(f"Insufficient data before {prediction_date} (need 50+, have {len(train_df)})")
    else:
        train_df = df

    # Create and train model
    hmm_model = VolatilityHMM(n_regimes=n_regimes)
    metrics = hmm_model.train(train_df, n_iter=n_iter)

    # Store metadata
    st.session_state.hmm_model = hmm_model
    st.session_state.hmm_metrics = metrics
    st.session_state.hmm_training_date_range = (train_df.index[0], train_df.index[-1])
    st.session_state.hmm_training_cutoff = prediction_date  # NEW

    return hmm_model, metrics
```

---

### Phase 4: Prediction Changes (model_controller.py)

**4.1 Update `predict_latest()` to `predict_for_date()`**
```python
def predict_for_date(
    self,
    features_df: pd.DataFrame,
    prediction_date: Optional[pd.Timestamp] = None
) -> Dict:
    """
    Generate prediction for a specific date or the latest.

    Args:
        features_df: Full features DataFrame
        prediction_date: Date to predict for (None = latest/tomorrow)
    """
    if st.session_state.hmm_model is None:
        raise ValueError("HMM model not trained. Train model first.")

    # Determine which row to use for prediction
    if prediction_date is None:
        # Current behavior - use last row
        target_idx = -1
        target_date = features_df.index[-1]
        prediction_label = "Tomorrow"
    else:
        # Find the row for the day BEFORE prediction date
        # (we predict using yesterday's features)
        prior_date = prediction_date - pd.Timedelta(days=1)

        if prior_date not in features_df.index:
            # Find closest prior date
            valid_dates = features_df.index[features_df.index < prediction_date]
            if len(valid_dates) == 0:
                raise ValueError(f"No data available before {prediction_date}")
            prior_date = valid_dates[-1]

        target_idx = features_df.index.get_loc(prior_date)
        target_date = prediction_date
        prediction_label = prediction_date.strftime('%Y-%m-%d')

    # Get context data (everything up to and including target row)
    context_df = features_df.iloc[:target_idx + 1]

    # Get HMM prediction on context
    hmm_prediction = st.session_state.hmm_model.predict_latest(context_df)

    # Extract feature signals from target row
    target_row = features_df.iloc[target_idx]
    feature_signals = {
        'overnight_gap_abs': target_row['overnight_gap_abs'],
        'vix_change_1d': target_row['vix_change_1d'],
        'vix_level': target_row['vix_level'],
        'range_expansion': target_row['range_expansion'],
        'volume_surge': target_row['volume_surge'],
        'volume_ratio': target_row['volume_ratio'],
        'high_range_days_5': target_row['high_range_days_5']
    }

    # Calculate confidence score
    scorer = st.session_state.confidence_scorer
    score = scorer.calculate_score(
        regime_volatility=hmm_prediction['expected_volatility'],
        regime_label=hmm_prediction['regime_label'],
        timesfm_forecast=None,  # Would need similar update for TimesFM
        feature_signals=feature_signals
    )

    return {
        'date': target_date,
        'prediction_label': prediction_label,  # NEW: "Tomorrow" or "2026-01-15"
        'regime_label': hmm_prediction['regime_label'],
        'regime_volatility': hmm_prediction['expected_volatility'],
        'regime_probabilities': hmm_prediction['regime_probabilities'],
        'regime_confidence': hmm_prediction['confidence'],
        'timesfm_forecast': None,
        'confidence_score': score.total_score,
        'confidence_breakdown': {
            'regime_score': score.regime_score,
            'timesfm_score': score.timesfm_score,
            'feature_score': score.feature_score
        },
        'should_trade': score.total_score >= scorer.threshold,
        'recommendation': score.recommendation,
        'explanation': score.explanation,
        'feature_signals': feature_signals
    }
```

---

### Phase 5: UI Display Updates (app.py)

**5.1 Update Load Data Button**
```python
if st.sidebar.button("🔄 Load/Refresh Data", type="primary", use_container_width=True):
    with st.spinner("Loading data..."):
        try:
            # Determine end_date based on prediction mode
            if st.session_state.use_latest_date:
                end_date = None  # Will use today
            else:
                # Load data up to day BEFORE prediction date
                end_date = st.session_state.prediction_date - pd.Timedelta(days=1)

            spy_data, vix_data, features_df = st.session_state.data_manager.load_complete_dataset(
                history_days,
                end_date=end_date  # NEW PARAMETER
            )
            # ... rest of existing code ...
```

**5.2 Update Train HMM Button**
```python
if st.sidebar.button("🚀 Train HMM", use_container_width=True):
    if not validate_data_loaded():
        return

    with st.spinner("Training HMM model..."):
        try:
            model, metrics = st.session_state.model_controller.train_hmm(
                df=st.session_state.features_df,
                prediction_date=st.session_state.prediction_date,  # NEW
                n_regimes=n_regimes,
                n_iter=training_iterations
            )
            # ... rest of existing code ...
```

**5.3 Update Run Prediction Button**
```python
if st.sidebar.button("🔮 Run Prediction", type="primary", use_container_width=True):
    if not validate_data_loaded() or not validate_model_trained():
        return

    with st.spinner("Generating prediction..."):
        try:
            prediction = st.session_state.model_controller.predict_for_date(
                st.session_state.features_df,
                prediction_date=st.session_state.prediction_date  # NEW
            )
            st.session_state.last_prediction = prediction
            st.sidebar.success(f"✅ Prediction for {prediction['prediction_label']}")
```

**5.4 Update Prediction Dashboard Display**
```python
def render_prediction_tab():
    st.header("📈 Prediction Dashboard")

    prediction = st.session_state.get('last_prediction')

    if prediction is None:
        st.info("ℹ️ No prediction available...")
        return

    # Show prediction target
    st.subheader(f"Predicting for: {prediction['prediction_label']}")

    # If historical date, show actual outcome
    if prediction['prediction_label'] != "Tomorrow":
        actual_date = prediction['date']
        # Check if we have actual data for this date
        if actual_date in st.session_state.features_df.index:
            actual_volatility = st.session_state.features_df.loc[actual_date, 'intraday_range_pct']
            actual_high_vol = actual_volatility >= 0.012  # threshold
            predicted_high_vol = prediction['regime_label'] == 'high_vol'

            if predicted_high_vol == actual_high_vol:
                st.success(f"✅ Prediction CORRECT! (Actual volatility: {actual_volatility:.2%})")
            else:
                st.error(f"❌ Prediction INCORRECT (Actual volatility: {actual_volatility:.2%})")

    # ... rest of existing display code ...
```

---

## Files to Modify Summary

| File | Changes Required | Priority |
|------|------------------|----------|
| `src/ui/utils.py` | Add 3 session state variables | P1 |
| `src/ui/app.py` | Add date picker, update buttons, update displays | P1 |
| `src/ui/data_manager.py` | Add `end_date` parameter to 3 methods | P1 |
| `src/ui/model_controller.py` | Add `prediction_date` to train & predict | P1 |
| `src/models/hmm_volatility.py` | No changes needed (already flexible) | - |
| `src/models/timesfm_volatility.py` | Add date-aware prediction (optional) | P2 |

---

## Testing Checklist

After implementation:

- [ ] Select "Predict for Tomorrow" → behavior unchanged from current
- [ ] Select historical date (e.g., 2026-01-15) → loads data before that date
- [ ] Train HMM → only uses data before prediction date
- [ ] Run Prediction → shows "Predicting for 2026-01-15"
- [ ] Historical prediction → shows actual outcome comparison
- [ ] Verify no data leakage by checking training date range
- [ ] Test edge cases: earliest available date, date outside data range

---

## Conclusion

This architecture ensures:

1. **No Data Leakage**: Training only uses data before prediction date
2. **Point-in-Time Accuracy**: Users can backtest any historical date
3. **Backward Compatibility**: "Predict for Tomorrow" mode works exactly as before
4. **Validation Feedback**: Historical predictions show actual outcomes

The implementation touches 4-5 files with targeted changes, maintaining the existing code structure while adding the new capability.
