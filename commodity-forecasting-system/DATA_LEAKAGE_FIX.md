# DATA LEAKAGE FIX - Critical Temporal Validation

## Problem Discovered: 2026-01-25

### The Issue
Historical predictions in the UI were producing **opposite results** from the validation script for the same date (2026-01-16):

| Component | Regime Prediction | Expected Vol | Actual Vol |
|-----------|------------------|--------------|------------|
| **UI (WRONG)** | extreme_vol | 0.91% | 0.598% |
| **Validation (CORRECT)** | very_low_vol | ~0.6% | 0.598% |

### Root Cause: Look-Ahead Bias (Data Leakage)

**Broken Workflow:**
```
1. Load data (180 days, ending 2026-01-25)
2. Train HMM → Model trained on ALL 180 days ⚠️
3. Select historical date (2026-01-16)
4. Run Prediction → Model has "seen the future"!
```

The model was trained on data that **included 9 days AFTER** the prediction target, creating temporal leakage.

### The Fix: Temporal Cutoff Validation

**Location:** `src/ui/model_controller.py:241-284`

Added validation that:
1. Detects when `prediction_date` is set (historical mode)
2. Checks if model was trained with proper `hmm_training_cutoff`
3. Raises clear error if model saw future data
4. Provides step-by-step fix instructions

**Error Messages Now Appear When:**
- Model trained without cutoff (on all data)
- Model cutoff > prediction date (trained on future data)

### Correct Workflow for Historical Predictions

```
✅ CORRECT ORDER:
1. Uncheck "Predict for Tomorrow (Latest)"
2. Select Prediction Date (e.g., 2026-01-16)
3. Click "Load/Refresh Data" (optional - loads data up to that date)
4. Click "Train HMM" (trains ONLY on data before 2026-01-16)
5. Click "Run Prediction" (uses correctly-trained model)

❌ WRONG ORDER:
1. Load Data
2. Train HMM (trains on ALL data)
3. Select Prediction Date
4. Run Prediction → DATA LEAKAGE ERROR!
```

### Technical Details

**Training with Temporal Cutoff:**
```python
# model_controller.py:85-97
if prediction_date is not None:
    train_df = df[df.index < prediction_date]
    logger.info(f"Temporal cutoff applied: training on {len(train_df)} rows before {prediction_date}")
```

**Validation Check:**
```python
# model_controller.py:241-284
if prediction_date is not None:
    model_cutoff = st.session_state.get('hmm_training_cutoff')

    if model_cutoff is None or model_cutoff > prediction_date:
        raise ValueError("⚠️ DATA LEAKAGE DETECTED! ...")
```

**Metadata Tracking:**
```python
# model_controller.py:112-117
st.session_state.hmm_training_cutoff = prediction_date
st.session_state.hmm_training_date_range = (train_df.index[0], train_df.index[-1])
st.session_state.hmm_training_rows = len(train_df)
```

### Why This Matters

**Without temporal validation:**
- Historical backtests are invalid (model saw the future)
- Win rates appear artificially high
- Strategy would fail in live trading
- No way to detect the issue

**With temporal validation:**
- Error raised immediately when detected
- Clear instructions to fix the workflow
- Ensures proper walk-forward methodology
- Matches validation script behavior

### Testing the Fix

**Expected Behavior:**
1. Load data ending 2026-01-25
2. Train HMM without selecting prediction date → Trains on all data
3. Select prediction date 2026-01-16
4. Click "Run Prediction" → **ERROR RAISED** ✅
5. Click "Train HMM" again → Retrains with cutoff
6. Click "Run Prediction" → **SUCCESS** ✅

**Error Message Example:**
```
⚠️ DATA LEAKAGE DETECTED!

MODEL TRAINING MISMATCH:
  • Model was trained on data up to: 2026-01-25
  • You are predicting for: 2026-01-16

The model was trained on data AFTER your prediction target!
This creates look-ahead bias - the model has 'seen the future'.

EXAMPLE OF THE PROBLEM:
  You're asking: 'What will happen on Jan 16?'
  But the model was trained on data through Jan 25!
  It already knows what happened on Jan 16-25.

TO FIX THIS:
1. Keep your prediction date set to 2026-01-16
2. Click '🚀 Train HMM' to retrain with correct temporal cutoff
3. Then click '🔮 Run Prediction' again

This is why your UI prediction (extreme_vol) differs from validation (very_low_vol).
The validation script trains correctly; the UI had stale model from wrong time period.
```

## Files Modified

1. **`src/ui/model_controller.py`**
   - Added data leakage detection in `predict_for_date()` (lines 241-284)
   - Validates `hmm_training_cutoff` matches `prediction_date`
   - Provides actionable error messages

## Related Files

- **`scripts/validate_volatility_mvp.py`** - Reference implementation (correct behavior)
- **`src/ui/app.py`** - UI controls (already passes prediction_date correctly)
- **`CLAUDE.md`** - Updated with this critical fix

## Prevention

This fix ensures:
- ✅ Historical predictions use proper temporal cutoffs
- ✅ Walk-forward validation matches UI behavior
- ✅ No accidental look-ahead bias
- ✅ Clear error messages guide users to correct workflow
- ✅ Production trading won't be based on leaked data

## Date: 2026-01-25
## Status: ✅ FIXED AND VALIDATED
