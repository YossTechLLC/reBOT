# Architecture Changes Checklist
**Date:** 2026-01-24
**Status:** ✅ COMPLETE (Phase 1-3) | 🔄 NEW: Phase 4 Added

---

## Summary of Requested Changes

| # | Feature | Complexity | Files Affected |
|---|---------|------------|----------------|
| 1 | Support 2-5 Regimes Correctly | Medium | hmm_volatility.py, confidence_scorer.py, utils.py |
| 2 | Fix Date Picker Off-by-One | Simple | app.py (3 lines) |
| 3 | HMM Feature Selection UI | Medium | app.py, model_controller.py, hmm_volatility.py |
| 4 | Feature Scaling Bounds UI | Medium | hmm_volatility.py, app.py, utils.py |

---

## Issue 1: Support 2-5 Regimes Correctly

### Current Problem
```python
# hmm_volatility.py:179
labels = ['low_vol', 'normal_vol', 'high_vol'][:self.n_regimes]
```

This slices from a fixed 3-element list:
- 2 regimes: `['low_vol', 'normal_vol']` ✓ works
- 3 regimes: `['low_vol', 'normal_vol', 'high_vol']` ✓ works
- 4 regimes: ✗ IndexError (only 3 labels available)
- 5 regimes: ✗ IndexError

### Solution: Dynamic Label Generation

**File: `src/models/hmm_volatility.py`**

Replace hardcoded labels with dynamic generation:

```python
def _get_regime_labels(self, n_regimes: int) -> list:
    """Generate regime labels based on number of regimes."""
    label_sets = {
        2: ['low_vol', 'high_vol'],
        3: ['low_vol', 'normal_vol', 'high_vol'],
        4: ['very_low_vol', 'low_vol', 'normal_vol', 'high_vol'],
        5: ['very_low_vol', 'low_vol', 'normal_vol', 'high_vol', 'extreme_vol']
    }
    return label_sets.get(n_regimes, [f'regime_{i}' for i in range(n_regimes)])
```

**File: `src/volatility/confidence_scorer.py`**

Update `_calculate_regime_score()` to handle new labels:

```python
# Current (lines 186-189):
if regime_label == 'high_vol':
    score = min(score * 1.1, 100)
elif regime_label == 'explosive_vol':
    score = min(score * 1.2, 100)

# Updated:
if regime_label in ['high_vol', 'extreme_vol']:
    score = min(score * 1.1, 100)
elif regime_label == 'extreme_vol':
    score = min(score * 1.2, 100)
```

**File: `src/ui/utils.py`**

Update `get_color_for_regime()` to handle new labels:

```python
colors = {
    'very_low_vol': '#B0E0B0',  # Pale green
    'low_vol': '#90EE90',       # Light green
    'normal_vol': '#FFD700',    # Gold
    'high_vol': '#FF6347',      # Tomato red
    'extreme_vol': '#DC143C'    # Crimson
}
```

**File: `src/ui/app.py`**

Remove the slider max_value restriction (currently 3, should be 5):

```python
n_regimes = st.sidebar.slider(
    "Number of Regimes",
    min_value=2,
    max_value=5,  # Restore full range
    value=DEFAULT_HMM_REGIMES,
    ...
)
```

### Changes Summary
| File | Change |
|------|--------|
| `hmm_volatility.py` | Add `_get_regime_labels()` method, update `_learn_regime_mappings()` |
| `confidence_scorer.py` | Update regime label handling in `_calculate_regime_score()` |
| `utils.py` | Add colors for new regime labels |
| `app.py` | Change slider max_value from 3 to 5 |

---

## Issue 2: Fix Date Picker Off-by-One

### Current Problem

User selects **1/16/2026** → Prediction shows **1/15/2026**

**Root cause:** `max_date` is set to the last date in loaded data. But we want to predict FOR tomorrow using today's features.

```python
# app.py:150
max_date = st.session_state.features_df.index[-1].to_pydatetime().date()
```

If data ends at 1/15, user can only select up to 1/15. But the semantics are:
- User selects 1/16 = "I want to predict what happens on 1/16"
- System uses 1/15 features to predict 1/16 regime

### Solution: Extend max_date by 1 day

**File: `src/ui/app.py`**

```python
# BEFORE (line 150):
max_date = st.session_state.features_df.index[-1].to_pydatetime().date()

# AFTER:
data_end = st.session_state.features_df.index[-1].to_pydatetime().date()
max_date = data_end + timedelta(days=1)  # Allow predicting for day AFTER data ends
```

This is a **3-line change** (including import adjustment if needed).

### Changes Summary
| File | Change |
|------|--------|
| `app.py` | Line ~150: `max_date = data_end + timedelta(days=1)` |

---

## Issue 3: HMM Feature Selection UI

### Current State
Features are hardcoded in `hmm_volatility.py`:

```python
required_cols = [
    'overnight_gap_abs',
    'range_ma_5',
    'vix_level',
    'volume_ratio',
    'range_std_5'
]
```

### Solution: Configurable Features via UI

#### Phase 1: Session State (utils.py)

Add to `initialize_session_state()`:

```python
'hmm_features': [
    'overnight_gap_abs',
    'range_ma_5',
    'vix_level',
    'volume_ratio',
    'range_std_5'
],  # Default features
```

#### Phase 2: Model Update (hmm_volatility.py)

Update `__init__` and `prepare_features`:

```python
def __init__(self, n_regimes: int = 3, features: list = None, random_state: int = 42):
    """
    Args:
        features: List of feature column names to use (default: standard 5)
    """
    self.features = features or [
        'overnight_gap_abs',
        'range_ma_5',
        'vix_level',
        'volume_ratio',
        'range_std_5'
    ]
    ...

def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.Series]:
    required_cols = self.features  # Use instance features instead of hardcoded
    ...
```

#### Phase 3: Controller Update (model_controller.py)

Pass features to HMM:

```python
def train_hmm(
    self,
    df: pd.DataFrame,
    n_regimes: int = 3,
    features: List[str] = None,  # Add parameter
    n_iter: int = 100,
    prediction_date: Optional[pd.Timestamp] = None
) -> Tuple[VolatilityHMM, Dict]:
    ...
    hmm_model = VolatilityHMM(n_regimes=n_regimes, features=features)
    ...
```

#### Phase 4: UI Sidebar (app.py)

Add feature selection panel:

```python
# HMM Feature Selection (collapsible)
with st.sidebar.expander("🔧 HMM Features", expanded=False):
    available_features = [
        'overnight_gap_abs',
        'range_ma_5',
        'vix_level',
        'volume_ratio',
        'range_std_5',
        # Additional available features from features_df
        'vix_change_1d',
        'range_expansion',
        'volume_surge',
        'high_range_days_5'
    ]

    selected_features = st.multiselect(
        "Select Features",
        options=available_features,
        default=st.session_state.get('hmm_features', available_features[:5]),
        help="Features used by HMM for regime detection"
    )

    if len(selected_features) < 2:
        st.warning("Select at least 2 features")
    else:
        st.session_state.hmm_features = selected_features
```

### Changes Summary
| File | Change |
|------|--------|
| `utils.py` | Add `hmm_features` to session state defaults |
| `hmm_volatility.py` | Add `features` parameter to `__init__`, update `prepare_features` |
| `model_controller.py` | Pass `features` parameter to HMM constructor |
| `app.py` | Add feature selection expander in sidebar, pass to train_hmm() |

---

## Implementation Order

1. **Issue 2 (Date Fix)** - Simplest, 3 lines, no refactoring
2. **Issue 1 (Regimes)** - Medium, 4 files, focused changes
3. **Issue 3 (Features UI)** - Medium, 4 files, requires testing

---

## Refactoring Principles Applied

1. **No code bloat**: Each change is minimal and focused
2. **Backward compatible**: Default behavior unchanged
3. **Single responsibility**: Each function does one thing
4. **No duplication**: Reuse existing patterns (session state, sidebar controls)

---

## Testing Plan

### Issue 1 Tests
```
1. Train with 2 regimes → should show low_vol, high_vol
2. Train with 3 regimes → should show low_vol, normal_vol, high_vol
3. Train with 4 regimes → should show very_low_vol, low_vol, normal_vol, high_vol
4. Train with 5 regimes → should show all 5 labels
5. P&L visualization should work for all regime counts
```

### Issue 2 Tests
```
1. Load data ending 1/15
2. Date picker should allow selecting 1/16 (max_date)
3. Select 1/16 → prediction shows "Predicting for: 2026-01-16"
4. Prediction uses 1/15 features (correctly, no leakage)
```

### Issue 3 Tests
```
1. Default features should be pre-selected
2. Deselecting a feature and training → HMM uses reduced feature set
3. Select <2 features → warning shown, training blocked
4. Verify model retraining picks up new feature selection
```

---

## Questions for User

1. **Regime Labels**: Are the proposed 4/5 regime labels acceptable?
   - 4: very_low_vol, low_vol, normal_vol, high_vol
   - 5: very_low_vol, low_vol, normal_vol, high_vol, extreme_vol

2. **Feature Defaults**: Should additional features be available in the selector beyond the 5 core ones?

3. **Priority**: Should I implement in the order listed (2 → 1 → 3)?

---

## Approval Checklist

- [ ] Issue 1: Regime support approach approved
- [ ] Issue 2: Date picker fix approved
- [ ] Issue 3: Feature selection UI approach approved
- [ ] Implementation order confirmed
