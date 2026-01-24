# BUGFIX INVESTIGATION: Missing Target Column in HMM Training

**Date:** 2026-01-17
**Error:** `KeyError: 'intraday_range_pct'`
**Status:** 🔍 ROOT CAUSE IDENTIFIED

---

## 🐛 ERROR DETAILS

**Error Message:**
```
KeyError: 'intraday_range_pct'
```

**Error Location:**
- File: `src/models/hmm_volatility.py`
- Line: 97
- Function: `prepare_features()`
- Code: `y = df['intraday_range_pct']`

**Error Context:**
User clicked "Train HMM" button in UI → `model_controller.train_hmm()` called → `hmm_model.train()` called → `prepare_features()` failed

---

## 🔍 ROOT CAUSE ANALYSIS

### Execution Flow Trace

**Step 1: UI Calls train_hmm() - app.py Line 196**
```python
model, metrics = st.session_state.model_controller.train_hmm(
    df=st.session_state.features_df,  # Full DataFrame with ALL features
    n_regimes=n_regimes,
    features=hmm_features,  # ← This is HMM_DEFAULT_FEATURES (5 columns)
    n_iter=training_iterations
)
```

**HMM_DEFAULT_FEATURES contains:** (utils.py line 344-350)
```python
HMM_DEFAULT_FEATURES = [
    'overnight_gap_abs',
    'range_ma_5',
    'vix_level',
    'volume_ratio',
    'range_std_5'
]
# NOTE: Does NOT include 'intraday_range_pct'
```

---

**Step 2: model_controller.train_hmm() Subsets Features - Lines 86-91**
```python
# Use specified features or model defaults
if features:
    train_data = df[features]  # ← PROBLEM: Selects ONLY these 5 columns
else:
    train_data = df  # Would work fine if features=None

metrics = hmm_model.train(train_data, n_iter=n_iter)
```

**Result:** `train_data` now contains ONLY 5 columns, missing `'intraday_range_pct'`

---

**Step 3: hmm_model.train() Calls prepare_features() - Line 122**
```python
def train(self, df: pd.DataFrame, n_iter: int = 100, tol: float = 1e-4) -> Dict:
    # Prepare features
    X, y = self.prepare_features(df)  # ← df only has 5 columns
```

---

**Step 4: prepare_features() Tries to Extract Target - Line 97**
```python
def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.Series]:
    required_cols = [
        'overnight_gap_abs',
        'range_ma_5',
        'vix_level',
        'volume_ratio',
        'range_std_5'
    ]

    # Extract features (these exist - no problem)
    X = df[required_cols].values

    # Target: intraday volatility
    y = df['intraday_range_pct']  # ← CRASH! Column doesn't exist in subset
```

**Why it crashes:**
- The DataFrame passed to `prepare_features()` only contains the 5 feature columns
- The method expects a 6th column (`intraday_range_pct`) to use as the target
- KeyError is raised because the column is missing

---

## 🎯 ROOT CAUSE SUMMARY

**The Mismatch:**
1. UI passes a list of **feature** columns (5 columns) to `train_hmm()`
2. `train_hmm()` subsets the DataFrame to ONLY those columns: `df[features]`
3. HMM's `prepare_features()` expects the **target** column to also be present
4. Target column (`intraday_range_pct`) is NOT in the feature list
5. KeyError occurs when trying to access the missing target column

**Design Flaw:**
- The `features` parameter in `train_hmm()` is ambiguous
- Should it include the target or not?
- Current implementation assumes it does NOT (line 86-87)
- But HMM implementation assumes it DOES (line 97)

---

## 🔧 AFFECTED CODE SECTIONS

### 1. model_controller.py - train_hmm() Lines 86-91

**Current (Broken):**
```python
# Use specified features or model defaults
if features:
    train_data = df[features]  # ← Missing target column
else:
    train_data = df

metrics = hmm_model.train(train_data, n_iter=n_iter)
```

**Issue:** When `features` is provided, the subset doesn't include the target column.

---

### 2. hmm_volatility.py - prepare_features() Line 97

**Current:**
```python
# Target: intraday volatility
y = df['intraday_range_pct']  # ← Expects column to exist
```

**Issue:** Assumes target column is in the DataFrame, but caller may not include it.

---

### 3. utils.py - HMM_DEFAULT_FEATURES Lines 344-350

**Current:**
```python
HMM_DEFAULT_FEATURES = [
    'overnight_gap_abs',
    'range_ma_5',
    'vix_level',
    'volume_ratio',
    'range_std_5'
]
```

**Issue:** List doesn't include the target column `'intraday_range_pct'`.

---

## ✅ RESOLUTION STRATEGY

### Option 1: Include Target in Feature Subset (RECOMMENDED)

**Change:** Modify `model_controller.train_hmm()` to always include the target column when subsetting.

**Pros:**
- Minimal code change (1 line)
- Preserves existing HMM API (no changes to hmm_volatility.py)
- Feature selection still works as expected
- Defensive - handles edge cases

**Cons:**
- None

**Implementation:**
```python
# In model_controller.py, lines 86-91
if features:
    # Ensure target column is included for HMM training
    feature_cols = list(features)
    if 'intraday_range_pct' not in feature_cols:
        feature_cols.append('intraday_range_pct')
    train_data = df[feature_cols]
else:
    train_data = df

metrics = hmm_model.train(train_data, n_iter=n_iter)
```

**Risk:** LOW - Isolated change, backwards compatible

---

### Option 2: Modify HMM API to Accept Target Separately

**Change:** Modify `hmm_volatility.py` to accept target as a separate parameter.

**Pros:**
- Clear separation of features vs target
- More explicit API
- Prevents this class of bugs

**Cons:**
- Requires changes to HMM class (multiple methods)
- Requires changes to all HMM callers
- More complex migration
- Breaks existing code patterns

**Implementation:**
```python
# In hmm_volatility.py
def prepare_features(
    self,
    df: pd.DataFrame,
    target_col: str = 'intraday_range_pct'
) -> Tuple[np.ndarray, pd.Series]:
    # ... extract features ...
    y = df[target_col]
    return X, y

def train(
    self,
    df: pd.DataFrame,
    target_col: str = 'intraday_range_pct',
    n_iter: int = 100,
    tol: float = 1e-4
) -> Dict:
    X, y = self.prepare_features(df, target_col)
    # ... rest of training ...
```

**Risk:** MEDIUM - Requires changes across multiple files, potential for regressions

---

### Option 3: Don't Subset Features

**Change:** Remove feature subsetting entirely, pass full DataFrame.

**Pros:**
- Simplest fix
- No risk of missing columns

**Cons:**
- Loses feature selection capability
- User cannot control which features are used
- Defeats purpose of `features` parameter
- Less flexible

**Implementation:**
```python
# In model_controller.py, lines 86-91
# Always use full DataFrame
train_data = df
metrics = hmm_model.train(train_data, n_iter=n_iter)
```

**Risk:** LOW - But removes useful functionality

---

### Option 4: Add Target to Feature List in UI

**Change:** Update `HMM_DEFAULT_FEATURES` to include the target.

**Pros:**
- Simple fix
- No code changes needed

**Cons:**
- Confusing - mixing features and target
- Not semantically correct
- Doesn't prevent future bugs (user could deselect it)
- Brittle solution

**Implementation:**
```python
# In utils.py
HMM_DEFAULT_FEATURES = [
    'overnight_gap_abs',
    'range_ma_5',
    'vix_level',
    'volume_ratio',
    'range_std_5',
    'intraday_range_pct'  # ← Add target to feature list (confusing!)
]
```

**Risk:** LOW - But semantically incorrect

---

## 📋 RECOMMENDED SOLUTION: Option 1

**Rationale:**
1. **Minimal Impact:** Single location change in `model_controller.py`
2. **Defensive:** Handles both cases (target in list or not)
3. **Preserves Functionality:** User can still select features
4. **No API Changes:** HMM class remains unchanged
5. **Backwards Compatible:** Existing code continues to work
6. **Clear Intent:** Comment explains why target is added

**Changes Required:**
1. Update `model_controller.train_hmm()` - Add 4 lines (lines 86-91)
2. Test with UI (verify training works)
3. Document the pattern

---

## 🧪 TESTING PLAN

### Test Case 1: UI Feature Selection
**Steps:**
1. Launch UI
2. Load data
3. Select all 5 default HMM features
4. Click "Train HMM"

**Expected:**
- ✅ Training completes successfully
- ✅ No KeyError
- ✅ Metrics displayed

---

### Test Case 2: Custom Feature Selection
**Steps:**
1. Launch UI
2. Load data
3. Select only 3 features (e.g., overnight_gap_abs, vix_level, range_ma_5)
4. Click "Train HMM"

**Expected:**
- ✅ Training completes successfully
- ✅ Only selected features used for X matrix
- ✅ Target column automatically included

---

### Test Case 3: No Feature Selection (Use Defaults)
**Steps:**
1. Launch UI
2. Load data
3. Don't modify feature multiselect
4. Click "Train HMM"

**Expected:**
- ✅ Training completes successfully
- ✅ All default features used

---

### Test Case 4: Programmatic Call (Non-UI)
**Steps:**
1. Create test script
2. Call `model_controller.train_hmm(df, features=['vix_level'])`

**Expected:**
- ✅ Training completes successfully
- ✅ Target column automatically included

---

## 🎓 LESSONS LEARNED

### API Design Patterns

**Anti-Pattern:**
```python
def train(df, features):
    subset = df[features]  # Assumes features is complete
    # ... but later code expects additional columns
```

**Better Pattern:**
```python
def train(df, features):
    # Be explicit about dependencies
    required_cols = list(features) + ['target_col']
    subset = df[required_cols]
```

**Best Pattern:**
```python
def train(df, features, target_col='target'):
    # Separate features from target
    X = df[features]
    y = df[target_col]
```

---

### Feature vs Target Separation

**Lesson:** Always separate feature columns from target column in ML APIs.

**Why:**
- Features: Input variables (X)
- Target: Output variable (y)
- They serve different purposes
- Mixing them creates confusion

**Apply to:**
- All ML model APIs
- Feature engineering pipelines
- Data validation functions

---

### Defensive Programming

**Lesson:** Validate assumptions at boundaries.

**Example:**
```python
def prepare_features(df):
    required_cols = ['feat1', 'feat2', 'target']

    # Validate EARLY
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Now safe to proceed
    X = df[['feat1', 'feat2']].values
    y = df['target']
```

---

## 📊 VERIFICATION CHECKLIST

See `docs/BUGFIX_MISSING_TARGET_CHECKLIST.md` for detailed implementation checklist.

---

_Investigation completed 2026-01-17_
