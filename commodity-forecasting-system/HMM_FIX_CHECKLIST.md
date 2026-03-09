# HMM NaN Error Fix Checklist

## Error Summary

**Error:** `startprob_ must sum to 1 (got nan)`

**Trigger:** User set wider feature scaling bounds:
- overnight_gap_abs: 0.00-6.00 (was 0.00-3.00)
- range_ma_5: 0.20-8.00 (was 0.30-3.00)
- vix_level: 9.00-90.00 (was 10.00-50.00)
- volume_ratio: 0.20-6.00 (was 0.50-3.00)
- range_std_5: 0.00-4.00 (was 0.00-1.50)

**Root Cause:** Numerical underflow during HMM training when feature scaling bounds are too wide relative to actual data distribution.

---

## Technical Analysis

### Data Flow
```
Raw OHLCV → VolatilityFeatureEngineer → prepare_features() → _scale_features() → GaussianHMM.fit()
```

### Why Wide Bounds Cause NaN

1. **Compression Problem:**
   - Wide bounds (e.g., 0-90 for VIX) compress real data (e.g., VIX 12-25) into tiny portion of [0,1]
   - Real VIX data [12-25] → scaled to [(12-9)/(90-9), (25-9)/(90-9)] = [0.037, 0.198]
   - Feature variance becomes very small (~0.01²)

2. **Covariance Collapse:**
   - GaussianHMM with `covariance_type='diag'` estimates per-feature variance
   - Tiny scaled values → tiny covariance (1e-4 or smaller)
   - Covariance matrix becomes ill-conditioned

3. **Likelihood Underflow:**
   - Gaussian PDF with tiny covariance → extremely small likelihoods (1e-100+)
   - Forward-backward algorithm: log(1e-100) → -230 → sum of logs → -Inf
   - Normalization: exp(-Inf) / exp(-Inf) = 0/0 = NaN

4. **NaN Propagation:**
   - NaN in forward pass → NaN in gamma → NaN in startprob_ calculation
   - Library check fails: `startprob_.sum() != 1`

---

## Checklist of Required Changes

### 1. Add Data Validation Before Scaling
**File:** `src/models/hmm_volatility.py`
**Location:** `_scale_features()` method
**Priority:** CRITICAL

```python
# Before scaling, validate that bounds are reasonable relative to data
def _validate_bounds(self, X: np.ndarray) -> list:
    """Validate feature bounds against actual data distribution."""
    warnings = []
    for i, feat in enumerate(self.features):
        data_min, data_max = X[:, i].min(), X[:, i].max()
        data_range = data_max - data_min

        bounds = self.feature_bounds.get(feat)
        if bounds is None:
            continue

        bound_min, bound_max = bounds['min'], bounds['max']
        bound_range = bound_max - bound_min

        # Check if data occupies less than 10% of bound range
        coverage = data_range / bound_range if bound_range > 0 else 0
        if coverage < 0.1:
            warnings.append({
                'feature': feat,
                'issue': 'sparse_coverage',
                'data_range': (data_min, data_max),
                'bound_range': (bound_min, bound_max),
                'coverage_pct': coverage * 100,
                'suggestion': f"Try bounds {data_min:.2f} to {data_max:.2f}"
            })

    return warnings
```

**Acceptance Criteria:**
- [ ] Method `_validate_bounds()` exists and is called before scaling
- [ ] Warnings are logged when data covers <10% of bound range
- [ ] Warnings include suggested tighter bounds

---

### 2. Add Scaled Data Variance Check
**File:** `src/models/hmm_volatility.py`
**Location:** After `_scale_features()` call in `train()`
**Priority:** CRITICAL

```python
def _check_scaled_variance(self, X_scaled: np.ndarray, min_variance: float = 1e-6) -> bool:
    """Check if scaled features have sufficient variance for HMM."""
    variances = X_scaled.var(axis=0)

    for i, feat in enumerate(self.features):
        if variances[i] < min_variance:
            logger.error(
                f"Feature '{feat}' has near-zero variance ({variances[i]:.2e}) after scaling. "
                f"This will cause numerical instability. Tighten the scaling bounds for this feature."
            )
            return False

    return True
```

**Acceptance Criteria:**
- [ ] Method `_check_scaled_variance()` exists
- [ ] Called after scaling in `train()` method
- [ ] Raises clear exception if variance is too small
- [ ] Exception message includes feature name and suggestion

---

### 3. Add Covariance Regularization to HMM
**File:** `src/models/hmm_volatility.py`
**Location:** `train()` method, HMM initialization
**Priority:** HIGH

**Option A: Add minimum covariance floor**
```python
# After HMM fit, enforce minimum covariance
self.model.fit(X_scaled)

# Add covariance floor to prevent numerical issues
MIN_COV = 1e-4
if hasattr(self.model, 'covars_'):
    self.model.covars_ = np.maximum(self.model.covars_, MIN_COV)
```

**Option B: Use more stable covariance type**
```python
# Consider 'spherical' which is more numerically stable
self.model = hmm.GaussianHMM(
    n_components=self.n_regimes,
    covariance_type='spherical',  # Instead of 'diag'
    ...
)
```

**Option C: Add regularization parameter**
```python
self.model = hmm.GaussianHMM(
    n_components=self.n_regimes,
    covariance_type='diag',
    min_covar=1e-4,  # hmmlearn supports this parameter
    ...
)
```

**Acceptance Criteria:**
- [ ] HMM initialization includes `min_covar=1e-4` parameter
- [ ] Documented why this parameter is needed
- [ ] Test confirms training succeeds with wide bounds after this change

---

### 4. Improve Error Handling with Actionable Messages
**File:** `src/models/hmm_volatility.py`
**Location:** `train()` method, wrap `.fit()` call
**Priority:** HIGH

```python
def train(self, df: pd.DataFrame, n_iter: int = 100, tol: float = 1e-4) -> Dict:
    # ... existing code ...

    # Validate bounds before fitting
    bound_warnings = self._validate_bounds(X)
    if bound_warnings:
        for w in bound_warnings:
            logger.warning(
                f"Feature '{w['feature']}' bounds may be too wide: "
                f"data range {w['data_range'][0]:.3f}-{w['data_range'][1]:.3f} "
                f"covers only {w['coverage_pct']:.1f}% of bound range "
                f"{w['bound_range'][0]:.3f}-{w['bound_range'][1]:.3f}. "
                f"Suggestion: {w['suggestion']}"
            )

    # Scale features
    X_scaled = self._scale_features(X, fit=True)

    # Check variance
    if not self._check_scaled_variance(X_scaled):
        raise ValueError(
            "Scaled features have insufficient variance. "
            "Your scaling bounds are too wide for the data. "
            "Either use tighter bounds or enable 'Use Data-Driven Bounds' option."
        )

    # Fit HMM with try/catch for better error messages
    try:
        self.model.fit(X_scaled)
    except ValueError as e:
        if 'startprob_' in str(e) and 'nan' in str(e).lower():
            raise ValueError(
                f"HMM training failed due to numerical underflow. "
                f"Your feature scaling bounds are too wide for the data. "
                f"Suggestions:\n"
                f"1. Use 'Data-Driven Bounds' option (auto-detects bounds from data)\n"
                f"2. Tighten bounds based on these warnings:\n" +
                '\n'.join([f"   - {w['feature']}: {w['suggestion']}" for w in bound_warnings])
            ) from e
        raise
```

**Acceptance Criteria:**
- [ ] Clear error message when NaN occurs
- [ ] Message includes specific feature(s) causing the problem
- [ ] Message includes suggested bounds based on actual data
- [ ] No generic "startprob_ must sum to 1" error shown to user

---

### 5. Add "Data-Driven Bounds" Option
**File:** `src/ui/app.py`
**Location:** Sidebar, after feature bounds section
**Priority:** HIGH

```python
# Add checkbox for data-driven bounds
use_data_driven_bounds = st.checkbox(
    "Use Data-Driven Bounds",
    value=False,
    help="Automatically calculate bounds from data (recommended for stability)"
)

if use_data_driven_bounds:
    st.info("Bounds will be calculated from data percentiles (1st-99th)")
    # Override user bounds with None (signals data-driven)
    for feat in selected_features:
        feature_bounds[feat] = None  # HMM will use data-driven bounds
```

**File:** `src/models/hmm_volatility.py`
**Location:** `_scale_features()` method

```python
# In _scale_features(), when bounds is None, use percentile-based bounds
if bounds is None:
    # Use 1st-99th percentile to exclude outliers
    data_min = np.percentile(X[:, i], 1)
    data_max = np.percentile(X[:, i], 99)
    mins.append(data_min)
    maxs.append(data_max)
    logger.info(f"Feature '{feat}': using data-driven bounds [{data_min:.3f}, {data_max:.3f}]")
```

**Acceptance Criteria:**
- [ ] "Use Data-Driven Bounds" checkbox in UI
- [ ] When enabled, HMM uses percentile-based bounds
- [ ] Percentile bounds logged for transparency
- [ ] Training succeeds with data-driven bounds

---

### 6. Show Actual Data Range in UI
**File:** `src/ui/app.py`
**Location:** Feature bounds configuration section
**Priority:** MEDIUM

```python
# After data is loaded, show actual data range for each feature
if st.session_state.data_loaded and st.session_state.features_df is not None:
    df = st.session_state.features_df
    for feat in selected_features:
        if feat in df.columns:
            data_min = df[feat].min()
            data_max = df[feat].max()
            st.caption(f"*Actual data range: {data_min:.3f} - {data_max:.3f}*")
```

**Acceptance Criteria:**
- [ ] Actual data range shown below each feature's bounds input
- [ ] Only shown after data is loaded
- [ ] Helps user choose reasonable bounds

---

### 7. Add Feature Scaling Diagnostics
**File:** `src/models/hmm_volatility.py`
**Location:** New method
**Priority:** MEDIUM

```python
def get_scaling_diagnostics(self, df: pd.DataFrame) -> Dict:
    """
    Get diagnostics about feature scaling for debugging.

    Returns:
        Dictionary with per-feature scaling stats
    """
    X, _ = self.prepare_features(df)
    X_scaled = self._scale_features(X, fit=False)

    diagnostics = {}
    for i, feat in enumerate(self.features):
        raw_data = X[:, i]
        scaled_data = X_scaled[:, i]

        bounds = self.feature_bounds.get(feat, {})

        diagnostics[feat] = {
            'raw_min': float(raw_data.min()),
            'raw_max': float(raw_data.max()),
            'raw_mean': float(raw_data.mean()),
            'raw_std': float(raw_data.std()),
            'scaled_min': float(scaled_data.min()),
            'scaled_max': float(scaled_data.max()),
            'scaled_mean': float(scaled_data.mean()),
            'scaled_std': float(scaled_data.std()),
            'scaled_variance': float(scaled_data.var()),
            'bound_min': bounds.get('min', 'data-driven'),
            'bound_max': bounds.get('max', 'data-driven'),
            'coverage_pct': float((raw_data.max() - raw_data.min()) /
                                  (bounds.get('max', raw_data.max()) - bounds.get('min', raw_data.min())) * 100)
                           if bounds.get('max') else 100.0,
            'is_healthy': float(scaled_data.var()) > 1e-4
        }

    return diagnostics
```

**Acceptance Criteria:**
- [ ] Method `get_scaling_diagnostics()` exists
- [ ] Returns per-feature raw and scaled statistics
- [ ] Includes `is_healthy` flag for quick checks
- [ ] Can be called from UI for debugging

---

### 8. Add Scaling Diagnostics Tab in UI
**File:** `src/ui/app.py`
**Location:** New section in sidebar or main area
**Priority:** LOW

```python
# In sidebar or as expandable section
with st.expander("🔬 Scaling Diagnostics"):
    if st.session_state.hmm_model is not None and st.session_state.data_loaded:
        diagnostics = st.session_state.hmm_model.get_scaling_diagnostics(
            st.session_state.features_df
        )

        for feat, stats in diagnostics.items():
            health = "✅" if stats['is_healthy'] else "❌"
            st.markdown(f"**{feat}** {health}")
            st.caption(
                f"Raw: {stats['raw_min']:.3f} - {stats['raw_max']:.3f} | "
                f"Scaled var: {stats['scaled_variance']:.6f} | "
                f"Coverage: {stats['coverage_pct']:.1f}%"
            )
            if not stats['is_healthy']:
                st.warning(f"⚠️ Variance too low - tighten bounds or use data-driven")
```

**Acceptance Criteria:**
- [ ] Diagnostics section visible in UI
- [ ] Shows per-feature scaling health
- [ ] Warnings for unhealthy features
- [ ] Helps user debug bound configuration

---

### 9. Update Unit Tests
**File:** `tests/test_hmm_volatility.py` (create if not exists)
**Priority:** HIGH

```python
def test_hmm_wide_bounds_error_message():
    """Test that wide bounds give helpful error message, not cryptic NaN error."""
    # Create data with small range
    df = create_test_df()  # Data with VIX ~15-25

    # Set very wide bounds
    hmm = VolatilityHMM(
        n_regimes=3,
        feature_bounds={'vix_level': {'min': 0, 'max': 100}}  # Way too wide
    )

    with pytest.raises(ValueError) as exc_info:
        hmm.train(df)

    # Check error message is helpful
    assert 'variance' in str(exc_info.value).lower() or 'bound' in str(exc_info.value).lower()
    assert 'vix_level' in str(exc_info.value)


def test_hmm_data_driven_bounds():
    """Test that data-driven bounds work correctly."""
    df = create_test_df()

    # Set bounds to None (data-driven)
    hmm = VolatilityHMM(
        n_regimes=3,
        feature_bounds={feat: None for feat in VolatilityHMM.DEFAULT_FEATURES}
    )

    # Should train successfully
    metrics = hmm.train(df)
    assert metrics['converged']


def test_hmm_min_covar_prevents_nan():
    """Test that min_covar parameter prevents NaN errors."""
    df = create_test_df()

    hmm = VolatilityHMM(n_regimes=3)
    # Should not raise even with default bounds
    metrics = hmm.train(df)
    assert not np.isnan(metrics['log_likelihood'])
```

**Acceptance Criteria:**
- [ ] Test for helpful error message with wide bounds
- [ ] Test for data-driven bounds working
- [ ] Test for min_covar preventing NaN
- [ ] All tests pass

---

### 10. Update Documentation
**File:** `src/ui/utils.py` - HMM_FEATURE_INFO
**Priority:** MEDIUM

Update the `HMM_FEATURE_INFO` dictionary with recommended bound ranges:

```python
HMM_FEATURE_INFO = {
    'overnight_gap_abs': {
        'description': 'Overnight gap magnitude (absolute %)',
        'min': 0.0,
        'max': 3.0,  # Reasonable for normal markets
        'recommended_max': 5.0,  # For crisis periods
        'unit': '%',
        'effect': 'HIGH = strong morning volatility',
        'bound_notes': 'Keep max <5% unless analyzing 2008/2020 crisis periods'
    },
    # ... similar for other features
}
```

**Acceptance Criteria:**
- [ ] Each feature has `bound_notes` explaining reasonable ranges
- [ ] UI shows these notes to help user configure bounds
- [ ] Documentation explains the tradeoff: wider bounds = more historical coverage but less numerical stability

---

## Implementation Order

1. **CRITICAL (must fix first):**
   - [x] Add `min_covar=1e-4` to HMM initialization (#3, Option C) ✅ DONE
   - [x] Add variance check after scaling (#2) ✅ DONE
   - [x] Improve error message (#4) ✅ DONE

2. **HIGH (should fix):**
   - [x] Add bound validation warnings (#1) ✅ DONE
   - [x] Add data-driven bounds option (#5) ✅ DONE
   - [ ] Add unit tests (#9)

3. **MEDIUM (nice to have):**
   - [x] Show actual data range in UI (#6) ✅ DONE
   - [x] Add scaling diagnostics method (#7) ✅ DONE
   - [ ] Update documentation (#10)

4. **LOW (polish):**
   - [ ] Add diagnostics tab in UI (#8)

---

## Verification Steps

After implementing all changes:

1. **Test with default bounds:**
   ```bash
   python -c "from src.models.hmm_volatility import test_hmm_volatility; test_hmm_volatility()"
   ```
   Expected: All tests pass

2. **Test with user's wide bounds:**
   - Set bounds: overnight_gap_abs 0-6, range_ma_5 0.2-8, vix_level 9-90, volume_ratio 0.2-6, range_std_5 0-4
   - Click "Train HMM"
   - Expected: Either succeeds (with min_covar) OR shows clear error message with suggestions

3. **Test with data-driven bounds:**
   - Enable "Use Data-Driven Bounds" checkbox
   - Click "Train HMM"
   - Expected: Succeeds, logs show calculated bounds

4. **Run validation script:**
   ```bash
   python scripts/validate_volatility_mvp.py
   ```
   Expected: Completes without NaN errors

---

## Summary

The `startprob_ must sum to 1 (got nan)` error is caused by **numerical underflow** when feature scaling bounds are too wide relative to actual data. The fix requires:

1. **Prevention:** Add `min_covar` parameter to prevent covariance collapse
2. **Detection:** Check scaled variance before fitting
3. **Recovery:** Offer data-driven bounds as alternative
4. **Communication:** Show helpful error messages with suggested bounds

All features of the HMM system will be preserved - the changes add safeguards without removing any functionality.
