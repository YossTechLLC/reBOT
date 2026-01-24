# BUGFIX INVESTIGATION: Session State Initialization Conflict

**Date:** 2026-01-17
**Error:** `AttributeError: 'NoneType' object has no attribute 'threshold'`
**Status:** 🔍 ROOT CAUSE IDENTIFIED

---

## 🐛 ERROR DETAILS

**Error Message:**
```
AttributeError: 'NoneType' object has no attribute 'threshold'
```

**Error Location:**
- File: `src/ui/model_controller.py`
- Line: 265
- Function: `set_confidence_threshold()`
- Code: `st.session_state.confidence_scorer.threshold = threshold`

**Problem:** `st.session_state.confidence_scorer` is `None` instead of a `VolatilityConfidenceScorer` instance.

---

## 🔍 ROOT CAUSE ANALYSIS

### Execution Flow Trace

**Step 1: app.py main() - Line 82**
```python
initialize_session_state()
```

This calls `utils.initialize_session_state()` which executes:
```python
defaults = {
    'hmm_model': None,
    'hmm_metrics': None,
    'timesfm_forecaster': None,
    'confidence_scorer': None,  # ← Sets to None
    # ...
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value  # ← Creates key with None value
```

**Result:** `st.session_state.confidence_scorer = None` ✅ Key exists, Value is None

---

**Step 2: app.py main() - Line 90-91**
```python
if 'model_controller' not in st.session_state:
    st.session_state.model_controller = ModelController()
```

This creates a `ModelController` instance, triggering `__init__`:

```python
def __init__(self):
    # ...
    if 'confidence_scorer' not in st.session_state:  # ← CHECK FAILS!
        st.session_state.confidence_scorer = VolatilityConfidenceScorer()
```

**Why check fails:**
- The key `'confidence_scorer'` **DOES** exist in session_state (created in Step 1)
- The check `'confidence_scorer' not in st.session_state` evaluates to `False`
- The `VolatilityConfidenceScorer()` instance is **NEVER CREATED**

**Result:** `st.session_state.confidence_scorer` remains `None` ❌

---

**Step 3: app.py render_sidebar() - Line 262**
```python
st.session_state.model_controller.set_confidence_threshold(confidence_threshold)
```

This calls `model_controller.py` line 265:
```python
st.session_state.confidence_scorer.threshold = threshold  # ← CRASH!
```

**Why crash:**
- `st.session_state.confidence_scorer` is `None`
- Trying to access `.threshold` attribute on `None`
- `AttributeError: 'NoneType' object has no attribute 'threshold'`

---

## 🎯 ROOT CAUSE SUMMARY

**The Conflict:**
1. `initialize_session_state()` creates keys with `None` values as defaults
2. `ModelController.__init__()` checks `if key not in st.session_state`
3. The check fails because the key exists (even though value is None)
4. The actual object never gets created

**Pattern Anti-Pattern:**
- ❌ `if 'key' not in st.session_state:` - Only checks if key exists
- ✅ `if st.session_state.get('key') is None:` - Checks if key is missing OR None

---

## 🔧 AFFECTED CODE SECTIONS

### 1. ModelController.__init__() - Lines 49-56

**Current (Broken):**
```python
if 'hmm_model' not in st.session_state:
    st.session_state.hmm_model = None
if 'hmm_metrics' not in st.session_state:
    st.session_state.hmm_metrics = None
if 'timesfm_forecaster' not in st.session_state:
    st.session_state.timesfm_forecaster = None
if 'confidence_scorer' not in st.session_state:
    st.session_state.confidence_scorer = VolatilityConfidenceScorer()
```

**Issue:** All 4 checks fail when `initialize_session_state()` has run first.

### 2. utils.initialize_session_state() - Lines 32-36

**Current:**
```python
defaults = {
    'hmm_model': None,
    'hmm_metrics': None,
    'timesfm_forecaster': None,
    'confidence_scorer': None,  # Conflicts with ModelController
    # ...
}
```

**Issue:** Pre-creates keys with None values, preventing ModelController from detecting uninitialized state.

---

## ✅ RESOLUTION STRATEGY

### Option 1: Fix ModelController Checks (RECOMMENDED)

**Change:** Modify `ModelController.__init__()` to check for None values, not just key existence.

**Pros:**
- Defensive programming - handles both missing keys AND None values
- Works with existing `initialize_session_state()` pattern
- Minimal changes needed
- Follows Streamlit best practices

**Cons:**
- Need to update multiple checks

**Implementation:**
```python
# Before: if 'confidence_scorer' not in st.session_state:
# After:  if st.session_state.get('confidence_scorer') is None:
```

---

### Option 2: Remove from initialize_session_state()

**Change:** Remove model-related keys from `utils.initialize_session_state()` defaults.

**Pros:**
- Clean separation - ModelController owns model state
- No conflicts

**Cons:**
- Loses safety net of guaranteed key existence
- Other parts of code might expect keys to exist
- Less defensive

---

### Option 3: Hybrid Approach

**Change:** Keep `initialize_session_state()` for data keys, but remove model keys.

**Pros:**
- Best of both worlds
- Clear ownership boundaries

**Cons:**
- More complex mental model
- Need to audit all session_state access patterns

---

## 📋 RECOMMENDED SOLUTION: Option 1

**Rationale:**
1. **Defensive:** Handles both missing keys and None values
2. **Minimal Risk:** Small, focused changes
3. **Idiomatic:** Follows Streamlit community patterns
4. **Backwards Compatible:** Works with existing code

**Changes Required:**
1. Update `ModelController.__init__()` - 4 checks
2. Test all initialization paths
3. Document the pattern

---

## 🧪 TESTING PLAN

### Test Case 1: Fresh Session
1. Clear session state
2. Load app
3. Verify `confidence_scorer` is created
4. Verify no AttributeError

### Test Case 2: Existing Session
1. Create session with None values
2. Reload app
3. Verify `confidence_scorer` is created
4. Verify no AttributeError

### Test Case 3: Model Operations
1. Load data
2. Train HMM
3. Set confidence threshold
4. Run prediction
5. Verify all operations work

---

## 🎯 IMPLEMENTATION CHECKLIST

See next section for detailed checklist.

---

## 📚 LESSONS LEARNED

### Streamlit Session State Patterns

**Wrong Pattern:**
```python
if 'key' not in st.session_state:
    st.session_state.key = create_object()
```
This fails when key exists with None value.

**Correct Pattern:**
```python
if st.session_state.get('key') is None:
    st.session_state.key = create_object()
```
This handles both missing keys and None values.

**Best Pattern (for required objects):**
```python
if st.session_state.get('key') is None:
    st.session_state.key = create_object()
# Then use it - guaranteed to exist
```

### Initialization Strategy

**Two Common Approaches:**

1. **Centralized Initialization:**
   - One function sets all defaults
   - All components assume keys exist
   - Values may be None initially

2. **Lazy Initialization:**
   - Components create their own state on first access
   - Check for None, not just key existence

**Our Code:** Uses both! (Hence the conflict)

**Resolution:** Make components defensive - check for None.

---

## 🔄 SIMILAR ISSUES TO CHECK

After fixing `confidence_scorer`, audit for similar patterns:

1. ✅ `hmm_model` - Same pattern, needs fix
2. ✅ `hmm_metrics` - Same pattern, needs fix
3. ✅ `timesfm_forecaster` - Same pattern, needs fix
4. ✅ `data_manager` - Different pattern (created in app.py), OK
5. ✅ `model_controller` - Created in app.py, OK

---

_Investigation completed 2026-01-17_
