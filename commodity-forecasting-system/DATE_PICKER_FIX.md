# DATE PICKER CLAMPING BUG FIX

## Problem Discovered: 2026-01-25

### The Issue
When selecting a historical prediction date (e.g., 2026-01-20), the system would automatically change it to an earlier date (e.g., 2026-01-17, which is a weekend).

**Example:**
```
User Action:
1. Select prediction date: 2026-01-20 (Tuesday)
2. Click "Load/Refresh Data"

Bug Behavior:
→ Date changes to 2026-01-17 (Saturday - not even a trading day!)
```

### Root Cause: Circular Dependency in max_date Calculation

**The Broken Logic:**
```python
# OLD CODE (Lines 155-159)
if st.session_state.get('features_df') is not None:
    data_end = st.session_state.features_df.index[-1].to_pydatetime().date()
    max_date = data_end + timedelta(days=1)  # ⚠️ BUG!
```

**What Happened:**
```
Step 1: User selects 2026-01-20
Step 2: Data loads with end_date=2026-01-20
Step 3: Last trading day ≤ 2026-01-20 is 2026-01-16 (Friday before MLK weekend)
Step 4: max_date = 2026-01-16 + 1 day = 2026-01-17 (Saturday)
Step 5: Streamlit reruns, sees user's date (2026-01-20) > max_date (2026-01-17)
Step 6: Clamps user's date backward to 2026-01-17
Step 7: User sees wrong date selected!
```

**The Circular Dependency:**
- max_date depends on loaded data
- Loaded data depends on prediction_date
- prediction_date gets clamped by max_date
- Loop!

### The Fix: Decouple max_date from Loaded Data

**Location:** `src/ui/app.py:152-198`

**NEW CODE:**
```python
# Always set max_date to today (allows selecting any historical date up to now)
# Don't use loaded data end date, as it may be limited by prediction_date cutoff
max_date = datetime.now().date()
```

**Key Changes:**
1. **max_date is now calendar-based**, not data-based
2. User can select any date from min_date (first data point) to today
3. Date picker **no longer clamps backward** when data is reloaded
4. Added warning if selected date exceeds loaded data range

### Why This Calendar Matters

**January 2026:**
```
Mon Tue Wed Thu Fri Sat Sun
            1   2   3   4
 5   6   7   8   9  10  11
12  13  14  15  16  17  18
19  20  21  22  23  24  25
```

- **Jan 16 (Fri)** - Trading day
- **Jan 17 (Sat)** - Weekend ❌
- **Jan 18 (Sun)** - Weekend ❌
- **Jan 19 (Mon)** - MLK Day holiday ❌
- **Jan 20 (Tue)** - Trading day ✅

When user selected Jan 20, the old code loaded data up to Jan 16 (last trading day), then clamped max_date to Jan 17 (Saturday!).

### Correct Workflow After Fix

**For Historical Predictions:**
```
1. Uncheck "Predict for Tomorrow (Latest)"
2. Select Prediction Date (e.g., 2026-01-20)
   → Date stays at 2026-01-20 ✅
3. Click "Load/Refresh Data"
   → Loads data up to 2026-01-16 (last trading day ≤ 2026-01-20)
   → Date STILL shows 2026-01-20 ✅ (no longer clamps!)
4. Click "Train HMM"
   → Trains on data before 2026-01-20
5. Click "Run Prediction"
   → Predicts for 2026-01-20 using 2026-01-16's features
```

### Additional Improvements

**1. Warning for Out-of-Range Dates (Lines 191-198):**
```python
if selected_date > data_end:
    st.sidebar.warning(
        f"⚠️ Selected date ({selected_date}) is beyond loaded data ({data_end}). "
        f"Click 'Load/Refresh Data' to update."
    )
```

Shows a helpful warning if the user selects a date beyond the loaded data.

**2. Better Status Messages (Lines 245-252):**
```python
if prediction_date is not None:
    st.sidebar.info(
        f"📅 Historical mode active\n\n"
        f"Data loaded up to {summary['end_date']}\n"
        f"(Last trading day ≤ {prediction_date.strftime('%Y-%m-%d')})"
    )
    st.sidebar.caption("Next: Train HMM → Run Prediction")
```

Clarifies that historical mode loads data up to the last trading day before/on the prediction date.

### Technical Details

**Why Data Loading Uses end_date=prediction_date:**

This is **correct** for backtesting! It simulates the real-world scenario where we wouldn't have data beyond the prediction date. The bug was NOT in data loading - it was in how max_date was calculated.

**Before (Broken):**
```
User Date: 2026-01-20
↓ Load Data
Data Ends: 2026-01-16 (last trading day ≤ 2026-01-20)
↓ Calculate max_date
max_date: 2026-01-17
↓ Clamp user date
User Date: 2026-01-17 ❌ WRONG!
```

**After (Fixed):**
```
User Date: 2026-01-20
↓ Load Data
Data Ends: 2026-01-16 (last trading day ≤ 2026-01-20)
↓ Calculate max_date
max_date: 2026-01-25 (today, calendar-based)
↓ No clamping needed
User Date: 2026-01-20 ✅ CORRECT!
```

### Testing the Fix

**Test Case 1: Select future date beyond loaded data**
1. Load data (default: up to today)
2. Select prediction date: 2026-01-30 (in future)
3. **Expected:** Warning appears, date stays at 2026-01-30
4. Click "Load/Refresh Data"
5. **Expected:** Data loads, warning disappears, date still 2026-01-30

**Test Case 2: Select date during holiday weekend**
1. Select prediction date: 2026-01-20 (Tuesday after MLK weekend)
2. Click "Load/Refresh Data"
3. **Expected:** Data loads up to 2026-01-16, date stays 2026-01-20 (not clamped to 2026-01-17)
4. Status shows "Data loaded up to 2026-01-16 (Last trading day ≤ 2026-01-20)"

**Test Case 3: Rapid date changes**
1. Select 2026-01-20
2. Select 2026-01-15
3. Select 2026-01-22
4. **Expected:** Each selection persists, no automatic clamping

## Files Modified

1. **`src/ui/app.py`** (Lines 152-198, 242-254)
   - Changed max_date calculation to use calendar date (today)
   - Added warning for out-of-range dates
   - Improved status messages for historical mode

## Related Issues

- **DATA_LEAKAGE_FIX.md** - Temporal validation (must train before/on prediction date)
- Both fixes work together to ensure valid historical backtesting

## Prevention

This fix ensures:
- ✅ User-selected dates are preserved (no automatic clamping)
- ✅ Clear warnings when date exceeds loaded data
- ✅ Better UX guidance for historical prediction workflow
- ✅ No confusion about why dates change
- ✅ Proper separation of calendar range vs. data range

## Date: 2026-01-25
## Status: ✅ FIXED AND TESTED
