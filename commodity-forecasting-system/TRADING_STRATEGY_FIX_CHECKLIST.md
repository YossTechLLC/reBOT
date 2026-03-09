# Trading Strategy Architecture Fix Checklist

## Your Desired Strategy (Reference)

```
ENTRY:
- Use HMM + TimesFM to predict volatility regime for day X
- IF regime predicts sufficient volatility → ENTER trade
- BUY 1 CALL + BUY 1 PUT at market open (same strike = straddle)

EXIT:
- Take profit: 7x return on the cost of THAT LEG (not total position)
- OR: Let position expire worthless
- NO stop loss
- NO time-based exit (hold until 7x or expiration)

EXAMPLE:
- Call cost: $1.50 → Exit call leg when call value reaches $10.50 (7x)
- Put cost: $1.20 → Exit put leg when put value reaches $8.40 (7x)
- Each leg exits INDEPENDENTLY when it hits 7x
- If neither hits 7x, both expire (max loss = total premium paid)
```

---

## Current Implementation Problems

| Problem | Current Behavior | Desired Behavior | Severity |
|---------|-----------------|------------------|----------|
| Exit rule | 30-100% profit on total position | 7x on EACH LEG independently | **CRITICAL** |
| Per-leg tracking | Not tracked | Must track call_cost and put_cost separately | **CRITICAL** |
| Time-based exit | "Close before 3:30/3:45 PM" | Hold until 7x or expiration | **CRITICAL** |
| Stop loss | 50% loss for normal_vol | NO stop loss ever | **HIGH** |
| Validation | Measures prediction accuracy only | Must simulate 7x exit rule P&L | **HIGH** |
| P&L calculation | Intrinsic - total debit | Per-leg: leg_value vs leg_cost × 7 | **HIGH** |

---

## Checklist of Required Changes

### 1. CRITICAL: Implement Per-Leg Cost Tracking

**File:** `src/ui/strategy.py`
**Location:** `_calculate_risk_metrics()` method (lines 213-283)

**Current:**
```python
debit_paid = round(spot * 0.012, 2)  # Single total premium
max_loss = debit_paid
```

**Required Change:**
```python
# Track each leg separately
call_premium = round(spot * 0.006, 2)  # Estimated call cost
put_premium = round(spot * 0.006, 2)   # Estimated put cost
total_debit = call_premium + put_premium

# 7x exit targets per leg
call_exit_target = call_premium * 7
put_exit_target = put_premium * 7
```

**Acceptance Criteria:**
- [ ] `call_premium` tracked separately from `put_premium`
- [ ] `call_exit_target = call_premium * 7` calculated
- [ ] `put_exit_target = put_premium * 7` calculated
- [ ] Both values returned in risk_metrics dictionary
- [ ] UI displays per-leg exit targets

---

### 2. CRITICAL: Remove Time-Based Exit Rules

**File:** `src/ui/strategy.py`
**Location:** `regime_strategies` dictionary (lines 37-88)

**Current (extreme_vol example):**
```python
'exit_rules': [
    '100%+ profit target (let winners run)',
    'Close before 3:30 PM (0DTE theta acceleration)',  # REMOVE
    'No stop loss - max loss is premium paid'
]
```

**Required Change:**
```python
'exit_rules': [
    'Exit CALL leg when value reaches 7x call cost',
    'Exit PUT leg when value reaches 7x put cost',
    'If neither leg hits 7x, let both expire',
    'No stop loss - max loss is total premium paid'
]
```

**Acceptance Criteria:**
- [ ] All time-based exit rules removed from ALL regime strategies
- [ ] Exit rules updated to reflect 7x per-leg rule
- [ ] Same exit rules for ALL tradeable regimes (extreme_vol, high_vol, normal_vol)

---

### 3. CRITICAL: Remove Stop Loss from normal_vol

**File:** `src/ui/strategy.py`
**Location:** `regime_strategies['normal_vol']` (lines 66-79)

**Current:**
```python
'exit_rules': [
    '30-50% profit target',
    'Close before 3:45 PM',
    'Consider closing at 50% loss if no movement'  # REMOVE
]
```

**Required Change:**
```python
'exit_rules': [
    'Exit CALL leg when value reaches 7x call cost',
    'Exit PUT leg when value reaches 7x put cost',
    'If neither leg hits 7x, let both expire',
    'No stop loss - max loss is total premium paid'
]
```

**Acceptance Criteria:**
- [ ] Stop loss rule removed from normal_vol
- [ ] All tradeable regimes have identical exit rules (7x per-leg)

---

### 4. HIGH: Update Risk Metrics Structure

**File:** `src/ui/strategy.py`
**Location:** `_calculate_risk_metrics()` return statement (lines 274-283)

**Current:**
```python
return {
    'position_type': position_type,
    'debit_paid': debit_paid,
    'max_loss': max_loss,
    'max_profit_estimate': max_profit_estimate,
    'profit_probability': win_probability,
    'breakeven_upper': breakeven_upper,
    'breakeven_lower': breakeven_lower,
    'risk_reward_ratio': round(max_profit_estimate / max_loss, 2)
}
```

**Required Change:**
```python
return {
    'position_type': 'long',
    # Per-leg costs
    'call_premium': call_premium,
    'put_premium': put_premium,
    'total_debit': call_premium + put_premium,
    # Per-leg exit targets (7x rule)
    'call_exit_target': call_premium * 7,
    'put_exit_target': put_premium * 7,
    # Max outcomes
    'max_loss': call_premium + put_premium,  # Both legs expire worthless
    'max_profit_one_leg': (call_premium * 7) - call_premium,  # 6x gain on one leg
    'max_profit_both_legs': (call_premium * 6) + (put_premium * 6),  # Rare: both hit 7x
    # Breakevens (for reference)
    'breakeven_upper': call_strike + (call_premium + put_premium),
    'breakeven_lower': put_strike - (call_premium + put_premium),
    # Simplified - actual win rate depends on historical backtesting
    'exit_rule': '7x per-leg or expire'
}
```

**Acceptance Criteria:**
- [ ] `call_premium` and `put_premium` tracked separately
- [ ] `call_exit_target` and `put_exit_target` calculated (7x)
- [ ] `max_loss` = total premium (both legs expire)
- [ ] Old `profit_probability` removed (not meaningful without backtest)
- [ ] `exit_rule` field added for clarity

---

### 5. HIGH: Simplify Strategy Selection

**File:** `src/ui/strategy.py`
**Location:** `regime_strategies` dictionary (lines 37-88)

**Current:** Different strike selection by regime (ATM straddle vs OTM strangle)

**Required Change:** Unify to single strategy for all tradeable regimes

```python
# Single strategy for all "TRADE" regimes
LONG_STRADDLE_STRATEGY = {
    'name': 'LONG ATM Straddle',
    'position_type': 'long',
    'call_delta': 0.50,  # ATM
    'put_delta': -0.50,  # ATM
    'contracts': 1,
    'entry_timing': 'Market open',
    'exit_rules': [
        'Exit CALL leg when value reaches 7x call cost',
        'Exit PUT leg when value reaches 7x put cost',
        'If neither leg hits 7x, let both expire',
        'No stop loss - max loss is total premium paid'
    ],
    'rationale': 'Volatility predicted - BUY ATM straddle for maximum gamma exposure'
}

self.regime_strategies = {
    'extreme_vol': {**LONG_STRADDLE_STRATEGY, 'priority': 'HIGH'},
    'high_vol': {**LONG_STRADDLE_STRATEGY, 'priority': 'HIGH'},
    'normal_vol': {**LONG_STRADDLE_STRATEGY, 'priority': 'MEDIUM'},
    'low_vol': {'name': 'SKIP', 'rationale': 'Insufficient volatility expected'},
    'very_low_vol': {'name': 'SKIP', 'rationale': 'Dead market - no edge'},
}
```

**Acceptance Criteria:**
- [ ] All tradeable regimes use same entry strategy (ATM straddle)
- [ ] All tradeable regimes use same exit rules (7x per-leg)
- [ ] Only difference is `priority` level for sizing decisions
- [ ] SKIP regimes remain unchanged

---

### 6. HIGH: Update Validation Script for 7x Rule

**File:** `scripts/validate_volatility_mvp.py`
**Location:** Entire validation methodology (lines 135-325)

**Current Validation:**
```python
# Binary classification: did volatility exceed 1.2%?
predicted_high_vol = score.total_score >= scorer.threshold
actual_high_vol = row['intraday_range_pct'] >= volatility_threshold
# Measures: accuracy, precision, recall, F1
```

**Required Change - Simulate Actual 7x Exit P&L:**

```python
def simulate_trade_pnl(
    entry_price: float,
    high_of_day: float,
    low_of_day: float,
    call_strike: float,
    put_strike: float,
    call_premium: float,
    put_premium: float
) -> dict:
    """
    Simulate P&L for a straddle with 7x exit rule.

    Assumptions:
    - Entry at market open (ATM straddle)
    - Call/put premiums estimated from IV
    - Check if 7x target was reachable during the day
    """
    # At market extremes, what would each leg be worth?
    call_max_value = max(0, high_of_day - call_strike)  # Intrinsic at high
    put_max_value = max(0, put_strike - low_of_day)      # Intrinsic at low

    # Did 7x trigger for each leg?
    call_target = call_premium * 7
    put_target = put_premium * 7

    call_hit_7x = call_max_value >= call_target
    put_hit_7x = put_max_value >= put_target

    # Calculate P&L
    if call_hit_7x:
        call_pnl = call_target - call_premium  # 6x gain
    else:
        call_pnl = -call_premium  # Expired worthless (simplified)

    if put_hit_7x:
        put_pnl = put_target - put_premium  # 6x gain
    else:
        put_pnl = -put_premium  # Expired worthless (simplified)

    total_pnl = call_pnl + put_pnl

    return {
        'call_hit_7x': call_hit_7x,
        'put_hit_7x': put_hit_7x,
        'call_pnl': call_pnl,
        'put_pnl': put_pnl,
        'total_pnl': total_pnl,
        'outcome': 'WIN' if total_pnl > 0 else 'LOSS'
    }
```

**New Validation Metrics:**
```python
# Per-trade simulation
for day in test_days:
    if should_trade(day):
        pnl = simulate_trade_pnl(...)
        trade_results.append(pnl)

# Aggregate metrics
total_trades = len(trade_results)
winning_trades = sum(1 for t in trade_results if t['total_pnl'] > 0)
win_rate = winning_trades / total_trades

call_7x_rate = sum(1 for t in trade_results if t['call_hit_7x']) / total_trades
put_7x_rate = sum(1 for t in trade_results if t['put_hit_7x']) / total_trades
either_7x_rate = sum(1 for t in trade_results if t['call_hit_7x'] or t['put_hit_7x']) / total_trades

total_pnl = sum(t['total_pnl'] for t in trade_results)
avg_pnl_per_trade = total_pnl / total_trades
```

**Acceptance Criteria:**
- [ ] `simulate_trade_pnl()` function implemented
- [ ] Validation tracks `call_hit_7x` and `put_hit_7x` separately
- [ ] Validation reports `either_7x_rate` (at least one leg hits 7x)
- [ ] Validation reports actual simulated P&L, not hardcoded $150/$80
- [ ] Results CSV includes per-trade P&L breakdown

---

### 7. HIGH: Update UI Display for 7x Rule

**File:** `src/ui/strategy.py`
**Location:** `format_strategy_output()` function (lines 388-479)

**Current Display:**
```
**Risk/Reward (LONG Position):**
- Premium Paid (Debit): $X.XX
- Max Loss: $X.XX (limited to premium)
- Max Profit (Est): $X.XX (on expected move)
- Breakeven: $X - $Y
- Win Probability: 45%
```

**Required Change:**
```
**Position Details:**
- Call Premium: $X.XX → Exit target: $X.XX (7x)
- Put Premium: $X.XX → Exit target: $X.XX (7x)
- Total Debit: $X.XX

**Exit Rules:**
- EXIT call leg when value reaches $X.XX (7x cost)
- EXIT put leg when value reaches $X.XX (7x cost)
- If neither hits 7x → let both expire (max loss: $X.XX)

**Scenarios:**
- One leg hits 7x: +$X.XX profit (6x on winning leg minus losing leg)
- Both legs hit 7x: +$X.XX profit (rare, requires whipsaw)
- Neither hits 7x: -$X.XX loss (total premium)
```

**Acceptance Criteria:**
- [ ] Display shows per-leg premiums and exit targets
- [ ] Exit rules clearly state "7x cost" with dollar amounts
- [ ] Scenario analysis shows realistic outcomes
- [ ] No mention of time-based exits or stop losses

---

### 8. MEDIUM: Update Confidence Scorer Recommendation Text

**File:** `src/volatility/confidence_scorer.py`
**Location:** `_get_recommendation()` method (lines 297-310)

**Current:**
```python
def _get_recommendation(self, score: float) -> str:
    if score < 40:
        return "SKIP - Low volatility expected"
    elif score < 60:
        return "TRADE (Small Size) - Moderate confidence"
    elif score < 80:
        return "TRADE (Full Size) - High confidence"
    else:
        return "TRADE (Full Size) - Exceptional setup"
```

**Required Change:**
```python
def _get_recommendation(self, score: float) -> str:
    if score < 40:
        return "SKIP - Insufficient volatility for 7x exit probability"
    elif score < 60:
        return "TRADE - Enter straddle, target 7x per-leg"
    elif score < 80:
        return "TRADE - High conviction, enter straddle, target 7x per-leg"
    else:
        return "TRADE - Exceptional setup, enter straddle, target 7x per-leg"
```

**Acceptance Criteria:**
- [ ] Recommendation text mentions "7x per-leg" exit rule
- [ ] Sizing recommendations simplified (TRADE or SKIP, not small/full)
- [ ] Or: Remove sizing recommendations entirely, let user decide

---

### 9. MEDIUM: Add 7x Exit Target Calculation to Visualization

**File:** `src/ui/visualization.py`
**Location:** P&L payoff diagram function

**Current:** Shows standard straddle payoff curve

**Required Change:** Add horizontal lines showing 7x exit targets for each leg

```python
# Add 7x exit target lines to payoff diagram
fig.add_hline(
    y=call_premium * 6,  # Net profit after 7x exit
    line_dash="dash",
    line_color="green",
    annotation_text=f"Call 7x target: ${call_premium * 7:.2f}"
)
fig.add_hline(
    y=put_premium * 6,
    line_dash="dash",
    line_color="green",
    annotation_text=f"Put 7x target: ${put_premium * 7:.2f}"
)
```

**Acceptance Criteria:**
- [ ] Payoff diagram shows 7x exit target levels
- [ ] Clear visual of "profit zone" if 7x is hit
- [ ] Shows max loss zone if neither hits 7x

---

### 10. LOW: Update Documentation Strings

**Files:** All strategy-related files

**Required Change:** Update docstrings to reflect 7x exit rule

Example for `SpreadRecommender` class:
```python
class SpreadRecommender:
    """
    Recommends straddle entry based on volatility regime prediction.

    Strategy:
    - Entry: BUY ATM call + BUY ATM put at market open
    - Exit: 7x return on cost of EACH LEG independently
    - No stop loss, no time-based exit
    - Max loss: Total premium paid (both legs expire worthless)

    The 7x rule means:
    - If call cost $1.50, exit call when value reaches $10.50
    - If put cost $1.20, exit put when value reaches $8.40
    - Each leg is managed independently
    """
```

**Acceptance Criteria:**
- [ ] Class docstrings updated
- [ ] Function docstrings updated
- [ ] README/documentation reflects 7x strategy

---

## Implementation Order

### Phase 1: Core Strategy Logic (CRITICAL)
1. [ ] Update `_calculate_risk_metrics()` with per-leg tracking
2. [ ] Remove time-based exit rules from all regimes
3. [ ] Remove stop loss from normal_vol
4. [ ] Update risk_metrics return structure

### Phase 2: Validation (HIGH)
5. [ ] Implement `simulate_trade_pnl()` function
6. [ ] Update validation metrics to track 7x hit rates
7. [ ] Update validation output to show simulated P&L

### Phase 3: UI/Display (HIGH)
8. [ ] Update `format_strategy_output()` for 7x display
9. [ ] Update confidence scorer recommendation text
10. [ ] Update payoff diagram visualization

### Phase 4: Polish (LOW)
11. [ ] Simplify regime strategies to single entry rule
12. [ ] Update all documentation

---

## Verification Tests

After implementation, verify:

1. **Strategy Output Check:**
   ```
   When regime = extreme_vol:
   - Shows "Exit CALL leg when value reaches $X.XX (7x)"
   - Shows "Exit PUT leg when value reaches $X.XX (7x)"
   - NO mention of "close before 3:30 PM"
   - NO mention of "stop loss"
   ```

2. **Validation Output Check:**
   ```
   Validation should report:
   - call_7x_hit_rate: X%
   - put_7x_hit_rate: X%
   - either_leg_7x_rate: X%
   - average_pnl_per_trade: $X.XX
   - total_simulated_pnl: $X.XX
   ```

3. **Risk Metrics Check:**
   ```python
   risk_metrics = {
       'call_premium': 1.50,
       'put_premium': 1.20,
       'call_exit_target': 10.50,  # 7x
       'put_exit_target': 8.40,    # 7x
       'max_loss': 2.70,           # Total premium
       'exit_rule': '7x per-leg or expire'
   }
   ```

---

## Summary

The current implementation has **fundamentally different exit logic** from your desired strategy:

| Aspect | Current | Your Strategy |
|--------|---------|---------------|
| Exit trigger | % profit on total position | 7x on EACH LEG |
| Time exit | Yes (3:30-3:45 PM) | No (hold to expiration) |
| Stop loss | Yes (50% for normal_vol) | No |
| Leg tracking | Combined | Independent per-leg |
| Validation | Prediction accuracy | Simulated 7x P&L |

**The 7x per-leg exit rule is the most important change** - it fundamentally changes how profits are calculated and when positions are closed.
