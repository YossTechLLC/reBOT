# PHASE 0 VALIDATION REPORT - Intraday Regime Shift Detection

**Generated:** 2026-01-17 07:17:52

---

## Executive Summary

**Decision:** NO-GO

**Recommendation:** Reconsider approach or pivot strategy

---

## Data Summary

- **Period:** 2025-10-22 09:30:00-04:00 to 2026-01-16 15:45:00-05:00
- **15-minute bars:** 1536
- **Trading days analyzed:** 59
- **Features engineered:** 21

## HMM Regime Detection

**Regimes Identified:**

- **Regime 0 (neutral):**
  - Mean return: 0.0027% per 15-min
  - Volatility: 0.0382%
  - Persistence: 89.3%
  - Frequency: 26.0%

- **Regime 1 (bear):**
  - Mean return: 0.0009% per 15-min
  - Volatility: 0.0913%
  - Persistence: 88.6%
  - Frequency: 43.3%

- **Regime 2 (bull):**
  - Mean return: 0.0055% per 15-min
  - Volatility: 0.2509%
  - Persistence: 91.8%
  - Frequency: 30.7%

## Regime Shift Analysis

- **Days with shift (9:30 AM → 12:00 PM):** 27 / 59 (45.8%)
- **Avg move (shift days):** 0.256%
- **Avg move (no-shift days):** 0.392%
- **T-test p-value:** 0.1444

**Shift Type Breakdown:**

- bull → bear: 12 (44.4%)
- bear → neutral: 8 (29.6%)
- bull → neutral: 5 (18.5%)
- bear → bull: 1 (3.7%)
- neutral → bear: 1 (3.7%)

## Predictability Analysis

- **AUC (ROC):** 0.754
- **Prediction threshold:** 0.50
- **Samples:** 58

**Feature Importance:**

- overnight_gap: +0.7995
- volatility_5: +0.4375
- volatility_10: -0.5938
- momentum_5: -0.1262
- volume_ratio: +0.6824

## Simulated Trading Performance

- **Total trades:** 25
- **Win rate:** 4.0%
- **Avg win:** $228.08
- **Avg loss:** $-162.15
- **Profit factor:** 0.06
- **Total P&L:** $-3663.51
- **Expectancy:** $-146.54/trade

## Validation Criteria Results

- ✅ PASS: Regime shifts occur >7 days (~13% rate)
- ✅ PASS: Shift prediction AUC >0.65
- ❌ FAIL: Average move during shift >0.5%
- ❌ FAIL: Simulated win rate >50%
- ❌ FAIL: Expectancy >$20/trade

---

## Conclusion

Some validation criteria failed. Consider:

**Options:**
1. Refine HMM parameters (more/fewer states)
2. Try different features for prediction
3. Adjust shift definition (different time windows)
4. Pivot to daily swing trading instead of intraday
