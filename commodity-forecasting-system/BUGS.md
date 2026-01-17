# BUGS.md
**HMM-Black-Scholes Commodity Forecasting System - Bug Tracking**

Format: `[status] YYYY-MM-DD - title - repro (1 line) - root cause (1 line) - fix (file/function)`

---

## Open Bugs

No open bugs currently.

---

## Fixed Bugs

No fixed bugs yet.

---

## Bug Reporting Guidelines

When reporting bugs, please include:

1. **Error Code**: Use taxonomy (TFM1001-5001) if applicable
2. **Reproduction Steps**: Minimal steps to reproduce
3. **Expected vs Actual**: What should happen vs what does happen
4. **Environment**: Python version, OS, key library versions
5. **Logs**: Relevant log excerpts from `logs/system.log`

### Example Format

```
[open] 2026-01-16 - HMM fails to converge on low-volatility data
Repro: Load COMMODITY=GC data from 2020-2021, fit HMM with n_states=3
Root cause: Low price variance causes numerical instability in covariance estimation
Fix: Add regularization term in src/models/hmm_core.py::CommodityHMM.fit()
```

---

**Last Updated**: 2026-01-16
**Status**: No bugs reported
