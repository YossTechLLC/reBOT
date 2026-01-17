# Black-Scholes Futures Implementation Checklist

**Purpose**: Systematic implementation guide for integrating Black-Scholes/Black-76 futures pricing with TimesFM forecasting.

**Last Updated**: 2026-01-16

**Reference Documents**:
- `BLACK_SCHOLES_FUTURES_IMPLEMENTATION.md` - Technical specification
- `CLAUDE.md` - Operating guide and engineering standards
- `MAP.md` - Codebase structure

---

## Phase 0: Pre-Implementation Setup

### 0.1 Environment Verification
- [ ] Verify Python version >= 3.11
- [ ] Activate virtual environment: `source .venv/bin/activate`
- [ ] Install dev dependencies: `pip install -e '.[torch,xreg]'`
- [ ] Verify TimesFM imports: `python -c "import timesfm; print(timesfm.__file__)"`

### 0.2 Codebase Orientation
- [ ] Read `src/timesfm/__init__.py` - understand export patterns
- [ ] Read `src/timesfm/configs.py` - understand dataclass patterns (frozen=True)
- [ ] Read `src/timesfm/utils/xreg_lib.py` - understand utility module patterns
- [ ] Read `src/timesfm/timesfm_2p5/timesfm_2p5_base.py` - understand forecast API

### 0.3 Project Journals Setup
- [ ] Create `BUGS.md` if not exists (per CLAUDE.md section 6)
- [ ] Create `PROGRESS.md` if not exists
- [ ] Create `DECISIONS.md` if not exists

---

## Phase 1: Core Module Structure

### 1.1 Create Module Directory
**Target**: `src/timesfm/pricing/`

- [ ] Create directory: `src/timesfm/pricing/`
- [ ] Create empty `__init__.py` (follow `torch/` pattern)

### 1.2 Create Configuration Dataclass
**File**: `src/timesfm/pricing/configs.py`

- [ ] Add 14-line Apache 2.0 copyright header
- [ ] Create `@dataclasses.dataclass(frozen=True)` for `BlackScholesConfig`:
  ```python
  risk_free_rate: float = 0.05
  dividend_yield: float = 0.0  # or convenience_yield for commodities
  model_type: Literal["black_scholes", "black_76"] = "black_76"
  volatility_source: Literal["realized", "implied", "forecast", "hybrid"] = "hybrid"
  volatility_window: int = 60  # days for realized vol calculation
  greek_calculations: tuple[str, ...] = ("delta", "gamma", "theta", "vega", "rho")
  numerical_tolerance: float = 1e-10
  max_volatility: float = 5.0  # 500% cap
  ```
- [ ] Create `@dataclasses.dataclass(frozen=True)` for `VolatilityConfig`:
  ```python
  realized_weight: float = 0.5
  forecast_weight: float = 0.5
  implied_weight: float = 0.0  # if implied data available
  annualization_factor: int = 252  # trading days
  ```
- [ ] Add Google-style docstrings with Attributes section
- [ ] Verify 2-space indentation and 88-char line limit

### 1.3 Create Base Pricing Module
**File**: `src/timesfm/pricing/black_scholes_base.py`

- [ ] Add copyright header
- [ ] Add module docstring: `"""Black-Scholes and Black-76 pricing base classes."""`
- [ ] Import structure (follow xreg_lib.py pattern):
  ```python
  import dataclasses
  import math
  from typing import Literal, Mapping, Sequence

  import numpy as np

  try:
    from scipy.stats import norm
  except ImportError:
    raise ImportError(
      "Failed to load pricing module. Install with: pip install timesfm[pricing]"
    )
  ```
- [ ] Define type aliases:
  ```python
  OptionType = Literal["call", "put"]
  _TOL = 1e-10
  ```

### 1.4 Core Pricing Functions
**File**: `src/timesfm/pricing/black_scholes_base.py` (continued)

- [ ] Implement `_calculate_d1(S, K, T, r, sigma, q) -> float`
- [ ] Implement `_calculate_d2(d1, sigma, T) -> float`
- [ ] Implement `black_scholes_price(S, K, T, r, sigma, q, option_type) -> float`:
  - [ ] Input validation (positive S, K, non-negative T, sigma)
  - [ ] Edge case: T < epsilon → return intrinsic value
  - [ ] Edge case: sigma < epsilon → return discounted intrinsic
  - [ ] Cap sigma at max_volatility (5.0)
  - [ ] Clamp d1, d2 to [-10, 10] for numerical stability
  - [ ] Ensure non-negative return value
- [ ] Implement `black76_price(F, K, T, r, sigma, option_type) -> float`:
  - [ ] Same stability patterns as black_scholes_price
  - [ ] No dividend yield (futures already incorporate carry)

### 1.5 Greeks Implementation
**File**: `src/timesfm/pricing/greeks.py`

- [ ] Add copyright header and module docstring
- [ ] Implement analytical Greeks for Black-Scholes:
  - [ ] `delta(S, K, T, r, sigma, q, option_type) -> float`
  - [ ] `gamma(S, K, T, r, sigma, q) -> float` (same for calls/puts)
  - [ ] `theta(S, K, T, r, sigma, q, option_type) -> float` (per day: /365)
  - [ ] `vega(S, K, T, r, sigma, q) -> float` (per 1%: /100)
  - [ ] `rho(S, K, T, r, sigma, q, option_type) -> float` (per 1%: /100)
- [ ] Implement analytical Greeks for Black-76:
  - [ ] `black76_delta(F, K, T, r, sigma) -> float`
  - [ ] `black76_gamma(F, K, T, r, sigma) -> float`
  - [ ] `black76_theta(F, K, T, r, sigma) -> float`
  - [ ] `black76_vega(F, K, T, r, sigma) -> float`
  - [ ] `black76_rho(F, K, T, r, sigma) -> float`
- [ ] Implement second-order Greeks:
  - [ ] `vanna(S, K, T, r, sigma, q) -> float`
  - [ ] `volga(S, K, T, r, sigma, q) -> float`
  - [ ] `charm(S, K, T, r, sigma, q) -> float`
- [ ] Implement numerical Greeks (finite difference) for verification:
  - [ ] `numerical_greeks(pricing_fn, params, epsilons) -> dict`
- [ ] Create `all_greeks(S, K, T, r, sigma, q, option_type) -> dict` wrapper

---

## Phase 2: Volatility Estimation

### 2.1 Realized Volatility
**File**: `src/timesfm/pricing/volatility.py`

- [ ] Add copyright header and module docstring
- [ ] Implement `compute_realized_volatility(prices, window, annualize) -> float`:
  - [ ] Handle log returns: `np.diff(np.log(prices))`
  - [ ] Rolling window std calculation
  - [ ] Annualization: `* np.sqrt(annualization_factor)`
  - [ ] Handle insufficient data gracefully

### 2.2 Forecast-Derived Volatility
**File**: `src/timesfm/pricing/volatility.py` (continued)

- [ ] Implement `volatility_from_quantiles(quantiles, method) -> float`:
  - [ ] Method "80_ci": `(Q0.9 - Q0.1) / (2 * 1.282)`
  - [ ] Method "60_ci": `(Q0.8 - Q0.2) / (2 * 0.842)`
  - [ ] Method "average": average spread across horizon
  - [ ] Handle quantile shape: `(batch, horizon, 10)`
  - [ ] Document quantile index mapping (index 1=Q0.1, index 9=Q0.9)

### 2.3 Hybrid Volatility Estimator
**File**: `src/timesfm/pricing/volatility.py` (continued)

- [ ] Implement `HybridVolatilityEstimator` class:
  ```python
  class HybridVolatilityEstimator:
    def __init__(self, config: VolatilityConfig): ...
    def estimate(self, prices, quantiles, implied=None) -> float: ...
  ```
  - [ ] Weighted combination: `w1*realized + w2*forecast + w3*implied`
  - [ ] Validate weights sum to 1.0
  - [ ] Handle missing components gracefully

---

## Phase 3: TimesFM Integration

### 3.1 Forecast-to-Pricing Pipeline
**File**: `src/timesfm/pricing/integration.py`

- [ ] Add copyright header and module docstring
- [ ] Create `ForecastPricingPipeline` class:
  ```python
  class ForecastPricingPipeline:
    def __init__(
      self,
      forecast_model,  # TimesFM instance
      bs_config: BlackScholesConfig,
      vol_config: VolatilityConfig,
      rate_curve: Callable[[float], float] | None = None,
    ): ...
  ```

### 3.2 Core Integration Methods
**File**: `src/timesfm/pricing/integration.py` (continued)

- [ ] Implement `forecast_spot_price(historical_data, horizon_days) -> np.ndarray`:
  - [ ] Call `model.forecast(horizon, inputs)`
  - [ ] Extract point forecast: `point_forecast[0, -1]` for final value
  - [ ] Return full horizon for path-dependent analysis

- [ ] Implement `extract_volatility(historical_data, quantiles) -> float`:
  - [ ] Use `HybridVolatilityEstimator`
  - [ ] Combine realized vol from history + forecast vol from quantiles

- [ ] Implement `get_rate(maturity_years) -> float`:
  - [ ] Use rate_curve callable if provided
  - [ ] Fall back to config default if not

### 3.3 Option Pricing Method
**File**: `src/timesfm/pricing/integration.py` (continued)

- [ ] Implement `price_option(historical_data, K, T_days, option_type, **kwargs)`:
  ```python
  def price_option(
    self,
    historical_data: np.ndarray,
    strike: float,
    days_to_maturity: int,
    option_type: OptionType = "call",
    convenience_yield: float = 0.0,
    storage_cost: float = 0.0,
  ) -> dict:
    """
    Returns:
      dict with keys: option_price, greeks, diagnostics
    """
  ```
  - [ ] Forecast spot price at maturity
  - [ ] Estimate volatility using hybrid approach
  - [ ] Get risk-free rate for maturity
  - [ ] Calculate futures price if using Black-76: `F = S * exp((r - q + c) * T)`
  - [ ] Price option using appropriate model
  - [ ] Calculate all Greeks
  - [ ] Return comprehensive result dict

### 3.4 Batch Pricing Support
**File**: `src/timesfm/pricing/integration.py` (continued)

- [ ] Implement `price_option_batch(...)`:
  - [ ] Accept list of strikes, maturities
  - [ ] Vectorize where possible using numpy
  - [ ] Return list of result dicts or DataFrame

---

## Phase 4: Validation and Error Handling

### 4.1 Input Validation Module
**File**: `src/timesfm/pricing/validation.py`

- [ ] Add copyright header and module docstring
- [ ] Implement `validate_pricing_inputs(S, K, T, r, sigma, q) -> list[str]`:
  - [ ] S > 0 (spot/futures must be positive)
  - [ ] K > 0 (strike must be positive)
  - [ ] T >= 0 (time cannot be negative)
  - [ ] 0 <= sigma <= 5.0 (volatility bounds)
  - [ ] -0.1 <= r <= 0.5 (rate sanity check)
  - [ ] -0.5 <= q <= 0.5 (yield sanity check)
  - [ ] Return list of warning/error strings

- [ ] Implement `validate_forecast_inputs(prices, horizon) -> list[str]`:
  - [ ] prices array non-empty
  - [ ] prices contain no inf values
  - [ ] horizon > 0
  - [ ] horizon <= max supported

### 4.2 Error Codes (per CLAUDE.md)
**File**: `src/timesfm/pricing/errors.py`

- [ ] Define error code constants:
  ```python
  # TFM1001 CONFIG - configuration errors
  # TFM2001 DATA - data shape/validation errors
  # TFM4001 INFERENCE - pricing calculation errors
  # TFM5001 PERF - performance warnings
  ```
- [ ] Implement `PricingError(Exception)` with error code
- [ ] Implement `ValidationError(Exception)` with error code

### 4.3 Logging Setup
**File**: All pricing modules

- [ ] Add `import logging` to each module
- [ ] Create module-level logger: `logger = logging.getLogger(__name__)`
- [ ] Log at appropriate levels:
  - [ ] DEBUG: intermediate calculations
  - [ ] INFO: successful pricing operations
  - [ ] WARNING: input validation warnings (non-fatal)
  - [ ] ERROR: pricing failures with error codes

---

## Phase 5: Put-Call Parity and Arbitrage Checks

### 5.1 Parity Verification
**File**: `src/timesfm/pricing/validation.py` (continued)

- [ ] Implement `verify_put_call_parity(call, put, S, K, T, r, q, tol) -> bool`:
  - [ ] Formula: `C - P = S*exp(-q*T) - K*exp(-r*T)`
  - [ ] Return True if within tolerance
  - [ ] Log warning if parity violated

### 5.2 Bounds Checking
**File**: `src/timesfm/pricing/validation.py` (continued)

- [ ] Implement `verify_option_bounds(price, S, K, T, r, q, option_type) -> bool`:
  - [ ] Call lower bound: `max(0, S*exp(-q*T) - K*exp(-r*T))`
  - [ ] Call upper bound: `S*exp(-q*T)`
  - [ ] Put lower bound: `max(0, K*exp(-r*T) - S*exp(-q*T))`
  - [ ] Put upper bound: `K*exp(-r*T)`
  - [ ] Log error if bounds violated

---

## Phase 6: Public API and Exports

### 6.1 Module __init__.py
**File**: `src/timesfm/pricing/__init__.py`

- [ ] Export public classes and functions:
  ```python
  from .configs import BlackScholesConfig, VolatilityConfig
  from .black_scholes_base import black_scholes_price, black76_price
  from .greeks import all_greeks, black76_greeks
  from .volatility import (
    compute_realized_volatility,
    volatility_from_quantiles,
    HybridVolatilityEstimator,
  )
  from .integration import ForecastPricingPipeline
  from .errors import PricingError, ValidationError
  ```

### 6.2 Top-Level Package Export
**File**: `src/timesfm/__init__.py`

- [ ] Add conditional import (follow existing pattern):
  ```python
  try:
    from .pricing import (
      BlackScholesConfig,
      VolatilityConfig,
      ForecastPricingPipeline,
      black_scholes_price,
      black76_price,
    )
  except ImportError:
    pass
  ```

### 6.3 Dependencies in pyproject.toml
**File**: `pyproject.toml`

- [ ] Add new optional dependency group:
  ```toml
  [project.optional-dependencies]
  pricing = [
      "scipy>=1.11.0",
  ]
  ```
- [ ] Update README install instructions

---

## Phase 7: Testing

### 7.1 Test File Structure
**Directory**: `tests/pricing/` (or `v1/tests/` if following existing pattern)

- [ ] Create `test_black_scholes.py`
- [ ] Create `test_greeks.py`
- [ ] Create `test_volatility.py`
- [ ] Create `test_integration.py`
- [ ] Create `conftest.py` with shared fixtures (optional)

### 7.2 Core Pricing Tests
**File**: `tests/pricing/test_black_scholes.py`

- [ ] Test helper: `create_sample_option_params() -> dict`
- [ ] Test: Black-Scholes call price known value
- [ ] Test: Black-Scholes put price known value
- [ ] Test: Black-76 call price known value
- [ ] Test: Black-76 put price known value
- [ ] Test: Put-call parity holds
- [ ] Test: Option bounds satisfied
- [ ] Parametrize over moneyness: ITM, ATM, OTM
- [ ] Parametrize over time: short (7d), medium (30d), long (90d)
- [ ] Test edge cases:
  - [ ] T ≈ 0 (at expiration)
  - [ ] sigma ≈ 0 (zero volatility)
  - [ ] Deep ITM and deep OTM
  - [ ] Extreme volatility (cap at 5.0)

### 7.3 Greeks Tests
**File**: `tests/pricing/test_greeks.py`

- [ ] Test: Delta in [0, 1] for calls, [-1, 0] for puts
- [ ] Test: Gamma always positive
- [ ] Test: Vega always positive
- [ ] Test: Analytical vs numerical Greeks match (within tolerance)
- [ ] Test: Greeks at ATM have expected properties
- [ ] Test: Greeks sum correctly for portfolio

### 7.4 Volatility Tests
**File**: `tests/pricing/test_volatility.py`

- [ ] Test: Realized volatility calculation correctness
- [ ] Test: Quantile-derived volatility extraction
- [ ] Test: Hybrid estimator weight validation
- [ ] Test: Handle insufficient data gracefully
- [ ] Test: Annualization factor applied correctly

### 7.5 Integration Tests
**File**: `tests/pricing/test_integration.py`

- [ ] Test: Full pipeline smoke test (mock TimesFM model)
- [ ] Test: Forecast → volatility → pricing flow
- [ ] Test: Batch pricing returns correct shapes
- [ ] Test: Rate curve integration
- [ ] Test: Convenience yield / storage cost impact

### 7.6 Run Tests
- [ ] Execute: `pytest tests/pricing/ -v`
- [ ] Verify all tests pass
- [ ] Check coverage if pytest-cov available

---

## Phase 8: Documentation and Examples

### 8.1 Docstrings
- [ ] All public functions have Google-style docstrings
- [ ] All classes have class-level docstrings
- [ ] Args, Returns, Raises sections complete
- [ ] Include usage examples in docstrings

### 8.2 Example Script
**File**: `examples/black_scholes_futures_example.py`

- [ ] Complete working example showing:
  - [ ] Loading TimesFM model
  - [ ] Historical price data preparation
  - [ ] Creating ForecastPricingPipeline
  - [ ] Pricing a call option
  - [ ] Pricing a put option
  - [ ] Extracting and interpreting Greeks
  - [ ] Running sensitivity analysis
- [ ] Include inline comments explaining each step
- [ ] Test that example runs without error

### 8.3 Update README
**File**: `README.md`

- [ ] Add section on Black-Scholes integration
- [ ] Document installation with pricing extra
- [ ] Link to example script
- [ ] Note commodity vs equity considerations

---

## Phase 9: Code Quality and Style

### 9.1 Linting
- [ ] Run ruff on all new files: `ruff check src/timesfm/pricing/`
- [ ] Fix all linting errors
- [ ] Verify 2-space indentation throughout
- [ ] Verify 88-character line limit

### 9.2 Type Hints
- [ ] All function parameters have type hints
- [ ] All return types specified
- [ ] Use `Sequence`, `Mapping` from typing where appropriate
- [ ] Run mypy if available: `mypy src/timesfm/pricing/`

### 9.3 Final Review
- [ ] No hardcoded secrets or credentials
- [ ] No debug print statements
- [ ] All TODOs addressed or documented
- [ ] Copyright headers on all files

---

## Phase 10: Project Journals Update

### 10.1 PROGRESS.md
- [ ] Update `now:` section with current status
- [ ] Update `next:` section with remaining work
- [ ] Update `done:` section with completed items

### 10.2 DECISIONS.md
- [ ] Document: Black-76 vs Black-Scholes choice
- [ ] Document: Volatility estimation approach chosen
- [ ] Document: Module location decision (pricing/ vs utils/)
- [ ] Document: Dependency choices (scipy vs vollib)

### 10.3 BUGS.md
- [ ] Document any bugs encountered during implementation
- [ ] Mark fixed bugs with resolution

---

## Acceptance Criteria

The implementation is complete when:

1. **Functional**:
   - [ ] `pip install -e '.[pricing]'` succeeds
   - [ ] `from timesfm import ForecastPricingPipeline` works
   - [ ] Can price a call option with forecasted inputs
   - [ ] Can price a put option with forecasted inputs
   - [ ] All Greeks calculate correctly
   - [ ] Put-call parity verified

2. **Tested**:
   - [ ] All unit tests pass
   - [ ] Integration test with mock model passes
   - [ ] Example script runs without error

3. **Quality**:
   - [ ] No ruff errors
   - [ ] All functions documented
   - [ ] Error handling with proper codes
   - [ ] Logging in place

4. **Documented**:
   - [ ] README updated
   - [ ] Example script provided
   - [ ] Project journals current

---

## File Summary

| File | Purpose | Est. Lines |
|------|---------|------------|
| `src/timesfm/pricing/__init__.py` | Public exports | ~30 |
| `src/timesfm/pricing/configs.py` | Dataclass configs | ~60 |
| `src/timesfm/pricing/black_scholes_base.py` | Core pricing functions | ~200 |
| `src/timesfm/pricing/greeks.py` | Greeks calculations | ~250 |
| `src/timesfm/pricing/volatility.py` | Volatility estimation | ~150 |
| `src/timesfm/pricing/integration.py` | TimesFM pipeline | ~300 |
| `src/timesfm/pricing/validation.py` | Input validation | ~100 |
| `src/timesfm/pricing/errors.py` | Error definitions | ~40 |
| `tests/pricing/test_black_scholes.py` | Pricing tests | ~150 |
| `tests/pricing/test_greeks.py` | Greeks tests | ~100 |
| `tests/pricing/test_volatility.py` | Volatility tests | ~80 |
| `tests/pricing/test_integration.py` | Integration tests | ~120 |
| `examples/black_scholes_futures_example.py` | Usage example | ~150 |

**Total estimated**: ~1,730 lines of new code

---

## Quick Reference: Key Patterns

### Import Pattern (from xreg_lib.py)
```python
try:
  from scipy.stats import norm
except ImportError:
  raise ImportError(
    "Failed to load pricing module. Install with: pip install timesfm[pricing]"
  )
```

### Dataclass Pattern (from configs.py)
```python
@dataclasses.dataclass(frozen=True)
class BlackScholesConfig:
  """Configuration for Black-Scholes pricing.

  Attributes:
    risk_free_rate: Annualized risk-free interest rate.
    ...
  """
  risk_free_rate: float = 0.05
```

### Error Pattern (from CLAUDE.md)
```python
logger.error(
  f"TFM4001 INFERENCE: Option pricing failed - price {price:.4f} "
  f"below intrinsic {intrinsic:.4f}. Check inputs: S={S}, K={K}, T={T}"
)
```

### Validation Pattern (from xreg_lib.py)
```python
def _assert_inputs(self, S, K, T):
  if S <= 0:
    raise ValueError(
      f"TFM2001 DATA: Spot price must be positive, got {S}."
    )
```

### Quantile-to-Volatility Pattern
```python
# TimesFM quantile indices: 0=point, 1=Q0.1, ..., 9=Q0.9
# Extract 80% CI volatility estimate:
vol_80ci = (quantiles[0, -1, 9] - quantiles[0, -1, 1]) / (2 * 1.282)
```

### Forecast Output Shapes
```python
point_forecast, quantile_forecast = model.forecast(horizon=30, inputs=[prices])
# point_forecast.shape = (1, 30)      # (batch, horizon)
# quantile_forecast.shape = (1, 30, 10)  # (batch, horizon, quantiles)
```

---

## Implementation Order (Recommended)

1. **Phase 0** - Setup and orientation (prerequisite)
2. **Phase 1.2-1.4** - Core pricing functions (standalone, testable)
3. **Phase 7.2** - Core pricing tests (validate Phase 1)
4. **Phase 1.5** - Greeks (depends on Phase 1)
5. **Phase 7.3** - Greeks tests (validate Phase 1.5)
6. **Phase 2** - Volatility estimation (standalone)
7. **Phase 7.4** - Volatility tests (validate Phase 2)
8. **Phase 4** - Validation and errors (used by Phase 3)
9. **Phase 5** - Parity checks (used by Phase 3)
10. **Phase 3** - Integration (brings it all together)
11. **Phase 7.5** - Integration tests (validate Phase 3)
12. **Phase 6** - Public API (exports)
13. **Phase 8** - Documentation
14. **Phase 9** - Code quality
15. **Phase 10** - Project journals

---

**End of Checklist**
