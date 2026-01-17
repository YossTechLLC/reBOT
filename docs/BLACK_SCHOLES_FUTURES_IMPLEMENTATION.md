# Black-Scholes Model for Futures Pricing: Clinical Implementation Guide

**Document Purpose**: Comprehensive guide for implementing Black-Scholes models for futures pricing with practical Python implementations, focusing on commodities vs equities, volatility estimation, and integration with forecasting models.

**Last Updated**: 2026-01-16

---

## 1. Black-Scholes Formula Variations: Commodities vs Equities

### 1.1 Fundamental Differences

**Black Model (Black-76) vs Standard Black-Scholes**

The Black model is the primary variant used for futures pricing, particularly for:
- Options on futures contracts
- Bond options
- Interest rate caps and floors
- Swaptions

**Key Distinction**:
- **Equities**: Spot price modeled as stochastic process
- **Commodities**: Futures price (not spot) modeled as log-normal stochastic process
  - Spot prices for commodities often don't fluctuate predictably enough to model directly
  - Fischer Black addressed this by modeling the futures price itself

### 1.2 Formula Adjustments by Asset Class

**Equities**:
```
Must account for:
- Expected dividends over option life
- Interest income
- Standard risk-free rate
```

**Commodities**:
```
Must account for:
- Storage costs over option life
- Convenience yield (analogous to dividend yield)
- No holding costs for "futures-style" options
```

**Conventional Options on Futures**:
```
- No holding costs
- Short-term rates not a factor in pricing
```

### 1.3 Volatility Smile Patterns

**Equities**:
- Skewed curves with substantially higher implied volatility for low strikes
- Reflects crash risk and leveraged exposure at lower prices

**Commodities**:
- Often reverse behavior to equities
- Higher implied volatility for higher strikes
- Reflects supply disruption risk and storage constraints

---

## 2. Volatility Estimation Methods

### 2.1 Three Primary Approaches

#### 2.1.1 Historical/Realized Volatility
**Advantages**:
- Superior and quickly-adapting estimate of current volatility
- Uses high-frequency data for more accurate measurements
- No model risk from option pricing assumptions

**Disadvantages**:
- Backward-looking (past may not predict future)
- Requires clean high-frequency data

**Implementation Note**: Realized volatility provides faster adaptation than GARCH models, which rely on slowly decaying weighted moving averages.

#### 2.1.2 GARCH Models
**Key Variants**:
- **Standard GARCH**: Captures volatility clustering and mean reversion
- **EGARCH**: Asymmetric GARCH for leverage effects
- **Realized EGARCH (REGARCH)**: Incorporates realized measures from high-frequency data
- **REGARCH-2C**: Two-component model with long memory structure

**Modern Enhancements**:
- **Realized GARCH**: Combines GARCH with realized volatility measures from high-frequency data
- **MIDAS-GARCH**: Mixed-frequency models combining daily returns with intraday volatility

**Performance**:
- GARCH models augmented with implied volatility outperform standalone GARCH
- Hybrid approaches combining GARCH + realized + implied volatility generally optimal

#### 2.1.3 Implied Volatility
**Definition**: Volatility backed out from observed option prices using Black-Scholes

**Advantages**:
- Forward-looking (market's expectation)
- Incorporates all available market information
- Real-time market sentiment

**Disadvantages**:
- Circular logic (using BS to estimate parameter for BS)
- Requires liquid option markets
- Subject to market microstructure noise

### 2.2 Recommended Hybrid Approach

**Best Practice**: Combine asymmetric GARCH with implied and realized volatility through ARMA models
```
σ_forecast = w1 * σ_GARCH + w2 * σ_realized + w3 * σ_implied
```

**Rationale**: Single-method approaches underperform. Ensemble methods:
- Capture different information sets
- Reduce model risk
- Improve forecast accuracy across market regimes

### 2.3 Research on Combined Approaches

Recent developments include:
- GARCH option pricing using volatility derivatives (VIX index, variance swaps, VIX futures/options)
- Four estimation methods incorporating both stock returns and volatility derivative prices
- Hybrid forecasting frameworks that dynamically weight different volatility sources

---

## 3. Interest Rate Curves for Different Maturities

### 3.1 Term Structure Fundamentals

**Definition**: The term structure of interest rates represents market interest rates across various maturities—a vital input for valuation of financial products.

**Also Known As**: Maturity structure, yield curve

### 3.2 Forward Curves and Futures Pricing

**Construction**:
Forward curves are derived from:
- Futures contracts
- Market swap rates
- Current outstanding Treasury instruments

**Forward Interest Rate**: Interest rate specified for a loan occurring at a future date
- Includes term structure showing different rates for different maturities
- Used for forecasting and underwriting floating/fixed-rate debt

**Important Caveat**: Forward curves should NOT be viewed as predictive of actual future interest rates—they reflect current market pricing, not forecasts.

### 3.3 Three-Factor Term Structure Model

Popular model describes yield curve changes via three independent movements:

1. **Level**: Parallel shifts (all rates move together)
2. **Steepness**: Short vs long rate differential changes
3. **Curvature**: Middle maturity rates vs short/long rates

### 3.4 2026 Interest Rate Environment

**Current Market Context** (as of early 2026):
- Fed "dot plot" median: ~3.6% end-2025, ~3.4% end-2026
- Gentle glide path, not steep drop
- Futures markets pricing additional cuts into late-2025
- Significant uncertainty around pace into 2026

**Implications for Futures Pricing**:
- Must use appropriate maturity-matched rates
- Consider term premium adjustments for longer durations
- Account for uncertainty/risk premium at longer maturities

### 3.5 Yield Curve Theories Relevant to Futures Pricing

**Expectations Hypothesis**:
- Various maturities are perfect substitutes
- Yield curve shape depends on expected future rates

**Risk Premium Theory**:
- Longer durations require risk premium
- More uncertainty and greater chance of impactful events
- Critical for pricing long-dated futures options

---

## 4. Dividend/Convenience Yield Adjustments for Commodities

### 4.1 Convenience Yield Concept

**Definition**: Implied return on holding physical inventories; adjustment to cost of carry in non-arbitrage pricing formula

**Key Properties**:
- Analogous to dividend yield on stocks
- Can be inferred from spot/forward price relationships
- Reflects benefits of physical ownership vs paper position

**Economic Interpretation**:
- Additional value from holding asset vs long forward/futures contract
- Captures optionality of physical possession
- Related to inventory management and supply chain flexibility

### 4.2 Impact on Futures Curves

**High Convenience Yield**:
- Backwardated futures curves
- Positive carry (spot > forward)
- Indicates tight supply or expected shortages
- Physical inventory highly valuable

**Low Convenience Yield**:
- Contango futures curves
- Negative carry (forward > spot)
- Ample supply, storage costs dominate
- Little premium for immediate possession

### 4.3 Storage Costs and Net Carry

**Total Cost of Carry**:
```
Cost of Carry = Storage Costs + Financing Costs - Convenience Yield
```

**For Black-Scholes Implementation**:
- Convenience yield enters as negative cost (like dividend yield)
- Reduces forward price relative to spot
- Must be estimated from forward curve shape when not directly observable

### 4.4 Dividend Adjustments in Black-Scholes

**For Equity Options**:
- Higher dividend yields → reduce fair value (lower call prices)
- Lower dividend yields → increase fair value (higher call prices)
- Can model as discrete payments or continuous yield

**Stochastic Dividend Models**:
- Advanced approaches model dividends as stochastic fraction of stock price
- Derive Black-Scholes-type equations for European options
- More realistic but computationally intensive

**Practical Implementation**:
```python
# Convenience yield (for commodities) or dividend yield (for equities)
# enters as q in modified Black-Scholes:
# F = S * exp((r - q) * T)
# where:
# S = spot price
# r = risk-free rate
# q = convenience yield or dividend yield
# T = time to maturity
```

---

## 5. Python Implementation Patterns and Numerical Stability

### 5.1 Standard Implementation Using SciPy

**Core Libraries**:
```python
import numpy as np
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma, q=0):
    """
    Black-Scholes call option pricing.

    Parameters:
    -----------
    S : float - Spot price
    K : float - Strike price
    T : float - Time to maturity (years)
    r : float - Risk-free rate
    sigma : float - Volatility
    q : float - Dividend/convenience yield (default 0)

    Returns:
    --------
    float - Call option price
    """
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price
```

### 5.2 Black-76 Model for Futures

**For Options on Futures**:
```python
def black76_call(F, K, T, r, sigma):
    """
    Black-76 model for options on futures.

    Parameters:
    -----------
    F : float - Futures price
    K : float - Strike price
    T : float - Time to maturity (years)
    r : float - Risk-free rate (for discounting)
    sigma : float - Volatility

    Returns:
    --------
    float - Call option price
    """
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return call_price
```

### 5.3 Numerical Stability Considerations

#### 5.3.1 Common Numerical Issues

**Problem 1: Near-Zero Time to Maturity**
```python
# Issue: Division by zero or near-zero when T → 0
# Solution: Add small epsilon or handle boundary case
epsilon = 1e-10
T_safe = max(T, epsilon)
```

**Problem 2: Extreme Moneyness**
```python
# Issue: norm.cdf() returns 0 or 1 for extreme values
# Solution: Use log probabilities or clamp values
if d1 > 10:  # Very deep in the money
    # Use intrinsic value approximation
    return max(0, S * np.exp(-q * T) - K * np.exp(-r * T))
```

**Problem 3: High Volatility Values**
```python
# Issue: sigma * sqrt(T) can overflow for large sigma
# Solution: Validate inputs and cap reasonable ranges
MAX_VOL = 5.0  # 500% annualized
sigma = min(sigma, MAX_VOL)
```

#### 5.3.2 Robust Implementation Pattern

```python
def robust_black_scholes(S, K, T, r, sigma, q=0, option_type='call'):
    """
    Numerically stable Black-Scholes implementation.
    """
    # Input validation
    if S <= 0 or K <= 0:
        raise ValueError("Spot and strike must be positive")
    if T < 0:
        raise ValueError("Time to maturity cannot be negative")
    if sigma < 0:
        raise ValueError("Volatility must be non-negative")

    # Handle edge cases
    if T < 1e-10:  # Essentially at expiration
        if option_type == 'call':
            return max(0, S - K)
        else:
            return max(0, K - S)

    if sigma < 1e-10:  # Essentially zero volatility
        if option_type == 'call':
            return max(0, S * np.exp(-q * T) - K * np.exp(-r * T))
        else:
            return max(0, K * np.exp(-r * T) - S * np.exp(-q * T))

    # Cap volatility at reasonable level
    sigma = min(sigma, 5.0)

    # Standard calculation
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    # Clamp d1, d2 to avoid numerical issues with norm.cdf
    d1 = np.clip(d1, -10, 10)
    d2 = np.clip(d2, -10, 10)

    discount_spot = S * np.exp(-q * T)
    discount_strike = K * np.exp(-r * T)

    if option_type == 'call':
        price = discount_spot * norm.cdf(d1) - discount_strike * norm.cdf(d2)
    else:
        price = discount_strike * norm.cdf(-d2) - discount_spot * norm.cdf(-d1)

    return max(0, price)  # Ensure non-negative
```

### 5.4 Specialized Python Libraries

#### 5.4.1 vollib
```python
# High-performance implied volatility and greeks
# Based on Peter Jaeckel's LetsBeRational algorithm
from vollib.black_scholes import black_scholes
from vollib.black_scholes.greeks.analytical import delta, gamma, theta, vega, rho

price = black_scholes('c', S, K, T, r, sigma)
```

#### 5.4.2 blackscholes (CarloLepelaars)
```python
# Supports Black-Scholes-Merton, Black-76, up to 3rd order Greeks
# Available on GitHub: github.com/CarloLepelaars/blackscholes
```

#### 5.4.3 QuantLib
```python
# Comprehensive C++ library with Python bindings
# Production-grade numerical stability
import QuantLib as ql

# Example: Valuing options on commodity futures
# See: gouthamanbalaraman.com/blog/value-options-commodity-futures-black-formula-quantlib-python.html
```

### 5.5 Finite Difference Methods for Stability

**Crank-Nicholson Method**:
- "Much more robust and stable than fully implicit and explicit methods"
- Small mistakes less impactful in later computations
- Recommended for American options or complex boundary conditions

```python
# Pseudocode structure for Crank-Nicholson
# Solve PDE: ∂V/∂t + 0.5σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0

# Time stepping with θ = 0.5 (Crank-Nicholson)
# (I - θ*dt*A) V^(n+1) = (I + (1-θ)*dt*A) V^n
```

---

## 6. Integration with Forecasting Models

### 6.1 Architecture Pattern: Forecasting → Black-Scholes Pipeline

**Conceptual Flow**:
```
Time Series Model → Spot Price Forecast → Black-Scholes Input
       |                    |                      |
   TimesFM             S(t+h) predictions      Option Pricing
   ARIMA/GARCH         + uncertainty           + Greeks
   Neural Network      bounds                  + Risk Metrics
```

### 6.2 Key Integration Challenges

#### 6.2.1 Time Horizon Alignment
```python
# Forecasting models predict at discrete horizons
forecast_horizons = [1, 7, 30, 90]  # days

# Black-Scholes requires continuous time
T_option = 30 / 365.0  # Convert to years

# Must interpolate/extrapolate forecasts to match option maturity
```

#### 6.2.2 Uncertainty Quantification
```python
# Forecasting models provide point predictions + uncertainty
# Example: TimesFM with prediction intervals
forecast_mean = model.forecast(inputs)
forecast_std = model.prediction_std  # If available

# Convert to Black-Scholes volatility input
# Option 1: Use forecast std as volatility proxy
# Option 2: Combine with historical/implied volatility
```

### 6.3 Practical Integration Pattern

```python
import numpy as np
from typing import Tuple

class ForecastBlackScholesIntegration:
    """
    Integrates spot price forecasting with Black-Scholes option pricing.
    """

    def __init__(self, forecast_model, volatility_model, rate_curve):
        """
        Parameters:
        -----------
        forecast_model : object with .forecast() method
        volatility_model : object with .predict_volatility() method
        rate_curve : callable returning rate for given maturity
        """
        self.forecast_model = forecast_model
        self.volatility_model = volatility_model
        self.rate_curve = rate_curve

    def forecast_spot_price(self, historical_data, horizon_days):
        """
        Generate spot price forecast at specified horizon.
        """
        forecast = self.forecast_model.forecast(
            inputs=historical_data,
            freq=0,  # Daily frequency
            horizon=horizon_days
        )
        return forecast

    def estimate_volatility(self, historical_data, forecast_horizon):
        """
        Estimate volatility for Black-Scholes using hybrid approach.
        """
        # Combine multiple volatility estimates
        historical_vol = self._compute_realized_volatility(historical_data)

        # If volatility model available (e.g., GARCH)
        garch_vol = self.volatility_model.predict_volatility(
            horizon=forecast_horizon
        )

        # Weighted combination
        vol_estimate = 0.5 * historical_vol + 0.5 * garch_vol

        return vol_estimate

    def price_option_from_forecast(
        self,
        historical_data: np.ndarray,
        K: float,
        T_days: int,
        option_type: str = 'call',
        q: float = 0.0
    ) -> Tuple[float, dict]:
        """
        Price option using forecasted spot price and estimated volatility.

        Returns:
        --------
        option_price : float
        diagnostics : dict with intermediate values
        """
        # 1. Forecast spot price at option maturity
        S_forecast = self.forecast_spot_price(historical_data, T_days)

        # 2. Estimate volatility
        sigma = self.estimate_volatility(historical_data, T_days)

        # 3. Get risk-free rate for appropriate maturity
        T_years = T_days / 365.0
        r = self.rate_curve(T_years)

        # 4. Price option using Black-Scholes
        option_price = robust_black_scholes(
            S=S_forecast.mean(),  # Use mean forecast
            K=K,
            T=T_years,
            r=r,
            sigma=sigma,
            q=q,
            option_type=option_type
        )

        diagnostics = {
            'forecast_spot': S_forecast.mean(),
            'forecast_std': S_forecast.std() if hasattr(S_forecast, 'std') else None,
            'volatility': sigma,
            'risk_free_rate': r,
            'time_to_maturity': T_years
        }

        return option_price, diagnostics

    @staticmethod
    def _compute_realized_volatility(prices, window=30):
        """
        Compute realized volatility from recent price history.
        """
        log_returns = np.diff(np.log(prices))
        # Annualize (assuming daily data)
        realized_vol = np.std(log_returns[-window:]) * np.sqrt(252)
        return realized_vol
```

### 6.4 Machine Learning Integration

**Recent Research Directions**:
- Neural networks outperform traditional BS in complex scenarios
- LSTM models for time-varying volatility and rate dynamics
- Hybrid approaches: ML forecasting + traditional pricing

**Extended Black-Scholes with ML**:
```python
# Pseudocode pattern
class MLEnhancedBlackScholes:
    def __init__(self, lstm_model, bs_model):
        self.lstm_model = lstm_model  # For volatility/price forecasting
        self.bs_model = bs_model      # Traditional pricing

    def price_option(self, market_data, strike, maturity):
        # Use LSTM for volatility forecasting
        vol_forecast = self.lstm_model.predict_volatility(market_data)

        # Use LSTM for spot price prediction
        spot_forecast = self.lstm_model.predict_spot(market_data)

        # Feed into Black-Scholes
        price = self.bs_model.price(
            S=spot_forecast,
            K=strike,
            T=maturity,
            sigma=vol_forecast
        )

        return price
```

### 6.5 Futures Prices and Spot Price Relationships

**Key Research Finding**: Futures prices incorporate all information useful for in-sample spot price prediction.

**Implication**: When forecasting spot prices for futures option pricing:
1. If futures market exists and is liquid: Use futures prices directly (no need to forecast spot)
2. If forecasting is needed: Ensure forecasting model accounts for forward curve information
3. Arbitrage relationships constrain forecast plausibility

**Cost of Carry Model**:
```python
def spot_from_futures_price(F, T, r, q, storage_cost=0):
    """
    Extract implied spot price from futures price.

    F = S * exp((r - q + storage_cost) * T)
    => S = F * exp(-(r - q + storage_cost) * T)
    """
    carry_cost = r - q + storage_cost
    S_implied = F * np.exp(-carry_cost * T)
    return S_implied
```

---

## 7. Greeks Calculation and Sensitivity Analysis

### 7.1 The Greeks: Definition and Interpretation

**Delta (Δ)**: Sensitivity to underlying price change
```
Δ = ∂V / ∂S

Call Delta: 0 to 1 (positive exposure)
Put Delta: -1 to 0 (negative exposure)

For hedging: Hedge ratio = -Δ
```

**Gamma (Γ)**: Rate of change of Delta
```
Γ = ∂²V / ∂S² = ∂Δ / ∂S

Always positive for long options
Measures convexity/curvature of option value
High Gamma → Delta changes rapidly → more frequent rehedging needed
```

**Theta (Θ)**: Time decay
```
Θ = ∂V / ∂t

Typically negative for long options (time decay)
Measures P&L from passage of one day
```

**Vega (ν)**: Sensitivity to volatility
```
ν = ∂V / ∂σ

Always positive for long options
Measures P&L from 1% change in implied volatility
```

**Rho (ρ)**: Sensitivity to interest rate
```
ρ = ∂V / ∂r

Call Rho: Positive (higher rates → higher call value)
Put Rho: Negative (higher rates → lower put value)
```

### 7.2 Analytical Greeks Formulas

```python
import numpy as np
from scipy.stats import norm

def bs_greeks(S, K, T, r, sigma, q=0, option_type='call'):
    """
    Calculate all Black-Scholes Greeks analytically.

    Returns:
    --------
    dict with keys: delta, gamma, theta, vega, rho
    """
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    # Common terms
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)

    discount_factor = np.exp(-q * T)
    discount_strike = np.exp(-r * T)

    if option_type == 'call':
        # Call Greeks
        delta = discount_factor * cdf_d1

        gamma = (discount_factor * pdf_d1) / (S * sigma * sqrt_T)

        theta = (
            -(S * pdf_d1 * sigma * discount_factor) / (2 * sqrt_T)
            - r * K * discount_strike * cdf_d2
            + q * S * discount_factor * cdf_d1
        ) / 365.0  # Per day

        vega = S * discount_factor * pdf_d1 * sqrt_T / 100.0  # Per 1% vol change

        rho = K * T * discount_strike * cdf_d2 / 100.0  # Per 1% rate change

    else:  # put
        delta = -discount_factor * norm.cdf(-d1)

        gamma = (discount_factor * pdf_d1) / (S * sigma * sqrt_T)  # Same as call

        theta = (
            -(S * pdf_d1 * sigma * discount_factor) / (2 * sqrt_T)
            + r * K * discount_strike * norm.cdf(-d2)
            - q * S * discount_factor * norm.cdf(-d1)
        ) / 365.0

        vega = S * discount_factor * pdf_d1 * sqrt_T / 100.0  # Same as call

        rho = -K * T * discount_strike * norm.cdf(-d2) / 100.0

    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }
```

### 7.3 Numerical Greeks via Finite Differences

**When to Use Numerical Greeks**:
- Complex payoffs (exotics, path-dependent)
- Model extensions without closed-form Greeks
- Verification of analytical formulas

```python
def numerical_greeks(pricing_function, S, K, T, r, sigma, q=0,
                    epsilon_S=0.01, epsilon_sigma=0.01, epsilon_r=0.0001, epsilon_T=1/365):
    """
    Calculate Greeks using finite difference approximations.

    Parameters:
    -----------
    pricing_function : callable that prices the option
    epsilon_* : step sizes for each parameter
    """
    # Base price
    V0 = pricing_function(S, K, T, r, sigma, q)

    # Delta: (V(S+ε) - V(S-ε)) / (2ε)
    V_up = pricing_function(S + epsilon_S, K, T, r, sigma, q)
    V_down = pricing_function(S - epsilon_S, K, T, r, sigma, q)
    delta = (V_up - V_down) / (2 * epsilon_S)

    # Gamma: (V(S+ε) - 2V(S) + V(S-ε)) / ε²
    gamma = (V_up - 2*V0 + V_down) / (epsilon_S**2)

    # Vega: (V(σ+ε) - V(σ-ε)) / (2ε)
    V_vol_up = pricing_function(S, K, T, r, sigma + epsilon_sigma, q)
    V_vol_down = pricing_function(S, K, T, r, sigma - epsilon_sigma, q)
    vega = (V_vol_up - V_vol_down) / (2 * epsilon_sigma) / 100.0  # Per 1%

    # Rho: (V(r+ε) - V(r-ε)) / (2ε)
    V_r_up = pricing_function(S, K, T, r + epsilon_r, sigma, q)
    V_r_down = pricing_function(S, K, T, r - epsilon_r, sigma, q)
    rho = (V_r_up - V_r_down) / (2 * epsilon_r) / 100.0  # Per 1%

    # Theta: -(V(T-ε) - V(T)) / ε (note the sign convention)
    V_t_down = pricing_function(S, K, T - epsilon_T, r, sigma, q)
    theta = -(V_t_down - V0) / epsilon_T

    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }
```

### 7.4 Python Libraries for Greeks

#### 7.4.1 vollib
```python
from vollib.black_scholes.greeks.analytical import (
    delta, gamma, theta, vega, rho
)

# Example
S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
flag = 'c'  # 'c' for call, 'p' for put

delta_val = delta(flag, S, K, T, r, sigma)
gamma_val = gamma(flag, S, K, T, r, sigma)
theta_val = theta(flag, S, K, T, r, sigma)
vega_val = vega(flag, S, K, T, r, sigma)
rho_val = rho(flag, S, K, T, r, sigma)
```

#### 7.4.2 Pathway Framework (Real-Time Greeks)
```python
# Pathway can compute Greeks for Black/Black-76 model in real-time
# Useful for live trading systems with streaming market data
# Supports Delta, Gamma, Theta, Vega, Rho
```

### 7.5 Greeks for Black-76 (Futures Options)

```python
def black76_greeks(F, K, T, r, sigma):
    """
    Greeks for Black-76 model (options on futures).

    Key difference from Black-Scholes:
    - No dividend yield (q)
    - Underlying is futures price (F) not spot (S)
    - All Greeks relative to futures price
    """
    sqrt_T = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    discount = np.exp(-r * T)

    # Delta (w.r.t. futures price)
    delta = discount * cdf_d1

    # Gamma
    gamma = (discount * pdf_d1) / (F * sigma * sqrt_T)

    # Theta
    theta = (
        -F * discount * pdf_d1 * sigma / (2 * sqrt_T)
        - r * discount * (F * cdf_d1 - K * cdf_d2)
    ) / 365.0

    # Vega
    vega = F * discount * pdf_d1 * sqrt_T / 100.0

    # Rho (less meaningful for futures, but included for completeness)
    rho = -T * discount * (F * cdf_d1 - K * cdf_d2) / 100.0

    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }
```

### 7.6 Sensitivity Analysis and Risk Management

#### 7.6.1 Portfolio Greeks

```python
class OptionPortfolio:
    """
    Manage portfolio-level Greeks for risk management.
    """

    def __init__(self):
        self.positions = []  # List of (option, quantity) tuples

    def add_position(self, option_params, quantity):
        """
        Add option position to portfolio.

        option_params: dict with S, K, T, r, sigma, q, option_type
        quantity: positive for long, negative for short
        """
        self.positions.append((option_params, quantity))

    def calculate_portfolio_greeks(self):
        """
        Calculate aggregate Greeks for entire portfolio.
        """
        portfolio_greeks = {
            'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0
        }

        for option_params, quantity in self.positions:
            greeks = bs_greeks(**option_params)

            for greek_name, greek_value in greeks.items():
                portfolio_greeks[greek_name] += quantity * greek_value

        return portfolio_greeks

    def delta_hedge(self, spot_price):
        """
        Calculate number of underlying units needed to delta hedge.
        """
        portfolio_greeks = self.calculate_portfolio_greeks()
        portfolio_delta = portfolio_greeks['delta']

        # To hedge, need -Delta units of underlying
        hedge_quantity = -portfolio_delta

        return hedge_quantity
```

#### 7.6.2 Scenario Analysis

```python
def scenario_analysis(S_base, K, T, r_base, sigma_base, q=0, option_type='call'):
    """
    Analyze option value under different market scenarios.
    """
    import pandas as pd

    # Define scenarios
    scenarios = {
        'Base Case': (S_base, r_base, sigma_base),
        'Spot +10%': (S_base * 1.1, r_base, sigma_base),
        'Spot -10%': (S_base * 0.9, r_base, sigma_base),
        'Vol +5%': (S_base, r_base, sigma_base + 0.05),
        'Vol -5%': (S_base, r_base, sigma_base - 0.05),
        'Rate +1%': (S_base, r_base + 0.01, sigma_base),
        'Rate -1%': (S_base, r_base - 0.01, sigma_base),
    }

    results = []

    for scenario_name, (S, r, sigma) in scenarios.items():
        price = robust_black_scholes(S, K, T, r, sigma, q, option_type)
        greeks = bs_greeks(S, K, T, r, sigma, q, option_type)

        results.append({
            'Scenario': scenario_name,
            'Spot': S,
            'Rate': r,
            'Volatility': sigma,
            'Option_Price': price,
            **greeks
        })

    return pd.DataFrame(results)
```

#### 7.6.3 Greeks Visualization

```python
def visualize_greeks(K, T, r, sigma, q=0, S_range=None):
    """
    Visualize how Greeks change with spot price.
    """
    import matplotlib.pyplot as plt

    if S_range is None:
        S_range = np.linspace(K * 0.5, K * 1.5, 100)

    call_deltas = []
    call_gammas = []
    call_vegas = []
    call_thetas = []

    for S in S_range:
        greeks = bs_greeks(S, K, T, r, sigma, q, 'call')
        call_deltas.append(greeks['delta'])
        call_gammas.append(greeks['gamma'])
        call_vegas.append(greeks['vega'])
        call_thetas.append(greeks['theta'])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(S_range, call_deltas)
    axes[0, 0].set_title('Delta')
    axes[0, 0].set_xlabel('Spot Price')
    axes[0, 0].axvline(K, color='r', linestyle='--', label='Strike')
    axes[0, 0].legend()

    axes[0, 1].plot(S_range, call_gammas)
    axes[0, 1].set_title('Gamma')
    axes[0, 1].set_xlabel('Spot Price')
    axes[0, 1].axvline(K, color='r', linestyle='--', label='Strike')
    axes[0, 1].legend()

    axes[1, 0].plot(S_range, call_vegas)
    axes[1, 0].set_title('Vega')
    axes[1, 0].set_xlabel('Spot Price')
    axes[1, 0].axvline(K, color='r', linestyle='--', label='Strike')
    axes[1, 0].legend()

    axes[1, 1].plot(S_range, call_thetas)
    axes[1, 1].set_title('Theta')
    axes[1, 1].set_xlabel('Spot Price')
    axes[1, 1].axvline(K, color='r', linestyle='--', label='Strike')
    axes[1, 1].legend()

    plt.tight_layout()
    return fig
```

### 7.7 Advanced Greeks: Second and Third Order

**Vanna**: Sensitivity of Delta to volatility change
```
Vanna = ∂²V / (∂S ∂σ) = ∂Δ / ∂σ = ∂ν / ∂S
```

**Volga (Vomma)**: Sensitivity of Vega to volatility change
```
Volga = ∂²V / ∂σ² = ∂ν / ∂σ
```

**Charm**: Rate of change of Delta over time
```
Charm = ∂²V / (∂S ∂t) = ∂Δ / ∂t
```

```python
def second_order_greeks(S, K, T, r, sigma, q=0):
    """
    Calculate second-order Greeks (Vanna, Volga, Charm).
    """
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    pdf_d1 = norm.pdf(d1)
    discount_factor = np.exp(-q * T)

    # Vanna
    vanna = -discount_factor * pdf_d1 * d2 / sigma

    # Volga
    volga = discount_factor * S * pdf_d1 * sqrt_T * d1 * d2 / sigma

    # Charm (per day)
    charm = -discount_factor * pdf_d1 * (
        2 * (r - q) * T - d2 * sigma * sqrt_T
    ) / (2 * T * sqrt_T) / 365.0

    return {
        'vanna': vanna,
        'volga': volga,
        'charm': charm
    }
```

---

## 8. Complete Working Example: TimesFM + Black-Scholes Pipeline

```python
"""
Complete example: Forecast commodity spot prices with TimesFM,
then price futures options using Black-76 model.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

# Assume TimesFM model loaded
# from timesfm import TimesFmHparams, TimesFmCheckpoint
# model = ...

class CommodityFuturesOptionPricer:
    """
    End-to-end pipeline for forecasting-based futures option pricing.
    """

    def __init__(self, forecast_model, rate_curve_func):
        self.forecast_model = forecast_model
        self.rate_curve_func = rate_curve_func
        self.price_history = []
        self.volatility_window = 60  # days for realized vol

    def update_price_history(self, prices):
        """Store historical prices for volatility estimation."""
        self.price_history = prices

    def forecast_futures_price(self, horizon_days):
        """
        Forecast futures price at specified horizon.

        For commodities, can forecast spot and apply cost of carry,
        or forecast futures directly if historical futures data used.
        """
        # Use TimesFM to forecast spot prices
        forecast = self.forecast_model.forecast(
            inputs=np.array([self.price_history]),
            freq=0,  # Daily
            horizon=horizon_days
        )

        # Extract mean prediction
        S_forecast = forecast[0, -1]  # Last point in horizon

        return S_forecast

    def estimate_realized_volatility(self):
        """
        Estimate annualized realized volatility from recent price history.
        """
        prices = np.array(self.price_history[-self.volatility_window:])
        log_returns = np.diff(np.log(prices))

        # Annualize assuming 252 trading days
        realized_vol = np.std(log_returns) * np.sqrt(252)

        return realized_vol

    def price_futures_option(
        self,
        K: float,
        T_days: int,
        option_type: str = 'call',
        convenience_yield: float = 0.0,
        storage_cost: float = 0.0
    ):
        """
        Price futures option using Black-76 model with forecasted inputs.

        Parameters:
        -----------
        K : Strike price
        T_days : Days to maturity
        option_type : 'call' or 'put'
        convenience_yield : Annualized convenience yield
        storage_cost : Annualized storage cost

        Returns:
        --------
        dict with price and diagnostics
        """
        # 1. Forecast spot price
        S_forecast = self.forecast_futures_price(T_days)

        # 2. Calculate futures price from spot using cost of carry
        T_years = T_days / 365.0
        r = self.rate_curve_func(T_years)

        # F = S * exp((r - q + storage) * T)
        carry_cost = r - convenience_yield + storage_cost
        F = S_forecast * np.exp(carry_cost * T_years)

        # 3. Estimate volatility
        sigma = self.estimate_realized_volatility()

        # 4. Price option using Black-76
        d1 = (np.log(F / K) + 0.5 * sigma**2 * T_years) / (sigma * np.sqrt(T_years))
        d2 = d1 - sigma * np.sqrt(T_years)

        discount = np.exp(-r * T_years)

        if option_type == 'call':
            option_price = discount * (F * norm.cdf(d1) - K * norm.cdf(d2))
        else:  # put
            option_price = discount * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

        # 5. Calculate Greeks
        greeks = self._calculate_black76_greeks(F, K, T_years, r, sigma)

        # 6. Return comprehensive results
        return {
            'option_price': option_price,
            'spot_forecast': S_forecast,
            'futures_price': F,
            'volatility': sigma,
            'risk_free_rate': r,
            'time_to_maturity_years': T_years,
            'greeks': greeks,
            'inputs': {
                'strike': K,
                'days_to_maturity': T_days,
                'option_type': option_type,
                'convenience_yield': convenience_yield,
                'storage_cost': storage_cost
            }
        }

    def _calculate_black76_greeks(self, F, K, T, r, sigma):
        """Calculate Black-76 Greeks."""
        sqrt_T = np.sqrt(T)
        d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        cdf_d2 = norm.cdf(d2)
        discount = np.exp(-r * T)

        return {
            'delta': discount * cdf_d1,
            'gamma': (discount * pdf_d1) / (F * sigma * sqrt_T),
            'theta': (
                -F * discount * pdf_d1 * sigma / (2 * sqrt_T)
                - r * discount * (F * cdf_d1 - K * cdf_d2)
            ) / 365.0,
            'vega': F * discount * pdf_d1 * sqrt_T / 100.0,
            'rho': -T * discount * (F * cdf_d1 - K * cdf_d2) / 100.0
        }

    def run_sensitivity_analysis(self, K, T_days, option_type='call'):
        """
        Run sensitivity analysis on key parameters.
        """
        results = []

        # Base case
        base_result = self.price_futures_option(K, T_days, option_type)
        results.append(('Base Case', base_result))

        # Volatility scenarios
        orig_window = self.volatility_window
        for vol_window in [30, 90, 120]:
            self.volatility_window = vol_window
            result = self.price_futures_option(K, T_days, option_type)
            results.append((f'Vol Window {vol_window}d', result))
        self.volatility_window = orig_window

        # Convenience yield scenarios
        for cy in [-0.05, 0.0, 0.05, 0.10]:
            result = self.price_futures_option(
                K, T_days, option_type, convenience_yield=cy
            )
            results.append((f'Convenience Yield {cy:.1%}', result))

        return results


# Example usage
if __name__ == '__main__':
    # Mock data and models for demonstration
    np.random.seed(42)

    # Generate synthetic price history
    price_history = 100 * np.exp(np.cumsum(np.random.randn(252) * 0.02))

    # Mock forecast model
    class MockForecastModel:
        def forecast(self, inputs, freq, horizon):
            # Simple random walk forecast
            last_price = inputs[0, -1]
            forecast_path = last_price * np.exp(
                np.cumsum(np.random.randn(horizon) * 0.02)
            )
            return np.array([forecast_path])

    # Mock rate curve (flat at 5%)
    rate_curve = lambda T: 0.05

    # Initialize pricer
    pricer = CommodityFuturesOptionPricer(MockForecastModel(), rate_curve)
    pricer.update_price_history(price_history)

    # Price a 30-day call option
    result = pricer.price_futures_option(
        K=price_history[-1],  # ATM strike
        T_days=30,
        option_type='call',
        convenience_yield=0.02
    )

    print("Option Pricing Result:")
    print(f"Option Price: ${result['option_price']:.2f}")
    print(f"Spot Forecast: ${result['spot_forecast']:.2f}")
    print(f"Futures Price: ${result['futures_price']:.2f}")
    print(f"Implied Volatility: {result['volatility']:.1%}")
    print("\nGreeks:")
    for greek, value in result['greeks'].items():
        print(f"  {greek.capitalize()}: {value:.4f}")
```

---

## 9. Production Considerations and Best Practices

### 9.1 Model Risk Management

**Validation Requirements**:
- Compare model prices against market prices regularly
- Track pricing errors and investigate large discrepancies
- Implement P&L attribution (greeks vs realized)
- Maintain model performance dashboards

### 9.2 Data Quality Checks

```python
def validate_inputs(S, K, T, r, sigma, q=0):
    """
    Validate all inputs before pricing.
    """
    errors = []

    if S <= 0:
        errors.append("Spot/Futures price must be positive")
    if K <= 0:
        errors.append("Strike price must be positive")
    if T < 0:
        errors.append("Time to maturity cannot be negative")
    if T > 10:
        errors.append("Warning: Time to maturity > 10 years unusual")
    if sigma < 0:
        errors.append("Volatility must be non-negative")
    if sigma > 5.0:
        errors.append("Warning: Volatility > 500% extremely high")
    if r < -0.1 or r > 0.5:
        errors.append("Warning: Risk-free rate outside normal range")
    if q < -0.5 or q > 0.5:
        errors.append("Warning: Dividend/convenience yield unusual")

    return errors
```

### 9.3 Performance Optimization

**Vectorization**:
```python
# Price multiple options simultaneously
def vectorized_black_scholes(S, K, T, r, sigma, q=0):
    """
    S, K, T, r, sigma, q can be arrays for bulk pricing.
    """
    # All operations are element-wise
    # NumPy/SciPy handle vectorization efficiently
    pass
```

**Caching**:
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_norm_cdf(x_rounded):
    """Cache normal CDF values for speed."""
    return norm.cdf(x_rounded)
```

### 9.4 Error Handling and Logging

```python
import logging

logger = logging.getLogger(__name__)

def safe_option_price(S, K, T, r, sigma, q=0, option_type='call'):
    """
    Production-ready pricing with comprehensive error handling.
    """
    try:
        # Validate inputs
        errors = validate_inputs(S, K, T, r, sigma, q)
        if errors:
            logger.warning(f"Input validation warnings: {errors}")

        # Price option
        price = robust_black_scholes(S, K, T, r, sigma, q, option_type)

        # Sanity check output
        if option_type == 'call':
            intrinsic = max(0, S * np.exp(-q * T) - K * np.exp(-r * T))
        else:
            intrinsic = max(0, K * np.exp(-r * T) - S * np.exp(-q * T))

        if price < intrinsic - 1e-6:
            logger.error(
                f"TFM4001 INFERENCE: Option price {price:.4f} below "
                f"intrinsic {intrinsic:.4f}. Check inputs."
            )
            return None

        logger.info(f"Successfully priced {option_type} option: {price:.4f}")
        return price

    except Exception as e:
        logger.error(f"TFM4001 INFERENCE: Option pricing failed: {str(e)}")
        return None
```

### 9.5 Integration Testing

```python
def test_black_scholes_parity():
    """
    Test put-call parity: C - P = S*e^(-qT) - K*e^(-rT)
    """
    S, K, T, r, sigma, q = 100, 100, 1.0, 0.05, 0.2, 0.02

    call_price = robust_black_scholes(S, K, T, r, sigma, q, 'call')
    put_price = robust_black_scholes(S, K, T, r, sigma, q, 'put')

    lhs = call_price - put_price
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)

    assert np.abs(lhs - rhs) < 1e-6, "Put-call parity violated"
    print("Put-call parity test: PASSED")

def test_option_bounds():
    """
    Test option price bounds.
    """
    S, K, T, r, sigma, q = 100, 100, 1.0, 0.05, 0.2, 0.0

    call_price = robust_black_scholes(S, K, T, r, sigma, q, 'call')

    # Lower bound: max(0, S - K*e^(-rT))
    lower_bound = max(0, S - K * np.exp(-r * T))
    # Upper bound: S
    upper_bound = S

    assert call_price >= lower_bound - 1e-6, "Call price below lower bound"
    assert call_price <= upper_bound + 1e-6, "Call price above upper bound"
    print("Option bounds test: PASSED")
```

---

## 10. Further Resources and References

### 10.1 Academic Resources
- Black, F., & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities"
- Black, F. (1976). "The Pricing of Commodity Contracts"
- Hull, J. "Options, Futures, and Other Derivatives" (latest edition)

### 10.2 Python Libraries
- **vollib**: Implied volatility and Greeks (Peter Jaeckel's LetsBeRational)
- **QuantLib**: Comprehensive quantitative finance library
- **NumPy/SciPy**: Numerical computing foundations
- **pandas**: Data manipulation for time series

### 10.3 Online Resources
- QuantLib Python tutorials: gouthamanbalaraman.com
- Options Greeks visualization: github.com/AmirDehkordi/OptionGreeks
- Black-Scholes implementations: github.com/CarloLepelaars/blackscholes

---

## Appendix: Error Taxonomy

**TFM1001 CONFIG**: Bad configuration/environment/flags
**TFM2001 DATA**: Bad shapes, missing columns, leakage risks
**TFM3001 CHECKPOINT**: Missing or incompatible checkpoint
**TFM4001 INFERENCE**: Runtime/OOM/NaN/precision issues
**TFM5001 PERF**: Regression or unexpected slowness

---

**End of Document**
