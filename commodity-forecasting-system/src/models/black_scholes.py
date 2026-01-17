"""
Black-Scholes Options Pricing Module
Black-76 model for commodity futures options with Greeks.

Model:
The Black-76 model is the standard for pricing European options on futures:

Call: C = e^(-rT)[F₀N(d₁) - KN(d₂)]
Put:  P = e^(-rT)[KN(-d₂) - F₀N(-d₁)]

where:
d₁ = [ln(F₀/K) + (σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T

F₀: Futures price
K: Strike price
σ: Volatility
r: Risk-free rate
T: Time to maturity
N(): Standard normal CDF
"""

import logging
from typing import Dict, Optional, Tuple, Literal
import numpy as np
import pandas as pd
from scipy.stats import norm
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OptionResult:
    """Container for option pricing results."""
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    implied_vol: Optional[float] = None


class Black76Pricer:
    """
    Black-76 model for futures options pricing with Greeks.

    The Black-76 model is a variant of Black-Scholes specifically
    designed for futures and forward contracts. It's the industry
    standard for commodity options.
    """

    def __init__(self, config: Dict):
        """
        Initialize Black-76 pricer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.bs_config = config.get('black_scholes', {})

        # Numerical parameters
        self.epsilon = self.bs_config.get('epsilon', 1e-8)
        self.max_iterations = self.bs_config.get('max_iterations', 100)
        self.iv_tolerance = self.bs_config.get('iv_tolerance', 1e-6)

        logger.info("Initialized Black76Pricer")

    def price(
        self,
        futures_price: float,
        strike: float,
        volatility: float,
        time_to_maturity: float,
        risk_free_rate: float,
        option_type: Literal['call', 'put'] = 'call'
    ) -> float:
        """
        Calculate option price using Black-76 model.

        Args:
            futures_price: Current futures price
            strike: Option strike price
            volatility: Implied/expected volatility (annualized)
            time_to_maturity: Time to maturity in years
            risk_free_rate: Risk-free rate (decimal)
            option_type: 'call' or 'put'

        Returns:
            Option price
        """
        # Validate inputs
        self._validate_inputs(
            futures_price, strike, volatility, time_to_maturity, risk_free_rate
        )

        # Handle edge cases
        if time_to_maturity < self.epsilon:
            # At expiration, option value is intrinsic value
            if option_type == 'call':
                return max(futures_price - strike, 0)
            else:
                return max(strike - futures_price, 0)

        # Calculate d1 and d2
        d1, d2 = self._calculate_d1_d2(
            futures_price, strike, volatility, time_to_maturity
        )

        # Calculate option price
        discount_factor = np.exp(-risk_free_rate * time_to_maturity)

        if option_type == 'call':
            price = discount_factor * (
                futures_price * norm.cdf(d1) - strike * norm.cdf(d2)
            )
        elif option_type == 'put':
            price = discount_factor * (
                strike * norm.cdf(-d2) - futures_price * norm.cdf(-d1)
            )
        else:
            raise ValueError(
                f"TFM1001 CONFIG: Invalid option_type '{option_type}'. "
                f"Must be 'call' or 'put'."
            )

        logger.debug(
            f"{option_type.upper()} price: {price:.4f} "
            f"(F={futures_price:.2f}, K={strike:.2f}, σ={volatility:.4f}, "
            f"T={time_to_maturity:.2f}, r={risk_free_rate:.4f})"
        )

        return price

    def greeks(
        self,
        futures_price: float,
        strike: float,
        volatility: float,
        time_to_maturity: float,
        risk_free_rate: float,
        option_type: Literal['call', 'put'] = 'call'
    ) -> OptionResult:
        """
        Calculate option price and all Greeks.

        Greeks measure sensitivities to various parameters:
        - Delta (Δ): Sensitivity to futures price (∂V/∂F)
        - Gamma (Γ): Rate of change of delta (∂²V/∂F²)
        - Theta (Θ): Time decay (∂V/∂t)
        - Vega (ν): Sensitivity to volatility (∂V/∂σ)
        - Rho (ρ): Sensitivity to interest rate (∂V/∂r)

        Args:
            futures_price: Current futures price
            strike: Option strike price
            volatility: Implied volatility
            time_to_maturity: Time to maturity (years)
            risk_free_rate: Risk-free rate
            option_type: 'call' or 'put'

        Returns:
            OptionResult with price and all Greeks
        """
        # Validate inputs
        self._validate_inputs(
            futures_price, strike, volatility, time_to_maturity, risk_free_rate
        )

        # Calculate option price
        option_price = self.price(
            futures_price, strike, volatility, time_to_maturity,
            risk_free_rate, option_type
        )

        # Handle edge cases
        if time_to_maturity < self.epsilon:
            # At expiration, Greeks are either 0 or undefined
            if option_type == 'call':
                delta = 1.0 if futures_price > strike else 0.0
            else:
                delta = -1.0 if futures_price < strike else 0.0

            return OptionResult(
                price=option_price,
                delta=delta,
                gamma=0.0,
                theta=0.0,
                vega=0.0,
                rho=0.0
            )

        # Calculate d1 and d2
        d1, d2 = self._calculate_d1_d2(
            futures_price, strike, volatility, time_to_maturity
        )

        # Common terms
        sqrt_t = np.sqrt(time_to_maturity)
        discount_factor = np.exp(-risk_free_rate * time_to_maturity)
        pdf_d1 = norm.pdf(d1)

        # Delta: ∂V/∂F
        if option_type == 'call':
            delta = discount_factor * norm.cdf(d1)
        else:
            delta = -discount_factor * norm.cdf(-d1)

        # Gamma: ∂²V/∂F² (same for calls and puts)
        gamma = (discount_factor * pdf_d1) / (futures_price * volatility * sqrt_t)

        # Vega: ∂V/∂σ (same for calls and puts)
        # Note: Vega is typically quoted per 1% change in volatility
        vega = discount_factor * futures_price * pdf_d1 * sqrt_t / 100.0

        # Theta: ∂V/∂t (convert to per-day basis)
        theta_term1 = -(discount_factor * futures_price * pdf_d1 * volatility) / (
            2 * sqrt_t
        )

        if option_type == 'call':
            theta_term2 = -risk_free_rate * discount_factor * (
                futures_price * norm.cdf(d1) - strike * norm.cdf(d2)
            )
        else:
            theta_term2 = -risk_free_rate * discount_factor * (
                strike * norm.cdf(-d2) - futures_price * norm.cdf(-d1)
            )

        theta = (theta_term1 + theta_term2) / 365.0  # Convert to per-day

        # Rho: ∂V/∂r (per 1% change in interest rate)
        rho = -time_to_maturity * option_price / 100.0

        logger.debug(
            f"{option_type.upper()} Greeks: Δ={delta:.4f}, Γ={gamma:.6f}, "
            f"Θ={theta:.4f}, ν={vega:.4f}, ρ={rho:.4f}"
        )

        return OptionResult(
            price=option_price,
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho
        )

    def implied_volatility(
        self,
        market_price: float,
        futures_price: float,
        strike: float,
        time_to_maturity: float,
        risk_free_rate: float,
        option_type: Literal['call', 'put'] = 'call',
        initial_guess: float = 0.30
    ) -> float:
        """
        Calculate implied volatility using Newton-Raphson method.

        Implied volatility is the volatility that makes the theoretical
        price equal to the observed market price.

        Args:
            market_price: Observed market price
            futures_price: Current futures price
            strike: Strike price
            time_to_maturity: Time to maturity (years)
            risk_free_rate: Risk-free rate
            option_type: 'call' or 'put'
            initial_guess: Starting point for iteration

        Returns:
            Implied volatility (annualized)
        """
        # Validate inputs
        if market_price <= 0:
            raise ValueError(
                f"TFM2001 DATA: Market price must be positive (got {market_price})"
            )

        # Check for intrinsic value violations
        intrinsic_value = self._intrinsic_value(
            futures_price, strike, option_type
        )

        if market_price < intrinsic_value - self.epsilon:
            raise ValueError(
                f"TFM2001 DATA: Market price ({market_price:.4f}) below "
                f"intrinsic value ({intrinsic_value:.4f}). Arbitrage opportunity?"
            )

        # Newton-Raphson iteration
        vol = initial_guess

        for iteration in range(self.max_iterations):
            # Calculate price and vega at current volatility
            price = self.price(
                futures_price, strike, vol, time_to_maturity,
                risk_free_rate, option_type
            )

            result = self.greeks(
                futures_price, strike, vol, time_to_maturity,
                risk_free_rate, option_type
            )

            vega_full = result.vega * 100.0  # Convert back to full vega

            # Check convergence
            price_diff = market_price - price

            if abs(price_diff) < self.iv_tolerance:
                logger.debug(
                    f"Implied volatility converged: {vol:.4f} "
                    f"({iteration+1} iterations)"
                )
                return vol

            # Newton-Raphson update: vol_new = vol_old + (target - price) / vega
            if abs(vega_full) < self.epsilon:
                logger.warning(
                    f"TFM4001 INFERENCE: Vega too small ({vega_full:.8f}). "
                    f"Cannot continue Newton-Raphson."
                )
                break

            vol += price_diff / vega_full

            # Keep volatility in reasonable bounds
            vol = np.clip(vol, 0.01, 5.0)

        # Failed to converge
        logger.warning(
            f"TFM4001 INFERENCE: Implied volatility did not converge after "
            f"{self.max_iterations} iterations. Last estimate: {vol:.4f}"
        )

        return vol

    def put_call_parity_check(
        self,
        call_price: float,
        put_price: float,
        futures_price: float,
        strike: float,
        time_to_maturity: float,
        risk_free_rate: float,
        tolerance: float = 0.01
    ) -> Tuple[bool, float]:
        """
        Check put-call parity for futures options.

        Put-Call Parity for futures options:
        C - P = e^(-rT)(F - K)

        Args:
            call_price: Call option price
            put_price: Put option price
            futures_price: Futures price
            strike: Strike price
            time_to_maturity: Time to maturity
            risk_free_rate: Risk-free rate
            tolerance: Maximum acceptable deviation

        Returns:
            Tuple of (parity_holds, deviation)
        """
        # Left side: C - P
        lhs = call_price - put_price

        # Right side: e^(-rT)(F - K)
        discount_factor = np.exp(-risk_free_rate * time_to_maturity)
        rhs = discount_factor * (futures_price - strike)

        # Deviation
        deviation = abs(lhs - rhs)

        parity_holds = deviation <= tolerance

        if not parity_holds:
            logger.warning(
                f"TFM4001 INFERENCE: Put-call parity violation detected. "
                f"Deviation: {deviation:.4f} (tolerance: {tolerance})"
            )
        else:
            logger.debug(
                f"Put-call parity satisfied. Deviation: {deviation:.6f}"
            )

        return parity_holds, deviation

    def _calculate_d1_d2(
        self,
        futures_price: float,
        strike: float,
        volatility: float,
        time_to_maturity: float
    ) -> Tuple[float, float]:
        """
        Calculate d1 and d2 for Black-76 formula.

        d₁ = [ln(F/K) + (σ²/2)T] / (σ√T)
        d₂ = d₁ - σ√T

        Args:
            futures_price: Futures price
            strike: Strike price
            volatility: Volatility
            time_to_maturity: Time to maturity

        Returns:
            Tuple of (d1, d2)
        """
        sqrt_t = np.sqrt(time_to_maturity)
        vol_sqrt_t = volatility * sqrt_t

        d1 = (np.log(futures_price / strike) + 0.5 * volatility**2 * time_to_maturity) / vol_sqrt_t
        d2 = d1 - vol_sqrt_t

        return d1, d2

    def _intrinsic_value(
        self,
        futures_price: float,
        strike: float,
        option_type: Literal['call', 'put']
    ) -> float:
        """Calculate intrinsic value of option."""
        if option_type == 'call':
            return max(futures_price - strike, 0)
        else:
            return max(strike - futures_price, 0)

    def _validate_inputs(
        self,
        futures_price: float,
        strike: float,
        volatility: float,
        time_to_maturity: float,
        risk_free_rate: float
    ):
        """Validate option pricing inputs."""
        if futures_price <= 0:
            raise ValueError(
                f"TFM2001 DATA: Futures price must be positive (got {futures_price})"
            )
        if strike <= 0:
            raise ValueError(
                f"TFM2001 DATA: Strike must be positive (got {strike})"
            )
        if volatility < 0:
            raise ValueError(
                f"TFM2001 DATA: Volatility must be non-negative (got {volatility})"
            )
        if time_to_maturity < 0:
            raise ValueError(
                f"TFM2001 DATA: Time to maturity must be non-negative (got {time_to_maturity})"
            )

    def option_chain(
        self,
        futures_price: float,
        strikes: np.ndarray,
        volatility: float,
        time_to_maturity: float,
        risk_free_rate: float
    ) -> pd.DataFrame:
        """
        Generate option chain for range of strikes.

        Args:
            futures_price: Current futures price
            strikes: Array of strike prices
            volatility: Implied volatility
            time_to_maturity: Time to maturity
            risk_free_rate: Risk-free rate

        Returns:
            DataFrame with calls and puts at each strike
        """
        chain_data = []

        for strike in strikes:
            # Call option
            call_result = self.greeks(
                futures_price, strike, volatility, time_to_maturity,
                risk_free_rate, 'call'
            )

            # Put option
            put_result = self.greeks(
                futures_price, strike, volatility, time_to_maturity,
                risk_free_rate, 'put'
            )

            # Moneyness
            moneyness = futures_price / strike

            chain_data.append({
                'strike': strike,
                'moneyness': moneyness,
                'call_price': call_result.price,
                'call_delta': call_result.delta,
                'call_gamma': call_result.gamma,
                'call_theta': call_result.theta,
                'call_vega': call_result.vega,
                'put_price': put_result.price,
                'put_delta': put_result.delta,
                'put_gamma': put_result.gamma,
                'put_theta': put_result.theta,
                'put_vega': put_result.vega,
            })

        chain_df = pd.DataFrame(chain_data)

        logger.info(
            f"Generated option chain for {len(strikes)} strikes "
            f"(F={futures_price:.2f}, σ={volatility:.4f}, T={time_to_maturity:.2f})"
        )

        return chain_df


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('../..')

    from src.config.loader import load_config
    from src.config.logging_config import setup_logging

    # Load configuration
    config = load_config('../../config/parameters.toml')
    logger = setup_logging(config)

    # Initialize pricer
    pricer = Black76Pricer(config)

    print("\n" + "="*80)
    print("BLACK-76 OPTIONS PRICING EXAMPLES")
    print("="*80)

    # Parameters
    futures_price = 100.0
    strike = 100.0
    volatility = 0.25
    time_to_maturity = 0.25  # 3 months
    risk_free_rate = 0.05

    # Example 1: Price and Greeks for ATM call
    print("\nATM Call Option:")
    print(f"  Futures: ${futures_price:.2f}")
    print(f"  Strike: ${strike:.2f}")
    print(f"  Volatility: {volatility*100:.1f}%")
    print(f"  Time to Maturity: {time_to_maturity:.2f}y")
    print(f"  Risk-Free Rate: {risk_free_rate*100:.2f}%")

    call_result = pricer.greeks(
        futures_price, strike, volatility, time_to_maturity,
        risk_free_rate, 'call'
    )

    print(f"\nCall Price: ${call_result.price:.4f}")
    print(f"  Delta: {call_result.delta:.4f}")
    print(f"  Gamma: {call_result.gamma:.6f}")
    print(f"  Theta: ${call_result.theta:.4f} per day")
    print(f"  Vega: ${call_result.vega:.4f} per 1% vol")
    print(f"  Rho: ${call_result.rho:.4f} per 1% rate")

    # Example 2: Put option
    print("\nATM Put Option:")
    put_result = pricer.greeks(
        futures_price, strike, volatility, time_to_maturity,
        risk_free_rate, 'put'
    )

    print(f"Put Price: ${put_result.price:.4f}")
    print(f"  Delta: {put_result.delta:.4f}")
    print(f"  Gamma: {put_result.gamma:.6f}")
    print(f"  Theta: ${put_result.theta:.4f} per day")

    # Example 3: Put-call parity
    print("\nPut-Call Parity Check:")
    parity_holds, deviation = pricer.put_call_parity_check(
        call_result.price, put_result.price, futures_price,
        strike, time_to_maturity, risk_free_rate
    )
    print(f"  Parity Holds: {parity_holds}")
    print(f"  Deviation: ${deviation:.6f}")

    # Example 4: Implied volatility
    print("\nImplied Volatility Calculation:")
    market_price = call_result.price  # Use theoretical price as market price
    implied_vol = pricer.implied_volatility(
        market_price, futures_price, strike, time_to_maturity,
        risk_free_rate, 'call'
    )
    print(f"  Market Price: ${market_price:.4f}")
    print(f"  Implied Vol: {implied_vol*100:.2f}%")
    print(f"  Input Vol: {volatility*100:.2f}%")
    print(f"  Difference: {abs(implied_vol - volatility)*100:.6f}%")

    # Example 5: Option chain
    print("\n" + "-"*80)
    print("Option Chain:")
    print("-"*80)

    strikes = np.array([90, 95, 100, 105, 110])
    chain = pricer.option_chain(
        futures_price, strikes, volatility, time_to_maturity, risk_free_rate
    )

    print("\nCalls:")
    print(chain[['strike', 'call_price', 'call_delta', 'call_gamma', 'call_theta']].to_string(index=False))

    print("\nPuts:")
    print(chain[['strike', 'put_price', 'put_delta', 'put_gamma', 'put_theta']].to_string(index=False))
