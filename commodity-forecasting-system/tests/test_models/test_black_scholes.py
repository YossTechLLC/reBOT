"""
Unit Tests for Black-Scholes Options Pricing Module
Tests for Black-76 pricer, volatility estimation, and related components.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.black_scholes import Black76Pricer, OptionResult
from src.models.volatility import VolatilityEstimator
from src.models.interest_rates import InterestRateCurve
from src.models.convenience_yield import ConvenienceYieldEstimator
from src.config.loader import load_config


@pytest.fixture
def config():
    """Load test configuration."""
    config_path = Path(__file__).parent.parent.parent / 'config' / 'parameters.toml'
    return load_config(str(config_path))


@pytest.fixture
def pricer(config):
    """Initialize Black76Pricer."""
    return Black76Pricer(config)


@pytest.fixture
def standard_params():
    """Standard option parameters for testing."""
    return {
        'futures_price': 100.0,
        'strike': 100.0,
        'volatility': 0.25,
        'time_to_maturity': 0.25,  # 3 months
        'risk_free_rate': 0.05
    }


class TestBlack76Pricing:
    """Test suite for Black-76 options pricing."""

    def test_atm_call_price(self, pricer, standard_params):
        """Test ATM call option pricing."""
        price = pricer.price(**standard_params, option_type='call')

        # ATM call should have positive price
        assert price > 0

        # Rough sanity check: ATM call with 25% vol, 3m maturity
        # should be around 5% of futures price
        assert 0.03 * standard_params['futures_price'] < price < 0.08 * standard_params['futures_price']

    def test_atm_put_price(self, pricer, standard_params):
        """Test ATM put option pricing."""
        price = pricer.price(**standard_params, option_type='put')

        # ATM put should have positive price
        assert price > 0

        # Should be similar to call price for ATM options
        call_price = pricer.price(**standard_params, option_type='call')
        assert abs(price - call_price) < 0.50  # Within 50 cents

    def test_put_call_parity(self, pricer, standard_params):
        """Test put-call parity for futures options."""
        call_price = pricer.price(**standard_params, option_type='call')
        put_price = pricer.price(**standard_params, option_type='put')

        # Check parity
        parity_holds, deviation = pricer.put_call_parity_check(
            call_price, put_price,
            standard_params['futures_price'],
            standard_params['strike'],
            standard_params['time_to_maturity'],
            standard_params['risk_free_rate']
        )

        assert parity_holds
        assert deviation < 0.01

    def test_itm_otm_prices(self, pricer, standard_params):
        """Test ITM and OTM option prices."""
        # Deep ITM call
        itm_call = pricer.price(
            futures_price=110.0,
            strike=100.0,
            volatility=standard_params['volatility'],
            time_to_maturity=standard_params['time_to_maturity'],
            risk_free_rate=standard_params['risk_free_rate'],
            option_type='call'
        )

        # Deep OTM call
        otm_call = pricer.price(
            futures_price=90.0,
            strike=100.0,
            volatility=standard_params['volatility'],
            time_to_maturity=standard_params['time_to_maturity'],
            risk_free_rate=standard_params['risk_free_rate'],
            option_type='call'
        )

        # ITM should be more expensive than OTM
        assert itm_call > otm_call

        # ITM call should be at least intrinsic value
        intrinsic_value = max(110.0 - 100.0, 0)
        assert itm_call >= intrinsic_value

    def test_zero_time_to_expiry(self, pricer, standard_params):
        """Test option price at expiration."""
        # ITM call at expiration
        price = pricer.price(
            futures_price=105.0,
            strike=100.0,
            volatility=standard_params['volatility'],
            time_to_maturity=0.0,
            risk_free_rate=standard_params['risk_free_rate'],
            option_type='call'
        )

        # Should equal intrinsic value
        intrinsic_value = max(105.0 - 100.0, 0)
        assert abs(price - intrinsic_value) < 1e-6

        # OTM call at expiration
        price_otm = pricer.price(
            futures_price=95.0,
            strike=100.0,
            volatility=standard_params['volatility'],
            time_to_maturity=0.0,
            risk_free_rate=standard_params['risk_free_rate'],
            option_type='call'
        )

        assert abs(price_otm) < 1e-6  # Should be zero

    def test_volatility_impact(self, pricer, standard_params):
        """Test that higher volatility increases option price."""
        low_vol_price = pricer.price(
            **{**standard_params, 'volatility': 0.15},
            option_type='call'
        )

        high_vol_price = pricer.price(
            **{**standard_params, 'volatility': 0.35},
            option_type='call'
        )

        # Higher volatility should increase price
        assert high_vol_price > low_vol_price

    def test_time_decay(self, pricer, standard_params):
        """Test that longer time increases option price."""
        short_time_price = pricer.price(
            **{**standard_params, 'time_to_maturity': 0.08},
            option_type='call'
        )

        long_time_price = pricer.price(
            **{**standard_params, 'time_to_maturity': 0.50},
            option_type='call'
        )

        # More time should increase price (for ATM options)
        assert long_time_price > short_time_price


class TestGreeks:
    """Test suite for option Greeks."""

    def test_call_delta_range(self, pricer, standard_params):
        """Test that call delta is in (0, 1) range."""
        result = pricer.greeks(**standard_params, option_type='call')

        # Call delta should be between 0 and 1
        assert 0 < result.delta < 1

        # For ATM call, delta should be around 0.5
        assert 0.4 < result.delta < 0.6

    def test_put_delta_range(self, pricer, standard_params):
        """Test that put delta is in (-1, 0) range."""
        result = pricer.greeks(**standard_params, option_type='put')

        # Put delta should be between -1 and 0
        assert -1 < result.delta < 0

        # For ATM put, delta should be around -0.5
        assert -0.6 < result.delta < -0.4

    def test_gamma_positive(self, pricer, standard_params):
        """Test that gamma is positive."""
        call_result = pricer.greeks(**standard_params, option_type='call')
        put_result = pricer.greeks(**standard_params, option_type='put')

        # Gamma should be positive for both calls and puts
        assert call_result.gamma > 0
        assert put_result.gamma > 0

        # Gamma should be the same for calls and puts
        assert abs(call_result.gamma - put_result.gamma) < 1e-6

    def test_vega_positive(self, pricer, standard_params):
        """Test that vega is positive."""
        call_result = pricer.greeks(**standard_params, option_type='call')
        put_result = pricer.greeks(**standard_params, option_type='put')

        # Vega should be positive for both calls and puts
        assert call_result.vega > 0
        assert put_result.vega > 0

        # Vega should be the same for calls and puts
        assert abs(call_result.vega - put_result.vega) < 1e-6

    def test_theta_negative(self, pricer, standard_params):
        """Test that theta is negative (time decay)."""
        call_result = pricer.greeks(**standard_params, option_type='call')
        put_result = pricer.greeks(**standard_params, option_type='put')

        # Theta should be negative for both (time decay)
        assert call_result.theta < 0
        assert put_result.theta < 0

    def test_delta_moneyness_relationship(self, pricer, standard_params):
        """Test delta changes with moneyness."""
        # Deep ITM call
        itm_result = pricer.greeks(
            futures_price=120.0,
            strike=100.0,
            volatility=standard_params['volatility'],
            time_to_maturity=standard_params['time_to_maturity'],
            risk_free_rate=standard_params['risk_free_rate'],
            option_type='call'
        )

        # Deep OTM call
        otm_result = pricer.greeks(
            futures_price=80.0,
            strike=100.0,
            volatility=standard_params['volatility'],
            time_to_maturity=standard_params['time_to_maturity'],
            risk_free_rate=standard_params['risk_free_rate'],
            option_type='call'
        )

        # ITM delta should be higher than OTM delta
        assert itm_result.delta > otm_result.delta

        # ITM delta should be close to 1
        assert itm_result.delta > 0.9

        # OTM delta should be close to 0
        assert otm_result.delta < 0.1


class TestImpliedVolatility:
    """Test suite for implied volatility calculation."""

    def test_iv_recovery(self, pricer, standard_params):
        """Test that IV calculation recovers input volatility."""
        # Calculate theoretical price
        theoretical_price = pricer.price(**standard_params, option_type='call')

        # Recover implied volatility
        implied_vol = pricer.implied_volatility(
            market_price=theoretical_price,
            futures_price=standard_params['futures_price'],
            strike=standard_params['strike'],
            time_to_maturity=standard_params['time_to_maturity'],
            risk_free_rate=standard_params['risk_free_rate'],
            option_type='call'
        )

        # Should recover original volatility
        assert abs(implied_vol - standard_params['volatility']) < 1e-4

    def test_iv_different_strikes(self, pricer, standard_params):
        """Test IV calculation for different strikes."""
        for strike in [90, 95, 100, 105, 110]:
            price = pricer.price(
                futures_price=standard_params['futures_price'],
                strike=strike,
                volatility=standard_params['volatility'],
                time_to_maturity=standard_params['time_to_maturity'],
                risk_free_rate=standard_params['risk_free_rate'],
                option_type='call'
            )

            implied_vol = pricer.implied_volatility(
                market_price=price,
                futures_price=standard_params['futures_price'],
                strike=strike,
                time_to_maturity=standard_params['time_to_maturity'],
                risk_free_rate=standard_params['risk_free_rate'],
                option_type='call'
            )

            # Should recover original volatility
            assert abs(implied_vol - standard_params['volatility']) < 1e-3

    def test_iv_negative_price_error(self, pricer, standard_params):
        """Test that negative market price raises error."""
        with pytest.raises(ValueError, match="Market price must be positive"):
            pricer.implied_volatility(
                market_price=-1.0,
                futures_price=standard_params['futures_price'],
                strike=standard_params['strike'],
                time_to_maturity=standard_params['time_to_maturity'],
                risk_free_rate=standard_params['risk_free_rate'],
                option_type='call'
            )


class TestVolatilityEstimator:
    """Test suite for volatility estimation."""

    @pytest.fixture
    def vol_estimator(self, config):
        """Initialize VolatilityEstimator."""
        return VolatilityEstimator(config)

    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns data."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.02)  # Daily returns
        return returns

    @pytest.fixture
    def sample_ohlc(self):
        """Generate sample OHLC data."""
        np.random.seed(42)
        n = 252
        close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02))

        data = pd.DataFrame({
            'returns': np.diff(np.log(close), prepend=np.log(close[0])),
            'high': close * (1 + np.abs(np.random.randn(n)) * 0.01),
            'low': close * (1 - np.abs(np.random.randn(n)) * 0.01)
        })

        return data

    def test_historical_volatility(self, vol_estimator, sample_returns):
        """Test historical volatility calculation."""
        vol = vol_estimator.historical_volatility(sample_returns, window=30)

        # Should be positive
        assert vol > 0

        # Should be reasonable magnitude (annualized)
        assert 0.05 < vol < 1.0

    def test_realized_volatility_parkinson(self, vol_estimator, sample_ohlc):
        """Test Parkinson realized volatility estimator."""
        vol = vol_estimator.realized_volatility_parkinson(
            sample_ohlc['high'],
            sample_ohlc['low'],
            window=30
        )

        # Should be positive
        assert vol > 0

        # Should be reasonable
        assert 0.05 < vol < 1.0

    def test_hybrid_volatility(self, vol_estimator, sample_ohlc):
        """Test hybrid volatility calculation."""
        hybrid_vol, individual_vols = vol_estimator.hybrid_volatility(sample_ohlc)

        # Should be positive
        assert hybrid_vol > 0

        # Individual volatilities should all be calculated
        assert 'historical' in individual_vols
        assert individual_vols['historical'] > 0


class TestInterestRateCurve:
    """Test suite for interest rate curve."""

    @pytest.fixture
    def rate_curve(self, config):
        """Initialize InterestRateCurve."""
        return InterestRateCurve(config)

    def test_default_curve(self, rate_curve):
        """Test default flat yield curve."""
        curve = rate_curve._default_curve()

        # Should have standard maturities
        assert len(curve) > 0

        # All rates should be equal (flat curve)
        assert all(curve == curve.iloc[0])

    def test_get_rate_interpolation(self, rate_curve):
        """Test rate interpolation."""
        # This will use default curve if no API key
        rate_1y = rate_curve.get_rate(1.0)
        rate_2y = rate_curve.get_rate(2.0)

        # Rates should be positive
        assert rate_1y > 0
        assert rate_2y > 0

        # Should be able to interpolate between
        rate_1_5y = rate_curve.get_rate(1.5)
        assert rate_1_5y > 0

    def test_forward_rate_calculation(self, rate_curve):
        """Test forward rate calculation."""
        forward = rate_curve.forward_rate(1.0, 2.0)

        # Forward rate should be positive
        assert forward > 0

        # Should be reasonable magnitude
        assert -0.05 < forward < 0.30


class TestConvenienceYield:
    """Test suite for convenience yield estimation."""

    @pytest.fixture
    def cy_estimator(self, config):
        """Initialize ConvenienceYieldEstimator."""
        return ConvenienceYieldEstimator(config)

    def test_single_futures_yield(self, cy_estimator):
        """Test convenience yield from single futures contract."""
        # Contango case: futures > spot
        conv_yield = cy_estimator.from_single_futures(
            spot_price=100.0,
            futures_price=102.0,
            time_to_maturity=0.25,
            risk_free_rate=0.05
        )

        # Should be negative or low (contango)
        assert conv_yield < 0.10

    def test_backwardation_contango_indicator(self, cy_estimator):
        """Test backwardation/contango detection."""
        # Contango
        state, basis = cy_estimator.backwardation_contango_indicator(
            spot_price=100.0,
            futures_price=102.0
        )
        assert state == "contango"
        assert basis > 0

        # Backwardation
        state, basis = cy_estimator.backwardation_contango_indicator(
            spot_price=100.0,
            futures_price=98.0
        )
        assert state == "backwardation"
        assert basis < 0


class TestInputValidation:
    """Test input validation and error handling."""

    def test_negative_futures_price_error(self, pricer, standard_params):
        """Test that negative futures price raises error."""
        with pytest.raises(ValueError, match="Futures price must be positive"):
            pricer.price(**{**standard_params, 'futures_price': -100}, option_type='call')

    def test_negative_strike_error(self, pricer, standard_params):
        """Test that negative strike raises error."""
        with pytest.raises(ValueError, match="Strike must be positive"):
            pricer.price(**{**standard_params, 'strike': -100}, option_type='call')

    def test_negative_volatility_error(self, pricer, standard_params):
        """Test that negative volatility raises error."""
        with pytest.raises(ValueError, match="Volatility must be non-negative"):
            pricer.price(**{**standard_params, 'volatility': -0.25}, option_type='call')

    def test_negative_time_error(self, pricer, standard_params):
        """Test that negative time to maturity raises error."""
        with pytest.raises(ValueError, match="Time to maturity must be non-negative"):
            pricer.price(**{**standard_params, 'time_to_maturity': -0.25}, option_type='call')

    def test_invalid_option_type_error(self, pricer, standard_params):
        """Test that invalid option type raises error."""
        with pytest.raises(ValueError, match="Invalid option_type"):
            pricer.price(**standard_params, option_type='futures')


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
