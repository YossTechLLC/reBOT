"""
Integration Tests for Options Pricing Pipeline
End-to-end tests from data acquisition to Black-Scholes pricing.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.loader import load_config
from src.data.acquisition import CommodityDataAcquisition
from src.data.preprocessing import DataPreprocessor
from src.data.features import FeatureEngineer
from src.models.hmm_core import CommodityHMM
from src.models.black_scholes import Black76Pricer
from src.models.volatility import VolatilityEstimator
from src.models.interest_rates import InterestRateCurve
from src.models.convenience_yield import ConvenienceYieldEstimator
from src.models.volatility_surface import VolatilitySurface


@pytest.fixture
def config():
    """Load test configuration."""
    config_path = Path(__file__).parent.parent.parent / 'config' / 'parameters.toml'
    return load_config(str(config_path))


@pytest.fixture
def sample_data():
    """Generate sample commodity data."""
    np.random.seed(42)

    dates = pd.date_range('2023-01-01', periods=300, freq='D')
    close = 100 * np.exp(np.cumsum(np.random.randn(300) * 0.02))

    data = pd.DataFrame({
        'date': dates,
        'open': close * (1 + np.random.randn(300) * 0.01),
        'high': close * (1 + np.abs(np.random.randn(300)) * 0.015),
        'low': close * (1 - np.abs(np.random.randn(300)) * 0.015),
        'close': close,
        'volume': np.random.randint(1000000, 5000000, 300)
    })

    return data


class TestFullOptionsPipeline:
    """Integration tests for full options pricing pipeline."""

    def test_data_to_volatility_pipeline(self, config, sample_data):
        """Test pipeline from data to volatility estimation."""
        # Preprocess data
        preprocessor = DataPreprocessor(config)
        processed_data, _ = preprocessor.preprocess(sample_data)

        assert 'returns' in processed_data.columns
        assert len(processed_data) > 0

        # Estimate volatility
        vol_estimator = VolatilityEstimator(config)

        # Historical volatility
        hist_vol = vol_estimator.historical_volatility(processed_data['returns'])
        assert hist_vol > 0
        assert 0.05 < hist_vol < 1.0

        # Hybrid volatility
        hybrid_vol, individual_vols = vol_estimator.hybrid_volatility(processed_data)
        assert hybrid_vol > 0
        assert len(individual_vols) > 0

    def test_data_to_options_price_pipeline(self, config, sample_data):
        """Test full pipeline from data to option pricing."""
        # 1. Preprocess data
        preprocessor = DataPreprocessor(config)
        processed_data, _ = preprocessor.preprocess(sample_data)

        # 2. Estimate volatility
        vol_estimator = VolatilityEstimator(config)
        volatility = vol_estimator.historical_volatility(processed_data['returns'])

        # 3. Get risk-free rate
        rate_curve = InterestRateCurve(config)
        risk_free_rate = rate_curve.get_rate(0.25)  # 3-month rate

        # 4. Price options
        pricer = Black76Pricer(config)

        current_price = sample_data['close'].iloc[-1]
        time_to_maturity = 0.25  # 3 months

        call_result = pricer.greeks(
            futures_price=current_price,
            strike=current_price,  # ATM
            volatility=volatility,
            time_to_maturity=time_to_maturity,
            risk_free_rate=risk_free_rate,
            option_type='call'
        )

        # Validate option price
        assert call_result.price > 0
        assert 0 < call_result.delta < 1
        assert call_result.gamma > 0
        assert call_result.vega > 0

    def test_hmm_regime_to_volatility_pipeline(self, config, sample_data):
        """Test HMM regime detection integrated with volatility estimation."""
        # 1. Preprocess
        preprocessor = DataPreprocessor(config)
        processed_data, _ = preprocessor.preprocess(sample_data)

        # 2. Feature engineering
        feature_engineer = FeatureEngineer(config)
        features = feature_engineer.engineer_features(processed_data)

        # 3. Train HMM
        feature_cols = ['returns', 'volatility_5']
        hmm_features = features[feature_cols].dropna()

        hmm = CommodityHMM(config)
        hmm.fit(hmm_features)

        # 4. Predict regimes
        regimes = []
        for idx in range(len(hmm_features)):
            state, label = hmm.predict_regime(hmm_features.iloc[idx:idx+1])
            regimes.append(label)

        regime_series = pd.Series(regimes, index=hmm_features.index)

        # 5. Estimate regime-conditional volatility
        vol_estimator = VolatilityEstimator(config)

        # Align data with regime labels
        aligned_data = features.loc[regime_series.index]

        regime_vols = vol_estimator.regime_conditional_volatility(
            aligned_data,
            regime_series,
            method='historical'
        )

        # Validate
        assert len(regime_vols) > 0

        # Different regimes should have different volatilities
        if len(regime_vols) >= 2:
            vols = list(regime_vols.values())
            assert not all(v == vols[0] for v in vols)

    def test_regime_conditional_option_pricing(self, config, sample_data):
        """Test option pricing with regime-conditional volatility."""
        # 1. Setup pipeline
        preprocessor = DataPreprocessor(config)
        processed_data, _ = preprocessor.preprocess(sample_data)

        feature_engineer = FeatureEngineer(config)
        features = feature_engineer.engineer_features(processed_data)

        # 2. Train HMM
        feature_cols = ['returns', 'volatility_5']
        hmm_features = features[feature_cols].dropna()

        hmm = CommodityHMM(config)
        hmm.fit(hmm_features)

        # 3. Get current regime
        current_state, current_label = hmm.predict_regime(hmm_features.iloc[-1:])

        # 4. Get regime statistics
        regime_stats = hmm.get_regime_stats()
        current_volatility = regime_stats[current_state]['volatility']

        # 5. Price option with regime-specific volatility
        pricer = Black76Pricer(config)
        rate_curve = InterestRateCurve(config)

        current_price = sample_data['close'].iloc[-1]

        option_result = pricer.greeks(
            futures_price=current_price,
            strike=current_price,
            volatility=current_volatility,
            time_to_maturity=0.25,
            risk_free_rate=rate_curve.get_rate(0.25),
            option_type='call'
        )

        # Validate
        assert option_result.price > 0
        assert option_result.delta > 0

        print(f"\nRegime-Conditional Option Pricing:")
        print(f"  Current Regime: {current_label}")
        print(f"  Regime Volatility: {current_volatility:.4f}")
        print(f"  Call Price: ${option_result.price:.4f}")
        print(f"  Delta: {option_result.delta:.4f}")

    def test_full_option_chain_generation(self, config, sample_data):
        """Test generating complete option chain."""
        # 1. Preprocess
        preprocessor = DataPreprocessor(config)
        processed_data, _ = preprocessor.preprocess(sample_data)

        # 2. Estimate volatility
        vol_estimator = VolatilityEstimator(config)
        volatility = vol_estimator.historical_volatility(processed_data['returns'])

        # 3. Setup pricing
        pricer = Black76Pricer(config)
        rate_curve = InterestRateCurve(config)

        current_price = sample_data['close'].iloc[-1]
        time_to_maturity = 0.25
        risk_free_rate = rate_curve.get_rate(time_to_maturity)

        # 4. Generate option chain
        # Create strikes around current price
        strikes = np.array([
            current_price * 0.90,
            current_price * 0.95,
            current_price * 1.00,
            current_price * 1.05,
            current_price * 1.10
        ])

        option_chain = pricer.option_chain(
            futures_price=current_price,
            strikes=strikes,
            volatility=volatility,
            time_to_maturity=time_to_maturity,
            risk_free_rate=risk_free_rate
        )

        # Validate chain
        assert len(option_chain) == len(strikes)
        assert 'call_price' in option_chain.columns
        assert 'put_price' in option_chain.columns
        assert 'call_delta' in option_chain.columns
        assert 'put_delta' in option_chain.columns

        # Check ordering: ITM calls more expensive than OTM calls
        # (assuming strikes are ordered)
        assert option_chain['call_price'].iloc[0] > option_chain['call_price'].iloc[-1]

        # Check put-call relationship: ITM puts more expensive than OTM puts
        assert option_chain['put_price'].iloc[-1] > option_chain['put_price'].iloc[0]

    def test_volatility_surface_construction(self, config, sample_data):
        """Test volatility surface construction."""
        # 1. Setup
        preprocessor = DataPreprocessor(config)
        processed_data, _ = preprocessor.preprocess(sample_data)

        vol_estimator = VolatilityEstimator(config)
        base_volatility = vol_estimator.historical_volatility(processed_data['returns'])

        # 2. Construct surface
        pricer = Black76Pricer(config)
        vol_surface = VolatilitySurface(config)
        rate_curve = InterestRateCurve(config)

        current_price = sample_data['close'].iloc[-1]

        K, T, IV = vol_surface.construct_surface(
            pricer=pricer,
            futures_price=current_price,
            base_volatility=base_volatility,
            risk_free_rate=rate_curve.get_rate(0.25),
            smile_amplitude=0.05,
            skew_slope=0.03
        )

        # Validate surface
        assert K.shape == T.shape == IV.shape
        assert K.size > 0
        assert np.all(IV > 0)  # All volatilities positive
        assert np.all(IV < 2.0)  # All volatilities reasonable

        # Check that surface has expected structure
        # ATM volatility should be close to base
        mid_idx = K.shape[1] // 2
        atm_vols = IV[:, mid_idx]
        assert np.all(np.abs(atm_vols - base_volatility) < 0.15)

    def test_convenience_yield_integration(self, config, sample_data):
        """Test convenience yield estimation integrated with futures pricing."""
        # 1. Setup
        cy_estimator = ConvenienceYieldEstimator(config)
        rate_curve = InterestRateCurve(config)

        current_price = sample_data['close'].iloc[-1]

        # 2. Simulate futures curve
        maturities = np.array([3/12, 6/12, 9/12, 1, 2])
        risk_free_rates = np.array([rate_curve.get_rate(m) for m in maturities])

        # Assume small convenience yield of 2%
        assumed_cy = 0.02
        futures_prices = current_price * np.exp(
            (risk_free_rates - assumed_cy) * maturities
        )

        futures_curve = pd.DataFrame({
            'maturity': maturities,
            'price': futures_prices
        })

        # 3. Estimate convenience yield
        estimated_cy, fitted_prices = cy_estimator.from_futures_curve(
            spot_price=current_price,
            futures_curve=futures_curve,
            rate_curve=rate_curve
        )

        # 4. Validate
        # Should recover approximately the assumed convenience yield
        assert abs(estimated_cy - assumed_cy) < 0.01

        # Fitted prices should be close to observed
        assert len(fitted_prices) == len(futures_curve)
        rmse = np.sqrt(np.mean(fitted_prices['residual'] ** 2))
        assert rmse < 0.50  # Good fit


class TestRobustness:
    """Test robustness and edge cases."""

    def test_zero_volatility_handling(self, config):
        """Test handling of zero volatility."""
        pricer = Black76Pricer(config)

        # Zero volatility should still price (intrinsic value only)
        price = pricer.price(
            futures_price=105.0,
            strike=100.0,
            volatility=0.0,
            time_to_maturity=0.25,
            risk_free_rate=0.05,
            option_type='call'
        )

        # Should be approximately intrinsic value (discounted)
        discount = np.exp(-0.05 * 0.25)
        intrinsic = (105.0 - 100.0) * discount
        assert abs(price - intrinsic) < 0.10

    def test_very_long_maturity(self, config):
        """Test option pricing with very long maturity."""
        pricer = Black76Pricer(config)

        price = pricer.price(
            futures_price=100.0,
            strike=100.0,
            volatility=0.25,
            time_to_maturity=10.0,  # 10 years
            risk_free_rate=0.05,
            option_type='call'
        )

        # Should have substantial time value
        assert price > 10.0

        # But should still be reasonable
        assert price < 100.0

    def test_extreme_strikes(self, config):
        """Test pricing with extreme strikes."""
        pricer = Black76Pricer(config)

        # Very deep ITM call
        deep_itm = pricer.price(
            futures_price=200.0,
            strike=100.0,
            volatility=0.25,
            time_to_maturity=0.25,
            risk_free_rate=0.05,
            option_type='call'
        )

        # Should be close to (F-K)*discount
        discount = np.exp(-0.05 * 0.25)
        expected = (200.0 - 100.0) * discount
        assert abs(deep_itm - expected) < 5.0

        # Very deep OTM call
        deep_otm = pricer.price(
            futures_price=50.0,
            strike=100.0,
            volatility=0.25,
            time_to_maturity=0.25,
            risk_free_rate=0.05,
            option_type='call'
        )

        # Should be very small
        assert deep_otm < 0.50


class TestPerformance:
    """Test computational performance."""

    def test_large_option_chain_performance(self, config):
        """Test performance of generating large option chain."""
        import time

        pricer = Black76Pricer(config)

        # Generate large chain
        strikes = np.linspace(80, 120, 50)  # 50 strikes

        start = time.time()

        option_chain = pricer.option_chain(
            futures_price=100.0,
            strikes=strikes,
            volatility=0.25,
            time_to_maturity=0.25,
            risk_free_rate=0.05
        )

        elapsed = time.time() - start

        # Should complete quickly (< 1 second for 50 strikes)
        assert elapsed < 1.0
        assert len(option_chain) == 50

    def test_volatility_surface_performance(self, config):
        """Test performance of volatility surface construction."""
        import time

        pricer = Black76Pricer(config)
        vol_surface = VolatilitySurface(config)
        rate_curve = InterestRateCurve(config)

        start = time.time()

        K, T, IV = vol_surface.construct_surface(
            pricer=pricer,
            futures_price=100.0,
            base_volatility=0.25,
            risk_free_rate=rate_curve.get_rate(0.25)
        )

        elapsed = time.time() - start

        # Should complete quickly (< 2 seconds for default grid)
        assert elapsed < 2.0
        assert K.size > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
