"""
Volatility Estimation Module
Multiple approaches for estimating and forecasting volatility.

Methodologies:
1. Historical Volatility: Simple rolling standard deviation
2. GARCH(1,1): Generalized AutoRegressive Conditional Heteroskedasticity
3. Realized Volatility: Parkinson estimator using high-low range
4. Hybrid Volatility: Weighted combination of multiple estimators
"""

import logging
from typing import Dict, Tuple, Union, Optional
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

logger = logging.getLogger(__name__)


class VolatilityEstimator:
    """
    Multi-method volatility estimation for options pricing.

    Volatility is the most critical input to options pricing models.
    Different estimation methods capture different aspects:
    - Historical: backward-looking, assumes stationarity
    - GARCH: time-varying, mean-reverting, forecasts future volatility
    - Realized: uses intraday range, more efficient than close-to-close
    - Hybrid: combines multiple methods to reduce estimation error
    """

    def __init__(self, config: Dict):
        """
        Initialize volatility estimator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.vol_config = config.get('volatility', {})

        # Default parameters
        self.annualization_factor = self.vol_config.get('annualization_factor', 252)
        self.historical_window = self.vol_config.get('historical_window', 30)
        self.garch_p = self.vol_config.get('garch_p', 1)
        self.garch_q = self.vol_config.get('garch_q', 1)

        # Hybrid weights
        self.hybrid_weights = self.vol_config.get('hybrid_weights', {
            'historical': 0.4,
            'garch': 0.4,
            'realized': 0.2
        })

        logger.info(f"Initialized VolatilityEstimator with annualization={self.annualization_factor}")

    def historical_volatility(
        self,
        returns: Union[pd.Series, np.ndarray],
        window: Optional[int] = None,
        annualize: bool = True
    ) -> float:
        """
        Calculate historical volatility using standard deviation.

        Formula: σ_hist = std(returns) * sqrt(annualization_factor)

        Args:
            returns: Return series (should be log returns)
            window: Rolling window size (default uses config)
            annualize: Whether to annualize volatility

        Returns:
            Historical volatility (annualized if requested)
        """
        if window is None:
            window = self.historical_window

        # Convert to numpy if needed
        if isinstance(returns, pd.Series):
            returns = returns.values

        # Handle insufficient data
        if len(returns) < window:
            logger.warning(
                f"TFM4001 INFERENCE: Insufficient data for historical volatility "
                f"(have {len(returns)}, need {window}). Using all available data."
            )
            window = len(returns)

        # Calculate rolling volatility
        returns_recent = returns[-window:]
        vol = np.std(returns_recent, ddof=1)

        # Annualize if requested
        if annualize:
            vol *= np.sqrt(self.annualization_factor)

        logger.debug(f"Historical volatility ({window} days): {vol:.4f}")
        return vol

    def garch_volatility(
        self,
        returns: Union[pd.Series, np.ndarray],
        horizon: int = 1,
        rescale: bool = False
    ) -> Tuple[float, Dict]:
        """
        Forecast volatility using GARCH(1,1) model.

        GARCH(1,1) specification:
        σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

        where:
        - ω: long-run variance level
        - α: ARCH coefficient (reaction to shocks)
        - β: GARCH coefficient (persistence)
        - Constraint: α + β < 1 for stationarity

        Args:
            returns: Return series
            horizon: Forecast horizon in days
            rescale: Whether to rescale returns to percentage (for arch library)

        Returns:
            Tuple of (forecasted_volatility, model_diagnostics)
        """
        try:
            from arch import arch_model
        except ImportError:
            logger.error(
                "TFM1001 CONFIG: arch library not installed. "
                "Install with: pip install arch"
            )
            raise

        # Convert to pandas Series if needed
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)

        # Remove NaN
        returns = returns.dropna()

        if len(returns) < 50:
            logger.warning(
                f"TFM4001 INFERENCE: GARCH requires >=50 observations, "
                f"have {len(returns)}. Results may be unreliable."
            )

        # Rescale to percentage returns if requested (arch library preference)
        if rescale:
            returns = returns * 100

        # Fit GARCH(1,1) model
        try:
            model = arch_model(
                returns,
                vol='Garch',
                p=self.garch_p,
                q=self.garch_q,
                rescale=not rescale
            )

            # Fit with convergence warnings suppressed
            result = model.fit(disp='off', show_warning=False)

            # Forecast volatility
            forecast = result.forecast(horizon=horizon)

            # Extract forecasted variance and convert to volatility
            forecasted_variance = forecast.variance.values[-1, -1]

            if rescale:
                # Convert back from percentage
                forecasted_vol = np.sqrt(forecasted_variance) / 100
            else:
                forecasted_vol = np.sqrt(forecasted_variance)

            # Annualize
            forecasted_vol_annual = forecasted_vol * np.sqrt(self.annualization_factor)

            # Diagnostics
            diagnostics = {
                'omega': result.params['omega'],
                'alpha': result.params['alpha[1]'],
                'beta': result.params['beta[1]'],
                'persistence': result.params['alpha[1]'] + result.params['beta[1]'],
                'log_likelihood': result.loglikelihood,
                'aic': result.aic,
                'bic': result.bic,
                'converged': result.convergence_flag == 0
            }

            logger.debug(
                f"GARCH(1,1) volatility forecast ({horizon}d): {forecasted_vol_annual:.4f}, "
                f"persistence={diagnostics['persistence']:.3f}"
            )

            return forecasted_vol_annual, diagnostics

        except Exception as e:
            logger.error(
                f"TFM4001 INFERENCE: GARCH fitting failed: {e}. "
                f"Falling back to historical volatility."
            )
            fallback_vol = self.historical_volatility(returns)
            return fallback_vol, {'error': str(e), 'fallback': True}

    def realized_volatility_parkinson(
        self,
        high: Union[pd.Series, np.ndarray],
        low: Union[pd.Series, np.ndarray],
        window: Optional[int] = None,
        annualize: bool = True
    ) -> float:
        """
        Calculate realized volatility using Parkinson estimator.

        The Parkinson estimator uses high-low range and is more efficient
        than close-to-close volatility (uses intraday information).

        Formula: σ_P = sqrt( (1/(4*ln(2))) * mean((ln(H/L))²) )

        Efficiency: ~5x more efficient than close-to-close estimator

        Args:
            high: High prices
            low: Low prices
            window: Rolling window size
            annualize: Whether to annualize volatility

        Returns:
            Realized volatility (Parkinson estimator)
        """
        if window is None:
            window = self.historical_window

        # Convert to numpy if needed
        if isinstance(high, pd.Series):
            high = high.values
        if isinstance(low, pd.Series):
            low = low.values

        # Validate inputs
        if len(high) != len(low):
            raise ValueError(
                f"TFM2001 DATA: High and low must have same length "
                f"(high={len(high)}, low={len(low)})"
            )

        if np.any(high < low):
            raise ValueError(
                "TFM2001 DATA: High prices must be >= low prices"
            )

        # Handle insufficient data
        if len(high) < window:
            logger.warning(
                f"TFM4001 INFERENCE: Insufficient data for realized volatility "
                f"(have {len(high)}, need {window}). Using all available data."
            )
            window = len(high)

        # Calculate Parkinson estimator
        high_recent = high[-window:]
        low_recent = low[-window:]

        # Avoid log(0) by adding small epsilon
        hl_ratio = np.log(high_recent / (low_recent + 1e-10))

        # Parkinson formula: sqrt(1/(4*ln(2)) * mean(hl_ratio^2))
        variance = np.mean(hl_ratio ** 2) / (4 * np.log(2))
        vol = np.sqrt(variance)

        # Annualize if requested
        if annualize:
            vol *= np.sqrt(self.annualization_factor)

        logger.debug(f"Realized volatility - Parkinson ({window} days): {vol:.4f}")
        return vol

    def hybrid_volatility(
        self,
        data: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate hybrid volatility as weighted combination of methods.

        Combines multiple estimators to reduce estimation error:
        - Historical: captures recent realized volatility
        - GARCH: forecasts future volatility with mean reversion
        - Realized: efficient estimator using high-low range

        Args:
            data: DataFrame with columns: returns, high, low
            weights: Optional custom weights (default from config)

        Returns:
            Tuple of (hybrid_volatility, individual_volatilities)
        """
        if weights is None:
            weights = self.hybrid_weights

        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}

        individual_vols = {}

        # Calculate historical volatility
        if 'historical' in weights and weights['historical'] > 0:
            if 'returns' in data.columns:
                hist_vol = self.historical_volatility(data['returns'])
                individual_vols['historical'] = hist_vol
            else:
                logger.warning(
                    "TFM2001 DATA: 'returns' column missing for historical volatility"
                )

        # Calculate GARCH volatility
        if 'garch' in weights and weights['garch'] > 0:
            if 'returns' in data.columns:
                garch_vol, _ = self.garch_volatility(data['returns'], horizon=1)
                individual_vols['garch'] = garch_vol
            else:
                logger.warning(
                    "TFM2001 DATA: 'returns' column missing for GARCH volatility"
                )

        # Calculate realized volatility
        if 'realized' in weights and weights['realized'] > 0:
            if 'high' in data.columns and 'low' in data.columns:
                realized_vol = self.realized_volatility_parkinson(
                    data['high'],
                    data['low']
                )
                individual_vols['realized'] = realized_vol
            else:
                logger.warning(
                    "TFM2001 DATA: 'high' and 'low' columns missing for realized volatility"
                )

        # Calculate weighted average
        hybrid_vol = 0.0
        actual_weight_sum = 0.0

        for method, vol in individual_vols.items():
            if method in weights:
                hybrid_vol += weights[method] * vol
                actual_weight_sum += weights[method]

        # Renormalize if some methods were missing
        if actual_weight_sum > 0:
            hybrid_vol /= actual_weight_sum
        else:
            logger.error(
                "TFM4001 INFERENCE: No volatility methods succeeded. "
                "Using fallback volatility of 0.20"
            )
            hybrid_vol = 0.20

        logger.info(
            f"Hybrid volatility: {hybrid_vol:.4f} "
            f"(weights: {weights}, individual: {individual_vols})"
        )

        return hybrid_vol, individual_vols

    def regime_conditional_volatility(
        self,
        data: pd.DataFrame,
        regime_labels: pd.Series,
        method: str = 'hybrid'
    ) -> Dict[str, float]:
        """
        Calculate regime-conditional volatility estimates.

        Different market regimes (bull/bear/neutral) exhibit different
        volatility characteristics. This method estimates volatility
        separately for each regime.

        Args:
            data: Price/return data
            regime_labels: HMM regime labels (bull/bear/neutral)
            method: Estimation method ('historical', 'garch', 'realized', 'hybrid')

        Returns:
            Dictionary mapping regime -> volatility
        """
        regime_vols = {}

        unique_regimes = regime_labels.unique()

        for regime in unique_regimes:
            # Filter data for this regime
            regime_mask = regime_labels == regime
            regime_data = data[regime_mask]

            if len(regime_data) < 10:
                logger.warning(
                    f"TFM4001 INFERENCE: Insufficient data for regime '{regime}' "
                    f"({len(regime_data)} observations). Skipping."
                )
                continue

            # Calculate volatility for this regime
            try:
                if method == 'historical':
                    vol = self.historical_volatility(regime_data['returns'])
                elif method == 'garch':
                    vol, _ = self.garch_volatility(regime_data['returns'])
                elif method == 'realized':
                    vol = self.realized_volatility_parkinson(
                        regime_data['high'],
                        regime_data['low']
                    )
                elif method == 'hybrid':
                    vol, _ = self.hybrid_volatility(regime_data)
                else:
                    raise ValueError(
                        f"TFM1001 CONFIG: Unknown volatility method: {method}"
                    )

                regime_vols[regime] = vol

                logger.debug(
                    f"Regime '{regime}' volatility ({method}): {vol:.4f} "
                    f"({len(regime_data)} observations)"
                )

            except Exception as e:
                logger.error(
                    f"TFM4001 INFERENCE: Failed to calculate volatility "
                    f"for regime '{regime}': {e}"
                )
                continue

        return regime_vols

    def volatility_term_structure(
        self,
        returns: Union[pd.Series, np.ndarray],
        maturities: np.ndarray,
        method: str = 'garch'
    ) -> pd.DataFrame:
        """
        Estimate volatility term structure across multiple maturities.

        Different option maturities may have different implied volatilities,
        creating a term structure (volatility smile/skew).

        Args:
            returns: Historical returns
            maturities: Array of maturities in days (e.g., [7, 30, 90, 180, 365])
            method: Forecasting method ('garch' or 'historical')

        Returns:
            DataFrame with columns: maturity, volatility
        """
        term_structure = []

        for maturity in maturities:
            if method == 'garch':
                # Forecast volatility for this horizon
                vol, _ = self.garch_volatility(returns, horizon=maturity)
            elif method == 'historical':
                # Use rolling window scaled by sqrt(time)
                base_vol = self.historical_volatility(returns, window=maturity)
                vol = base_vol
            else:
                raise ValueError(
                    f"TFM1001 CONFIG: Unknown method for term structure: {method}"
                )

            term_structure.append({
                'maturity_days': maturity,
                'maturity_years': maturity / self.annualization_factor,
                'volatility': vol
            })

        term_structure_df = pd.DataFrame(term_structure)

        logger.info(
            f"Volatility term structure estimated for {len(maturities)} maturities"
        )

        return term_structure_df


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('../..')

    from src.config.loader import load_config
    from src.config.logging_config import setup_logging
    from src.data.acquisition import CommodityDataAcquisition
    from src.data.preprocessing import DataPreprocessor

    # Load configuration
    config = load_config('../../config/parameters.toml')
    logger = setup_logging(config)

    # Fetch and preprocess data
    data_client = CommodityDataAcquisition(config)
    data = data_client.fetch_commodity_prices()

    preprocessor = DataPreprocessor(config)
    data, _ = preprocessor.preprocess(data)

    # Initialize volatility estimator
    vol_estimator = VolatilityEstimator(config)

    print("\n" + "="*80)
    print("VOLATILITY ESTIMATION EXAMPLES")
    print("="*80)

    # Historical volatility
    hist_vol = vol_estimator.historical_volatility(data['returns'])
    print(f"\nHistorical Volatility (30d): {hist_vol:.4f}")

    # GARCH volatility
    garch_vol, diagnostics = vol_estimator.garch_volatility(data['returns'])
    print(f"\nGARCH(1,1) Volatility Forecast: {garch_vol:.4f}")
    print(f"  Persistence (α + β): {diagnostics['persistence']:.4f}")
    print(f"  Converged: {diagnostics['converged']}")

    # Realized volatility
    realized_vol = vol_estimator.realized_volatility_parkinson(
        data['high'],
        data['low']
    )
    print(f"\nRealized Volatility (Parkinson): {realized_vol:.4f}")

    # Hybrid volatility
    hybrid_vol, individual_vols = vol_estimator.hybrid_volatility(data)
    print(f"\nHybrid Volatility: {hybrid_vol:.4f}")
    print(f"  Individual estimates: {individual_vols}")

    # Volatility term structure
    maturities = np.array([7, 30, 60, 90, 180, 365])
    term_structure = vol_estimator.volatility_term_structure(
        data['returns'],
        maturities,
        method='garch'
    )
    print("\nVolatility Term Structure:")
    print(term_structure)
