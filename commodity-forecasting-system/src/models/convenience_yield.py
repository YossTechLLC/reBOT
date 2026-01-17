"""
Convenience Yield Estimation Module
Estimate the convenience yield for commodity futures pricing.

Theory:
Convenience yield represents the benefit of holding the physical commodity
rather than a futures contract. It captures storage costs, supply/demand
imbalances, and optionality value of inventory.

Relationship:
F = S * e^((r - y) * T)

where:
- F: Futures price
- S: Spot price
- r: Risk-free rate
- y: Convenience yield
- T: Time to maturity

Rearranging: y = r - (1/T) * ln(F/S)
"""

import logging
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from datetime import datetime

logger = logging.getLogger(__name__)


class ConvenienceYieldEstimator:
    """
    Estimate convenience yield from futures prices.

    Methods:
    1. Futures Curve Method: Back out from single futures contract
    2. Multiple Contracts Method: Estimate from entire futures curve
    3. Time-Series Method: Model convenience yield dynamics
    """

    def __init__(self, config: Dict):
        """
        Initialize convenience yield estimator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.cy_config = config.get('convenience_yield', {})

        # Parameters
        self.min_yield = self.cy_config.get('min_yield', -0.20)
        self.max_yield = self.cy_config.get('max_yield', 0.40)
        self.default_yield = self.cy_config.get('default_yield', 0.05)

        logger.info("Initialized ConvenienceYieldEstimator")

    def from_single_futures(
        self,
        spot_price: float,
        futures_price: float,
        time_to_maturity: float,
        risk_free_rate: float
    ) -> float:
        """
        Estimate convenience yield from a single futures contract.

        Formula: y = r - (1/T) * ln(F/S)

        Args:
            spot_price: Current spot price
            futures_price: Futures price
            time_to_maturity: Time to maturity (years)
            risk_free_rate: Risk-free rate (decimal)

        Returns:
            Convenience yield (decimal)
        """
        # Validate inputs
        if spot_price <= 0:
            raise ValueError(
                f"TFM2001 DATA: Spot price must be positive (got {spot_price})"
            )
        if futures_price <= 0:
            raise ValueError(
                f"TFM2001 DATA: Futures price must be positive (got {futures_price})"
            )
        if time_to_maturity <= 0:
            raise ValueError(
                f"TFM2001 DATA: Time to maturity must be positive (got {time_to_maturity})"
            )

        # Calculate convenience yield
        try:
            yield_est = risk_free_rate - (1.0 / time_to_maturity) * np.log(
                futures_price / spot_price
            )

            # Sanity check
            if yield_est < self.min_yield or yield_est > self.max_yield:
                logger.warning(
                    f"TFM4001 INFERENCE: Estimated convenience yield {yield_est:.4f} "
                    f"outside reasonable range [{self.min_yield}, {self.max_yield}]. "
                    f"Check input data."
                )

            logger.debug(
                f"Convenience yield: {yield_est:.4f} "
                f"(F={futures_price:.2f}, S={spot_price:.2f}, "
                f"T={time_to_maturity:.2f}, r={risk_free_rate:.4f})"
            )

            return yield_est

        except Exception as e:
            logger.error(
                f"TFM4001 INFERENCE: Failed to calculate convenience yield: {e}. "
                f"Using default: {self.default_yield}"
            )
            return self.default_yield

    def from_futures_curve(
        self,
        spot_price: float,
        futures_curve: pd.DataFrame,
        rate_curve: object
    ) -> Tuple[float, pd.DataFrame]:
        """
        Estimate convenience yield from entire futures curve.

        Uses multiple futures contracts to get more robust estimate.
        Fits a constant convenience yield that best explains observed prices.

        Args:
            spot_price: Current spot price
            futures_curve: DataFrame with columns: maturity (years), price
            rate_curve: InterestRateCurve object for risk-free rates

        Returns:
            Tuple of (estimated_yield, fitted_prices_df)
        """
        if len(futures_curve) == 0:
            logger.error(
                "TFM2001 DATA: Empty futures curve provided. "
                f"Using default yield: {self.default_yield}"
            )
            return self.default_yield, pd.DataFrame()

        # Validate required columns
        required_cols = ['maturity', 'price']
        missing_cols = [col for col in required_cols if col not in futures_curve.columns]
        if missing_cols:
            raise ValueError(
                f"TFM2001 DATA: Futures curve missing columns: {missing_cols}"
            )

        # Extract data
        maturities = futures_curve['maturity'].values
        observed_prices = futures_curve['price'].values

        # Get risk-free rates for each maturity
        risk_free_rates = np.array([
            rate_curve.get_rate(maturity) for maturity in maturities
        ])

        # Objective function: minimize squared pricing errors
        def objective(y):
            """Calculate sum of squared errors."""
            theoretical_prices = spot_price * np.exp(
                (risk_free_rates - y) * maturities
            )
            errors = (observed_prices - theoretical_prices) ** 2
            return np.sum(errors)

        # Optimize
        try:
            result = minimize(
                objective,
                x0=self.default_yield,
                bounds=[(self.min_yield, self.max_yield)],
                method='L-BFGS-B'
            )

            if not result.success:
                logger.warning(
                    f"TFM4001 INFERENCE: Optimization did not converge: {result.message}. "
                    f"Using result anyway."
                )

            estimated_yield = float(result.x[0])

            # Calculate fitted prices
            fitted_prices = spot_price * np.exp(
                (risk_free_rates - estimated_yield) * maturities
            )

            # Calculate fit statistics
            residuals = observed_prices - fitted_prices
            rmse = np.sqrt(np.mean(residuals ** 2))
            mae = np.mean(np.abs(residuals))
            mape = np.mean(np.abs(residuals / observed_prices)) * 100

            logger.info(
                f"Estimated convenience yield: {estimated_yield:.4f} "
                f"(RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%)"
            )

            # Create results DataFrame
            results_df = pd.DataFrame({
                'maturity': maturities,
                'observed_price': observed_prices,
                'fitted_price': fitted_prices,
                'residual': residuals,
                'error_pct': (residuals / observed_prices) * 100
            })

            return estimated_yield, results_df

        except Exception as e:
            logger.error(
                f"TFM4001 INFERENCE: Futures curve optimization failed: {e}. "
                f"Using default yield: {self.default_yield}"
            )
            return self.default_yield, pd.DataFrame()

    def time_varying_yield(
        self,
        historical_data: pd.DataFrame,
        window: int = 30
    ) -> pd.Series:
        """
        Estimate time-varying convenience yield from historical data.

        Args:
            historical_data: DataFrame with columns: date, spot, futures, maturity
            window: Rolling window for estimation

        Returns:
            Series of convenience yields over time
        """
        if len(historical_data) < window:
            logger.warning(
                f"TFM2001 DATA: Insufficient data for time-varying yield "
                f"(have {len(historical_data)}, need {window})"
            )

        # Validate required columns
        required_cols = ['date', 'spot', 'futures', 'maturity', 'risk_free_rate']
        missing_cols = [col for col in required_cols if col not in historical_data.columns]
        if missing_cols:
            raise ValueError(
                f"TFM2001 DATA: Historical data missing columns: {missing_cols}"
            )

        yields = []

        for idx in range(len(historical_data)):
            row = historical_data.iloc[idx]

            try:
                yield_est = self.from_single_futures(
                    spot_price=row['spot'],
                    futures_price=row['futures'],
                    time_to_maturity=row['maturity'],
                    risk_free_rate=row['risk_free_rate']
                )
                yields.append(yield_est)

            except Exception as e:
                logger.debug(f"Failed to estimate yield for row {idx}: {e}")
                yields.append(np.nan)

        yield_series = pd.Series(yields, index=historical_data['date'])

        # Apply rolling smoothing
        yield_series_smooth = yield_series.rolling(window=window, min_periods=1).mean()

        logger.info(
            f"Estimated time-varying convenience yield for {len(yield_series)} periods "
            f"(mean={yield_series_smooth.mean():.4f}, std={yield_series_smooth.std():.4f})"
        )

        return yield_series_smooth

    def term_structure_of_convenience_yield(
        self,
        spot_price: float,
        futures_curve: pd.DataFrame,
        rate_curve: object
    ) -> pd.DataFrame:
        """
        Estimate term structure of convenience yield.

        Different maturities may imply different convenience yields,
        creating a term structure.

        Args:
            spot_price: Current spot price
            futures_curve: DataFrame with maturity and price
            rate_curve: InterestRateCurve object

        Returns:
            DataFrame with maturity and convenience_yield
        """
        term_structure = []

        for idx, row in futures_curve.iterrows():
            maturity = row['maturity']
            futures_price = row['price']

            # Get risk-free rate for this maturity
            risk_free_rate = rate_curve.get_rate(maturity)

            # Estimate convenience yield
            conv_yield = self.from_single_futures(
                spot_price=spot_price,
                futures_price=futures_price,
                time_to_maturity=maturity,
                risk_free_rate=risk_free_rate
            )

            term_structure.append({
                'maturity': maturity,
                'convenience_yield': conv_yield,
                'futures_price': futures_price
            })

        term_structure_df = pd.DataFrame(term_structure)

        logger.info(
            f"Estimated convenience yield term structure for {len(term_structure_df)} maturities"
        )

        return term_structure_df

    def backwardation_contango_indicator(
        self,
        spot_price: float,
        futures_price: float
    ) -> Tuple[str, float]:
        """
        Determine if market is in backwardation or contango.

        Backwardation: Futures < Spot (positive convenience yield expected)
        Contango: Futures > Spot (negative or low convenience yield expected)

        Args:
            spot_price: Current spot price
            futures_price: Near-term futures price

        Returns:
            Tuple of (market_state, basis)
        """
        basis = futures_price - spot_price
        basis_pct = (basis / spot_price) * 100

        if futures_price < spot_price:
            state = "backwardation"
            logger.debug(
                f"Market in BACKWARDATION: futures ({futures_price:.2f}) < "
                f"spot ({spot_price:.2f}), basis={basis_pct:.2f}%"
            )
        elif futures_price > spot_price:
            state = "contango"
            logger.debug(
                f"Market in CONTANGO: futures ({futures_price:.2f}) > "
                f"spot ({spot_price:.2f}), basis={basis_pct:.2f}%"
            )
        else:
            state = "neutral"
            logger.debug("Market NEUTRAL: futures = spot")

        return state, basis_pct

    def plot_convenience_yield_term_structure(
        self,
        term_structure: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """
        Plot convenience yield term structure.

        Args:
            term_structure: DataFrame from term_structure_of_convenience_yield
            save_path: Optional path to save figure
        """
        import plotly.graph_objects as go

        fig = go.Figure()

        # Convenience yield curve
        fig.add_trace(go.Scatter(
            x=term_structure['maturity'],
            y=term_structure['convenience_yield'] * 100,
            mode='lines+markers',
            name='Convenience Yield',
            line=dict(color='green', width=2),
            marker=dict(size=8)
        ))

        # Zero line
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
            annotation_text="Zero Yield"
        )

        # Layout
        fig.update_layout(
            title='Convenience Yield Term Structure',
            xaxis_title='Maturity (Years)',
            yaxis_title='Convenience Yield (%)',
            hovermode='x unified',
            showlegend=True
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Convenience yield term structure plot saved to {save_path}")

        return fig


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('../..')

    from src.config.loader import load_config
    from src.config.logging_config import setup_logging
    from src.models.interest_rates import InterestRateCurve

    # Load configuration
    config = load_config('../../config/parameters.toml')
    logger = setup_logging(config)

    # Initialize
    cy_estimator = ConvenienceYieldEstimator(config)
    rate_curve = InterestRateCurve(config)

    print("\n" + "="*80)
    print("CONVENIENCE YIELD ESTIMATION EXAMPLES")
    print("="*80)

    # Example 1: Single futures contract
    spot_price = 100.0
    futures_price = 102.5
    maturity = 0.25  # 3 months
    risk_free_rate = rate_curve.get_rate(maturity)

    conv_yield = cy_estimator.from_single_futures(
        spot_price=spot_price,
        futures_price=futures_price,
        time_to_maturity=maturity,
        risk_free_rate=risk_free_rate
    )

    print(f"\nSingle Futures Contract:")
    print(f"  Spot: ${spot_price:.2f}")
    print(f"  Futures: ${futures_price:.2f}")
    print(f"  Maturity: {maturity:.2f}y")
    print(f"  Risk-Free Rate: {risk_free_rate*100:.3f}%")
    print(f"  Convenience Yield: {conv_yield*100:.3f}%")

    # Example 2: Backwardation/Contango
    state, basis = cy_estimator.backwardation_contango_indicator(spot_price, futures_price)
    print(f"\nMarket State: {state.upper()}")
    print(f"  Basis: {basis:.2f}%")

    # Example 3: Futures curve
    print("\n" + "-"*80)
    print("Futures Curve Analysis:")
    print("-"*80)

    # Simulate futures curve
    futures_curve = pd.DataFrame({
        'maturity': [3/12, 6/12, 9/12, 1, 2, 3],
        'price': [102.5, 104.0, 105.0, 106.5, 110.0, 113.0]
    })

    print("\nFutures Curve:")
    print(futures_curve)

    # Estimate from curve
    fitted_yield, fitted_prices = cy_estimator.from_futures_curve(
        spot_price=spot_price,
        futures_curve=futures_curve,
        rate_curve=rate_curve
    )

    print(f"\nFitted Convenience Yield: {fitted_yield*100:.3f}%")
    print("\nFitted Prices vs Observed:")
    print(fitted_prices)

    # Term structure
    ts = cy_estimator.term_structure_of_convenience_yield(
        spot_price=spot_price,
        futures_curve=futures_curve,
        rate_curve=rate_curve
    )

    print("\nConvenience Yield Term Structure:")
    print(ts)
