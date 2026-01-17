"""
Interest Rate Curve Construction Module
Risk-free rate estimation for options pricing.

Data Sources:
- FRED (Federal Reserve Economic Data): U.S. Treasury rates
- Interpolation: Cubic spline for arbitrary maturities
"""

import logging
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class InterestRateCurve:
    """
    Construct and interpolate risk-free rate curves.

    U.S. Treasury rates are the standard risk-free benchmark.
    Available maturities from FRED:
    - DGS1MO: 1-Month
    - DGS3MO: 3-Month
    - DGS6MO: 6-Month
    - DGS1: 1-Year
    - DGS2: 2-Year
    - DGS5: 5-Year
    - DGS10: 10-Year
    - DGS30: 30-Year

    Cubic spline interpolation provides smooth rates for any maturity.
    """

    # FRED Treasury series IDs
    TREASURY_SERIES = {
        1/12: 'DGS1MO',    # 1-month
        3/12: 'DGS3MO',    # 3-month
        6/12: 'DGS6MO',    # 6-month
        1: 'DGS1',         # 1-year
        2: 'DGS2',         # 2-year
        5: 'DGS5',         # 5-year
        10: 'DGS10',       # 10-year
        30: 'DGS30'        # 30-year
    }

    def __init__(self, config: Dict):
        """
        Initialize interest rate curve.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.rates_config = config.get('interest_rates', {})

        self.fred_api_key = config.get('api_keys', {}).get('fred')
        self.default_rate = self.rates_config.get('default_rate', 0.05)
        self.cache_duration_hours = self.rates_config.get('cache_duration_hours', 24)

        # Cache
        self.curve_cache = None
        self.cache_timestamp = None

        logger.info("Initialized InterestRateCurve with FRED data source")

    def fetch_treasury_rates(
        self,
        date: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.Series:
        """
        Fetch U.S. Treasury rates from FRED.

        Args:
            date: Date to fetch rates for (YYYY-MM-DD). If None, uses latest.
            use_cache: Whether to use cached data if available

        Returns:
            Series with maturity (years) as index, rates as values
        """
        # Check cache
        if use_cache and self._is_cache_valid():
            logger.debug("Using cached Treasury rates")
            return self.curve_cache

        if not self.fred_api_key:
            logger.warning(
                "TFM1001 CONFIG: FRED API key not configured. "
                f"Using default rate: {self.default_rate}"
            )
            return self._default_curve()

        try:
            from fredapi import Fred
        except ImportError:
            logger.error(
                "TFM1001 CONFIG: fredapi library not installed. "
                "Install with: pip install fredapi"
            )
            return self._default_curve()

        try:
            fred = Fred(api_key=self.fred_api_key)

            rates_dict = {}

            for maturity_years, series_id in self.TREASURY_SERIES.items():
                try:
                    # Fetch series
                    series = fred.get_series(series_id)

                    # Get rate for specific date or latest
                    if date:
                        rate_date = pd.to_datetime(date)
                        if rate_date in series.index:
                            rate = series[rate_date]
                        else:
                            # Find nearest date
                            nearest_date = series.index[series.index <= rate_date][-1]
                            rate = series[nearest_date]
                            logger.debug(
                                f"Date {date} not available for {series_id}, "
                                f"using {nearest_date}"
                            )
                    else:
                        # Use latest available
                        rate = series.dropna().iloc[-1]

                    # Convert from percentage to decimal
                    rates_dict[maturity_years] = rate / 100.0

                except Exception as e:
                    logger.warning(
                        f"TFM2001 DATA: Failed to fetch {series_id}: {e}. Skipping."
                    )
                    continue

            if not rates_dict:
                logger.error(
                    "TFM2001 DATA: No Treasury rates fetched. Using default curve."
                )
                return self._default_curve()

            # Create Series sorted by maturity
            rates_series = pd.Series(rates_dict).sort_index()

            # Cache the result
            self.curve_cache = rates_series
            self.cache_timestamp = datetime.now()

            logger.info(
                f"Fetched {len(rates_series)} Treasury rates from FRED "
                f"(maturities: {list(rates_series.index)})"
            )

            return rates_series

        except Exception as e:
            logger.error(
                f"TFM2001 DATA: Error fetching Treasury rates from FRED: {e}. "
                f"Using default curve."
            )
            return self._default_curve()

    def get_rate(
        self,
        maturity: float,
        date: Optional[str] = None,
        interpolation: str = 'cubic'
    ) -> float:
        """
        Get risk-free rate for a specific maturity.

        Args:
            maturity: Time to maturity in years
            date: Date for rate (YYYY-MM-DD). If None, uses latest.
            interpolation: Interpolation method ('cubic', 'linear', 'nearest')

        Returns:
            Risk-free rate (decimal)
        """
        # Fetch current curve
        curve = self.fetch_treasury_rates(date=date)

        if len(curve) == 0:
            logger.warning(
                f"TFM2001 DATA: No curve data available. "
                f"Using default rate: {self.default_rate}"
            )
            return self.default_rate

        # If exact maturity exists, return it
        if maturity in curve.index:
            return curve[maturity]

        # Interpolate
        if interpolation == 'cubic':
            rate = self._cubic_spline_interpolation(curve, maturity)
        elif interpolation == 'linear':
            rate = np.interp(maturity, curve.index, curve.values)
        elif interpolation == 'nearest':
            nearest_maturity = curve.index[np.argmin(np.abs(curve.index - maturity))]
            rate = curve[nearest_maturity]
        else:
            raise ValueError(
                f"TFM1001 CONFIG: Unknown interpolation method: {interpolation}"
            )

        logger.debug(
            f"Interpolated rate for maturity={maturity:.2f}y: {rate:.4f} "
            f"({interpolation} method)"
        )

        return rate

    def _cubic_spline_interpolation(
        self,
        curve: pd.Series,
        maturity: float
    ) -> float:
        """
        Interpolate rate using cubic spline.

        Cubic spline provides smooth, continuous interpolation
        with continuous first and second derivatives.

        Args:
            curve: Series of known rates
            maturity: Target maturity

        Returns:
            Interpolated rate
        """
        if len(curve) < 4:
            logger.warning(
                f"TFM4001 INFERENCE: Cubic spline requires >=4 points, "
                f"have {len(curve)}. Falling back to linear interpolation."
            )
            return np.interp(maturity, curve.index, curve.values)

        try:
            # Construct cubic spline
            cs = CubicSpline(curve.index, curve.values, extrapolate=True)

            # Interpolate
            rate = float(cs(maturity))

            # Sanity check: rates should be reasonable
            if rate < -0.05 or rate > 0.30:
                logger.warning(
                    f"TFM4001 INFERENCE: Interpolated rate {rate:.4f} "
                    f"seems unreasonable for maturity {maturity:.2f}y. "
                    f"Using nearest neighbor instead."
                )
                nearest_maturity = curve.index[np.argmin(np.abs(curve.index - maturity))]
                rate = curve[nearest_maturity]

            return rate

        except Exception as e:
            logger.error(
                f"TFM4001 INFERENCE: Cubic spline interpolation failed: {e}. "
                f"Using linear interpolation."
            )
            return np.interp(maturity, curve.index, curve.values)

    def _default_curve(self) -> pd.Series:
        """
        Generate default flat yield curve.

        Returns:
            Series with flat rates
        """
        maturities = np.array([1/12, 3/12, 6/12, 1, 2, 5, 10, 30])
        rates = np.full_like(maturities, self.default_rate)

        logger.debug(
            f"Using default flat yield curve: {self.default_rate:.4f}"
        )

        return pd.Series(rates, index=maturities)

    def _is_cache_valid(self) -> bool:
        """Check if cached curve is still valid."""
        if self.curve_cache is None or self.cache_timestamp is None:
            return False

        age = datetime.now() - self.cache_timestamp
        max_age = timedelta(hours=self.cache_duration_hours)

        return age < max_age

    def plot_yield_curve(
        self,
        date: Optional[str] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot the yield curve.

        Args:
            date: Date for curve (YYYY-MM-DD)
            save_path: Optional path to save figure
        """
        import plotly.graph_objects as go

        # Fetch curve
        curve = self.fetch_treasury_rates(date=date)

        if len(curve) == 0:
            logger.error("TFM2001 DATA: No curve data to plot")
            return

        # Create interpolated curve
        maturities_interp = np.linspace(curve.index[0], curve.index[-1], 100)
        rates_interp = [self._cubic_spline_interpolation(curve, m) for m in maturities_interp]

        # Create plot
        fig = go.Figure()

        # Actual points
        fig.add_trace(go.Scatter(
            x=curve.index,
            y=curve.values * 100,  # Convert to percentage
            mode='markers',
            name='Treasury Rates',
            marker=dict(size=10, color='blue')
        ))

        # Interpolated curve
        fig.add_trace(go.Scatter(
            x=maturities_interp,
            y=np.array(rates_interp) * 100,
            mode='lines',
            name='Cubic Spline',
            line=dict(color='lightblue', width=2)
        ))

        # Layout
        title = f"U.S. Treasury Yield Curve"
        if date:
            title += f" ({date})"
        else:
            title += " (Latest)"

        fig.update_layout(
            title=title,
            xaxis_title="Maturity (Years)",
            yaxis_title="Yield (%)",
            hovermode='x unified',
            showlegend=True
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Yield curve plot saved to {save_path}")

        return fig

    def forward_rate(
        self,
        t1: float,
        t2: float,
        date: Optional[str] = None
    ) -> float:
        """
        Calculate forward rate between two maturities.

        Forward rate f(t1,t2) is the implied rate for period [t1, t2]
        based on current spot rates.

        Formula: (1+r2)^t2 = (1+r1)^t1 * (1+f)^(t2-t1)
        Solving: f = [(1+r2)^t2 / (1+r1)^t1]^(1/(t2-t1)) - 1

        Args:
            t1: Start time (years)
            t2: End time (years)
            date: Date for rates

        Returns:
            Forward rate (decimal)
        """
        if t2 <= t1:
            raise ValueError(
                f"TFM1001 CONFIG: t2 must be > t1 (got t1={t1}, t2={t2})"
            )

        r1 = self.get_rate(t1, date=date)
        r2 = self.get_rate(t2, date=date)

        # Calculate forward rate
        forward = ((1 + r2)**t2 / (1 + r1)**t1)**(1/(t2-t1)) - 1

        logger.debug(
            f"Forward rate f({t1},{t2}): {forward:.4f} "
            f"(r({t1})={r1:.4f}, r({t2})={r2:.4f})"
        )

        return forward


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('../..')

    from src.config.loader import load_config
    from src.config.logging_config import setup_logging

    # Load configuration
    config = load_config('../../config/parameters.toml')
    logger = setup_logging(config)

    # Initialize rate curve
    rate_curve = InterestRateCurve(config)

    print("\n" + "="*80)
    print("INTEREST RATE CURVE EXAMPLES")
    print("="*80)

    # Fetch Treasury rates
    curve = rate_curve.fetch_treasury_rates()
    print("\nU.S. Treasury Rates (Latest):")
    print(curve)

    # Get specific maturities
    print("\nInterpolated Rates:")
    test_maturities = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    for maturity in test_maturities:
        rate = rate_curve.get_rate(maturity)
        print(f"  {maturity:5.2f}y: {rate*100:6.3f}%")

    # Forward rates
    print("\nForward Rates:")
    print(f"  1y-2y forward: {rate_curve.forward_rate(1, 2)*100:.3f}%")
    print(f"  2y-5y forward: {rate_curve.forward_rate(2, 5)*100:.3f}%")
    print(f"  5y-10y forward: {rate_curve.forward_rate(5, 10)*100:.3f}%")
