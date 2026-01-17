"""
Volatility Surface Visualization Module
3D visualization of implied volatility across strikes and maturities.

The volatility surface captures how implied volatility varies with:
- Strike price (moneyness): Creates volatility smile/skew
- Time to maturity: Creates term structure

Common patterns:
- Volatility Smile: Higher IV for OTM options (both calls and puts)
- Volatility Skew: Asymmetric smile (often higher for OTM puts)
- Term Structure: IV typically higher for shorter maturities (uncertainty)
"""

import logging
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class VolatilitySurface:
    """
    Construct and visualize implied volatility surface.

    The volatility surface is a critical tool for:
    - Options pricing consistency checks
    - Trading strategy development
    - Risk management
    - Market sentiment analysis
    """

    def __init__(self, config: Dict):
        """
        Initialize volatility surface.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.surface_config = config.get('volatility_surface', {})

        # Grid parameters
        self.n_strikes = self.surface_config.get('n_strikes', 15)
        self.n_maturities = self.surface_config.get('n_maturities', 10)
        self.strike_range_pct = self.surface_config.get('strike_range_pct', 0.30)

        logger.info("Initialized VolatilitySurface")

    def construct_surface(
        self,
        pricer: object,
        futures_price: float,
        base_volatility: float,
        risk_free_rate: float,
        min_maturity: float = 1/12,  # 1 month
        max_maturity: float = 2.0,   # 2 years
        smile_amplitude: float = 0.05,
        skew_slope: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Construct theoretical volatility surface.

        Creates a parametric volatility surface with smile and skew.

        Volatility model:
        σ(K,T) = σ_base + smile_amplitude * (K/F - 1)² - skew_slope * (K/F - 1)

        Args:
            pricer: Black76Pricer instance
            futures_price: Current futures price
            base_volatility: Base/ATM volatility
            risk_free_rate: Risk-free rate
            min_maturity: Minimum maturity (years)
            max_maturity: Maximum maturity (years)
            smile_amplitude: Amplitude of volatility smile
            skew_slope: Slope of volatility skew

        Returns:
            Tuple of (strikes, maturities, implied_vols) grids
        """
        # Create strike grid centered around futures price
        strike_min = futures_price * (1 - self.strike_range_pct)
        strike_max = futures_price * (1 + self.strike_range_pct)
        strikes = np.linspace(strike_min, strike_max, self.n_strikes)

        # Create maturity grid
        maturities = np.linspace(min_maturity, max_maturity, self.n_maturities)

        # Create meshgrid
        K, T = np.meshgrid(strikes, maturities)

        # Calculate moneyness
        moneyness = K / futures_price

        # Parametric volatility surface with smile and skew
        # σ(K,T) = σ_base + smile * (m-1)² - skew * (m-1)
        # where m = moneyness = K/F
        IV = base_volatility + smile_amplitude * (moneyness - 1)**2 - skew_slope * (moneyness - 1)

        # Add term structure effect (shorter maturities have higher vol)
        term_structure_factor = 1.0 + 0.1 * np.exp(-2 * T)
        IV = IV * term_structure_factor

        # Ensure volatilities are positive and reasonable
        IV = np.clip(IV, 0.05, 2.0)

        logger.info(
            f"Constructed volatility surface: "
            f"{self.n_strikes}x{self.n_maturities} grid, "
            f"IV range: [{IV.min():.3f}, {IV.max():.3f}]"
        )

        return K, T, IV

    def construct_from_market_prices(
        self,
        pricer: object,
        futures_price: float,
        market_data: pd.DataFrame,
        risk_free_rate: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Construct volatility surface from observed market option prices.

        Args:
            pricer: Black76Pricer instance
            futures_price: Current futures price
            market_data: DataFrame with columns: strike, maturity, option_price, option_type
            risk_free_rate: Risk-free rate

        Returns:
            Tuple of (strikes, maturities, implied_vols) grids
        """
        # Validate input
        required_cols = ['strike', 'maturity', 'option_price', 'option_type']
        missing_cols = [col for col in required_cols if col not in market_data.columns]
        if missing_cols:
            raise ValueError(
                f"TFM2001 DATA: Market data missing columns: {missing_cols}"
            )

        # Extract unique strikes and maturities
        unique_strikes = sorted(market_data['strike'].unique())
        unique_maturities = sorted(market_data['maturity'].unique())

        # Initialize IV grid
        iv_grid = np.full((len(unique_maturities), len(unique_strikes)), np.nan)

        # Calculate implied volatility for each option
        for i, maturity in enumerate(unique_maturities):
            for j, strike in enumerate(unique_strikes):
                # Find matching option
                mask = (
                    (market_data['strike'] == strike) &
                    (market_data['maturity'] == maturity)
                )

                matching_options = market_data[mask]

                if len(matching_options) == 0:
                    continue

                # Prefer calls for OTM calls and puts for OTM puts
                if strike >= futures_price:
                    # OTM call
                    option = matching_options[matching_options['option_type'] == 'call']
                else:
                    # OTM put
                    option = matching_options[matching_options['option_type'] == 'put']

                if len(option) == 0:
                    # Fallback to any available option
                    option = matching_options.iloc[0:1]

                if len(option) > 0:
                    try:
                        market_price = option.iloc[0]['option_price']
                        option_type = option.iloc[0]['option_type']

                        # Calculate implied volatility
                        iv = pricer.implied_volatility(
                            market_price, futures_price, strike,
                            maturity, risk_free_rate, option_type
                        )

                        iv_grid[i, j] = iv

                    except Exception as e:
                        logger.warning(
                            f"TFM4001 INFERENCE: Failed to calculate IV for "
                            f"K={strike}, T={maturity}: {e}"
                        )
                        continue

        # Create meshgrid
        K, T = np.meshgrid(unique_strikes, unique_maturities)

        logger.info(
            f"Constructed market-implied volatility surface from {len(market_data)} options"
        )

        return K, T, iv_grid

    def plot_surface_3d(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        implied_vols: np.ndarray,
        futures_price: float,
        title: str = "Implied Volatility Surface",
        save_path: Optional[str] = None
    ):
        """
        Create 3D surface plot of implied volatility.

        Args:
            strikes: Strike price grid
            maturities: Maturity grid
            implied_vols: Implied volatility grid
            futures_price: Current futures price
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Plotly figure
        """
        # Convert to percentage for display
        iv_pct = implied_vols * 100

        # Create 3D surface
        fig = go.Figure(data=[go.Surface(
            x=strikes,
            y=maturities,
            z=iv_pct,
            colorscale='Viridis',
            colorbar=dict(title="IV (%)"),
            hovertemplate=(
                'Strike: $%{x:.2f}<br>'
                'Maturity: %{y:.2f}y<br>'
                'IV: %{z:.2f}%<br>'
                '<extra></extra>'
            )
        )])

        # Add marker for ATM
        fig.add_trace(go.Scatter3d(
            x=[futures_price],
            y=[maturities.mean()],
            z=[iv_pct.mean()],
            mode='markers',
            marker=dict(size=8, color='red', symbol='diamond'),
            name='ATM',
            hovertext=f'ATM: F=${futures_price:.2f}'
        ))

        # Layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Strike Price ($)',
                yaxis_title='Time to Maturity (years)',
                zaxis_title='Implied Volatility (%)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                )
            ),
            width=1000,
            height=700
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Volatility surface 3D plot saved to {save_path}")

        return fig

    def plot_volatility_smile(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        implied_vols: np.ndarray,
        futures_price: float,
        selected_maturities: Optional[list] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot volatility smile for selected maturities.

        Volatility smile shows how IV varies with strike for a given maturity.

        Args:
            strikes: Strike price grid
            maturities: Maturity grid
            implied_vols: Implied volatility grid
            futures_price: Current futures price
            selected_maturities: List of maturity indices to plot
            save_path: Optional path to save figure

        Returns:
            Plotly figure
        """
        if selected_maturities is None:
            # Select a few representative maturities
            n_mats = min(4, maturities.shape[0])
            selected_maturities = np.linspace(
                0, maturities.shape[0]-1, n_mats, dtype=int
            )

        fig = go.Figure()

        for mat_idx in selected_maturities:
            maturity_val = maturities[mat_idx, 0]
            strike_slice = strikes[mat_idx, :]
            iv_slice = implied_vols[mat_idx, :] * 100

            # Calculate moneyness
            moneyness = strike_slice / futures_price

            fig.add_trace(go.Scatter(
                x=moneyness,
                y=iv_slice,
                mode='lines+markers',
                name=f'T = {maturity_val:.2f}y',
                hovertemplate=(
                    'Moneyness: %{x:.3f}<br>'
                    'IV: %{y:.2f}%<br>'
                    '<extra></extra>'
                )
            ))

        # Add vertical line at ATM (moneyness = 1)
        fig.add_vline(
            x=1.0,
            line_dash="dash",
            line_color="gray",
            annotation_text="ATM"
        )

        # Layout
        fig.update_layout(
            title='Volatility Smile',
            xaxis_title='Moneyness (Strike / Futures)',
            yaxis_title='Implied Volatility (%)',
            hovermode='x unified',
            showlegend=True,
            width=900,
            height=600
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Volatility smile plot saved to {save_path}")

        return fig

    def plot_volatility_term_structure(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        implied_vols: np.ndarray,
        futures_price: float,
        selected_moneyness: Optional[list] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot volatility term structure for selected strikes.

        Term structure shows how IV varies with maturity for a given strike.

        Args:
            strikes: Strike price grid
            maturities: Maturity grid
            implied_vols: Implied volatility grid
            futures_price: Current futures price
            selected_moneyness: List of moneyness levels (e.g., [0.9, 1.0, 1.1])
            save_path: Optional path to save figure

        Returns:
            Plotly figure
        """
        if selected_moneyness is None:
            # ATM and a few OTM levels
            selected_moneyness = [0.90, 0.95, 1.00, 1.05, 1.10]

        fig = go.Figure()

        for moneyness_target in selected_moneyness:
            # Find closest strike for this moneyness at first maturity
            target_strike = futures_price * moneyness_target
            strike_idx = np.argmin(np.abs(strikes[0, :] - target_strike))

            # Extract IV term structure for this strike
            maturity_vals = maturities[:, 0]
            iv_vals = implied_vols[:, strike_idx] * 100

            label = f"K/F = {moneyness_target:.2f}"
            if abs(moneyness_target - 1.0) < 0.01:
                label += " (ATM)"

            fig.add_trace(go.Scatter(
                x=maturity_vals,
                y=iv_vals,
                mode='lines+markers',
                name=label,
                hovertemplate=(
                    'Maturity: %{x:.2f}y<br>'
                    'IV: %{y:.2f}%<br>'
                    '<extra></extra>'
                )
            ))

        # Layout
        fig.update_layout(
            title='Volatility Term Structure',
            xaxis_title='Time to Maturity (years)',
            yaxis_title='Implied Volatility (%)',
            hovermode='x unified',
            showlegend=True,
            width=900,
            height=600
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Volatility term structure plot saved to {save_path}")

        return fig

    def plot_combined_surface_analysis(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        implied_vols: np.ndarray,
        futures_price: float,
        save_path: Optional[str] = None
    ):
        """
        Create comprehensive multi-panel volatility surface analysis.

        Combines:
        - 3D surface
        - Volatility smile
        - Term structure
        - Heatmap

        Args:
            strikes: Strike price grid
            maturities: Maturity grid
            implied_vols: Implied volatility grid
            futures_price: Current futures price
            save_path: Optional path to save figure

        Returns:
            Plotly figure
        """
        from plotly.subplots import make_subplots

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Implied Volatility Heatmap',
                'Volatility Smile (Multiple Maturities)',
                'Volatility Term Structure (Multiple Strikes)',
                'Statistics Summary'
            ),
            specs=[
                [{"type": "heatmap"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "table"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.10
        )

        # Convert to percentage
        iv_pct = implied_vols * 100

        # 1. Heatmap
        moneyness_grid = strikes / futures_price

        fig.add_trace(
            go.Heatmap(
                x=moneyness_grid[0, :],
                y=maturities[:, 0],
                z=iv_pct,
                colorscale='Viridis',
                colorbar=dict(x=0.46, y=0.77, len=0.4, title="IV (%)"),
                hovertemplate='Moneyness: %{x:.3f}<br>Maturity: %{y:.2f}y<br>IV: %{z:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )

        # 2. Volatility smile (3 maturities)
        mat_indices = [0, len(maturities)//2, -1]
        for mat_idx in mat_indices:
            fig.add_trace(
                go.Scatter(
                    x=moneyness_grid[mat_idx, :],
                    y=iv_pct[mat_idx, :],
                    mode='lines+markers',
                    name=f'T={maturities[mat_idx, 0]:.2f}y',
                    showlegend=True
                ),
                row=1, col=2
            )

        # 3. Term structure (3 strikes: ITM, ATM, OTM)
        strike_indices = [len(strikes[0])//4, len(strikes[0])//2, 3*len(strikes[0])//4]
        for strike_idx in strike_indices:
            moneyness_val = moneyness_grid[0, strike_idx]
            fig.add_trace(
                go.Scatter(
                    x=maturities[:, 0],
                    y=iv_pct[:, strike_idx],
                    mode='lines+markers',
                    name=f'K/F={moneyness_val:.2f}',
                    showlegend=True
                ),
                row=2, col=1
            )

        # 4. Statistics table
        stats_data = {
            'Metric': [
                'Min IV',
                'Max IV',
                'Mean IV',
                'ATM IV (avg)',
                'IV Range',
                'Grid Size'
            ],
            'Value': [
                f'{np.nanmin(iv_pct):.2f}%',
                f'{np.nanmax(iv_pct):.2f}%',
                f'{np.nanmean(iv_pct):.2f}%',
                f'{iv_pct[:, len(strikes[0])//2].mean():.2f}%',
                f'{np.nanmax(iv_pct) - np.nanmin(iv_pct):.2f}%',
                f'{strikes.shape[1]}x{strikes.shape[0]}'
            ]
        }

        fig.add_trace(
            go.Table(
                header=dict(
                    values=list(stats_data.keys()),
                    fill_color='lightgray',
                    align='left'
                ),
                cells=dict(
                    values=list(stats_data.values()),
                    fill_color='white',
                    align='left'
                )
            ),
            row=2, col=2
        )

        # Update axes labels
        fig.update_xaxes(title_text="Moneyness (K/F)", row=1, col=1)
        fig.update_yaxes(title_text="Maturity (years)", row=1, col=1)

        fig.update_xaxes(title_text="Moneyness (K/F)", row=1, col=2)
        fig.update_yaxes(title_text="IV (%)", row=1, col=2)

        fig.update_xaxes(title_text="Maturity (years)", row=2, col=1)
        fig.update_yaxes(title_text="IV (%)", row=2, col=1)

        # Layout
        fig.update_layout(
            title='Comprehensive Volatility Surface Analysis',
            height=1000,
            width=1400,
            showlegend=True
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Combined volatility analysis saved to {save_path}")

        return fig


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('../..')

    from src.config.loader import load_config
    from src.config.logging_config import setup_logging
    from src.models.black_scholes import Black76Pricer

    # Load configuration
    config = load_config('../../config/parameters.toml')
    logger = setup_logging(config)

    # Initialize
    pricer = Black76Pricer(config)
    vol_surface = VolatilitySurface(config)

    print("\n" + "="*80)
    print("VOLATILITY SURFACE EXAMPLES")
    print("="*80)

    # Parameters
    futures_price = 100.0
    base_volatility = 0.25
    risk_free_rate = 0.05

    # Construct theoretical surface with smile and skew
    K, T, IV = vol_surface.construct_surface(
        pricer=pricer,
        futures_price=futures_price,
        base_volatility=base_volatility,
        risk_free_rate=risk_free_rate,
        smile_amplitude=0.08,  # 8% smile amplitude
        skew_slope=0.05        # 5% negative skew
    )

    print(f"\nVolatility Surface Statistics:")
    print(f"  Grid Size: {K.shape}")
    print(f"  IV Range: [{IV.min():.3f}, {IV.max():.3f}]")
    print(f"  Strike Range: [${K.min():.2f}, ${K.max():.2f}]")
    print(f"  Maturity Range: [{T.min():.2f}y, {T.max():.2f}y]")

    # Create visualizations
    print("\nGenerating visualizations...")

    # 3D surface plot
    fig_3d = vol_surface.plot_surface_3d(
        K, T, IV, futures_price,
        title="Implied Volatility Surface (Parametric)"
    )

    # Volatility smile
    fig_smile = vol_surface.plot_volatility_smile(
        K, T, IV, futures_price
    )

    # Term structure
    fig_ts = vol_surface.plot_volatility_term_structure(
        K, T, IV, futures_price
    )

    # Combined analysis
    fig_combined = vol_surface.plot_combined_surface_analysis(
        K, T, IV, futures_price
    )

    print("\nVisualization objects created successfully")
    print("Call .show() on any figure object to display interactively")
