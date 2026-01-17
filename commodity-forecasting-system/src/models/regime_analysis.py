"""
Regime Analysis Module
Analysis and visualization tools for HMM regimes.
"""

import logging
from typing import Dict, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.models.hmm_core import CommodityHMM

logger = logging.getLogger(__name__)


class RegimeAnalyzer:
    """
    Comprehensive regime analysis and visualization.

    Features:
    - Transition dynamics analysis
    - Regime persistence and duration
    - Asymmetry detection
    - Visualization tools
    """

    def __init__(self, hmm_model: CommodityHMM):
        """
        Initialize analyzer with fitted HMM model.

        Args:
            hmm_model: Fitted CommodityHMM instance
        """
        if not hmm_model.is_fitted:
            raise ValueError(
                "TFM4001 INFERENCE: HMM model must be fitted before analysis"
            )

        self.model = hmm_model
        self.n_states = hmm_model.n_states
        self.regime_stats = hmm_model.get_regime_stats()
        self.trans_matrix = hmm_model.get_transition_matrix()

        logger.info(f"Initialized RegimeAnalyzer for {self.n_states}-state HMM")

    def analyze_transitions(self) -> Dict:
        """
        Analyze regime transition dynamics.

        Returns:
            Dictionary with transition analysis:
            - persistence: Diagonal elements (prob of staying in state)
            - expected_duration: Expected time in each state
            - asymmetry: Transition asymmetry between states
            - steady_state: Long-run state probabilities
        """
        logger.info("Analyzing regime transitions...")

        # Persistence (diagonal elements)
        persistence = np.diag(self.trans_matrix)

        # Expected duration in each state
        # E[duration] = 1 / (1 - P(stay))
        expected_duration = 1.0 / (1.0 - persistence + 1e-10)  # Add small epsilon

        # Asymmetry: How much easier to enter than exit?
        asymmetry = {}
        for i in range(self.n_states):
            for j in range(self.n_states):
                if i != j:
                    ratio = self.trans_matrix[i, j] / (self.trans_matrix[j, i] + 1e-10)
                    label_i = self.regime_stats[i]['label']
                    label_j = self.regime_stats[j]['label']
                    asymmetry[f'{label_i}→{label_j}'] = ratio

        # Steady-state distribution (eigenvector of transition matrix)
        eigenvalues, eigenvectors = np.linalg.eig(self.trans_matrix.T)
        steady_state_idx = np.argmax(eigenvalues)
        steady_state = np.abs(eigenvectors[:, steady_state_idx])
        steady_state = steady_state / steady_state.sum()

        results = {
            'persistence': {
                self.regime_stats[i]['label']: float(persistence[i])
                for i in range(self.n_states)
            },
            'expected_duration': {
                self.regime_stats[i]['label']: float(expected_duration[i])
                for i in range(self.n_states)
            },
            'asymmetry': {k: float(v) for k, v in asymmetry.items()},
            'steady_state': {
                self.regime_stats[i]['label']: float(steady_state[i])
                for i in range(self.n_states)
            }
        }

        # Log summary
        logger.info("Transition Analysis:")
        for label, persist in results['persistence'].items():
            duration = results['expected_duration'][label]
            logger.info(f"  {label}: persistence={persist:.2f}, duration={duration:.1f} days")

        return results

    def plot_regime_overlay(
        self,
        price_series: pd.Series,
        states: np.ndarray,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Plot price series with regime overlay.

        Args:
            price_series: Price time series
            states: HMM state sequence (same length as price_series)
            save_path: Optional path to save figure

        Returns:
            Plotly figure
        """
        logger.info("Creating regime overlay plot...")

        # Create figure
        fig = go.Figure()

        # Add price line
        fig.add_trace(go.Scatter(
            x=price_series.index,
            y=price_series.values,
            mode='lines',
            name='Price',
            line=dict(color='black', width=1.5),
            yaxis='y1'
        ))

        # Color map for regimes
        colors = {
            0: 'rgba(255, 0, 0, 0.2)',    # Red
            1: 'rgba(128, 128, 128, 0.2)', # Gray
            2: 'rgba(0, 255, 0, 0.2)',    # Green
            3: 'rgba(0, 0, 255, 0.2)',    # Blue
            4: 'rgba(255, 165, 0, 0.2)',  # Orange
            5: 'rgba(128, 0, 128, 0.2)'   # Purple
        }

        # Add regime bands
        for state in range(self.n_states):
            state_mask = (states == state)
            regime_label = self.regime_stats[state]['label']

            # Find contiguous regions
            changes = np.diff(state_mask.astype(int))
            start_indices = np.where(changes == 1)[0] + 1
            end_indices = np.where(changes == -1)[0] + 1

            # Handle edge cases
            if state_mask[0]:
                start_indices = np.insert(start_indices, 0, 0)
            if state_mask[-1]:
                end_indices = np.append(end_indices, len(state_mask))

            # Add shaded regions
            for start, end in zip(start_indices, end_indices):
                fig.add_vrect(
                    x0=price_series.index[start],
                    x1=price_series.index[end-1],
                    fillcolor=colors.get(state, 'rgba(128, 128, 128, 0.2)'),
                    layer="below",
                    line_width=0,
                    annotation_text=regime_label if end - start > 20 else "",
                    annotation_position="top left"
                )

        # Update layout
        fig.update_layout(
            title='Price Series with Regime Detection',
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified',
            height=600,
            showlegend=True
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Regime overlay plot saved to {save_path}")

        return fig

    def plot_transition_matrix(
        self,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Visualize transition matrix as heatmap.

        Args:
            save_path: Optional path to save figure

        Returns:
            Plotly figure
        """
        logger.info("Creating transition matrix heatmap...")

        # Get regime labels
        labels = [self.regime_stats[i]['label'] for i in range(self.n_states)]

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=self.trans_matrix,
            x=[f"To {label}" for label in labels],
            y=[f"From {label}" for label in labels],
            colorscale='RdYlGn',
            text=np.round(self.trans_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 12},
            colorbar=dict(title="Probability")
        ))

        fig.update_layout(
            title='State Transition Matrix',
            xaxis_title='Destination State',
            yaxis_title='Source State',
            height=500,
            width=600
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Transition matrix plot saved to {save_path}")

        return fig

    def plot_regime_characteristics(
        self,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Plot regime characteristics (returns, volatility, persistence).

        Args:
            save_path: Optional path to save figure

        Returns:
            Plotly figure
        """
        logger.info("Creating regime characteristics plot...")

        labels = [self.regime_stats[i]['label'] for i in range(self.n_states)]
        mean_returns = [self.regime_stats[i]['mean_return'] for i in range(self.n_states)]
        volatilities = [self.regime_stats[i]['volatility'] for i in range(self.n_states)]
        persistences = [self.regime_stats[i]['persistence'] for i in range(self.n_states)]

        # Create subplots
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Mean Return', 'Volatility', 'Persistence'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}]]
        )

        # Mean returns
        fig.add_trace(
            go.Bar(
                x=labels,
                y=mean_returns,
                name='Mean Return',
                marker=dict(
                    color=mean_returns,
                    colorscale='RdYlGn',
                    showscale=False
                )
            ),
            row=1, col=1
        )

        # Volatility
        fig.add_trace(
            go.Bar(
                x=labels,
                y=volatilities,
                name='Volatility',
                marker=dict(
                    color=volatilities,
                    colorscale='Reds',
                    showscale=False
                )
            ),
            row=1, col=2
        )

        # Persistence
        fig.add_trace(
            go.Bar(
                x=labels,
                y=persistences,
                name='Persistence',
                marker=dict(
                    color=persistences,
                    colorscale='Blues',
                    showscale=False
                )
            ),
            row=1, col=3
        )

        # Update layout
        fig.update_xaxes(title_text="Regime", row=1, col=1)
        fig.update_xaxes(title_text="Regime", row=1, col=2)
        fig.update_xaxes(title_text="Regime", row=1, col=3)

        fig.update_yaxes(title_text="Return", row=1, col=1)
        fig.update_yaxes(title_text="Volatility", row=1, col=2)
        fig.update_yaxes(title_text="Probability", row=1, col=3)

        fig.update_layout(
            title='Regime Characteristics',
            height=400,
            showlegend=False
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Regime characteristics plot saved to {save_path}")

        return fig

    def plot_state_probabilities(
        self,
        features: pd.DataFrame,
        window: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Plot posterior state probabilities over time.

        Args:
            features: Feature matrix
            window: Optional window to plot (last N days)
            save_path: Optional path to save figure

        Returns:
            Plotly figure
        """
        logger.info("Calculating posterior state probabilities...")

        # Get posteriors
        posteriors = self.model.predict_proba(features)

        # Apply window if specified
        if window:
            posteriors = posteriors[-window:]
            dates = features.index[-window:] if hasattr(features, 'index') else range(len(posteriors))
        else:
            dates = features.index if hasattr(features, 'index') else range(len(posteriors))

        # Create figure
        fig = go.Figure()

        # Add trace for each state
        for state in range(self.n_states):
            label = self.regime_stats[state]['label']
            fig.add_trace(go.Scatter(
                x=dates,
                y=posteriors[:, state],
                mode='lines',
                name=label,
                stackgroup='one',
                fillcolor=self._get_regime_color(state)
            ))

        # Update layout
        fig.update_layout(
            title='Posterior State Probabilities Over Time',
            xaxis_title='Date',
            yaxis_title='Probability',
            hovermode='x unified',
            height=500,
            yaxis=dict(range=[0, 1])
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"State probabilities plot saved to {save_path}")

        return fig

    def _get_regime_color(self, state: int) -> str:
        """Get color for regime visualization."""
        colors = {
            0: 'rgba(255, 0, 0, 0.5)',    # Red
            1: 'rgba(128, 128, 128, 0.5)', # Gray
            2: 'rgba(0, 255, 0, 0.5)',    # Green
            3: 'rgba(0, 0, 255, 0.5)',    # Blue
            4: 'rgba(255, 165, 0, 0.5)',  # Orange
            5: 'rgba(128, 0, 128, 0.5)'   # Purple
        }
        return colors.get(state, 'rgba(128, 128, 128, 0.5)')

    def generate_report(self) -> str:
        """
        Generate comprehensive text report of regime analysis.

        Returns:
            Formatted text report
        """
        report = []
        report.append("="*80)
        report.append("REGIME ANALYSIS REPORT")
        report.append("="*80)
        report.append("")

        # Model summary
        report.append(f"Model: {self.model}")
        report.append(f"Number of States: {self.n_states}")
        report.append("")

        # Regime statistics
        report.append("-"*80)
        report.append("REGIME STATISTICS")
        report.append("-"*80)

        for state in range(self.n_states):
            stats = self.regime_stats[state]
            report.append(f"\n{stats['label'].upper()} (State {state}):")
            report.append(f"  Mean Return: {stats['mean_return']:.4f}")
            report.append(f"  Volatility: {stats['volatility']:.4f}")
            report.append(f"  Sharpe Ratio: {stats['sharpe']:.2f}")
            report.append(f"  Persistence: {stats['persistence']:.2f}")
            report.append(f"  Frequency: {stats['frequency']:.1%}")
            report.append(f"  Median Return: {stats['median_return']:.4f}")
            report.append(f"  Skewness: {stats['skewness']:.2f}")
            report.append(f"  Kurtosis: {stats['kurtosis']:.2f}")

        # Transition analysis
        trans_analysis = self.analyze_transitions()

        report.append("")
        report.append("-"*80)
        report.append("TRANSITION DYNAMICS")
        report.append("-"*80)
        report.append("")

        report.append("Expected Duration (days):")
        for label, duration in trans_analysis['expected_duration'].items():
            report.append(f"  {label}: {duration:.1f}")

        report.append("")
        report.append("Steady-State Distribution:")
        for label, prob in trans_analysis['steady_state'].items():
            report.append(f"  {label}: {prob:.1%}")

        report.append("")
        report.append("Transition Asymmetry (ratio):")
        for transition, ratio in trans_analysis['asymmetry'].items():
            report.append(f"  {transition}: {ratio:.2f}")

        report.append("")
        report.append("-"*80)
        report.append("TRANSITION MATRIX")
        report.append("-"*80)
        report.append("")

        # Format transition matrix
        labels = [self.regime_stats[i]['label'] for i in range(self.n_states)]
        header = "From \\ To |" + "|".join(f"{label:>12}" for label in labels)
        report.append(header)
        report.append("-" * len(header))

        for i in range(self.n_states):
            row = f"{labels[i]:>10} |"
            for j in range(self.n_states):
                row += f"{self.trans_matrix[i, j]:>12.3f}|"
            report.append(row)

        report.append("")
        report.append("="*80)

        return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('../..')

    from src.config.loader import load_config
    from src.config.logging_config import setup_logging
    from src.data.acquisition import CommodityDataAcquisition
    from src.data.preprocessing import DataPreprocessor
    from src.data.features import FeatureEngineer

    # Load configuration
    config = load_config('../../config/parameters.toml')
    logger = setup_logging(config)

    # Fetch and preprocess data
    data_client = CommodityDataAcquisition(config)
    data = data_client.fetch_commodity_prices()

    preprocessor = DataPreprocessor(config)
    data, _ = preprocessor.preprocess(data)

    # Engineer features
    feature_engineer = FeatureEngineer(config)
    features = feature_engineer.engineer_features(data)

    # Select subset of features
    feature_cols = ['returns', 'volatility_5', 'rsi_14', 'macd']
    hmm_features = features[feature_cols].dropna()

    # Train HMM
    hmm_model = CommodityHMM(config)
    hmm_model.fit_with_multiple_inits(hmm_features, n_inits=5)

    # Create analyzer
    analyzer = RegimeAnalyzer(hmm_model)

    # Generate report
    report = analyzer.generate_report()
    print(report)

    # Save report
    with open('../../outputs/reports/regime_analysis.txt', 'w') as f:
        f.write(report)

    print("\nReport saved to outputs/reports/regime_analysis.txt")

    # Create visualizations
    print("\nGenerating visualizations...")

    # Get states
    features_scaled = hmm_model.scaler.transform(hmm_features.values)
    states = hmm_model.model.predict(features_scaled)

    # Plot regime overlay
    fig1 = analyzer.plot_regime_overlay(
        data['close'].iloc[-len(states):],
        states,
        save_path='../../outputs/figures/regime_overlay.html'
    )

    # Plot transition matrix
    fig2 = analyzer.plot_transition_matrix(
        save_path='../../outputs/figures/transition_matrix.html'
    )

    # Plot regime characteristics
    fig3 = analyzer.plot_regime_characteristics(
        save_path='../../outputs/figures/regime_characteristics.html'
    )

    # Plot state probabilities
    fig4 = analyzer.plot_state_probabilities(
        hmm_features,
        window=250,  # Last 250 days
        save_path='../../outputs/figures/state_probabilities.html'
    )

    print("Visualizations saved to outputs/figures/")
