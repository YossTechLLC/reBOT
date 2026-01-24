"""
Visualization Components for Volatility Prediction UI
======================================================
Plotly-based interactive charts and visualizations.

Chart Types:
- Candlestick charts with regime overlay
- Volatility time series
- Regime transition heatmaps
- Confidence gauges
- Feature contribution charts
- Confusion matrices

Author: Claude + User
Date: 2026-01-17
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class VolatilityCharts:
    """
    Collection of chart generation functions for volatility prediction UI.
    """

    # Color scheme for volatility regimes
    REGIME_COLORS = {
        'low_vol': 'rgba(0, 255, 0, 0.15)',      # Green
        'normal_vol': 'rgba(255, 255, 0, 0.15)', # Yellow
        'high_vol': 'rgba(255, 0, 0, 0.15)'      # Red
    }

    REGIME_LINE_COLORS = {
        'low_vol': 'rgb(0, 200, 0)',
        'normal_vol': 'rgb(200, 200, 0)',
        'high_vol': 'rgb(200, 0, 0)'
    }

    @staticmethod
    def plot_candlestick_with_regime(
        df: pd.DataFrame,
        regime_col: str = 'regime_label',
        title: str = 'SPY Price Action with Volatility Regimes'
    ) -> go.Figure:
        """
        Create candlestick chart with regime-colored background.

        Args:
            df: DataFrame with OHLCV data and regime labels
            regime_col: Column name for regime labels
            title: Chart title

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        # Add candlesticks
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='SPY',
            showlegend=True
        ))

        # Add regime background colors
        if regime_col in df.columns:
            for regime, color in VolatilityCharts.REGIME_COLORS.items():
                regime_periods = df[df[regime_col] == regime]
                for idx in regime_periods.index:
                    fig.add_vrect(
                        x0=idx,
                        x1=idx + pd.Timedelta(days=1),
                        fillcolor=color,
                        layer='below',
                        line_width=0,
                        annotation_text=None
                    )

        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified',
            template='plotly_white',
            height=500,
            xaxis_rangeslider_visible=False
        )

        return fig

    @staticmethod
    def plot_volatility_timeseries(
        df: pd.DataFrame,
        volatility_col: str = 'intraday_range_pct',
        regime_col: str = 'regime_label',
        threshold: float = 0.012,
        title: str = 'Intraday Volatility with Regime Detection'
    ) -> go.Figure:
        """
        Create volatility time series with regime-colored segments.

        Args:
            df: DataFrame with volatility and regime data
            volatility_col: Column name for volatility values
            regime_col: Column name for regime labels
            threshold: Volatility threshold line
            title: Chart title

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        # Add volatility line
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[volatility_col] * 100,  # Convert to percentage
            mode='lines',
            name='Intraday Range %',
            line=dict(color='blue', width=2),
            hovertemplate='%{y:.2f}%<extra></extra>'
        ))

        # Add threshold line
        fig.add_hline(
            y=threshold * 100,
            line_dash='dash',
            line_color='red',
            annotation_text=f'Threshold ({threshold*100:.1f}%)',
            annotation_position='right'
        )

        # Add regime backgrounds
        if regime_col in df.columns:
            for regime, color in VolatilityCharts.REGIME_COLORS.items():
                regime_periods = df[df[regime_col] == regime]
                for idx in regime_periods.index:
                    fig.add_vrect(
                        x0=idx,
                        x1=idx + pd.Timedelta(days=1),
                        fillcolor=color,
                        layer='below',
                        line_width=0
                    )

        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Volatility (%)',
            hovermode='x unified',
            template='plotly_white',
            height=400
        )

        return fig

    @staticmethod
    def plot_confidence_gauge(
        score: float,
        threshold: float = 40.0,
        title: str = 'Confidence Score'
    ) -> go.Figure:
        """
        Create a gauge chart for confidence score.

        Args:
            score: Confidence score (0-100)
            threshold: Threshold for trading decision
            title: Chart title

        Returns:
            Plotly Figure object
        """
        # Determine color based on score
        if score >= threshold:
            color = 'green'
        elif score >= threshold * 0.7:
            color = 'yellow'
        else:
            color = 'red'

        fig = go.Figure(go.Indicator(
            mode='gauge+number+delta',
            value=score,
            title={'text': title},
            delta={'reference': threshold, 'valueformat': '.1f'},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, threshold * 0.7], 'color': 'lightgray'},
                    {'range': [threshold * 0.7, threshold], 'color': 'lightyellow'},
                    {'range': [threshold, 100], 'color': 'lightgreen'}
                ],
                'threshold': {
                    'line': {'color': 'red', 'width': 4},
                    'thickness': 0.75,
                    'value': threshold
                }
            }
        ))

        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20)
        )

        return fig

    @staticmethod
    def plot_feature_contribution(
        feature_signals: Dict[str, float],
        title: str = 'Feature Contributions'
    ) -> go.Figure:
        """
        Create horizontal bar chart showing feature contributions.

        Args:
            feature_signals: Dictionary of feature names and values
            title: Chart title

        Returns:
            Plotly Figure object
        """
        features = list(feature_signals.keys())
        values = list(feature_signals.values())

        # Determine colors (positive = green, negative = red)
        colors = ['green' if v > 0 else 'red' for v in values]

        fig = go.Figure(go.Bar(
            x=values,
            y=features,
            orientation='h',
            marker=dict(color=colors),
            hovertemplate='%{y}: %{x:.4f}<extra></extra>'
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Value',
            yaxis_title='Feature',
            template='plotly_white',
            height=400,
            showlegend=False
        )

        return fig

    @staticmethod
    def plot_regime_probabilities(
        regime_probs: Dict[str, float],
        title: str = 'Regime Probabilities'
    ) -> go.Figure:
        """
        Create bar chart showing regime probabilities.

        Args:
            regime_probs: Dictionary of regime labels and probabilities
            title: Chart title

        Returns:
            Plotly Figure object
        """
        regimes = list(regime_probs.keys())
        probs = [regime_probs[r] * 100 for r in regimes]  # Convert to percentage

        # Use regime colors
        colors = [VolatilityCharts.REGIME_LINE_COLORS.get(r, 'blue') for r in regimes]

        fig = go.Figure(go.Bar(
            x=regimes,
            y=probs,
            marker=dict(color=colors),
            hovertemplate='%{x}: %{y:.1f}%<extra></extra>',
            text=[f'{p:.1f}%' for p in probs],
            textposition='auto'
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Regime',
            yaxis_title='Probability (%)',
            template='plotly_white',
            height=350,
            showlegend=False
        )

        return fig

    @staticmethod
    def plot_confusion_matrix(
        confusion_matrix: np.ndarray,
        labels: List[str] = ['Predicted No Trade', 'Predicted Trade'],
        title: str = 'Validation Confusion Matrix'
    ) -> go.Figure:
        """
        Create heatmap for confusion matrix.

        Args:
            confusion_matrix: 2x2 numpy array with confusion matrix values
            labels: List of labels for axes
            title: Chart title

        Returns:
            Plotly Figure object
        """
        # Annotation text for cells
        annotations = []
        for i in range(2):
            for j in range(2):
                annotations.append(
                    dict(
                        x=j,
                        y=i,
                        text=str(confusion_matrix[i, j]),
                        font=dict(size=20, color='white' if confusion_matrix[i, j] > confusion_matrix.max()/2 else 'black'),
                        showarrow=False
                    )
                )

        fig = go.Figure(data=go.Heatmap(
            z=confusion_matrix,
            x=['Actual No Trade', 'Actual Trade'],
            y=labels,
            colorscale='Blues',
            hovertemplate='%{y} vs %{x}: %{z}<extra></extra>'
        ))

        fig.update_layout(
            title=title,
            annotations=annotations,
            template='plotly_white',
            height=400
        )

        return fig

    @staticmethod
    def plot_validation_metrics(
        results_df: pd.DataFrame,
        title: str = 'Walk-Forward Validation Performance'
    ) -> go.Figure:
        """
        Create time series of validation accuracy over time.

        Args:
            results_df: DataFrame with validation results
            title: Chart title

        Returns:
            Plotly Figure object
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Confidence Score', 'Correct Predictions'),
            vertical_spacing=0.15
        )

        # Confidence score over time
        fig.add_trace(
            go.Scatter(
                x=results_df.index,
                y=results_df['confidence_score'],
                mode='lines+markers',
                name='Confidence Score',
                line=dict(color='blue')
            ),
            row=1, col=1
        )

        # Correct predictions (1 = correct, 0 = incorrect)
        fig.add_trace(
            go.Scatter(
                x=results_df.index,
                y=results_df['correct_prediction'].astype(int),
                mode='markers',
                name='Correct',
                marker=dict(
                    color=results_df['correct_prediction'].map({True: 'green', False: 'red'}),
                    size=10
                )
            ),
            row=2, col=1
        )

        fig.update_layout(
            title=title,
            template='plotly_white',
            height=600,
            showlegend=True
        )

        fig.update_xaxes(title_text='Date', row=2, col=1)
        fig.update_yaxes(title_text='Score', row=1, col=1)
        fig.update_yaxes(title_text='Correct (1) / Incorrect (0)', row=2, col=1)

        return fig
