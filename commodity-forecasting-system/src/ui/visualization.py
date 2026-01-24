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

    @staticmethod
    def plot_pnl_payoff_diagram(
        current_price: float,
        call_strike: float,
        put_strike: float,
        credit_received: float,
        strategy_name: str = 'Short Strangle'
    ) -> go.Figure:
        """
        Create P&L payoff diagram for options strategy.

        Shows profit/loss at different underlying prices at expiration.
        For a short strangle, max profit is at the current price, with
        losses increasing as price moves beyond the strikes.

        Args:
            current_price: Current SPY price
            call_strike: Call strike price
            put_strike: Put strike price
            credit_received: Credit collected (premium in dollars)
            strategy_name: Name of strategy for title

        Returns:
            Plotly Figure with payoff curve
        """
        # Generate price range (20% below put to 20% above call)
        price_min = put_strike * 0.92
        price_max = call_strike * 1.08
        price_range = np.linspace(price_min, price_max, 200)

        # Calculate P&L at each price point
        pnl = []
        for price in price_range:
            # Short strangle P&L at expiration:
            # If price < put_strike: loss = (put_strike - price) * 100 - credit
            # If price > call_strike: loss = (price - call_strike) * 100 - credit
            # If between strikes: profit = credit
            call_intrinsic = max(price - call_strike, 0) * 100
            put_intrinsic = max(put_strike - price, 0) * 100
            position_pnl = credit_received - call_intrinsic - put_intrinsic
            pnl.append(position_pnl)

        pnl = np.array(pnl)

        # Create figure
        fig = go.Figure()

        # Color the fill based on profit/loss
        profit_mask = pnl >= 0
        loss_mask = pnl < 0

        # Add profit region
        profit_prices = price_range[profit_mask]
        profit_pnl = pnl[profit_mask]
        if len(profit_prices) > 0:
            fig.add_trace(go.Scatter(
                x=profit_prices,
                y=profit_pnl,
                mode='lines',
                name='Profit Zone',
                line=dict(color='green', width=3),
                fill='tozeroy',
                fillcolor='rgba(0, 200, 0, 0.2)'
            ))

        # Add loss regions
        loss_prices = price_range[loss_mask]
        loss_pnl = pnl[loss_mask]
        if len(loss_prices) > 0:
            fig.add_trace(go.Scatter(
                x=loss_prices,
                y=loss_pnl,
                mode='lines',
                name='Loss Zone',
                line=dict(color='red', width=3),
                fill='tozeroy',
                fillcolor='rgba(200, 0, 0, 0.2)'
            ))

        # Add zero line
        fig.add_hline(y=0, line_dash='dash', line_color='gray', line_width=1)

        # Add current price marker
        fig.add_vline(
            x=current_price,
            line_dash='dash',
            line_color='blue',
            annotation_text=f'Current: ${current_price:.2f}',
            annotation_position='top'
        )

        # Add strike markers
        fig.add_vline(
            x=put_strike,
            line_dash='dot',
            line_color='orange',
            annotation_text=f'Put: ${put_strike:.0f}',
            annotation_position='bottom left'
        )
        fig.add_vline(
            x=call_strike,
            line_dash='dot',
            line_color='orange',
            annotation_text=f'Call: ${call_strike:.0f}',
            annotation_position='bottom right'
        )

        # Add max profit annotation
        fig.add_annotation(
            x=current_price,
            y=credit_received,
            text=f'Max Profit: ${credit_received:.0f}',
            showarrow=True,
            arrowhead=2,
            arrowcolor='green'
        )

        # Format layout
        fig.update_layout(
            title=f'{strategy_name} P&L at Expiration',
            xaxis_title='SPY Price at Expiration ($)',
            yaxis_title='Profit/Loss ($)',
            template='plotly_white',
            height=400,
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )

        return fig

    @staticmethod
    def plot_pnl_distribution(
        pnl_array: np.ndarray,
        title: str = 'P&L Distribution',
        show_stats: bool = True
    ) -> go.Figure:
        """
        Create histogram of P&L outcomes from backtesting.

        Args:
            pnl_array: Array of P&L values from historical trades
            title: Chart title
            show_stats: Whether to show statistics overlay

        Returns:
            Plotly Figure with P&L histogram
        """
        # Calculate statistics
        mean_pnl = np.mean(pnl_array)
        median_pnl = np.median(pnl_array)
        std_pnl = np.std(pnl_array)
        win_rate = np.sum(pnl_array > 0) / len(pnl_array) * 100
        total_pnl = np.sum(pnl_array)

        # Create histogram
        fig = go.Figure()

        # Color bins by profit/loss
        fig.add_trace(go.Histogram(
            x=pnl_array,
            nbinsx=30,
            name='P&L',
            marker=dict(
                color=np.where(pnl_array >= 0, 'green', 'red'),
                line=dict(color='white', width=1)
            ),
            opacity=0.75
        ))

        # Add mean line
        fig.add_vline(
            x=mean_pnl,
            line_dash='dash',
            line_color='blue',
            annotation_text=f'Mean: ${mean_pnl:.2f}',
            annotation_position='top'
        )

        # Add zero line
        fig.add_vline(
            x=0,
            line_dash='solid',
            line_color='gray',
            line_width=2
        )

        # Add statistics annotation
        if show_stats:
            stats_text = (
                f"<b>Statistics</b><br>"
                f"Mean: ${mean_pnl:.2f}<br>"
                f"Median: ${median_pnl:.2f}<br>"
                f"Std Dev: ${std_pnl:.2f}<br>"
                f"Win Rate: {win_rate:.1f}%<br>"
                f"Total P&L: ${total_pnl:.2f}"
            )
            fig.add_annotation(
                x=0.98, y=0.98,
                xref='paper', yref='paper',
                text=stats_text,
                showarrow=False,
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='gray',
                borderwidth=1,
                font=dict(size=11),
                align='left'
            )

        fig.update_layout(
            title=title,
            xaxis_title='P&L ($)',
            yaxis_title='Frequency',
            template='plotly_white',
            height=400,
            showlegend=False
        )

        return fig

    @staticmethod
    def plot_cumulative_pnl(
        trades_df: pd.DataFrame,
        pnl_col: str = 'pnl',
        date_col: str = 'date',
        title: str = 'Cumulative P&L Over Time'
    ) -> go.Figure:
        """
        Create cumulative P&L curve from trade history.

        Args:
            trades_df: DataFrame with trade history
            pnl_col: Column name for P&L values
            date_col: Column name for dates
            title: Chart title

        Returns:
            Plotly Figure with cumulative P&L line
        """
        # Calculate cumulative P&L
        if date_col in trades_df.columns:
            trades_sorted = trades_df.sort_values(date_col)
            x_values = trades_sorted[date_col]
        else:
            trades_sorted = trades_df
            x_values = range(len(trades_df))

        cumulative_pnl = trades_sorted[pnl_col].cumsum()

        # Create figure
        fig = go.Figure()

        # Add cumulative P&L line
        fig.add_trace(go.Scatter(
            x=x_values,
            y=cumulative_pnl,
            mode='lines+markers',
            name='Cumulative P&L',
            line=dict(color='blue', width=2),
            marker=dict(size=6),
            hovertemplate='Trade %{pointNumber+1}<br>Cumulative: $%{y:.2f}<extra></extra>'
        ))

        # Fill based on positive/negative
        fig.add_trace(go.Scatter(
            x=x_values,
            y=cumulative_pnl,
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(0, 100, 200, 0.1)',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Add zero line
        fig.add_hline(y=0, line_dash='dash', line_color='gray')

        # Add final P&L annotation
        final_pnl = cumulative_pnl.iloc[-1]
        color = 'green' if final_pnl >= 0 else 'red'
        fig.add_annotation(
            x=x_values.iloc[-1] if hasattr(x_values, 'iloc') else x_values[-1],
            y=final_pnl,
            text=f'Total: ${final_pnl:.2f}',
            showarrow=True,
            arrowhead=2,
            arrowcolor=color,
            font=dict(color=color, size=12, weight='bold')
        )

        fig.update_layout(
            title=title,
            xaxis_title='Date' if date_col in trades_df.columns else 'Trade Number',
            yaxis_title='Cumulative P&L ($)',
            template='plotly_white',
            height=400,
            showlegend=True
        )

        return fig
