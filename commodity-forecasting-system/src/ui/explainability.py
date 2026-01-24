"""
Explainability Components for Volatility Prediction UI
=======================================================
SHAP-based feature importance and model explanation.

Components:
- SHAP value calculation
- Feature importance visualization
- Local explanation (waterfall plots)
- Global explanation (beeswarm plots)

Author: Claude + User
Date: 2026-01-17
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import streamlit as st
import plotly.graph_objects as go

# SHAP imports - will be available after pip install
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class SHAPExplainer:
    """
    SHAP-based explainability for volatility predictions.

    Note: This is a placeholder implementation. Full SHAP integration
    requires training a compatible model (e.g., sklearn wrapper around HMM).
    """

    def __init__(self, model=None, features: List[str] = None):
        """
        Initialize SHAP explainer.

        Args:
            model: Model to explain (must be SHAP-compatible)
            features: List of feature names
        """
        self.model = model
        self.features = features or []
        self.explainer = None
        self.shap_values = None

        if not SHAP_AVAILABLE:
            st.warning("SHAP library not installed. Install with: pip install shap")

    def calculate_shap_values(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Calculate SHAP values for feature matrix.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            SHAP values array or None if SHAP unavailable
        """
        if not SHAP_AVAILABLE:
            return None

        if self.explainer is None and self.model is not None:
            # Initialize explainer (choose appropriate explainer type)
            try:
                self.explainer = shap.Explainer(self.model, X)
            except Exception as e:
                st.error(f"Failed to initialize SHAP explainer: {str(e)}")
                return None

        if self.explainer is not None:
            try:
                self.shap_values = self.explainer(X)
                return self.shap_values.values
            except Exception as e:
                st.error(f"Failed to calculate SHAP values: {str(e)}")
                return None

        return None

    @staticmethod
    def plot_feature_importance_simple(
        feature_values: Dict[str, float],
        title: str = 'Feature Importance (Simplified)'
    ) -> go.Figure:
        """
        Create simple feature importance chart without SHAP.

        Uses feature values directly as a proxy for importance.

        Args:
            feature_values: Dictionary of feature names and values
            title: Chart title

        Returns:
            Plotly Figure object
        """
        # Sort by absolute value
        sorted_features = sorted(
            feature_values.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        features = [f[0] for f in sorted_features]
        values = [f[1] for f in sorted_features]

        # Normalize to 0-1 range for "importance"
        max_val = max(abs(v) for v in values) if values else 1
        importance = [abs(v) / max_val for v in values]

        fig = go.Figure(go.Bar(
            y=features,
            x=importance,
            orientation='h',
            marker=dict(color=importance, colorscale='Viridis'),
            hovertemplate='%{y}: %{x:.3f}<extra></extra>',
            text=[f'{i:.3f}' for i in importance],
            textposition='auto'
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Relative Importance',
            yaxis_title='Feature',
            template='plotly_white',
            height=400
        )

        return fig


class FeatureAnalyzer:
    """
    Analyzes feature contributions to predictions without requiring SHAP.

    This provides a simplified explanation of feature impact based on
    the confidence scoring logic.
    """

    @staticmethod
    def analyze_feature_signals(
        feature_signals: Dict[str, float],
        regime_volatility: float,
        regime_label: str
    ) -> Dict:
        """
        Analyze how each feature contributes to the prediction.

        Args:
            feature_signals: Dictionary of feature values
            regime_volatility: Expected volatility from regime
            regime_label: Detected regime

        Returns:
            Dictionary with analysis results
        """
        analysis = {
            'regime_impact': {
                'label': regime_label,
                'volatility': regime_volatility,
                'interpretation': _interpret_regime(regime_label, regime_volatility)
            },
            'feature_impacts': []
        }

        # Analyze each feature
        for feature, value in feature_signals.items():
            impact = _analyze_feature_impact(feature, value)
            analysis['feature_impacts'].append({
                'feature': feature,
                'value': value,
                'impact': impact['impact'],
                'interpretation': impact['interpretation']
            })

        return analysis

    @staticmethod
    def create_explanation_text(analysis: Dict) -> str:
        """
        Create human-readable explanation of the prediction.

        Args:
            analysis: Analysis dictionary from analyze_feature_signals

        Returns:
            Formatted explanation string
        """
        regime_info = analysis['regime_impact']
        explanation = f"**Regime Analysis:**\n"
        explanation += f"- Current regime: {regime_info['label']}\n"
        explanation += f"- Expected volatility: {regime_info['volatility']*100:.2f}%\n"
        explanation += f"- {regime_info['interpretation']}\n\n"

        explanation += "**Key Feature Signals:**\n"
        for feature_info in analysis['feature_impacts']:
            explanation += f"- {feature_info['feature']}: {feature_info['interpretation']}\n"

        return explanation


def _interpret_regime(regime_label: str, volatility: float) -> str:
    """Interpret regime detection result."""
    if regime_label == 'high_vol':
        return f"High volatility regime detected ({volatility*100:.1f}%) - Strong trading opportunity"
    elif regime_label == 'normal_vol':
        return f"Normal volatility regime ({volatility*100:.1f}%) - Moderate trading opportunity"
    else:
        return f"Low volatility regime ({volatility*100:.1f}%) - Limited trading opportunity"


def _analyze_feature_impact(feature: str, value: float) -> Dict:
    """Analyze impact of a specific feature."""
    # Feature-specific interpretation logic
    interpretations = {
        'overnight_gap_abs': _interpret_overnight_gap(value),
        'vix_change_1d': _interpret_vix_change(value),
        'vix_level': _interpret_vix_level(value),
        'range_expansion': _interpret_range_expansion(value),
        'volume_surge': _interpret_volume_surge(value),
        'volume_ratio': _interpret_volume_ratio(value),
        'high_range_days_5': _interpret_high_range_days(value)
    }

    return interpretations.get(feature, {
        'impact': 'neutral',
        'interpretation': f'Value: {value:.4f}'
    })


def _interpret_overnight_gap(value: float) -> Dict:
    """Interpret overnight gap value."""
    if value > 0.005:  # > 0.5%
        return {
            'impact': 'bullish',
            'interpretation': f'Large overnight gap ({value*100:.2f}%) suggests increased volatility'
        }
    elif value > 0.002:
        return {
            'impact': 'neutral',
            'interpretation': f'Moderate overnight gap ({value*100:.2f}%)'
        }
    else:
        return {
            'impact': 'bearish',
            'interpretation': f'Small overnight gap ({value*100:.2f}%) suggests low volatility'
        }


def _interpret_vix_change(value: float) -> Dict:
    """Interpret VIX change value."""
    if value > 0.05:  # > 5% increase
        return {
            'impact': 'bullish',
            'interpretation': f'VIX rising sharply ({value*100:.1f}%) - fear increasing'
        }
    elif value < -0.05:
        return {
            'impact': 'bearish',
            'interpretation': f'VIX dropping ({value*100:.1f}%) - complacency increasing'
        }
    else:
        return {
            'impact': 'neutral',
            'interpretation': f'VIX stable (change: {value*100:.1f}%)'
        }


def _interpret_vix_level(value: float) -> Dict:
    """Interpret VIX level."""
    if value > 25:
        return {
            'impact': 'bullish',
            'interpretation': f'VIX elevated ({value:.1f}) - high fear/volatility'
        }
    elif value > 15:
        return {
            'impact': 'neutral',
            'interpretation': f'VIX normal ({value:.1f})'
        }
    else:
        return {
            'impact': 'bearish',
            'interpretation': f'VIX low ({value:.1f}) - low expected volatility'
        }


def _interpret_range_expansion(value: float) -> Dict:
    """Interpret range expansion value."""
    if value > 1.2:
        return {
            'impact': 'bullish',
            'interpretation': f'Strong range expansion ({value:.2f}x) - volatility increasing'
        }
    elif value > 0.8:
        return {
            'impact': 'neutral',
            'interpretation': f'Normal range ({value:.2f}x average)'
        }
    else:
        return {
            'impact': 'bearish',
            'interpretation': f'Range contraction ({value:.2f}x) - volatility decreasing'
        }


def _interpret_volume_surge(value: int) -> Dict:
    """Interpret volume surge value."""
    if value == 1:
        return {
            'impact': 'bullish',
            'interpretation': 'Volume surge detected - increased interest/volatility'
        }
    else:
        return {
            'impact': 'neutral',
            'interpretation': 'Normal volume'
        }


def _interpret_volume_ratio(value: float) -> Dict:
    """Interpret volume ratio."""
    if value > 1.5:
        return {
            'impact': 'bullish',
            'interpretation': f'High volume ({value:.2f}x average) - strong activity'
        }
    elif value > 0.7:
        return {
            'impact': 'neutral',
            'interpretation': f'Normal volume ({value:.2f}x average)'
        }
    else:
        return {
            'impact': 'bearish',
            'interpretation': f'Low volume ({value:.2f}x average) - weak activity'
        }


def _interpret_high_range_days(value: float) -> Dict:
    """Interpret high range days count."""
    if value >= 3:
        return {
            'impact': 'bullish',
            'interpretation': f'{int(value)} high-vol days in last 5 - volatility clustering'
        }
    elif value >= 1:
        return {
            'impact': 'neutral',
            'interpretation': f'{int(value)} high-vol days in last 5'
        }
    else:
        return {
            'impact': 'bearish',
            'interpretation': 'No high-vol days recently - low volatility environment'
        }
