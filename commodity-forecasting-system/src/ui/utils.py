"""
Utility Functions for Volatility Prediction UI
===============================================
Helper functions for the Streamlit UI.

Functions:
- Session state management
- Data formatting
- Error handling
- Export utilities

Author: Claude + User
Date: 2026-01-17
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, Optional
from datetime import datetime
import streamlit as st
import logging

logger = logging.getLogger(__name__)


def initialize_session_state():
    """
    Initialize all session state variables.

    Call this at the start of the Streamlit app to ensure
    all required session state keys exist.
    """
    defaults = {
        'hmm_model': None,
        'hmm_metrics': None,
        'timesfm_forecaster': None,
        'confidence_scorer': None,
        'data_loaded': False,
        'spy_data': None,
        'vix_data': None,
        'features_df': None,
        'last_prediction': None,
        'validation_results': None
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format decimal as percentage string.

    Args:
        value: Decimal value (e.g., 0.025 for 2.5%)
        decimals: Number of decimal places

    Returns:
        Formatted percentage string (e.g., "2.50%")
    """
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: float, decimals: int = 2) -> str:
    """
    Format value as currency string.

    Args:
        value: Dollar amount
        decimals: Number of decimal places

    Returns:
        Formatted currency string (e.g., "$123.45")
    """
    return f"${value:,.{decimals}f}"


def format_date(date: pd.Timestamp, format: str = '%Y-%m-%d') -> str:
    """
    Format pandas Timestamp as string.

    Args:
        date: Pandas Timestamp
        format: Date format string

    Returns:
        Formatted date string
    """
    return date.strftime(format)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if division by zero.

    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if denominator is zero

    Returns:
        Result of division or default
    """
    return numerator / denominator if denominator != 0 else default


def export_to_csv(df: pd.DataFrame, filename: str):
    """
    Export DataFrame to CSV and provide download button.

    Args:
        df: DataFrame to export
        filename: Filename for download
    """
    csv = df.to_csv()
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=filename,
        mime='text/csv'
    )


def display_metrics_grid(metrics: Dict[str, Any], columns: int = 3):
    """
    Display metrics in a grid layout using Streamlit columns.

    Args:
        metrics: Dictionary of metric_name: value pairs
        columns: Number of columns in grid
    """
    cols = st.columns(columns)
    for i, (label, value) in enumerate(metrics.items()):
        with cols[i % columns]:
            if isinstance(value, float):
                if 0 < abs(value) < 1:
                    st.metric(label, format_percentage(value))
                else:
                    st.metric(label, f"{value:.2f}")
            else:
                st.metric(label, value)


def display_status_header(
    hmm_status: Dict,
    timesfm_status: Dict,
    data_date: Optional[pd.Timestamp] = None
):
    """
    Display system status header showing model and data status.

    Args:
        hmm_status: HMM model status dictionary
        timesfm_status: TimesFM forecaster status dictionary
        data_date: Latest data date
    """
    col1, col2, col3 = st.columns(3)

    with col1:
        if hmm_status['loaded']:
            st.success("✅ HMM Loaded")
            st.caption(f"{hmm_status['n_regimes']} regimes, Converged: {hmm_status['converged']}")
        else:
            st.error("❌ HMM Not Loaded")

    with col2:
        if timesfm_status['loaded'] and timesfm_status['available']:
            st.success("✅ TimesFM Ready")
        elif timesfm_status['loaded']:
            st.warning("⚠️ TimesFM Loaded (Checkpoint Missing)")
        else:
            st.info("ℹ️ TimesFM Not Loaded")

    with col3:
        if data_date:
            st.info(f"📅 Data: {format_date(data_date)}")
        else:
            st.warning("⚠️ No Data Loaded")


def handle_error(error: Exception, context: str = ""):
    """
    Display error message in UI with context.

    Args:
        error: Exception that occurred
        context: Context description for the error
    """
    error_msg = f"**Error in {context}:**\n\n{str(error)}"
    st.error(error_msg)
    logger.error(f"{context}: {str(error)}", exc_info=True)


def create_info_expander(title: str, content: str):
    """
    Create an expandable info section.

    Args:
        title: Title for the expander
        content: Content to display when expanded
    """
    with st.expander(f"ℹ️ {title}"):
        st.markdown(content)


def display_dataframe_with_download(
    df: pd.DataFrame,
    title: str,
    filename: str,
    height: int = 400
):
    """
    Display DataFrame with download button.

    Args:
        df: DataFrame to display
        title: Title for the section
        filename: Filename for CSV download
        height: Height of the dataframe display
    """
    st.subheader(title)
    st.dataframe(df, height=height, use_container_width=True)
    export_to_csv(df, filename)


def validate_data_loaded() -> bool:
    """
    Check if data is loaded in session state.

    Returns:
        True if data is loaded, False otherwise
    """
    if not st.session_state.get('data_loaded', False):
        st.warning("⚠️ No data loaded. Please load data first using the sidebar.")
        return False
    return True


def validate_model_trained() -> bool:
    """
    Check if HMM model is trained.

    Returns:
        True if model is trained, False otherwise
    """
    if st.session_state.get('hmm_model') is None:
        st.warning("⚠️ HMM model not trained. Please train the model first using the sidebar.")
        return False
    return True


def get_color_for_regime(regime: str) -> str:
    """
    Get color code for a regime label.

    Args:
        regime: Regime label ('low_vol', 'normal_vol', 'high_vol')

    Returns:
        Color code (CSS color name or hex)
    """
    colors = {
        'low_vol': '#90EE90',      # Light green
        'normal_vol': '#FFD700',   # Gold
        'high_vol': '#FF6347'      # Tomato red
    }
    return colors.get(regime, '#808080')  # Default gray


def get_confidence_color(score: float, threshold: float = 40.0) -> str:
    """
    Get color code for confidence score.

    Args:
        score: Confidence score (0-100)
        threshold: Threshold for trading decision

    Returns:
        Color code
    """
    if score >= threshold:
        return '#90EE90'  # Green
    elif score >= threshold * 0.7:
        return '#FFD700'  # Yellow
    else:
        return '#FF6347'  # Red


def render_regime_badge(regime: str):
    """
    Render a colored badge for regime label.

    Args:
        regime: Regime label
    """
    color = get_color_for_regime(regime)
    st.markdown(
        f'<span style="background-color: {color}; padding: 5px 10px; border-radius: 5px; font-weight: bold;">'
        f'{regime.replace("_", " ").title()}'
        f'</span>',
        unsafe_allow_html=True
    )


def render_confidence_badge(score: float, threshold: float = 40.0):
    """
    Render a colored badge for confidence score.

    Args:
        score: Confidence score (0-100)
        threshold: Threshold for trading decision
    """
    color = get_confidence_color(score, threshold)
    st.markdown(
        f'<span style="background-color: {color}; padding: 5px 10px; border-radius: 5px; font-weight: bold;">'
        f'{score:.1f}/100'
        f'</span>',
        unsafe_allow_html=True
    )


def create_tooltip_help(text: str) -> str:
    """
    Create help text for tooltip.

    Args:
        text: Help text content

    Returns:
        Formatted help text
    """
    return f"💡 {text}"


# Constants for UI
DEFAULT_HISTORY_DAYS = 180
DEFAULT_HMM_REGIMES = 3
DEFAULT_HMM_ITERATIONS = 100
DEFAULT_CONFIDENCE_THRESHOLD = 40.0
DEFAULT_ACCOUNT_SIZE = 10000
DEFAULT_MAX_RISK_PCT = 0.02

HMM_DEFAULT_FEATURES = [
    'overnight_gap_abs',
    'range_ma_5',
    'vix_level',
    'volume_ratio',
    'range_std_5'
]
