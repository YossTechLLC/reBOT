"""
Volatility Prediction System - Main Streamlit App
==================================================
Interactive UI for HMM + TimesFM volatility prediction system.

Features:
- Data loading and visualization
- HMM model training and control
- TimesFM forecasting (optional)
- Prediction dashboard with confidence scoring
- Feature importance and explainability
- Trading strategy recommendations
- Walk-forward validation interface

Author: Claude + User
Date: 2026-01-17

Usage:
    streamlit run src/ui/app.py
    # or
    streamlit run app.py (if symlink exists)
"""

import sys
import os
import logging
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np

# Add src to path for imports (works with symlinks)
# Use realpath to resolve symlinks, then go up 3 levels to project root
real_file = os.path.realpath(__file__)  # Resolves app.py symlink to src/ui/app.py
project_root = os.path.dirname(os.path.dirname(os.path.dirname(real_file)))
sys.path.insert(0, os.path.join(project_root, 'src'))

# Import UI modules
from ui.data_manager import DataManager
from ui.model_controller import ModelController
from ui.visualization import VolatilityCharts
from ui.explainability import FeatureAnalyzer, SHAPExplainer
from ui.strategy import SpreadRecommender, PositionSizer, format_strategy_output
from ui.utils import *

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Volatility Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application entry point."""
    # Initialize session state
    initialize_session_state()

    # Initialize controllers
    if 'data_manager' not in st.session_state:
        st.session_state.data_manager = DataManager(
            alpaca_key='PKDTSYSP4AYPZDNELOHNRPW2BR',
            alpaca_secret='Ae8HNurExREVLghBw5dw9D3Pinkc2Kv8kNcJykMB3XQE'
        )
    if 'model_controller' not in st.session_state:
        st.session_state.model_controller = ModelController()

    # Header
    st.markdown('<div class="main-header">📈 Volatility Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">HMM + TimesFM for 0DTE/1DTE Options Trading</div>', unsafe_allow_html=True)

    # Status header
    hmm_status = st.session_state.model_controller.get_hmm_status()
    timesfm_status = st.session_state.model_controller.get_timesfm_status()
    data_date = st.session_state.get('features_df').index[-1] if st.session_state.get('features_df') is not None else None

    display_status_header(hmm_status, timesfm_status, data_date)

    st.divider()

    # Sidebar
    render_sidebar()

    # Main content - Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Prediction Dashboard",
        "📊 Model Explanation",
        "🔍 Validation Results",
        "💰 Trading Strategy"
    ])

    with tab1:
        render_prediction_tab()

    with tab2:
        render_explanation_tab()

    with tab3:
        render_validation_tab()

    with tab4:
        render_strategy_tab()


def render_sidebar():
    """Render sidebar with all controls."""
    st.sidebar.title("⚙️ Controls")

    # Data Settings
    st.sidebar.header("📊 Data Settings")

    history_days = st.sidebar.slider(
        "History Days",
        min_value=30,
        max_value=365,
        value=DEFAULT_HISTORY_DAYS,
        step=30,
        help=create_tooltip_help("Number of historical days to download for analysis")
    )

    if st.sidebar.button("🔄 Load/Refresh Data", type="primary", use_container_width=True):
        with st.spinner("Loading data..."):
            try:
                spy_data, vix_data, features_df = st.session_state.data_manager.load_complete_dataset(history_days)
                st.session_state.spy_data = spy_data
                st.session_state.vix_data = vix_data
                st.session_state.features_df = features_df
                st.session_state.data_loaded = True

                summary = st.session_state.data_manager.get_data_summary(features_df)
                st.sidebar.success(f"✅ Loaded {summary['total_rows']} days")
                st.sidebar.caption(f"Date range: {summary['start_date']} to {summary['end_date']}")
            except Exception as e:
                handle_error(e, "Data Loading")

    st.sidebar.divider()

    # HMM Parameters
    st.sidebar.header("🎯 HMM Parameters")

    n_regimes = st.sidebar.slider(
        "Number of Regimes",
        min_value=2,
        max_value=5,
        value=DEFAULT_HMM_REGIMES,
        help=create_tooltip_help("Number of volatility regimes (typically 3: low, normal, high)")
    )

    # Display HMM features (informational only - HMM uses these internally)
    with st.sidebar.expander("📊 HMM Features", expanded=False):
        for feat in HMM_DEFAULT_FEATURES:
            st.caption(f"• {feat}")
        st.caption("*HMM uses these 5 features internally*")

    training_iterations = st.sidebar.slider(
        "Training Iterations",
        min_value=50,
        max_value=500,
        value=DEFAULT_HMM_ITERATIONS,
        step=50,
        help=create_tooltip_help("Maximum iterations for HMM training")
    )

    if st.sidebar.button("🚀 Train HMM", use_container_width=True):
        if not validate_data_loaded():
            return

        with st.spinner("Training HMM model..."):
            try:
                model, metrics = st.session_state.model_controller.train_hmm(
                    df=st.session_state.features_df,
                    n_regimes=n_regimes,
                    n_iter=training_iterations
                )

                if metrics['converged']:
                    st.sidebar.success("✅ HMM Training Complete")
                    st.sidebar.caption(f"Log-Likelihood: {metrics['log_likelihood']:.2f}")
                else:
                    st.sidebar.warning("⚠️ HMM did not converge. Try increasing iterations.")
            except Exception as e:
                handle_error(e, "HMM Training")

    # Load pre-trained model option
    if st.sidebar.button("📁 Load Pre-trained HMM", use_container_width=True):
        try:
            st.session_state.model_controller.load_hmm('models/hmm_volatility.pkl')
            st.sidebar.success("✅ Loaded pre-trained HMM")
        except FileNotFoundError:
            st.sidebar.error("❌ Pre-trained model not found at models/hmm_volatility.pkl")
        except Exception as e:
            handle_error(e, "Model Loading")

    st.sidebar.divider()

    # TimesFM Parameters
    st.sidebar.header("🔮 TimesFM Parameters")

    enable_timesfm = st.sidebar.checkbox(
        "Enable TimesFM",
        value=False,
        help=create_tooltip_help("Enable TimesFM foundation model forecasting (requires checkpoint download ~800MB)")
    )

    if enable_timesfm:
        device = st.sidebar.selectbox(
            "Device",
            options=['cpu', 'cuda'],
            index=0,
            help=create_tooltip_help("Device to run TimesFM on (cuda requires GPU)")
        )

        if st.sidebar.button("📥 Load TimesFM", use_container_width=True):
            with st.spinner("Loading TimesFM (this may take a few minutes on first load)..."):
                try:
                    st.session_state.model_controller.load_timesfm(device=device)
                    st.sidebar.success("✅ TimesFM Loaded")
                except Exception as e:
                    handle_error(e, "TimesFM Loading")

    st.sidebar.divider()

    # Model Configuration
    st.sidebar.header("⚙️ Model Configuration")

    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=100.0,
        value=DEFAULT_CONFIDENCE_THRESHOLD,
        step=5.0,
        help=create_tooltip_help("Minimum confidence score to trigger trade signal")
    )

    st.session_state.model_controller.set_confidence_threshold(confidence_threshold)

    # Weight configuration
    with st.sidebar.expander("Advanced: Confidence Weights"):
        regime_weight = st.slider("Regime Weight", 0.0, 1.0, 0.4, 0.05)
        timesfm_weight = st.slider("TimesFM Weight", 0.0, 1.0, 0.4, 0.05)
        feature_weight = st.slider("Feature Weight", 0.0, 1.0, 0.2, 0.05)

        total_weight = regime_weight + timesfm_weight + feature_weight
        if abs(total_weight - 1.0) > 0.01:
            st.warning(f"⚠️ Weights sum to {total_weight:.2f}, should be 1.0")
        else:
            st.session_state.model_controller.set_confidence_weights(
                regime_weight, timesfm_weight, feature_weight
            )

    st.sidebar.divider()

    # Actions
    st.sidebar.header("🚀 Actions")

    if st.sidebar.button("🔮 Run Prediction", type="primary", use_container_width=True):
        if not validate_data_loaded() or not validate_model_trained():
            return

        with st.spinner("Generating prediction..."):
            try:
                prediction = st.session_state.model_controller.predict_latest(
                    st.session_state.features_df
                )
                st.session_state.last_prediction = prediction
                st.sidebar.success("✅ Prediction Complete")
                st.sidebar.caption(f"Confidence: {prediction['confidence_score']:.1f}/100")
            except Exception as e:
                handle_error(e, "Prediction")


def render_prediction_tab():
    """Render the main prediction dashboard tab."""
    st.header("📈 Current Prediction Dashboard")

    prediction = st.session_state.get('last_prediction')

    if prediction is None:
        st.info("ℹ️ No prediction available. Load data, train model, and run prediction using the sidebar.")
        return

    # Prediction card
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Regime Detection")
        render_regime_badge(prediction['regime_label'])
        st.metric("Expected Volatility", format_percentage(prediction['regime_volatility']))
        st.metric("Regime Confidence", format_percentage(prediction['regime_confidence']))

    with col2:
        st.subheader("Confidence Score")
        render_confidence_badge(
            prediction['confidence_score'],
            st.session_state.confidence_scorer.threshold
        )
        st.metric("Total Score", f"{prediction['confidence_score']:.1f}/100")
        st.caption(f"Threshold: {st.session_state.confidence_scorer.threshold:.0f}")

    with col3:
        st.subheader("Trading Decision")
        if prediction['should_trade']:
            st.success("✅ TRADE SIGNAL")
        else:
            st.warning("⏸️ SKIP TRADE")
        st.caption(prediction['recommendation'])

    st.divider()

    # Confidence gauge
    st.subheader("Confidence Gauge")
    gauge_fig = VolatilityCharts.plot_confidence_gauge(
        prediction['confidence_score'],
        st.session_state.confidence_scorer.threshold
    )
    st.plotly_chart(gauge_fig, use_container_width=True)

    st.divider()

    # Charts
    if st.session_state.get('features_df') is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Price Action with Regimes")
            # Add regime labels to SPY data for visualization
            spy_with_regime = st.session_state.spy_data.copy()
            # Note: This is simplified - in practice, you'd add regime labels from HMM predictions
            candlestick_fig = VolatilityCharts.plot_candlestick_with_regime(spy_with_regime)
            st.plotly_chart(candlestick_fig, use_container_width=True)

        with col2:
            st.subheader("Volatility Time Series")
            vol_fig = VolatilityCharts.plot_volatility_timeseries(
                st.session_state.features_df,
                threshold=0.012
            )
            st.plotly_chart(vol_fig, use_container_width=True)

    # Regime probabilities
    st.divider()
    st.subheader("Regime Probabilities")
    prob_fig = VolatilityCharts.plot_regime_probabilities(prediction['regime_probabilities'])
    st.plotly_chart(prob_fig, use_container_width=True)


def render_explanation_tab():
    """Render the model explanation tab."""
    st.header("📊 Model Explanation")

    prediction = st.session_state.get('last_prediction')

    if prediction is None:
        st.info("ℹ️ Run a prediction first to see explanations.")
        return

    # Feature contribution chart
    st.subheader("Feature Contributions")
    feature_fig = VolatilityCharts.plot_feature_contribution(prediction['feature_signals'])
    st.plotly_chart(feature_fig, use_container_width=True)

    st.divider()

    # Detailed analysis
    st.subheader("Detailed Feature Analysis")
    analyzer = FeatureAnalyzer()
    analysis = analyzer.analyze_feature_signals(
        prediction['feature_signals'],
        prediction['regime_volatility'],
        prediction['regime_label']
    )

    explanation_text = analyzer.create_explanation_text(analysis)
    st.markdown(explanation_text)

    st.divider()

    # Confidence breakdown
    st.subheader("Confidence Score Breakdown")
    breakdown = prediction['confidence_breakdown']

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Regime Score", f"{breakdown['regime_score']:.1f}")
    with col2:
        st.metric("TimesFM Score", f"{breakdown['timesfm_score']:.1f}" if breakdown['timesfm_score'] else "N/A")
    with col3:
        st.metric("Feature Score", f"{breakdown['feature_score']:.1f}")


def render_validation_tab():
    """Render the validation results tab."""
    st.header("🔍 Walk-Forward Validation")

    st.info("ℹ️ Validation functionality will run the validate_volatility_mvp.py script.")

    if st.button("▶️ Run Validation", type="primary"):
        st.warning("⚠️ This feature is not yet implemented. Use scripts/validate_volatility_mvp.py directly.")

    # If validation results exist, display them
    if st.session_state.get('validation_results') is not None:
        results_df = st.session_state.validation_results

        st.subheader("Validation Metrics")
        # Display metrics here

        st.subheader("Validation Results")
        display_dataframe_with_download(
            results_df,
            "Full Validation Results",
            "validation_results.csv"
        )


def render_strategy_tab():
    """Render the trading strategy recommendations tab."""
    st.header("💰 Trading Strategy Recommendations")

    prediction = st.session_state.get('last_prediction')

    if prediction is None:
        st.info("ℹ️ Run a prediction first to see strategy recommendations.")
        return

    # Get current price from latest SPY data
    current_price = st.session_state.spy_data.iloc[-1]['close']

    # Generate strategy recommendation
    recommender = SpreadRecommender()
    strategy = recommender.recommend_spread(
        regime=prediction['regime_label'],
        current_price=current_price,
        predicted_volatility=prediction['regime_volatility'],
        confidence=prediction['confidence_score']
    )

    # Position sizing
    st.subheader("Position Sizing")
    col1, col2 = st.columns(2)

    with col1:
        account_size = st.number_input(
            "Account Size ($)",
            min_value=1000,
            max_value=1000000,
            value=DEFAULT_ACCOUNT_SIZE,
            step=1000
        )

    with col2:
        max_risk_pct = st.slider(
            "Max Risk per Trade (%)",
            min_value=0.5,
            max_value=5.0,
            value=DEFAULT_MAX_RISK_PCT * 100,
            step=0.5
        ) / 100

    sizer = PositionSizer(account_size, max_risk_pct)
    position_size = sizer.calculate_position_size(strategy, prediction['confidence_score'])

    # Display strategy
    st.subheader("Strategy Recommendation")
    strategy_output = format_strategy_output(strategy, position_size)
    st.markdown(strategy_output)

    # P&L visualization (placeholder)
    st.divider()
    st.subheader("Expected P&L Distribution")
    st.info("ℹ️ P&L visualization coming soon...")


if __name__ == "__main__":
    main()
