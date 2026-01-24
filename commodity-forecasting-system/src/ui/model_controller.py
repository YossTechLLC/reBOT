"""
Model Controller for Volatility Prediction UI
==============================================
Manages model training, prediction, and state for the Streamlit UI.

Components:
- HMM model training and loading
- TimesFM forecaster management
- Confidence scoring
- Model state management via Streamlit session state

Author: Claude + User
Date: 2026-01-17
"""

import sys
import os
import logging
from typing import Dict, Tuple, Optional, List
import pandas as pd
import numpy as np
import streamlit as st

# Add src to path for imports
# Use realpath to handle symlinks correctly
current_file = os.path.realpath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
sys.path.insert(0, os.path.join(project_root, 'src'))

from models.hmm_volatility import VolatilityHMM
from models.timesfm_volatility import TimesFMVolatilityForecaster
from volatility.confidence_scorer import VolatilityConfidenceScorer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelController:
    """
    Controls model lifecycle and state management for the UI.

    Uses Streamlit session state to persist models across interactions.
    """

    def __init__(self):
        """Initialize ModelController and set up session state."""
        # Initialize session state for models if not exists OR is None
        # Using .get() with None check handles both missing keys and None values
        if st.session_state.get('hmm_model') is None:
            st.session_state.hmm_model = None
        if st.session_state.get('hmm_metrics') is None:
            st.session_state.hmm_metrics = None
        if st.session_state.get('timesfm_forecaster') is None:
            st.session_state.timesfm_forecaster = None
        if st.session_state.get('confidence_scorer') is None:
            st.session_state.confidence_scorer = VolatilityConfidenceScorer()

        logger.info("ModelController initialized")

    def train_hmm(
        self,
        df: pd.DataFrame,
        n_regimes: int = 3,
        features: List[str] = None,
        n_iter: int = 100
    ) -> Tuple[VolatilityHMM, Dict]:
        """
        Train HMM model and store in session state.

        Args:
            df: DataFrame with features
            n_regimes: Number of volatility regimes (default: 3)
            features: List of feature column names to use (default: HMM default features)
            n_iter: Maximum training iterations (default: 100)

        Returns:
            Tuple of (trained_model, metrics_dict)
        """
        logger.info(f"Training HMM with {n_regimes} regimes, {n_iter} iterations")

        # Create and train model
        hmm_model = VolatilityHMM(n_regimes=n_regimes)

        # HMM extracts its required features internally - always use full DataFrame
        metrics = hmm_model.train(df, n_iter=n_iter)

        # Store in session state
        st.session_state.hmm_model = hmm_model
        st.session_state.hmm_metrics = metrics

        logger.info(f"HMM training complete. Converged: {metrics['converged']}")
        return hmm_model, metrics

    def load_hmm(self, model_path: str) -> VolatilityHMM:
        """
        Load pre-trained HMM model from file.

        Args:
            model_path: Path to saved model file

        Returns:
            Loaded VolatilityHMM model
        """
        logger.info(f"Loading HMM model from {model_path}")
        hmm_model = VolatilityHMM()
        hmm_model.load(model_path)

        # Store in session state
        st.session_state.hmm_model = hmm_model

        logger.info("HMM model loaded successfully")
        return hmm_model

    @st.cache_resource
    def load_timesfm(_self, checkpoint: str = None, device: str = 'cpu') -> TimesFMVolatilityForecaster:
        """
        Load TimesFM forecaster with caching.

        Args:
            checkpoint: Path to TimesFM checkpoint (default: auto-download from HuggingFace)
            device: Device to load model on ('cpu' or 'cuda')

        Returns:
            TimesFMVolatilityForecaster instance
        """
        logger.info(f"Loading TimesFM forecaster on {device}")
        try:
            forecaster = TimesFMVolatilityForecaster(checkpoint=checkpoint, device=device)
            st.session_state.timesfm_forecaster = forecaster
            logger.info("TimesFM loaded successfully")
            return forecaster
        except Exception as e:
            logger.error(f"Failed to load TimesFM: {str(e)}")
            st.session_state.timesfm_forecaster = None
            raise

    def get_hmm_status(self) -> Dict:
        """
        Get current HMM model status.

        Returns:
            Dictionary with model status information
        """
        if st.session_state.hmm_model is None:
            return {
                'loaded': False,
                'message': 'HMM model not loaded'
            }

        metrics = st.session_state.hmm_metrics or {}
        return {
            'loaded': True,
            'n_regimes': st.session_state.hmm_model.n_regimes,
            'converged': metrics.get('converged', False),
            'log_likelihood': metrics.get('log_likelihood', None),
            'regime_labels': st.session_state.hmm_model.regime_labels,
            'regime_volatilities': st.session_state.hmm_model.regime_volatilities
        }

    def get_timesfm_status(self) -> Dict:
        """
        Get current TimesFM forecaster status.

        Returns:
            Dictionary with forecaster status information
        """
        if st.session_state.timesfm_forecaster is None:
            return {
                'loaded': False,
                'available': False,
                'message': 'TimesFM not loaded'
            }

        is_available = st.session_state.timesfm_forecaster.is_available()
        return {
            'loaded': True,
            'available': is_available,
            'message': 'TimesFM ready' if is_available else 'TimesFM loaded but checkpoint unavailable'
        }

    def predict_latest(self, features_df: pd.DataFrame) -> Dict:
        """
        Generate prediction for the latest data point.

        Args:
            features_df: DataFrame with all features

        Returns:
            Dictionary with prediction results
        """
        if st.session_state.hmm_model is None:
            raise ValueError("HMM model not trained. Train model first.")

        logger.info("Generating prediction for latest data")

        # Get HMM prediction
        hmm_prediction = st.session_state.hmm_model.predict_latest(features_df)

        # Get TimesFM forecast (optional)
        timesfm_forecast = None
        if st.session_state.timesfm_forecaster and st.session_state.timesfm_forecaster.is_available():
            try:
                timesfm_forecast = st.session_state.timesfm_forecaster.predict_next_day(
                    features_df,
                    volatility_col='intraday_range_pct'
                )
                logger.info(f"TimesFM forecast: {timesfm_forecast:.4f}")
            except Exception as e:
                logger.warning(f"TimesFM forecast failed: {str(e)}")
                timesfm_forecast = None

        # Get latest feature signals
        latest = features_df.iloc[-1]
        feature_signals = {
            'overnight_gap_abs': latest['overnight_gap_abs'],
            'vix_change_1d': latest['vix_change_1d'],
            'vix_level': latest['vix_level'],
            'range_expansion': latest['range_expansion'],
            'volume_surge': latest['volume_surge'],
            'volume_ratio': latest['volume_ratio'],
            'high_range_days_5': latest['high_range_days_5']
        }

        # Calculate confidence score
        scorer = st.session_state.confidence_scorer
        score = scorer.calculate_score(
            regime_volatility=hmm_prediction['expected_volatility'],
            regime_label=hmm_prediction['regime_label'],
            timesfm_forecast=timesfm_forecast,
            feature_signals=feature_signals
        )

        # Combine all results
        return {
            'date': features_df.index[-1],
            'regime_label': hmm_prediction['regime_label'],
            'regime_volatility': hmm_prediction['expected_volatility'],
            'regime_probabilities': hmm_prediction['regime_probabilities'],
            'regime_confidence': hmm_prediction['confidence'],
            'timesfm_forecast': timesfm_forecast,
            'confidence_score': score.total_score,
            'confidence_breakdown': {
                'regime_score': score.regime_score,
                'timesfm_score': score.timesfm_score,
                'feature_score': score.feature_score
            },
            'should_trade': score.total_score >= scorer.threshold,
            'recommendation': score.recommendation,
            'explanation': score.explanation,
            'feature_signals': feature_signals
        }

    def set_confidence_threshold(self, threshold: float):
        """
        Update confidence threshold for trading decisions.

        Args:
            threshold: New threshold value (0-100)
        """
        # Ensure confidence scorer exists (defensive check)
        if st.session_state.get('confidence_scorer') is None:
            st.session_state.confidence_scorer = VolatilityConfidenceScorer()
            logger.warning("confidence_scorer was None, created new instance")

        st.session_state.confidence_scorer.threshold = threshold
        logger.info(f"Confidence threshold updated to {threshold}")

    def set_confidence_weights(self, regime_weight: float, timesfm_weight: float, feature_weight: float):
        """
        Update weights for confidence score components.

        Args:
            regime_weight: Weight for regime score (0-1)
            timesfm_weight: Weight for TimesFM score (0-1)
            feature_weight: Weight for feature score (0-1)

        Note: Weights should sum to 1.0
        """
        # Ensure confidence scorer exists (defensive check)
        if st.session_state.get('confidence_scorer') is None:
            st.session_state.confidence_scorer = VolatilityConfidenceScorer()
            logger.warning("confidence_scorer was None, created new instance")

        scorer = st.session_state.confidence_scorer
        scorer.regime_weight = regime_weight
        scorer.timesfm_weight = timesfm_weight
        scorer.feature_weight = feature_weight
        logger.info(f"Confidence weights updated: regime={regime_weight}, timesfm={timesfm_weight}, feature={feature_weight}")
