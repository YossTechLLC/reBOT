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
        n_iter: int = 100,
        prediction_date: Optional[pd.Timestamp] = None
    ) -> Tuple[VolatilityHMM, Dict]:
        """
        Train HMM model and store in session state.

        Args:
            df: DataFrame with features
            n_regimes: Number of volatility regimes (default: 3)
            features: List of feature column names to use (default: HMM default features)
            n_iter: Maximum training iterations (default: 100)
            prediction_date: Only train on data BEFORE this date (None = use all data).
                           This prevents data leakage when backtesting.

        Returns:
            Tuple of (trained_model, metrics_dict)
        """
        # Apply temporal cutoff if prediction_date specified
        if prediction_date is not None:
            train_df = df[df.index < prediction_date]
            logger.info(f"Temporal cutoff applied: training on {len(train_df)} rows before {prediction_date}")

            if len(train_df) < 50:
                raise ValueError(
                    f"Insufficient training data before {prediction_date}. "
                    f"Need at least 50 rows, have {len(train_df)}. "
                    f"Try loading more history or selecting a later prediction date."
                )
        else:
            train_df = df

        features_str = f", {len(features)} features" if features else ""
        logger.info(f"Training HMM with {n_regimes} regimes{features_str}, {n_iter} iterations on {len(train_df)} rows")

        # Create and train model with specified features
        hmm_model = VolatilityHMM(n_regimes=n_regimes, features=features)

        # HMM extracts features internally using self.features
        metrics = hmm_model.train(train_df, n_iter=n_iter)

        # Store in session state
        st.session_state.hmm_model = hmm_model
        st.session_state.hmm_metrics = metrics

        # Track training data for coherency checks
        st.session_state.hmm_training_data_hash = hash(train_df.values.tobytes())
        st.session_state.hmm_training_date_range = (train_df.index[0], train_df.index[-1])
        st.session_state.hmm_training_rows = len(train_df)
        st.session_state.hmm_training_cutoff = prediction_date  # Track the cutoff date
        logger.info(f"HMM training metadata: {len(train_df)} rows, {train_df.index[0]} to {train_df.index[-1]}")

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

    def predict_for_date(
        self,
        features_df: pd.DataFrame,
        prediction_date: Optional[pd.Timestamp] = None
    ) -> Dict:
        """
        Generate prediction for a specific date or the latest.

        This method supports both current (latest) prediction and historical
        backtesting. When prediction_date is specified, it uses features from
        the day BEFORE that date to predict the regime.

        Args:
            features_df: DataFrame with all features
            prediction_date: Date to predict for (None = latest/tomorrow)

        Returns:
            Dictionary with prediction results including:
            - prediction_label: Human-readable date label ("Tomorrow" or "2026-01-15")
            - is_historical: Whether this is a historical backtest
            - All standard prediction fields (regime, confidence, etc.)
        """
        if st.session_state.hmm_model is None:
            raise ValueError("HMM model not trained. Train model first.")

        # Determine which row to use for prediction
        if prediction_date is None:
            # Current behavior - use last row, predict for "tomorrow"
            target_idx = len(features_df) - 1
            target_date = features_df.index[-1]
            prediction_label = "Tomorrow"
            is_historical = False
            context_df = features_df
        else:
            # Historical prediction - find the row for day BEFORE prediction date
            # We predict using yesterday's features for the selected date
            prior_dates = features_df.index[features_df.index < prediction_date]

            if len(prior_dates) == 0:
                raise ValueError(
                    f"No data available before {prediction_date}. "
                    f"Data starts at {features_df.index[0]}. "
                    f"Select a later prediction date."
                )

            prior_date = prior_dates[-1]  # Day before prediction date
            target_idx = features_df.index.get_loc(prior_date)
            target_date = prediction_date
            prediction_label = prediction_date.strftime('%Y-%m-%d')
            is_historical = True

            # Use data only up to the prior date for HMM context
            context_df = features_df.iloc[:target_idx + 1]
            logger.info(f"Historical prediction: using context from {context_df.index[0]} to {context_df.index[-1]}")

        logger.info(f"Generating prediction for {prediction_label}")

        # Get HMM prediction on context data
        hmm_prediction = st.session_state.hmm_model.predict_latest(context_df)

        # Get TimesFM forecast (optional) - only use historical context
        timesfm_forecast = None
        if st.session_state.timesfm_forecaster and st.session_state.timesfm_forecaster.is_available():
            try:
                timesfm_forecast = st.session_state.timesfm_forecaster.predict_next_day(
                    context_df,
                    volatility_col='intraday_range_pct'
                )
                logger.info(f"TimesFM forecast: {timesfm_forecast:.4f}")
            except Exception as e:
                logger.warning(f"TimesFM forecast failed: {str(e)}")
                timesfm_forecast = None

        # Get feature signals from the target row (day before prediction)
        target_row = features_df.iloc[target_idx]
        feature_signals = {
            'overnight_gap_abs': target_row['overnight_gap_abs'],
            'vix_change_1d': target_row['vix_change_1d'],
            'vix_level': target_row['vix_level'],
            'range_expansion': target_row['range_expansion'],
            'volume_surge': target_row['volume_surge'],
            'volume_ratio': target_row['volume_ratio'],
            'high_range_days_5': target_row['high_range_days_5']
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
            'date': target_date,
            'prediction_label': prediction_label,
            'is_historical': is_historical,
            'features_date': features_df.index[target_idx],  # Date of features used
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

    def predict_latest(self, features_df: pd.DataFrame) -> Dict:
        """
        Generate prediction for the latest data point.

        This is a backward-compatible wrapper around predict_for_date().

        Args:
            features_df: DataFrame with all features

        Returns:
            Dictionary with prediction results
        """
        return self.predict_for_date(features_df, prediction_date=None)

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
        # Fix: Update the weights dictionary that calculate_score() actually uses
        # Previously this set individual attributes that were never read
        scorer.weights['regime'] = regime_weight
        scorer.weights['timesfm'] = timesfm_weight
        scorer.weights['features'] = feature_weight
        logger.info(f"Confidence weights updated: regime={regime_weight}, timesfm={timesfm_weight}, feature={feature_weight}")
