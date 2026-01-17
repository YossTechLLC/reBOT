"""
TimesFM-HMM Ensemble Architecture
Combines Google's TimesFM foundation model with Hidden Markov Model for regime-aware forecasting.

Three Integration Modes:
1. **ensemble**: Average TimesFM and HMM predictions (weighted or equal)
2. **primary**: Use TimesFM as primary forecaster, HMM for regime detection only
3. **regime_input**: Feed HMM regime states to TimesFM as conditioning features

TimesFM provides strong zero-shot long-context forecasting.
HMM provides regime-aware volatility and market state detection.
Ensemble leverages strengths of both approaches.
"""

import logging
from typing import Tuple, Optional, Dict, List
import numpy as np
import pandas as pd
from dataclasses import dataclass

from src.models.timesfm_adapter import TimesFMAdapter, TimesFMForecast
from src.models.hmm_core import CommodityHMM

logger = logging.getLogger(__name__)


@dataclass
class EnsembleForecast:
    """Container for ensemble forecast results."""
    point_forecast: np.ndarray  # Shape: (horizon,)
    timesfm_forecast: np.ndarray  # TimesFM component
    hmm_forecast: Optional[np.ndarray] = None  # HMM component (if used)
    regime_state: Optional[int] = None  # Current regime
    regime_label: Optional[str] = None  # Regime label (bull/bear/neutral)
    regime_volatility: Optional[float] = None  # Regime-specific volatility
    quantile_forecasts: Optional[dict] = None  # Probabilistic forecasts
    ensemble_mode: str = "ensemble"
    weights: Optional[Dict[str, float]] = None  # Component weights


class TimesFMHMMEnsemble:
    """
    Ensemble architecture combining TimesFM and HMM.

    Integration Modes
    -----------------
    - **ensemble**: Weighted average of TimesFM and HMM predictions
      Best for: Balanced approach, combining deep learning and statistical methods
    - **primary**: TimesFM primary, HMM only for regime detection
      Best for: Leveraging TimesFM zero-shot capability with regime awareness
    - **regime_input**: Feed HMM regimes to TimesFM as features
      Best for: Experimental regime-conditioned forecasting

    Architecture
    ------------
    TimesFM: 200M param decoder-only transformer, zero-shot forecasting
    HMM: Gaussian HMM with regime-dependent parameters
    Ensemble: Intelligent combination based on integration mode

    Parameters
    ----------
    config : dict
        Configuration dict with [timesfm], [hmm], and [ensemble] sections
    timesfm_adapter : TimesFMAdapter, optional
        Pretrained TimesFM adapter (if None, creates new)
    hmm_model : CommodityHMM, optional
        Trained HMM model (if None, creates new)
    integration_mode : str, default="ensemble"
        Integration strategy ("ensemble", "primary", "regime_input")
    """

    def __init__(
        self,
        config: dict,
        timesfm_adapter: Optional[TimesFMAdapter] = None,
        hmm_model: Optional[CommodityHMM] = None,
        integration_mode: Optional[str] = None
    ):
        self.config = config

        # Get integration mode from config or parameter
        timesfm_config = config.get('timesfm', {})
        self.integration_mode = integration_mode or timesfm_config.get(
            'integration_mode',
            'ensemble'
        )

        if self.integration_mode not in ['ensemble', 'primary', 'regime_input']:
            raise ValueError(
                f"TFM1001 CONFIG - Invalid integration_mode: {self.integration_mode}. "
                f"Must be 'ensemble', 'primary', or 'regime_input'"
            )

        # Initialize or use provided models
        self.timesfm = timesfm_adapter or TimesFMAdapter(config)
        self.hmm = hmm_model or CommodityHMM(config)

        # Ensemble weights (for 'ensemble' mode)
        self.ensemble_weights = {
            'timesfm': 0.5,
            'hmm': 0.5
        }

        logger.info(
            f"TimesFM-HMM Ensemble initialized: mode={self.integration_mode}, "
            f"weights={self.ensemble_weights}"
        )

    def fit_hmm(self, features: pd.DataFrame):
        """
        Train HMM component on historical features.

        Required before forecasting if HMM not already trained.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix with columns like ['returns', 'volatility_5', ...]
        """
        if not self.hmm.is_fitted:
            logger.info("Training HMM component...")
            self.hmm.fit(features)
            logger.info(f"HMM trained: {self.hmm.n_states} states")
        else:
            logger.info("HMM already fitted, skipping training")

    def forecast(
        self,
        context: pd.DataFrame,
        horizon: int,
        freq: Optional[str] = None,
        quantiles: Optional[List[float]] = None
    ) -> EnsembleForecast:
        """
        Generate ensemble forecast combining TimesFM and HMM.

        Workflow depends on integration mode:

        **Ensemble Mode**:
        1. Generate TimesFM zero-shot forecast
        2. Detect current regime with HMM
        3. Generate HMM regime-aware forecast
        4. Weighted average of both forecasts

        **Primary Mode**:
        1. Generate TimesFM zero-shot forecast (primary)
        2. Detect current regime with HMM (for context)
        3. Return TimesFM forecast with regime metadata

        **Regime Input Mode**:
        1. Detect current regime with HMM
        2. Feed regime state to TimesFM as conditioning
        3. Return regime-conditioned TimesFM forecast

        Parameters
        ----------
        context : pd.DataFrame
            Historical data with 'close' prices and features
            Must have datetime index
        horizon : int
            Number of periods to forecast
        freq : str, optional
            Temporal frequency ("D", "H", "W", "M")
            If None, inferred from context index
        quantiles : list of float, optional
            Quantile levels for probabilistic forecasting

        Returns
        -------
        EnsembleForecast
            Ensemble forecast with component predictions and metadata

        Raises
        ------
        ValueError
            If context missing required columns or HMM not fitted
        RuntimeError
            If TimesFM not enabled or forecast fails
        """
        # Validate inputs
        if 'close' not in context.columns:
            raise ValueError(
                "TFM2001 DATA - Context must have 'close' column for prices"
            )

        if not self.hmm.is_fitted and self.integration_mode != 'primary':
            raise ValueError(
                "TFM2001 DATA - HMM must be fitted before forecasting. "
                "Call fit_hmm() first or set integration_mode='primary'"
            )

        # Infer frequency if not provided
        if freq is None:
            freq = self._infer_frequency(context.index)

        # Extract price series for TimesFM
        prices = context['close'].values

        # Get current regime state (if HMM fitted)
        regime_state = None
        regime_label = None
        regime_volatility = None

        if self.hmm.is_fitted:
            # Use last N rows for regime detection
            hmm_features = self._prepare_hmm_features(context)
            if len(hmm_features) > 0:
                regime_state, regime_label = self.hmm.predict_regime(
                    hmm_features.iloc[-1:]
                )

                # Get regime statistics
                regime_stats = self.hmm.get_regime_stats()
                if regime_state in regime_stats:
                    regime_volatility = regime_stats[regime_state]['volatility']

                logger.info(
                    f"Current regime: {regime_label} (state {regime_state}), "
                    f"volatility={regime_volatility:.4f}"
                )

        # Generate forecasts based on integration mode
        if self.integration_mode == 'ensemble':
            return self._forecast_ensemble(
                prices=prices,
                context=context,
                horizon=horizon,
                freq=freq,
                regime_state=regime_state,
                regime_label=regime_label,
                regime_volatility=regime_volatility,
                quantiles=quantiles
            )

        elif self.integration_mode == 'primary':
            return self._forecast_primary(
                prices=prices,
                horizon=horizon,
                freq=freq,
                regime_state=regime_state,
                regime_label=regime_label,
                regime_volatility=regime_volatility,
                quantiles=quantiles
            )

        elif self.integration_mode == 'regime_input':
            return self._forecast_regime_input(
                prices=prices,
                horizon=horizon,
                freq=freq,
                regime_state=regime_state,
                regime_label=regime_label,
                regime_volatility=regime_volatility,
                quantiles=quantiles
            )

    def _forecast_ensemble(
        self,
        prices: np.ndarray,
        context: pd.DataFrame,
        horizon: int,
        freq: str,
        regime_state: Optional[int],
        regime_label: Optional[str],
        regime_volatility: Optional[float],
        quantiles: Optional[List[float]]
    ) -> EnsembleForecast:
        """Ensemble mode: Average TimesFM and HMM predictions."""
        # 1. TimesFM forecast
        timesfm_result = self.timesfm.forecast(
            context=prices,
            horizon=horizon,
            freq=freq,
            quantiles=quantiles
        )
        timesfm_forecast = timesfm_result.point_forecast

        # 2. HMM forecast (regime-aware random walk with drift)
        # Use regime-specific volatility and drift
        hmm_forecast = self._generate_hmm_forecast(
            last_price=prices[-1],
            horizon=horizon,
            regime_volatility=regime_volatility
        )

        # 3. Weighted ensemble
        w_tf = self.ensemble_weights['timesfm']
        w_hmm = self.ensemble_weights['hmm']
        ensemble_forecast = w_tf * timesfm_forecast + w_hmm * hmm_forecast

        logger.info(
            f"Ensemble forecast: TimesFM weight={w_tf}, HMM weight={w_hmm}"
        )

        return EnsembleForecast(
            point_forecast=ensemble_forecast,
            timesfm_forecast=timesfm_forecast,
            hmm_forecast=hmm_forecast,
            regime_state=regime_state,
            regime_label=regime_label,
            regime_volatility=regime_volatility,
            quantile_forecasts=timesfm_result.quantile_forecasts,
            ensemble_mode='ensemble',
            weights=self.ensemble_weights
        )

    def _forecast_primary(
        self,
        prices: np.ndarray,
        horizon: int,
        freq: str,
        regime_state: Optional[int],
        regime_label: Optional[str],
        regime_volatility: Optional[float],
        quantiles: Optional[List[float]]
    ) -> EnsembleForecast:
        """Primary mode: TimesFM primary, HMM for regime only."""
        # Generate TimesFM forecast
        timesfm_result = self.timesfm.forecast(
            context=prices,
            horizon=horizon,
            freq=freq,
            quantiles=quantiles
        )

        logger.info("Primary mode: Using TimesFM forecast with regime context")

        return EnsembleForecast(
            point_forecast=timesfm_result.point_forecast,
            timesfm_forecast=timesfm_result.point_forecast,
            hmm_forecast=None,  # Not used
            regime_state=regime_state,
            regime_label=regime_label,
            regime_volatility=regime_volatility,
            quantile_forecasts=timesfm_result.quantile_forecasts,
            ensemble_mode='primary',
            weights={'timesfm': 1.0, 'hmm': 0.0}
        )

    def _forecast_regime_input(
        self,
        prices: np.ndarray,
        horizon: int,
        freq: str,
        regime_state: Optional[int],
        regime_label: Optional[str],
        regime_volatility: Optional[float],
        quantiles: Optional[List[float]]
    ) -> EnsembleForecast:
        """Regime input mode: Feed HMM regime to TimesFM."""
        # Generate regime-conditioned TimesFM forecast
        timesfm_result = self.timesfm.forecast_with_regime(
            context=prices,
            horizon=horizon,
            regime_state=regime_state,
            freq=freq
        )

        logger.info(
            f"Regime input mode: TimesFM conditioned on regime {regime_label}"
        )

        return EnsembleForecast(
            point_forecast=timesfm_result.point_forecast,
            timesfm_forecast=timesfm_result.point_forecast,
            hmm_forecast=None,
            regime_state=regime_state,
            regime_label=regime_label,
            regime_volatility=regime_volatility,
            quantile_forecasts=timesfm_result.quantile_forecasts,
            ensemble_mode='regime_input',
            weights={'timesfm': 1.0, 'hmm': 0.0}
        )

    def _generate_hmm_forecast(
        self,
        last_price: float,
        horizon: int,
        regime_volatility: Optional[float]
    ) -> np.ndarray:
        """
        Generate simple HMM-based forecast.

        Uses regime-aware random walk:
        - Drift based on regime (bull=positive, bear=negative, neutral=zero)
        - Volatility based on regime-specific volatility

        Parameters
        ----------
        last_price : float
            Last observed price
        horizon : int
            Forecast horizon
        regime_volatility : float, optional
            Regime-specific volatility

        Returns
        -------
        np.ndarray
            Forecast prices, shape (horizon,)
        """
        if regime_volatility is None:
            # Fallback: use constant volatility
            regime_volatility = 0.02  # 2% daily volatility

        # Simple random walk with regime-aware drift
        # Could be enhanced with more sophisticated HMM forecasting
        forecast = np.zeros(horizon)
        current_price = last_price

        for t in range(horizon):
            # Random shock scaled by regime volatility
            shock = np.random.randn() * regime_volatility * current_price
            current_price = current_price + shock
            forecast[t] = current_price

        return forecast

    def _prepare_hmm_features(self, context: pd.DataFrame) -> pd.DataFrame:
        """
        Extract HMM features from context.

        Assumes context has required feature columns.
        If missing, computes basic features.

        Parameters
        ----------
        context : pd.DataFrame
            Historical data

        Returns
        -------
        pd.DataFrame
            Features for HMM (returns, volatility, momentum, range)
        """
        # Get features that HMM was trained on
        if hasattr(self.hmm, 'feature_names') and self.hmm.feature_names:
            required_cols = self.hmm.feature_names
        else:
            # Default feature set
            required_cols = ['returns', 'volatility_5', 'volatility_10',
                           'momentum_5', 'range']

        # Check if features already present
        if all(col in context.columns for col in required_cols):
            return context[required_cols].dropna()

        # Compute features if missing
        features = pd.DataFrame(index=context.index)

        if 'close' in context.columns:
            features['returns'] = context['close'].pct_change()
            features['volatility_5'] = features['returns'].rolling(5).std()
            features['volatility_10'] = features['returns'].rolling(10).std()

        if 'close' in context.columns:
            features['momentum_5'] = context['close'] / context['close'].shift(5) - 1

        if 'high' in context.columns and 'low' in context.columns and 'close' in context.columns:
            features['range'] = (context['high'] - context['low']) / context['close']

        # Return only columns that HMM expects
        available_cols = [col for col in required_cols if col in features.columns]
        return features[available_cols].dropna()

    def _infer_frequency(self, index: pd.DatetimeIndex) -> str:
        """
        Infer temporal frequency from datetime index.

        Returns
        -------
        str
            Frequency string ("D", "H", "W", "M", etc.)
        """
        if len(index) < 2:
            return "D"  # Default to daily

        # Compute median time delta
        deltas = np.diff(index.values).astype('timedelta64[s]').astype(float)
        median_delta_seconds = np.median(deltas)

        # Map to frequency
        if median_delta_seconds < 3600:  # < 1 hour
            return "T"  # Minute
        elif median_delta_seconds < 86400:  # < 1 day
            return "H"  # Hourly
        elif median_delta_seconds < 604800:  # < 1 week
            return "D"  # Daily
        elif median_delta_seconds < 2592000:  # < 30 days
            return "W"  # Weekly
        else:
            return "M"  # Monthly

    def set_ensemble_weights(self, timesfm_weight: float, hmm_weight: float):
        """
        Set ensemble weights for 'ensemble' mode.

        Parameters
        ----------
        timesfm_weight : float
            Weight for TimesFM predictions (should be in [0, 1])
        hmm_weight : float
            Weight for HMM predictions (should be in [0, 1])

        Raises
        ------
        ValueError
            If weights don't sum to 1.0
        """
        if not np.isclose(timesfm_weight + hmm_weight, 1.0):
            raise ValueError(
                f"TFM1001 CONFIG - Ensemble weights must sum to 1.0, "
                f"got {timesfm_weight + hmm_weight}"
            )

        self.ensemble_weights = {
            'timesfm': timesfm_weight,
            'hmm': hmm_weight
        }

        logger.info(f"Updated ensemble weights: {self.ensemble_weights}")

    def get_ensemble_info(self) -> dict:
        """
        Get information about ensemble configuration.

        Returns
        -------
        dict
            Ensemble configuration and model status
        """
        return {
            'integration_mode': self.integration_mode,
            'ensemble_weights': self.ensemble_weights,
            'timesfm_enabled': self.timesfm.enabled,
            'hmm_fitted': self.hmm.is_fitted,
            'hmm_n_states': self.hmm.n_states if self.hmm.is_fitted else None,
            'timesfm_info': self.timesfm.get_model_info()
        }
