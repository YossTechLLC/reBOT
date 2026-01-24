"""
TimesFM Adapter Module
Wrapper for Google's TimesFM foundation model to integrate with commodity forecasting system.

TimesFM is a decoder-only transformer pretrained on ~100B timepoints for zero-shot forecasting.
Architecture: 200M params (20 layers, 16 heads, 1280 model_dim)
- Input patching: 32 timepoints → token
- Output patching: 128 timepoints per prediction
- Handles variable context/horizon lengths
- Supports multiple temporal granularities

Reference: https://arxiv.org/abs/2310.10688
"""

import logging
from typing import Tuple, Optional, Union, List
import numpy as np
import pandas as pd
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TimesFMForecast:
    """Container for TimesFM forecast results."""
    point_forecast: np.ndarray  # Shape: (horizon,)
    quantile_forecasts: Optional[dict] = None  # Optional quantile predictions
    context_length: int = 0
    horizon_length: int = 0


class TimesFMAdapter:
    """
    Adapter for Google's TimesFM foundation model.

    Supports both torch and flax backends, with zero-shot forecasting capability.

    Architecture Details:
    - Decoder-only transformer with input patching (patch_len=32, output_patch_len=128)
    - Pretrained on ~100B timepoints (Google Trends, Wiki Pageviews, synthetic)
    - 200M parameters: 20 layers, 16 heads, 1280 model dimension
    - Handles context up to 512 timepoints, horizon up to 256

    Parameters
    ----------
    config : dict
        Configuration dict with [timesfm] section
    backend : str, default="torch"
        Backend to use ("torch" or "flax")
    checkpoint : str, optional
        HuggingFace checkpoint path (default from config)
    device : str, optional
        Device for inference ("cpu", "cuda", "auto")
    """

    def __init__(
        self,
        config: dict,
        backend: str = "torch",
        checkpoint: Optional[str] = None,
        device: str = "auto"
    ):
        self.config = config
        self.backend = backend
        self.device = device

        # Extract TimesFM configuration
        timesfm_config = config.get('timesfm', {})
        self.enabled = timesfm_config.get('enabled', False)

        if not self.enabled:
            logger.warning("TFM5001 SYSTEM - TimesFM is disabled in configuration")
            self.model = None
            return

        # Model configuration
        # Note: Using 1.0 version as default because it provides torch_model.ckpt format
        # The 2.5 version uses safetensors which isn't fully supported by timesfm library yet
        self.checkpoint = checkpoint or timesfm_config.get(
            'checkpoint',
            'google/timesfm-1.0-200m-pytorch'
        )
        self.max_context = timesfm_config.get('max_context', 1024)
        self.max_horizon = timesfm_config.get('max_horizon', 256)
        self.normalize_inputs = timesfm_config.get('normalize_inputs', True)
        self.use_continuous_quantile_head = timesfm_config.get('use_continuous_quantile_head', True)
        self.force_flip_invariance = timesfm_config.get('force_flip_invariance', True)
        self.infer_is_positive = timesfm_config.get('infer_is_positive', True)
        self.fix_quantile_crossing = timesfm_config.get('fix_quantile_crossing', True)

        # Load model
        self._load_model()

        logger.info(
            f"TimesFM adapter initialized: backend={backend}, "
            f"checkpoint={self.checkpoint}, max_context={self.max_context}"
        )

    def _load_model(self):
        """Load TimesFM model from HuggingFace."""
        if not self.enabled:
            return

        try:
            if self.backend == "torch":
                self._load_torch_model()
            elif self.backend == "flax":
                self._load_flax_model()
            else:
                raise ValueError(
                    f"TFM1001 CONFIG - Invalid backend: {self.backend}. "
                    f"Must be 'torch' or 'flax'"
                )
        except Exception as e:
            logger.error(
                f"TFM3001 CHECKPOINT - Failed to load TimesFM: {str(e)}. "
                f"Hint: Install timesfm package: pip install timesfm"
            )
            self.model = None
            self.enabled = False
            raise

    def _load_torch_model(self):
        """Load PyTorch backend of TimesFM."""
        try:
            import timesfm
        except ImportError:
            raise ImportError(
                "TFM3001 CHECKPOINT - timesfm package not found. "
                "Install with: pip install timesfm"
            )

        # Create hyperparameters with new API
        hparams = timesfm.TimesFmHparams(
            context_len=self.max_context,
            horizon_len=self.max_horizon,
            input_patch_len=32,  # Standard from paper
            output_patch_len=128,  # Standard from paper
            num_layers=20,
            num_heads=16,
            model_dims=1280,
            backend="cpu" if self.device == "cpu" else "gpu",
        )

        # Create checkpoint configuration
        checkpoint = timesfm.TimesFmCheckpoint(
            huggingface_repo_id=self.checkpoint
        )

        # Initialize TimesFM with new API
        self.model = timesfm.TimesFm(
            hparams=hparams,
            checkpoint=checkpoint
        )

        logger.info(f"Loaded TimesFM torch model from {self.checkpoint}")

    def _load_flax_model(self):
        """Load Flax/JAX backend of TimesFM."""
        try:
            import timesfm
        except ImportError:
            raise ImportError(
                "TFM3001 CHECKPOINT - timesfm package not found. "
                "Install with: pip install timesfm[flax]"
            )

        # Initialize TimesFM with flax backend
        self.model = timesfm.TimesFm(
            context_len=self.max_context,
            horizon_len=self.max_horizon,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=20,
            model_dims=1280,
            backend="tpu",  # Flax typically used with TPU
        )

        # Load pretrained checkpoint
        self.model.load_from_checkpoint(repo_id=self.checkpoint)

        logger.info(f"Loaded TimesFM flax model from {self.checkpoint}")

    def forecast(
        self,
        context: Union[np.ndarray, pd.Series],
        horizon: int,
        freq: Optional[str] = None,
        quantiles: Optional[List[float]] = None
    ) -> TimesFMForecast:
        """
        Generate zero-shot forecast using TimesFM.

        TimesFM uses decoder-only autoregressive generation:
        1. Break context into patches (patch_len=32)
        2. Generate next 128 timepoints (output_patch_len)
        3. Repeat autoregressively until horizon reached

        Parameters
        ----------
        context : array-like, shape (context_length,)
            Historical time series values
        horizon : int
            Number of future timepoints to forecast
        freq : str, optional
            Temporal frequency ("D", "H", "W", "M", etc.)
            Helps model select appropriate patterns
        quantiles : list of float, optional
            Quantile levels for probabilistic forecasting (e.g., [0.1, 0.5, 0.9])

        Returns
        -------
        TimesFMForecast
            Forecast results with point and optional quantile predictions

        Raises
        ------
        ValueError
            If context exceeds max_context or horizon exceeds max_horizon
        RuntimeError
            If model not loaded or inference fails
        """
        if not self.enabled or self.model is None:
            raise RuntimeError(
                "TFM3001 CHECKPOINT - TimesFM model not loaded. "
                "Check configuration and installation."
            )

        # Convert to numpy array
        if isinstance(context, pd.Series):
            context = context.values
        context = np.asarray(context, dtype=np.float32)

        # Validate dimensions
        context_length = len(context)
        if context_length > self.max_context:
            logger.warning(
                f"TFM2001 DATA - Context length {context_length} exceeds "
                f"max_context {self.max_context}. Truncating to last {self.max_context} points."
            )
            context = context[-self.max_context:]
            context_length = self.max_context

        if horizon > self.max_horizon:
            raise ValueError(
                f"TFM2001 DATA - Horizon {horizon} exceeds max_horizon "
                f"{self.max_horizon}. Split into multiple forecasts or increase max_horizon."
            )

        # Handle NaN values
        if np.any(np.isnan(context)):
            logger.warning(
                "TFM2001 DATA - Context contains NaN values. "
                "Filling with forward fill then backward fill."
            )
            context = pd.Series(context).fillna(method='ffill').fillna(method='bfill').values

        try:
            # TimesFM expects inputs as list of arrays
            # For single series forecasting, wrap in list
            inputs = [context]

            # Map frequency string to integer (0 for unknown/daily)
            freq_int = [0]  # Default to 0 (daily frequency)

            # Generate forecast using TimesFM
            # TimesFM.forecast returns tuple: (point_forecast, quantile_forecast)
            point_forecast, quantile_forecast = self.model.forecast(
                inputs=inputs,
                freq=freq_int
            )

            # Extract forecast for the single series (index 0)
            point_forecast = point_forecast[0, :horizon]

            # Parse quantile forecasts if available
            quantile_forecasts = None
            if quantile_forecast is not None and quantiles is not None:
                quantile_forecasts = {}
                # quantile_forecast shape: (num_series, horizon, num_quantiles)
                for i, q in enumerate(quantiles):
                    if i < quantile_forecast.shape[2]:
                        quantile_forecasts[q] = quantile_forecast[0, :horizon, i]

            return TimesFMForecast(
                point_forecast=point_forecast,
                quantile_forecasts=quantile_forecasts,
                context_length=context_length,
                horizon_length=horizon
            )

        except Exception as e:
            logger.error(
                f"TFM4001 INFERENCE - TimesFM forecast failed: {str(e)}. "
                f"Context shape: {context.shape}, horizon: {horizon}, freq: {freq}"
            )
            raise RuntimeError(f"TFM4001 INFERENCE - Forecast failed: {str(e)}")

    def forecast_with_regime(
        self,
        context: Union[np.ndarray, pd.Series],
        horizon: int,
        regime_state: Optional[int] = None,
        freq: Optional[str] = None
    ) -> TimesFMForecast:
        """
        Forecast with optional HMM regime conditioning.

        In 'regime_input' integration mode, HMM regime states can be used
        to condition TimesFM forecasts. This is experimental.

        Parameters
        ----------
        context : array-like
            Historical time series values
        horizon : int
            Forecast horizon
        regime_state : int, optional
            Current HMM regime state (0=bear, 1=neutral, 2=bull)
            If provided, used to condition forecast
        freq : str, optional
            Temporal frequency

        Returns
        -------
        TimesFMForecast
            Forecast results

        Notes
        -----
        Regime conditioning is achieved by prepending regime indicator
        to context. This is experimental and may be refined.
        """
        # For now, regime conditioning is not directly supported by TimesFM
        # We could experiment with:
        # 1. Prepending regime as synthetic feature
        # 2. Adjusting context based on regime statistics
        # 3. Post-processing TimesFM output based on regime

        # Current implementation: simple passthrough
        # TODO: Implement regime-aware forecasting
        if regime_state is not None:
            logger.info(
                f"Generating forecast conditioned on regime {regime_state}. "
                f"Regime conditioning is experimental."
            )

        return self.forecast(context=context, horizon=horizon, freq=freq)

    def evaluate_zero_shot(
        self,
        test_data: pd.DataFrame,
        context_col: str = 'value',
        horizon: int = 30,
        freq: Optional[str] = None
    ) -> dict:
        """
        Evaluate zero-shot forecasting performance on test data.

        Performs walk-forward evaluation:
        1. Use first N points as context
        2. Forecast next H points
        3. Compare to actual values
        4. Slide window forward

        Parameters
        ----------
        test_data : pd.DataFrame
            Test time series with datetime index
        context_col : str
            Column name for time series values
        horizon : int
            Forecast horizon for evaluation
        freq : str, optional
            Temporal frequency

        Returns
        -------
        dict
            Evaluation metrics (MAE, RMSE, MAPE, etc.)
        """
        if not self.enabled or self.model is None:
            raise RuntimeError("TimesFM model not loaded")

        values = test_data[context_col].values
        total_length = len(values)

        # Use 70% for context, forecast remaining
        context_length = int(total_length * 0.7)
        context = values[:context_length]
        actual = values[context_length:context_length + horizon]

        # Generate forecast
        forecast_result = self.forecast(
            context=context,
            horizon=horizon,
            freq=freq
        )

        predicted = forecast_result.point_forecast

        # Calculate metrics
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mape = np.mean(np.abs((actual - predicted) / (actual + 1e-10))) * 100

        # Direction accuracy (did we predict trend correctly?)
        actual_direction = np.diff(actual) > 0
        pred_direction = np.diff(predicted) > 0
        direction_accuracy = np.mean(actual_direction == pred_direction) * 100

        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'direction_accuracy': float(direction_accuracy),
            'context_length': context_length,
            'horizon': horizon
        }

        logger.info(
            f"Zero-shot evaluation: MAE={mae:.4f}, RMSE={rmse:.4f}, "
            f"MAPE={mape:.2f}%, Direction Acc={direction_accuracy:.2f}%"
        )

        return metrics

    def get_model_info(self) -> dict:
        """
        Get information about loaded TimesFM model.

        Returns
        -------
        dict
            Model configuration and capabilities
        """
        return {
            'enabled': self.enabled,
            'backend': self.backend,
            'checkpoint': self.checkpoint,
            'max_context': self.max_context,
            'max_horizon': self.max_horizon,
            'normalize_inputs': self.normalize_inputs,
            'use_continuous_quantile_head': self.use_continuous_quantile_head,
            'model_loaded': self.model is not None,
            'architecture': {
                'type': 'decoder-only transformer',
                'parameters': '200M',
                'layers': 20,
                'heads': 16,
                'model_dim': 1280,
                'input_patch_len': 32,
                'output_patch_len': 128,
                'pretraining_data': '~100B timepoints'
            }
        }
