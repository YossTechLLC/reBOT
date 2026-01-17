"""
Configuration Validator Module
Validates configuration parameters using Pydantic models.
"""

import logging
from typing import Literal, List, Optional
from pydantic import BaseModel, Field, validator, root_validator

logger = logging.getLogger(__name__)


class MetaConfig(BaseModel):
    """Metadata configuration."""
    version: str
    last_updated: str
    author: str
    description: Optional[str] = None


class CommodityConfig(BaseModel):
    """Commodity selection configuration."""
    ticker: str = Field(..., description="Commodity ticker symbol")
    name: str = Field(..., description="Commodity name")

    @validator('ticker')
    def validate_ticker(cls, v):
        """Validate ticker format."""
        valid_tickers = ["GC=F", "CL=F", "SI=F", "NG=F", "HG=F", "ZC=F"]
        if v not in valid_tickers:
            logger.warning(
                f"TFM1001 CONFIG: Ticker '{v}' not in standard list. "
                f"Valid tickers: {valid_tickers}"
            )
        return v


class DataConfig(BaseModel):
    """Data acquisition and preprocessing configuration."""
    start_date: str
    end_date: str


class PreprocessingConfig(BaseModel):
    """Data preprocessing configuration."""
    missing_method: Literal["ffill", "interpolate", "drop"] = "ffill"
    outlier_method: Literal["zscore", "iqr", "isolation_forest"] = "zscore"
    outlier_threshold: float = Field(ge=2.0, le=4.0, default=3.0)
    transformation: Literal["returns", "log_returns", "diff", "none"] = "log_returns"


class FeaturesConfig(BaseModel):
    """Feature engineering configuration."""
    indicators: List[str] = Field(default_factory=list)
    fred_series: List[str] = Field(default_factory=list)
    max_lags: int = Field(ge=1, le=60, default=10)


class HMMConfig(BaseModel):
    """Hidden Markov Model configuration."""
    n_states: int = Field(ge=2, le=6, description="Number of hidden states")
    covariance_type: Literal["diag", "full", "spherical", "tied"] = "diag"
    n_iter: int = Field(ge=100, le=5000, default=1000)
    tol: float = Field(ge=1e-6, le=1e-2, default=1e-4)
    n_random_inits: int = Field(ge=5, le=50, default=10)
    random_seed: Optional[int] = 42

    @validator('n_states')
    def validate_n_states(cls, v):
        """Validate number of states."""
        if v > 4:
            logger.warning(
                f"TFM1001 CONFIG: n_states={v} is high. "
                f"Risk of overfitting. Recommended range: [2, 4]"
            )
        return v

    @validator('covariance_type')
    def validate_covariance(cls, v):
        """Validate covariance type."""
        if v == 'full':
            logger.warning(
                "TFM1001 CONFIG: 'full' covariance requires more data. "
                "Ensure you have >1000 observations."
            )
        return v


class GARCHConfig(BaseModel):
    """GARCH model configuration."""
    p: int = Field(ge=1, le=5, default=1)
    q: int = Field(ge=1, le=5, default=1)


class OptionsConfig(BaseModel):
    """Options pricing configuration."""
    maturities: List[int] = Field(default_factory=lambda: [30, 60, 90, 180, 365])
    strike_ratios: List[float] = Field(default_factory=lambda: [0.90, 0.95, 1.00, 1.05, 1.10])


class BlackScholesConfig(BaseModel):
    """Black-Scholes options pricing configuration."""
    model_type: Literal["black76", "bsm"] = "black76"
    rate_source: Literal["constant", "fred_dgs10", "curve"] = "fred_dgs10"
    constant_rate: float = Field(ge=0.0, le=0.10, default=0.035)
    convenience_yield_method: Literal["fitted", "futures_curve", "constant"] = "futures_curve"
    volatility_method: Literal["historical", "garch", "realized", "implied", "hybrid"] = "hybrid"
    hist_vol_window: int = Field(ge=20, le=252, default=60)
    garch: Optional[GARCHConfig] = None
    options: Optional[OptionsConfig] = None


class TimesFMConfig(BaseModel):
    """TimesFM integration configuration."""
    enabled: bool = False
    backend: Literal["torch", "flax"] = "torch"
    checkpoint: str = "google/timesfm-2.5-200m-pytorch"
    max_context: int = Field(ge=1, le=16384, default=1024)
    max_horizon: int = Field(ge=1, le=1024, default=256)
    normalize_inputs: bool = True
    use_continuous_quantile_head: bool = True
    force_flip_invariance: bool = True
    infer_is_positive: bool = True
    fix_quantile_crossing: bool = True
    integration_mode: Literal["ensemble", "primary", "regime_input"] = "ensemble"

    @validator('max_context')
    def validate_max_context(cls, v):
        """Validate context length."""
        if v > 2048:
            logger.info(
                f"Using large context length: {v}. "
                f"This may increase memory usage and latency."
            )
        return v


class BacktestingConfig(BaseModel):
    """Backtesting configuration."""
    initial_capital: float = Field(ge=1000.0, default=100000.0)
    transaction_cost_bps: int = Field(ge=0, le=100, default=5)
    slippage_bps: int = Field(ge=0, le=100, default=2)
    position_size_pct: float = Field(ge=0.0, le=1.0, default=0.25)


class ValidationConfig(BaseModel):
    """Validation configuration."""
    strategy: Literal["walk_forward", "expanding_window", "rolling_window"] = "walk_forward"
    train_window_days: int = Field(ge=100, default=756)
    test_window_days: int = Field(ge=10, default=63)
    step_size_days: int = Field(ge=1, default=21)
    n_splits: int = Field(ge=2, le=20, default=5)
    metrics: List[str] = Field(default_factory=list)
    backtesting: Optional[BacktestingConfig] = None


class OptunaConfig(BaseModel):
    """Optuna hyperparameter optimization configuration."""
    n_trials: int = Field(ge=10, le=1000, default=100)
    timeout_seconds: int = Field(ge=60, default=3600)
    pruner: Literal["median", "hyperband", "none"] = "median"
    sampler: Literal["tpe", "random", "grid"] = "tpe"


class OptimizationConfig(BaseModel):
    """Optimization configuration."""
    enabled: bool = True
    framework: Literal["optuna", "ray_tune"] = "optuna"
    optuna: Optional[OptunaConfig] = None


class SensitivityConfig(BaseModel):
    """Sensitivity analysis configuration."""
    method: Literal["one_at_a_time", "sobol", "morris"] = "one_at_a_time"
    parameters: List[str] = Field(default_factory=list)
    perturbation_pct: float = Field(ge=0.01, le=0.50, default=0.1)


class UIConfig(BaseModel):
    """User interface configuration."""
    theme: Literal["light", "dark"] = "dark"
    port: int = Field(ge=1024, le=65535, default=8501)
    refresh_interval: int = Field(ge=1, default=60)
    realtime_enabled: bool = False


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/system.log"


class MLflowConfig(BaseModel):
    """MLflow experiment tracking configuration."""
    tracking_uri: str = "./experiments"
    experiment_name: str = "commodity-forecasting"
    auto_log: bool = True


class SystemConfig(BaseModel):
    """Complete system configuration."""
    meta: MetaConfig
    commodity: CommodityConfig
    data: DataConfig
    features: FeaturesConfig
    hmm: HMMConfig
    black_scholes: BlackScholesConfig
    timesfm: TimesFMConfig
    validation: ValidationConfig
    optimization: OptimizationConfig
    sensitivity: SensitivityConfig
    ui: UIConfig
    logging: LoggingConfig
    mlflow: MLflowConfig

    @root_validator
    def validate_dates(cls, values):
        """Validate start_date < end_date."""
        data = values.get('data')
        if data:
            from datetime import datetime
            try:
                start = datetime.fromisoformat(data.start_date)
                end = datetime.fromisoformat(data.end_date)
                if start >= end:
                    raise ValueError(
                        f"TFM1001 CONFIG: start_date ({data.start_date}) must be before "
                        f"end_date ({data.end_date})"
                    )
            except ValueError as e:
                logger.error(f"TFM1001 CONFIG: Invalid date format: {e}")
                raise
        return values

    @root_validator
    def validate_timesfm_integration(cls, values):
        """Validate TimesFM configuration if enabled."""
        timesfm = values.get('timesfm')
        if timesfm and timesfm.enabled:
            if timesfm.backend == 'flax':
                logger.info("Using Flax backend for TimesFM. Ensure JAX is installed.")
            logger.info(f"TimesFM integration enabled with mode: {timesfm.integration_mode}")
        return values


def validate_config(config_dict: dict) -> SystemConfig:
    """
    Validate configuration dictionary using Pydantic models.

    Args:
        config_dict: Configuration dictionary from TOML

    Returns:
        Validated SystemConfig object

    Raises:
        pydantic.ValidationError: If configuration is invalid
    """
    try:
        validated_config = SystemConfig(**config_dict)
        logger.info("Configuration validation successful")
        return validated_config
    except Exception as e:
        logger.error(f"TFM1001 CONFIG: Configuration validation failed: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    from src.config.loader import load_config

    logging.basicConfig(level=logging.INFO)

    try:
        config_dict = load_config()
        validated_config = validate_config(config_dict)
        print(f"Validation successful!")
        print(f"HMM n_states: {validated_config.hmm.n_states}")
        print(f"Black-Scholes model: {validated_config.black_scholes.model_type}")
    except Exception as e:
        print(f"Validation failed: {e}")
