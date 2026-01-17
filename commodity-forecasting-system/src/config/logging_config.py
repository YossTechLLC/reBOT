"""
Logging Configuration Module
Sets up structured logging for the commodity forecasting system.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Configure structured logging based on configuration.

    Args:
        config: Configuration dictionary containing logging settings

    Returns:
        Configured logger instance

    Example:
        >>> from src.config.loader import load_config
        >>> config = load_config()
        >>> logger = setup_logging(config)
        >>> logger.info("System initialized")
    """
    logging_config = config.get('logging', {})

    # Extract logging parameters
    level = logging_config.get('level', 'INFO')
    format_string = logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = logging_config.get('file', 'logs/system.log')

    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Create formatters
    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(getattr(logging, level.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("="*80)
    logger.info("Logging system initialized")
    logger.info(f"Log level: {level}")
    logger.info(f"Log file: {log_file}")
    logger.info("="*80)

    return logger


def log_error(logger: logging.Logger, error_code: str, message: str, exception: Exception = None):
    """
    Log error with standardized error codes (from CLAUDE.md taxonomy).

    Error Codes:
        TFM1001 CONFIG: bad config/env/flags
        TFM2001 DATA: bad shapes, missing columns, leakage risks
        TFM3001 CHECKPOINT: missing or incompatible checkpoint
        TFM4001 INFERENCE: runtime/OOM/NaN/precision issues
        TFM5001 PERF: regression or unexpected slowness

    Args:
        logger: Logger instance
        error_code: Error code from taxonomy
        message: Error message
        exception: Optional exception object

    Example:
        >>> log_error(logger, "TFM2001", "Missing price data for 2024-01-15")
    """
    full_message = f"{error_code}: {message}"

    if exception:
        logger.error(full_message, exc_info=exception)
    else:
        logger.error(full_message)


def log_performance_metric(logger: logging.Logger, metric_name: str, value: float, unit: str = ""):
    """
    Log performance metric in structured format.

    Args:
        logger: Logger instance
        metric_name: Name of the metric
        value: Metric value
        unit: Optional unit of measurement

    Example:
        >>> log_performance_metric(logger, "RMSE", 25.3, "$")
        >>> log_performance_metric(logger, "Training Time", 125.5, "seconds")
    """
    unit_str = f" {unit}" if unit else ""
    logger.info(f"METRIC: {metric_name} = {value:.4f}{unit_str}")


def log_model_info(logger: logging.Logger, model_name: str, parameters: Dict[str, Any]):
    """
    Log model configuration and parameters.

    Args:
        logger: Logger instance
        model_name: Name of the model
        parameters: Dictionary of model parameters

    Example:
        >>> log_model_info(logger, "HMM", {"n_states": 3, "covariance_type": "diag"})
    """
    logger.info(f"MODEL: {model_name}")
    for key, value in parameters.items():
        logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    # Example usage
    from src.config.loader import load_config

    config = load_config()
    logger = setup_logging(config)

    # Test logging
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")

    # Test error logging
    log_error(logger, "TFM1001", "Configuration file missing parameter: hmm.n_states")

    # Test metric logging
    log_performance_metric(logger, "RMSE", 25.345, "$")

    # Test model info logging
    log_model_info(logger, "HMM", {"n_states": 3, "covariance_type": "diag", "n_iter": 1000})
