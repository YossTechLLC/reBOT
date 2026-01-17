"""
Configuration Loader Module
Loads and parses TOML configuration files for the commodity forecasting system.
"""

import logging
from pathlib import Path
from typing import Dict, Any

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Fallback for Python <3.11

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/parameters.toml") -> Dict[str, Any]:
    """
    Load configuration from TOML file.

    Args:
        config_path: Path to TOML configuration file

    Returns:
        Dictionary containing configuration parameters

    Raises:
        FileNotFoundError: If configuration file does not exist
        tomllib.TOMLDecodeError: If TOML file is malformed
    """
    config_file = Path(config_path)

    if not config_file.exists():
        logger.error(f"TFM1001 CONFIG: Configuration file not found: {config_path}")
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}. "
            f"Please ensure the file exists at the specified location."
        )

    try:
        with open(config_file, "rb") as f:
            config = tomllib.load(f)

        logger.info(f"Successfully loaded configuration from {config_path}")
        logger.debug(f"Configuration version: {config.get('meta', {}).get('version', 'unknown')}")

        return config

    except tomllib.TOMLDecodeError as e:
        logger.error(f"TFM1001 CONFIG: Malformed TOML file: {e}")
        raise ValueError(
            f"Configuration file {config_path} is malformed. "
            f"Error: {e}. Please check TOML syntax."
        )
    except Exception as e:
        logger.error(f"TFM1001 CONFIG: Unexpected error loading configuration: {e}")
        raise


def save_config(config: Dict[str, Any], config_path: str = "config/parameters.toml") -> None:
    """
    Save configuration to TOML file.

    Args:
        config: Configuration dictionary
        config_path: Path to save TOML configuration file

    Raises:
        ImportError: If tomli_w is not installed
    """
    try:
        import tomli_w
    except ImportError:
        logger.error("TFM1001 CONFIG: tomli_w not installed. Cannot save configuration.")
        raise ImportError(
            "tomli_w is required to save configuration files. "
            "Install with: pip install tomli-w"
        )

    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(config_file, "wb") as f:
            tomli_w.dump(config, f)

        logger.info(f"Successfully saved configuration to {config_path}")

    except Exception as e:
        logger.error(f"TFM1001 CONFIG: Error saving configuration: {e}")
        raise


def get_nested_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get nested configuration value using dot notation.

    Args:
        config: Configuration dictionary
        key_path: Dot-separated path (e.g., "hmm.n_states")
        default: Default value if key not found

    Returns:
        Configuration value or default

    Example:
        >>> config = {"hmm": {"n_states": 3}}
        >>> get_nested_value(config, "hmm.n_states")
        3
    """
    keys = key_path.split('.')
    value = config

    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        logger.warning(
            f"TFM1001 CONFIG: Key path '{key_path}' not found in configuration. "
            f"Using default: {default}"
        )
        return default


def set_nested_value(config: Dict[str, Any], key_path: str, value: Any) -> None:
    """
    Set nested configuration value using dot notation.

    Args:
        config: Configuration dictionary
        key_path: Dot-separated path (e.g., "hmm.n_states")
        value: Value to set

    Example:
        >>> config = {"hmm": {"n_states": 3}}
        >>> set_nested_value(config, "hmm.n_states", 4)
        >>> config["hmm"]["n_states"]
        4
    """
    keys = key_path.split('.')
    current = config

    # Navigate to parent dictionary
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    # Set value
    current[keys[-1]] = value
    logger.debug(f"Set configuration value: {key_path} = {value}")


def validate_config_keys(config: Dict[str, Any], required_keys: list) -> bool:
    """
    Validate that required keys exist in configuration.

    Args:
        config: Configuration dictionary
        required_keys: List of required key paths (dot notation)

    Returns:
        True if all required keys exist

    Raises:
        ValueError: If any required key is missing
    """
    missing_keys = []

    for key_path in required_keys:
        value = get_nested_value(config, key_path, default=None)
        if value is None:
            missing_keys.append(key_path)

    if missing_keys:
        logger.error(
            f"TFM1001 CONFIG: Missing required configuration keys: {missing_keys}"
        )
        raise ValueError(
            f"Configuration validation failed. Missing required keys: {missing_keys}. "
            f"Please ensure all required keys are present in parameters.toml"
        )

    logger.info("Configuration validation passed: all required keys present")
    return True


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    try:
        config = load_config()
        print(f"Loaded configuration version: {config['meta']['version']}")
        print(f"HMM n_states: {get_nested_value(config, 'hmm.n_states')}")
        print(f"Commodity ticker: {get_nested_value(config, 'commodity.ticker')}")
    except Exception as e:
        print(f"Error: {e}")
