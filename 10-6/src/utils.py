"""
Utility functions for the property scraper application.
Includes logging setup, retry logic, and helper functions.
"""

import logging
import time
import functools
from pathlib import Path
from typing import Callable, Any, Optional
from datetime import datetime


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: str = "INFO",
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    date_format: str = "%Y-%m-%d %H:%M:%S"
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.

    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log message format
        date_format: Date format for log messages

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Create formatter
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if log_file is provided)
    if log_file:
        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def retry_on_exception(
    max_retries: int = 3,
    delay: int = 5,
    exceptions: tuple = (Exception,),
    backoff: bool = True
) -> Callable:
    """
    Decorator to retry a function on exception.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        exceptions: Tuple of exceptions to catch
        backoff: Whether to use exponential backoff

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    if attempt == max_retries:
                        # Last attempt failed, raise the exception
                        raise

                    # Log the retry attempt
                    logger = logging.getLogger(func.__module__)
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {current_delay} seconds..."
                    )

                    # Wait before retrying
                    time.sleep(current_delay)

                    # Apply exponential backoff if enabled
                    if backoff:
                        current_delay *= 2

        return wrapper
    return decorator


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize a string to be used as a filename.

    Args:
        filename: String to sanitize
        max_length: Maximum filename length

    Returns:
        Sanitized filename string
    """
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')

    # Remove leading/trailing spaces and dots
    filename = filename.strip('. ')

    # Truncate if too long
    if len(filename) > max_length:
        filename = filename[:max_length]

    return filename


def extract_with_regex(text: str, patterns: list) -> Optional[str]:
    """
    Extract text using a list of regex patterns.
    Returns the first successful match.

    Args:
        text: Text to search in
        patterns: List of regex pattern strings

    Returns:
        Extracted text or None if no match found
    """
    import re

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Return the first captured group
            if match.groups():
                return match.group(1).strip()
            else:
                return match.group(0).strip()

    return None


def clean_text(text: str) -> str:
    """
    Clean extracted text by removing extra whitespace and special characters.

    Args:
        text: Text to clean

    Returns:
        Cleaned text
    """
    import re

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def random_delay(min_seconds: float, max_seconds: float) -> None:
    """
    Sleep for a random duration between min and max seconds.
    Useful for rate limiting and avoiding detection.

    Args:
        min_seconds: Minimum delay in seconds
        max_seconds: Maximum delay in seconds
    """
    import random

    delay = random.uniform(min_seconds, max_seconds)
    time.sleep(delay)


def format_timestamp(dt: Optional[datetime] = None, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format a datetime object as a string.

    Args:
        dt: Datetime object (defaults to current time)
        format_str: Format string for output

    Returns:
        Formatted timestamp string
    """
    if dt is None:
        dt = datetime.now()

    return dt.strftime(format_str)


def calculate_progress(current: int, total: int) -> float:
    """
    Calculate progress percentage.

    Args:
        current: Current item number
        total: Total number of items

    Returns:
        Progress percentage (0-100)
    """
    if total == 0:
        return 0.0

    return (current / total) * 100


def print_progress_bar(
    current: int,
    total: int,
    prefix: str = "Progress",
    suffix: str = "Complete",
    length: int = 50,
    fill: str = "█"
) -> None:
    """
    Print a progress bar to the console.

    Args:
        current: Current item number
        total: Total number of items
        prefix: Prefix string
        suffix: Suffix string
        length: Character length of bar
        fill: Bar fill character
    """
    percent = calculate_progress(current, total)
    filled_length = int(length * current // total)
    bar = fill * filled_length + '-' * (length - filled_length)

    print(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}', end='')

    # Print new line on completion
    if current == total:
        print()


def validate_url(url: str) -> bool:
    """
    Validate if a string is a valid URL.

    Args:
        url: URL string to validate

    Returns:
        True if valid URL, False otherwise
    """
    import re

    # Simple URL validation regex
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # or IP
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    return url_pattern.match(url) is not None


def get_file_size(file_path: Path) -> str:
    """
    Get human-readable file size.

    Args:
        file_path: Path to file

    Returns:
        File size string (e.g., "1.5 MB")
    """
    if not file_path.exists():
        return "0 B"

    size_bytes = file_path.stat().st_size

    # Convert to human readable format
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0

    return f"{size_bytes:.1f} PB"


def ensure_directory(path: Path) -> Path:
    """
    Ensure a directory exists, create if it doesn't.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


class Timer:
    """Context manager for timing code execution."""

    def __init__(self, name: str = "Operation", logger: Optional[logging.Logger] = None):
        """
        Initialize timer.

        Args:
            name: Name of the operation being timed
            logger: Logger instance (optional)
        """
        self.name = name
        self.logger = logger
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        """Start the timer."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the timer and log the duration."""
        self.end_time = time.time()
        duration = self.end_time - self.start_time

        message = f"{self.name} completed in {duration:.2f} seconds"

        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0

        end = self.end_time if self.end_time else time.time()
        return end - self.start_time
