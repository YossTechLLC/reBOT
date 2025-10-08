"""
Session management module for browser cookie persistence.
Handles saving and loading browser cookies to maintain authentication sessions.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from selenium import webdriver

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages browser session persistence through cookies."""

    def __init__(self, cookie_file_path: Path):
        """
        Initialize session manager.

        Args:
            cookie_file_path: Path to the cookie storage file
        """
        self.cookie_file_path = cookie_file_path

        # Ensure parent directory exists
        self.cookie_file_path.parent.mkdir(parents=True, exist_ok=True)

    def save_cookies(self, driver: webdriver.Chrome, domain: Optional[str] = None) -> bool:
        """
        Save browser cookies to file.

        Args:
            driver: Selenium WebDriver instance
            domain: Optional domain filter (save only cookies for this domain)

        Returns:
            True if cookies saved successfully, False otherwise
        """
        try:
            cookies = driver.get_cookies()

            # Filter by domain if specified
            if domain:
                cookies = [c for c in cookies if domain in c.get('domain', '')]

            # Add metadata
            cookie_data = {
                'saved_at': datetime.now().isoformat(),
                'domain': domain,
                'cookie_count': len(cookies),
                'cookies': cookies
            }

            # Save to file
            with open(self.cookie_file_path, 'w') as f:
                json.dump(cookie_data, f, indent=2)

            logger.info(
                f"Saved {len(cookies)} cookie(s) to {self.cookie_file_path.name}"
            )
            return True

        except Exception as e:
            logger.error(f"Error saving cookies: {e}")
            return False

    def load_cookies(self, driver: webdriver.Chrome) -> bool:
        """
        Load cookies from file into browser session.

        Args:
            driver: Selenium WebDriver instance

        Returns:
            True if cookies loaded successfully, False otherwise
        """
        try:
            # Check if cookie file exists
            if not self.cookie_file_path.exists():
                logger.info("No saved cookies found")
                return False

            # Load cookie data
            with open(self.cookie_file_path, 'r') as f:
                cookie_data = json.load(f)

            cookies = cookie_data.get('cookies', [])

            if not cookies:
                logger.warning("Cookie file exists but contains no cookies")
                return False

            # Add cookies to browser
            loaded_count = 0
            for cookie in cookies:
                try:
                    # Remove problematic fields that Selenium doesn't accept
                    cookie_copy = cookie.copy()
                    cookie_copy.pop('sameSite', None)  # Can cause issues
                    cookie_copy.pop('expiry', None)     # Let browser handle expiration

                    driver.add_cookie(cookie_copy)
                    loaded_count += 1
                except Exception as e:
                    logger.debug(f"Skipped cookie {cookie.get('name', 'unknown')}: {e}")

            logger.info(
                f"Loaded {loaded_count}/{len(cookies)} cookie(s) from "
                f"{self.cookie_file_path.name} "
                f"(saved at: {cookie_data.get('saved_at', 'unknown')})"
            )

            return loaded_count > 0

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in cookie file: {self.cookie_file_path}")
            return False

        except Exception as e:
            logger.error(f"Error loading cookies: {e}")
            return False

    def clear_cookies(self) -> bool:
        """
        Delete the cookie file.

        Returns:
            True if file deleted successfully, False otherwise
        """
        try:
            if self.cookie_file_path.exists():
                self.cookie_file_path.unlink()
                logger.info(f"Deleted cookie file: {self.cookie_file_path.name}")
                return True
            else:
                logger.info("No cookie file to delete")
                return False

        except Exception as e:
            logger.error(f"Error deleting cookie file: {e}")
            return False

    def get_cookie_info(self) -> Optional[dict]:
        """
        Get information about saved cookies without loading them.

        Returns:
            Dictionary with cookie metadata, or None if no cookies exist
        """
        try:
            if not self.cookie_file_path.exists():
                return None

            with open(self.cookie_file_path, 'r') as f:
                cookie_data = json.load(f)

            return {
                'saved_at': cookie_data.get('saved_at'),
                'domain': cookie_data.get('domain'),
                'cookie_count': cookie_data.get('cookie_count', 0),
                'file_path': str(self.cookie_file_path)
            }

        except Exception as e:
            logger.error(f"Error reading cookie info: {e}")
            return None

    def is_session_valid(self, max_age_hours: int = 720) -> bool:
        """
        Check if saved session is still valid based on age.

        Args:
            max_age_hours: Maximum age of cookies in hours (default: 30 days = 720 hours)

        Returns:
            True if session appears valid, False otherwise
        """
        try:
            info = self.get_cookie_info()

            if not info or not info.get('saved_at'):
                return False

            # Parse saved timestamp
            saved_at = datetime.fromisoformat(info['saved_at'])
            age_hours = (datetime.now() - saved_at).total_seconds() / 3600

            is_valid = age_hours < max_age_hours

            if is_valid:
                logger.debug(
                    f"Session is valid (age: {age_hours:.1f} hours, "
                    f"max: {max_age_hours} hours)"
                )
            else:
                logger.debug(
                    f"Session expired (age: {age_hours:.1f} hours, "
                    f"max: {max_age_hours} hours)"
                )

            return is_valid

        except Exception as e:
            logger.error(f"Error checking session validity: {e}")
            return False


def create_session_manager(cookie_dir: Path, session_name: str = "fmls_session") -> SessionManager:
    """
    Helper function to create a SessionManager instance.

    Args:
        cookie_dir: Directory where cookies should be stored
        session_name: Name for the session file (without extension)

    Returns:
        SessionManager instance
    """
    cookie_file = cookie_dir / f"{session_name}.json"
    return SessionManager(cookie_file)
