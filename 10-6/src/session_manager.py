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
            # DEBUG SESSION PERSISTENCE: Get all cookies from browser
            cookies = driver.get_cookies()
            logger.info(f"[DEBUG SESSION] Retrieved {len(cookies)} total cookies from browser")

            # DEBUG SESSION PERSISTENCE: Log current URL and domains
            current_url = driver.current_url
            logger.info(f"[DEBUG SESSION] Current browser URL: {current_url}")
            logger.info(f"[DEBUG SESSION] Cookie filter domain: {domain if domain else 'None (all domains)'}")

            # DEBUG SESSION PERSISTENCE: Show all cookie domains before filtering
            all_domains = set(c.get('domain', 'unknown') for c in cookies)
            logger.info(f"[DEBUG SESSION] Cookie domains present: {', '.join(all_domains)}")
            # END DEBUG SESSION PERSISTENCE

            # Filter by domain if specified
            if domain:
                original_count = len(cookies)
                cookies = [c for c in cookies if domain in c.get('domain', '')]
                # DEBUG SESSION PERSISTENCE: Log filtering results
                logger.info(f"[DEBUG SESSION] Filtered cookies: {original_count} -> {len(cookies)} (kept cookies with '{domain}' in domain)")
                if len(cookies) == 0:
                    logger.warning(f"[DEBUG SESSION] ⚠️ No cookies matched domain filter '{domain}'! This may cause session persistence issues.")
                # END DEBUG SESSION PERSISTENCE

            # DEBUG SESSION PERSISTENCE: Log sample cookie details (first 3)
            for i, cookie in enumerate(cookies[:3]):
                logger.info(f"[DEBUG SESSION] Cookie {i+1}: name={cookie.get('name')}, domain={cookie.get('domain')}, httpOnly={cookie.get('httpOnly')}, secure={cookie.get('secure')}")
            if len(cookies) > 3:
                logger.info(f"[DEBUG SESSION] ... and {len(cookies) - 3} more cookies")
            # END DEBUG SESSION PERSISTENCE

            # Add metadata
            cookie_data = {
                'saved_at': datetime.now().isoformat(),
                'domain': domain,
                'cookie_count': len(cookies),
                'cookies': cookies
            }

            # DEBUG SESSION PERSISTENCE: Log file path
            logger.info(f"[DEBUG SESSION] Writing cookies to: {self.cookie_file_path}")
            logger.info(f"[DEBUG SESSION] File path exists before write: {self.cookie_file_path.exists()}")
            # END DEBUG SESSION PERSISTENCE

            # Save to file
            with open(self.cookie_file_path, 'w') as f:
                json.dump(cookie_data, f, indent=2)

            # DEBUG SESSION PERSISTENCE: Verify file was created
            if self.cookie_file_path.exists():
                file_size = self.cookie_file_path.stat().st_size
                logger.info(f"[DEBUG SESSION] ✓ Cookie file created successfully: {file_size} bytes")
            else:
                logger.error(f"[DEBUG SESSION] ✗ Cookie file was not created!")
            # END DEBUG SESSION PERSISTENCE

            logger.info(
                f"Saved {len(cookies)} cookie(s) to {self.cookie_file_path.name}"
            )
            return True

        except Exception as e:
            logger.error(f"Error saving cookies: {e}")
            # DEBUG SESSION PERSISTENCE: Log exception details
            logger.error(f"[DEBUG SESSION] Exception type: {type(e).__name__}")
            logger.error(f"[DEBUG SESSION] Exception details: {str(e)}")
            import traceback
            logger.error(f"[DEBUG SESSION] Traceback:\n{traceback.format_exc()}")
            # END DEBUG SESSION PERSISTENCE
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
            # DEBUG SESSION PERSISTENCE: Log load attempt
            logger.info(f"[DEBUG SESSION] === COOKIE LOADING ATTEMPT ===")
            logger.info(f"[DEBUG SESSION] Cookie file path: {self.cookie_file_path}")
            logger.info(f"[DEBUG SESSION] File exists: {self.cookie_file_path.exists()}")
            # END DEBUG SESSION PERSISTENCE

            # Check if cookie file exists
            if not self.cookie_file_path.exists():
                logger.info("No saved cookies found")
                # DEBUG SESSION PERSISTENCE: Check parent directory
                logger.info(f"[DEBUG SESSION] Parent directory exists: {self.cookie_file_path.parent.exists()}")
                logger.info(f"[DEBUG SESSION] Parent directory contents: {list(self.cookie_file_path.parent.iterdir()) if self.cookie_file_path.parent.exists() else 'N/A'}")
                # END DEBUG SESSION PERSISTENCE
                return False

            # DEBUG SESSION PERSISTENCE: Log file details
            file_size = self.cookie_file_path.stat().st_size
            file_mtime = datetime.fromtimestamp(self.cookie_file_path.stat().st_mtime)
            logger.info(f"[DEBUG SESSION] Cookie file size: {file_size} bytes")
            logger.info(f"[DEBUG SESSION] Cookie file modified: {file_mtime.isoformat()}")
            logger.info(f"[DEBUG SESSION] Current browser URL before loading: {driver.current_url}")
            # END DEBUG SESSION PERSISTENCE

            # Load cookie data
            with open(self.cookie_file_path, 'r') as f:
                cookie_data = json.load(f)

            # DEBUG SESSION PERSISTENCE: Log cookie data metadata
            logger.info(f"[DEBUG SESSION] Cookie data saved_at: {cookie_data.get('saved_at', 'unknown')}")
            logger.info(f"[DEBUG SESSION] Cookie data domain filter: {cookie_data.get('domain', 'None')}")
            logger.info(f"[DEBUG SESSION] Cookie data count: {cookie_data.get('cookie_count', 0)}")
            # END DEBUG SESSION PERSISTENCE

            cookies = cookie_data.get('cookies', [])

            if not cookies:
                logger.warning("Cookie file exists but contains no cookies")
                return False

            # DEBUG SESSION PERSISTENCE: Show cookie domains we're about to load
            cookie_domains = set(c.get('domain', 'unknown') for c in cookies)
            logger.info(f"[DEBUG SESSION] Cookie domains to load: {', '.join(cookie_domains)}")
            # END DEBUG SESSION PERSISTENCE

            # DEBUG SESSION PERSISTENCE: Check current URL for domain matching
            current_url = driver.current_url
            logger.info(f"[DEBUG SESSION] Current URL for cookie loading: {current_url}")

            # Extract current domain from URL
            from urllib.parse import urlparse
            current_domain = urlparse(current_url).netloc
            logger.info(f"[DEBUG SESSION] Current domain: {current_domain}")
            # END DEBUG SESSION PERSISTENCE

            # Add cookies to browser
            loaded_count = 0
            skipped_count = 0
            # DEBUG SESSION PERSISTENCE: Track skip reasons
            skip_reasons = {}
            # END DEBUG SESSION PERSISTENCE

            for cookie in cookies:
                try:
                    # Remove problematic fields that Selenium doesn't accept
                    cookie_copy = cookie.copy()
                    cookie_copy.pop('sameSite', None)  # Can cause issues
                    cookie_copy.pop('expiry', None)     # Let browser handle expiration

                    # DEBUG SESSION PERSISTENCE: Check domain compatibility
                    cookie_domain = cookie.get('domain', '')
                    cookie_name = cookie.get('name', 'unknown')

                    # Clean domain (remove leading dot)
                    clean_domain = cookie_domain.lstrip('.')

                    # Check if cookie domain matches current page domain
                    # Cookie domain should be a suffix of current domain or vice versa
                    domain_match = (
                        clean_domain in current_domain or
                        current_domain.endswith(clean_domain) or
                        clean_domain == current_domain
                    )

                    logger.debug(f"[DEBUG SESSION] Cookie '{cookie_name}': domain={cookie_domain}, current={current_domain}, match={domain_match}")

                    if not domain_match:
                        skipped_count += 1
                        reason = f"Domain mismatch: cookie domain '{cookie_domain}' doesn't match current page '{current_domain}'"
                        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                        logger.debug(f"[DEBUG SESSION] Skipping cookie '{cookie_name}': {reason}")
                        continue
                    # END DEBUG SESSION PERSISTENCE

                    # DEBUG SESSION PERSISTENCE: Log cookie being added
                    logger.debug(f"[DEBUG SESSION] Adding cookie: name={cookie_name}, domain={cookie_domain}")
                    # END DEBUG SESSION PERSISTENCE

                    driver.add_cookie(cookie_copy)
                    loaded_count += 1
                except Exception as e:
                    skipped_count += 1
                    # DEBUG SESSION PERSISTENCE: Track skip reasons
                    reason = str(e)
                    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                    # END DEBUG SESSION PERSISTENCE
                    logger.debug(f"Skipped cookie {cookie.get('name', 'unknown')}: {e}")

            # DEBUG SESSION PERSISTENCE: Log skip summary
            if skipped_count > 0:
                logger.warning(f"[DEBUG SESSION] Skipped {skipped_count} cookies. Reasons:")
                for reason, count in skip_reasons.items():
                    logger.warning(f"[DEBUG SESSION]   - {reason}: {count} cookie(s)")
            # END DEBUG SESSION PERSISTENCE

            # DEBUG SESSION PERSISTENCE: Verify cookies were added to browser
            browser_cookies_after = driver.get_cookies()
            logger.info(f"[DEBUG SESSION] Browser now has {len(browser_cookies_after)} total cookies")
            logger.info(f"[DEBUG SESSION] Successfully loaded {loaded_count}/{len(cookies)} cookies")
            # END DEBUG SESSION PERSISTENCE

            logger.info(
                f"Loaded {loaded_count}/{len(cookies)} cookie(s) from "
                f"{self.cookie_file_path.name} "
                f"(saved at: {cookie_data.get('saved_at', 'unknown')})"
            )

            return loaded_count > 0

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in cookie file: {self.cookie_file_path}")
            # DEBUG SESSION PERSISTENCE: Show JSON error details
            logger.error(f"[DEBUG SESSION] JSON decode error: {e}")
            logger.error(f"[DEBUG SESSION] Try to read raw file content...")
            try:
                with open(self.cookie_file_path, 'r') as f:
                    content = f.read(500)  # First 500 chars
                    logger.error(f"[DEBUG SESSION] File content preview: {content[:500]}")
            except:
                pass
            # END DEBUG SESSION PERSISTENCE
            return False

        except Exception as e:
            logger.error(f"Error loading cookies: {e}")
            # DEBUG SESSION PERSISTENCE: Log exception details
            logger.error(f"[DEBUG SESSION] Exception type: {type(e).__name__}")
            logger.error(f"[DEBUG SESSION] Exception details: {str(e)}")
            import traceback
            logger.error(f"[DEBUG SESSION] Traceback:\n{traceback.format_exc()}")
            # END DEBUG SESSION PERSISTENCE
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
            # DEBUG SESSION PERSISTENCE: Log validation attempt
            logger.info(f"[DEBUG SESSION] === SESSION VALIDITY CHECK ===")
            logger.info(f"[DEBUG SESSION] Cookie file path: {self.cookie_file_path}")
            logger.info(f"[DEBUG SESSION] Max age allowed: {max_age_hours} hours ({max_age_hours/24:.1f} days)")
            # END DEBUG SESSION PERSISTENCE

            info = self.get_cookie_info()

            if not info:
                # DEBUG SESSION PERSISTENCE: Log why info is None
                logger.info(f"[DEBUG SESSION] ✗ No cookie info available (file doesn't exist or error reading)")
                # END DEBUG SESSION PERSISTENCE
                return False

            if not info.get('saved_at'):
                # DEBUG SESSION PERSISTENCE: Log missing timestamp
                logger.warning(f"[DEBUG SESSION] ✗ Cookie file exists but missing 'saved_at' timestamp")
                logger.warning(f"[DEBUG SESSION] Cookie info: {info}")
                # END DEBUG SESSION PERSISTENCE
                return False

            # Parse saved timestamp
            saved_at = datetime.fromisoformat(info['saved_at'])
            current_time = datetime.now()
            age_hours = (current_time - saved_at).total_seconds() / 3600
            age_minutes = age_hours * 60

            # DEBUG SESSION PERSISTENCE: Log age calculation
            logger.info(f"[DEBUG SESSION] Cookie saved at: {saved_at.isoformat()}")
            logger.info(f"[DEBUG SESSION] Current time: {current_time.isoformat()}")
            logger.info(f"[DEBUG SESSION] Session age: {age_hours:.2f} hours ({age_minutes:.1f} minutes)")
            logger.info(f"[DEBUG SESSION] Cookie count in file: {info.get('cookie_count', 0)}")
            # END DEBUG SESSION PERSISTENCE

            is_valid = age_hours < max_age_hours

            if is_valid:
                # DEBUG SESSION PERSISTENCE: Log validity
                remaining_hours = max_age_hours - age_hours
                logger.info(f"[DEBUG SESSION] ✓ Session is VALID (expires in {remaining_hours:.1f} hours / {remaining_hours/24:.1f} days)")
                # END DEBUG SESSION PERSISTENCE
                logger.debug(
                    f"Session is valid (age: {age_hours:.1f} hours, "
                    f"max: {max_age_hours} hours)"
                )
            else:
                # DEBUG SESSION PERSISTENCE: Log expiration
                expired_by_hours = age_hours - max_age_hours
                logger.info(f"[DEBUG SESSION] ✗ Session EXPIRED (expired {expired_by_hours:.1f} hours ago)")
                # END DEBUG SESSION PERSISTENCE
                logger.debug(
                    f"Session expired (age: {age_hours:.1f} hours, "
                    f"max: {max_age_hours} hours)"
                )

            return is_valid

        except Exception as e:
            logger.error(f"Error checking session validity: {e}")
            # DEBUG SESSION PERSISTENCE: Log exception details
            logger.error(f"[DEBUG SESSION] Exception type: {type(e).__name__}")
            logger.error(f"[DEBUG SESSION] Exception details: {str(e)}")
            import traceback
            logger.error(f"[DEBUG SESSION] Traceback:\n{traceback.format_exc()}")
            # END DEBUG SESSION PERSISTENCE
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
