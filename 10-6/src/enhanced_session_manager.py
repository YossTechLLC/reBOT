"""
Enhanced Session Manager with Chrome DevTools Protocol (CDP)

This module uses CDP to capture and restore the COMPLETE browser state:
1. ALL cookies from ALL domains (via CDP Network.getAllCookies)
2. localStorage from current page
3. sessionStorage from current page

This approach is MUCH faster and more reliable than navigating to multiple domains.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime
from selenium import webdriver

logger = logging.getLogger(__name__)


class EnhancedSessionManager:
    """
    Manages complete browser session persistence using Chrome DevTools Protocol.
    """

    def __init__(self, session_dir: Path, session_name: str = "fmls_session"):
        """
        Initialize enhanced session manager.

        Args:
            session_dir: Directory to store session files
            session_name: Base name for session files
        """
        self.session_dir = session_dir
        self.session_name = session_name

        # Ensure directory exists
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.state_file = self.session_dir / f"{session_name}_state.json"
        self.metadata_file = self.session_dir / f"{session_name}_metadata.json"

    def save_complete_state(
        self,
        driver: webdriver.Chrome,
        user_agent: Optional[str] = None
    ) -> bool:
        """
        Save complete browser state using Chrome DevTools Protocol.

        Uses CDP to get ALL cookies from ALL domains in a single call.
        This is much faster and more reliable than navigating to each domain.

        Args:
            driver: Selenium WebDriver instance
            user_agent: User agent string (for consistency check)

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            logger.info("=== SAVING COMPLETE BROWSER STATE (CDP) ===")

            current_url = driver.current_url
            logger.info(f"Current URL: {current_url}")

            # Use CDP to get ALL cookies from ALL domains in one call
            logger.info("Getting all cookies via Chrome DevTools Protocol...")
            try:
                cdp_result = driver.execute_cdp_cmd('Network.getAllCookies', {})
                all_cookies = cdp_result.get('cookies', [])
                logger.info(f"✓ Retrieved {len(all_cookies)} cookies from all domains via CDP")
            except Exception as e:
                logger.error(f"Failed to get cookies via CDP: {e}")
                return False

            # Get localStorage from current page
            try:
                local_storage = driver.execute_script("""
                    let items = {};
                    try {
                        for (let i = 0; i < localStorage.length; i++) {
                            let key = localStorage.key(i);
                            items[key] = localStorage.getItem(key);
                        }
                    } catch(e) {}
                    return items;
                """)
                logger.info(f"✓ Retrieved {len(local_storage)} localStorage items")
            except Exception as e:
                logger.warning(f"Failed to get localStorage: {e}")
                local_storage = {}

            # Get sessionStorage from current page
            try:
                session_storage = driver.execute_script("""
                    let items = {};
                    try {
                        for (let i = 0; i < sessionStorage.length; i++) {
                            let key = sessionStorage.key(i);
                            items[key] = sessionStorage.getItem(key);
                        }
                    } catch(e) {}
                    return items;
                """)
                logger.info(f"✓ Retrieved {len(session_storage)} sessionStorage items")
            except Exception as e:
                logger.warning(f"Failed to get sessionStorage: {e}")
                session_storage = {}

            # Group cookies by domain for analysis
            cookie_domains = {}
            for cookie in all_cookies:
                domain = cookie.get('domain', 'unknown')
                if domain not in cookie_domains:
                    cookie_domains[domain] = 0
                cookie_domains[domain] += 1

            logger.info(f"Cookie distribution across domains:")
            for domain, count in sorted(cookie_domains.items()):
                logger.info(f"  {domain}: {count} cookies")

            # Check for critical Auth0 cookies
            auth0_cookies = [c for c in all_cookies if 'auth0' in c.get('name', '').lower()]
            if auth0_cookies:
                logger.info(f"✓ Found {len(auth0_cookies)} Auth0 cookies")
                for cookie in auth0_cookies:
                    logger.info(f"  - {cookie.get('name')} on {cookie.get('domain')}")
            else:
                logger.warning("⚠️ No Auth0 cookies found - authentication may fail!")

            # Create state object
            state = {
                'saved_at': datetime.now().isoformat(),
                'user_agent': user_agent,
                'current_url': current_url,
                'cookies': all_cookies,
                'localStorage': local_storage,
                'sessionStorage': session_storage,
                'cookie_count': len(all_cookies),
                'localStorage_count': len(local_storage),
                'sessionStorage_count': len(session_storage),
            }

            # Save state to file
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)

            file_size = self.state_file.stat().st_size
            logger.info(f"✓ Saved complete state: {file_size:,} bytes")
            logger.info(f"  Total: {len(all_cookies)} cookies, {len(local_storage)} localStorage, {len(session_storage)} sessionStorage")

            # Save metadata
            self._save_metadata({
                'saved_at': datetime.now().isoformat(),
                'user_agent': user_agent,
                'current_url': current_url,
                'total_cookies': len(all_cookies),
                'total_localStorage': len(local_storage),
                'total_sessionStorage': len(session_storage),
            })

            return True

        except Exception as e:
            logger.error(f"Error saving complete state: {e}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            return False

    def load_complete_state(
        self,
        driver: webdriver.Chrome,
        verify_user_agent: bool = True
    ) -> bool:
        """
        Load complete browser state using Chrome DevTools Protocol.

        Uses CDP to set ALL cookies at once, BEFORE any navigation.
        This ensures cookies are present when the page loads.

        Args:
            driver: Selenium WebDriver instance
            verify_user_agent: Whether to verify user agent matches

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            logger.info("=== LOADING COMPLETE BROWSER STATE (CDP) ===")

            # Check if state file exists
            if not self.state_file.exists():
                logger.info("No saved state found")
                return False

            # Load state
            with open(self.state_file, 'r') as f:
                state = json.load(f)

            saved_at = state.get('saved_at')
            saved_url = state.get('current_url')
            saved_ua = state.get('user_agent')

            logger.info(f"State saved at: {saved_at}")
            logger.info(f"Saved URL: {saved_url}")

            # Verify user agent if requested
            if verify_user_agent and saved_ua:
                current_ua = driver.execute_script("return navigator.userAgent;")
                if current_ua != saved_ua:
                    logger.warning("⚠️ User-Agent mismatch!")
                    logger.warning(f"  Saved: {saved_ua[:80]}...")
                    logger.warning(f"  Current: {current_ua[:80]}...")
                    logger.warning("  This may cause authentication to fail!")

            cookies = state.get('cookies', [])
            local_storage = state.get('localStorage', {})
            session_storage = state.get('sessionStorage', {})

            logger.info(f"Restoring: {len(cookies)} cookies, {len(local_storage)} localStorage, {len(session_storage)} sessionStorage")

            # CRITICAL: Set ALL cookies via CDP BEFORE navigating
            # This ensures cookies are present when we load the page
            logger.info("Setting cookies via CDP...")
            restored_cookies = 0
            failed_cookies = 0

            for cookie in cookies:
                try:
                    # CDP requires specific cookie format
                    cdp_cookie = {
                        'name': cookie.get('name'),
                        'value': cookie.get('value'),
                        'domain': cookie.get('domain'),
                        'path': cookie.get('path', '/'),
                        'secure': cookie.get('secure', False),
                        'httpOnly': cookie.get('httpOnly', False),
                    }

                    # Add expiry if present
                    if 'expiry' in cookie or 'expires' in cookie:
                        cdp_cookie['expires'] = cookie.get('expiry', cookie.get('expires'))

                    # Add sameSite if present
                    if 'sameSite' in cookie:
                        cdp_cookie['sameSite'] = cookie.get('sameSite')

                    # Set cookie via CDP
                    driver.execute_cdp_cmd('Network.setCookie', cdp_cookie)
                    restored_cookies += 1

                except Exception as e:
                    failed_cookies += 1
                    logger.debug(f"Failed to set cookie {cookie.get('name')}: {e}")

            logger.info(f"✓ Set {restored_cookies} cookies via CDP ({failed_cookies} failed)")

            # Now navigate to the saved URL
            # Cookies are already present, so authentication should work
            logger.info(f"Navigating to saved URL: {saved_url}")
            driver.get(saved_url)

            import time
            time.sleep(2)  # Give page time to load

            # Restore localStorage
            if local_storage:
                logger.info(f"Restoring {len(local_storage)} localStorage items...")
                for key, value in local_storage.items():
                    try:
                        driver.execute_script(
                            "localStorage.setItem(arguments[0], arguments[1]);",
                            key, value
                        )
                    except Exception as e:
                        logger.debug(f"Failed to restore localStorage[{key}]: {e}")

            # Restore sessionStorage
            if session_storage:
                logger.info(f"Restoring {len(session_storage)} sessionStorage items...")
                for key, value in session_storage.items():
                    try:
                        driver.execute_script(
                            "sessionStorage.setItem(arguments[0], arguments[1]);",
                            key, value
                        )
                    except Exception as e:
                        logger.debug(f"Failed to restore sessionStorage[{key}]: {e}")

            logger.info("✓ Complete state restored successfully")
            logger.info(f"Current URL: {driver.current_url}")

            return restored_cookies > 0

        except Exception as e:
            logger.error(f"Error loading complete state: {e}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            return False

    def _save_metadata(self, metadata: Dict) -> None:
        """Save session metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save metadata: {e}")

    def get_metadata(self) -> Optional[Dict]:
        """Get session metadata."""
        try:
            if not self.metadata_file.exists():
                return None

            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read metadata: {e}")
            return None

    def is_session_valid(self, max_age_hours: int = 720) -> bool:
        """
        Check if saved session is still valid based on age.

        Args:
            max_age_hours: Maximum age in hours (default: 30 days)

        Returns:
            True if session appears valid, False otherwise
        """
        try:
            metadata = self.get_metadata()
            if not metadata:
                return False

            saved_at_str = metadata.get('saved_at')
            if not saved_at_str:
                return False

            saved_at = datetime.fromisoformat(saved_at_str)
            age_hours = (datetime.now() - saved_at).total_seconds() / 3600

            is_valid = age_hours < max_age_hours

            if is_valid:
                remaining = max_age_hours - age_hours
                logger.info(f"✓ Session valid (expires in {remaining:.1f} hours)")
            else:
                expired_by = age_hours - max_age_hours
                logger.info(f"✗ Session expired ({expired_by:.1f} hours ago)")

            return is_valid

        except Exception as e:
            logger.error(f"Error checking session validity: {e}")
            return False

    def clear_session(self) -> bool:
        """Delete all session files."""
        try:
            deleted = []

            if self.state_file.exists():
                self.state_file.unlink()
                deleted.append(self.state_file.name)

            if self.metadata_file.exists():
                self.metadata_file.unlink()
                deleted.append(self.metadata_file.name)

            if deleted:
                logger.info(f"Deleted: {', '.join(deleted)}")
                return True
            else:
                logger.info("No session files to delete")
                return False

        except Exception as e:
            logger.error(f"Error deleting session: {e}")
            return False


def create_enhanced_session_manager(
    session_dir: Path,
    session_name: str = "fmls_session"
) -> EnhancedSessionManager:
    """
    Factory function to create an EnhancedSessionManager instance.

    Args:
        session_dir: Directory for session storage
        session_name: Name for the session

    Returns:
        EnhancedSessionManager instance
    """
    return EnhancedSessionManager(session_dir, session_name)
