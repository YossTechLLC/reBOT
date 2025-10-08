"""
Enhanced Session Manager with Full Browser State Persistence

This module saves and restores:
1. Cookies (all domains)
2. localStorage (all domains)
3. sessionStorage (all domains)
4. Browser fingerprint consistency
5. User-agent and metadata

This is the CORRECT way to persist authentication state across runs.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
from selenium import webdriver
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class EnhancedSessionManager:
    """
    Manages complete browser session persistence including cookies and storage.
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
        domains: List[str],
        user_agent: Optional[str] = None
    ) -> bool:
        """
        Save complete browser state including cookies and storage from all domains.

        Args:
            driver: Selenium WebDriver instance
            domains: List of domain URLs to capture state from
            user_agent: User agent string (for consistency check)

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            logger.info(f"[SESSION] === SAVING COMPLETE BROWSER STATE ===")
            logger.info(f"[SESSION] Capturing state from {len(domains)} domains")

            state = {
                'saved_at': datetime.now().isoformat(),
                'user_agent': user_agent,
                'domains': {},
            }

            total_cookies = 0
            total_local_storage = 0
            total_session_storage = 0

            # Capture state from each domain
            for domain_url in domains:
                logger.info(f"[SESSION] Capturing: {domain_url}")

                try:
                    # Navigate to domain
                    driver.get(domain_url)
                    import time
                    time.sleep(1)

                    # Get domain key
                    parsed = urlparse(domain_url)
                    domain_key = parsed.netloc

                    # Capture cookies
                    cookies = driver.get_cookies()
                    total_cookies += len(cookies)

                    # Capture localStorage
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
                    total_local_storage += len(local_storage)

                    # Capture sessionStorage
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
                    total_session_storage += len(session_storage)

                    # Store domain state
                    state['domains'][domain_key] = {
                        'url': domain_url,
                        'actual_url': driver.current_url,
                        'cookies': cookies,
                        'localStorage': local_storage,
                        'sessionStorage': session_storage,
                        'counts': {
                            'cookies': len(cookies),
                            'localStorage': len(local_storage),
                            'sessionStorage': len(session_storage),
                        }
                    }

                    logger.info(f"[SESSION]   ✓ {domain_key}: "
                               f"{len(cookies)} cookies, "
                               f"{len(local_storage)} localStorage, "
                               f"{len(session_storage)} sessionStorage")

                except Exception as e:
                    logger.warning(f"[SESSION]   ✗ Failed to capture {domain_url}: {e}")
                    state['domains'][domain_key] = {'error': str(e)}

            # Save state to file
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)

            file_size = self.state_file.stat().st_size

            logger.info(f"[SESSION] ✓ Saved complete state: {file_size:,} bytes")
            logger.info(f"[SESSION]   Total: {total_cookies} cookies, "
                       f"{total_local_storage} localStorage, "
                       f"{total_session_storage} sessionStorage")

            # Save metadata
            self._save_metadata({
                'saved_at': datetime.now().isoformat(),
                'user_agent': user_agent,
                'total_cookies': total_cookies,
                'total_localStorage': total_local_storage,
                'total_sessionStorage': total_session_storage,
                'domains_captured': len(domains),
            })

            return True

        except Exception as e:
            logger.error(f"[SESSION] Error saving complete state: {e}")
            import traceback
            logger.error(f"[SESSION] Traceback:\n{traceback.format_exc()}")
            return False

    def load_complete_state(
        self,
        driver: webdriver.Chrome,
        verify_user_agent: bool = True
    ) -> bool:
        """
        Load complete browser state including cookies and storage.

        Args:
            driver: Selenium WebDriver instance
            verify_user_agent: Whether to verify user agent matches

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            logger.info(f"[SESSION] === LOADING COMPLETE BROWSER STATE ===")

            # Check if state file exists
            if not self.state_file.exists():
                logger.info(f"[SESSION] No saved state found")
                return False

            # Load state
            with open(self.state_file, 'r') as f:
                state = json.load(f)

            saved_at = state.get('saved_at')
            saved_ua = state.get('user_agent')

            logger.info(f"[SESSION] State saved at: {saved_at}")

            # Verify user agent if requested
            if verify_user_agent and saved_ua:
                current_ua = driver.execute_script("return navigator.userAgent;")
                if current_ua != saved_ua:
                    logger.warning(f"[SESSION] ⚠️ User-Agent mismatch!")
                    logger.warning(f"[SESSION]   Saved: {saved_ua[:80]}...")
                    logger.warning(f"[SESSION]   Current: {current_ua[:80]}...")
                    logger.warning(f"[SESSION]   This may cause authentication to fail!")

            domains_data = state.get('domains', {})
            logger.info(f"[SESSION] Restoring state for {len(domains_data)} domains")

            total_restored_cookies = 0
            total_restored_local = 0
            total_restored_session = 0

            # Restore state for each domain
            for domain_key, domain_data in domains_data.items():
                if 'error' in domain_data:
                    logger.warning(f"[SESSION] Skipping {domain_key}: {domain_data['error']}")
                    continue

                domain_url = domain_data.get('url')
                logger.info(f"[SESSION] Restoring: {domain_key}")

                try:
                    # Navigate to domain
                    driver.get(domain_url)
                    import time
                    time.sleep(1)

                    # Restore cookies
                    cookies = domain_data.get('cookies', [])
                    for cookie in cookies:
                        try:
                            # Clean cookie
                            cookie_copy = cookie.copy()
                            cookie_copy.pop('sameSite', None)
                            cookie_copy.pop('expiry', None)

                            driver.add_cookie(cookie_copy)
                            total_restored_cookies += 1
                        except Exception as e:
                            logger.debug(f"[SESSION] Skipped cookie {cookie.get('name')}: {e}")

                    # Restore localStorage
                    local_storage = domain_data.get('localStorage', {})
                    if local_storage:
                        for key, value in local_storage.items():
                            try:
                                driver.execute_script(
                                    f"localStorage.setItem(arguments[0], arguments[1]);",
                                    key, value
                                )
                                total_restored_local += 1
                            except Exception as e:
                                logger.debug(f"[SESSION] Failed to restore localStorage[{key}]: {e}")

                    # Restore sessionStorage
                    session_storage = domain_data.get('sessionStorage', {})
                    if session_storage:
                        for key, value in session_storage.items():
                            try:
                                driver.execute_script(
                                    f"sessionStorage.setItem(arguments[0], arguments[1]);",
                                    key, value
                                )
                                total_restored_session += 1
                            except Exception as e:
                                logger.debug(f"[SESSION] Failed to restore sessionStorage[{key}]: {e}")

                    logger.info(f"[SESSION]   ✓ {domain_key}: "
                               f"{len(cookies)} cookies, "
                               f"{len(local_storage)} localStorage, "
                               f"{len(session_storage)} sessionStorage")

                except Exception as e:
                    logger.warning(f"[SESSION]   ✗ Failed to restore {domain_key}: {e}")

            logger.info(f"[SESSION] ✓ Restored complete state:")
            logger.info(f"[SESSION]   {total_restored_cookies} cookies")
            logger.info(f"[SESSION]   {total_restored_local} localStorage items")
            logger.info(f"[SESSION]   {total_restored_session} sessionStorage items")

            return total_restored_cookies > 0

        except Exception as e:
            logger.error(f"[SESSION] Error loading complete state: {e}")
            import traceback
            logger.error(f"[SESSION] Traceback:\n{traceback.format_exc()}")
            return False

    def _save_metadata(self, metadata: Dict) -> None:
        """Save session metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"[SESSION] Failed to save metadata: {e}")

    def get_metadata(self) -> Optional[Dict]:
        """Get session metadata."""
        try:
            if not self.metadata_file.exists():
                return None

            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"[SESSION] Failed to read metadata: {e}")
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
                logger.info(f"[SESSION] No metadata found")
                return False

            saved_at_str = metadata.get('saved_at')
            if not saved_at_str:
                logger.warning(f"[SESSION] Metadata missing 'saved_at'")
                return False

            saved_at = datetime.fromisoformat(saved_at_str)
            age_hours = (datetime.now() - saved_at).total_seconds() / 3600

            is_valid = age_hours < max_age_hours

            if is_valid:
                remaining = max_age_hours - age_hours
                logger.info(f"[SESSION] ✓ Session valid (expires in {remaining:.1f} hours)")
            else:
                expired_by = age_hours - max_age_hours
                logger.info(f"[SESSION] ✗ Session expired ({expired_by:.1f} hours ago)")

            return is_valid

        except Exception as e:
            logger.error(f"[SESSION] Error checking session validity: {e}")
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
                logger.info(f"[SESSION] Deleted: {', '.join(deleted)}")
                return True
            else:
                logger.info(f"[SESSION] No session files to delete")
                return False

        except Exception as e:
            logger.error(f"[SESSION] Error deleting session: {e}")
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
