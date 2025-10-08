"""
FMLS authentication module.
Handles the complete login flow including 2FA for FMLS/Remine access.
"""

import logging
import time
import json
from datetime import datetime
from typing import Optional, Dict
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    WebDriverException
)

from .gcp_secrets import SecretManagerClient
from .session_manager import SessionManager
from .utils import random_delay

logger = logging.getLogger(__name__)


class FMLSAuthenticator:
    """Handles FMLS authentication and session management."""

    def __init__(
        self,
        driver: webdriver.Chrome,
        session_manager: SessionManager,
        gcp_project_id: str,
        login_secret_name: str,
        password_secret_name: str,
        config: Dict
    ):
        """
        Initialize FMLS authenticator.

        Args:
            driver: Selenium WebDriver instance
            session_manager: SessionManager for cookie persistence
            gcp_project_id: Google Cloud project ID
            login_secret_name: Name of login secret in Secret Manager
            password_secret_name: Name of password secret in Secret Manager
            config: Configuration dictionary with URLs and selectors
        """
        self.driver = driver
        self.session_manager = session_manager
        self.secret_client = SecretManagerClient(gcp_project_id)
        self.login_secret_name = login_secret_name
        self.password_secret_name = password_secret_name
        self.config = config

    def is_authenticated(self) -> bool:
        """
        Check if already authenticated by trying to access the dashboard.

        Returns:
            True if authenticated, False otherwise
        """
        try:
            logger.info("Checking if already authenticated...")

            # DEBUG SESSION PERSISTENCE: Enhanced auth checking
            logger.info(f"[DEBUG SESSION] === AUTHENTICATION CHECK ===")

            # Show current cookies before navigation
            current_cookies = self.driver.get_cookies()
            logger.info(f"[DEBUG SESSION] Browser has {len(current_cookies)} cookies before dashboard check")
            cookie_domains = set(c.get('domain', '') for c in current_cookies)
            logger.info(f"[DEBUG SESSION] Cookie domains in browser: {', '.join(sorted(cookie_domains))}")

            # Check for SSO cookies specifically
            sso_cookies = [c for c in current_cookies if 'firstmls.sso.remine.com' in c.get('domain', '')]
            logger.info(f"[DEBUG SESSION] SSO cookies present: {len(sso_cookies)}")
            if sso_cookies:
                for cookie in sso_cookies[:3]:
                    logger.info(f"[DEBUG SESSION]   - {cookie.get('name')}")
            # END DEBUG SESSION PERSISTENCE

            # Try to navigate to dashboard
            dashboard_url = self.config['dashboard_url']
            logger.info(f"[DEBUG SESSION] Navigating to dashboard: {dashboard_url}")
            self.driver.get(dashboard_url)

            # Wait a moment for page to load
            time.sleep(3)

            # Check if we're on the dashboard (not error page)
            current_url = self.driver.current_url
            page_source = self.driver.page_source.lower()
            page_title = self.driver.title

            # DEBUG SESSION PERSISTENCE: Log page details
            logger.info(f"[DEBUG SESSION] After navigation:")
            logger.info(f"[DEBUG SESSION]   Current URL: {current_url}")
            logger.info(f"[DEBUG SESSION]   Page title: {page_title}")
            logger.info(f"[DEBUG SESSION]   Page source length: {len(page_source)} chars")
            # END DEBUG SESSION PERSISTENCE

            # Check for error message
            if "oops! something's wrong" in page_source or "can't seem to find the page" in page_source:
                logger.info("Not authenticated - error page detected")
                # DEBUG SESSION PERSISTENCE: Show error details
                logger.info(f"[DEBUG SESSION] Error page detected in source")
                logger.info(f"[DEBUG SESSION] This means cookies didn't provide valid authentication")
                # END DEBUG SESSION PERSISTENCE
                return False

            # Check for login redirect
            if 'login' in current_url.lower() or 'auth' in current_url.lower():
                logger.info(f"[DEBUG SESSION] Redirected to login/auth page - not authenticated")
                logger.info(f"[DEBUG SESSION] Redirect URL: {current_url}")
                return False

            # Check if URL contains dashboard
            if 'dashboard' in current_url:
                logger.info("Already authenticated - dashboard accessible")
                # DEBUG SESSION PERSISTENCE: Confirm success
                logger.info(f"[DEBUG SESSION] ✓ Dashboard accessible - authentication successful!")
                # END DEBUG SESSION PERSISTENCE
                return True

            # DEBUG SESSION PERSISTENCE: Unclear status
            logger.info(f"[DEBUG SESSION] Authentication status unclear")
            logger.info(f"[DEBUG SESSION] URL doesn't contain 'dashboard' but no error detected")
            logger.info(f"[DEBUG SESSION] Current URL: {current_url}")
            # END DEBUG SESSION PERSISTENCE

            logger.info("Authentication status unclear - will attempt login")
            return False

        except Exception as e:
            logger.warning(f"Error checking authentication status: {e}")
            # DEBUG SESSION PERSISTENCE: Log exception
            logger.warning(f"[DEBUG SESSION] Exception during auth check: {type(e).__name__}: {e}")
            import traceback
            logger.warning(f"[DEBUG SESSION] Traceback:\n{traceback.format_exc()}")
            # END DEBUG SESSION PERSISTENCE
            return False

    def navigate_to_login(self) -> bool:
        """
        Navigate to login page and handle OAuth redirect.

        Returns:
            True if navigation successful, False otherwise
        """
        try:
            logger.info("Navigating to FMLS SSO page (will redirect to Auth0)...")

            # Navigate to FMLS SSO URL
            login_url = self.config['login_url']
            logger.debug(f"[DEBUG] Initial navigation to: {login_url}")

            self.driver.get(login_url)

            # Log initial URL
            initial_url = self.driver.current_url
            logger.debug(f"[DEBUG] Initial URL after navigation: {initial_url}")

            # Wait for OAuth redirect to Auth0 login page
            logger.info("Waiting for OAuth redirect to Auth0...")
            wait = WebDriverWait(self.driver, self.config.get('login_timeout', 30))

            # Wait for URL to change to Auth0 login page
            try:
                wait.until(lambda d: 'firstmls-login.sso.remine.com' in d.current_url)
                logger.debug(f"[DEBUG] Detected redirect to Auth0 login")
            except TimeoutException:
                logger.warning("Did not detect redirect to firstmls-login.sso.remine.com")
                logger.debug(f"[DEBUG] Current URL: {self.driver.current_url}")
                # Continue anyway - might already be on login page

            # Log the redirected URL with all OAuth parameters
            redirected_url = self.driver.current_url
            logger.debug(f"[DEBUG] Redirected URL: {redirected_url[:100]}...")  # First 100 chars
            logger.info(f"✓ Reached Auth0 login page: {redirected_url.split('?')[0]}")

            # Wait for page to be fully loaded
            wait.until(lambda d: d.execute_script("return document.readyState") == "complete")
            logger.debug("[DEBUG] Page document.readyState = complete")

            # Additional wait for dynamic content
            time.sleep(2)

            # Log page title for verification
            page_title = self.driver.title
            logger.debug(f"[DEBUG] Page title: {page_title}")

            return True

        except TimeoutException:
            logger.error("❌ Timeout waiting for login page to load")
            logger.debug(f"[DEBUG] Current URL: {self.driver.current_url}")
            logger.debug(f"[DEBUG] Page title: {self.driver.title}")

            # Save screenshot for debugging
            self._save_debug_screenshot("navigate_to_login_timeout")

            return False

        except Exception as e:
            logger.error(f"❌ Error navigating to login page: {e}")
            logger.debug(f"[DEBUG] Current URL: {self.driver.current_url}")
            logger.debug(f"[DEBUG] Exception type: {type(e).__name__}")

            # Save screenshot for debugging
            self._save_debug_screenshot("navigate_to_login_error")

            return False

    def perform_login(self, login_id: str, password: str) -> bool:
        """
        Fill in login credentials and submit using Auth0 selectors.

        Args:
            login_id: User login ID
            password: User password

        Returns:
            True if login form submitted successfully, False otherwise
        """
        try:
            logger.info("Filling in login credentials on Auth0 page...")

            # DEBUG: Log all input fields on the page
            self._log_page_inputs()

            wait = WebDriverWait(self.driver, self.config.get('login_timeout', 30))

            # Try multiple possible selectors for username/email field (Auth0 variations)
            username_selectors = [
                'input#loginId',                # FMLS specific - exact match
                'input[name="loginId"]',        # FMLS specific - by name
                '#loginId',                     # FMLS specific - by ID
                'input[type="email"]',          # Auth0 generic
                'input[name="username"]',       # Auth0 generic
                'input[name="email"]',          # Auth0 generic
                '#username',                    # Auth0 generic
                'input[id="username"]',         # Auth0 generic
                self.config['login_id_input']   # Fallback to config
            ]

            login_input = None
            used_selector = None

            logger.debug("[DEBUG] Attempting to find username/email input...")
            for selector in username_selectors:
                try:
                    logger.debug(f"[DEBUG] Trying selector: {selector}")
                    login_input = wait.until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    used_selector = selector
                    logger.info(f"✓ Found username field with selector: {selector}")
                    break
                except (TimeoutException, NoSuchElementException):
                    logger.debug(f"[DEBUG] Selector '{selector}' not found, trying next...")
                    continue

            if not login_input:
                logger.error("❌ Could not find username/email input field with any selector")
                self._save_debug_screenshot("login_username_not_found")
                self._save_page_source("login_username_not_found")
                return False

            # Fill in login ID
            login_input.clear()
            login_input.send_keys(login_id)
            logger.info(f"✓ Entered login ID (field: {used_selector})")

            random_delay(0.5, 1.0)

            # Try multiple possible selectors for password field
            password_selectors = [
                'input#password',               # FMLS specific - exact match
                '#password',                    # FMLS specific - by ID
                'input[name="password"]',       # FMLS/Auth0 - by name
                'input[type="password"]',       # Generic password field
                'input[id="password"]',         # Generic by ID
                self.config['password_input']   # Fallback to config
            ]

            password_input = None
            used_password_selector = None

            logger.debug("[DEBUG] Attempting to find password input...")
            for selector in password_selectors:
                try:
                    logger.debug(f"[DEBUG] Trying selector: {selector}")
                    password_input = wait.until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    used_password_selector = selector
                    logger.info(f"✓ Found password field with selector: {selector}")
                    break
                except (TimeoutException, NoSuchElementException):
                    logger.debug(f"[DEBUG] Selector '{selector}' not found, trying next...")
                    continue

            if not password_input:
                logger.error("❌ Could not find password input field with any selector")
                self._save_debug_screenshot("login_password_not_found")
                self._save_page_source("login_password_not_found")
                return False

            # Fill in password
            password_input.clear()
            password_input.send_keys(password)
            logger.info(f"✓ Entered password (field: {used_password_selector})")

            random_delay(0.5, 1.0)

            # Try multiple possible selectors for submit button
            submit_selectors = [
                'button#btn-login',             # FMLS specific (from original config)
                'button[type="submit"]',        # Generic submit button
                'button[name="action"]',        # Auth0 pattern
                'button.btn-primary',           # Bootstrap primary button
                'button.auth0-lock-submit',     # Auth0 specific
                'input[type="submit"]',         # Input submit
                self.config['login_button']     # Fallback to config
            ]

            login_button = None
            used_button_selector = None

            logger.debug("[DEBUG] Attempting to find submit button...")
            for selector in submit_selectors:
                try:
                    logger.debug(f"[DEBUG] Trying selector: {selector}")
                    login_button = wait.until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                    used_button_selector = selector
                    logger.info(f"✓ Found submit button with selector: {selector}")
                    break
                except (TimeoutException, NoSuchElementException):
                    logger.debug(f"[DEBUG] Selector '{selector}' not found, trying next...")
                    continue

            if not login_button:
                logger.error("❌ Could not find submit button with any selector")
                self._save_debug_screenshot("login_button_not_found")
                self._save_page_source("login_button_not_found")
                return False

            # Click login button
            login_button.click()
            logger.info(f"✓ Clicked login button (button: {used_button_selector})")

            # Wait for page to process login
            logger.debug("[DEBUG] Waiting for login to process...")
            time.sleep(3)

            logger.info("✓ Login form submitted successfully")
            return True

        except TimeoutException:
            logger.error("❌ Timeout waiting for login form elements")
            logger.debug(f"[DEBUG] Current URL: {self.driver.current_url}")
            self._save_debug_screenshot("login_form_timeout")
            self._save_page_source("login_form_timeout")
            return False

        except Exception as e:
            logger.error(f"❌ Error during login: {e}")
            logger.debug(f"[DEBUG] Exception type: {type(e).__name__}")
            logger.debug(f"[DEBUG] Current URL: {self.driver.current_url}")
            self._save_debug_screenshot("login_form_error")
            self._save_page_source("login_form_error")
            return False

    def handle_2fa(self) -> bool:
        """
        Handle 2FA prompt by requesting OTP from user via terminal.

        Returns:
            True if 2FA completed successfully, False otherwise
        """
        try:
            logger.info("Waiting for 2FA prompt...")

            wait = WebDriverWait(self.driver, self.config.get('login_timeout', 30))

            # Wait for OTP input field to appear
            otp_input_selector = self.config['otp_input']
            logger.debug(f"Looking for OTP input: {otp_input_selector}")

            otp_input = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, otp_input_selector))
            )

            logger.info("2FA prompt detected")

            # Prompt user for OTP code
            print("\n" + "=" * 80)
            print("TWO-FACTOR AUTHENTICATION REQUIRED")
            print("=" * 80)
            print("\nPlease check your authentication method (email, SMS, app) for the OTP code.")
            print(f"You have {self.config.get('otp_timeout', 120)} seconds to enter the code.\n")

            otp_code = input("Enter OTP code: ").strip()

            if not otp_code:
                logger.error("No OTP code provided")
                return False

            # Enter OTP code
            otp_input.clear()
            otp_input.send_keys(otp_code)
            logger.info("Entered OTP code")

            random_delay(0.5, 1.0)

            # Check "remember browser" checkbox
            try:
                remember_checkbox_selector = self.config['remember_checkbox']
                remember_checkbox = self.driver.find_element(By.CSS_SELECTOR, remember_checkbox_selector)

                if not remember_checkbox.is_selected():
                    remember_checkbox.click()
                    logger.info("Checked 'remember browser' checkbox")
                else:
                    logger.debug("'Remember browser' already checked")

            except NoSuchElementException:
                logger.warning("Could not find 'remember browser' checkbox - continuing anyway")

            random_delay(0.5, 1.0)

            # Click continue/verify button
            otp_button_selector = self.config['otp_continue_button']
            otp_button = wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, otp_button_selector))
            )

            otp_button.click()
            logger.info("Clicked OTP continue button")

            # Wait for 2FA to process
            time.sleep(5)

            print("\n2FA verification submitted successfully!")
            print("=" * 80 + "\n")

            return True

        except TimeoutException:
            logger.error("Timeout waiting for 2FA elements or user input")
            return False

        except Exception as e:
            logger.error(f"Error during 2FA: {e}")
            return False

    def _save_debug_screenshot(self, name: str) -> None:
        """
        Save a screenshot for debugging purposes.

        Args:
            name: Name identifier for the screenshot
        """
        try:
            # Import Path for settings
            from pathlib import Path
            from config import Settings

            timestamp = int(time.time())
            filename = f"{name}_{timestamp}.png"
            filepath = Settings.LOGS_DIR / filename

            self.driver.save_screenshot(str(filepath))
            logger.info(f"📸 Screenshot saved: {filepath}")

        except Exception as e:
            logger.warning(f"Failed to save screenshot: {e}")

    def _save_page_source(self, name: str) -> None:
        """
        Save the current page HTML source for debugging.

        Args:
            name: Name identifier for the HTML file
        """
        try:
            from pathlib import Path
            from config import Settings

            timestamp = int(time.time())
            filename = f"{name}_{timestamp}.html"
            filepath = Settings.LOGS_DIR / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.driver.page_source)

            logger.info(f"📄 Page source saved: {filepath}")

        except Exception as e:
            logger.warning(f"Failed to save page source: {e}")

    def _log_page_inputs(self) -> None:
        """
        Log all input fields found on the current page for debugging.
        """
        try:
            logger.debug("[DEBUG] === Inspecting page input fields ===")

            # Find all input elements
            inputs = self.driver.find_elements(By.TAG_NAME, 'input')
            logger.debug(f"[DEBUG] Found {len(inputs)} input elements:")

            for i, inp in enumerate(inputs[:10], 1):  # Limit to first 10
                input_type = inp.get_attribute('type') or 'text'
                input_name = inp.get_attribute('name') or 'N/A'
                input_id = inp.get_attribute('id') or 'N/A'
                input_placeholder = inp.get_attribute('placeholder') or 'N/A'

                logger.debug(f"[DEBUG]   {i}. type={input_type}, name={input_name}, "
                           f"id={input_id}, placeholder={input_placeholder}")

            # Find all buttons
            buttons = self.driver.find_elements(By.TAG_NAME, 'button')
            logger.debug(f"[DEBUG] Found {len(buttons)} button elements:")

            for i, btn in enumerate(buttons[:5], 1):  # Limit to first 5
                btn_type = btn.get_attribute('type') or 'N/A'
                btn_name = btn.get_attribute('name') or 'N/A'
                btn_text = btn.text or 'N/A'

                logger.debug(f"[DEBUG]   {i}. type={btn_type}, name={btn_name}, text={btn_text}")

            logger.debug("[DEBUG] === End page inspection ===")

        except Exception as e:
            logger.debug(f"[DEBUG] Failed to inspect page inputs: {e}")

    def navigate_to_remine(self) -> bool:
        """
        Navigate from dashboard to Remine daily page.

        Returns:
            True if navigation successful, False otherwise
        """
        try:
            logger.info("Navigating to Remine product...")

            wait = WebDriverWait(self.driver, self.config.get('login_timeout', 30))

            # Find and click Remine product link
            remine_link_selector = self.config['remine_product_link']
            logger.debug(f"Looking for Remine link: {remine_link_selector}")

            remine_link = wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, remine_link_selector))
            )

            # Store current window handle
            original_window = self.driver.current_window_handle

            # Click the link (opens in new tab)
            remine_link.click()
            logger.info("Clicked Remine product link")

            # Wait for new tab to open
            time.sleep(3)

            # Switch to new tab
            all_windows = self.driver.window_handles

            if len(all_windows) > 1:
                # Find the new window
                for window in all_windows:
                    if window != original_window:
                        self.driver.switch_to.window(window)
                        logger.debug("Switched to new tab")
                        break

            # Wait for Remine page to load
            time.sleep(3)

            # Check if we're on the daily page
            current_url = self.driver.current_url
            expected_url = self.config['remine_daily_url']

            if expected_url in current_url or 'fmls.remine.com' in current_url:
                logger.info(f"Successfully navigated to Remine (URL: {current_url})")
                return True
            else:
                logger.warning(f"Unexpected URL after Remine navigation: {current_url}")
                return True  # Still return True as we might be on a Remine page

        except TimeoutException:
            logger.error("Timeout waiting for Remine link")
            return False

        except Exception as e:
            logger.error(f"Error navigating to Remine: {e}")
            return False

    def authenticate(self) -> bool:
        """
        Complete authentication flow.

        Returns:
            True if authentication successful, False otherwise
        """
        try:
            logger.info("=" * 80)
            logger.info("Starting FMLS authentication process")
            logger.info("=" * 80)

            # DEBUG: Log configuration being used
            logger.debug("=== [DEBUG] Authentication Configuration ===")
            logger.debug(f"Home URL: {self.config.get('home_url')}")
            logger.debug(f"Login URL: {self.config.get('login_url')}")
            logger.debug(f"Dashboard URL: {self.config.get('dashboard_url')}")
            logger.debug(f"Remine Daily URL: {self.config.get('remine_daily_url')}")
            logger.debug(f"GCP Project ID: {self.secret_client.project_id}")
            logger.debug(f"Login Secret: {self.login_secret_name}")
            logger.debug(f"Password Secret: {self.password_secret_name}")
            logger.debug("=== [END DEBUG] ===\n")

            # Step 1: Try to load saved session
            # DEBUG SESSION PERSISTENCE: Log session check start
            logger.info("=" * 80)
            logger.info("[DEBUG SESSION] STEP 1: CHECKING FOR SAVED SESSION")
            logger.info("=" * 80)
            # END DEBUG SESSION PERSISTENCE

            logger.info("Step 1: Checking for saved session...")

            # DEBUG SESSION PERSISTENCE: Check session validity first
            session_is_valid = self.session_manager.is_session_valid()
            logger.info(f"[DEBUG SESSION] Session validity result: {session_is_valid}")
            # END DEBUG SESSION PERSISTENCE

            if session_is_valid:
                logger.info("Found valid saved session - attempting to load cookies...")

                # DEBUG SESSION PERSISTENCE: Load cookies for EACH domain where they exist
                logger.info(f"[DEBUG SESSION] === MULTI-DOMAIN COOKIE LOADING STRATEGY ===")

                # Get cookie info to see what domains we have
                cookie_info = self.session_manager.get_cookie_info()
                if cookie_info:
                    import json
                    with open(self.session_manager.cookie_file_path, 'r') as f:
                        cookie_data = json.load(f)
                    saved_cookies = cookie_data.get('cookies', [])

                    # Get unique domains
                    cookie_domains = list(set(c.get('domain', '').lstrip('.') for c in saved_cookies if c.get('domain')))
                    logger.info(f"[DEBUG SESSION] Found cookies for {len(cookie_domains)} domains: {cookie_domains}")

                    # Load cookies for each domain
                    total_loaded = 0
                    for domain in cookie_domains:
                        logger.info(f"[DEBUG SESSION] --- Loading cookies for domain: {domain} ---")

                        # Construct URL for this domain
                        domain_url = f"https://{domain}"
                        logger.info(f"[DEBUG SESSION] Navigating to: {domain_url}")

                        try:
                            self.driver.get(domain_url)
                            time.sleep(1)

                            logger.info(f"[DEBUG SESSION] Current URL: {self.driver.current_url}")

                            # Now add cookies for this domain
                            domain_cookies = [c for c in saved_cookies if domain in c.get('domain', '')]
                            logger.info(f"[DEBUG SESSION] Adding {len(domain_cookies)} cookie(s) for this domain")

                            for cookie in domain_cookies:
                                try:
                                    cookie_copy = cookie.copy()
                                    cookie_copy.pop('sameSite', None)
                                    cookie_copy.pop('expiry', None)
                                    self.driver.add_cookie(cookie_copy)
                                    total_loaded += 1
                                    logger.debug(f"[DEBUG SESSION] ✓ Added cookie: {cookie.get('name')}")
                                except Exception as e:
                                    logger.debug(f"[DEBUG SESSION] ✗ Failed to add cookie {cookie.get('name')}: {e}")

                        except Exception as e:
                            logger.warning(f"[DEBUG SESSION] Failed to navigate to {domain_url}: {e}")

                    logger.info(f"[DEBUG SESSION] Total cookies loaded across all domains: {total_loaded}")
                    cookie_load_result = total_loaded > 0
                else:
                    logger.warning(f"[DEBUG SESSION] No cookie info available")
                    cookie_load_result = False

                # END DEBUG SESSION PERSISTENCE

                logger.info(f"[DEBUG SESSION] Overall cookie load result: {cookie_load_result}")

                if cookie_load_result:
                    logger.info("Cookies loaded - verifying authentication status...")

                    # DEBUG SESSION PERSISTENCE: Navigate to dashboard to check auth
                    logger.info(f"[DEBUG SESSION] Navigating to dashboard to verify authentication...")
                    dashboard_url = self.config['dashboard_url']
                    logger.info(f"[DEBUG SESSION] Dashboard URL: {dashboard_url}")
                    self.driver.get(dashboard_url)
                    time.sleep(3)
                    logger.info(f"[DEBUG SESSION] Current URL after navigation: {self.driver.current_url}")
                    # END DEBUG SESSION PERSISTENCE

                    # Check if session is still valid
                    # DEBUG SESSION PERSISTENCE: Check authentication
                    logger.info(f"[DEBUG SESSION] Checking if cookies provide valid authentication...")
                    auth_check_result = self.is_authenticated()
                    logger.info(f"[DEBUG SESSION] Authentication check result: {auth_check_result}")
                    # END DEBUG SESSION PERSISTENCE

                    if auth_check_result:
                        logger.info("✓ Successfully authenticated using saved session")

                        # Navigate to Remine
                        if self.navigate_to_remine():
                            logger.info("✓ Successfully navigated to Remine dashboard")
                            # DEBUG SESSION PERSISTENCE: Success path
                            logger.info(f"[DEBUG SESSION] ✓✓✓ SESSION PERSISTENCE SUCCESSFUL - No 2FA needed! ✓✓✓")
                            # END DEBUG SESSION PERSISTENCE
                            return True
                        else:
                            logger.warning("Failed to navigate to Remine with saved session - will try fresh login")
                    else:
                        # DEBUG SESSION PERSISTENCE: Failed authentication
                        logger.warning(f"[DEBUG SESSION] Loaded cookies did not provide valid authentication")
                        logger.warning(f"[DEBUG SESSION] Will proceed with fresh login and 2FA")
                        # END DEBUG SESSION PERSISTENCE
                else:
                    logger.info("Failed to load cookies - will perform fresh login")
                    # DEBUG SESSION PERSISTENCE: Failed cookie loading
                    logger.info(f"[DEBUG SESSION] Cookie loading failed - cookies may be corrupted or incompatible")
                    # END DEBUG SESSION PERSISTENCE
            else:
                logger.info("No valid saved session found - will perform fresh login")
                # DEBUG SESSION PERSISTENCE: No valid session
                logger.info(f"[DEBUG SESSION] Reason: Session file doesn't exist, expired, or invalid")
                # END DEBUG SESSION PERSISTENCE

            # Step 2: Check if already authenticated (without cookies)
            logger.info("\nStep 2: Checking if already authenticated...")
            if self.is_authenticated():
                logger.info("Already authenticated - skipping login")

                # Navigate to Remine
                if self.navigate_to_remine():
                    logger.info("✓ Successfully navigated to Remine dashboard")
                    return True

            # Step 3: Perform fresh login
            logger.info("\nStep 3: Performing fresh login...")

            # Retrieve credentials from Secret Manager
            logger.info("Step 3.1: Retrieving credentials from Google Secret Manager...")
            logger.debug(f"[DEBUG] Attempting to access project: {self.secret_client.project_id}")
            logger.debug(f"[DEBUG] Login secret: projects/{self.secret_client.project_id}/secrets/{self.login_secret_name}/versions/latest")
            logger.debug(f"[DEBUG] Password secret: projects/{self.secret_client.project_id}/secrets/{self.password_secret_name}/versions/latest")

            credentials = self.secret_client.get_credentials(
                self.login_secret_name,
                self.password_secret_name
            )

            if not credentials:
                logger.error("❌ Failed to retrieve credentials from Secret Manager")
                logger.error("Check:")
                logger.error("  1. GOOGLE_APPLICATION_CREDENTIALS environment variable is set")
                logger.error("  2. Service account has 'Secret Manager Secret Accessor' role")
                logger.error("  3. Secret names are correct in configuration")
                logger.error("  4. GCP project ID is correct")
                return False

            logger.info("✓ Credentials retrieved successfully")

            # Navigate to login page
            logger.info("\nStep 3.2: Navigating to login page...")
            if not self.navigate_to_login():
                logger.error("❌ Failed to navigate to login page")
                return False

            logger.info("✓ Reached login page")

            # Perform login
            logger.info("\nStep 3.3: Submitting login credentials...")
            if not self.perform_login(credentials['login_id'], credentials['password']):
                logger.error("❌ Failed to submit login form")
                return False

            logger.info("✓ Login form submitted")

            # Handle 2FA
            logger.info("\nStep 3.4: Handling 2FA...")
            if not self.handle_2fa():
                logger.error("❌ Failed to complete 2FA")
                return False

            logger.info("✓ 2FA completed")

            # Verify we reached dashboard
            logger.info("\nStep 3.5: Verifying dashboard access...")
            time.sleep(3)
            if not self.is_authenticated():
                logger.error("❌ Login appeared successful but dashboard not accessible")
                logger.debug(f"[DEBUG] Current URL: {self.driver.current_url}")
                logger.debug(f"[DEBUG] Page title: {self.driver.title}")
                return False

            logger.info("✓ Dashboard accessible")

            # DEBUG SESSION PERSISTENCE: CRITICAL FIX V3 - Save cookies at BOTH locations
            logger.info("\n[DEBUG SESSION] === DUAL COOKIE SAVE STRATEGY ===")
            logger.info("[DEBUG SESSION] Problem: SSO cookies exist only on dashboard domain")
            logger.info("[DEBUG SESSION] Solution: Save cookies at dashboard AND after Remine")
            logger.info("[DEBUG SESSION] This captures BOTH SSO cookies AND Remine session cookies")
            # END DEBUG SESSION PERSISTENCE

            # SAVE POINT 1: Capture SSO/Dashboard cookies
            logger.info("\nStep 3.6a: Saving SSO/Dashboard cookies...")
            logger.info(f"[DEBUG SESSION] Current URL (dashboard): {self.driver.current_url}")
            sso_cookies = self.driver.get_cookies()
            sso_domains = set(c.get('domain', 'unknown') for c in sso_cookies)
            logger.info(f"[DEBUG SESSION] SSO cookies count: {len(sso_cookies)}")
            logger.info(f"[DEBUG SESSION] SSO domains: {', '.join(sorted(sso_domains))}")

            # Store SSO cookies temporarily
            import json
            sso_cookie_list = sso_cookies.copy()
            logger.info(f"[DEBUG SESSION] ✓ Stored {len(sso_cookie_list)} SSO cookies for later merge")
            # END DEBUG SESSION PERSISTENCE

            # Navigate to Remine
            logger.info("\nStep 3.6b: Navigating to Remine product...")
            if not self.navigate_to_remine():
                logger.error("❌ Failed to navigate to Remine dashboard")
                return False

            logger.info("✓ Successfully navigated to Remine")

            # SAVE POINT 2: Capture Remine session cookies
            logger.info("\nStep 3.7: Saving combined cookies (SSO + Remine)...")
            logger.info(f"[DEBUG SESSION] Current URL (Remine): {self.driver.current_url}")
            remine_cookies = self.driver.get_cookies()
            remine_domains = set(c.get('domain', 'unknown') for c in remine_cookies)
            logger.info(f"[DEBUG SESSION] Remine cookies count: {len(remine_cookies)}")
            logger.info(f"[DEBUG SESSION] Remine domains: {', '.join(sorted(remine_domains))}")

            # CRITICAL: Merge SSO and Remine cookies
            logger.info(f"[DEBUG SESSION] === MERGING COOKIES ===")
            logger.info(f"[DEBUG SESSION] SSO cookies: {len(sso_cookie_list)}")
            logger.info(f"[DEBUG SESSION] Remine cookies: {len(remine_cookies)}")

            # Create a dict to track unique cookies by name+domain
            merged_cookies = {}
            for cookie in sso_cookie_list + remine_cookies:
                key = (cookie.get('name'), cookie.get('domain'))
                merged_cookies[key] = cookie

            final_cookie_list = list(merged_cookies.values())
            logger.info(f"[DEBUG SESSION] Merged total: {len(final_cookie_list)} unique cookies")

            # Show all domains in final set
            all_domains = set(c.get('domain', 'unknown') for c in final_cookie_list)
            logger.info(f"[DEBUG SESSION] Final cookie domains: {', '.join(sorted(all_domains))}")

            # Check for critical SSO domain
            has_sso = any('firstmls.sso.remine.com' in c.get('domain', '') for c in final_cookie_list)
            has_remine = any('remine.com' in c.get('domain', '') for c in final_cookie_list)
            logger.info(f"[DEBUG SESSION] Has SSO cookies: {has_sso}")
            logger.info(f"[DEBUG SESSION] Has Remine cookies: {has_remine}")

            if not has_sso:
                logger.warning(f"[DEBUG SESSION] ⚠️ WARNING: No SSO cookies found! Auth check will fail!")
            if not has_remine:
                logger.warning(f"[DEBUG SESSION] ⚠️ WARNING: No Remine cookies found!")
            # END DEBUG SESSION PERSISTENCE

            # Save merged cookie list
            logger.info(f"[DEBUG SESSION] Saving {len(final_cookie_list)} merged cookies to file...")

            # Manually save merged cookies
            cookie_data = {
                'saved_at': datetime.now().isoformat(),
                'domain': None,
                'cookie_count': len(final_cookie_list),
                'cookies': final_cookie_list
            }

            cookie_file = self.session_manager.cookie_file_path
            with open(cookie_file, 'w') as f:
                json.dump(cookie_data, f, indent=2)

            logger.info(f"[DEBUG SESSION] ✓ Saved merged cookies to: {cookie_file}")
            logger.info("✓ Cookies saved")

            logger.info("=" * 80)
            logger.info("✓ AUTHENTICATION SUCCESSFUL")
            logger.info("=" * 80)

            return True

        except Exception as e:
            logger.error(f"❌ Authentication failed with unexpected error: {e}", exc_info=True)
            logger.debug(f"[DEBUG] Current URL at error: {self.driver.current_url if self.driver else 'N/A'}")
            return False


def create_authenticator(
    driver: webdriver.Chrome,
    cookie_dir: Path,
    gcp_project_id: str,
    login_secret: str,
    password_secret: str,
    config: Dict
) -> FMLSAuthenticator:
    """
    Helper function to create an FMLSAuthenticator instance.

    Args:
        driver: Selenium WebDriver instance
        cookie_dir: Directory for cookie storage
        gcp_project_id: Google Cloud project ID
        login_secret: Name of login secret
        password_secret: Name of password secret
        config: Configuration dictionary

    Returns:
        FMLSAuthenticator instance
    """
    from .session_manager import create_session_manager

    session_manager = create_session_manager(cookie_dir, session_name="fmls_session")

    return FMLSAuthenticator(
        driver=driver,
        session_manager=session_manager,
        gcp_project_id=gcp_project_id,
        login_secret_name=login_secret,
        password_secret_name=password_secret,
        config=config
    )
