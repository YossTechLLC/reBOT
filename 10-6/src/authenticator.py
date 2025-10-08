"""
FMLS authentication module.
Handles the complete login flow including 2FA for FMLS/Remine access.
"""

import logging
import time
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

            # Try to navigate to dashboard
            dashboard_url = self.config['dashboard_url']
            self.driver.get(dashboard_url)

            # Wait a moment for page to load
            time.sleep(3)

            # Check if we're on the dashboard (not error page)
            current_url = self.driver.current_url
            page_source = self.driver.page_source.lower()

            # Check for error message
            if "oops! something's wrong" in page_source or "can't seem to find the page" in page_source:
                logger.info("Not authenticated - error page detected")
                return False

            # Check if URL contains dashboard
            if 'dashboard' in current_url:
                logger.info("Already authenticated - dashboard accessible")
                return True

            logger.info("Authentication status unclear - will attempt login")
            return False

        except Exception as e:
            logger.warning(f"Error checking authentication status: {e}")
            return False

    def navigate_to_login(self) -> bool:
        """
        Navigate directly to login page.

        Returns:
            True if navigation successful, False otherwise
        """
        try:
            logger.info("Navigating directly to FMLS login page...")

            # Go directly to login URL (skip home page)
            login_url = self.config['login_url']
            logger.debug(f"[DEBUG] Navigating to: {login_url}")

            self.driver.get(login_url)

            # Wait for page to load
            logger.debug("[DEBUG] Waiting for page to load...")
            time.sleep(3)

            # Wait for page to be ready
            wait = WebDriverWait(self.driver, self.config.get('login_timeout', 30))
            wait.until(lambda driver: driver.execute_script("return document.readyState") == "complete")

            logger.info("✓ Reached login page directly")
            logger.debug(f"[DEBUG] Current URL: {self.driver.current_url}")

            return True

        except TimeoutException:
            logger.error("Timeout waiting for login page to load")
            logger.debug(f"[DEBUG] Current URL: {self.driver.current_url}")
            return False

        except Exception as e:
            logger.error(f"Error navigating to login page: {e}")
            logger.debug(f"[DEBUG] Current URL: {self.driver.current_url}")
            return False

    def perform_login(self, login_id: str, password: str) -> bool:
        """
        Fill in login credentials and submit.

        Args:
            login_id: User login ID
            password: User password

        Returns:
            True if login form submitted successfully, False otherwise
        """
        try:
            logger.info("Filling in login credentials...")

            wait = WebDriverWait(self.driver, self.config.get('login_timeout', 30))

            # Find login ID input
            login_input_selector = self.config['login_id_input']
            logger.debug(f"Looking for login input: {login_input_selector}")

            login_input = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, login_input_selector))
            )

            # Fill in login ID
            login_input.clear()
            login_input.send_keys(login_id)
            logger.debug("Entered login ID")

            random_delay(0.5, 1.0)

            # Find password input
            password_input_selector = self.config['password_input']
            password_input = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, password_input_selector))
            )

            # Fill in password
            password_input.clear()
            password_input.send_keys(password)
            logger.debug("Entered password")

            random_delay(0.5, 1.0)

            # Click login button
            login_button_selector = self.config['login_button']
            login_button = wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, login_button_selector))
            )

            login_button.click()
            logger.info("Clicked login button")

            # Wait for page to process login
            time.sleep(3)

            return True

        except TimeoutException:
            logger.error("Timeout waiting for login form elements")
            return False

        except Exception as e:
            logger.error(f"Error during login: {e}")
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
            logger.info("Step 1: Checking for saved session...")
            if self.session_manager.is_session_valid():
                logger.info("Found valid saved session - attempting to load cookies...")

                # Navigate to domain first (required before adding cookies)
                logger.debug(f"[DEBUG] Navigating to login URL to set domain: {self.config['login_url']}")
                self.driver.get(self.config['login_url'])
                time.sleep(2)

                # Load cookies
                if self.session_manager.load_cookies(self.driver):
                    logger.info("Cookies loaded - verifying authentication status...")
                    # Check if session is still valid
                    if self.is_authenticated():
                        logger.info("✓ Successfully authenticated using saved session")

                        # Navigate to Remine
                        if self.navigate_to_remine():
                            logger.info("✓ Successfully navigated to Remine dashboard")
                            return True
                        else:
                            logger.warning("Failed to navigate to Remine with saved session - will try fresh login")
                else:
                    logger.info("Failed to load cookies - will perform fresh login")
            else:
                logger.info("No valid saved session found - will perform fresh login")

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

            # Save session cookies
            logger.info("\nStep 3.6: Saving session cookies for future use...")
            self.session_manager.save_cookies(self.driver, domain='remine.com')
            logger.info("✓ Cookies saved")

            # Navigate to Remine
            logger.info("\nStep 3.7: Navigating to Remine product...")
            if not self.navigate_to_remine():
                logger.error("❌ Failed to navigate to Remine dashboard")
                return False

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
