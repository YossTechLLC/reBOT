"""
Web scraper module for property search automation.
Handles browser automation, page navigation, and data extraction.
"""

import logging
import re
from typing import Optional, Dict, Any, List
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    WebDriverException
)
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService

from .utils import retry_on_exception, random_delay, extract_with_regex, clean_text

logger = logging.getLogger(__name__)


class PropertyScraper:
    """Handles web scraping for property information."""

    def __init__(
        self,
        target_url: str,
        headless: bool = True,
        window_size: str = "1920,1080",
        user_agent: Optional[str] = None,
        page_load_timeout: int = 30,
        element_wait_timeout: int = 10
    ):
        """
        Initialize the property scraper.

        Args:
            target_url: Base URL for property search
            headless: Whether to run browser in headless mode
            window_size: Browser window size (width,height)
            user_agent: Custom user agent string
            page_load_timeout: Maximum time to wait for page load
            element_wait_timeout: Maximum time to wait for elements
        """
        self.target_url = target_url
        self.headless = headless
        self.window_size = window_size
        self.user_agent = user_agent
        self.page_load_timeout = page_load_timeout
        self.element_wait_timeout = element_wait_timeout
        self.driver = None

    def _setup_driver(self) -> webdriver.Chrome:
        """
        Set up and configure the Chrome WebDriver.

        Returns:
            Configured Chrome WebDriver instance
        """
        chrome_options = ChromeOptions()

        # Headless mode
        if self.headless:
            chrome_options.add_argument('--headless=new')
            chrome_options.add_argument('--disable-gpu')

        # Window size
        chrome_options.add_argument(f'--window-size={self.window_size}')

        # Additional options for stability
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')

        # User agent
        if self.user_agent:
            chrome_options.add_argument(f'user-agent={self.user_agent}')

        # Disable automation flags
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)

        # Disable images for faster loading (optional)
        # prefs = {"profile.managed_default_content_settings.images": 2}
        # chrome_options.add_experimental_option("prefs", prefs)

        try:
            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(self.page_load_timeout)

            # Execute CDP commands to prevent detection
            driver.execute_cdp_cmd('Network.setUserAgentOverride', {
                "userAgent": self.user_agent or driver.execute_script("return navigator.userAgent")
            })

            logger.info("Chrome WebDriver initialized successfully")
            return driver

        except WebDriverException as e:
            logger.error(f"Failed to initialize Chrome WebDriver: {e}")
            raise

    def start(self):
        """Start the browser session."""
        if self.driver is None:
            self.driver = self._setup_driver()
            logger.debug("Browser session started")

    def stop(self):
        """Stop the browser session and cleanup."""
        if self.driver:
            try:
                self.driver.quit()
                logger.debug("Browser session closed")
            except Exception as e:
                logger.warning(f"Error closing browser: {e}")
            finally:
                self.driver = None

    @retry_on_exception(max_retries=2, delay=3, exceptions=(TimeoutException, WebDriverException))
    def search_address(
        self,
        address: str,
        search_input_selector: str,
        search_button_selector: str,
        follow_first_result: bool = False,
        first_result_selector: Optional[str] = None
    ) -> str:
        """
        Search for an address on the target website.

        Args:
            address: Property address to search
            search_input_selector: CSS selector for search input field
            search_button_selector: CSS selector for search button
            follow_first_result: Whether to click the first result link
            first_result_selector: CSS selector for first result link

        Returns:
            Page HTML content after search

        Raises:
            TimeoutException: If page elements don't load in time
        """
        if not self.driver:
            self.start()

        try:
            # Navigate to search page
            logger.info(f"Navigating to {self.target_url}")
            self.driver.get(self.target_url)

            # Wait for search input to be present
            wait = WebDriverWait(self.driver, self.element_wait_timeout)

            logger.debug(f"Waiting for search input: {search_input_selector}")
            search_input = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, search_input_selector))
            )

            # Clear and enter address
            search_input.clear()
            search_input.send_keys(address)
            logger.info(f"Entered address: {address}")

            # Small delay to mimic human behavior
            random_delay(0.5, 1.5)

            # Click search button or press Enter
            try:
                search_button = wait.until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, search_button_selector))
                )
                search_button.click()
                logger.debug("Clicked search button")
            except (TimeoutException, NoSuchElementException):
                # Fallback to pressing Enter
                search_input.send_keys(Keys.RETURN)
                logger.debug("Pressed Enter to submit search")

            # Wait for results to load (wait for body or specific element)
            wait.until(lambda driver: driver.execute_script("return document.readyState") == "complete")
            random_delay(1, 2)

            # Optionally follow first result
            if follow_first_result and first_result_selector:
                try:
                    first_result = wait.until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, first_result_selector))
                    )
                    first_result.click()
                    logger.debug("Clicked first result link")

                    # Wait for new page to load
                    wait.until(lambda driver: driver.execute_script("return document.readyState") == "complete")
                    random_delay(1, 2)

                except (TimeoutException, NoSuchElementException):
                    logger.warning("Could not find or click first result link")

            # Get page HTML
            page_html = self.driver.page_source
            logger.info(f"Successfully retrieved page for address: {address}")

            return page_html

        except TimeoutException as e:
            logger.error(f"Timeout while searching for address '{address}': {e}")
            raise

        except Exception as e:
            logger.error(f"Error searching for address '{address}': {e}")
            raise

    def extract_owner_buyer(
        self,
        html_content: str,
        owner_patterns: List[str],
        buyer_patterns: List[str]
    ) -> Dict[str, Optional[str]]:
        """
        Extract owner and buyer information from HTML content.

        Args:
            html_content: HTML content to search
            owner_patterns: List of regex patterns for owner extraction
            buyer_patterns: List of regex patterns for buyer extraction

        Returns:
            Dictionary with 'owner' and 'buyer' keys
        """
        # Convert HTML to plain text for easier pattern matching
        from bs4 import BeautifulSoup

        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            # Get visible text
            text = soup.get_text(separator=' ', strip=True)

            # Extract owner
            owner = extract_with_regex(text, owner_patterns)
            if owner:
                owner = clean_text(owner)
                logger.debug(f"Extracted owner: {owner}")

            # Extract buyer
            buyer = extract_with_regex(text, buyer_patterns)
            if buyer:
                buyer = clean_text(buyer)
                logger.debug(f"Extracted buyer: {buyer}")

            return {
                'owner': owner,
                'buyer': buyer
            }

        except Exception as e:
            logger.error(f"Error extracting owner/buyer information: {e}")
            return {'owner': None, 'buyer': None}

    def scrape_address(
        self,
        address: str,
        search_input_selector: str,
        search_button_selector: str,
        owner_patterns: List[str],
        buyer_patterns: List[str],
        follow_first_result: bool = False,
        first_result_selector: Optional[str] = None,
        save_html: bool = True,
        min_delay: float = 2.0,
        max_delay: float = 5.0
    ) -> Dict[str, Any]:
        """
        Complete scraping workflow for a single address.

        Args:
            address: Property address to search
            search_input_selector: CSS selector for search input
            search_button_selector: CSS selector for search button
            owner_patterns: Regex patterns for owner extraction
            buyer_patterns: Regex patterns for buyer extraction
            follow_first_result: Whether to click first result
            first_result_selector: CSS selector for first result
            save_html: Whether to save HTML snapshot
            min_delay: Minimum delay after scraping (seconds)
            max_delay: Maximum delay after scraping (seconds)

        Returns:
            Dictionary with scrape results
        """
        result = {
            'address': address,
            'owner': None,
            'buyer': None,
            'raw_html': None,
            'status': 'failed',
            'error_message': None,
            'timestamp': datetime.now()
        }

        try:
            # Search for address
            html_content = self.search_address(
                address=address,
                search_input_selector=search_input_selector,
                search_button_selector=search_button_selector,
                follow_first_result=follow_first_result,
                first_result_selector=first_result_selector
            )

            # Extract owner and buyer
            extracted_data = self.extract_owner_buyer(
                html_content=html_content,
                owner_patterns=owner_patterns,
                buyer_patterns=buyer_patterns
            )

            result['owner'] = extracted_data['owner']
            result['buyer'] = extracted_data['buyer']

            # Save HTML if requested
            if save_html:
                result['raw_html'] = html_content

            # Determine status
            if result['owner'] or result['buyer']:
                result['status'] = 'success'
            else:
                result['status'] = 'no_results'
                result['error_message'] = 'No owner or buyer information found'

            logger.info(f"Successfully scraped address: {address} (status: {result['status']})")

            # Rate limiting delay
            random_delay(min_delay, max_delay)

        except TimeoutException as e:
            result['status'] = 'failed'
            result['error_message'] = f'Timeout error: {str(e)}'
            logger.error(f"Timeout scraping address '{address}': {e}")

        except WebDriverException as e:
            result['status'] = 'failed'
            result['error_message'] = f'WebDriver error: {str(e)}'
            logger.error(f"WebDriver error scraping address '{address}': {e}")

        except Exception as e:
            result['status'] = 'failed'
            result['error_message'] = f'Unexpected error: {str(e)}'
            logger.error(f"Unexpected error scraping address '{address}': {e}")

        return result

    def __enter__(self):
        """Context manager entry - start browser."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop browser."""
        self.stop()

    def __del__(self):
        """Destructor - ensure browser is closed."""
        self.stop()
