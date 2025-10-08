#!/usr/bin/env python3
"""
Test script for FMLS authentication.
Tests the complete authentication flow without requiring address scraping.
"""

import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from config import Settings
from src import PropertyScraper, create_authenticator, setup_logger

def test_fmls_authentication():
    """Test FMLS authentication flow."""

    # Setup logging
    logger = setup_logger(
        name=__name__,
        log_file=Settings.LOG_FILE,
        level=Settings.LOG_LEVEL
    )

    logger.info("=" * 80)
    logger.info("FMLS AUTHENTICATION TEST")
    logger.info("=" * 80)

    scraper = None

    try:
        # Validate configuration
        logger.info("\n1. Validating configuration...")
        Settings.validate()

        config = Settings.display_config()
        logger.info("Configuration:")
        for key, value in config.items():
            logger.info(f"  {key}: {value}")

        # Initialize browser
        logger.info("\n2. Initializing browser...")
        scraper = PropertyScraper(
            target_url=Settings.TARGET_URL,
            headless=Settings.HEADLESS_MODE,
            window_size=Settings.BROWSER_WINDOW_SIZE,
            user_agent=Settings.USER_AGENT,
            page_load_timeout=Settings.PAGE_LOAD_TIMEOUT,
            element_wait_timeout=Settings.ELEMENT_WAIT_TIMEOUT
        )
        scraper.start()
        logger.info("✓ Browser initialized successfully")

        # Create authenticator
        logger.info("\n3. Creating authenticator...")
        authenticator = create_authenticator(
            driver=scraper.driver,
            cookie_dir=Settings.COOKIES_DIR,
            gcp_project_id=Settings.GCP_PROJECT_ID,
            login_secret=Settings.SECRET_LOGIN_ID,
            password_secret=Settings.SECRET_PASSWORD,
            config=Settings.get_fmls_config()
        )
        logger.info("✓ Authenticator created successfully")

        # Perform authentication
        logger.info("\n4. Performing authentication...")
        if authenticator.authenticate():
            logger.info("\n" + "=" * 80)
            logger.info("✓✓✓ AUTHENTICATION TEST SUCCESSFUL ✓✓✓")
            logger.info("=" * 80)
            logger.info("\nAuthentication Details:")
            logger.info(f"  Current URL: {scraper.driver.current_url}")
            logger.info(f"  Page Title: {scraper.driver.title}")
            logger.info("\nThe browser will remain open for 10 seconds for verification...")
            logger.info("=" * 80)

            # Keep browser open for a moment to verify
            import time
            time.sleep(10)

            return True
        else:
            logger.error("\n" + "=" * 80)
            logger.error("✗✗✗ AUTHENTICATION TEST FAILED ✗✗✗")
            logger.error("=" * 80)
            return False

    except KeyboardInterrupt:
        logger.warning("\n\nTest interrupted by user (Ctrl+C)")
        return False

    except Exception as e:
        logger.error(f"\n\nTest failed with error: {e}", exc_info=True)
        return False

    finally:
        # Clean up
        if scraper:
            try:
                scraper.stop()
                logger.info("\nBrowser session closed")
            except Exception as e:
                logger.warning(f"Error closing browser: {e}")

        logger.info(f"\nLog file saved to: {Settings.LOG_FILE}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("FMLS AUTHENTICATION TEST SCRIPT")
    print("=" * 80)
    print("\nThis script will test the complete FMLS authentication flow.")
    print("You will be prompted for 2FA code during the test.")
    print("\nPress Ctrl+C at any time to cancel.")
    print("=" * 80 + "\n")

    success = test_fmls_authentication()

    sys.exit(0 if success else 1)
