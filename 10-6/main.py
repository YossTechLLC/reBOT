#!/usr/bin/env python3
"""
Property Scraper Application - Main Entry Point

This application automates the process of searching property addresses,
extracting owner and buyer information, and exporting results to Excel.

Usage:
    python main.py
"""

import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from config import Settings
from src import DatabaseManager, ExcelHandler, PropertyScraper, setup_logger, create_authenticator
from src.utils import Timer, print_progress_bar

logger = None


def initialize_application():
    """
    Initialize the application by setting up logging and validating configuration.

    Returns:
        bool: True if initialization successful, False otherwise
    """
    global logger

    try:
        # Setup logging
        logger = setup_logger(
            name=__name__,
            log_file=Settings.LOG_FILE,
            level=Settings.LOG_LEVEL,
            log_format=Settings.LOG_FORMAT,
            date_format=Settings.LOG_DATE_FORMAT
        )

        logger.info("=" * 80)
        logger.info("Property Scraper Application Starting")
        logger.info("=" * 80)

        # Validate configuration
        Settings.validate()

        # Display configuration
        config = Settings.display_config()
        logger.info("Configuration:")
        for key, value in config.items():
            logger.info(f"  {key}: {value}")

        return True

    except Exception as e:
        if logger:
            logger.error(f"Initialization failed: {e}")
        else:
            print(f"ERROR: Initialization failed: {e}")
        return False


def authenticate_fmls(scraper: PropertyScraper) -> bool:
    """
    Authenticate with FMLS if authentication is enabled.

    Args:
        scraper: PropertyScraper instance with initialized driver

    Returns:
        True if authentication successful or not required, False otherwise
    """
    if not Settings.ENABLE_FMLS_AUTH:
        logger.info("FMLS authentication disabled - skipping")
        return True

    try:
        logger.info("\n" + "=" * 80)
        logger.info("FMLS AUTHENTICATION")
        logger.info("=" * 80)

        # DEBUG: Verify scraper has driver
        logger.debug(f"[DEBUG] Scraper driver initialized: {scraper.driver is not None}")
        if scraper.driver:
            logger.debug(f"[DEBUG] Current browser URL: {scraper.driver.current_url}")

        # Create authenticator
        logger.debug("[DEBUG] Creating authenticator instance...")
        authenticator = create_authenticator(
            driver=scraper.driver,
            cookie_dir=Settings.COOKIES_DIR,
            gcp_project_id=Settings.GCP_PROJECT_ID,
            login_secret=Settings.SECRET_LOGIN_ID,
            password_secret=Settings.SECRET_PASSWORD,
            config=Settings.get_fmls_config()
        )
        logger.debug("[DEBUG] Authenticator created successfully")

        # Perform authentication
        logger.info("Starting authentication process...")
        if authenticator.authenticate():
            logger.info("✓ FMLS authentication completed successfully")
            return True
        else:
            logger.error("❌ FMLS authentication failed")
            logger.error("Review the logs above for specific error details")
            return False

    except Exception as e:
        logger.error(f"❌ Unexpected error during FMLS authentication: {e}", exc_info=True)
        logger.debug(f"[DEBUG] Exception type: {type(e).__name__}")
        return False


def validate_input_file():
    """
    Validate the input Excel file exists and has the correct structure.

    Returns:
        tuple: (bool, list) - Success status and list of addresses
    """
    logger.info("\nValidating input file...")

    # Check if input file exists
    if not Settings.INPUT_EXCEL.exists():
        logger.error(f"Input file not found: {Settings.INPUT_EXCEL}")
        logger.info("Creating sample input file...")

        try:
            ExcelHandler.create_sample_input(
                output_path=Settings.INPUT_EXCEL,
                address_column=Settings.ADDRESS_COLUMN
            )
            logger.info(f"Sample input file created at: {Settings.INPUT_EXCEL}")
            logger.info("Please populate it with addresses and run the application again.")
            return False, []
        except Exception as e:
            logger.error(f"Failed to create sample input file: {e}")
            return False, []

    # Validate file structure
    validation_result = ExcelHandler.validate_input_file(
        file_path=Settings.INPUT_EXCEL,
        address_column=Settings.ADDRESS_COLUMN,
        sheet_name=Settings.EXCEL_SHEET_NAME
    )

    if not validation_result['valid']:
        logger.error("Input file validation failed:")
        for error in validation_result['errors']:
            logger.error(f"  - {error}")
        return False, []

    # Read addresses
    try:
        addresses = ExcelHandler.read_addresses(
            file_path=Settings.INPUT_EXCEL,
            address_column=Settings.ADDRESS_COLUMN,
            sheet_name=Settings.EXCEL_SHEET_NAME
        )

        logger.info(f"✓ Input file validated successfully")
        logger.info(f"✓ Found {len(addresses)} addresses to process")

        return True, addresses

    except Exception as e:
        logger.error(f"Error reading addresses from file: {e}")
        return False, []


def process_addresses(addresses: list, db_manager: DatabaseManager):
    """
    Process all addresses by scraping data and storing results.

    Args:
        addresses: List of addresses to process
        db_manager: Database manager instance

    Returns:
        list: List of scrape results
    """
    results = []
    total_addresses = len(addresses)

    logger.info(f"\nStarting to process {total_addresses} addresses...")
    logger.info("=" * 80)

    # Initialize scraper
    with PropertyScraper(
        target_url=Settings.TARGET_URL,
        headless=Settings.HEADLESS_MODE,
        window_size=Settings.BROWSER_WINDOW_SIZE,
        user_agent=Settings.USER_AGENT,
        page_load_timeout=Settings.PAGE_LOAD_TIMEOUT,
        element_wait_timeout=Settings.ELEMENT_WAIT_TIMEOUT
    ) as scraper:

        for index, address in enumerate(addresses, start=1):
            logger.info(f"\n[{index}/{total_addresses}] Processing: {address}")

            # Check if already processed (if skip feature is enabled)
            if Settings.SKIP_PROCESSED_ADDRESSES:
                if db_manager.is_address_processed(address):
                    logger.info(f"  ⏭️  Address already processed, skipping...")
                    # Get existing result from database
                    existing_result = db_manager.get_result_by_address(address)
                    if existing_result:
                        results.append(existing_result)
                    print_progress_bar(index, total_addresses, prefix='Progress', suffix='Complete')
                    continue

            # Scrape address
            result = scraper.scrape_address(
                address=address,
                search_input_selector=Settings.SEARCH_INPUT_SELECTOR,
                search_button_selector=Settings.SEARCH_BUTTON_SELECTOR,
                owner_patterns=Settings.OWNER_PATTERNS,
                buyer_patterns=Settings.BUYER_PATTERNS,
                follow_first_result=Settings.FOLLOW_FIRST_RESULT,
                first_result_selector=Settings.FIRST_RESULT_SELECTOR,
                save_html=Settings.SAVE_HTML_SNAPSHOTS,
                min_delay=Settings.MIN_DELAY_BETWEEN_REQUESTS,
                max_delay=Settings.MAX_DELAY_BETWEEN_REQUESTS
            )

            # Log result
            if result['status'] == 'success':
                logger.info(f"  ✓ Owner: {result['owner']}")
                logger.info(f"  ✓ Buyer: {result['buyer']}")
            elif result['status'] == 'no_results':
                logger.warning(f"  ⚠️  No results found")
            else:
                logger.error(f"  ✗ Failed: {result['error_message']}")

            # Store in database
            try:
                db_manager.insert_result(
                    address=result['address'],
                    owner=result['owner'],
                    buyer=result['buyer'],
                    raw_html=result['raw_html'],
                    status=result['status'],
                    error_message=result['error_message'],
                    timestamp=result['timestamp']
                )
            except Exception as e:
                logger.error(f"  ✗ Database error: {e}")

            # Add to results list (don't include raw HTML in Excel output)
            result_for_excel = {k: v for k, v in result.items() if k != 'raw_html'}
            results.append(result_for_excel)

            # Progress bar
            print_progress_bar(index, total_addresses, prefix='Progress', suffix='Complete')

    logger.info("\n" + "=" * 80)
    logger.info("Address processing completed")

    return results


def export_results(results: list):
    """
    Export results to Excel file.

    Args:
        results: List of scrape results

    Returns:
        bool: True if export successful, False otherwise
    """
    if not results:
        logger.warning("No results to export")
        return False

    try:
        logger.info(f"\nExporting results to {Settings.OUTPUT_EXCEL}...")

        ExcelHandler.write_results(
            results=results,
            output_path=Settings.OUTPUT_EXCEL,
            include_timestamp=True
        )

        logger.info(f"✓ Results exported successfully to {Settings.OUTPUT_EXCEL}")
        return True

    except Exception as e:
        logger.error(f"Failed to export results: {e}")
        return False


def display_summary(db_manager: DatabaseManager):
    """
    Display summary statistics of the scraping operation.

    Args:
        db_manager: Database manager instance
    """
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    try:
        stats = db_manager.get_statistics()

        logger.info(f"Total addresses processed: {stats['total']}")
        logger.info(f"  ✓ Successful: {stats['success']}")
        logger.info(f"  ⚠️  No results: {stats['no_results']}")
        logger.info(f"  ✗ Failed: {stats['failed']}")

        if stats['total'] > 0:
            success_rate = (stats['success'] / stats['total']) * 100
            logger.info(f"\nSuccess rate: {success_rate:.1f}%")

    except Exception as e:
        logger.error(f"Error generating summary: {e}")


def main():
    """Main application entry point."""
    exit_code = 0
    scraper = None

    try:
        # Initialize application
        if not initialize_application():
            return 1

        # If FMLS auth is enabled, we need to authenticate first
        if Settings.ENABLE_FMLS_AUTH:
            logger.info("\n" + "=" * 80)
            logger.info("FMLS authentication is enabled")
            logger.info("Initializing browser for authentication...")
            logger.info("=" * 80)

            # Initialize scraper (which starts the browser)
            scraper = PropertyScraper(
                target_url=Settings.TARGET_URL,
                headless=Settings.HEADLESS_MODE,
                window_size=Settings.BROWSER_WINDOW_SIZE,
                user_agent=Settings.USER_AGENT,
                page_load_timeout=Settings.PAGE_LOAD_TIMEOUT,
                element_wait_timeout=Settings.ELEMENT_WAIT_TIMEOUT
            )
            scraper.start()

            # Authenticate with FMLS
            if not authenticate_fmls(scraper):
                logger.error("Failed to authenticate with FMLS - cannot continue")
                return 1

            logger.info("\n✓ Ready to begin scraping operations\n")

        # Validate input file and get addresses
        success, addresses = validate_input_file()
        if not success or not addresses:
            return 1

        # Initialize database
        with DatabaseManager(
            db_path=Settings.DATABASE_PATH,
            timeout=Settings.DB_TIMEOUT
        ) as db_manager:

            # If we didn't create a scraper for auth, create it now
            if scraper is None:
                # Process addresses with timer
                with Timer("Total processing time", logger):
                    results = process_addresses(addresses, db_manager)
            else:
                # Use existing authenticated scraper
                # For now, we'll just display success message
                # The actual scraping implementation will be in your next prompt
                logger.info("\n" + "=" * 80)
                logger.info("AUTHENTICATION TEST COMPLETED")
                logger.info("=" * 80)
                logger.info("\n✓ Successfully authenticated with FMLS")
                logger.info("✓ Browser session is active and ready")
                logger.info("✓ All authentication systems working correctly")
                logger.info("\nNext steps: Implement FMLS-specific scraping logic")
                logger.info("=" * 80)

                # For now, return empty results
                results = []

            # Export results to Excel (if any)
            if results:
                export_results(results)

            # Display summary (if any results)
            if results:
                display_summary(db_manager)

        logger.info("\n" + "=" * 80)
        logger.info("Application completed successfully!")
        logger.info("=" * 80)

    except KeyboardInterrupt:
        logger.warning("\n\nApplication interrupted by user (Ctrl+C)")
        exit_code = 130

    except Exception as e:
        logger.error(f"\n\nUnexpected error: {e}", exc_info=True)
        exit_code = 1

    finally:
        # Clean up scraper if it was created
        if scraper:
            try:
                scraper.stop()
                logger.info("Browser session closed")
            except Exception as e:
                logger.warning(f"Error closing browser: {e}")

        if logger:
            logger.info(f"Log file saved to: {Settings.LOG_FILE}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
