"""
Configuration settings for the property scraper application.
Centralized configuration management with environment variable support.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


class Settings:
    """Central configuration class for all application settings."""

    # Project paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    INPUT_DIR = DATA_DIR / "input"
    OUTPUT_DIR = DATA_DIR / "output"
    DATABASE_DIR = DATA_DIR / "database"
    LOGS_DIR = BASE_DIR / "logs"

    # Ensure directories exist
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # File paths
    INPUT_EXCEL = INPUT_DIR / os.getenv("INPUT_EXCEL_NAME", "input_addresses.xlsx")
    OUTPUT_EXCEL = OUTPUT_DIR / os.getenv("OUTPUT_EXCEL_NAME", "output_results.xlsx")
    DATABASE_PATH = DATABASE_DIR / os.getenv("DATABASE_NAME", "scrape.db")
    LOG_FILE = LOGS_DIR / "scraper.log"
    COOKIES_DIR = DATA_DIR / "cookies"

    # Ensure cookies directory exists
    COOKIES_DIR.mkdir(parents=True, exist_ok=True)

    # Excel configuration
    ADDRESS_COLUMN = os.getenv("ADDRESS_COLUMN", "address")
    EXCEL_SHEET_NAME = os.getenv("EXCEL_SHEET_NAME", "Sheet1")

    # =============================================================================
    # FMLS AUTHENTICATION CONFIGURATION
    # =============================================================================

    # Google Cloud Project ID for Secret Manager
    GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "291176869049")

    # Secret names in Google Secret Manager
    SECRET_LOGIN_ID = os.getenv("SECRET_LOGIN_ID", "reBOT_LOGIN")
    SECRET_PASSWORD = os.getenv("SECRET_PASSWORD", "reBOT_PASSWORD")

    # FMLS URLs
    FMLS_HOME_URL = os.getenv("FMLS_HOME_URL", "https://firstmls.com/")
    FMLS_LOGIN_URL = os.getenv("FMLS_LOGIN_URL", "https://firstmls.sso.remine.com")
    FMLS_DASHBOARD_URL = os.getenv("FMLS_DASHBOARD_URL", "https://firstmls.sso.remine.com/dashboard-v2")
    FMLS_REMINE_URL = os.getenv("FMLS_REMINE_URL", "https://fmls.remine.com")
    FMLS_REMINE_DAILY_URL = os.getenv("FMLS_REMINE_DAILY_URL", "https://fmls.remine.com/daily")

    # Authentication timeouts (in seconds)
    OTP_TIMEOUT = int(os.getenv("OTP_TIMEOUT", "120"))
    LOGIN_TIMEOUT = int(os.getenv("LOGIN_TIMEOUT", "30"))

    # FMLS CSS Selectors
    FMLS_LOGIN_LINK_SELECTOR = os.getenv(
        "FMLS_LOGIN_LINK_SELECTOR",
        'a[href="https://firstmls.sso.remine.com"]'
    )
    FMLS_LOGIN_ID_INPUT = os.getenv("FMLS_LOGIN_ID_INPUT", "input[name='username']")
    FMLS_PASSWORD_INPUT = os.getenv("FMLS_PASSWORD_INPUT", "input[name='password']")
    FMLS_LOGIN_BUTTON = os.getenv("FMLS_LOGIN_BUTTON", "button#btn-login")
    FMLS_OTP_INPUT = os.getenv("FMLS_OTP_INPUT", "input#otp-input")
    FMLS_REMEMBER_CHECKBOX = os.getenv("FMLS_REMEMBER_CHECKBOX", "input#remember-browser-checkbox-2")
    FMLS_OTP_CONTINUE_BUTTON = os.getenv("FMLS_OTP_CONTINUE_BUTTON", "button#btn-verify-login-otp")
    FMLS_REMINE_PRODUCT_LINK = os.getenv(
        "FMLS_REMINE_PRODUCT_LINK",
        'a._productItem_15qlz_1[href="https://fmls.remine.com"]'
    )

    # FMLS Authentication feature flag
    ENABLE_FMLS_AUTH = os.getenv("ENABLE_FMLS_AUTH", "true").lower() == "true"

    # =============================================================================
    # WEB SCRAPING CONFIGURATION
    # =============================================================================

    # Web scraping configuration
    TARGET_URL = os.getenv("TARGET_URL", "https://example.com/search")
    SEARCH_INPUT_SELECTOR = os.getenv("SEARCH_INPUT_SELECTOR", "input[name='search']")
    SEARCH_BUTTON_SELECTOR = os.getenv("SEARCH_BUTTON_SELECTOR", "button[type='submit']")

    # Wait for results page to load (in seconds)
    PAGE_LOAD_TIMEOUT = int(os.getenv("PAGE_LOAD_TIMEOUT", "30"))
    ELEMENT_WAIT_TIMEOUT = int(os.getenv("ELEMENT_WAIT_TIMEOUT", "10"))

    # Browser configuration
    HEADLESS_MODE = os.getenv("HEADLESS_MODE", "true").lower() == "true"
    BROWSER_WINDOW_SIZE = os.getenv("BROWSER_WINDOW_SIZE", "1920,1080")
    USER_AGENT = os.getenv("USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

    # Rate limiting (to avoid being blocked)
    MIN_DELAY_BETWEEN_REQUESTS = float(os.getenv("MIN_DELAY_BETWEEN_REQUESTS", "2.0"))
    MAX_DELAY_BETWEEN_REQUESTS = float(os.getenv("MAX_DELAY_BETWEEN_REQUESTS", "5.0"))

    # Retry configuration
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_DELAY = int(os.getenv("RETRY_DELAY", "5"))

    # Data extraction patterns
    # Regex patterns to find owner and buyer information
    OWNER_PATTERNS = [
        r"owner[:\s]+([A-Za-z\s\.,]+)",
        r"property owner[:\s]+([A-Za-z\s\.,]+)",
        r"owned by[:\s]+([A-Za-z\s\.,]+)",
    ]

    BUYER_PATTERNS = [
        r"buyer[:\s]+([A-Za-z\s\.,]+)",
        r"purchased by[:\s]+([A-Za-z\s\.,]+)",
        r"purchaser[:\s]+([A-Za-z\s\.,]+)",
    ]

    # Logging configuration
    # DEBUG: Temporarily set to DEBUG for authentication troubleshooting
    # TODO: Change back to INFO once authentication is working
    LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")  # Was: "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    # Database configuration
    DB_TIMEOUT = int(os.getenv("DB_TIMEOUT", "30"))

    # Feature flags
    SAVE_HTML_SNAPSHOTS = os.getenv("SAVE_HTML_SNAPSHOTS", "true").lower() == "true"
    FOLLOW_FIRST_RESULT = os.getenv("FOLLOW_FIRST_RESULT", "false").lower() == "true"
    SKIP_PROCESSED_ADDRESSES = os.getenv("SKIP_PROCESSED_ADDRESSES", "true").lower() == "true"

    # First result link selector (if FOLLOW_FIRST_RESULT is True)
    FIRST_RESULT_SELECTOR = os.getenv("FIRST_RESULT_SELECTOR", "a.result-link:first-child")

    @classmethod
    def validate(cls):
        """
        Validate critical configuration settings.
        Raises ValueError if required settings are missing or invalid.
        """
        if not cls.TARGET_URL or cls.TARGET_URL == "https://example.com/search":
            raise ValueError(
                "TARGET_URL must be configured. Please set it in .env file or environment variables."
            )

        if cls.MIN_DELAY_BETWEEN_REQUESTS > cls.MAX_DELAY_BETWEEN_REQUESTS:
            raise ValueError(
                "MIN_DELAY_BETWEEN_REQUESTS cannot be greater than MAX_DELAY_BETWEEN_REQUESTS"
            )

        return True

    @classmethod
    def get_fmls_config(cls):
        """
        Get FMLS authentication configuration as a dictionary.

        Returns:
            Dictionary with FMLS configuration
        """
        return {
            'home_url': cls.FMLS_HOME_URL,
            'login_url': cls.FMLS_LOGIN_URL,
            'dashboard_url': cls.FMLS_DASHBOARD_URL,
            'remine_url': cls.FMLS_REMINE_URL,
            'remine_daily_url': cls.FMLS_REMINE_DAILY_URL,
            'login_link_selector': cls.FMLS_LOGIN_LINK_SELECTOR,
            'login_id_input': cls.FMLS_LOGIN_ID_INPUT,
            'password_input': cls.FMLS_PASSWORD_INPUT,
            'login_button': cls.FMLS_LOGIN_BUTTON,
            'otp_input': cls.FMLS_OTP_INPUT,
            'remember_checkbox': cls.FMLS_REMEMBER_CHECKBOX,
            'otp_continue_button': cls.FMLS_OTP_CONTINUE_BUTTON,
            'remine_product_link': cls.FMLS_REMINE_PRODUCT_LINK,
            'otp_timeout': cls.OTP_TIMEOUT,
            'login_timeout': cls.LOGIN_TIMEOUT,
        }

    @classmethod
    def display_config(cls):
        """Display current configuration (for debugging)."""
        config = {
            "Input Excel": cls.INPUT_EXCEL,
            "Output Excel": cls.OUTPUT_EXCEL,
            "Database Path": cls.DATABASE_PATH,
            "Target URL": cls.TARGET_URL,
            "Headless Mode": cls.HEADLESS_MODE,
            "Page Load Timeout": cls.PAGE_LOAD_TIMEOUT,
            "Delay Range": f"{cls.MIN_DELAY_BETWEEN_REQUESTS}-{cls.MAX_DELAY_BETWEEN_REQUESTS}s",
            "Max Retries": cls.MAX_RETRIES,
            "Log Level": cls.LOG_LEVEL,
            "FMLS Auth Enabled": cls.ENABLE_FMLS_AUTH,
        }
        return config
