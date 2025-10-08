#!/usr/bin/env python3
"""
Diagnostic script for FMLS authentication.
Checks each component independently to identify issues.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("FMLS AUTHENTICATION DIAGNOSTIC TOOL")
print("=" * 80)
print()

# Test 1: Environment Configuration
print("Test 1: Environment Configuration")
print("-" * 80)

try:
    from dotenv import load_dotenv
    load_dotenv()

    gcp_creds = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    print(f"✓ GOOGLE_APPLICATION_CREDENTIALS: {gcp_creds if gcp_creds else 'Not set (using ADC)'}")

    from config import Settings
    print(f"✓ GCP Project ID: {Settings.GCP_PROJECT_ID}")
    print(f"✓ Login Secret Name: {Settings.SECRET_LOGIN_ID}")
    print(f"✓ Password Secret Name: {Settings.SECRET_PASSWORD}")
    print(f"✓ FMLS Auth Enabled: {Settings.ENABLE_FMLS_AUTH}")
    print(f"✓ Cookies Directory: {Settings.COOKIES_DIR}")
    print(f"✓ Log Level: {Settings.LOG_LEVEL}")

except Exception as e:
    print(f"❌ Configuration error: {e}")
    sys.exit(1)

print()

# Test 2: Google Cloud Secret Manager
print("Test 2: Google Cloud Secret Manager Connection")
print("-" * 80)

try:
    from src.gcp_secrets import SecretManagerClient

    client = SecretManagerClient(Settings.GCP_PROJECT_ID)
    print(f"✓ Secret Manager client created")

    # Test connection
    if client.test_connection():
        print(f"✓ Connection to Secret Manager successful")
    else:
        print(f"❌ Cannot connect to Secret Manager")
        print(f"   Check credentials and permissions")
        sys.exit(1)

    # Try to retrieve secrets
    print(f"\nAttempting to retrieve credentials...")
    credentials = client.get_credentials(
        Settings.SECRET_LOGIN_ID,
        Settings.SECRET_PASSWORD
    )

    if credentials:
        print(f"✓ Login ID retrieved: {len(credentials['login_id'])} characters")
        print(f"✓ Password retrieved: {len(credentials['password'])} characters")
    else:
        print(f"❌ Failed to retrieve credentials")
        print(f"   Check secret names and permissions")
        sys.exit(1)

except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"   Run: pip install google-cloud-secret-manager")
    sys.exit(1)

except Exception as e:
    print(f"❌ Secret Manager error: {e}")
    print(f"   Check GCP configuration and credentials")
    sys.exit(1)

print()

# Test 3: Selenium WebDriver
print("Test 3: Selenium WebDriver")
print("-" * 80)

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.chrome.options import Options as ChromeOptions

    options = ChromeOptions()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    print("Initializing Chrome WebDriver...")
    service = ChromeService()
    driver = webdriver.Chrome(service=service, options=options)

    print(f"✓ WebDriver initialized successfully")

    # Test navigation
    print(f"Testing navigation to Google...")
    driver.get("https://www.google.com")
    print(f"✓ Navigation successful")
    print(f"  Page title: {driver.title}")

    driver.quit()
    print(f"✓ WebDriver closed successfully")

except Exception as e:
    print(f"❌ WebDriver error: {e}")
    print(f"   Check Chrome/ChromeDriver installation")
    sys.exit(1)

print()

# Test 4: Session Manager
print("Test 4: Session Manager")
print("-" * 80)

try:
    from src.session_manager import SessionManager

    cookie_file = Settings.COOKIES_DIR / "test_session.json"
    session_mgr = SessionManager(cookie_file)

    print(f"✓ SessionManager created")
    print(f"  Cookie file: {cookie_file}")

    # Check for existing cookies
    info = session_mgr.get_cookie_info()
    if info:
        print(f"✓ Found existing session:")
        print(f"  Saved at: {info['saved_at']}")
        print(f"  Cookie count: {info['cookie_count']}")
        print(f"  Valid: {session_mgr.is_session_valid()}")
    else:
        print(f"  No existing session found (this is normal for first run)")

    # Clean up test file if created
    if cookie_file.name == "test_session.json" and cookie_file.exists():
        cookie_file.unlink()

except Exception as e:
    print(f"❌ SessionManager error: {e}")
    sys.exit(1)

print()

# Test 5: FMLS Configuration
print("Test 5: FMLS Configuration")
print("-" * 80)

try:
    fmls_config = Settings.get_fmls_config()

    print("URLs:")
    print(f"  Home: {fmls_config['home_url']}")
    print(f"  Login: {fmls_config['login_url']}")
    print(f"  Dashboard: {fmls_config['dashboard_url']}")
    print(f"  Remine: {fmls_config['remine_url']}")
    print(f"  Remine Daily: {fmls_config['remine_daily_url']}")

    print("\nSelectors:")
    print(f"  Login Link: {fmls_config['login_link_selector']}")
    print(f"  Login ID Input: {fmls_config['login_id_input']}")
    print(f"  Password Input: {fmls_config['password_input']}")
    print(f"  Login Button: {fmls_config['login_button']}")
    print(f"  OTP Input: {fmls_config['otp_input']}")
    print(f"  OTP Button: {fmls_config['otp_continue_button']}")

    print("\nTimeouts:")
    print(f"  Login Timeout: {fmls_config['login_timeout']}s")
    print(f"  OTP Timeout: {fmls_config['otp_timeout']}s")

    print(f"\n✓ FMLS configuration loaded successfully")

except Exception as e:
    print(f"❌ Configuration error: {e}")
    sys.exit(1)

print()

# Final Summary
print("=" * 80)
print("DIAGNOSTIC SUMMARY")
print("=" * 80)
print()
print("✓ All diagnostic tests passed!")
print()
print("Your system is configured correctly for FMLS authentication.")
print()
print("Next steps:")
print("  1. Run: python test_fmls_auth.py")
print("  2. Or run: python main.py")
print()
print("If authentication still fails, check logs/scraper.log for detailed error messages.")
print("=" * 80)
