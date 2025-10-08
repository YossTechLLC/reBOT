#!/usr/bin/env python3
"""
Session Persistence Diagnostic Tool

This tool helps diagnose issues with FMLS session persistence by:
1. Checking if cookie file exists
2. Displaying cookie file contents
3. Showing session age and validity
4. Testing cookie loading in isolated browser
5. Verifying domain configurations

Usage:
    python diagnose_session.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from config import Settings


def print_header(title):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80)


def check_cookie_file():
    """Check if cookie file exists and display basic info."""
    print_header("COOKIE FILE CHECK")

    cookie_file = Settings.COOKIES_DIR / "fmls_session.json"
    print(f"Cookie file path: {cookie_file}")
    print(f"Cookie directory: {Settings.COOKIES_DIR}")
    print(f"Directory exists: {Settings.COOKIES_DIR.exists()}")
    print(f"Cookie file exists: {cookie_file.exists()}")

    if not cookie_file.exists():
        print("\n❌ Cookie file does NOT exist")
        print("   This is expected if you haven't logged in yet.")
        print("   After your first successful login with 2FA, the cookie file will be created.")
        return None

    print("✓ Cookie file exists")

    # Get file metadata
    stat = cookie_file.stat()
    file_size = stat.st_size
    modified_time = datetime.fromtimestamp(stat.st_mtime)

    print(f"\nFile size: {file_size:,} bytes")
    print(f"Last modified: {modified_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Age: {(datetime.now() - modified_time).total_seconds() / 3600:.2f} hours")

    return cookie_file


def display_cookie_contents(cookie_file):
    """Display contents of cookie file."""
    if not cookie_file:
        return None

    print_header("COOKIE FILE CONTENTS")

    try:
        with open(cookie_file, 'r') as f:
            data = json.load(f)

        print(f"Saved at: {data.get('saved_at', 'unknown')}")
        print(f"Domain filter: {data.get('domain', 'None (all domains)')}")
        print(f"Cookie count: {data.get('cookie_count', 0)}")

        cookies = data.get('cookies', [])

        if not cookies:
            print("\n❌ No cookies in file!")
            return None

        print(f"\n✓ Found {len(cookies)} cookie(s)")

        # Group cookies by domain
        domains = {}
        for cookie in cookies:
            domain = cookie.get('domain', 'unknown')
            if domain not in domains:
                domains[domain] = []
            domains[domain].append(cookie)

        print(f"\nCookies by domain:")
        for domain, domain_cookies in domains.items():
            print(f"  • {domain}: {len(domain_cookies)} cookie(s)")

        # Show first 5 cookies in detail
        print(f"\nFirst {min(5, len(cookies))} cookies (detailed):")
        for i, cookie in enumerate(cookies[:5], 1):
            print(f"\n  Cookie {i}:")
            print(f"    Name: {cookie.get('name', 'N/A')}")
            print(f"    Domain: {cookie.get('domain', 'N/A')}")
            print(f"    Path: {cookie.get('path', 'N/A')}")
            print(f"    Secure: {cookie.get('secure', False)}")
            print(f"    HttpOnly: {cookie.get('httpOnly', False)}")
            print(f"    SameSite: {cookie.get('sameSite', 'N/A')}")

            # Check if cookie has expiry
            if 'expiry' in cookie:
                expiry_timestamp = cookie['expiry']
                expiry_date = datetime.fromtimestamp(expiry_timestamp)
                print(f"    Expiry: {expiry_date.strftime('%Y-%m-%d %H:%M:%S')}")

                if expiry_date < datetime.now():
                    print(f"    ⚠️ EXPIRED!")
                else:
                    remaining = (expiry_date - datetime.now()).total_seconds() / 86400
                    print(f"    Valid for: {remaining:.1f} more days")

        return data

    except json.JSONDecodeError as e:
        print(f"\n❌ ERROR: Cookie file contains invalid JSON")
        print(f"   Error: {e}")
        return None

    except Exception as e:
        print(f"\n❌ ERROR reading cookie file: {e}")
        return None


def check_session_validity(cookie_data):
    """Check if session is still valid."""
    if not cookie_data:
        return

    print_header("SESSION VALIDITY CHECK")

    saved_at_str = cookie_data.get('saved_at')
    if not saved_at_str:
        print("❌ No 'saved_at' timestamp in cookie file")
        return

    try:
        saved_at = datetime.fromisoformat(saved_at_str)
        current_time = datetime.now()
        age_seconds = (current_time - saved_at).total_seconds()
        age_hours = age_seconds / 3600
        age_days = age_hours / 24

        max_age_hours = 720  # 30 days
        max_age_days = max_age_hours / 24

        print(f"Session saved at: {saved_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Current time:     {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nSession age: {age_hours:.2f} hours ({age_days:.2f} days)")
        print(f"Max allowed age: {max_age_hours} hours ({max_age_days} days)")

        is_valid = age_hours < max_age_hours

        if is_valid:
            remaining_hours = max_age_hours - age_hours
            remaining_days = remaining_hours / 24
            print(f"\n✓ Session is VALID")
            print(f"  Expires in: {remaining_hours:.1f} hours ({remaining_days:.1f} days)")
        else:
            expired_hours = age_hours - max_age_hours
            expired_days = expired_hours / 24
            print(f"\n❌ Session is EXPIRED")
            print(f"  Expired: {expired_hours:.1f} hours ago ({expired_days:.1f} days)")

    except Exception as e:
        print(f"❌ ERROR checking session validity: {e}")


def show_configuration():
    """Show relevant configuration settings."""
    print_header("CONFIGURATION")

    print(f"FMLS Auth Enabled: {Settings.ENABLE_FMLS_AUTH}")
    print(f"GCP Project ID: {Settings.GCP_PROJECT_ID}")
    print(f"Cookie Directory: {Settings.COOKIES_DIR}")
    print(f"\nFMLS URLs:")
    print(f"  Home URL: {Settings.FMLS_HOME_URL}")
    print(f"  Login URL: {Settings.FMLS_LOGIN_URL}")
    print(f"  Dashboard URL: {Settings.FMLS_DASHBOARD_URL}")
    print(f"  Remine URL: {Settings.FMLS_REMINE_URL}")
    print(f"  Remine Daily URL: {Settings.FMLS_REMINE_DAILY_URL}")

    print(f"\nTimeouts:")
    print(f"  OTP Timeout: {Settings.OTP_TIMEOUT} seconds")
    print(f"  Login Timeout: {Settings.LOGIN_TIMEOUT} seconds")


def test_session_manager():
    """Test SessionManager without opening browser."""
    print_header("SESSION MANAGER TEST")

    try:
        from src.session_manager import create_session_manager

        session_manager = create_session_manager(
            cookie_dir=Settings.COOKIES_DIR,
            session_name="fmls_session"
        )

        print("✓ SessionManager created successfully")

        # Check session validity
        is_valid = session_manager.is_session_valid()
        print(f"Session valid: {is_valid}")

        # Get cookie info
        info = session_manager.get_cookie_info()
        if info:
            print(f"\nCookie info:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        else:
            print("No cookie info available")

    except Exception as e:
        print(f"❌ ERROR testing SessionManager: {e}")
        import traceback
        traceback.print_exc()


def show_recommendations(cookie_file, cookie_data):
    """Show recommendations based on findings."""
    print_header("RECOMMENDATIONS")

    if not cookie_file:
        print("📝 Cookie file doesn't exist yet.")
        print("   Action: Run main.py and complete the 2FA login.")
        print("   After successful login, cookies will be saved automatically.")
        print("   Next run should use saved cookies without 2FA.")
        return

    if not cookie_data:
        print("📝 Cookie file exists but has issues.")
        print("   Action: Delete the cookie file and login again:")
        print(f"   rm {Settings.COOKIES_DIR}/fmls_session.json")
        print("   python main.py")
        return

    cookies = cookie_data.get('cookies', [])
    saved_at_str = cookie_data.get('saved_at')

    if not cookies:
        print("📝 Cookie file exists but contains no cookies.")
        print("   Possible cause: Cookies were saved before login completed.")
        print("   Action: Delete cookie file and login again.")
        return

    if not saved_at_str:
        print("📝 Cookie file missing timestamp.")
        print("   Action: Delete cookie file and login again.")
        return

    # CRITICAL CHECK: Number of cookies
    cookie_count = len(cookies)
    print(f"\n🔍 CRITICAL CHECK: Cookie Count")
    print(f"   Found: {cookie_count} cookies")

    if cookie_count < 5:
        print(f"\n   ❌ TOO FEW COOKIES!")
        print(f"   Problem: Only {cookie_count} cookie(s) were saved.")
        print(f"   Expected: 10-20+ cookies after full authentication.")
        print(f"\n   Root Cause: Cookies were likely saved BEFORE completing")
        print(f"              the full authentication flow (before reaching Remine).")
        print(f"\n   ⚠️  This is why session persistence is failing!")
        print(f"\n   Action Required:")
        print(f"   1. Delete the incomplete cookie file:")
        print(f"      python clear_session.py")
        print(f"   2. Run main.py and complete FULL 2FA login")
        print(f"   3. Let the process complete (reach Remine dashboard)")
        print(f"   4. Cookies will be saved AFTER full authentication")
        print(f"   5. Next run will have 10-20+ cookies and work without 2FA")
        return
    else:
        print(f"   ✓ Good! You have enough cookies for authentication.")

    # Check session age
    try:
        saved_at = datetime.fromisoformat(saved_at_str)
        age_hours = (datetime.now() - saved_at).total_seconds() / 3600

        if age_hours >= 720:  # 30 days
            print("\n📝 Session has expired (>30 days old).")
            print("   Action: Run main.py - you'll need to login with 2FA again.")
            print("   New cookies will be saved after successful login.")
        else:
            print("\n✓ Session age is good (not expired).")
            print("\n   If you're still being asked for 2FA, check:")
            print("   1. Cookie domains match the login domain")
            print("   2. Cookies aren't being cleared by browser settings")
            print("   3. Review main.py logs for detailed debug output")

            # Check domain mismatch
            domains = set(c.get('domain', '') for c in cookies)
            expected_domains = ['remine.com', 'sso.remine.com', 'firstmls.sso.remine.com', 'fmls.remine.com']

            print(f"\n   Cookie domains found: {', '.join(sorted(domains))}")
            print(f"   Expected domains: {', '.join(expected_domains)}")

            # Check if we have critical cookie domains
            has_sso = any('firstmls.sso.remine.com' in d for d in domains)
            has_remine = any('remine.com' in d for d in domains)

            print(f"\n   🔍 Critical Domain Check:")
            print(f"      SSO domain (firstmls.sso.remine.com): {'✓ Present' if has_sso else '✗ MISSING'}")
            print(f"      Remine domain (*.remine.com): {'✓ Present' if has_remine else '✗ MISSING'}")

            if has_sso and has_remine:
                print(f"\n   ✓✓ Found BOTH SSO and Remine cookies - authentication should work!")
            elif has_remine and not has_sso:
                print(f"\n   ⚠️ CRITICAL PROBLEM: No SSO cookies found!")
                print(f"   You have Remine cookies but missing SSO authentication cookies.")
                print(f"   The is_authenticated() check navigates to firstmls.sso.remine.com/dashboard-v2")
                print(f"   Without SSO cookies, this check will FAIL and trigger 2FA again.")
                print(f"\n   Root Cause: Cookies were saved AFTER leaving the SSO domain.")
                print(f"   Fix: Need to save cookies at BOTH dashboard (SSO) AND Remine.")
                print(f"\n   Action Required:")
                print(f"   1. Clear cookies: python clear_session.py")
                print(f"   2. Update to latest code (should have dual cookie save)")
                print(f"   3. Run main.py with 2FA")
                print(f"   4. Check that BOTH SSO and Remine cookies are saved")
            elif has_sso and not has_remine:
                print(f"\n   ⚠️ WARNING: No Remine cookies found!")
                print(f"   You have SSO cookies but missing Remine session cookies.")
                print(f"   Cookies were saved before reaching Remine dashboard.")
            else:
                print(f"\n   ✗✗ Missing BOTH SSO and Remine cookies!")
                print(f"   This will definitely cause authentication to fail.")
                print(f"   Cookies file is incomplete or corrupted.")

    except Exception as e:
        print(f"Unable to check session age: {e}")


def main():
    """Main diagnostic function."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " FMLS SESSION PERSISTENCE DIAGNOSTIC TOOL ".center(78) + "║")
    print("╚" + "═" * 78 + "╝")

    # Run all checks
    cookie_file = check_cookie_file()
    cookie_data = display_cookie_contents(cookie_file)
    check_session_validity(cookie_data)
    show_configuration()
    test_session_manager()
    show_recommendations(cookie_file, cookie_data)

    print("\n" + "=" * 80)
    print("Diagnostic complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDiagnostic interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
