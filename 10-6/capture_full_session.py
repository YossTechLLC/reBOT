#!/usr/bin/env python3
"""
Complete Session State Capture Tool

This tool captures EVERYTHING during a successful authentication:
1. All cookies from ALL domains with full details
2. localStorage for each domain
3. sessionStorage for each domain
4. Browser fingerprint details
5. Network state
6. Timing information

Run this IMMEDIATELY after successful 2FA to capture the real auth state.

Usage:
    python capture_full_session.py
"""

import json
import time
from pathlib import Path
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from config import Settings

OUTPUT_DIR = Path("data/session_debug")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def setup_browser():
    """Create browser instance matching production config."""
    options = Options()

    if Settings.HEADLESS_MODE:
        options.add_argument('--headless=new')

    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option('excludeSwitches', ['enable-automation'])
    options.add_experimental_option('useAutomationExtension', False)

    if Settings.USER_AGENT:
        options.add_argument(f'user-agent={Settings.USER_AGENT}')

    if Settings.BROWSER_WINDOW_SIZE:
        options.add_argument(f'window-size={Settings.BROWSER_WINDOW_SIZE}')

    driver = webdriver.Chrome(options=options)
    return driver


def get_all_storage(driver, url):
    """
    Extract ALL storage (localStorage + sessionStorage) from a given URL.

    Args:
        driver: Selenium WebDriver
        url: URL to navigate to and extract storage from

    Returns:
        Dictionary with localStorage and sessionStorage
    """
    print(f"\n🔍 Extracting storage from: {url}")

    try:
        driver.get(url)
        time.sleep(2)

        # Get localStorage
        local_storage = driver.execute_script("""
            let items = {};
            for (let i = 0; i < localStorage.length; i++) {
                let key = localStorage.key(i);
                items[key] = localStorage.getItem(key);
            }
            return items;
        """)

        # Get sessionStorage
        session_storage = driver.execute_script("""
            let items = {};
            for (let i = 0; i < sessionStorage.length; i++) {
                let key = sessionStorage.key(i);
                items[key] = sessionStorage.getItem(key);
            }
            return items;
        """)

        print(f"   localStorage: {len(local_storage)} items")
        print(f"   sessionStorage: {len(session_storage)} items")

        return {
            'url': url,
            'current_url': driver.current_url,
            'localStorage': local_storage,
            'sessionStorage': session_storage,
            'item_count': {
                'localStorage': len(local_storage),
                'sessionStorage': len(session_storage)
            }
        }

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {
            'url': url,
            'error': str(e),
            'localStorage': {},
            'sessionStorage': {}
        }


def get_browser_fingerprint(driver):
    """Get browser fingerprint details."""
    print("\n🔍 Capturing browser fingerprint...")

    fingerprint = driver.execute_script("""
        return {
            userAgent: navigator.userAgent,
            platform: navigator.platform,
            language: navigator.language,
            languages: navigator.languages,
            hardwareConcurrency: navigator.hardwareConcurrency,
            deviceMemory: navigator.deviceMemory,
            screenResolution: screen.width + 'x' + screen.height,
            colorDepth: screen.colorDepth,
            pixelRatio: window.devicePixelRatio,
            timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
            timezoneOffset: new Date().getTimezoneOffset(),
            cookieEnabled: navigator.cookieEnabled,
            doNotTrack: navigator.doNotTrack,
            webdriver: navigator.webdriver,
            plugins: Array.from(navigator.plugins).map(p => p.name),
            mimeTypes: Array.from(navigator.mimeTypes).map(m => m.type),
        };
    """)

    print(f"   User-Agent: {fingerprint.get('userAgent', 'N/A')[:80]}...")
    print(f"   Webdriver detected: {fingerprint.get('webdriver', 'N/A')}")
    print(f"   Timezone: {fingerprint.get('timezone', 'N/A')}")

    return fingerprint


def capture_complete_state(driver):
    """
    Capture complete browser state across all relevant domains.

    Returns:
        Complete state dictionary
    """
    timestamp = datetime.now().isoformat()

    print("\n" + "=" * 80)
    print("CAPTURING COMPLETE SESSION STATE")
    print("=" * 80)

    # All domains involved in auth flow
    domains_to_check = [
        'https://firstmls.com',
        'https://firstmls.sso.remine.com',
        'https://firstmls.sso.remine.com/dashboard-v2',
        'https://sso.remine.com',
        'https://fmls.remine.com',
        'https://fmls.remine.com/daily',
        'https://remine.com',
    ]

    state = {
        'captured_at': timestamp,
        'fingerprint': get_browser_fingerprint(driver),
        'cookies': {},
        'storage': {},
        'domains_checked': domains_to_check,
    }

    # Capture cookies from each domain
    print("\n" + "=" * 80)
    print("CAPTURING COOKIES PER DOMAIN")
    print("=" * 80)

    for domain_url in domains_to_check:
        print(f"\n🍪 Domain: {domain_url}")
        try:
            driver.get(domain_url)
            time.sleep(1)

            cookies = driver.get_cookies()

            # Parse domain from URL
            from urllib.parse import urlparse
            parsed = urlparse(domain_url)
            domain_key = parsed.netloc

            state['cookies'][domain_key] = {
                'url': domain_url,
                'actual_url': driver.current_url,
                'cookie_count': len(cookies),
                'cookies': cookies
            }

            print(f"   Captured: {len(cookies)} cookies")

            # Show cookie names and domains
            for cookie in cookies[:5]:
                print(f"   - {cookie.get('name')}: domain={cookie.get('domain')}, "
                      f"httpOnly={cookie.get('httpOnly')}, secure={cookie.get('secure')}, "
                      f"sameSite={cookie.get('sameSite')}")

            if len(cookies) > 5:
                print(f"   ... and {len(cookies) - 5} more cookies")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            state['cookies'][domain_key] = {'error': str(e)}

    # Capture storage from each domain
    print("\n" + "=" * 80)
    print("CAPTURING STORAGE PER DOMAIN")
    print("=" * 80)

    for domain_url in domains_to_check:
        parsed = urlparse(domain_url)
        domain_key = parsed.netloc

        storage = get_all_storage(driver, domain_url)
        state['storage'][domain_key] = storage

        # Show storage keys
        if storage.get('localStorage'):
            print(f"   localStorage keys: {list(storage['localStorage'].keys())[:5]}")
        if storage.get('sessionStorage'):
            print(f"   sessionStorage keys: {list(storage['sessionStorage'].keys())[:5]}")

    return state


def save_state(state, filename="complete_session_state.json"):
    """Save captured state to file."""
    output_file = OUTPUT_DIR / filename

    print("\n" + "=" * 80)
    print("SAVING STATE")
    print("=" * 80)

    with open(output_file, 'w') as f:
        json.dump(state, f, indent=2, default=str)

    file_size = output_file.stat().st_size
    print(f"\n✓ Saved to: {output_file}")
    print(f"  File size: {file_size:,} bytes")

    return output_file


def analyze_state(state):
    """Analyze captured state for auth-related data."""
    print("\n" + "=" * 80)
    print("STATE ANALYSIS")
    print("=" * 80)

    # Cookie analysis
    print("\n📊 COOKIE SUMMARY:")
    total_cookies = 0
    auth_cookies = []

    for domain, data in state['cookies'].items():
        if 'error' in data:
            continue

        cookies = data.get('cookies', [])
        total_cookies += len(cookies)

        print(f"\n  {domain}: {len(cookies)} cookies")

        # Look for auth-related cookies
        for cookie in cookies:
            name = cookie.get('name', '').lower()
            if any(keyword in name for keyword in ['auth', 'session', 'token', 'id', 'login', 'sso', 'oauth', 'bearer']):
                auth_cookies.append({
                    'domain': domain,
                    'name': cookie.get('name'),
                    'cookie_domain': cookie.get('domain'),
                    'httpOnly': cookie.get('httpOnly'),
                    'secure': cookie.get('secure'),
                    'sameSite': cookie.get('sameSite'),
                })
                print(f"    🔑 AUTH COOKIE: {cookie.get('name')}")
                print(f"       Domain: {cookie.get('domain')}, httpOnly: {cookie.get('httpOnly')}, "
                      f"secure: {cookie.get('secure')}, sameSite: {cookie.get('sameSite')}")

    print(f"\n  Total cookies: {total_cookies}")
    print(f"  Auth-related cookies: {len(auth_cookies)}")

    # Storage analysis
    print("\n📊 STORAGE SUMMARY:")
    total_local = 0
    total_session = 0
    auth_storage = []

    for domain, data in state['storage'].items():
        if 'error' in data:
            continue

        local = data.get('localStorage', {})
        session = data.get('sessionStorage', {})

        total_local += len(local)
        total_session += len(session)

        if local or session:
            print(f"\n  {domain}:")
            print(f"    localStorage: {len(local)} items")
            print(f"    sessionStorage: {len(session)} items")

            # Look for auth-related storage
            for key, value in {**local, **session}.items():
                key_lower = key.lower()
                if any(keyword in key_lower for keyword in ['auth', 'token', 'id', 'session', 'login', 'user', 'sso']):
                    auth_storage.append({
                        'domain': domain,
                        'key': key,
                        'value_preview': str(value)[:100] + '...' if len(str(value)) > 100 else str(value)
                    })
                    print(f"    🔑 AUTH STORAGE: {key}")
                    print(f"       Value preview: {str(value)[:100]}...")

    print(f"\n  Total localStorage items: {total_local}")
    print(f"  Total sessionStorage items: {total_session}")
    print(f"  Auth-related storage items: {len(auth_storage)}")

    # Fingerprint analysis
    print("\n📊 FINGERPRINT:")
    fp = state.get('fingerprint', {})
    print(f"  Webdriver detected: {fp.get('webdriver')}")
    print(f"  User-Agent: {fp.get('userAgent', 'N/A')[:100]}...")
    print(f"  Timezone: {fp.get('timezone')}")
    print(f"  Language: {fp.get('language')}")

    return {
        'total_cookies': total_cookies,
        'auth_cookies': auth_cookies,
        'total_localStorage': total_local,
        'total_sessionStorage': total_session,
        'auth_storage': auth_storage,
    }


def main():
    """Main capture function."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " COMPLETE SESSION STATE CAPTURE TOOL ".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    print("\n⚠️  Run this IMMEDIATELY after successful 2FA authentication!")
    print("   This will capture the REAL authentication state.\n")

    input("Press Enter when you're at the Remine dashboard after 2FA...")

    print("\nInitializing browser...")
    driver = setup_browser()

    try:
        # Capture complete state
        state = capture_complete_state(driver)

        # Save to file
        output_file = save_state(state)

        # Analyze
        analysis = analyze_state(state)

        # Save analysis separately
        analysis_file = OUTPUT_DIR / "session_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)

        print("\n" + "=" * 80)
        print("✓ CAPTURE COMPLETE!")
        print("=" * 80)
        print(f"\nFiles saved:")
        print(f"  • {output_file}")
        print(f"  • {analysis_file}")
        print(f"\n📋 Next steps:")
        print(f"  1. Review {analysis_file} for auth cookies/storage")
        print(f"  2. Check which domain has the REAL auth tokens")
        print(f"  3. Update session manager to capture ALL domains + storage")

    finally:
        driver.quit()
        print("\nBrowser closed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCapture interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
