# OAuth Redirect Fix - Auth0 Login Implementation

## Problem Identified

The FMLS login system uses **OAuth 2.0 with Auth0** authentication, which causes automatic redirects with dynamic session parameters. When navigating to `https://firstmls.sso.remine.com`, the browser is redirected to:

```
https://firstmls-login.sso.remine.com/login?state=[SESSION_STATE]&client=MlYefPoR5ztZdoyauZRB54OdKzMIDVKJ&protocol=oauth2&...#signin
```

### Original Issue
- Code tried to find login link on home page
- Link didn't exist - automatic OAuth redirect happens instead
- Selenium didn't wait for redirect to complete
- Wrong selectors used (FMLS selectors instead of Auth0 selectors)

## Solution Implemented

### 1. OAuth Redirect Handling (`navigate_to_login()`)

**Changes:**
- ✅ Navigate directly to `https://firstmls.sso.remine.com`
- ✅ Wait for automatic redirect to Auth0 (`firstmls-login.sso.remine.com`)
- ✅ Detect URL change using WebDriverWait
- ✅ Log both initial and redirected URLs
- ✅ Wait for page to fully load after redirect
- ✅ Save screenshot on failure

**Key Code:**
```python
# Navigate to FMLS SSO (will auto-redirect)
self.driver.get(self.config['login_url'])

# Wait for OAuth redirect to Auth0
wait.until(lambda d: 'firstmls-login.sso.remine.com' in d.current_url)

# Log the OAuth parameters
logger.debug(f"Redirected URL: {redirected_url}")
```

### 2. Auth0 Form Detection (`perform_login()`)

**Changes:**
- ✅ Try multiple Auth0 selectors for username/email field
- ✅ Try multiple Auth0 selectors for password field
- ✅ Try multiple Auth0 selectors for submit button
- ✅ Log which selector successfully finds each element
- ✅ Save screenshot and page source on any failure
- ✅ Inspect all input fields on page for debugging

**Selector Priority:**

**Username/Email Field:**
1. `input[type="email"]` ← Most likely for Auth0
2. `input[name="username"]`
3. `input[name="email"]`
4. `#username`
5. `input[id="username"]`
6. Config fallback

**Password Field:**
1. `input[type="password"]` ← Most likely for Auth0
2. `input[name="password"]`
3. `#password`
4. `input[id="password"]`
5. Config fallback

**Submit Button:**
1. `button[type="submit"]` ← Most likely for Auth0
2. `button[name="action"]`
3. `button.auth0-lock-submit`
4. `input[type="submit"]`
5. Config fallback

### 3. Debug Helper Methods

**New Methods Added:**

**`_save_debug_screenshot(name)`**
- Saves PNG screenshot to `logs/` directory
- Includes timestamp in filename
- Called automatically on any failure

**`_save_page_source(name)`**
- Saves HTML source to `logs/` directory
- Includes timestamp in filename
- Allows offline inspection of page structure

**`_log_page_inputs()`**
- Lists all input fields on page
- Shows type, name, id, placeholder for each
- Lists all buttons with their attributes
- Helps identify correct selectors

## What Changed in the Code

### File: `src/authenticator.py`

#### `navigate_to_login()` - Lines 95-164
- **Before:** Navigated to home page, tried to find and click login link
- **After:** Navigates to SSO URL, waits for OAuth redirect, detects Auth0 page

#### `perform_login()` - Lines 166-320
- **Before:** Used single selector for each field
- **After:** Tries multiple selectors, logs which one works, saves debug info on failure

#### New Helper Methods - Lines 405-484
- `_save_debug_screenshot()` - Screenshot utility
- `_save_page_source()` - HTML save utility
- `_log_page_inputs()` - Page inspection utility

## Expected Behavior Now

### Successful Login Flow

```
1. Navigate to https://firstmls.sso.remine.com
   ↓
2. Detect automatic redirect to:
   https://firstmls-login.sso.remine.com/login?...
   ↓
3. Inspect page inputs (DEBUG mode)
   ↓
4. Find username field (try multiple selectors)
   ✓ Found with: input[type="email"]
   ↓
5. Find password field (try multiple selectors)
   ✓ Found with: input[type="password"]
   ↓
6. Find submit button (try multiple selectors)
   ✓ Found with: button[type="submit"]
   ↓
7. Fill credentials and submit
   ↓
8. Wait for 2FA prompt
```

### Enhanced Logging Output

You'll now see:

```
[DEBUG] Initial navigation to: https://firstmls.sso.remine.com
[DEBUG] Initial URL after navigation: https://firstmls.sso.remine.com
Waiting for OAuth redirect to Auth0...
[DEBUG] Detected redirect to Auth0 login
[DEBUG] Redirected URL: https://firstmls-login.sso.remine.com/login?state=...
✓ Reached Auth0 login page: https://firstmls-login.sso.remine.com/login
[DEBUG] Page title: Sign In - First MLS

[DEBUG] === Inspecting page input fields ===
[DEBUG] Found 2 input elements:
[DEBUG]   1. type=email, name=username, id=username, placeholder=Email
[DEBUG]   2. type=password, name=password, id=password, placeholder=Password
[DEBUG] Found 1 button elements:
[DEBUG]   1. type=submit, name=action, text=Continue
[DEBUG] === End page inspection ===

[DEBUG] Attempting to find username/email input...
[DEBUG] Trying selector: input[type="email"]
✓ Found username field with selector: input[type="email"]
✓ Entered login ID (field: input[type="email"])

[DEBUG] Attempting to find password input...
[DEBUG] Trying selector: input[type="password"]
✓ Found password field with selector: input[type="password"]
✓ Entered password (field: input[type="password"])

[DEBUG] Attempting to find submit button...
[DEBUG] Trying selector: button[type="submit"]
✓ Found submit button with selector: button[type="submit"]
✓ Clicked login button (button: button[type="submit"])
```

### Debug Files on Failure

If authentication fails, you'll get:

**Screenshots:**
```
logs/navigate_to_login_timeout_1736345678.png
logs/login_username_not_found_1736345690.png
logs/login_password_not_found_1736345692.png
logs/login_button_not_found_1736345694.png
```

**HTML Source:**
```
logs/login_username_not_found_1736345690.html
logs/login_password_not_found_1736345692.html
```

## Testing the Changes

### Run the application:
```bash
python main.py
```

### Watch for these key log lines:
1. ✅ "Detected redirect to Auth0 login"
2. ✅ "Found username field with selector: ..."
3. ✅ "Found password field with selector: ..."
4. ✅ "Found submit button with selector: ..."
5. ✅ "Login form submitted successfully"

### If it fails:
1. Check `logs/scraper.log` for detailed debug output
2. Look for screenshot files in `logs/` directory
3. Open HTML files in browser to see exact page structure
4. Check which selector attempt failed

## Reverting to INFO Logging (Once Working)

After authentication works consistently, clean up debug code:

1. **In `config/settings.py` (line 129):**
   ```python
   LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")  # Changed from DEBUG
   ```

2. **Optional:** Remove or comment out `logger.debug()` calls in:
   - `src/authenticator.py`
   - `src/gcp_secrets.py`
   - `main.py`

3. **Keep:** Error logging and screenshot/page source saves (helpful for production)

## OAuth Flow Details

### What Happens Behind the Scenes:

1. **Client initiates:** Browser navigates to `https://firstmls.sso.remine.com`
2. **FMLS redirects:** HTTP 302 redirect to Auth0
3. **Auth0 generates session:** Creates `state`, `nonce`, `code_challenge` (PKCE)
4. **Auth0 displays login:** Shows `https://firstmls-login.sso.remine.com/login?...#signin`
5. **User authenticates:** Enters credentials
6. **Auth0 validates:** Checks credentials and 2FA
7. **Auth0 redirects back:** Returns to `https://firstmls.sso.remine.com/callback`
8. **FMLS issues token:** OAuth token exchange completes
9. **Dashboard accessible:** User is authenticated

Our code now properly handles steps 1-5!

## Common Issues & Solutions

### Issue: "Did not detect redirect"
- **Cause:** Redirect happened too fast or Selenium blocked it
- **Solution:** Code continues anyway; check if login form appears

### Issue: "Could not find username/email input field"
- **Cause:** Page structure changed or JavaScript not loaded
- **Solution:** Check screenshots and HTML files to see actual structure

### Issue: Screenshot shows blank page
- **Cause:** Page didn't load or JavaScript disabled
- **Solution:** Try non-headless mode (`HEADLESS_MODE=false`)

### Issue: Redirect loops
- **Cause:** Cookies or session issues
- **Solution:** Clear cookies directory and retry

---

**Implementation Status:** ✅ Complete

**Next Step:** Test with real credentials and monitor logs for selector matches.
