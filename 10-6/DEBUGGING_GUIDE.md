# FMLS Authentication Debugging Guide

This guide helps troubleshoot FMLS authentication issues using the enhanced debugging features.

## DEBUG Code Locations

All debugging code is marked with `# DEBUG:` or `[DEBUG]` for easy identification and removal once authentication is stable.

### Files with DEBUG Code

1. **`config/settings.py`** (Line ~129)
   ```python
   # DEBUG: Temporarily set to DEBUG for authentication troubleshooting
   # TODO: Change back to INFO once authentication is working
   LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")  # Was: "INFO"
   ```

2. **`src/authenticator.py`** (Lines ~361-369, and throughout)
   ```python
   # DEBUG: Log configuration being used
   logger.debug("=== [DEBUG] Authentication Configuration ===")
   ```

3. **`src/gcp_secrets.py`** (Lines ~40-42, and throughout)
   ```python
   # DEBUG: Check for credentials
   logger.debug(f"[DEBUG] GOOGLE_APPLICATION_CREDENTIALS: ...")
   ```

4. **`main.py`** (Lines ~87-89, and throughout)
   ```python
   # DEBUG: Verify scraper has driver
   logger.debug(f"[DEBUG] Scraper driver initialized: ...")
   ```

### Removing DEBUG Code (Once Working)

Search for these patterns and remove/revert:
- `# DEBUG:` comments
- `logger.debug("[DEBUG]` calls
- `# TODO: Change back to INFO` in settings.py

Quick find command:
```bash
grep -r "DEBUG" src/ config/ main.py
```

## Understanding the Logs

### Authentication Flow Steps

The logs now show clear step-by-step progression:

```
Step 1: Checking for saved session...
Step 2: Checking if already authenticated...
Step 3: Performing fresh login...
  Step 3.1: Retrieving credentials from Google Secret Manager...
  Step 3.2: Navigating to login page...
  Step 3.3: Submitting login credentials...
  Step 3.4: Handling 2FA...
  Step 3.5: Verifying dashboard access...
  Step 3.6: Saving session cookies...
  Step 3.7: Navigating to Remine product...
```

### Success Indicators

Look for these markers:
- ✓ checkmarks indicate successful steps
- Green "Step X.Y" messages show progress
- Final message: `✓ AUTHENTICATION SUCCESSFUL`

### Failure Indicators

Look for these markers:
- ❌ crosses indicate failures
- Red error messages with specific issues
- "Troubleshooting steps:" or "Check:" sections

## Common Issues and Solutions

### Issue 1: No Credentials Retrieved

**Log Output:**
```
❌ Failed to retrieve credentials from Secret Manager
Check:
  1. GOOGLE_APPLICATION_CREDENTIALS environment variable is set
  2. Service account has 'Secret Manager Secret Accessor' role
  3. Secret names are correct in configuration
  4. GCP project ID is correct
```

**Debug Info to Check:**
```
[DEBUG] GOOGLE_APPLICATION_CREDENTIALS: /path/to/key.json
[DEBUG] Attempting to access project: 291176869049
[DEBUG] Login secret: projects/291176869049/secrets/reBOT_LOGIN/versions/latest
[DEBUG] Password secret: projects/291176869049/secrets/reBOT_PASSWORD/versions/latest
```

**Solutions:**

1. **Check environment variable:**
   ```bash
   echo $GOOGLE_APPLICATION_CREDENTIALS
   ```

2. **Verify service account has access:**
   ```bash
   gcloud secrets list --project=291176869049
   ```

3. **Grant permissions:**
   ```bash
   gcloud secrets add-iam-policy-binding reBOT_LOGIN \
     --member='serviceAccount:YOUR_SA@291176869049.iam.gserviceaccount.com' \
     --role='roles/secretmanager.secretAccessor' \
     --project=291176869049

   gcloud secrets add-iam-policy-binding reBOT_PASSWORD \
     --member='serviceAccount:YOUR_SA@291176869049.iam.gserviceaccount.com' \
     --role='roles/secretmanager.secretAccessor' \
     --project=291176869049
   ```

4. **Test Secret Manager access:**
   ```python
   python test_fmls_auth.py
   ```

### Issue 2: Element Not Found

**Log Output:**
```
❌ Failed to navigate to login page
Timeout waiting for login link
```

**Debug Info to Check:**
```
[DEBUG] Looking for login link: a[href="https://firstmls.sso.remine.com"]
[DEBUG] Current URL: https://firstmls.com/
```

**Solutions:**

1. **Run in non-headless mode to see what's happening:**
   ```bash
   # In .env
   HEADLESS_MODE=false
   ```

2. **Check if selector is correct:**
   - Open browser manually
   - Navigate to https://firstmls.com/
   - Right-click login link → Inspect
   - Verify the selector matches

3. **Increase timeout:**
   ```bash
   # In .env
   LOGIN_TIMEOUT=60
   ```

### Issue 3: 2FA Timeout

**Log Output:**
```
❌ Failed to complete 2FA
Timeout waiting for 2FA elements or user input
```

**Debug Info to Check:**
```
[DEBUG] Looking for OTP input: input#otp-input
```

**Solutions:**

1. **Increase OTP timeout:**
   ```bash
   # In .env
   OTP_TIMEOUT=180
   ```

2. **Verify OTP input selector:**
   - Check if the element ID changed
   - Update `FMLS_OTP_INPUT` in .env if needed

### Issue 4: Session/Cookie Problems

**Log Output:**
```
Failed to load cookies - will perform fresh login
```

**Debug Info to Check:**
```
[DEBUG] Cookie file exists but contains no cookies
[DEBUG] Navigating to login URL to set domain: https://firstmls.sso.remine.com
```

**Solutions:**

1. **Clear old cookies:**
   ```bash
   rm data/cookies/fmls_session.json
   ```

2. **Check cookie file:**
   ```bash
   cat data/cookies/fmls_session.json
   ```

3. **Verify domain in cookies:**
   - Cookies must match the domain (remine.com)
   - Check `saved_at` timestamp

### Issue 5: Browser Driver Issues

**Log Output:**
```
Failed to initialize Chrome WebDriver
```

**Debug Info to Check:**
```
[DEBUG] Scraper driver initialized: False
```

**Solutions:**

1. **Check Chrome/ChromeDriver:**
   ```bash
   google-chrome --version
   chromedriver --version
   ```

2. **Update Selenium:**
   ```bash
   pip install --upgrade selenium
   ```

3. **Use Selenium Manager (automatic):**
   - Already configured in the code
   - Should auto-download correct driver

## Verbose Logging

### Enable Maximum Verbosity

1. **Environment variable:**
   ```bash
   export LOG_LEVEL=DEBUG
   ```

2. **Or in .env:**
   ```bash
   LOG_LEVEL=DEBUG
   ```

### Log File Location

All logs are saved to:
```
logs/scraper.log
```

### Viewing Logs in Real-time

```bash
tail -f logs/scraper.log
```

### Filtering DEBUG Messages

```bash
# Show only DEBUG messages
grep "DEBUG" logs/scraper.log

# Show only errors
grep "ERROR\|❌" logs/scraper.log

# Show authentication flow
grep "Step [0-9]" logs/scraper.log
```

## Testing Checklist

Before running the full application, verify each component:

### 1. GCP Credentials
```bash
# Test Secret Manager access
gcloud secrets list --project=291176869049

# Test secret retrieval
gcloud secrets versions access latest --secret=reBOT_LOGIN --project=291176869049
```

### 2. Browser Driver
```bash
# Test browser initialization
python -c "from selenium import webdriver; driver = webdriver.Chrome(); print('OK'); driver.quit()"
```

### 3. Authentication Flow
```bash
# Run isolated auth test
python test_fmls_auth.py
```

### 4. Full Application
```bash
# Run with debug logging
python main.py
```

## Debug Output Example

Successful authentication should show:

```
[DEBUG] Authentication Configuration ===
[DEBUG] Home URL: https://firstmls.com/
[DEBUG] Login URL: https://firstmls.sso.remine.com
[DEBUG] Dashboard URL: https://firstmls.sso.remine.com/dashboard-v2
...

Step 1: Checking for saved session...
No valid saved session found - will perform fresh login

Step 2: Checking if already authenticated...
Not authenticated - error page detected

Step 3: Performing fresh login...
Step 3.1: Retrieving credentials from Google Secret Manager...
[DEBUG] GOOGLE_APPLICATION_CREDENTIALS: /path/to/key.json
[DEBUG] Attempting to access project: 291176869049
✓ Successfully retrieved secret: reBOT_LOGIN
✓ Successfully retrieved secret: reBOT_PASSWORD
✓ Credentials retrieved successfully

Step 3.2: Navigating to login page...
[DEBUG] Looking for login link: a[href="https://firstmls.sso.remine.com"]
✓ Reached login page

Step 3.3: Submitting login credentials...
[DEBUG] Entered login ID
[DEBUG] Entered password
✓ Login form submitted

Step 3.4: Handling 2FA...
TWO-FACTOR AUTHENTICATION REQUIRED
Enter OTP code: 123456
✓ 2FA completed

Step 3.5: Verifying dashboard access...
✓ Dashboard accessible

Step 3.6: Saving session cookies...
✓ Cookies saved

Step 3.7: Navigating to Remine product...
[DEBUG] Switched to new tab
✓ Successfully navigated to Remine

================================================================================
✓ AUTHENTICATION SUCCESSFUL
================================================================================
```

## Quick Troubleshooting Commands

```bash
# Check environment
env | grep GOOGLE_APPLICATION_CREDENTIALS

# Verify secrets exist
gcloud secrets list --project=291176869049 | grep reBOT

# Test ADC login
gcloud auth application-default login

# Check Chrome
which google-chrome
google-chrome --version

# Check logs
tail -100 logs/scraper.log

# Clear cookies and retry
rm data/cookies/*.json
python main.py
```

## When to Remove DEBUG Code

Once authentication is working consistently (3+ successful runs), remove DEBUG code:

1. **Change log level back to INFO:**
   ```python
   # In config/settings.py
   LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
   ```

2. **Remove or comment out logger.debug() calls** in:
   - src/authenticator.py
   - src/gcp_secrets.py
   - main.py

3. **Keep error logging** (logger.error, logger.warning) for production

---

**Remember:** DEBUG logging includes sensitive paths and URLs. Don't share debug logs publicly without redacting sensitive information.
