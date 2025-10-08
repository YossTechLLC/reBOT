# FMLS Authentication Implementation

This document describes the FMLS (First MLS) authentication system that has been integrated into the reBOT scraper.

## Overview

The FMLS authentication system enables automatic login to the FMLS/Remine platform with the following features:

- **Google Secret Manager Integration**: Credentials stored securely in GCP Secret Manager
- **Session Persistence**: Browser cookies saved locally for "remember browser" functionality
- **2FA Support**: Interactive terminal prompt for OTP codes
- **Automatic Re-authentication**: Detects expired sessions and re-authenticates automatically
- **Error Handling**: Comprehensive error handling and logging

## Architecture

### New Modules

1. **`src/gcp_secrets.py`** - Google Cloud Secret Manager integration
   - Retrieves credentials using Application Default Credentials (ADC)
   - Handles GCP authentication errors

2. **`src/session_manager.py`** - Browser session/cookie management
   - Saves cookies to JSON file after successful login
   - Loads cookies on subsequent runs to skip re-authentication
   - Validates session age (30 days default)

3. **`src/authenticator.py`** - FMLS authentication orchestration
   - Manages complete authentication flow
   - Handles login form submission
   - Processes 2FA prompts
   - Navigates to final Remine dashboard

### Authentication Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Check for Saved Session                                  │
│    - Load cookies from file                                  │
│    - Validate session age                                    │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Verify Authentication Status                             │
│    - Navigate to dashboard URL                               │
│    - Check for error page                                    │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
         ┌────┴────┐
         │ Valid?  │
         └────┬────┘
              │
      ┌───────┴────────┐
      │ YES            │ NO
      │                │
      ▼                ▼
┌───────────┐    ┌──────────────────────────────┐
│ Navigate  │    │ 3. Perform Fresh Login       │
│ to Remine │◄───│    - Get credentials from SM  │
└───────────┘    │    - Fill login form          │
                 │    - Submit credentials       │
                 └──────────┬───────────────────┘
                            │
                            ▼
                 ┌──────────────────────────────┐
                 │ 4. Handle 2FA                 │
                 │    - Prompt user for OTP      │
                 │    - Enter OTP code           │
                 │    - Check "remember browser" │
                 │    - Submit 2FA               │
                 └──────────┬───────────────────┘
                            │
                            ▼
                 ┌──────────────────────────────┐
                 │ 5. Save Session               │
                 │    - Save cookies to file     │
                 └──────────┬───────────────────┘
                            │
                            ▼
                 ┌──────────────────────────────┐
                 │ 6. Navigate to Remine         │
                 │    - Click Remine link        │
                 │    - Switch to new tab        │
                 │    - Verify final URL         │
                 └──────────────────────────────┘
```

## Configuration

### Environment Variables

Add these to your `.env` file (see `.env.example` for reference):

```bash
# FMLS Authentication
ENABLE_FMLS_AUTH=true
GCP_PROJECT_ID=291176869049
SECRET_LOGIN_ID=reBOT_LOGIN
SECRET_PASSWORD=reBOT_PASSWORD

# FMLS URLs
FMLS_HOME_URL=https://firstmls.com/
FMLS_LOGIN_URL=https://firstmls.sso.remine.com
FMLS_DASHBOARD_URL=https://firstmls.sso.remine.com/dashboard-v2
FMLS_REMINE_URL=https://fmls.remine.com
FMLS_REMINE_DAILY_URL=https://fmls.remine.com/daily

# Authentication Timeouts
OTP_TIMEOUT=120
LOGIN_TIMEOUT=30
```

### CSS Selectors

The following selectors are pre-configured but can be customized in `.env`:

```bash
FMLS_LOGIN_LINK_SELECTOR=a[href="https://firstmls.sso.remine.com"]
FMLS_LOGIN_ID_INPUT=input[name='username']
FMLS_PASSWORD_INPUT=input[name='password']
FMLS_LOGIN_BUTTON=button#btn-login
FMLS_OTP_INPUT=input#otp-input
FMLS_REMEMBER_CHECKBOX=input#remember-browser-checkbox-2
FMLS_OTP_CONTINUE_BUTTON=button#btn-verify-login-otp
FMLS_REMINE_PRODUCT_LINK=a._productItem_15qlz_1[href="https://fmls.remine.com"]
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `google-cloud-secret-manager>=2.16.0`
- All existing dependencies

### 2. Configure Google Cloud Authentication

#### Option A: Service Account (Recommended for VMs)

1. Create a service account in GCP Console
2. Grant "Secret Manager Secret Accessor" role
3. Download JSON key file
4. Set environment variable:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

#### Option B: Application Default Credentials (ADC)

```bash
gcloud auth application-default login
```

### 3. Store Credentials in Secret Manager

```bash
# Create secrets
echo -n "your-login-id" | gcloud secrets create reBOT_LOGIN \
    --data-file=- \
    --project=291176869049

echo -n "your-password" | gcloud secrets create reBOT_PASSWORD \
    --data-file=- \
    --project=291176869049
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

## Usage

### Running the Main Application

```bash
python main.py
```

When FMLS authentication is enabled, the application will:
1. Initialize the browser
2. Attempt to use saved session (if available)
3. Perform fresh login if needed (with 2FA prompt)
4. Navigate to Remine dashboard
5. Proceed with scraping operations

### Testing Authentication Only

To test authentication without scraping:

```bash
python test_fmls_auth.py
```

This script:
- Tests the complete authentication flow
- Verifies all components work correctly
- Keeps browser open for 10 seconds for manual verification
- Provides detailed logging

### 2FA Workflow

When 2FA is required, you'll see:

```
================================================================================
TWO-FACTOR AUTHENTICATION REQUIRED
================================================================================

Please check your authentication method (email, SMS, app) for the OTP code.
You have 120 seconds to enter the code.

Enter OTP code: [your input here]
```

After entering the code, the system:
- Submits the OTP
- Checks "remember browser" to avoid 2FA for 30 days
- Saves cookies for future sessions

## Directory Structure

```
10-6/
├── data/
│   └── cookies/              # Session cookies storage
│       ├── .gitkeep         # Placeholder (committed)
│       └── fmls_session.json # Saved cookies (NOT committed)
├── src/
│   ├── authenticator.py     # FMLS authentication logic
│   ├── gcp_secrets.py       # Secret Manager integration
│   ├── session_manager.py   # Cookie persistence
│   └── ...
├── config/
│   └── settings.py          # Updated with FMLS config
├── main.py                  # Updated with auth flow
├── test_fmls_auth.py       # Authentication test script
└── FMLS_AUTH_README.md     # This file
```

## Session Persistence

### Cookie Storage

Cookies are saved to `data/cookies/fmls_session.json`:

```json
{
  "saved_at": "2025-01-08T12:34:56.789012",
  "domain": "remine.com",
  "cookie_count": 15,
  "cookies": [...]
}
```

### Session Validity

- Default session lifetime: **30 days** (720 hours)
- Configurable via `is_session_valid(max_age_hours=720)`
- Expired sessions trigger fresh login

### Security Notes

- Cookie files are excluded from git (via `.gitignore`)
- Cookies contain authentication tokens - keep secure
- Do not commit cookie files to version control

## Troubleshooting

### "Permission denied" accessing Secret Manager

**Solution**: Ensure service account has "Secret Manager Secret Accessor" role:

```bash
gcloud projects add-iam-policy-binding 291176869049 \
    --member="serviceAccount:your-sa@project.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

### "Secret not found"

**Solution**: Verify secret names and project ID:

```bash
gcloud secrets list --project=291176869049
```

### "Timeout waiting for login elements"

**Possible causes**:
- Website structure changed (update CSS selectors)
- Network latency (increase `LOGIN_TIMEOUT`)
- Headless mode issues (try `HEADLESS_MODE=false`)

### "2FA timeout"

**Solution**: Increase OTP timeout in `.env`:

```bash
OTP_TIMEOUT=180  # 3 minutes
```

### Browser closes immediately after login

**Solution**: This is expected. The session is saved in cookies. On next run, cookies will be loaded and browser will skip login.

## Feature Flags

### Disable FMLS Authentication

To disable FMLS auth and use standard scraping:

```bash
# In .env
ENABLE_FMLS_AUTH=false
```

## Logging

Authentication events are logged to `logs/scraper.log`:

```
2025-01-08 12:34:56 - __main__ - INFO - Starting FMLS authentication process
2025-01-08 12:34:57 - src.authenticator - INFO - Found valid saved session
2025-01-08 12:34:58 - src.session_manager - INFO - Loaded 15/15 cookies
2025-01-08 12:35:00 - src.authenticator - INFO - Already authenticated - dashboard accessible
2025-01-08 12:35:02 - src.authenticator - INFO - Successfully navigated to Remine
```

## Next Steps

The authentication system is now complete and ready for use. The next phase is to implement the FMLS-specific scraping logic:

1. Navigate within Remine interface
2. Search for addresses
3. Extract property data
4. Store results in database

---

**Built with ❤️ for FMLS/Remine automation**
