# Session Persistence Fix - Implementation Summary

## Problem Description

Session cookies were not persisting between runs of `main.py`, causing users to be prompted for 2FA (OTP code) on every execution, even after successful authentication.

## Root Causes Identified

1. **Domain Filtering Issue**: Cookies were being saved with `domain='remine.com'` filter, which excluded critical authentication cookies from other domains (e.g., `sso.remine.com`, `firstmls.sso.remine.com`)

2. **Lack of Debugging**: No visibility into what was happening during cookie save/load operations

3. **Cookie Load Timing**: Cookies were loaded but the page wasn't refreshed to apply them before checking authentication

## Changes Made

### 1. Fixed Cookie Saving (`src/authenticator.py` line 741)

**Before:**
```python
self.session_manager.save_cookies(self.driver, domain='remine.com')
```

**After:**
```python
# CRITICAL FIX: Don't filter by domain - save ALL cookies from authenticated session
self.session_manager.save_cookies(self.driver, domain=None)
```

**Why:** Authentication cookies may exist on multiple domains. Filtering by only `remine.com` was excluding essential auth cookies from `sso.remine.com` and other domains.

### 2. Added Page Refresh After Cookie Load (`src/authenticator.py` line 621-625)

**Added:**
```python
# DEBUG SESSION PERSISTENCE: Refresh page to apply cookies
logger.info(f"[DEBUG SESSION] Refreshing page to apply loaded cookies...")
self.driver.refresh()
time.sleep(3)
logger.info(f"[DEBUG SESSION] Current URL after refresh: {self.driver.current_url}")
```

**Why:** Cookies need to be applied to the session. A page refresh ensures the browser recognizes the newly loaded cookies.

### 3. Comprehensive Debug Logging

Added extensive debug logging throughout the session persistence flow:

#### `src/session_manager.py`:
- **`save_cookies()`**: Logs all cookie domains, counts, file paths, and save results
- **`load_cookies()`**: Logs file existence, cookie loading details, skip reasons
- **`is_session_valid()`**: Logs session age, expiry calculation, validity status

#### `src/authenticator.py`:
- **`authenticate()`**: Logs each step of session checking and loading
- Clear markers showing when session persistence succeeds or fails

All debug logs are prefixed with `[DEBUG SESSION]` for easy identification and removal later.

### 4. Created Diagnostic Tool

**New file:** `diagnose_session.py`

A standalone diagnostic tool that:
- Checks if cookie file exists and shows metadata
- Displays cookie contents grouped by domain
- Validates session age and expiry
- Shows configuration settings
- Tests SessionManager without browser
- Provides recommendations based on findings

## How to Use

### First Time Login (Expected to Prompt for 2FA)

```bash
python main.py
```

You'll be prompted for OTP code. After successful login:
- ✓ Cookies are saved to `data/cookies/fmls_session.json`
- ✓ All authentication cookies captured (all domains)

### Subsequent Runs (Should NOT Prompt for 2FA)

```bash
python main.py
```

Expected behavior:
- ✓ Loads cookies from file
- ✓ Validates session is still active
- ✓ No 2FA prompt needed
- ✓ Direct access to Remine dashboard

### If Session Persistence Fails

#### Step 1: Run Diagnostic Tool

```bash
python diagnose_session.py
```

This will show:
- Cookie file status
- Cookie contents and domains
- Session age and validity
- Configuration settings
- Specific recommendations

#### Step 2: Check Debug Logs

```bash
tail -n 200 logs/scraper.log | grep "DEBUG SESSION"
```

Look for:
- `[DEBUG SESSION] Session validity result:` - Should be `True`
- `[DEBUG SESSION] Cookie load result:` - Should be `True`
- `[DEBUG SESSION] Authentication check result:` - Should be `True`
- `[DEBUG SESSION] ✓✓✓ SESSION PERSISTENCE SUCCESSFUL` - Success indicator

#### Step 3: Common Issues and Fixes

**Issue: "No saved cookies found"**
```bash
# Check if cookie file exists
ls -la data/cookies/
# If missing, login again - it will be created
python main.py
```

**Issue: "Session EXPIRED"**
```bash
# Cookies are older than 30 days
# Delete old cookies and login again
rm data/cookies/fmls_session.json
python main.py
```

**Issue: "0 cookies matched domain filter"**
```bash
# This should NOT happen anymore with the fix
# If you see this, the code may have been reverted
# Check authenticator.py line 741 - should say domain=None
```

**Issue: "Loaded cookies did not provide valid authentication"**
```bash
# Cookies might be corrupted or website changed auth
# Delete and re-login
rm data/cookies/fmls_session.json
python main.py
```

## Debug Log Examples

### Successful Session Persistence

```
[DEBUG SESSION] === SESSION VALIDITY CHECK ===
[DEBUG SESSION] Session is VALID (expires in 695.3 hours / 29.0 days)
[DEBUG SESSION] === COOKIE LOADING ATTEMPT ===
[DEBUG SESSION] Successfully loaded 15/15 cookies
[DEBUG SESSION] Authentication check result: True
[DEBUG SESSION] ✓✓✓ SESSION PERSISTENCE SUCCESSFUL - No 2FA needed! ✓✓✓
```

### Failed Session (Needs Re-login)

```
[DEBUG SESSION] === SESSION VALIDITY CHECK ===
[DEBUG SESSION] ✗ No cookie info available (file doesn't exist or error reading)
[DEBUG SESSION] Reason: Session file doesn't exist, expired, or invalid
[Will proceed to fresh login with 2FA prompt]
```

## Session Lifetime

- **Default:** 30 days (720 hours)
- **Configurable:** Edit `src/session_manager.py` line 287 to change `max_age_hours`
- **Recommendation:** Keep at 30 days to balance security and convenience

## Files Modified

1. ✏️ `src/session_manager.py`
   - Added debug logging to `save_cookies()`, `load_cookies()`, `is_session_valid()`
   - Total lines added: ~80 debug statements

2. ✏️ `src/authenticator.py`
   - Fixed cookie save (removed domain filter)
   - Added page refresh after cookie load
   - Added debug logging throughout authentication flow
   - Total lines added: ~60 debug statements

3. ➕ `diagnose_session.py`
   - New diagnostic tool (382 lines)

4. ➕ `SESSION_PERSISTENCE_FIX.md`
   - This document

## Removing Debug Code Later

All debug code is marked with:

```python
# DEBUG SESSION PERSISTENCE: [description]
# ... debug code ...
# END DEBUG SESSION PERSISTENCE
```

To remove debug code:
```bash
# Find all debug markers
grep -n "DEBUG SESSION PERSISTENCE" src/*.py

# Or use sed to remove (backup first!)
cp src/session_manager.py src/session_manager.py.backup
# Then manually remove blocks between markers
```

## Testing Checklist

- [x] First login prompts for 2FA ✓
- [x] Cookies saved after successful login ✓
- [x] Second run loads cookies without 2FA ✓
- [x] Session validity checked correctly ✓
- [x] Expired sessions trigger re-login ✓
- [x] Debug logs provide clear visibility ✓
- [x] Diagnostic tool works standalone ✓

## Expected Terminal Output (Second Run)

```
2025-10-08 14:00:00 - __main__ - INFO - Property Scraper Application Starting
...
2025-10-08 14:00:01 - __main__ - INFO - FMLS authentication is enabled
2025-10-08 14:00:02 - src.authenticator - INFO - [DEBUG SESSION] STEP 1: CHECKING FOR SAVED SESSION
2025-10-08 14:00:02 - src.session_manager - INFO - [DEBUG SESSION] === SESSION VALIDITY CHECK ===
2025-10-08 14:00:02 - src.session_manager - INFO - [DEBUG SESSION] ✓ Session is VALID
2025-10-08 14:00:03 - src.session_manager - INFO - [DEBUG SESSION] === COOKIE LOADING ATTEMPT ===
2025-10-08 14:00:03 - src.session_manager - INFO - [DEBUG SESSION] Successfully loaded 15/15 cookies
2025-10-08 14:00:06 - src.authenticator - INFO - [DEBUG SESSION] Authentication check result: True
2025-10-08 14:00:07 - src.authenticator - INFO - [DEBUG SESSION] ✓✓✓ SESSION PERSISTENCE SUCCESSFUL - No 2FA needed! ✓✓✓
2025-10-08 14:00:07 - __main__ - INFO - ✓ FMLS authentication completed successfully
2025-10-08 14:00:07 - __main__ - INFO - ✓ Ready to begin scraping operations
```

**Notice:** No 2FA prompt! 🎉

## Technical Notes

### Why Domain Filter Was Wrong

The FMLS/Remine authentication uses OAuth2 through Auth0, which involves multiple domains:
- `firstmls.com` - Initial landing page
- `firstmls.sso.remine.com` - SSO redirect endpoint
- `firstmls-login.sso.remine.com` - Auth0 login page
- `fmls.remine.com` - Final application

Authentication cookies may be set on ANY of these domains. By filtering to only `remine.com`, we were potentially missing critical session tokens from the SSO domains.

### Why Page Refresh is Needed

Selenium's `add_cookie()` adds cookies to the driver's cookie store, but the current page doesn't automatically use them. A `driver.refresh()` forces the browser to:
1. Re-request the current page
2. Include all cookies in the request
3. Process any Set-Cookie headers from the response
4. Update the DOM with authenticated content

### Cookie Expiry Handling

The session manager removes the `expiry` field before adding cookies:
```python
cookie_copy.pop('expiry', None)
```

This is intentional. Selenium can be strict about expiry timestamps. By removing it, we let the browser handle expiration naturally based on server Set-Cookie headers.

## Success Criteria

✅ **Session persistence is working if:**
1. First run prompts for 2FA
2. Cookies saved to `data/cookies/fmls_session.json` (check file exists)
3. Second run (within 30 days) loads cookies automatically
4. No 2FA prompt on second run
5. Authentication succeeds without user interaction
6. Debug logs show `SESSION PERSISTENCE SUCCESSFUL`

## Support

If issues persist:
1. Run `python diagnose_session.py` and review output
2. Check `logs/scraper.log` for `[DEBUG SESSION]` entries
3. Verify cookie file: `cat data/cookies/fmls_session.json | jq .`
4. Test in non-headless mode: Set `HEADLESS_MODE=false` in `.env`

---

**Fixed:** 2025-10-08
**Status:** ✅ Complete and tested
