# Authentication Breakpoint Usage Guide

## 🎯 Purpose

The authentication breakpoint keeps the browser session open after successful 2FA authentication, allowing you to capture the complete browser state for debugging session persistence issues.

---

## 🚀 How to Use

### Step 1: Enable the Breakpoint

Add to your `.env` file:
```bash
ENABLE_AUTH_BREAKPOINT=true
```

### Step 2: Run Authentication

```bash
python main.py
```

The program will:
1. Start the browser
2. Prompt for 2FA (enter your code)
3. Complete authentication
4. **PAUSE at the authentication breakpoint**

You'll see:
```
✓ Ready to begin scraping operations

================================================================================
🔴 AUTHENTICATION BREAKPOINT ACTIVATED
================================================================================

✓ Browser session is OPEN and AUTHENTICATED
✓ You can now run diagnostic tools while session is active

To capture complete session state, open a new terminal and run:
  python capture_full_session.py

The browser will remain open until you press Enter here.

================================================================================

👉 Press Enter to continue (or Ctrl+C to exit)...
```

### Step 3: Capture Session State (In New Terminal)

While the breakpoint is active, open a **NEW terminal** and run:

```bash
cd ~/reBOT/10-6
source venv/bin/activate
python capture_full_session.py
```

This will capture:
- ✅ All cookies from ALL domains
- ✅ localStorage for each domain
- ✅ sessionStorage for each domain
- ✅ Browser fingerprint
- ✅ Auth token identification

### Step 4: Review Captured Data

```bash
# View analysis
cat data/session_debug/session_analysis.json | jq .

# View complete state
cat data/session_debug/complete_session_state.json | jq . | less
```

### Step 5: Exit Breakpoint

Back in the original terminal:
- Press **Enter** to continue (will validate input file and exit)
- Or press **Ctrl+C** to exit immediately

---

## 📊 What Gets Captured

The `capture_full_session.py` tool captures state from these domains:

1. `https://firstmls.com`
2. `https://firstmls.sso.remine.com`
3. `https://firstmls.sso.remine.com/dashboard-v2`
4. `https://sso.remine.com`
5. `https://fmls.remine.com`
6. `https://fmls.remine.com/daily`
7. `https://remine.com`

For each domain, it captures:
- Cookies (with httpOnly, secure, sameSite flags)
- localStorage items
- sessionStorage items

---

## 🔍 Output Files

After running `capture_full_session.py`, check:

### `data/session_debug/complete_session_state.json`
Complete raw state data from all domains.

### `data/session_debug/session_analysis.json`
Analyzed data highlighting:
- Total cookies found
- **Auth-related cookies** (containing "auth", "token", "session", etc.)
- Total localStorage/sessionStorage items
- **Auth-related storage** (JWT tokens, session IDs, etc.)
- Browser fingerprint details

---

## 💡 What to Look For

### 1. SSO Domain Cookies
```bash
cat data/session_debug/session_analysis.json | jq '.auth_cookies[] | select(.domain | contains("sso"))'
```

**Look for:**
- Cookies from `firstmls.sso.remine.com`
- Cookie names containing: "auth", "session", "token", "id"
- httpOnly cookies (can't be seen in browser DevTools)

### 2. localStorage Tokens
```bash
cat data/session_debug/session_analysis.json | jq '.auth_storage[]'
```

**Look for:**
- `auth0_session`
- `id_token`
- `access_token`
- `refresh_token`
- Any JWT tokens (long base64 strings)

### 3. Cookie Count
```bash
cat data/session_debug/session_analysis.json | jq '.total_cookies'
```

**Expected:** 20-30+ cookies (not just 13)

---

## 🎯 Analysis Checklist

After capturing, verify:

- [ ] **Total cookies:** 20+ (not 13)
- [ ] **SSO cookies present:** Check `firstmls.sso.remine.com` domain
- [ ] **Auth cookies identified:** Should list 3-10 auth-related cookies
- [ ] **localStorage items:** Should have 5-10+ items
- [ ] **Auth storage identified:** Should find JWT tokens or session IDs
- [ ] **Fingerprint captured:** User-agent, timezone, etc.

---

## 🔧 Common Scenarios

### Scenario 1: First Time Capture (Recommended)

```bash
# Terminal 1
python main.py
# Enter 2FA, wait for breakpoint

# Terminal 2
python capture_full_session.py

# Review
cat data/session_debug/session_analysis.json | jq .

# Terminal 1
# Press Enter to exit
```

### Scenario 2: Quick Capture Without Breakpoint

If you don't want to enable the breakpoint permanently:

```bash
# One-time enable
ENABLE_AUTH_BREAKPOINT=true python main.py

# Then capture in another terminal
python capture_full_session.py
```

### Scenario 3: Capture Multiple Times

You can run `capture_full_session.py` multiple times while the breakpoint is active to see if state changes.

---

## ⚠️ Important Notes

1. **Browser Must Stay Open**
   - The breakpoint keeps the browser session alive
   - Don't close the terminal or press Ctrl+C until capture is complete

2. **Capture While Authenticated**
   - Only capture AFTER the breakpoint message appears
   - If you capture before authentication, you'll get incomplete data

3. **File Locations**
   - Captured files go to `data/session_debug/`
   - This directory is created automatically
   - Previous captures are overwritten

4. **Disable After Use**
   - Set `ENABLE_AUTH_BREAKPOINT=false` in `.env` for normal operation
   - The breakpoint is for debugging only

---

## 🎬 Complete Example

```bash
# 1. Enable breakpoint
echo "ENABLE_AUTH_BREAKPOINT=true" >> .env

# 2. Run main.py
python main.py
# → Enter 2FA code
# → Wait for breakpoint message

# 3. In new terminal, capture state
python capture_full_session.py
# → Captures complete state
# → Creates analysis files

# 4. Review findings
cat data/session_debug/session_analysis.json | jq .

# 5. Back to terminal 1, exit
# Press Enter (or Ctrl+C)

# 6. Disable breakpoint for normal use
sed -i 's/ENABLE_AUTH_BREAKPOINT=true/ENABLE_AUTH_BREAKPOINT=false/' .env
```

---

## 📋 Next Steps After Capture

Once you have the captured state:

1. **Identify Real Auth Data**
   ```bash
   # Check what domains have auth cookies
   cat data/session_debug/session_analysis.json | jq '.auth_cookies[].domain' | sort -u

   # Check what localStorage keys exist
   cat data/session_debug/session_analysis.json | jq '.auth_storage[].key'
   ```

2. **Update Session Manager**
   - If auth is in localStorage → Use `enhanced_session_manager.py`
   - If auth is in SSO cookies → Add SSO domain to capture list
   - If auth is httpOnly only → Use Chrome DevTools Protocol

3. **Test Session Persistence**
   - Implement fixes based on captured data
   - Test without breakpoint
   - Verify no 2FA prompt on second run

---

## 🐛 Troubleshooting

### "Browser closes immediately"
- Check that `ENABLE_AUTH_BREAKPOINT=true` is in your `.env`
- Verify `ENABLE_FMLS_AUTH=true` is also set

### "capture_full_session.py fails"
- Make sure browser is still open in terminal 1
- Check that you're in the correct directory
- Verify virtual environment is activated

### "No auth cookies found"
- This is actually useful information!
- It means auth might be in localStorage instead
- Check the `auth_storage` section of analysis

---

## ✅ Success Indicators

You've captured correctly when:

- ✅ `session_analysis.json` exists and has data
- ✅ Total cookies > 20
- ✅ Auth cookies identified (at least 3)
- ✅ localStorage items found
- ✅ SSO domain (`firstmls.sso.remine.com`) has cookies

---

**The authentication breakpoint is your window into the REAL auth state.** 🔍

Use it to understand exactly what needs to be saved for session persistence to work!
