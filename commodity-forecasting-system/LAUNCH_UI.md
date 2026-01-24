# HOW TO LAUNCH THE VOLATILITY PREDICTION UI

**Last Updated:** 2026-01-17
**Status:** ✅ Ready to Launch

---

## 🚀 QUICK START

### Step 1: Ensure You're in the Project Directory

```bash
cd /mnt/c/Users/YossTech/Desktop/2025/reBOT/commodity-forecasting-system
```

### Step 2: Kill Any Existing Streamlit Processes (if needed)

If you see "Port 8501 is not available", run:

```bash
pkill -f streamlit
# Wait a moment
sleep 2
```

### Step 3: Launch the UI

```bash
streamlit run app.py
```

**Expected Output:**
```
  You can now view your Streamlit app in your browser.

  URL: http://localhost:8501
```

### Step 4: Open in Browser

The UI should automatically open in your default browser.
If not, manually open: `http://localhost:8501`

---

## ✅ WHAT YOU SHOULD SEE

When the UI loads successfully, you'll see:

**Header:**
- "📈 Volatility Prediction System"
- "HMM + TimesFM for 0DTE/1DTE Options Trading"

**Status Bar:**
- ❌ HMM Not Loaded (initially)
- ℹ️ TimesFM Not Loaded (initially)
- ⚠️ No Data Loaded (initially)

**Sidebar:** Controls for data loading, HMM training, etc.

**Main Panel:** 4 tabs (Prediction, Explanation, Validation, Strategy)

---

## 🐛 TROUBLESHOOTING

### Problem: "Port 8501 is not available"

**Solution:**
```bash
pkill -f streamlit
sleep 2
streamlit run app.py
```

### Problem: "ModuleNotFoundError: No module named 'ui'"

**This should be fixed!** But if you still see it:

1. Verify you're running from the project root:
   ```bash
   pwd
   # Should show: .../commodity-forecasting-system
   ```

2. Check the symlink exists:
   ```bash
   ls -la app.py
   # Should show: app.py -> src/ui/app.py
   ```

3. Run the test script:
   ```bash
   python test_ui_imports.py
   ```

If all imports succeed in the test but Streamlit still fails, there may be a caching issue. Try:
```bash
rm -rf .streamlit/cache
streamlit run app.py
```

### Problem: Browser doesn't open automatically

**Solution:**
Manually open `http://localhost:8501` in your browser

### Problem: "Connection refused" in browser

**Solution:**
- Check that Streamlit is still running in the terminal
- Verify no firewall is blocking port 8501
- Try `http://127.0.0.1:8501` instead

---

## 📱 ACCESSING FROM ANOTHER DEVICE

If you want to access the UI from another device on your network:

1. Find your local IP address:
   ```bash
   ip addr show | grep "inet " | grep -v 127.0.0.1
   ```

2. Update `.streamlit/config.toml`:
   ```toml
   [server]
   serverAddress = "0.0.0.0"
   ```

3. Launch Streamlit and access from another device:
   ```
   http://YOUR_IP_ADDRESS:8501
   ```

**Warning:** Only do this on a trusted network!

---

## 🎯 FIRST-TIME USAGE WORKFLOW

Once the UI is running:

### 1. Load Data (Required First Step)

- In sidebar, click **"🔄 Load/Refresh Data"**
- Wait 5-10 seconds for data download
- You should see: "✅ Loaded XXX days"

### 2. Train or Load HMM Model

**Option A: Train New Model**
- Click **"🚀 Train HMM"** in sidebar
- Wait ~5 seconds
- You should see: "✅ HMM Training Complete"

**Option B: Load Pre-trained Model**
- Click **"📁 Load Pre-trained HMM"**
- Requires `models/hmm_volatility.pkl` from Week 2

### 3. Generate Prediction

- Click **"🔮 Run Prediction"** in sidebar
- Results appear in all tabs
- Check confidence score and trading decision

### 4. Explore Tabs

- **Tab 1:** View prediction dashboard and charts
- **Tab 2:** Understand why the model made this prediction
- **Tab 3:** Validation results (placeholder)
- **Tab 4:** Get trading strategy recommendation

---

## ⚙️ ADVANCED OPTIONS

### Adjust Confidence Threshold

In sidebar under "Model Configuration":
- Lower threshold (20-30) = more trades
- Higher threshold (50-70) = fewer trades, higher confidence

### Customize HMM Parameters

Before training:
- Number of Regimes (2-5)
- Select features
- Training iterations

### Enable TimesFM (Optional)

1. Check "Enable TimesFM" in sidebar
2. Select device (CPU or CUDA)
3. Click "📥 Load TimesFM"
4. Wait for ~800MB download (first time only)

---

## 🛑 STOPPING THE UI

To stop Streamlit:
- Press `Ctrl+C` in the terminal where it's running
- Or close the terminal window

---

## 📚 MORE DOCUMENTATION

- **Complete Usage Guide:** `docs/UI_USAGE_GUIDE.md`
- **Bugfix Documentation:** `docs/BUGFIX_IMPORT_PATHS.md`
- **Progress Tracker:** `docs/CHECKLIST_PROGRESS.md`

---

## ✅ VERIFICATION CHECKLIST

Before launching, verify:
- [ ] You're in the `commodity-forecasting-system` directory
- [ ] Virtual environment is activated (`.venv`)
- [ ] Dependencies installed: `pip install -r requirements-ui.txt`
- [ ] No other Streamlit instance running on port 8501

After launching, verify:
- [ ] No import errors in terminal
- [ ] Browser opens to UI
- [ ] Sidebar controls are visible
- [ ] 4 tabs are present
- [ ] Status bar shows initial state

---

**Ready to go! Run:** `streamlit run app.py`
