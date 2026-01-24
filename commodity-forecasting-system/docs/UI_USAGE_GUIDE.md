# VOLATILITY PREDICTION UI - USAGE GUIDE

**Created:** 2026-01-17
**Status:** ✅ Phase 2 Complete (80%)
**Version:** 1.0.0

---

## 📋 OVERVIEW

This interactive UI provides a complete interface for the HMM + TimesFM volatility prediction system. The UI enables:

- 📊 Data loading and visualization
- 🎯 HMM model training with parameter control
- 🔮 TimesFM forecasting (optional)
- 📈 Real-time predictions with confidence scoring
- 🧠 Feature importance analysis
- 💰 Trading strategy recommendations
- 🔍 Walk-forward validation

---

## 🚀 QUICK START

### 1. Installation

First, ensure you have all dependencies installed:

```bash
# Navigate to project directory
cd commodity-forecasting-system

# Install UI dependencies
pip install -r requirements-ui.txt

# This installs:
# - streamlit>=1.40.0
# - plotly>=5.18.0
# - shap>=0.44.0
# - matplotlib>=3.8.0
# - seaborn>=0.13.0
# - kaleido>=0.2.1
# - streamlit-aggrid>=0.3.4
# - streamlit-option-menu>=0.3.6
```

### 2. Launch the UI

```bash
# Option 1: Using symlink (recommended)
streamlit run app.py

# Option 2: Direct path
streamlit run src/ui/app.py

# The UI will open in your default browser at: http://localhost:8501
```

### 3. First-Time Setup

Once the UI loads, follow these steps:

1. **Load Data** (Sidebar)
   - Click "🔄 Load/Refresh Data"
   - Wait for SPY and VIX data to download
   - Default: 180 days of history

2. **Train HMM Model** (Sidebar)
   - Set number of regimes (default: 3)
   - Select features (default: all 5 features)
   - Set training iterations (default: 100)
   - Click "🚀 Train HMM"
   - Wait for convergence confirmation

3. **Run Prediction** (Sidebar)
   - Click "🔮 Run Prediction"
   - View results in "Prediction Dashboard" tab

---

## 🎨 UI STRUCTURE

### Header
- **Status Bar** - Shows model status (HMM/TimesFM) and latest data date
- **System Info** - Quick overview of what's loaded and ready

### Sidebar Controls

#### 📊 Data Settings
- **History Days** - Slider to select amount of historical data (30-365 days)
- **Load/Refresh Data** - Download latest data from Alpaca and Yahoo Finance

#### 🎯 HMM Parameters
- **Number of Regimes** - 2-5 volatility states (typically 3)
- **HMM Features** - Multi-select for feature selection
  - `overnight_gap_abs` - Overnight gap magnitude
  - `range_ma_5` - 5-day average range
  - `vix_level` - VIX level
  - `volume_ratio` - Volume relative to average
  - `range_std_5` - 5-day range standard deviation
- **Training Iterations** - Max iterations for EM algorithm (50-500)
- **Train HMM** - Button to train new model
- **Load Pre-trained HMM** - Load existing model from `models/hmm_volatility.pkl`

#### 🔮 TimesFM Parameters
- **Enable TimesFM** - Checkbox to activate foundation model forecasting
- **Device** - Select CPU or CUDA (GPU)
- **Load TimesFM** - Download and initialize TimesFM (~800MB)

#### ⚙️ Model Configuration
- **Confidence Threshold** - Minimum score to trigger trade signal (0-100)
- **Advanced: Confidence Weights** - Adjust component weights:
  - Regime Weight (default: 0.4)
  - TimesFM Weight (default: 0.4)
  - Feature Weight (default: 0.2)

#### 🚀 Actions
- **Run Prediction** - Generate prediction for latest data point

### Main Panel - Tabs

#### Tab 1: 📈 Prediction Dashboard
**Purpose:** View current prediction and confidence metrics

**Components:**
1. **Regime Detection Card**
   - Current regime (low/normal/high volatility)
   - Expected volatility percentage
   - Regime confidence score

2. **Confidence Score Card**
   - Total confidence score (0-100)
   - Visual badge (green/yellow/red based on threshold)
   - Current threshold value

3. **Trading Decision Card**
   - TRADE or SKIP recommendation
   - Brief rationale

4. **Confidence Gauge**
   - Visual gauge showing score relative to threshold
   - Delta from threshold

5. **Price Action Chart**
   - Candlestick chart with regime-colored backgrounds
   - Interactive (zoom, pan, hover)

6. **Volatility Time Series**
   - Intraday range % over time
   - Threshold line overlay
   - Regime backgrounds

7. **Regime Probabilities**
   - Bar chart showing probability of each regime
   - Color-coded by regime type

#### Tab 2: 📊 Model Explanation
**Purpose:** Understand why the model made its prediction

**Components:**
1. **Feature Contributions**
   - Horizontal bar chart showing feature values
   - Positive/negative impact visualization

2. **Detailed Feature Analysis**
   - **Regime Analysis** - Current regime interpretation
   - **Feature Signals** - Individual feature analysis with context
     - Overnight gap interpretation
     - VIX change analysis
     - Volume surge detection
     - Range expansion analysis

3. **Confidence Score Breakdown**
   - Regime component score
   - TimesFM component score (if enabled)
   - Feature component score

#### Tab 3: 🔍 Validation Results
**Purpose:** Review walk-forward validation performance

**Status:** 🚧 Placeholder (run `scripts/validate_volatility_mvp.py` directly)

**Planned Components:**
- Validation metrics table
- Confusion matrix heatmap
- Win rate over time
- Accuracy chart

#### Tab 4: 💰 Trading Strategy
**Purpose:** Get actionable trading recommendations

**Components:**
1. **Position Sizing Controls**
   - Account Size input ($1,000 - $1,000,000)
   - Max Risk per Trade slider (0.5% - 5.0%)

2. **Strategy Recommendation**
   - **SKIP Mode (Low Volatility)**
     - Reason for skipping
     - Current price and predicted volatility

   - **TRADE Mode (Normal/High Volatility)**
     - Strategy name (Wide Strangle or Narrow Strangle)
     - Current price and expected move
     - Call and Put strike prices
     - Spread width
     - Entry timing (market open)
     - Exit rules (profit targets, time exit, stop loss)
     - Risk/reward metrics:
       - Credit received
       - Max profit
       - Max loss estimate
       - Breakeven prices
       - Win probability
     - Position sizing:
       - Recommended contracts
       - Total risk in dollars and %
       - Max profit potential
       - Confidence adjustment

3. **P&L Distribution**
   - 🚧 Placeholder for expected P&L visualization

---

## 🔧 CONFIGURATION

### Streamlit Config (`.streamlit/config.toml`)

```toml
[theme]
primaryColor = "#1f77b4"          # Blue accents
backgroundColor = "#ffffff"        # White background
secondaryBackgroundColor = "#f0f2f6"  # Light gray
textColor = "#262730"              # Dark text

[server]
port = 8501
enableCORS = false
maxUploadSize = 200

[browser]
gatherUsageStats = false
```

### Data Sources

- **SPY Data** - Alpaca Markets API (daily bars)
  - Requires: `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`
  - Current credentials hardcoded in `data_manager.py` (change for production)

- **VIX Data** - Yahoo Finance (yfinance library)
  - No API key required
  - Downloaded on-demand

---

## 📊 WORKFLOW EXAMPLES

### Example 1: Daily Prediction Workflow

1. **Morning Routine (Before Market Open)**
   ```
   Launch UI → Load Data (yesterday's close) → Run Prediction
   ```

2. **Review Results**
   - Check confidence score
   - Review regime detection
   - Examine feature signals

3. **Make Decision**
   - If confidence >= threshold → Go to Strategy tab
   - Review spread recommendation
   - Note entry/exit prices
   - Calculate position size

4. **Execute Trade**
   - Use strikes from recommendation
   - Enter at market open
   - Set exit rules per recommendation

### Example 2: Model Tuning Workflow

1. **Load Historical Data**
   ```
   Set History Days: 365 → Load Data
   ```

2. **Experiment with HMM Parameters**
   ```
   Try n_regimes = 3 → Train → Review regime separation
   Try n_regimes = 4 → Train → Compare
   ```

3. **Adjust Confidence Threshold**
   ```
   Lower threshold: More trades, lower win rate
   Higher threshold: Fewer trades, higher win rate
   ```

4. **Test Weights**
   ```
   Adjust Regime/TimesFM/Feature weights
   Re-run predictions to see impact
   ```

### Example 3: Validation Workflow

1. **Train on Full Dataset**
   ```
   History: 365 days → Train HMM
   ```

2. **Run External Validation**
   ```bash
   # In terminal
   python scripts/validate_volatility_mvp.py
   ```

3. **Review Validation Results**
   - Check `outputs/validation_summary.json`
   - Review `outputs/validation_results.csv`

---

## 🐛 TROUBLESHOOTING

### Issue: "Data not loaded" warning

**Solution:**
- Click "Load/Refresh Data" in sidebar
- Ensure internet connection is active
- Check Alpaca API credentials if SPY data fails

### Issue: "HMM model not trained" warning

**Solution:**
- Click "Train HMM" in sidebar after loading data
- Or click "Load Pre-trained HMM" if you have a saved model

### Issue: TimesFM not loading

**Solution:**
- TimesFM is optional - system works without it
- If needed:
  - Ensure checkbox is enabled
  - Click "Load TimesFM"
  - Wait for 800MB checkpoint download
  - First load may take 5-10 minutes

### Issue: Charts not displaying

**Solution:**
- Ensure Plotly is installed: `pip install plotly>=5.18.0`
- Check browser JavaScript is enabled
- Try refreshing the page

### Issue: Slow performance

**Solution:**
- Reduce history days (try 90 instead of 180)
- Disable TimesFM if not needed
- Use CPU instead of CUDA if GPU memory is limited

---

## 📈 PERFORMANCE NOTES

### Data Loading
- **Initial load:** 5-10 seconds for 180 days
- **Cached load:** < 1 second (data cached for 1 hour)

### HMM Training
- **3 regimes, 100 iterations:** ~5 seconds on 180 days
- **5 regimes, 500 iterations:** ~30 seconds on 180 days

### Prediction
- **HMM-only mode:** < 1 second
- **With TimesFM:** 2-5 seconds (first run), < 1 second (cached)

---

## 🔄 NEXT STEPS

### Phase 3: Data Visualization Enhancements
- [ ] Add regime transition heatmap
- [ ] Implement SHAP waterfall plots
- [ ] Create P&L distribution charts
- [ ] Add historical regime timeline

### Phase 4: Advanced Features
- [ ] Multi-symbol support (QQQ, IWM, DIA)
- [ ] Real-time data mode
- [ ] Email/SMS alerts for trade signals
- [ ] CSV export for all charts
- [ ] PDF report generation

### Phase 5: Validation Integration
- [ ] In-UI validation runner
- [ ] Interactive confusion matrix
- [ ] Parameter optimization tool
- [ ] Backtesting simulator

---

## 📚 REFERENCES

- **Architecture Checklist:** `docs/UI_ARCHITECTURE_CHECKLIST.md`
- **Progress Tracker:** `docs/CHECKLIST_PROGRESS.md`
- **Week 2 Status:** `docs/WEEK2_COMPLETION_STATUS.md`
- **Streamlit Docs:** https://docs.streamlit.io/
- **Plotly Docs:** https://plotly.com/python/

---

## 🎯 KEY METRICS

**Code Statistics (Phase 2):**
- Total Lines: ~2,500
- Total Size: ~75 KB
- Modules: 7 files
- Dependencies: 10 packages

**Feature Coverage:**
- Data Loading: ✅ Complete
- HMM Control: ✅ Complete
- TimesFM Control: ✅ Complete
- Prediction Dashboard: ✅ Complete
- Explainability: ✅ Complete (simplified)
- Strategy Recommendations: ✅ Complete
- Validation UI: ⏸️ Placeholder

---

_Last Updated: 2026-01-17 10:50 AM_
