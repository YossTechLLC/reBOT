# 🎯 INTRADAY REGIME-SHIFT OPTIONS TRADING SYSTEM - IMPLEMENTATION CHECKLIST

**Project Start Date:** 2026-01-17
**Target Completion:** Week 14 (2026-04-18)
**Current Phase:** Phase 0 - Validation Sprint

---

## 📊 PROJECT OVERVIEW

**Objective:** Build an automated system to identify intraday S&P 500 regime shifts (9:30 AM → 12:00 PM) that create profitable 0DTE/1DTE option spread opportunities.

**Success Criteria:**
- ✅ Backtest Sharpe Ratio > 1.5
- ✅ Win Rate > 55%
- ✅ Max Drawdown < 25%
- ✅ Paper Trading validates backtest (±30%)

**Key Innovation:** Hierarchical HMM (daily/hourly/15-min) + TimesFM forecasting + Black-Scholes Greeks

---

## 🚦 PHASE 0: VALIDATION SPRINT (Week 1)
**Timeline:** 2026-01-17 → 2026-01-24
**Effort:** 15 hours
**Cost:** $0 (free data)

### **Objective**
Prove regime shifts are predictable and profitable BEFORE building full infrastructure.

### **Tasks**

#### Data Collection
- [ ] Download 1 year SPY 15-minute bars (yfinance)
  - Start: 2025-01-17, End: 2026-01-16
  - Expected: ~252 days × 26 bars/day = 6,552 bars
  - Validate: No gaps, correct timestamps (9:30-16:00 ET)

- [ ] Download VIX daily data (same period)
  - Source: yfinance (^VIX)
  - Fields: Open, High, Low, Close

- [ ] Create data validation report
  - Missing bars: < 1%
  - Price outliers: Flag and review
  - Volume spikes: Identify unusual days

**Files Created:**
- `/commodity-forecasting-system/data/intraday/spy_15min_2025-2026.csv`
- `/commodity-forecasting-system/notebooks/phase0_validation.ipynb`

#### Feature Engineering
- [ ] Compute intraday features (15-min bars)
  - Returns (close-to-close)
  - Volatility (rolling 5-bar std)
  - Volume ratio (vs 20-day avg at same time)
  - Range (high-low / close)
  - Momentum (5-bar, 10-bar)

- [ ] Compute regime-detecting features
  - Overnight gap (prev_close → open)
  - Pre-market indicator (if available)
  - Hour-of-day (cyclical encoding)
  - VIX level and change

**Success Criteria:**
- ✅ All features compute without errors
- ✅ No NaN/Inf values after dropna()
- ✅ Feature distributions look reasonable (no crazy outliers)

#### 15-Minute HMM Training
- [ ] Train Gaussian HMM on 15-min bars
  - States: 3 (bull/neutral/bear initially)
  - Covariance: 'diag' (independent features)
  - Iterations: 1000 max
  - Random inits: 5

- [ ] Validate HMM convergence
  - Check log-likelihood increases
  - Inspect transition matrix (reasonable persistence?)
  - Analyze regime statistics (mean return, volatility per state)

- [ ] Label regimes (bull/bear/neutral)
  - State with highest mean return = bull
  - State with lowest mean return = bear
  - State in between = neutral

**Success Criteria:**
- ✅ HMM converges (log-likelihood stable)
- ✅ Bull regime has positive mean returns
- ✅ Bear regime has negative mean returns
- ✅ Persistence (diagonal of transition matrix) > 0.85

**Files Created:**
- `/commodity-forecasting-system/models/phase0_15min_hmm.pkl`

#### Regime Shift Analysis
- [ ] Identify regime shifts (9:30 AM → 12:00 PM)
  - For each day, compare regime at 9:30 AM vs 12:00 PM
  - Count: How many days had a shift?
  - Direction: Bull→Bear, Bear→Bull, Neutral→Bull, etc.

- [ ] Measure shift magnitude
  - Average SPY move during shift days
  - Average SPY move during non-shift days
  - Statistical significance (t-test)

- [ ] Analyze predictability
  - Features at 9:30 AM that predict shift by 12:00 PM
  - Train logistic regression: shift_occurred ~ overnight_gap + vix_change + volume
  - Compute AUC (Area Under ROC Curve)

**Success Criteria:**
- ✅ Regime shifts occur >30 days/year (>12% of trading days)
- ✅ Average move during shift >0.5% (enough for options)
- ✅ Shift prediction AUC >0.65 (better than random)

**Files Created:**
- `/commodity-forecasting-system/outputs/phase0_shift_analysis.csv`
- `/commodity-forecasting-system/outputs/phase0_shift_report.md`

#### Profitability Estimation
- [ ] Simulate simple option strategy
  - Assume: Buy ATM call spread at 9:30 AM if shift predicted
  - Width: 10 points ($1000 max profit)
  - Cost: Assume $300 debit (30% of width)
  - Exit: 12:00 PM or profit target +40%

- [ ] Calculate hypothetical P&L
  - For each predicted shift day:
    - Entry: 9:30 AM price
    - Exit: 12:00 PM price
    - Profit: min(move × 100, max_profit) - debit

- [ ] Compute performance metrics
  - Win rate
  - Average win / Average loss
  - Profit factor
  - Expectancy ($/trade)

**Success Criteria:**
- ✅ Win rate >50% (better than coin flip)
- ✅ Average win > Average loss
- ✅ Expectancy >$20/trade (after rough transaction costs)

**Files Created:**
- `/commodity-forecasting-system/outputs/phase0_profitability.csv`

#### Decision Report
- [ ] Compile Phase 0 findings
  - Summary statistics
  - Key charts (regime transitions, P&L distribution)
  - Recommendation: GO or NO-GO

- [ ] Present to stakeholder (user)
  - Clear go/no-go decision
  - If GO: Proceed to Phase 1
  - If NO-GO: Document reasons, suggest alternatives

**Success Criteria:**
- ✅ Report is clear and actionable
- ✅ All 3 validation criteria met (shifts occur, predictable, profitable)

**Files Created:**
- `/commodity-forecasting-system/docs/PHASE0_VALIDATION_REPORT.md`

---

### **DECISION GATE 1: GO / NO-GO**

**GO Criteria (ALL must pass):**
- ✅ Regime shifts occur >30 days/year
- ✅ Shift prediction AUC >0.65
- ✅ Average move during shift >0.5%
- ✅ Simulated win rate >50%
- ✅ Expectancy >$20/trade

**NO-GO Triggers (ANY fails project):**
- ❌ Shifts rare (<20 days/year)
- ❌ Prediction random (AUC <0.60)
- ❌ Moves too small (<0.3%)
- ❌ Win rate <45%

**Action if GO:** Proceed to Phase 1 (Data Foundation)
**Action if NO-GO:** Pivot to daily swing trading or abandon options focus

**Decision Date:** 2026-01-24
**Decision:** [ ] GO  [ ] NO-GO  [ ] NEEDS MORE ANALYSIS

---

## 🏗️ PHASE 1: DATA FOUNDATION (Weeks 2-3)
**Timeline:** 2026-01-24 → 2026-02-07
**Effort:** 20 hours
**Cost:** $0-200/month (optional paid data)

### **Objective**
Build robust intraday data pipeline for 15-min bars, options chains, and market features.

### **Tasks**

#### Module: Intraday Data Acquisition
- [ ] Create `src/data/intraday_acquisition.py`
  - Class: `IntradayDataAcquisition`
  - Methods:
    - `fetch_intraday_bars(symbol, interval='15m', period='60d')`
    - `fetch_premarket_futures(symbol='ES=F', lookback_hours=12)`
    - `fetch_options_chain(symbol='SPX', expiry='0DTE')`
    - `fetch_vix_term_structure()`
    - `fetch_economic_calendar(date, impact_filter='high')`

- [ ] Implement data quality checks
  - Gap detection (missing bars)
  - Outlier detection (price spikes)
  - Volume validation (reasonable ranges)
  - Timestamp validation (market hours only)

- [ ] Add caching layer
  - Store downloaded data locally
  - Avoid redundant API calls
  - Implement refresh logic (daily updates)

**Success Criteria:**
- ✅ Can fetch 15-min bars for SPY, QQQ, ES futures
- ✅ Options chain returns strikes, bid/ask, IV, Greeks
- ✅ VIX term structure (VIX, VIX1D, VIX9D)
- ✅ Data quality >98% (missing <2%)

**Files Created:**
- `/commodity-forecasting-system/src/data/intraday_acquisition.py`
- `/commodity-forecasting-system/tests/test_data/test_intraday_acquisition.py`

#### Module: Intraday Feature Engineering
- [ ] Create `src/data/intraday_features.py`
  - Class: `IntradayFeatureEngineer`
  - Features:
    - **Momentum**: 15min return, 1hr return, overnight gap
    - **Volatility**: Parkinson (high-low), Garman-Klass (OHLC), rolling std
    - **Volume**: Relative volume (vs 20-day avg at same time), VWAP deviation
    - **Microstructure**: Bid-ask proxy (range), tick direction changes
    - **Temporal**: Hour-of-day (cyclical), day-of-week, days-to-event
    - **Options**: VIX change, term structure slope, put/call ratio
    - **Regime**: HMM state probabilities (from daily/hourly models)

- [ ] Validate feature computation
  - Check for NaN/Inf propagation
  - Verify feature distributions
  - Test on edge cases (market open, close, gaps)

**Success Criteria:**
- ✅ >20 features computed per 15-min bar
- ✅ Features compute in <1 second per day
- ✅ No data leakage (only use past information)

**Files Created:**
- `/commodity-forecasting-system/src/data/intraday_features.py`
- `/commodity-forecasting-system/tests/test_data/test_intraday_features.py`

#### Configuration Extension
- [ ] Add `[intraday]` section to `config/parameters.toml`
  ```toml
  [intraday]
  enabled = true
  data_interval = "15m"
  lookback_days = 252
  market_open = "09:30"
  market_close = "16:00"
  signal_generation_time = "09:15"
  exit_time = "12:00"

  [intraday.data_sources]
  bars = "yfinance"  # or "polygon", "alpaca"
  options = "yfinance"  # or "tradier", "ibkr"
  vix = "cboe"
  ```

- [ ] Add `[options_strategy]` section
  ```toml
  [options_strategy]
  min_shift_probability = 0.65
  spread_types = ["bull_call", "bear_put", "iron_condor"]
  strike_spacing = 5  # SPX points
  max_spread_width = 50  # SPX points
  min_dte = 0  # 0DTE allowed
  max_dte = 1  # Up to 1DTE
  ```

**Success Criteria:**
- ✅ Config validates with Pydantic
- ✅ All parameters documented

**Files Modified:**
- `/commodity-forecasting-system/config/parameters.toml`

#### Data Storage & Persistence
- [ ] Create data directory structure
  ```
  /data/intraday/
    /bars/
      spy_15min_2025.csv
      spy_15min_2026.csv
    /options/
      spy_chain_2026-01-17_0930.csv
    /vix/
      vix_term_structure_2025-2026.csv
  ```

- [ ] Implement data versioning
  - Track data source, fetch time, version
  - Enable reproducibility

**Success Criteria:**
- ✅ Data persisted in standardized format
- ✅ Can reload data without re-fetching

**Files Created:**
- `/commodity-forecasting-system/data/intraday/README.md` (data dictionary)

---

## 🧠 PHASE 2: HIERARCHICAL HMM DEVELOPMENT (Weeks 4-5)
**Timeline:** 2026-02-07 → 2026-02-21
**Effort:** 30 hours
**Cost:** $0

### **Objective**
Build 3-tier HMM architecture (daily/hourly/15-min) for regime detection across timescales.

### **Tasks**

#### Module: Hierarchical HMM
- [ ] Create `src/models/hierarchical_hmm.py`
  - Class: `HierarchicalHMM`
  - Architecture:
    - **Daily HMM**: 3 states (bull/bear/neutral)
    - **Hourly HMM**: 4 states (strong_bull/bull/bear/strong_bear)
    - **15-Minute HMM**: 5 states (panic_sell/weak_sell/neutral/weak_buy/panic_buy)

  - Methods:
    - `fit_hierarchical(daily_data, hourly_data, intraday_data)`
    - `predict_regime_cascade(current_features)`
    - `predict_regime_shift_probability(premarket_features, target_time='12:00')`

- [ ] Implement conditional transitions
  - 15-min transitions depend on hourly regime
  - Hourly transitions depend on daily regime
  - `P(S_t^(15min) | S_{t-1}^(15min), S_t^(hourly))`

**Success Criteria:**
- ✅ All 3 HMMs converge independently
- ✅ Hierarchical conditioning improves predictive power (AUC gain >0.05)
- ✅ Regime labels make intuitive sense

**Files Created:**
- `/commodity-forecasting-system/src/models/hierarchical_hmm.py`
- `/commodity-forecasting-system/tests/test_models/test_hierarchical_hmm.py`

#### Training Pipeline
- [ ] Train daily HMM
  - Data: 5 years daily SPY OHLCV (existing pipeline)
  - Features: returns, volatility_5, volatility_10, momentum_5, range
  - States: 3
  - Random inits: 10

- [ ] Train hourly HMM
  - Data: 1 year hourly SPY bars
  - Features: returns, volatility_3hr, volume_ratio, hour_of_day
  - States: 4
  - Condition on daily regime

- [ ] Train 15-minute HMM
  - Data: 60 days 15-min SPY bars
  - Features: returns, volatility_5bar, volume_ratio, range
  - States: 5
  - Condition on hourly regime

**Success Criteria:**
- ✅ Daily HMM matches existing performance (baseline)
- ✅ Hourly HMM shows regime persistence >0.80
- ✅ 15-min HMM detects microstructure regimes

**Files Created:**
- `/commodity-forecasting-system/models/trained/daily_hmm_v1.pkl`
- `/commodity-forecasting-system/models/trained/hourly_hmm_v1.pkl`
- `/commodity-forecasting-system/models/trained/15min_hmm_v1.pkl`

#### Regime Shift Prediction
- [ ] Implement `predict_regime_shift_probability()`
  - Inputs: Pre-market snapshot (overnight gap, VIX, volume, daily regime)
  - Output: P(shift by noon), expected_regime_noon, confidence

- [ ] Build pre-market feature extractor
  - Overnight ES futures move
  - Pre-market SPY volume (if available)
  - VIX change overnight
  - Economic calendar events (high-impact only)

- [ ] Validate on historical data
  - For each trading day, predict at 9:15 AM
  - Actual regime at 12:00 PM
  - Compute AUC, precision, recall

**Success Criteria:**
- ✅ AUC >0.65 (better than random)
- ✅ Precision >0.60 (when predicting shift, it happens 60% of time)
- ✅ Predictions stable (not over-reactive to noise)

**Files Created:**
- `/commodity-forecasting-system/notebooks/regime_shift_validation.ipynb`

#### Regime Visualization
- [ ] Create regime analysis plots
  - Regime timeline (color-coded by state)
  - Transition heatmap (from → to probabilities)
  - Regime statistics table (mean return, vol, persistence per state)

- [ ] Integrate with existing `regime_analysis.py` (562 LOC)
  - Extend for hierarchical regimes
  - Cross-timescale analysis

**Success Criteria:**
- ✅ Visualizations clearly show regime structure
- ✅ Transition patterns make economic sense

**Files Modified:**
- `/commodity-forecasting-system/src/models/regime_analysis.py`

---

## 🔮 PHASE 3: TIMESFM INTRADAY ADAPTATION (Week 6)
**Timeline:** 2026-02-21 → 2026-02-28
**Effort:** 15 hours
**Cost:** $0

### **Objective**
Validate TimesFM zero-shot forecasting on 15-min bars and optimize for intraday use.

### **Tasks**

#### Zero-Shot Performance Testing
- [ ] Test TimesFM on 15-min SPY bars
  - Context: 512 bars (≈8 trading days)
  - Horizon: 12 bars (9:30 AM → 12:00 PM, 3 hours)
  - Frequency hint: `freq=0` (treat as generic timeseries)

- [ ] Evaluate forecast accuracy
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - Directional accuracy (sign of change)
  - Compare to baselines: random walk, ARIMA, exponential smoothing

**Success Criteria:**
- ✅ TimesFM MAE < random walk MAE
- ✅ Directional accuracy >55%
- ✅ Forecasts not collapsed to mean (check variance)

**Files Created:**
- `/commodity-forecasting-system/notebooks/timesfm_intraday_evaluation.ipynb`

#### Volatility Forecasting
- [ ] Implement volatility forecast from price path
  - Use TimesFM forecasted high-low range
  - Compute Parkinson volatility estimator
  - Annualize for comparison to VIX

- [ ] Validate volatility forecasts
  - Compare to realized volatility (9:30-12:00)
  - Correlation with VIX changes
  - Directional accuracy (vol up vs vol down)

**Success Criteria:**
- ✅ Volatility forecast correlation >0.50 with realized
- ✅ Directional accuracy >60%

**Files Created:**
- `/commodity-forecasting-system/src/models/timesfm_volatility_forecast.py`

#### Ensemble Integration (Preliminary)
- [ ] Extend `TimesFMHMMEnsemble` for intraday
  - Add mode: "intraday_signal"
  - Combine hierarchical HMM + TimesFM forecast
  - Weight by confidence

- [ ] Test ensemble vs individual models
  - Regime shift detection: HMM only
  - Price forecast: TimesFM only
  - Combined: Ensemble
  - Compare AUC/accuracy

**Success Criteria:**
- ✅ Ensemble improves on individual models (AUC gain >0.03)

**Files Modified:**
- `/commodity-forecasting-system/src/models/ensemble.py`

---

## 📈 PHASE 4: OPTIONS STRATEGY IMPLEMENTATION (Weeks 7-8)
**Timeline:** 2026-02-28 → 2026-03-14
**Effort:** 25 hours
**Cost:** $0

### **Objective**
Build options strategy logic: strike selection, spread construction, Greeks tracking, risk management.

### **Tasks**

#### Module: Strike Selection
- [ ] Create `src/trading/strike_selection.py`
  - Class: `StrikeSelector`
  - Method: `select_spread_strikes(spot, forecast_noon, forecast_std, spread_type)`
  - Strategy:
    - Long strike: ATM or slightly OTM
    - Short strike: At forecasted noon price
    - Width: 10-50 points (balance probability vs payoff)

- [ ] Implement spread types
  - Bull Call Spread (expect upward move)
  - Bear Put Spread (expect downward move)
  - Iron Condor (expect volatility collapse)
  - Calendar Spread (theta decay play)

**Success Criteria:**
- ✅ Strikes reasonable (within ±5% of spot)
- ✅ Spreads have positive edge (expected value >0)

**Files Created:**
- `/commodity-forecasting-system/src/trading/strike_selection.py`
- `/commodity-forecasting-system/tests/test_trading/test_strike_selection.py`

#### Module: Greeks Tracker
- [ ] Create `src/trading/greeks_tracker.py`
  - Class: `GreeksEvolutionTracker`
  - Track: Delta, Gamma, Theta, Vega, Rho
  - Update: Real-time as spot/IV/time change

- [ ] P&L Attribution
  - Delta P&L: Δ × (S_t - S_0)
  - Theta P&L: Θ × (t - t_0)
  - Vega P&L: ν × (IV_t - IV_0)
  - Total P&L: Sum of components

**Success Criteria:**
- ✅ Greeks match Black-Scholes calculations
- ✅ P&L attribution sums to total P&L

**Files Created:**
- `/commodity-forecasting-system/src/trading/greeks_tracker.py`

#### Module: Risk Manager
- [ ] Create `src/trading/risk_manager.py`
  - Class: `RiskManager`
  - Position Limits:
    - Max capital at risk: 2% per trade
    - Max spread width: 50 points
    - Max contracts: Based on capital

  - Greeks Limits:
    - Max portfolio delta: ±0.50 (delta-neutral preferred)
    - Max portfolio gamma: ±0.10
    - Max portfolio theta: -$100/day

  - Stop Losses:
    - Per-trade: -60% of capital at risk
    - Per-day: -5% of capital
    - Per-week: -10% of capital

- [ ] Implement Kelly Criterion position sizing
  - `f* = (p × b - q) / b`
  - Use 1/4 Kelly for conservative sizing

- [ ] Regime-aware risk adjustment
  - High confidence regime → increase size 20%
  - Low confidence regime → decrease size 50%
  - Panic regime → halt trading

**Success Criteria:**
- ✅ Risk limits enforced programmatically
- ✅ Position size respects Kelly criterion
- ✅ Never exceeds max loss limits

**Files Created:**
- `/commodity-forecasting-system/src/trading/risk_manager.py`
- `/commodity-forecasting-system/tests/test_trading/test_risk_manager.py`

#### Entry/Exit Rules Engine
- [ ] Create `src/trading/rules_engine.py`
  - Class: `TradingRulesEngine`

  - Entry Rules:
    - Signal confidence >0.70
    - Bid-ask spread <5% of mid
    - Options volume >100 contracts
    - No major event in next 3 hours
    - VIX <40

  - Exit Rules:
    - Profit target: +40% of max profit
    - Stop loss: -60% of capital at risk
    - Time stop: 12:00 PM (always exit)
    - Regime reversal: HMM flips back
    - Volatility spike: VIX up >15%

**Success Criteria:**
- ✅ Rules trigger correctly on test data
- ✅ No false positives/negatives

**Files Created:**
- `/commodity-forecasting-system/src/trading/rules_engine.py`

---

## 🧪 PHASE 5: BACKTESTING ENGINE (Weeks 9-10)
**Timeline:** 2026-03-14 → 2026-03-28
**Effort:** 30 hours
**Cost:** $0

### **Objective**
Build walk-forward backtesting framework to validate strategy on historical data.

### **Tasks**

#### Module: Intraday Backtest
- [ ] Create `src/backtesting/intraday_backtest.py`
  - Class: `IntradayOptionsBacktest`
  - Walk-Forward Logic:
    1. Train HMMs on rolling 1-year window
    2. Test on next 60 days
    3. Slide window forward 60 days
    4. Repeat

  - Daily Workflow:
    1. 9:15 AM: Generate signal from pre-market
    2. 9:30 AM: Execute trade (if signal)
    3. 9:45-12:00: Monitor every 15 min, check exits
    4. 12:00 PM: Force close
    5. Record results, update equity

- [ ] Implement fill simulation
  - Bid-ask spread: Estimate from historical data or assume 2-5%
  - Slippage: 1-2 ticks on limit orders
  - Market orders: Mid + half spread
  - Limit orders: Fill if price crosses limit

**Success Criteria:**
- ✅ Backtest runs on 1+ year data without errors
- ✅ Realistic fills (not cheating with future data)
- ✅ Transaction costs included

**Files Created:**
- `/commodity-forecasting-system/src/backtesting/intraday_backtest.py`

#### Performance Metrics
- [ ] Create `src/backtesting/performance_metrics.py`
  - Metrics:
    - Total trades, Win rate, Avg win, Avg loss
    - Profit factor (gross profit / gross loss)
    - Sharpe ratio, Sortino ratio
    - Max drawdown, Max drawdown duration
    - Calmar ratio (annual return / max drawdown)
    - Expectancy (E[P&L] per trade)
    - Avg holding time

  - Regime-Specific Performance:
    - Analyze by shift type (bull→bear, bear→bull, etc.)
    - Identify which regimes are most profitable

**Success Criteria:**
- ✅ All metrics computed correctly
- ✅ Regime breakdown provides insights

**Files Created:**
- `/commodity-forecasting-system/src/backtesting/performance_metrics.py`

#### Backtesting Execution
- [ ] Run full backtest on 1 year data
  - Period: 2025-01-01 to 2026-01-01
  - Initial capital: $10,000
  - Position size: Kelly-based
  - Transaction costs: $2/contract + slippage

- [ ] Generate backtest report
  - Equity curve
  - Drawdown chart
  - Trade distribution (wins/losses)
  - Monthly returns
  - Performance metrics table

**Success Criteria (DECISION GATE 2):**
- ✅ Sharpe ratio >1.5
- ✅ Win rate >55%
- ✅ Max drawdown <25%
- ✅ Profitable in multiple regime types (not overfitted)

**Files Created:**
- `/commodity-forecasting-system/outputs/backtest_report_v1.html`
- `/commodity-forecasting-system/outputs/backtest_trades_v1.csv`

#### Sensitivity Analysis
- [ ] Parameter sweeps
  - HMM states: 2, 3, 4, 5
  - Signal threshold: 0.60, 0.65, 0.70, 0.75
  - Spread width: 10, 20, 30, 40, 50 points
  - Exit time: 11:00, 11:30, 12:00, 12:30

- [ ] Analyze robustness
  - Which parameters most sensitive?
  - Is performance stable across parameter ranges?
  - Identify overfitting risks

**Success Criteria:**
- ✅ Performance stable across reasonable parameter ranges
- ✅ No single parameter drives all performance

**Files Created:**
- `/commodity-forecasting-system/notebooks/sensitivity_analysis.ipynb`

---

### **DECISION GATE 2: GO / NO-GO**

**GO Criteria (ALL must pass):**
- ✅ Backtest Sharpe ratio >1.5
- ✅ Win rate >55%
- ✅ Max drawdown <25%
- ✅ Profitable across multiple regime types
- ✅ Performance stable across parameter variations

**NO-GO Triggers (ANY fails):**
- ❌ Sharpe <1.0
- ❌ Win rate <50%
- ❌ Max drawdown >35%
- ❌ Only works in bull markets
- ❌ Performance collapses with small parameter changes

**Action if GO:** Proceed to Phase 6 (Paper Trading)
**Action if NO-GO:** Iterate on signal generation, adjust parameters, or pivot strategy

**Decision Date:** 2026-03-28
**Decision:** [ ] GO  [ ] NO-GO  [ ] ITERATE

---

## 📡 PHASE 6: PAPER TRADING PREPARATION (Weeks 11-12)
**Timeline:** 2026-03-28 → 2026-04-11
**Effort:** 20 hours
**Cost:** $0 (paper trading free)

### **Objective**
Build real-time signal generation pipeline and paper trading infrastructure.

### **Tasks**

#### Module: Real-Time Signal Generator
- [ ] Create `src/live/signal_generator.py`
  - Class: `LiveSignalGenerator`
  - Workflow:
    1. 9:00 AM: Fetch overnight data (ES futures, VIX)
    2. 9:10 AM: Compute pre-market features
    3. 9:15 AM: Generate trade signal
    4. Output: Signal object with action, strikes, confidence, etc.

- [ ] Add data staleness checks
  - Verify data is fresh (<5 min old)
  - Alert if data feed down
  - Fallback to last known good data (with warning)

**Success Criteria:**
- ✅ Signal generated daily by 9:15 AM
- ✅ Latency <10 seconds (pre-market → signal)
- ✅ No crashes on missing/stale data

**Files Created:**
- `/commodity-forecasting-system/src/live/signal_generator.py`

#### Module: Paper Trader
- [ ] Create `src/live/paper_trader.py`
  - Class: `PaperTradingEngine`
  - Mock Execution:
    - Simulate order placement (no real broker)
    - Track fills based on historical bid-ask
    - Update position and P&L

  - Position Tracking:
    - Open positions, Greeks, P&L
    - Entry time, exit time, holding period
    - Stop loss and profit target monitoring

**Success Criteria:**
- ✅ Tracks positions accurately
- ✅ Exits trigger correctly
- ✅ P&L matches manual calculations

**Files Created:**
- `/commodity-forecasting-system/src/live/paper_trader.py`

#### Dashboard: Signal Monitor
- [ ] Create Streamlit dashboard `src/ui/intraday_dashboard.py`
  - Tabs:
    - **Signal**: Current signal, confidence, strikes
    - **Positions**: Open trades, P&L, Greeks
    - **Performance**: Equity curve, metrics
    - **Regimes**: Current regime, transition probs

  - Refresh: Every 15 min during trading hours

**Success Criteria:**
- ✅ Dashboard displays all relevant info
- ✅ Updates in real-time
- ✅ User-friendly (clear visuals)

**Files Created:**
- `/commodity-forecasting-system/src/ui/intraday_dashboard.py`

#### Alerts & Monitoring
- [ ] Create `src/live/alerts.py`
  - Triggers:
    - Signal generated (9:15 AM)
    - Trade entered (9:30 AM)
    - Profit target hit
    - Stop loss hit
    - Data feed failure
    - Risk limit breached

  - Delivery: Console log, email (optional), SMS (optional)

**Success Criteria:**
- ✅ Alerts trigger correctly
- ✅ No false alerts

**Files Created:**
- `/commodity-forecasting-system/src/live/alerts.py`

#### Trade Journal
- [ ] Create `src/live/trade_journal.py`
  - Record every trade:
    - Date, time, signal, strikes, direction
    - Entry price, exit price, P&L
    - Regime at entry, regime at exit
    - Stop/target hit, holding time
    - Notes (manual annotation)

  - Export: CSV for analysis

**Success Criteria:**
- ✅ All trades logged
- ✅ Can review historical performance

**Files Created:**
- `/commodity-forecasting-system/outputs/trade_journal.csv`

---

## 🎮 PHASE 7: PAPER TRADING EXECUTION (Weeks 13-16)
**Timeline:** 2026-04-11 → 2026-05-09
**Effort:** 10 hours/week (40 hours total)
**Cost:** $0

### **Objective**
Run paper trading for minimum 20 trading days, validate backtest expectations.

### **Tasks**

#### Week 13-14: Initial Paper Trading
- [ ] Day 1-10: Monitor daily
  - Check signal at 9:15 AM
  - Verify entry at 9:30 AM (if signal present)
  - Monitor every 15 min until 12:00 PM
  - Record results in trade journal

- [ ] Track performance vs backtest
  - Compare win rate (paper vs backtest)
  - Compare avg P&L (paper vs backtest)
  - Identify discrepancies (slippage, timing, etc.)

**Success Criteria (Checkpoint):**
- ✅ >5 trades executed
- ✅ Win rate >45% (within ±15% of backtest)
- ✅ No technical failures

**Files Updated:**
- `/commodity-forecasting-system/outputs/trade_journal.csv`

#### Week 15-16: Extended Paper Trading
- [ ] Day 11-20: Continue monitoring
  - Same process as Week 13-14
  - Refine based on learnings

- [ ] Weekly performance review
  - Equity curve vs backtest
  - Sharpe ratio calculation
  - Max drawdown tracking
  - Identify patterns (best/worst days)

**Success Criteria (Final):**
- ✅ >20 trades executed
- ✅ Win rate >50%
- ✅ Paper trading Sharpe >1.5
- ✅ Results within ±30% of backtest

**Files Updated:**
- `/commodity-forecasting-system/outputs/trade_journal.csv`
- `/commodity-forecasting-system/docs/PAPER_TRADING_REPORT.md`

#### Issue Tracking & Resolution
- [ ] Document all issues
  - Data gaps, late signals, missed exits
  - Unexpected behavior, bugs
  - Performance deviations

- [ ] Fix critical issues
  - Priority: Data reliability, signal timing, exit logic
  - Test fixes in paper trading

- [ ] Update documentation
  - Known issues, workarounds, limitations

**Files Created:**
- `/commodity-forecasting-system/docs/PAPER_TRADING_ISSUES.md`

---

### **DECISION GATE 3: GO / NO-GO (Live Trading)**

**GO Criteria (ALL must pass):**
- ✅ Paper trading Sharpe >1.5
- ✅ Win rate >55%
- ✅ Results within ±30% of backtest
- ✅ No major technical issues
- ✅ Emotionally comfortable with losses

**NO-GO Triggers (ANY fails):**
- ❌ Paper results <50% of backtest (something broken)
- ❌ Win rate <45%
- ❌ Frequent data gaps or missed signals
- ❌ Not ready psychologically

**Action if GO:** Start live trading with $2,000-$10,000
**Action if NO-GO:** Extended paper trading (4+ weeks) or abandon

**Decision Date:** 2026-05-09
**Decision:** [ ] GO  [ ] NO-GO  [ ] EXTEND PAPER TRADING

---

## 🚀 PHASE 8: LIVE TRADING (OPTIONAL - Conditional on Gate 3)
**Timeline:** Week 17+
**Capital:** $2,000 minimum, $10,000 recommended

### **Prerequisites**
- ✅ Paper trading successful (>20 trades, Sharpe >1.5)
- ✅ All technical issues resolved
- ✅ Broker account setup (Interactive Brokers, Tastyworks, etc.)
- ✅ Risk limits configured in code
- ✅ Emotionally prepared for real losses

### **Tasks**

#### Live Trading Setup
- [ ] Open broker account
  - Recommended: Interactive Brokers (low commissions, good API)
  - Account type: Margin (required for spreads)
  - Minimum balance: $2,000 (start small)

- [ ] Connect to broker API
  - Authenticate (API keys, OAuth)
  - Test order placement (1 paper trade via API)
  - Verify fills, Greeks, P&L reporting

**Success Criteria:**
- ✅ Broker account approved
- ✅ API connected and tested
- ✅ Can place/cancel orders programmatically

#### Start Small
- [ ] Week 1-2: 1 contract per trade
  - Max risk: $200/trade
  - Track performance daily
  - Compare to paper trading

- [ ] Week 3-4: Scale to 2-3 contracts (if successful)
  - Only if Week 1-2 profitable
  - Increase position size gradually

**Success Criteria:**
- ✅ Live results match paper trading (±20%)
- ✅ No execution errors
- ✅ Risk limits respected

#### Ongoing Monitoring
- [ ] Daily checklist:
  - 9:00 AM: Review overnight markets
  - 9:15 AM: Check signal
  - 9:30 AM: Execute trade (if signal)
  - 9:45-12:00: Monitor every 15 min
  - 12:00 PM: Force exit
  - End of day: Review P&L, update journal

- [ ] Weekly review:
  - Performance metrics (Sharpe, win rate, drawdown)
  - Compare to backtest and paper trading
  - Adjust if underperforming (reduce size or pause)

**Success Criteria:**
- ✅ Positive P&L over 4 weeks
- ✅ Sharpe >1.5
- ✅ Max drawdown <15%

---

## 📊 KEY PERFORMANCE INDICATORS (KPIs)

### **Target Metrics** (Track Weekly)

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| **Win Rate** | >60% | ___ | 🟢/🟡/🔴 |
| **Sharpe Ratio** | >2.0 | ___ | 🟢/🟡/🔴 |
| **Avg Win** | >$80 | ___ | 🟢/🟡/🔴 |
| **Avg Loss** | <$70 | ___ | 🟢/🟡/🔴 |
| **Max Drawdown** | <15% | ___ | 🟢/🟡/🔴 |
| **Profit Factor** | >2.0 | ___ | 🟢/🟡/🔴 |
| **Trades/Month** | 6-10 | ___ | 🟢/🟡/🔴 |
| **Avg Hold Time** | <2.5 hrs | ___ | 🟢/🟡/🔴 |

### **Regime-Specific Performance**

| Regime Transition | Trades | Win Rate | Avg P&L | Status |
|-------------------|--------|----------|---------|--------|
| Bull → Bear | ___ | ___ | ___ | 🟢/🟡/🔴 |
| Bear → Bull | ___ | ___ | ___ | 🟢/🟡/🔴 |
| Neutral → Bull | ___ | ___ | ___ | 🟢/🟡/🔴 |
| Neutral → Bear | ___ | ___ | ___ | 🟢/🟡/🔴 |
| High Vol → Neutral | ___ | ___ | ___ | 🟢/🟡/🔴 |

---

## 🗂️ PROJECT FILES & DELIVERABLES

### **Code Modules** (Est. 3,000-4,000 LOC new code)

**Data Layer:**
- [ ] `src/data/intraday_acquisition.py` (500 LOC)
- [ ] `src/data/intraday_features.py` (400 LOC)

**Models:**
- [ ] `src/models/hierarchical_hmm.py` (600 LOC)
- [ ] `src/models/timesfm_volatility_forecast.py` (200 LOC)

**Trading:**
- [ ] `src/trading/strike_selection.py` (300 LOC)
- [ ] `src/trading/greeks_tracker.py` (250 LOC)
- [ ] `src/trading/risk_manager.py` (400 LOC)
- [ ] `src/trading/rules_engine.py` (300 LOC)

**Backtesting:**
- [ ] `src/backtesting/intraday_backtest.py` (500 LOC)
- [ ] `src/backtesting/performance_metrics.py` (300 LOC)

**Live Trading:**
- [ ] `src/live/signal_generator.py` (300 LOC)
- [ ] `src/live/paper_trader.py` (400 LOC)
- [ ] `src/live/alerts.py` (200 LOC)
- [ ] `src/live/trade_journal.py` (150 LOC)

**UI:**
- [ ] `src/ui/intraday_dashboard.py` (400 LOC)

**Tests:**
- [ ] `tests/test_intraday/` (800 LOC across all test files)

### **Documentation**

- [ ] `docs/PHASE0_VALIDATION_REPORT.md`
- [ ] `docs/BACKTEST_REPORT.md`
- [ ] `docs/PAPER_TRADING_REPORT.md`
- [ ] `docs/LIVE_TRADING_GUIDE.md`
- [ ] `docs/TROUBLESHOOTING.md`

### **Notebooks**

- [ ] `notebooks/phase0_validation.ipynb`
- [ ] `notebooks/regime_shift_validation.ipynb`
- [ ] `notebooks/timesfm_intraday_evaluation.ipynb`
- [ ] `notebooks/sensitivity_analysis.ipynb`

### **Data**

- [ ] `data/intraday/spy_15min_2025-2026.csv`
- [ ] `data/intraday/vix_term_structure.csv`
- [ ] `data/intraday/options_chains/` (directory)

### **Models**

- [ ] `models/trained/daily_hmm_v1.pkl`
- [ ] `models/trained/hourly_hmm_v1.pkl`
- [ ] `models/trained/15min_hmm_v1.pkl`

### **Outputs**

- [ ] `outputs/trade_journal.csv`
- [ ] `outputs/backtest_report_v1.html`
- [ ] `outputs/paper_trading_equity_curve.png`

---

## ⚠️ RISK CONTROLS & CIRCUIT BREAKERS

### **Automated Risk Limits** (Enforced in Code)

**Per-Trade:**
- ✅ Max capital at risk: 2% ($200 on $10k)
- ✅ Max spread width: 50 points
- ✅ Stop loss: -60% of capital at risk
- ✅ Profit target: +40% of max profit

**Per-Day:**
- ✅ Max 1 trade per day
- ✅ No re-entry if stopped out
- ✅ Daily loss limit: -5% of capital

**Per-Week:**
- ✅ 3 consecutive losses → reduce size 50%
- ✅ Weekly loss limit: -10% → halt trading

**Market Conditions:**
- ✅ VIX >40 → no new trades
- ✅ Bid-ask spread >5% → skip trade
- ✅ Options volume <100 → skip trade

### **Manual Circuit Breakers** (Require Human Decision)

- ⚠️ Equity drawdown >15% → Review strategy, consider pause
- ⚠️ 5 consecutive losses → Pause, analyze what changed
- ⚠️ Paper trading results <70% of backtest → Do NOT go live
- ⚠️ Emotional discomfort → Reduce size or pause

---

## 📞 WEEKLY REVIEW TEMPLATE

**Week Ending:** ___________

**Performance Summary:**
- Total trades: ___
- Wins: ___ | Losses: ___
- Win rate: ___%
- P&L: $___
- Sharpe (rolling 4-week): ___
- Max drawdown: ___%

**Best Trade:**
- Date: ___
- Regime: ___
- P&L: $___
- Why it worked: ___

**Worst Trade:**
- Date: ___
- Regime: ___
- P&L: $___
- What went wrong: ___

**Issues Encountered:**
- Data: ___
- Execution: ___
- Strategy: ___

**Actions for Next Week:**
1. ___
2. ___
3. ___

---

## 🎓 LESSONS LEARNED & ITERATION LOG

### **Phase 0 Learnings:**
- Date: ___
- Key Insight: ___
- Action Taken: ___

### **Phase 1 Learnings:**
- Date: ___
- Key Insight: ___
- Action Taken: ___

(Continue for all phases)

---

## ✅ FINAL CHECKLIST - PROJECT COMPLETION

**Phase 0: Validation** [ ]
- [x] Data downloaded
- [ ] HMM trained
- [ ] Regime shifts analyzed
- [ ] Profitability estimated
- [ ] Decision: GO / NO-GO

**Phase 1: Data Foundation** [ ]
- [ ] Intraday acquisition module
- [ ] Feature engineering module
- [ ] Config extended
- [ ] Data storage setup

**Phase 2: Hierarchical HMM** [ ]
- [ ] Daily HMM trained
- [ ] Hourly HMM trained
- [ ] 15-min HMM trained
- [ ] Regime shift predictor built
- [ ] Validated on historical data

**Phase 3: TimesFM Adaptation** [ ]
- [ ] Zero-shot tested on 15-min
- [ ] Volatility forecasting implemented
- [ ] Ensemble integration

**Phase 4: Options Strategy** [ ]
- [ ] Strike selection module
- [ ] Greeks tracker
- [ ] Risk manager
- [ ] Entry/exit rules

**Phase 5: Backtesting** [ ]
- [ ] Backtest engine built
- [ ] 1-year backtest run
- [ ] Performance metrics computed
- [ ] Sensitivity analysis
- [ ] Decision: GO / NO-GO

**Phase 6: Paper Trading Prep** [ ]
- [ ] Signal generator (real-time)
- [ ] Paper trading engine
- [ ] Dashboard built
- [ ] Alerts configured

**Phase 7: Paper Trading** [ ]
- [ ] 20+ trades executed
- [ ] Performance validated
- [ ] Issues resolved
- [ ] Decision: GO / NO-GO for live

**Phase 8: Live Trading** [ ]
- [ ] Broker account setup
- [ ] API connected
- [ ] First live trades executed
- [ ] Performance tracking

---

## 📅 PROJECT TIMELINE SUMMARY

| Phase | Weeks | Start | End | Status |
|-------|-------|-------|-----|--------|
| **Phase 0: Validation** | 1 | 2026-01-17 | 2026-01-24 | 🟡 IN PROGRESS |
| **Phase 1: Data Foundation** | 2 | 2026-01-24 | 2026-02-07 | ⚪ PENDING |
| **Phase 2: Hierarchical HMM** | 2 | 2026-02-07 | 2026-02-21 | ⚪ PENDING |
| **Phase 3: TimesFM Adaptation** | 1 | 2026-02-21 | 2026-02-28 | ⚪ PENDING |
| **Phase 4: Options Strategy** | 2 | 2026-02-28 | 2026-03-14 | ⚪ PENDING |
| **Phase 5: Backtesting** | 2 | 2026-03-14 | 2026-03-28 | ⚪ PENDING |
| **Phase 6: Paper Trading Prep** | 2 | 2026-03-28 | 2026-04-11 | ⚪ PENDING |
| **Phase 7: Paper Trading** | 4 | 2026-04-11 | 2026-05-09 | ⚪ PENDING |
| **Phase 8: Live Trading** | Ongoing | 2026-05-09+ | — | ⚪ PENDING |

**Total Estimated Time:** 16 weeks (4 months)
**Total Estimated Effort:** ~175 hours

---

## 🏁 SUCCESS DEFINITION

**Project is SUCCESSFUL if:**
1. ✅ Phase 0 validation passes (regime shifts predictable)
2. ✅ Backtest achieves targets (Sharpe >1.5, win rate >55%)
3. ✅ Paper trading validates backtest (±30%)
4. ✅ Live trading is profitable over 4 weeks
5. ✅ Max drawdown stays <15%
6. ✅ Risk management prevents catastrophic loss

**Project is COMPLETE when:**
- All phases executed
- Live trading operational (if Gate 3 passed)
- Full documentation in place
- Code tested and robust

---

**Last Updated:** 2026-01-17
**Maintained By:** Claude Code Assistant
**Version:** 1.0

---

## 📝 NOTES & AMENDMENTS

_Use this section to track changes to the plan, lessons learned, and key decisions._

### Amendment Log:
- 2026-01-17: Initial checklist created based on comprehensive analysis
- ___: ___

---

**END OF CHECKLIST**
