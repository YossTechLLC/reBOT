# VOLATILITY PREDICTION UI - ARCHITECTURE & IMPLEMENTATION CHECKLIST

**Created:** 2026-01-17
**Purpose:** Comprehensive architectural plan for building an interactive UI for the HMM + TimesFM volatility prediction system

**Research Sources:**
- [Dashboard Best Practices - Made With ML](https://madewithml.com/courses/mlops/dashboard/)
- [ML Model Monitoring Dashboard Tutorial - Evidently AI](https://www.evidentlyai.com/blog/ml-model-monitoring-dashboard-tutorial)
- [Commodity Trading Visualization Tools - Dev3lop](https://dev3lop.com/commodity-trading-visualization-market-volatility-analysis-tools/)
- [Streamlit vs Gradio Comparison - UI Bakery](https://uibakery.io/blog/streamlit-vs-gradio)
- [ExplainerDashboard Documentation](https://github.com/oegedijk/explainerdashboard)
- [SHAP Documentation](https://shap.readthedocs.io/)

---

## 📋 EXECUTIVE SUMMARY

### Technology Stack Decision

**Primary Framework: Streamlit**
- **Why:** Fastest prototyping, zero frontend work, perfect for data science dashboards
- **Alternatives Considered:** Gradio (better for ML demos only), Dash (overkill for our needs)

**Visualization Libraries:**
- **Plotly** - Interactive candlestick charts, time series, 3D visualizations
- **SHAP** - Explainable AI feature importance
- **Matplotlib/Seaborn** - Statistical plots (confusion matrix, distribution plots)

**Architecture Pattern:**
```
User Interface (Streamlit)
    ↓
Parameter Controls → Model Pipeline → Visualization Engine
    ↓                     ↓                    ↓
HMM Params         Data Processing      Charts & Metrics
TimesFM Params     Model Training       Explainability
Training Params    Prediction           Trading Output
```

---

## 🎯 PHASE 1: ARCHITECTURE & DESIGN PLANNING

### 1.1 Requirements Analysis ✅

**Functional Requirements:**
- [ ] **FR-1:** Display historical volatility data with interactive charts
- [ ] **FR-2:** Allow users to adjust HMM training parameters (n_regimes, features, training days)
- [ ] **FR-3:** Allow users to adjust model parameters (confidence threshold, weights)
- [ ] **FR-4:** Allow users to adjust TimesFM parameters (context length, checkpoint)
- [ ] **FR-5:** Run forward/backward training passes with visual feedback
- [ ] **FR-6:** Display predictions with confidence scores
- [ ] **FR-7:** Visualize regime detection with color-coded time series
- [ ] **FR-8:** Show feature importance and SHAP values
- [ ] **FR-9:** Display recommended trading strategy (spread type, entry/exit)
- [ ] **FR-10:** Export predictions and charts to CSV/PDF

**Non-Functional Requirements:**
- [ ] **NFR-1:** UI must load in < 3 seconds
- [ ] **NFR-2:** Charts must be interactive (zoom, pan, hover)
- [ ] **NFR-3:** UI must be locally runnable without internet (except data download)
- [ ] **NFR-4:** All visualizations must be colorblind-friendly
- [ ] **NFR-5:** UI must handle errors gracefully with clear messages

### 1.2 UI Layout Design

**Proposed Layout Structure:**
```
┌─────────────────────────────────────────────────────────────────┐
│ HEADER: Volatility Prediction System - SPY 0DTE/1DTE           │
│ Status: ✅ HMM Loaded | ⚠️ TimesFM Unavailable | Data: 2026-01-16│
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────┬───────────────────────────────────────────┐
│ SIDEBAR (Controls)  │ MAIN PANEL (Visualizations)              │
│                     │                                           │
│ 📊 Data Settings    │ 📈 Tab 1: Prediction Dashboard            │
│   - Symbol          │   - Current Prediction Card              │
│   - Date Range      │   - Confidence Gauge (0-100)             │
│   - History Days    │   - Regime Indicator                     │
│                     │   - Candlestick Chart with Regime Overlay│
│ 🎯 HMM Parameters   │   - Feature Contribution Bar Chart       │
│   - N Regimes       │                                           │
│   - Features Select │ 📊 Tab 2: Model Explanation              │
│   - Training Days   │   - SHAP Waterfall Plot                  │
│   - Iterations      │   - Feature Importance Ranking           │
│                     │   - Regime Transition Heatmap            │
│ 🔮 TimesFM Params   │   - Probability Distribution             │
│   - Enable/Disable  │                                           │
│   - Context Length  │ 🔍 Tab 3: Validation Results             │
│   - Checkpoint      │   - Confusion Matrix                     │
│                     │   - Walk-Forward Performance             │
│ ⚙️ Model Config     │   - Win Rate Over Time                   │
│   - Threshold       │   - Accuracy Metrics Table               │
│   - Weight Mix      │                                           │
│                     │ 💰 Tab 4: Trading Strategy               │
│ 🚀 Actions          │   - Recommended Spread Type              │
│   - Train Model     │   - Entry Price Levels                   │
│   - Run Prediction  │   - Exit Rules Visualization             │
│   - Validate        │   - Expected P&L Chart                   │
│   - Export Results  │   - Risk Metrics                         │
└─────────────────────┴───────────────────────────────────────────┘
```

**Design Decisions:**
- [ ] **DD-1:** Use tabs to organize different views (prediction, explanation, validation, strategy)
- [ ] **DD-2:** Sidebar for all controls to keep main panel clean
- [ ] **DD-3:** Real-time updates when parameters change (no "Apply" button needed)
- [ ] **DD-4:** Color scheme: Green (high confidence), Yellow (medium), Red (low confidence)
- [ ] **DD-5:** Use tooltips extensively for parameter explanations

### 1.3 Data Flow Architecture

**Component Diagram:**
```
┌──────────────┐
│ Streamlit UI │
└──────┬───────┘
       │
       ├─→ DataManager (src/ui/data_manager.py)
       │     ├─→ AlpacaDataClient
       │     └─→ VolatilityFeatureEngineer
       │
       ├─→ ModelController (src/ui/model_controller.py)
       │     ├─→ VolatilityHMM
       │     ├─→ TimesFMVolatilityForecaster
       │     └─→ VolatilityConfidenceScorer
       │
       ├─→ VisualizationEngine (src/ui/visualization.py)
       │     ├─→ PlotlyCharts
       │     ├─→ SHAPExplainer
       │     └─→ MetricsDisplay
       │
       └─→ TradingStrategyEngine (src/ui/strategy.py)
             ├─→ SpreadRecommender
             └─→ RiskCalculator
```

**File Structure:**
```
commodity-forecasting-system/
├── src/
│   └── ui/
│       ├── __init__.py
│       ├── app.py                    # Main Streamlit app
│       ├── data_manager.py           # Data loading & caching
│       ├── model_controller.py       # Model training & prediction
│       ├── visualization.py          # Chart components
│       ├── explainability.py         # SHAP & feature importance
│       ├── strategy.py               # Trading strategy logic
│       └── utils.py                  # Helper functions
├── app.py                            # Entry point (symlink to src/ui/app.py)
└── requirements-ui.txt               # UI-specific dependencies
```

---

## 🛠️ PHASE 2: ENVIRONMENT & DEPENDENCIES SETUP

### 2.1 Install Core UI Framework

**Dependencies to Add:**
```txt
# requirements-ui.txt
streamlit>=1.40.0
plotly>=5.18.0
shap>=0.44.0
matplotlib>=3.8.0
seaborn>=0.13.0
kaleido>=0.2.1              # For static image export
streamlit-aggrid>=0.3.4     # For interactive tables
streamlit-option-menu>=0.3.6 # For better navigation
```

**Tasks:**
- [ ] **T-2.1.1:** Create `requirements-ui.txt` with all UI dependencies
- [ ] **T-2.1.2:** Install dependencies: `pip install -r requirements-ui.txt`
- [ ] **T-2.1.3:** Verify installation: `streamlit hello`
- [ ] **T-2.1.4:** Test Plotly: `python -c "import plotly.express as px; print('OK')"`
- [ ] **T-2.1.5:** Test SHAP: `python -c "import shap; print('OK')"`

### 2.2 Project Structure Setup

**Tasks:**
- [ ] **T-2.2.1:** Create `src/ui/` directory
- [ ] **T-2.2.2:** Create `__init__.py` in `src/ui/`
- [ ] **T-2.2.3:** Create empty template files:
  - `src/ui/app.py`
  - `src/ui/data_manager.py`
  - `src/ui/model_controller.py`
  - `src/ui/visualization.py`
  - `src/ui/explainability.py`
  - `src/ui/strategy.py`
  - `src/ui/utils.py`
- [ ] **T-2.2.4:** Create symlink: `ln -s src/ui/app.py app.py` (for easy launch)

### 2.3 Streamlit Configuration

**Tasks:**
- [ ] **T-2.3.1:** Create `.streamlit/config.toml`:
  ```toml
  [theme]
  primaryColor = "#1f77b4"
  backgroundColor = "#ffffff"
  secondaryBackgroundColor = "#f0f2f6"
  textColor = "#262730"
  font = "sans serif"

  [server]
  maxUploadSize = 200
  enableCORS = false

  [browser]
  gatherUsageStats = false
  ```
- [ ] **T-2.3.2:** Test configuration loads correctly

---

## 📊 PHASE 3: DATA VISUALIZATION COMPONENTS

### 3.1 Core Chart Components

**Chart Types Needed:**
1. **Candlestick Chart** - Price action with volume
2. **Volatility Time Series** - Intraday range over time with regime overlay
3. **Regime Heatmap** - Transition probabilities between regimes
4. **Confidence Gauge** - 0-100 score visualization
5. **Feature Contribution Bar Chart** - Horizontal bar chart showing feature impact
6. **SHAP Waterfall Plot** - Feature importance for single prediction
7. **SHAP Beeswarm Plot** - Feature importance across all predictions
8. **Confusion Matrix Heatmap** - Validation performance
9. **Walk-Forward Performance** - Accuracy/Win Rate over time
10. **P&L Distribution** - Expected value histogram

### 3.2 Visualization Module (`src/ui/visualization.py`)

**Tasks:**
- [ ] **T-3.2.1:** Create `plot_candlestick_with_regime()` function
  - Input: DataFrame with OHLCV + regime labels
  - Output: Plotly figure with candlesticks and colored regime background
  - Features: Zoom, pan, hover with regime info

- [ ] **T-3.2.2:** Create `plot_volatility_timeseries()` function
  - Input: DataFrame with intraday_range_pct, regime labels, predictions
  - Output: Plotly line chart with regime-colored segments
  - Features: Threshold line, actual vs predicted overlay

- [ ] **T-3.2.3:** Create `plot_regime_heatmap()` function
  - Input: Regime transition matrix from HMM
  - Output: Plotly heatmap with annotations
  - Features: Hover shows transition probability

- [ ] **T-3.2.4:** Create `plot_confidence_gauge()` function
  - Input: Confidence score (0-100)
  - Output: Plotly indicator gauge
  - Features: Color zones (0-25 red, 25-40 yellow, 40-100 green)

- [ ] **T-3.2.5:** Create `plot_feature_contributions()` function
  - Input: Dictionary of {feature: contribution_score}
  - Output: Plotly horizontal bar chart
  - Features: Sorted by absolute value, color by positive/negative

- [ ] **T-3.2.6:** Create `plot_confusion_matrix()` function
  - Input: Confusion matrix (TP, TN, FP, FN)
  - Output: Seaborn/Plotly heatmap
  - Features: Annotations with counts and percentages

- [ ] **T-3.2.7:** Create `plot_validation_metrics()` function
  - Input: DataFrame with daily accuracy, win rate over time
  - Output: Plotly multi-line chart
  - Features: Reference lines for targets (50% accuracy, 40% win rate)

- [ ] **T-3.2.8:** Create `plot_pnl_distribution()` function
  - Input: Array of P&L outcomes
  - Output: Plotly histogram with expected value line
  - Features: Stats overlay (mean, median, std)

**Code Pattern Template:**
```python
def plot_candlestick_with_regime(
    df: pd.DataFrame,
    regime_col: str = 'regime_label',
    title: str = 'SPY Price Action with Volatility Regimes'
) -> go.Figure:
    """
    Create candlestick chart with regime-colored background.

    Args:
        df: DataFrame with OHLCV data and regime labels
        regime_col: Column name for regime labels
        title: Chart title

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Add candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='SPY'
    ))

    # Add regime background colors
    regime_colors = {
        'low_vol': 'rgba(0, 255, 0, 0.1)',
        'normal_vol': 'rgba(255, 255, 0, 0.1)',
        'high_vol': 'rgba(255, 0, 0, 0.1)'
    }

    for regime, color in regime_colors.items():
        regime_periods = df[df[regime_col] == regime]
        for idx in regime_periods.index:
            fig.add_vrect(
                x0=idx, x1=idx + pd.Timedelta(days=1),
                fillcolor=color, layer='below',
                line_width=0
            )

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x unified',
        template='plotly_white'
    )

    return fig
```

### 3.3 SHAP Explainability Module (`src/ui/explainability.py`)

**Tasks:**
- [ ] **T-3.3.1:** Create `SHAPExplainer` class wrapper around SHAP library
- [ ] **T-3.3.2:** Implement `calculate_shap_values()` method
  - Use TreeExplainer for HMM features
  - Cache results for performance
- [ ] **T-3.3.3:** Implement `plot_shap_waterfall()` method
  - Show feature contributions for single prediction
- [ ] **T-3.3.4:** Implement `plot_shap_beeswarm()` method
  - Show feature importance across all predictions
- [ ] **T-3.3.5:** Implement `get_feature_importance_ranking()` method
  - Return sorted DataFrame of feature importance

**Code Template:**
```python
import shap
import streamlit as st

class SHAPExplainer:
    """SHAP-based explainability for volatility predictions."""

    def __init__(self, model, features: List[str]):
        self.model = model
        self.features = features
        self.explainer = None
        self.shap_values = None

    @st.cache_data
    def calculate_shap_values(_self, X: np.ndarray) -> np.ndarray:
        """Calculate SHAP values for feature matrix."""
        if _self.explainer is None:
            _self.explainer = shap.Explainer(_self.model, X)
        _self.shap_values = _self.explainer(X)
        return _self.shap_values

    def plot_waterfall(self, prediction_idx: int = -1) -> go.Figure:
        """Plot SHAP waterfall for single prediction."""
        shap.plots.waterfall(self.shap_values[prediction_idx])
        # Convert matplotlib to plotly for interactivity
        return convert_mpl_to_plotly(plt.gcf())
```

---

## 🎛️ PHASE 4: MODEL CONTROL PANEL

### 4.1 Data Settings Panel

**UI Components:**
- [ ] **T-4.1.1:** Symbol selector (default: SPY, expandable to other tickers)
- [ ] **T-4.1.2:** Date range picker (start date, end date)
- [ ] **T-4.1.3:** History days slider (30-365 days)
- [ ] **T-4.1.4:** Data refresh button with loading spinner
- [ ] **T-4.1.5:** Data quality indicator (rows downloaded, missing data %)

**Streamlit Code Template:**
```python
# src/ui/app.py - Sidebar Data Settings Section
st.sidebar.header("📊 Data Settings")

symbol = st.sidebar.selectbox(
    "Symbol",
    options=['SPY', 'QQQ', 'IWM', 'DIA'],
    index=0,
    help="Select the ticker symbol to analyze"
)

history_days = st.sidebar.slider(
    "History Days",
    min_value=30,
    max_value=365,
    value=180,
    step=30,
    help="Number of historical days to download"
)

if st.sidebar.button("🔄 Refresh Data", type="primary"):
    with st.spinner("Downloading data..."):
        data = load_data(symbol, history_days)
        st.success(f"✅ Downloaded {len(data)} bars")
```

### 4.2 HMM Parameters Panel

**UI Components:**
- [ ] **T-4.2.1:** N Regimes selector (2-5, default: 3)
- [ ] **T-4.2.2:** Feature multi-select (checkboxes for all 5 HMM features)
- [ ] **T-4.2.3:** Training days slider (90-300, default: 180)
- [ ] **T-4.2.4:** Iterations slider (50-200, default: 100)
- [ ] **T-4.2.5:** Random seed input (for reproducibility)
- [ ] **T-4.2.6:** "Train HMM" button
- [ ] **T-4.2.7:** Training status indicator (convergence, log-likelihood)

**Streamlit Code Template:**
```python
st.sidebar.header("🎯 HMM Parameters")

n_regimes = st.sidebar.slider(
    "Number of Regimes",
    min_value=2,
    max_value=5,
    value=3,
    help="Number of volatility regimes (typically 3: low, normal, high)"
)

hmm_features = st.sidebar.multiselect(
    "HMM Features",
    options=[
        'overnight_gap_abs',
        'range_ma_5',
        'vix_level',
        'volume_ratio',
        'range_std_5'
    ],
    default=[
        'overnight_gap_abs',
        'range_ma_5',
        'vix_level',
        'volume_ratio',
        'range_std_5'
    ],
    help="Select features for HMM training"
)

if st.sidebar.button("🚀 Train HMM"):
    with st.spinner("Training HMM..."):
        hmm_model, metrics = train_hmm_model(
            data,
            n_regimes=n_regimes,
            features=hmm_features
        )
        if metrics['converged']:
            st.sidebar.success("✅ HMM Trained Successfully")
            st.sidebar.metric("Log-Likelihood", f"{metrics['log_likelihood']:.2f}")
        else:
            st.sidebar.warning("⚠️ HMM did not converge")
```

### 4.3 TimesFM Parameters Panel

**UI Components:**
- [ ] **T-4.3.1:** Enable/Disable toggle for TimesFM
- [ ] **T-4.3.2:** Checkpoint selector (dropdown of available checkpoints)
- [ ] **T-4.3.3:** Context length slider (30-120 days, default: 60)
- [ ] **T-4.3.4:** Device selector (CPU/CUDA)
- [ ] **T-4.3.5:** Download checkpoint button (if not available)
- [ ] **T-4.3.6:** Status indicator (loaded/unavailable)

**Streamlit Code Template:**
```python
st.sidebar.header("🔮 TimesFM Parameters")

timesfm_enabled = st.sidebar.toggle(
    "Enable TimesFM",
    value=False,
    help="Use TimesFM foundation model for forecasting (requires ~800MB download)"
)

if timesfm_enabled:
    context_length = st.sidebar.slider(
        "Context Length",
        min_value=30,
        max_value=120,
        value=60,
        help="Days of historical volatility to use as context"
    )

    if not is_timesfm_available():
        st.sidebar.warning("⚠️ TimesFM checkpoint not downloaded")
        if st.sidebar.button("📥 Download Checkpoint (~800MB)"):
            download_timesfm_checkpoint()
else:
    st.sidebar.info("ℹ️ Using HMM-only mode")
```

### 4.4 Model Configuration Panel

**UI Components:**
- [ ] **T-4.4.1:** Confidence threshold slider (0-100, default: 40)
- [ ] **T-4.4.2:** Weight sliders for ensemble (regime, timesfm, features)
- [ ] **T-4.4.3:** Volatility threshold input (default: 1.2%)
- [ ] **T-4.4.4:** Save/Load configuration buttons
- [ ] **T-4.4.5:** Reset to defaults button

**Streamlit Code Template:**
```python
st.sidebar.header("⚙️ Model Configuration")

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0,
    max_value=100,
    value=40,
    help="Minimum confidence score to trigger TRADE signal"
)

st.sidebar.subheader("Ensemble Weights")
weight_regime = st.sidebar.slider("Regime Weight", 0.0, 1.0, 0.4, 0.05)
weight_timesfm = st.sidebar.slider("TimesFM Weight", 0.0, 1.0, 0.4, 0.05)
weight_features = st.sidebar.slider("Features Weight", 0.0, 1.0, 0.2, 0.05)

# Normalize weights to sum to 1.0
total_weight = weight_regime + weight_timesfm + weight_features
if total_weight != 1.0:
    st.sidebar.warning(f"⚠️ Weights sum to {total_weight:.2f}, will be normalized")
```

### 4.5 Model Controller Backend (`src/ui/model_controller.py`)

**Tasks:**
- [ ] **T-4.5.1:** Create `ModelController` class
- [ ] **T-4.5.2:** Implement `train_hmm()` method with progress callback
- [ ] **T-4.5.3:** Implement `load_timesfm()` method with caching
- [ ] **T-4.5.4:** Implement `run_prediction()` method
- [ ] **T-4.5.5:** Implement `run_validation()` method with walk-forward
- [ ] **T-4.5.6:** Add state management (st.session_state)

**Code Template:**
```python
import streamlit as st
from typing import Dict, Tuple

class ModelController:
    """Manages model training, prediction, and validation."""

    def __init__(self):
        if 'hmm_model' not in st.session_state:
            st.session_state.hmm_model = None
        if 'timesfm_forecaster' not in st.session_state:
            st.session_state.timesfm_forecaster = None

    def train_hmm(
        self,
        df: pd.DataFrame,
        n_regimes: int,
        features: List[str],
        n_iter: int = 100
    ) -> Tuple[VolatilityHMM, Dict]:
        """Train HMM and store in session state."""
        hmm_model = VolatilityHMM(n_regimes=n_regimes)
        metrics = hmm_model.train(df[features], n_iter=n_iter)

        # Store in session state
        st.session_state.hmm_model = hmm_model
        st.session_state.hmm_metrics = metrics

        return hmm_model, metrics

    @st.cache_resource
    def load_timesfm(_self, checkpoint: str, device: str) -> TimesFMVolatilityForecaster:
        """Load TimesFM with caching."""
        forecaster = TimesFMVolatilityForecaster(
            checkpoint=checkpoint,
            device=device
        )
        st.session_state.timesfm_forecaster = forecaster
        return forecaster
```

---

## 🧠 PHASE 5: EXPLAINABILITY FEATURES

### 5.1 Feature Importance Dashboard

**Components:**
- [ ] **T-5.1.1:** Global feature importance ranking (averaged across all predictions)
- [ ] **T-5.1.2:** Local feature importance (SHAP waterfall for selected prediction)
- [ ] **T-5.1.3:** Feature correlation heatmap
- [ ] **T-5.1.4:** Feature distribution plots (violin plots for each feature by regime)
- [ ] **T-5.1.5:** Feature contribution trends over time

**UI Layout:**
```python
# Tab 2: Model Explanation
with tab2:
    st.header("🧠 Model Explanation")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Feature Importance Ranking")
        fig_importance = plot_feature_importance(shap_values)
        st.plotly_chart(fig_importance, use_container_width=True)

    with col2:
        st.subheader("Feature Correlations")
        fig_corr = plot_feature_correlation(features_df)
        st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Local Explanation (Selected Prediction)")
    selected_date = st.select_slider(
        "Select Date",
        options=df.index.tolist(),
        value=df.index[-1]
    )
    fig_waterfall = plot_shap_waterfall(shap_values, selected_date)
    st.plotly_chart(fig_waterfall, use_container_width=True)
```

### 5.2 Regime Analysis Dashboard

**Components:**
- [ ] **T-5.2.1:** Regime transition matrix (heatmap)
- [ ] **T-5.2.2:** Regime duration statistics (box plot)
- [ ] **T-5.2.3:** Regime-specific feature distributions
- [ ] **T-5.2.4:** Historical regime timeline
- [ ] **T-5.2.5:** Regime prediction confidence over time

**Code Tasks:**
- [ ] **T-5.2.6:** Create `plot_regime_transitions()` function
- [ ] **T-5.2.7:** Create `plot_regime_durations()` function
- [ ] **T-5.2.8:** Create `plot_regime_timeline()` function

### 5.3 Prediction Explainability

**Components:**
- [ ] **T-5.3.1:** Confidence score breakdown (regime + timesfm + features)
- [ ] **T-5.3.2:** Individual feature contributions with direction (positive/negative)
- [ ] **T-5.3.3:** Comparison to historical similar days
- [ ] **T-5.3.4:** "What-if" scenario tool (adjust features, see prediction change)

**What-If Tool UI:**
```python
st.subheader("🔬 What-If Scenario Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    vix_override = st.number_input(
        "VIX Level Override",
        value=float(latest_features['vix_level']),
        min_value=10.0,
        max_value=50.0,
        step=1.0
    )

with col2:
    gap_override = st.number_input(
        "Overnight Gap Override (%)",
        value=float(latest_features['overnight_gap_abs']) * 100,
        min_value=0.0,
        max_value=5.0,
        step=0.1
    )

if st.button("Run What-If Prediction"):
    modified_features = latest_features.copy()
    modified_features['vix_level'] = vix_override
    modified_features['overnight_gap_abs'] = gap_override / 100

    new_prediction = model.predict(modified_features)
    st.metric("New Confidence Score", f"{new_prediction:.1f}/100")
```

---

## 💰 PHASE 6: TRADING STRATEGY OUTPUT

### 6.1 Spread Recommendation Engine (`src/ui/strategy.py`)

**Tasks:**
- [ ] **T-6.1.1:** Create `SpreadRecommender` class
- [ ] **T-6.1.2:** Implement regime-to-spread mapping logic:
  - High volatility → Wide strangle (OTM strikes)
  - Normal volatility → Narrow strangle
  - Low volatility → SKIP
- [ ] **T-6.1.3:** Calculate optimal strike distances based on predicted volatility
- [ ] **T-6.1.4:** Implement position sizing recommendations

**Code Template:**
```python
class SpreadRecommender:
    """Recommends options spread strategy based on volatility prediction."""

    def __init__(self):
        self.regime_strategies = {
            'high_vol': {
                'name': 'Wide Strangle',
                'call_delta': 0.30,  # 30 delta call
                'put_delta': -0.30,  # 30 delta put
                'contracts': 1,
                'rationale': 'High volatility expected - wider strikes capture large moves'
            },
            'normal_vol': {
                'name': 'Narrow Strangle',
                'call_delta': 0.40,
                'put_delta': -0.40,
                'contracts': 1,
                'rationale': 'Normal volatility - standard strangle setup'
            },
            'low_vol': {
                'name': 'SKIP',
                'rationale': 'Low volatility - insufficient edge for trade'
            }
        }

    def recommend_spread(
        self,
        regime: str,
        current_price: float,
        predicted_volatility: float,
        confidence: float
    ) -> Dict:
        """Generate spread recommendation."""
        strategy = self.regime_strategies[regime]

        if regime == 'low_vol':
            return strategy

        # Calculate strike prices (simplified)
        implied_move = current_price * predicted_volatility
        call_strike = current_price + implied_move
        put_strike = current_price - implied_move

        return {
            **strategy,
            'current_price': current_price,
            'call_strike': round(call_strike, 2),
            'put_strike': round(put_strike, 2),
            'expected_move': round(implied_move, 2),
            'confidence': confidence
        }
```

### 6.2 Trading Strategy Visualization

**UI Components:**
- [ ] **T-6.2.1:** Strategy card (spread type, strikes, rationale)
- [ ] **T-6.2.2:** P&L diagram (profit/loss at different price points)
- [ ] **T-6.2.3:** Greeks display (delta, gamma, theta, vega)
- [ ] **T-6.2.4:** Entry/exit rules checklist
- [ ] **T-6.2.5:** Risk metrics (max loss, max gain, breakevens)

**UI Code:**
```python
# Tab 4: Trading Strategy
with tab4:
    st.header("💰 Trading Strategy Recommendation")

    if confidence_score >= threshold:
        recommendation = spread_recommender.recommend_spread(
            regime=regime_label,
            current_price=current_price,
            predicted_volatility=predicted_vol,
            confidence=confidence_score
        )

        # Strategy Card
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Strategy", recommendation['name'])
        with col2:
            st.metric("Call Strike", f"${recommendation['call_strike']}")
        with col3:
            st.metric("Put Strike", f"${recommendation['put_strike']}")

        st.info(f"📖 **Rationale:** {recommendation['rationale']}")

        # P&L Diagram
        st.subheader("Expected P&L Diagram")
        fig_pnl = plot_pnl_diagram(recommendation)
        st.plotly_chart(fig_pnl, use_container_width=True)

        # Entry/Exit Rules
        with st.expander("📋 Entry & Exit Rules"):
            st.markdown(f"""
            **Entry:**
            - Time: 9:30 AM EST (market open)
            - Spread: {recommendation['name']}
            - Size: {recommendation['contracts']} contract(s)

            **Exit:**
            - Time: 1:00 PM EST (regardless of P&L)
            - Max Loss: $80/contract
            - Target Profit: $150/contract
            """)
    else:
        st.warning("⏸️ **SKIP TRADE** - Confidence below threshold")
```

### 6.3 Risk Metrics Calculator

**Tasks:**
- [ ] **T-6.3.1:** Calculate max loss (premium paid)
- [ ] **T-6.3.2:** Calculate max gain (based on strike width)
- [ ] **T-6.3.3:** Calculate breakeven points
- [ ] **T-6.3.4:** Calculate risk/reward ratio
- [ ] **T-6.3.5:** Display portfolio heat (% of account at risk)

---

## 🧪 PHASE 7: VALIDATION & BACKTESTING INTERFACE

### 7.1 Walk-Forward Validation UI

**Components:**
- [ ] **T-7.1.1:** Date range selector for validation period
- [ ] **T-7.1.2:** Train/Test split slider (e.g., 180/30 days)
- [ ] **T-7.1.3:** Run validation button
- [ ] **T-7.1.4:** Progress bar for validation
- [ ] **T-7.1.5:** Results display (accuracy, precision, recall, F1)
- [ ] **T-7.1.6:** Confusion matrix visualization
- [ ] **T-7.1.7:** Day-by-day predictions table (actual vs predicted)

**UI Code:**
```python
# Tab 3: Validation Results
with tab3:
    st.header("🔍 Walk-Forward Validation")

    col1, col2 = st.columns(2)
    with col1:
        train_days = st.number_input("Training Days", value=180, min_value=60, max_value=365)
    with col2:
        test_days = st.number_input("Test Days", value=30, min_value=10, max_value=90)

    if st.button("▶️ Run Validation", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        validation_results = run_validation(
            df,
            train_days=train_days,
            test_days=test_days,
            progress_callback=lambda pct, msg: (
                progress_bar.progress(pct),
                status_text.text(msg)
            )
        )

        # Display metrics
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("Accuracy", f"{validation_results['accuracy']:.1f}%")
        with metric_cols[1]:
            st.metric("Win Rate", f"{validation_results['win_rate']:.1f}%")
        with metric_cols[2]:
            st.metric("Trades", validation_results['total_trades'])
        with metric_cols[3]:
            st.metric("Expected Value", f"${validation_results['ev_per_trade']:.2f}")

        # Confusion matrix
        fig_confusion = plot_confusion_matrix(validation_results['confusion_matrix'])
        st.plotly_chart(fig_confusion, use_container_width=True)
```

### 7.2 Historical Performance Charts

**Tasks:**
- [ ] **T-7.2.1:** Create cumulative accuracy chart (rolling window)
- [ ] **T-7.2.2:** Create win rate over time chart
- [ ] **T-7.2.3:** Create monthly performance table
- [ ] **T-7.2.4:** Create regime accuracy breakdown (accuracy by regime)
- [ ] **T-7.2.5:** Create calendar heatmap (green = correct, red = wrong)

### 7.3 Export & Reporting

**Tasks:**
- [ ] **T-7.3.1:** Export validation results to CSV
- [ ] **T-7.3.2:** Export all charts to PDF report
- [ ] **T-7.3.3:** Export configuration to JSON
- [ ] **T-7.3.4:** Generate markdown summary report

**UI Code:**
```python
st.sidebar.header("📤 Export")

if st.sidebar.button("Download Validation Results (CSV)"):
    csv = validation_results_df.to_csv(index=False)
    st.sidebar.download_button(
        label="💾 Download CSV",
        data=csv,
        file_name=f"validation_results_{datetime.now():%Y%m%d}.csv",
        mime="text/csv"
    )

if st.sidebar.button("Generate PDF Report"):
    with st.spinner("Generating report..."):
        pdf_buffer = generate_pdf_report(
            charts=[fig1, fig2, fig3],
            metrics=validation_results,
            config=current_config
        )
        st.sidebar.download_button(
            label="📄 Download PDF",
            data=pdf_buffer,
            file_name=f"volatility_report_{datetime.now():%Y%m%d}.pdf",
            mime="application/pdf"
        )
```

---

## 🎨 PHASE 8: UI POLISH & USER EXPERIENCE

### 8.1 Loading States & Performance

**Tasks:**
- [ ] **T-8.1.1:** Add loading spinners for all async operations
- [ ] **T-8.1.2:** Implement data caching with `@st.cache_data`
- [ ] **T-8.1.3:** Implement model caching with `@st.cache_resource`
- [ ] **T-8.1.4:** Add progress bars for long operations (training, validation)
- [ ] **T-8.1.5:** Optimize chart rendering (limit data points if > 1000)

**Caching Example:**
```python
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_historical_data(symbol: str, days: int) -> pd.DataFrame:
    """Load data with caching."""
    return alpaca_client.get_daily_bars(symbol, days=days)

@st.cache_resource
def load_hmm_model(_path: str) -> VolatilityHMM:
    """Load model with caching (use _ prefix for unhashable args)."""
    model = VolatilityHMM()
    model.load(_path)
    return model
```

### 8.2 Error Handling & Validation

**Tasks:**
- [ ] **T-8.2.1:** Validate user inputs (e.g., weights sum to 1.0)
- [ ] **T-8.2.2:** Handle API errors gracefully (Alpaca, yfinance)
- [ ] **T-8.2.3:** Display clear error messages with recovery suggestions
- [ ] **T-8.2.4:** Add input validation for date ranges
- [ ] **T-8.2.5:** Handle missing data gracefully

**Error Handling Pattern:**
```python
try:
    data = load_data(symbol, days)
except AlpacaAPIError as e:
    st.error(f"❌ Failed to download data from Alpaca: {str(e)}")
    st.info("💡 Try again in a few minutes or check your API credentials.")
    st.stop()
except Exception as e:
    st.error(f"❌ Unexpected error: {str(e)}")
    with st.expander("🔍 Debug Info"):
        st.code(traceback.format_exc())
    st.stop()
```

### 8.3 Tooltips & Help System

**Tasks:**
- [ ] **T-8.3.1:** Add help tooltips to all parameters
- [ ] **T-8.3.2:** Create "How to Use" expander in sidebar
- [ ] **T-8.3.3:** Add glossary of terms (regime, SHAP, strangle, etc.)
- [ ] **T-8.3.4:** Create video tutorial placeholder
- [ ] **T-8.3.5:** Add FAQ section

**Help System Example:**
```python
with st.sidebar.expander("❓ How to Use This Dashboard"):
    st.markdown("""
    ### Quick Start Guide

    1. **Load Data:** Adjust data settings and click "Refresh Data"
    2. **Train Model:** Set HMM parameters and click "Train HMM"
    3. **View Prediction:** Check the Prediction Dashboard tab
    4. **Understand Why:** Check the Model Explanation tab
    5. **Validate:** Run walk-forward validation in Validation tab
    6. **Trade:** Review recommended strategy in Trading Strategy tab

    ### Key Terms
    - **Regime:** Market volatility state (low/normal/high)
    - **SHAP:** Explainability method showing feature contributions
    - **Strangle:** Options strategy with OTM call + put
    - **Confidence Score:** 0-100 score indicating trade signal strength
    """)
```

### 8.4 Responsive Design & Accessibility

**Tasks:**
- [ ] **T-8.4.1:** Test on different screen sizes
- [ ] **T-8.4.2:** Ensure colorblind-friendly palette
- [ ] **T-8.4.3:** Add alt text to charts (for screen readers)
- [ ] **T-8.4.4:** Ensure keyboard navigation works
- [ ] **T-8.4.5:** Test with browser zoom (150%, 200%)

---

## 🚀 PHASE 9: TESTING & DEPLOYMENT

### 9.1 Unit Testing

**Tasks:**
- [ ] **T-9.1.1:** Test data loading functions
- [ ] **T-9.1.2:** Test model training functions
- [ ] **T-9.1.3:** Test visualization functions (output type)
- [ ] **T-9.1.4:** Test spread recommendation logic
- [ ] **T-9.1.5:** Test validation metrics calculation

**Test File Structure:**
```
tests/
├── test_ui_data_manager.py
├── test_ui_model_controller.py
├── test_ui_visualization.py
├── test_ui_explainability.py
└── test_ui_strategy.py
```

### 9.2 Integration Testing

**Tasks:**
- [ ] **T-9.2.1:** Test end-to-end prediction flow
- [ ] **T-9.2.2:** Test validation flow
- [ ] **T-9.2.3:** Test export functionality
- [ ] **T-9.2.4:** Test what-if scenarios
- [ ] **T-9.2.5:** Test state persistence across reruns

### 9.3 User Acceptance Testing

**Tasks:**
- [ ] **T-9.3.1:** Run full user workflow (data → train → predict → validate)
- [ ] **T-9.3.2:** Test with different parameter combinations
- [ ] **T-9.3.3:** Test error scenarios (bad data, failed API calls)
- [ ] **T-9.3.4:** Verify all charts render correctly
- [ ] **T-9.3.5:** Verify all exports work

### 9.4 Documentation

**Tasks:**
- [ ] **T-9.4.1:** Create `docs/UI_USER_GUIDE.md`
- [ ] **T-9.4.2:** Create `docs/UI_DEVELOPER_GUIDE.md`
- [ ] **T-9.4.3:** Add docstrings to all UI functions
- [ ] **T-9.4.4:** Create screenshot gallery
- [ ] **T-9.4.5:** Record demo video (optional)

### 9.5 Deployment

**Tasks:**
- [ ] **T-9.5.1:** Test local deployment: `streamlit run app.py`
- [ ] **T-9.5.2:** Create startup script (sets environment, activates venv)
- [ ] **T-9.5.3:** Document port configuration (default: 8501)
- [ ] **T-9.5.4:** Add firewall/network access instructions
- [ ] **T-9.5.5:** Create desktop shortcut (optional)

**Startup Script (`run_ui.sh`):**
```bash
#!/bin/bash
# Volatility Prediction UI Launcher

cd /path/to/commodity-forecasting-system
source .venv/bin/activate
export STREAMLIT_SERVER_PORT=8501
streamlit run app.py
```

---

## 📊 PHASE 10: ADVANCED FEATURES (OPTIONAL)

### 10.1 Multi-Symbol Support

**Tasks:**
- [ ] **T-10.1.1:** Add ability to compare multiple symbols (SPY, QQQ, IWM)
- [ ] **T-10.1.2:** Create correlation matrix between symbols
- [ ] **T-10.1.3:** Display multi-symbol predictions in table
- [ ] **T-10.1.4:** Rank symbols by confidence score

### 10.2 Custom Alerts

**Tasks:**
- [ ] **T-10.2.1:** Add email alert configuration
- [ ] **T-10.2.2:** Add SMS alert configuration (Twilio)
- [ ] **T-10.2.3:** Create alert trigger rules (threshold exceeded)
- [ ] **T-10.2.4:** Test alert delivery

### 10.3 Historical Backtesting

**Tasks:**
- [ ] **T-10.3.1:** Create backtesting engine (simulate trades)
- [ ] **T-10.3.2:** Calculate actual P&L based on historical option prices
- [ ] **T-10.3.3:** Display equity curve
- [ ] **T-10.3.4:** Calculate Sharpe ratio, max drawdown

### 10.4 Real-Time Mode

**Tasks:**
- [ ] **T-10.4.1:** Add auto-refresh capability (every 5 minutes)
- [ ] **T-10.4.2:** Display countdown to next refresh
- [ ] **T-10.4.3:** Add manual refresh button
- [ ] **T-10.4.4:** Store predictions in database for historical tracking

---

## 🎯 IMPLEMENTATION TIMELINE

### Sprint 1 (Days 1-3): Foundation
- **Day 1:** Phase 2 (Environment Setup) + Phase 3.1 (Chart Components)
- **Day 2:** Phase 4 (Model Control Panel)
- **Day 3:** Phase 3.2-3.3 (Visualization Module + SHAP)

### Sprint 2 (Days 4-6): Core Features
- **Day 4:** Phase 5 (Explainability Features)
- **Day 5:** Phase 6 (Trading Strategy Output)
- **Day 6:** Phase 7 (Validation Interface)

### Sprint 3 (Days 7-9): Polish & Testing
- **Day 7:** Phase 8 (UI Polish)
- **Day 8:** Phase 9.1-9.3 (Testing)
- **Day 9:** Phase 9.4-9.5 (Documentation & Deployment)

**Total Estimated Time:** 9 days (3 sprints × 3 days)

---

## ✅ ACCEPTANCE CRITERIA

The UI will be considered complete when:

1. **✅ AC-1:** User can load SPY data and view it in candlestick chart
2. **✅ AC-2:** User can adjust HMM parameters and retrain model
3. **✅ AC-3:** User can adjust confidence threshold and see prediction change
4. **✅ AC-4:** User can see current prediction with confidence score
5. **✅ AC-5:** User can see SHAP feature importance for prediction
6. **✅ AC-6:** User can see recommended trading strategy (spread type, strikes)
7. **✅ AC-7:** User can run walk-forward validation and see metrics
8. **✅ AC-8:** User can export results to CSV
9. **✅ AC-9:** All charts are interactive (zoom, pan, hover)
10. **✅ AC-10:** UI loads in < 3 seconds on local machine

---

## 🔄 ITERATION PLAN

This checklist is designed for iterative development:

1. **Iteration 1:** Basic prediction dashboard (Phases 2-4)
2. **Iteration 2:** Add explainability (Phase 5)
3. **Iteration 3:** Add trading strategy (Phase 6)
4. **Iteration 4:** Add validation (Phase 7)
5. **Iteration 5:** Polish & deploy (Phases 8-9)

After each iteration, review with user and adjust priorities.

---

## 📚 REFERENCE MATERIALS

### Key Resources
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Python Documentation](https://plotly.com/python/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [ExplainerDashboard](https://github.com/oegedijk/explainerdashboard)

### Inspiration Dashboards
- [Trading Stock Market Dashboard UI Kit (Figma)](https://www.figma.com/community/file/1359230717010633204)
- [Commodity Trading Visualization Tools](https://dev3lop.com/commodity-trading-visualization-market-volatility-analysis-tools/)
- [Implied Volatility Dashboard (CQG)](https://news.cqg.com/workspaces/main/2015/02/cqg-implied-volatility-dashboard.html)

---

## 🎬 GETTING STARTED

To begin implementation, start with **Phase 2** tasks in order:
1. Create `requirements-ui.txt`
2. Install dependencies
3. Create project structure
4. Set up Streamlit configuration
5. Build first simple "Hello World" dashboard

**First Command to Run:**
```bash
cd commodity-forecasting-system
touch requirements-ui.txt
# Add dependencies to file
pip install -r requirements-ui.txt
streamlit hello  # Verify installation
```

---

**END OF CHECKLIST**

*Last Updated: 2026-01-17*
*Status: Ready for Implementation*
*Estimated Completion: 9 days (3 sprints)*
