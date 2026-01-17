# HMM-Black-Scholes Commodity Forecasting System: Complete Architectural Checklist

**Objective**: Build a sophisticated, production-grade forecasting system that combines Hidden Markov Models for spot price prediction with Black-Scholes options pricing for futures, featuring configurable parameters, comprehensive UI, and rigorous validation.

**Perspective**: High-level market analyst & quantitative analyst with expertise in stochastic processes, matrix operations, and financial engineering.

**Integration Note**: This system can leverage TimesFM (documented in MAP.md) as an alternative/complementary forecasting engine, particularly for long-context prediction and regime-informed forecasting.

---

## PHASE 0: MATHEMATICAL FOUNDATION & ARCHITECTURE DESIGN

### 0.1 Theoretical Framework Definition
- [ ] **Define HMM mathematical structure**
  - [ ] Hidden state space S = {s₁, s₂, ..., sₙ} (typically n=3: bull/bear/neutral)
  - [ ] Observation space O (price returns, technical indicators, volatility)
  - [ ] Transition matrix A: P(sₜ = sⱼ | sₜ₋₁ = sᵢ) - n×n stochastic matrix
  - [ ] Emission probability distributions B: P(oₜ | sₜ = sᵢ) - Gaussian or GMM
  - [ ] Initial state distribution π: P(s₁ = sᵢ)
  - [ ] Document forward-backward algorithm (α, β recursions)
  - [ ] Document Viterbi algorithm for optimal state sequence
  - [ ] Document Baum-Welch (EM) for parameter estimation

- [ ] **Define Black-Scholes framework for commodities**
  - [ ] Black-76 model for futures: C = e^(-rT)[F₀N(d₁) - KN(d₂)]
  - [ ] d₁ = [ln(F₀/K) + (σ²/2)T] / (σ√T)
  - [ ] d₂ = d₁ - σ√T
  - [ ] Convenience yield modeling: F₀ = S₀e^((r-y)T)
  - [ ] Storage cost adjustments for physical commodities
  - [ ] Volatility surface construction (term structure + strike dimension)
  - [ ] Greeks calculation: Δ, Γ, Θ, ν, ρ (first-order), Vanna, Volga (second-order)

- [ ] **Integration architecture design**
  - [ ] HMM predicts spot price distribution → Black-Scholes input
  - [ ] Regime-dependent volatility: σ(regime) feeds into Black-Scholes
  - [ ] Hybrid pipeline: TimesFM context → HMM regime → Black-Scholes pricing
  - [ ] Decision: TimesFM as primary forecaster vs HMM primary vs ensemble

- [ ] **System architecture diagram**
  - [ ] Data acquisition layer (APIs, storage)
  - [ ] Preprocessing layer (cleaning, feature engineering)
  - [ ] Model layer (HMM, Black-Scholes, optional TimesFM)
  - [ ] Parameter management layer (TOML configs, validation)
  - [ ] Evaluation layer (backtesting, metrics)
  - [ ] UI layer (Streamlit/Gradio)
  - [ ] Deployment layer (serving, monitoring)

### 0.2 Technology Stack Selection
- [ ] **Core Python environment**
  - [ ] Python 3.11+ (for consistency with TimesFM requirements)
  - [ ] Virtual environment strategy (venv vs conda vs uv)
  - [ ] Dependency management: requirements.txt + constraints file

- [ ] **HMM implementation library**
  - [ ] Primary: `hmmlearn` (0.3.2+) - BSD license, well-maintained
  - [ ] Backup: `pomegranate` (if need more flexible distributions)
  - [ ] Consider: custom implementation for exotic emission distributions

- [ ] **Black-Scholes implementation**
  - [ ] `vollib` for volatility calculations and Greeks
  - [ ] `QuantLib` for advanced options pricing (if needed)
  - [ ] `scipy.stats` for numerical methods
  - [ ] Custom Crank-Nicholson PDE solver for exotic options

- [ ] **Data acquisition**
  - [ ] Primary data source decision:
    - [ ] yfinance (prototyping, rate-limited, free)
    - [ ] Quandl/NASDAQ Data Link (production, paid, comprehensive)
    - [ ] Alpha Vantage (mid-tier, good technical indicators)
    - [ ] Polygon.io (real-time, professional)
  - [ ] Fallback sources for redundancy
  - [ ] FRED API for macroeconomic features

- [ ] **Feature engineering**
  - [ ] `ta` (Technical Analysis Library) - recommended for OHLCV
  - [ ] `pandas-ta` (115+ indicators)
  - [ ] `talipp` (for real-time O(1) incremental computation)
  - [ ] `TuneTA` (automated indicator optimization)

- [ ] **Configuration management**
  - [ ] TOML format (explicit typing, safe, readable)
  - [ ] `tomli` for reading, `tomli-w` for writing
  - [ ] Schema validation: `pydantic` or `cerberus`

- [ ] **Hyperparameter tuning**
  - [ ] Optuna (primary - 35% faster than deprecated Hyperopt)
  - [ ] Ray Tune (if distributed tuning needed)

- [ ] **Experiment tracking**
  - [ ] MLflow (self-hosted, open-source)
  - [ ] Weights & Biases (cloud, superior visualization) - optional
  - [ ] Custom logging to SQLite/PostgreSQL

- [ ] **Backtesting framework**
  - [ ] `vectorbt` (ultra-fast, Numba-accelerated)
  - [ ] `Backtrader` (customization, community)
  - [ ] Custom walk-forward validator

- [ ] **UI framework**
  - [ ] Streamlit (primary - rich dashboards, parameter tuning)
  - [ ] Gradio (quick model demos, Hugging Face integration)

- [ ] **Visualization**
  - [ ] Plotly (interactive, publication-quality)
  - [ ] Matplotlib/Seaborn (static, detailed)
  - [ ] Bokeh (web-based real-time)

- [ ] **Optional: TimesFM integration**
  - [ ] `timesfm[torch]` or `timesfm[flax]` from MAP.md
  - [ ] Decision point: use for context length >512 or multi-horizon forecasting

---

## PHASE 1: PROJECT SETUP & INFRASTRUCTURE

### 1.1 Repository Structure
```
commodity-forecasting-system/
├── config/
│   ├── parameters.toml           # Master parameter file
│   ├── schema.json               # Parameter validation schema
│   └── commodities.toml          # Commodity-specific configs
├── data/
│   ├── raw/                      # Downloaded data (gitignored)
│   ├── processed/                # Cleaned, feature-engineered
│   └── external/                 # FRED, alternative sources
├── src/
│   ├── data/
│   │   ├── acquisition.py        # API clients, data download
│   │   ├── preprocessing.py      # Cleaning, outliers, stationarity
│   │   └── features.py           # Technical indicators, engineering
│   ├── models/
│   │   ├── hmm_core.py           # HMM implementation wrapper
│   │   ├── black_scholes.py      # Options pricing engine
│   │   ├── timesfm_adapter.py    # Optional TimesFM integration
│   │   └── ensemble.py           # Hybrid model orchestration
│   ├── evaluation/
│   │   ├── metrics.py            # RMSE, MAE, Sharpe, etc.
│   │   ├── backtester.py         # Walk-forward validation
│   │   └── visualizations.py     # Performance plots
│   ├── optimization/
│   │   ├── hyperparameter.py     # Optuna integration
│   │   └── sensitivity.py        # Parameter sensitivity analysis
│   ├── config/
│   │   ├── loader.py             # TOML config loading
│   │   ├── validator.py          # Schema validation
│   │   └── manager.py            # Runtime config management
│   └── ui/
│       ├── streamlit_app.py      # Main dashboard
│       ├── components/           # Reusable UI components
│       └── gradio_demo.py        # Quick demo interface
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_hmm_prototype.ipynb
│   ├── 03_black_scholes_validation.ipynb
│   └── 04_integration_test.ipynb
├── tests/
│   ├── test_data/
│   ├── test_models/
│   └── test_integration/
├── experiments/                   # MLflow tracking directory
├── outputs/
│   ├── forecasts/                # Generated predictions
│   ├── reports/                  # Performance reports
│   └── figures/                  # Visualizations
├── docs/
│   ├── ARCHITECTURE.md           # System design
│   ├── PARAMETERS.md             # Parameter documentation
│   ├── API.md                    # Code API documentation
│   └── DEPLOYMENT.md             # Production deployment guide
├── BUGS.md                        # Bug tracking (per CLAUDE.md)
├── PROGRESS.md                    # Progress tracking
├── DECISIONS.md                   # Architecture decision log
├── pyproject.toml                # Package metadata
├── requirements.txt              # Dependencies
└── README.md                     # Project overview
```

- [ ] **Initialize repository**
  - [ ] `git init` and create `.gitignore` (data/, .venv/, __pycache__/)
  - [ ] Set up git branches: `main`, `dev`, `experiments`
  - [ ] Initialize pre-commit hooks for code quality

- [ ] **Create virtual environment**
  ```bash
  python -m venv .venv
  source .venv/bin/activate  # or .venv\Scripts\activate on Windows
  python -m pip install -U pip setuptools wheel
  ```

- [ ] **Install dependencies**
  ```bash
  # Core
  pip install numpy pandas scipy scikit-learn statsmodels

  # HMM
  pip install hmmlearn

  # Options pricing
  pip install vollib py_vollib

  # Data acquisition
  pip install yfinance alpha_vantage fredapi

  # Feature engineering
  pip install ta pandas-ta talipp

  # Hyperparameter tuning
  pip install optuna

  # Experiment tracking
  pip install mlflow

  # Backtesting
  pip install vectorbt backtrader

  # Config management
  pip install tomli tomli-w pydantic

  # Visualization
  pip install plotly matplotlib seaborn bokeh

  # UI
  pip install streamlit gradio

  # Optional: TimesFM
  pip install timesfm[torch]  # or [flax]

  # Development
  pip install pytest black ruff jupyter
  ```

- [ ] **Create requirements.txt**
  ```bash
  pip freeze > requirements.txt
  ```

### 1.2 Configuration System Foundation

- [ ] **Create parameters.toml master file**
  ```toml
  # config/parameters.toml

  [meta]
  version = "1.0.0"
  last_updated = "2026-01-16"
  author = "Quantitative Research Team"

  [commodity]
  # Primary commodity to forecast
  # Options: "GC=F" (Gold), "CL=F" (Crude Oil), "SI=F" (Silver), "NG=F" (Natural Gas)
  ticker = "GC=F"
  name = "Gold Futures"

  [data]
  # Historical data period
  start_date = "2015-01-01"
  end_date = "2026-01-15"

  # Data source priority (1=highest)
  [data.sources]
  yfinance = 1
  alpha_vantage = 2
  quandl = 3

  # Preprocessing parameters
  [data.preprocessing]
  # Handle missing data: "ffill", "interpolate", "drop"
  missing_method = "ffill"

  # Outlier detection method: "zscore", "iqr", "isolation_forest"
  outlier_method = "zscore"
  outlier_threshold = 3.0  # z-score threshold

  # Stationarity transformation: "returns", "log_returns", "diff", "none"
  transformation = "log_returns"

  [features]
  # Technical indicators to compute
  # Type: list of strings
  # Available: see ta library documentation
  indicators = [
    "sma_20", "sma_50", "sma_200",
    "ema_12", "ema_26",
    "rsi_14",
    "macd",
    "bollinger_bands",
    "atr_14",
    "obv"
  ]

  # Macroeconomic features from FRED
  fred_series = [
    "DGS10",      # 10-Year Treasury Rate
    "DTWEXBGS",   # Trade Weighted USD Index
    "CPIAUCSL",   # CPI
    "VIXCLS"      # VIX
  ]

  # Lagged features
  # Type: integer
  # Range: [1, 60]
  # Impact: More lags = more context but risk of overfitting
  max_lags = 10

  [hmm]
  # Number of hidden states (market regimes)
  # Type: integer
  # Range: [2, 6]
  # Default: 3 (bull, bear, neutral)
  # Impact: More states = more granular regimes but overfitting risk
  n_states = 3

  # Covariance structure
  # Type: string
  # Options: "diag", "full", "spherical", "tied"
  # Default: "diag" (balance of flexibility and stability)
  # Impact: "full" captures correlations but needs more data
  covariance_type = "diag"

  # Maximum EM iterations
  # Type: integer
  # Range: [100, 5000]
  # Default: 1000
  # Impact: More iterations = better convergence but longer training
  n_iter = 1000

  # Convergence tolerance
  # Type: float
  # Range: [1e-6, 1e-2]
  # Default: 1e-4
  # Impact: Smaller = more precise but slower
  tol = 1e-4

  # Multiple random initializations (avoid local optima)
  # Type: integer
  # Range: [5, 50]
  # Default: 10
  # Impact: More inits = better global optimum but slower
  n_random_inits = 10

  # Random seed for reproducibility
  random_seed = 42

  [black_scholes]
  # Model variant: "black76" (futures), "bsm" (equity-style)
  model_type = "black76"

  # Risk-free rate source: "constant", "fred_dgs10", "curve"
  # Type: string
  # Default: "fred_dgs10" (use 10-year Treasury as proxy)
  rate_source = "fred_dgs10"

  # If constant rate
  # Type: float
  # Range: [0.0, 0.10]
  # Default: 0.035 (3.5% reflecting 2026 environment)
  constant_rate = 0.035

  # Convenience yield estimation method
  # Type: string
  # Options: "fitted", "futures_curve", "constant"
  # Default: "futures_curve"
  convenience_yield_method = "futures_curve"

  # Volatility estimation approach
  # Type: string
  # Options: "historical", "garch", "realized", "implied", "hybrid"
  # Default: "hybrid" (combines GARCH + realized)
  volatility_method = "hybrid"

  # Historical volatility window (days)
  # Type: integer
  # Range: [20, 252]
  # Default: 60
  hist_vol_window = 60

  # GARCH model specification
  [black_scholes.garch]
  p = 1  # GARCH lag order
  q = 1  # ARCH lag order

  # Option parameters
  [black_scholes.options]
  # Time to maturity (days)
  # Type: list of integers
  # Default: multiple maturities for surface construction
  maturities = [30, 60, 90, 180, 365]

  # Strike prices relative to spot (percentage)
  # Type: list of floats
  # Example: [0.9, 0.95, 1.0, 1.05, 1.1] for 90%-110% strikes
  strike_ratios = [0.90, 0.95, 1.00, 1.05, 1.10]

  [timesfm]
  # Optional TimesFM integration
  # Enable TimesFM as complementary forecaster
  enabled = false

  # Model variant: "torch", "flax"
  backend = "torch"

  # Hugging Face checkpoint
  checkpoint = "google/timesfm-2.5-200m-pytorch"

  # Forecast configuration
  max_context = 1024    # Up to 16384 available
  max_horizon = 256
  normalize_inputs = true
  use_continuous_quantile_head = true
  force_flip_invariance = true
  infer_is_positive = true
  fix_quantile_crossing = true

  # Integration mode
  # Options: "ensemble", "primary", "regime_input"
  # ensemble: average TimesFM and HMM predictions
  # primary: use TimesFM, HMM only for regime
  # regime_input: feed HMM regime states to TimesFM as features
  integration_mode = "ensemble"

  [validation]
  # Train/test split strategy
  # Type: string
  # Options: "walk_forward", "expanding_window", "rolling_window"
  # Default: "walk_forward"
  strategy = "walk_forward"

  # Walk-forward parameters
  train_window_days = 756  # ~3 years
  test_window_days = 63    # ~3 months
  step_size_days = 21      # ~1 month

  # Cross-validation folds
  n_splits = 5

  # Metrics to compute
  metrics = [
    "rmse", "mae", "mape", "dstat",
    "sharpe_ratio", "calmar_ratio", "max_drawdown"
  ]

  # Backtesting
  [validation.backtesting]
  initial_capital = 100000.0
  transaction_cost_bps = 5  # 5 basis points
  slippage_bps = 2
  position_size_pct = 0.25  # 25% of capital per trade

  [optimization]
  # Enable hyperparameter optimization
  enabled = true

  # Framework: "optuna", "ray_tune"
  framework = "optuna"

  # Optuna settings
  [optimization.optuna]
  n_trials = 100
  timeout_seconds = 3600  # 1 hour

  # Pruning to stop unpromising trials early
  pruner = "median"  # "median", "hyperband", "none"

  # Sampler for search space exploration
  sampler = "tpe"  # "tpe", "random", "grid"

  # Search spaces for key parameters
  [optimization.search_spaces]

  [optimization.search_spaces.hmm_n_states]
  type = "int"
  low = 2
  high = 6

  [optimization.search_spaces.hmm_covariance]
  type = "categorical"
  choices = ["diag", "full", "spherical"]

  [optimization.search_spaces.learning_rate]  # If using ML models
  type = "loguniform"
  low = 1e-5
  high = 1e-1

  [sensitivity]
  # Sensitivity analysis method
  # Options: "ml_ampsit", "sobol", "one_at_a_time"
  method = "one_at_a_time"

  # Parameters to analyze
  parameters = [
    "hmm.n_states",
    "hmm.n_iter",
    "black_scholes.hist_vol_window",
    "features.max_lags"
  ]

  # Perturbation percentage
  perturbation_pct = 0.1  # ±10%

  [ui]
  # Streamlit configuration
  theme = "dark"
  port = 8501

  # Dashboard refresh interval (seconds)
  refresh_interval = 60

  # Enable real-time mode
  realtime_enabled = false

  [logging]
  level = "INFO"  # DEBUG, INFO, WARNING, ERROR
  format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file = "logs/system.log"

  [mlflow]
  tracking_uri = "./experiments"
  experiment_name = "commodity-forecasting"
  auto_log = true
  ```

- [ ] **Create parameter validation schema**
  ```python
  # src/config/validator.py
  from pydantic import BaseModel, Field, validator
  from typing import Literal, List

  class HMMConfig(BaseModel):
      n_states: int = Field(ge=2, le=6)
      covariance_type: Literal["diag", "full", "spherical", "tied"]
      n_iter: int = Field(ge=100, le=5000)
      tol: float = Field(ge=1e-6, le=1e-2)
      n_random_inits: int = Field(ge=5, le=50)
      random_seed: int

  class BlackScholesConfig(BaseModel):
      model_type: Literal["black76", "bsm"]
      rate_source: Literal["constant", "fred_dgs10", "curve"]
      constant_rate: float = Field(ge=0.0, le=0.10)
      volatility_method: Literal["historical", "garch", "realized", "implied", "hybrid"]
      hist_vol_window: int = Field(ge=20, le=252)

  # ... additional model configs
  ```

- [ ] **Create parameter documentation (PARAMETERS.md)**
  - [ ] Table format with: Name, Type, Range, Default, Impact, Sensitivity
  - [ ] Cross-reference with degrees of freedom
  - [ ] Examples of typical configurations for different commodities

### 1.3 Logging and Monitoring Setup

- [ ] **Configure structured logging**
  ```python
  # src/config/logging_config.py
  import logging

  def setup_logging(config):
      logging.basicConfig(
          level=getattr(logging, config['logging']['level']),
          format=config['logging']['format'],
          handlers=[
              logging.FileHandler(config['logging']['file']),
              logging.StreamHandler()
          ]
      )
      return logging.getLogger(__name__)
  ```

- [ ] **Error taxonomy implementation** (from CLAUDE.md)
  - [ ] TFM1001 CONFIG: bad config/env/flags
  - [ ] TFM2001 DATA: bad shapes, missing columns, leakage risks
  - [ ] TFM3001 CHECKPOINT: missing or incompatible checkpoint
  - [ ] TFM4001 INFERENCE: runtime/OOM/NaN/precision issues
  - [ ] TFM5001 PERF: regression or unexpected slowness

- [ ] **Create BUGS.md, PROGRESS.md, DECISIONS.md** (per CLAUDE.md)

---

## PHASE 2: DATA ACQUISITION & PREPROCESSING PIPELINE

### 2.1 Data Acquisition Module

- [ ] **Implement multi-source data client**
  ```python
  # src/data/acquisition.py

  class CommodityDataAcquisition:
      def __init__(self, config):
          self.config = config
          self.sources = self._init_sources()

      def _init_sources(self):
          """Initialize API clients based on config priority"""
          # yfinance, alpha_vantage, quandl, etc.
          pass

      def fetch_commodity_prices(self, ticker, start, end):
          """Fetch OHLCV data with fallback sources"""
          pass

      def fetch_fred_data(self, series_id, start, end):
          """Fetch macroeconomic data from FRED"""
          pass

      def fetch_futures_curve(self, ticker):
          """Fetch futures curve for convenience yield estimation"""
          pass
  ```

- [ ] **Implement rate limiting and retry logic**
  - [ ] Exponential backoff for API failures
  - [ ] Respect rate limits (yfinance: 2000/hour, Alpha Vantage: 25/day free)
  - [ ] Cache responses locally to minimize API calls

- [ ] **Data validation on acquisition**
  - [ ] Check for missing dates
  - [ ] Validate OHLCV relationships (High >= Low, Close within range)
  - [ ] Flag suspicious price movements (>5σ)
  - [ ] Timestamp integrity checks

### 2.2 Feature Engineering Pipeline

- [ ] **Technical indicators module**
  ```python
  # src/data/features.py

  import ta
  import pandas as pd

  class FeatureEngineer:
      def __init__(self, config):
          self.config = config

      def engineer_technical_indicators(self, df):
          """Add technical indicators from config"""
          # Use ta library
          df = ta.add_all_ta_features(
              df, open="open", high="high", low="low",
              close="close", volume="volume", fillna=True
          )
          return df

      def add_lagged_features(self, df, n_lags):
          """Add lagged values of key features"""
          for lag in range(1, n_lags + 1):
              df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
          return df

      def add_macroeconomic_features(self, price_df, fred_df):
          """Merge FRED data with price data"""
          # Align on dates, forward fill
          pass

      def create_regime_features(self, df, hmm_states):
          """One-hot encode HMM regime states as features"""
          pass
  ```

- [ ] **Implement feature selection**
  - [ ] Correlation analysis (remove highly correlated features >0.95)
  - [ ] Variance threshold (remove near-zero variance features)
  - [ ] Optional: Use TuneTA for automated indicator optimization
  - [ ] Feature importance from Random Forest

- [ ] **Prevent data leakage**
  - [ ] Audit: ensure all features use only past information
  - [ ] Validate: no .shift(-n) operations
  - [ ] Document: clear cut-off dates for train/test

### 2.3 Preprocessing Pipeline

- [ ] **Missing data handler**
  ```python
  # src/data/preprocessing.py

  def handle_missing_data(df, method='ffill'):
      if method == 'ffill':
          return df.fillna(method='ffill')
      elif method == 'interpolate':
          return df.interpolate(method='linear')
      elif method == 'drop':
          return df.dropna()
      else:
          raise ValueError(f"Unknown method: {method}")
  ```

- [ ] **Outlier detection and treatment**
  ```python
  def detect_outliers_zscore(series, threshold=3.0):
      z_scores = np.abs(stats.zscore(series.dropna()))
      return z_scores > threshold

  def detect_outliers_iqr(series):
      Q1, Q3 = series.quantile([0.25, 0.75])
      IQR = Q3 - Q1
      lower = Q1 - 1.5 * IQR
      upper = Q3 + 1.5 * IQR
      return (series < lower) | (series > upper)

  def detect_outliers_isolation_forest(df, contamination=0.1):
      from sklearn.ensemble import IsolationForest
      iso = IsolationForest(contamination=contamination, random_state=42)
      outliers = iso.fit_predict(df)
      return outliers == -1
  ```

- [ ] **Stationarity testing and transformation**
  ```python
  from statsmodels.tsa.stattools import adfuller

  def test_stationarity(series, alpha=0.05):
      """Augmented Dickey-Fuller test"""
      result = adfuller(series.dropna())
      adf_stat, p_value = result[0], result[1]
      is_stationary = p_value < alpha
      return is_stationary, p_value

  def transform_to_stationary(series, method='log_returns'):
      if method == 'log_returns':
          return np.log(series / series.shift(1))
      elif method == 'returns':
          return series.pct_change()
      elif method == 'diff':
          return series.diff()
      else:
          return series
  ```

- [ ] **Scaling and normalization**
  ```python
  from sklearn.preprocessing import StandardScaler, RobustScaler

  def scale_features(df, method='standard'):
      if method == 'standard':
          scaler = StandardScaler()
      elif method == 'robust':
          scaler = RobustScaler()
      else:
          raise ValueError(f"Unknown scaling method: {method}")

      scaled = scaler.fit_transform(df)
      return scaled, scaler
  ```

- [ ] **Data pipeline integration test**
  - [ ] Unit test: each preprocessing step
  - [ ] Integration test: full pipeline end-to-end
  - [ ] Validate: output shapes, no NaNs, correct dtypes

---

## PHASE 3: HIDDEN MARKOV MODEL DEVELOPMENT

### 3.1 HMM Core Implementation

- [ ] **Wrapper class for hmmlearn**
  ```python
  # src/models/hmm_core.py

  from hmmlearn import hmm
  import numpy as np

  class CommodityHMM:
      def __init__(self, config):
          self.config = config
          self.n_states = config['hmm']['n_states']
          self.model = None
          self.scaler = None
          self.regime_stats = {}

      def fit_with_multiple_inits(self, features, n_inits=None):
          """Fit HMM with multiple random initializations"""
          n_inits = n_inits or self.config['hmm']['n_random_inits']

          # Scale features
          from sklearn.preprocessing import StandardScaler
          self.scaler = StandardScaler()
          features_scaled = self.scaler.fit_transform(features)

          best_score = -np.inf
          best_model = None

          for seed in range(n_inits):
              model = hmm.GaussianHMM(
                  n_components=self.n_states,
                  covariance_type=self.config['hmm']['covariance_type'],
                  n_iter=self.config['hmm']['n_iter'],
                  tol=self.config['hmm']['tol'],
                  random_state=seed + self.config['hmm']['random_seed']
              )

              try:
                  model.fit(features_scaled)
                  score = model.score(features_scaled)

                  if score > best_score:
                      best_score = score
                      best_model = model
              except Exception as e:
                  logging.warning(f"TFM4001 INFERENCE: HMM fit failed for seed {seed}: {e}")
                  continue

          if best_model is None:
              raise RuntimeError("TFM4001 INFERENCE: All HMM initializations failed")

          self.model = best_model
          self._analyze_regimes(features_scaled, features)
          return self

      def _analyze_regimes(self, features_scaled, features_raw):
          """Analyze and label regimes"""
          states = self.model.predict(features_scaled)

          for state in range(self.n_states):
              state_mask = (states == state)
              state_returns = features_raw[state_mask, 0]  # Assuming returns first

              self.regime_stats[state] = {
                  'mean_return': np.mean(state_returns),
                  'volatility': np.std(state_returns),
                  'sharpe': np.mean(state_returns) / np.std(state_returns) if np.std(state_returns) > 0 else 0,
                  'count': np.sum(state_mask),
                  'persistence': self.model.transmat_[state, state]
              }

          self._label_regimes()

      def _label_regimes(self):
          """Label regimes as bull/bear/neutral"""
          returns = [stats['mean_return'] for stats in self.regime_stats.values()]
          sorted_states = sorted(range(self.n_states), key=lambda x: returns[x])

          labels = {}
          if self.n_states == 2:
              labels[sorted_states[0]] = 'bear'
              labels[sorted_states[1]] = 'bull'
          elif self.n_states == 3:
              labels[sorted_states[0]] = 'bear'
              labels[sorted_states[1]] = 'neutral'
              labels[sorted_states[2]] = 'bull'
          else:
              for i, state in enumerate(sorted_states):
                  labels[state] = f'regime_{i}'

          for state in range(self.n_states):
              self.regime_stats[state]['label'] = labels.get(state, f'state_{state}')

      def predict_regime(self, features):
          """Predict current regime"""
          features_scaled = self.scaler.transform(features)
          state = self.model.predict(features_scaled)[-1]
          return state, self.regime_stats[state]['label']

      def predict_proba(self, features):
          """Posterior probabilities for each regime"""
          features_scaled = self.scaler.transform(features)
          return self.model.predict_proba(features_scaled)

      def forecast_spot_price(self, current_features, horizon=1):
          """Forecast spot price using regime-based expectations"""
          # Get current regime
          current_state, _ = self.predict_regime(current_features[-1:])

          # Expected return in current regime
          expected_return = self.regime_stats[current_state]['mean_return']
          expected_vol = self.regime_stats[current_state]['volatility']

          # Monte Carlo simulation
          n_simulations = 10000
          current_price = current_features[-1, 0]  # Assuming price is first feature

          simulated_prices = []
          for _ in range(n_simulations):
              # Geometric Brownian Motion with regime parameters
              dt = 1/252  # Daily
              drift = expected_return * dt
              diffusion = expected_vol * np.sqrt(dt) * np.random.randn()
              future_price = current_price * np.exp(drift + diffusion)
              simulated_prices.append(future_price)

          # Return mean and quantiles
          forecast_mean = np.mean(simulated_prices)
          forecast_std = np.std(simulated_prices)
          forecast_quantiles = np.quantile(simulated_prices, [0.05, 0.25, 0.5, 0.75, 0.95])

          return {
              'mean': forecast_mean,
              'std': forecast_std,
              'quantiles': forecast_quantiles,
              'regime': self.regime_stats[current_state]['label'],
              'regime_persistence': self.regime_stats[current_state]['persistence']
          }

      def get_transition_matrix(self):
          """Return state transition matrix"""
          return self.model.transmat_

      def get_emission_params(self):
          """Return emission distribution parameters"""
          return {
              'means': self.model.means_,
              'covars': self.model.covars_
          }
  ```

- [ ] **Model selection via AIC/BIC**
  ```python
  def select_optimal_states(features, n_states_range=range(2, 7)):
      """Select optimal number of states using AIC/BIC"""
      aic_scores = []
      bic_scores = []
      models = []

      for n_states in n_states_range:
          hmm_model = CommodityHMM(config_with_n_states(n_states))
          hmm_model.fit_with_multiple_inits(features, n_inits=5)

          # Calculate AIC/BIC
          log_likelihood = hmm_model.model.score(features) * len(features)
          n_params = (n_states - 1) + n_states * (n_states - 1) + \
                     n_states * features.shape[1] * 2

          aic = 2 * n_params - 2 * log_likelihood
          bic = np.log(len(features)) * n_params - 2 * log_likelihood

          aic_scores.append(aic)
          bic_scores.append(bic)
          models.append(hmm_model)

      # Select model with lowest BIC (stronger penalty)
      best_idx = np.argmin(bic_scores)
      return models[best_idx], n_states_range[best_idx]
  ```

### 3.2 Regime Analysis Tools

- [ ] **Regime visualization**
  ```python
  def plot_regime_analysis(hmm_model, price_series, states):
      """Visualize regimes over time with prices"""
      import plotly.graph_objects as go

      fig = go.Figure()

      # Add price line
      fig.add_trace(go.Scatter(x=price_series.index, y=price_series.values,
                               mode='lines', name='Price'))

      # Color background by regime
      for state in range(hmm_model.n_states):
          state_mask = (states == state)
          label = hmm_model.regime_stats[state]['label']
          # Add regime bands
          # ... implementation

      fig.update_layout(title='Price with Regime Detection')
      return fig
  ```

- [ ] **Transition analysis**
  ```python
  def analyze_regime_transitions(hmm_model):
      """Analyze regime transition patterns"""
      trans_matrix = hmm_model.get_transition_matrix()

      # Persistence (diagonal)
      persistence = np.diag(trans_matrix)

      # Expected duration in each state
      expected_duration = 1 / (1 - persistence)

      # Asymmetry (easier to enter bear than exit?)
      asymmetry = {}
      for i in range(hmm_model.n_states):
          for j in range(hmm_model.n_states):
              if i != j:
                  asymmetry[f'{i}->{j}'] = trans_matrix[i,j] / trans_matrix[j,i]

      return {
          'persistence': persistence,
          'expected_duration': expected_duration,
          'asymmetry': asymmetry
      }
  ```

### 3.3 HMM Testing and Validation

- [ ] **Unit tests**
  - [ ] Test: fit converges for synthetic data
  - [ ] Test: predict returns correct state shape
  - [ ] Test: transition matrix is stochastic (rows sum to 1)
  - [ ] Test: regime labeling is consistent

- [ ] **Integration tests**
  - [ ] Test: full pipeline from raw data to regime prediction
  - [ ] Test: retraining with new data doesn't break
  - [ ] Test: save/load model state

---

## PHASE 4: BLACK-SCHOLES OPTIONS PRICING ENGINE

### 4.1 Volatility Estimation Module

- [ ] **Historical volatility**
  ```python
  # src/models/black_scholes.py

  def calculate_historical_volatility(returns, window=60):
      """Calculate annualized historical volatility"""
      vol = returns.rolling(window=window).std() * np.sqrt(252)
      return vol
  ```

- [ ] **GARCH volatility**
  ```python
  from arch import arch_model

  def calculate_garch_volatility(returns, p=1, q=1):
      """Fit GARCH(p,q) model for volatility forecasting"""
      # Remove NaN
      returns_clean = returns.dropna() * 100  # Scale for numerical stability

      # Fit GARCH
      model = arch_model(returns_clean, vol='Garch', p=p, q=q)
      results = model.fit(disp='off')

      # Forecast volatility
      forecast = results.forecast(horizon=1)
      forecast_var = forecast.variance.values[-1, 0]
      forecast_vol = np.sqrt(forecast_var) / 100 * np.sqrt(252)  # Annualize

      return forecast_vol, results
  ```

- [ ] **Realized volatility**
  ```python
  def calculate_realized_volatility(high, low, close, window=20):
      """Parkinson's volatility using high-low range"""
      hl_ratio = np.log(high / low)
      parkinson_vol = np.sqrt((1 / (4 * np.log(2))) * (hl_ratio ** 2))
      realized_vol = parkinson_vol.rolling(window=window).mean() * np.sqrt(252)
      return realized_vol
  ```

- [ ] **Hybrid volatility**
  ```python
  def calculate_hybrid_volatility(returns, high, low, close, weights=None):
      """Combine multiple volatility estimates"""
      if weights is None:
          weights = {'historical': 0.3, 'garch': 0.4, 'realized': 0.3}

      hist_vol = calculate_historical_volatility(returns)
      garch_vol, _ = calculate_garch_volatility(returns)
      realized_vol = calculate_realized_volatility(high, low, close)

      # Weighted average
      hybrid_vol = (weights['historical'] * hist_vol.iloc[-1] +
                    weights['garch'] * garch_vol +
                    weights['realized'] * realized_vol.iloc[-1])

      return hybrid_vol
  ```

### 4.2 Interest Rate Curve Construction

- [ ] **Fetch Treasury rates from FRED**
  ```python
  def fetch_treasury_curve(fred_client):
      """Fetch Treasury curve from FRED"""
      maturities = {
          'DGS1MO': 1/12,
          'DGS3MO': 0.25,
          'DGS6MO': 0.5,
          'DGS1': 1,
          'DGS2': 2,
          'DGS5': 5,
          'DGS10': 10,
          'DGS30': 30
      }

      rates = {}
      for series_id, maturity in maturities.items():
          data = fred_client.get_series(series_id)
          rates[maturity] = data.iloc[-1] / 100  # Convert to decimal

      return rates
  ```

- [ ] **Interpolate interest rates**
  ```python
  from scipy.interpolate import CubicSpline

  def interpolate_rate_curve(rates_dict, target_maturity):
      """Interpolate rate for any maturity"""
      maturities = np.array(sorted(rates_dict.keys()))
      rates = np.array([rates_dict[m] for m in maturities])

      cs = CubicSpline(maturities, rates)
      return cs(target_maturity)
  ```

### 4.3 Convenience Yield Estimation

- [ ] **Fitted convenience yield**
  ```python
  def calculate_convenience_yield(spot_price, futures_price, rate, time_to_maturity):
      """Back out convenience yield from futures curve"""
      # F = S * e^((r - y) * T)
      # y = r - (1/T) * ln(F/S)

      convenience_yield = rate - (1 / time_to_maturity) * np.log(futures_price / spot_price)
      return convenience_yield
  ```

- [ ] **Futures curve method**
  ```python
  def estimate_convenience_yield_from_curve(futures_curve, spot_price, rate_curve):
      """Average convenience yield across futures curve"""
      yields = []

      for maturity, futures_price in futures_curve.items():
          rate = interpolate_rate_curve(rate_curve, maturity)
          y = calculate_convenience_yield(spot_price, futures_price, rate, maturity)
          yields.append(y)

      return np.mean(yields)
  ```

### 4.4 Black-Scholes-Merton Implementation

- [ ] **Black-76 for futures options**
  ```python
  from scipy.stats import norm

  class Black76Pricer:
      def __init__(self, config):
          self.config = config

      def calculate_d1_d2(self, F, K, T, r, sigma):
          """Calculate d1 and d2"""
          d1 = (np.log(F / K) + (sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
          d2 = d1 - sigma * np.sqrt(T)
          return d1, d2

      def price_call(self, F, K, T, r, sigma):
          """Black-76 call option price"""
          d1, d2 = self.calculate_d1_d2(F, K, T, r, sigma)

          call_price = np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
          return call_price

      def price_put(self, F, K, T, r, sigma):
          """Black-76 put option price"""
          d1, d2 = self.calculate_d1_d2(F, K, T, r, sigma)

          put_price = np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
          return put_price

      def calculate_greeks(self, F, K, T, r, sigma, option_type='call'):
          """Calculate option Greeks"""
          d1, d2 = self.calculate_d1_d2(F, K, T, r, sigma)

          # Delta
          if option_type == 'call':
              delta = np.exp(-r * T) * norm.cdf(d1)
          else:
              delta = -np.exp(-r * T) * norm.cdf(-d1)

          # Gamma
          gamma = np.exp(-r * T) * norm.pdf(d1) / (F * sigma * np.sqrt(T))

          # Theta
          if option_type == 'call':
              theta = (-F * norm.pdf(d1) * sigma * np.exp(-r * T) / (2 * np.sqrt(T)) -
                       r * F * norm.cdf(d1) * np.exp(-r * T) +
                       r * K * norm.cdf(d2) * np.exp(-r * T))
          else:
              theta = (-F * norm.pdf(d1) * sigma * np.exp(-r * T) / (2 * np.sqrt(T)) +
                       r * F * norm.cdf(-d1) * np.exp(-r * T) -
                       r * K * norm.cdf(-d2) * np.exp(-r * T))

          # Vega
          vega = F * np.exp(-r * T) * norm.pdf(d1) * np.sqrt(T)

          # Rho
          if option_type == 'call':
              rho = -T * np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
          else:
              rho = -T * np.exp(-r * T) * (-F * norm.cdf(-d1) + K * norm.cdf(-d2))

          return {
              'delta': delta,
              'gamma': gamma,
              'theta': theta / 365,  # Daily theta
              'vega': vega / 100,    # Per 1% vol change
              'rho': rho / 100       # Per 1% rate change
          }
  ```

- [ ] **Volatility surface construction**
  ```python
  def build_volatility_surface(spot_price, maturities, strike_ratios,
                                market_prices_grid, pricer):
      """Construct implied volatility surface"""
      vol_surface = np.zeros((len(maturities), len(strike_ratios)))

      for i, T in enumerate(maturities):
          for j, strike_ratio in enumerate(strike_ratios):
              K = spot_price * strike_ratio
              market_price = market_prices_grid[i, j]

              # Use vollib for implied vol calculation
              from py_vollib.black.implied_volatility import implied_volatility

              try:
                  iv = implied_volatility(market_price, spot_price, K, T, 0.0, 'c')
                  vol_surface[i, j] = iv
              except:
                  vol_surface[i, j] = np.nan

      return vol_surface
  ```

### 4.5 Integration with HMM Forecasts

- [ ] **Regime-dependent pricing**
  ```python
  class RegimeAwareBlackScholes:
      def __init__(self, hmm_model, bs_pricer, config):
          self.hmm_model = hmm_model
          self.bs_pricer = bs_pricer
          self.config = config

      def price_with_regime(self, current_features, K, T, option_type='call'):
          """Price option using regime-dependent parameters"""
          # Get current regime
          current_state, regime_label = self.hmm_model.predict_regime(current_features[-1:])

          # Get regime statistics
          regime_stats = self.hmm_model.regime_stats[current_state]

          # Forecast spot price
          spot_forecast = self.hmm_model.forecast_spot_price(current_features, horizon=int(T*252))
          F = spot_forecast['mean']  # Expected futures price

          # Use regime volatility
          sigma = regime_stats['volatility'] * np.sqrt(252)  # Annualize

          # Interest rate (from config or FRED)
          r = self.config['black_scholes']['constant_rate']

          # Price option
          if option_type == 'call':
              price = self.bs_pricer.price_call(F, K, T, r, sigma)
          else:
              price = self.bs_pricer.price_put(F, K, T, r, sigma)

          # Calculate Greeks
          greeks = self.bs_pricer.calculate_greeks(F, K, T, r, sigma, option_type)

          return {
              'price': price,
              'greeks': greeks,
              'regime': regime_label,
              'spot_forecast': spot_forecast,
              'implied_vol': sigma,
              'futures_price': F
          }
  ```

### 4.6 Black-Scholes Testing

- [ ] **Numerical validation**
  - [ ] Test: Put-Call parity holds
  - [ ] Test: Greeks are correct (finite difference check)
  - [ ] Test: Option prices converge for known analytical cases
  - [ ] Test: Volatility smile is reasonable

- [ ] **Integration tests**
  - [ ] Test: HMM → Black-Scholes pipeline
  - [ ] Test: Regime changes affect option prices correctly

---

## PHASE 5: TIMESFM INTEGRATION (OPTIONAL BUT RECOMMENDED)

### 5.1 TimesFM Adapter Module

- [ ] **Create TimesFM wrapper**
  ```python
  # src/models/timesfm_adapter.py

  import timesfm
  import torch

  class TimesFMAdapter:
      def __init__(self, config):
          self.config = config
          self.enabled = config['timesfm']['enabled']

          if self.enabled:
              self._load_model()

      def _load_model(self):
          """Load TimesFM from Hugging Face"""
          backend = self.config['timesfm']['backend']
          checkpoint = self.config['timesfm']['checkpoint']

          if backend == 'torch':
              from timesfm import TimesFM_2p5_200M_torch
              self.model = TimesFM_2p5_200M_torch.from_pretrained(checkpoint)
              torch.set_float32_matmul_precision("high")
          elif backend == 'flax':
              from timesfm import TimesFM_2p5_200M_flax
              self.model = TimesFM_2p5_200M_flax.from_pretrained(checkpoint)
          else:
              raise ValueError(f"Unknown backend: {backend}")

          # Compile model
          forecast_config = timesfm.ForecastConfig(
              max_context=self.config['timesfm']['max_context'],
              max_horizon=self.config['timesfm']['max_horizon'],
              normalize_inputs=self.config['timesfm']['normalize_inputs'],
              use_continuous_quantile_head=self.config['timesfm']['use_continuous_quantile_head'],
              force_flip_invariance=self.config['timesfm']['force_flip_invariance'],
              infer_is_positive=self.config['timesfm']['infer_is_positive'],
              fix_quantile_crossing=self.config['timesfm']['fix_quantile_crossing']
          )
          self.model.compile(forecast_config)

      def forecast(self, inputs, horizon):
          """Generate TimesFM forecast"""
          if not self.enabled:
              return None

          point_forecast, quantile_forecast = self.model.forecast(
              horizon=horizon,
              inputs=inputs
          )
          return {
              'point': point_forecast,
              'quantiles': quantile_forecast
          }
  ```

### 5.2 Ensemble Architecture

- [ ] **Implement ensemble combiner**
  ```python
  # src/models/ensemble.py

  class EnsembleForecaster:
      def __init__(self, hmm_model, timesfm_adapter, config):
          self.hmm_model = hmm_model
          self.timesfm = timesfm_adapter
          self.config = config
          self.mode = config['timesfm']['integration_mode']

      def forecast(self, features, horizon):
          """Combine HMM and TimesFM forecasts"""
          if self.mode == 'ensemble':
              # Average predictions
              hmm_forecast = self.hmm_model.forecast_spot_price(features, horizon)
              timesfm_forecast = self.timesfm.forecast(features[:, 0], horizon)

              combined_mean = 0.5 * hmm_forecast['mean'] + 0.5 * timesfm_forecast['point']

              return {
                  'mean': combined_mean,
                  'hmm': hmm_forecast,
                  'timesfm': timesfm_forecast
              }

          elif self.mode == 'primary':
              # Use TimesFM as primary, HMM for regime only
              regime_state, regime_label = self.hmm_model.predict_regime(features[-1:])
              timesfm_forecast = self.timesfm.forecast(features[:, 0], horizon)

              return {
                  'mean': timesfm_forecast['point'],
                  'regime': regime_label,
                  'timesfm': timesfm_forecast
              }

          elif self.mode == 'regime_input':
              # Feed HMM states as features to TimesFM
              states = self.hmm_model.model.predict(features)
              features_with_regime = np.column_stack([features, states])

              # Note: TimesFM expects specific input format, may need adaptation
              timesfm_forecast = self.timesfm.forecast(features_with_regime, horizon)

              return {
                  'mean': timesfm_forecast['point'],
                  'timesfm': timesfm_forecast
              }
  ```

### 5.3 TimesFM Validation

- [ ] **Compare TimesFM vs HMM performance**
  - [ ] Metric: RMSE, MAE, Directional Accuracy
  - [ ] Visualize: Forecast comparison plots
  - [ ] Analyze: When does each model perform better?

---

## PHASE 6: EVALUATION & VALIDATION FRAMEWORK

### 6.1 Metrics Module

- [ ] **Implement forecast accuracy metrics**
  ```python
  # src/evaluation/metrics.py

  import numpy as np

  def calculate_rmse(y_true, y_pred):
      """Root Mean Square Error"""
      return np.sqrt(np.mean((y_true - y_pred) ** 2))

  def calculate_mae(y_true, y_pred):
      """Mean Absolute Error"""
      return np.mean(np.abs(y_true - y_pred))

  def calculate_mape(y_true, y_pred):
      """Mean Absolute Percentage Error"""
      return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

  def calculate_directional_accuracy(y_true, y_pred):
      """Percentage of correct direction predictions"""
      direction_true = np.sign(np.diff(y_true))
      direction_pred = np.sign(np.diff(y_pred))
      return np.mean(direction_true == direction_pred) * 100
  ```

- [ ] **Implement financial metrics**
  ```python
  def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
      """Sharpe Ratio"""
      excess_returns = returns - risk_free_rate / 252  # Daily
      return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)

  def calculate_max_drawdown(equity_curve):
      """Maximum Drawdown"""
      running_max = np.maximum.accumulate(equity_curve)
      drawdown = (equity_curve - running_max) / running_max
      return np.min(drawdown) * 100  # Percentage

  def calculate_calmar_ratio(returns, equity_curve):
      """Calmar Ratio"""
      annual_return = (1 + returns.mean()) ** 252 - 1
      max_dd = abs(calculate_max_drawdown(equity_curve))
      return annual_return / (max_dd / 100)
  ```

### 6.2 Walk-Forward Validation

- [ ] **Implement walk-forward backtester**
  ```python
  # src/evaluation/backtester.py

  class WalkForwardValidator:
      def __init__(self, config):
          self.config = config
          self.train_window = config['validation']['train_window_days']
          self.test_window = config['validation']['test_window_days']
          self.step_size = config['validation']['step_size_days']

      def validate(self, data, model_factory):
          """Perform walk-forward validation"""
          results = []

          n_samples = len(data)
          start = self.train_window

          while start + self.test_window < n_samples:
              # Split data
              train_data = data[start - self.train_window:start]
              test_data = data[start:start + self.test_window]

              # Train model
              model = model_factory()
              model.fit(train_data)

              # Predict
              predictions = model.predict(test_data)

              # Evaluate
              metrics = self._calculate_metrics(test_data, predictions)

              results.append({
                  'train_end': start,
                  'test_start': start,
                  'test_end': start + self.test_window,
                  'metrics': metrics,
                  'predictions': predictions
              })

              # Step forward
              start += self.step_size

          return results

      def _calculate_metrics(self, actual, predicted):
          """Calculate all configured metrics"""
          metrics = {}

          if 'rmse' in self.config['validation']['metrics']:
              metrics['rmse'] = calculate_rmse(actual, predicted)

          if 'mae' in self.config['validation']['metrics']:
              metrics['mae'] = calculate_mae(actual, predicted)

          if 'mape' in self.config['validation']['metrics']:
              metrics['mape'] = calculate_mape(actual, predicted)

          if 'dstat' in self.config['validation']['metrics']:
              metrics['dstat'] = calculate_directional_accuracy(actual, predicted)

          return metrics
  ```

### 6.3 Backtesting with Trading Strategy

- [ ] **Implement vectorbt backtester**
  ```python
  import vectorbt as vbt

  class TradingBacktester:
      def __init__(self, config):
          self.config = config
          self.initial_capital = config['validation']['backtesting']['initial_capital']
          self.transaction_cost = config['validation']['backtesting']['transaction_cost_bps'] / 10000

      def backtest_strategy(self, price_data, signals):
          """Backtest trading strategy using vectorbt"""
          # Convert signals to entries/exits
          entries = signals > 0
          exits = signals < 0

          # Run backtest
          portfolio = vbt.Portfolio.from_signals(
              price_data,
              entries,
              exits,
              init_cash=self.initial_capital,
              fees=self.transaction_cost
          )

          # Extract metrics
          results = {
              'total_return': portfolio.total_return(),
              'sharpe_ratio': portfolio.sharpe_ratio(),
              'max_drawdown': portfolio.max_drawdown(),
              'calmar_ratio': portfolio.calmar_ratio(),
              'win_rate': portfolio.trades.win_rate(),
              'trades': portfolio.trades.count()
          }

          return results, portfolio
  ```

### 6.4 Statistical Significance Testing

- [ ] **Implement Diebold-Mariano test**
  ```python
  from scipy import stats

  def diebold_mariano_test(errors1, errors2, h=1):
      """
      Test if forecast 1 is significantly better than forecast 2
      H0: Both forecasts have same accuracy
      """
      # Loss differential
      d = errors1 ** 2 - errors2 ** 2

      # Mean loss differential
      d_mean = np.mean(d)

      # Variance with autocorrelation adjustment
      gamma0 = np.var(d)
      if h > 1:
          gamma = [np.cov(d[:-i], d[i:])[0, 1] for i in range(1, h)]
          var_d = gamma0 + 2 * sum(gamma)
      else:
          var_d = gamma0

      # DM statistic
      dm_stat = d_mean / np.sqrt(var_d / len(d))

      # P-value (two-tailed)
      p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

      return dm_stat, p_value
  ```

- [ ] **Implement Model Confidence Set**
  ```python
  # Note: Use R package 'MCS' via rpy2, or implement simplified version
  def model_confidence_set(losses_matrix, alpha=0.10):
      """
      Construct Model Confidence Set
      losses_matrix: (n_samples, n_models)
      Returns: indices of models in MCS
      """
      # Simplified implementation - full version requires more complexity
      # For production, recommend using R's MCS package via rpy2
      pass
  ```

### 6.5 Visualization Module

- [ ] **Forecast visualization**
  ```python
  # src/evaluation/visualizations.py

  import plotly.graph_objects as go

  def plot_forecast_with_intervals(dates, actual, forecast, confidence_intervals):
      """Plot forecast with confidence bands"""
      fig = go.Figure()

      # Actual values
      fig.add_trace(go.Scatter(x=dates, y=actual, mode='lines',
                               name='Actual', line=dict(color='black')))

      # Forecast
      fig.add_trace(go.Scatter(x=dates, y=forecast, mode='lines',
                               name='Forecast', line=dict(color='blue', dash='dash')))

      # Confidence intervals
      fig.add_trace(go.Scatter(
          x=np.concatenate([dates, dates[::-1]]),
          y=np.concatenate([confidence_intervals['upper'], confidence_intervals['lower'][::-1]]),
          fill='toself',
          fillcolor='rgba(0,100,255,0.2)',
          line=dict(color='rgba(255,255,255,0)'),
          name='95% CI'
      ))

      fig.update_layout(
          title='Commodity Price Forecast',
          xaxis_title='Date',
          yaxis_title='Price',
          hovermode='x unified'
      )

      return fig

  def plot_regime_performance(regime_stats):
      """Bar chart of regime characteristics"""
      # Implementation
      pass

  def plot_options_surface(maturities, strikes, vol_surface):
      """3D plot of implied volatility surface"""
      # Implementation using plotly
      pass

  def plot_greeks_heatmap(greeks_matrix, strikes, maturities):
      """Heatmap of Greeks across strike and maturity"""
      # Implementation
      pass
  ```

---

## PHASE 7: HYPERPARAMETER OPTIMIZATION & SENSITIVITY ANALYSIS

### 7.1 Optuna Integration

- [ ] **Create optimization framework**
  ```python
  # src/optimization/hyperparameter.py

  import optuna

  class HyperparameterOptimizer:
      def __init__(self, config):
          self.config = config
          self.study = None

      def optimize(self, objective_fn, n_trials=100):
          """Run Optuna optimization"""
          # Create study
          self.study = optuna.create_study(
              direction='minimize',  # Minimize RMSE
              sampler=optuna.samplers.TPESampler(seed=42),
              pruner=optuna.pruners.MedianPruner()
          )

          # Optimize
          self.study.optimize(objective_fn, n_trials=n_trials, timeout=3600)

          # Best parameters
          best_params = self.study.best_params
          best_value = self.study.best_value

          return best_params, best_value

      def plot_optimization_history(self):
          """Visualize optimization process"""
          fig = optuna.visualization.plot_optimization_history(self.study)
          return fig

      def plot_param_importances(self):
          """Parameter importance plot"""
          fig = optuna.visualization.plot_param_importances(self.study)
          return fig
  ```

- [ ] **Define objective function**
  ```python
  def create_objective(data, config):
      """Create Optuna objective function"""
      def objective(trial):
          # Sample hyperparameters
          n_states = trial.suggest_int('n_states', 2, 6)
          covariance_type = trial.suggest_categorical('covariance_type',
                                                       ['diag', 'full', 'spherical'])
          n_iter = trial.suggest_int('n_iter', 500, 2000)

          # Update config
          config_trial = config.copy()
          config_trial['hmm']['n_states'] = n_states
          config_trial['hmm']['covariance_type'] = covariance_type
          config_trial['hmm']['n_iter'] = n_iter

          # Train model
          model = CommodityHMM(config_trial)
          model.fit_with_multiple_inits(data['features'], n_inits=5)

          # Validate
          predictions = model.forecast_spot_price(data['features'], horizon=30)
          rmse = calculate_rmse(data['actual'], predictions)

          return rmse

      return objective
  ```

### 7.2 Sensitivity Analysis

- [ ] **Implement one-at-a-time sensitivity**
  ```python
  # src/optimization/sensitivity.py

  class SensitivityAnalyzer:
      def __init__(self, config, baseline_model):
          self.config = config
          self.baseline_model = baseline_model
          self.baseline_performance = None

      def analyze(self, data, parameters, perturbation_pct=0.1):
          """Perform sensitivity analysis on specified parameters"""
          results = {}

          # Baseline performance
          self.baseline_performance = self._evaluate(self.baseline_model, data)

          for param in parameters:
              param_results = self._perturb_parameter(param, data, perturbation_pct)
              results[param] = param_results

          return results

      def _perturb_parameter(self, param, data, pct):
          """Perturb parameter and measure impact"""
          # Get baseline value
          param_path = param.split('.')
          baseline_value = self._get_nested_param(self.config, param_path)

          # Perturb up and down
          perturbations = {
              'up': baseline_value * (1 + pct),
              'down': baseline_value * (1 - pct)
          }

          results = {'baseline': self.baseline_performance}

          for direction, new_value in perturbations.items():
              # Update config
              config_perturbed = self.config.copy()
              self._set_nested_param(config_perturbed, param_path, new_value)

              # Retrain model
              model = CommodityHMM(config_perturbed)
              model.fit_with_multiple_inits(data['features'], n_inits=5)

              # Evaluate
              performance = self._evaluate(model, data)
              results[direction] = performance

          # Calculate sensitivity
          sensitivity = {
              'absolute': (results['up'] - results['down']) / 2,
              'relative': ((results['up'] - results['down']) / 2) / self.baseline_performance
          }

          results['sensitivity'] = sensitivity
          return results

      def _evaluate(self, model, data):
          """Evaluate model performance"""
          predictions = model.forecast_spot_price(data['features'], horizon=30)
          rmse = calculate_rmse(data['actual'], predictions)
          return rmse

      def plot_sensitivity(self, results):
          """Tornado plot of parameter sensitivities"""
          # Implementation
          pass
  ```

### 7.3 Optuna Visualization and Reporting

- [ ] **Generate optimization report**
  ```python
  def generate_optimization_report(study, output_path='outputs/reports/optimization.html'):
      """Generate comprehensive optimization report"""
      from optuna.visualization import (
          plot_optimization_history,
          plot_param_importances,
          plot_parallel_coordinate,
          plot_slice
      )

      # Create HTML report
      html_content = f"""
      <html>
      <head><title>Hyperparameter Optimization Report</title></head>
      <body>
          <h1>Optimization Results</h1>
          <p>Best Value: {study.best_value:.4f}</p>
          <p>Best Parameters: {study.best_params}</p>

          <h2>Optimization History</h2>
          {plot_optimization_history(study).to_html()}

          <h2>Parameter Importances</h2>
          {plot_param_importances(study).to_html()}

          <h2>Parallel Coordinate Plot</h2>
          {plot_parallel_coordinate(study).to_html()}
      </body>
      </html>
      """

      with open(output_path, 'w') as f:
          f.write(html_content)
  ```

---

## PHASE 8: USER INTERFACE DEVELOPMENT

### 8.1 Streamlit Dashboard

- [ ] **Main dashboard layout**
  ```python
  # src/ui/streamlit_app.py

  import streamlit as st
  import sys
  sys.path.append('../')

  from src.config.loader import load_config
  from src.models.hmm_core import CommodityHMM
  from src.models.black_scholes import Black76Pricer, RegimeAwareBlackScholes

  # Page config
  st.set_page_config(
      page_title="Commodity Forecasting System",
      page_icon="📈",
      layout="wide",
      initial_sidebar_state="expanded"
  )

  # Load config
  config = load_config('config/parameters.toml')

  # Sidebar - Parameter Configuration
  st.sidebar.title("⚙️ Configuration")

  st.sidebar.header("Data Settings")
  ticker = st.sidebar.selectbox("Commodity", ["GC=F", "CL=F", "SI=F", "NG=F"],
                                 index=0)
  date_range = st.sidebar.date_input("Date Range",
                                      value=(config['data']['start_date'],
                                             config['data']['end_date']))

  st.sidebar.header("HMM Settings")
  n_states = st.sidebar.slider("Number of States", 2, 6,
                                config['hmm']['n_states'])
  covariance_type = st.sidebar.selectbox("Covariance Type",
                                          ["diag", "full", "spherical", "tied"],
                                          index=0)
  n_iter = st.sidebar.slider("Max Iterations", 100, 5000,
                               config['hmm']['n_iter'], step=100)

  st.sidebar.header("Black-Scholes Settings")
  volatility_method = st.sidebar.selectbox("Volatility Method",
                                            ["historical", "garch", "realized", "hybrid"],
                                            index=3)

  # Update config with UI values
  config['commodity']['ticker'] = ticker
  config['hmm']['n_states'] = n_states
  config['hmm']['covariance_type'] = covariance_type
  config['hmm']['n_iter'] = n_iter
  config['black_scholes']['volatility_method'] = volatility_method

  # Main content
  st.title("📈 Commodity Forecasting System")
  st.markdown("**HMM-Based Spot Price Prediction + Black-Scholes Futures Pricing**")

  # Tabs
  tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔮 Forecasts", "📐 Options Pricing", "📈 Performance"])

  with tab1:
      st.header("System Overview")

      col1, col2, col3 = st.columns(3)

      with col1:
          st.metric("Commodity", config['commodity']['name'])
          st.metric("Current Price", "$1,925.50", "+$15.30")

      with col2:
          st.metric("Current Regime", "Bull", "")
          st.metric("Regime Confidence", "87%", "+5%")

      with col3:
          st.metric("30-Day Forecast", "$1,985.20", "+3.1%")
          st.metric("Implied Volatility", "18.5%", "-0.8%")

      # Price chart with regimes
      st.subheader("Price History with Regime Detection")
      # ... plot implementation

  with tab2:
      st.header("Spot Price Forecasts")

      horizon = st.slider("Forecast Horizon (days)", 1, 365, 30)

      if st.button("Generate Forecast"):
          with st.spinner("Running HMM forecast..."):
              # Load data and train model
              # ... implementation

              st.success("Forecast complete!")

              # Display forecast plot
              st.plotly_chart(forecast_plot, use_container_width=True)

              # Forecast statistics
              col1, col2 = st.columns(2)
              with col1:
                  st.metric("Mean Forecast", "$1,985.20")
                  st.metric("Std Deviation", "$45.30")
              with col2:
                  st.metric("5th Percentile", "$1,905.10")
                  st.metric("95th Percentile", "$2,065.30")

  with tab3:
      st.header("Options Pricing (Black-Scholes)")

      col1, col2 = st.columns(2)

      with col1:
          option_type = st.radio("Option Type", ["Call", "Put"])
          strike_price = st.number_input("Strike Price", value=2000.0)
          time_to_maturity = st.number_input("Time to Maturity (days)", value=30)

      with col2:
          risk_free_rate = st.number_input("Risk-Free Rate (%)", value=3.5)
          volatility = st.number_input("Volatility (%)", value=18.5)

      if st.button("Calculate Option Price"):
          # Price option using Black-Scholes
          # ... implementation

          st.subheader("Option Price")
          st.metric("Premium", "$87.50")

          st.subheader("Greeks")
          col1, col2, col3 = st.columns(3)
          with col1:
              st.metric("Delta", "0.65")
              st.metric("Gamma", "0.003")
          with col2:
              st.metric("Theta", "-0.15")
              st.metric("Vega", "1.25")
          with col3:
              st.metric("Rho", "0.45")

          # Volatility surface
          st.subheader("Implied Volatility Surface")
          st.plotly_chart(vol_surface_plot, use_container_width=True)

  with tab4:
      st.header("Model Performance")

      # Metrics table
      st.subheader("Forecast Accuracy Metrics")
      metrics_df = pd.DataFrame({
          'Metric': ['RMSE', 'MAE', 'MAPE', 'Directional Accuracy'],
          'Value': [25.3, 18.7, 2.1, 68.5],
          'Unit': ['$', '$', '%', '%']
      })
      st.dataframe(metrics_df, use_container_width=True)

      # Financial metrics
      st.subheader("Trading Performance")
      fin_metrics_df = pd.DataFrame({
          'Metric': ['Sharpe Ratio', 'Max Drawdown', 'Calmar Ratio', 'Win Rate'],
          'Value': [1.85, -12.5, 2.40, 62.3],
          'Unit': ['', '%', '', '%']
      })
      st.dataframe(fin_metrics_df, use_container_width=True)

      # Backtest equity curve
      st.subheader("Backtest Equity Curve")
      st.plotly_chart(equity_curve_plot, use_container_width=True)

  # Footer
  st.markdown("---")
  st.markdown("**Version**: 1.0.0 | **Last Updated**: 2026-01-16")
  ```

- [ ] **Component modules**
  ```python
  # src/ui/components/forecast_plot.py

  def create_forecast_plot(dates, actual, forecast, confidence_intervals):
      """Reusable forecast plot component"""
      # ... implementation from visualizations.py
      pass
  ```

### 8.2 Gradio Demo Interface

- [ ] **Quick demo for model testing**
  ```python
  # src/ui/gradio_demo.py

  import gradio as gr

  def forecast_commodity(ticker, horizon, n_states):
      """Simple forecasting interface"""
      # Load model and generate forecast
      # ... implementation

      return forecast_plot, metrics_text

  # Create interface
  demo = gr.Interface(
      fn=forecast_commodity,
      inputs=[
          gr.Dropdown(["GC=F", "CL=F", "SI=F"], label="Commodity"),
          gr.Slider(1, 365, value=30, label="Forecast Horizon (days)"),
          gr.Slider(2, 6, value=3, step=1, label="HMM States")
      ],
      outputs=[
          gr.Plot(label="Forecast"),
          gr.Textbox(label="Metrics")
      ],
      title="Commodity Price Forecasting Demo",
      description="Quick demo of HMM-based commodity forecasting"
  )

  if __name__ == "__main__":
      demo.launch()
  ```

### 8.3 Real-Time Mode (Optional)

- [ ] **WebSocket integration for live data**
  ```python
  # For production real-time forecasting
  # Requires WebSocket API (Polygon.io, Alpha Vantage, etc.)

  import asyncio
  import websockets

  async def stream_live_data():
      """Stream real-time commodity prices"""
      # Implementation depends on chosen data provider
      pass
  ```

---

## PHASE 9: DOCUMENTATION & DEPLOYMENT

### 9.1 Comprehensive Documentation

- [ ] **PARAMETERS.md - Complete parameter guide**
  ```markdown
  # Parameter Configuration Guide

  ## HMM Parameters

  ### n_states
  - **Type**: Integer
  - **Range**: [2, 6]
  - **Default**: 3
  - **Description**: Number of hidden market regimes
  - **Impact**:
    - More states → more granular regime detection
    - Risk of overfitting with too many states
    - 3 states (bull/bear/neutral) recommended for most commodities
  - **Sensitivity**: HIGH
  - **Optimal Range**: [2, 4] for most use cases

  ### covariance_type
  - **Type**: Categorical
  - **Options**: "diag", "full", "spherical", "tied"
  - **Default**: "diag"
  - **Description**: Structure of emission covariance matrices
  - **Impact**:
    - "diag": Independent variances per feature (recommended)
    - "full": Captures feature correlations (needs more data)
    - "spherical": Same variance for all features (strong regularization)
    - "tied": All states share covariance structure
  - **Sensitivity**: MEDIUM
  - **Guidelines**: Use "diag" unless you have >1000 observations and many features

  [... continue for all parameters ...]
  ```

- [ ] **API.md - Code API documentation**
  ```markdown
  # API Reference

  ## CommodityHMM

  ### Constructor
  \```python
  CommodityHMM(config: dict)
  \```

  ### Methods

  #### fit_with_multiple_inits
  \```python
  fit_with_multiple_inits(features: np.ndarray, n_inits: int = 10) -> CommodityHMM
  \```

  Fit HMM with multiple random initializations to avoid local optima.

  **Parameters:**
  - `features` (np.ndarray): Feature matrix, shape (n_samples, n_features)
  - `n_inits` (int): Number of random initializations

  **Returns:**
  - `self`: Fitted model

  [... continue for all classes and methods ...]
  ```

- [ ] **ARCHITECTURE.md - System design document**
  - [ ] Overview of system components
  - [ ] Data flow diagrams
  - [ ] Decision rationales
  - [ ] Integration points

- [ ] **DEPLOYMENT.md - Production deployment guide**
  - [ ] Server requirements
  - [ ] Docker containerization instructions
  - [ ] Environment setup
  - [ ] Monitoring and logging
  - [ ] Scaling considerations

### 9.2 Testing Suite

- [ ] **Unit tests (pytest)**
  ```bash
  # tests/test_models/test_hmm.py

  import pytest
  from src.models.hmm_core import CommodityHMM

  def test_hmm_fit_converges():
      """Test that HMM converges on synthetic data"""
      # Generate synthetic data
      # Fit model
      # Assert convergence
      pass

  def test_transition_matrix_stochastic():
      """Test that transition matrix rows sum to 1"""
      pass

  # ... more tests
  ```

- [ ] **Integration tests**
  - [ ] Test: Full pipeline from data acquisition to forecast
  - [ ] Test: Parameter loading and validation
  - [ ] Test: Model saving and loading

- [ ] **Performance tests**
  - [ ] Benchmark: Training time for different n_states
  - [ ] Benchmark: Inference latency
  - [ ] Memory profiling

### 9.3 Experiment Tracking Setup

- [ ] **MLflow initialization**
  ```python
  # Integrate MLflow tracking throughout codebase

  import mlflow

  mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
  mlflow.set_experiment(config['mlflow']['experiment_name'])

  with mlflow.start_run(run_name="hmm_training"):
      # Log parameters
      mlflow.log_params(config['hmm'])

      # Train model
      model.fit(data)

      # Log metrics
      mlflow.log_metrics({
          'rmse': rmse,
          'sharpe_ratio': sharpe
      })

      # Log model
      mlflow.sklearn.log_model(model, "hmm_model")
  ```

- [ ] **Model registry setup**
  - [ ] Register best models
  - [ ] Version control
  - [ ] Promotion workflow (staging → production)

### 9.4 Deployment Options

- [ ] **Option 1: Local deployment**
  ```bash
  # Run Streamlit locally
  streamlit run src/ui/streamlit_app.py --server.port 8501
  ```

- [ ] **Option 2: Docker containerization**
  ```dockerfile
  # Dockerfile
  FROM python:3.11-slim

  WORKDIR /app

  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt

  COPY . .

  EXPOSE 8501

  CMD ["streamlit", "run", "src/ui/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
  ```

- [ ] **Option 3: Cloud deployment (AWS/GCP/Azure)**
  - [ ] Set up cloud infrastructure
  - [ ] Configure autoscaling
  - [ ] Set up monitoring (CloudWatch, Stackdriver)

---

## PHASE 10: PRODUCTION READINESS & MAINTENANCE

### 10.1 Monitoring and Alerting

- [ ] **Model performance monitoring**
  ```python
  # Track model drift and degradation

  def monitor_model_performance(predictions, actuals, threshold=0.1):
      """Alert if performance degrades beyond threshold"""
      current_rmse = calculate_rmse(actuals, predictions)

      if current_rmse > baseline_rmse * (1 + threshold):
          send_alert(f"Model RMSE degraded: {current_rmse:.2f}")
  ```

- [ ] **Data quality monitoring**
  - [ ] Alert on missing data
  - [ ] Alert on distribution shifts
  - [ ] Alert on API failures

- [ ] **System health monitoring**
  - [ ] CPU/memory usage
  - [ ] API latency
  - [ ] Error rates

### 10.2 Retraining Pipeline

- [ ] **Automated retraining schedule**
  ```python
  # Retrain model monthly

  from apscheduler.schedulers.blocking import BlockingScheduler

  scheduler = BlockingScheduler()

  @scheduler.scheduled_job('cron', day=1, hour=0)  # First of month at midnight
  def retrain_model():
      """Automated retraining"""
      logging.info("Starting scheduled retraining")

      # Fetch latest data
      data = fetch_latest_data()

      # Retrain model
      model = CommodityHMM(config)
      model.fit_with_multiple_inits(data['features'])

      # Validate performance
      validation_results = validate_model(model, data)

      # If better, promote to production
      if validation_results['rmse'] < current_production_rmse:
          promote_to_production(model)
          logging.info(f"New model promoted. RMSE: {validation_results['rmse']:.2f}")
      else:
          logging.warning("New model did not outperform current production model")

  scheduler.start()
  ```

### 10.3 Logging and Debugging

- [ ] **Structured logging implementation**
  - [ ] Log all predictions with timestamps
  - [ ] Log parameter changes
  - [ ] Log errors with full context

- [ ] **Debug mode**
  ```python
  # Enable verbose logging for debugging

  if config['debug_mode']:
      logging.basicConfig(level=logging.DEBUG)
      # Save intermediate results
      # Generate debug plots
  ```

### 10.4 Security Considerations

- [ ] **API key management**
  - [ ] Store keys in environment variables, not code
  - [ ] Use secrets manager (AWS Secrets Manager, GCP Secret Manager)

- [ ] **Input validation**
  - [ ] Validate all user inputs
  - [ ] Sanitize before processing
  - [ ] Rate limiting on API endpoints

- [ ] **Access control**
  - [ ] Authentication for UI
  - [ ] Role-based access control

### 10.5 Maintenance Checklist

- [ ] **Weekly**
  - [ ] Review error logs
  - [ ] Check data pipeline health
  - [ ] Verify API quotas

- [ ] **Monthly**
  - [ ] Retrain models
  - [ ] Review performance metrics
  - [ ] Update documentation

- [ ] **Quarterly**
  - [ ] Dependency updates
  - [ ] Security audits
  - [ ] Performance optimization

---

## PHASE 11: ADVANCED FEATURES (FUTURE ENHANCEMENTS)

### 11.1 Multi-Commodity Portfolio

- [ ] **Extend to multiple commodities simultaneously**
  - [ ] Correlation analysis
  - [ ] Portfolio optimization
  - [ ] Cross-commodity regime detection

### 11.2 Advanced Hybrid Models

- [ ] **HMM + LSTM ensemble**
  - [ ] Train LSTM for each regime
  - [ ] Compare performance

- [ ] **HMM + Random Forest**
  - [ ] Feature importance analysis
  - [ ] Non-linear regime patterns

### 11.3 Real-Time Trading Integration

- [ ] **Broker API integration**
  - [ ] Interactive Brokers
  - [ ] Alpaca
  - [ ] Order execution

- [ ] **Risk management system**
  - [ ] Position sizing
  - [ ] Stop-loss automation
  - [ ] Portfolio rebalancing

### 11.4 Advanced Visualization

- [ ] **Augmented Reality/VR dashboards**
  - [ ] Immersive data exploration
  - [ ] 3D volatility surfaces

- [ ] **AI-driven anomaly detection**
  - [ ] Automatic pattern recognition
  - [ ] Predictive alerts

---

## FINAL CHECKLIST: PRE-LAUNCH VERIFICATION

### Code Quality
- [ ] All unit tests passing
- [ ] Integration tests passing
- [ ] Code coverage >80%
- [ ] Linting (ruff/black) clean
- [ ] Type hints added (mypy check)

### Documentation
- [ ] README.md complete with quickstart
- [ ] PARAMETERS.md comprehensive
- [ ] API.md up to date
- [ ] ARCHITECTURE.md finalized
- [ ] DEPLOYMENT.md ready
- [ ] BUGS.md, PROGRESS.md, DECISIONS.md maintained

### Validation
- [ ] Walk-forward validation complete
- [ ] Backtesting results documented
- [ ] Statistical significance tests performed
- [ ] Sensitivity analysis complete
- [ ] Performance benchmarks established

### Security
- [ ] No secrets in code
- [ ] Environment variables configured
- [ ] Input validation implemented
- [ ] Error handling robust

### UI/UX
- [ ] Streamlit dashboard functional
- [ ] Gradio demo working
- [ ] Responsive on different screen sizes
- [ ] Parameter tooltips clear
- [ ] Visualizations informative

### Deployment
- [ ] Local deployment tested
- [ ] Docker image built and tested
- [ ] Cloud deployment configured (if applicable)
- [ ] Monitoring and logging operational
- [ ] Backup and recovery plan in place

### Performance
- [ ] Training time acceptable (<10 min for typical dataset)
- [ ] Inference latency <1 second
- [ ] Memory usage within limits
- [ ] Can handle 10+ years of daily data

---

## SUCCESS CRITERIA

### Technical
- ✅ HMM correctly identifies 3 distinct market regimes
- ✅ Forecast RMSE < 5% of mean price
- ✅ Directional accuracy >60%
- ✅ Sharpe ratio >1.5 in backtests
- ✅ Black-Scholes Greeks pass finite difference validation
- ✅ System processes 5 years of data in <5 minutes

### User Experience
- ✅ Non-technical users can modify parameters via UI
- ✅ Parameter documentation is clear and comprehensive
- ✅ Visualizations are publication-quality
- ✅ System provides actionable insights
- ✅ Error messages are informative and actionable

### Production Readiness
- ✅ Can run continuously for 30+ days without intervention
- ✅ Automated retraining pipeline operational
- ✅ Monitoring alerts are working
- ✅ Model performance tracking in place
- ✅ Documentation is comprehensive enough for handoff

---

## ESTIMATED TIMELINE

**Total Duration**: 8-12 weeks (depending on team size and experience)

- **Phase 0-1**: 1 week (Foundation & Setup)
- **Phase 2**: 1 week (Data Pipeline)
- **Phase 3**: 2 weeks (HMM Development)
- **Phase 4**: 2 weeks (Black-Scholes Engine)
- **Phase 5**: 1 week (TimesFM Integration - optional)
- **Phase 6**: 2 weeks (Evaluation Framework)
- **Phase 7**: 1 week (Optimization & Sensitivity)
- **Phase 8**: 1 week (UI Development)
- **Phase 9**: 1 week (Documentation & Deployment)
- **Phase 10**: Ongoing (Maintenance)

---

## CONCLUSION

This checklist provides a systematic, production-grade approach to building a sophisticated HMM-Black-Scholes commodity forecasting system. By following this roadmap:

1. **Mathematical rigor**: Foundation in stochastic processes and financial engineering
2. **Engineering excellence**: Clean code, testing, documentation
3. **User-centric design**: Clear UI, comprehensive parameter documentation
4. **Production readiness**: Monitoring, retraining, deployment

The system integrates cutting-edge research (2025-2026) with battle-tested quantitative finance techniques, providing a robust platform for commodity price forecasting and options pricing.

**Key differentiators**:
- Regime-aware forecasting combining HMM with Black-Scholes
- Optional TimesFM integration for long-context forecasting
- Comprehensive parameter tuning with sensitivity analysis
- Production-grade evaluation and validation framework
- User-friendly UI with real-time capabilities

This represents a professional-grade quantitative trading system suitable for hedge funds, proprietary trading firms, and sophisticated individual traders.

---

**Document Version**: 1.0.0
**Last Updated**: 2026-01-16
**Author**: Quantitative Research Team
**Status**: READY FOR IMPLEMENTATION
