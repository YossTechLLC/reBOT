# HMM-Black-Scholes Commodity Forecasting System

**Version**: 1.0.0
**Status**: In Development
**Started**: 2026-01-16

A sophisticated production-grade commodity forecasting system that combines Hidden Markov Models (HMM) for spot price prediction with Black-Scholes options pricing for futures, featuring configurable parameters, comprehensive UI, and rigorous validation.

---

## Features

### Core Capabilities
- **Hidden Markov Model (HMM)** for regime detection and spot price forecasting
- **Black-Scholes/Black-76** options pricing engine for futures
- **Optional TimesFM Integration** for long-context time-series forecasting
- **Multi-source data acquisition** (yfinance, Quandl, Alpha Vantage, FRED)
- **Comprehensive feature engineering** (technical indicators, macroeconomic features)
- **Production-grade validation** (walk-forward, backtesting, statistical significance tests)
- **Hyperparameter optimization** (Optuna with TPE sampler)
- **Interactive dashboards** (Streamlit and Gradio)
- **Experiment tracking** (MLflow)

### Mathematical Foundation
- **HMM**: Gaussian/GMM emission distributions, Baum-Welch EM algorithm, Viterbi decoding
- **Black-76**: Futures options pricing with convenience yield modeling
- **Volatility estimation**: Historical, GARCH, realized, hybrid methods
- **Greeks calculation**: Delta, Gamma, Theta, Vega, Rho

---

## Quick Start

### Installation

1. Clone repository:
```bash
git clone <repository-url>
cd commodity-forecasting-system
```

2. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -U pip setuptools wheel
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from src.config.loader import load_config
from src.models.hmm_core import CommodityHMM
from src.models.black_scholes import Black76Pricer

# Load configuration
config = load_config()

# Initialize HMM
hmm_model = CommodityHMM(config)

# Train on historical data
hmm_model.fit_with_multiple_inits(features)

# Forecast spot price
forecast = hmm_model.forecast_spot_price(current_features, horizon=30)

# Price options
bs_pricer = Black76Pricer(config)
option_price = bs_pricer.price_call(F=2000, K=2050, T=0.25, r=0.035, sigma=0.18)
```

### Run Dashboard

```bash
streamlit run src/ui/streamlit_app.py
```

---

## Configuration

All system parameters are configured in `config/parameters.toml`. Key parameters:

### HMM Parameters
```toml
[hmm]
n_states = 3              # Number of market regimes (bull/bear/neutral)
covariance_type = "diag"  # Covariance structure
n_iter = 1000             # Maximum EM iterations
n_random_inits = 10       # Multiple initializations to avoid local optima
```

### Black-Scholes Parameters
```toml
[black_scholes]
model_type = "black76"              # Black-76 for futures
volatility_method = "hybrid"        # Combines GARCH + realized volatility
hist_vol_window = 60                # Historical volatility window (days)
convenience_yield_method = "futures_curve"
```

### TimesFM Integration (Optional)
```toml
[timesfm]
enabled = false                     # Enable TimesFM integration
backend = "torch"                   # or "flax"
max_context = 1024                  # Context length
integration_mode = "ensemble"       # ensemble, primary, or regime_input
```

See `docs/PARAMETERS.md` for complete parameter documentation.

---

## Project Structure

```
commodity-forecasting-system/
├── config/
│   ├── parameters.toml           # Master configuration
│   └── schema.json               # Parameter validation schema
├── data/
│   ├── raw/                      # Downloaded data
│   ├── processed/                # Cleaned, feature-engineered
│   └── external/                 # FRED, alternative sources
├── src/
│   ├── data/                     # Data acquisition and preprocessing
│   ├── models/                   # HMM, Black-Scholes, TimesFM
│   ├── evaluation/               # Metrics, backtesting, visualization
│   ├── optimization/             # Hyperparameter tuning
│   ├── config/                   # Configuration management
│   └── ui/                       # Streamlit dashboard
├── notebooks/                     # Jupyter notebooks for exploration
├── tests/                        # Unit and integration tests
├── experiments/                  # MLflow tracking
├── outputs/                      # Forecasts, reports, figures
├── docs/                         # Documentation
└── logs/                         # System logs
```

---

## Architecture

### System Components

1. **Data Layer**: Multi-source acquisition with fallback and validation
2. **Preprocessing Layer**: Missing data handling, outlier detection, stationarity testing
3. **Feature Engineering**: Technical indicators, macroeconomic features, lagged values
4. **Model Layer**: HMM for regimes, Black-Scholes for pricing, optional TimesFM
5. **Evaluation Layer**: Walk-forward validation, backtesting, statistical tests
6. **Optimization Layer**: Optuna for hyperparameter tuning
7. **UI Layer**: Streamlit dashboard for interaction and visualization
8. **Monitoring Layer**: MLflow for experiment tracking, logging for debugging

### Integration Modes

**Ensemble Mode**: Combines HMM and TimesFM predictions
```
Forecast = 0.5 * HMM_forecast + 0.5 * TimesFM_forecast
```

**Primary Mode**: Uses TimesFM for forecasting, HMM for regime detection
```
Forecast = TimesFM_forecast
Regime = HMM_regime  (for volatility estimation)
```

**Regime Input Mode**: Feeds HMM regime states as features to TimesFM
```
Features_augmented = [Features, HMM_states]
Forecast = TimesFM(Features_augmented)
```

---

## Development

### Running Tests

```bash
# Unit tests
pytest tests/test_models/

# Integration tests
pytest tests/test_integration/

# All tests with coverage
pytest --cov=src tests/
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type checking
mypy src/
```

### Experiment Tracking

```bash
# View MLflow UI
mlflow ui --backend-store-uri ./experiments

# Navigate to http://localhost:5000
```

---

## Performance Benchmarks

**Target Metrics** (per CHECKLIST.md):
- ✅ Forecast RMSE < 5% of mean price
- ✅ Directional accuracy > 60%
- ✅ Sharpe ratio > 1.5 in backtests
- ✅ Training time < 10 minutes for 5 years of daily data
- ✅ Inference latency < 1 second

---

## Error Taxonomy

Following CLAUDE.md guidelines, errors are categorized as:

- **TFM1001 CONFIG**: Bad configuration/environment/flags
- **TFM2001 DATA**: Bad shapes, missing columns, leakage risks
- **TFM3001 CHECKPOINT**: Missing or incompatible checkpoint
- **TFM4001 INFERENCE**: Runtime/OOM/NaN/precision issues
- **TFM5001 PERF**: Regression or unexpected slowness

See `BUGS.md` for current issues.

---

## Documentation

- **PARAMETERS.md**: Complete parameter guide with ranges, impacts, sensitivities
- **API.md**: Code API documentation
- **ARCHITECTURE.md**: System design and integration details
- **DEPLOYMENT.md**: Production deployment guide
- **CHECKLIST.md**: Implementation roadmap (in parent directory)
- **CHECKLIST_PROGRESS.md**: Implementation progress tracking

---

## Contributing

See development workflow in CHECKLIST.md Phase 0-1.

1. Create feature branch from `main`
2. Implement changes with tests
3. Update documentation
4. Submit pull request

---

## License

Apache 2.0 (consistent with TimesFM parent project)

---

## Acknowledgments

- **TimesFM**: Google Research time-series foundation model
- **hmmlearn**: BSD-licensed HMM implementation
- **Optuna**: Hyperparameter optimization framework
- **Streamlit**: Interactive dashboard framework

---

## Support

For issues, see BUGS.md or create a GitHub issue.

**Project Status**: Active Development
**Progress Tracking**: See CHECKLIST_PROGRESS.md
**Last Updated**: 2026-01-16
