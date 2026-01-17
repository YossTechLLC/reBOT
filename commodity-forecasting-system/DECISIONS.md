# DECISIONS.md
**HMM-Black-Scholes Commodity Forecasting System - Architecture Decision Log**

Format: `YYYY-MM-DD - decision - context - options - chosen + why (max 5 lines)`

---

## Decision 001: 2026-01-16 - Project Location and Structure

**Context**: Need to organize the HMM-Black-Scholes system within the TimesFM repository structure. The system is related to TimesFM but has its own distinct functionality.

**Options**:
1. Create as a separate top-level directory alongside timesfm/
2. Create as a subdirectory within timesfm/
3. Create within v1/ as an extension

**Decision**: Create `commodity-forecasting-system/` as a subdirectory directly under the TimesFM project root

**Rationale**:
- Keeps all code together while maintaining clear separation
- Allows for independent development and testing
- Can leverage TimesFM as optional integration
- Easy to extract as standalone project later if needed
- Follows modular architecture principles

---

## Decision 002: 2026-01-16 - Configuration Format

**Context**: Need to choose configuration file format for parameter management. System has 100+ configurable parameters requiring clear documentation and type safety.

**Options**:
1. YAML (human-readable, widely used)
2. JSON (strict typing, machine-readable)
3. TOML (explicit, type-safe, Python-native in 3.11+)
4. Python .py config files (executable, flexible)

**Decision**: TOML with Pydantic validation

**Rationale**:
- TOML is more explicit than YAML (no implicit type coercion)
- Native support in Python 3.11+ (tomllib)
- Inline comments for parameter documentation
- Pydantic provides runtime type validation and IDE support
- Clear separation between data (TOML) and validation (Pydantic models)

---

## Decision 003: 2026-01-16 - HMM Library Selection

**Context**: Need to select HMM implementation library for regime detection. Requirements: stable, well-maintained, supports Gaussian/GMM emissions.

**Options**:
1. hmmlearn (sklearn-style, BSD license)
2. pomegranate (more flexible distributions)
3. Custom implementation (full control)
4. statsmodels (integrated with scipy ecosystem)

**Decision**: hmmlearn as primary, with fallback to custom for exotic cases

**Rationale**:
- hmmlearn is well-maintained (v0.3.2+) and widely used
- Sklearn-style API is familiar and documented
- Supports Gaussian and GMM emissions out-of-the-box
- BSD license compatible with Apache 2.0 project
- Custom implementation can be added later if needed for specific emission distributions

---

## Decision 004: 2026-01-16 - Hyperparameter Optimization Framework

**Context**: Need hyperparameter optimization for HMM (n_states, covariance_type, etc.) and other model parameters. Hyperopt is deprecated as of 2024.

**Options**:
1. Optuna (active development, 35% faster than Hyperopt)
2. Ray Tune (distributed, scalable)
3. Scikit-optimize (Bayesian optimization)
4. Grid search (exhaustive, simple)

**Decision**: Optuna with TPE sampler

**Rationale**:
- Optuna is actively maintained with latest research algorithms
- TPE (Tree-structured Parzen Estimator) is efficient for our parameter space
- Built-in pruning saves time by stopping unpromising trials
- Excellent visualization tools for analysis
- Can easily switch to Ray Tune later if distributed tuning needed

---

## Decision 005: 2026-01-16 - Data Source Strategy

**Context**: Need reliable commodity price data. Different sources have different rate limits, costs, and data quality.

**Options**:
1. yfinance only (free, rate-limited, prototyping-grade)
2. Quandl/NASDAQ Data Link (paid, comprehensive, production-grade)
3. Alpha Vantage (freemium, good technical indicators)
4. Multi-source with priority fallback

**Decision**: Multi-source with priority fallback, yfinance for prototyping, Quandl for production

**Rationale**:
- yfinance is free and sufficient for development/prototyping
- Quandl provides production-grade data quality and coverage
- Alpha Vantage as backup source for redundancy
- FRED for macroeconomic features (free, authoritative)
- Priority system allows easy swap between sources via config

---

## Decision 006: 2026-01-16 - Volatility Estimation Method

**Context**: Black-Scholes requires volatility input. Multiple estimation methods available with different characteristics.

**Options**:
1. Historical volatility (simple, backward-looking)
2. GARCH (captures volatility clustering)
3. Realized volatility (high-frequency, intraday)
4. Hybrid (weighted combination)

**Decision**: Hybrid (GARCH + realized) as default, all methods available via config

**Rationale**:
- GARCH captures volatility clustering and persistence
- Realized volatility uses high-low range (Parkinson estimator)
- Hybrid approach balances historical and forward-looking estimates
- Different commodities may benefit from different methods
- Config parameter allows experimentation and adaptation

---

## Decision 007: 2026-01-16 - TimesFM Integration Architecture

**Context**: TimesFM (from MAP.md) offers long-context forecasting. Need to decide integration approach.

**Options**:
1. TimesFM as primary forecaster, HMM for regime only
2. HMM as primary, TimesFM optional
3. Ensemble (average predictions)
4. Regime-aware TimesFM (feed HMM states as features)

**Decision**: All three modes available via config parameter `timesfm.integration_mode`

**Rationale**:
- Different use cases benefit from different approaches
- Ensemble provides robustness through diversification
- Primary mode leverages TimesFM's long-context capability
- Regime-input mode conditions TimesFM on market regime
- Config flexibility allows empirical comparison and selection

---

## Decision 008: 2026-01-16 - Testing Strategy

**Context**: Need comprehensive testing for financial system. Requirements: unit tests, integration tests, numerical validation.

**Options**:
1. Pytest only (standard Python testing)
2. Pytest + hypothesis (property-based testing)
3. Pytest + numerical validation suite
4. Full suite: pytest + property tests + benchmark tests

**Decision**: Pytest with numerical validation and benchmark tests

**Rationale**:
- Pytest is industry standard with excellent fixture support
- Numerical validation critical for financial models (Put-Call parity, Greeks finite difference)
- Benchmark tests ensure performance doesn't regress (TFM5001)
- Property-based testing (hypothesis) can be added later for edge cases
- Integration tests ensure end-to-end pipeline correctness

---

## Decision 009: 2026-01-16 - UI Framework Selection

**Context**: Need interactive dashboard for parameter tuning and visualization. Users range from quants to traders.

**Options**:
1. Streamlit (rapid development, rich components)
2. Gradio (ML-focused, Hugging Face integration)
3. Dash (Plotly-based, enterprise features)
4. Flask/FastAPI + React (full custom, most flexible)

**Decision**: Streamlit for main dashboard, Gradio for quick demos

**Rationale**:
- Streamlit enables rapid prototyping with rich widgets
- Built-in caching and state management
- Gradio excellent for quick model demos and testing
- Both are Python-native (no JavaScript required)
- Can migrate to Dash later if enterprise features needed
- Custom React frontend not justified given existing tools

---

## Decision 010: 2026-01-16 - Experiment Tracking

**Context**: Need to track model experiments, parameters, metrics for reproducibility and comparison.

**Options**:
1. MLflow (open-source, self-hosted)
2. Weights & Biases (cloud, superior visualization)
3. Neptune.ai (versioning, collaboration)
4. Custom SQLite/PostgreSQL logging

**Decision**: MLflow for self-hosted tracking, W&B as optional upgrade

**Rationale**:
- MLflow is open-source and can run locally (no external dependencies)
- Excellent integration with sklearn-style models
- Model registry for versioning and promotion
- W&B has better visualization but requires cloud/account
- Config parameter allows easy switch between backends
- Custom DB logging as ultimate fallback

---

**Last Updated**: 2026-01-16
**Total Decisions**: 10
**Status**: Active Development
