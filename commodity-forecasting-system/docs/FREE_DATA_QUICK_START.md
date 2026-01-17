# Free Data Sources - Quick Start Guide

## TL;DR - Best Free Options (2026)

### For SPY Intraday Data (1-minute bars, 2+ years):
**USE ALPACA - It's the best free option available**

```bash
# 1. Sign up for free paper trading account: https://alpaca.markets/
# 2. Get API keys from dashboard
# 3. Install and use:

pip install alpaca-py

export ALPACA_API_KEY="your_key_here"
export ALPACA_SECRET_KEY="your_secret_here"

python scripts/download_free_market_data.py --source alpaca --symbol SPY --interval 1min --years 2
```

**Result:** 7+ years of 1-minute SPY data, completely FREE, 10K API calls/minute

---

### For VIX Daily Data (1990-present):
**USE CBOE - It's the official source**

```bash
# No API key required!

python scripts/download_free_market_data.py --source cboe --symbol VIX
```

**Result:** Complete VIX history from 1990, updated daily, completely FREE

---

### For SPY Daily Data (5+ years):
**USE YFINANCE - Simplest option, no API key**

```bash
pip install yfinance

python scripts/download_free_market_data.py --source yfinance --symbol SPY --interval 1d --years 10
```

**Result:** Unlimited daily history, completely FREE, no signup required

---

## Quick Comparison

| Need | Best Source | Coverage | API Key? | Signup? |
|------|-------------|----------|----------|---------|
| **SPY 1-min bars (2+ years)** | Alpaca | 7+ years | Yes | Free account |
| **SPY daily (5+ years)** | yfinance | Unlimited | No | No |
| **VIX daily (1990+)** | CBOE | 1990-present | No | No |
| **SPY 1-min bars (60 days)** | yfinance | 60 days | No | No |
| **SPY 1-min bars (2 years)** | Polygon.io | 2 years | Yes | Free account |

---

## Installation

```bash
# Install all libraries (choose what you need)
pip install alpaca-py          # For Alpaca (RECOMMENDED)
pip install yfinance            # For Yahoo Finance (no API key)
pip install pandas-datareader   # For FRED (VIX data)
pip install polygon-api-client  # For Polygon.io
pip install tiingo              # For Tiingo

# Or install all at once
pip install alpaca-py yfinance pandas-datareader polygon-api-client tiingo
```

---

## 60-Second Quick Start

### Get 2 Years of SPY 1-Minute Data (Best Free Option)

```python
# 1. Sign up at https://alpaca.markets/ (100% free, no credit card)
# 2. Get API keys from dashboard
# 3. Run this code:

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta

API_KEY = "your_alpaca_api_key"
SECRET_KEY = "your_alpaca_secret_key"

client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

request = StockBarsRequest(
    symbol_or_symbols=["SPY"],
    timeframe=TimeFrame.Minute,
    start=datetime.now() - timedelta(days=730),
    end=datetime.now()
)

bars = client.get_stock_bars(request)
df = bars.df
df.to_csv('spy_2years_1min.csv')

print(f"Downloaded {len(df):,} 1-minute bars")
# Expected output: ~200,000+ bars
```

---

### Get VIX Daily Data (1990-present)

```python
# No API key required!

import pandas as pd

# Method 1: Direct from CBOE
vix = pd.read_csv(
    "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
    skiprows=1
)
vix.to_csv('vix_daily.csv', index=False)

# Method 2: From FRED
from pandas_datareader import data as pdr
from datetime import datetime

vix = pdr.DataReader('VIXCLS', 'fred', datetime(1990, 1, 1), datetime.now())
vix.to_csv('vix_daily_fred.csv')

print(f"Downloaded {len(vix)} days of VIX data")
# Expected output: ~8,000+ days
```

---

### Get SPY Daily Data (No API Key Needed)

```python
import yfinance as yf

# Download 10 years of daily SPY data
spy = yf.download('SPY', start='2014-01-01', end='2024-01-01', interval='1d')
spy.to_csv('spy_10years_daily.csv')

print(f"Downloaded {len(spy)} days")
# Expected output: ~2,500 days
```

---

## Rate Limits Summary

| Source | Free Tier Limit | Notes |
|--------|----------------|-------|
| **Alpaca** | 10,000 calls/min | Extremely generous - best for bulk downloads |
| **Polygon.io** | 5 calls/min | Slow but works for 2 years of data |
| **yfinance** | Varies (100-2000/hr) | Unofficial, can be blocked |
| **Tiingo** | 50 symbols/hour | Good for daily data |
| **Alpha Vantage** | 25 calls/day | Too slow for intraday |
| **Finnhub** | 60 calls/min | Good balance |
| **CBOE** | Unlimited | Direct CSV download |

---

## Common Use Cases

### Use Case 1: Train a ML Model on SPY (Need 2+ years of 1-min data)
**Solution:** Use Alpaca
- Sign up (free)
- Download 7 years of 1-min data in ~5 minutes
- 200K+ bars for training

### Use Case 2: Quick Backtest on SPY Daily (Need 5+ years)
**Solution:** Use yfinance
- No signup
- Instant download
- Unlimited history

### Use Case 3: Analyze VIX Trends (Need full history)
**Solution:** Use CBOE or FRED
- No signup
- Complete 1990-present data
- Official source

### Use Case 4: Build a Dashboard with Recent Data (60 days)
**Solution:** Use yfinance
- No API key
- 1-minute bars for last 60 days
- Simple code

### Use Case 5: Download Multiple Symbols (SPY, QQQ, IWM, etc.)
**Solution:** Use Alpaca
- 10K calls/min (can download multiple symbols in parallel)
- Or use yfinance (no rate limits for daily data)

---

## Troubleshooting

### "Rate limit exceeded"
- **Alpaca:** You won't hit this (10K/min)
- **Polygon:** Slow down to 5 calls/min, add delays
- **yfinance:** Add delays between requests, use random intervals
- **Alpha Vantage:** Only 25/day - use different source

### "API key invalid"
- Double-check environment variables
- Make sure you're using paper trading keys (Alpaca)
- Check if key has expired

### "No data returned"
- Check symbol spelling (SPY not spy)
- Check date range (market hours only for intraday)
- VIX: Use '^VIX' for yfinance, 'VIX' for others

### "Out of memory"
- Download data in chunks (month by month)
- Use date ranges instead of period='max'
- Save to CSV and process in batches

---

## Data Quality Comparison

| Source | Accuracy | Reliability | Completeness |
|--------|----------|-------------|--------------|
| **Alpaca** | ★★★★★ | ★★★★★ | ★★★★★ |
| **CBOE (VIX)** | ★★★★★ | ★★★★★ | ★★★★★ |
| **Polygon.io** | ★★★★★ | ★★★★☆ | ★★★★☆ |
| **Tiingo** | ★★★★☆ | ★★★★☆ | ★★★★☆ |
| **yfinance** | ★★★★☆ | ★★★☆☆ | ★★★★☆ |
| **Finnhub** | ★★★★☆ | ★★★★☆ | ★★★☆☆ |
| **Alpha Vantage** | ★★★★☆ | ★★★★☆ | ★★★★★ |

---

## Recommended Setup for This Project

Based on your requirements (SPY intraday + VIX daily for 2+ years):

```bash
# 1. Install required packages
pip install alpaca-py pandas-datareader yfinance

# 2. Sign up for Alpaca (free paper trading)
# Visit: https://alpaca.markets/

# 3. Set environment variables
export ALPACA_API_KEY="your_key"
export ALPACA_SECRET_KEY="your_secret"

# 4. Download SPY intraday (2+ years of 1-min data)
python scripts/download_free_market_data.py --source alpaca --symbol SPY --interval 1min --years 2

# 5. Download VIX daily (1990-present)
python scripts/download_free_market_data.py --source cboe --symbol VIX

# 6. Backup: Download SPY daily via yfinance (no API key needed)
python scripts/download_free_market_data.py --source yfinance --symbol SPY --interval 1d --years 5
```

**Total time:** ~10 minutes (including signup)
**Total cost:** $0
**Data obtained:**
- 200K+ SPY 1-minute bars (2 years)
- 8K+ VIX daily bars (1990-present)
- 1250+ SPY daily bars (5 years)

---

## Full Command Reference

```bash
# Alpaca - 2 years SPY 1-min (BEST)
python download_free_market_data.py --source alpaca --symbol SPY --interval 1min --years 2

# Alpaca - 7 years SPY 5-min
python download_free_market_data.py --source alpaca --symbol SPY --interval 5min --years 7

# Alpaca - 7 years SPY daily
python download_free_market_data.py --source alpaca --symbol SPY --interval 1d --years 7

# Polygon - 2 years SPY 1-min
python download_free_market_data.py --source polygon --symbol SPY --interval 1min --years 2

# yfinance - 10 years SPY daily
python download_free_market_data.py --source yfinance --symbol SPY --interval 1d --years 10

# yfinance - 60 days SPY 1-min
python download_free_market_data.py --source yfinance --symbol SPY --interval 1m --years 1

# yfinance - VIX daily
python download_free_market_data.py --source yfinance --symbol ^VIX --interval 1d --years 10

# CBOE - VIX official (1990-present)
python download_free_market_data.py --source cboe --symbol VIX

# Tiingo - 30 years SPY daily
python download_free_market_data.py --source tiingo --symbol SPY --interval 1d --years 30
```

---

## Next Steps

1. **For immediate use:** Sign up for Alpaca (5 minutes), download 2 years of SPY 1-min data
2. **For VIX analysis:** Run CBOE download script (no signup needed)
3. **For backtesting:** Use yfinance for quick daily data downloads
4. **For production:** Consider Alpaca's data quality and rate limits vs alternatives

See `FREE_DATA_SOURCES_GUIDE.md` for complete documentation and Python examples.
