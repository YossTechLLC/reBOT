# Free Historical Stock Market Data Sources Guide (2026)

## Executive Summary

This guide provides a comprehensive ranking of FREE data sources for historical stock market data, specifically targeting SPY, VIX, and options data. All sources are truly free (no trial periods that expire).

---

## Top Ranked Data Sources

### 1. Alpaca Markets (BEST OVERALL FOR INTRADAY)
**Rating: 9/10**

**What You Get (FREE):**
- 7+ years of historical data
- Intraday bars: 1min, 5min, 15min, 30min, 60min
- Daily, weekly, monthly bars
- Up to 10,000 API calls/minute (extremely generous)
- FREE forever with paper trading account
- Real-time and delayed data

**SPY Coverage:** ✅ Excellent (7+ years daily, 7+ years intraday)
**VIX Coverage:** ⚠️ Limited (VIX is not a tradeable equity)
**Intraday Frequency:** ✅ 1-minute bars available
**Python Support:** ✅ Official SDK (alpaca-py)

**How to Access:**

```python
# Installation
pip install alpaca-py

# Usage Example
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime

# Get your free API keys at https://alpaca.markets/
# Sign up for paper trading account (100% free)
API_KEY = "your_api_key"
SECRET_KEY = "your_secret_key"

client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# Request 1-minute bars for SPY
request_params = StockBarsRequest(
    symbol_or_symbols=["SPY"],
    timeframe=TimeFrame.Minute,
    start=datetime(2023, 1, 1),
    end=datetime(2024, 1, 1)
)

bars = client.get_stock_bars(request_params)
df = bars.df  # Convert to pandas DataFrame

print(f"Retrieved {len(df)} bars of 1-minute SPY data")
```

**Pros:**
- Best free intraday data available
- 7+ years of history
- Very high rate limits
- Official Python SDK
- Reliable and stable

**Cons:**
- Requires account signup (free)
- No VIX data (it's an index, not a stock)
- US markets only on free tier

**URLs:**
- Website: https://alpaca.markets/
- Docs: https://docs.alpaca.markets/
- Python SDK: https://alpaca.markets/sdks/python/

---

### 2. Polygon.io / Massive.com (BEST FOR LONG HISTORY)
**Rating: 8/10**

**What You Get (FREE):**
- 2 years of historical intraday data (1-minute granularity)
- End-of-day data for US equities, forex, crypto
- 5 API calls/minute (rate limited)
- 50,000 data points per request

**SPY Coverage:** ✅ Good (2 years daily, 2 years intraday)
**VIX Coverage:** ⚠️ Limited (index data may be available)
**Intraday Frequency:** ✅ 1-minute bars
**Python Support:** ✅ Official SDK

**How to Access:**

```python
# Installation
pip install polygon-api-client

# Usage Example
from polygon import RESTClient
from datetime import datetime

# Get free API key at https://polygon.io/
API_KEY = "your_api_key"

client = RESTClient(API_KEY)

# Get 1-minute bars for SPY
# Note: 50,000 point limit = ~35 days of 1-min data per request
aggs = client.get_aggs(
    ticker="SPY",
    multiplier=1,
    timespan="minute",
    from_="2024-01-01",
    to="2024-02-05",
    limit=50000
)

# Convert to DataFrame
import pandas as pd
data = []
for agg in aggs:
    data.append({
        'timestamp': datetime.fromtimestamp(agg.timestamp / 1000),
        'open': agg.open,
        'high': agg.high,
        'low': agg.low,
        'close': agg.close,
        'volume': agg.volume
    })
df = pd.DataFrame(data)

print(f"Retrieved {len(df)} bars")
```

**Pros:**
- 2 years of 1-minute data
- Simple API
- Official Python client

**Cons:**
- Only 5 API calls/minute (slow for large downloads)
- 50,000 point limit per request (requires chunking)
- Limited to ~3 months of 1-min data per request

**URLs:**
- Website: https://polygon.io/ (now https://massive.com/)
- Docs: https://polygon.io/docs/

---

### 3. Tiingo (GOOD FOR DAILY DATA)
**Rating: 7/10**

**What You Get (FREE):**
- 30+ years of end-of-day data
- IEX intraday data (last ~2000 ticks)
- 50 symbols per hour limit
- Daily OHLCV, adjusted prices

**SPY Coverage:** ✅ Excellent (30+ years daily), ⚠️ Limited intraday (~7 days at 5-min)
**VIX Coverage:** ⚠️ VIX not available on IEX
**Intraday Frequency:** ⚠️ Limited to ~2000 most recent ticks
**Python Support:** ✅ Official library (tiingo-python)

**How to Access:**

```python
# Installation
pip install tiingo

# Usage Example
from tiingo import TiingoClient
import pandas as pd

# Get free API key at https://www.tiingo.com/
config = {
    'api_key': 'your_api_key'
}

client = TiingoClient(config)

# Get daily historical data for SPY (30+ years available)
historical_prices = client.get_dataframe(
    'SPY',
    startDate='2020-01-01',
    endDate='2024-01-01'
)

print(f"Retrieved {len(historical_prices)} days of SPY data")

# Get intraday data (limited to last 2000 ticks)
intraday = client.get_dataframe(
    'SPY',
    frequency='5min',
    startDate='2024-01-10',
    endDate='2024-01-17'
)

print(f"Retrieved {len(intraday)} 5-minute bars")
```

**Pros:**
- 30+ years of daily data
- Simple Python library
- Good for EOD analysis

**Cons:**
- Intraday limited to ~2000 recent ticks
- 50 symbols/hour rate limit
- No VIX (IEX doesn't trade indices)

**URLs:**
- Website: https://www.tiingo.com/
- Docs: https://api.tiingo.com/
- Python: https://github.com/hydrosquall/tiingo-python

---

### 4. Alpha Vantage (LIMITED BUT FUNCTIONAL)
**Rating: 6/10**

**What You Get (FREE):**
- 20+ years of monthly intraday data (if you specify month)
- Latest 30 days of intraday (if no month specified)
- 25 API calls per day (very restrictive)
- Daily, weekly, monthly data

**SPY Coverage:** ✅ Good (20+ years daily), ⚠️ Limited intraday (need to download month-by-month)
**VIX Coverage:** ✅ Available
**Intraday Frequency:** ✅ 1min, 5min, 15min, 30min, 60min
**Python Support:** ⚠️ Third-party libraries only

**How to Access:**

```python
# Installation
pip install alpha_vantage

# Usage Example
from alpha_vantage.timeseries import TimeSeries
import pandas as pd

# Get free API key at https://www.alphavantage.co/support/#api-key
API_KEY = "your_api_key"

ts = TimeSeries(key=API_KEY, output_format='pandas')

# Get daily data (full history)
data_daily, meta = ts.get_daily(symbol='SPY', outputsize='full')
print(f"Daily data: {len(data_daily)} rows")

# Get intraday data (last 30 days, or specific month)
# For historical: use month parameter (YYYY-MM format)
data_intraday, meta = ts.get_intraday(
    symbol='SPY',
    interval='5min',
    outputsize='full',
    month='2024-01'  # Specify month for historical data
)
print(f"Intraday data for Jan 2024: {len(data_intraday)} rows")

# VIX data is also available
data_vix, meta = ts.get_daily(symbol='VIX', outputsize='full')
print(f"VIX data: {len(data_vix)} rows")
```

**Pros:**
- Can access 20+ years of intraday by specifying month
- VIX data available
- Multiple timeframes

**Cons:**
- Only 25 API calls/day (severely limiting)
- Must download month-by-month for historical intraday
- Slow data collection

**URLs:**
- Website: https://www.alphavantage.co/
- Docs: https://www.alphavantage.co/documentation/

---

### 5. Finnhub.io (GOOD FOR RECENT DATA)
**Rating: 6/10**

**What You Get (FREE):**
- 1 year of historical data per call
- 60 API calls/minute
- Real-time data (15-minute delayed)
- Daily, weekly, monthly candles

**SPY Coverage:** ✅ Good (1+ years daily and intraday)
**VIX Coverage:** ⚠️ May be limited
**Intraday Frequency:** ✅ Minute-level resolutions
**Python Support:** ✅ Official library

**How to Access:**

```python
# Installation
pip install finnhub-python

# Usage Example
import finnhub
from datetime import datetime, timedelta

# Get free API key at https://finnhub.io/
API_KEY = "your_api_key"

client = finnhub.Client(api_key=API_KEY)

# Get daily candles (1 year of data)
end_date = int(datetime.now().timestamp())
start_date = int((datetime.now() - timedelta(days=365)).timestamp())

candles = client.stock_candles('SPY', 'D', start_date, end_date)

import pandas as pd
df = pd.DataFrame({
    'timestamp': pd.to_datetime(candles['t'], unit='s'),
    'open': candles['o'],
    'high': candles['h'],
    'low': candles['l'],
    'close': candles['c'],
    'volume': candles['v']
})

print(f"Retrieved {len(df)} daily candles")

# Get 5-minute candles
candles_5min = client.stock_candles('SPY', '5', start_date, end_date)
```

**Pros:**
- 60 calls/minute (reasonable)
- 1 year per call
- Official Python library

**Cons:**
- Limited to ~1 year history
- May require multiple calls for multi-year data

**URLs:**
- Website: https://finnhub.io/
- Docs: https://finnhub.io/docs/api

---

### 6. CBOE (BEST FOR VIX DATA)
**Rating: 8/10 (for VIX only)**

**What You Get (FREE):**
- VIX daily data from 1990 to present
- Free CSV downloads
- Updated daily
- Official source

**SPY Coverage:** ❌ Not available
**VIX Coverage:** ✅ EXCELLENT (1990-present, daily)
**Intraday Frequency:** ❌ No intraday
**Python Support:** ⚠️ Manual CSV download

**How to Access:**

```python
# Direct download from CBOE
import pandas as pd

# Method 1: Download CSV manually from CBOE website
# https://www.cboe.com/tradable_products/vix/vix_historical_data/

# Method 2: Use pandas to read directly
vix_url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"

# Read CSV (may need to adjust skiprows based on current format)
vix_data = pd.read_csv(vix_url, skiprows=1)
vix_data.columns = ['Date', 'Open', 'High', 'Low', 'Close']
vix_data['Date'] = pd.to_datetime(vix_data['Date'])

print(f"VIX data: {len(vix_data)} days from {vix_data['Date'].min()} to {vix_data['Date'].max()}")
```

**Alternative: Use FRED (Federal Reserve Economic Data)**

```python
# Using pandas_datareader
pip install pandas-datareader

from pandas_datareader import data as pdr
from datetime import datetime

# Get VIX from FRED (no API key required for basic usage)
vix = pdr.DataReader('VIXCLS', 'fred', datetime(1990, 1, 1), datetime(2024, 1, 1))
print(f"VIX from FRED: {len(vix)} days")
```

**Pros:**
- Official VIX source
- Complete history from 1990
- Free, no API key needed
- Also available via FRED

**Cons:**
- VIX only (not for SPY)
- No intraday data
- Manual download or scraping

**URLs:**
- CBOE VIX: https://www.cboe.com/tradable_products/vix/vix_historical_data/
- FRED VIX: https://fred.stlouisfed.org/series/VIXCLS

---

### 7. yfinance (FALLBACK OPTION)
**Rating: 5/10**

**What You Get (FREE):**
- Unlimited daily data (years of history)
- 60 days of intraday (1min, 2min, 5min, 15min, 30min, 60min)
- No API key required
- Fast and simple

**SPY Coverage:** ✅ Excellent daily, ⚠️ Limited intraday (60 days)
**VIX Coverage:** ✅ Good (daily only)
**Intraday Frequency:** ⚠️ Limited to 60 days
**Python Support:** ✅ Popular library

**How to Access:**

```python
# Installation
pip install yfinance

# Usage Example
import yfinance as yf
import pandas as pd

# Download daily data (unlimited history)
spy_daily = yf.download('SPY', start='2019-01-01', end='2024-01-01', interval='1d')
print(f"SPY daily: {len(spy_daily)} days")

# Download intraday data (limited to 60 days)
spy_1min = yf.download('SPY', period='60d', interval='1m')
print(f"SPY 1-minute: {len(spy_1min)} bars")

# VIX data
vix_daily = yf.download('^VIX', start='2019-01-01', end='2024-01-01', interval='1d')
print(f"VIX daily: {len(vix_daily)} days")
```

**Pros:**
- No API key
- Simple to use
- Fast for daily data
- Works for both SPY and VIX

**Cons:**
- Only 60 days of intraday
- Unreliable (Yahoo can break it anytime)
- Rate limiting issues
- Built on web scraping

**URLs:**
- GitHub: https://github.com/ranaroussi/yfinance

---

## Specific Answers to Your Questions

### 1. Can we get 2+ years of SPY daily data for free?
**✅ YES - Multiple sources:**
- **Alpaca**: 7+ years daily ⭐ BEST
- **Tiingo**: 30+ years daily ⭐ BEST
- **Polygon.io**: 2+ years daily
- **yfinance**: Unlimited years daily
- **Alpha Vantage**: 20+ years daily
- **Finnhub**: 1+ years daily

**Recommendation:** Use Alpaca or Tiingo for daily data.

### 2. Can we get 1+ years of SPY intraday (any frequency) for free?
**✅ YES - With caveats:**
- **Alpaca**: 7+ years of 1-minute data ⭐ BEST OPTION
- **Polygon.io**: 2 years of 1-minute data (but slow download due to rate limits)
- **Alpha Vantage**: 20+ years available but must download month-by-month (25 calls/day limit)

**Recommendation:** Use Alpaca for best intraday coverage.

### 3. Which source has the best VIX historical data?
**✅ CBOE + FRED:**
- **CBOE**: Official source, 1990-present, daily, free CSV download ⭐ BEST
- **FRED**: Same data, easy API access via pandas_datareader
- **yfinance**: Good fallback, daily data
- **Alpha Vantage**: Available but limited to 25 calls/day

**Recommendation:** Use CBOE direct download or FRED API.

### 4. Are there any sources with free historical options data?
**⚠️ LIMITED FREE OPTIONS:**

**Free Sources:**
- **CBOE Historical Data**: Free downloads of aggregate options statistics (volume, put/call ratios)
  - URL: https://www.cboe.com/us/options/market_statistics/historical_data/
  - ⚠️ NOT individual option chains, just aggregate stats

- **Option Strategist**: Free weekly implied volatility data
  - URL: https://www.optionstrategist.com/calculators/free-volatility-data
  - ⚠️ Weekly data only, limited scope

**Paid Sources (No Free Tier):**
- OptionMetrics (IvyDB): Industry standard, institutional pricing
- IVolatility.com: Historical option chains since 2005
- FirstRateData: Historical option chains with Greeks

**Recommendation:** No truly comprehensive free historical options data exists. For serious options analysis, you'll need a paid service. CBOE provides free aggregate statistics which may be sufficient for some use cases.

---

## Recommended Strategy for Your Use Case

### For SPY Historical Data:

**Daily Data (5+ years):**
1. Primary: **Alpaca** (7+ years, free, excellent API)
2. Backup: **Tiingo** (30+ years, free)

**Intraday Data (1-minute bars, 1+ years):**
1. Primary: **Alpaca** (7+ years, 1-min bars, 10K calls/min) ⭐ BEST
2. Secondary: **Polygon.io** (2 years, 1-min bars, slower due to rate limits)
3. Workaround: **Alpha Vantage** (download month-by-month, 25 calls/day = ~1 month of data per day)

### For VIX Historical Data:

**Daily Data:**
1. Primary: **CBOE** (official source, 1990-present, free CSV)
2. Secondary: **FRED** (same data, easy API access)
3. Backup: **yfinance** (simple, no API key)

### For Options Data:

**Free (Limited):**
1. **CBOE** aggregate statistics (free downloads)
2. **Option Strategist** weekly IV data (free)

**Paid (Comprehensive):**
- Budget: FirstRateData (~$49-199 for historical packs)
- Professional: OptionMetrics, IVolatility

---

## Complete Python Example: Download 2 Years of SPY 1-Minute Data

```python
"""
Complete example: Download 2+ years of SPY 1-minute intraday data using Alpaca
This is FREE and provides the best coverage available.
"""

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
import pandas as pd

# Sign up at https://alpaca.markets/ (free paper trading account)
API_KEY = "your_alpaca_api_key"
SECRET_KEY = "your_alpaca_secret_key"

client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# Download 2 years of 1-minute SPY data
# Note: Alpaca has 7+ years available
end_date = datetime.now()
start_date = end_date - timedelta(days=730)  # 2 years

print(f"Downloading SPY 1-minute data from {start_date} to {end_date}...")

request = StockBarsRequest(
    symbol_or_symbols=["SPY"],
    timeframe=TimeFrame.Minute,
    start=start_date,
    end=end_date
)

bars = client.get_stock_bars(request)
df = bars.df

print(f"Downloaded {len(df)} 1-minute bars")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"Data size: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Save to CSV
df.to_csv('spy_1min_2years.csv')
print("Saved to spy_1min_2years.csv")

# Preview
print(df.head())
print(df.tail())
```

---

## Summary Comparison Table

| Source | SPY Daily | SPY Intraday | VIX Daily | Intraday Freq | Rate Limit | API Key | Python SDK |
|--------|-----------|--------------|-----------|---------------|------------|---------|------------|
| **Alpaca** | 7+ years | 7+ years | ❌ | 1-min | 10K/min | ✅ Free | ✅ Official |
| **Polygon.io** | 2+ years | 2 years | ⚠️ | 1-min | 5/min | ✅ Free | ✅ Official |
| **Tiingo** | 30+ years | ~7 days | ❌ | 5-min | 50 sym/hr | ✅ Free | ✅ Official |
| **Alpha Vantage** | 20+ years | 20+ years* | ✅ | 1-min | 25/day | ✅ Free | ⚠️ 3rd party |
| **Finnhub** | 1+ years | 1 year | ⚠️ | 1-min | 60/min | ✅ Free | ✅ Official |
| **CBOE** | ❌ | ❌ | ✅ 1990+ | ❌ | Unlimited | ❌ | Manual CSV |
| **yfinance** | Unlimited | 60 days | ✅ | 1-min | Varies | ❌ | ✅ Popular |

*Alpha Vantage: Must download month-by-month

---

## Final Recommendation

**For your specific requirements:**

1. **SPY Daily (5+ years)**: Use **Alpaca** or **Tiingo**
2. **SPY Intraday (1-min, 2+ years)**: Use **Alpaca** (BEST FREE OPTION)
3. **VIX Daily**: Use **CBOE** direct download or **FRED API**
4. **Options Data**: No comprehensive free source; use **CBOE** for aggregate stats or budget $49-199 for historical packs from FirstRateData

**Bottom Line:** Sign up for a free Alpaca paper trading account. It provides the best free historical intraday data available (7+ years of 1-minute bars, 10K API calls/minute). Combine with CBOE/FRED for VIX data, and you'll have everything you need for free.

---

## Sources

- [Alpaca Markets Data Documentation](https://alpaca.markets/data)
- [Alpaca Python SDK](https://alpaca.markets/sdks/python/market_data.html)
- [Polygon.io Data API](https://polygon.io/)
- [Polygon Free Data APIs](https://massive.com/blog/free-data-apis)
- [Tiingo API Documentation](https://www.tiingo.com/)
- [Tiingo Python Library](https://github.com/hydrosquall/tiingo-python)
- [Alpha Vantage API Documentation](https://www.alphavantage.co/documentation/)
- [Alpha Vantage Complete Guide 2026](https://alphalog.ai/blog/alphavantage-api-complete-guide)
- [Finnhub API Documentation](https://finnhub.io/docs/api)
- [Finnhub Python Client](https://github.com/Finnhub-Stock-API/finnhub-python)
- [CBOE VIX Historical Data](https://www.cboe.com/tradable_products/vix/vix_historical_data/)
- [FRED VIX Data](https://fred.stlouisfed.org/series/VIXCLS)
- [CBOE Options Historical Data](https://www.cboe.com/us/options/market_statistics/historical_data/)
- [Option Strategist Free Volatility Data](https://www.optionstrategist.com/calculators/free-volatility-data)
- [yfinance GitHub](https://github.com/ranaroussi/yfinance)
- [Comparing Intraday Data Sources](https://www.crackingmarkets.com/comparing-affordable-intraday-data-sources-tradestation-vs-polygon-vs-alpaca/)
- [Beyond yFinance Alternatives](https://medium.com/@trading.dude/beyond-yfinance-comparing-the-best-financial-data-apis-for-traders-and-developers-06a3b8bc07e2)
