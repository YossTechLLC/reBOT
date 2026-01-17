"""
Alpaca Markets Data Client
===========================
Free historical market data via Alpaca Markets API.

Features:
- 7+ years of daily OHLCV data
- 7+ years of 1-minute intraday data
- 10,000 API calls/minute (effectively unlimited)
- No credit card required (free paper trading account)

Setup:
1. Sign up at https://alpaca.markets/
2. Get API key + secret from paper trading account
3. Set environment variables:
   export ALPACA_API_KEY="your_key"
   export ALPACA_SECRET_KEY="your_secret"
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

logger = logging.getLogger(__name__)


class AlpacaDataClient:
    """Client for fetching free historical market data from Alpaca Markets."""

    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None):
        """
        Initialize Alpaca client.

        Args:
            api_key: Alpaca API key (or set ALPACA_API_KEY env var)
            secret_key: Alpaca secret key (or set ALPACA_SECRET_KEY env var)
        """
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')

        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca API credentials not found. Either:\n"
                "1. Pass api_key and secret_key to constructor\n"
                "2. Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables\n"
                "Get free credentials at: https://alpaca.markets/"
            )

        self.client = StockHistoricalDataClient(self.api_key, self.secret_key)
        logger.info("Alpaca client initialized")

    def get_daily_bars(
        self,
        symbol: str,
        days: int = 60,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV bars.

        Args:
            symbol: Stock symbol (e.g., 'SPY')
            days: Number of days of history
            end_date: End date (default: today)

        Returns:
            DataFrame with columns: open, high, low, close, volume, vwap
        """
        if end_date is None:
            end_date = datetime.now()

        start_date = end_date - timedelta(days=days)

        logger.info(f"Fetching {days} days of {symbol} daily bars")

        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date
        )

        bars = self.client.get_stock_bars(request_params)

        # Convert to DataFrame
        df = bars.df

        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()

        # Alpaca returns multi-index (symbol, timestamp) - flatten it
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index(level=0, drop=True)  # Drop symbol level

        # Standardize column names (lowercase)
        df.columns = [col.lower() for col in df.columns]

        # Sort by date
        df = df.sort_index()

        logger.info(f"Retrieved {len(df)} daily bars for {symbol}")
        return df

    def get_intraday_bars(
        self,
        symbol: str,
        days: int = 5,
        timeframe: str = '1Min',
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch intraday bars (1-minute, 5-minute, 15-minute, 1-hour).

        Args:
            symbol: Stock symbol (e.g., 'SPY')
            days: Number of days of history (max 7 years)
            timeframe: Bar frequency ('1Min', '5Min', '15Min', '1Hour')
            end_date: End date (default: today)

        Returns:
            DataFrame with columns: open, high, low, close, volume, vwap, trade_count
        """
        if end_date is None:
            end_date = datetime.now()

        start_date = end_date - timedelta(days=days)

        # Map timeframe string to Alpaca TimeFrame object
        timeframe_map = {
            '1Min': TimeFrame(1, TimeFrameUnit.Minute),
            '5Min': TimeFrame(5, TimeFrameUnit.Minute),
            '15Min': TimeFrame(15, TimeFrameUnit.Minute),
            '1Hour': TimeFrame(1, TimeFrameUnit.Hour),
        }

        if timeframe not in timeframe_map:
            raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of {list(timeframe_map.keys())}")

        logger.info(f"Fetching {days} days of {symbol} {timeframe} bars")

        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe_map[timeframe],
            start=start_date,
            end=end_date
        )

        bars = self.client.get_stock_bars(request_params)

        # Convert to DataFrame
        df = bars.df

        if df.empty:
            logger.warning(f"No intraday data returned for {symbol}")
            return pd.DataFrame()

        # Flatten multi-index
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index(level=0, drop=True)

        # Standardize column names
        df.columns = [col.lower() for col in df.columns]

        # Sort by timestamp
        df = df.sort_index()

        logger.info(f"Retrieved {len(df)} {timeframe} bars for {symbol}")
        return df

    def get_latest_price(self, symbol: str) -> float:
        """
        Get current price for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Current price (close of last bar)
        """
        df = self.get_daily_bars(symbol, days=1)

        if df.empty:
            raise ValueError(f"Could not fetch current price for {symbol}")

        return df['close'].iloc[-1]

    def save_to_cache(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        cache_dir: str = 'data/cache'
    ):
        """
        Save DataFrame to local cache (Parquet format).

        Args:
            df: DataFrame to save
            symbol: Stock symbol
            timeframe: Bar frequency
            cache_dir: Directory to save cache files
        """
        os.makedirs(cache_dir, exist_ok=True)

        filename = f"{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d')}.parquet"
        filepath = os.path.join(cache_dir, filename)

        df.to_parquet(filepath, compression='gzip')
        logger.info(f"Saved {len(df)} bars to {filepath}")

    def load_from_cache(
        self,
        symbol: str,
        timeframe: str,
        cache_dir: str = 'data/cache'
    ) -> Optional[pd.DataFrame]:
        """
        Load DataFrame from local cache.

        Args:
            symbol: Stock symbol
            timeframe: Bar frequency
            cache_dir: Directory containing cache files

        Returns:
            DataFrame if cache exists, None otherwise
        """
        # Find most recent cache file for this symbol/timeframe
        import glob

        pattern = os.path.join(cache_dir, f"{symbol}_{timeframe}_*.parquet")
        files = glob.glob(pattern)

        if not files:
            logger.info(f"No cache found for {symbol} {timeframe}")
            return None

        # Get most recent file
        latest_file = max(files, key=os.path.getctime)

        df = pd.read_parquet(latest_file)
        logger.info(f"Loaded {len(df)} bars from cache: {latest_file}")

        return df


def test_alpaca_client():
    """Test Alpaca client with SPY data."""
    try:
        client = AlpacaDataClient()

        # Test daily data
        print("Testing daily bars...")
        daily = client.get_daily_bars('SPY', days=60)
        print(f"✅ Downloaded {len(daily)} daily bars")
        print(f"   Date range: {daily.index[0]} to {daily.index[-1]}")
        print(f"   Columns: {list(daily.columns)}")
        print()

        # Test intraday data
        print("Testing 1-minute bars...")
        intraday = client.get_intraday_bars('SPY', days=5, timeframe='1Min')
        print(f"✅ Downloaded {len(intraday)} 1-minute bars")
        print(f"   Date range: {intraday.index[0]} to {intraday.index[-1]}")
        print()

        # Test caching
        print("Testing cache...")
        client.save_to_cache(daily, 'SPY', 'daily')
        cached = client.load_from_cache('SPY', 'daily')
        print(f"✅ Cache works: {len(cached)} bars loaded")
        print()

        print("🎉 All tests passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    # Run tests if executed directly
    test_alpaca_client()
