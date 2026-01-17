"""
Free Market Data Downloader
===========================

This script downloads historical SPY and VIX data using FREE data sources.
No paid subscriptions required - all sources are 100% free.

Usage:
    python download_free_market_data.py --source alpaca --symbol SPY --interval 1min --years 2
    python download_free_market_data.py --source cboe --symbol VIX
    python download_free_market_data.py --source yfinance --symbol SPY --interval 1d --years 5

Requirements:
    pip install alpaca-py pandas yfinance pandas-datareader requests

Author: reBOT Team
Date: 2026-01-17
"""

import argparse
import pandas as pd
from datetime import datetime, timedelta
import os
import sys


class DataDownloader:
    """Unified interface for downloading data from multiple free sources"""

    def __init__(self, output_dir='data/raw'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def download_alpaca(self, symbol, interval='1min', years=2, api_key=None, secret_key=None):
        """
        Download data from Alpaca Markets (FREE - requires paper trading account)

        Coverage: 7+ years of 1-minute bars, 10,000 API calls/minute
        Sign up: https://alpaca.markets/

        Args:
            symbol: Stock symbol (e.g., 'SPY')
            interval: '1min', '5min', '15min', '1h', '1d'
            years: Number of years of historical data
            api_key: Alpaca API key (or set ALPACA_API_KEY env var)
            secret_key: Alpaca secret key (or set ALPACA_SECRET_KEY env var)
        """
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
        except ImportError:
            print("Error: alpaca-py not installed. Install with: pip install alpaca-py")
            return None

        # Get API keys from arguments or environment
        api_key = api_key or os.getenv('ALPACA_API_KEY')
        secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')

        if not api_key or not secret_key:
            print("Error: Alpaca API keys required. Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.")
            print("Sign up for FREE at: https://alpaca.markets/")
            return None

        # Map interval to TimeFrame
        interval_map = {
            '1min': TimeFrame.Minute,
            '5min': TimeFrame(5, 'Min'),
            '15min': TimeFrame(15, 'Min'),
            '30min': TimeFrame(30, 'Min'),
            '1h': TimeFrame.Hour,
            '1d': TimeFrame.Day
        }

        if interval not in interval_map:
            print(f"Error: Unsupported interval '{interval}'. Use: {list(interval_map.keys())}")
            return None

        client = StockHistoricalDataClient(api_key, secret_key)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years)

        print(f"Downloading {symbol} {interval} data from Alpaca...")
        print(f"Date range: {start_date.date()} to {end_date.date()}")

        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=interval_map[interval],
            start=start_date,
            end=end_date
        )

        bars = client.get_stock_bars(request)
        df = bars.df

        # Reset index to get timestamp as column
        df = df.reset_index()

        output_file = os.path.join(self.output_dir, f"{symbol}_{interval}_alpaca_{years}y.csv")
        df.to_csv(output_file, index=False)

        print(f"✓ Downloaded {len(df):,} bars")
        print(f"✓ Saved to: {output_file}")
        print(f"✓ Size: {os.path.getsize(output_file) / 1024**2:.2f} MB")

        return df

    def download_polygon(self, symbol, interval='1min', years=2, api_key=None):
        """
        Download data from Polygon.io (FREE - 2 years of 1-min data)

        Coverage: 2 years of 1-minute bars, 5 API calls/minute
        Sign up: https://polygon.io/

        Args:
            symbol: Stock symbol (e.g., 'SPY')
            interval: '1min', '5min', '15min', '1h', '1d'
            years: Number of years (max 2 on free tier)
            api_key: Polygon API key (or set POLYGON_API_KEY env var)
        """
        try:
            from polygon import RESTClient
        except ImportError:
            print("Error: polygon-api-client not installed. Install with: pip install polygon-api-client")
            return None

        api_key = api_key or os.getenv('POLYGON_API_KEY')

        if not api_key:
            print("Error: Polygon API key required. Set POLYGON_API_KEY environment variable.")
            print("Sign up for FREE at: https://polygon.io/")
            return None

        if years > 2:
            print("Warning: Free tier limited to 2 years. Setting years=2")
            years = 2

        # Map interval to Polygon format
        interval_map = {
            '1min': ('minute', 1),
            '5min': ('minute', 5),
            '15min': ('minute', 15),
            '30min': ('minute', 30),
            '1h': ('hour', 1),
            '1d': ('day', 1)
        }

        if interval not in interval_map:
            print(f"Error: Unsupported interval '{interval}'. Use: {list(interval_map.keys())}")
            return None

        timespan, multiplier = interval_map[interval]

        client = RESTClient(api_key)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years)

        print(f"Downloading {symbol} {interval} data from Polygon.io...")
        print(f"Date range: {start_date.date()} to {end_date.date()}")
        print("Note: Free tier has 50,000 data points per request limit")
        print("      Large date ranges will be chunked automatically")

        # Polygon has 50k point limit, so we need to chunk for intraday data
        # 50k minutes = ~35 days, so chunk by month for safety
        all_data = []
        current_start = start_date

        while current_start < end_date:
            # Chunk by 30 days for minute data, 365 days for daily
            chunk_days = 30 if timespan == 'minute' else 365
            current_end = min(current_start + timedelta(days=chunk_days), end_date)

            print(f"  Fetching: {current_start.date()} to {current_end.date()}...")

            aggs = client.get_aggs(
                ticker=symbol,
                multiplier=multiplier,
                timespan=timespan,
                from_=current_start.strftime('%Y-%m-%d'),
                to=current_end.strftime('%Y-%m-%d'),
                limit=50000
            )

            for agg in aggs:
                all_data.append({
                    'timestamp': datetime.fromtimestamp(agg.timestamp / 1000),
                    'open': agg.open,
                    'high': agg.high,
                    'low': agg.low,
                    'close': agg.close,
                    'volume': agg.volume
                })

            current_start = current_end + timedelta(days=1)

        df = pd.DataFrame(all_data)
        df = df.sort_values('timestamp').reset_index(drop=True)

        output_file = os.path.join(self.output_dir, f"{symbol}_{interval}_polygon_{years}y.csv")
        df.to_csv(output_file, index=False)

        print(f"✓ Downloaded {len(df):,} bars")
        print(f"✓ Saved to: {output_file}")
        print(f"✓ Size: {os.path.getsize(output_file) / 1024**2:.2f} MB")

        return df

    def download_yfinance(self, symbol, interval='1d', years=5):
        """
        Download data from Yahoo Finance (FREE - no API key required)

        Coverage:
            - Daily: Unlimited history
            - Intraday: 60 days only (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h)

        Args:
            symbol: Stock symbol (e.g., 'SPY', '^VIX')
            interval: '1m', '5m', '15m', '1h', '1d', '1wk', '1mo'
            years: Number of years (for daily only)
        """
        try:
            import yfinance as yf
        except ImportError:
            print("Error: yfinance not installed. Install with: pip install yfinance")
            return None

        print(f"Downloading {symbol} {interval} data from Yahoo Finance...")

        # yfinance has 60-day limit for intraday
        if interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h']:
            print("Warning: Intraday data limited to 60 days on Yahoo Finance")
            df = yf.download(symbol, period='60d', interval=interval, progress=False)
            period_str = '60d'
        else:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * years)
            df = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False)
            period_str = f'{years}y'

        if df.empty:
            print(f"Error: No data returned for {symbol}")
            return None

        # Reset index to get date/timestamp as column
        df = df.reset_index()

        # Rename columns to standard format
        column_map = {
            'Date': 'timestamp',
            'Datetime': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume'
        }
        df = df.rename(columns=column_map)

        output_file = os.path.join(self.output_dir, f"{symbol}_{interval}_yfinance_{period_str}.csv")
        df.to_csv(output_file, index=False)

        print(f"✓ Downloaded {len(df):,} bars")
        print(f"✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"✓ Saved to: {output_file}")
        print(f"✓ Size: {os.path.getsize(output_file) / 1024**2:.2f} MB")

        return df

    def download_cboe_vix(self):
        """
        Download VIX data from CBOE (FREE - official source)

        Coverage: Daily VIX from 1990 to present
        No API key required
        """
        import requests

        print("Downloading VIX data from CBOE (official source)...")

        # CBOE VIX historical data URL
        url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"

        try:
            df = pd.read_csv(url, skiprows=1)
            df.columns = ['date', 'open', 'high', 'low', 'close']
            df['date'] = pd.to_datetime(df['date'])

            output_file = os.path.join(self.output_dir, "VIX_daily_cboe.csv")
            df.to_csv(output_file, index=False)

            print(f"✓ Downloaded {len(df):,} days of VIX data")
            print(f"✓ Date range: {df['date'].min().date()} to {df['date'].max().date()}")
            print(f"✓ Saved to: {output_file}")
            print(f"✓ Size: {os.path.getsize(output_file) / 1024**2:.2f} MB")

            return df

        except Exception as e:
            print(f"Error downloading from CBOE: {e}")
            print("Trying alternative method via FRED...")
            return self.download_fred_vix()

    def download_fred_vix(self):
        """
        Download VIX data from FRED (Federal Reserve Economic Data)

        Coverage: Daily VIX from 1990 to present
        No API key required for basic usage
        """
        try:
            from pandas_datareader import data as pdr
        except ImportError:
            print("Error: pandas-datareader not installed. Install with: pip install pandas-datareader")
            return None

        print("Downloading VIX data from FRED...")

        try:
            df = pdr.DataReader('VIXCLS', 'fred', datetime(1990, 1, 1), datetime.now())
            df = df.reset_index()
            df.columns = ['date', 'close']

            output_file = os.path.join(self.output_dir, "VIX_daily_fred.csv")
            df.to_csv(output_file, index=False)

            print(f"✓ Downloaded {len(df):,} days of VIX data")
            print(f"✓ Date range: {df['date'].min().date()} to {df['date'].max().date()}")
            print(f"✓ Saved to: {output_file}")

            return df

        except Exception as e:
            print(f"Error: {e}")
            return None

    def download_tiingo(self, symbol, interval='1d', years=5, api_key=None):
        """
        Download data from Tiingo (FREE - 30+ years daily, limited intraday)

        Coverage:
            - Daily: 30+ years
            - Intraday: ~2000 most recent ticks (~7 days at 5-min)

        Sign up: https://www.tiingo.com/

        Args:
            symbol: Stock symbol (e.g., 'SPY')
            interval: '1d' or '5min' (intraday limited)
            years: Number of years (for daily only)
            api_key: Tiingo API key (or set TIINGO_API_KEY env var)
        """
        try:
            from tiingo import TiingoClient
        except ImportError:
            print("Error: tiingo not installed. Install with: pip install tiingo")
            return None

        api_key = api_key or os.getenv('TIINGO_API_KEY')

        if not api_key:
            print("Error: Tiingo API key required. Set TIINGO_API_KEY environment variable.")
            print("Sign up for FREE at: https://www.tiingo.com/")
            return None

        config = {'api_key': api_key}
        client = TiingoClient(config)

        print(f"Downloading {symbol} {interval} data from Tiingo...")

        if interval == '1d':
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * years)

            df = client.get_dataframe(
                symbol,
                startDate=start_date.strftime('%Y-%m-%d'),
                endDate=end_date.strftime('%Y-%m-%d')
            )
            period_str = f'{years}y'
        else:
            # Intraday limited to ~2000 recent ticks
            print("Warning: Intraday data limited to ~2000 most recent ticks on Tiingo free tier")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=14)  # Try last 2 weeks

            df = client.get_dataframe(
                symbol,
                frequency='5min',
                startDate=start_date.strftime('%Y-%m-%d'),
                endDate=end_date.strftime('%Y-%m-%d')
            )
            period_str = 'recent'

        df = df.reset_index()

        output_file = os.path.join(self.output_dir, f"{symbol}_{interval}_tiingo_{period_str}.csv")
        df.to_csv(output_file, index=False)

        print(f"✓ Downloaded {len(df):,} bars")
        print(f"✓ Saved to: {output_file}")
        print(f"✓ Size: {os.path.getsize(output_file) / 1024**2:.2f} MB")

        return df


def main():
    parser = argparse.ArgumentParser(
        description='Download free historical market data from multiple sources',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 2 years of SPY 1-minute data from Alpaca (BEST)
  python download_free_market_data.py --source alpaca --symbol SPY --interval 1min --years 2

  # Download 5 years of SPY daily data from Tiingo
  python download_free_market_data.py --source tiingo --symbol SPY --interval 1d --years 5

  # Download VIX data from CBOE (official source)
  python download_free_market_data.py --source cboe --symbol VIX

  # Download 60 days of SPY 5-min data from Yahoo Finance
  python download_free_market_data.py --source yfinance --symbol SPY --interval 5m

  # Download 2 years of SPY 1-min data from Polygon.io
  python download_free_market_data.py --source polygon --symbol SPY --interval 1min --years 2

Environment Variables:
  ALPACA_API_KEY      Alpaca API key
  ALPACA_SECRET_KEY   Alpaca secret key
  POLYGON_API_KEY     Polygon.io API key
  TIINGO_API_KEY      Tiingo API key

  Sign up for FREE at:
    - Alpaca: https://alpaca.markets/
    - Polygon: https://polygon.io/
    - Tiingo: https://www.tiingo.com/
        """
    )

    parser.add_argument('--source', required=True,
                       choices=['alpaca', 'polygon', 'yfinance', 'cboe', 'tiingo'],
                       help='Data source to use')
    parser.add_argument('--symbol', default='SPY',
                       help='Stock symbol (default: SPY)')
    parser.add_argument('--interval', default='1d',
                       help='Data interval (default: 1d)')
    parser.add_argument('--years', type=int, default=2,
                       help='Number of years of history (default: 2)')
    parser.add_argument('--output-dir', default='data/raw',
                       help='Output directory (default: data/raw)')

    args = parser.parse_args()

    downloader = DataDownloader(output_dir=args.output_dir)

    # Route to appropriate downloader
    if args.source == 'alpaca':
        downloader.download_alpaca(args.symbol, args.interval, args.years)
    elif args.source == 'polygon':
        downloader.download_polygon(args.symbol, args.interval, args.years)
    elif args.source == 'yfinance':
        downloader.download_yfinance(args.symbol, args.interval, args.years)
    elif args.source == 'cboe':
        if args.symbol.upper() == 'VIX' or args.symbol.upper() == '^VIX':
            downloader.download_cboe_vix()
        else:
            print("Error: CBOE source only supports VIX data")
            sys.exit(1)
    elif args.source == 'tiingo':
        downloader.download_tiingo(args.symbol, args.interval, args.years)

    print("\n✓ Done!")


if __name__ == '__main__':
    main()
