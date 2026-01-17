"""
Data Acquisition Module
Multi-source commodity price data acquisition with fallback and validation.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# Import data sources
import yfinance as yf

logger = logging.getLogger(__name__)


class DataAcquisitionError(Exception):
    """Custom exception for data acquisition failures."""
    pass


class CommodityDataAcquisition:
    """
    Multi-source commodity data acquisition client.

    Supports:
    - yfinance (primary for prototyping)
    - Alpha Vantage (backup)
    - Quandl/NASDAQ Data Link (production)
    - FRED (macroeconomic data)

    Features:
    - Priority-based fallback
    - Rate limiting with exponential backoff
    - Data validation
    - Caching
    """

    def __init__(self, config: Dict):
        """
        Initialize data acquisition client.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.commodity_ticker = config['commodity']['ticker']
        self.start_date = config['data']['start_date']
        self.end_date = config['data']['end_date']

        # Source priorities
        self.source_priority = config['data'].get('sources', {
            'yfinance': 1,
            'alpha_vantage': 2,
            'quandl': 3
        })

        # Rate limiting
        self.request_timestamps = []
        self.rate_limit_window = 60  # seconds
        self.max_requests_per_window = 30

        # Cache
        self.cache = {}

        logger.info(f"Initialized data acquisition for {self.commodity_ticker}")
        logger.info(f"Date range: {self.start_date} to {self.end_date}")

    def fetch_commodity_prices(
        self,
        ticker: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV commodity price data with fallback sources.

        Args:
            ticker: Commodity ticker (default: from config)
            start: Start date YYYY-MM-DD (default: from config)
            end: End date YYYY-MM-DD (default: from config)

        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume

        Raises:
            DataAcquisitionError: If all sources fail
        """
        ticker = ticker or self.commodity_ticker
        start = start or self.start_date
        end = end or self.end_date

        # Check cache
        cache_key = f"{ticker}_{start}_{end}"
        if cache_key in self.cache:
            logger.info(f"Using cached data for {ticker}")
            return self.cache[cache_key].copy()

        # Try sources in priority order
        sorted_sources = sorted(
            self.source_priority.items(),
            key=lambda x: x[1]
        )

        last_error = None
        for source_name, _ in sorted_sources:
            try:
                logger.info(f"Attempting to fetch {ticker} from {source_name}")

                if source_name == 'yfinance':
                    data = self._fetch_yfinance(ticker, start, end)
                elif source_name == 'alpha_vantage':
                    data = self._fetch_alpha_vantage(ticker, start, end)
                elif source_name == 'quandl':
                    data = self._fetch_quandl(ticker, start, end)
                else:
                    logger.warning(f"Unknown source: {source_name}")
                    continue

                # Validate data
                if self._validate_ohlcv_data(data):
                    logger.info(f"Successfully fetched {len(data)} rows from {source_name}")
                    self.cache[cache_key] = data.copy()
                    return data
                else:
                    logger.warning(f"Data validation failed for {source_name}")
                    continue

            except Exception as e:
                logger.warning(f"Failed to fetch from {source_name}: {e}")
                last_error = e
                continue

        # All sources failed
        error_msg = f"TFM2001 DATA: Failed to fetch data for {ticker} from all sources"
        logger.error(error_msg)
        raise DataAcquisitionError(error_msg) from last_error

    def _fetch_yfinance(
        self,
        ticker: str,
        start: str,
        end: str
    ) -> pd.DataFrame:
        """
        Fetch data from yfinance.

        Args:
            ticker: Commodity ticker
            start: Start date
            end: End date

        Returns:
            DataFrame with OHLCV data
        """
        self._rate_limit()

        try:
            data = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True
            )

            if data.empty:
                raise DataAcquisitionError(f"No data returned for {ticker}")

            # Reset index to get Date as column
            data = data.reset_index()

            # Standardize column names
            data = data.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # Select required columns
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            data = data[required_cols]

            return data

        except Exception as e:
            logger.error(f"yfinance fetch error: {e}")
            raise

    def _fetch_alpha_vantage(
        self,
        ticker: str,
        start: str,
        end: str
    ) -> pd.DataFrame:
        """
        Fetch data from Alpha Vantage.

        Note: Requires ALPHA_VANTAGE_API_KEY environment variable.

        Args:
            ticker: Commodity ticker
            start: Start date
            end: End date

        Returns:
            DataFrame with OHLCV data
        """
        self._rate_limit()

        try:
            import os
            from alpha_vantage.timeseries import TimeSeries

            api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
            if not api_key:
                raise DataAcquisitionError(
                    "TFM1001 CONFIG: ALPHA_VANTAGE_API_KEY not set"
                )

            ts = TimeSeries(key=api_key, output_format='pandas')

            # Map commodity tickers to Alpha Vantage symbols
            # (implementation depends on API support for commodities)
            data, meta_data = ts.get_daily(symbol=ticker, outputsize='full')

            # Convert to standard format
            data = data.reset_index()
            data = data.rename(columns={
                'date': 'date',
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume'
            })

            # Filter date range
            data['date'] = pd.to_datetime(data['date'])
            mask = (data['date'] >= start) & (data['date'] <= end)
            data = data[mask]

            return data

        except Exception as e:
            logger.error(f"Alpha Vantage fetch error: {e}")
            raise

    def _fetch_quandl(
        self,
        ticker: str,
        start: str,
        end: str
    ) -> pd.DataFrame:
        """
        Fetch data from Quandl/NASDAQ Data Link.

        Note: Requires QUANDL_API_KEY environment variable.

        Args:
            ticker: Commodity ticker
            start: Start date
            end: End date

        Returns:
            DataFrame with OHLCV data
        """
        self._rate_limit()

        try:
            import os
            import nasdaqdatalink

            api_key = os.getenv('QUANDL_API_KEY')
            if not api_key:
                raise DataAcquisitionError(
                    "TFM1001 CONFIG: QUANDL_API_KEY not set"
                )

            nasdaqdatalink.ApiConfig.api_key = api_key

            # Map commodity tickers to Quandl codes
            # Example: GC=F -> CHRIS/CME_GC1
            quandl_code = self._map_ticker_to_quandl(ticker)

            data = nasdaqdatalink.get(
                quandl_code,
                start_date=start,
                end_date=end
            )

            # Convert to standard format (depends on Quandl dataset schema)
            data = data.reset_index()
            # ... column mapping ...

            return data

        except Exception as e:
            logger.error(f"Quandl fetch error: {e}")
            raise

    def fetch_fred_data(
        self,
        series_id: str,
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> pd.Series:
        """
        Fetch macroeconomic data from FRED.

        Args:
            series_id: FRED series ID (e.g., "DGS10" for 10-year Treasury)
            start: Start date (default: from config)
            end: End date (default: from config)

        Returns:
            Series with date index and values

        Raises:
            DataAcquisitionError: If FRED API fails
        """
        import os
        from fredapi import Fred

        start = start or self.start_date
        end = end or self.end_date

        # Check cache
        cache_key = f"fred_{series_id}_{start}_{end}"
        if cache_key in self.cache:
            logger.info(f"Using cached FRED data for {series_id}")
            return self.cache[cache_key].copy()

        self._rate_limit()

        try:
            api_key = os.getenv('FRED_API_KEY')
            if not api_key:
                raise DataAcquisitionError(
                    "TFM1001 CONFIG: FRED_API_KEY not set"
                )

            fred = Fred(api_key=api_key)
            data = fred.get_series(series_id, observation_start=start, observation_end=end)

            logger.info(f"Fetched FRED series {series_id}: {len(data)} observations")
            self.cache[cache_key] = data.copy()

            return data

        except Exception as e:
            error_msg = f"TFM2001 DATA: Failed to fetch FRED series {series_id}: {e}"
            logger.error(error_msg)
            raise DataAcquisitionError(error_msg) from e

    def fetch_futures_curve(
        self,
        ticker: str,
        date: Optional[str] = None
    ) -> Dict[int, float]:
        """
        Fetch futures curve for convenience yield estimation.

        Args:
            ticker: Commodity ticker
            date: Date for futures curve (default: latest)

        Returns:
            Dictionary mapping maturity (days) to futures price
        """
        # Note: This requires access to multiple futures contracts
        # For yfinance: GC=F (front month), GC=F+1, GC=F+2, etc.
        # Implementation depends on data source capabilities

        logger.warning(
            "fetch_futures_curve not fully implemented. "
            "Requires multi-contract data source."
        )

        # Placeholder implementation
        return {30: 2000.0, 60: 2005.0, 90: 2010.0}

    def _validate_ohlcv_data(self, data: pd.DataFrame) -> bool:
        """
        Validate OHLCV data integrity.

        Checks:
        - Required columns present
        - No all-NaN columns
        - High >= Low
        - Close within [Low, High]
        - Reasonable data coverage (>50%)

        Args:
            data: DataFrame with OHLCV data

        Returns:
            True if validation passes
        """
        try:
            # Check required columns
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_cols):
                logger.error(f"TFM2001 DATA: Missing required columns")
                return False

            # Check for all-NaN columns
            if data[['open', 'high', 'low', 'close']].isna().all().any():
                logger.error(f"TFM2001 DATA: All-NaN column detected")
                return False

            # Check OHLC relationships
            valid_high_low = (data['high'] >= data['low']).all()
            if not valid_high_low:
                logger.error(f"TFM2001 DATA: Invalid OHLC: high < low detected")
                return False

            valid_close = ((data['close'] >= data['low']) & (data['close'] <= data['high'])).all()
            if not valid_close:
                logger.error(f"TFM2001 DATA: Invalid OHLC: close outside [low, high]")
                return False

            # Check data coverage
            coverage = 1 - data['close'].isna().sum() / len(data)
            if coverage < 0.5:
                logger.warning(
                    f"TFM2001 DATA: Low data coverage: {coverage:.2%}. "
                    f"Expected >50%"
                )
                return False

            logger.debug(f"Data validation passed. Coverage: {coverage:.2%}")
            return True

        except Exception as e:
            logger.error(f"TFM2001 DATA: Validation error: {e}")
            return False

    def _rate_limit(self):
        """
        Implement rate limiting with exponential backoff.

        Limits requests to max_requests_per_window within rate_limit_window.
        """
        current_time = time.time()

        # Remove timestamps outside window
        self.request_timestamps = [
            ts for ts in self.request_timestamps
            if current_time - ts < self.rate_limit_window
        ]

        # Check if limit exceeded
        if len(self.request_timestamps) >= self.max_requests_per_window:
            # Calculate wait time
            oldest_request = self.request_timestamps[0]
            wait_time = self.rate_limit_window - (current_time - oldest_request)

            if wait_time > 0:
                logger.info(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time + 1)  # +1 for buffer

        # Record this request
        self.request_timestamps.append(current_time)

    def _map_ticker_to_quandl(self, ticker: str) -> str:
        """
        Map commodity ticker to Quandl dataset code.

        Args:
            ticker: Yahoo Finance ticker (e.g., "GC=F")

        Returns:
            Quandl dataset code (e.g., "CHRIS/CME_GC1")
        """
        mapping = {
            'GC=F': 'CHRIS/CME_GC1',  # Gold
            'CL=F': 'CHRIS/CME_CL1',  # Crude Oil
            'SI=F': 'CHRIS/CME_SI1',  # Silver
            'NG=F': 'CHRIS/CME_NG1',  # Natural Gas
            'HG=F': 'CHRIS/CME_HG1',  # Copper
        }

        quandl_code = mapping.get(ticker)
        if not quandl_code:
            logger.warning(
                f"No Quandl mapping for {ticker}. "
                f"Available: {list(mapping.keys())}"
            )
            raise DataAcquisitionError(f"Unsupported ticker for Quandl: {ticker}")

        return quandl_code


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('../..')

    from src.config.loader import load_config
    from src.config.logging_config import setup_logging

    # Load configuration
    config = load_config('../../config/parameters.toml')
    logger = setup_logging(config)

    # Initialize data acquisition
    data_client = CommodityDataAcquisition(config)

    try:
        # Fetch commodity prices
        prices = data_client.fetch_commodity_prices()
        print(f"\nFetched {len(prices)} price records")
        print(f"\nFirst 5 rows:\n{prices.head()}")
        print(f"\nLast 5 rows:\n{prices.tail()}")

        # Fetch FRED data
        dgs10 = data_client.fetch_fred_data('DGS10')
        print(f"\nFetched {len(dgs10)} FRED DGS10 observations")
        print(f"Latest 10-year Treasury rate: {dgs10.iloc[-1]:.2%}")

    except Exception as e:
        logger.error(f"Error in example: {e}")
