"""
Data Manager for Volatility Prediction UI
==========================================
Handles data loading, caching, and feature engineering for the Streamlit UI.

Components:
- Data downloading from Alpaca and Yahoo Finance
- Feature engineering pipeline
- Data caching for performance
- Data validation and preprocessing

Author: Claude + User
Date: 2026-01-17
"""

import sys
import os
import logging
from typing import Tuple, Optional
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st

# Add src to path for imports
# Use realpath to handle symlinks correctly
current_file = os.path.realpath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
sys.path.insert(0, os.path.join(project_root, 'src'))

from data.alpaca_client import AlpacaDataClient
from data.volatility_features import VolatilityFeatureEngineer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataManager:
    """
    Manages data loading and caching for the UI.

    Uses Streamlit caching decorators to avoid re-downloading data
    on every interaction.
    """

    def __init__(self, alpaca_key: str = None, alpaca_secret: str = None):
        """
        Initialize DataManager.

        Args:
            alpaca_key: Alpaca API key (default: from env var ALPACA_API_KEY)
            alpaca_secret: Alpaca secret key (default: from env var ALPACA_SECRET_KEY)
        """
        self.alpaca_client = AlpacaDataClient(alpaca_key, alpaca_secret)
        self.feature_engineer = VolatilityFeatureEngineer()
        logger.info("DataManager initialized")

    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def load_spy_data(_self, days: int = 180, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Load SPY daily bars with caching.

        Args:
            days: Number of historical days to load
            end_date: Load data UP TO this date (None = today/now)

        Returns:
            DataFrame with OHLCV data
        """
        end_str = end_date.strftime('%Y-%m-%d') if end_date else 'now'
        logger.info(f"Loading {days} days of SPY data ending at {end_str}")
        spy_daily = _self.alpaca_client.get_daily_bars('SPY', days=days, end_date=end_date)
        logger.info(f"Loaded {len(spy_daily)} SPY bars")
        return spy_daily

    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def load_vix_data(_self, days: int = 180, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Load VIX data from Yahoo Finance with caching.

        Args:
            days: Number of historical days to load
            end_date: Load data UP TO this date (None = today/now)

        Returns:
            DataFrame with VIX OHLCV data
        """
        import yfinance as yf

        # Calculate date range
        if end_date is None:
            end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        end_str = end_date.strftime('%Y-%m-%d')
        start_str = start_date.strftime('%Y-%m-%d')
        logger.info(f"Loading VIX data from {start_str} to {end_str}")

        # Use explicit date range instead of period for precise control
        vix_daily = yf.download(
            '^VIX',
            start=start_str,
            end=end_str,
            progress=False
        )

        # Standardize column names
        if isinstance(vix_daily.columns, pd.MultiIndex):
            vix_daily.columns = [col[0].lower() if isinstance(col, tuple) else col.lower()
                                  for col in vix_daily.columns]
        else:
            vix_daily.columns = [col.lower() for col in vix_daily.columns]

        logger.info(f"Loaded {len(vix_daily)} VIX bars")
        return vix_daily

    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def engineer_features(
        _self,
        spy_data: pd.DataFrame,
        vix_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply feature engineering pipeline with caching.

        Args:
            spy_data: SPY OHLCV DataFrame
            vix_data: VIX OHLCV DataFrame

        Returns:
            DataFrame with all engineered features
        """
        logger.info("Engineering features")
        features_df = _self.feature_engineer.add_all_features(spy_data, vix_data)
        logger.info(f"Created {len(features_df.columns)} features on {len(features_df)} rows")
        return features_df

    def load_complete_dataset(
        self,
        days: int = 180,
        end_date: Optional[datetime] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load complete dataset: raw SPY, raw VIX, and engineered features.

        Args:
            days: Number of historical days to load
            end_date: Load data UP TO this date (None = today/now).
                     For backtesting, set this to day BEFORE the prediction date
                     to prevent data leakage.

        Returns:
            Tuple of (spy_data, vix_data, features_df)
        """
        end_str = end_date.strftime('%Y-%m-%d') if end_date else 'now'
        logger.info(f"Loading complete dataset: {days} days ending at {end_str}")

        spy_data = self.load_spy_data(days, end_date=end_date)
        vix_data = self.load_vix_data(days, end_date=end_date)
        features_df = self.engineer_features(spy_data, vix_data)

        return spy_data, vix_data, features_df

    def get_latest_data_date(self, features_df: pd.DataFrame) -> datetime:
        """
        Get the date of the latest data in the features DataFrame.

        Args:
            features_df: Features DataFrame

        Returns:
            Latest date as datetime
        """
        return features_df.index[-1]

    def get_data_summary(self, features_df: pd.DataFrame) -> dict:
        """
        Get summary statistics about the loaded data.

        Args:
            features_df: Features DataFrame

        Returns:
            Dictionary with summary stats
        """
        return {
            'total_rows': len(features_df),
            'start_date': features_df.index[0].strftime('%Y-%m-%d'),
            'end_date': features_df.index[-1].strftime('%Y-%m-%d'),
            'total_features': len(features_df.columns),
            'missing_values': features_df.isnull().sum().sum()
        }
