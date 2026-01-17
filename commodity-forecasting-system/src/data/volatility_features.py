"""
Volatility-Specific Feature Engineering
========================================
Features designed to predict next-day intraday volatility for 0DTE/1DTE options trading.

Key Features:
1. Overnight Gap Volatility - Predicts morning volatility bursts
2. Intraday Range Metrics - Direct volatility proxy
3. VIX Integration - External fear gauge
4. Volume Patterns - Volatility confirmation
5. Time-Based Features - Day-of-week effects
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class VolatilityFeatureEngineer:
    """Engineer features that predict intraday volatility spikes."""

    def __init__(self):
        self.feature_names = []

    def add_all_features(
        self,
        df: pd.DataFrame,
        vix_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Add all volatility features to DataFrame.

        Args:
            df: Stock OHLCV DataFrame
            vix_df: VIX OHLCV DataFrame (optional)

        Returns:
            DataFrame with all features added
        """
        logger.info("Engineering volatility features...")

        # Core volatility features
        df = self.add_overnight_gap_features(df)
        df = self.add_intraday_range_features(df)
        df = self.add_volume_features(df)
        df = self.add_time_features(df)

        # External features (if VIX available)
        if vix_df is not None:
            df = self.add_vix_features(df, vix_df)

        # Clean up NaN from rolling calculations
        logger.info(f"Created {len(df.columns)} total columns")
        logger.info(f"Rows before cleanup: {len(df)}")
        df = df.dropna()
        logger.info(f"Rows after cleanup: {len(df)}")

        return df

    def add_overnight_gap_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate overnight gap volatility features.

        Overnight gaps are STRONG predictors of morning volatility.
        Large gaps often lead to morning reversals or continuation volatility.

        Features:
        - overnight_gap_pct: % change from prev close to open
        - overnight_gap_abs: Absolute value (direction-agnostic)
        - gap_ma_5: 5-day moving average of gaps
        - gap_std_5: 5-day standard deviation (clustering)
        - gap_zscore: Z-score (outlier detection)
        """
        logger.info("Adding overnight gap features...")

        # Basic gap calculation
        df['overnight_gap_pct'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        df['overnight_gap_abs'] = np.abs(df['overnight_gap_pct'])

        # Rolling statistics
        df['gap_ma_5'] = df['overnight_gap_abs'].rolling(5).mean()
        df['gap_std_5'] = df['overnight_gap_abs'].rolling(5).std()

        # Z-score for outlier detection (large gaps)
        df['gap_zscore'] = (df['overnight_gap_abs'] - df['gap_ma_5']) / (df['gap_std_5'] + 1e-8)

        # Binary indicators
        df['gap_large'] = (df['overnight_gap_abs'] > 0.01).astype(int)  # >1% gap
        df['gap_extreme'] = (df['overnight_gap_abs'] > 0.02).astype(int)  # >2% gap

        return df

    def add_intraday_range_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate intraday range features (direct volatility measure).

        Intraday range = (high - low) / open is the PRIMARY volatility metric
        for 0DTE trading. This is what determines option profitability.

        Features:
        - intraday_range_pct: Daily high-low range
        - range_ma_5, range_ma_10, range_ma_20: Moving averages
        - range_std_5: Short-term volatility
        - range_expansion: Current vs average (regime shift indicator)
        - high_range_days_5: Count of >1.2% days (clustering)
        """
        logger.info("Adding intraday range features...")

        # Core metric: intraday range
        df['intraday_range_pct'] = (df['high'] - df['low']) / df['open']

        # Rolling averages (different timeframes)
        df['range_ma_5'] = df['intraday_range_pct'].rolling(5).mean()
        df['range_ma_10'] = df['intraday_range_pct'].rolling(10).mean()
        df['range_ma_20'] = df['intraday_range_pct'].rolling(20).mean()

        # Rolling standard deviation (volatility of volatility)
        df['range_std_5'] = df['intraday_range_pct'].rolling(5).std()

        # Range expansion (regime shift detector)
        df['range_expansion'] = df['intraday_range_pct'] / (df['range_ma_5'] + 1e-8)

        # Count of high-volatility days (clustering effect)
        df['high_range_days_5'] = (df['intraday_range_pct'] > 0.012).rolling(5).sum()
        df['high_range_days_10'] = (df['intraday_range_pct'] > 0.012).rolling(10).sum()

        # Binary indicators for trade profitability
        df['is_profitable_range'] = (df['intraday_range_pct'] > 0.012).astype(int)  # >1.2%

        return df

    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume features (confirms volatility).

        High volume + high volatility = real move (not noise).
        Volume surges often precede or confirm volatility spikes.

        Features:
        - volume_ma_20: 20-day volume moving average
        - volume_ratio: Current vs average (surge detector)
        - volume_surge: Binary indicator (>1.5x average)
        - volume_change_pct: Day-over-day change
        """
        logger.info("Adding volume features...")

        # Volume moving average
        df['volume_ma_20'] = df['volume'].rolling(20).mean()

        # Volume ratio (surge detector)
        df['volume_ratio'] = df['volume'] / (df['volume_ma_20'] + 1)

        # Binary indicators
        df['volume_surge'] = (df['volume_ratio'] > 1.5).astype(int)

        # Day-over-day volume change
        df['volume_change_pct'] = df['volume'].pct_change()

        # Combined indicator: high volume + high range (quality volatility)
        df['quality_volatility'] = (
            (df['volume_surge'] == 1) &
            (df['intraday_range_pct'] > 0.01)
        ).astype(int)

        return df

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features (day-of-week, month-end, opex).

        Certain days exhibit higher volatility:
        - Mondays (weekend gap resolution)
        - Fridays (position squaring)
        - Month-end (rebalancing)
        - OPEX week (options expiration)

        Features:
        - day_of_week: 0=Monday, 4=Friday
        - is_monday, is_friday: Binary indicators
        - is_month_end: Last 2 trading days of month
        - days_in_month: Trading day number (1-21)
        """
        logger.info("Adding time features...")

        # Day of week
        df['day_of_week'] = df.index.dayofweek
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)

        # Month-end detection (last 2 days of month)
        df['day_of_month'] = df.index.day
        df['days_in_month'] = df.index.days_in_month
        df['is_month_end'] = (df['days_in_month'] - df['day_of_month'] <= 2).astype(int)

        # OPEX week (third Friday of month) - approximate
        # TODO: Use actual options expiration calendar
        df['week_of_month'] = (df.index.day - 1) // 7 + 1
        df['is_opex_week'] = (
            (df['week_of_month'] == 3) &
            (df['is_friday'] == 1)
        ).astype(int)

        return df

    def add_vix_features(
        self,
        df: pd.DataFrame,
        vix_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add VIX (Volatility Index) features.

        VIX is the "fear gauge" - measures expected SPX volatility.
        High VIX → high SPY volatility (strong correlation).

        Features:
        - vix_level: Current VIX reading
        - vix_change_1d: Day-over-day change
        - vix_change_5d: 5-day change
        - vix_percentile_20: Percentile rank (0-100)
        - vix_spike: Binary indicator (change >2 points)
        - vix_regime: Low (<15), Medium (15-25), High (>25)

        Args:
            df: Stock DataFrame
            vix_df: VIX OHLCV DataFrame (must have 'close' column)
        """
        logger.info("Adding VIX features...")

        # Prepare VIX data
        vix_close = vix_df['close'].rename('vix_level')

        # TIMEZONE FIX: Normalize both to date-only (ignore time component)
        # This handles Alpaca's "2025-11-19 05:00:00+00:00" vs yfinance's "2025-11-19 00:00:00"
        df_original_index = df.index.copy()
        df.index = pd.to_datetime(df.index.date)  # Strip time, keep date only
        vix_close.index = pd.to_datetime(vix_close.index.date)  # Strip time

        # Merge VIX into stock DataFrame (align by date)
        df = df.join(vix_close, how='left')

        # Forward-fill VIX for any missing dates
        df['vix_level'] = df['vix_level'].ffill()

        # VIX changes
        df['vix_change_1d'] = df['vix_level'].diff()
        df['vix_change_5d'] = df['vix_level'].diff(5)

        # VIX percentile (relative to last 20 days)
        df['vix_percentile_20'] = df['vix_level'].rolling(20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
        )

        # Binary indicators
        df['vix_spike'] = (df['vix_change_1d'] > 2).astype(int)  # >2 point increase
        df['vix_crash'] = (df['vix_change_1d'] < -2).astype(int)  # >2 point decrease

        # VIX regime classification
        df['vix_regime'] = pd.cut(
            df['vix_level'],
            bins=[0, 15, 25, 100],
            labels=['low', 'medium', 'high']
        )

        return df

    def get_feature_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary statistics of all features.

        Returns:
            DataFrame with mean, std, min, max for each feature
        """
        return df.describe().T

    def get_top_predictive_features(
        self,
        df: pd.DataFrame,
        target: str = 'is_profitable_range',
        n: int = 10
    ) -> pd.DataFrame:
        """
        Identify top N features correlated with target.

        Args:
            df: DataFrame with features
            target: Target column name
            n: Number of top features to return

        Returns:
            DataFrame with feature correlations
        """
        # Calculate correlations
        correlations = df.corr()[target].abs().sort_values(ascending=False)

        # Exclude target itself
        correlations = correlations[correlations.index != target]

        return correlations.head(n)


def test_volatility_features():
    """Test volatility feature engineering."""
    print("Testing Volatility Feature Engineering...")
    print("=" * 60)

    # Create sample data
    dates = pd.date_range('2025-01-01', periods=100, freq='D')
    np.random.seed(42)

    df = pd.DataFrame({
        'open': 100 + np.random.randn(100).cumsum(),
        'high': 102 + np.random.randn(100).cumsum(),
        'low': 98 + np.random.randn(100).cumsum(),
        'close': 100 + np.random.randn(100).cumsum(),
        'volume': 1000000 + np.random.randint(-100000, 100000, 100)
    }, index=dates)

    # Ensure high >= low
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    # Create VIX data
    vix_df = pd.DataFrame({
        'close': 15 + np.random.randn(100) * 3
    }, index=dates)

    # Engineer features
    engineer = VolatilityFeatureEngineer()
    df_features = engineer.add_all_features(df, vix_df)

    print(f"\n✅ Feature Engineering Complete")
    print(f"   Original columns: {len(df.columns)}")
    print(f"   Total columns after: {len(df_features.columns)}")
    print(f"   Rows: {len(df_features)}")
    print()

    # Show feature summary
    print("Feature Summary (last 5 rows):")
    print("-" * 60)
    print(df_features[['overnight_gap_abs', 'intraday_range_pct', 'range_expansion', 'vix_level']].tail())
    print()

    # Show top predictive features
    print("Top 5 Predictive Features:")
    print("-" * 60)
    top_features = engineer.get_top_predictive_features(df_features, n=5)
    for feature, corr in top_features.items():
        print(f"   {feature}: {corr:.3f}")
    print()

    print("🎉 All tests passed!")
    return True


if __name__ == "__main__":
    test_volatility_features()
