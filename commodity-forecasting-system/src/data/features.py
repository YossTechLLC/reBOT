"""
Feature Engineering Module
Technical indicators, lagged features, and macroeconomic integration.
"""

import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import ta

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Comprehensive feature engineering for commodity forecasting.

    Features:
    - Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
    - Lagged features (autoregressive terms)
    - Macroeconomic features (FRED integration)
    - Regime features (HMM state one-hot encoding)
    - Custom derived features
    """

    def __init__(self, config: Dict):
        """
        Initialize feature engineer with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.features_config = config.get('features', {})

        # Parameters
        self.indicators = self.features_config.get('indicators', [])
        self.fred_series = self.features_config.get('fred_series', [])
        self.max_lags = self.features_config.get('max_lags', 10)

        logger.info("Initialized feature engineer")
        logger.info(f"Technical indicators: {len(self.indicators)}")
        logger.info(f"FRED series: {len(self.fred_series)}")
        logger.info(f"Max lags: {self.max_lags}")

    def engineer_features(
        self,
        price_data: pd.DataFrame,
        fred_data: Optional[Dict[str, pd.Series]] = None,
        hmm_states: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.

        Args:
            price_data: OHLCV price data
            fred_data: Optional dict of FRED series {series_id: Series}
            hmm_states: Optional HMM regime states

        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering...")

        # Make copy to avoid modifying original
        features = price_data.copy()

        # 1. Technical indicators
        features = self.add_technical_indicators(features)

        # 2. Lagged features
        features = self.add_lagged_features(features)

        # 3. Macroeconomic features
        if fred_data:
            features = self.add_macroeconomic_features(features, fred_data)

        # 4. Regime features
        if hmm_states is not None:
            features = self.add_regime_features(features, hmm_states)

        # 5. Custom derived features
        features = self.add_custom_features(features)

        # Remove rows with NaN (from lagged features)
        initial_len = len(features)
        features = features.dropna()
        rows_removed = initial_len - len(features)

        logger.info(f"Feature engineering complete: {len(features.columns)} features")
        logger.info(f"Rows removed due to NaN: {rows_removed}")

        return features

    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators using ta library.

        Supported indicators:
        - SMA (Simple Moving Average)
        - EMA (Exponential Moving Average)
        - RSI (Relative Strength Index)
        - MACD (Moving Average Convergence Divergence)
        - Bollinger Bands
        - ATR (Average True Range)
        - OBV (On-Balance Volume)

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame with technical indicators
        """
        logger.info("Adding technical indicators...")

        for indicator in self.indicators:
            try:
                # Parse indicator string (e.g., "sma_20")
                indicator_type, *params = indicator.split('_')

                if indicator_type == 'sma':
                    # Simple Moving Average
                    window = int(params[0]) if params else 20
                    data[f'sma_{window}'] = ta.trend.sma_indicator(
                        data['close'], window=window
                    )
                    logger.debug(f"Added SMA({window})")

                elif indicator_type == 'ema':
                    # Exponential Moving Average
                    window = int(params[0]) if params else 12
                    data[f'ema_{window}'] = ta.trend.ema_indicator(
                        data['close'], window=window
                    )
                    logger.debug(f"Added EMA({window})")

                elif indicator_type == 'rsi':
                    # Relative Strength Index
                    window = int(params[0]) if params else 14
                    data[f'rsi_{window}'] = ta.momentum.rsi(
                        data['close'], window=window
                    )
                    logger.debug(f"Added RSI({window})")

                elif indicator_type == 'macd':
                    # MACD
                    macd = ta.trend.MACD(data['close'])
                    data['macd'] = macd.macd()
                    data['macd_signal'] = macd.macd_signal()
                    data['macd_diff'] = macd.macd_diff()
                    logger.debug("Added MACD")

                elif indicator == 'bollinger_bands':
                    # Bollinger Bands
                    bb = ta.volatility.BollingerBands(data['close'])
                    data['bb_high'] = bb.bollinger_hband()
                    data['bb_mid'] = bb.bollinger_mavg()
                    data['bb_low'] = bb.bollinger_lband()
                    data['bb_width'] = bb.bollinger_wband()
                    logger.debug("Added Bollinger Bands")

                elif indicator_type == 'atr':
                    # Average True Range
                    window = int(params[0]) if params else 14
                    data[f'atr_{window}'] = ta.volatility.average_true_range(
                        data['high'], data['low'], data['close'], window=window
                    )
                    logger.debug(f"Added ATR({window})")

                elif indicator == 'obv':
                    # On-Balance Volume
                    data['obv'] = ta.volume.on_balance_volume(
                        data['close'], data['volume']
                    )
                    logger.debug("Added OBV")

                else:
                    logger.warning(f"Unknown indicator: {indicator}")

            except Exception as e:
                logger.error(f"Failed to add indicator {indicator}: {e}")
                continue

        logger.info(f"Added {len(self.indicators)} technical indicators")
        return data

    def add_lagged_features(
        self,
        data: pd.DataFrame,
        n_lags: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Add lagged features (autoregressive terms).

        Args:
            data: DataFrame with features
            n_lags: Number of lags (default: from config)

        Returns:
            DataFrame with lagged features
        """
        n_lags = n_lags or self.max_lags
        logger.info(f"Adding {n_lags} lagged features...")

        # Key features to lag
        features_to_lag = ['returns', 'close', 'volume']

        for feature in features_to_lag:
            if feature not in data.columns:
                logger.warning(f"Feature {feature} not found, skipping lags")
                continue

            for lag in range(1, n_lags + 1):
                data[f'{feature}_lag_{lag}'] = data[feature].shift(lag)

        logger.info(f"Added {n_lags * len(features_to_lag)} lagged features")
        return data

    def add_macroeconomic_features(
        self,
        price_data: pd.DataFrame,
        fred_data: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """
        Merge FRED macroeconomic data with price data.

        Args:
            price_data: OHLCV price DataFrame
            fred_data: Dict mapping FRED series ID to Series

        Returns:
            DataFrame with macro features
        """
        logger.info("Adding macroeconomic features...")

        # Ensure price_data has date index
        if 'date' in price_data.columns:
            price_data = price_data.set_index('date')

        merged = price_data.copy()

        for series_id, series in fred_data.items():
            # Align dates (FRED data may be at different frequency)
            # Forward fill to match daily price data
            series = series.reindex(merged.index, method='ffill')

            # Add to dataframe
            merged[f'fred_{series_id.lower()}'] = series

            logger.debug(f"Added FRED series: {series_id}")

        logger.info(f"Added {len(fred_data)} macroeconomic features")

        # Reset index if needed
        if price_data.index.name == 'date':
            merged = merged.reset_index()

        return merged

    def add_regime_features(
        self,
        data: pd.DataFrame,
        hmm_states: pd.Series
    ) -> pd.DataFrame:
        """
        Add HMM regime states as one-hot encoded features.

        Args:
            data: DataFrame with features
            hmm_states: Series with HMM state labels

        Returns:
            DataFrame with regime features
        """
        logger.info("Adding regime features...")

        # Align indices
        if len(hmm_states) != len(data):
            logger.warning(
                f"TFM2001 DATA: HMM states length ({len(hmm_states)}) "
                f"!= data length ({len(data)}). Attempting alignment..."
            )

            # Assume hmm_states has same index as data
            hmm_states = hmm_states.reindex(data.index)

        # One-hot encode regime states
        unique_states = hmm_states.unique()
        for state in unique_states:
            data[f'regime_{state}'] = (hmm_states == state).astype(int)

        logger.info(f"Added {len(unique_states)} regime features")
        return data

    def add_custom_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add custom derived features.

        Custom features:
        - Price momentum (rate of change)
        - Volatility (rolling std of returns)
        - Volume momentum
        - High-Low spread
        - Close relative to High-Low range

        Args:
            data: DataFrame with features

        Returns:
            DataFrame with custom features
        """
        logger.info("Adding custom features...")

        # Price momentum (rate of change)
        if 'close' in data.columns:
            data['price_momentum_5'] = data['close'].pct_change(periods=5)
            data['price_momentum_20'] = data['close'].pct_change(periods=20)

        # Volatility (rolling std of returns)
        if 'returns' in data.columns:
            data['volatility_5'] = data['returns'].rolling(window=5).std()
            data['volatility_20'] = data['returns'].rolling(window=20).std()

        # Volume momentum
        if 'volume' in data.columns:
            data['volume_momentum'] = data['volume'].pct_change(periods=5)
            data['volume_ma_ratio'] = data['volume'] / data['volume'].rolling(window=20).mean()

        # High-Low spread
        if 'high' in data.columns and 'low' in data.columns:
            data['hl_spread'] = data['high'] - data['low']
            data['hl_spread_pct'] = (data['high'] - data['low']) / data['close']

        # Close relative to High-Low range (Williams %R style)
        if all(col in data.columns for col in ['high', 'low', 'close']):
            high_max = data['high'].rolling(window=14).max()
            low_min = data['low'].rolling(window=14).min()
            data['close_hl_ratio'] = (high_max - data['close']) / (high_max - low_min)

        logger.info("Added custom derived features")
        return data

    def select_features(
        self,
        data: pd.DataFrame,
        method: str = 'correlation',
        threshold: float = 0.95
    ) -> pd.DataFrame:
        """
        Feature selection to remove redundant or low-variance features.

        Methods:
        - correlation: Remove highly correlated features (>threshold)
        - variance: Remove low-variance features

        Args:
            data: DataFrame with features
            method: Selection method
            threshold: Threshold for selection

        Returns:
            DataFrame with selected features
        """
        logger.info(f"Performing feature selection ({method})...")

        initial_features = len(data.columns)

        if method == 'correlation':
            # Remove highly correlated features
            numeric_data = data.select_dtypes(include=[np.number])
            corr_matrix = numeric_data.corr().abs()

            # Upper triangle of correlation matrix
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            # Find features with correlation > threshold
            to_drop = [
                column for column in upper_triangle.columns
                if any(upper_triangle[column] > threshold)
            ]

            data = data.drop(columns=to_drop)
            logger.info(f"Removed {len(to_drop)} highly correlated features")

        elif method == 'variance':
            # Remove low-variance features
            numeric_data = data.select_dtypes(include=[np.number])
            variances = numeric_data.var()

            low_variance_features = variances[variances < threshold].index
            data = data.drop(columns=low_variance_features)
            logger.info(f"Removed {len(low_variance_features)} low-variance features")

        final_features = len(data.columns)
        logger.info(f"Feature selection complete: {initial_features} -> {final_features}")

        return data

    def get_feature_importance(
        self,
        data: pd.DataFrame,
        target: str = 'returns'
    ) -> pd.Series:
        """
        Calculate feature importance using Random Forest.

        Args:
            data: DataFrame with features
            target: Target column name

        Returns:
            Series with feature importances
        """
        from sklearn.ensemble import RandomForestRegressor

        logger.info("Calculating feature importance...")

        # Prepare data
        features = data.drop(columns=[target])
        target_values = data[target]

        # Remove NaN
        valid_idx = ~(features.isna().any(axis=1) | target_values.isna())
        features = features[valid_idx]
        target_values = target_values[valid_idx]

        # Fit Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(features, target_values)

        # Get importances
        importances = pd.Series(
            rf.feature_importances_,
            index=features.columns
        ).sort_values(ascending=False)

        logger.info("Feature importance calculation complete")
        logger.info(f"Top 5 features:\n{importances.head()}")

        return importances


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('../..')

    from src.config.loader import load_config
    from src.config.logging_config import setup_logging
    from src.data.acquisition import CommodityDataAcquisition
    from src.data.preprocessing import DataPreprocessor

    # Load configuration
    config = load_config('../../config/parameters.toml')
    logger = setup_logging(config)

    # Fetch and preprocess data
    data_client = CommodityDataAcquisition(config)
    data = data_client.fetch_commodity_prices()

    preprocessor = DataPreprocessor(config)
    data, _ = preprocessor.preprocess(data)

    # Fetch FRED data
    fred_data = {}
    for series_id in config['features']['fred_series']:
        try:
            fred_data[series_id] = data_client.fetch_fred_data(series_id)
        except Exception as e:
            logger.warning(f"Failed to fetch {series_id}: {e}")

    # Engineer features
    feature_engineer = FeatureEngineer(config)
    features = feature_engineer.engineer_features(data, fred_data=fred_data)

    print(f"\nEngineered {len(features.columns)} features")
    print(f"\nFeature columns:\n{features.columns.tolist()}")
    print(f"\nFirst 5 rows:\n{features.head()}")

    # Feature importance
    if 'returns' in features.columns:
        importances = feature_engineer.get_feature_importance(features)
        print(f"\nTop 10 important features:\n{importances.head(10)}")
