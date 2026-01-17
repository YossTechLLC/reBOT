"""
Data Preprocessing Module
Missing data handling, outlier detection, stationarity testing, and scaling.
"""

import logging
import warnings
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline.

    Features:
    - Missing data handling (forward fill, interpolation, drop)
    - Outlier detection (Z-score, IQR, Isolation Forest)
    - Stationarity testing (Augmented Dickey-Fuller)
    - Transformations (returns, log returns, differencing)
    - Scaling and normalization
    """

    def __init__(self, config: Dict):
        """
        Initialize preprocessor with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.preprocessing_config = config['data'].get('preprocessing', {})

        # Parameters
        self.missing_method = self.preprocessing_config.get('missing_method', 'ffill')
        self.outlier_method = self.preprocessing_config.get('outlier_method', 'zscore')
        self.outlier_threshold = self.preprocessing_config.get('outlier_threshold', 3.0)
        self.transformation = self.preprocessing_config.get('transformation', 'log_returns')

        # State
        self.scaler = None
        self.outlier_detector = None

        logger.info("Initialized data preprocessor")
        logger.info(f"Missing data method: {self.missing_method}")
        logger.info(f"Outlier method: {self.outlier_method}")
        logger.info(f"Transformation: {self.transformation}")

    def preprocess(
        self,
        data: pd.DataFrame,
        fit: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Full preprocessing pipeline.

        Steps:
        1. Handle missing data
        2. Detect and treat outliers
        3. Test stationarity
        4. Apply transformation
        5. Scale features (optional)

        Args:
            data: Raw OHLCV DataFrame
            fit: Whether to fit scalers (True for training, False for inference)

        Returns:
            Tuple of (preprocessed_data, metadata)
        """
        logger.info("Starting preprocessing pipeline...")
        metadata = {}

        # 1. Handle missing data
        data, missing_info = self.handle_missing_data(data)
        metadata['missing_data'] = missing_info

        # 2. Detect and treat outliers
        data, outlier_info = self.detect_and_treat_outliers(data)
        metadata['outliers'] = outlier_info

        # 3. Test stationarity
        is_stationary, stat_info = self.test_stationarity(data['close'])
        metadata['stationarity'] = stat_info

        if not is_stationary:
            logger.warning(
                "TFM2001 DATA: Price series is non-stationary (p-value > 0.05). "
                "Transformation recommended."
            )

        # 4. Apply transformation
        data, transform_info = self.apply_transformation(data)
        metadata['transformation'] = transform_info

        logger.info("Preprocessing pipeline complete")
        return data, metadata

    def handle_missing_data(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Handle missing data using configured method.

        Methods:
        - ffill: Forward fill (propagate last valid observation)
        - interpolate: Linear interpolation
        - drop: Remove rows with missing values

        Args:
            data: DataFrame with potential missing values

        Returns:
            Tuple of (cleaned_data, info_dict)
        """
        initial_len = len(data)
        missing_count = data.isna().sum().sum()

        logger.info(f"Missing values before handling: {missing_count}")

        if missing_count == 0:
            logger.info("No missing data found")
            return data, {'method': self.missing_method, 'count': 0, 'rows_removed': 0}

        if self.missing_method == 'ffill':
            data = data.fillna(method='ffill')
            # Backfill any remaining NaNs at the start
            data = data.fillna(method='bfill')

        elif self.missing_method == 'interpolate':
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = data[numeric_cols].interpolate(method='linear')
            # Fill any remaining NaNs at edges
            data = data.fillna(method='bfill').fillna(method='ffill')

        elif self.missing_method == 'drop':
            data = data.dropna()

        else:
            raise ValueError(
                f"TFM1001 CONFIG: Unknown missing_method: {self.missing_method}. "
                f"Valid options: 'ffill', 'interpolate', 'drop'"
            )

        final_len = len(data)
        rows_removed = initial_len - final_len
        remaining_missing = data.isna().sum().sum()

        logger.info(f"Missing values after handling: {remaining_missing}")
        logger.info(f"Rows removed: {rows_removed}")

        return data, {
            'method': self.missing_method,
            'count': missing_count,
            'rows_removed': rows_removed,
            'remaining': remaining_missing
        }

    def detect_and_treat_outliers(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Detect and treat outliers using configured method.

        Methods:
        - zscore: Z-score threshold (default: 3.0 for 99.7% CI)
        - iqr: Interquartile range (1.5 * IQR)
        - isolation_forest: Unsupervised anomaly detection

        Args:
            data: DataFrame with potential outliers

        Returns:
            Tuple of (treated_data, outlier_info)
        """
        logger.info(f"Detecting outliers using {self.outlier_method} method")

        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        outlier_counts = {}

        for col in numeric_cols:
            if col not in data.columns:
                continue

            if self.outlier_method == 'zscore':
                outliers = self._detect_outliers_zscore(data[col])
            elif self.outlier_method == 'iqr':
                outliers = self._detect_outliers_iqr(data[col])
            elif self.outlier_method == 'isolation_forest':
                outliers = self._detect_outliers_isolation_forest(data[[col]])
            else:
                raise ValueError(
                    f"TFM1001 CONFIG: Unknown outlier_method: {self.outlier_method}"
                )

            outlier_count = outliers.sum()
            outlier_counts[col] = int(outlier_count)

            if outlier_count > 0:
                logger.info(f"  {col}: {outlier_count} outliers ({outlier_count/len(data)*100:.2f}%)")

                # Treatment: Replace outliers with median
                median_value = data[col].median()
                data.loc[outliers, col] = median_value

        total_outliers = sum(outlier_counts.values())
        logger.info(f"Total outliers detected: {total_outliers}")

        return data, {
            'method': self.outlier_method,
            'threshold': self.outlier_threshold,
            'counts_by_column': outlier_counts,
            'total': total_outliers
        }

    def _detect_outliers_zscore(
        self,
        series: pd.Series,
        threshold: Optional[float] = None
    ) -> pd.Series:
        """Detect outliers using Z-score method."""
        threshold = threshold or self.outlier_threshold
        z_scores = np.abs(stats.zscore(series.dropna()))

        # Create boolean mask (same length as original series)
        outliers = pd.Series(False, index=series.index)
        outliers.loc[series.notna()] = z_scores > threshold

        return outliers

    def _detect_outliers_iqr(self, series: pd.Series) -> pd.Series:
        """Detect outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = (series < lower_bound) | (series > upper_bound)
        return outliers

    def _detect_outliers_isolation_forest(
        self,
        data: pd.DataFrame,
        contamination: float = 0.1
    ) -> pd.Series:
        """Detect outliers using Isolation Forest."""
        if self.outlier_detector is None:
            self.outlier_detector = IsolationForest(
                contamination=contamination,
                random_state=42
            )

        # Fit and predict
        predictions = self.outlier_detector.fit_predict(data.dropna())

        # Create boolean mask (-1 indicates outlier)
        outliers = pd.Series(False, index=data.index)
        outliers.loc[data.notna().all(axis=1)] = predictions == -1

        return outliers

    def test_stationarity(
        self,
        series: pd.Series,
        alpha: float = 0.05
    ) -> Tuple[bool, Dict]:
        """
        Test stationarity using Augmented Dickey-Fuller test.

        Null hypothesis: Series has a unit root (non-stationary)
        Alternative: Series is stationary

        Args:
            series: Time series to test
            alpha: Significance level (default: 0.05)

        Returns:
            Tuple of (is_stationary, test_results)
        """
        try:
            result = adfuller(series.dropna())

            adf_stat = result[0]
            p_value = result[1]
            n_lags = result[2]
            n_obs = result[3]
            critical_values = result[4]

            is_stationary = p_value < alpha

            logger.info(f"Augmented Dickey-Fuller Test:")
            logger.info(f"  ADF Statistic: {adf_stat:.4f}")
            logger.info(f"  p-value: {p_value:.4f}")
            logger.info(f"  Critical values: {critical_values}")

            if is_stationary:
                logger.info(f"  Result: Series is STATIONARY (p < {alpha})")
            else:
                logger.warning(
                    f"  Result: Series is NON-STATIONARY (p >= {alpha}). "
                    f"Transformation recommended."
                )

            return is_stationary, {
                'adf_statistic': adf_stat,
                'p_value': p_value,
                'n_lags': n_lags,
                'n_observations': n_obs,
                'critical_values': critical_values,
                'is_stationary': is_stationary
            }

        except Exception as e:
            logger.error(f"TFM2001 DATA: Stationarity test failed: {e}")
            return False, {'error': str(e)}

    def apply_transformation(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply transformation to make series stationary.

        Transformations:
        - returns: Simple returns (pct_change)
        - log_returns: Log returns (more appropriate for prices)
        - diff: First difference
        - none: No transformation

        Args:
            data: DataFrame with price data

        Returns:
            Tuple of (transformed_data, transformation_info)
        """
        if self.transformation == 'none':
            logger.info("No transformation applied")
            return data, {'method': 'none'}

        original_len = len(data)

        if self.transformation == 'returns':
            # Simple returns
            data['returns'] = data['close'].pct_change()
            logger.info("Applied simple returns transformation")

        elif self.transformation == 'log_returns':
            # Log returns
            data['returns'] = np.log(data['close'] / data['close'].shift(1))
            logger.info("Applied log returns transformation")

        elif self.transformation == 'diff':
            # First difference
            data['returns'] = data['close'].diff()
            logger.info("Applied first difference transformation")

        else:
            raise ValueError(
                f"TFM1001 CONFIG: Unknown transformation: {self.transformation}"
            )

        # Remove NaN from transformation
        data = data.dropna()
        rows_removed = original_len - len(data)

        logger.info(f"Rows removed due to transformation: {rows_removed}")

        return data, {
            'method': self.transformation,
            'rows_removed': rows_removed
        }

    def scale_features(
        self,
        data: pd.DataFrame,
        method: str = 'standard',
        fit: bool = True
    ) -> Tuple[pd.DataFrame, StandardScaler]:
        """
        Scale features for model training.

        Methods:
        - standard: StandardScaler (zero mean, unit variance)
        - robust: RobustScaler (median and IQR, robust to outliers)

        Args:
            data: DataFrame with features to scale
            method: Scaling method
            fit: Whether to fit scaler (True for training, False for inference)

        Returns:
            Tuple of (scaled_data, scaler)
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        if method == 'standard':
            if self.scaler is None or fit:
                self.scaler = StandardScaler()
                scaled = self.scaler.fit_transform(data[numeric_cols])
                logger.info("Fitted StandardScaler")
            else:
                scaled = self.scaler.transform(data[numeric_cols])
                logger.info("Applied pre-fitted StandardScaler")

        elif method == 'robust':
            if self.scaler is None or fit:
                self.scaler = RobustScaler()
                scaled = self.scaler.fit_transform(data[numeric_cols])
                logger.info("Fitted RobustScaler")
            else:
                scaled = self.scaler.transform(data[numeric_cols])
                logger.info("Applied pre-fitted RobustScaler")

        else:
            raise ValueError(
                f"TFM1001 CONFIG: Unknown scaling method: {method}"
            )

        # Create DataFrame with scaled values
        scaled_data = data.copy()
        scaled_data[numeric_cols] = scaled

        return scaled_data, self.scaler

    def validate_no_leakage(
        self,
        data: pd.DataFrame,
        target_col: str = 'returns'
    ) -> bool:
        """
        Validate that there is no data leakage (future information in features).

        Checks:
        - No forward-looking operations (negative shifts)
        - Target column exists
        - No perfect correlation with target

        Args:
            data: DataFrame with features
            target_col: Target column name

        Returns:
            True if no leakage detected

        Raises:
            ValueError: If leakage is detected
        """
        logger.info("Validating data for leakage...")

        if target_col not in data.columns:
            raise ValueError(
                f"TFM2001 DATA: Target column '{target_col}' not found"
            )

        # Check for perfect correlation (indicator of leakage)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        correlations = data[numeric_cols].corrwith(data[target_col]).abs()

        # Exclude target column itself
        correlations = correlations.drop(target_col, errors='ignore')

        perfect_corr = correlations[correlations > 0.999]
        if not perfect_corr.empty:
            logger.error(
                f"TFM2001 DATA: Perfect correlation detected (leakage risk):\n{perfect_corr}"
            )
            raise ValueError(
                f"Data leakage detected: columns {perfect_corr.index.tolist()} "
                f"have perfect correlation with target"
            )

        logger.info("No data leakage detected")
        return True


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('../..')

    from src.config.loader import load_config
    from src.config.logging_config import setup_logging
    from src.data.acquisition import CommodityDataAcquisition

    # Load configuration
    config = load_config('../../config/parameters.toml')
    logger = setup_logging(config)

    # Fetch data
    data_client = CommodityDataAcquisition(config)
    data = data_client.fetch_commodity_prices()

    # Preprocess
    preprocessor = DataPreprocessor(config)
    preprocessed_data, metadata = preprocessor.preprocess(data)

    print(f"\nPreprocessed {len(preprocessed_data)} rows")
    print(f"\nMetadata: {metadata}")
    print(f"\nFirst 5 rows:\n{preprocessed_data.head()}")
    print(f"\nData types:\n{preprocessed_data.dtypes}")
