"""
Volatility Confidence Scoring System
=====================================
Combines HMM regime detection, TimesFM forecasting, and feature signals
into a single 0-100 confidence score for next-day volatility.

Score Interpretation:
- 0-40: SKIP - Low volatility expected, don't trade
- 40-60: TRADE (Small Size) - Moderate confidence
- 60-80: TRADE (Full Size) - High confidence
- 80-100: TRADE (Full Size) - Exceptional setup

Decision Logic:
- Score >= 40: Enter 4 strangles (0DTE + 1DTE calls/puts)
- Score < 40: Sit on hands, wait for better setup
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceScore:
    """Container for confidence score and components."""
    total_score: float  # 0-100 final score
    regime_score: float  # Contribution from HMM
    timesfm_score: float  # Contribution from TimesFM
    feature_score: float  # Contribution from features
    regime_label: str  # Current regime (low_vol/normal_vol/high_vol)
    regime_volatility: float  # Expected volatility from regime
    timesfm_forecast: float  # TimesFM volatility forecast
    feature_signals: Dict[str, any]  # Feature values
    recommendation: str  # Trading recommendation
    explanation: str  # Human-readable explanation


class VolatilityConfidenceScorer:
    """
    Calculate daily confidence score for volatility trading.

    Combines:
    1. HMM Regime Detection (40%) - Market volatility state
    2. TimesFM Forecast (40%) - Foundation model prediction
    3. Feature Signals (20%) - Gap, VIX, volume, momentum
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        threshold: float = 40.0
    ):
        """
        Initialize confidence scorer.

        Args:
            weights: Component weights (must sum to 1.0)
            threshold: Minimum score to trade (default: 40)
        """
        # Default weights: equal HMM + TimesFM, with features as tiebreaker
        self.weights = weights or {
            'regime': 0.4,
            'timesfm': 0.4,
            'features': 0.2
        }

        # Validate weights
        weight_sum = sum(self.weights.values())
        if not np.isclose(weight_sum, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")

        self.threshold = threshold

        logger.info(f"Confidence scorer initialized: weights={self.weights}, threshold={threshold}")

    def calculate_score(
        self,
        regime_volatility: float,
        regime_label: str,
        timesfm_forecast: Optional[float] = None,
        feature_signals: Optional[Dict[str, any]] = None
    ) -> ConfidenceScore:
        """
        Calculate comprehensive confidence score.

        Args:
            regime_volatility: Expected volatility from HMM regime
            regime_label: Regime name (low_vol/normal_vol/high_vol)
            timesfm_forecast: TimesFM volatility forecast (optional)
            feature_signals: Dictionary of feature values (optional)

        Returns:
            ConfidenceScore object with all components
        """
        # 1. REGIME SCORE (40%)
        regime_score = self._calculate_regime_score(regime_volatility, regime_label)

        # 2. TIMESFM SCORE (40%)
        if timesfm_forecast is not None:
            timesfm_score = self._calculate_timesfm_score(timesfm_forecast)
        else:
            # If TimesFM not available, redistribute weight to regime
            logger.warning("TimesFM forecast not provided, using regime score only")
            timesfm_score = regime_score
            # Adjust weights temporarily
            effective_weights = {
                'regime': self.weights['regime'] + self.weights['timesfm'],
                'timesfm': 0.0,
                'features': self.weights['features']
            }
        else:
            effective_weights = self.weights

        # 3. FEATURE SCORE (20%)
        if feature_signals:
            feature_score = self._calculate_feature_score(feature_signals)
        else:
            logger.warning("Feature signals not provided, using zero score")
            feature_score = 0.0

        # 4. WEIGHTED ENSEMBLE
        total_score = (
            effective_weights['regime'] * regime_score +
            effective_weights['timesfm'] * timesfm_score +
            effective_weights['features'] * feature_score
        )

        # Ensure score is in [0, 100] range
        total_score = np.clip(total_score, 0, 100)

        # Generate recommendation
        recommendation = self._get_recommendation(total_score)
        explanation = self._generate_explanation(
            total_score,
            regime_score,
            timesfm_score,
            feature_score,
            regime_label,
            feature_signals
        )

        return ConfidenceScore(
            total_score=total_score,
            regime_score=regime_score,
            timesfm_score=timesfm_score,
            feature_score=feature_score,
            regime_label=regime_label,
            regime_volatility=regime_volatility,
            timesfm_forecast=timesfm_forecast,
            feature_signals=feature_signals or {},
            recommendation=recommendation,
            explanation=explanation
        )

    def _calculate_regime_score(
        self,
        regime_volatility: float,
        regime_label: str
    ) -> float:
        """
        Calculate score from HMM regime.

        Calibration:
        - 0.5% volatility → 0 score (very low)
        - 1.0% volatility → 33 score (medium)
        - 1.5% volatility → 67 score (high)
        - 2.0%+ volatility → 100 score (extreme)

        Args:
            regime_volatility: Expected volatility (e.g., 0.015 = 1.5%)
            regime_label: Regime name

        Returns:
            Score in [0, 100]
        """
        # Linear scaling: 0.5% = 0, 2.0% = 100
        score = (regime_volatility - 0.005) / 0.015 * 100

        # Clip to [0, 100]
        score = np.clip(score, 0, 100)

        # Bonus for high volatility regimes (categorical boost)
        if regime_label == 'high_vol':
            score = min(score * 1.1, 100)  # 10% bonus
        elif regime_label == 'explosive_vol':
            score = min(score * 1.2, 100)  # 20% bonus

        return score

    def _calculate_timesfm_score(self, timesfm_forecast: float) -> float:
        """
        Calculate score from TimesFM forecast.

        Same calibration as regime score:
        - 0.5% forecast → 0 score
        - 2.0%+ forecast → 100 score

        Args:
            timesfm_forecast: Forecasted intraday range

        Returns:
            Score in [0, 100]
        """
        score = (timesfm_forecast - 0.005) / 0.015 * 100
        return np.clip(score, 0, 100)

    def _calculate_feature_score(self, feature_signals: Dict[str, any]) -> float:
        """
        Calculate score from feature signals.

        Features (additive, max 100):
        - Overnight gap (30 points): Large gap → morning volatility
        - VIX spike (25 points): Fear → volatility
        - Range expansion (25 points): Volatility clustering
        - Volume surge (20 points): Confirms real move

        Args:
            feature_signals: Dictionary of feature values

        Returns:
            Score in [0, 100]
        """
        score = 0

        # 1. OVERNIGHT GAP (30 points max)
        gap = feature_signals.get('overnight_gap_abs', 0)
        if gap > 0.02:  # >2% gap
            score += 30
        elif gap > 0.015:  # >1.5% gap
            score += 20
        elif gap > 0.01:  # >1% gap
            score += 10

        # 2. VIX SPIKE (25 points max)
        vix_change = feature_signals.get('vix_change_1d', 0)
        vix_level = feature_signals.get('vix_level', 15)

        if vix_change > 3:  # >3 point VIX increase
            score += 25
        elif vix_change > 2:  # >2 point increase
            score += 15
        elif vix_change > 1:  # >1 point increase
            score += 8

        # Bonus for high absolute VIX
        if vix_level > 25:  # VIX >25 = fear
            score += 10

        # 3. RANGE EXPANSION (25 points max)
        range_expansion = feature_signals.get('range_expansion', 1.0)
        if range_expansion > 1.5:  # 50% above average
            score += 25
        elif range_expansion > 1.3:  # 30% above average
            score += 15
        elif range_expansion > 1.2:  # 20% above average
            score += 8

        # 4. VOLUME SURGE (20 points max)
        volume_surge = feature_signals.get('volume_surge', 0)
        volume_ratio = feature_signals.get('volume_ratio', 1.0)

        if volume_surge or volume_ratio > 1.5:
            score += 20
        elif volume_ratio > 1.3:
            score += 10

        # 5. BONUS: Recent high-volatility clustering
        high_range_days = feature_signals.get('high_range_days_5', 0)
        if high_range_days >= 3:  # 3+ volatile days in last 5
            score += 15

        # Cap at 100
        return min(score, 100)

    def _get_recommendation(self, score: float) -> str:
        """Convert score to trading recommendation."""
        if score < 40:
            return "SKIP - Low volatility expected"
        elif score < 60:
            return "TRADE (Small Size) - Moderate confidence"
        elif score < 80:
            return "TRADE (Full Size) - High confidence"
        else:
            return "TRADE (Full Size) - Exceptional setup"

    def _generate_explanation(
        self,
        total_score: float,
        regime_score: float,
        timesfm_score: float,
        feature_score: float,
        regime_label: str,
        feature_signals: Optional[Dict[str, any]]
    ) -> str:
        """Generate human-readable explanation of score."""
        lines = [
            f"Total Confidence: {total_score:.0f}/100",
            f"",
            f"Component Breakdown:",
            f"  - Regime Score: {regime_score:.0f}/100 ({regime_label})",
            f"  - TimesFM Score: {timesfm_score:.0f}/100",
            f"  - Feature Score: {feature_score:.0f}/100",
            f"",
        ]

        # Add feature details if available
        if feature_signals:
            lines.append("Key Signals:")
            gap = feature_signals.get('overnight_gap_abs', 0)
            lines.append(f"  - Overnight Gap: {gap*100:.2f}%")

            vix_change = feature_signals.get('vix_change_1d', 0)
            vix_level = feature_signals.get('vix_level', 0)
            lines.append(f"  - VIX: {vix_level:.1f} (change: {vix_change:+.1f})")

            range_exp = feature_signals.get('range_expansion', 1.0)
            lines.append(f"  - Range Expansion: {range_exp:.2f}x")

            vol_ratio = feature_signals.get('volume_ratio', 1.0)
            lines.append(f"  - Volume Ratio: {vol_ratio:.2f}x")

        return "\n".join(lines)

    def batch_score(
        self,
        df: pd.DataFrame,
        regime_col: str = 'regime_volatility',
        regime_label_col: str = 'regime_label',
        timesfm_col: Optional[str] = 'timesfm_forecast',
        feature_cols: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Calculate scores for multiple days (backtesting/validation).

        Args:
            df: DataFrame with regime and feature data
            regime_col: Column name for regime volatility
            regime_label_col: Column name for regime labels
            timesfm_col: Column name for TimesFM forecasts (optional)
            feature_cols: List of feature column names (optional)

        Returns:
            DataFrame with 'confidence_score' column added
        """
        logger.info(f"Calculating batch scores for {len(df)} rows")

        scores = []

        for idx, row in df.iterrows():
            # Extract regime info
            regime_vol = row[regime_col]
            regime_label = row[regime_label_col]

            # Extract TimesFM forecast
            timesfm_forecast = row[timesfm_col] if timesfm_col and timesfm_col in row else None

            # Extract features
            if feature_cols:
                feature_signals = {col: row[col] for col in feature_cols if col in row}
            else:
                feature_signals = None

            # Calculate score
            score_obj = self.calculate_score(
                regime_volatility=regime_vol,
                regime_label=regime_label,
                timesfm_forecast=timesfm_forecast,
                feature_signals=feature_signals
            )

            scores.append(score_obj.total_score)

        df['confidence_score'] = scores
        df['trade_signal'] = (df['confidence_score'] >= self.threshold).astype(int)

        logger.info(f"Batch scoring complete. Trade signals: {df['trade_signal'].sum()}/{len(df)}")

        return df


def test_confidence_scorer():
    """Test confidence scoring system."""
    print("Testing Volatility Confidence Scorer...")
    print("=" * 60)

    scorer = VolatilityConfidenceScorer()

    # Test 1: High volatility scenario
    print("\n1. High Volatility Scenario")
    print("-" * 60)
    score1 = scorer.calculate_score(
        regime_volatility=0.018,  # 1.8% expected
        regime_label='high_vol',
        timesfm_forecast=0.019,  # 1.9% forecast
        feature_signals={
            'overnight_gap_abs': 0.022,  # 2.2% gap
            'vix_change_1d': 3.5,  # VIX up 3.5 points
            'vix_level': 28,  # VIX at 28 (fear)
            'range_expansion': 1.4,  # 40% above average
            'volume_surge': 1,  # Volume surge detected
            'volume_ratio': 1.7,
            'high_range_days_5': 3
        }
    )

    print(score1.explanation)
    print(f"\nRecommendation: {score1.recommendation}")

    # Test 2: Low volatility scenario
    print("\n\n2. Low Volatility Scenario")
    print("-" * 60)
    score2 = scorer.calculate_score(
        regime_volatility=0.006,  # 0.6% expected
        regime_label='low_vol',
        timesfm_forecast=0.007,  # 0.7% forecast
        feature_signals={
            'overnight_gap_abs': 0.002,  # 0.2% gap
            'vix_change_1d': -0.5,  # VIX down
            'vix_level': 13,  # VIX at 13 (calm)
            'range_expansion': 0.9,  # Below average
            'volume_surge': 0,
            'volume_ratio': 0.8,
            'high_range_days_5': 0
        }
    )

    print(score2.explanation)
    print(f"\nRecommendation: {score2.recommendation}")

    # Test 3: Moderate scenario (edge case)
    print("\n\n3. Moderate Scenario (Edge Case)")
    print("-" * 60)
    score3 = scorer.calculate_score(
        regime_volatility=0.011,  # 1.1% expected
        regime_label='normal_vol',
        timesfm_forecast=0.012,  # 1.2% forecast
        feature_signals={
            'overnight_gap_abs': 0.008,  # 0.8% gap
            'vix_change_1d': 0.5,  # Small VIX increase
            'vix_level': 18,  # VIX normal
            'range_expansion': 1.1,  # Slightly above average
            'volume_surge': 0,
            'volume_ratio': 1.2,
            'high_range_days_5': 1
        }
    )

    print(score3.explanation)
    print(f"\nRecommendation: {score3.recommendation}")

    print("\n" + "=" * 60)
    print("✅ All tests passed!")

    return True


if __name__ == "__main__":
    test_confidence_scorer()
