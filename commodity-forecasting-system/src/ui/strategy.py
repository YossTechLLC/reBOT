"""
Trading Strategy Components for Volatility Prediction UI
=========================================================
Generates trading strategy recommendations based on volatility predictions.

Strategy Types:
- 0DTE/1DTE Volatility Strangles
- Wide strangle (high volatility expected)
- Narrow strangle (normal volatility)
- Skip (low volatility)

Author: Claude + User
Date: 2026-01-17
"""

from typing import Dict, Optional
import numpy as np


class SpreadRecommender:
    """
    Recommends options spread strategies based on volatility regime and prediction.
    """

    def __init__(self):
        """Initialize SpreadRecommender with strategy rules."""
        # Strategy definitions by regime
        self.regime_strategies = {
            'high_vol': {
                'name': 'Wide Strangle',
                'call_delta': 0.30,  # 30 delta call
                'put_delta': -0.30,  # 30 delta put
                'contracts': 1,
                'entry_timing': 'Market open',
                'exit_rules': [
                    '50% profit target',
                    '3:45 PM force exit',
                    '80% loss stop'
                ],
                'rationale': 'High volatility expected - wider strikes capture large moves while limiting risk'
            },
            'normal_vol': {
                'name': 'Narrow Strangle',
                'call_delta': 0.40,  # 40 delta call
                'put_delta': -0.40,  # 40 delta put
                'contracts': 1,
                'entry_timing': 'Market open',
                'exit_rules': [
                    '30% profit target',
                    '3:45 PM force exit',
                    '70% loss stop'
                ],
                'rationale': 'Normal volatility - standard strangle setup with moderate strikes'
            },
            'low_vol': {
                'name': 'SKIP',
                'rationale': 'Low volatility expected - insufficient edge for profitable trade. Wait for better setup.'
            }
        }

    def recommend_spread(
        self,
        regime: str,
        current_price: float,
        predicted_volatility: float,
        confidence: float
    ) -> Dict:
        """
        Generate spread recommendation based on regime and prediction.

        Args:
            regime: Volatility regime ('low_vol', 'normal_vol', 'high_vol')
            current_price: Current SPY price
            predicted_volatility: Predicted next-day volatility (as decimal, e.g., 0.015 for 1.5%)
            confidence: Confidence score (0-100)

        Returns:
            Dictionary with strategy recommendation
        """
        strategy = self.regime_strategies[regime].copy()

        if regime == 'low_vol':
            return {
                **strategy,
                'should_trade': False,
                'current_price': current_price,
                'predicted_volatility': predicted_volatility,
                'confidence': confidence
            }

        # Calculate expected move
        expected_move = current_price * predicted_volatility

        # Estimate strike prices (simplified - actual strikes would come from options chain)
        call_strike = self._round_to_strike(current_price + expected_move)
        put_strike = self._round_to_strike(current_price - expected_move)

        # Calculate position Greeks estimates (simplified)
        greeks = self._estimate_greeks(
            current_price,
            call_strike,
            put_strike,
            predicted_volatility
        )

        # Risk calculations
        risk_metrics = self._calculate_risk_metrics(
            current_price,
            call_strike,
            put_strike,
            greeks
        )

        return {
            **strategy,
            'should_trade': True,
            'current_price': current_price,
            'predicted_volatility': predicted_volatility,
            'confidence': confidence,
            'expected_move': round(expected_move, 2),
            'call_strike': call_strike,
            'put_strike': put_strike,
            'spread_width': round(call_strike - put_strike, 2),
            'greeks': greeks,
            'risk_metrics': risk_metrics
        }

    @staticmethod
    def _round_to_strike(price: float) -> float:
        """
        Round price to nearest option strike (typically $1 or $0.50 increments).

        Args:
            price: Price to round

        Returns:
            Rounded strike price
        """
        # For SPY, use $1 increments
        return round(price)

    @staticmethod
    def _estimate_greeks(
        spot: float,
        call_strike: float,
        put_strike: float,
        volatility: float
    ) -> Dict:
        """
        Estimate position Greeks (simplified - not actual Black-Scholes).

        Args:
            spot: Current stock price
            call_strike: Call strike price
            put_strike: Put strike price
            volatility: Implied volatility

        Returns:
            Dictionary with estimated Greeks
        """
        # Simplified estimates (in practice, use actual options pricing)
        return {
            'delta': 0.0,  # Strangle is delta-neutral
            'gamma': round(-0.05 * volatility * 100, 4),  # Short gamma
            'vega': round(-0.10 * volatility * 100, 4),   # Short vega
            'theta': round(0.02 * spot, 2)                # Positive theta (time decay)
        }

    @staticmethod
    def _calculate_risk_metrics(
        spot: float,
        call_strike: float,
        put_strike: float,
        greeks: Dict
    ) -> Dict:
        """
        Calculate risk/reward metrics for the position.

        Args:
            spot: Current stock price
            call_strike: Call strike price
            put_strike: Put strike price
            greeks: Position Greeks

        Returns:
            Dictionary with risk metrics
        """
        # Simplified risk calculations
        # In practice, these would be derived from options pricing model

        # Assume credit received is ~0.5% of spot (typical for strangle)
        credit_received = round(spot * 0.005, 2)

        # Max profit = credit received
        max_profit = credit_received

        # Max loss estimate (simplified)
        # Actual max loss is unlimited for naked strangle, but using expected move
        spread_width = call_strike - put_strike
        max_loss_estimate = round(spread_width * 0.1, 2)  # Rough estimate

        return {
            'credit_received': credit_received,
            'max_profit': max_profit,
            'max_loss_estimate': max_loss_estimate,
            'profit_probability': 0.65,  # Estimated (typically strangle wins 60-70%)
            'breakeven_upper': round(call_strike + credit_received, 2),
            'breakeven_lower': round(put_strike - credit_received, 2),
            'risk_reward_ratio': round(max_profit / max_loss_estimate, 2) if max_loss_estimate > 0 else 0
        }


class PositionSizer:
    """
    Calculates position size based on account size and risk parameters.
    """

    def __init__(self, account_size: float = 10000, max_risk_pct: float = 0.02):
        """
        Initialize PositionSizer.

        Args:
            account_size: Total account size in dollars
            max_risk_pct: Maximum risk per trade as percentage (e.g., 0.02 for 2%)
        """
        self.account_size = account_size
        self.max_risk_pct = max_risk_pct

    def calculate_position_size(
        self,
        strategy: Dict,
        confidence: float
    ) -> Dict:
        """
        Calculate recommended position size.

        Args:
            strategy: Strategy dictionary from SpreadRecommender
            confidence: Confidence score (0-100)

        Returns:
            Dictionary with position sizing recommendation
        """
        if not strategy.get('should_trade', False):
            return {
                'contracts': 0,
                'total_risk': 0,
                'reason': 'Trade skipped - low volatility'
            }

        # Max risk per trade in dollars
        max_risk_dollars = self.account_size * self.max_risk_pct

        # Adjust risk based on confidence
        # High confidence (>70) = 100% of max risk
        # Medium confidence (40-70) = 50-100% of max risk
        # Low confidence (<40) = skip
        if confidence >= 70:
            risk_multiplier = 1.0
        elif confidence >= 40:
            risk_multiplier = 0.5 + (confidence - 40) / 60  # Linear scaling
        else:
            return {
                'contracts': 0,
                'total_risk': 0,
                'reason': 'Confidence too low'
            }

        adjusted_risk = max_risk_dollars * risk_multiplier

        # Calculate number of contracts
        max_loss_per_contract = strategy['risk_metrics']['max_loss_estimate'] * 100  # Convert to dollars per contract
        contracts = int(adjusted_risk / max_loss_per_contract) if max_loss_per_contract > 0 else 0

        # Minimum 1 contract if risk allows
        if contracts == 0 and adjusted_risk >= max_loss_per_contract:
            contracts = 1

        return {
            'contracts': contracts,
            'total_risk': round(contracts * max_loss_per_contract, 2),
            'risk_pct': round((contracts * max_loss_per_contract / self.account_size) * 100, 2),
            'max_profit': round(contracts * strategy['risk_metrics']['max_profit'] * 100, 2),
            'confidence_adjustment': round(risk_multiplier * 100, 1)
        }


def format_strategy_output(recommendation: Dict, position_size: Dict = None) -> str:
    """
    Format strategy recommendation as readable text.

    Args:
        recommendation: Strategy dictionary from SpreadRecommender
        position_size: Optional position sizing from PositionSizer

    Returns:
        Formatted strategy description
    """
    if not recommendation.get('should_trade', False):
        return f"""
### Strategy: {recommendation['name']}

**Decision:** SKIP TRADE

**Reason:** {recommendation['rationale']}

**Current Price:** ${recommendation['current_price']:.2f}
**Predicted Volatility:** {recommendation['predicted_volatility']*100:.2f}%
**Confidence:** {recommendation['confidence']:.1f}/100
"""

    output = f"""
### Strategy: {recommendation['name']}

**Decision:** ENTER TRADE

**Setup:**
- Current Price: ${recommendation['current_price']:.2f}
- Expected Move: ${recommendation['expected_move']:.2f} ({recommendation['predicted_volatility']*100:.2f}%)
- Call Strike: ${recommendation['call_strike']:.2f}
- Put Strike: ${recommendation['put_strike']:.2f}
- Spread Width: ${recommendation['spread_width']:.2f}

**Entry:**
- Timing: {recommendation['entry_timing']}
- Sell {recommendation['contracts']} x ${recommendation['call_strike']:.2f} Call
- Sell {recommendation['contracts']} x ${recommendation['put_strike']:.2f} Put

**Exit Rules:**
"""
    for rule in recommendation['exit_rules']:
        output += f"- {rule}\n"

    output += f"""
**Risk/Reward:**
- Credit Received: ${recommendation['risk_metrics']['credit_received']:.2f}
- Max Profit: ${recommendation['risk_metrics']['max_profit']:.2f}
- Max Loss (Est): ${recommendation['risk_metrics']['max_loss_estimate']:.2f}
- Breakeven: ${recommendation['risk_metrics']['breakeven_lower']:.2f} - ${recommendation['risk_metrics']['breakeven_upper']:.2f}
- Win Probability: {recommendation['risk_metrics']['profit_probability']*100:.0f}%

**Rationale:** {recommendation['rationale']}
"""

    if position_size:
        output += f"""
**Position Sizing:**
- Recommended Contracts: {position_size['contracts']}
- Total Risk: ${position_size['total_risk']:.2f} ({position_size['risk_pct']:.2f}% of account)
- Max Profit Potential: ${position_size['max_profit']:.2f}
- Confidence Adjustment: {position_size['confidence_adjustment']:.1f}%
"""

    return output
