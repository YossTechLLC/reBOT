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
        # Strategy definitions by regime (supports 2-5 regime HMM)
        #
        # KEY INSIGHT: We BUY options when expecting HIGH volatility
        # - Long straddle/strangle = pay debit, profit from big moves
        # - One leg goes deep ITM, other expires worthless
        # - Net profit = ITM intrinsic value - total premium paid
        #
        # extreme_vol/high_vol = BUY options (long gamma, profit from moves)
        # normal_vol = BUY options (smaller position)
        # low_vol/very_low_vol = SKIP (no edge, theta decay kills long options)
        self.regime_strategies = {
            'extreme_vol': {
                'name': 'LONG ATM Straddle',
                'position_type': 'long',  # BUY options
                'call_delta': 0.50,  # ATM call (delta ~0.50)
                'put_delta': -0.50,  # ATM put (delta ~-0.50)
                'contracts': 1,
                'entry_timing': 'Market open - PRIORITY',
                'exit_rules': [
                    '100%+ profit target (let winners run)',
                    'Close before 3:30 PM (0DTE theta acceleration)',
                    'No stop loss - max loss is premium paid'
                ],
                'rationale': 'EXTREME volatility expected - BUY ATM straddle for maximum gamma exposure. One leg will go deep ITM on the big move. Best risk/reward setup.'
            },
            'high_vol': {
                'name': 'LONG Wide Strangle',
                'position_type': 'long',  # BUY options
                'call_delta': 0.30,  # 30 delta OTM call
                'put_delta': -0.30,  # 30 delta OTM put
                'contracts': 1,
                'entry_timing': 'Market open',
                'exit_rules': [
                    '50-100% profit target',
                    'Close before 3:45 PM',
                    'No stop loss - max loss is premium paid'
                ],
                'rationale': 'High volatility expected - BUY OTM strangle for leveraged exposure. Cheaper premium, higher % return if move exceeds strikes.'
            },
            'normal_vol': {
                'name': 'LONG Narrow Strangle',
                'position_type': 'long',  # BUY options
                'call_delta': 0.40,  # 40 delta call (closer to ATM)
                'put_delta': -0.40,  # 40 delta put
                'contracts': 1,
                'entry_timing': 'Market open',
                'exit_rules': [
                    '30-50% profit target',
                    'Close before 3:45 PM',
                    'Consider closing at 50% loss if no movement'
                ],
                'rationale': 'Normal volatility - BUY closer-to-ATM strangle. Moderate premium, good delta exposure for expected move.'
            },
            'low_vol': {
                'name': 'SKIP',
                'rationale': 'Low volatility expected - theta decay will kill long options. Wait for better setup.'
            },
            'very_low_vol': {
                'name': 'SKIP',
                'rationale': 'Very low volatility (dead market) - no edge. Long options will decay to zero. Wait for volatility cluster.'
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
            regime: Volatility regime. Supports 2-5 regime HMM:
                    - 'very_low_vol': SKIP (dead market)
                    - 'low_vol': SKIP (insufficient edge)
                    - 'normal_vol': TRADE (narrow strangle)
                    - 'high_vol': TRADE (wide strangle)
                    - 'extreme_vol': TRADE (ATM straddle - BEST opportunity)
            current_price: Current SPY price
            predicted_volatility: Predicted next-day volatility (as decimal, e.g., 0.015 for 1.5%)
            confidence: Confidence score (0-100)

        Returns:
            Dictionary with strategy recommendation
        """
        # Handle unknown regimes gracefully (fallback to normal_vol strategy)
        if regime not in self.regime_strategies:
            regime = 'normal_vol'

        strategy = self.regime_strategies[regime].copy()

        # SKIP regimes: low_vol and very_low_vol
        if regime in ('low_vol', 'very_low_vol'):
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

        # Get position type from strategy (default to 'long' for high vol)
        position_type = strategy.get('position_type', 'long')

        # Risk calculations for LONG position
        risk_metrics = self._calculate_risk_metrics(
            current_price,
            call_strike,
            put_strike,
            greeks,
            expected_move,
            position_type
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
        # For LONG positions: positive gamma/vega (benefit from vol), negative theta (time decay hurts)
        return {
            'delta': 0.0,  # Straddle/strangle is delta-neutral at entry
            'gamma': round(0.05 * volatility * 100, 4),   # LONG gamma (benefit from moves)
            'vega': round(0.10 * volatility * 100, 4),    # LONG vega (benefit from vol increase)
            'theta': round(-0.03 * spot, 2)               # NEGATIVE theta (time decay hurts us)
        }

    @staticmethod
    def _calculate_risk_metrics(
        spot: float,
        call_strike: float,
        put_strike: float,
        greeks: Dict,
        expected_move: float,
        position_type: str = 'long'
    ) -> Dict:
        """
        Calculate risk/reward metrics for LONG straddle/strangle.

        For LONG positions:
        - Pay debit (premium) upfront = max loss
        - Profit when underlying moves beyond breakeven
        - One leg goes ITM, other expires worthless
        - Net profit = ITM intrinsic value - total premium paid

        Args:
            spot: Current stock price
            call_strike: Call strike price
            put_strike: Put strike price
            greeks: Position Greeks
            expected_move: Expected price move in dollars
            position_type: 'long' (buy options) or 'short' (sell options)

        Returns:
            Dictionary with risk metrics
        """
        # For LONG straddle/strangle:
        # Premium cost estimate: ~0.8-1.5% of spot for 0DTE ATM straddle
        # Wider OTM strangle is cheaper (~0.3-0.6% of spot)

        spread_width = call_strike - put_strike
        is_atm = spread_width < spot * 0.02  # ATM if strikes within 2% of each other

        if is_atm:
            # ATM straddle: higher premium, higher gamma
            debit_paid = round(spot * 0.012, 2)  # ~1.2% of spot for 0DTE ATM straddle
        else:
            # OTM strangle: cheaper premium
            debit_paid = round(spot * 0.006, 2)  # ~0.6% of spot for OTM strangle

        # Max loss = premium paid (defined risk)
        max_loss = debit_paid

        # Max profit estimate: if move hits expected, one leg goes ITM
        # Profit = intrinsic value of ITM leg - total premium paid
        # Conservative estimate: capture 70% of expected move as intrinsic
        intrinsic_value = expected_move * 0.7
        max_profit_estimate = round(max(intrinsic_value - debit_paid, 0), 2)

        # Breakeven points: spot +/- debit paid (for ATM straddle)
        # For OTM strangle, breakeven is at strikes +/- portion of premium
        breakeven_upper = round(call_strike + debit_paid, 2)
        breakeven_lower = round(put_strike - debit_paid, 2)

        # Win probability for long straddle on high vol day: ~40-50%
        # (need big move to overcome premium paid)
        win_probability = 0.45

        return {
            'position_type': position_type,
            'debit_paid': debit_paid,           # Cost to enter (premium)
            'max_loss': max_loss,               # Max loss = premium paid
            'max_profit_estimate': max_profit_estimate,  # Estimated profit on expected move
            'profit_probability': win_probability,
            'breakeven_upper': breakeven_upper,
            'breakeven_lower': breakeven_lower,
            'risk_reward_ratio': round(max_profit_estimate / max_loss, 2) if max_loss > 0 else 0
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
        # Check if trade should be skipped
        if not strategy.get('should_trade', False):
            return {
                'contracts': 0,
                'total_risk': 0,
                'risk_pct': 0,
                'max_profit': 0,
                'confidence_adjustment': 0,
                'reason': 'Trade skipped - low volatility regime'
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
                'risk_pct': 0,
                'max_profit': 0,
                'confidence_adjustment': 0,
                'reason': 'Confidence too low (<40)'
            }

        adjusted_risk = max_risk_dollars * risk_multiplier

        # Calculate number of contracts based on position type
        # For LONG: max_loss = debit paid (premium)
        # For SHORT: max_loss = max_loss_estimate
        risk_metrics = strategy['risk_metrics']
        if risk_metrics.get('position_type') == 'long':
            max_loss_per_contract = risk_metrics['max_loss'] * 100  # Convert to dollars per contract
            max_profit_per_contract = risk_metrics['max_profit_estimate'] * 100
        else:
            max_loss_per_contract = risk_metrics.get('max_loss_estimate', risk_metrics.get('max_loss', 1)) * 100
            max_profit_per_contract = risk_metrics.get('max_profit', 0) * 100

        contracts = int(adjusted_risk / max_loss_per_contract) if max_loss_per_contract > 0 else 0

        # Minimum 1 contract if risk allows
        if contracts == 0 and adjusted_risk >= max_loss_per_contract:
            contracts = 1

        # Build result
        result = {
            'contracts': contracts,
            'total_risk': round(contracts * max_loss_per_contract, 2),
            'risk_pct': round((contracts * max_loss_per_contract / self.account_size) * 100, 2),
            'max_profit': round(contracts * max_profit_per_contract, 2),
            'confidence_adjustment': round(risk_multiplier * 100, 1)
        }

        # Add reason if no contracts due to risk limits
        if contracts == 0:
            result['reason'] = (
                f'Risk limit too restrictive: ${adjusted_risk:.0f} available vs '
                f'${max_loss_per_contract:.0f}/contract required. '
                f'Consider increasing account size or risk %.'
            )

        return result


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

    # Determine if this is a BUY (long) or SELL (short) position
    position_type = recommendation.get('position_type', 'long')
    action = "BUY" if position_type == 'long' else "SELL"

    output = f"""
### Strategy: {recommendation['name']}

**Decision:** ENTER TRADE ({action} TO OPEN)

**Setup:**
- Current Price: ${recommendation['current_price']:.2f}
- Expected Move: ${recommendation['expected_move']:.2f} ({recommendation['predicted_volatility']*100:.2f}%)
- Call Strike: ${recommendation['call_strike']:.2f}
- Put Strike: ${recommendation['put_strike']:.2f}
- Spread Width: ${recommendation['spread_width']:.2f}

**Entry:**
- Timing: {recommendation['entry_timing']}
- **{action}** {recommendation['contracts']} x ${recommendation['call_strike']:.2f} Call
- **{action}** {recommendation['contracts']} x ${recommendation['put_strike']:.2f} Put

**Exit Rules:**
"""
    for rule in recommendation['exit_rules']:
        output += f"- {rule}\n"

    # Format risk metrics based on position type
    risk = recommendation['risk_metrics']
    if position_type == 'long':
        output += f"""
**Risk/Reward (LONG Position):**
- Premium Paid (Debit): ${risk['debit_paid']:.2f}
- Max Loss: ${risk['max_loss']:.2f} (limited to premium)
- Max Profit (Est): ${risk['max_profit_estimate']:.2f} (on expected move)
- Breakeven: ${risk['breakeven_lower']:.2f} - ${risk['breakeven_upper']:.2f}
- Win Probability: {risk['profit_probability']*100:.0f}%

**Rationale:** {recommendation['rationale']}
"""
    else:
        output += f"""
**Risk/Reward (SHORT Position):**
- Credit Received: ${risk.get('credit_received', 0):.2f}
- Max Profit: ${risk.get('max_profit', 0):.2f}
- Max Loss (Est): ${risk.get('max_loss_estimate', 0):.2f}
- Breakeven: ${risk['breakeven_lower']:.2f} - ${risk['breakeven_upper']:.2f}
- Win Probability: {risk['profit_probability']*100:.0f}%

**Rationale:** {recommendation['rationale']}
"""

    if position_size:
        if position_size.get('contracts', 0) > 0:
            output += f"""
**Position Sizing:**
- Recommended Contracts: {position_size['contracts']}
- Total Risk: ${position_size['total_risk']:.2f} ({position_size['risk_pct']:.2f}% of account)
- Max Profit Potential: ${position_size['max_profit']:.2f}
- Confidence Adjustment: {position_size['confidence_adjustment']:.1f}%
"""
        elif position_size.get('reason'):
            output += f"""
**Position Sizing:** SKIP
- Reason: {position_size['reason']}
"""

    return output
