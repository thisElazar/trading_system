"""
Position Sizer
==============
Calculates position sizes based on risk management rules.

Methods:
- Fixed fractional (risk X% per trade)
- ATR-based (volatility-adjusted)
- Volatility-targeted (scale to target portfolio vol)
- Kelly criterion (optimal sizing based on edge)
"""

import logging
from typing import Optional, Dict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Import from parent
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    TOTAL_CAPITAL, RISK_PER_TRADE, MAX_POSITION_SIZE,
    MAX_POSITIONS, CASH_BUFFER_PCT, TRANSACTION_COSTS_BPS,
    MAX_POSITION_PCT, MAX_DRAWDOWN_PCT,
    get_transaction_cost
)

logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    """Result of position sizing calculation."""
    symbol: str
    shares: int
    dollar_value: float
    risk_amount: float
    stop_distance: float
    position_pct: float  # % of portfolio
    method: str
    
    # Risk metrics
    expected_loss_at_stop: float = 0.0
    position_risk_pct: float = 0.0  # Risk as % of portfolio
    
    # Constraints applied
    was_capped: bool = False
    cap_reason: str = ""


class PositionSizer:
    """
    Position sizing calculator with multiple methods.
    
    Default method: ATR-based sizing that risks 2% per trade
    with a 2-ATR stop loss.
    """
    
    def __init__(self,
                 total_capital: float = TOTAL_CAPITAL,
                 risk_per_trade: float = RISK_PER_TRADE,
                 max_position_size: float = MAX_POSITION_SIZE,
                 max_positions: int = MAX_POSITIONS,
                 cash_buffer_pct: float = CASH_BUFFER_PCT,
                 max_position_pct: float = MAX_POSITION_PCT,
                 drawdown_pct: float = 0.0):

        self.total_capital = total_capital
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.max_positions = max_positions
        self.cash_buffer_pct = cash_buffer_pct
        self.max_position_pct = max_position_pct
        self.drawdown_pct = drawdown_pct  # Current drawdown (0.0 = no drawdown)

        # Track current allocation
        self._current_positions: Dict[str, float] = {}
        self._cash = total_capital
    
    def update_capital(self, total_capital: float, cash: float = None):
        """Update capital levels."""
        self.total_capital = total_capital
        if cash is not None:
            self._cash = cash
    
    def update_positions(self, positions: Dict[str, float]):
        """Update current position values."""
        self._current_positions = positions
        positions_value = sum(positions.values())
        self._cash = self.total_capital - positions_value

    def update_drawdown(self, drawdown_pct: float):
        """Update current drawdown level (e.g., 0.10 for 10% drawdown)."""
        self.drawdown_pct = max(0.0, drawdown_pct)

    @property
    def drawdown_multiplier(self) -> float:
        """
        Calculate position size multiplier based on drawdown.

        If drawdown exceeds MAX_DRAWDOWN_PCT, reduce position sizes by 50%.
        """
        if self.drawdown_pct > MAX_DRAWDOWN_PCT:
            logger.warning(
                f"Drawdown {self.drawdown_pct:.1%} exceeds limit {MAX_DRAWDOWN_PCT:.1%}, "
                f"reducing position sizes by 50%"
            )
            return 0.5
        return 1.0
    
    @property
    def available_capital(self) -> float:
        """Capital available for new positions."""
        buffer = self.total_capital * self.cash_buffer_pct
        return max(self._cash - buffer, 0)
    
    @property
    def can_open_position(self) -> bool:
        """Check if we can open a new position."""
        return (len(self._current_positions) < self.max_positions and
                self.available_capital > 1000)  # Min $1000 for new position
    
    def calculate_atr_based(self,
                            symbol: str,
                            price: float,
                            atr: float,
                            stop_multiplier: float = 2.0) -> PositionSize:
        """
        Calculate position size based on ATR stop loss.
        
        Formula: Shares = (Risk Amount) / (Stop Distance)
        Where: Stop Distance = ATR * multiplier
        
        Args:
            symbol: Stock symbol
            price: Current price
            atr: Average True Range
            stop_multiplier: ATR multiplier for stop (default 2x)
            
        Returns:
            PositionSize with calculated values
        """
        # Risk amount = portfolio * risk_per_trade
        risk_amount = self.total_capital * self.risk_per_trade
        
        # Stop distance
        stop_distance = atr * stop_multiplier
        
        # Shares to risk exactly risk_amount at stop
        if stop_distance > 0:
            shares = risk_amount / stop_distance
        else:
            shares = risk_amount / (price * 0.05)  # Default to 5% stop
        
        # Dollar value
        dollar_value = shares * price
        
        # Apply drawdown reduction if in significant drawdown
        drawdown_mult = self.drawdown_multiplier
        dollar_value *= drawdown_mult

        # Apply constraints
        was_capped = False
        cap_reason = ""

        # Cap at max position size (absolute dollar limit)
        if dollar_value > self.max_position_size:
            dollar_value = self.max_position_size
            shares = dollar_value / price
            was_capped = True
            cap_reason = "max_position_size"

        # Cap at available capital
        if dollar_value > self.available_capital:
            dollar_value = self.available_capital * 0.95  # Leave small buffer
            shares = dollar_value / price
            was_capped = True
            cap_reason = "available_capital"

        # Cap at MAX_POSITION_PCT of portfolio (default 5%)
        max_pct_value = self.total_capital * self.max_position_pct
        if dollar_value > max_pct_value:
            dollar_value = max_pct_value
            shares = dollar_value / price
            was_capped = True
            cap_reason = f"max_position_pct ({self.max_position_pct:.0%})"
        
        # Round to whole shares
        shares = int(shares)
        dollar_value = shares * price
        
        # Recalculate risk with actual shares
        actual_risk = shares * stop_distance
        position_risk_pct = actual_risk / self.total_capital * 100
        
        return PositionSize(
            symbol=symbol,
            shares=shares,
            dollar_value=dollar_value,
            risk_amount=actual_risk,
            stop_distance=stop_distance,
            position_pct=dollar_value / self.total_capital * 100,
            method='atr_based',
            expected_loss_at_stop=actual_risk,
            position_risk_pct=position_risk_pct,
            was_capped=was_capped,
            cap_reason=cap_reason
        )
    
    def calculate_volatility_targeted(self,
                                       symbol: str,
                                       price: float,
                                       realized_vol: float,
                                       target_vol: float = 0.15,
                                       strategy_allocation: float = 1.0) -> PositionSize:
        """
        Calculate position size to target specific volatility contribution.
        
        Used by vol-managed momentum strategy.
        
        Formula: Weight = (Target Vol / Stock Vol) * Strategy Allocation
        
        Args:
            symbol: Stock symbol
            price: Current price
            realized_vol: Stock's annualized volatility (e.g., 0.25 for 25%)
            target_vol: Target portfolio volatility (default 15%)
            strategy_allocation: Strategy's allocation as fraction of portfolio
            
        Returns:
            PositionSize with calculated values
        """
        # Base allocation for this strategy
        strategy_capital = self.total_capital * strategy_allocation
        
        # Vol-weighted position size
        if realized_vol > 0:
            vol_weight = target_vol / realized_vol
        else:
            vol_weight = 1.0
        
        # Cap vol weight to prevent extreme sizing
        vol_weight = min(vol_weight, 2.0)
        vol_weight = max(vol_weight, 0.25)
        
        # Dollar value
        dollar_value = strategy_capital * vol_weight * 0.10  # 10% base position

        # Apply drawdown reduction if in significant drawdown
        dollar_value *= self.drawdown_multiplier

        # Apply constraints
        was_capped = False
        cap_reason = ""

        if dollar_value > self.max_position_size:
            dollar_value = self.max_position_size
            was_capped = True
            cap_reason = "max_position_size"

        if dollar_value > self.available_capital:
            dollar_value = self.available_capital * 0.95
            was_capped = True
            cap_reason = "available_capital"

        # Cap at MAX_POSITION_PCT of portfolio (default 5%)
        max_pct_value = self.total_capital * self.max_position_pct
        if dollar_value > max_pct_value:
            dollar_value = max_pct_value
            was_capped = True
            cap_reason = f"max_position_pct ({self.max_position_pct:.0%})"

        shares = int(dollar_value / price)
        dollar_value = shares * price
        
        # Estimate risk (using 2 std dev move)
        risk_amount = dollar_value * realized_vol * 2 / np.sqrt(252) * 21  # 21-day horizon
        
        return PositionSize(
            symbol=symbol,
            shares=shares,
            dollar_value=dollar_value,
            risk_amount=risk_amount,
            stop_distance=price * realized_vol * 2 / np.sqrt(252),
            position_pct=dollar_value / self.total_capital * 100,
            method='volatility_targeted',
            expected_loss_at_stop=risk_amount,
            position_risk_pct=risk_amount / self.total_capital * 100,
            was_capped=was_capped,
            cap_reason=cap_reason
        )
    
    def calculate_fixed_fractional(self,
                                    symbol: str,
                                    price: float,
                                    fraction: float = 0.10) -> PositionSize:
        """
        Simple fixed fractional sizing.
        
        Args:
            symbol: Stock symbol
            price: Current price
            fraction: Fraction of capital per position (default 10%)
            
        Returns:
            PositionSize with calculated values
        """
        dollar_value = self.total_capital * fraction

        # Apply drawdown reduction if in significant drawdown
        dollar_value *= self.drawdown_multiplier

        # Apply constraints
        was_capped = False
        cap_reason = ""

        if dollar_value > self.max_position_size:
            dollar_value = self.max_position_size
            was_capped = True
            cap_reason = "max_position_size"

        if dollar_value > self.available_capital:
            dollar_value = self.available_capital * 0.95
            was_capped = True
            cap_reason = "available_capital"

        # Cap at MAX_POSITION_PCT of portfolio (default 5%)
        max_pct_value = self.total_capital * self.max_position_pct
        if dollar_value > max_pct_value:
            dollar_value = max_pct_value
            was_capped = True
            cap_reason = f"max_position_pct ({self.max_position_pct:.0%})"

        shares = int(dollar_value / price)
        dollar_value = shares * price

        # Estimate risk at 10% drawdown
        risk_amount = dollar_value * 0.10
        
        return PositionSize(
            symbol=symbol,
            shares=shares,
            dollar_value=dollar_value,
            risk_amount=risk_amount,
            stop_distance=price * 0.10,
            position_pct=dollar_value / self.total_capital * 100,
            method='fixed_fractional',
            expected_loss_at_stop=risk_amount,
            position_risk_pct=risk_amount / self.total_capital * 100,
            was_capped=was_capped,
            cap_reason=cap_reason
        )
    
    def calculate_kelly(self,
                        symbol: str,
                        price: float,
                        win_rate: float,
                        avg_win: float,
                        avg_loss: float,
                        kelly_fraction: float = 0.25) -> PositionSize:
        """
        Kelly criterion position sizing.
        
        Formula: f* = (p * b - q) / b
        Where: p = win rate, q = 1 - p, b = win/loss ratio
        
        We use fractional Kelly (typically 25%) to reduce variance.
        
        Args:
            symbol: Stock symbol
            price: Current price
            win_rate: Historical win rate (e.g., 0.55)
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive number)
            kelly_fraction: Fraction of full Kelly to use (default 0.25)
            
        Returns:
            PositionSize with calculated values
        """
        if avg_loss == 0:
            avg_loss = 0.01  # Prevent division by zero
        
        p = win_rate
        q = 1 - win_rate
        b = abs(avg_win / avg_loss)  # Win/loss ratio
        
        # Full Kelly
        kelly = (p * b - q) / b if b > 0 else 0
        
        # Apply fraction
        position_fraction = kelly * kelly_fraction

        # Sanity limits - cap at max_position_pct
        position_fraction = max(min(position_fraction, self.max_position_pct), 0)

        dollar_value = self.total_capital * position_fraction

        # Apply drawdown reduction if in significant drawdown
        dollar_value *= self.drawdown_multiplier

        # Apply constraints
        was_capped = False
        cap_reason = ""

        if dollar_value > self.max_position_size:
            dollar_value = self.max_position_size
            was_capped = True
            cap_reason = "max_position_size"

        if dollar_value > self.available_capital:
            dollar_value = self.available_capital * 0.95
            was_capped = True
            cap_reason = "available_capital"

        # Cap at MAX_POSITION_PCT of portfolio (default 5%)
        max_pct_value = self.total_capital * self.max_position_pct
        if dollar_value > max_pct_value:
            dollar_value = max_pct_value
            was_capped = True
            cap_reason = f"max_position_pct ({self.max_position_pct:.0%})"

        shares = int(dollar_value / price)
        dollar_value = shares * price
        
        risk_amount = dollar_value * avg_loss
        
        return PositionSize(
            symbol=symbol,
            shares=shares,
            dollar_value=dollar_value,
            risk_amount=risk_amount,
            stop_distance=price * avg_loss,
            position_pct=dollar_value / self.total_capital * 100,
            method='kelly',
            expected_loss_at_stop=risk_amount,
            position_risk_pct=risk_amount / self.total_capital * 100,
            was_capped=was_capped,
            cap_reason=cap_reason
        )
    
    def calculate_for_signal(self,
                             symbol: str,
                             price: float,
                             atr: float = None,
                             realized_vol: float = None,
                             signal_strength: float = 1.0,
                             strategy_allocation: float = None) -> PositionSize:
        """
        Calculate position size based on available information.
        
        Selects appropriate method based on inputs.
        
        Args:
            symbol: Stock symbol
            price: Current price
            atr: ATR if available
            realized_vol: Realized volatility if available
            signal_strength: Signal strength (0-1) to scale position
            strategy_allocation: Strategy allocation if specified
            
        Returns:
            PositionSize with calculated values
        """
        if atr is not None and atr > 0:
            result = self.calculate_atr_based(symbol, price, atr)
        elif realized_vol is not None and realized_vol > 0:
            result = self.calculate_volatility_targeted(
                symbol, price, realized_vol,
                strategy_allocation=strategy_allocation or 1.0
            )
        else:
            result = self.calculate_fixed_fractional(symbol, price)
        
        # Scale by signal strength
        if signal_strength < 1.0:
            result.shares = int(result.shares * signal_strength)
            result.dollar_value = result.shares * price
            result.risk_amount *= signal_strength
            result.position_pct = result.dollar_value / self.total_capital * 100
        
        return result


# Convenience function
def calculate_position_size(symbol: str, price: float, atr: float,
                            capital: float = TOTAL_CAPITAL) -> PositionSize:
    """Quick position size calculation."""
    sizer = PositionSizer(total_capital=capital)
    return sizer.calculate_atr_based(symbol, price, atr)


if __name__ == "__main__":
    print("=" * 60)
    print("Position Sizer Test")
    print("=" * 60)
    
    sizer = PositionSizer(
        total_capital=100000,
        risk_per_trade=0.02,
        max_position_size=15000
    )
    
    print(f"\nCapital: ${sizer.total_capital:,.0f}")
    print(f"Risk per trade: {sizer.risk_per_trade:.1%}")
    print(f"Max position: ${sizer.max_position_size:,.0f}")
    
    # Test ATR-based sizing
    print("\n" + "-" * 40)
    print("ATR-Based Sizing:")
    
    result = sizer.calculate_atr_based(
        symbol="AAPL",
        price=175.00,
        atr=3.50,
        stop_multiplier=2.0
    )
    
    print(f"  Symbol: {result.symbol}")
    print(f"  Price: ${175:.2f}")
    print(f"  ATR: ${3.50:.2f}")
    print(f"  Stop distance: ${result.stop_distance:.2f}")
    print(f"  Shares: {result.shares}")
    print(f"  Dollar value: ${result.dollar_value:,.2f}")
    print(f"  Position %: {result.position_pct:.1f}%")
    print(f"  Risk at stop: ${result.risk_amount:,.2f}")
    print(f"  Risk %: {result.position_risk_pct:.2f}%")
    print(f"  Capped: {result.was_capped} ({result.cap_reason})")
    
    # Test vol-targeted sizing
    print("\n" + "-" * 40)
    print("Volatility-Targeted Sizing:")
    
    result = sizer.calculate_volatility_targeted(
        symbol="NVDA",
        price=450.00,
        realized_vol=0.45,  # 45% vol
        target_vol=0.15,
        strategy_allocation=0.25
    )
    
    print(f"  Symbol: {result.symbol}")
    print(f"  Price: ${450:.2f}")
    print(f"  Realized Vol: 45%")
    print(f"  Target Vol: 15%")
    print(f"  Shares: {result.shares}")
    print(f"  Dollar value: ${result.dollar_value:,.2f}")
    print(f"  Position %: {result.position_pct:.1f}%")
    
    # Test Kelly sizing
    print("\n" + "-" * 40)
    print("Kelly Criterion Sizing:")
    
    result = sizer.calculate_kelly(
        symbol="MSFT",
        price=375.00,
        win_rate=0.55,
        avg_win=0.08,
        avg_loss=0.05,
        kelly_fraction=0.25
    )
    
    print(f"  Symbol: {result.symbol}")
    print(f"  Win rate: 55%")
    print(f"  Avg win: 8%, Avg loss: 5%")
    print(f"  Shares: {result.shares}")
    print(f"  Dollar value: ${result.dollar_value:,.2f}")
    print(f"  Position %: {result.position_pct:.1f}%")
