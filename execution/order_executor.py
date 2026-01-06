"""
Order Executor
==============
Executes orders via Alpaca API (paper trading).

Features:
- Market and limit orders
- Bracket orders (entry + stop + target)
- Order status tracking
- Position management
"""

import logging
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Import from parent
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL
from strategies.base import Signal, SignalType

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status values."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass
class OrderResult:
    """Result of order submission."""
    success: bool
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: float = 0
    filled_price: float = 0
    message: str = ""
    
    # Timestamps
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None


class OrderExecutor:
    """
    Executes orders via Alpaca API.
    
    Supports:
    - Market orders
    - Limit orders
    - Bracket orders (OCO: entry + stop-loss + take-profit)
    """
    
    def __init__(self, paper: bool = True, circuit_breaker=None):
        """
        Initialize order executor.

        Args:
            paper: If True, use paper trading (default)
            circuit_breaker: Optional CircuitBreakerManager instance
        """
        self.paper = paper
        self.circuit_breaker = circuit_breaker
        self._client = None
        self._trading_client = None
        self._account = None
    
    def _ensure_client(self):
        """Ensure Alpaca client is initialized."""
        if self._trading_client is None:
            from alpaca.trading.client import TradingClient
            
            self._trading_client = TradingClient(
                ALPACA_API_KEY,
                ALPACA_SECRET_KEY,
                paper=self.paper
            )
            
            logger.info(f"Connected to Alpaca ({'Paper' if self.paper else 'Live'} Trading)")
    
    def get_account(self) -> dict:
        """Get account information."""
        self._ensure_client()
        
        account = self._trading_client.get_account()
        
        return {
            'equity': float(account.equity),
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'portfolio_value': float(account.portfolio_value),
            'pattern_day_trader': account.pattern_day_trader,
            'trading_blocked': account.trading_blocked,
            'account_blocked': account.account_blocked,
        }
    
    def get_positions(self) -> Dict[str, dict]:
        """Get all open positions."""
        self._ensure_client()
        
        positions = self._trading_client.get_all_positions()
        
        result = {}
        for pos in positions:
            result[pos.symbol] = {
                'symbol': pos.symbol,
                'qty': float(pos.qty),
                'side': 'long' if float(pos.qty) > 0 else 'short',
                'market_value': float(pos.market_value),
                'cost_basis': float(pos.cost_basis),
                'avg_entry_price': float(pos.avg_entry_price),
                'current_price': float(pos.current_price),
                'unrealized_pl': float(pos.unrealized_pl),
                'unrealized_plpc': float(pos.unrealized_plpc),
                'change_today': float(pos.change_today),
            }
        
        return result
    
    def get_position(self, symbol: str) -> Optional[dict]:
        """Get position for a specific symbol."""
        positions = self.get_positions()
        return positions.get(symbol)
    
    def submit_market_order(self,
                            symbol: str,
                            qty: float,
                            side: str,
                            time_in_force: str = 'day') -> OrderResult:
        """
        Submit a market order.
        
        Args:
            symbol: Stock symbol
            qty: Number of shares (positive)
            side: 'buy' or 'sell'
            time_in_force: 'day', 'gtc', 'ioc', etc.
            
        Returns:
            OrderResult with status
        """
        self._ensure_client()
        
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        
        try:
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
            tif = TimeInForce.DAY if time_in_force.lower() == 'day' else TimeInForce.GTC
            
            request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=tif
            )
            
            order = self._trading_client.submit_order(request)
            
            logger.info(f"Submitted market {side.upper()} {qty} {symbol}: {order.id}")
            
            return OrderResult(
                success=True,
                order_id=str(order.id),
                status=OrderStatus.SUBMITTED,
                message=f"Order submitted: {order.id}",
                submitted_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to submit order: {e}")
            return OrderResult(
                success=False,
                status=OrderStatus.FAILED,
                message=str(e)
            )
    
    def submit_bracket_order(self,
                             symbol: str,
                             qty: float,
                             side: str,
                             stop_loss: float,
                             take_profit: float,
                             time_in_force: str = 'gtc') -> OrderResult:
        """
        Submit a bracket order (entry + stop-loss + take-profit).
        
        Args:
            symbol: Stock symbol
            qty: Number of shares
            side: 'buy' or 'sell'
            stop_loss: Stop-loss price
            take_profit: Take-profit price
            time_in_force: Usually 'gtc' for brackets
            
        Returns:
            OrderResult with status
        """
        self._ensure_client()
        
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
        
        try:
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
            
            request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.GTC,
                order_class=OrderClass.BRACKET,
                stop_loss={'stop_price': round(stop_loss, 2)},
                take_profit={'limit_price': round(take_profit, 2)}
            )
            
            order = self._trading_client.submit_order(request)
            
            logger.info(f"Submitted bracket {side.upper()} {qty} {symbol}: "
                        f"SL=${stop_loss:.2f}, TP=${take_profit:.2f}")
            
            return OrderResult(
                success=True,
                order_id=str(order.id),
                status=OrderStatus.SUBMITTED,
                message=f"Bracket order submitted: {order.id}",
                submitted_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to submit bracket order: {e}")
            return OrderResult(
                success=False,
                status=OrderStatus.FAILED,
                message=str(e)
            )
    
    def submit_limit_order(self,
                           symbol: str,
                           qty: float,
                           side: str,
                           limit_price: float,
                           time_in_force: str = 'day') -> OrderResult:
        """
        Submit a limit order.
        
        Args:
            symbol: Stock symbol
            qty: Number of shares
            side: 'buy' or 'sell'
            limit_price: Limit price
            time_in_force: 'day', 'gtc', etc.
            
        Returns:
            OrderResult with status
        """
        self._ensure_client()
        
        from alpaca.trading.requests import LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        
        try:
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
            tif = TimeInForce.DAY if time_in_force.lower() == 'day' else TimeInForce.GTC
            
            request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                limit_price=round(limit_price, 2),
                time_in_force=tif
            )
            
            order = self._trading_client.submit_order(request)
            
            logger.info(f"Submitted limit {side.upper()} {qty} {symbol} @ ${limit_price:.2f}")
            
            return OrderResult(
                success=True,
                order_id=str(order.id),
                status=OrderStatus.SUBMITTED,
                message=f"Limit order submitted: {order.id}",
                submitted_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to submit limit order: {e}")
            return OrderResult(
                success=False,
                status=OrderStatus.FAILED,
                message=str(e)
            )
    
    def close_position(self, symbol: str) -> OrderResult:
        """
        Close entire position for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            OrderResult with status
        """
        self._ensure_client()
        
        try:
            order = self._trading_client.close_position(symbol)
            
            logger.info(f"Closed position: {symbol}")
            
            return OrderResult(
                success=True,
                order_id=str(order.id) if order else None,
                status=OrderStatus.SUBMITTED,
                message=f"Position closed: {symbol}",
                submitted_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to close position {symbol}: {e}")
            return OrderResult(
                success=False,
                status=OrderStatus.FAILED,
                message=str(e)
            )
    
    def close_all_positions(self) -> List[OrderResult]:
        """Close all open positions."""
        self._ensure_client()
        
        results = []
        
        try:
            orders = self._trading_client.close_all_positions(cancel_orders=True)
            
            for order in orders:
                results.append(OrderResult(
                    success=True,
                    order_id=str(order.id) if hasattr(order, 'id') else None,
                    status=OrderStatus.SUBMITTED,
                    message="Position closed",
                    submitted_at=datetime.now()
                ))
            
            logger.info(f"Closed all positions: {len(results)} orders")
            
        except Exception as e:
            logger.error(f"Failed to close all positions: {e}")
            results.append(OrderResult(
                success=False,
                status=OrderStatus.FAILED,
                message=str(e)
            ))
        
        return results
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        self._ensure_client()
        
        try:
            self._trading_client.cancel_order_by_id(order_id)
            logger.info(f"Cancelled order: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    def cancel_all_orders(self) -> int:
        """Cancel all open orders."""
        self._ensure_client()
        
        try:
            result = self._trading_client.cancel_orders()
            count = len(result) if result else 0
            logger.info(f"Cancelled {count} orders")
            return count
        except Exception as e:
            logger.error(f"Failed to cancel orders: {e}")
            return 0
    
    def get_order_status(self, order_id: str) -> Optional[dict]:
        """Get status of an order."""
        self._ensure_client()
        
        try:
            order = self._trading_client.get_order_by_id(order_id)
            
            return {
                'id': str(order.id),
                'symbol': order.symbol,
                'side': order.side.value,
                'qty': float(order.qty),
                'filled_qty': float(order.filled_qty) if order.filled_qty else 0,
                'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else 0,
                'status': order.status.value,
                'created_at': str(order.created_at),
                'filled_at': str(order.filled_at) if order.filled_at else None,
            }
        except Exception as e:
            logger.error(f"Failed to get order status: {e}")
            return None
    
    def execute_signal(self, signal: Signal) -> OrderResult:
        """
        Execute a trading signal.

        Args:
            signal: Signal from strategy

        Returns:
            OrderResult with status
        """
        # CIRCUIT BREAKER CHECK - First line of defense
        if self.circuit_breaker:
            if not self.circuit_breaker.can_trade():
                logger.warning(f"Order blocked by circuit breaker: {signal.symbol}")
                return OrderResult(
                    success=False,
                    status=OrderStatus.REJECTED,
                    message="Trading halted by circuit breaker"
                )

            strategy_name = signal.strategy if hasattr(signal, 'strategy') else None
            if strategy_name and not self.circuit_breaker.can_run_strategy(strategy_name):
                logger.warning(f"Strategy {strategy_name} blocked by circuit breaker")
                return OrderResult(
                    success=False,
                    status=OrderStatus.REJECTED,
                    message=f"Strategy {strategy_name} paused by circuit breaker"
                )

        if signal.signal_type == SignalType.BUY:
            # Use bracket order if stop/target provided
            if signal.stop_loss and signal.target_price:
                return self.submit_bracket_order(
                    symbol=signal.symbol,
                    qty=signal.metadata.get('shares', 1),
                    side='buy',
                    stop_loss=signal.stop_loss,
                    take_profit=signal.target_price
                )
            else:
                return self.submit_market_order(
                    symbol=signal.symbol,
                    qty=signal.metadata.get('shares', 1),
                    side='buy'
                )
        
        elif signal.signal_type == SignalType.SELL:
            return self.submit_market_order(
                symbol=signal.symbol,
                qty=signal.metadata.get('shares', 1),
                side='sell'
            )
        
        elif signal.signal_type == SignalType.CLOSE:
            return self.close_position(signal.symbol)
        
        else:
            return OrderResult(
                success=False,
                status=OrderStatus.FAILED,
                message=f"Unknown signal type: {signal.signal_type}"
            )


# Convenience functions

def get_account_info() -> dict:
    """Quick access to account info."""
    executor = OrderExecutor()
    return executor.get_account()


def get_all_positions() -> dict:
    """Quick access to positions."""
    executor = OrderExecutor()
    return executor.get_positions()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Order Executor Test")
    print("=" * 60)
    
    executor = OrderExecutor(paper=True)
    
    # Get account info
    print("\nAccount Info:")
    try:
        account = executor.get_account()
        for key, value in account.items():
            if isinstance(value, float):
                print(f"  {key}: ${value:,.2f}")
            else:
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"  Error: {e}")
        print("  (Set ALPACA_API_KEY and ALPACA_SECRET_KEY)")
    
    # Get positions
    print("\nOpen Positions:")
    try:
        positions = executor.get_positions()
        if positions:
            for symbol, pos in positions.items():
                print(f"  {symbol}: {pos['qty']} shares, "
                      f"P&L: ${pos['unrealized_pl']:,.2f}")
        else:
            print("  No open positions")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n" + "-" * 40)
    print("Note: To test order submission, use valid API credentials.")
    print("See OrderExecutor.submit_market_order() for usage.")
