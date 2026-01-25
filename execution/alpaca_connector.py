"""
Alpaca Broker Integration
=========================
Connects the trading system to Alpaca for paper/live trading.

Features:
- Order execution (market, limit, bracket)
- Position sync from broker
- Account info and buying power
- Order status tracking
- Timeout protection on all API calls

Safety features:
- All external API calls wrapped with timeouts
- Configurable timeout values via utils.timeout
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pandas as pd

# Import timeout utilities
from utils.timeout import timeout_wrapper, TIMEOUTS

# Alpaca imports
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest, 
        LimitOrderRequest,
        StopLimitOrderRequest,
        GetOrdersRequest
    )
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus, QueryOrderStatus
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("Warning: alpaca-py not installed. Run: pip install alpaca-py")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DIRS
from execution.signal_tracker import Position

logger = logging.getLogger(__name__)


@dataclass
class BrokerPosition:
    """Position as reported by broker."""
    symbol: str
    qty: float
    side: str  # 'long' or 'short'
    avg_entry_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    current_price: float


@dataclass  
class BrokerOrder:
    """Order as reported by broker."""
    id: str
    symbol: str
    side: str
    qty: float
    order_type: str
    status: str
    filled_qty: float
    filled_avg_price: Optional[float]
    submitted_at: datetime
    filled_at: Optional[datetime]


@dataclass
class AccountInfo:
    """Account summary from broker."""
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    day_trade_count: int
    pattern_day_trader: bool
    trading_blocked: bool


class AlpacaConnector:
    """
    Connects to Alpaca for order execution and account management.
    
    Usage:
        connector = AlpacaConnector(api_key, secret_key, paper=True)
        
        # Get account
        account = connector.get_account()
        print(f"Buying power: ${account.buying_power:,.2f}")
        
        # Get positions
        positions = connector.get_positions()
        
        # Execute order
        order = connector.submit_market_order('AAPL', 10, 'buy')
        
        # Sync with local database
        connector.sync_positions(signal_db)
    """
    
    def __init__(
        self, 
        api_key: str = None, 
        secret_key: str = None,
        paper: bool = True
    ):
        if not ALPACA_AVAILABLE:
            raise ImportError("alpaca-py required: pip install alpaca-py")
        
        # Load from env if not provided
        if not api_key or not secret_key:
            import os
            from dotenv import load_dotenv
            load_dotenv()
            api_key = api_key or os.getenv('ALPACA_API_KEY')
            secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')
        
        if not api_key or not secret_key:
            raise ValueError("Alpaca API credentials required")
        
        self.paper = paper
        self.trading_client = TradingClient(api_key, secret_key, paper=paper)
        self.data_client = StockHistoricalDataClient(api_key, secret_key)
        
        mode = "PAPER" if paper else "LIVE"
        logger.info(f"Alpaca connector initialized ({mode} trading)")
    
    # -------------------------------------------------------------------------
    # Account & Positions
    # -------------------------------------------------------------------------
    
    def get_account(self) -> Optional[AccountInfo]:
        """Get account information."""
        try:
            account = timeout_wrapper(
                self.trading_client.get_account,
                TIMEOUTS.API_CALL,
                "get_account"
            )

            return AccountInfo(
                equity=float(account.equity),
                cash=float(account.cash),
                buying_power=float(account.buying_power),
                portfolio_value=float(account.portfolio_value),
                day_trade_count=account.daytrade_count,
                pattern_day_trader=account.pattern_day_trader,
                trading_blocked=account.trading_blocked
            )
        except TimeoutError:
            logger.error("get_account timed out")
            return None
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return None

    def get_positions(self) -> List[BrokerPosition]:
        """Get all open positions from broker."""
        try:
            positions = timeout_wrapper(
                self.trading_client.get_all_positions,
                TIMEOUTS.API_CALL,
                "get_positions"
            )
        except TimeoutError:
            logger.error("get_positions timed out")
            return []
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

        result = []
        for pos in positions:
            try:
                result.append(BrokerPosition(
                    symbol=pos.symbol,
                    qty=float(pos.qty),
                    side='long' if float(pos.qty) > 0 else 'short',
                    avg_entry_price=float(pos.avg_entry_price),
                    market_value=float(pos.market_value),
                    unrealized_pnl=float(pos.unrealized_pl),
                    unrealized_pnl_pct=float(pos.unrealized_plpc) * 100,
                    current_price=float(pos.current_price)
                ))
            except Exception as e:
                logger.warning(f"Failed to parse position {pos.symbol}: {e}")
                continue

        return result
    
    def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        """Get position for a specific symbol."""
        try:
            pos = timeout_wrapper(
                lambda: self.trading_client.get_open_position(symbol),
                TIMEOUTS.API_CALL,
                f"get_position({symbol})"
            )
            return BrokerPosition(
                symbol=pos.symbol,
                qty=float(pos.qty),
                side='long' if float(pos.qty) > 0 else 'short',
                avg_entry_price=float(pos.avg_entry_price),
                market_value=float(pos.market_value),
                unrealized_pnl=float(pos.unrealized_pl),
                unrealized_pnl_pct=float(pos.unrealized_plpc) * 100,
                current_price=float(pos.current_price)
            )
        except TimeoutError:
            logger.debug(f"get_position({symbol}) timed out")
            return None
        except Exception as e:
            logger.debug(f"Failed to get position for {symbol}: {e}")
            return None
    
    # -------------------------------------------------------------------------
    # Order Execution
    # -------------------------------------------------------------------------
    
    def submit_market_order(
        self,
        symbol: str,
        qty: int,
        side: str,  # 'buy' or 'sell'
        time_in_force: str = 'day',
        wait_for_fill: bool = True,
        fill_timeout_seconds: float = 30.0
    ) -> Optional[BrokerOrder]:
        """
        Submit a market order.

        Args:
            symbol: Stock symbol
            qty: Number of shares
            side: 'buy' or 'sell'
            time_in_force: 'day' or 'gtc'
            wait_for_fill: If True, wait for order to fill and return actual filled qty
            fill_timeout_seconds: Max time to wait for fill

        Returns:
            BrokerOrder with actual filled_qty (may differ from requested qty)
        """
        try:
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
            tif = TimeInForce.DAY if time_in_force == 'day' else TimeInForce.GTC

            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=tif
            )

            order = timeout_wrapper(
                lambda: self.trading_client.submit_order(order_data),
                TIMEOUTS.ORDER_SUBMIT,
                f"submit_market_order({symbol})"
            )
            order_id = str(order.id)

            logger.info(f"✓ Market order submitted: {side.upper()} {qty} {symbol} (order_id={order_id})")

            # Wait for fill if requested
            if wait_for_fill:
                order = self._wait_for_order_fill(order_id, fill_timeout_seconds)
                if order is None:
                    logger.error(f"Order {order_id} did not fill within {fill_timeout_seconds}s")
                    return None

            filled_qty = float(order.filled_qty) if order.filled_qty else 0
            filled_price = float(order.filled_avg_price) if order.filled_avg_price else None

            # Log partial fill as ERROR for manual review
            if filled_qty > 0 and filled_qty < qty:
                logger.error(
                    f"PARTIAL FILL - NEEDS REVIEW: {symbol} requested {qty} shares, "
                    f"only {int(filled_qty)} filled ({filled_qty/qty*100:.1f}%). "
                    f"Order ID: {order_id}"
                )
            elif filled_qty == 0:
                logger.error(f"ORDER NOT FILLED: {symbol} {qty} shares - status={order.status.value}")

            return BrokerOrder(
                id=order_id,
                symbol=order.symbol,
                side=order.side.value,
                qty=float(order.qty),  # Requested qty
                order_type='market',
                status=order.status.value,
                filled_qty=filled_qty,  # ACTUAL filled qty
                filled_avg_price=filled_price,
                submitted_at=order.submitted_at,
                filled_at=order.filled_at
            )

        except Exception as e:
            logger.error(f"Order failed: {side} {qty} {symbol} - {e}")
            return None

    def _wait_for_order_fill(
        self,
        order_id: str,
        timeout_seconds: float = 30.0,
        poll_interval: float = 0.5
    ):
        """
        Wait for an order to reach a terminal state (filled, cancelled, etc).

        Args:
            order_id: The order ID to track
            timeout_seconds: Maximum time to wait
            poll_interval: Time between status checks

        Returns:
            The final order object, or None if timeout
        """
        import time
        start_time = time.time()

        terminal_statuses = {
            'filled', 'canceled', 'cancelled', 'expired', 'rejected',
            'done_for_day', 'replaced'
        }

        while time.time() - start_time < timeout_seconds:
            try:
                order = timeout_wrapper(
                    lambda: self.trading_client.get_order_by_id(order_id),
                    TIMEOUTS.API_CALL,
                    f"get_order_by_id({order_id})"
                )
                status = order.status.value.lower()

                if status in terminal_statuses:
                    return order

                # Also check if partially filled and no more fills coming
                if status == 'partially_filled':
                    # For market orders during market hours, partial fills
                    # should complete quickly. If stuck, return what we have.
                    if time.time() - start_time > 10.0:
                        logger.warning(f"Order {order_id} stuck in partial fill, returning current state")
                        return order

                time.sleep(poll_interval)

            except TimeoutError:
                logger.warning(f"Timeout checking order {order_id}, retrying...")
                time.sleep(poll_interval)
            except Exception as e:
                logger.warning(f"Error checking order {order_id}: {e}")
                time.sleep(poll_interval)

        # Timeout - return last known state
        try:
            return timeout_wrapper(
                lambda: self.trading_client.get_order_by_id(order_id),
                TIMEOUTS.API_CALL,
                f"get_order_by_id_final({order_id})"
            )
        except Exception:
            return None
    
    def submit_limit_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        limit_price: float,
        time_in_force: str = 'day'
    ) -> Optional[BrokerOrder]:
        """Submit a limit order."""
        try:
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
            tif = TimeInForce.DAY if time_in_force == 'day' else TimeInForce.GTC
            
            order_data = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                limit_price=limit_price,
                time_in_force=tif
            )

            order = timeout_wrapper(
                lambda: self.trading_client.submit_order(order_data),
                TIMEOUTS.ORDER_SUBMIT,
                f"submit_limit_order({symbol})"
            )

            logger.info(f"✓ Limit order submitted: {side.upper()} {qty} {symbol} @ ${limit_price:.2f}")
            
            return BrokerOrder(
                id=str(order.id),
                symbol=order.symbol,
                side=order.side.value,
                qty=float(order.qty),
                order_type='limit',
                status=order.status.value,
                filled_qty=float(order.filled_qty) if order.filled_qty else 0,
                filled_avg_price=float(order.filled_avg_price) if order.filled_avg_price else None,
                submitted_at=order.submitted_at,
                filled_at=order.filled_at
            )
            
        except Exception as e:
            logger.error(f"Limit order failed: {side} {qty} {symbol} @ ${limit_price} - {e}")
            return None
    
    def submit_bracket_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        take_profit: float,
        stop_loss: float,
        limit_price: float = None  # If None, uses market order for entry
    ) -> Optional[str]:
        """
        Submit a bracket order (entry + take profit + stop loss).
        
        Returns order ID if successful.
        """
        try:
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
            
            # Alpaca bracket order
            from alpaca.trading.requests import (
                MarketOrderRequest,
                TakeProfitRequest,
                StopLossRequest
            )
            
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.GTC,
                order_class='bracket',
                take_profit=TakeProfitRequest(limit_price=take_profit),
                stop_loss=StopLossRequest(stop_price=stop_loss)
            )

            order = timeout_wrapper(
                lambda: self.trading_client.submit_order(order_data),
                TIMEOUTS.ORDER_SUBMIT,
                f"submit_bracket_order({symbol})"
            )

            logger.info(
                f"✓ Bracket order: {side.upper()} {qty} {symbol} | "
                f"TP: ${take_profit:.2f} | SL: ${stop_loss:.2f}"
            )

            return str(order.id)
            
        except Exception as e:
            logger.error(f"Bracket order failed: {symbol} - {e}")
            return None
    
    def close_position(self, symbol: str) -> bool:
        """Close entire position for a symbol."""
        try:
            timeout_wrapper(
                lambda: self.trading_client.close_position(symbol),
                TIMEOUTS.ORDER_SUBMIT,
                f"close_position({symbol})"
            )
            logger.info(f"✓ Closed position: {symbol}")
            return True
        except TimeoutError:
            logger.error(f"close_position({symbol}) timed out")
            return False
        except Exception as e:
            logger.error(f"Failed to close {symbol}: {e}")
            return False
    
    def close_all_positions(self) -> bool:
        """Close all positions (emergency)."""
        try:
            timeout_wrapper(
                lambda: self.trading_client.close_all_positions(cancel_orders=True),
                TIMEOUTS.ORDER_SUBMIT,
                "close_all_positions"
            )
            logger.warning("⚠ Closed ALL positions")
            return True
        except TimeoutError:
            logger.error("close_all_positions timed out")
            return False
        except Exception as e:
            logger.error(f"Failed to close all positions: {e}")
            return False
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        try:
            timeout_wrapper(
                lambda: self.trading_client.cancel_order_by_id(order_id),
                TIMEOUTS.API_CALL,
                f"cancel_order({order_id})"
            )
            logger.info(f"✓ Cancelled order: {order_id}")
            return True
        except TimeoutError:
            logger.error(f"cancel_order({order_id}) timed out")
            return False
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    def get_order(self, order_id: str) -> Optional[BrokerOrder]:
        """Get order status."""
        try:
            order = timeout_wrapper(
                lambda: self.trading_client.get_order_by_id(order_id),
                TIMEOUTS.API_CALL,
                f"get_order({order_id})"
            )
            return BrokerOrder(
                id=str(order.id),
                symbol=order.symbol,
                side=order.side.value,
                qty=float(order.qty),
                order_type=order.order_type.value,
                status=order.status.value,
                filled_qty=float(order.filled_qty) if order.filled_qty else 0,
                filled_avg_price=float(order.filled_avg_price) if order.filled_avg_price else None,
                submitted_at=order.submitted_at,
                filled_at=order.filled_at
            )
        except TimeoutError:
            logger.error(f"get_order({order_id}) timed out")
            return None
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return None
    
    def get_open_orders(self) -> List[BrokerOrder]:
        """Get all open orders."""
        try:
            request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            orders = timeout_wrapper(
                lambda: self.trading_client.get_orders(request),
                TIMEOUTS.API_CALL,
                "get_open_orders"
            )

            return [
                BrokerOrder(
                    id=str(o.id),
                    symbol=o.symbol,
                    side=o.side.value,
                    qty=float(o.qty),
                    order_type=o.order_type.value,
                    status=o.status.value,
                    filled_qty=float(o.filled_qty) if o.filled_qty else 0,
                    filled_avg_price=float(o.filled_avg_price) if o.filled_avg_price else None,
                    submitted_at=o.submitted_at,
                    filled_at=o.filled_at
                )
                for o in orders
            ]
        except TimeoutError:
            logger.error("get_open_orders timed out")
            return []
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []
    
    # -------------------------------------------------------------------------
    # Market Data
    # -------------------------------------------------------------------------
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest trade price for a symbol."""
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = timeout_wrapper(
                lambda: self.data_client.get_stock_latest_quote(request),
                TIMEOUTS.API_CALL,
                f"get_latest_price({symbol})"
            )

            if symbol in quotes:
                quote = quotes[symbol]
                # Use midpoint of bid/ask
                return (float(quote.bid_price) + float(quote.ask_price)) / 2
            return None
        except TimeoutError:
            logger.debug(f"get_latest_price({symbol}) timed out")
            return None
        except Exception as e:
            logger.debug(f"Failed to get quote for {symbol}: {e}")
            return None
    
    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get latest prices for multiple symbols."""
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbols)
            quotes = timeout_wrapper(
                lambda: self.data_client.get_stock_latest_quote(request),
                TIMEOUTS.DATA_FETCH,
                "get_latest_prices"
            )

            result = {}
            for symbol, quote in quotes.items():
                result[symbol] = (float(quote.bid_price) + float(quote.ask_price)) / 2
            return result
        except TimeoutError:
            logger.error("get_latest_prices timed out")
            return {}
        except Exception as e:
            logger.error(f"Failed to get quotes: {e}")
            return {}
    
    def get_bars(
        self,
        symbol: str,
        days: int = 60,
        timeframe: str = 'day'
    ) -> pd.DataFrame:
        """Get historical bars."""
        try:
            tf = TimeFrame.Day if timeframe == 'day' else TimeFrame.Hour

            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=datetime.now() - timedelta(days=days),
                end=datetime.now()
            )
            bars = timeout_wrapper(
                lambda: self.data_client.get_stock_bars(request),
                TIMEOUTS.DATA_FETCH,
                f"get_bars({symbol})"
            )

            df = bars.df.reset_index()
            return df

        except TimeoutError:
            logger.error(f"get_bars({symbol}) timed out")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to get bars for {symbol}: {e}")
            return pd.DataFrame()
    
    # -------------------------------------------------------------------------
    # Position Sync
    # -------------------------------------------------------------------------
    
    def sync_positions(self, signal_db) -> Tuple[int, int, int]:
        """
        Sync broker positions with local database.
        
        Returns:
            (new_positions, updated_positions, closed_positions)
        """
        broker_positions = {p.symbol: p for p in self.get_positions()}
        
        new_count = 0
        updated_count = 0
        closed_count = 0
        
        # Get local open positions
        local_positions = signal_db.get_open_positions()
        local_symbols = {p.symbol for p in local_positions}
        
        # Check for positions in broker but not in DB (opened externally)
        for symbol, bp in broker_positions.items():
            if symbol not in local_symbols:
                logger.warning(f"Position {symbol} in broker but not in DB - syncing")
                position = Position(
                    signal_id=0,  # No signal for externally opened positions
                    strategy_name='manual',
                    symbol=symbol,
                    direction='long' if bp.side == 'long' else 'short',
                    quantity=int(bp.qty),
                    entry_price=bp.avg_entry_price,
                    current_price=bp.current_price,
                    stop_loss=bp.avg_entry_price * 0.95,  # Default 5% stop
                    take_profit=bp.avg_entry_price * 1.10  # Default 10% target
                )
                signal_db.open_position(position)
                new_count += 1
        
        # Check for positions in DB but not in broker (closed externally)
        for lp in local_positions:
            if lp.symbol not in broker_positions:
                logger.warning(f"Position {lp.symbol} in DB but not in broker - marking closed")
                # Use the last known current_price as exit_price, or entry_price if not available
                exit_price = lp.current_price if lp.current_price else lp.entry_price
                signal_db.close_position(lp.id, exit_price, 'broker_sync')
                closed_count += 1
            else:
                # Sync with broker data - use broker as source of truth
                bp = broker_positions[lp.symbol]

                # Check if entry price or quantity differs (broker is authoritative)
                entry_diff = abs(lp.entry_price - bp.avg_entry_price) / bp.avg_entry_price if bp.avg_entry_price > 0 else 0
                qty_diff = lp.quantity != int(bp.qty)

                if entry_diff > 0.01 or qty_diff:
                    # Use full sync to correct entry_price, quantity, and recalculate TP/SL
                    signal_db.sync_position_with_broker(
                        lp.id,
                        entry_price=bp.avg_entry_price,
                        quantity=int(bp.qty),
                        current_price=bp.current_price,
                        unrealized_pnl=bp.unrealized_pnl
                    )
                else:
                    # Just update current price
                    signal_db.update_position_price(lp.id, bp.current_price)
                updated_count += 1
        
        logger.info(f"Position sync: {new_count} new, {updated_count} updated, {closed_count} closed")
        return new_count, updated_count, closed_count
    
    def summary(self) -> str:
        """Get account and position summary."""
        account = self.get_account()
        positions = self.get_positions()
        
        lines = [
            "=" * 50,
            f"ALPACA ACCOUNT ({'PAPER' if self.paper else 'LIVE'})",
            "=" * 50,
            f"Portfolio Value: ${account.portfolio_value:,.2f}",
            f"Cash: ${account.cash:,.2f}",
            f"Buying Power: ${account.buying_power:,.2f}",
            f"Day Trades: {account.day_trade_count}",
            "",
            f"Open Positions: {len(positions)}"
        ]
        
        if positions:
            for p in positions:
                pnl_sign = "+" if p.unrealized_pnl >= 0 else ""
                lines.append(
                    f"  {p.symbol}: {p.qty} @ ${p.avg_entry_price:.2f} → "
                    f"${p.current_price:.2f} ({pnl_sign}{p.unrealized_pnl_pct:.1f}%)"
                )
        
        return "\n".join(lines)


# Convenience function
def create_connector(paper: bool = True) -> AlpacaConnector:
    """Create connector using environment variables."""
    return AlpacaConnector(paper=paper)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Alpaca Connector...")
    print("(Requires ALPACA_API_KEY and ALPACA_SECRET_KEY in .env)")
    
    try:
        connector = create_connector(paper=True)
        print(connector.summary())
        
        # Test getting latest price
        price = connector.get_latest_price('AAPL')
        if price:
            print(f"\nAAPL latest price: ${price:.2f}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nSetup:")
        print("1. pip install alpaca-py python-dotenv")
        print("2. Create .env with ALPACA_API_KEY and ALPACA_SECRET_KEY")
        print("3. Get keys from https://alpaca.markets")
