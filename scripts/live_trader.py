#!/usr/bin/env python3
"""
Live Trading Runner
===================
Connects all components for live/paper trading.

Usage:
    python scripts/live_trader.py --mode paper
    python scripts/live_trader.py --mode paper --once  # Single scan
    python scripts/live_trader.py --status              # Show status only
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TOTAL_CAPITAL, VIX_REGIMES
from data.cached_data_manager import CachedDataManager
from execution.signal_tracker import SignalDatabase
from execution.ensemble import StrategyEnsemble, create_ensemble
from execution.alerts import create_alert_manager, AlertType, AlertLevel
from execution.scheduler import MarketHours

# Strategies
from strategies.gap_fill import GapFillStrategy
from strategies.pairs_trading import PairsTradingStrategy
from strategies.relative_volume_breakout import RelativeVolumeBreakout
from strategies.sector_rotation import SectorRotationStrategy

# Alpaca (optional)
try:
    from execution.alpaca_connector import AlpacaConnector, create_connector
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LiveTrader:
    """
    Live trading coordinator.
    
    Flow:
    1. Sync positions from broker
    2. Manage existing positions (check stops/targets)
    3. Generate new signals from strategies
    4. Aggregate via ensemble
    5. Execute top signals
    """
    
    def __init__(self, paper: bool = True, capital: float = None):
        self.paper = paper
        self.capital = capital or TOTAL_CAPITAL
        
        # Initialize components
        self.data_mgr = CachedDataManager()
        self.signal_db = SignalDatabase()
        self.alerts = create_alert_manager()
        self.ensemble = create_ensemble(self.capital)
        self.market_hours = MarketHours()
        
        # Alpaca connector
        self.broker = None
        if ALPACA_AVAILABLE:
            try:
                self.broker = create_connector(paper=paper)
                logger.info(f"Broker connected ({'PAPER' if paper else 'LIVE'})")
            except Exception as e:
                logger.warning(f"Broker connection failed: {e}")
        
        # Initialize strategies
        self.strategies = {
            'gap_fill': GapFillStrategy(),
            'pairs_trading': PairsTradingStrategy(),
            'relative_volume_breakout': RelativeVolumeBreakout(),
            'sector_rotation': SectorRotationStrategy(),
        }
        
        # Register with ensemble
        for name in self.strategies:
            self.ensemble.register_strategy(name, enabled=True)
    
    def get_vix_regime(self) -> str:
        """Get current VIX regime."""
        vix = self.data_mgr.get_vix()
        if vix is not None:
            if vix < VIX_REGIMES["low"]:
                return 'low'
            elif vix < VIX_REGIMES["normal"]:
                return 'normal'
            elif vix < VIX_REGIMES["high"]:
                return 'high'
            else:
                return 'extreme'
        return 'normal'
    
    def sync_positions(self):
        """Sync positions from broker."""
        if not self.broker:
            return
        
        try:
            positions = self.broker.get_positions()
            for pos in positions:
                self.ensemble.update_position(pos.symbol, 'broker')
            logger.info(f"Synced {len(positions)} positions from broker")
        except Exception as e:
            logger.error(f"Position sync failed: {e}")
    
    def manage_positions(self):
        """Check existing positions for stop/target hits."""
        if not self.broker:
            return
        
        positions = self.broker.get_positions()
        
        for pos in positions:
            # Get current price
            price = self.broker.get_latest_price(pos.symbol)
            if not price:
                continue
            
            # Check trailing stop (15% from peak)
            # Note: Would need to track peak price in DB
            
            # Check if down >10% for >3 days
            if pos.unrealized_pnl_pct < -10:
                logger.warning(f"Position {pos.symbol} down {pos.unrealized_pnl_pct:.1f}% - consider closing")
                self.alerts.send(
                    AlertType.POSITION_UPDATE,
                    f"{pos.symbol} down {pos.unrealized_pnl_pct:.1f}%",
                    level=AlertLevel.WARNING
                )
    
    def generate_signals(self, vix_regime: str) -> dict:
        """Generate signals from all strategies."""
        all_signals = {}
        
        for name, strategy in self.strategies.items():
            if not self.ensemble.strategies.get(name, {}).get('enabled', True):
                continue
            
            try:
                signals = strategy.generate_signals(
                    data={},  # Strategies load their own data
                    current_positions=list(self.ensemble.current_positions.keys()),
                    vix_regime=vix_regime
                )
                
                if signals:
                    all_signals[name] = signals
                    logger.info(f"{name}: {len(signals)} signals")
                    
            except Exception as e:
                logger.error(f"Strategy {name} failed: {e}")
        
        return all_signals
    
    def execute_signals(self, decisions: list, max_orders: int = 3):
        """Execute top signals via broker."""
        if not self.broker:
            logger.warning("No broker - signals not executed")
            for d in decisions[:max_orders]:
                logger.info(f"  [DRY RUN] {d.direction.upper()} {d.symbol} (${d.capital_allocation:,.0f})")
            return
        
        account = self.broker.get_account()
        
        for decision in decisions[:max_orders]:
            # Check buying power
            if decision.capital_allocation > account.buying_power:
                logger.warning(f"Insufficient buying power for {decision.symbol}")
                continue
            
            # Get price and calculate shares
            price = self.broker.get_latest_price(decision.symbol)
            if not price:
                continue
            
            qty = int(decision.capital_allocation / price)
            if qty < 1:
                continue
            
            # Execute
            side = 'buy' if decision.direction == 'long' else 'sell'
            
            order = self.broker.submit_market_order(
                symbol=decision.symbol,
                qty=qty,
                side=side
            )
            
            if order:
                self.ensemble.update_position(decision.symbol, decision.sources[0])
                self.alerts.send(
                    AlertType.EXECUTION,
                    f"{side.upper()} {qty} {decision.symbol} @ ~${price:.2f}",
                    level=AlertLevel.INFO
                )
    
    def run_once(self):
        """Run single trading cycle."""
        logger.info("=" * 60)
        logger.info("TRADING CYCLE START")
        logger.info("=" * 60)
        
        # Check market hours
        if not self.market_hours.is_market_open():
            logger.info("Market closed")
            return
        
        # Get regime
        vix_regime = self.get_vix_regime()
        logger.info(f"VIX Regime: {vix_regime}")
        
        # Sync positions
        self.sync_positions()
        
        # Manage existing positions
        self.manage_positions()
        
        # Generate signals
        all_signals = self.generate_signals(vix_regime)
        
        # Add to ensemble
        for strategy_name, signals in all_signals.items():
            self.ensemble.add_signals(strategy_name, signals)
        
        # Get aggregated decisions
        decisions = self.ensemble.get_portfolio_decisions()
        
        if decisions:
            logger.info(f"\nTop Signals:")
            for d in decisions[:5]:
                logger.info(f"  {d.direction.upper()} {d.symbol} | "
                           f"strength={d.strength:.2f} | ${d.capital_allocation:,.0f}")
            
            # Execute
            self.execute_signals(decisions)
        else:
            logger.info("No signals meeting criteria")
        
        logger.info("=" * 60)
    
    def status(self):
        """Print current status."""
        print(self.ensemble.summary())
        
        if self.broker:
            print("\n" + self.broker.summary())


def main():
    parser = argparse.ArgumentParser(description="Live Trading Runner")
    parser.add_argument('--mode', choices=['paper', 'live'], default='paper')
    parser.add_argument('--once', action='store_true', help='Run single cycle')
    parser.add_argument('--status', action='store_true', help='Show status only')
    parser.add_argument('--capital', type=float, default=None, help='Override capital')
    
    args = parser.parse_args()
    
    if args.mode == 'live':
        confirm = input("⚠️  LIVE TRADING MODE - Are you sure? (type 'yes'): ")
        if confirm.lower() != 'yes':
            print("Aborted")
            return
    
    trader = LiveTrader(paper=(args.mode == 'paper'), capital=args.capital)
    
    if args.status:
        trader.status()
    elif args.once:
        trader.run_once()
    else:
        # Continuous mode - would use scheduler
        logger.info("Continuous mode not yet implemented - use --once")
        trader.run_once()


if __name__ == "__main__":
    main()
