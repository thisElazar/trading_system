#!/usr/bin/env python3
"""
Paper Trading Runner
====================
Main entry point for running the trading system in paper mode.

Usage:
    python3 scripts/paper_trade.py          # Run once (generate signals)
    python3 scripts/paper_trade.py --live   # Continuous monitoring
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import STRATEGIES, TOTAL_CAPITAL, get_enabled_strategies
from execution.ensemble import StrategyEnsemble, AllocationMethod
from execution.alpaca_connector import AlpacaConnector, ALPACA_AVAILABLE
from data.cached_data_manager import CachedDataManager

# Strategy imports
from strategies.pairs_trading import PairsTradingStrategy, PairsAnalyzer
from strategies.relative_volume_breakout import RelativeVolumeBreakout
from strategies.vix_regime_rotation import VIXRegimeRotationStrategy
from strategies.sector_rotation import SectorRotationStrategy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class PaperTradingRunner:
    """Coordinates strategy execution for paper trading."""
    
    def __init__(self):
        self.data_mgr = CachedDataManager()
        self.ensemble = StrategyEnsemble(
            total_capital=TOTAL_CAPITAL,
            allocation_method=AllocationMethod.FIXED
        )
        self.connector = None
        
        # Set allocations from config
        weights = {
            name: cfg['allocation_pct'] 
            for name, cfg in STRATEGIES.items() 
            if cfg['enabled']
        }
        self.ensemble.allocator.set_fixed_weights(weights)
        
        # Initialize strategies
        self.strategies = {}
        self._init_strategies()
    
    def _init_strategies(self):
        """Initialize enabled strategies."""
        enabled = get_enabled_strategies()
        logger.info(f"Enabled strategies: {enabled}")
        
        if 'pairs_trading' in enabled:
            self.strategies['pairs_trading'] = PairsTradingStrategy()
            self.ensemble.register_strategy('pairs_trading')
            
        if 'relative_volume_breakout' in enabled:
            self.strategies['relative_volume_breakout'] = RelativeVolumeBreakout()
            self.ensemble.register_strategy('relative_volume_breakout')
            
        if 'vix_regime_rotation' in enabled:
            self.strategies['vix_regime_rotation'] = VIXRegimeRotationStrategy()
            self.ensemble.register_strategy('vix_regime_rotation')
            
        if 'sector_rotation' in enabled:
            self.strategies['sector_rotation'] = SectorRotationStrategy()
            self.ensemble.register_strategy('sector_rotation')
    
    def connect_broker(self) -> bool:
        """Connect to Alpaca paper trading."""
        if not ALPACA_AVAILABLE:
            logger.error("alpaca-py not installed")
            return False
        
        try:
            from config import ALPACA_API_KEY, ALPACA_SECRET_KEY
            self.connector = AlpacaConnector(
                ALPACA_API_KEY, 
                ALPACA_SECRET_KEY, 
                paper=True
            )
            account = self.connector.get_account()
            logger.info(f"Connected to Alpaca - Equity: ${account.equity:,.2f}")
            
            # Sync existing positions to ensemble
            positions = self.connector.get_positions()
            for pos in positions:
                self.ensemble.update_position(pos.symbol, 'synced')
            if positions:
                logger.info(f"Synced {len(positions)} existing positions")
            
            return True
        except Exception as e:
            logger.error(f"Broker connection failed: {e}")
            return False
    
    def load_data(self):
        """Load market data."""
        if not self.data_mgr.cache:
            logger.info("Loading market data...")
            self.data_mgr.load_all()
        logger.info(f"Loaded {len(self.data_mgr.cache)} symbols")
    
    def generate_signals(self) -> dict:
        """Generate signals from all strategies."""
        self.load_data()
        
        all_signals = {}
        
        # Pairs trading
        if 'pairs_trading' in self.strategies:
            from strategies.base import Signal, SignalType
            strategy = self.strategies['pairs_trading']
            strategy.refresh_pairs(max_per_sector=2)
            raw_signals = strategy.scan_for_signals()
            
            # Convert dicts to Signal objects
            signals = []
            for s in raw_signals:
                pair = s['pair']
                # Get prices
                price_a = self.data_mgr.get_bars(pair.stock_a)['close'].iloc[-1] if self.data_mgr.get_bars(pair.stock_a) is not None else 100
                price_b = self.data_mgr.get_bars(pair.stock_b)['close'].iloc[-1] if self.data_mgr.get_bars(pair.stock_b) is not None else 100
                
                if s['direction'] == 'long_spread':
                    # Buy A, Sell B
                    signals.append(Signal(
                        timestamp=datetime.now(),
                        symbol=pair.stock_a,
                        strategy='pairs_trading',
                        signal_type=SignalType.BUY,
                        strength=min(0.9, abs(s['zscore']) / 3),
                        price=float(price_a),
                        reason=f"Pairs: {pair.stock_a}/{pair.stock_b} z={s['zscore']:.2f}"
                    ))
                else:
                    # Short spread: Buy B
                    signals.append(Signal(
                        timestamp=datetime.now(),
                        symbol=pair.stock_b,
                        strategy='pairs_trading',
                        signal_type=SignalType.BUY,
                        strength=min(0.9, abs(s['zscore']) / 3),
                        price=float(price_b),
                        reason=f"Pairs: {pair.stock_a}/{pair.stock_b} z={s['zscore']:.2f}"
                    ))
            all_signals['pairs_trading'] = signals
            logger.info(f"Pairs trading: {len(signals)} signals")
        
        # RV Breakout
        if 'relative_volume_breakout' in self.strategies:
            strategy = self.strategies['relative_volume_breakout']
            signals = strategy.generate_signals()
            all_signals['relative_volume_breakout'] = signals
            logger.info(f"RV breakout: {len(signals)} signals")
        
        # VIX regime (needs VIX data)
        if 'vix_regime_rotation' in self.strategies:
            # Load VIX
            from config import DIRS, get_vix_regime
            vix_path = DIRS['vix'] / 'vix.parquet'
            if vix_path.exists():
                import pandas as pd
                vix_df = pd.read_parquet(vix_path)
                current_vix = vix_df['close'].iloc[-1]
                vix_regime = get_vix_regime(current_vix)  # Convert to regime string
                
                strategy = self.strategies['vix_regime_rotation']
                data = self._prepare_data()
                signals = strategy.generate_signals(data, vix_regime=vix_regime)
                all_signals['vix_regime_rotation'] = signals
                logger.info(f"VIX regime: {len(signals)} signals (VIX={current_vix:.1f}, regime={vix_regime})")
        
        return all_signals
    
    def _prepare_data(self, n_symbols: int = 100) -> dict:
        """Prepare data dict for strategies."""
        import pandas as pd
        metadata = self.data_mgr.get_all_metadata()
        sorted_symbols = sorted(
            metadata.items(),
            key=lambda x: x[1].get('dollar_volume', 0),
            reverse=True
        )[:n_symbols]
        
        data = {}
        for symbol, _ in sorted_symbols:
            df = self.data_mgr.get_bars(symbol)
            if df is not None and len(df) >= 100:
                if 'timestamp' in df.columns:
                    df = df.copy()
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.set_index('timestamp')
                data[symbol] = df
        return data
    
    def run_once(self, execute: bool = False):
        """Run one signal generation cycle."""
        print("="*60)
        print(f"PAPER TRADING - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("="*60)
        
        # Generate signals
        all_signals = self.generate_signals()
        
        # Add to ensemble
        for strategy_name, signals in all_signals.items():
            if signals:
                self.ensemble.add_signals(strategy_name, signals)
        
        # Get portfolio decisions
        decisions = self.ensemble.get_portfolio_decisions()
        
        print(f"\n{'='*60}")
        print("PORTFOLIO DECISIONS")
        print("="*60)
        
        if not decisions:
            print("No actionable signals")
        else:
            for d in decisions:
                print(f"  {d.direction.upper():6s} {d.symbol:6s} | "
                      f"strength={d.strength:.2f} | ${d.capital_allocation:,.0f} | "
                      f"sources={d.sources}")
        
        # Execute if requested and broker connected
        if execute and self.connector and decisions:
            print(f"\n{'='*60}")
            print("EXECUTION")
            print("="*60)
            for d in decisions:
                # Skip if already have position
                if d.symbol in self.ensemble.current_positions:
                    print(f"  - SKIP {d.symbol}: already held")
                    continue
                    
                try:
                    if d.direction == 'long':
                        # Calculate shares from dollar amount
                        price = self.connector.get_latest_price(d.symbol) or 100
                        qty = int(d.capital_allocation / price)
                        if qty > 0:
                            order = self.connector.submit_market_order(
                                d.symbol, qty, 'buy'
                            )
                            if order:
                                self.ensemble.update_position(d.symbol, d.sources[0])
                            print(f"  ✓ BUY {qty} {d.symbol} @ ~${price:.2f}")
                    elif d.direction == 'short':
                        price = self.connector.get_latest_price(d.symbol) or 100
                        qty = int(d.capital_allocation / price)
                        if qty > 0:
                            order = self.connector.submit_market_order(
                                d.symbol, qty, 'sell'
                            )
                            if order:
                                self.ensemble.update_position(d.symbol, d.sources[0])
                            print(f"  ✓ SELL {qty} {d.symbol} @ ~${price:.2f}")
                except Exception as e:
                    print(f"  ✗ {d.symbol}: {e}")
        
        print(f"\n{self.ensemble.summary()}")
        return decisions


def main():
    parser = argparse.ArgumentParser(description='Paper Trading Runner')
    parser.add_argument('--execute', action='store_true', 
                        help='Actually execute orders (default: signals only)')
    parser.add_argument('--live', action='store_true',
                        help='Run continuously during market hours')
    args = parser.parse_args()
    
    runner = PaperTradingRunner()
    
    if args.execute or args.live:
        if not runner.connect_broker():
            sys.exit(1)
    
    if args.live:
        from execution.scheduler import MarketHours
        import time
        
        print("Running in live mode. Ctrl+C to stop.")
        while True:
            if MarketHours.is_market_open():
                runner.run_once(execute=args.execute)
                time.sleep(300)  # 5 min between checks
            else:
                seconds = MarketHours.time_to_open()
                print(f"Market closed. Opens in {seconds/3600:.1f} hours")
                time.sleep(min(seconds, 3600))
    else:
        runner.run_once(execute=args.execute)


if __name__ == "__main__":
    main()
