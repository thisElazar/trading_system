#!/usr/bin/env python3
"""
Gap-Fill Morning Scanner
========================
Run at 9:31 AM ET to detect and execute gap-fill trades.

Usage:
    python scripts/gap_fill_scanner.py           # Scan only
    python scripts/gap_fill_scanner.py --execute # Scan and execute
    python scripts/gap_fill_scanner.py --backtest # Quick backtest
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.gap_fill import GapFillStrategy
from data.fetchers.intraday_bars import IntradayDataManager
from data.fetchers.vix import get_current_vix
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, get_vix_regime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def check_market_hours() -> bool:
    """Check if we're in the gap-fill trading window (9:30-9:45 AM ET)."""
    now = datetime.now()
    market_open = now.replace(hour=9, minute=30, second=0)
    window_end = now.replace(hour=9, minute=45, second=0)
    return market_open <= now <= window_end


def ensure_data_fresh(manager: IntradayDataManager) -> bool:
    """Ensure we have today's opening data."""
    today = datetime.now()
    
    # Check if market is open (weekday)
    if today.weekday() >= 5:
        logger.warning("Market closed (weekend)")
        return False
    
    # Try to fetch fresh data
    for symbol in manager.GAP_FILL_UNIVERSE[:2]:  # SPY, QQQ
        gap = manager.calculate_gap(symbol, today)
        if gap is None:
            logger.warning(f"Missing data for {symbol} - fetching...")
            df = manager.fetch_day(symbol, today)
            if df is not None:
                manager._get_file_path(symbol, today).parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(manager._get_file_path(symbol, today))
    
    return True


def scan(execute: bool = False, paper: bool = True):
    """
    Main scanning function.
    
    Args:
        execute: If True, submit orders for valid signals
        paper: Use paper trading (always True for safety)
    """
    logger.info("=" * 60)
    logger.info("GAP-FILL SCANNER")
    logger.info("=" * 60)
    
    # Initialize
    strategy = GapFillStrategy()
    manager = IntradayDataManager()
    
    # Get VIX
    try:
        vix = get_current_vix()
        regime = get_vix_regime(vix)
        logger.info(f"VIX: {vix:.1f} ({regime} regime)")
    except Exception as e:
        vix = None
        logger.warning(f"Could not fetch VIX ({e}) - proceeding without regime filter")
    
    # Ensure data
    ensure_data_fresh(manager)
    
    # Scan for gaps
    today = datetime.now()
    opportunities = strategy.scan_for_gaps(today)
    
    if not opportunities:
        logger.info("No tradeable gaps found")
        return
    
    # Display opportunities
    print("\n" + "=" * 60)
    print("GAP OPPORTUNITIES")
    print("=" * 60)
    
    for opp in opportunities:
        conf = "★★★" if opp['outside_3day_range'] else "★★"
        print(f"\n{opp['symbol']} | {opp['direction'].upper()}")
        print(f"  Gap: {opp['gap_percent']:+.2f}% (${opp['gap_dollars']:+.2f})")
        print(f"  Prev Close: ${opp['prev_close']:.2f}")
        print(f"  Open: ${opp['open']:.2f}")
        print(f"  Target: ${opp['target']:.2f}")
        print(f"  Confidence: {conf}")
    
    # Generate signals
    signals = strategy.generate_signals(today, vix_level=vix, long_only=True)
    
    if not signals:
        logger.info("No signals generated (may be filtered by VIX or direction)")
        return
    
    print("\n" + "=" * 60)
    print("SIGNALS")
    print("=" * 60)
    
    for sig in signals:
        print(f"\n{sig.signal_type.name} {sig.symbol}")
        print(f"  Entry: ${sig.price:.2f}")
        print(f"  Stop: ${sig.stop_loss:.2f}" if sig.stop_loss else "  Stop: N/A")
        print(f"  Target: ${sig.target_price:.2f}" if sig.target_price else "  Target: Time-based exit")
        print(f"  Hold Days: {sig.metadata.get('hold_days', 'N/A')}")
        print(f"  Strength: {sig.strength:.0%}")
    
    # Execute if requested
    if execute:
        logger.info("\n" + "=" * 60)
        logger.info("EXECUTION MODE")
        logger.info("=" * 60)
        
        if not check_market_hours():
            logger.warning("Outside trading window (9:30-9:45 AM ET)")
            logger.warning("Skipping execution - run during market hours")
            return
        
        if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
            logger.error("Alpaca API keys not configured")
            return
        
        # Import execution module
        from execution.order_executor import OrderExecutor
        
        executor = OrderExecutor(paper=paper)
        
        for sig in signals:
            logger.info(f"Submitting {sig.signal_type.name} order for {sig.symbol}...")
            
            try:
                # Calculate position size (gap-fill uses fixed allocation)
                from config import STRATEGIES, TOTAL_CAPITAL
                allocation = STRATEGIES['gap_fill']['allocation_pct']
                position_value = TOTAL_CAPITAL * allocation
                shares = int(position_value / sig.price)
                
                if shares < 1:
                    logger.warning(f"Position too small for {sig.symbol}")
                    continue
                
                # Submit order with bracket (stop + time exit)
                order = executor.submit_bracket_order(
                    symbol=sig.symbol,
                    qty=shares,
                    side='buy' if sig.signal_type.name == 'BUY' else 'sell',
                    stop_loss=sig.stop_loss,
                    take_profit=sig.target_price
                )
                
                logger.info(f"Order submitted: {order}")
                
            except Exception as e:
                logger.error(f"Failed to execute {sig.symbol}: {e}")
    
    else:
        print("\n" + "-" * 60)
        print("SCAN ONLY - Use --execute to submit orders")
        print("-" * 60)


def backtest_recent():
    """Quick backtest on available data."""
    strategy = GapFillStrategy()
    status = strategy.get_strategy_status()
    
    print("Data Status:")
    for symbol, info in status['data_status'].items():
        print(f"  {symbol}: {info['days_available']} days")
    
    if all(s['days_available'] == 0 for s in status['data_status'].values()):
        print("\nNo data available. Downloading...")
        manager = IntradayDataManager()
        manager.download_recent(days=10)
    
    # Run backtest
    from strategies.gap_fill import run_gap_fill_backtest
    run_gap_fill_backtest()


def main():
    parser = argparse.ArgumentParser(description='Gap-Fill Morning Scanner')
    parser.add_argument('--execute', action='store_true', 
                       help='Execute trades (paper trading)')
    parser.add_argument('--backtest', action='store_true',
                       help='Run backtest instead of scan')
    parser.add_argument('--download', action='store_true',
                       help='Download fresh intraday data')
    
    args = parser.parse_args()
    
    if args.download:
        from data.fetchers.intraday_bars import download_gap_fill_data
        download_gap_fill_data()
    elif args.backtest:
        backtest_recent()
    else:
        scan(execute=args.execute)


if __name__ == "__main__":
    main()
