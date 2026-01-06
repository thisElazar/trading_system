#!/usr/bin/env python3
"""
Position Monitor
================
Checks open positions against stops and targets.
Run periodically during market hours.

Usage:
    python3 scripts/monitor_positions.py
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from execution.alpaca_connector import AlpacaConnector, ALPACA_AVAILABLE
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# Simple stop/target tracking (in production, persist to DB)
POSITION_STOPS = {
    # VIX regime rotation - defensive stops (3 ATR ~8%)
    'XLK': {'stop_pct': -0.08, 'target_pct': 0.12},
    'XLF': {'stop_pct': -0.08, 'target_pct': 0.12},
    'XLY': {'stop_pct': -0.08, 'target_pct': 0.12},
    'IWM': {'stop_pct': -0.08, 'target_pct': 0.12},
    'QQQ': {'stop_pct': -0.08, 'target_pct': 0.12},
    # Pairs trading - tighter stops
    'MS':  {'stop_pct': -0.05, 'target_pct': 0.08},
    'DHR': {'stop_pct': -0.05, 'target_pct': 0.08},
    'XEL': {'stop_pct': -0.05, 'target_pct': 0.08},
}


def monitor_positions():
    """Check all positions against stops and targets."""
    print("="*60)
    print(f"POSITION MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*60)
    
    if not ALPACA_AVAILABLE:
        print("Alpaca not available")
        return
    
    connector = AlpacaConnector(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
    
    # Get account
    account = connector.get_account()
    print(f"\nAccount: ${account.equity:,.2f} (${account.cash:,.2f} cash)")
    
    # Get positions
    positions = connector.get_positions()
    
    if not positions:
        print("\nNo open positions")
        return
    
    print(f"\n{'Symbol':<8} {'Qty':>6} {'Entry':>10} {'Current':>10} {'P&L':>10} {'P&L%':>8} {'Status':<12}")
    print("-"*70)
    
    alerts = []
    total_pnl = 0
    
    for pos in positions:
        pnl_pct = pos.unrealized_pnl_pct / 100  # Convert to decimal
        total_pnl += pos.unrealized_pnl
        
        # Check stops/targets
        limits = POSITION_STOPS.get(pos.symbol, {'stop_pct': -0.10, 'target_pct': 0.15})
        
        if pnl_pct <= limits['stop_pct']:
            status = "⚠️  STOP HIT"
            alerts.append((pos.symbol, 'STOP', pnl_pct))
        elif pnl_pct >= limits['target_pct']:
            status = "🎯 TARGET"
            alerts.append((pos.symbol, 'TARGET', pnl_pct))
        elif pnl_pct < 0:
            status = "📉 losing"
        else:
            status = "📈 profit"
        
        print(f"{pos.symbol:<8} {int(pos.qty):>6} ${pos.avg_entry_price:>9.2f} "
              f"${pos.current_price:>9.2f} ${pos.unrealized_pnl:>+9.2f} "
              f"{pos.unrealized_pnl_pct:>+7.1f}% {status}")
    
    print("-"*70)
    print(f"{'TOTAL':<8} {'':<6} {'':<10} {'':<10} ${total_pnl:>+9.2f}")
    
    # Alerts
    if alerts:
        print(f"\n{'='*60}")
        print("ALERTS - Action Required!")
        print("="*60)
        for symbol, alert_type, pnl in alerts:
            if alert_type == 'STOP':
                print(f"  🔴 {symbol}: Stop loss triggered ({pnl:.1%}) - CLOSE POSITION")
            else:
                print(f"  🟢 {symbol}: Target reached ({pnl:.1%}) - Consider taking profits")
    
    return positions, alerts


def close_stopped_positions(auto_close: bool = False):
    """Close positions that hit stops."""
    positions, alerts = monitor_positions()
    
    if not alerts:
        return
    
    stops = [(s, t, p) for s, t, p in alerts if t == 'STOP']
    
    if not stops:
        print("\nNo stop-loss alerts")
        return
    
    if not auto_close:
        print(f"\n{len(stops)} positions hit stop loss. Run with --auto-close to close them.")
        return
    
    connector = AlpacaConnector(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
    
    print("\nClosing stopped positions...")
    for symbol, _, pnl in stops:
        try:
            connector.close_position(symbol)
            print(f"  ✓ Closed {symbol} at {pnl:.1%} loss")
        except Exception as e:
            print(f"  ✗ Failed to close {symbol}: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--auto-close', action='store_true', help='Auto-close stopped positions')
    args = parser.parse_args()
    
    if args.auto_close:
        close_stopped_positions(auto_close=True)
    else:
        monitor_positions()
