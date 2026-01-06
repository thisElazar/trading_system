#!/usr/bin/env python3
"""Quick test of Alpaca paper trading connection."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ALPACA_API_KEY, ALPACA_SECRET_KEY

def test_connection():
    print("="*50)
    print("ALPACA CONNECTION TEST")
    print("="*50)
    
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("❌ API keys not set in .env")
        return False
    
    print(f"API Key: {ALPACA_API_KEY[:8]}...")
    
    try:
        from alpaca.trading.client import TradingClient
        
        client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
        account = client.get_account()
        
        print(f"\n✓ Connected to Alpaca Paper Trading")
        print(f"\nAccount Status:")
        print(f"  Equity:       ${float(account.equity):,.2f}")
        print(f"  Cash:         ${float(account.cash):,.2f}")
        print(f"  Buying Power: ${float(account.buying_power):,.2f}")
        print(f"  Day Trades:   {account.daytrade_count}/3")
        print(f"  PDT:          {'Yes' if account.pattern_day_trader else 'No'}")
        print(f"  Blocked:      {'Yes' if account.trading_blocked else 'No'}")
        
        # Get positions
        positions = client.get_all_positions()
        if positions:
            print(f"\nOpen Positions ({len(positions)}):")
            for p in positions:
                pnl = float(p.unrealized_pl)
                pnl_pct = float(p.unrealized_plpc) * 100
                print(f"  {p.symbol}: {p.qty} @ ${float(p.avg_entry_price):.2f} "
                      f"(P&L: ${pnl:+,.2f} / {pnl_pct:+.1f}%)")
        else:
            print(f"\nNo open positions")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    test_connection()
