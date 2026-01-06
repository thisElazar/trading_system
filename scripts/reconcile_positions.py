#!/usr/bin/env python3
"""
Position Reconciliation Script
Cleans up phantom and duplicate positions in the database to match broker reality.
"""

import sqlite3
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from execution.alpaca_connector import AlpacaConnector


def reconcile_positions():
    """Reconcile database positions with broker reality."""

    db_path = Path(__file__).parent.parent / "db" / "trades.db"

    print("=" * 60)
    print("POSITION RECONCILIATION")
    print("=" * 60)

    # Get broker positions
    print("\n1. Fetching broker positions...")
    connector = AlpacaConnector()
    broker_positions = connector.get_positions()
    broker_symbols = {p.symbol: p for p in broker_positions}

    print(f"   Broker has {len(broker_symbols)} positions:")
    for symbol, pos in broker_symbols.items():
        print(f"   - {symbol}: {pos.qty} shares @ ${pos.avg_entry_price:.2f}")

    # Get database positions
    print("\n2. Fetching database positions...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, strategy_name, symbol, quantity, entry_price, opened_at
        FROM positions
        WHERE status = 'open'
        ORDER BY symbol, opened_at
    """)
    db_positions = cursor.fetchall()

    print(f"   Database has {len(db_positions)} open positions")

    # Analyze positions
    print("\n3. Analyzing positions...")

    # Group by symbol
    positions_by_symbol = {}
    for pos in db_positions:
        pos_id, strategy, symbol, qty, price, opened = pos
        if symbol not in positions_by_symbol:
            positions_by_symbol[symbol] = []
        positions_by_symbol[symbol].append({
            'id': pos_id,
            'strategy': strategy,
            'qty': qty,
            'price': price,
            'opened': opened
        })

    to_close = []  # Positions to mark as closed (phantom)
    to_keep = []   # Positions to keep

    for symbol, positions in positions_by_symbol.items():
        if symbol not in broker_symbols:
            # All positions for this symbol are phantom
            print(f"   PHANTOM: {symbol} - {len(positions)} DB entries, 0 at broker")
            for p in positions:
                to_close.append((p['id'], p['strategy'], symbol, p['qty'], 'phantom_cleanup'))
        else:
            broker_qty = float(broker_symbols[symbol].qty)
            broker_price = float(broker_symbols[symbol].avg_entry_price)

            # Find best match or consolidate
            if len(positions) == 1:
                # Single position, keep it but update qty if needed
                p = positions[0]
                if p['qty'] != broker_qty:
                    print(f"   UPDATE: {symbol} - DB qty {p['qty']} -> broker qty {broker_qty}")
                to_keep.append((p['id'], symbol, broker_qty, broker_price))
            else:
                # Multiple positions - keep the most recent one that matches qty, close others
                print(f"   DUPLICATE: {symbol} - {len(positions)} DB entries, keeping best match")

                # Find best match (prefer one with matching qty, else most recent)
                best_match = None
                for p in positions:
                    if p['qty'] == broker_qty:
                        best_match = p
                        break

                if not best_match:
                    # Take most recent
                    best_match = positions[-1]

                for p in positions:
                    if p['id'] == best_match['id']:
                        to_keep.append((p['id'], symbol, broker_qty, broker_price))
                    else:
                        to_close.append((p['id'], p['strategy'], symbol, p['qty'], 'duplicate_cleanup'))

    # Summary
    print("\n4. Reconciliation plan:")
    print(f"   Positions to CLOSE (phantom/duplicate): {len(to_close)}")
    print(f"   Positions to KEEP/UPDATE: {len(to_keep)}")

    if not to_close and not to_keep:
        print("\n   Nothing to do - database is clean!")
        conn.close()
        return

    # Execute cleanup
    print("\n5. Executing cleanup...")

    now = datetime.now().isoformat()

    # Close phantom/duplicate positions
    for pos_id, strategy, symbol, qty, reason in to_close:
        cursor.execute("""
            UPDATE positions
            SET status = 'closed',
                closed_at = ?,
                exit_reason = ?,
                exit_price = entry_price,
                realized_pnl = 0
            WHERE id = ?
        """, (now, reason, pos_id))
        print(f"   CLOSED: ID {pos_id} - {strategy}/{symbol} x{qty} ({reason})")

    # Update kept positions with correct qty
    for pos_id, symbol, correct_qty, correct_price in to_keep:
        cursor.execute("""
            UPDATE positions
            SET quantity = ?,
                entry_price = ?,
                current_price = ?
            WHERE id = ?
        """, (correct_qty, correct_price, correct_price, pos_id))
        print(f"   UPDATED: ID {pos_id} - {symbol} qty={correct_qty}")

    conn.commit()
    conn.close()

    print("\n" + "=" * 60)
    print("RECONCILIATION COMPLETE")
    print(f"Closed {len(to_close)} phantom/duplicate positions")
    print(f"Updated {len(to_keep)} valid positions")
    print("=" * 60)


if __name__ == "__main__":
    reconcile_positions()
