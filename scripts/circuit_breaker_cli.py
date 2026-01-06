#!/usr/bin/env python3
"""
Circuit Breaker CLI
===================
Manual control of circuit breakers and kill switches.

Usage:
    python scripts/circuit_breaker_cli.py status           # Show current status
    python scripts/circuit_breaker_cli.py halt             # Activate HALT
    python scripts/circuit_breaker_cli.py close-all        # Emergency close all
    python scripts/circuit_breaker_cli.py graceful         # Graceful shutdown
    python scripts/circuit_breaker_cli.py disable gap_fill # Disable strategy
    python scripts/circuit_breaker_cli.py enable gap_fill  # Re-enable strategy
    python scripts/circuit_breaker_cli.py clear HALT       # Clear kill switch
    python scripts/circuit_breaker_cli.py clear-breaker daily_loss  # Clear breaker
    python scripts/circuit_breaker_cli.py history          # Show trigger history
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DIRS, CIRCUIT_BREAKER
from execution.circuit_breaker import (
    CircuitBreakerManager,
    CircuitBreakerConfig,
    CircuitBreakerDB
)


def print_header(title: str):
    """Print a formatted header."""
    print("=" * 60)
    print(f" {title}")
    print("=" * 60)


def cmd_status(args):
    """Show current circuit breaker status."""
    config = CircuitBreakerConfig.from_dict(CIRCUIT_BREAKER)
    manager = CircuitBreakerManager(config)

    status = manager.get_status()

    print_header("CIRCUIT BREAKER STATUS")

    # Trading status
    if status['trading_allowed']:
        print(f"\n  Trading:    ALLOWED")
    else:
        print(f"\n  Trading:    ** HALTED **")

    print(f"  Position Multiplier: {status['position_multiplier']:.0%}")

    # Active breakers
    print(f"\n  Active Breakers: {len(status['active_breakers'])}")
    if status['active_breakers']:
        for b in status['active_breakers']:
            print(f"\n    [{b['type'].upper()}]")
            print(f"      Reason:  {b['reason']}")
            print(f"      Action:  {b['action']}")
            print(f"      Target:  {b['target']}")
            if b['expires_at']:
                print(f"      Expires: {b['expires_at']}")
            else:
                print(f"      Expires: Manual reset required")
    else:
        print("    (none)")

    # File kill switches
    print(f"\n  File Kill Switches: {len(status['file_kill_switches'])}")
    if status['file_kill_switches']:
        for f in status['file_kill_switches']:
            print(f"    - {f}")
    else:
        print("    (none)")

    # Configuration summary
    print(f"\n  Configuration:")
    print(f"    Daily Loss Halt:     {CIRCUIT_BREAKER['daily_loss_pct']:.0%}")
    print(f"    Drawdown Reduce:     {CIRCUIT_BREAKER['drawdown_pct']:.0%}")
    print(f"    Rapid Loss:          {CIRCUIT_BREAKER['rapid_loss_pct']:.0%} in {CIRCUIT_BREAKER['rapid_loss_window_min']} min")
    print(f"    Consecutive Losses:  {CIRCUIT_BREAKER['max_consecutive_losses']}")
    print(f"    Strategy Loss:       {CIRCUIT_BREAKER['strategy_loss_pct']:.0%}")

    print()


def cmd_halt(args):
    """Activate HALT kill switch."""
    kill_switch_dir = DIRS.get("killswitch", Path("killswitch"))
    kill_switch_dir.mkdir(parents=True, exist_ok=True)

    halt_file = kill_switch_dir / "HALT"
    halt_file.touch()

    print_header("HALT ACTIVATED")
    print(f"\n  File created: {halt_file}")
    print("\n  All new orders will be blocked.")
    print("  Existing positions are kept.")
    print(f"\n  To clear: python scripts/circuit_breaker_cli.py clear HALT")
    print()


def cmd_close_all(args):
    """Activate CLOSE_ALL kill switch (requires confirmation)."""
    print_header("EMERGENCY CLOSE ALL")
    print("\n  WARNING: This will CLOSE ALL POSITIONS immediately!")
    print("  This action cannot be undone.")

    confirm = input("\n  Type 'CONFIRM' to proceed: ")

    if confirm.strip() != 'CONFIRM':
        print("\n  Aborted.")
        return

    kill_switch_dir = DIRS.get("killswitch", Path("killswitch"))
    kill_switch_dir.mkdir(parents=True, exist_ok=True)

    close_file = kill_switch_dir / "CLOSE_ALL"
    close_file.write_text("CONFIRM")

    print(f"\n  CLOSE_ALL activated: {close_file}")
    print("  All positions will be liquidated on next check.")
    print()


def cmd_graceful(args):
    """Activate GRACEFUL shutdown."""
    kill_switch_dir = DIRS.get("killswitch", Path("killswitch"))
    kill_switch_dir.mkdir(parents=True, exist_ok=True)

    graceful_file = kill_switch_dir / "GRACEFUL"
    graceful_file.touch()

    print_header("GRACEFUL SHUTDOWN ACTIVATED")
    print(f"\n  File created: {graceful_file}")
    print("\n  No new signals will be generated.")
    print("  Pending orders will complete.")
    print(f"\n  To clear: python scripts/circuit_breaker_cli.py clear GRACEFUL")
    print()


def cmd_disable(args):
    """Disable a specific strategy."""
    strategy = args.strategy

    kill_switch_dir = DIRS.get("killswitch", Path("killswitch"))
    kill_switch_dir.mkdir(parents=True, exist_ok=True)

    strategy_file = kill_switch_dir / f"STRATEGY_{strategy}"
    strategy_file.touch()

    print_header(f"STRATEGY DISABLED: {strategy}")
    print(f"\n  File created: {strategy_file}")
    print(f"\n  Strategy '{strategy}' will not generate signals.")
    print(f"\n  To re-enable: python scripts/circuit_breaker_cli.py enable {strategy}")
    print()


def cmd_enable(args):
    """Re-enable a disabled strategy."""
    strategy = args.strategy

    kill_switch_dir = DIRS.get("killswitch", Path("killswitch"))
    strategy_file = kill_switch_dir / f"STRATEGY_{strategy}"

    if strategy_file.exists():
        strategy_file.unlink()
        print_header(f"STRATEGY ENABLED: {strategy}")
        print(f"\n  File removed: {strategy_file}")
        print(f"\n  Strategy '{strategy}' is now active.")
    else:
        print(f"\n  Strategy '{strategy}' was not disabled.")

    print()


def cmd_clear(args):
    """Clear a kill switch file."""
    switch = args.switch

    kill_switch_dir = DIRS.get("killswitch", Path("killswitch"))

    if switch.startswith("STRATEGY_"):
        file_path = kill_switch_dir / switch
    else:
        file_path = kill_switch_dir / switch

    if file_path.exists():
        file_path.unlink()
        print_header(f"CLEARED: {switch}")
        print(f"\n  File removed: {file_path}")
    else:
        print(f"\n  No active switch: {switch}")

    print()


def cmd_clear_breaker(args):
    """Clear a circuit breaker from database."""
    breaker = args.breaker
    target = args.target

    db = CircuitBreakerDB()
    success = db.clear_breaker(breaker, target, cleared_by='cli')

    if success:
        print_header(f"BREAKER CLEARED: {breaker}")
        print(f"\n  Breaker: {breaker}")
        print(f"  Target:  {target}")
        print("\n  Circuit breaker has been cleared.")
    else:
        print(f"\n  No active breaker: {breaker} ({target})")

    print()


def cmd_history(args):
    """Show recent kill switch and circuit breaker history."""
    db = CircuitBreakerDB()
    history = db.get_kill_switch_log(limit=args.limit)

    print_header("KILL SWITCH / CIRCUIT BREAKER HISTORY")

    if not history:
        print("\n  No recent history.")
    else:
        for entry in history:
            print(f"\n  [{entry['timestamp']}]")
            print(f"    Type:       {entry['switch_type']}")
            print(f"    Triggered:  {entry['triggered_by']}")
            print(f"    Positions:  {entry['positions_affected']}")
            print(f"    Orders:     {entry['orders_cancelled']}")

    # Also show active breakers
    active = db.get_active_breakers()
    if active:
        print(f"\n  Currently Active Breakers: {len(active)}")
        for state in active:
            print(f"    - {state.breaker_type}: {state.reason}")

    print()


def cmd_test(args):
    """Test circuit breaker with simulated context."""
    print_header("CIRCUIT BREAKER TEST")

    config = CircuitBreakerConfig.from_dict(CIRCUIT_BREAKER)
    manager = CircuitBreakerManager(config)

    # Simulate context
    context = {
        'current_equity': 95000,
        'start_of_day_equity': 97000,  # Down 2.06%
        'peak_equity': 100000,          # Down 5% from peak
    }

    print(f"\n  Simulated Context:")
    print(f"    Current Equity:     ${context['current_equity']:,}")
    print(f"    Start of Day:       ${context['start_of_day_equity']:,}")
    print(f"    Peak Equity:        ${context['peak_equity']:,}")

    daily_loss = (context['start_of_day_equity'] - context['current_equity']) / context['start_of_day_equity']
    drawdown = (context['peak_equity'] - context['current_equity']) / context['peak_equity']

    print(f"\n  Calculated Metrics:")
    print(f"    Daily Loss:         {daily_loss:.2%}")
    print(f"    Drawdown from Peak: {drawdown:.2%}")

    print(f"\n  Would Trigger:")
    if daily_loss >= config.daily_loss_pct:
        print(f"    - Daily Loss Breaker (>= {config.daily_loss_pct:.0%})")
    if drawdown >= config.drawdown_pct:
        print(f"    - Drawdown Breaker (>= {config.drawdown_pct:.0%})")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Circuit Breaker Control CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s status                    # Show current status
  %(prog)s halt                      # Stop all new orders
  %(prog)s close-all                 # Emergency liquidation
  %(prog)s disable gap_fill          # Disable a strategy
  %(prog)s clear HALT                # Clear the HALT switch
  %(prog)s clear-breaker daily_loss  # Clear a circuit breaker
  %(prog)s history                   # Show recent events
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Status
    subparsers.add_parser('status', help='Show current circuit breaker status')

    # Kill switches
    subparsers.add_parser('halt', help='Activate HALT (stop all new orders)')
    subparsers.add_parser('close-all', help='Emergency close all positions')
    subparsers.add_parser('graceful', help='Graceful shutdown')

    # Strategy control
    disable_parser = subparsers.add_parser('disable', help='Disable a strategy')
    disable_parser.add_argument('strategy', help='Strategy name (e.g., gap_fill)')

    enable_parser = subparsers.add_parser('enable', help='Re-enable a strategy')
    enable_parser.add_argument('strategy', help='Strategy name (e.g., gap_fill)')

    # Clear commands
    clear_parser = subparsers.add_parser('clear', help='Clear a kill switch')
    clear_parser.add_argument('switch', help='Switch type (HALT, CLOSE_ALL, GRACEFUL, or STRATEGY_*)')

    clear_breaker = subparsers.add_parser('clear-breaker', help='Clear a circuit breaker')
    clear_breaker.add_argument('breaker', help='Breaker type (daily_loss, drawdown, rapid_loss, etc)')
    clear_breaker.add_argument('--target', default='all', help='Target (all or strategy name)')

    # History
    history_parser = subparsers.add_parser('history', help='Show trigger history')
    history_parser.add_argument('--limit', type=int, default=20, help='Number of entries to show')

    # Test
    subparsers.add_parser('test', help='Test circuit breaker with simulated data')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Command dispatch
    commands = {
        'status': cmd_status,
        'halt': cmd_halt,
        'close-all': cmd_close_all,
        'graceful': cmd_graceful,
        'disable': cmd_disable,
        'enable': cmd_enable,
        'clear': cmd_clear,
        'clear-breaker': cmd_clear_breaker,
        'history': cmd_history,
        'test': cmd_test,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
