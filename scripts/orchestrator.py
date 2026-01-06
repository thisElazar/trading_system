#!/usr/bin/env python3
"""
Trading System Orchestrator
===========================
Master control for the entire trading system.

Commands:
    python scripts/orchestrator.py status       # Show system status
    python scripts/orchestrator.py run          # Run scheduler (blocking)
    python scripts/orchestrator.py scan         # Run all strategies once
    python scripts/orchestrator.py report       # Generate report
    python scripts/orchestrator.py test         # Test all components
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import STRATEGIES, DATA_ROOT, DATABASES
from execution.signal_tracker import SignalDatabase, ExecutionTracker
from execution.scheduler import StrategyScheduler, MarketHours, create_default_scheduler
from execution.alerts import create_alert_manager, AlertLevel
from data.cached_data_manager import CachedDataManager

logger = logging.getLogger(__name__)


class TradingOrchestrator:
    """
    Master controller for the trading system.
    
    Coordinates:
    - Data management
    - Strategy execution
    - Signal tracking
    - Alerts
    """
    
    def __init__(self, paper_mode: bool = True):
        self.paper_mode = paper_mode
        
        # Initialize components
        self.data_mgr = CachedDataManager()
        self.db = SignalDatabase()
        self.tracker = ExecutionTracker(self.db)
        self.scheduler = create_default_scheduler()
        self.alerts = create_alert_manager(console=True, file=True)
        
        logger.info("Orchestrator initialized")
    
    def status(self) -> dict:
        """Get comprehensive system status."""
        # Data status
        data_files = list((DATA_ROOT / "data" / "historical" / "daily").glob("*.parquet"))
        
        # Database status
        db_exists = {name: path.exists() for name, path in DATABASES.items()}
        
        # Positions
        open_positions = self.db.get_open_positions()
        pending_signals = self.db.get_pending_signals()
        
        # Strategies
        enabled = [name for name, cfg in STRATEGIES.items() if cfg.get('enabled')]
        
        return {
            'timestamp': datetime.now().isoformat(),
            'mode': 'paper' if self.paper_mode else 'live',
            'market_open': MarketHours.is_market_open(),
            'data': {
                'symbols_cached': len(data_files),
                'data_root': str(DATA_ROOT)
            },
            'databases': db_exists,
            'strategies': {
                'enabled': enabled,
                'total': len(STRATEGIES)
            },
            'positions': {
                'open': len(open_positions),
                'pending_signals': len(pending_signals)
            },
            'scheduler': self.scheduler.status()
        }
    
    def print_status(self):
        """Print formatted status."""
        s = self.status()
        
        print("=" * 60)
        print("TRADING SYSTEM STATUS")
        print("=" * 60)
        print(f"Time: {s['timestamp']}")
        print(f"Mode: {s['mode'].upper()}")
        print(f"Market: {'OPEN' if s['market_open'] else 'CLOSED'}")
        print()
        
        print("DATA:")
        print(f"  Symbols cached: {s['data']['symbols_cached']}")
        print(f"  Root: {s['data']['data_root']}")
        print()
        
        print("DATABASES:")
        for name, exists in s['databases'].items():
            status = "✓" if exists else "✗"
            print(f"  {status} {name}")
        print()
        
        print("STRATEGIES:")
        print(f"  Enabled: {', '.join(s['strategies']['enabled']) or 'None'}")
        print(f"  Total: {s['strategies']['total']}")
        print()
        
        print("POSITIONS:")
        print(f"  Open: {s['positions']['open']}")
        print(f"  Pending signals: {s['positions']['pending_signals']}")
        print()
        
        if s['positions']['open'] > 0:
            print("OPEN POSITIONS:")
            for pos in self.db.get_open_positions():
                print(f"  {pos.symbol} {pos.direction} @ {pos.entry_price:.2f} | P&L: {pos.unrealized_pnl:.2f}%")
    
    def scan_all(self):
        """Run all enabled strategies once."""
        print("=" * 60)
        print("SCANNING ALL STRATEGIES")
        print("=" * 60)
        
        for name, cfg in STRATEGIES.items():
            if not cfg.get('enabled'):
                continue
            
            print(f"\n>>> {name}")
            try:
                self.scheduler.run_now(name)
            except Exception as e:
                logger.error(f"{name} failed: {e}")
                self.alerts.error(f"{name} scan failed", e, name)
    
    def generate_report(self) -> str:
        """Generate comprehensive report."""
        lines = []
        
        lines.append("=" * 60)
        lines.append("TRADING SYSTEM REPORT")
        lines.append("=" * 60)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Performance by strategy
        lines.append("## Strategy Performance (Last 30 Days)")
        for name in STRATEGIES.keys():
            comp = self.tracker.compare_to_backtest(name)
            if comp['status'] == 'no_data':
                lines.append(f"  {name}: No trades")
            else:
                icon = {'ok': '✓', 'degraded': '⚠', 'critical': '✗'}[comp['status']]
                lines.append(f"  {icon} {name}:")
                lines.append(f"      Trades: {comp['trades']}")
                lines.append(f"      Win Rate: {comp['live_win_rate']:.1%} (backtest: {comp['backtest_win_rate']:.1%})")
                lines.append(f"      Avg P&L: {comp['live_avg_pnl']:.2f}%")
        
        lines.append("")
        
        # Closed positions
        closed = self.db.get_all_closed_positions(days=30)
        if len(closed) > 0:
            lines.append("## Recent Closed Positions")
            total_pnl = closed['realized_pnl'].sum()
            win_rate = (closed['realized_pnl'] > 0).mean()
            lines.append(f"  Total P&L: {total_pnl:.2f}%")
            lines.append(f"  Win Rate: {win_rate:.1%}")
            lines.append(f"  Trades: {len(closed)}")
        
        lines.append("")
        
        # Open positions
        open_pos = self.db.get_open_positions()
        if open_pos:
            lines.append("## Open Positions")
            for pos in open_pos:
                lines.append(f"  {pos.symbol} {pos.direction} @ {pos.entry_price:.2f} | "
                           f"P&L: {pos.unrealized_pnl:.2f}%")
        
        report = "\n".join(lines)
        
        # Save to file
        report_path = DATA_ROOT / "research" / "reports" / f"report_{datetime.now().strftime('%Y%m%d')}.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report)
        
        print(report)
        print(f"\nSaved to: {report_path}")
        
        return report
    
    def test_components(self):
        """Test all system components."""
        print("=" * 60)
        print("COMPONENT TESTS")
        print("=" * 60)
        
        tests = []
        
        # Test data manager
        print("\n1. Data Manager...")
        try:
            symbols = list((DATA_ROOT / "data" / "historical" / "daily").glob("*.parquet"))
            assert len(symbols) > 0, "No data files found"
            tests.append(("Data Manager", True, f"{len(symbols)} symbols"))
        except Exception as e:
            tests.append(("Data Manager", False, str(e)))
        
        # Test database
        print("2. Database...")
        try:
            self.db._init_tables()
            pending = self.db.get_pending_signals()
            tests.append(("Database", True, f"{len(pending)} pending signals"))
        except Exception as e:
            tests.append(("Database", False, str(e)))
        
        # Test strategies
        print("3. Strategies...")
        for name in ['gap_fill', 'pairs_trading', 'relative_volume_breakout']:
            try:
                if name == 'gap_fill':
                    from strategies.gap_fill import GapFillStrategy
                    s = GapFillStrategy()
                elif name == 'pairs_trading':
                    from strategies.pairs_trading import PairsTradingStrategy
                    s = PairsTradingStrategy()
                else:
                    from strategies.relative_volume_breakout import RelativeVolumeBreakout
                    s = RelativeVolumeBreakout()
                tests.append((f"Strategy: {name}", True, "Imported"))
            except Exception as e:
                tests.append((f"Strategy: {name}", False, str(e)))
        
        # Test alerts
        print("4. Alert System...")
        try:
            self.alerts.signal("TEST", "long", 100.0, "test")
            tests.append(("Alerts", True, "Working"))
        except Exception as e:
            tests.append(("Alerts", False, str(e)))
        
        # Test scheduler
        print("5. Scheduler...")
        try:
            status = self.scheduler.status()
            tests.append(("Scheduler", True, f"{status['pending_jobs']} jobs"))
        except Exception as e:
            tests.append(("Scheduler", False, str(e)))
        
        # Results
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        
        passed = 0
        for name, success, msg in tests:
            icon = "✓" if success else "✗"
            print(f"  {icon} {name}: {msg}")
            if success:
                passed += 1
        
        print(f"\nPassed: {passed}/{len(tests)}")
        
        return all(success for _, success, _ in tests)
    
    def run(self):
        """Start the scheduler in blocking mode."""
        print("=" * 60)
        print("STARTING TRADING SYSTEM")
        print("=" * 60)
        print(f"Mode: {'PAPER' if self.paper_mode else 'LIVE'}")
        print(f"Market: {'OPEN' if MarketHours.is_market_open() else 'CLOSED'}")
        print()
        print("Press Ctrl+C to stop")
        print("=" * 60)
        
        self.alerts.critical("System Started", f"Trading system started in {'paper' if self.paper_mode else 'live'} mode")
        
        try:
            self.scheduler.start(blocking=True)
        except KeyboardInterrupt:
            print("\nShutting down...")
            self.scheduler.stop()
            self.alerts.critical("System Stopped", "Trading system stopped by user")


def main():
    parser = argparse.ArgumentParser(description='Trading System Orchestrator')
    parser.add_argument('command', choices=['status', 'run', 'scan', 'report', 'test'],
                       help='Command to execute')
    parser.add_argument('--live', action='store_true', help='Live mode (default: paper)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Create orchestrator
    orch = TradingOrchestrator(paper_mode=not args.live)
    
    # Execute command
    if args.command == 'status':
        orch.print_status()
    elif args.command == 'run':
        orch.run()
    elif args.command == 'scan':
        orch.scan_all()
    elif args.command == 'report':
        orch.generate_report()
    elif args.command == 'test':
        success = orch.test_components()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
