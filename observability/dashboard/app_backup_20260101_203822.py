#!/usr/bin/env python3
"""
Trading System Dashboard
========================
Real-time monitoring dashboard using Dash + Plotly.

Usage:
    python observability/dashboard/app.py

Then open http://localhost:5000 in your browser.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import sqlite3

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, html, dcc, dash_table, callback, Output, Input
import dash_bootstrap_components as dbc

# Import trading system components
from execution.alpaca_connector import AlpacaConnector
from execution.signal_tracker import SignalDatabase
from execution.alerts import AlertManager, get_alerts, AlertLevel
from execution.scheduler import MarketHours
from config import STRATEGIES, DATABASES
from execution.circuit_breaker import CircuitBreakerManager, CircuitBreakerDB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Dash app with Bootstrap theme
app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
    ],
    title="Trading System Dashboard"
)

# Global state
_broker: Optional[AlpacaConnector] = None
_signal_db: Optional[SignalDatabase] = None


def get_broker() -> AlpacaConnector:
    """Get or create broker connection."""
    global _broker
    if _broker is None:
        _broker = AlpacaConnector(paper=True)
    return _broker


def get_signal_db() -> SignalDatabase:
    """Get or create signal database connection."""
    global _signal_db
    if _signal_db is None:
        _signal_db = SignalDatabase()
    return _signal_db


def get_account_data() -> Dict[str, Any]:
    """Fetch current account data from broker."""
    try:
        broker = get_broker()
        account = broker.get_account()
        if account:
            return {
                "equity": account.equity,
                "cash": account.cash,
                "buying_power": account.buying_power,
                "portfolio_value": account.portfolio_value,
                "day_trade_count": account.day_trade_count,
                "pattern_day_trader": account.pattern_day_trader,
                "trading_blocked": account.trading_blocked,
            }
    except Exception as e:
        logger.error(f"Error fetching account: {e}")
    return {}


def get_positions_data() -> List[Dict[str, Any]]:
    """Fetch current positions from broker."""
    try:
        broker = get_broker()
        positions = broker.get_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": p.qty,
                "side": p.side,
                "entry_price": p.avg_entry_price,
                "current_price": p.current_price,
                "market_value": p.market_value,
                "unrealized_pnl": p.unrealized_pnl,
                "pnl_pct": p.unrealized_pnl_pct if p.unrealized_pnl_pct else 0,
            }
            for p in positions
        ]
    except Exception as e:
        logger.error(f"Error fetching positions: {e}")
    return []


def get_equity_history() -> pd.DataFrame:
    """Fetch equity history from performance database."""
    try:
        db_path = PROJECT_ROOT / "db" / "performance.db"
        if not db_path.exists():
            return pd.DataFrame()

        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(
            "SELECT date, equity, daily_pnl, drawdown_pct FROM portfolio_daily ORDER BY date",
            conn
        )
        conn.close()

        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        logger.error(f"Error fetching equity history: {e}")
    return pd.DataFrame()


def get_strategy_stats() -> List[Dict[str, Any]]:
    """Fetch strategy performance stats."""
    try:
        db_path = PROJECT_ROOT / "db" / "performance.db"
        if not db_path.exists():
            return []

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if strategy_stats table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='strategy_stats'"
        )
        if not cursor.fetchone():
            conn.close()
            return []

        df = pd.read_sql_query(
            """SELECT strategy, total_trades, winning_trades,
                      win_rate, total_pnl, sharpe_ratio, max_drawdown_pct,
                      is_enabled, last_trade_date
               FROM strategy_stats
               ORDER BY total_pnl DESC""",
            conn
        )
        conn.close()

        return df.to_dict('records') if not df.empty else []
    except Exception as e:
        logger.error(f"Error fetching strategy stats: {e}")
    return []


def get_recent_alerts(count: int = 20) -> List[Dict[str, Any]]:
    """Fetch recent alerts."""
    try:
        alerts = get_alerts()
        recent = alerts.get_recent(count=count)
        return [
            {
                "timestamp": a.timestamp[:19],  # Trim to seconds
                "type": a.alert_type.value,
                "level": a.level.value,
                "title": a.title,
                "message": a.message[:100] + "..." if len(a.message) > 100 else a.message,
            }
            for a in recent
        ]
    except Exception as e:
        logger.error(f"Error fetching alerts: {e}")
    return []


def get_research_overview() -> Dict[str, Any]:
    """Fetch research/GA overview statistics."""
    try:
        db_path = PROJECT_ROOT / "db" / "research.db"
        if not db_path.exists():
            return {}

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # GA runs stats
        cursor.execute("SELECT COUNT(*) FROM ga_runs")
        total_ga_runs = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM ga_runs WHERE status='running'")
        running_ga_runs = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM ga_runs WHERE status='completed'")
        completed_ga_runs = cursor.fetchone()[0]

        # Backtest stats
        cursor.execute("SELECT COUNT(*) FROM backtests")
        total_backtests = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(sharpe_ratio) FROM backtests WHERE sharpe_ratio IS NOT NULL")
        avg_sharpe = cursor.fetchone()[0] or 0

        cursor.execute("SELECT MAX(sharpe_ratio) FROM backtests WHERE sharpe_ratio IS NOT NULL")
        best_sharpe = cursor.fetchone()[0] or 0

        # Best GA fitness (live from ga_populations)
        cursor.execute("SELECT MAX(best_fitness) FROM ga_populations WHERE best_fitness > 0")
        best_ga_fitness = cursor.fetchone()[0] or 0

        # Discovered strategies
        cursor.execute("SELECT COUNT(*) FROM discovered_strategies")
        discovered_count = cursor.fetchone()[0]

        # Latest GA run time
        cursor.execute("SELECT start_time FROM ga_runs ORDER BY start_time DESC LIMIT 1")
        row = cursor.fetchone()
        last_ga_run = row[0][:16] if row else "Never"

        conn.close()

        return {
            "total_ga_runs": total_ga_runs,
            "running_ga_runs": running_ga_runs,
            "completed_ga_runs": completed_ga_runs,
            "total_backtests": total_backtests,
            "avg_sharpe": avg_sharpe,
            "best_sharpe": best_sharpe,
            "best_ga_fitness": best_ga_fitness,
            "discovered_strategies": discovered_count,
            "last_ga_run": last_ga_run,
        }
    except Exception as e:
        logger.error(f"Error fetching research overview: {e}")
    return {}


def get_live_research_progress() -> Optional[str]:
    """Get current progress from research log (lightweight)."""
    try:
        import subprocess
        log_file = PROJECT_ROOT / "logs" / "nightly_research.log"
        if not log_file.exists():
            return None
        # Get last line matching progress pattern - very fast
        result = subprocess.run(
            ["grep", "-E", r"\[\d+/\d+\]", str(log_file)],
            capture_output=True, text=True, timeout=1
        )
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            if lines:
                last = lines[-1]
                # Extract "[2/5] Testing:" or "[2/5] WF:" pattern
                import re
                match = re.search(r'\[(\d+/\d+)\]\s*(\w+)', last)
                if match:
                    return f"{match.group(1)} {match.group(2)}"
    except Exception:
        pass
    return None


def get_ga_runs(limit: int = 10) -> List[Dict[str, Any]]:
    """Fetch recent GA optimization runs with real-time progress."""
    try:
        db_path = PROJECT_ROOT / "db" / "research.db"
        if not db_path.exists():
            return []

        conn = sqlite3.connect(db_path)

        # Get runs from ga_runs table
        df = pd.read_sql_query(
            f"""SELECT run_id, start_time, strategies_evolved,
                       total_generations, improvements_found, status
                FROM ga_runs
                ORDER BY start_time DESC
                LIMIT {limit}""",
            conn
        )

        # Get real-time progress from ga_populations for running jobs
        live_progress = pd.read_sql_query(
            """SELECT strategy, MAX(generation) as max_gen, MAX(best_fitness) as best_fitness
               FROM ga_populations
               GROUP BY strategy""",
            conn
        )
        conn.close()

        if df.empty:
            return []

        # Format for display
        results = []
        for _, row in df.iterrows():
            # For running jobs, get progress for only the strategies in this run
            if row['status'] == 'running':
                run_strategies = (row['strategies_evolved'] or '').split(',')
                run_strategies = [s.strip() for s in run_strategies if s.strip()]
                if run_strategies and not live_progress.empty:
                    # Filter to only strategies in this run
                    run_progress = live_progress[live_progress['strategy'].isin(run_strategies)]
                    gens = int(run_progress['max_gen'].sum()) if not run_progress.empty else 0
                    impr = int((run_progress['best_fitness'] > 0).sum()) if not run_progress.empty else 0
                else:
                    gens = 0
                    impr = 0
            else:
                gens = row['total_generations'] or 0
                impr = row['improvements_found'] or 0

            # Determine display status with resumability indicator
            # Only show "resumable" if there's actual progress to resume
            status = row['status'] or "unknown"
            has_progress = gens > 0

            if status == 'running':
                # Show live progress for running jobs
                live_prog = get_live_research_progress()
                display_status = f"running ({live_prog})" if live_prog else "running"
            elif status == 'interrupted':
                display_status = "interrupted (resumable)" if has_progress else "interrupted"
            elif status == 'paused':
                display_status = "paused (will resume)" if has_progress else "paused"
            elif status == 'abandoned':
                display_status = "abandoned (resumable)" if has_progress else "abandoned"
            else:
                display_status = status

            results.append({
                "run_id": row['run_id'][:8] + "...",
                "start_time": row['start_time'][:16] if row['start_time'] else "",
                "strategies": row['strategies_evolved'][:30] if row['strategies_evolved'] else "-",
                "generations": gens,
                "improvements": impr,
                "status": display_status,
            })
        return results
    except Exception as e:
        logger.error(f"Error fetching GA runs: {e}")
    return []


def get_backtest_results(limit: int = 10) -> List[Dict[str, Any]]:
    """Fetch recent backtest results."""
    try:
        db_path = PROJECT_ROOT / "db" / "research.db"
        if not db_path.exists():
            return []

        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(
            f"""SELECT strategy, timestamp, sharpe_ratio, sortino_ratio,
                       max_drawdown_pct, total_trades, win_rate, total_return
                FROM backtests
                ORDER BY timestamp DESC
                LIMIT {limit}""",
            conn
        )
        conn.close()

        if df.empty:
            return []

        return df.to_dict('records')
    except Exception as e:
        logger.error(f"Error fetching backtest results: {e}")
    return []


def get_ga_fitness_history() -> pd.DataFrame:
    """Fetch GA fitness evolution over generations."""
    try:
        db_path = PROJECT_ROOT / "db" / "research.db"
        if not db_path.exists():
            return pd.DataFrame()

        conn = sqlite3.connect(db_path)
        # Use ga_populations for current state, filter out zero values
        df = pd.read_sql_query(
            """SELECT strategy, generation, best_fitness
               FROM ga_populations
               WHERE best_fitness > 0
               ORDER BY strategy, generation ASC""",
            conn
        )
        conn.close()
        return df
    except Exception as e:
        logger.error(f"Error fetching GA fitness history: {e}")
    return pd.DataFrame()


def get_trade_history(limit: int = 20) -> List[Dict[str, Any]]:
    """Fetch recent trade history from database."""
    try:
        db_path = PROJECT_ROOT / "db" / "trades.db"
        if not db_path.exists():
            return []

        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(
            f"""SELECT timestamp, symbol, strategy, side, quantity,
                       entry_price, exit_price, pnl, pnl_percent, status, exit_reason
                FROM trades
                ORDER BY timestamp DESC
                LIMIT {limit}""",
            conn
        )
        conn.close()

        if df.empty:
            return []

        # Format for display
        results = []
        for _, row in df.iterrows():
            results.append({
                "time": row['timestamp'][:16] if row['timestamp'] else "",
                "symbol": row['symbol'],
                "strategy": row['strategy'][:15] if row['strategy'] else "",
                "side": row['side'],
                "qty": row['quantity'],
                "entry": row['entry_price'],
                "exit": row['exit_price'] if row['exit_price'] else "-",
                "pnl": row['pnl'] if row['pnl'] else 0,
                "status": row['status'],
            })
        return results
    except Exception as e:
        logger.error(f"Error fetching trade history: {e}")
    return []


def get_pending_orders() -> List[Dict[str, Any]]:
    """Fetch pending/open orders from Alpaca."""
    try:
        broker = get_broker()
        orders = broker.api.list_orders(status='open')
        return [
            {
                "symbol": o.symbol,
                "side": o.side,
                "type": o.type,
                "qty": float(o.qty),
                "limit": float(o.limit_price) if o.limit_price else "-",
                "stop": float(o.stop_price) if o.stop_price else "-",
                "status": o.status,
                "submitted": str(o.submitted_at)[:16] if o.submitted_at else "",
            }
            for o in orders
        ]
    except Exception as e:
        logger.error(f"Error fetching pending orders: {e}")
    return []


def get_position_price_history(days: int = 10) -> Dict[str, pd.DataFrame]:
    """Fetch recent price history for all positions."""
    try:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        broker = get_broker()
        positions = broker.get_positions()

        if not positions:
            return {}

        symbols = [p.symbol for p in positions]
        end = datetime.now()
        start = end - timedelta(days=days)

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start,
            end=end
        )
        bars = broker.data_client.get_stock_bars(request)

        # Convert to dict of DataFrames
        result = {}
        for sym in symbols:
            if sym in bars.data:
                data = []
                for bar in bars.data[sym]:
                    data.append({
                        'date': bar.timestamp.date(),
                        'open': bar.open,
                        'high': bar.high,
                        'low': bar.low,
                        'close': bar.close,
                        'volume': bar.volume,
                    })
                if data:
                    result[sym] = pd.DataFrame(data)
        return result
    except Exception as e:
        logger.error(f"Error fetching price history: {e}")
    return {}


def get_risk_metrics() -> Dict[str, Any]:
    """Calculate portfolio risk metrics."""
    try:
        broker = get_broker()
        account = broker.get_account()
        positions = broker.get_positions()

        if not account:
            return {}

        equity = float(account.equity)
        cash = float(account.cash)
        positions_value = sum(float(p.market_value) for p in positions)

        # Calculate metrics
        exposure_pct = (positions_value / equity * 100) if equity > 0 else 0
        cash_pct = (cash / equity * 100) if equity > 0 else 0

        # Position concentration (largest position %)
        if positions:
            largest_pos = max(float(p.market_value) for p in positions)
            concentration = (largest_pos / equity * 100) if equity > 0 else 0
        else:
            concentration = 0

        # Count by sector (using symbol as proxy - ETFs have sector exposure)
        sector_exposure = {}
        for p in positions:
            # Simple categorization based on common ETFs
            sym = p.symbol
            if sym in ['XLF']:
                sector = 'Financials'
            elif sym in ['XLK', 'QQQ']:
                sector = 'Technology'
            elif sym in ['XLY']:
                sector = 'Consumer'
            elif sym in ['XEL']:
                sector = 'Utilities'
            elif sym in ['IWM']:
                sector = 'Small Cap'
            else:
                sector = 'Other'
            sector_exposure[sector] = sector_exposure.get(sector, 0) + float(p.market_value)

        return {
            "equity": equity,
            "exposure_pct": exposure_pct,
            "cash_pct": cash_pct,
            "concentration": concentration,
            "num_positions": len(positions),
            "sector_exposure": sector_exposure,
            "largest_position": max((p.symbol for p in positions), key=lambda s: next(float(p.market_value) for p in positions if p.symbol == s), default="-") if positions else "-",
        }
    except Exception as e:
        logger.error(f"Error calculating risk metrics: {e}")
    return {}


def get_market_status() -> Dict[str, Any]:
    """Get current market status."""
    try:
        is_open = MarketHours.is_market_open()

        # Get market hours info without using DailyOrchestrator (which uses signals)
        now = datetime.now()

        # Determine phase based on time (simplified)
        hour = now.hour
        minute = now.minute
        weekday = now.weekday()

        if weekday >= 5:  # Weekend
            phase = "weekend"
            time_to_next = "Monday 9:30 AM ET"
        elif hour < 9 or (hour == 9 and minute < 30):
            phase = "pre_market"
            time_to_next = "Market opens 9:30 AM ET"
        elif hour < 16:
            phase = "market_hours"
            close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)
            remaining = close_time - now
            time_to_next = str(remaining).split('.')[0]
        else:
            phase = "after_hours"
            time_to_next = "Tomorrow 9:30 AM ET"

        return {
            "is_open": is_open,
            "phase": phase,
            "time_to_next": time_to_next,
        }
    except Exception as e:
        logger.error(f"Error getting market status: {e}")
    return {"is_open": False, "phase": "unknown", "time_to_next": "N/A"}


def get_circuit_breaker_status() -> Dict[str, Any]:
    """Get current circuit breaker status."""
    try:
        manager = CircuitBreakerManager()
        return manager.get_status()
    except Exception as e:
        logger.error(f"Error fetching circuit breaker status: {e}")
    return {
        "trading_allowed": True,
        "position_multiplier": 1.0,
        "active_breakers": [],
        "file_kill_switches": []
    }


def get_circuit_breaker_history(limit: int = 10) -> List[Dict[str, Any]]:
    """Get recent circuit breaker history."""
    try:
        db = CircuitBreakerDB()
        return db.get_kill_switch_log(limit=limit)
    except Exception as e:
        logger.error(f"Error fetching circuit breaker history: {e}")
    return []


def get_system_errors(limit: int = 50, include_resolved: bool = False) -> List[Dict[str, Any]]:
    """Fetch recent system errors and warnings from database.

    Args:
        limit: Maximum number of errors to return
        include_resolved: Whether to include resolved errors

    Returns:
        List of error records with timestamp, level, message, component, etc.
    """
    try:
        db_path = PROJECT_ROOT / "db" / "performance.db"
        if not db_path.exists():
            return []

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if error_log table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='error_log'"
        )
        if not cursor.fetchone():
            conn.close()
            return []

        # Build query based on include_resolved
        where_clause = "" if include_resolved else "WHERE is_resolved = 0"

        df = pd.read_sql_query(
            f"""SELECT id, timestamp, level, logger_name, message, source_file,
                       line_number, exception_type, component, is_resolved
                FROM error_log
                {where_clause}
                ORDER BY timestamp DESC
                LIMIT {limit}""",
            conn
        )
        conn.close()

        if df.empty:
            return []

        # Format for display
        results = []
        for _, row in df.iterrows():
            # Truncate message for display
            message = row['message'][:150] + "..." if len(str(row['message'])) > 150 else row['message']

            # Extract just filename from full path
            source_file = row['source_file']
            if source_file:
                source_file = Path(source_file).name

            results.append({
                "id": row['id'],
                "timestamp": row['timestamp'][:19] if row['timestamp'] else "",
                "level": row['level'],
                "message": message,
                "component": row['component'] or "system",
                "source": f"{source_file}:{row['line_number']}" if source_file else "",
                "exception": row['exception_type'] or "",
                "is_resolved": bool(row['is_resolved']),
            })
        return results
    except Exception as e:
        logger.error(f"Error fetching system errors: {e}")
    return []


def mark_error_resolved(error_id: int, resolved_by: str = "dashboard") -> bool:
    """Mark an error as resolved.

    Args:
        error_id: ID of the error to mark resolved
        resolved_by: Who resolved it

    Returns:
        True if successful
    """
    try:
        db_path = PROJECT_ROOT / "db" / "performance.db"
        if not db_path.exists():
            return False

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE error_log
            SET is_resolved = 1, resolved_at = ?, resolved_by = ?
            WHERE id = ?
        """, (datetime.now().isoformat(), resolved_by, error_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error marking error resolved: {e}")
    return False


def clear_all_errors() -> int:
    """Clear all errors from the error log.

    Returns:
        Number of errors cleared
    """
    try:
        db_path = PROJECT_ROOT / "db" / "performance.db"
        if not db_path.exists():
            return 0

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM error_log")
        count = cursor.fetchone()[0]
        cursor.execute("DELETE FROM error_log")
        conn.commit()
        conn.close()
        return count
    except Exception as e:
        logger.error(f"Error clearing errors: {e}")
    return 0


# ============================================================================
# Layout Components
# ============================================================================

def create_account_card():
    """Create account summary card."""
    return dbc.Card([
        dbc.CardHeader(html.H5("Account Summary", className="mb-0")),
        dbc.CardBody(id="account-summary")
    ], className="mb-3")


def create_positions_table():
    """Create positions data table."""
    return dbc.Card([
        dbc.CardHeader(html.H5("Open Positions", className="mb-0")),
        dbc.CardBody([
            dash_table.DataTable(
                id='positions-table',
                columns=[
                    {"name": "Symbol", "id": "symbol"},
                    {"name": "Qty", "id": "qty", "type": "numeric"},
                    {"name": "Entry", "id": "entry_price", "type": "numeric",
                     "format": {"specifier": ",.2f"}},
                    {"name": "Current", "id": "current_price", "type": "numeric",
                     "format": {"specifier": ",.2f"}},
                    {"name": "Value", "id": "market_value", "type": "numeric",
                     "format": {"specifier": ",.0f"}},
                    {"name": "P&L", "id": "unrealized_pnl", "type": "numeric",
                     "format": {"specifier": "+,.2f"}},
                    {"name": "P&L %", "id": "pnl_pct", "type": "numeric",
                     "format": {"specifier": "+.2f"}},
                ],
                style_table={'overflowX': 'auto'},
                style_cell={
                    'backgroundColor': '#303030',
                    'color': 'white',
                    'textAlign': 'right',
                    'padding': '8px',
                },
                style_header={
                    'backgroundColor': '#444',
                    'fontWeight': 'bold',
                    'textAlign': 'center',
                },
                style_data_conditional=[
                    {
                        'if': {'filter_query': '{unrealized_pnl} > 0'},
                        'color': '#00bc8c',
                    },
                    {
                        'if': {'filter_query': '{unrealized_pnl} < 0'},
                        'color': '#e74c3c',
                    },
                ],
                page_size=10,
            )
        ])
    ], className="mb-3")


def create_equity_chart():
    """Create equity curve chart."""
    return dbc.Card([
        dbc.CardHeader(html.H5("Equity Curve", className="mb-0")),
        dbc.CardBody([
            dcc.Graph(id='equity-chart', style={'height': '300px'})
        ])
    ], className="mb-3")


def create_strategy_table():
    """Create strategy performance table."""
    return dbc.Card([
        dbc.CardHeader(html.H5("Strategy Performance", className="mb-0")),
        dbc.CardBody([
            dash_table.DataTable(
                id='strategy-table',
                columns=[
                    {"name": "Strategy", "id": "strategy"},
                    {"name": "Trades", "id": "total_trades", "type": "numeric"},
                    {"name": "Win Rate", "id": "win_rate", "type": "numeric",
                     "format": {"specifier": ".1%"}},
                    {"name": "P&L", "id": "total_pnl", "type": "numeric",
                     "format": {"specifier": "+,.0f"}},
                    {"name": "Sharpe", "id": "sharpe_ratio", "type": "numeric",
                     "format": {"specifier": ".2f"}},
                    {"name": "Max DD", "id": "max_drawdown_pct", "type": "numeric",
                     "format": {"specifier": ".1%"}},
                    {"name": "Enabled", "id": "is_enabled"},
                ],
                style_table={'overflowX': 'auto'},
                style_cell={
                    'backgroundColor': '#303030',
                    'color': 'white',
                    'textAlign': 'right',
                    'padding': '8px',
                },
                style_header={
                    'backgroundColor': '#444',
                    'fontWeight': 'bold',
                    'textAlign': 'center',
                },
                page_size=10,
            )
        ])
    ], className="mb-3")


def create_alerts_card():
    """Create recent alerts card."""
    return dbc.Card([
        dbc.CardHeader(html.H5("Recent Alerts", className="mb-0")),
        dbc.CardBody(id="alerts-list", style={'maxHeight': '300px', 'overflowY': 'auto'})
    ], className="mb-3")


def create_status_card():
    """Create system status card."""
    return dbc.Card([
        dbc.CardHeader(html.H5("System Status", className="mb-0")),
        dbc.CardBody(id="system-status")
    ], className="mb-3")


def create_circuit_breaker_card():
    """Create circuit breaker status card - prominent at top of dashboard."""
    return dbc.Card([
        dbc.CardHeader([
            html.H4([
                html.I(className="fas fa-shield-alt me-2"),
                "Circuit Breaker Status"
            ], className="mb-0 text-white")
        ], className="bg-dark"),
        dbc.CardBody(id="circuit-breaker-status")
    ], className="mb-3 border-2", id="circuit-breaker-card")


def create_research_overview_card():
    """Create research overview summary card."""
    return dbc.Card([
        dbc.CardHeader(html.H5("Research Overview", className="mb-0")),
        dbc.CardBody(id="research-overview")
    ], className="mb-3")


def create_ga_runs_table():
    """Create GA optimization runs table."""
    return dbc.Card([
        dbc.CardHeader(html.H5("GA Optimization Runs", className="mb-0")),
        dbc.CardBody([
            dash_table.DataTable(
                id='ga-runs-table',
                columns=[
                    {"name": "Run ID", "id": "run_id"},
                    {"name": "Started", "id": "start_time"},
                    {"name": "Generations", "id": "generations", "type": "numeric"},
                    {"name": "Improvements", "id": "improvements", "type": "numeric"},
                    {"name": "Status", "id": "status"},
                ],
                style_table={'overflowX': 'auto'},
                style_cell={
                    'backgroundColor': '#303030',
                    'color': 'white',
                    'textAlign': 'left',
                    'padding': '8px',
                    'fontSize': '13px',
                },
                style_header={
                    'backgroundColor': '#444',
                    'fontWeight': 'bold',
                    'textAlign': 'center',
                },
                style_data_conditional=[
                    {
                        'if': {'filter_query': '{status} = "running"'},
                        'backgroundColor': 'rgba(0,188,140,0.2)',
                    },
                    {
                        'if': {'filter_query': '{status} = "completed"'},
                        'backgroundColor': 'rgba(144,238,144,0.2)',
                        'color': '#90EE90',
                    },
                    {
                        'if': {'filter_query': '{status} contains "error"'},
                        'backgroundColor': 'rgba(231,76,60,0.2)',
                    },
                    # Yellow highlight for resumable runs (interrupted with progress)
                    {
                        'if': {'filter_query': '{status} contains "resumable"'},
                        'backgroundColor': 'rgba(255,193,7,0.3)',
                        'color': '#FFD700',
                    },
                    # Yellow highlight for paused runs (will resume)
                    {
                        'if': {'filter_query': '{status} contains "will resume"'},
                        'backgroundColor': 'rgba(255,193,7,0.25)',
                        'color': '#FFD700',
                    },
                    # Gray for abandoned/interrupted without progress
                    {
                        'if': {'filter_query': '{status} = "abandoned"'},
                        'backgroundColor': 'rgba(128,128,128,0.2)',
                        'color': '#A0A0A0',
                    },
                    {
                        'if': {'filter_query': '{status} = "interrupted"'},
                        'backgroundColor': 'rgba(128,128,128,0.2)',
                        'color': '#A0A0A0',
                    },
                ],
                page_size=10,
            )
        ])
    ], className="mb-3")


def create_backtest_results_table():
    """Create backtest results table."""
    return dbc.Card([
        dbc.CardHeader(html.H5("Recent Backtests", className="mb-0")),
        dbc.CardBody([
            dash_table.DataTable(
                id='backtest-table',
                columns=[
                    {"name": "Strategy", "id": "strategy"},
                    {"name": "Sharpe", "id": "sharpe_ratio", "type": "numeric",
                     "format": {"specifier": ".2f"}},
                    {"name": "Sortino", "id": "sortino_ratio", "type": "numeric",
                     "format": {"specifier": ".2f"}},
                    {"name": "Max DD%", "id": "max_drawdown_pct", "type": "numeric",
                     "format": {"specifier": ".1f"}},
                    {"name": "Trades", "id": "total_trades", "type": "numeric"},
                    {"name": "Win%", "id": "win_rate", "type": "numeric",
                     "format": {"specifier": ".1f"}},
                    {"name": "Return%", "id": "total_return", "type": "numeric",
                     "format": {"specifier": ".1f"}},
                ],
                style_table={'overflowX': 'auto'},
                style_cell={
                    'backgroundColor': '#303030',
                    'color': 'white',
                    'textAlign': 'right',
                    'padding': '8px',
                    'fontSize': '13px',
                },
                style_header={
                    'backgroundColor': '#444',
                    'fontWeight': 'bold',
                    'textAlign': 'center',
                },
                style_data_conditional=[
                    {
                        'if': {'filter_query': '{sharpe_ratio} > 0.5'},
                        'color': '#00bc8c',
                    },
                    {
                        'if': {'filter_query': '{sharpe_ratio} < 0'},
                        'color': '#e74c3c',
                    },
                ],
                page_size=10,
            )
        ])
    ], className="mb-3")


def create_ga_fitness_chart():
    """Create GA fitness evolution chart."""
    return dbc.Card([
        dbc.CardHeader(html.H5("GA Fitness Evolution", className="mb-0")),
        dbc.CardBody([
            dcc.Graph(id='ga-fitness-chart', style={'height': '250px'})
        ])
    ], className="mb-3")


def create_trade_history_table():
    """Create trade history table."""
    return dbc.Card([
        dbc.CardHeader(html.H5("Trade History", className="mb-0")),
        dbc.CardBody([
            dash_table.DataTable(
                id='trade-history-table',
                columns=[
                    {"name": "Time", "id": "time"},
                    {"name": "Symbol", "id": "symbol"},
                    {"name": "Side", "id": "side"},
                    {"name": "Qty", "id": "qty", "type": "numeric"},
                    {"name": "Entry", "id": "entry", "type": "numeric",
                     "format": {"specifier": ",.2f"}},
                    {"name": "Exit", "id": "exit"},
                    {"name": "P&L", "id": "pnl", "type": "numeric",
                     "format": {"specifier": "+,.2f"}},
                    {"name": "Status", "id": "status"},
                ],
                style_table={'overflowX': 'auto'},
                style_cell={
                    'backgroundColor': '#303030',
                    'color': 'white',
                    'textAlign': 'left',
                    'padding': '6px',
                    'fontSize': '12px',
                },
                style_header={
                    'backgroundColor': '#444',
                    'fontWeight': 'bold',
                    'textAlign': 'center',
                },
                style_data_conditional=[
                    {'if': {'filter_query': '{pnl} > 0'}, 'color': '#00bc8c'},
                    {'if': {'filter_query': '{pnl} < 0'}, 'color': '#e74c3c'},
                    {'if': {'filter_query': '{side} = "BUY"'}, 'backgroundColor': 'rgba(0,188,140,0.1)'},
                    {'if': {'filter_query': '{side} = "SELL"'}, 'backgroundColor': 'rgba(231,76,60,0.1)'},
                ],
                page_size=8,
            )
        ])
    ], className="mb-3")


def create_pending_orders_table():
    """Create pending orders table."""
    return dbc.Card([
        dbc.CardHeader(html.H5("Pending Orders", className="mb-0")),
        dbc.CardBody([
            dash_table.DataTable(
                id='pending-orders-table',
                columns=[
                    {"name": "Symbol", "id": "symbol"},
                    {"name": "Side", "id": "side"},
                    {"name": "Type", "id": "type"},
                    {"name": "Qty", "id": "qty", "type": "numeric"},
                    {"name": "Limit", "id": "limit"},
                    {"name": "Status", "id": "status"},
                ],
                style_table={'overflowX': 'auto'},
                style_cell={
                    'backgroundColor': '#303030',
                    'color': 'white',
                    'textAlign': 'left',
                    'padding': '6px',
                    'fontSize': '12px',
                },
                style_header={
                    'backgroundColor': '#444',
                    'fontWeight': 'bold',
                    'textAlign': 'center',
                },
                page_size=5,
            )
        ])
    ], className="mb-3")


def create_risk_metrics_card():
    """Create risk metrics card."""
    return dbc.Card([
        dbc.CardHeader(html.H5("Risk Metrics", className="mb-0")),
        dbc.CardBody(id="risk-metrics")
    ], className="mb-3")


def create_system_errors_card():
    """Create system errors/warnings display card."""
    return dbc.Card([
        dbc.CardHeader([
            html.Div([
                html.H5([
                    html.I(className="fas fa-exclamation-triangle me-2 text-warning"),
                    "System Errors & Warnings"
                ], className="mb-0 d-inline"),
                dbc.Button(
                    [html.I(className="fas fa-trash me-1"), "Clear All"],
                    id="clear-errors-btn",
                    color="secondary",
                    size="sm",
                    className="float-end",
                ),
            ], className="d-flex justify-content-between align-items-center")
        ]),
        dbc.CardBody([
            dash_table.DataTable(
                id='system-errors-table',
                columns=[
                    {"name": "Time", "id": "timestamp"},
                    {"name": "Level", "id": "level"},
                    {"name": "Component", "id": "component"},
                    {"name": "Message", "id": "message"},
                    {"name": "Source", "id": "source"},
                    {"name": "Exception", "id": "exception"},
                ],
                style_table={'overflowX': 'auto'},
                style_cell={
                    'backgroundColor': '#303030',
                    'color': 'white',
                    'textAlign': 'left',
                    'padding': '6px',
                    'fontSize': '11px',
                    'maxWidth': '300px',
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis',
                },
                style_header={
                    'backgroundColor': '#444',
                    'fontWeight': 'bold',
                    'textAlign': 'center',
                },
                style_data_conditional=[
                    {
                        'if': {'filter_query': '{level} = "ERROR"'},
                        'backgroundColor': 'rgba(231,76,60,0.2)',
                        'color': '#e74c3c',
                    },
                    {
                        'if': {'filter_query': '{level} = "CRITICAL"'},
                        'backgroundColor': 'rgba(231,76,60,0.4)',
                        'color': '#ff6b6b',
                        'fontWeight': 'bold',
                    },
                    {
                        'if': {'filter_query': '{level} = "WARNING"'},
                        'backgroundColor': 'rgba(243,156,18,0.2)',
                        'color': '#f39c12',
                    },
                ],
                page_size=10,
                tooltip_data=[],
                tooltip_duration=None,
            ),
            html.Div(id="system-errors-summary", className="mt-2 text-muted small"),
        ], style={'maxHeight': '400px', 'overflowY': 'auto'})
    ], className="mb-3")


def create_position_charts():
    """Create position price charts."""
    return dbc.Card([
        dbc.CardHeader(html.H5("Position Price Charts (10 Day)", className="mb-0")),
        dbc.CardBody([
            dcc.Graph(id='position-charts', style={'height': '350px'})
        ])
    ], className="mb-3")


# ============================================================================
# Main Layout
# ============================================================================

app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H2("Trading System Dashboard", className="text-primary mb-0"),
            html.Small(id="last-updated", className="text-muted"),
        ], width=8),
        dbc.Col([
            dbc.Button("Refresh", id="refresh-btn", color="primary", className="float-end"),
        ], width=4),
    ], className="mb-4 mt-3"),

    # Circuit Breaker Status - FIRST AND MOST PROMINENT
    dbc.Row([
        dbc.Col(create_circuit_breaker_card(), width=12),
    ]),

    # Top row: Account + Status
    dbc.Row([
        dbc.Col(create_account_card(), width=8),
        dbc.Col(create_status_card(), width=4),
    ]),

    # Middle row: Positions + Equity
    dbc.Row([
        dbc.Col(create_positions_table(), width=7),
        dbc.Col(create_equity_chart(), width=5),
    ]),

    # Strategy row: Strategies + Alerts
    dbc.Row([
        dbc.Col(create_strategy_table(), width=7),
        dbc.Col(create_alerts_card(), width=5),
    ]),

    # Trading Activity Section
    dbc.Row([
        dbc.Col([
            html.Hr(className="my-4"),
            html.H4("Trading Activity", className="text-warning mb-3"),
        ], width=12),
    ]),

    # Trading row: Trade History + Orders + Risk
    dbc.Row([
        dbc.Col(create_trade_history_table(), width=5),
        dbc.Col(create_pending_orders_table(), width=3),
        dbc.Col(create_risk_metrics_card(), width=4),
    ]),

    # Position Charts row
    dbc.Row([
        dbc.Col(create_position_charts(), width=12),
    ]),

    # Research Section Header
    dbc.Row([
        dbc.Col([
            html.Hr(className="my-4"),
            html.H4("Research & Optimization", className="text-info mb-3"),
        ], width=12),
    ]),

    # Research row 1: Overview + GA Fitness Chart
    dbc.Row([
        dbc.Col(create_research_overview_card(), width=4),
        dbc.Col(create_ga_fitness_chart(), width=8),
    ]),

    # Research row 2: GA Runs + Backtests
    dbc.Row([
        dbc.Col(create_ga_runs_table(), width=5),
        dbc.Col(create_backtest_results_table(), width=7),
    ]),

    # System Health Section Header
    dbc.Row([
        dbc.Col([
            html.Hr(className="my-4"),
            html.H4([
                html.I(className="fas fa-heartbeat me-2"),
                "System Health"
            ], className="text-danger mb-3"),
        ], width=12),
    ]),

    # System Health row: Errors/Warnings
    dbc.Row([
        dbc.Col(create_system_errors_card(), width=12),
    ]),

    # Auto-refresh interval (30 seconds)
    dcc.Interval(id='interval-component', interval=30*1000, n_intervals=0),

], fluid=True, className="bg-dark")


# ============================================================================
# Callbacks
# ============================================================================

@callback(
    [
        Output('circuit-breaker-status', 'children'),
        Output('circuit-breaker-card', 'className'),
        Output('account-summary', 'children'),
        Output('positions-table', 'data'),
        Output('equity-chart', 'figure'),
        Output('strategy-table', 'data'),
        Output('alerts-list', 'children'),
        Output('system-status', 'children'),
        Output('trade-history-table', 'data'),
        Output('pending-orders-table', 'data'),
        Output('risk-metrics', 'children'),
        Output('position-charts', 'figure'),
        Output('research-overview', 'children'),
        Output('ga-runs-table', 'data'),
        Output('backtest-table', 'data'),
        Output('ga-fitness-chart', 'figure'),
        Output('system-errors-table', 'data'),
        Output('system-errors-summary', 'children'),
        Output('last-updated', 'children'),
    ],
    [
        Input('interval-component', 'n_intervals'),
        Input('refresh-btn', 'n_clicks'),
    ]
)
def update_dashboard(n_intervals, n_clicks):
    """Update all dashboard components."""

    # Circuit Breaker Status - FIRST and most important
    cb_status = get_circuit_breaker_status()
    cb_history = get_circuit_breaker_history(10)

    trading_allowed = cb_status.get('trading_allowed', True)
    position_multiplier = cb_status.get('position_multiplier', 1.0)
    active_breakers = cb_status.get('active_breakers', [])
    file_switches = cb_status.get('file_kill_switches', [])

    # Determine overall status and colors
    if not trading_allowed:
        status_color = "danger"
        status_text = "TRADING HALTED"
        status_icon = "fas fa-stop-circle"
        card_class = "mb-3 border-2 border-danger blink-danger"
    elif position_multiplier < 1.0:
        status_color = "warning"
        status_text = "REDUCED TRADING"
        status_icon = "fas fa-exclamation-triangle"
        card_class = "mb-3 border-2 border-warning"
    else:
        status_color = "success"
        status_text = "TRADING ALLOWED"
        status_icon = "fas fa-check-circle"
        card_class = "mb-3 border-2 border-success"

    # Build circuit breaker content
    cb_content = html.Div([
        # Main status row
        dbc.Row([
            # Large status indicator
            dbc.Col([
                dbc.Alert([
                    html.Div([
                        html.I(className=f"{status_icon} fa-3x mb-2"),
                        html.H3(status_text, className="mb-0"),
                    ], className="text-center")
                ], color=status_color, className="mb-0 py-3")
            ], width=4),
            # Position multiplier
            dbc.Col([
                html.Div([
                    html.H2(f"{int(position_multiplier * 100)}%", className=f"text-{status_color} mb-0"),
                    html.P("Position Multiplier", className="text-muted mb-0"),
                    html.Small(
                        "Normal trading" if position_multiplier >= 1.0 else "Reduced position sizes",
                        className="text-muted"
                    ),
                ], className="text-center py-3")
            ], width=2),
            # Active breakers list
            dbc.Col([
                html.H6("Active Breakers", className="text-warning mb-2"),
                html.Div([
                    dbc.Badge(
                        f"{b['type']}: {b['reason'][:40]}..." if len(b.get('reason', '')) > 40 else f"{b['type']}: {b.get('reason', 'Unknown')}",
                        color="danger" if b.get('action') == 'halt' else "warning",
                        className="me-1 mb-1 d-block text-start",
                        style={'whiteSpace': 'normal', 'fontSize': '11px'}
                    )
                    for b in active_breakers
                ] if active_breakers else [html.Span("None", className="text-success")],
                style={'maxHeight': '100px', 'overflowY': 'auto'})
            ], width=3),
            # File kill switches
            dbc.Col([
                html.H6("File Kill Switches", className="text-danger mb-2"),
                html.Div([
                    dbc.Badge(
                        switch,
                        color="danger",
                        className="me-1 mb-1 d-block"
                    )
                    for switch in file_switches
                ] if file_switches else [html.Span("None", className="text-success")],
                style={'maxHeight': '100px', 'overflowY': 'auto'})
            ], width=3),
        ], className="mb-3"),

        # Active breakers details (if any)
        html.Div([
            html.Hr(className="my-2"),
            html.H6("Active Breaker Details", className="text-info mb-2"),
            dash_table.DataTable(
                columns=[
                    {"name": "Type", "id": "type"},
                    {"name": "Reason", "id": "reason"},
                    {"name": "Target", "id": "target"},
                    {"name": "Expires", "id": "expires_at"},
                ],
                data=[
                    {
                        "type": b['type'],
                        "reason": b.get('reason', 'N/A')[:60] + ('...' if len(b.get('reason', '')) > 60 else ''),
                        "target": b.get('target', 'all'),
                        "expires_at": b.get('expires_at', 'Never')[:16] if b.get('expires_at') else 'Never',
                    }
                    for b in active_breakers
                ],
                style_table={'overflowX': 'auto'},
                style_cell={
                    'backgroundColor': '#303030',
                    'color': 'white',
                    'textAlign': 'left',
                    'padding': '6px',
                    'fontSize': '12px',
                },
                style_header={
                    'backgroundColor': '#444',
                    'fontWeight': 'bold',
                    'textAlign': 'center',
                },
                page_size=5,
            )
        ]) if active_breakers else html.Div(),

        # Recent history
        html.Div([
            html.Hr(className="my-2"),
            html.H6("Recent Circuit Breaker History", className="text-muted mb-2"),
            dash_table.DataTable(
                columns=[
                    {"name": "Time", "id": "timestamp"},
                    {"name": "Type", "id": "switch_type"},
                    {"name": "Triggered By", "id": "triggered_by"},
                    {"name": "Positions", "id": "positions_affected"},
                    {"name": "Orders Cancelled", "id": "orders_cancelled"},
                ],
                data=[
                    {
                        "timestamp": h.get('timestamp', '')[:16] if h.get('timestamp') else '',
                        "switch_type": h.get('switch_type', 'N/A'),
                        "triggered_by": h.get('triggered_by', 'N/A'),
                        "positions_affected": h.get('positions_affected', 0),
                        "orders_cancelled": h.get('orders_cancelled', 0),
                    }
                    for h in cb_history
                ],
                style_table={'overflowX': 'auto'},
                style_cell={
                    'backgroundColor': '#303030',
                    'color': 'white',
                    'textAlign': 'left',
                    'padding': '6px',
                    'fontSize': '11px',
                },
                style_header={
                    'backgroundColor': '#444',
                    'fontWeight': 'bold',
                    'textAlign': 'center',
                },
                style_data_conditional=[
                    {
                        'if': {'filter_query': '{switch_type} contains "HALT"'},
                        'backgroundColor': 'rgba(231,76,60,0.2)',
                    },
                    {
                        'if': {'filter_query': '{switch_type} contains "CLOSE"'},
                        'backgroundColor': 'rgba(231,76,60,0.3)',
                    },
                ],
                page_size=5,
            )
        ]) if cb_history else html.Div([
            html.Hr(className="my-2"),
            html.Small("No recent circuit breaker events", className="text-muted")
        ]),
    ])

    # Account summary
    account = get_account_data()
    if account:
        account_content = html.Div([
            dbc.Row([
                dbc.Col([
                    html.H3(f"${account.get('equity', 0):,.2f}", className="text-success mb-0"),
                    html.Small("Equity", className="text-muted"),
                ], width=3),
                dbc.Col([
                    html.H4(f"${account.get('cash', 0):,.2f}", className="mb-0"),
                    html.Small("Cash", className="text-muted"),
                ], width=3),
                dbc.Col([
                    html.H4(f"${account.get('buying_power', 0):,.2f}", className="mb-0"),
                    html.Small("Buying Power", className="text-muted"),
                ], width=3),
                dbc.Col([
                    html.H4(f"{account.get('day_trade_count', 0)}/3", className="mb-0"),
                    html.Small("Day Trades", className="text-muted"),
                ], width=3),
            ]),
        ])
    else:
        account_content = html.Div("Unable to fetch account data", className="text-danger")

    # Positions
    positions = get_positions_data()

    # Equity chart
    equity_df = get_equity_history()
    if not equity_df.empty:
        equity_fig = go.Figure()
        equity_fig.add_trace(go.Scatter(
            x=equity_df['date'],
            y=equity_df['equity'],
            mode='lines',
            name='Equity',
            line=dict(color='#00bc8c', width=2),
            fill='tozeroy',
            fillcolor='rgba(0,188,140,0.1)',
        ))
        equity_fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=20, t=20, b=40),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            showlegend=False,
        )
    else:
        equity_fig = go.Figure()
        equity_fig.add_annotation(
            text="No equity history available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="gray"),
        )
        equity_fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )

    # Strategy stats
    strategies = get_strategy_stats()

    # Alerts
    alerts = get_recent_alerts(15)
    if alerts:
        alert_items = []
        for a in alerts:
            level_color = {
                'info': 'info',
                'warning': 'warning',
                'error': 'danger',
                'critical': 'danger',
            }.get(a['level'], 'secondary')

            alert_items.append(
                dbc.Alert([
                    html.Strong(f"[{a['type']}] ", className="me-1"),
                    html.Span(a['title']),
                    html.Br(),
                    html.Small(a['timestamp'], className="text-muted"),
                ], color=level_color, className="py-2 mb-1")
            )
        alerts_content = html.Div(alert_items)
    else:
        alerts_content = html.Div("No recent alerts", className="text-muted")

    # System status
    status = get_market_status()
    market_badge = dbc.Badge(
        "OPEN" if status['is_open'] else "CLOSED",
        color="success" if status['is_open'] else "secondary",
        className="me-2"
    )
    status_content = html.Div([
        html.Div([
            html.Strong("Market: "),
            market_badge,
        ], className="mb-2"),
        html.Div([
            html.Strong("Phase: "),
            html.Span(status['phase'].replace('_', ' ').title()),
        ], className="mb-2"),
        html.Div([
            html.Strong("Next Phase: "),
            html.Span(status['time_to_next']),
        ]),
    ])

    # Trade history
    trade_history = get_trade_history(15)

    # Pending orders
    pending_orders = get_pending_orders()

    # Risk metrics
    risk = get_risk_metrics()
    if risk:
        risk_content = html.Div([
            dbc.Row([
                dbc.Col([
                    html.H4(f"{risk.get('exposure_pct', 0):.1f}%", className="text-warning mb-0"),
                    html.Small("Exposure", className="text-muted"),
                ], width=6),
                dbc.Col([
                    html.H4(f"{risk.get('cash_pct', 0):.1f}%", className="text-success mb-0"),
                    html.Small("Cash", className="text-muted"),
                ], width=6),
            ], className="mb-2"),
            dbc.Row([
                dbc.Col([
                    html.H5(f"{risk.get('num_positions', 0)}", className="mb-0"),
                    html.Small("Positions", className="text-muted"),
                ], width=6),
                dbc.Col([
                    html.H5(f"{risk.get('concentration', 0):.1f}%", className="mb-0"),
                    html.Small("Top Position", className="text-muted"),
                ], width=6),
            ], className="mb-2"),
            html.Hr(),
            html.Small([
                html.Strong("Largest: "),
                html.Span(risk.get('largest_position', '-')),
            ], className="text-muted"),
        ])
    else:
        risk_content = html.Div("Unable to calculate risk", className="text-muted")

    # Position price charts
    from plotly.subplots import make_subplots
    price_history = get_position_price_history(10)
    if price_history:
        num_positions = len(price_history)
        cols = min(4, num_positions)
        rows = (num_positions + cols - 1) // cols

        position_fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=list(price_history.keys()),
            vertical_spacing=0.12,
            horizontal_spacing=0.05,
        )

        for i, (symbol, df) in enumerate(price_history.items()):
            row = i // cols + 1
            col = i % cols + 1

            # Calculate color based on performance
            if len(df) >= 2:
                change = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
                color = '#00bc8c' if change >= 0 else '#e74c3c'
            else:
                color = '#888'

            position_fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['close'],
                    mode='lines',
                    line=dict(color=color, width=2),
                    fill='tozeroy',
                    fillcolor=f'rgba{tuple(list(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.1])}',
                    name=symbol,
                    showlegend=False,
                ),
                row=row, col=col
            )

        position_fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=30, r=20, t=30, b=30),
            height=350,
        )
        position_fig.update_xaxes(showticklabels=False, showgrid=False)
        position_fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    else:
        position_fig = go.Figure()
        position_fig.add_annotation(
            text="No position data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="gray"),
        )
        position_fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )

    # Research overview
    research = get_research_overview()
    if research:
        research_content = html.Div([
            dbc.Row([
                dbc.Col([
                    html.H4(f"{research.get('total_ga_runs', 0)}", className="text-info mb-0"),
                    html.Small("GA Runs", className="text-muted"),
                ], width=6),
                dbc.Col([
                    html.H4(f"{research.get('running_ga_runs', 0)}", className="text-warning mb-0"),
                    html.Small("Running", className="text-muted"),
                ], width=6),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.H4(f"{research.get('total_backtests', 0)}", className="mb-0"),
                    html.Small("Backtests", className="text-muted"),
                ], width=6),
                dbc.Col([
                    html.H4(f"{research.get('best_sharpe', 0):.2f}", className="text-success mb-0"),
                    html.Small("Best Sharpe", className="text-muted"),
                ], width=6),
            ], className="mb-3"),
            html.Hr(),
            html.Small([
                html.Strong("Last GA Run: "),
                html.Span(research.get('last_ga_run', 'Never')),
            ], className="text-muted"),
        ])
    else:
        research_content = html.Div("No research data available", className="text-muted")

    # GA runs table
    ga_runs = get_ga_runs(10)

    # Backtest results table
    backtests = get_backtest_results(10)

    # GA fitness chart
    ga_history = get_ga_fitness_history()
    if not ga_history.empty:
        ga_fig = go.Figure()
        # Color palette for strategies
        colors = ['#00bc8c', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
        # Group by strategy and plot each
        for i, strategy in enumerate(ga_history['strategy'].unique()):
            strat_data = ga_history[ga_history['strategy'] == strategy].sort_values('generation')
            color = colors[i % len(colors)]
            ga_fig.add_trace(go.Scatter(
                x=strat_data['generation'],
                y=strat_data['best_fitness'],
                mode='lines+markers',
                name=strategy.replace('_', ' ').title(),
                line=dict(width=3, color=color),
                marker=dict(size=10, color=color),
                hovertemplate='Gen %{x}: %{y:.4f}<extra></extra>',
            ))
        ga_fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=50, r=20, t=30, b=50),
            xaxis=dict(title='Generation', showgrid=True, gridcolor='rgba(255,255,255,0.1)',
                      dtick=1, tickmode='linear'),
            yaxis=dict(title='Best Fitness (Walk-Forward)', showgrid=True,
                      gridcolor='rgba(255,255,255,0.1)', tickformat='.3f'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
            showlegend=True,
            hovermode='x unified',
        )
    else:
        ga_fig = go.Figure()
        ga_fig.add_annotation(
            text="No GA fitness history available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="gray"),
        )
        ga_fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )

    # System errors and warnings
    system_errors = get_system_errors(limit=50)

    # Create summary text
    if system_errors:
        error_count = sum(1 for e in system_errors if e['level'] == 'ERROR')
        warning_count = sum(1 for e in system_errors if e['level'] == 'WARNING')
        critical_count = sum(1 for e in system_errors if e['level'] == 'CRITICAL')

        summary_parts = []
        if critical_count:
            summary_parts.append(f"{critical_count} critical")
        if error_count:
            summary_parts.append(f"{error_count} errors")
        if warning_count:
            summary_parts.append(f"{warning_count} warnings")

        errors_summary = f"Showing {len(system_errors)} recent issues: " + ", ".join(summary_parts) if summary_parts else "No unresolved issues"
    else:
        errors_summary = "No system errors or warnings recorded"

    # Last updated timestamp
    last_updated = f"Last updated: {datetime.now().strftime('%H:%M:%S')}"

    return (
        cb_content,
        card_class,
        account_content,
        positions,
        equity_fig,
        strategies,
        alerts_content,
        status_content,
        trade_history,
        pending_orders,
        risk_content,
        position_fig,
        research_content,
        ga_runs,
        backtests,
        ga_fig,
        system_errors,
        errors_summary,
        last_updated,
    )


@callback(
    [
        Output('system-errors-table', 'data', allow_duplicate=True),
        Output('system-errors-summary', 'children', allow_duplicate=True),
    ],
    Input('clear-errors-btn', 'n_clicks'),
    prevent_initial_call=True,
)
def handle_clear_errors(n_clicks):
    """Handle clear errors button click."""
    if n_clicks:
        count = clear_all_errors()
        logger.info(f"Cleared {count} errors from dashboard")
    return [], f"Showing 0 recent issues: 0 errors, 0 warnings"


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Trading System Dashboard")
    parser.add_argument('--port', type=int, default=5050, help='Port to run on (default: 5050)')
    args = parser.parse_args()

    print("=" * 60)
    print("TRADING SYSTEM DASHBOARD")
    print("=" * 60)
    print(f"Starting server at http://localhost:{args.port}")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    app.run(debug=False, host='0.0.0.0', port=args.port)
