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
from dash import Dash, html, dcc, dash_table, callback, Output, Input, State, no_update
import dash_bootstrap_components as dbc

# Health check endpoint
from observability.dashboard.health_endpoint import register_health_endpoint
# Import trading system components
from execution.alpaca_connector import AlpacaConnector
from execution.signal_tracker import SignalDatabase
from execution.alerts import AlertManager, get_alerts, AlertLevel
from execution.scheduler import MarketHours
from config import STRATEGIES, DATABASES, DIRS
from execution.circuit_breaker import CircuitBreakerManager, CircuitBreakerDB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Dash app with Bootstrap theme
# Note: suppress_callback_exceptions helps handle browser cache mismatches during development
app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
    ],
    title="Trading System Dashboard",
    suppress_callback_exceptions=True,  # Handle cached JS with different callback signatures
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


def get_trading_mode() -> Dict[str, Any]:
    """Get current trading mode (paper/live) and connection status."""
    try:
        broker = get_broker()
        is_paper = broker.paper
        # Test connection by getting account
        account = broker.get_account()
        connected = account is not None
        return {
            'is_paper': is_paper,
            'mode': 'PAPER' if is_paper else 'LIVE',
            'connected': connected,
            'status': 'Connected' if connected else 'Disconnected'
        }
    except Exception as e:
        return {
            'is_paper': True,
            'mode': 'PAPER',
            'connected': False,
            'status': f'Error: {str(e)[:30]}'
        }


def get_signal_db() -> SignalDatabase:
    """Get or create signal database connection."""
    global _signal_db
    if _signal_db is None:
        _signal_db = SignalDatabase()
    return _signal_db


def get_recent_signals(hours: int = 24) -> List[Dict[str, Any]]:
    """Get recent signals from the signal database."""
    try:
        db = get_signal_db()
        conn = db._get_conn()
        cursor = conn.cursor()

        # Get signals from the last N hours
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        cursor.execute("""
            SELECT id, strategy_name, symbol, direction, signal_type,
                   price, stop_loss, take_profit, quantity, confidence,
                   status, created_at, executed_at, execution_route
            FROM signals
            WHERE created_at > ?
            ORDER BY created_at DESC
            LIMIT 100
        """, (cutoff,))

        rows = cursor.fetchall()
        signals = []
        for row in rows:
            signals.append({
                'id': row[0],
                'strategy': row[1],
                'symbol': row[2],
                'direction': row[3],
                'type': row[4],
                'price': row[5],
                'stop_loss': row[6],
                'take_profit': row[7],
                'quantity': row[8],
                'confidence': row[9],
                'status': row[10],
                'created_at': row[11],
                'executed_at': row[12],
                'execution_route': row[13],
            })
        return signals
    except Exception as e:
        logger.error(f"Error fetching signals: {e}")
        return []


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


def get_vix_regime_history() -> pd.DataFrame:
    """Fetch VIX regime history for overlay on charts."""
    try:
        # Try to load VIX data with regimes
        vix_path = PROJECT_ROOT / "data" / "market" / "vix.csv"
        if not vix_path.exists():
            return pd.DataFrame()

        df = pd.read_csv(vix_path)
        if 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        else:
            return pd.DataFrame()

        # Determine regime if not already present
        if 'regime' not in df.columns and 'close' in df.columns:
            df['regime'] = 'normal'
            df.loc[df['close'] < 15, 'regime'] = 'low'
            df.loc[df['close'] > 25, 'regime'] = 'elevated'
            df.loc[df['close'] > 35, 'regime'] = 'extreme'

        return df[['date', 'regime']].drop_duplicates()
    except Exception as e:
        logger.error(f"Error fetching VIX regime history: {e}")
    return pd.DataFrame()


def add_vix_regime_overlay(fig: go.Figure, date_range: tuple, ymin: float, ymax: float):
    """Add VIX regime background shading to a figure."""
    try:
        vix_df = get_vix_regime_history()
        if vix_df.empty:
            return

        # Filter to date range
        start_date, end_date = date_range
        vix_df = vix_df[(vix_df['date'] >= start_date) & (vix_df['date'] <= end_date)]

        if vix_df.empty:
            return

        # Regime colors (subtle backgrounds)
        regime_colors = {
            'low': 'rgba(0, 188, 140, 0.08)',      # Green - low volatility
            'normal': 'rgba(128, 128, 128, 0.0)', # Transparent - normal
            'elevated': 'rgba(255, 193, 7, 0.1)', # Yellow - elevated
            'extreme': 'rgba(231, 76, 60, 0.15)', # Red - extreme
        }

        # Group consecutive same-regime periods
        vix_df = vix_df.sort_values('date')
        vix_df['regime_change'] = vix_df['regime'] != vix_df['regime'].shift()
        vix_df['period'] = vix_df['regime_change'].cumsum()

        for period_id, group in vix_df.groupby('period'):
            regime = group['regime'].iloc[0]
            if regime == 'normal':
                continue  # Skip normal regime

            color = regime_colors.get(regime, 'rgba(128, 128, 128, 0.05)')
            start = group['date'].min()
            end = group['date'].max()

            fig.add_vrect(
                x0=start, x1=end,
                fillcolor=color,
                layer='below',
                line_width=0,
            )
    except Exception as e:
        logger.debug(f"Could not add VIX overlay: {e}")


def get_pnl_summary() -> Dict[str, Any]:
    """Get comprehensive P&L summary."""
    try:
        broker = get_broker()
        account = broker.get_account()
        positions = broker.get_positions()

        # Calculate unrealized P&L from positions
        unrealized_pnl = sum(float(p.unrealized_pnl or 0) for p in positions)

        # Get today's P&L from account (last_equity vs current equity)
        equity = float(account.equity) if account else 0
        last_equity = float(account.last_equity) if account and hasattr(account, 'last_equity') else equity
        today_pnl = equity - last_equity if last_equity else 0
        today_pnl_pct = (today_pnl / last_equity * 100) if last_equity else 0

        # Get historical P&L from database
        db_path = PROJECT_ROOT / "db" / "performance.db"
        week_pnl = 0
        month_pnl = 0
        if db_path.exists():
            conn = sqlite3.connect(db_path)
            try:
                # Last 7 days
                week_df = pd.read_sql_query(
                    "SELECT SUM(daily_pnl) as total FROM portfolio_daily WHERE date >= date('now', '-7 days')",
                    conn
                )
                week_pnl = float(week_df['total'].iloc[0] or 0) if not week_df.empty else 0

                # Last 30 days
                month_df = pd.read_sql_query(
                    "SELECT SUM(daily_pnl) as total FROM portfolio_daily WHERE date >= date('now', '-30 days')",
                    conn
                )
                month_pnl = float(month_df['total'].iloc[0] or 0) if not month_df.empty else 0
            except:
                pass
            conn.close()

        return {
            "equity": equity,
            "unrealized_pnl": unrealized_pnl,
            "today_pnl": today_pnl,
            "today_pnl_pct": today_pnl_pct,
            "week_pnl": week_pnl,
            "month_pnl": month_pnl,
            "position_count": len(positions),
        }
    except Exception as e:
        logger.error(f"Error getting P&L summary: {e}")
    return {"equity": 0, "unrealized_pnl": 0, "today_pnl": 0, "today_pnl_pct": 0,
            "week_pnl": 0, "month_pnl": 0, "position_count": 0}


def get_data_freshness() -> Dict[str, Any]:
    """Get data freshness information."""
    try:
        # Check market data age
        data_path = PROJECT_ROOT / "data" / "historical"
        latest_file = None
        latest_time = None

        if data_path.exists():
            for f in data_path.glob("*.parquet"):
                mtime = f.stat().st_mtime
                if latest_time is None or mtime > latest_time:
                    latest_time = mtime
                    latest_file = f.name

        if latest_time:
            age_seconds = (datetime.now() - datetime.fromtimestamp(latest_time)).total_seconds()
            if age_seconds < 3600:
                age_str = f"{int(age_seconds / 60)}m ago"
            elif age_seconds < 86400:
                age_str = f"{int(age_seconds / 3600)}h ago"
            else:
                age_str = f"{int(age_seconds / 86400)}d ago"
        else:
            age_str = "N/A"

        return {
            "last_update": age_str,
            "latest_file": latest_file or "None",
        }
    except Exception as e:
        logger.error(f"Error getting data freshness: {e}")
    return {"last_update": "N/A", "latest_file": "None"}


def get_strategy_leaderboard() -> List[Dict[str, Any]]:
    """Get strategy leaderboard with rankings and trends."""
    try:
        db_path = PROJECT_ROOT / "db" / "research.db"
        if not db_path.exists():
            return []

        conn = sqlite3.connect(db_path)
        # Get best fitness per strategy from ga_populations
        df = pd.read_sql_query(
            """SELECT strategy,
                      MAX(best_fitness) as best_fitness,
                      COUNT(DISTINCT generation) as total_generations,
                      MAX(created_at) as last_updated
               FROM ga_populations
               GROUP BY strategy
               ORDER BY best_fitness DESC""",
            conn
        )

        # Get recent fitness trend (compare last 2 generations)
        results = []
        for _, row in df.iterrows():
            strategy = row['strategy']
            trend_df = pd.read_sql_query(
                f"""SELECT generation, best_fitness FROM ga_populations
                    WHERE strategy = ? ORDER BY generation DESC LIMIT 2""",
                conn, params=(strategy,)
            )

            if len(trend_df) >= 2:
                recent = trend_df['best_fitness'].iloc[0]
                previous = trend_df['best_fitness'].iloc[1]
                if recent > previous:
                    trend = "↑"
                elif recent < previous:
                    trend = "↓"
                else:
                    trend = "→"
            else:
                trend = "→"

            results.append({
                "strategy": strategy,
                "fitness": row['best_fitness'],
                "generations": row['total_generations'],
                "trend": trend,
                "last_updated": row['last_updated'][:10] if row['last_updated'] else "N/A",
            })

        conn.close()
        return results
    except Exception as e:
        logger.error(f"Error getting strategy leaderboard: {e}")
    return []


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


def get_live_research_progress(run_id: str = None, strategy: str = None) -> Optional[str]:
    """Get current progress from research log (lightweight).

    Args:
        run_id: If provided, only return progress for this specific run
        strategy: Strategy name to filter interleaved log entries
    """
    try:
        import subprocess
        import re
        log_file = PROJECT_ROOT / "logs" / "nightly_research.log"
        if not log_file.exists():
            return None

        # Strategy-specific parameter patterns to identify which run a line belongs to
        strategy_params = {
            'mean_reversion': 'lookback_period|entry_std|exit_std',
            'vol_managed_momentum': 'formation_period|skip_period|vol_lookback',
            'sector_rotation': 'rebalance_freq|top_n',
            'trend_following': 'fast_period|slow_period',
            'vix_regime_rotation': 'low_vix_threshold|high_vix_threshold|extreme_vix',
            'gap_fill': 'min_gap_pct|max_gap_pct|stop_loss_pct',
            'pairs_trading': 'min_correlation|max_half_life|stop_z',
            'relative_volume_breakout': 'min_rv|atr_stop_mult|atr_target_mult',
            'quality_smallcap_value': 'min_roa|min_profit_margin|max_debt_to_equity',
            'factor_momentum': 'formation_period_long|formation_period_med|max_factor_weight',
        }

        # Grep for run markers and progress patterns (new format: [G1 5/5] or old: [5/5])
        result = subprocess.run(
            ["grep", "-E", r"NIGHTLY RESEARCH RUN:|^\d{4}-\d{2}-\d{2}.*\[G?\d+\s*\d*/\d+\]", str(log_file)],
            capture_output=True, text=True, timeout=1
        )

        if result.stdout:
            lines = result.stdout.strip().split('\n')

            if run_id and lines:
                # Find where this run starts
                run_start_idx = -1
                for i, line in enumerate(lines):
                    if f"NIGHTLY RESEARCH RUN: {run_id}" in line:
                        run_start_idx = i
                        break

                if run_start_idx >= 0:
                    # Get all lines after run start
                    run_lines = lines[run_start_idx:]

                    # Filter progress lines - if strategy provided, match its params
                    param_pattern = strategy_params.get(strategy, '') if strategy else ''

                    progress_lines = []
                    for l in run_lines:
                        if re.search(r'\[\d+/\d+\]', l):
                            # If we have a param pattern, only include matching lines
                            if param_pattern:
                                if re.search(param_pattern, l):
                                    progress_lines.append(l)
                            else:
                                progress_lines.append(l)

                    if progress_lines:
                        last = progress_lines[-1]
                        # Match new format: [G1 5/5] WF or old format: [5/5] WF
                        match = re.search(r'\[G(\d+)\s+(\d+/\d+)\]\s*(\w+)', last)
                        if match:
                            return f"G{match.group(1)} {match.group(2)} {match.group(3)}"
                        # Fallback to old format
                        match = re.search(r'\[(\d+/\d+)\]\s*(\w+)', last)
                        if match:
                            return f"{match.group(1)} {match.group(2)}"
            elif lines:
                # No run_id filter - just get last progress line
                progress_lines = [l for l in lines if re.search(r'\[G?\d+\s*\d*/\d+\]', l)]
                if progress_lines:
                    last = progress_lines[-1]
                    # Match new format: [G1 5/5] WF or old format: [5/5] WF
                    match = re.search(r'\[G(\d+)\s+(\d+/\d+)\]\s*(\w+)', last)
                    if match:
                        return f"G{match.group(1)} {match.group(2)} {match.group(3)}"
                    # Fallback to old format
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
                       total_generations, planned_generations, improvements_found, status
                FROM ga_runs
                ORDER BY start_time DESC
                LIMIT {limit}""",
            conn
        )

        if df.empty:
            conn.close()
            return []

        # Format for display
        results = []
        for _, row in df.iterrows():
            # Parse strategies (handle JSON array or comma-separated)
            strategies_raw = row['strategies_evolved'] or ''
            if strategies_raw.startswith('['):
                try:
                    run_strategies = json_module.loads(strategies_raw)
                except:
                    run_strategies = []
            else:
                run_strategies = [s.strip() for s in strategies_raw.split(',') if s.strip()]

            # Get planned and completed generations
            planned_gens = row.get('planned_generations') or 1

            # For running jobs, get progress from ga_populations filtered by run start time
            if row['status'] == 'running':
                if run_strategies:
                    start_time = row['start_time']
                    placeholders = ','.join('?' * len(run_strategies))
                    run_progress = pd.read_sql_query(
                        f"""SELECT strategy, COUNT(DISTINCT generation) as gen_count,
                                   MAX(best_fitness) as best_fitness
                            FROM ga_populations
                            WHERE strategy IN ({placeholders})
                              AND created_at >= ?
                            GROUP BY strategy""",
                        conn,
                        params=run_strategies + [start_time]
                    )
                    # Use max generation across strategies (they run in parallel)
                    completed_gens = int(run_progress['gen_count'].max()) if not run_progress.empty else 0
                    impr = int((run_progress['best_fitness'] > 0).sum()) if not run_progress.empty else 0
                else:
                    completed_gens = 0
                    impr = 0
            else:
                completed_gens = row['total_generations'] or 0
                impr = row['improvements_found'] or 0
                # For completed runs, use total_generations as the display
                planned_gens = completed_gens if completed_gens > 0 else planned_gens

            # Determine display status with resumability indicator
            status = row['status'] or "unknown"
            has_progress = completed_gens > 0

            if status == 'running':
                # Show live progress: Gen X/Y (Z/5 Testing)
                first_strat = run_strategies[0] if run_strategies else None
                live_prog = get_live_research_progress(run_id=row['run_id'], strategy=first_strat)
                # current_gen is what we're working on (1-indexed for display)
                # If completed_gens >= planned_gens, we're done
                if completed_gens >= planned_gens:
                    display_status = f"Gen {planned_gens}/{planned_gens} (Finalizing)"
                else:
                    current_gen = completed_gens + 1
                    if live_prog:
                        display_status = f"Gen {current_gen}/{planned_gens} ({live_prog})"
                    else:
                        display_status = f"Gen {current_gen}/{planned_gens}"
            elif status == 'interrupted':
                display_status = "interrupted (resumable)" if has_progress else "interrupted"
            elif status == 'paused':
                display_status = "paused (will resume)" if has_progress else "paused"
            elif status == 'abandoned':
                display_status = "abandoned (resumable)" if has_progress else "abandoned"
            else:
                display_status = status

            # Determine available action for this row
            if status == 'running':
                action = "⏸ Pause"
            elif status in ('paused', 'interrupted', 'abandoned') and has_progress:
                action = "▶ Resume"
            else:
                action = "-"

            results.append({
                "run_id": row['run_id'][:8] + "...",
                "full_run_id": row['run_id'],  # Store full ID for details lookup
                "start_time": row['start_time'][:16] if row['start_time'] else "",
                "strategies": row['strategies_evolved'][:30] if row['strategies_evolved'] else "-",
                "generations": planned_gens,
                "improvements": impr,
                "status": display_status,
                "action": action,
            })

        conn.close()
        return results
    except Exception as e:
        logger.error(f"Error fetching GA runs: {e}")
    return []


def get_ga_run_details(run_id: str) -> Dict[str, Any]:
    """Fetch detailed information about a specific GA run."""
    try:
        import json as json_module
        db_path = PROJECT_ROOT / "db" / "research.db"
        if not db_path.exists():
            return {}

        conn = sqlite3.connect(db_path)

        # Get run info - try exact match first, then prefix match
        run_df = pd.read_sql_query(
            "SELECT * FROM ga_runs WHERE run_id = ? OR run_id LIKE ?",
            conn, params=(run_id, f"{run_id}%")
        )

        if run_df.empty:
            conn.close()
            return {}

        run_info = run_df.iloc[0].to_dict()

        # Get population info for strategies in this run
        # Handle both JSON array format and comma-separated format
        strategies_raw = run_info.get('strategies_evolved') or ''
        if strategies_raw.startswith('['):
            # JSON array format
            try:
                strategies = json_module.loads(strategies_raw)
            except:
                strategies = []
        else:
            # Comma-separated format
            strategies = [s.strip() for s in strategies_raw.split(',') if s.strip()]

        pop_data = []
        for strat in strategies:
            pop_df = pd.read_sql_query(
                """SELECT strategy, generation, best_fitness, best_genes_json as params, created_at
                   FROM ga_populations
                   WHERE strategy = ?
                   ORDER BY generation DESC
                   LIMIT 5""",
                conn, params=(strat,)
            )
            if not pop_df.empty:
                pop_data.extend(pop_df.to_dict('records'))

        conn.close()

        return {
            "run_id": run_id,
            "start_time": run_info.get('start_time'),
            "end_time": run_info.get('end_time'),
            "status": run_info.get('status'),
            "strategies": strategies,
            "total_generations": run_info.get('total_generations', 0),
            "improvements_found": run_info.get('improvements_found', 0),
            "populations": pop_data,
        }
    except Exception as e:
        logger.error(f"Error fetching GA run details: {e}")
    return {}


def get_best_strategy_for_testing() -> tuple:
    """Pick the best strategy for the next quick test.

    Priority:
    1. Untested strategies (no generations yet)
    2. Strategies with fewest generations (less explored)
    3. Strategies not tested recently
    4. Skip any currently running

    Returns:
        (strategy_name, reason) tuple
    """
    try:
        from config import STRATEGIES

        db_path = PROJECT_ROOT / "db" / "research.db"
        all_strategies = list(STRATEGIES.keys())

        # Get current evolution state
        tested = {}
        if db_path.exists():
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query(
                """SELECT strategy,
                          MAX(generation) as max_gen,
                          MAX(best_fitness) as best_fit,
                          MAX(created_at) as last_updated
                   FROM ga_populations
                   GROUP BY strategy""",
                conn
            )

            # Check for currently running
            running_df = pd.read_sql_query(
                "SELECT strategies_evolved FROM ga_runs WHERE status = 'running'",
                conn
            )
            conn.close()

            running_strategies = set()
            for _, row in running_df.iterrows():
                strats = row['strategies_evolved'] or ''
                if strats.startswith('['):
                    import json
                    running_strategies.update(json.loads(strats))
                else:
                    running_strategies.update(s.strip() for s in strats.split(',') if s.strip())

            for _, row in df.iterrows():
                tested[row['strategy']] = {
                    'generations': row['max_gen'],
                    'fitness': row['best_fit'] or 0,
                    'last_updated': row['last_updated'],
                }
        else:
            running_strategies = set()

        # Find untested strategies
        untested = [s for s in all_strategies if s not in tested and s not in running_strategies]
        if untested:
            return untested[0], "untested strategy"

        # Find strategy with fewest generations (excluding running)
        candidates = [(s, info) for s, info in tested.items() if s not in running_strategies]
        if not candidates:
            # All strategies are running, just pick least explored
            candidates = [(s, info) for s, info in tested.items()]

        if candidates:
            # Sort by generations (ascending), then by last_updated (oldest first)
            candidates.sort(key=lambda x: (x[1]['generations'], x[1]['last_updated'] or ''))
            best = candidates[0]
            return best[0], f"gen {best[1]['generations']}, fitness {best[1]['fitness']:.2f}"

        # Fallback
        return 'mean_reversion', "fallback"

    except Exception as e:
        logger.error(f"Error picking best strategy: {e}")
        return 'mean_reversion', "error fallback"


def get_backtest_results(limit: int = 10) -> List[Dict[str, Any]]:
    """Fetch recent backtest results."""
    try:
        db_path = PROJECT_ROOT / "db" / "research.db"
        if not db_path.exists():
            return []

        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(
            f"""SELECT run_id, strategy, timestamp, sharpe_ratio, sortino_ratio,
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


def get_backtest_details(strategy: str, timestamp: str) -> Dict[str, Any]:
    """Get detailed backtest results for a specific run."""
    try:
        db_path = PROJECT_ROOT / "db" / "research.db"
        if not db_path.exists():
            return {}

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """SELECT * FROM backtests
               WHERE strategy = ? AND timestamp = ?
               LIMIT 1""",
            (strategy, timestamp)
        )
        row = cursor.fetchone()
        if row:
            columns = [desc[0] for desc in cursor.description]
            result = dict(zip(columns, row))
            conn.close()
            return result
        conn.close()
    except Exception as e:
        logger.error(f"Error fetching backtest details: {e}")
    return {}


def get_ga_fitness_history() -> pd.DataFrame:
    """Fetch GA fitness evolution over generations."""
    try:
        db_path = PROJECT_ROOT / "db" / "research.db"
        if not db_path.exists():
            return pd.DataFrame()

        conn = sqlite3.connect(db_path)
        # Use ga_populations for current state, include all generations
        df = pd.read_sql_query(
            """SELECT strategy, generation, best_fitness
               FROM ga_populations
               ORDER BY strategy, generation ASC""",
            conn
        )
        conn.close()
        return df
    except Exception as e:
        logger.error(f"Error fetching GA fitness history: {e}")
    return pd.DataFrame()


def get_best_params_comparison() -> List[Dict[str, Any]]:
    """Get best parameters for each strategy for comparison."""
    try:
        import json as json_module
        db_path = PROJECT_ROOT / "db" / "research.db"
        if not db_path.exists():
            return []

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Get the best performing population for each strategy
        cursor.execute(
            """SELECT strategy, generation, best_fitness, best_genes_json
               FROM ga_populations
               WHERE best_fitness > 0
               GROUP BY strategy
               HAVING MAX(best_fitness)
               ORDER BY best_fitness DESC
               LIMIT 5"""
        )
        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            strategy, generation, fitness, params_str = row
            try:
                params = json_module.loads(params_str) if params_str else {}
            except:
                params = {}
            results.append({
                'strategy': strategy,
                'generation': generation,
                'fitness': fitness,
                'params': params,
            })
        return results
    except Exception as e:
        logger.error(f"Error fetching best params: {e}")
    return []


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
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        broker = get_broker()
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        orders = broker.trading_client.get_orders(request)
        return [
            {
                "symbol": o.symbol,
                "side": str(o.side.value) if hasattr(o.side, 'value') else str(o.side),
                "type": str(o.type.value) if hasattr(o.type, 'value') else str(o.type),
                "qty": float(o.qty),
                "limit": float(o.limit_price) if o.limit_price else "-",
                "stop": float(o.stop_price) if o.stop_price else "-",
                "status": str(o.status.value) if hasattr(o.status, 'value') else str(o.status),
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

        # Get current time in Eastern timezone
        from zoneinfo import ZoneInfo
        et_tz = ZoneInfo('America/New_York')
        now_et = datetime.now(et_tz)

        hour = now_et.hour
        minute = now_et.minute
        weekday = now_et.weekday()
        time_minutes = hour * 60 + minute

        # Market phases in Eastern Time:
        # Overnight: 8:00 PM - 4:00 AM ET (previous day close to pre-market)
        # Pre-market: 4:00 AM - 9:30 AM ET
        # Market hours: 9:30 AM - 4:00 PM ET
        # After hours: 4:00 PM - 8:00 PM ET

        if weekday >= 5:  # Weekend
            phase = "weekend"
            time_to_next = "Monday 4:00 AM ET"
        elif time_minutes < 240:  # Before 4:00 AM ET
            phase = "overnight"
            pre_market_start = 240 - time_minutes
            hours, mins = divmod(pre_market_start, 60)
            time_to_next = f"Pre-market in {hours}h {mins}m"
        elif time_minutes < 570:  # 4:00 AM - 9:30 AM ET
            phase = "pre_market"
            market_open = 570 - time_minutes
            hours, mins = divmod(market_open, 60)
            time_to_next = f"Opens in {hours}h {mins}m"
        elif time_minutes < 960:  # 9:30 AM - 4:00 PM ET
            phase = "market_hours"
            market_close = 960 - time_minutes
            hours, mins = divmod(market_close, 60)
            time_to_next = f"Closes in {hours}h {mins}m"
        elif time_minutes < 1200:  # 4:00 PM - 8:00 PM ET
            phase = "after_hours"
            after_close = 1200 - time_minutes
            hours, mins = divmod(after_close, 60)
            time_to_next = f"Ends in {hours}h {mins}m"
        else:  # After 8:00 PM ET
            phase = "overnight"
            # Minutes until 4 AM next day
            pre_market_start = (24 * 60 - time_minutes) + 240
            hours, mins = divmod(pre_market_start, 60)
            time_to_next = f"Pre-market in {hours}h {mins}m"

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


def get_strategy_pnl_attribution() -> List[Dict[str, Any]]:
    """Get P&L attribution by strategy from positions and historical data."""
    try:
        results = []

        # First try to get from strategy_daily table
        db_path = PROJECT_ROOT / "db" / "performance.db"
        if db_path.exists():
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query(
                """SELECT strategy,
                          SUM(net_pnl) as total_pnl,
                          SUM(trades_closed) as total_trades,
                          AVG(win_rate_30d) as avg_win_rate,
                          MAX(date) as last_active
                   FROM strategy_daily
                   GROUP BY strategy
                   ORDER BY total_pnl DESC""",
                conn
            )
            conn.close()

            if not df.empty:
                for _, row in df.iterrows():
                    results.append({
                        'strategy': row['strategy'],
                        'total_pnl': row['total_pnl'] or 0,
                        'total_trades': int(row['total_trades'] or 0),
                        'win_rate': row['avg_win_rate'] or 0,
                        'last_active': row['last_active'] or 'N/A',
                        'source': 'historical'
                    })

        # Also get current positions grouped by any strategy tag
        try:
            broker = get_broker()
            positions = broker.trading_client.get_all_positions()

            # Group positions - for now just show as "Live Positions"
            if positions:
                live_pnl = sum(float(p.unrealized_pl) for p in positions)
                results.append({
                    'strategy': 'Live Positions',
                    'total_pnl': live_pnl,
                    'total_trades': len(positions),
                    'win_rate': None,
                    'last_active': 'Now',
                    'source': 'live'
                })
        except Exception as e:
            logger.debug(f"Could not fetch live positions for P&L: {e}")

        return results
    except Exception as e:
        logger.error(f"Error getting strategy P&L attribution: {e}")
    return []


def get_system_processes() -> List[Dict[str, Any]]:
    """Check status of key system processes using psutil (works in WSGI context)."""
    import psutil
    from datetime import datetime

    processes = []

    # Define processes to monitor
    process_checks = [
        {
            'name': 'Daily Orchestrator',
            'pattern': 'daily_orchestrator.py',
            'description': 'Main trading orchestrator'
        },
        {
            'name': 'Nightly Research',
            'pattern': 'run_nightly_research.py',
            'description': 'GA optimization engine'
        },
        {
            'name': 'Dashboard',
            'pattern': 'dashboard/app.py',  # Match partial path
            'alt_pattern': 'Python app.py',  # Or when run from dashboard dir
            'description': 'This dashboard'
        },
        {
            'name': 'Intraday Stream',
            'pattern': 'intraday_stream.py',
            'description': 'Real-time market data'
        },
        {
            'name': 'Data Refresh',
            'pattern': 'fetch_',  # Matches fetch_fundamentals, fetch_index_constituents, etc.
            'description': 'Data update scripts'
        },
    ]

    for proc in process_checks:
        try:
            found_pid = None
            found_uptime = None

            # Iterate through all running processes
            for p in psutil.process_iter(['pid', 'cmdline', 'create_time']):
                try:
                    cmdline = p.info.get('cmdline') or []
                    cmdline_str = ' '.join(cmdline)
                    # Check primary pattern and optional alt_pattern
                    matches = proc['pattern'] in cmdline_str
                    if not matches and proc.get('alt_pattern'):
                        matches = proc['alt_pattern'] in cmdline_str
                    if matches:
                        found_pid = str(p.info['pid'])
                        # Calculate uptime
                        create_time = p.info.get('create_time')
                        if create_time:
                            uptime_seconds = datetime.now().timestamp() - create_time
                            hours, remainder = divmod(int(uptime_seconds), 3600)
                            minutes, seconds = divmod(remainder, 60)
                            if hours > 0:
                                found_uptime = f"{hours}:{minutes:02d}:{seconds:02d}"
                            else:
                                found_uptime = f"{minutes:02d}:{seconds:02d}"
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            is_running = found_pid is not None

            processes.append({
                'name': proc['name'],
                'description': proc['description'],
                'is_running': is_running,
                'pid': found_pid,
                'uptime': found_uptime,
                'status': 'running' if is_running else 'stopped'
            })
        except Exception as e:
            logger.error(f"Error checking process {proc['name']}: {e}")
            processes.append({
                'name': proc['name'],
                'description': proc['description'],
                'is_running': False,
                'pid': None,
                'uptime': None,
                'status': f'error: {str(e)[:20]}'
            })

    return processes


def get_orchestrator_status() -> Dict[str, Any]:
    """Get detailed orchestrator status from log file."""
    import re
    from pathlib import Path

    status = {
        'is_running': False,
        'pid': None,
        'current_phase': 'unknown',
        'tasks_total': 0,
        'tasks_succeeded': 0,
        'last_check': None,
        'recent_errors': [],
        'task_details': [],
    }

    # Check if orchestrator is running
    try:
        import psutil
        for p in psutil.process_iter(['pid', 'cmdline']):
            try:
                cmdline = p.info.get('cmdline') or []
                if any('daily_orchestrator.py' in arg for arg in cmdline):
                    status['is_running'] = True
                    status['pid'] = p.info['pid']
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception as e:
        logger.warning(f"Error checking orchestrator process: {e}")

    # Parse orchestrator log for status details
    log_path = Path(__file__).parent.parent.parent / "logs" / "orchestrator.log"
    if not log_path.exists():
        log_path = Path('/tmp/orchestrator.log')  # fallback
    if log_path.exists():
        try:
            # Read last 100 lines of log
            with open(log_path, 'r') as f:
                lines = f.readlines()[-100:]

            # Find most recent phase and task info
            phase_pattern = re.compile(r'Running tasks for phase: (\w+)')
            result_pattern = re.compile(r'Phase (\w+): (\d+)/(\d+) tasks succeeded')
            transition_pattern = re.compile(r'Phase transition: \w+ -> (\w+)')
            weekend_active_pattern = re.compile(r'Weekend phase active - sub-phase: (\w+)')
            sleeping_pattern = re.compile(r'(sleeping until|Sleeping for)')
            task_pattern = re.compile(r'(Starting|Task) (\w+).*?(completed|failed|returned False)')
            error_pattern = re.compile(r'ERROR.*?\|.*?\|(.*)')
            timestamp_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')

            task_results = {}

            for line in reversed(lines):
                # Get timestamp
                ts_match = timestamp_pattern.match(line)
                if ts_match and not status['last_check']:
                    status['last_check'] = ts_match.group(1)

                # Get phase result (most recent)
                result_match = result_pattern.search(line)
                if result_match and status['tasks_total'] == 0:
                    status['current_phase'] = result_match.group(1)
                    status['tasks_succeeded'] = int(result_match.group(2))
                    status['tasks_total'] = int(result_match.group(3))

                # Check for phase transition (weekend/idle states)
                if status['current_phase'] == 'unknown':
                    # Check for active weekend phase first
                    weekend_match = weekend_active_pattern.search(line)
                    if weekend_match:
                        sub_phase = weekend_match.group(1)
                        status['current_phase'] = f'weekend ({sub_phase})'
                    else:
                        transition_match = transition_pattern.search(line)
                        if transition_match:
                            status['current_phase'] = transition_match.group(1)
                        elif sleeping_pattern.search(line):
                            status['current_phase'] = 'weekend (sleeping)'

                # Collect errors (up to 5)
                if 'ERROR' in line and len(status['recent_errors']) < 5:
                    error_match = error_pattern.search(line)
                    if error_match:
                        status['recent_errors'].append(error_match.group(1).strip()[:80])

                # Track individual task results
                if 'Task ' in line and 'completed' in line:
                    task_match = re.search(r'Task (\w+) completed', line)
                    if task_match:
                        task_name = task_match.group(1)
                        if task_name not in task_results:
                            task_results[task_name] = 'success'
                elif 'returned False' in line or 'failed' in line.lower():
                    task_match = re.search(r'Task (\w+)', line)
                    if task_match:
                        task_name = task_match.group(1)
                        if task_name not in task_results:
                            task_results[task_name] = 'failed'

            # Convert task results to list
            status['task_details'] = [
                {'name': name, 'status': result}
                for name, result in task_results.items()
            ][:10]  # Limit to 10 tasks

        except Exception as e:
            logger.warning(f"Error parsing orchestrator log: {e}")

    return status


def get_weekend_status() -> Dict[str, Any]:
    """Get weekend phase status from orchestrator log and coordinator state."""
    import re
    import json
    from pathlib import Path

    status = {
        'is_weekend': False,
        'sub_phase': None,
        'sub_phase_display': 'Not Active',
        'tasks_completed': [],
        'research_status': None,
        'research_progress': {},
        'started_at': None,
        'timeline': [],  # For visual timeline display
    }

    # Check if it's actually the weekend (or Friday after 4pm)
    from datetime import datetime
    import pytz
    now = datetime.now(pytz.timezone('US/Eastern'))
    day = now.weekday()
    hour = now.hour

    # Weekend = Friday after 4 PM through Sunday
    if (day == 4 and hour >= 16) or day == 5 or day == 6:
        status['is_weekend'] = True

    # Try to read coordinator state file first (more detailed progress)
    state_file = DIRS.get('logs', Path('./logs')) / 'weekend_research_state.json'
    if state_file.exists():
        try:
            with open(state_file, 'r') as f:
                coordinator_state = json.load(f)

            # Extract research progress from coordinator
            if coordinator_state.get('phase') in ['optimization', 'discovery', 'adaptive']:
                status['research_status'] = 'running'
                status['research_progress'] = {
                    'phase': coordinator_state.get('phase', ''),
                    'current_strategy': coordinator_state.get('current_strategy', ''),
                    'generation': coordinator_state.get('generation', 0),
                    'total_generations': coordinator_state.get('total_generations', 0),
                    'individual': coordinator_state.get('individual', 0),
                    'population_size': coordinator_state.get('population_size', 0),
                    'best_fitness': coordinator_state.get('best_fitness', 0),
                    'strategies_completed': coordinator_state.get('strategies_completed', []),
                    'strategies_remaining': coordinator_state.get('strategies_remaining', []),
                    'discoveries_found': coordinator_state.get('discoveries_found', 0),
                }
                status['started_at'] = coordinator_state.get('started_at')
            elif coordinator_state.get('phase') == 'complete':
                status['research_status'] = 'complete'
            elif coordinator_state.get('phase') == 'paused':
                status['research_status'] = 'paused'
            elif coordinator_state.get('phase') == 'error':
                status['research_status'] = 'error'
                status['research_progress']['error'] = coordinator_state.get('error', '')

        except Exception as e:
            logger.warning(f"Error reading coordinator state: {e}")

    # Parse orchestrator log for weekend status
    log_path = Path(__file__).parent.parent.parent / "logs" / "orchestrator.log"
    if not log_path.exists():
        log_path = Path('/tmp/orchestrator.log')  # fallback
    if log_path.exists():
        try:
            with open(log_path, 'r') as f:
                lines = f.readlines()[-200:]

            # Patterns to match
            sub_phase_pattern = re.compile(r'Weekend phase active - sub-phase: (\w+)')
            weekend_task_pattern = re.compile(r'Weekend sub-phase transition: \w+ -> (\w+)')
            task_complete_pattern = re.compile(r'Task (\w+) completed')
            research_pattern = re.compile(r'Starting weekend research|Launching weekend research')

            for line in reversed(lines):
                # Get current sub-phase
                if status['sub_phase'] is None:
                    match = sub_phase_pattern.search(line)
                    if match:
                        status['sub_phase'] = match.group(1)

                # Get sub-phase from transition
                if status['sub_phase'] is None:
                    match = weekend_task_pattern.search(line)
                    if match:
                        status['sub_phase'] = match.group(1)

                # Check for research running (backup if coordinator state not available)
                if status['research_status'] is None and research_pattern.search(line):
                    status['research_status'] = 'running'

        except Exception as e:
            logger.warning(f"Error parsing weekend status: {e}")

    # Set display name for sub-phase
    phase_names = {
        'friday_cleanup': 'Friday Cleanup',
        'research': 'Research',
        'data_refresh': 'Data Refresh',
        'preweek_prep': 'Pre-Week Prep',
        'complete': 'Complete',
    }
    if status['sub_phase']:
        status['sub_phase_display'] = phase_names.get(status['sub_phase'], status['sub_phase'].replace('_', ' ').title())

    # Build timeline for visual display
    phases = ['friday_cleanup', 'research', 'data_refresh', 'preweek_prep', 'complete']
    current_idx = phases.index(status['sub_phase']) if status['sub_phase'] in phases else -1

    for i, phase in enumerate(phases):
        timeline_item = {
            'name': phase_names.get(phase, phase),
            'key': phase,
            'status': 'completed' if i < current_idx else ('active' if i == current_idx else 'pending')
        }
        status['timeline'].append(timeline_item)

    return status


def get_shadow_trading_data() -> Dict[str, Any]:
    """Get shadow trading status and metrics."""
    import sqlite3

    data = {
        'strategies': [],
        'total_strategies': 0,
        'active_strategies': 0,
        'total_trades': 0,
        'total_pnl': 0.0,
        'winning_trades': 0,
        'positions': [],
    }

    # Shadow trading data is in the main trades database
    db_path = DIRS.get('db', Path(__file__).parent.parent.parent / 'db') / 'trades.db'
    if not db_path.exists():
        return data

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Check if shadow_strategies table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='shadow_strategies'
        """)
        if not cursor.fetchone():
            conn.close()
            return data

        # Get shadow strategies
        cursor.execute("""
            SELECT name, status, initial_capital, current_capital,
                   total_trades, winning_trades, total_pnl, max_drawdown,
                   start_time
            FROM shadow_strategies
            ORDER BY total_pnl DESC
        """)

        strategies = []
        for row in cursor.fetchall():
            win_rate = (row['winning_trades'] / row['total_trades'] * 100) if row['total_trades'] > 0 else 0
            pnl_pct = ((row['current_capital'] - row['initial_capital']) / row['initial_capital'] * 100) if row['initial_capital'] > 0 else 0

            strategies.append({
                'name': row['name'],
                'status': row['status'],
                'trades': row['total_trades'],
                'winning_trades': row['winning_trades'],
                'win_rate': win_rate,
                'pnl': row['total_pnl'],
                'pnl_pct': pnl_pct,
                'max_drawdown': row['max_drawdown'],
                'start_time': row['start_time'],
            })

            data['total_trades'] += row['total_trades']
            data['total_pnl'] += row['total_pnl']
            data['winning_trades'] += row['winning_trades']
            if row['status'] == 'active':
                data['active_strategies'] += 1

        data['strategies'] = strategies
        data['total_strategies'] = len(strategies)

        # Get open shadow positions
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='shadow_positions'
        """)
        if cursor.fetchone():
            cursor.execute("""
                SELECT strategy, symbol, side, shares, entry_price, current_price
                FROM shadow_positions
                ORDER BY entry_time DESC
            """)
            for row in cursor.fetchall():
                pnl = (row['current_price'] - row['entry_price']) * row['shares']
                pnl_pct = ((row['current_price'] - row['entry_price']) / row['entry_price'] * 100) if row['entry_price'] > 0 else 0
                data['positions'].append({
                    'strategy': row['strategy'],
                    'symbol': row['symbol'],
                    'side': row['side'],
                    'shares': row['shares'],
                    'entry_price': row['entry_price'],
                    'current_price': row['current_price'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                })

        conn.close()
    except Exception as e:
        logger.warning(f"Error getting shadow trading data: {e}")

    return data


def get_system_performance() -> Dict[str, Any]:
    """Get system performance metrics (CPU, memory, temp, disk, swap hierarchy)."""
    import psutil
    from pathlib import Path

    perf = {
        'cpu_percent': 0,
        'memory_percent': 0,
        'memory_used_gb': 0,
        'memory_total_gb': 0,
        'memory_available_gb': 0,
        'disk_percent': 0,
        'disk_used_gb': 0,
        'disk_total_gb': 0,
        'cpu_temp': None,
        'load_avg': None,
        # Swap hierarchy
        'swap_total_gb': 0,
        'swap_used_gb': 0,
        'swap_percent': 0,
        # Zram details
        'zram_total_gb': 0,
        'zram_used_gb': 0,
        'zram_comp_ratio': 0,
        'zram_algorithm': None,
        # NVMe swap details
        'nvme_swap_total_gb': 0,
        'nvme_swap_used_gb': 0,
        # Zswap status
        'zswap_enabled': False,
        # Effective memory calculation
        'effective_memory_gb': 0,
    }

    try:
        # CPU usage (average over 0.1 seconds for responsiveness)
        perf['cpu_percent'] = psutil.cpu_percent(interval=0.1)

        # Memory
        mem = psutil.virtual_memory()
        perf['memory_percent'] = mem.percent
        perf['memory_used_gb'] = mem.used / (1024 ** 3)
        perf['memory_total_gb'] = mem.total / (1024 ** 3)
        perf['memory_available_gb'] = mem.available / (1024 ** 3)

        # Swap (total from psutil)
        swap = psutil.swap_memory()
        perf['swap_total_gb'] = swap.total / (1024 ** 3)
        perf['swap_used_gb'] = swap.used / (1024 ** 3)
        perf['swap_percent'] = swap.percent if swap.total > 0 else 0

        # Zram details from /sys
        try:
            zram_disksize = Path('/sys/block/zram0/disksize')
            if zram_disksize.exists():
                perf['zram_total_gb'] = int(zram_disksize.read_text().strip()) / (1024 ** 3)

            zram_stat = Path('/sys/block/zram0/mm_stat')
            if zram_stat.exists():
                # mm_stat format: orig_data compr_data mem_used mem_limit max_used same_pages pages_compacted huge_pages
                stats = zram_stat.read_text().strip().split()
                if len(stats) >= 3:
                    orig_data = int(stats[0])  # Original data size
                    compr_data = int(stats[1])  # Compressed data size
                    mem_used = int(stats[2])   # Memory used by zram
                    perf['zram_used_gb'] = mem_used / (1024 ** 3)
                    if compr_data > 0:
                        perf['zram_comp_ratio'] = orig_data / compr_data

            zram_algo = Path('/sys/block/zram0/comp_algorithm')
            if zram_algo.exists():
                algo_text = zram_algo.read_text().strip()
                # Algorithm is shown as: lzo lzo-rle lz4 lz4hc [zstd] - bracketed one is active
                import re
                match = re.search(r'\[(\w+)\]', algo_text)
                perf['zram_algorithm'] = match.group(1) if match else algo_text.split()[0]
        except Exception:
            pass

        # NVMe swap (file-based swap)
        try:
            # Parse /proc/swaps to find file-based swap
            swaps_file = Path('/proc/swaps')
            if swaps_file.exists():
                for line in swaps_file.read_text().strip().split('\n')[1:]:  # Skip header
                    parts = line.split()
                    if len(parts) >= 4 and parts[1] == 'file':
                        size_kb = int(parts[2])
                        used_kb = int(parts[3])
                        perf['nvme_swap_total_gb'] = size_kb / (1024 ** 2)
                        perf['nvme_swap_used_gb'] = used_kb / (1024 ** 2)
        except Exception:
            pass

        # Zswap status
        try:
            zswap_enabled = Path('/sys/module/zswap/parameters/enabled')
            if zswap_enabled.exists():
                perf['zswap_enabled'] = zswap_enabled.read_text().strip().upper() == 'Y'
        except Exception:
            pass

        # Calculate effective memory (RAM + compressed swap equivalent)
        # Zram with ~2:1 compression gives ~2x its size in effective memory
        zram_effective = perf['zram_total_gb'] * max(perf['zram_comp_ratio'], 1.5) if perf['zram_total_gb'] > 0 else 0
        perf['effective_memory_gb'] = perf['memory_total_gb'] + zram_effective + perf['nvme_swap_total_gb']

        # Disk (check the trading system path)
        disk = psutil.disk_usage('/')
        perf['disk_percent'] = disk.percent
        perf['disk_used_gb'] = disk.used / (1024 ** 3)
        perf['disk_total_gb'] = disk.total / (1024 ** 3)

        # Load average (Unix only)
        try:
            load = psutil.getloadavg()
            perf['load_avg'] = load[0]  # 1-minute average
        except (AttributeError, OSError):
            pass

        # CPU temperature (works on Pi, may not on Mac)
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # Look for CPU temp (different keys on different systems)
                for name in ['coretemp', 'cpu_thermal', 'cpu-thermal', 'soc_thermal']:
                    if name in temps and temps[name]:
                        perf['cpu_temp'] = temps[name][0].current
                        break
        except (AttributeError, KeyError):
            pass

    except Exception as e:
        logger.error(f"Error getting system performance: {e}")

    return perf


def get_recent_errors_summary() -> Dict[str, Any]:
    """Get a summary of recent errors for quick display."""
    try:
        db_path = PROJECT_ROOT / "db" / "performance.db"
        if not db_path.exists():
            return {'total': 0, 'unresolved': 0, 'by_level': {}}

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='error_log'")
        if not cursor.fetchone():
            conn.close()
            return {'total': 0, 'unresolved': 0, 'by_level': {}}

        # Get counts
        cursor.execute("SELECT COUNT(*) FROM error_log")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM error_log WHERE is_resolved = 0")
        unresolved = cursor.fetchone()[0]

        cursor.execute("""
            SELECT level, COUNT(*) as cnt
            FROM error_log
            WHERE is_resolved = 0
            GROUP BY level
        """)
        by_level = {row[0]: row[1] for row in cursor.fetchall()}

        # Get most recent error
        cursor.execute("""
            SELECT timestamp, level, message
            FROM error_log
            WHERE is_resolved = 0
            ORDER BY timestamp DESC LIMIT 1
        """)
        recent = cursor.fetchone()

        conn.close()

        return {
            'total': total,
            'unresolved': unresolved,
            'by_level': by_level,
            'most_recent': {
                'timestamp': recent[0][:19] if recent else None,
                'level': recent[1] if recent else None,
                'message': recent[2][:100] if recent else None
            } if recent else None
        }
    except Exception as e:
        logger.error(f"Error getting error summary: {e}")
    return {'total': 0, 'unresolved': 0, 'by_level': {}}


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


def create_quick_status_bar():
    """Create compact status bar with market status, circuit breaker, P&L, orchestrator, and system performance."""
    return dbc.Row([
        # Quick P&L
        dbc.Col([
            html.Div([
                html.Div("P&L", className="status-panel-label"),
                html.Div(id="pnl-display", className="small")
            ], className="status-panel status-panel-pnl", id="pnl-panel")
        ], width=3, className="px-1"),

        # Market Status
        dbc.Col([
            html.Div([
                html.Div("Market", className="status-panel-label"),
                html.Div(id="market-status-display")
            ], className="status-panel status-panel-market")
        ], width=2, className="px-1"),

        # Circuit Breaker
        dbc.Col([
            html.Div([
                html.Div("Circuit Breaker", className="status-panel-label"),
                html.Div(id="circuit-breaker-status")
            ], className="status-panel status-panel-cb", id="circuit-breaker-card")
        ], width=2, className="px-1"),

        # Orchestrator Status (compact)
        dbc.Col([
            html.Div([
                html.Div("Orchestrator", className="status-panel-label"),
                html.Div(id="orchestrator-compact", className="small")
            ], className="status-panel status-panel-orch", id="orchestrator-panel")
        ], width=3, className="px-1"),

        # System Performance (compact)
        dbc.Col([
            html.Div([
                html.Div("System", className="status-panel-label"),
                html.Div(id="system-perf-compact", className="small")
            ], className="status-panel status-panel-perf")
        ], width=2, className="px-1"),
    ], className="mb-3 py-2 px-2 bg-dark rounded")


def create_system_performance_card():
    """Create detailed system performance card for System Health section."""
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-microchip me-2"),
            "System Performance"
        ]),
        dbc.CardBody(id="system-perf-detailed")
    ], className="h-100")


def create_circuit_breaker_card():
    """Legacy - kept for compatibility but replaced by create_pnl_market_status_bar."""
    return html.Div()  # Empty placeholder


def create_research_overview_card():
    """Create research overview summary card."""
    return dbc.Card([
        dbc.CardHeader(html.H5("Research Overview", className="mb-0")),
        dbc.CardBody(id="research-overview")
    ], className="mb-3")


def create_shadow_trading_card():
    """Create shadow trading card showing paper trading strategies."""
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="fas fa-ghost me-2 text-info"),
                "Shadow Trading"
            ], className="mb-0 d-inline"),
            html.Span(id="shadow-badge", className="ms-2"),
        ]),
        dbc.CardBody(id="shadow-trading-content")
    ], className="mb-3 h-100")


def create_strategy_leaderboard():
    """Create strategy leaderboard card with rankings and trends."""
    return dbc.Card([
        dbc.CardHeader(html.H5([
            html.I(className="fas fa-trophy me-2 text-warning"),
            "Strategy Leaderboard"
        ], className="mb-0")),
        dbc.CardBody([
            dash_table.DataTable(
                id='strategy-leaderboard-table',
                columns=[
                    {"name": "Rank", "id": "rank"},
                    {"name": "Strategy", "id": "strategy"},
                    {"name": "Fitness", "id": "fitness", "type": "numeric", "format": {"specifier": ".4f"}},
                    {"name": "Gens", "id": "generations", "type": "numeric"},
                    {"name": "Trend", "id": "trend"},
                    {"name": "Updated", "id": "last_updated"},
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
                    {'if': {'filter_query': '{rank} = 1'}, 'backgroundColor': 'rgba(255,215,0,0.2)'},  # Gold
                    {'if': {'filter_query': '{rank} = 2'}, 'backgroundColor': 'rgba(192,192,192,0.2)'},  # Silver
                    {'if': {'filter_query': '{rank} = 3'}, 'backgroundColor': 'rgba(205,127,50,0.2)'},  # Bronze
                    {'if': {'filter_query': '{trend} = "↑"'}, 'color': '#00ff00'},
                    {'if': {'filter_query': '{trend} = "↓"'}, 'color': '#ff4444'},
                ],
                page_size=10,
            ),
        ])
    ], className="mb-3")


def create_ga_runs_table():
    """Create GA optimization runs table with per-row control buttons."""
    return dbc.Card([
        dbc.CardHeader([
            html.H5("GA Optimization Runs", className="mb-0"),
        ]),
        dbc.CardBody([
            dash_table.DataTable(
                id='ga-runs-table',
                columns=[
                    {"name": "Strategy", "id": "strategies"},
                    {"name": "Run ID", "id": "run_id"},
                    {"name": "Started", "id": "start_time"},
                    {"name": "Gens", "id": "generations", "type": "numeric"},
                    {"name": "Impr", "id": "improvements", "type": "numeric"},
                    {"name": "Status", "id": "status"},
                    {"name": "Action", "id": "action"},  # Per-row action button
                    {"name": "", "id": "full_run_id", "hidden": True},  # Hidden column for lookup
                ],
                filter_action="native",
                sort_action="native",
                style_table={'overflowX': 'auto'},
                style_cell={
                    'backgroundColor': '#303030',
                    'color': 'white',
                    'textAlign': 'left',
                    'padding': '8px',
                    'fontSize': '13px',
                    'cursor': 'pointer',
                    'border': '1px solid #444',
                },
                style_header={
                    'backgroundColor': '#444',
                    'fontWeight': 'bold',
                    'textAlign': 'center',
                },
                style_cell_conditional=[
                    {'if': {'column_id': 'action'}, 'textAlign': 'center', 'fontWeight': 'bold'},
                ],
                style_data_conditional=[
                    {
                        'if': {'filter_query': '{status} contains "Gen"'},
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
                    # Yellow highlight for resumable runs
                    {
                        'if': {'filter_query': '{status} contains "resumable"'},
                        'backgroundColor': 'rgba(255,193,7,0.3)',
                        'color': '#FFD700',
                    },
                    # Yellow highlight for paused runs
                    {
                        'if': {'filter_query': '{status} contains "paused"'},
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
                    # Action column styling - green for Resume, yellow for Pause
                    {
                        'if': {'filter_query': '{action} = "▶ Resume"', 'column_id': 'action'},
                        'color': '#00bc8c',
                    },
                    {
                        'if': {'filter_query': '{action} = "⏸ Pause"', 'column_id': 'action'},
                        'color': '#f39c12',
                    },
                    # Remove active cell highlight - highlight whole row instead
                    {
                        'if': {'state': 'active'},
                        'backgroundColor': 'rgba(100,100,100,0.3)',
                        'border': '1px solid #444',
                    },
                ],
                css=[
                    # Fix filter input text visibility
                    {'selector': 'input.dash-filter--case--sensitive', 'rule': 'color: white !important; background-color: #444 !important;'},
                    {'selector': '.dash-filter', 'rule': 'color: white !important; background-color: #444 !important;'},
                    {'selector': 'input[placeholder="filter data..."]', 'rule': 'color: white !important; background-color: #444 !important;'},
                    # Remove the focus outline/border on active cells
                    {'selector': 'td.cell--selected', 'rule': 'background-color: inherit !important; border: 1px solid #444 !important;'},
                    {'selector': 'td.focused', 'rule': 'background-color: inherit !important; border: 1px solid #444 !important;'},
                ],
                page_size=10,
            ),
            html.Small("Click Action column to Resume/Pause a run, or click row for details", className="text-muted mt-1"),
            html.Div(id="ga-control-status", className="mt-2"),
            dcc.Store(id='ga-runs-store'),  # Store full data for lookup
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
                filter_action="native",
                sort_action="native",
                style_table={'overflowX': 'auto'},
                style_cell={
                    'backgroundColor': '#303030',
                    'color': 'white',
                    'textAlign': 'right',
                    'padding': '8px',
                    'fontSize': '13px',
                    'border': '1px solid #444',
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
                    # Remove active cell highlight
                    {
                        'if': {'state': 'active'},
                        'backgroundColor': 'inherit',
                        'border': '1px solid #444',
                    },
                ],
                css=[
                    # Fix filter input text visibility
                    {'selector': 'input.dash-filter--case--sensitive', 'rule': 'color: white !important; background-color: #444 !important;'},
                    {'selector': '.dash-filter', 'rule': 'color: white !important; background-color: #444 !important;'},
                    {'selector': 'input[placeholder="filter data..."]', 'rule': 'color: white !important; background-color: #444 !important;'},
                    {'selector': 'td.cell--selected', 'rule': 'background-color: inherit !important; border: 1px solid #444 !important;'},
                    {'selector': 'td.focused', 'rule': 'background-color: inherit !important; border: 1px solid #444 !important;'},
                ],
                page_size=10,
            ),
            html.Small("Click a row to view details", className="text-muted mt-1"),
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


def create_param_comparison_card():
    """Create parameter comparison card showing best params per strategy."""
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="fas fa-sliders-h me-2"),
                "Best Parameters by Strategy",
            ], className="mb-0"),
        ]),
        dbc.CardBody(id="param-comparison-display", style={'maxHeight': '300px', 'overflowY': 'auto'})
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


def create_strategy_pnl_card():
    """Create strategy P&L attribution card."""
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="fas fa-chart-pie me-2 text-info"),
                "Strategy P&L Attribution"
            ], className="mb-0"),
        ]),
        dbc.CardBody([
            html.Div(id='strategy-pnl-content')
        ])
    ], className="mb-3")


def create_system_monitor_card():
    """Create system process monitor card."""
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="fas fa-server me-2 text-primary"),
                "System Processes"
            ], className="mb-0"),
        ]),
        dbc.CardBody([
            html.Div(id='system-monitor-content')
        ])
    ], className="mb-3")


def create_orchestrator_status_card():
    """Create orchestrator status card showing phase and task status."""
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="fas fa-tasks me-2 text-warning"),
                "Orchestrator Status"
            ], className="mb-0"),
        ]),
        dbc.CardBody([
            html.Div(id='orchestrator-status-content')
        ])
    ], className="mb-3")


def create_weekend_status_card():
    """Create weekend phase status card with timeline and control panel."""
    # Get available strategies for multi-select
    from config import STRATEGIES
    strategy_options = [
        {"label": s.replace("_", " ").title(), "value": s}
        for s in sorted(STRATEGIES.keys())
    ]

    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="fas fa-flask me-2 text-info"),
                "Weekend Research Phase"
            ], className="mb-0 d-inline"),
            dbc.Button(
                html.I(className="fas fa-cog"),
                id="weekend-config-toggle",
                color="link",
                size="sm",
                className="float-end text-info p-0",
            ),
        ]),
        dbc.CardBody([
            # Status display
            html.Div(id='weekend-status-content', className="mb-3"),

            # Configuration panel (collapsible)
            dbc.Collapse([
                html.Hr(),
                html.H6("Research Configuration", className="text-muted mb-3"),

                # Generations slider
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Generations", className="small"),
                        dcc.Slider(
                            id='weekend-generations',
                            min=3, max=50, step=1, value=10,
                            marks={3: '3', 10: '10', 25: '25', 50: '50'},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ], width=6),
                    dbc.Col([
                        dbc.Label("Population", className="small"),
                        dcc.Slider(
                            id='weekend-population',
                            min=10, max=100, step=5, value=30,
                            marks={10: '10', 30: '30', 50: '50', 100: '100'},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ], width=6),
                ], className="mb-3"),

                # Toggles
                dbc.Row([
                    dbc.Col([
                        dbc.Checklist(
                            id='weekend-discovery-toggle',
                            options=[{"label": " Discovery", "value": "discovery"}],
                            value=["discovery"],
                            switch=True,
                            inline=True,
                            className="small",
                        ),
                    ], width=4),
                    dbc.Col([
                        dbc.Checklist(
                            id='weekend-adaptive-toggle',
                            options=[{"label": " Adaptive GA", "value": "adaptive"}],
                            value=["adaptive"],
                            switch=True,
                            inline=True,
                            className="small",
                        ),
                    ], width=4),
                    dbc.Col([
                        dbc.Label("Discovery Hours", className="small"),
                        dbc.Input(
                            id='weekend-discovery-hours',
                            type="number", min=1, max=8, value=4, size="sm",
                        ),
                    ], width=4),
                ], className="mb-3"),

                # Strategy selection
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Strategies (empty = all)", className="small"),
                        dcc.Dropdown(
                            id='weekend-strategies',
                            options=strategy_options,
                            value=[],
                            multi=True,
                            placeholder="All evolvable strategies",
                            style={"fontSize": "0.8rem"},
                        ),
                    ], width=12),
                ], className="mb-3"),

                # Presets
                html.Div([
                    html.Label("Presets:", className="small text-muted me-2"),
                    dbc.ButtonGroup([
                        dbc.Button("Quick", id="weekend-preset-quick", color="secondary", size="sm", outline=True),
                        dbc.Button("Standard", id="weekend-preset-standard", color="info", size="sm", outline=True),
                        dbc.Button("Deep", id="weekend-preset-deep", color="primary", size="sm", outline=True),
                    ], size="sm"),
                ], className="mb-3"),

                html.Hr(),
            ], id="weekend-config-collapse", is_open=False),

            # Action buttons (always visible)
            dbc.ButtonGroup([
                dbc.Button([html.I(className="fas fa-play me-1"), "Start"],
                          id="btn-weekend-start", color="success", size="sm", outline=True),
                dbc.Button([html.I(className="fas fa-pause me-1"), "Pause"],
                          id="btn-weekend-pause", color="warning", size="sm", outline=True),
                dbc.Button([html.I(className="fas fa-forward me-1"), "Skip"],
                          id="btn-weekend-skip", color="info", size="sm", outline=True),
                dbc.Button([html.I(className="fas fa-stop me-1"), "Stop"],
                          id="btn-weekend-stop", color="danger", size="sm", outline=True),
            ], className="w-100"),
            html.Div(id="weekend-action-status", className="mt-2 small text-muted"),
        ])
    ], className="mb-3", id="weekend-status-card")


def create_position_charts():
    """Create position price charts."""
    return dbc.Card([
        dbc.CardHeader(html.H5("Position Price Charts (10 Day)", className="mb-0")),
        dbc.CardBody([
            dcc.Graph(id='position-charts', style={'height': '350px'})
        ])
    ], className="mb-3")


def create_signals_card():
    """Create trading signals card with recent signals table."""
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="fas fa-broadcast-tower me-2 text-info"),
                "Trading Signals"
            ], className="mb-0 d-inline"),
            html.Span(id="signals-count-badge", className="ms-2"),
        ]),
        dbc.CardBody([
            # Signal summary row
            html.Div(id="signals-summary", className="mb-3"),
            # Signals table
            dash_table.DataTable(
                id='signals-table',
                columns=[
                    {"name": "Time", "id": "time"},
                    {"name": "Strategy", "id": "strategy"},
                    {"name": "Symbol", "id": "symbol"},
                    {"name": "Direction", "id": "direction"},
                    {"name": "Type", "id": "type"},
                    {"name": "Price", "id": "price"},
                    {"name": "Confidence", "id": "confidence"},
                    {"name": "Status", "id": "status"},
                ],
                data=[],
                style_table={'overflowX': 'auto'},
                style_cell={
                    'backgroundColor': '#303030',
                    'color': 'white',
                    'border': '1px solid #444',
                    'textAlign': 'left',
                    'padding': '8px',
                    'fontSize': '12px',
                },
                style_header={
                    'backgroundColor': '#1a1a1a',
                    'fontWeight': 'bold',
                    'border': '1px solid #444',
                },
                style_data_conditional=[
                    # Status coloring
                    {'if': {'filter_query': '{status} = "pending"'},
                     'backgroundColor': '#1a3a5c', 'color': '#6cb2eb'},
                    {'if': {'filter_query': '{status} = "submitted"'},
                     'backgroundColor': '#3d3a1a', 'color': '#f6e05e'},
                    {'if': {'filter_query': '{status} = "executed"'},
                     'backgroundColor': '#1a3d1a', 'color': '#68d391'},
                    {'if': {'filter_query': '{status} = "expired"'},
                     'backgroundColor': '#3d1a1a', 'color': '#fc8181'},
                    {'if': {'filter_query': '{status} = "rejected"'},
                     'backgroundColor': '#3d1a1a', 'color': '#fc8181'},
                    # Direction coloring
                    {'if': {'filter_query': '{direction} = "long"', 'column_id': 'direction'},
                     'color': '#68d391'},
                    {'if': {'filter_query': '{direction} = "short"', 'column_id': 'direction'},
                     'color': '#fc8181'},
                ],
                page_size=10,
                sort_action='native',
            ),
        ])
    ], className="mb-3")


# ============================================================================
# Main Layout
# ============================================================================

app.layout = dbc.Container([
    # Trading Mode Banner - prominent indicator
    html.Div(id="trading-mode-banner", className="text-center py-2 mb-2"),

    # Header
    dbc.Row([
        dbc.Col([
            html.H2("Trading System Dashboard", className="text-primary mb-0"),
            html.Small(id="last-updated", className="text-muted"),
        ], width=6),
        dbc.Col([
            dbc.ButtonGroup([
                dbc.Button([
                    html.I(className="fas fa-sync-alt me-1", id="auto-refresh-icon"),
                    "Auto"
                ], id="btn-auto-refresh", color="info", size="sm", outline=True, className="me-1"),
                dbc.Button([
                    html.I(className="fas fa-redo me-1"),
                    "Refresh"
                ], id="refresh-btn", color="primary", size="sm"),
            ], className="float-end"),
            dbc.Button([
                html.I(className="fas fa-cog me-1"),
                "Quick Test Config"
            ], id="btn-quick-test-config", color="secondary", size="sm", outline=True,
               className="float-end me-2"),
        ], width=6),
    ], className="mb-3 mt-3"),

    # I/O Control Buttons - Desktop view (hidden on mobile)
    dbc.Row([
        dbc.Col([
            dbc.ButtonGroup([
                dbc.Button([html.I(className="fas fa-robot me-1"), "Start Orchestrator"],
                          id="btn-start-orchestrator", color="success", size="sm", outline=True),
                dbc.Button([html.I(className="fas fa-pause me-1"), "Stop Orchestrator"],
                          id="btn-stop-orchestrator", color="warning", size="sm", outline=True),
            ], className="me-2"),
            dbc.ButtonGroup([
                dbc.Button([html.I(className="fas fa-play me-1"), "Quick Test"],
                          id="btn-quick-test", color="success", size="sm", outline=True),
                dbc.Button([html.I(className="fas fa-stop me-1"), "Stop Research"],
                          id="btn-stop-research", color="warning", size="sm", outline=True),
                dbc.Button([html.I(className="fas fa-scroll me-1"), "View Logs"],
                          id="btn-view-logs", color="info", size="sm", outline=True),
                dbc.Button([html.I(className="fas fa-database me-1"), "Cleanup DBs"],
                          id="btn-cleanup-dbs", color="secondary", size="sm", outline=True),
            ], className="me-3"),
            html.Span(id="io-status", className="ms-3 text-muted small"),
        ], width=9),
        dbc.Col([
            # Emergency Controls - separate section
            dbc.ButtonGroup([
                dbc.Button([html.I(className="fas fa-skull-crossbones me-1"), "KILL SWITCH"],
                          id="btn-kill-switch", color="danger", size="sm"),
                dbc.Button([html.I(className="fas fa-redo me-1"), "Reset"],
                          id="btn-reset-system", color="warning", size="sm", outline=True),
            ]),
        ], width=3, className="text-end"),
    ], className="mb-3 d-none d-md-flex"),

    # I/O Control Buttons - Mobile view (collapsible menu)
    html.Div([
        dbc.Button(
            [html.I(className="fas fa-bars me-2"), "Actions"],
            id="mobile-actions-toggle",
            color="secondary",
            size="sm",
            className="mb-2 w-100"
        ),
        dbc.Collapse([
            dbc.Card([
                dbc.CardBody([
                    # Orchestrator controls
                    html.Div("Orchestrator", className="text-muted small mb-1"),
                    dbc.ButtonGroup([
                        dbc.Button([html.I(className="fas fa-robot me-1"), "Start"],
                                  id="btn-start-orchestrator-mobile", color="success", size="sm", outline=True),
                        dbc.Button([html.I(className="fas fa-pause me-1"), "Stop"],
                                  id="btn-stop-orchestrator-mobile", color="warning", size="sm", outline=True),
                    ], className="w-100 mb-2"),

                    # Research controls
                    html.Div("Research", className="text-muted small mb-1"),
                    dbc.ButtonGroup([
                        dbc.Button([html.I(className="fas fa-play me-1"), "Quick Test"],
                                  id="btn-quick-test-mobile", color="success", size="sm", outline=True),
                        dbc.Button([html.I(className="fas fa-stop me-1"), "Stop"],
                                  id="btn-stop-research-mobile", color="warning", size="sm", outline=True),
                    ], className="w-100 mb-2"),

                    # Utility controls
                    html.Div("Utilities", className="text-muted small mb-1"),
                    dbc.ButtonGroup([
                        dbc.Button([html.I(className="fas fa-scroll me-1"), "Logs"],
                                  id="btn-view-logs-mobile", color="info", size="sm", outline=True),
                        dbc.Button([html.I(className="fas fa-database me-1"), "Cleanup"],
                                  id="btn-cleanup-dbs-mobile", color="secondary", size="sm", outline=True),
                    ], className="w-100 mb-2"),

                    # Emergency controls
                    html.Hr(className="my-2"),
                    html.Div("Emergency", className="text-danger small mb-1"),
                    dbc.ButtonGroup([
                        dbc.Button([html.I(className="fas fa-skull-crossbones me-1"), "KILL"],
                                  id="btn-kill-switch-mobile", color="danger", size="sm"),
                        dbc.Button([html.I(className="fas fa-redo me-1"), "Reset"],
                                  id="btn-reset-system-mobile", color="warning", size="sm", outline=True),
                    ], className="w-100"),
                ], className="py-2")
            ], className="bg-dark border-secondary")
        ], id="mobile-actions-collapse", is_open=False),
        html.Div(id="io-status-mobile", className="text-muted small mt-1"),
    ], className="mb-3 d-md-none"),

    # Close Positions Confirmation Modal
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Confirm Close All Positions", className="text-danger")),
        dbc.ModalBody([
            html.P([
                html.I(className="fas fa-exclamation-triangle text-warning me-2", style={"fontSize": "24px"}),
                "This will close ALL open positions immediately at market price.",
            ], className="mb-3"),
            html.P("Are you sure you want to proceed?", className="text-muted"),
            html.Hr(),
            html.P("Type 'CONFIRM' to proceed:", className="mb-2"),
            dbc.Input(id="close-positions-confirm-input", type="text", placeholder="Type CONFIRM"),
        ]),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="close-positions-cancel", color="secondary", n_clicks=0),
            dbc.Button("Close All Positions", id="close-positions-execute", color="danger", n_clicks=0, disabled=True),
        ]),
    ], id="close-positions-modal", is_open=False),

    # Kill Switch Confirmation Modal
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("KILL SWITCH", className="text-danger")),
        dbc.ModalBody([
            html.P([
                html.I(className="fas fa-skull-crossbones text-danger me-2", style={"fontSize": "24px"}),
                "This will STOP all trading system processes immediately.",
            ], className="mb-3"),
            html.Ul([
                html.Li("Stop all research/optimization"),
                html.Li("Trip circuit breaker"),
                html.Li("Halt order execution"),
            ]),
        ]),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="kill-switch-cancel", color="secondary", n_clicks=0),
            dbc.Button("ENGAGE KILL SWITCH", id="kill-switch-execute", color="danger", n_clicks=0),
        ]),
    ], id="kill-switch-modal", is_open=False),

    # Quick Test Config Modal
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle([
            html.I(className="fas fa-flask me-2"),
            "Quick Test Configuration"
        ])),
        dbc.ModalBody([
            dbc.Form([
                # Mode selection
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Strategy Selection Mode", className="fw-bold"),
                        dbc.RadioItems(
                            id="quick-test-mode",
                            options=[
                                {"label": "Auto (best performing)", "value": "auto"},
                                {"label": "Manual (choose strategy)", "value": "manual"},
                            ],
                            value="auto",
                            inline=True,
                            className="mb-2"
                        ),
                    ]),
                ], className="mb-3"),

                # Manual strategy selection (shown when manual mode)
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Strategy", className="fw-bold"),
                        dbc.Select(
                            id="quick-test-strategy",
                            options=[
                                {"label": "Mean Reversion", "value": "mean_reversion"},
                                {"label": "Momentum", "value": "momentum"},
                                {"label": "Trend Following", "value": "trend_following"},
                                {"label": "Breakout", "value": "breakout"},
                                {"label": "VWAP", "value": "vwap"},
                            ],
                            value="mean_reversion",
                        ),
                    ]),
                ], className="mb-3", id="manual-strategy-row"),

                # Generations
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Generations", className="fw-bold"),
                        dbc.Input(
                            id="quick-test-generations",
                            type="number",
                            min=1,
                            max=50,
                            value=1,
                            className="w-50"
                        ),
                        html.Small("Number of GA generations to run", className="text-muted"),
                    ]),
                ], className="mb-3"),
            ]),
            html.Hr(),
            html.Div(id="quick-test-config-status", className="text-muted small"),
        ]),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="quick-test-config-cancel", color="secondary", n_clicks=0),
            dbc.Button("Save Config", id="quick-test-config-save", color="primary", n_clicks=0),
        ]),
    ], id="quick-test-config-modal", is_open=False),

    # GA Run Details Modal
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("GA Run Details", id="ga-modal-title")),
        dbc.ModalBody(id="ga-modal-body"),
        dbc.ModalFooter(
            dbc.Button("Close", id="ga-modal-close", className="ms-auto", n_clicks=0)
        ),
    ], id="ga-details-modal", size="lg", is_open=False),

    # Backtest Details Modal
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Backtest Details", id="backtest-modal-title")),
        dbc.ModalBody(id="backtest-modal-body"),
        dbc.ModalFooter(
            dbc.Button("Close", id="backtest-modal-close", className="ms-auto", n_clicks=0)
        ),
    ], id="backtest-details-modal", size="lg", is_open=False),

    # Logs Modal with live updates
    dbc.Modal([
        dbc.ModalHeader([
            dbc.ModalTitle("Recent Research Logs"),
            html.Small(" (auto-refreshes every 30s)", className="text-muted ms-2"),
        ]),
        dbc.ModalBody([
            html.Pre(id="logs-content", style={
                'backgroundColor': '#1a1a1a',
                'color': '#00ff00',
                'padding': '15px',
                'borderRadius': '5px',
                'maxHeight': '500px',
                'overflowY': 'auto',
                'fontSize': '12px',
                'fontFamily': 'monospace',
            }),
            dcc.Interval(id='logs-interval', interval=30000, n_intervals=0, disabled=True),
        ]),
        dbc.ModalFooter(
            dbc.Button("Close", id="logs-modal-close", className="ms-auto", n_clicks=0)
        ),
    ], id="logs-modal", size="xl", is_open=False),

    # Quick Status Bar - always visible at top
    create_quick_status_bar(),

    # Portfolio Overview Section (Collapsible)
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H4([
                    html.I(className="fas fa-wallet me-2"),
                    "Portfolio Overview",
                    html.Span(id="portfolio-badge", className="ms-2"),
                ], className="text-success mb-0 d-inline"),
                html.I(className="fas fa-chevron-down text-success", id="portfolio-chevron",
                       style={"fontSize": "1.2rem"}),
            ], id="portfolio-collapse-btn", className="d-flex justify-content-between align-items-center mb-3 mt-3",
               style={"cursor": "pointer"}),
        ], width=12),
    ]),
    dbc.Collapse(
        [
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
            # Emergency Position Control (hidden inside Portfolio)
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-exclamation-triangle text-danger me-2"),
                                html.Strong("Emergency Position Control", className="text-danger"),
                                dbc.Button([html.I(className="fas fa-times-circle me-1"), "Close All Positions"],
                                          id="btn-close-positions", color="danger", size="sm", className="float-end"),
                            ], className="d-flex align-items-center"),
                        ], className="py-2"),
                    ], className="border-danger", style={"borderWidth": "1px"}),
                ], width=12),
            ], className="mt-3"),
        ],
        id="portfolio-collapse",
        is_open=False,
    ),

    # Trading Activity Section (Collapsible)
    dbc.Row([
        dbc.Col([
            html.Hr(className="my-4"),
            html.Div([
                html.H4([
                    html.I(className="fas fa-chart-line me-2"),
                    "Trading Activity",
                    html.Span(id="trading-badge", className="ms-2"),
                ], className="text-warning mb-0 d-inline"),
                html.I(className="fas fa-chevron-down text-warning", id="trading-chevron",
                       style={"fontSize": "1.2rem"}),
            ], id="trading-collapse-btn", className="d-flex justify-content-between align-items-center mb-3",
               style={"cursor": "pointer"}),
        ], width=12),
    ]),
    dbc.Collapse(
        [
            dbc.Row([
                dbc.Col(create_trade_history_table(), width=5),
                dbc.Col(create_pending_orders_table(), width=3),
                dbc.Col(create_risk_metrics_card(), width=4),
            ]),
            dbc.Row([
                dbc.Col(create_position_charts(), width=12),
            ]),
        ],
        id="trading-collapse",
        is_open=False,
    ),

    # Signals Section (Collapsible) - Between Trading Activity and P&L
    dbc.Row([
        dbc.Col([
            html.Hr(className="my-4"),
            html.Div([
                html.H4([
                    html.I(className="fas fa-broadcast-tower me-2"),
                    "Trading Signals",
                    html.Span(id="signals-badge", className="ms-2"),
                ], className="text-info mb-0 d-inline"),
                html.I(className="fas fa-chevron-down text-info", id="signals-chevron",
                       style={"fontSize": "1.2rem"}),
            ], id="signals-collapse-btn", className="d-flex justify-content-between align-items-center mb-3",
               style={"cursor": "pointer"}),
        ], width=12),
    ]),
    dbc.Collapse(
        [
            dbc.Row([
                dbc.Col(create_signals_card(), width=12),
            ]),
        ],
        id="signals-collapse",
        is_open=False,
    ),

    # P&L Section (Collapsible) - Below Trading Activity
    dbc.Row([
        dbc.Col([
            html.Hr(className="my-4"),
            html.Div([
                html.H4([
                    html.I(className="fas fa-dollar-sign me-2"),
                    "P&L & Performance",
                    html.Span(id="pnl-badge", className="ms-2"),
                ], className="mb-0 d-inline", style={"color": "#1a7a52"}),  # Darker green
                html.I(className="fas fa-chevron-down", id="pnl-chevron",
                       style={"fontSize": "1.2rem", "color": "#1a7a52"}),
            ], id="pnl-collapse-btn", className="d-flex justify-content-between align-items-center mb-3",
               style={"cursor": "pointer"}),
        ], width=12),
    ]),
    dbc.Collapse(
        [
            dbc.Row([
                # Detailed P&L Card
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("P&L Summary", className="mb-0")),
                        dbc.CardBody(id="pnl-detailed-display")
                    ])
                ], width=4),
                # Equity Chart
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("Equity Curve", className="mb-0")),
                        dbc.CardBody([
                            dcc.Graph(id='pnl-equity-chart', style={'height': '200px'})
                        ])
                    ])
                ], width=8),
            ]),
        ],
        id="pnl-collapse",
        is_open=False,
    ),

    # Research Section (Collapsible)
    dbc.Row([
        dbc.Col([
            html.Hr(className="my-4"),
            html.Div([
                html.H4([
                    html.I(className="fas fa-flask me-2"),
                    "Research & Optimization",
                    html.Span(id="research-badge", className="ms-2"),
                ], className="text-info mb-0 d-inline"),
                html.I(className="fas fa-chevron-down text-info", id="research-chevron",
                       style={"fontSize": "1.2rem"}),
            ], id="research-collapse-btn", className="d-flex justify-content-between align-items-center mb-3",
               style={"cursor": "pointer"}),
        ], width=12),
    ]),
    dbc.Collapse(
        [
            # Row 1: Overview + Shadow Trading + Leaderboard
            dbc.Row([
                dbc.Col(create_research_overview_card(), width=3),
                dbc.Col(create_shadow_trading_card(), width=3),
                dbc.Col(create_strategy_leaderboard(), width=3),
                dbc.Col(create_ga_fitness_chart(), width=3),
            ], className="mb-3"),
            # Row 2: GA Runs + Backtest Results
            dbc.Row([
                dbc.Col(create_ga_runs_table(), width=5),
                dbc.Col(create_backtest_results_table(), width=7),
            ]),
            # Row 3: Param Comparison
            dbc.Row([
                dbc.Col(create_param_comparison_card(), width=12),
            ]),
        ],
        id="research-collapse",
        is_open=False,
    ),

    # System Health Section (Collapsible)
    dbc.Row([
        dbc.Col([
            html.Hr(className="my-4"),
            html.Div([
                html.H4([
                    html.I(className="fas fa-heartbeat me-2"),
                    "System Health",
                    html.Span(id="health-badge", className="ms-2"),
                ], className="text-danger mb-0 d-inline"),
                html.I(className="fas fa-chevron-down text-danger", id="health-chevron",
                       style={"fontSize": "1.2rem"}),
            ], id="health-collapse-btn", className="d-flex justify-content-between align-items-center mb-3",
               style={"cursor": "pointer"}),
        ], width=12),
    ]),
    dbc.Collapse(
        [
            # System Health Row 1: Process Monitor + Orchestrator + Weekend + System Performance
            dbc.Row([
                dbc.Col(create_system_monitor_card(), width=3),
                dbc.Col(create_orchestrator_status_card(), width=3),
                dbc.Col(create_weekend_status_card(), width=3),
                dbc.Col(create_system_performance_card(), width=3),
            ], className="mb-3"),
            # System Health Row 2: Strategy P&L
            dbc.Row([
                dbc.Col(create_strategy_pnl_card(), width=12),
            ], className="mb-3"),
            # System Health Row 3: Error Log
            dbc.Row([
                dbc.Col(create_system_errors_card(), width=12),
            ]),
        ],
        id="health-collapse",
        is_open=False,
    ),

    # Auto-refresh interval (30 seconds)
    dcc.Interval(id='interval-component', interval=30*1000, n_intervals=0),
    # Startup interval - triggers once after 1.5s to allow callbacks to register
    dcc.Interval(id='startup-interval', interval=1500, n_intervals=0, max_intervals=1),

], fluid=True, className="bg-dark")


# ============================================================================
# Callbacks
# ============================================================================

@callback(
    [
        Output('trading-mode-banner', 'children'),
        Output('trading-mode-banner', 'className'),
        Output('pnl-display', 'children'),
        Output('pnl-detailed-display', 'children'),
        Output('pnl-equity-chart', 'figure'),
        Output('market-status-display', 'children'),
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
        Output('shadow-trading-content', 'children'),
        Output('shadow-badge', 'children'),
        Output('ga-runs-table', 'data'),
        Output('strategy-leaderboard-table', 'data'),
        Output('backtest-table', 'data'),
        Output('ga-fitness-chart', 'figure'),
        Output('param-comparison-display', 'children'),
        Output('system-errors-table', 'data'),
        Output('system-errors-summary', 'children'),
        Output('strategy-pnl-content', 'children'),
        Output('system-monitor-content', 'children'),
        Output('orchestrator-status-content', 'children'),
        Output('orchestrator-compact', 'children'),
        Output('orchestrator-panel', 'className'),
        Output('system-perf-compact', 'children'),
        Output('system-perf-detailed', 'children'),
        Output('signals-table', 'data'),
        Output('signals-summary', 'children'),
        Output('signals-badge', 'children'),
        Output('last-updated', 'children'),
        Output('weekend-status-content', 'children'),
    ],
    [
        Input('interval-component', 'n_intervals'),
        Input('refresh-btn', 'n_clicks'),
        Input('startup-interval', 'n_intervals'),
    ],
    prevent_initial_call=True,  # Prevent race condition on initial load
)
def update_dashboard(n_intervals, n_clicks, startup_intervals):
    """Update all dashboard components. First update triggered by interval after page load."""

    # Trading Mode Banner
    mode_info = get_trading_mode()
    if mode_info['is_paper']:
        mode_banner = html.Div([
            html.I(className="fas fa-flask me-2"),
            html.Strong("PAPER TRADING"),
            html.Span(" - Simulated trades only", className="ms-2 small"),
            html.Span(f" | API: {mode_info['status']}", className="ms-2 small text-muted"),
        ])
        mode_class = "text-center py-2 mb-2 bg-info text-dark fw-bold rounded"
    else:
        mode_banner = html.Div([
            html.I(className="fas fa-exclamation-triangle me-2"),
            html.Strong("LIVE TRADING"),
            html.Span(" - Real money at risk!", className="ms-2 small"),
            html.Span(f" | API: {mode_info['status']}", className="ms-2 small"),
        ])
        mode_class = "text-center py-2 mb-2 bg-danger text-white fw-bold rounded blink-danger"

    # P&L Summary
    pnl = get_pnl_summary()
    today_color = "success" if pnl['today_pnl'] >= 0 else "danger"
    week_color = "success" if pnl['week_pnl'] >= 0 else "danger"
    month_color = "success" if pnl['month_pnl'] >= 0 else "danger"

    pnl_content = html.Div([
        html.Div([
            html.Span("Today: ", className="text-muted small"),
            html.Span(f"${pnl['today_pnl']:+,.0f}", className=f"text-{today_color} fw-bold"),
            html.Span(f" ({pnl['today_pnl_pct']:+.1f}%)", className=f"text-{today_color} small"),
        ], className="me-3"),
        html.Div([
            html.Span("Week: ", className="text-muted small"),
            html.Span(f"${pnl['week_pnl']:+,.0f}", className=f"text-{week_color}"),
        ], className="me-3"),
        html.Div([
            html.Span("Month: ", className="text-muted small"),
            html.Span(f"${pnl['month_pnl']:+,.0f}", className=f"text-{month_color}"),
        ], className="me-3"),
        html.Div([
            html.Span("Unrealized: ", className="text-muted small"),
            html.Span(f"${pnl['unrealized_pnl']:+,.0f}",
                     className=f"text-{'success' if pnl['unrealized_pnl'] >= 0 else 'danger'}"),
        ]),
    ], className="d-flex flex-wrap")

    # Detailed P&L display for collapsible section
    pnl_detailed = html.Div([
        dbc.Row([
            dbc.Col([
                html.H2(f"${pnl['equity']:,.0f}", className="text-primary mb-0"),
                html.Small("Total Equity", className="text-muted"),
            ], width=6),
            dbc.Col([
                html.H3(f"${pnl['today_pnl']:+,.0f}", className=f"text-{today_color} mb-0"),
                html.Small(f"Today ({pnl['today_pnl_pct']:+.2f}%)", className="text-muted"),
            ], width=6),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([
                html.H5(f"${pnl['week_pnl']:+,.0f}", className=f"text-{week_color} mb-0"),
                html.Small("Week", className="text-muted"),
            ], width=4),
            dbc.Col([
                html.H5(f"${pnl['month_pnl']:+,.0f}", className=f"text-{month_color} mb-0"),
                html.Small("Month", className="text-muted"),
            ], width=4),
            dbc.Col([
                html.H5(f"${pnl['unrealized_pnl']:+,.0f}",
                       className=f"text-{'success' if pnl['unrealized_pnl'] >= 0 else 'danger'} mb-0"),
                html.Small("Unrealized", className="text-muted"),
            ], width=4),
        ]),
        html.Hr(),
        html.Small(f"{pnl['position_count']} open positions", className="text-muted"),
    ])

    # P&L Equity Chart
    equity_hist = get_equity_history()
    if not equity_hist.empty:
        pnl_equity_fig = go.Figure()
        pnl_equity_fig.add_trace(go.Scatter(
            x=equity_hist['date'],
            y=equity_hist['equity'],
            mode='lines',
            fill='tozeroy',
            line=dict(color='#00bc8c', width=2),
            fillcolor='rgba(0,188,140,0.2)',
        ))
        # Add VIX regime overlay
        if len(equity_hist) > 1:
            add_vix_regime_overlay(
                pnl_equity_fig,
                (equity_hist['date'].min(), equity_hist['date'].max()),
                equity_hist['equity'].min(),
                equity_hist['equity'].max()
            )
        pnl_equity_fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=20, t=10, b=30),
            showlegend=False,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        )
    else:
        pnl_equity_fig = go.Figure()
        pnl_equity_fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            annotations=[dict(text="No equity data", showarrow=False, font=dict(color='gray'))]
        )

    # Market Status
    market = get_market_status()
    data_fresh = get_data_freshness()

    # Different icons and colors for each market phase
    phase = market['phase']
    if market['is_open']:
        market_icon = "fas fa-sun"
        market_color = "success"
        market_text = "OPEN"
    elif phase == "overnight":
        market_icon = "fas fa-bed"  # Sleep icon for overnight
        market_color = "secondary"
        market_text = "OVERNIGHT"
    elif phase == "pre_market":
        market_icon = "fas fa-cloud-sun"  # Dawn icon
        market_color = "info"
        market_text = "PRE-MARKET"
    elif phase == "after_hours":
        market_icon = "fas fa-moon"
        market_color = "warning"
        market_text = "AFTER HOURS"
    elif phase == "weekend":
        market_icon = "fas fa-calendar-week"
        market_color = "secondary"
        market_text = "WEEKEND"
    else:
        market_icon = "fas fa-question-circle"
        market_color = "secondary"
        market_text = phase.upper().replace('_', ' ')

    market_content = html.Div([
        html.Div([
            html.I(className=f"{market_icon} text-{market_color} me-2"),
            html.Span(market_text, className=f"fw-bold text-{market_color}"),
        ]),
        html.Small(market['time_to_next'], className="text-muted d-block"),
        html.Small(f"Data: {data_fresh['last_update']}", className="text-muted"),
    ])

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
        card_class = "status-panel status-panel-cb halted blink-danger"
    elif position_multiplier < 1.0:
        status_color = "warning"
        status_text = "REDUCED TRADING"
        status_icon = "fas fa-exclamation-triangle"
        card_class = "status-panel status-panel-cb warning"
    else:
        status_color = "success"
        status_text = "TRADING ALLOWED"
        status_icon = "fas fa-check-circle"
        card_class = "status-panel status-panel-cb"

    # Build compact circuit breaker content
    cb_content = html.Div([
        html.Div([
            html.I(className=f"{status_icon} me-2"),
            html.Span(status_text, className=f"fw-bold text-{status_color}"),
        ]),
        html.Div([
            html.Span(f"{int(position_multiplier * 100)}%", className=f"text-{status_color} fw-bold"),
            html.Span(" position size", className="text-muted small"),
        ]),
        html.Div([
            html.Span(f"{len(active_breakers)} active", className="text-warning small") if active_breakers else html.Span("No breakers", className="text-success small"),
            html.Span(" | ", className="text-muted small"),
            html.Span(f"{len(file_switches)} files", className="text-danger small") if file_switches else html.Span("No files", className="text-success small"),
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
        # Add VIX regime overlay
        if len(equity_df) > 1:
            add_vix_regime_overlay(
                equity_fig,
                (equity_df['date'].min(), equity_df['date'].max()),
                equity_df['equity'].min(),
                equity_df['equity'].max()
            )
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

    # Research overview + Shadow Trading summary
    research = get_research_overview()
    shadow_data = get_shadow_trading_data()

    # Calculate shadow trading metrics
    shadow_win_rate = 0
    if shadow_data['total_trades'] > 0:
        shadow_win_rate = (shadow_data['winning_trades'] / shadow_data['total_trades']) * 100

    if research:
        research_content = html.Div([
            # GA & Backtests row
            dbc.Row([
                dbc.Col([
                    html.H4(f"{research.get('total_ga_runs', 0)}", className="text-info mb-0"),
                    html.Small("GA Runs", className="text-muted"),
                ], width=6),
                dbc.Col([
                    html.H4(f"{research.get('total_backtests', 0)}", className="mb-0"),
                    html.Small("Backtests", className="text-muted"),
                ], width=6),
            ], className="mb-2"),
            # Shadow Trading row
            dbc.Row([
                dbc.Col([
                    html.H4(f"{shadow_data['total_strategies']}", className="text-info mb-0"),
                    html.Small("Shadow Strategies", className="text-muted"),
                ], width=6),
                dbc.Col([
                    html.H4(
                        f"${shadow_data['total_pnl']:+,.0f}",
                        className=f"text-{'success' if shadow_data['total_pnl'] >= 0 else 'danger'} mb-0"
                    ),
                    html.Small("Shadow P&L", className="text-muted"),
                ], width=6),
            ], className="mb-2"),
            # Shadow stats row
            dbc.Row([
                dbc.Col([
                    html.Span(f"{shadow_data['total_trades']}", className="fw-bold"),
                    html.Small(" trades", className="text-muted"),
                ], width=6),
                dbc.Col([
                    html.Span(f"{shadow_win_rate:.0f}%", className="fw-bold"),
                    html.Small(" win rate", className="text-muted"),
                ], width=6),
            ], className="mb-2") if shadow_data['total_trades'] > 0 else None,
            html.Hr(),
            html.Small([
                html.Strong("Best Sharpe: "),
                html.Span(f"{research.get('best_sharpe', 0):.2f}", className="text-success"),
            ], className="text-muted d-block"),
            html.Small([
                html.Strong("Last GA: "),
                html.Span(research.get('last_ga_run', 'Never')),
            ], className="text-muted"),
        ])
    else:
        research_content = html.Div("No research data available", className="text-muted")

    # Shadow Trading card content (separate card)
    if shadow_data['total_strategies'] > 0:
        shadow_rows = []
        for strat in shadow_data['strategies']:
            pnl_color = "success" if strat['pnl'] >= 0 else "danger"
            status_badge = html.Span(
                strat['status'].title(),
                className=f"badge bg-{'success' if strat['status'] == 'active' else 'secondary'} me-2"
            )
            shadow_rows.append(
                html.Tr([
                    html.Td([status_badge, strat['name'].replace('_', ' ').title()]),
                    html.Td(f"{strat['trades']}", className="text-center"),
                    html.Td(f"{strat['win_rate']:.0f}%", className="text-center"),
                    html.Td(
                        f"${strat['pnl']:+,.2f}",
                        className=f"text-{pnl_color} text-end"
                    ),
                ])
            )

        shadow_trading_content = html.Div([
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Strategy", className="text-muted"),
                    html.Th("Trades", className="text-muted text-center"),
                    html.Th("Win%", className="text-muted text-center"),
                    html.Th("P&L", className="text-muted text-end"),
                ])),
                html.Tbody(shadow_rows)
            ], className="table table-sm table-dark table-hover mb-2"),
            # Open shadow positions
            html.Div([
                html.Small("Open Positions:", className="text-muted fw-bold"),
            ] + ([
                html.Div([
                    html.Span(f"{pos['symbol']} ", className="fw-bold"),
                    html.Span(f"{pos['side'].upper()} {pos['shares']}", className="text-info"),
                    html.Span(f" @ ${pos['entry_price']:.2f}", className="text-muted"),
                    html.Span(
                        f" ({pos['pnl_pct']:+.1f}%)",
                        className=f"text-{'success' if pos['pnl_pct'] >= 0 else 'danger'}"
                    ),
                ]) for pos in shadow_data.get('positions', [])
            ] if shadow_data.get('positions') else [
                html.Small("No open positions", className="text-muted")
            ]), className="mt-2"),
            html.Hr(className="my-2"),
            html.Div([
                html.Span(f"Total: ", className="text-muted"),
                html.Span(
                    f"${shadow_data['total_pnl']:+,.2f}",
                    className=f"fw-bold text-{'success' if shadow_data['total_pnl'] >= 0 else 'danger'}"
                ),
                html.Span(f" ({shadow_data['total_trades']} trades)", className="text-muted ms-2"),
            ], className="small"),
        ])
        shadow_badge = html.Span(
            f"{shadow_data['active_strategies']} active",
            className="badge bg-info"
        ) if shadow_data['active_strategies'] > 0 else None
    else:
        shadow_trading_content = html.Div([
            html.I(className="fas fa-ghost fa-2x text-muted mb-2"),
            html.P("No shadow strategies", className="text-muted mb-0"),
            html.Small("Add strategies via ShadowTrader", className="text-muted"),
        ], className="text-center py-3")
        shadow_badge = None

    # GA runs table
    ga_runs = get_ga_runs(10)

    # Strategy leaderboard
    leaderboard_raw = get_strategy_leaderboard()
    leaderboard = [
        {
            "rank": i + 1,
            "strategy": item['strategy'].replace('_', ' ').title(),
            "fitness": item['fitness'],
            "generations": item['generations'],
            "trend": item['trend'],
            "last_updated": item['last_updated'],
        }
        for i, item in enumerate(leaderboard_raw)
    ]

    # Backtest results table
    backtests = get_backtest_results(10)

    # GA fitness chart
    ga_history = get_ga_fitness_history()
    if not ga_history.empty:
        ga_fig = go.Figure()
        # Color palette grouped by strategy type for better contrast
        # Intraday (warm reds/oranges) - fast, short-term
        # Swing (cool blues/cyans) - medium-term, days to weeks
        # Long-term (greens/purples) - trend following, weeks to months
        strategy_colors = {
            # Intraday - Warm colors (reds/oranges/yellows)
            'gap_fill': '#FF6B6B',              # Coral red
            'relative_volume_breakout': '#FFA94D',  # Orange

            # Swing - Cool colors (blues/cyans)
            'mean_reversion': '#4DABF7',        # Sky blue
            'pairs_trading': '#3BC9DB',         # Cyan
            'vix_regime_rotation': '#748FFC',   # Indigo blue

            # Long-term/Trend - Greens/Purples
            'vol_managed_momentum': '#51CF66',  # Green
            'sector_rotation': '#9775FA',       # Purple
            'trend_following': '#20C997',       # Teal
            'quality_smallcap_value': '#F783AC', # Pink
            'factor_momentum': '#FFE066',       # Yellow
        }
        fallback_colors = ['#69DB7C', '#FF8787', '#74C0FC', '#B197FC', '#FFC078']

        # Group by strategy and plot each
        for i, strategy in enumerate(ga_history['strategy'].unique()):
            strat_data = ga_history[ga_history['strategy'] == strategy].sort_values('generation')
            color = strategy_colors.get(strategy, fallback_colors[i % len(fallback_colors)])
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

    # Parameter comparison content
    param_data = get_best_params_comparison()
    if param_data:
        param_cards = []
        for item in param_data:
            strategy_name = item['strategy'].replace('_', ' ').title()
            params = item['params']
            if params:
                param_list = html.Ul([
                    html.Li([
                        html.Span(f"{k}: ", className="text-muted"),
                        html.Span(f"{v:.4f}" if isinstance(v, float) else str(v), className="text-info"),
                    ], className="small")
                    for k, v in params.items()
                ], className="mb-0 ps-3", style={'listStyleType': 'none'})
            else:
                param_list = html.Span("No parameters recorded", className="text-muted small")

            param_cards.append(
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6(strategy_name, className="text-success mb-1"),
                            html.Small([
                                html.Span("Fitness: ", className="text-muted"),
                                html.Span(f"{item['fitness']:.4f}", className="text-warning fw-bold"),
                                html.Span(f" (Gen {item['generation']})", className="text-muted"),
                            ], className="d-block mb-2"),
                            param_list,
                        ], className="py-2")
                    ], className="bg-dark h-100")
                ], width=4 if len(param_data) <= 3 else 3, className="mb-2")
            )
        param_comparison_content = dbc.Row(param_cards)
    else:
        param_comparison_content = html.Div("No optimized parameters available yet.", className="text-muted")

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

    # Strategy P&L Attribution
    strategy_pnl_data = get_strategy_pnl_attribution()
    if strategy_pnl_data:
        pnl_rows = []
        for s in strategy_pnl_data:
            pnl_color = 'success' if s['total_pnl'] >= 0 else 'danger'
            win_rate_text = f"{s['win_rate']:.1f}%" if s['win_rate'] is not None else '-'
            pnl_rows.append(
                html.Tr([
                    html.Td(s['strategy'], className="text-light"),
                    html.Td(f"${s['total_pnl']:+,.0f}", className=f"text-{pnl_color}"),
                    html.Td(str(s['total_trades']), className="text-muted"),
                    html.Td(win_rate_text, className="text-muted"),
                    html.Td(s['last_active'], className="text-muted small"),
                ])
            )
        strategy_pnl_content = html.Table([
            html.Thead(html.Tr([
                html.Th("Strategy", className="text-muted"),
                html.Th("Total P&L", className="text-muted"),
                html.Th("Trades", className="text-muted"),
                html.Th("Win Rate", className="text-muted"),
                html.Th("Last Active", className="text-muted"),
            ])),
            html.Tbody(pnl_rows)
        ], className="table table-sm table-dark table-hover mb-0")
    else:
        strategy_pnl_content = html.Div("No strategy P&L data available yet.", className="text-muted")

    # System Process Monitor
    processes = get_system_processes()
    logger.info(f"System processes check: {[(p['name'], p['is_running'], p['pid']) for p in processes]}")
    if processes:
        proc_rows = []
        for p in processes:
            if p['is_running']:
                status_badge = html.Span("Running", className="badge bg-success")
                uptime_text = p.get('uptime', '-')
            else:
                status_badge = html.Span("Stopped", className="badge bg-secondary")
                uptime_text = '-'
            proc_rows.append(
                html.Tr([
                    html.Td([
                        html.I(className=f"fas fa-{'circle text-success' if p['is_running'] else 'circle text-secondary'} me-2"),
                        p['name']
                    ]),
                    html.Td(status_badge),
                    html.Td(uptime_text, className="text-muted small"),
                    html.Td(p.get('pid', '-'), className="text-muted small"),
                ])
            )
        system_monitor_content = html.Table([
            html.Thead(html.Tr([
                html.Th("Process", className="text-muted"),
                html.Th("Status", className="text-muted"),
                html.Th("Uptime", className="text-muted"),
                html.Th("PID", className="text-muted"),
            ])),
            html.Tbody(proc_rows)
        ], className="table table-sm table-dark table-hover mb-0")
    else:
        system_monitor_content = html.Div("Unable to check processes.", className="text-muted")

    # Orchestrator Status
    orch_status = get_orchestrator_status()
    if orch_status['is_running']:
        # Determine status color
        if orch_status['tasks_total'] > 0:
            success_rate = orch_status['tasks_succeeded'] / orch_status['tasks_total']
            status_color = "success" if success_rate == 1.0 else ("warning" if success_rate >= 0.5 else "danger")
        else:
            status_color = "info"

        # Phase badge
        phase_colors = {
            'pre_market': 'info',
            'intraday_open': 'primary',
            'intraday_active': 'primary',
            'market_open': 'success',
            'post_market': 'secondary',
            'evening': 'dark',
            'overnight': 'dark',
            'weekend': 'secondary',
        }
        phase_color = phase_colors.get(orch_status['current_phase'], 'secondary')

        # Build task list
        task_items = []
        for task in orch_status['task_details'][:6]:  # Show up to 6 tasks
            task_icon = "check-circle text-success" if task['status'] == 'success' else "times-circle text-danger"
            task_items.append(
                html.Div([
                    html.I(className=f"fas fa-{task_icon} me-2"),
                    html.Span(task['name'].replace('_', ' ').title(), className="small"),
                ], className="mb-1")
            )

        orchestrator_status_content = html.Div([
            # Status row
            html.Div([
                html.Span([
                    html.I(className="fas fa-circle text-success me-1"),
                    "Running"
                ], className="badge bg-success me-2"),
                html.Span(
                    orch_status['current_phase'].replace('_', ' ').title(),
                    className=f"badge bg-{phase_color} me-2"
                ),
                html.Span(
                    f"{orch_status['tasks_succeeded']}/{orch_status['tasks_total']} tasks",
                    className=f"badge bg-{status_color}"
                ),
            ], className="mb-3"),
            # Last check time
            html.Div([
                html.I(className="fas fa-clock me-2 text-muted"),
                html.Span(f"Last: {orch_status['last_check']}", className="text-muted small"),
            ], className="mb-2") if orch_status['last_check'] else None,
            # Task details
            html.Div([
                html.Strong("Recent Tasks:", className="small text-muted d-block mb-2"),
                html.Div(task_items),
            ]) if task_items else None,
            # Recent errors (if any)
            html.Div([
                html.Strong("Errors:", className="small text-danger d-block mt-2 mb-1"),
                html.Div([
                    html.Div([
                        html.I(className="fas fa-exclamation-triangle text-danger me-1"),
                        html.Span(err[:60] + "..." if len(err) > 60 else err, className="small text-danger"),
                    ]) for err in orch_status['recent_errors'][:3]
                ])
            ]) if orch_status['recent_errors'] else None,
        ])
    else:
        orchestrator_status_content = html.Div([
            html.Span([
                html.I(className="fas fa-circle text-danger me-1"),
                "Stopped"
            ], className="badge bg-danger"),
            html.P("Orchestrator is not running", className="text-muted mt-2 mb-0 small"),
        ])

    # Compact orchestrator display for header
    if orch_status['is_running']:
        if orch_status['tasks_total'] > 0:
            success_rate = orch_status['tasks_succeeded'] / orch_status['tasks_total']
            tasks_color = "success" if success_rate == 1.0 else ("warning" if success_rate >= 0.5 else "danger")
        else:
            tasks_color = "info"

        orchestrator_compact = html.Div([
            html.I(className="fas fa-circle text-success me-1", style={"fontSize": "0.5rem"}),
            html.Span(orch_status['current_phase'].replace('_', ' ').title(), className="me-2"),
            html.Span(
                f"{orch_status['tasks_succeeded']}/{orch_status['tasks_total']}",
                className=f"badge bg-{tasks_color}"
            ),
        ], className="d-flex align-items-center")
        orchestrator_panel_class = "status-panel status-panel-orch running"
    else:
        orchestrator_compact = html.Div([
            html.I(className="fas fa-circle text-danger me-1", style={"fontSize": "0.5rem"}),
            html.Span("Stopped", className="text-danger"),
        ], className="d-flex align-items-center")
        orchestrator_panel_class = "status-panel status-panel-orch stopped"

    # System Performance
    perf = get_system_performance()
    cpu_color = "success" if perf['cpu_percent'] < 70 else ("warning" if perf['cpu_percent'] < 90 else "danger")
    mem_color = "success" if perf['memory_percent'] < 70 else ("warning" if perf['memory_percent'] < 90 else "danger")

    # Compact view for header bar
    perf_compact_items = [
        html.Span([
            html.I(className="fas fa-microchip me-1"),
            f"{perf['cpu_percent']:.0f}%"
        ], className=f"text-{cpu_color} me-2"),
        html.Span([
            html.I(className="fas fa-memory me-1"),
            f"{perf['memory_percent']:.0f}%"
        ], className=f"text-{mem_color} me-2"),
    ]
    if perf['cpu_temp'] is not None:
        temp_color = "success" if perf['cpu_temp'] < 60 else ("warning" if perf['cpu_temp'] < 75 else "danger")
        perf_compact_items.append(
            html.Span([
                html.I(className="fas fa-thermometer-half me-1"),
                f"{perf['cpu_temp']:.0f}°C"
            ], className=f"text-{temp_color}")
        )
    system_perf_compact = html.Div(perf_compact_items, className="d-flex")

    # Swap hierarchy colors
    swap_color = "success" if perf['swap_percent'] < 50 else ("warning" if perf['swap_percent'] < 80 else "danger")
    zram_used_pct = (perf['zram_used_gb'] / perf['zram_total_gb'] * 100) if perf['zram_total_gb'] > 0 else 0
    zram_color = "success" if zram_used_pct < 50 else ("warning" if zram_used_pct < 80 else "danger")
    nvme_used_pct = (perf['nvme_swap_used_gb'] / perf['nvme_swap_total_gb'] * 100) if perf['nvme_swap_total_gb'] > 0 else 0
    nvme_color = "success" if nvme_used_pct < 50 else ("warning" if nvme_used_pct < 80 else "danger")

    # Detailed view for System Health section
    system_perf_detailed = html.Div([
        # CPU
        html.Div([
            html.Div([
                html.I(className="fas fa-microchip me-2"),
                html.Span("CPU", className="text-muted"),
                html.Span(f" {perf['cpu_percent']:.1f}%", className=f"text-{cpu_color} fw-bold ms-2"),
            ]),
            dbc.Progress(value=perf['cpu_percent'], color=cpu_color, className="mb-2", style={"height": "8px"}),
        ], className="mb-3"),

        # Memory Hierarchy Section
        html.Div([
            html.Div([
                html.I(className="fas fa-layer-group me-2"),
                html.Span("Memory Hierarchy", className="text-info fw-bold"),
                html.Span(f" (~{perf['effective_memory_gb']:.0f} GB effective)", className="text-muted small ms-2"),
            ], className="mb-2"),

            # RAM
            html.Div([
                html.Div([
                    html.Span("RAM", className="text-muted", style={"width": "80px", "display": "inline-block"}),
                    html.Span(f"{perf['memory_used_gb']:.1f}/{perf['memory_total_gb']:.0f} GB", className=f"text-{mem_color} fw-bold"),
                    html.Span(f" ({perf['memory_percent']:.0f}%)", className="text-muted small ms-1"),
                ]),
                dbc.Progress(value=perf['memory_percent'], color=mem_color, style={"height": "6px"}),
            ], className="mb-2"),

            # Zram (if available)
            html.Div([
                html.Div([
                    html.Span("Zram", className="text-muted", style={"width": "80px", "display": "inline-block"}),
                    html.Span(f"{perf['zram_used_gb']*1024:.0f}MB/{perf['zram_total_gb']:.0f}GB", className=f"text-{zram_color} fw-bold"),
                    html.Span(f" ({perf['zram_algorithm']})", className="text-muted small ms-1") if perf['zram_algorithm'] else None,
                    html.Span(f" {perf['zram_comp_ratio']:.1f}:1", className="text-success small ms-1") if perf['zram_comp_ratio'] > 1 else None,
                ]),
                dbc.Progress(value=zram_used_pct, color=zram_color, style={"height": "6px"}),
            ], className="mb-2") if perf['zram_total_gb'] > 0 else None,

            # Zswap indicator
            html.Div([
                html.I(className=f"fas fa-{'check-circle text-success' if perf['zswap_enabled'] else 'times-circle text-muted'} me-1"),
                html.Span("Zswap", className="text-muted small"),
                html.Span(" (cache)", className="text-muted small") if perf['zswap_enabled'] else None,
            ], className="mb-2") if perf['zram_total_gb'] > 0 else None,

            # NVMe Swap (if available)
            html.Div([
                html.Div([
                    html.Span("NVMe", className="text-muted", style={"width": "80px", "display": "inline-block"}),
                    html.Span(f"{perf['nvme_swap_used_gb']*1024:.0f}MB/{perf['nvme_swap_total_gb']:.0f}GB", className=f"text-{nvme_color} fw-bold"),
                    html.Span(" (fallback)", className="text-muted small ms-1"),
                ]),
                dbc.Progress(value=nvme_used_pct, color=nvme_color, style={"height": "6px"}),
            ], className="mb-2") if perf['nvme_swap_total_gb'] > 0 else None,

        ], className="mb-3 p-2", style={"backgroundColor": "rgba(255,255,255,0.03)", "borderRadius": "4px"}),

        # Disk
        html.Div([
            html.Div([
                html.I(className="fas fa-hdd me-2"),
                html.Span("Disk", className="text-muted"),
                html.Span(f" {perf['disk_percent']:.1f}%", className="fw-bold ms-2"),
                html.Span(f" ({perf['disk_used_gb']:.0f}/{perf['disk_total_gb']:.0f} GB)", className="text-muted small ms-1"),
            ]),
            dbc.Progress(value=perf['disk_percent'], color="info", className="mb-2", style={"height": "8px"}),
        ], className="mb-3"),

        # Temperature (if available)
        html.Div([
            html.I(className="fas fa-thermometer-half me-2"),
            html.Span("Temp", className="text-muted"),
            html.Span(f" {perf['cpu_temp']:.1f}°C" if perf['cpu_temp'] else " N/A",
                     className=f"text-{temp_color if perf['cpu_temp'] else 'muted'} fw-bold ms-2"),
        ]) if perf['cpu_temp'] is not None else html.Div([
            html.I(className="fas fa-thermometer-half me-2 text-muted"),
            html.Span("Temp: N/A (not available on this system)", className="text-muted small"),
        ]),

        # Load average
        html.Div([
            html.I(className="fas fa-tachometer-alt me-2 text-muted"),
            html.Span(f"Load: {perf['load_avg']:.2f}" if perf['load_avg'] else "Load: N/A", className="text-muted small"),
        ], className="mt-2") if perf['load_avg'] else None,
    ])

    # Trading Signals
    signals = get_recent_signals(hours=24)
    signals_table_data = []
    for sig in signals:
        try:
            created = datetime.fromisoformat(sig['created_at']) if sig['created_at'] else None
            time_str = created.strftime('%H:%M:%S') if created else 'N/A'
        except:
            time_str = sig.get('created_at', 'N/A')[:8] if sig.get('created_at') else 'N/A'

        # Build status with execution route for clarity
        status = sig.get('status', 'unknown')
        route = sig.get('execution_route')
        if status == 'executed' and route:
            if route == 'shadow':
                display_status = 'shadow'
            elif route == 'live':
                display_status = 'LIVE'
            else:
                display_status = f'executed ({route})'
        else:
            display_status = status

        signals_table_data.append({
            'time': time_str,
            'strategy': sig.get('strategy', 'Unknown'),
            'symbol': sig.get('symbol', ''),
            'direction': sig.get('direction', ''),
            'type': sig.get('type', ''),
            'price': f"${sig.get('price', 0):.2f}" if sig.get('price') else 'N/A',
            'confidence': f"{sig.get('confidence', 0) * 100:.0f}%" if sig.get('confidence') else 'N/A',
            'status': display_status,
        })

    # Signal counts by status
    pending_count = sum(1 for s in signals if s.get('status') == 'pending')
    executed_count = sum(1 for s in signals if s.get('status') == 'executed')
    expired_count = sum(1 for s in signals if s.get('status') in ['expired', 'rejected'])

    signals_summary = html.Div([
        html.Span([
            html.I(className="fas fa-clock me-1"),
            f"{pending_count} pending"
        ], className="badge bg-info me-2"),
        html.Span([
            html.I(className="fas fa-check me-1"),
            f"{executed_count} executed"
        ], className="badge bg-success me-2"),
        html.Span([
            html.I(className="fas fa-times me-1"),
            f"{expired_count} expired/rejected"
        ], className="badge bg-secondary me-2"),
        html.Span(f"{len(signals)} signals (24h)", className="text-muted small ms-2"),
    ])

    # Badge for section header
    if pending_count > 0:
        signals_badge = html.Span(f"{pending_count}", className="badge bg-info")
    elif len(signals) > 0:
        signals_badge = html.Span(f"{len(signals)}", className="badge bg-secondary")
    else:
        signals_badge = html.Span("0", className="badge bg-secondary")

    # Weekend Status
    weekend_status_content = render_weekend_status()

    # Last updated timestamp
    last_updated = f"Last updated: {datetime.now().strftime('%H:%M:%S')}"

    return (
        mode_banner,
        mode_class,
        pnl_content,
        pnl_detailed,
        pnl_equity_fig,
        market_content,
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
        shadow_trading_content,
        shadow_badge,
        ga_runs,
        leaderboard,
        backtests,
        ga_fig,
        param_comparison_content,
        system_errors,
        errors_summary,
        strategy_pnl_content,
        system_monitor_content,
        orchestrator_status_content,
        orchestrator_compact,
        orchestrator_panel_class,
        system_perf_compact,
        system_perf_detailed,
        signals_table_data,
        signals_summary,
        signals_badge,
        last_updated,
        weekend_status_content,
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


# Collapse section toggle callbacks
@callback(
    Output('pnl-collapse', 'is_open'),
    Input('pnl-collapse-btn', 'n_clicks'),
    State('pnl-collapse', 'is_open'),
    prevent_initial_call=True,
)
def toggle_pnl(n_clicks, is_open):
    """Toggle P&L section."""
    return not is_open


@callback(
    Output('portfolio-collapse', 'is_open'),
    Input('portfolio-collapse-btn', 'n_clicks'),
    State('portfolio-collapse', 'is_open'),
    prevent_initial_call=True,
)
def toggle_portfolio(n_clicks, is_open):
    """Toggle portfolio overview section."""
    return not is_open


@callback(
    Output('trading-collapse', 'is_open'),
    Input('trading-collapse-btn', 'n_clicks'),
    State('trading-collapse', 'is_open'),
    prevent_initial_call=True,
)
def toggle_trading(n_clicks, is_open):
    """Toggle trading activity section."""
    return not is_open


@callback(
    Output('signals-collapse', 'is_open'),
    Input('signals-collapse-btn', 'n_clicks'),
    State('signals-collapse', 'is_open'),
    prevent_initial_call=True,
)
def toggle_signals(n_clicks, is_open):
    """Toggle trading signals section."""
    return not is_open


@callback(
    Output('research-collapse', 'is_open'),
    Input('research-collapse-btn', 'n_clicks'),
    State('research-collapse', 'is_open'),
    prevent_initial_call=True,
)
def toggle_research(n_clicks, is_open):
    """Toggle research & optimization section."""
    return not is_open


@callback(
    Output('health-collapse', 'is_open'),
    Input('health-collapse-btn', 'n_clicks'),
    State('health-collapse', 'is_open'),
    prevent_initial_call=True,
)
def toggle_health(n_clicks, is_open):
    """Toggle system health section."""
    return not is_open


# Badge update callbacks - show indicators when sections are collapsed
@callback(
    [
        Output('health-badge', 'children'),
        Output('research-badge', 'children'),
        Output('trading-badge', 'children'),
        Output('portfolio-badge', 'children'),
    ],
    [
        Input('interval-component', 'n_intervals'),
        Input('health-collapse', 'is_open'),
        Input('research-collapse', 'is_open'),
        Input('trading-collapse', 'is_open'),
        Input('portfolio-collapse', 'is_open'),
    ],
)
def update_section_badges(n_intervals, health_open, research_open, trading_open, portfolio_open):
    """Update badges to show indicators when sections are collapsed."""
    health_badge = ""
    research_badge = ""
    trading_badge = ""
    portfolio_badge = ""

    # Health badge - show error count when collapsed
    if not health_open:
        errors = get_system_errors()
        error_count = len([e for e in errors if e.get('level') == 'ERROR'])
        warning_count = len([e for e in errors if e.get('level') == 'WARNING'])
        if error_count > 0:
            health_badge = dbc.Badge(f"{error_count}!", color="danger", className="ms-1")
        elif warning_count > 0:
            health_badge = dbc.Badge(f"{warning_count}", color="warning", className="ms-1")

    # Research badge - show running count when collapsed
    if not research_open:
        ga_runs = get_ga_runs(limit=5)
        running = len([r for r in ga_runs if 'running' in r.get('status', '')])
        if running > 0:
            research_badge = dbc.Badge(f"{running} running", color="info", className="ms-1")

    # Trading badge - show pending orders when collapsed
    if not trading_open:
        orders = get_pending_orders()
        if orders:
            trading_badge = dbc.Badge(f"{len(orders)} pending", color="warning", className="ms-1")

    # Portfolio badge - show position count when collapsed
    if not portfolio_open:
        positions = get_positions_data()
        if positions:
            portfolio_badge = dbc.Badge(f"{len(positions)} pos", color="success", className="ms-1")

    return health_badge, research_badge, trading_badge, portfolio_badge


# GA Details Modal callback - use active_cell for direct row click
@callback(
    [
        Output('ga-details-modal', 'is_open'),
        Output('ga-modal-title', 'children'),
        Output('ga-modal-body', 'children'),
    ],
    [
        Input('ga-runs-table', 'active_cell'),
        Input('ga-modal-close', 'n_clicks'),
    ],
    [
        State('ga-runs-table', 'data'),
        State('ga-details-modal', 'is_open'),
    ],
    prevent_initial_call=True,
)
def toggle_ga_modal(active_cell, close_clicks, table_data, is_open):
    """Show GA run details when a row is clicked."""
    from dash import ctx

    if ctx.triggered_id == 'ga-modal-close':
        return False, "", ""

    if ctx.triggered_id == 'ga-runs-table' and active_cell and table_data:
        # Don't open modal when clicking the Action column (that's for Resume/Pause)
        if active_cell.get('column_id') == 'action':
            return False, "", ""

        row_idx = active_cell.get('row', -1)
        if 0 <= row_idx < len(table_data):
            full_run_id = table_data[row_idx].get('full_run_id', '')
            if full_run_id:
                details = get_ga_run_details(full_run_id)

                # Build modal content - show what we have even if details are sparse
                content = [
                    html.P([html.Strong("Run ID: "), full_run_id]),
                    html.P([html.Strong("Started: "), table_data[row_idx].get('start_time', 'N/A')]),
                    html.P([html.Strong("Status: "), table_data[row_idx].get('status', 'N/A')]),
                    html.P([html.Strong("Generations: "), str(table_data[row_idx].get('generations', 0))]),
                    html.P([html.Strong("Improvements: "), str(table_data[row_idx].get('improvements', 0))]),
                ]

                if details:
                    strategies = details.get('strategies', [])
                    if strategies:
                        content.append(html.P([html.Strong("Strategies: "), ", ".join(strategies)]))

                    content.append(html.Hr())
                    content.append(html.H6("Population State:"))

                    # Add population details
                    populations = details.get('populations', [])
                    if populations:
                        for pop in populations:
                            params_str = pop.get('params', '{}')
                            try:
                                import json
                                params = json.loads(params_str) if params_str else {}
                                params_display = ", ".join(f"{k}={v}" for k, v in params.items())
                            except:
                                params_display = params_str[:100] if params_str else "N/A"

                            content.append(
                                dbc.Card([
                                    dbc.CardBody([
                                        html.P([
                                            html.Strong(f"{pop.get('strategy', 'Unknown')}"),
                                            f" - Gen {pop.get('generation', 0)}",
                                        ], className="mb-1"),
                                        html.P([
                                            html.Small(f"Best Fitness: {pop.get('best_fitness', 0):.4f}"),
                                        ], className="mb-1 text-success"),
                                        html.P([
                                            html.Small(f"Best Params: {params_display}"),
                                        ], className="mb-0 text-muted", style={'fontSize': '11px'}),
                                    ])
                                ], className="mb-2", style={'backgroundColor': '#2a2a2a'})
                            )
                    else:
                        content.append(html.P("No population data available yet.", className="text-muted"))
                else:
                    content.append(html.Hr())
                    content.append(html.P("Run details not found in database.", className="text-warning"))

                return True, f"GA Run: {full_run_id[:12]}...", content

    return False, "", ""


# Backtest Details Modal callback
@callback(
    [
        Output('backtest-details-modal', 'is_open'),
        Output('backtest-modal-title', 'children'),
        Output('backtest-modal-body', 'children'),
    ],
    [
        Input('backtest-table', 'active_cell'),
        Input('backtest-modal-close', 'n_clicks'),
    ],
    [
        State('backtest-table', 'data'),
        State('backtest-details-modal', 'is_open'),
    ],
    prevent_initial_call=True,
)
def toggle_backtest_modal(active_cell, close_clicks, table_data, is_open):
    """Show backtest details when a row is clicked."""
    from dash import ctx
    import json as json_module

    if ctx.triggered_id == 'backtest-modal-close':
        return False, "", ""

    if ctx.triggered_id == 'backtest-table' and active_cell and table_data:
        row_idx = active_cell.get('row', -1)
        if 0 <= row_idx < len(table_data):
            row = table_data[row_idx]
            strategy = row.get('strategy', '')
            run_id = row.get('run_id', '')

            # Fetch full details from database
            details = get_backtest_details(strategy, row.get('timestamp', ''))

            # Helper to safely get numeric values (handles None)
            def safe_num(key, default=0):
                val = details.get(key, default)
                return val if val is not None else default

            # Parse parameters
            params_display = "N/A"
            if details.get('params'):
                try:
                    params = json_module.loads(details['params'])
                    params_display = html.Ul([
                        html.Li(f"{k}: {v}") for k, v in params.items()
                    ], className="mb-0 small")
                except:
                    params_display = details['params'][:200]

            # Build modal content
            content = [
                # Header info
                dbc.Row([
                    dbc.Col([
                        html.H6("Strategy", className="text-muted mb-1"),
                        html.H4(strategy.replace('_', ' ').title(), className="text-info"),
                    ], width=6),
                    dbc.Col([
                        html.H6("Run ID", className="text-muted mb-1"),
                        html.Small(run_id[:20] + "..." if len(run_id) > 20 else run_id, className="font-monospace"),
                    ], width=6),
                ], className="mb-3"),

                # Date range
                dbc.Row([
                    dbc.Col([
                        html.Small("Period:", className="text-muted"),
                        html.Span(f" {details.get('start_date', 'N/A')} → {details.get('end_date', 'N/A')}"),
                    ]),
                ], className="mb-3"),

                html.Hr(),

                # Performance Metrics
                html.H5("Performance Metrics", className="text-success mb-3"),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H3(f"{safe_num('sharpe_ratio'):.2f}", className="text-center mb-0"),
                                html.Small("Sharpe Ratio", className="text-muted d-block text-center"),
                            ])
                        ], className="bg-dark"),
                    ], width=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H3(f"{safe_num('sortino_ratio'):.2f}", className="text-center mb-0"),
                                html.Small("Sortino Ratio", className="text-muted d-block text-center"),
                            ])
                        ], className="bg-dark"),
                    ], width=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H3(f"{safe_num('total_return'):.1f}%", className="text-center mb-0"),
                                html.Small("Total Return", className="text-muted d-block text-center"),
                            ])
                        ], className="bg-dark"),
                    ], width=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H3(f"{safe_num('max_drawdown_pct'):.1f}%",
                                       className=f"text-center mb-0 text-{'danger' if safe_num('max_drawdown_pct') < -20 else 'warning'}"),
                                html.Small("Max Drawdown", className="text-muted d-block text-center"),
                            ])
                        ], className="bg-dark"),
                    ], width=3),
                ], className="mb-4"),

                # Trade Statistics
                html.H5("Trade Statistics", className="text-warning mb-3"),
                dbc.Row([
                    dbc.Col([
                        html.P([html.Strong("Total Trades: "), str(details.get('total_trades', 0))]),
                        html.P([html.Strong("Win Rate: "), f"{safe_num('win_rate'):.1f}%"]),
                        html.P([html.Strong("Profit Factor: "), f"{safe_num('profit_factor'):.2f}"]),
                    ], width=6),
                    dbc.Col([
                        html.P([html.Strong("Winning Trades: "), str(details.get('winning_trades', 0))]),
                        html.P([html.Strong("Losing Trades: "), str(details.get('losing_trades', 0))]),
                        html.P([html.Strong("Avg Trade P&L: "), f"${safe_num('avg_trade_pnl'):,.2f}"]),
                    ], width=6),
                ], className="mb-3"),

                html.Hr(),

                # Risk Metrics
                html.H5("Risk Metrics", className="text-info mb-3"),
                dbc.Row([
                    dbc.Col([
                        html.P([html.Strong("Volatility: "), f"{safe_num('volatility'):.2f}%"]),
                        html.P([html.Strong("VaR (95%): "), f"${safe_num('var_95'):,.0f}"]),
                        html.P([html.Strong("CVaR (95%): "), f"${safe_num('cvar_95'):,.0f}"]),
                    ], width=6),
                    dbc.Col([
                        html.P([html.Strong("Beta: "), f"{safe_num('beta'):.3f}"]),
                        html.P([html.Strong("Alpha: "), f"{safe_num('alpha'):.3f}"]),
                        html.P([html.Strong("Calmar Ratio: "), f"{safe_num('calmar_ratio'):.2f}"]),
                    ], width=6),
                ]),

                html.Hr(),

                # Parameters
                html.H5("Strategy Parameters", className="text-secondary mb-2"),
                html.Div(params_display, className="bg-dark p-2 rounded"),
            ]

            return True, f"Backtest: {strategy.replace('_', ' ').title()}", content

    return False, "", ""


# Logs Modal callback - controls modal open/close and interval
@callback(
    [
        Output('logs-modal', 'is_open'),
        Output('logs-interval', 'disabled'),
    ],
    [
        Input('btn-view-logs', 'n_clicks'),
        Input('btn-view-logs-mobile', 'n_clicks'),
        Input('logs-modal-close', 'n_clicks'),
    ],
    State('logs-modal', 'is_open'),
    prevent_initial_call=True,
)
def toggle_logs_modal(view_clicks, view_mobile, close_clicks, is_open):
    """Toggle logs modal and interval."""
    from dash import ctx

    if ctx.triggered_id == 'logs-modal-close':
        return False, True  # Close modal, disable interval

    if ctx.triggered_id in ['btn-view-logs', 'btn-view-logs-mobile']:
        return True, False  # Open modal, enable interval

    return is_open, True


# Logs content update callback - runs on interval
@callback(
    Output('logs-content', 'children'),
    [
        Input('logs-interval', 'n_intervals'),
        Input('logs-modal', 'is_open'),
    ],
    prevent_initial_call=True,
)
def update_logs_content(n_intervals, is_open):
    """Update log content when modal is open."""
    import subprocess

    if not is_open:
        return ""

    try:
        log_file = PROJECT_ROOT / "logs" / "nightly_research.log"
        if log_file.exists():
            result = subprocess.run(
                ["tail", "-100", str(log_file)],
                capture_output=True, text=True, timeout=5
            )
            return result.stdout or "No log content"
        return "Log file not found"
    except Exception as e:
        return f"Error reading logs: {e}"


def render_system_monitor(processes):
    """Render the system monitor table."""
    if processes:
        proc_rows = []
        for p in processes:
            if p['is_running']:
                status_badge = html.Span("Running", className="badge bg-success")
                uptime_text = p.get('uptime', '-')
            else:
                status_badge = html.Span("Stopped", className="badge bg-secondary")
                uptime_text = '-'
            proc_rows.append(
                html.Tr([
                    html.Td([
                        html.I(className=f"fas fa-{'circle text-success' if p['is_running'] else 'circle text-secondary'} me-2"),
                        p['name']
                    ]),
                    html.Td(status_badge),
                    html.Td(uptime_text, className="text-muted small"),
                    html.Td(p.get('pid', '-'), className="text-muted small"),
                ])
            )
        return html.Table([
            html.Thead(html.Tr([
                html.Th("Process", className="text-muted"),
                html.Th("Status", className="text-muted"),
                html.Th("Uptime", className="text-muted"),
                html.Th("PID", className="text-muted"),
            ])),
            html.Tbody(proc_rows)
        ], className="table table-sm table-dark table-hover mb-0")
    else:
        return html.Div("Unable to check processes.", className="text-muted")


def render_orchestrator_status():
    """Render the orchestrator status content."""
    orch_status = get_orchestrator_status()
    if orch_status['is_running']:
        # Determine status color
        if orch_status['tasks_total'] > 0:
            success_rate = orch_status['tasks_succeeded'] / orch_status['tasks_total']
            status_color = "success" if success_rate == 1.0 else ("warning" if success_rate >= 0.5 else "danger")
        else:
            status_color = "info"

        # Phase badge
        phase_colors = {
            'pre_market': 'info',
            'intraday_open': 'primary',
            'intraday_active': 'primary',
            'market_open': 'success',
            'post_market': 'secondary',
            'evening': 'dark',
            'overnight': 'dark',
            'weekend': 'secondary',
        }
        phase_color = phase_colors.get(orch_status['current_phase'], 'secondary')

        # Build task list
        task_items = []
        for task in orch_status['task_details'][:6]:
            task_icon = "check-circle text-success" if task['status'] == 'success' else "times-circle text-danger"
            task_items.append(
                html.Div([
                    html.I(className=f"fas fa-{task_icon} me-2"),
                    html.Span(task['name'].replace('_', ' ').title(), className="small"),
                ], className="mb-1")
            )

        return html.Div([
            html.Div([
                html.Span([
                    html.I(className="fas fa-circle text-success me-1"),
                    "Running"
                ], className="badge bg-success me-2"),
                html.Span(
                    orch_status['current_phase'].replace('_', ' ').title(),
                    className=f"badge bg-{phase_color} me-2"
                ),
                html.Span(
                    f"{orch_status['tasks_succeeded']}/{orch_status['tasks_total']} tasks",
                    className=f"badge bg-{status_color}"
                ),
            ], className="mb-3"),
            html.Div([
                html.I(className="fas fa-clock me-2 text-muted"),
                html.Span(f"Last: {orch_status['last_check']}", className="text-muted small"),
            ], className="mb-2") if orch_status['last_check'] else None,
            html.Div([
                html.Strong("Recent Tasks:", className="small text-muted d-block mb-2"),
                html.Div(task_items),
            ]) if task_items else None,
            html.Div([
                html.Strong("Errors:", className="small text-danger d-block mt-2 mb-1"),
                html.Div([
                    html.Div([
                        html.I(className="fas fa-exclamation-triangle text-danger me-1"),
                        html.Span(err[:60] + "..." if len(err) > 60 else err, className="small text-danger"),
                    ]) for err in orch_status['recent_errors'][:3]
                ])
            ]) if orch_status['recent_errors'] else None,
        ])
    else:
        return html.Div([
            html.Span([
                html.I(className="fas fa-circle text-danger me-1"),
                "Stopped"
            ], className="badge bg-danger"),
            html.P("Orchestrator is not running", className="text-muted mt-2 mb-0 small"),
        ])


def render_weekend_status():
    """Render weekend phase status with visual timeline and research progress."""
    weekend = get_weekend_status()

    if not weekend['is_weekend'] and not weekend['sub_phase']:
        return html.Div([
            html.Span("Not Weekend", className="badge bg-secondary me-2"),
            html.P("Weekend phase runs Friday 4 PM → Sunday evening", className="text-muted small mt-2 mb-0"),
        ])

    # Build timeline visualization
    timeline_items = []
    for item in weekend['timeline']:
        if item['status'] == 'completed':
            icon_class = "fas fa-check-circle text-success"
            badge_class = "bg-success"
        elif item['status'] == 'active':
            icon_class = "fas fa-circle-notch fa-spin text-primary"
            badge_class = "bg-primary"
        else:
            icon_class = "far fa-circle text-muted"
            badge_class = "bg-secondary"

        timeline_items.append(
            html.Div([
                html.I(className=f"{icon_class} me-2"),
                html.Span(item['name'], className=f"badge {badge_class}"),
            ], className="d-inline-block me-2 mb-2")
        )

    # Current phase info
    phase_info = []
    if weekend['sub_phase']:
        phase_info.append(
            html.Div([
                html.Strong("Current: ", className="text-muted"),
                html.Span(weekend['sub_phase_display'], className="badge bg-info"),
            ], className="mb-2")
        )

    # Research progress details
    research_progress = weekend.get('research_progress', {})
    if weekend['research_status'] == 'running' and research_progress:
        # Research phase indicator
        research_phase = research_progress.get('phase', 'optimization')
        phase_labels = {'optimization': 'Parameter Optimization', 'discovery': 'Strategy Discovery', 'adaptive': 'Adaptive GA'}
        phase_info.append(
            html.Div([
                html.I(className="fas fa-dna fa-spin text-warning me-2"),
                html.Span(phase_labels.get(research_phase, research_phase), className="text-warning fw-bold"),
            ], className="mb-2")
        )

        # Current strategy and generation
        current_strat = research_progress.get('current_strategy', '')
        gen = research_progress.get('generation', 0)
        total_gen = research_progress.get('total_generations', 0)
        ind = research_progress.get('individual', 0)
        pop = research_progress.get('population_size', 0)

        if current_strat:
            phase_info.append(
                html.Div([
                    html.Span("Strategy: ", className="text-muted small"),
                    html.Span(current_strat.replace('_', ' ').title(), className="small fw-bold"),
                ], className="mb-1")
            )

        if gen > 0 or ind > 0:
            progress_text = f"Gen {gen}/{total_gen}" if total_gen else f"Gen {gen}"
            if ind and pop:
                progress_text += f" | Individual {ind}/{pop}"
            phase_info.append(
                html.Div([
                    html.I(className="fas fa-chart-line me-2 text-muted"),
                    html.Span(progress_text, className="small"),
                ], className="mb-1")
            )

        # Best fitness
        best_fitness = research_progress.get('best_fitness', 0)
        if best_fitness > 0:
            phase_info.append(
                html.Div([
                    html.Span("Best Fitness: ", className="text-muted small"),
                    html.Span(f"{best_fitness:.2f}", className="small text-success fw-bold"),
                ], className="mb-1")
            )

        # Strategies completed
        strats_completed = research_progress.get('strategies_completed', [])
        strats_remaining = research_progress.get('strategies_remaining', [])
        if strats_completed or strats_remaining:
            total = len(strats_completed) + len(strats_remaining)
            phase_info.append(
                html.Div([
                    html.Span(f"Strategies: {len(strats_completed)}/{total} complete", className="small text-muted"),
                ], className="mb-1")
            )

        # Discoveries
        discoveries = research_progress.get('discoveries_found', 0)
        if discoveries > 0:
            phase_info.append(
                html.Div([
                    html.I(className="fas fa-lightbulb text-warning me-2"),
                    html.Span(f"{discoveries} new strategies discovered!", className="small text-warning"),
                ], className="mb-1")
            )

    elif weekend['research_status'] == 'running':
        phase_info.append(
            html.Div([
                html.I(className="fas fa-cog fa-spin text-warning me-2"),
                html.Span("Research running...", className="text-warning"),
            ], className="mb-2")
        )
    elif weekend['research_status'] == 'complete':
        phase_info.append(
            html.Div([
                html.I(className="fas fa-check-circle text-success me-2"),
                html.Span("Research complete", className="text-success"),
            ], className="mb-2")
        )
    elif weekend['research_status'] == 'paused':
        phase_info.append(
            html.Div([
                html.I(className="fas fa-pause-circle text-warning me-2"),
                html.Span("Research paused", className="text-warning"),
            ], className="mb-2")
        )
    elif weekend['research_status'] == 'error':
        error_msg = research_progress.get('error', 'Unknown error')
        phase_info.append(
            html.Div([
                html.I(className="fas fa-exclamation-circle text-danger me-2"),
                html.Span(f"Error: {error_msg[:50]}", className="text-danger small"),
            ], className="mb-2")
        )

    return html.Div([
        # Weekend active indicator
        html.Div([
            html.Span([
                html.I(className="fas fa-flask me-1"),
                "Weekend Active" if weekend['is_weekend'] else "Weekend Phase"
            ], className="badge bg-info me-2"),
        ], className="mb-3"),

        # Timeline
        html.Div([
            html.Strong("Progress:", className="small text-muted d-block mb-2"),
            html.Div(timeline_items),
        ], className="mb-3"),

        # Phase info and research progress
        html.Div(phase_info) if phase_info else None,
    ])


# Mobile Actions Menu Toggle
@callback(
    Output('mobile-actions-collapse', 'is_open'),
    Input('mobile-actions-toggle', 'n_clicks'),
    State('mobile-actions-collapse', 'is_open'),
    prevent_initial_call=True,
)
def toggle_mobile_actions(n_clicks, is_open):
    """Toggle mobile actions collapsible menu."""
    return not is_open


# Orchestrator Control callbacks
@callback(
    [
        Output('io-status', 'children', allow_duplicate=True),
        Output('io-status-mobile', 'children', allow_duplicate=True),
        Output('system-monitor-content', 'children', allow_duplicate=True),
        Output('orchestrator-status-content', 'children', allow_duplicate=True),
    ],
    [
        Input('btn-start-orchestrator', 'n_clicks'),
        Input('btn-stop-orchestrator', 'n_clicks'),
        Input('btn-start-orchestrator-mobile', 'n_clicks'),
        Input('btn-stop-orchestrator-mobile', 'n_clicks'),
    ],
    prevent_initial_call=True,
)
def handle_orchestrator_buttons(start_clicks, stop_clicks, start_mobile, stop_mobile):
    """Handle orchestrator start/stop button clicks."""
    from dash import ctx
    import subprocess
    import time
    from datetime import datetime

    timestamp = datetime.now().strftime("%H:%M:%S")

    if ctx.triggered_id in ['btn-start-orchestrator', 'btn-start-orchestrator-mobile']:
        try:
            # Check if already running
            import psutil
            for p in psutil.process_iter(['pid', 'cmdline']):
                try:
                    cmdline = ' '.join(p.info.get('cmdline') or [])
                    if 'daily_orchestrator.py' in cmdline:
                        msg = f"[{timestamp}] Already running (PID {p.info['pid']})"
                        return msg, msg, no_update, no_update
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Start orchestrator
            subprocess.Popen(
                ["python3", str(PROJECT_ROOT / "daily_orchestrator.py")],
                cwd=str(PROJECT_ROOT),
                stdout=open("/tmp/orchestrator.log", "a"),
                stderr=subprocess.STDOUT,
            )
            time.sleep(1)  # Brief wait for process to start

            # Update system monitor display
            processes = get_system_processes()
            system_monitor_content = render_system_monitor(processes)
            orchestrator_status_content = render_orchestrator_status()

            msg = f"[{timestamp}] Orchestrator started"
            return msg, msg, system_monitor_content, orchestrator_status_content
        except Exception as e:
            msg = f"[{timestamp}] Error: {e}"
            return msg, msg, no_update, no_update

    elif ctx.triggered_id in ['btn-stop-orchestrator', 'btn-stop-orchestrator-mobile']:
        try:
            subprocess.run(["pkill", "-f", "daily_orchestrator.py"], capture_output=True)
            time.sleep(1)  # Brief wait for process to stop

            # Update system monitor display
            processes = get_system_processes()
            system_monitor_content = render_system_monitor(processes)
            orchestrator_status_content = render_orchestrator_status()

            msg = f"[{timestamp}] Orchestrator stopped"
            return msg, msg, system_monitor_content, orchestrator_status_content
        except Exception as e:
            msg = f"[{timestamp}] Error: {e}"
            return msg, msg, no_update, no_update

    return "", "", no_update, no_update


# I/O Control callbacks
@callback(
    [
        Output('io-status', 'children'),
        Output('io-status-mobile', 'children'),
    ],
    [
        Input('btn-quick-test', 'n_clicks'),
        Input('btn-stop-research', 'n_clicks'),
        Input('btn-cleanup-dbs', 'n_clicks'),
        Input('btn-quick-test-mobile', 'n_clicks'),
        Input('btn-stop-research-mobile', 'n_clicks'),
        Input('btn-cleanup-dbs-mobile', 'n_clicks'),
    ],
    prevent_initial_call=True,
)
def handle_io_buttons(quick_test, stop_research, cleanup_dbs, quick_test_m, stop_research_m, cleanup_dbs_m):
    """Handle I/O control button clicks."""
    from dash import ctx
    import subprocess
    from datetime import datetime

    timestamp = datetime.now().strftime("%H:%M:%S")

    if ctx.triggered_id in ['btn-quick-test', 'btn-quick-test-mobile']:
        try:
            import json as json_module
            # Load config
            config_path = Path(__file__).parent / "quick_test_config.json"
            config = {"mode": "auto", "generations": 1}
            if config_path.exists():
                with open(config_path) as f:
                    config = json_module.load(f)

            # Pick strategy based on mode
            if config.get("mode") == "manual":
                strategy = config.get("manual_strategy", "mean_reversion")
                reason = "manual selection"
            else:
                strategy, reason = get_best_strategy_for_testing()

            generations = config.get("generations", 1)

            # Start a quick test in background
            subprocess.Popen(
                ["python3", str(PROJECT_ROOT / "run_nightly_research.py"),
                 "--quick", "--strategies", strategy, "--generations", str(generations)],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            msg = f"[{timestamp}] Quick test: {strategy} ({reason}, {generations} gen)"
            return msg, msg
        except Exception as e:
            msg = f"[{timestamp}] Error starting test: {e}"
            return msg, msg

    elif ctx.triggered_id in ['btn-stop-research', 'btn-stop-research-mobile']:
        try:
            # Kill any running research processes
            result = subprocess.run(
                ["pkill", "-f", "run_nightly_research.py"],
                capture_output=True, text=True
            )
            msg = f"[{timestamp}] Stop signal sent to research processes"
            return msg, msg
        except Exception as e:
            msg = f"[{timestamp}] Error: {e}"
            return msg, msg

    elif ctx.triggered_id in ['btn-cleanup-dbs', 'btn-cleanup-dbs-mobile']:
        try:
            # Run database cleanup
            from data.storage.db_manager import DatabaseManager
            db = DatabaseManager()

            # Clean up 0-progress GA runs
            conn = db.get_connection("research")
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM ga_runs
                WHERE status IN ('interrupted', 'abandoned')
                AND (total_generations IS NULL OR total_generations = 0)
            """)
            deleted = cursor.rowcount
            conn.commit()
            db.close_all()

            msg = f"[{timestamp}] Cleaned up {deleted} empty GA runs"
            return msg, msg
        except Exception as e:
            msg = f"[{timestamp}] Error: {e}"
            return msg, msg

    return "", ""


# GA Control callbacks - Per-row Resume/Pause via Action column click
@callback(
    Output('ga-control-status', 'children'),
    Input('ga-runs-table', 'active_cell'),
    State('ga-runs-table', 'data'),
    prevent_initial_call=True,
)
def handle_ga_row_action(active_cell, table_data):
    """Handle per-row Resume/Pause when clicking the Action column."""
    import subprocess
    from datetime import datetime

    if not active_cell or not table_data:
        return ""

    # Only trigger on Action column clicks
    if active_cell.get('column_id') != 'action':
        return ""

    row_idx = active_cell.get('row', -1)
    if row_idx < 0 or row_idx >= len(table_data):
        return ""

    row = table_data[row_idx]
    action = row.get('action', '-')
    run_id = row.get('full_run_id', '')
    strategy = row.get('strategies', '').strip('[]"').split(',')[0].strip('" ').lower().replace(' ', '_')

    timestamp = datetime.now().strftime("%H:%M:%S")

    if action == "▶ Resume" and run_id and strategy:
        try:
            subprocess.Popen(
                ["python3", str(PROJECT_ROOT / "run_nightly_research.py"),
                 "--resume", run_id, "--strategies", strategy],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return dbc.Alert(f"[{timestamp}] Resuming {strategy} (run {run_id[:8]}...)", color="success", className="py-1 mb-0")
        except Exception as e:
            return dbc.Alert(f"Error: {e}", color="danger", className="py-1 mb-0")

    elif action == "⏸ Pause" and run_id:
        try:
            # Update this specific run to 'paused' status
            db_path = PROJECT_ROOT / "db" / "research.db"
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("UPDATE ga_runs SET status = 'paused' WHERE run_id = ?", (run_id,))
            conn.commit()
            conn.close()

            # Send signal to stop the process gracefully
            subprocess.run(["pkill", "-f", "run_nightly_research.py"], capture_output=True)

            return dbc.Alert(f"[{timestamp}] Paused {strategy} - will resume on restart", color="warning", className="py-1 mb-0")
        except Exception as e:
            return dbc.Alert(f"Error: {e}", color="danger", className="py-1 mb-0")

    return ""


# Kill Switch Modal callbacks
@callback(
    Output('kill-switch-modal', 'is_open'),
    [
        Input('btn-kill-switch', 'n_clicks'),
        Input('btn-kill-switch-mobile', 'n_clicks'),
        Input('kill-switch-cancel', 'n_clicks'),
        Input('kill-switch-execute', 'n_clicks'),
    ],
    State('kill-switch-modal', 'is_open'),
    prevent_initial_call=True,
)
def toggle_kill_switch_modal(open_clicks, open_mobile, cancel_clicks, execute_clicks, is_open):
    """Toggle kill switch confirmation modal."""
    from dash import ctx
    if ctx.triggered_id in ['btn-kill-switch', 'btn-kill-switch-mobile']:
        return True
    return False


def render_circuit_breaker_display():
    """Render the circuit breaker status display components."""
    cb_status = get_circuit_breaker_status()
    trading_allowed = cb_status.get('trading_allowed', True)
    position_multiplier = cb_status.get('position_multiplier', 1.0)
    active_breakers = cb_status.get('active_breakers', [])
    file_switches = cb_status.get('file_kill_switches', [])

    if not trading_allowed:
        status_color = "danger"
        status_text = "TRADING HALTED"
        status_icon = "fas fa-stop-circle"
        card_class = "status-panel status-panel-cb halted blink-danger"
    elif position_multiplier < 1.0:
        status_color = "warning"
        status_text = "REDUCED TRADING"
        status_icon = "fas fa-exclamation-triangle"
        card_class = "status-panel status-panel-cb warning"
    else:
        status_color = "success"
        status_text = "TRADING ALLOWED"
        status_icon = "fas fa-check-circle"
        card_class = "status-panel status-panel-cb"

    cb_content = html.Div([
        html.Div([
            html.I(className=f"{status_icon} me-2"),
            html.Span(status_text, className=f"fw-bold text-{status_color}"),
        ]),
        html.Div([
            html.Span(f"{int(position_multiplier * 100)}%", className=f"text-{status_color} fw-bold"),
            html.Span(" position size", className="text-muted small"),
        ]),
        html.Div([
            html.Span(f"{len(active_breakers)} active", className="text-warning small") if active_breakers else html.Span("No breakers", className="text-success small"),
            html.Span(" | ", className="text-muted small"),
            html.Span(f"{len(file_switches)} files", className="text-danger small") if file_switches else html.Span("No files", className="text-success small"),
        ]),
    ])
    return cb_content, card_class


@callback(
    [
        Output('io-status', 'children', allow_duplicate=True),
        Output('circuit-breaker-status', 'children', allow_duplicate=True),
        Output('circuit-breaker-card', 'className', allow_duplicate=True),
    ],
    Input('kill-switch-execute', 'n_clicks'),
    prevent_initial_call=True,
)
def execute_kill_switch(n_clicks):
    """Execute kill switch - stop all processes and trip circuit breaker."""
    if not n_clicks:
        return "", no_update, no_update

    import subprocess
    from datetime import datetime
    timestamp = datetime.now().strftime("%H:%M:%S")

    try:
        # 1. Kill all research processes
        subprocess.run(["pkill", "-f", "run_nightly_research.py"], capture_output=True)

        # 2. Kill orchestrator
        subprocess.run(["pkill", "-f", "daily_orchestrator.py"], capture_output=True)

        # 3. Engage circuit breaker halt
        try:
            from execution.circuit_breaker import CircuitBreakerManager
            cb = CircuitBreakerManager()
            cb.emergency_halt()
        except Exception as e:
            logger.error(f"Circuit breaker error: {e}")

        # Get updated display
        cb_content, card_class = render_circuit_breaker_display()
        return f"[{timestamp}] KILL SWITCH ENGAGED - All processes stopped, trading halted", cb_content, card_class
    except Exception as e:
        return f"[{timestamp}] Kill switch error: {e}", no_update, no_update


# Reset System callback
@callback(
    [
        Output('io-status', 'children', allow_duplicate=True),
        Output('io-status-mobile', 'children', allow_duplicate=True),
        Output('circuit-breaker-status', 'children', allow_duplicate=True),
        Output('circuit-breaker-card', 'className', allow_duplicate=True),
    ],
    [
        Input('btn-reset-system', 'n_clicks'),
        Input('btn-reset-system-mobile', 'n_clicks'),
    ],
    prevent_initial_call=True,
)
def reset_system(n_clicks, n_clicks_mobile):
    """Reset system - clear circuit breaker and kill switches."""
    from dash import ctx
    if not ctx.triggered_id:
        return "", "", no_update, no_update

    from datetime import datetime
    timestamp = datetime.now().strftime("%H:%M:%S")

    try:
        from execution.circuit_breaker import CircuitBreakerManager
        cb = CircuitBreakerManager()
        # Clear any active kill switches
        cb.kill_switch.clear_switch('HALT')
        cb.kill_switch.clear_switch('GRACEFUL')

        # Get updated display
        cb_content, card_class = render_circuit_breaker_display()
        msg = f"[{timestamp}] System reset - Trading re-enabled"
        return msg, msg, cb_content, card_class
    except Exception as e:
        msg = f"[{timestamp}] Reset error: {e}"
        return msg, msg, no_update, no_update


# Close Positions Modal callbacks
@callback(
    Output('close-positions-modal', 'is_open'),
    [
        Input('btn-close-positions', 'n_clicks'),
        Input('close-positions-cancel', 'n_clicks'),
        Input('close-positions-execute', 'n_clicks'),
    ],
    State('close-positions-modal', 'is_open'),
    prevent_initial_call=True,
)
def toggle_close_positions_modal(open_clicks, cancel_clicks, execute_clicks, is_open):
    """Toggle close positions confirmation modal."""
    from dash import ctx
    if ctx.triggered_id == 'btn-close-positions':
        return True
    return False


@callback(
    Output('close-positions-execute', 'disabled'),
    Input('close-positions-confirm-input', 'value'),
    prevent_initial_call=True,
)
def validate_close_positions_confirm(value):
    """Enable execute button only when CONFIRM is typed."""
    return value != "CONFIRM"


@callback(
    Output('io-status', 'children', allow_duplicate=True),
    Input('close-positions-execute', 'n_clicks'),
    State('close-positions-confirm-input', 'value'),
    prevent_initial_call=True,
)
def execute_close_all_positions(n_clicks, confirm_value):
    """Close all positions at market price."""
    if not n_clicks or confirm_value != "CONFIRM":
        return ""

    from datetime import datetime
    timestamp = datetime.now().strftime("%H:%M:%S")

    try:
        broker = get_broker()
        positions = broker.get_positions()

        if not positions:
            return f"[{timestamp}] No positions to close"

        closed = 0
        errors = 0
        for pos in positions:
            try:
                # Submit market order to close position
                side = 'sell' if float(pos.qty) > 0 else 'buy'
                broker.submit_market_order(
                    symbol=pos.symbol,
                    qty=abs(float(pos.qty)),
                    side=side
                )
                closed += 1
            except Exception as e:
                logger.error(f"Error closing {pos.symbol}: {e}")
                errors += 1

        return f"[{timestamp}] Closed {closed} positions ({errors} errors)"
    except Exception as e:
        return f"[{timestamp}] Error closing positions: {e}"


# Auto-refresh toggle callback
@callback(
    [
        Output('interval-component', 'disabled'),
        Output('btn-auto-refresh', 'outline'),
        Output('btn-auto-refresh', 'color'),
    ],
    Input('btn-auto-refresh', 'n_clicks'),
    State('interval-component', 'disabled'),
    prevent_initial_call=True,
)
def toggle_auto_refresh(n_clicks, is_disabled):
    """Toggle auto-refresh on/off."""
    if is_disabled:
        # Turn ON auto-refresh
        return False, True, "info"  # enabled, outline, info color
    else:
        # Turn OFF auto-refresh
        return True, False, "secondary"  # disabled, solid, secondary color


# Quick Test Config Modal callbacks
@callback(
    [
        Output('quick-test-config-modal', 'is_open'),
        Output('quick-test-mode', 'value'),
        Output('quick-test-strategy', 'value'),
        Output('quick-test-generations', 'value'),
        Output('quick-test-config-status', 'children'),
    ],
    [
        Input('btn-quick-test-config', 'n_clicks'),
        Input('quick-test-config-cancel', 'n_clicks'),
        Input('quick-test-config-save', 'n_clicks'),
    ],
    [
        State('quick-test-config-modal', 'is_open'),
        State('quick-test-mode', 'value'),
        State('quick-test-strategy', 'value'),
        State('quick-test-generations', 'value'),
    ],
    prevent_initial_call=True,
)
def handle_quick_test_config_modal(open_clicks, cancel_clicks, save_clicks,
                                   is_open, mode, strategy, generations):
    """Handle Quick Test Config modal open/close/save."""
    from dash import ctx
    import json as json_module

    config_path = Path(__file__).parent / "quick_test_config.json"

    if ctx.triggered_id == 'btn-quick-test-config':
        # Opening modal - load current config
        try:
            if config_path.exists():
                with open(config_path) as f:
                    config = json_module.load(f)
                return (True, config.get('mode', 'auto'),
                       config.get('manual_strategy', 'mean_reversion'),
                       config.get('generations', 1),
                       f"Loaded from {config_path.name}")
        except Exception as e:
            pass
        return True, 'auto', 'mean_reversion', 1, "Using defaults"

    elif ctx.triggered_id == 'quick-test-config-cancel':
        return False, no_update, no_update, no_update, ""

    elif ctx.triggered_id == 'quick-test-config-save':
        # Save config
        try:
            config = {
                "mode": mode,
                "manual_strategy": strategy,
                "generations": int(generations) if generations else 1,
                "notes": "mode options: 'auto' (picks best strategy) or 'manual' (uses manual_strategy)"
            }
            with open(config_path, 'w') as f:
                json_module.dump(config, f, indent=4)
            return False, no_update, no_update, no_update, "Config saved!"
        except Exception as e:
            return True, no_update, no_update, no_update, f"Error saving: {e}"

    return False, no_update, no_update, no_update, ""


# Show/hide manual strategy selection based on mode
@callback(
    Output('manual-strategy-row', 'style'),
    Input('quick-test-mode', 'value'),
)
def toggle_manual_strategy_visibility(mode):
    """Show manual strategy dropdown only when manual mode is selected."""
    if mode == 'manual':
        return {'display': 'block'}
    return {'display': 'none'}


# ============================================================================
# Weekend Control Panel Callbacks
# ============================================================================

@callback(
    Output('weekend-config-collapse', 'is_open'),
    Input('weekend-config-toggle', 'n_clicks'),
    State('weekend-config-collapse', 'is_open'),
    prevent_initial_call=True,
)
def toggle_weekend_config(n_clicks, is_open):
    """Toggle weekend configuration panel."""
    return not is_open


@callback(
    [
        Output('weekend-generations', 'value'),
        Output('weekend-population', 'value'),
        Output('weekend-discovery-toggle', 'value'),
        Output('weekend-adaptive-toggle', 'value'),
        Output('weekend-discovery-hours', 'value'),
    ],
    [
        Input('weekend-preset-quick', 'n_clicks'),
        Input('weekend-preset-standard', 'n_clicks'),
        Input('weekend-preset-deep', 'n_clicks'),
    ],
    prevent_initial_call=True,
)
def apply_weekend_preset(quick_clicks, standard_clicks, deep_clicks):
    """Apply a weekend research preset configuration."""
    from dash import ctx

    # Determine which button was clicked
    if not ctx.triggered:
        raise PreventUpdate

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'weekend-preset-quick':
        return 5, 15, [], [], 2  # Quick: 5 gens, 15 pop, no discovery/adaptive, 2 hrs
    elif button_id == 'weekend-preset-standard':
        return 10, 30, ["discovery"], ["adaptive"], 4  # Standard: 10 gens, 30 pop, both enabled, 4 hrs
    elif button_id == 'weekend-preset-deep':
        return 25, 50, ["discovery"], ["adaptive"], 8  # Deep: 25 gens, 50 pop, both enabled, 8 hrs

    raise PreventUpdate


@callback(
    Output('weekend-action-status', 'children'),
    [
        Input('btn-weekend-start', 'n_clicks'),
        Input('btn-weekend-pause', 'n_clicks'),
        Input('btn-weekend-skip', 'n_clicks'),
        Input('btn-weekend-stop', 'n_clicks'),
    ],
    [
        State('weekend-generations', 'value'),
        State('weekend-population', 'value'),
        State('weekend-discovery-toggle', 'value'),
        State('weekend-adaptive-toggle', 'value'),
        State('weekend-discovery-hours', 'value'),
        State('weekend-strategies', 'value'),
    ],
    prevent_initial_call=True,
)
def handle_weekend_action(start_clicks, pause_clicks, skip_clicks, stop_clicks,
                          generations, population, discovery_toggle, adaptive_toggle,
                          discovery_hours, strategies):
    """Handle weekend control panel action buttons."""
    from dash import ctx
    from pathlib import Path
    import json
    from datetime import datetime

    if not ctx.triggered:
        raise PreventUpdate

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    action = button_id.replace('btn-weekend-', '')

    # Build config from current settings
    config = {
        "generations": generations or 10,
        "population": population or 30,
        "discovery_enabled": "discovery" in (discovery_toggle or []),
        "adaptive_ga_enabled": "adaptive" in (adaptive_toggle or []),
        "discovery_hours": discovery_hours or 4,
        "strategies": strategies or [],
        "action": action,
        "requested_at": datetime.now().isoformat(),
    }

    # Write command to control file (orchestrator will read this)
    control_file = Path(__file__).parent.parent.parent / "logs" / "weekend_control.json"
    try:
        control_file.parent.mkdir(parents=True, exist_ok=True)
        with open(control_file, 'w') as f:
            json.dump(config, f, indent=2)

        if action == 'start':
            return f"Starting weekend research with {generations} generations, {population} population..."
        elif action == 'pause':
            return "Pausing weekend research..."
        elif action == 'skip':
            return "Skipping to next sub-phase..."
        elif action == 'stop':
            return "Stopping weekend research..."
        else:
            return f"Unknown action: {action}"

    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================================
# Main
# ============================================================================

# Register health check endpoint
register_health_endpoint(app)

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
