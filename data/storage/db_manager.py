"""
Database Manager
================
Handles all SQLite database operations.
Four databases:
- trades.db: Signals, trades, positions
- performance.db: Strategy and portfolio metrics
- research.db: Backtests, optimizations, discoveries
- pairs.db: Pairs trading specific data

Thread Safety:
- Uses per-thread connections via threading.local()
- Each thread gets its own SQLite connection (safe for concurrent access)
- Connections are created lazily and cached per-thread
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import json
import logging
import threading

# Import from parent
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATABASES, ensure_dirs

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages all database connections and operations.

    Thread Safety:
    - Uses threading.local() to maintain per-thread connections
    - Each thread gets its own connection to each database
    - Eliminates race conditions from shared connections
    """

    def __init__(self):
        ensure_dirs()
        # Thread-local storage for connections (each thread gets its own dict)
        self._local = threading.local()
        # Lock for schema initialization (one-time operation)
        self._init_lock = threading.Lock()
        self._initialized = False
        self._init_all_databases()

    def _get_thread_connections(self) -> Dict[str, sqlite3.Connection]:
        """Get the connection dict for the current thread."""
        if not hasattr(self._local, 'connections'):
            self._local.connections = {}
        return self._local.connections

    def close_thread_connections(self) -> None:
        """
        Close all database connections for the current thread.

        IMPORTANT: Call this before any multiprocessing fork() to prevent
        SQLite deadlocks. Forked processes inherit file descriptors, and
        SQLite connections become corrupted when children exit.

        Connections will be automatically recreated on next database access.
        """
        connections = self._get_thread_connections()
        closed = []
        for name, conn in list(connections.items()):
            try:
                conn.close()
                closed.append(name)
            except Exception as e:
                logger.warning(f"Error closing connection {name}: {e}")
        connections.clear()
        if closed:
            logger.debug(f"Closed DB connections before fork: {closed}")

    def _get_connection(self, db_name: str) -> sqlite3.Connection:
        """Get or create a connection to a database for the current thread."""
        connections = self._get_thread_connections()

        if db_name not in connections:
            try:
                db_path = DATABASES[db_name]
                db_path.parent.mkdir(parents=True, exist_ok=True)

                # Create connection for this thread (no check_same_thread needed
                # since each thread has its own connection)
                conn = sqlite3.connect(str(db_path), timeout=30.0)
                conn.row_factory = sqlite3.Row

                # Enable WAL mode for better concurrent read performance
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                # Set busy timeout to 30s (prevents indefinite hangs on lock contention)
                conn.execute("PRAGMA busy_timeout=30000")

                connections[db_name] = conn

                thread_id = threading.current_thread().name
                logger.debug(f"Database connection established: {db_name} ({db_path}) [thread={thread_id}]")
            except sqlite3.Error as e:
                logger.error(f"Failed to connect to database {db_name}: {e}")
                raise
            except KeyError:
                logger.error(f"Unknown database name: {db_name}")
                raise ValueError(f"Unknown database: {db_name}")
        return connections[db_name]

    def _init_all_databases(self):
        """Initialize all database schemas (thread-safe, runs once)."""
        # Use lock to ensure only one thread initializes schemas
        with self._init_lock:
            if self._initialized:
                return

            errors = []
            for db_name, init_func in [
                ('trades', self._init_trades_db),
                ('performance', self._init_performance_db),
                ('research', self._init_research_db),
                ('pairs', self._init_pairs_db),
                ('performance', self._init_orchestrator_state_log),  # Add state log to performance DB
            ]:
                try:
                    init_func()
                    logger.debug(f"Database schema initialized: {db_name}")
                except Exception as e:
                    logger.error(f"Failed to initialize {db_name} database: {e}")
                    errors.append((db_name, str(e)))

            if errors:
                logger.warning(f"Database initialization completed with {len(errors)} error(s)")
            else:
                logger.info("All databases initialized successfully")

            self._initialized = True
    
    # ========================================================================
    # TRADES DATABASE
    # ========================================================================
    
    def _init_trades_db(self):
        """Initialize trades.db schema."""
        try:
            conn = self._get_connection("trades")
            cursor = conn.cursor()

            # Signals table - all generated signals
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    signal_type TEXT NOT NULL,  -- BUY, SELL, CLOSE
                    strength REAL,
                    price REAL,
                    metadata TEXT,  -- JSON for strategy-specific data
                    executed INTEGER DEFAULT 0,
                    execution_id INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Trades table - executed trades
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id INTEGER,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    side TEXT NOT NULL,  -- BUY, SELL
                    quantity REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL,
                    target_price REAL,
                    exit_price REAL,
                    exit_timestamp TEXT,
                    peak_price REAL,
                    trough_price REAL,
                    pnl REAL,
                    pnl_pct REAL,
                    status TEXT DEFAULT 'OPEN',  -- OPEN, CLOSED, CANCELLED
                    exit_reason TEXT,
                    commission REAL DEFAULT 0,
                    slippage REAL DEFAULT 0,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (signal_id) REFERENCES signals(id)
                )
            """)

            # Positions table - current open positions (denormalized for speed)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    strategy TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_timestamp TEXT NOT NULL,
                    stop_loss REAL,
                    target_price REAL,
                    peak_price REAL,
                    trough_price REAL,
                    current_price REAL,
                    unrealized_pnl REAL,
                    unrealized_pnl_pct REAL,
                    trade_id INTEGER,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (trade_id) REFERENCES trades(id)
                )
            """)

            # Orders table - track order submissions and fills
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id INTEGER,
                    alpaca_order_id TEXT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT NOT NULL,  -- MARKET, LIMIT, STOP, etc.
                    quantity REAL NOT NULL,
                    limit_price REAL,
                    stop_price REAL,
                    status TEXT,  -- NEW, FILLED, PARTIALLY_FILLED, CANCELLED, etc.
                    filled_qty REAL,
                    filled_avg_price REAL,
                    submitted_at TEXT,
                    filled_at TEXT,
                    cancelled_at TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (trade_id) REFERENCES trades(id)
                )
            """)

            # Create indices
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_strategy ON signals(strategy)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")

            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize trades database schema: {e}")
            raise
    
    # ========================================================================
    # PERFORMANCE DATABASE
    # ========================================================================
    
    def _init_performance_db(self):
        """Initialize performance.db schema."""
        try:
            conn = self._get_connection("performance")
            cursor = conn.cursor()

            # Strategy daily performance
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_daily (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                strategy TEXT NOT NULL,
                trades_opened INTEGER DEFAULT 0,
                trades_closed INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                gross_pnl REAL DEFAULT 0,
                net_pnl REAL DEFAULT 0,
                commission REAL DEFAULT 0,
                sharpe_20d REAL,
                sharpe_60d REAL,
                win_rate_30d REAL,
                avg_win REAL,
                avg_loss REAL,
                profit_factor REAL,
                max_drawdown REAL,
                is_enabled INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, strategy)
            )
            """)

            # Portfolio daily snapshot
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_daily (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL UNIQUE,
                equity REAL NOT NULL,
                cash REAL NOT NULL,
                positions_value REAL NOT NULL,
                num_positions INTEGER,
                exposure_pct REAL,
                daily_pnl REAL,
                daily_pnl_pct REAL,
                cumulative_pnl REAL,
                cumulative_pnl_pct REAL,
                drawdown REAL,
                drawdown_pct REAL,
                high_water_mark REAL,
                sharpe_ratio REAL,
                sortino_ratio REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # Regime log - track VIX regime changes and actions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS regime_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    vix_level REAL NOT NULL,
                    vix_regime TEXT NOT NULL,  -- low, normal, high, extreme
                    previous_regime TEXT,
                    action_taken TEXT,  -- e.g., "reduced exposure 50%"
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Strategy cumulative stats (updated periodically)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategy_stats (
                    strategy TEXT PRIMARY KEY,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    losing_trades INTEGER DEFAULT 0,
                    total_pnl REAL DEFAULT 0,
                    avg_pnl REAL DEFAULT 0,
                    avg_pnl_pct REAL DEFAULT 0,
                    win_rate REAL DEFAULT 0,
                    profit_factor REAL,
                    sharpe_ratio REAL,
                    sortino_ratio REAL,
                    max_drawdown REAL,
                    max_drawdown_pct REAL,
                    avg_hold_time_hours REAL,
                    best_trade REAL,
                    worst_trade REAL,
                    is_enabled INTEGER DEFAULT 1,
                    last_trade_date TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indices
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_strategy_daily_date ON strategy_daily(date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_strategy_daily_strategy ON strategy_daily(strategy)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_daily_date ON portfolio_daily(date)")

            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize performance database schema: {e}")
            raise

    # ========================================================================
    # RESEARCH DATABASE
    # ========================================================================

    def _init_research_db(self):
        """Initialize research.db schema."""
        try:
            conn = self._get_connection("research")
            cursor = conn.cursor()

            # Backtest results
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,  -- UUID for grouping related tests
                timestamp TEXT NOT NULL,
                strategy TEXT NOT NULL,
                symbol TEXT,  -- NULL for portfolio-level backtests
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                params TEXT,  -- JSON of strategy parameters
                
                -- Performance metrics
                total_return REAL,
                annual_return REAL,
                sharpe_ratio REAL,
                sortino_ratio REAL,
                max_drawdown REAL,
                max_drawdown_pct REAL,
                calmar_ratio REAL,
                
                -- Trade statistics
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                win_rate REAL,
                profit_factor REAL,
                avg_trade_pnl REAL,
                avg_winner REAL,
                avg_loser REAL,
                
                -- Risk metrics
                volatility REAL,
                var_95 REAL,
                cvar_95 REAL,
                beta REAL,
                alpha REAL,
                
                -- Validation
                meets_threshold INTEGER,  -- 1 if meets min_sharpe
                vs_research_pct REAL,     -- % of research benchmark achieved
                
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # Parameter optimization results
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS optimizations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    param_name TEXT NOT NULL,
                    param_type TEXT,
                    old_value TEXT,
                    new_value TEXT,
                    old_sharpe REAL,
                    new_sharpe REAL,
                    improvement_pct REAL,
                    sample_size INTEGER,
                    confidence REAL,
                    p_value REAL,
                    applied INTEGER DEFAULT 0,
                    applied_at TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Genetic algorithm discoveries
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS discoveries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    individual_id TEXT NOT NULL,
                    chromosome TEXT NOT NULL,
                    fitness REAL NOT NULL,
                    sharpe_ratio REAL,
                    total_return REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    in_sample_sharpe REAL,
                    out_of_sample_sharpe REAL,
                    oos_ratio REAL,
                    status TEXT DEFAULT 'candidate',
                    code_path TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indices
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_backtests_strategy ON backtests(strategy)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_backtests_run_id ON backtests(run_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_optimizations_strategy ON optimizations(strategy)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_discoveries_generation ON discoveries(generation)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_discoveries_fitness ON discoveries(fitness)")

            # GA populations
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ga_populations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    population_json TEXT NOT NULL,
                    best_fitness REAL,
                    best_genes_json TEXT,
                    population_size INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(strategy, generation)
                )
            """)

            # GA evolution history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ga_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    best_fitness REAL NOT NULL,
                    mean_fitness REAL,
                    std_fitness REAL,
                    best_genes_json TEXT,
                    generations_without_improvement INTEGER DEFAULT 0,
                    run_date TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(strategy, generation, run_date)
                )
            """)

            # GA run log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ga_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL UNIQUE,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    strategies_evolved TEXT,
                    total_generations INTEGER DEFAULT 0,
                    improvements_found INTEGER DEFAULT 0,
                    errors TEXT,
                    status TEXT DEFAULT 'running',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create GA indices
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ga_populations_strategy ON ga_populations(strategy)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ga_history_strategy ON ga_history(strategy)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ga_history_run_date ON ga_history(run_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ga_runs_status ON ga_runs(status)")

            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize research database schema: {e}")
            raise

    # ========================================================================
    # PAIRS DATABASE
    # ========================================================================

    def _init_pairs_db(self):
        """Initialize pairs.db schema."""
        try:
            conn = self._get_connection("pairs")
            cursor = conn.cursor()

            # Qualified pairs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pairs (
                    pair_id TEXT PRIMARY KEY,
                    symbol_a TEXT NOT NULL,
                    symbol_b TEXT NOT NULL,
                    sector TEXT,
                    industry TEXT,
                    coint_pvalue REAL,
                    adf_statistic REAL,
                    half_life REAL,
                    correlation REAL,
                    correlation_60d REAL,
                    hedge_ratio REAL,
                    hedge_ratio_std REAL,
                    spread_mean REAL,
                    spread_std REAL,
                    z_score_current REAL,
                    is_active INTEGER DEFAULT 1,
                    last_tested TEXT,
                    last_traded TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Spread history for each pair
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS spread_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    price_a REAL,
                    price_b REAL,
                    spread REAL,
                    z_score REAL,
                    position_status TEXT,
                    entry_z_score REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (pair_id) REFERENCES pairs(pair_id),
                    UNIQUE(pair_id, date)
                )
            """)

            # Pair trades
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pair_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair_id TEXT NOT NULL,
                    entry_timestamp TEXT NOT NULL,
                    exit_timestamp TEXT,
                    direction TEXT NOT NULL,
                    entry_z_score REAL,
                    entry_price_a REAL,
                    entry_price_b REAL,
                    quantity_a REAL,
                    quantity_b REAL,
                    exit_z_score REAL,
                    exit_price_a REAL,
                    exit_price_b REAL,
                    exit_reason TEXT,
                    pnl_a REAL,
                    pnl_b REAL,
                    total_pnl REAL,
                    total_pnl_pct REAL,
                    status TEXT DEFAULT 'OPEN',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (pair_id) REFERENCES pairs(pair_id)
                )
            """)

            # Create indices
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_spread_history_pair ON spread_history(pair_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_spread_history_date ON spread_history(date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pair_trades_status ON pair_trades(status)")

            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize pairs database schema: {e}")
            raise

    # ========================================================================
    # ORCHESTRATOR STATE LOG
    # ========================================================================

    def _init_orchestrator_state_log(self):
        """Initialize orchestrator state log in performance.db.

        This append-only log tracks:
        - Phase transitions (PRE_MARKET, TRADING, etc.)
        - Sub-phase transitions (weekend: FRIDAY_CLEANUP, RESEARCH, etc.)
        - Task completions (with success/failure status)

        Benefits:
        - State persistence across restarts (avoids redundant task execution)
        - Audit trail for debugging and timing analysis
        - Lightweight (just inserts, no complex queries during normal operation)
        """
        try:
            conn = self._get_connection("performance")
            cursor = conn.cursor()

            # Orchestrator state log - append-only for persistence + audit
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS orchestrator_state_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT (datetime('now')),
                    run_date TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    phase TEXT,
                    sub_phase TEXT,
                    task_name TEXT,
                    success INTEGER,
                    details TEXT
                )
            """)

            # Indices for efficient queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_state_log_run_date ON orchestrator_state_log(run_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_state_log_event_type ON orchestrator_state_log(event_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_state_log_task ON orchestrator_state_log(task_name)")

            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize orchestrator state log: {e}")
            raise

    # ========================================================================
    # COMMON OPERATIONS
    # ========================================================================

    def execute(self, db_name: str, query: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a query on the specified database."""
        try:
            conn = self._get_connection(db_name)
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor
        except sqlite3.Error as e:
            logger.error(f"Database execute failed: db={db_name}, query={query[:100]}... - {e}")
            raise

    def fetchone(self, db_name: str, query: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        """Fetch one row from the specified database."""
        try:
            conn = self._get_connection(db_name)
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchone()
        except sqlite3.Error as e:
            logger.error(f"Database fetchone failed: db={db_name}, query={query[:100]}... - {e}")
            raise

    def fetchall(self, db_name: str, query: str, params: tuple = ()) -> List[sqlite3.Row]:
        """Fetch all rows from the specified database."""
        try:
            conn = self._get_connection(db_name)
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Database fetchall failed: db={db_name}, query={query[:100]}... - {e}")
            raise

    def close_all(self):
        """Close all database connections for the current thread."""
        connections = self._get_thread_connections()
        for db_name, conn in list(connections.items()):
            try:
                conn.close()
                logger.debug(f"Closed connection to {db_name} [thread={threading.current_thread().name}]")
            except Exception as e:
                logger.warning(f"Error closing connection to {db_name}: {e}")
        connections.clear()

    def close_all_threads(self):
        """
        Close connections for all threads (call on shutdown).

        Note: This only works for threads that have registered their connections.
        For complete cleanup, each thread should call close_all() before exiting.
        """
        # Close current thread's connections
        self.close_all()
        logger.info("Database connections closed for current thread")
    
    # ========================================================================
    # TRADE OPERATIONS
    # ========================================================================
    
    def log_signal(self, symbol: str, strategy: str, signal_type: str,
                   strength: float, price: float, metadata: dict = None) -> int:
        """Log a trading signal."""
        cursor = self.execute(
            "trades",
            """
            INSERT INTO signals (timestamp, symbol, strategy, signal_type, 
                                strength, price, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (datetime.now().isoformat(), symbol, strategy, signal_type,
             strength, price, json.dumps(metadata) if metadata else None)
        )
        return cursor.lastrowid
    
    def log_trade(self, symbol: str, strategy: str, side: str, quantity: float,
                  entry_price: float, stop_loss: float = None, 
                  target_price: float = None, signal_id: int = None) -> int:
        """Log a new trade."""
        cursor = self.execute(
            "trades",
            """
            INSERT INTO trades (signal_id, timestamp, symbol, strategy, side,
                               quantity, entry_price, stop_loss, target_price,
                               peak_price, trough_price, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
            """,
            (signal_id, datetime.now().isoformat(), symbol, strategy, side,
             quantity, entry_price, stop_loss, target_price,
             entry_price, entry_price)
        )
        return cursor.lastrowid
    
    def close_trade(self, trade_id: int, exit_price: float, 
                    exit_reason: str) -> None:
        """Close an open trade."""
        # Get trade details
        trade = self.fetchone(
            "trades",
            "SELECT entry_price, quantity, side FROM trades WHERE id = ?",
            (trade_id,)
        )
        
        if not trade:
            raise ValueError(f"Trade {trade_id} not found")
        
        entry_price, quantity, side = trade

        # Validate entry_price to prevent division by zero
        if entry_price is None or entry_price <= 0:
            raise ValueError(f"Invalid entry_price {entry_price} for trade {trade_id}")

        # Calculate P&L
        if side == 'BUY':
            pnl = (exit_price - entry_price) * quantity
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        else:
            pnl = (entry_price - exit_price) * quantity
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100
        
        self.execute(
            "trades",
            """
            UPDATE trades 
            SET exit_price = ?, exit_timestamp = ?, pnl = ?, pnl_pct = ?,
                status = 'CLOSED', exit_reason = ?, updated_at = ?
            WHERE id = ?
            """,
            (exit_price, datetime.now().isoformat(), pnl, pnl_pct,
             exit_reason, datetime.now().isoformat(), trade_id)
        )
    
    def get_open_trades(self, strategy: str = None) -> List[sqlite3.Row]:
        """Get all open trades, optionally filtered by strategy."""
        if strategy:
            return self.fetchall(
                "trades",
                "SELECT * FROM trades WHERE status = 'OPEN' AND strategy = ?",
                (strategy,)
            )
        return self.fetchall(
            "trades",
            "SELECT * FROM trades WHERE status = 'OPEN'"
        )
    
    def get_open_positions(self) -> List[sqlite3.Row]:
        """Get all open positions."""
        return self.fetchall("trades", "SELECT * FROM positions")
    
    # ========================================================================
    # PERFORMANCE OPERATIONS
    # ========================================================================
    
    def update_strategy_stats(self, strategy: str) -> None:
        """Update cumulative stats for a strategy."""
        trades = self.fetchall(
            "trades",
            """
            SELECT pnl, pnl_pct, timestamp, exit_timestamp
            FROM trades 
            WHERE strategy = ? AND status = 'CLOSED'
            """,
            (strategy,)
        )
        
        if not trades:
            return
        
        total = len(trades)
        winners = sum(1 for t in trades if t['pnl'] > 0)
        total_pnl = sum(t['pnl'] for t in trades)
        avg_pnl = total_pnl / total if total > 0 else 0
        avg_pnl_pct = sum(t['pnl_pct'] for t in trades) / total if total > 0 else 0
        win_rate = winners / total if total > 0 else 0
        
        self.execute(
            "performance",
            """
            INSERT OR REPLACE INTO strategy_stats 
            (strategy, total_trades, winning_trades, losing_trades,
             total_pnl, avg_pnl, avg_pnl_pct, win_rate, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (strategy, total, winners, total - winners, total_pnl,
             avg_pnl, avg_pnl_pct, win_rate, datetime.now().isoformat())
        )
    
    def log_portfolio_snapshot(self, equity: float, cash: float,
                               positions_value: float, num_positions: int) -> None:
        """Log daily portfolio snapshot."""
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Get previous snapshot for calculations
        prev = self.fetchone(
            "performance",
            "SELECT equity, high_water_mark FROM portfolio_daily ORDER BY date DESC LIMIT 1"
        )
        
        prev_equity = prev['equity'] if prev else equity
        hwm = max(prev['high_water_mark'] if prev else equity, equity)
        
        daily_pnl = equity - prev_equity
        daily_pnl_pct = (daily_pnl / prev_equity * 100) if prev_equity > 0 else 0
        drawdown = hwm - equity
        drawdown_pct = (drawdown / hwm * 100) if hwm > 0 else 0
        exposure_pct = (positions_value / equity * 100) if equity > 0 else 0
        
        self.execute(
            "performance",
            """
            INSERT OR REPLACE INTO portfolio_daily
            (date, equity, cash, positions_value, num_positions, exposure_pct,
             daily_pnl, daily_pnl_pct, drawdown, drawdown_pct, high_water_mark)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (today, equity, cash, positions_value, num_positions, exposure_pct,
             daily_pnl, daily_pnl_pct, drawdown, drawdown_pct, hwm)
        )
    
    def log_regime_change(self, vix_level: float, vix_regime: str,
                          previous_regime: str = None, action: str = None) -> None:
        """Log VIX regime change."""
        self.execute(
            "performance",
            """
            INSERT INTO regime_log (timestamp, vix_level, vix_regime, 
                                   previous_regime, action_taken)
            VALUES (?, ?, ?, ?, ?)
            """,
            (datetime.now().isoformat(), vix_level, vix_regime,
             previous_regime, action)
        )
    
    # ========================================================================
    # RESEARCH OPERATIONS
    # ========================================================================
    
    def log_backtest(self, run_id: str, strategy: str, start_date: str,
                     end_date: str, metrics: dict, params: dict = None,
                     symbol: str = None) -> int:
        """Log backtest results."""
        cursor = self.execute(
            "research",
            """
            INSERT INTO backtests 
            (run_id, timestamp, strategy, symbol, start_date, end_date, params,
             total_return, annual_return, sharpe_ratio, sortino_ratio,
             max_drawdown, max_drawdown_pct, total_trades, winning_trades,
             losing_trades, win_rate, profit_factor, avg_trade_pnl,
             volatility, meets_threshold, vs_research_pct, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, datetime.now().isoformat(), strategy, symbol,
             start_date, end_date, json.dumps(params) if params else None,
             metrics.get('total_return'), metrics.get('annual_return'),
             metrics.get('sharpe_ratio'), metrics.get('sortino_ratio'),
             metrics.get('max_drawdown'), metrics.get('max_drawdown_pct'),
             metrics.get('total_trades'), metrics.get('winning_trades'),
             metrics.get('losing_trades'), metrics.get('win_rate'),
             metrics.get('profit_factor'), metrics.get('avg_trade_pnl'),
             metrics.get('volatility'), metrics.get('meets_threshold'),
             metrics.get('vs_research_pct'), json.dumps(metrics.get('metadata')))
        )
        return cursor.lastrowid
    
    # ========================================================================
    # GENETIC ALGORITHM OPERATIONS
    # ========================================================================
    
    def save_ga_population(self, strategy: str, generation: int, 
                           population: list, best_fitness: float,
                           best_genes: dict) -> int:
        """Save GA population state for a strategy."""
        cursor = self.execute(
            "research",
            """
            INSERT OR REPLACE INTO ga_populations 
            (strategy, generation, population_json, best_fitness, 
             best_genes_json, population_size)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (strategy, generation, json.dumps(population), best_fitness,
             json.dumps(best_genes), len(population))
        )
        return cursor.lastrowid
    
    def load_ga_population(self, strategy: str) -> Optional[Dict]:
        """Load the most recent GA population for a strategy."""
        row = self.fetchone(
            "research",
            """
            SELECT generation, population_json, best_fitness, best_genes_json,
                   population_size, created_at
            FROM ga_populations 
            WHERE strategy = ? 
            ORDER BY generation DESC 
            LIMIT 1
            """,
            (strategy,)
        )
        
        if row:
            return {
                'generation': row['generation'],
                'population': json.loads(row['population_json']),
                'best_fitness': row['best_fitness'],
                'best_genes': json.loads(row['best_genes_json']) if row['best_genes_json'] else None,
                'population_size': row['population_size'],
                'created_at': row['created_at']
            }
        return None
    
    def log_ga_history(self, strategy: str, generation: int, 
                       best_fitness: float, mean_fitness: float,
                       std_fitness: float, best_genes: dict,
                       generations_without_improvement: int = 0) -> int:
        """Log GA evolution history for tracking progress."""
        run_date = datetime.now().strftime('%Y-%m-%d')
        cursor = self.execute(
            "research",
            """
            INSERT OR REPLACE INTO ga_history 
            (strategy, generation, best_fitness, mean_fitness, std_fitness,
             best_genes_json, generations_without_improvement, run_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (strategy, generation, best_fitness, mean_fitness, std_fitness,
             json.dumps(best_genes), generations_without_improvement, run_date)
        )
        return cursor.lastrowid
    
    def get_ga_history(self, strategy: str, days: int = 30) -> List[sqlite3.Row]:
        """Get GA evolution history for a strategy."""
        return self.fetchall(
            "research",
            """
            SELECT * FROM ga_history 
            WHERE strategy = ? 
              AND created_at >= datetime('now', '-' || ? || ' days')
            ORDER BY generation ASC
            """,
            (strategy, days)
        )
    
    def get_ga_best_all_time(self, strategy: str) -> Optional[Dict]:
        """Get the best fitness ever achieved for a strategy."""
        row = self.fetchone(
            "research",
            """
            SELECT best_fitness, best_genes_json, generation, run_date
            FROM ga_history 
            WHERE strategy = ? 
            ORDER BY best_fitness DESC 
            LIMIT 1
            """,
            (strategy,)
        )
        
        if row:
            return {
                'best_fitness': row['best_fitness'],
                'best_genes': json.loads(row['best_genes_json']) if row['best_genes_json'] else None,
                'generation': row['generation'],
                'run_date': row['run_date']
            }
        return None
    
    def start_ga_run(self, run_id: str, strategies: list = None, planned_generations: int = 1) -> int:
        """Log the start of a GA run."""
        strategies_str = ','.join(strategies) if strategies else None
        cursor = self.execute(
            "research",
            """
            INSERT INTO ga_runs (run_id, start_time, status, strategies_evolved, planned_generations)
            VALUES (?, ?, 'running', ?, ?)
            """,
            (run_id, datetime.now().isoformat(), strategies_str, planned_generations)
        )
        return cursor.lastrowid
    
    def complete_ga_run(self, run_id: str, strategies: list, 
                        total_generations: int, improvements: int,
                        errors: list = None) -> None:
        """Mark a GA run as complete."""
        status = 'completed' if not errors else 'completed_with_errors'
        self.execute(
            "research",
            """
            UPDATE ga_runs 
            SET end_time = ?, strategies_evolved = ?, total_generations = ?,
                improvements_found = ?, errors = ?, status = ?
            WHERE run_id = ?
            """,
            (datetime.now().isoformat(), json.dumps(strategies),
             total_generations, improvements, 
             json.dumps(errors) if errors else None, status, run_id)
        )
    
    def fail_ga_run(self, run_id: str, error: str) -> None:
        """Mark a GA run as failed."""
        self.execute(
            "research",
            """
            UPDATE ga_runs
            SET end_time = ?, errors = ?, status = 'failed'
            WHERE run_id = ?
            """,
            (datetime.now().isoformat(), json.dumps([error]), run_id)
        )

    def interrupt_ga_run(self, run_id: str, reason: str = "Process terminated") -> None:
        """Mark a GA run as interrupted (can be resumed)."""
        self.execute(
            "research",
            """
            UPDATE ga_runs
            SET end_time = ?, errors = ?, status = 'interrupted'
            WHERE run_id = ? AND status = 'running'
            """,
            (datetime.now().isoformat(), json.dumps([reason]), run_id)
        )

    def cleanup_stale_ga_runs(self) -> int:
        """Mark any 'running' GA runs as interrupted (for startup cleanup).

        Returns:
            Number of runs marked as interrupted
        """
        cursor = self.execute(
            "research",
            """
            UPDATE ga_runs
            SET end_time = ?, errors = ?, status = 'interrupted'
            WHERE status = 'running'
            """,
            (datetime.now().isoformat(), json.dumps(["Process terminated unexpectedly"]))
        )
        return cursor.rowcount if cursor else 0

    def get_recent_ga_runs(self, limit: int = 10) -> List[sqlite3.Row]:
        """Get recent GA run history."""
        return self.fetchall(
            "research",
            """
            SELECT * FROM ga_runs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,)
        )

    # ========================================================================
    # ORCHESTRATOR STATE LOG OPERATIONS
    # ========================================================================

    def log_state_event(self, run_date: str, event_type: str,
                        phase: str = None, sub_phase: str = None,
                        task_name: str = None, success: bool = None,
                        details: str = None) -> int:
        """
        Log an orchestrator state event.

        Args:
            run_date: Grouping key (e.g., 'weekend_2026-01-10', '2026-01-10')
            event_type: 'task_complete', 'phase_enter', 'phase_exit', 'error'
            phase: Market phase (e.g., 'weekend', 'overnight')
            sub_phase: Weekend sub-phase (e.g., 'friday_cleanup', 'research')
            task_name: Name of completed task (if event_type='task_complete')
            success: True if task succeeded
            details: Optional JSON string with additional context

        Returns:
            Row ID of inserted event
        """
        cursor = self.execute(
            "performance",
            """
            INSERT INTO orchestrator_state_log
            (run_date, event_type, phase, sub_phase, task_name, success, details)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (run_date, event_type, phase, sub_phase, task_name,
             1 if success else (0 if success is False else None), details)
        )
        return cursor.lastrowid

    def get_completed_tasks(self, run_date: str, phase: str = None,
                            sub_phase: str = None) -> List[str]:
        """
        Get list of successfully completed tasks for a run date.

        Args:
            run_date: The run date key to query
            phase: Optional filter by phase
            sub_phase: Optional filter by sub-phase

        Returns:
            List of task names that completed successfully
        """
        query = """
            SELECT DISTINCT task_name FROM orchestrator_state_log
            WHERE run_date = ?
              AND event_type = 'task_complete'
              AND success = 1
              AND task_name IS NOT NULL
        """
        params = [run_date]

        if phase:
            query += " AND phase = ?"
            params.append(phase)

        if sub_phase:
            query += " AND sub_phase = ?"
            params.append(sub_phase)

        rows = self.fetchall("performance", query, tuple(params))
        return [row['task_name'] for row in rows]

    def get_state_log_events(self, run_date: str, event_type: str = None,
                             limit: int = 100) -> List[sqlite3.Row]:
        """
        Get state log events for a run date.

        Args:
            run_date: The run date key to query
            event_type: Optional filter by event type
            limit: Max events to return

        Returns:
            List of event rows
        """
        if event_type:
            return self.fetchall(
                "performance",
                """
                SELECT * FROM orchestrator_state_log
                WHERE run_date = ? AND event_type = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (run_date, event_type, limit)
            )
        else:
            return self.fetchall(
                "performance",
                """
                SELECT * FROM orchestrator_state_log
                WHERE run_date = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (run_date, limit)
            )

    def get_last_phase_state(self, run_date: str) -> Optional[Dict[str, Any]]:
        """
        Get the most recent phase/sub_phase state for a run date.

        Returns:
            Dict with 'phase', 'sub_phase', 'timestamp' or None
        """
        row = self.fetchone(
            "performance",
            """
            SELECT phase, sub_phase, timestamp FROM orchestrator_state_log
            WHERE run_date = ? AND event_type IN ('phase_enter', 'subphase_enter')
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (run_date,)
        )
        if row:
            return {
                'phase': row['phase'],
                'sub_phase': row['sub_phase'],
                'timestamp': row['timestamp']
            }
        return None


# Singleton instance
_db_manager: Optional[DatabaseManager] = None

def get_db() -> DatabaseManager:
    """Get the singleton database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


if __name__ == "__main__":
    # Test database initialization
    print("Initializing databases...")
    db = get_db()
    
    print("\nDatabase files created:")
    for name, path in DATABASES.items():
        status = "✓" if path.exists() else "✗"
        print(f"  {status} {name}: {path}")
    
    print("\nTesting operations...")
    
    # Test signal logging
    signal_id = db.log_signal("AAPL", "test_strategy", "BUY", 0.75, 150.00)
    print(f"  Logged signal ID: {signal_id}")
    
    # Test trade logging
    trade_id = db.log_trade("AAPL", "test_strategy", "BUY", 10, 150.00, 
                            stop_loss=145.00, target_price=160.00,
                            signal_id=signal_id)
    print(f"  Logged trade ID: {trade_id}")
    
    # Test trade close
    db.close_trade(trade_id, 155.00, "target_reached")
    print(f"  Closed trade ID: {trade_id}")
    
    # Verify
    trade = db.fetchone("trades", "SELECT * FROM trades WHERE id = ?", (trade_id,))
    print(f"  Trade P&L: ${trade['pnl']:.2f} ({trade['pnl_pct']:.2f}%)")
    
    print("\n✓ All database operations working")
