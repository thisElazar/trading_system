"""
Logging Configuration
=====================
Centralized logging setup for the trading system.

Features:
- Console and file logging
- Log rotation
- Different levels for different components
- Structured log format
- Database logging for errors/warnings (for dashboard display)
"""

import logging
import sys
import sqlite3
import traceback
import threading
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from queue import Queue, Empty
from typing import Optional

# Import from parent
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DIRS, LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT, LOG_MAX_BYTES, LOG_BACKUP_COUNT, DATABASES


class DatabaseErrorHandler(logging.Handler):
    """
    Custom logging handler that writes ERROR and WARNING logs to SQLite database.

    This enables the dashboard to display system errors/warnings in real-time.
    Uses a background thread and queue for non-blocking database writes.
    """

    def __init__(self, db_path: Optional[Path] = None, min_level: int = logging.WARNING):
        super().__init__(min_level)
        self.db_path = db_path or DATABASES.get('performance')
        self._queue: Queue = Queue()
        self._shutdown = False
        self._worker: Optional[threading.Thread] = None

        if self.db_path and self.db_path.exists():
            self._start_worker()

    def _start_worker(self):
        """Start background worker thread for database writes."""
        self._worker = threading.Thread(target=self._process_queue, daemon=True)
        self._worker.start()

    def _process_queue(self):
        """Background worker that writes log records to database."""
        while not self._shutdown:
            try:
                record = self._queue.get(timeout=1.0)
                if record is None:
                    break
                self._write_to_db(record)
            except Empty:
                continue
            except Exception:
                pass  # Silently fail - don't recursively log

    def _get_component(self, logger_name: str) -> str:
        """Extract component from logger name."""
        parts = logger_name.split('.')
        if len(parts) >= 2:
            return parts[1]  # e.g., 'trading_system.execution' -> 'execution'
        return 'system'

    def _write_to_db(self, record: logging.LogRecord):
        """Write a log record to the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Extract exception info if present
            exception_type = None
            exception_traceback = None
            if record.exc_info:
                exception_type = record.exc_info[0].__name__ if record.exc_info[0] else None
                exception_traceback = ''.join(traceback.format_exception(*record.exc_info))

            cursor.execute("""
                INSERT INTO error_log
                (timestamp, level, logger_name, message, source_file, line_number,
                 exception_type, exception_traceback, component)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                record.levelname,
                record.name,
                record.getMessage(),
                record.pathname,
                record.lineno,
                exception_type,
                exception_traceback,
                self._get_component(record.name)
            ))

            conn.commit()
            conn.close()
        except Exception:
            pass  # Silently fail - can't log errors about logging

    def emit(self, record: logging.LogRecord):
        """Queue a log record for database writing."""
        if self.db_path and self.db_path.exists():
            self._queue.put(record)

    def close(self):
        """Shutdown the handler and worker thread."""
        self._shutdown = True
        self._queue.put(None)
        if self._worker:
            self._worker.join(timeout=2.0)
        super().close()


def setup_logging(
    name: str = "trading_system",
    level: str = None,
    log_to_file: bool = True,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Set up logging for the trading system.
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        
    Returns:
        Configured logger
    """
    level = level or LOG_LEVEL
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        log_dir = DIRS.get("logs")
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Main log file (rotating)
            log_file = log_dir / f"{name}.log"
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=LOG_MAX_BYTES,
                backupCount=LOG_BACKUP_COUNT
            )
            file_handler.setLevel(getattr(logging, level.upper()))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            # Error log file (separate)
            error_file = log_dir / f"{name}_errors.log"
            error_handler = RotatingFileHandler(
                error_file,
                maxBytes=LOG_MAX_BYTES,
                backupCount=LOG_BACKUP_COUNT
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            logger.addHandler(error_handler)

    # Database handler for dashboard display (warnings and errors)
    db_handler = DatabaseErrorHandler(min_level=logging.WARNING)
    db_handler.setFormatter(formatter)
    logger.addHandler(db_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        name: Module name (e.g., 'strategies.vol_momentum')
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"trading_system.{name}")


def setup_trade_logger() -> logging.Logger:
    """
    Set up a dedicated trade logger.
    
    Logs all trades to a separate file for audit purposes.
    
    Returns:
        Trade logger
    """
    logger = logging.getLogger("trading_system.trades")
    logger.setLevel(logging.INFO)
    
    log_dir = DIRS.get("logs")
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Trade log with daily rotation
        trade_file = log_dir / "trades.log"
        handler = TimedRotatingFileHandler(
            trade_file,
            when='midnight',
            interval=1,
            backupCount=30  # Keep 30 days of trade logs
        )
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(handler)
    
    return logger


def log_trade(symbol: str, action: str, qty: float, price: float,
              strategy: str = "", reason: str = ""):
    """
    Log a trade to the trade log.
    
    Args:
        symbol: Stock symbol
        action: BUY, SELL, CLOSE
        qty: Number of shares
        price: Execution price
        strategy: Strategy name
        reason: Trade reason
    """
    trade_logger = logging.getLogger("trading_system.trades")
    
    msg = f"{action} | {symbol} | {qty} shares @ ${price:.2f}"
    if strategy:
        msg += f" | {strategy}"
    if reason:
        msg += f" | {reason}"
    
    trade_logger.info(msg)


class LogContext:
    """
    Context manager for temporary log level changes.
    
    Usage:
        with LogContext(logging.DEBUG):
            # Debug logging enabled here
            ...
        # Back to normal level
    """
    
    def __init__(self, level: int):
        self.level = level
        self.previous_level = None
    
    def __enter__(self):
        logger = logging.getLogger("trading_system")
        self.previous_level = logger.level
        logger.setLevel(self.level)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logger = logging.getLogger("trading_system")
        logger.setLevel(self.previous_level)
        return False


# Initialize logging on import
_root_logger = None

def init():
    """Initialize the logging system."""
    global _root_logger
    if _root_logger is None:
        _root_logger = setup_logging()
    return _root_logger


if __name__ == "__main__":
    # Test logging setup
    logger = setup_logging(level="DEBUG")
    
    print("Testing logging...")
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test trade logging
    setup_trade_logger()
    log_trade("AAPL", "BUY", 100, 175.50, "vol_momentum", "12-1 momentum signal")
    log_trade("MSFT", "SELL", 50, 375.25, "vix_regime", "Regime rotation")
    
    print("\nLog files created in:", DIRS.get("logs"))
