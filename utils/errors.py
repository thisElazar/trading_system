"""
Unified Error Handling Utilities
================================
Provides consistent error handling, custom exceptions, and decorators
for error logging and context management.

Usage:
    from utils.errors import (
        TradingSystemError, ExecutionError, DataFetchError,
        log_errors, error_context
    )

    # Custom exceptions
    raise ExecutionError("Order failed", symbol="AAPL", order_id="123")

    # Decorator for error logging
    @log_errors(reraise=True)
    def risky_function():
        ...

    # Context manager for error context
    with error_context("submitting order", symbol="AAPL"):
        submit_order()

Design Principles:
- Clear exception hierarchy for different error types
- Structured error context (symbol, operation, etc.)
- Consistent logging format
- No silent failures (explicit reraise control)
"""

import logging
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple, Type, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# EXCEPTION HIERARCHY
# =============================================================================

class TradingSystemError(Exception):
    """
    Base exception for all trading system errors.

    Supports structured context for logging and debugging.

    Example:
        raise TradingSystemError(
            "Operation failed",
            operation="fetch_data",
            symbol="AAPL",
            details={"response_code": 500}
        )
    """

    def __init__(
        self,
        message: str,
        *,
        operation: Optional[str] = None,
        symbol: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        self.message = message
        self.operation = operation
        self.symbol = symbol
        self.details = details or {}
        self.cause = cause

        # Build full message
        parts = [message]
        if operation:
            parts.append(f"operation={operation}")
        if symbol:
            parts.append(f"symbol={symbol}")
        if details:
            parts.append(f"details={details}")

        super().__init__(" | ".join(parts))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "operation": self.operation,
            "symbol": self.symbol,
            "details": self.details,
            "cause": str(self.cause) if self.cause else None
        }


class TimeoutError(TradingSystemError):
    """Raised when an operation times out."""

    def __init__(
        self,
        message: str = "Operation timed out",
        timeout_seconds: Optional[float] = None,
        **kwargs
    ):
        self.timeout_seconds = timeout_seconds
        if timeout_seconds:
            kwargs.setdefault('details', {})['timeout_seconds'] = timeout_seconds
        super().__init__(message, **kwargs)


class ExecutionError(TradingSystemError):
    """Raised when order execution fails."""

    def __init__(
        self,
        message: str = "Execution failed",
        order_id: Optional[str] = None,
        side: Optional[str] = None,
        quantity: Optional[int] = None,
        **kwargs
    ):
        self.order_id = order_id
        self.side = side
        self.quantity = quantity

        kwargs.setdefault('details', {})
        if order_id:
            kwargs['details']['order_id'] = order_id
        if side:
            kwargs['details']['side'] = side
        if quantity:
            kwargs['details']['quantity'] = quantity

        super().__init__(message, **kwargs)


class DataFetchError(TradingSystemError):
    """Raised when data fetching fails."""

    def __init__(
        self,
        message: str = "Data fetch failed",
        source: Optional[str] = None,
        **kwargs
    ):
        self.source = source
        if source:
            kwargs.setdefault('details', {})['source'] = source
        super().__init__(message, **kwargs)


class DatabaseError(TradingSystemError):
    """Raised when database operations fail."""

    def __init__(
        self,
        message: str = "Database error",
        db_name: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs
    ):
        self.db_name = db_name
        self.query = query

        kwargs.setdefault('details', {})
        if db_name:
            kwargs['details']['db_name'] = db_name
        if query:
            # Truncate long queries
            kwargs['details']['query'] = query[:200] + "..." if len(query) > 200 else query

        super().__init__(message, **kwargs)


class ConfigurationError(TradingSystemError):
    """Raised when configuration is invalid or missing."""

    def __init__(
        self,
        message: str = "Configuration error",
        config_key: Optional[str] = None,
        **kwargs
    ):
        self.config_key = config_key
        if config_key:
            kwargs.setdefault('details', {})['config_key'] = config_key
        super().__init__(message, **kwargs)


class ValidationError(TradingSystemError):
    """Raised when validation fails."""

    def __init__(
        self,
        message: str = "Validation failed",
        field: Optional[str] = None,
        value: Any = None,
        **kwargs
    ):
        self.field = field
        self.value = value

        kwargs.setdefault('details', {})
        if field:
            kwargs['details']['field'] = field
        if value is not None:
            kwargs['details']['value'] = str(value)[:100]

        super().__init__(message, **kwargs)


class CircuitBreakerOpenError(TradingSystemError):
    """Raised when circuit breaker is open and blocking operations."""

    def __init__(
        self,
        message: str = "Circuit breaker is open",
        breaker_name: Optional[str] = None,
        reset_after: Optional[float] = None,
        **kwargs
    ):
        self.breaker_name = breaker_name
        self.reset_after = reset_after

        kwargs.setdefault('details', {})
        if breaker_name:
            kwargs['details']['breaker_name'] = breaker_name
        if reset_after:
            kwargs['details']['reset_after_seconds'] = reset_after

        super().__init__(message, **kwargs)


class PositionLimitError(TradingSystemError):
    """Raised when position limits are exceeded."""

    def __init__(
        self,
        message: str = "Position limit exceeded",
        current: Optional[int] = None,
        limit: Optional[int] = None,
        **kwargs
    ):
        self.current = current
        self.limit = limit

        kwargs.setdefault('details', {})
        if current is not None:
            kwargs['details']['current'] = current
        if limit is not None:
            kwargs['details']['limit'] = limit

        super().__init__(message, **kwargs)


# =============================================================================
# ERROR LOGGING DECORATOR
# =============================================================================

def log_errors(
    reraise: bool = True,
    log_level: int = logging.ERROR,
    include_traceback: bool = True,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    default_return: Any = None,
    error_callback: Optional[Callable[[Exception], None]] = None
):
    """
    Decorator that logs exceptions with context.

    Args:
        reraise: If True, re-raise the exception after logging
        log_level: Logging level for errors
        include_traceback: Include traceback in log
        exceptions: Exception types to catch
        default_return: Value to return if exception is caught and not reraised
        error_callback: Optional callback to call with the exception

    Example:
        @log_errors(reraise=False, default_return=None)
        def might_fail():
            ...

        @log_errors(reraise=True)
        def must_succeed():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                # Build context
                context_parts = [f"{func.__module__}.{func.__name__}"]

                if isinstance(e, TradingSystemError):
                    context_parts.append(str(e))
                else:
                    context_parts.append(f"{type(e).__name__}: {e}")

                # Log error
                msg = " | ".join(context_parts)
                if include_traceback:
                    logger.log(log_level, msg, exc_info=True)
                else:
                    logger.log(log_level, msg)

                # Call error callback if provided
                if error_callback:
                    try:
                        error_callback(e)
                    except Exception as cb_error:
                        logger.warning(f"Error callback failed: {cb_error}")

                if reraise:
                    raise

                return default_return

        return wrapper
    return decorator


# =============================================================================
# ERROR CONTEXT MANAGER
# =============================================================================

@contextmanager
def error_context(
    operation: str,
    *,
    symbol: Optional[str] = None,
    reraise: bool = True,
    log_level: int = logging.ERROR,
    **extra_context
):
    """
    Context manager that adds context to any exception.

    Args:
        operation: Description of the operation being performed
        symbol: Optional symbol context
        reraise: If True, re-raise with added context
        log_level: Logging level for errors
        **extra_context: Additional context to include

    Example:
        with error_context("submitting order", symbol="AAPL", order_type="market"):
            submit_order()

        # On error, logs:
        # ERROR: Failed while submitting order | symbol=AAPL | order_type=market | ...
    """
    try:
        yield
    except TradingSystemError as e:
        # Add context to existing trading system error
        if not e.operation:
            e.operation = operation
        if not e.symbol and symbol:
            e.symbol = symbol
        e.details.update(extra_context)

        context = f"Failed while {operation}"
        if symbol:
            context += f" | symbol={symbol}"
        if extra_context:
            context += f" | {extra_context}"
        context += f" | {e}"

        logger.log(log_level, context)

        if reraise:
            raise

    except Exception as e:
        # Wrap in TradingSystemError with context
        context = f"Failed while {operation}"
        if symbol:
            context += f" | symbol={symbol}"
        if extra_context:
            context += f" | {extra_context}"
        context += f" | {type(e).__name__}: {e}"

        logger.log(log_level, context, exc_info=True)

        if reraise:
            raise TradingSystemError(
                f"Failed while {operation}: {e}",
                operation=operation,
                symbol=symbol,
                details=extra_context,
                cause=e
            ) from e


# =============================================================================
# ERROR CLASSIFICATION
# =============================================================================

def is_retryable(error: Exception) -> bool:
    """
    Determine if an error is retryable.

    Returns True for transient errors (network, timeout, temporary DB issues).
    Returns False for permanent errors (validation, config, etc.).
    """
    # Always retryable
    retryable_types = (
        TimeoutError,
        ConnectionError,
        OSError,
        DataFetchError,
    )

    if isinstance(error, retryable_types):
        return True

    # Check for common transient error patterns
    error_str = str(error).lower()
    transient_patterns = [
        "timeout",
        "timed out",
        "connection refused",
        "connection reset",
        "temporarily unavailable",
        "service unavailable",
        "rate limit",
        "too many requests",
        "database is locked",
        "busy",
    ]

    return any(pattern in error_str for pattern in transient_patterns)


def is_circuit_breaker_trigger(error: Exception) -> bool:
    """
    Determine if an error should trigger circuit breaker.

    Returns True for errors indicating service degradation.
    """
    # These errors indicate service problems
    trigger_types = (
        TimeoutError,
        ExecutionError,
        DatabaseError,
    )

    if isinstance(error, trigger_types):
        return True

    # Check patterns that indicate service issues
    error_str = str(error).lower()
    service_issue_patterns = [
        "service unavailable",
        "internal server error",
        "gateway timeout",
        "bad gateway",
        "too many requests",
        "rate limit exceeded",
    ]

    return any(pattern in error_str for pattern in service_issue_patterns)


# =============================================================================
# ERROR FORMATTING
# =============================================================================

def format_exception_chain(error: Exception) -> str:
    """
    Format exception with its full chain for logging.

    Returns a multi-line string showing the exception chain.
    """
    lines = []
    current = error

    while current:
        lines.append(f"{type(current).__name__}: {current}")
        current = getattr(current, '__cause__', None) or getattr(current, 'cause', None)

    return "\n  Caused by: ".join(lines)


def format_error_for_alert(error: Exception) -> Dict[str, Any]:
    """
    Format error for alerting systems (Slack, PagerDuty, etc.).

    Returns structured dict suitable for alert payloads.
    """
    result = {
        "error_type": type(error).__name__,
        "message": str(error),
        "is_retryable": is_retryable(error),
        "is_circuit_breaker_trigger": is_circuit_breaker_trigger(error),
    }

    if isinstance(error, TradingSystemError):
        result.update(error.to_dict())

    return result


# =============================================================================
# SAFE EXECUTION HELPERS
# =============================================================================

def safe_execute(
    func: Callable[..., T],
    *args,
    default: T = None,
    log_errors_flag: bool = True,
    **kwargs
) -> T:
    """
    Execute a function safely, returning default on any error.

    Useful for non-critical operations that shouldn't crash the system.

    Args:
        func: Function to execute
        *args: Positional arguments
        default: Value to return on error
        log_errors_flag: Whether to log errors
        **kwargs: Keyword arguments

    Returns:
        Function result or default on error

    Example:
        # Non-critical LED update
        safe_execute(update_led_status, "green", default=None)
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors_flag:
            logger.warning(f"{func.__name__} failed: {e}")
        return default


def collect_errors(
    operations: Dict[str, Callable[[], T]],
    continue_on_error: bool = True
) -> Tuple[Dict[str, T], Dict[str, Exception]]:
    """
    Execute multiple operations, collecting results and errors.

    Useful for batch operations where you want all results/errors.

    Args:
        operations: Dict mapping name to callable
        continue_on_error: If True, continue after errors

    Returns:
        Tuple of (results dict, errors dict)

    Example:
        ops = {
            "fetch_AAPL": lambda: fetch("AAPL"),
            "fetch_GOOGL": lambda: fetch("GOOGL"),
        }
        results, errors = collect_errors(ops)
        if errors:
            logger.warning(f"Some fetches failed: {list(errors.keys())}")
    """
    results: Dict[str, T] = {}
    errors: Dict[str, Exception] = {}

    for name, op in operations.items():
        try:
            results[name] = op()
        except Exception as e:
            errors[name] = e
            if not continue_on_error:
                break

    return results, errors
