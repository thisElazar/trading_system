"""
Unified Timeout Utilities
=========================
Provides consistent timeout handling, retry logic, and timeout wrappers
for all external calls across the trading system.

Usage:
    from utils.timeout import TimeoutConfig, timeout_wrapper, with_timeout, retry_with_backoff

    # Wrap a function call with timeout
    result = timeout_wrapper(slow_function, TimeoutConfig.API_CALL, "get_account")

    # Use as decorator
    @with_timeout(TimeoutConfig.API_CALL)
    def slow_function():
        ...

    # Retry with exponential backoff
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def flaky_function():
        ...

Design Principles:
- Simple timeout + log + raise (no circuit breaker integration here)
- Existing circuit breakers handle their own logic separately
- Consistent timeout values by operation type
"""

import logging
import time
import random
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Optional, Tuple, Type, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# TIMEOUT CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class TimeoutConfig:
    """Standard timeout values by operation type (in seconds)."""

    # API calls (account info, quotes, etc.)
    API_CALL: float = 15.0

    # Data fetching (historical bars, batch queries)
    DATA_FETCH: float = 30.0

    # Order submission (market/limit orders)
    ORDER_SUBMIT: float = 30.0

    # Worker pool join operations
    POOL_JOIN: float = 30.0

    # Database queries
    DB_QUERY: float = 10.0

    # Stream reconnection
    STREAM_RECONNECT: float = 60.0

    # Per-symbol data fetch (includes retries)
    SYMBOL_FETCH: float = 60.0


# Default instance for easy import
TIMEOUTS = TimeoutConfig()


# =============================================================================
# RETRY CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class RetryConfig:
    """Standard retry configuration."""

    # Default retry settings
    MAX_RETRIES: int = 3
    BASE_DELAY: float = 1.0  # Base delay in seconds (doubles each retry)
    MAX_DELAY: float = 30.0  # Maximum delay between retries
    JITTER: float = 0.5      # Jitter factor (0.5 means 50-100% of calculated delay)


RETRIES = RetryConfig()


# =============================================================================
# TIMEOUT WRAPPER
# =============================================================================

def timeout_wrapper(
    func: Callable[..., T],
    timeout_seconds: float,
    operation_name: str = "operation",
    *args,
    **kwargs
) -> T:
    """
    Execute a function with a timeout.

    Uses ThreadPoolExecutor to run the function in a separate thread,
    allowing timeout even for blocking I/O operations.

    Args:
        func: Function to execute
        timeout_seconds: Maximum time to wait in seconds
        operation_name: Name for logging purposes
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        Result from func

    Raises:
        TimeoutError: If function doesn't complete within timeout
        Exception: Any exception raised by func
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError:
            logger.error(
                f"Timeout: {operation_name} did not complete within {timeout_seconds}s"
            )
            raise TimeoutError(
                f"{operation_name} timed out after {timeout_seconds} seconds"
            )


def with_timeout(timeout_seconds: float, operation_name: Optional[str] = None):
    """
    Decorator to add timeout to a function.

    Args:
        timeout_seconds: Maximum time to wait in seconds
        operation_name: Name for logging (defaults to function name)

    Example:
        @with_timeout(TimeoutConfig.API_CALL)
        def get_account():
            return client.get_account()
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            name = operation_name or func.__name__
            return timeout_wrapper(func, timeout_seconds, name, *args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# RETRY WITH BACKOFF
# =============================================================================

def retry_with_backoff(
    max_retries: int = RETRIES.MAX_RETRIES,
    base_delay: float = RETRIES.BASE_DELAY,
    max_delay: float = RETRIES.MAX_DELAY,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = (Exception,),
    jitter: float = RETRIES.JITTER,
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Decorator that retries a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (total calls = max_retries + 1)
        base_delay: Base delay in seconds (doubles each retry)
        max_delay: Maximum delay between retries
        exceptions: Exception types to catch and retry
        jitter: Jitter factor (0.5 means delay is multiplied by 0.5-1.0)
        on_retry: Optional callback called on each retry with (exception, attempt)

    Example:
        @retry_with_backoff(max_retries=3, exceptions=(ConnectionError, TimeoutError))
        def fetch_data():
            return requests.get(url)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt < max_retries:
                        # Exponential backoff with jitter
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        delay = delay * (jitter + random.random() * (1 - jitter))

                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )

                        if on_retry:
                            on_retry(e, attempt)

                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_retries + 1} attempts failed for {func.__name__}: {e}"
                        )

            raise last_exception

        return wrapper
    return decorator


# =============================================================================
# COMBINED TIMEOUT + RETRY
# =============================================================================

def with_timeout_and_retry(
    timeout_seconds: float,
    max_retries: int = RETRIES.MAX_RETRIES,
    base_delay: float = RETRIES.BASE_DELAY,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = (Exception,),
    operation_name: Optional[str] = None
):
    """
    Decorator combining timeout and retry logic.

    Each retry attempt has its own timeout. Total time = timeout * (max_retries + 1).

    Args:
        timeout_seconds: Timeout for each attempt
        max_retries: Number of retry attempts
        base_delay: Base delay between retries
        exceptions: Exceptions to retry on (TimeoutError always included)
        operation_name: Name for logging

    Example:
        @with_timeout_and_retry(timeout_seconds=15, max_retries=3)
        def call_api():
            return requests.get(url)
    """
    # Always retry on TimeoutError
    if isinstance(exceptions, tuple):
        retry_exceptions = exceptions + (TimeoutError,)
    else:
        retry_exceptions = (exceptions, TimeoutError)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        name = operation_name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return timeout_wrapper(func, timeout_seconds, name, *args, **kwargs)
                except retry_exceptions as e:
                    last_exception = e

                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), RETRIES.MAX_DELAY)
                        delay = delay * (RETRIES.JITTER + random.random() * (1 - RETRIES.JITTER))

                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for {name}: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {name}: {e}")

            raise last_exception

        return wrapper
    return decorator


# =============================================================================
# POOL JOIN WITH TIMEOUT
# =============================================================================

def pool_join_with_timeout(
    pool,
    timeout: float = TIMEOUTS.POOL_JOIN,
    force_terminate: bool = True,
    operation_name: str = "pool"
) -> bool:
    """
    Wait for a multiprocessing Pool to finish with timeout.

    Uses polling to check worker status with deadline-based timeout,
    avoiding the deadlock that can occur with bare pool.join().

    Args:
        pool: multiprocessing.Pool instance
        timeout: Maximum seconds to wait
        force_terminate: If True, terminate workers after timeout
        operation_name: Name for logging

    Returns:
        True if pool exited gracefully, False if forced termination

    Example:
        pool.close()
        success = pool_join_with_timeout(pool, timeout=30)
        if not success:
            logger.warning("Pool required forced termination")
    """
    deadline = time.time() + timeout
    poll_interval = 0.1

    while time.time() < deadline:
        # Check if all workers have exited
        if hasattr(pool, '_pool'):
            # Standard Pool
            if not any(w.is_alive() for w in pool._pool):
                pool.join()
                logger.debug(f"{operation_name} workers exited gracefully")
                return True
        else:
            # Assume pool is ready if no _pool attribute
            try:
                pool.join()
                return True
            except Exception:
                pass

        time.sleep(poll_interval)

    # Timeout reached
    logger.warning(f"{operation_name} workers didn't terminate within {timeout}s")

    if force_terminate:
        logger.warning(f"Force terminating {operation_name} workers")
        try:
            pool.terminate()
            pool.join()
        except Exception as e:
            logger.error(f"Error terminating {operation_name}: {e}")

    return False


# =============================================================================
# ASYNC TIMEOUT UTILITIES
# =============================================================================

async def async_timeout_wrapper(
    coro,
    timeout_seconds: float,
    operation_name: str = "async_operation"
):
    """
    Wrap an async coroutine with a timeout.

    Args:
        coro: Coroutine to execute
        timeout_seconds: Maximum time to wait
        operation_name: Name for logging

    Returns:
        Result from coroutine

    Raises:
        asyncio.TimeoutError: If coroutine doesn't complete within timeout
    """
    import asyncio

    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.error(
            f"Async timeout: {operation_name} did not complete within {timeout_seconds}s"
        )
        raise


def async_with_timeout(timeout_seconds: float, operation_name: Optional[str] = None):
    """
    Decorator to add timeout to an async function.

    Args:
        timeout_seconds: Maximum time to wait
        operation_name: Name for logging

    Example:
        @async_with_timeout(5.0)
        async def fetch_data():
            async with aiohttp.ClientSession() as session:
                ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            import asyncio
            name = operation_name or func.__name__
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                logger.error(f"Async timeout: {name} did not complete within {timeout_seconds}s")
                raise
        return wrapper
    return decorator
