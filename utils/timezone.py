"""
Centralized Timezone Handling for Tradebot

POLICY: All internal data uses TIMEZONE-NAIVE timestamps (interpreted as UTC).
        Timezone-aware timestamps are converted to naive at data boundaries.
        Eastern time is only used for display and market hours logic.

This eliminates the "Invalid comparison between dtype=datetime64[ns, UTC] and Timestamp"
errors that occur when mixing aware and naive datetimes.

Usage:
    from utils.timezone import normalize_index, normalize_dataframe, now_naive

    # Normalize a DataFrame's index (handles both aware and naive)
    df = normalize_dataframe(df)

    # Get current time as naive timestamp
    cutoff = now_naive() - pd.Timedelta(days=730)

    # Safe comparison
    df = df[df.index >= cutoff]
"""

import pandas as pd
from datetime import datetime, timezone
from typing import Union, Optional
import pytz

# Timezone constants
TZ_UTC = pytz.UTC
TZ_EASTERN = pytz.timezone('America/New_York')


def normalize_timestamp(ts: Union[pd.Timestamp, datetime, str, None]) -> Optional[pd.Timestamp]:
    """
    Convert any timestamp to a timezone-naive pd.Timestamp.

    Handles:
    - pd.Timestamp (aware or naive)
    - datetime (aware or naive)
    - String dates
    - None (returns None)

    Args:
        ts: Input timestamp in any format

    Returns:
        Timezone-naive pd.Timestamp, or None if input is None

    Examples:
        >>> normalize_timestamp(pd.Timestamp('2024-01-01', tz='UTC'))
        Timestamp('2024-01-01 00:00:00')

        >>> normalize_timestamp(datetime.now(timezone.utc))
        Timestamp('2024-01-01 12:34:56')  # naive
    """
    if ts is None:
        return None

    # Convert to pd.Timestamp if needed
    if not isinstance(ts, pd.Timestamp):
        ts = pd.Timestamp(ts)

    # Remove timezone if present
    if ts.tz is not None:
        # Convert to UTC first, then make naive
        ts = ts.tz_convert('UTC').tz_localize(None)

    return ts


def normalize_index(index: pd.Index) -> pd.DatetimeIndex:
    """
    Convert a pandas Index to a timezone-naive DatetimeIndex.

    Handles:
    - DatetimeIndex (aware or naive)
    - Index with datetime-like values
    - RangeIndex (raises error)

    Args:
        index: Input pandas Index

    Returns:
        Timezone-naive DatetimeIndex

    Examples:
        >>> idx = pd.DatetimeIndex(['2024-01-01', '2024-01-02'], tz='UTC')
        >>> normalize_index(idx)
        DatetimeIndex(['2024-01-01', '2024-01-02'], dtype='datetime64[ns]')
    """
    # Convert to DatetimeIndex if needed
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(index)

    # Remove timezone if present
    if index.tz is not None:
        index = index.tz_convert('UTC').tz_localize(None)

    return index


def normalize_dataframe(df: pd.DataFrame, index_col: Optional[str] = None) -> pd.DataFrame:
    """
    Normalize a DataFrame's datetime index and any timestamp columns.

    This is the primary function to call when loading data from any source
    (Alpaca, Yahoo, parquet files, etc.) to ensure consistent timezone handling.

    Args:
        df: Input DataFrame
        index_col: If provided, set this column as index first

    Returns:
        DataFrame with timezone-naive datetime index

    Examples:
        >>> df = pd.read_parquet('data.parquet')  # Might have UTC timestamps
        >>> df = normalize_dataframe(df)  # Now timezone-naive
    """
    if df.empty:
        return df

    df = df.copy()

    # Set index if column specified
    if index_col and index_col in df.columns:
        df = df.set_index(index_col)

    # Normalize index if it's datetime-like
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is not None:
            df.index = df.index.tz_convert('UTC').tz_localize(None)
    elif df.index.dtype == 'object':
        # Try to convert to datetime
        try:
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_convert('UTC').tz_localize(None)
        except (ValueError, TypeError):
            pass  # Not a datetime index, leave as-is

    # Normalize any timestamp columns
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            if hasattr(df[col].dt, 'tz') and df[col].dt.tz is not None:
                df[col] = df[col].dt.tz_convert('UTC').dt.tz_localize(None)

    return df


def normalize_series(series: pd.Series) -> pd.Series:
    """
    Normalize a pandas Series with datetime index or values.

    Args:
        series: Input Series

    Returns:
        Series with timezone-naive datetime handling
    """
    series = series.copy()

    # Normalize index
    if isinstance(series.index, pd.DatetimeIndex):
        if series.index.tz is not None:
            series.index = series.index.tz_convert('UTC').tz_localize(None)

    # Normalize values if datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        if hasattr(series.dt, 'tz') and series.dt.tz is not None:
            series = series.dt.tz_convert('UTC').dt.tz_localize(None)

    return series


def ensure_comparable(*timestamps) -> tuple:
    """
    Ensure multiple timestamps are comparable (all naive or all aware).

    Converts all to naive for consistency with our policy.

    Args:
        *timestamps: Variable number of timestamps

    Returns:
        Tuple of normalized timestamps

    Examples:
        >>> t1 = pd.Timestamp('2024-01-01', tz='UTC')
        >>> t2 = pd.Timestamp('2024-01-02')  # naive
        >>> t1_norm, t2_norm = ensure_comparable(t1, t2)
        >>> t1_norm < t2_norm  # Now works!
        True
    """
    return tuple(normalize_timestamp(ts) for ts in timestamps)


def now_naive() -> pd.Timestamp:
    """
    Get current time as a timezone-naive pd.Timestamp.

    Use this instead of pd.Timestamp.now() for consistency.

    Returns:
        Current UTC time as naive Timestamp
    """
    return pd.Timestamp.now('UTC').tz_localize(None)


def now_utc() -> pd.Timestamp:
    """
    Get current time as a UTC-aware pd.Timestamp.

    Use this when you specifically need timezone awareness.

    Returns:
        Current time as UTC-aware Timestamp
    """
    return pd.Timestamp.now('UTC')


def now_eastern() -> datetime:
    """
    Get current time in US Eastern timezone.

    Use this for market hours logic and display purposes.

    Returns:
        Current time as Eastern-aware datetime
    """
    return datetime.now(TZ_EASTERN)


def to_market_time(ts: Union[pd.Timestamp, datetime]) -> datetime:
    """
    Convert any timestamp to US Eastern market time.

    Use this when you need to check market hours or display times.

    Args:
        ts: Input timestamp (aware or naive, assumes UTC if naive)

    Returns:
        Eastern-aware datetime
    """
    if isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime()

    # If naive, assume UTC
    if ts.tzinfo is None:
        ts = TZ_UTC.localize(ts)

    return ts.astimezone(TZ_EASTERN)


def is_market_hours(ts: Optional[Union[pd.Timestamp, datetime]] = None) -> bool:
    """
    Check if a timestamp falls within regular market hours (9:30 AM - 4:00 PM ET).

    Args:
        ts: Timestamp to check (defaults to now)

    Returns:
        True if within market hours
    """
    if ts is None:
        et_time = now_eastern()
    else:
        et_time = to_market_time(ts)

    # Check weekday (0=Monday, 6=Sunday)
    if et_time.weekday() >= 5:
        return False

    # Check time
    market_open = et_time.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = et_time.replace(hour=16, minute=0, second=0, microsecond=0)

    return market_open <= et_time <= market_close


def safe_date_filter(df: pd.DataFrame,
                     start_date: Optional[Union[str, pd.Timestamp, datetime]] = None,
                     end_date: Optional[Union[str, pd.Timestamp, datetime]] = None) -> pd.DataFrame:
    """
    Safely filter a DataFrame by date range, handling timezone mismatches.

    This is the recommended way to filter date ranges to avoid comparison errors.

    Args:
        df: DataFrame with datetime index
        start_date: Start of range (inclusive)
        end_date: End of range (inclusive)

    Returns:
        Filtered DataFrame

    Examples:
        >>> df = load_some_data()  # Might have UTC index
        >>> filtered = safe_date_filter(df, start_date='2024-01-01', end_date='2024-12-31')
    """
    if df.empty:
        return df

    # Normalize the DataFrame first
    df = normalize_dataframe(df)

    # Normalize filter dates
    if start_date is not None:
        start_date = normalize_timestamp(pd.Timestamp(start_date))
        df = df[df.index >= start_date]

    if end_date is not None:
        end_date = normalize_timestamp(pd.Timestamp(end_date))
        df = df[df.index <= end_date]

    return df


# Convenience function for the most common use case
def prepare_data_for_analysis(df: pd.DataFrame,
                               timestamp_col: str = 'timestamp',
                               cutoff_days: Optional[int] = None) -> pd.DataFrame:
    """
    Prepare a DataFrame for analysis by normalizing timezones and optionally filtering.

    This is the recommended entry point when loading data for backtesting or research.

    Args:
        df: Raw DataFrame from data source
        timestamp_col: Name of timestamp column (if not already index)
        cutoff_days: If provided, filter to last N days

    Returns:
        Normalized, optionally filtered DataFrame

    Examples:
        >>> raw_data = pd.read_parquet('spy.parquet')
        >>> data = prepare_data_for_analysis(raw_data, cutoff_days=365)
    """
    # Set timestamp as index if it's a column
    if timestamp_col in df.columns:
        df = df.set_index(timestamp_col)

    # Normalize
    df = normalize_dataframe(df)

    # Filter by cutoff if specified
    if cutoff_days is not None:
        cutoff = now_naive() - pd.Timedelta(days=cutoff_days)
        df = df[df.index >= cutoff]

    return df


# =============================================================================
# Research Time Boundaries
# =============================================================================
# Research should NEVER run during or close to market hours.
# These hard boundaries are enforced independent of the orchestrator.

# Times in Eastern (ET)
RESEARCH_STOP_TIME_WEEKDAY = (7, 30)   # 7:30 AM ET - hard stop on weekdays
RESEARCH_START_TIME_WEEKDAY = (17, 0)  # 5:00 PM ET - earliest start on weekdays
RESEARCH_STOP_TIME_SUNDAY = (19, 30)   # 7:30 PM ET - stop Sunday evening before futures open


def is_research_allowed() -> bool:
    """
    Check if research is currently allowed based on time boundaries.

    Research windows (all times in Eastern):
    - Saturday: All day (00:00-23:59)
    - Sunday: 00:00-19:30 (before futures open)
    - Weekdays: 00:00-07:30 and 17:00-23:59 (before/after market +buffer)

    Returns:
        True if research is allowed now, False otherwise
    """
    now = now_eastern()
    weekday = now.weekday()  # 0=Mon, 5=Sat, 6=Sun
    hour, minute = now.hour, now.minute
    current_minutes = hour * 60 + minute

    # Saturday: always allowed
    if weekday == 5:
        return True

    # Sunday: allowed until 7:30 PM ET (before futures open at 8 PM)
    if weekday == 6:
        stop_minutes = RESEARCH_STOP_TIME_SUNDAY[0] * 60 + RESEARCH_STOP_TIME_SUNDAY[1]
        return current_minutes < stop_minutes

    # Weekday (Mon-Fri):
    # Allowed if before 7:30 AM OR after 5:00 PM
    stop_minutes = RESEARCH_STOP_TIME_WEEKDAY[0] * 60 + RESEARCH_STOP_TIME_WEEKDAY[1]
    start_minutes = RESEARCH_START_TIME_WEEKDAY[0] * 60 + RESEARCH_START_TIME_WEEKDAY[1]

    return current_minutes < stop_minutes or current_minutes >= start_minutes


def get_research_deadline() -> Optional[datetime]:
    """
    Get the next time when research must stop.

    Returns:
        Datetime when research must stop, or None if no deadline today
    """
    now = now_eastern()
    weekday = now.weekday()

    # Saturday: no deadline
    if weekday == 5:
        return None

    # Sunday: deadline is 7:30 PM
    if weekday == 6:
        deadline = now.replace(
            hour=RESEARCH_STOP_TIME_SUNDAY[0],
            minute=RESEARCH_STOP_TIME_SUNDAY[1],
            second=0, microsecond=0
        )
        return deadline if now < deadline else None

    # Weekday: deadline is 7:30 AM (if we're in overnight session)
    deadline = now.replace(
        hour=RESEARCH_STOP_TIME_WEEKDAY[0],
        minute=RESEARCH_STOP_TIME_WEEKDAY[1],
        second=0, microsecond=0
    )
    return deadline if now < deadline else None
