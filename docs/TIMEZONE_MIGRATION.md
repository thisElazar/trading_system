# Timezone Handling Migration Guide

**Created:** January 4, 2026
**Status:** In Progress
**Location:** `utils/timezone.py`

---

## The Problem

The codebase had 100+ instances of ad-hoc timezone handling, leading to:
- "Invalid comparison between dtype=datetime64[ns, UTC] and Timestamp" errors
- Inconsistent patterns (`tz_localize(None)` vs `tz_convert(None)`)
- Scattered timezone stripping logic in every strategy and data loader

## The Solution

A centralized `utils/timezone.py` module with a clear policy:

> **POLICY:** All internal data uses TIMEZONE-NAIVE timestamps (interpreted as UTC).
> Timezone-aware timestamps are converted to naive at data boundaries.
> Eastern time is only used for display and market hours logic.

## Quick Start

```python
from utils.timezone import normalize_dataframe, now_naive, safe_date_filter

# Loading data from any source
df = pd.read_parquet('some_data.parquet')
df = normalize_dataframe(df)  # Now safe to compare

# Getting current time for comparisons
cutoff = now_naive() - pd.Timedelta(days=365)
df = df[df.index >= cutoff]  # No more errors!

# Or use the safe filter helper
df = safe_date_filter(df, start_date='2024-01-01')
```

## Available Functions

| Function | Purpose | When to Use |
|----------|---------|-------------|
| `normalize_dataframe(df)` | Strip timezone from DataFrame | After loading any external data |
| `normalize_index(idx)` | Strip timezone from Index | When working with indices directly |
| `normalize_timestamp(ts)` | Strip timezone from single timestamp | For individual timestamps |
| `now_naive()` | Current time, timezone-naive | For date comparisons |
| `now_utc()` | Current time, UTC-aware | When you need aware timestamps |
| `now_eastern()` | Current time, Eastern | For market hours display |
| `safe_date_filter(df, start, end)` | Filter with auto-normalization | Filtering date ranges |
| `ensure_comparable(*ts)` | Normalize multiple timestamps | Before comparisons |
| `to_market_time(ts)` | Convert to Eastern | For display only |
| `is_market_hours(ts)` | Check market hours | Market hours logic |

## Migration Priority

### High Priority (Cause Errors)
These patterns cause comparison errors and should be migrated first:

```python
# BEFORE (error-prone)
cutoff = pd.Timestamp.now() - pd.Timedelta(days=730)
df = df[df.index >= cutoff]

# AFTER (safe)
from utils.timezone import now_naive
cutoff = now_naive() - pd.Timedelta(days=730)
df = df[df.index >= cutoff]
```

### Medium Priority (Scattered Logic)
These patterns work but are inconsistent:

```python
# BEFORE (scattered)
if df.index.tz is not None:
    df.index = df.index.tz_localize(None)

# AFTER (centralized)
from utils.timezone import normalize_dataframe
df = normalize_dataframe(df)
```

### Low Priority (Already Working)
These patterns are fine but could be simplified:

```python
# BEFORE (verbose)
if 'timestamp' in df.columns:
    df = df.set_index('timestamp')
if df.index.tz is not None:
    df.index = df.index.tz_convert('UTC').tz_localize(None)

# AFTER (concise)
from utils.timezone import normalize_dataframe
df = normalize_dataframe(df, index_col='timestamp')
```

## Files to Migrate

Based on grep results, these files have timezone handling:

### Critical (Data Loading)
- [x] `daily_orchestrator.py` - Fixed `_get_historical_regime_data()`
- [ ] `data/fetchers/daily_bars.py` - 3 instances
- [ ] `data/fetchers/vix.py` - 3 instances
- [ ] `data/unified_data_loader.py` - 1 instance

### Strategies (Many Instances)
- [ ] `strategies/mean_reversion.py` - 2 instances
- [ ] `strategies/vol_managed_momentum_v2.py` - 2 instances
- [ ] `strategies/vol_managed_momentum.py` - 1 instance
- [ ] `strategies/pairs_trading.py` - 1 instance
- [ ] `strategies/relative_volume_breakout.py` - 2 instances
- [ ] `strategies/sector_rotation.py` - 2 instances

### Research (Many Instances)
- [ ] `research/backtester.py` - 3 instances
- [ ] `research/backtester_fast.py` - 1 instance
- [ ] `research/backtester_optimized.py` - 5 instances
- [ ] `research/parallel_backtester.py` - 2 instances
- [ ] `research/unified_tester.py` - 1 instance
- [ ] `run_nightly_research.py` - 15+ instances (most complex)

### Scripts
- [ ] `scripts/validate_strategies.py` - 3 instances
- [ ] `scripts/run_baseline_backtest.py` - 1 instance
- [ ] `scripts/optimize_vix_thresholds.py` - 2 instances
- [ ] `scripts/download_all_history.py` - 2 instances

## Migration Pattern

For each file:

1. Add import at top:
```python
from utils.timezone import normalize_dataframe, now_naive
```

2. Replace ad-hoc normalization:
```python
# Find patterns like:
if df.index.tz is not None:
    df.index = df.index.tz_localize(None)

# Replace with:
df = normalize_dataframe(df)
```

3. Replace `pd.Timestamp.now()` for comparisons:
```python
# Find:
cutoff = pd.Timestamp.now() - pd.Timedelta(days=X)

# Replace:
cutoff = now_naive() - pd.Timedelta(days=X)
```

4. Test the file still works

## Testing

After migrating a file, verify:

```bash
# Run relevant tests
python -m pytest tests/ -k "test_file_name" -v

# Or quick smoke test
python -c "from module import function; print('OK')"
```

## Changelog

| Date | File | Change |
|------|------|--------|
| 2026-01-04 | `utils/timezone.py` | Created centralized module |
| 2026-01-04 | `daily_orchestrator.py` | Migrated `_get_historical_regime_data()` |

---

*This is a living document. Update as files are migrated.*
