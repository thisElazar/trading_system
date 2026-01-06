# Pi 5 Trading System Optimization - January 4, 2026

## Summary
Comprehensive optimization session resolving system crashes, implementing parallel execution, and validating memory architecture.

---

## 1. Root Cause: Watchdog Daemon (RESOLVED)

### Problem
System was crashing randomly during research runs, even with conservative settings.

### Investigation
- Checked kernel parameters, ZRAM/ZSWAP config
- Compared against original SD card setup
- Found `watchdog` package installed Jan 3 with aggressive memory threshold

### Root Cause
```
/etc/watchdog.conf: min-memory = 51200  (50MB minimum)
```
Watchdog was rebooting the system when Python's VIRT memory exceeded thresholds, even though actual RAM usage was fine.

### Fix
```bash
sudo systemctl disable watchdog
sudo systemctl stop watchdog
```

---

## 2. Performance Configuration

### Optimized pi_safe Profile
| Setting | Before | After | Impact |
|---------|--------|-------|--------|
| max_symbols | 50 | 100 | 2x market coverage |
| max_years | 2 | 3 | 50% more history |
| n_workers | 2 | 4 | Use all CPU cores |
| population_size | 10 | 20 | Better GA exploration |
| generations | 2 | 3 | Deeper optimization |
| sleep_between_evals | 0.1s | 0.02s | 5x faster loops |

### Location
`config.py` - PI_PERFORMANCE_CONFIG['pi_safe']

---

## 3. Parallel Execution Refactor

### Problem
ProcessPoolExecutor couldn't pickle the fitness function (closure).

### Solution
Created module-level parallel infrastructure in `run_nightly_research.py`:
- `_parallel_fitness_context` - Global dict holding evaluation data
- `evaluate_genes_parallel()` - Module-level picklable function
- `setup_parallel_fitness_context()` - Initializes globals before parallel run

### Modified Files
- `run_nightly_research.py` - Added parallel infrastructure (~lines 55-150)
- `research/genetic/optimizer.py` - Updated `_evaluate_population_parallel()` to use fork Pool
- `research/genetic/persistent_optimizer.py` - Added parallel path in `evolve_incremental()`

### Performance Results
| Test | Sequential | Parallel (4 workers) | Speedup |
|------|------------|---------------------|---------|
| 4 evals | 141.40s | 58.68s | 2.41x |
| 20 evals | 707.01s | 312.40s | 2.26x |

---

## 4. Memory Architecture Validation

### Test
Loaded all 2556 symbols (~12.8M rows) to stress test memory hierarchy.

### Results
| Component | Usage | Status |
|-----------|-------|--------|
| RAM | 3.8/4GB (95%) | Maxed out |
| ZRAM | 1083MB (zstd compressed) | Working |
| ZSWAP | 2219MB | Working |
| NVMe Swap | Not needed | Available |

### Conclusion
Memory hierarchy works correctly:
1. RAM fills first
2. ZRAM compresses overflow (zstd ~1.4:1 ratio)
3. ZSWAP catches more before disk
4. NVMe swap as final fallback

### Safe Operating Limits
- **Max symbols**: ~500 (tested 2556 = thrashing)
- **Current setting**: 100 symbols (3x safety margin)
- **Target RAM**: <3GB to avoid swap

---

## 5. Database Fixes

### trades.db Schema Fix
The `signals` table was missing columns expected by db_manager.py.

```sql
ALTER TABLE signals ADD COLUMN timestamp TEXT;
ALTER TABLE signals ADD COLUMN strategy TEXT;
UPDATE signals SET timestamp = created_at WHERE timestamp IS NULL;
UPDATE signals SET strategy = strategy_name WHERE strategy IS NULL;
CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp);
```

### ML Regime Model Fix
Updated `_get_historical_regime_data()` in `daily_orchestrator.py`:
- Changed from non-existent `get_market_data()` to `get_bars()` and direct parquet load
- Fixed `train()` call to pass separate vix_series and sp500_series

---

## 6. Dashboard Improvements

### Orchestrator Logs Modal
Added clickable orchestrator panel that opens a log viewer modal.

**Files Modified**: `observability/dashboard/app.py`
- Added `orch-logs-modal` component
- Added `orch-logs-content` with auto-refresh (10s)
- Added callbacks for toggle and content update

### Weekend Staleness Check
Updated `verify_system_readiness()` to allow 120 hours (5 days) staleness on weekends vs 72 hours on weekdays.

---

## 7. Systemd Services

### Created Services
```
/etc/systemd/system/trading-orchestrator.service
/etc/systemd/system/trading-dashboard.service
```

### Features
- Auto-start on boot
- Restart on failure (10s delay)
- Runs as thiselazar user
- Logs to journald/log files

### Commands
```bash
# Status
sudo systemctl status trading-orchestrator trading-dashboard

# Restart
sudo systemctl restart trading-orchestrator

# Logs
journalctl -u trading-orchestrator -f

# Disable auto-start
sudo systemctl disable trading-orchestrator
```

---

## 8. Backup Files

Local backups saved before parallel refactor:
```
~/pi_backup_20260104/
  run_nightly_research.py
  optimizer.py
  persistent_optimizer.py
```

---

## Current System Status

| Component | Status |
|-----------|--------|
| Orchestrator | Running (systemd) |
| Dashboard | Running (systemd) |
| Auto-start | Enabled |
| Parallel Execution | 2.3x speedup |
| Memory Safety | Validated |
| All 8 Strategies | Evolved |

### Performance Profile
- 100 symbols, 3 years history
- 4 parallel workers
- Population 20, 3 generations
- Full research run: ~3-4 minutes (down from ~8 min)

---

## Files Modified This Session

1. `config.py` - Performance settings
2. `run_nightly_research.py` - Parallel fitness infrastructure
3. `research/genetic/optimizer.py` - Fork pool parallel evaluation
4. `research/genetic/persistent_optimizer.py` - Parallel path
5. `daily_orchestrator.py` - ML regime fix, staleness check
6. `observability/dashboard/app.py` - Orchestrator logs modal
7. `/etc/systemd/system/trading-*.service` - Auto-start services
8. `db/trades.db` - Schema fixes
