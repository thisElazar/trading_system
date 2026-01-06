# Pi 5 Memory Crisis: Debugging Journey & Lessons Learned

## The Problem
The Raspberry Pi 5 (4GB RAM) trading system was experiencing constant reboots when services started. The dashboard wouldn't load, and the fan was cycling rapidly between high and low modes.

## Timeline of Investigation

### Phase 1: Wrong Turns (Chasing Symptoms)
We initially suspected:
- **ZRAM configuration** (changed from 2GB to 3GB) - Not the cause
- **Service file configuration** (StartLimitIntervalSec in wrong section) - Minor issue, not root cause
- **CPU governor** (changed from "performance" to "ondemand") - Not the cause
- **Systemd service hardening** (stripped to bare minimum) - Not the cause

**Lesson:** When a system is crashing, resist the urge to tweak random settings. Find the actual memory/resource usage first.

### Phase 2: Finding the Smoking Gun
A `top` screenshot during a crash revealed:
```
PID   USER      VIRT    RES    %MEM  COMMAND
xxxx  thiselaz  2.5g    616m   15.0  python run_nightly_research.py
```

**2.5GB VIRT on a 4GB system = OOM killer territory**

### Phase 3: Root Cause Analysis
Traced the memory explosion to `run_nightly_research.py` line ~1295:
```python
self.data_manager.load_all()  # Loads ALL 2556 symbols FIRST
# ... then later ...
symbols = symbols[:max_symbols]  # Filters to 150 AFTER loading
```

**The bug:** Loading 2556 symbols (~1.4GB on disk) into memory as float64 DataFrames = ~2.5GB VIRT, THEN filtering down to 150 symbols. Classic memory explosion pattern.

### Phase 4: Optimizations We Tried (And Their Fate)

| Optimization | Applied | Kept | Why |
|-------------|---------|------|-----|
| Limit symbols BEFORE loading (50) | ✓ | ✓ | **ROOT FIX** - Prevents 2.5GB allocation |
| Disable `parallel_enabled` | ✓ | ✓ | Reduces worker process memory |
| Reduce workers to 1 | ✓ | ✓ | Single process, lower memory |
| Remove Numba JIT | ✓ | ? | May have been unnecessary |
| Remove SQLite mmap/cache pragmas | ✓ | ? | May have been unnecessary |
| DataFrame float32 downcasting | ✓ then ✗ | ✗ | Reverted - wasn't the issue |
| Disable SharedDataManager pool | ✓ | ✓ | Shared memory adds overhead |
| Use PI_ADAPTIVE_CONFIG (pop=20) | ✓ | ✓ | Smaller populations = less memory |
| Reduce strategies to 3 | ✓ | ✓ | Fewer optimizer instances |
| Disable pairs_refresh task | ✓ | ✓ | Was loading 699 symbols |
| Add gc.collect() after strategies | ✓ then ✗ | ✗ | Caused syntax errors, removed |

### Phase 5: Dashboard Issues (Separate Problem)

**Issue 1:** Callbacks not registering (0 callbacks)
- **Cause:** `@callback` decorator from `from dash import callback` doesn't work reliably in service contexts
- **Fix:** Changed to `@app.callback` (29 occurrences)

**Issue 2:** Dashboard showing no data
- **Cause:** Empty database file at wrong path (`/data/storage/research.db`)
- **Fix:** Removed empty file, dashboard found real DB at `/db/research.db`

## The Actual Fix (What Mattered)

```python
# BEFORE (Memory Explosion)
self.data_manager.load_all()  # Loads ALL 2556 symbols
# ... processing ...
symbols = symbols[:max_symbols]  # Too late!

# AFTER (Memory Safe)
max_symbols = 50
all_symbols = self.data_manager.get_available_symbols()
symbols_to_load = sorted(all_symbols)[:max_symbols]
self.data_manager.load_all(symbols=symbols_to_load)  # Only load what we need
```

This single change fixed 90% of the problem. Everything else was defense-in-depth.

## Key Lessons

### 1. Profile Before Optimizing
We wasted time on ZRAM, governors, and service configs when a single `top` screenshot would have shown the 2.5GB Python process immediately.

### 2. Load Data Lazily
Never load all data then filter. Filter first, load second. This is especially critical on memory-constrained devices.

### 3. Parallel Processing Has Hidden Costs
Each worker in a multiprocessing pool can duplicate data. On a 4GB Pi:
- 4 workers × 500MB data = 2GB just in data copies
- Plus Python overhead, numpy arrays, etc.

### 4. The "Undo Test"
When you apply multiple fixes and it works, try undoing them one by one to find which actually mattered. We applied ~10 fixes but likely only 2-3 were essential.

### 5. Dash Callbacks in Services
`@callback` decorator requires the app context to be properly initialized. In systemd services, use `@app.callback` explicitly.

## Final State

| Metric | Before | After |
|--------|--------|-------|
| Memory Used | 2.5GB+ (crash) | 770MB |
| Swap Used | Thrashing | 0B |
| Services | Crashing | Stable |
| Symbols Loaded | 2556 | 50 |

## What We Could Restore

Now that the system is stable, we could potentially restore:
1. **Numba JIT** - If it provides real speedup
2. **SQLite optimizations** - mmap and cache_size for faster queries
3. **More symbols** - Maybe 100-150 instead of 50
4. **More strategies** - Add back some of the disabled ones
5. **Parallel processing** - With proper memory limits

The key is to add these back ONE AT A TIME and monitor memory after each change.

## Commands for Future Debugging

```bash
# Monitor memory in real-time
watch -n 1 'free -h && ps aux --sort=-%mem | head -5'

# Check what's using memory
ps aux --sort=-%mem | head -10

# Check for OOM kills
dmesg | grep -i "out of memory\|oom\|killed"

# Monitor during research run
htop  # Press 'M' to sort by memory
```

---

## Update: January 4, 2026 - Backtest Duration Discovery

### The Plot Thickens

After implementing the fixes above, we attempted to re-enable performance optimizations (3-4 workers, parallel processing). The system still crashed repeatedly.

### New Root Cause: Backtest Duration

Through profiling, we discovered:

| Backtest Duration | VIRT Memory | Status |
|-------------------|-------------|--------|
| 1 year, 10 symbols | 234 MB | Works |
| 10 years, 10 symbols | 2.2 GB | Crashes |

The real culprit: 10 years of backtest data causes 2.2GB VIRT allocation regardless of symbol count. Combined with 100% CPU for extended periods, this triggers Pi watchdog/thermal shutdown.

### The Backup Was Not Working Either

When comparing with trading_backup from Jan 3:
- Backup optimizer.py had n_workers = 4 (not 1)
- The memory_alerts.log showed CRITICAL alerts from Jan 3
- Research was never stable - we just had not noticed

### Solution: Centralized Performance Configuration

Created PI_PERFORMANCE_CONFIG in config.py to consolidate all scattered magic numbers with three profiles: pi_safe, pi_balanced, and workstation. Auto-detects Raspberry Pi and selects appropriate profile.

Key settings in pi_safe profile:
- max_symbols: 50
- max_years: 1 (KEY LIMIT)
- parallel_enabled: False
- n_workers: 1

### Files Updated to Use PERF

1. config.py - Central authority for all performance settings
2. run_nightly_research.py - Uses PERF for symbols, years, workers
3. research/genetic/optimizer.py - Uses PERF for parallelism
4. research/genetic/adaptive_optimizer.py - Uses PERF for parallelism

### Result

Performance profile: pi_safe (max_symbols=50, max_years=1, workers=1)
Duration: 2.4s
Strategies evolved: 8/8

### Key Insight

The memory issue is not just about how many symbols we load - it is about how much history we process during backtesting. A 10-year walk-forward optimization creates massive intermediate arrays regardless of symbol count. Limiting max_years=1 keeps memory bounded.

---

## Update: January 4, 2026 - Root Cause Found: Watchdog Service

### The Real Culprit

After extensive debugging (kernel params, memory cgroups, ZRAM, CPU throttling, subprocess isolation), we discovered the actual cause: the watchdog daemon.

### Timeline

1. Crashes occurred even during idle periods
2. CPU and memory metrics looked fine at crash time
3. Kernel param changes did not help
4. Comparing against original setup revealed watchdog was installed Jan 3

### The Smoking Gun

The watchdog config had min-memory = 51200 (50MB minimum free RAM). When Python allocates large VIRT, free memory briefly dips below 50MB, triggering reboot.

### The Fix

Disabled watchdog service: systemctl disable watchdog

### Result

All 8 strategies evolved in 79.9 seconds with zero crashes.

### Lessons Learned

1. Check recent package installs first (dpkg.log)
2. System daemons can cause mysterious crashes
3. VIRT vs RSS - large VIRT affects memory reporting
4. Pi 5 is stable once overzealous watchdog removed

### Current Stable Config

- Watchdog: DISABLED
- ZRAM: 2GB
- zswap: enabled
- cgroup_enable=memory in cmdline.txt
- Performance profile: pi_safe (25 symbols, 1 year, 1 worker)

---

## Update: January 4, 2026 - Root Cause Found: Watchdog Service

### The Real Culprit

After extensive debugging (kernel params, memory cgroups, ZRAM, CPU throttling, subprocess isolation), we discovered the actual cause of the repeated crashes: **the watchdog daemon**.

### Timeline of Discovery

1. Crashes occurred even during idle periods (3-second cooldowns between tests)
2. CPU and memory metrics looked fine at crash time
3. Kernel params (, ) didn't help
4. User suggested comparing against original SD card setup
5. Found  package was installed Jan 3 at 19:51

### The Smoking Gun



The  setting triggers a system reboot when free RAM drops below 50MB. When Python's backtester allocates large virtual memory (2GB+ VIRT), the system's reported free memory briefly dips, triggering the watchdog to reboot.

### The Fix



### Result

After disabling watchdog:
- 10/10 stress test runs completed
- All 8 strategies evolved in 79.9 seconds
- Zero crashes

### Lessons Learned

1. **Check recent package installs first** -  would have shown the watchdog installation immediately
2. **System daemons can cause mysterious crashes** - The watchdog was doing exactly what it was configured to do
3. **VIRT vs RSS matters** - Large VIRT allocations can affect system memory reporting even if RSS is small
4. **The Pi 5 is stable** - Once the overzealous watchdog was removed, the system handled research workloads fine

### Current Stable Configuration

- Watchdog: DISABLED
- ZRAM: 2GB (ram/2)
- zswap: enabled with zstd compression
- cgroup_enable=memory in cmdline.txt
- Performance profile: pi_safe (25 symbols, 1 year, 1 worker)
