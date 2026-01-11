# Tradebot Punchlist

**Created:** January 4, 2026
**Purpose:** Identified bugs and improvements for R&D focus
**Priority Framework:** P0 (blocks progress) → P1 (significant impact) → P2 (nice to have)

---

## Summary

| Category | P0 | P1 | P2 | Resolved | Total |
|----------|----|----|----|---------:|------:|
| Bugs | 0 | 0 | 0 | 8 | 8 |
| Architecture | 0 | 0 | 1 | 5 | 6 |
| Stability | 0 | 0 | 0 | 10 | 10 |
| Research/GA | 0 | 0 | 0 | 6 | 6 |
| GP Research Gaps | 0 | 0 | 2 | 9 | 11 |
| **Total** | **0** | **0** | **3** | **38** | **41** |

**Resolved Jan 4:** BUG-001 (pairs/cash account), BUG-002 (timezone), BUG-005 (signals table), BUG-007 (test data cleanup), ARCH-003 (error logging), STAB-003 (log rotation), STAB-005 (version pinning)

**Resolved Jan 8:** ARCH-001 (startup recovery), STAB-006 (DB thread safety), STAB-007 (TOCTOU race), STAB-008 (partial fills), STAB-009 (screen cache), STAB-010 (VIX timeout)

**Resolved Jan 9:** GP-007 (paper trading duration), GP-008 (CPCV validation), GP-009 (Calmar ratio), GP-010 (HMM regime), GP-011 (migration rate), GP-012 (novelty pulsation), GA-001 (stagnation detection), STAB-001 (memory monitoring), ARCH-002 (real-time P&L), STAB-004 (graceful shutdown), BUG-003 (sector rotation), GA-002 (fitness bounds), BUG-004 (gap-fill intraday data), ARCH-005 (intraday refresh)

**Resolved Jan 10:** STAB-002 (database backups enabled), GA-006 (persistent pool refactor), ARCH-006 (Telegram alerts), GA-003 (GP discovery validated), GA-004 (portfolio fitness verified working), GA-005 (dynamic novelty tuning), GP-013 (self-adaptive mutation), GP-014 (Omega ratio), GP-017 (behavioral descriptors)

---

## BUGS

### P0 - Critical (Blocks Progress)

#### BUG-001: Pairs Trading Strategy Incompatible with Cash Account
**Status:** RESOLVED - Architectural Constraint (2026-01-04)
**Impact:** 5% allocation reallocated to mean_reversion

**Root Cause:**
Traditional pairs trading requires **shorting one leg** of each pair to create a market-neutral hedge. This system uses a **cash account** (no margin), which cannot short stocks.

**What Was Happening:**
- Strategy only executed the long leg (by design - no short capability)
- Without the short hedge, positions were directional bets, not pairs trades
- The cointegration-based mean reversion logic doesn't apply to single-leg positions
- Sharpe of -3.02 was expected behavior for a broken hedge

**Resolution:**
1. Pairs trading **disabled** in `config.py`
2. 5% allocation **reallocated to mean_reversion** (now 35%)
3. `pairs.db` and discovery infrastructure retained for future use if margin enabled

**Future Options (if margin account enabled):**
- Re-enable with proper two-legged execution
- Use inverse ETF pairs (long-only substitute)
- Use options for synthetic shorts

---

#### BUG-002: ML Regime Detector Datetime Error
**Status:** FIXED (2026-01-04)
**Impact:** ML-enhanced regime detection not functioning

**Evidence:**
```
WARNING | orchestrator | Failed to get historical regime data: Invalid comparison between dtype=datetime64[ns, UTC] and Timestamp
WARNING | orchestrator | Insufficient historical data for ML training
```

**Root Cause:**
Timezone-aware vs timezone-naive datetime comparison in `daily_orchestrator.py:_get_historical_regime_data()`

**Solution Applied:**
Created centralized timezone handling utility (`utils/timezone.py`) with policy:
- All internal data uses TIMEZONE-NAIVE timestamps (interpreted as UTC)
- Timezone-aware timestamps are converted to naive at data boundaries
- Eastern time only used for display and market hours logic

**Files Modified:**
- NEW: `utils/timezone.py` - Centralized timezone handling
- NEW: `utils/__init__.py` - Package exports
- `daily_orchestrator.py` - Now uses `normalize_dataframe()` and `now_naive()`

**Remaining Work:**
- Gradually migrate other files to use `utils.timezone` functions
- See TIMEZONE_MIGRATION.md for adoption guide

---

### P1 - High (Significant Impact)

#### BUG-003: Sector Rotation Negative Performance
**Status:** RESOLVED (2026-01-09)
**Impact:** Strategy re-enabled with 10% allocation, Sharpe improved from -0.38 → 1.08

**Root Cause:**
Original strategy spread allocations across all sectors based on VIX regime. GA discovered that concentrating in only top-2 momentum sectors performs dramatically better.

**Solution Applied (Jan 4, 2026):**
GA optimization discovered optimal parameters:
- `momentum_period: 105` (5-month lookback vs generic 21-day)
- `top_n_sectors: 2` (concentrated vs diversified)
- `rebalance_days: 28` (monthly vs event-driven)

**Configuration in config.py:**
```python
"sector_rotation": {
    "enabled": True,
    "tier": 1,  # Upgraded from tier 2
    "allocation_pct": 0.10,
    "max_positions": 2,
    "params": {
        "momentum_period": 105,
        "top_n_sectors": 2,
        "rebalance_days": 28,
    },
}
```

**Validation:**
- 138 GA evolution runs across 8+ days
- Best Sharpe: 1.0824 (consistent across all runs since Jan 4)
- Exceeds min threshold (0.5) and research benchmark (0.73)

---

#### BUG-004: Gap-Fill Strategy Not Live Validated
**Status:** CRITICAL - Strategy enabled but NOT WORKING (2026-01-09)
**Impact:** 10% allocation receiving 0 signals

**Investigation Findings (Jan 9):**
- **Zero signals generated** since deployment
- **Zero trades executed** by gap_fill strategy
- Logs show: "gap_fill completed in 4.8s (no signals)" consistently
- **Intraday data is 14 days stale** (last update: Dec 26, 2025)

**Root Cause Analysis:**
1. **Dual Implementation Conflict:**
   - Intraday async streaming implementation (`strategies/intraday/gap_fill/`)
   - Daily bar implementation (`strategies/gap_fill.py`)
   - Orchestrator tries intraday first, falls back to daily

2. **Data Staleness:**
   - IntradayDataManager hasn't downloaded new data since Dec 26
   - No backfill mechanism for missed days
   - No staleness alerting implemented

3. **Architectural Mismatch:**
   - Intraday strategy needs MarketDataStream (async callbacks)
   - Daily scheduler calls it as synchronous strategy

**Immediate Action Required:**
1. Refresh intraday data (currently 14 days old)
2. Clarify which implementation to use (intraday vs daily)
3. Add data staleness check to premarket phase
4. Debug why gap detection returns no candidates

---

#### BUG-005: Signals Table Empty
**Status:** FIXED (2026-01-04)
**Impact:** Signals now stored for all 7 enabled strategies

**Root Cause:**
The `StrategyScheduler` was initialized without registering strategies.

**Fix Applied:**
1. Added runner functions for all strategies in `execution/scheduler.py`:
   - `run_mean_reversion()`, `run_vix_regime_rotation()`, `run_vol_managed_momentum()`
   - `run_quality_smallcap_value()`, `run_factor_momentum()`

2. Updated `create_default_scheduler()` to register all 7 strategies:
   - 09:31: gap_fill
   - 10:00: mean_reversion, relative_volume_breakout
   - 10:30: vix_regime_rotation, vol_managed_momentum
   - 11:00: quality_smallcap_value, factor_momentum

3. Updated `daily_orchestrator.py` to use `create_default_scheduler()`

**Files Modified:**
- `execution/scheduler.py` - Added 5 runner functions, updated `create_default_scheduler()`
- `daily_orchestrator.py` - Import and use `create_default_scheduler()`

---

### P2 - Medium (Nice to Have)

#### BUG-006: ML Regime Detector Insufficient Samples Warning
**Status:** EXPECTED BEHAVIOR (2026-01-04)
**Impact:** None - model trains successfully despite warning

**Investigation:**
- Warning fires when a regime has < 30 samples (`min_samples_per_regime = 30` in config)
- Crisis regime (VIX > 35) is inherently rare in historical data
- Model continues training with cross-validation after the warning
- This is informational, not an error

**Resolution:**
Accept as expected behavior. Crisis regimes are rare by definition. The model handles this via:
- TimeSeriesSplit cross-validation
- RandomForest/GradientBoosting handles class imbalance reasonably well

**Optional Future Enhancement:**
Add SMOTE or class weighting if crisis regime prediction accuracy is insufficient.

---

#### BUG-007: Circuit Breaker Test Data in Production DB
**Status:** FIXED (2026-01-04)
**Impact:** None (cosmetic cleanup)

**Resolution:**
Deleted 1 test entry from `circuit_breaker_state` table:
```sql
DELETE FROM circuit_breaker_state WHERE reason LIKE '%test%'
```

---

## ARCHITECTURE

### P0 - Critical

#### ARCH-001: Single Point of Failure - Daily Orchestrator
**Status:** RESOLVED (2026-01-08)
**Impact:** Crash recovery now handled gracefully

**Solution Applied:**
Added comprehensive startup recovery sequence to `daily_orchestrator.py`:

1. **Orphan Signal Cleanup** (`signal_tracker.py`)
   - Expires stale pending signals (>24h)
   - Cancels old submitted signals
   - Identifies orphaned positions

2. **Broker Credential Validation**
   - Validates API connection before trading
   - Logs account equity for confirmation

3. **Position Reconciliation**
   - Syncs local DB with broker positions
   - Warns about positions closed externally

4. **Market Holiday Check**
   - Uses Alpaca calendar API to detect holidays
   - Prevents unnecessary trading attempts

**Files Modified:**
- `daily_orchestrator.py` - Added `_startup_recovery()` method
- `execution/signal_tracker.py` - Added `cleanup_orphaned_signals()` method

---

### P1 - High

#### ARCH-002: No Real-Time P&L Dashboard
**Status:** RESOLVED (2026-01-09)
**Impact:** Live P&L now visible in dashboard

**Solution Applied:**
- `live_pnl` calculated from positions in `observability/dashboard/app.py:1551`
- Sums `unrealized_pl` from all open positions
- Equity curve chart exists at line 2283
- Dashboard refreshes periodically during market hours

**Files:**
- `observability/dashboard/app.py` - Lines 1551-1554 calculate live P&L

---

#### ARCH-003: Error Log Not Populated
**Status:** VERIFIED WORKING (2026-01-09)
**Impact:** Error handler is wired correctly

**Investigation:**
- `DatabaseErrorHandler` is imported and attached at `daily_orchestrator.py:145,172`
- Handler configured for `min_level=logging.WARNING`
- `error_log` table exists but currently empty (no errors since handler added)
- This indicates system stability, not a bug

**Files:**
- `daily_orchestrator.py` - Lines 145, 172 wire up DatabaseErrorHandler
- `observability/logger.py` - DatabaseErrorHandler implementation

---

#### ARCH-004: Position Sync on Startup
**Status:** VERIFIED WORKING (2026-01-04)
**Impact:** Positions correctly sync from broker

**Investigation:**
The `sync_positions()` method in `execution/alpaca_connector.py:474-524` is complete:
1. Gets broker positions and local positions
2. Creates new local positions for broker positions not in DB (marks as 'manual')
3. Closes local positions that no longer exist in broker
4. Updates current prices for all positions

**Sync runs at PRE_MARKET phase** (`_task_sync_positions_from_broker` in orchestrator).

**Verified:**
- 8 broker positions were synced on Jan 4 after schema fix
- Logs show: "Position sync complete: X new, Y updated, Z closed"

**Remaining:** Monitor Monday morning sync to confirm positions populate correctly.

---

### P2 - Medium

#### ARCH-005: Intraday Data Rolling Window
**Status:** CRITICAL - DATA 14 DAYS STALE (2026-01-09)
**Impact:** Gap-fill strategy has no fresh data

**Investigation Findings (Jan 9):**

**What Exists (`data/fetchers/intraday_bars.py`):**
- ✅ `IntradayDataManager` class with 6 methods
- ✅ `download_recent()` - Downloads N days of 1-min bars
- ✅ `cleanup_old_data()` - Removes data older than retention
- ✅ `get_data_status()` - Reports days available per symbol

**Current Data Age (CRITICAL):**
- **Latest SPY data: December 26, 2025** (14 days old!)
- File: `data/historical/intraday/SPY/20251226.parquet`
- Last modified: Dec 28, 2025 19:19:38

**What's Missing:**
- ❌ No backfill mechanism for missed days
- ❌ No staleness alerting (should alert if data > 1 day old)
- ❌ No retry logic for failed/partial downloads
- ❌ No data freshness check in premarket phase

**Impact:**
Gap-fill strategy is enabled (10% allocation, research Sharpe 2.38) but has NO fresh intraday data to work with. This explains why it generates 0 signals.

**Immediate Fix Required:**
1. Add intraday data refresh to premarket phase
2. Add staleness check that alerts if data > 1 day old
3. Implement backfill retry for missed days

---

#### ARCH-006: No Alerting System
**Status:** RESOLVED (2026-01-10)
**Impact:** Now receiving Telegram alerts for warnings, errors, and critical events

**Solution Applied (Jan 10):**
1. Added `TelegramHandler` class to `execution/alerts.py`
   - Uses Telegram Bot API (`/sendMessage` endpoint)
   - Formats alerts with emoji icons and Markdown
   - Sends WARNING+ level alerts to Telegram
2. Added Telegram config to `config.py`:
   - `TELEGRAM_BOT_TOKEN` from environment
   - `TELEGRAM_CHAT_ID` from environment
3. Credentials stored in `.env`:
   - Bot: @thisElazar_TradeBot
   - Chat ID: User's personal chat
4. Wired up in `daily_orchestrator.py`:
   - `alert_manager` property now initializes handlers
   - Console (INFO+), File (DEBUG+), Telegram (WARNING+)

**Alert Triggers:**
- Circuit breaker events
- Position exits (stop loss, take profit)
- Strategy errors
- Critical system issues
- Daily performance summary

---

## STABILITY

### P1 - High

#### STAB-001: Memory Pressure Monitoring
**Status:** RESOLVED (2026-01-09)
**Impact:** Memory alerts now logged to database

**Solution Applied:**
- `_log_memory_alert()` method in `daily_orchestrator.py` logs to `memory_alerts` table
- Periodic memory checks integrated into orchestrator loop
- 3 memory alerts already captured in database
- Dashboard can query alerts for visibility

**Files Modified:**
- `daily_orchestrator.py` - Added `_log_memory_alert()` at line 3945

---

#### STAB-002: Database Backup Strategy
**Status:** RESOLVED (2026-01-10)
**Impact:** Data loss risk mitigated - daily backups now active

**Solution Applied (Jan 10):**
```bash
sudo systemctl enable trading-backup.service
sudo systemctl start trading-backup.timer
```

**First Backup Verified:**
- Location: `~/trading_system/data/backups/databases/20260110_190511/`
- Total size: ~17 MB (research.db 16MB, trades.db 344KB, performance.db 160KB, pairs.db 40KB)
- Schedule: Daily at 5 AM, 14-day retention

**Additionally Configured:**
- GitHub repository for code backups: `github.com/thisElazar/trading_system`
- SSH key authentication configured
- `.gitignore` updated to exclude databases, logs, and secrets

**Backup Script Features:**
- SQLite backup API with WAL mode support
- 14-day retention (MAX_BACKUPS = 14)
- Verification, restoration, and cleanup
- Supports `--full`, `--list`, `--restore` flags

---

#### STAB-003: Log Rotation
**Status:** VERIFIED WORKING (2026-01-09)
**Impact:** Disk fill risk mitigated

**Investigation Findings (Jan 9):**
- ✅ RotatingFileHandler configured in `daily_orchestrator.py` (lines 146-174)
- ✅ RotatingFileHandler configured in `observability/logger.py` (lines 168-186)
- ✅ TimedRotatingFileHandler for trades.log (30-day retention)
- ✅ Config: `LOG_MAX_BYTES = 10_000_000` (10 MB), `LOG_BACKUP_COUNT = 5`

**Current Log Sizes:**
- `orchestrator.log`: 6.2 MB (within limit)
- `nightly_research.log`: 8.2 MB (within limit)
- `nightly_research.log.1`: 30 MB (old backup before rotation implemented)

**Note:** Rotation is code-based, not system-level (`/etc/logrotate.d/`). Working correctly.

---

### P2 - Medium

#### STAB-004: Graceful Shutdown Handling
**Status:** RESOLVED (2026-01-09)
**Impact:** Clean shutdown with order cancellation

**Solution Applied:**
Signal handlers and comprehensive cleanup already existed. Added open order cancellation:

1. ✅ Signal handlers for SIGTERM/SIGINT (lines 405-406)
2. ✅ `_handle_shutdown()` sets shutdown event (line 4577)
3. ✅ `_cleanup()` now includes (lines 4582-4623):
   - Stop intraday stream
   - Stop scheduler thread (30s timeout)
   - **Cancel any open orders** (NEW - prevents unexpected fills)
   - Generate final daily report
   - Shutdown hardware
4. ✅ Main loop wrapped in try/finally ensuring cleanup runs
5. ✅ Startup recovery handles crash recovery (orphan cleanup, position reconciliation)

**Files Modified:**
- `daily_orchestrator.py` - Added order cancellation to `_cleanup()` at lines 4599-4614

---

#### STAB-005: Dependency Version Pinning
**Status:** FIXED (2026-01-04)
**Impact:** Future pip installs now reproducible

**Resolution:**
Created `requirements.lock` with 84 pinned dependencies:
- Python 3.13.5
- Platform: Raspberry Pi 5
- Install with: `pip install -r requirements.lock`

---

## RESEARCH / GENETIC ALGORITHM

### P0 - Critical

#### GA-001: GA Stagnation Detection
**Status:** IMPLEMENTED (2026-01-08)
**Impact:** Reduced wasted compute on converged populations

**Evidence:**
```
mean_reversion: generations 26-31 all at fitness 0.53
```

**Solution Applied:**
The `PersistentGAOptimizer` now has comprehensive anti-stagnation:

1. **Fitness-based diversity injection** - Replace low-fitness individuals when >50% failing
2. **Similarity-based diversity injection** - Detect and break converged populations (>85% identical genes)
3. **Adaptive mutation** - Increase from 15% → 40% when stagnating
4. **Hard reset** (NEW) - After 10 generations without improvement, reset entire population except top 2 elites

**Files Modified:**
- `research/genetic/persistent_optimizer.py` - Added `_hard_reset_population()` method

---

### P1 - High

#### GA-002: Erratic Fitness Jumps
**Status:** RESOLVED (2026-01-09)
**Impact:** Issue no longer reproducible - fitness bounds enforced

**Investigation Findings (Jan 9):**
- ✅ **Zero fitness values > 3** in entire ga_history table
- ✅ **vol_managed_momentum has only 5% volatility** (most stable strategy!)
- ✅ Fitness values bounded: 0.2291 → 0.2409 (only 4 unique values across 176 gens)
- ✅ Population converged to stable optimum with zero variance in last 20 generations

**Fitness Capping Logic (fitness_utils.py):**
- Sharpe capped at 3.0
- Sortino capped at 4.0
- Calmar capped at 3.0
- Win Rate capped at 2.0
- Maximum theoretical composite fitness: 3.2 (impossible to exceed)

**Why Original Issue No Longer Occurs:**
- Composite fitness formula with hard caps prevents unrealistic values
- Soft constraint penalties (suspicious_sharpe > 2.5 penalized)
- Hard rejection for constraint violations replaced with REJECTION_FITNESS = 0.01

**Files:**
- `research/genetic/fitness_utils.py` - Capping logic (lines 37-85)
- `research/genetic/persistent_optimizer.py` - Constraint penalties (lines 104-207)

---

#### GA-003: GP Discovery Pipeline Not Validated
**Status:** RESOLVED (2026-01-10)
**Impact:** Core R&D capability validated and working

**Validation Results (Jan 10):**
- 107 discovered strategies in `discovered_strategies` table
- 103 meet promotion thresholds (Sharpe >= 0.5, Sortino >= 0.8)
- Best strategy: Sharpe 1.69, Sortino 2.11
- Each genome has 5 trees: entry, exit, position, stop_loss, target
- Serialization/deserialization verified working
- Database stores complete genome JSON with all tree structures

---

### P2 - Medium

#### GA-004: Portfolio-Level Fitness Not Active
**Status:** RESOLVED - VERIFIED WORKING (2026-01-10)
**Impact:** Portfolio-level optimization is active in production

**Verification (Jan 10):**
The punchlist entry was incorrect. The `use_portfolio_fitness=False` was only in a TEST function (`overnight_runner.py:669`), not in production code.

**Evidence:**
- Production code in `run_nightly_research.py` uses default (`use_portfolio_fitness=True`)
- Logs confirmed: "Portfolio fitness evaluation enabled"
- `PortfolioFitnessEvaluator` is being used during nightly research

**Features Active:**
- Marginal Sharpe calculation: MSR = (σ_s/σ_p) × SR_s - β × SR_p
- Correlation threshold: 0.70 max to reject redundancy
- Composite weights: 25% standalone Sortino, 30% marginal Sharpe, 20% diversification, 15% max DD, 10% novelty

---

#### GA-005: Novelty Archive Tuning
**Status:** RESOLVED (2026-01-10)
**Impact:** Dynamic adaptive tuning now implemented

**Solution Applied (Jan 10):**
Enhanced `NoveltyArchive` class in `research/discovery/novelty_search.py`:

1. **Adaptive k_neighbors:**
   - `k = base_k * (0.5 + 0.5 * fill_ratio)`
   - Scales from ~10 (half-full) to ~20 (full archive)
   - Minimum k = 5 to ensure valid novelty calculation

2. **Archive Expansion:**
   - Tracks `_diversity_history` over 10-generation window
   - When diversity drops >15%, expands archive by 1.5x
   - Prevents premature convergence

3. **New Method:** `update_adaptive_params()`
   - Returns dict with adjustments made
   - Logs k_before, k_after, size_before, size_after
   - Serialized via `to_dict()`/`from_dict()` for checkpoint persistence

**Verification:**
Tested with archive at various fill levels - k and size adjust correctly

---

#### GA-006: Backtest Speed Optimization
**Status:** RESOLVED (2026-01-10)
**Impact:** Pool resource exhaustion fixed - stable parallel evolution

**Root Cause (Jan 10):**
`GeneticOptimizer._evaluate_population_parallel()` created a NEW `multiprocessing.Pool` on every generation (20-30 times per session), causing:
- Resource exhaustion (orphaned workers)
- Pipe/fork deadlocks
- Process stalls requiring manual intervention

**Solution Applied:**
Refactored `PersistentGAOptimizer` to own a persistent pool:

1. **Pool lifecycle management** (`persistent_optimizer.py`):
   - `init_pool()` - Creates pool once with staggered worker warmup
   - `shutdown_pool()` - Clean shutdown
   - `__enter__/__exit__` - Context manager for automatic cleanup
   - `_evaluate_population_with_pool()` - Reuses pool across generations

2. **Context manager pattern** (`run_nightly_research.py`):
   ```python
   with PersistentGAOptimizer(name, fitness_fn, config) as optimizer:
       optimizer.load_population()
       optimizer.evolve_incremental(generations=10)
   # Pool automatically cleaned up
   ```

3. **Lambda pickling fix**:
   - Replaced `lambda x: True` with module-level `_warmup_worker()` function

**Benefits:**
- 1 pool per strategy vs 20-30 pools per session
- Eliminates deadlocks and resource exhaustion
- Automatic cleanup via context manager
- Follows existing `RapidBacktester`/`GAWorkerPool` pattern

---

## GP RESEARCH GAPS (Literature Review - January 2026)

*Based on synthesis of academic literature and practitioner insights on genetic programming for autonomous trading strategy discovery. See `GP_RESEARCH_SYNTHESIS.md` for full research document.*

### P0 - Critical (Significant Research Gaps)

#### GP-007: Paper Trading Duration Too Short
**Status:** RESOLVED (2026-01-09)
**Impact:** Paper trading now requires statistically significant sample

**Solution Applied:**
```python
# promotion_pipeline.py:91-92
min_paper_days: int = 90       # 3 months minimum
min_paper_trades: int = 60     # Approaching statistical significance
```

**Files Modified:** `research/discovery/promotion_pipeline.py`

---

#### GP-008: CPCV Validation Not Implemented
**Status:** RESOLVED (2026-01-09)
**Impact:** Overfitting protection now in place

**Solution Applied:**
Full CPCV implementation at `research/validation/cpcv.py`:
- `generate_cpcv_splits()` - Creates S-subset combinations with purging
- `calculate_pbo()` - Probability of Backtest Overfitting calculation
- `run_cpcv_validation()` - Full validation pipeline
- `validate_strategy_with_cpcv()` - High-level API

**Config in `research/discovery/config.py:89-95`:**
```python
cpcv_n_subsets: int = 16          # S parameter (12,870 combinations)
cpcv_purge_days: int = 5          # Gap for leakage prevention
cpcv_embargo_pct: float = 0.01    # Skip 1% after train end
cpcv_max_combinations: int = 1000 # Sample for efficiency
cpcv_pbo_threshold: float = 0.05  # Reject if PBO > 5%
cpcv_n_workers: int = 2           # Parallel workers
```

**Integration:** `promotion_pipeline.py:1044` has `validate_with_cpcv()` method

**Files:**
- `research/validation/cpcv.py` - Full implementation
- `research/discovery/config.py` - Configuration
- `research/discovery/promotion_pipeline.py` - Integration

---

### P1 - High (Important Improvements)

#### GP-009: Add Calmar Ratio to Fitness
**Status:** RESOLVED (2026-01-09)
**Impact:** Calmar ratio now included in fitness evaluation

**Solution Applied:**
- `calculate_calmar_ratio()` implemented in `multi_objective.py:111-140`
- Included in `FitnessVector` dataclass at line 54
- Also in `fitness_utils.py:49-69` with 20% weight in composite fitness
- Capped at [-5, 10] to prevent outliers

**Files:**
- `research/discovery/multi_objective.py` - Lines 43, 54, 107, 111-140, 333, 341
- `research/genetic/fitness_utils.py` - Lines 49-69, 83

---

#### GP-010: HMM-Based Regime Detection
**Status:** RESOLVED (2026-01-09)
**Impact:** Full HMM regime detection now available

**Solution Applied:**
Complete implementation at `research/hmm_regime_detector.py`:
- `HMMRegimeDetector` class with 2-3 state GaussianHMM
- `HMMConfig` dataclass for configuration
- Fallback covariance types for robust fitting
- State mapping to interpretable regime names
- Model persistence (save/load)
- Confidence thresholds and VIX blending

**Config in `config.py:232-253`:**
```python
HMM_REGIME_CONFIG = {
    "enabled": True,
    "n_states": 3,
    "confidence_threshold": 0.7,
    "blend_with_vix": True,
    "blend_hmm_weight": 0.6,
    "fallback_to_vix": True,
}
```

**Files:**
- `research/hmm_regime_detector.py` - Full 500+ line implementation
- `config.py` - HMM_REGIME_CONFIG at lines 232-253

---

#### GP-011: Regime-Specialist Islands
**Status:** RESOLVED (2026-01-09)
**Impact:** Migration rates now match research recommendations

**Solution Applied:**
Migration parameters updated in `research/discovery/config.py:190-191`:
```python
migration_interval: int = 50    # Every 50 generations (was 5)
migration_rate: float = 0.03    # 3% migration (was 0.15)
```

This preserves island specialization as recommended by research (2-5% every 50 generations).

**Files Modified:** `research/discovery/config.py`

---

#### GP-012: Novelty Pulsation for Plateaus
**Status:** RESOLVED (2026-01-09)
**Impact:** Adaptive novelty weight now shifts during plateaus

**Solution Applied:**
- `_detect_plateau()` implemented in `diversity_metrics.py:438`
- `get_effective_novelty_weight()` at line 476 returns elevated weight during plateaus
- `novelty_weight_plateau: 0.7` configured in `config.py:40`
- Integration in `evolution_engine.py:803-810` uses adaptive weight

**How It Works:**
- Normal operation: `novelty_weight = 0.3`
- Plateau detected (10 gens, <5% improvement): `novelty_weight = 0.7`
- This shifts selection pressure toward exploration when stuck

**Files:**
- `research/discovery/diversity_metrics.py` - Lines 358, 438, 476
- `research/discovery/config.py` - Line 40
- `research/discovery/evolution_engine.py` - Lines 803-810

---

### P2 - Medium (Nice to Have)

#### GP-013: Self-Adaptive Mutation Rates
**Status:** RESOLVED (2026-01-10)
**Impact:** Per-individual mutation rates now evolve with strategy genes

**Solution Applied (Jan 10):**
Enhanced `StrategyGenome` dataclass in `research/discovery/strategy_genome.py`:

1. **New Fields:**
   ```python
   mutation_rate: float = field(default_factory=lambda: random.uniform(0.15, 0.35))
   crossover_rate: float = field(default_factory=lambda: random.uniform(0.6, 0.9))
   ```

2. **ES Log-Normal Adaptation in `mutate()`:**
   ```python
   tau = 1.0 / (2 * 5**0.5)  # Learning rate
   new_mutation_rate = mutation_rate * (1 + tau * random.gauss(0, 1))
   new_mutation_rate = max(0.05, min(0.5, new_mutation_rate))  # Clamp
   ```

3. **Serialization:**
   - `to_dict()` includes mutation_rate and crossover_rate
   - Backwards compatible with old genomes (uses defaults)

**Verification:**
Tested mutation - rates evolve each generation within valid bounds

---

#### GP-014: Omega Ratio for Non-Normal Distributions
**Status:** RESOLVED (2026-01-10)
**Impact:** Full distribution capture now available for non-normal returns

**Solution Applied (Jan 10):**
Added `calculate_omega_ratio()` to `research/genetic/fitness_utils.py`:

```python
def calculate_omega_ratio(
    returns: Union[pd.Series, np.ndarray, list],
    threshold: float = 0.0,
    annualize: bool = False,
    periods_per_year: int = 252
) -> float:
    """Omega(r) = sum(gains above threshold) / sum(losses below threshold)"""
```

**Features:**
- Handles pd.Series, np.ndarray, and list inputs
- Threshold can be annualized (converts to per-period rate)
- Capped at [0.1, 10.0] to prevent outliers
- Returns 1.0 for insufficient data (<10 points)
- Returns 10.0 for all-gains case (perfect strategy)

**Interpretation:**
- Omega = 1.0: Gains equal losses (breakeven)
- Omega > 1.0: More gains than losses (desirable)
- Omega < 1.0: More losses than gains (undesirable)

**Verification:**
Tested with good (Omega=2.4), bad (Omega=0.48), and skewed (Omega=1.4) strategies

---

#### GP-015: 2-Day Lag Before Regime Changes
**Status:** NOT IMPLEMENTED (2026-01-09)
**Impact:** Potential whipsaws from noisy regime detection

**Investigation Findings (Jan 9):**

**ML Regime Detector (`ml_regime_detector.py`):**
- ❌ No lag/delay logic before acting on regime changes
- `predict_regime()` returns immediate classification
- `transition_lookback_days: int = 5` defined but unused in actual logic
- Transition detection only checks probability spread (reactive, not anticipatory)

**HMM Regime Detector (`hmm_regime_detector.py`):**
- ❌ Pure probabilistic detection with immediate regime assignment
- `detect_regime()` returns current regime without lag
- `predict_next_regime()` exists for forecasts but not for delaying signals

**What's Missing:**
- No confirmation counter requiring regime stability across 2+ consecutive days
- No state machine tracking "detected" vs "confirmed" regimes
- No smoothing filter for brief regime fluctuations

**Recommendation:**
Add `regime_confirmation_days: int = 2` parameter and require stable regime classification before propagating to strategies

---

#### GP-016: Alpha Decay Detection System
**Status:** BASIC ONLY - Missing Rolling Windows (2026-01-09)
**Impact:** May not catch decay early enough

**Investigation Findings (Jan 9):**

**What Exists (`promotion_pipeline.py`):**
- `check_live_for_retirement()` (lines 571-604)
- Only checks: `live_sharpe < 0.0` (line 601-602)
- `min_rolling_sharpe: float = 0.0` threshold (line 99)

**What's Missing:**
- ❌ No rolling Sharpe calculations (36-month, 12-month, 3-month windows)
- ❌ No trend analysis of Sharpe degradation over time
- ❌ No factor correlation monitoring (momentum/value crowding detection)
- ❌ No slippage trend tracking (paper vs live execution comparison)
- ❌ No time-series analysis in database schema

**Comment at Line 577:**
"Rolling Sharpe negative" mentioned but only as variable name, not calculated

**Research Recommendation:**
- Rolling 36-month Sharpe for trend identification
- Factor correlation > 0.6 signals crowding risk
- Slippage growth indicates capacity constraints

---

#### GP-017: Behavioral Descriptor Enhancement
**Status:** RESOLVED (2026-01-10)
**Impact:** Full 10-dimensional behavioral diversity capture

**Solution Applied (Jan 10):**
Enhanced `BehaviorVector` dataclass in `research/discovery/novelty_search.py`:

**New Dimensions Added:**
```python
recovery_time: float = 0.0      # Days to recover from max DD (normalized by 252)
profit_factor: float = 1.0      # Gross profit / gross loss (clamped 0.5-2.0)
sharpe_ratio: float = 0.0       # Risk-adjusted return (normalized -1 to +1)
```

**Full 10-Dimension Vector:**
1. `trade_frequency` - Trades per week
2. `avg_hold_period` - Days, log-normalized
3. `long_short_ratio` - -1 to +1 balance
4. `return_autocorr` - -1 to +1 momentum vs mean-reversion
5. `drawdown_depth` - 0 to 1 normalized
6. `benchmark_corr` - -1 to +1 systematic vs idiosyncratic
7. `signal_variance` - Normalized variance of position changes
8. `recovery_time` - Days to recover from max drawdown (NEW)
9. `profit_factor` - Gross profit / gross loss ratio (NEW)
10. `sharpe_ratio` - Direct Sharpe in behavior space (NEW)

**Backwards Compatibility:**
- `from_array()` handles old 7-dim arrays (uses defaults for new fields)
- `to_array()` always outputs 10-dim for new genomes

**Verification:**
Tested creation, serialization, and backwards compatibility all working

---

### Quick Wins (Effort vs Impact)

| ID | Fix | Effort | Impact | Status |
|----|-----|--------|--------|--------|
| GP-007 | Paper trading duration | 5 min | HIGH | ✅ DONE |
| GP-009 | Add Calmar ratio | 30 min | MEDIUM | ✅ DONE |
| GP-011 | Reduce migration rate | 5 min | MEDIUM | ✅ DONE |
| GP-012 | Add plateau detection | 1 hour | MEDIUM | ✅ DONE |
| GP-013 | Self-adaptive mutation | 30 min | HIGH | ✅ DONE (Jan 10) |
| GP-014 | Omega Ratio | 20 min | MEDIUM | ✅ DONE (Jan 10) |
| GP-017 | Behavioral descriptors | 30 min | MEDIUM | ✅ DONE (Jan 10) |
| GP-015 | Add regime change lag | 30 min | LOW | Open |

---

### References

1. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
2. Bailey, D. H. & López de Prado, M. (2014). "The Probability of Backtest Overfitting." *Journal of Computational Finance*.
3. López de Prado, M. (2024). "The Three Types of Backtests."
4. Vivek et al. (2024). "Comparison of AGE-MOEA, NSGA-II, and MOEA/D for Trading." *Computational Economics*.
5. Nystrup et al. (2016). "Regime-switching strategies in global equity portfolios."
6. Mouret, J.-B. & Clune, J. (2015). "Illuminating search spaces by mapping elites."
7. Shahrzad et al. (2019). "Novelty Pulsation: A Method for Maintaining Diversity."
8. Menoita & Vanneschi (2025). "Vectorial GP for Trading" (strongly-typed VGP always among the best).
9. Esfahanipour et al. (2011). "Conditional Sharpe Ratio using CVaR."
10. arXiv:2501.03919 (Jan 2025). "Structured portfolios and predictor-level diversity."
11. arXiv:2008.09471 (2020). "GA-MSSR: Combining Sharpe and Sterling ratios."

---

## Quick Reference: Top 10 Priorities

| # | ID | Description | Effort | Impact | Status |
|---|-----|-------------|--------|--------|--------|
| 1 | BUG-004 | Gap-fill 0 signals + stale data | HIGH | HIGH | ✅ RESOLVED (Jan 9) |
| 2 | STAB-002 | Enable backup service | 5 min | HIGH | ✅ RESOLVED (Jan 10) |
| 3 | ARCH-005 | Refresh intraday data (14d stale) | Medium | HIGH | ✅ RESOLVED (Jan 9) |
| 4 | GP-007 | Paper trading duration (14d → 90d) | 5 min | HIGH | ✅ RESOLVED |
| 5 | GP-008 | CPCV validation + PBO threshold | High | HIGH | ✅ RESOLVED |
| 6 | BUG-003 | Sector rotation GA optimization | Low | Medium | ✅ RESOLVED |
| 7 | GA-002 | Erratic fitness jumps | Low | Medium | ✅ RESOLVED |
| 8 | ARCH-002 | Real-time P&L dashboard | Medium | Medium | ✅ RESOLVED |
| 9 | GA-006 | Persistent pool refactor | Medium | HIGH | ✅ RESOLVED (Jan 10) |
| 10 | GA-004 | Enable portfolio fitness | 5 min | Medium | ⚪ DISABLED |

---

---

## PRE-LAUNCH AUDIT (Updated January 7, 2026)

### System Status: LIVE TRADING ACTIVE

| Component | Status | Notes |
|-----------|--------|-------|
| Services | Running | Orchestrator + Dashboard both active 3+ hours |
| Alpaca Connection | Active | $96,715 equity, 8 positions |
| Memory | Healthy | 1.3GB used, 2.7GB available |
| Disk | Healthy | 9% used (203GB free) |
| Kill Switch | Clear | No halt files present |
| Data | Live | Data refreshing during market hours |
| Hardware | Active | LED breathing working, LCD display operational |
| RapidGainScaler | Disabled | Removed to let strategy performance shine through |

### Recent Session Fixes (Jan 7)

1. **LED Breathing Fix**
   - `flash_all()` now preserves and restores breathing state
   - Research LED properly shows blue breathing during overnight evolution
   - Added `_breathing_params` dict to track breathing configuration

2. **RapidGainScaler Disabled**
   - Removed `check_rapid_gains` from MARKET_OPEN phase tasks
   - Allows strategy performance to flow through without early trimming
   - Can re-enable later if desired with modified thresholds

3. **Documentation Created**
   - `docs/DAY_CYCLE.md` - Complete operational cycle reference
   - Updated all related docs with cross-references

### Items Already Resolved (from Jan 4)

- ✅ Position sync working
- ✅ Timezone bug fixed
- ✅ ML regime detector training
- ✅ Pairs trading disabled (cash account constraint)
- ✅ GP discovery validated (100+ strategies)

### Emergency Procedures

```bash
# Stop all trading immediately
touch ~/trading_system/killswitch/HALT

# Check system status
systemctl status trading-orchestrator trading-dashboard

# View live logs
tail -f ~/trading_system/logs/orchestrator.log

# Check Alpaca positions
cd ~/trading_system && source venv/bin/activate
python3 -c "from execution.alpaca_connector import AlpacaConnector; c = AlpacaConnector(); print(c.summary())"
```

---

## Changelog

| Date | Change |
|------|--------|
| 2026-01-04 | Initial punchlist created |
| 2026-01-04 | Pre-launch audit added |
| 2026-01-04 | BUG-002 (timezone) FIXED with utils/timezone.py |
| 2026-01-04 | ARCH-003 (error logging) FIXED - DatabaseErrorHandler added to orchestrator & research |
| 2026-01-04 | STAB-003 (log rotation) FIXED - RotatingFileHandler added to orchestrator & research |
| 2026-01-04 | STAB-002 (database backups) FIXED - scripts/backup_databases.py + systemd timer created |
| 2026-01-04 | BUG-001 (pairs trading) RESOLVED - Architectural constraint: cash account cannot short. Disabled strategy, reallocated 5% to mean_reversion |
| 2026-01-04 | GA-003 (GP discovery) VALIDATED - Pipeline tested end-to-end, 117 discovered strategies, working correctly |
| 2026-01-04 | BUG-007 (test data) FIXED - Deleted 1 test entry from circuit_breaker_state |
| 2026-01-04 | STAB-005 (version pinning) FIXED - Created requirements.lock with 84 pinned dependencies (Python 3.13.5) |
| 2026-01-04 | BUG-005 (empty signals) FIXED - Added strategy runners and registration for all 7 strategies |
| 2026-01-04 | BUG-006 (ML samples warning) DOCUMENTED - Expected behavior for rare crisis regimes |
| 2026-01-04 | ARCH-004 (position sync) VERIFIED WORKING - sync_positions() method complete and functional |
| 2026-01-07 | LED breathing fix - flash_all() now preserves and restores breathing state after flashing |
| 2026-01-07 | RapidGainScaler DISABLED - Removed check_rapid_gains from MARKET_OPEN tasks to let strategy performance shine through |
| 2026-01-07 | DAY_CYCLE.md CREATED - Complete documentation of trading system day cycle phases and tasks |
| 2026-01-07 | Documentation refresh - Updated AUTONOMOUS_RESEARCH_ENGINE.md, STRATEGY_PORTFOLIO_OVERVIEW.md, TRADEBOT_RD_OVERVIEW.md |
| 2026-01-08 | GA-001 IMPLEMENTED - Added hard reset mechanism for deeply stuck populations (10+ gens without improvement) |
| 2026-01-08 | LED fix - Removed blanket pkill that caused cross-process GPIO conflicts |
| 2026-01-08 | LCD fix - Research display now queries ga_history table, cleaner format |
| 2026-01-08 | GP RESEARCH GAPS section added - 11 new items (GP-007 to GP-017) based on academic literature synthesis |
| 2026-01-08 | **ARCH-001 RESOLVED** - Startup recovery sequence added (orphan cleanup, broker validation, position reconciliation, market holiday check) |
| 2026-01-08 | **STAB-006 FIXED** - Database thread safety: SQLite now uses per-thread connections with WAL mode |
| 2026-01-08 | **STAB-007 FIXED** - Position race condition (TOCTOU): Added locks and pending symbol tracking to ExecutionManager |
| 2026-01-08 | **STAB-008 FIXED** - Partial fill handling: Orders now wait for fill with timeout, partial fills properly recorded |
| 2026-01-08 | **STAB-009 FIXED** - Screen cache: Persistent JSON cache for instant LCD display on restart |
| 2026-01-08 | **STAB-010 FIXED** - VIX fetch timeout: 10-second timeout prevents orchestrator hanging |
| 2026-01-09 | **GP-007 RESOLVED** - Paper trading duration updated to 90 days / 60 trades minimum |
| 2026-01-09 | **GP-008 RESOLVED** - Full CPCV implementation at `research/validation/cpcv.py` with PBO calculation |
| 2026-01-09 | **GP-009 RESOLVED** - Calmar ratio added to `multi_objective.py` and `fitness_utils.py` |
| 2026-01-09 | **GP-010 RESOLVED** - HMM regime detector implemented at `research/hmm_regime_detector.py` |
| 2026-01-09 | **GP-011 RESOLVED** - Migration rate reduced to 3% every 50 generations |
| 2026-01-09 | **GP-012 RESOLVED** - Novelty pulsation with plateau detection in `diversity_metrics.py` |
| 2026-01-09 | **STAB-001 RESOLVED** - Memory monitoring with `_log_memory_alert()` and database logging |
| 2026-01-09 | **ARCH-002 RESOLVED** - Live P&L calculation in dashboard `app.py:1551` |
| 2026-01-09 | **ARCH-003 VERIFIED** - DatabaseErrorHandler wired correctly (empty log = system stability) |
| 2026-01-09 | Punchlist audit - verified 9 additional items already implemented but not documented |
| 2026-01-09 | **STAB-004 RESOLVED** - Graceful shutdown: Added open order cancellation to `_cleanup()` |
| 2026-01-09 | **COMPREHENSIVE SWARM AUDIT** - 8 parallel agents investigated all open punchlist items |
| 2026-01-09 | **BUG-003 RESOLVED** - Sector rotation GA optimized: Sharpe -0.38 → 1.08, now enabled with 10% allocation |
| 2026-01-09 | **GA-002 RESOLVED** - Erratic fitness jumps no longer reproducible, capping in place (max theoretical 3.2) |
| 2026-01-09 | **BUG-004 CRITICAL** - Gap-fill generating 0 signals; intraday data 14 days stale (Dec 26, 2025) |
| 2026-01-09 | **STAB-002 BROKEN** - Backup infrastructure exists but never runs (service disabled, directory empty) |
| 2026-01-09 | **STAB-003 VERIFIED** - Log rotation working with RotatingFileHandler (10MB, 5 backups) |
| 2026-01-09 | **GA-004** - Portfolio fitness fully implemented but explicitly disabled in overnight_runner.py |
| 2026-01-09 | **GA-006** - Worker count mismatch: config says 4, rapid_backtester defaults to 1 |
| 2026-01-09 | **ARCH-005 CRITICAL** - Intraday data 14 days stale, no backfill mechanism, no staleness alerts |
| 2026-01-09 | **ARCH-006** - Alert infrastructure (476 lines) exists but no webhook configured, logs empty |
| 2026-01-09 | GP-013/014/015/016/017 investigated - various partial implementations documented |
| 2026-01-09 | **ARCH-005 FIXED** - Added `refresh_intraday_data` task to PRE_MARKET phase; downloads 5 days of minute bars for SPY/QQQ/IWM/DIA |
| 2026-01-09 | **BUG-004 FIXED** - Intraday data refreshed (was 14 days stale, now current through Jan 9); GapFillStrategy initialization fixed |
| 2026-01-09 | Intraday data now: SPY/QQQ/IWM/DIA all have 24 days of minute bars (Dec 1, 2025 - Jan 9, 2026) |
| 2026-01-09 | **INTRADAY UNIVERSE EXPANDED** - 42 symbols across 8 categories (broad_market, sectors, mega_caps, volatility, commodities, bonds, international, thematic) |
| 2026-01-09 | Added `INTRADAY_UNIVERSE` and `INTRADAY_SYMBOLS` to config.py with configurable retention (30d) and refresh (5d) settings |
| 2026-01-09 | Updated `IntradayDataManager` to use config-driven universe; backwards compatible via `GAP_FILL_UNIVERSE` alias |
| 2026-01-09 | Data architecture: Daily bars (2,562 symbols, 627MB) + Intraday bars (42 symbols, 5MB) |
| 2026-01-10 | **GA-006 RESOLVED** - Persistent pool refactor: `PersistentGAOptimizer` now owns pool lifecycle with context manager |
| 2026-01-10 | Pool deadlock fix: 1 pool per strategy vs 20-30 per session; eliminates resource exhaustion |
| 2026-01-10 | Lambda pickling fix: Replaced lambda with module-level `_warmup_worker()` function |
| 2026-01-10 | Screen controller: Fixed false "STALLED" during GP discovery (checks `evolution_history` + CPU usage) |
| 2026-01-10 | Screen controller: Market page always shows countdown to next phase (staleness moved to time line) |
| 2026-01-10 | Orchestrator: Added pgrep safeguard to prevent orphaned research processes after restart |
| 2026-01-10 | Screen controller: Fixed generation display ORDER BY tiebreaker (was showing 215 instead of 217) |
| 2026-01-10 | **STAB-002 RESOLVED** - Database backup service enabled; first backup verified (17MB total) |
| 2026-01-10 | GitHub repository configured for code backups: `github.com/thisElazar/trading_system` |
| 2026-01-10 | SSH key authentication set up for Pi → GitHub push access |
| 2026-01-10 | `.gitignore` updated: excludes SQLite WAL files, trained models, backups, Claude settings |
| 2026-01-10 | **GA-003 VALIDATED** - GP Discovery Pipeline verified: 107 strategies discovered, 103 meet promotion thresholds |
| 2026-01-10 | **GA-004 VERIFIED** - Portfolio fitness is ACTIVE in production (punchlist was incorrect - only test function had it disabled) |
| 2026-01-10 | **GA-005 RESOLVED** - Dynamic novelty archive tuning: adaptive k_neighbors and archive expansion on diversity drop |
| 2026-01-10 | **GP-013 RESOLVED** - Self-adaptive mutation rates: per-individual rates evolve with ES log-normal adaptation |
| 2026-01-10 | **GP-014 RESOLVED** - Omega Ratio implemented in `fitness_utils.py` for non-normal distribution capture |
| 2026-01-10 | **GP-017 RESOLVED** - BehaviorVector expanded to 10 dimensions: added recovery_time, profit_factor, sharpe_ratio |

---

*This is a living document. Update as items are resolved or new issues discovered.*
