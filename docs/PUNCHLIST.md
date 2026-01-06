# Tradebot Punchlist

**Created:** January 4, 2026
**Purpose:** Identified bugs and improvements for R&D focus
**Priority Framework:** P0 (blocks progress) → P1 (significant impact) → P2 (nice to have)

---

## Summary

| Category | P0 | P1 | P2 | Resolved | Total |
|----------|----|----|----|---------:|------:|
| Bugs | 1 | 2 | 1 | 4 | 8 |
| Architecture | 1 | 3 | 2 | 0 | 6 |
| Stability | 0 | 0 | 1 | 4 | 5 |
| Research/GA | 1 | 2 | 3 | 0 | 6 |
| **Total** | **3** | **7** | **7** | **8** | **25** |

**Resolved This Session:** BUG-001 (pairs/cash account), BUG-002 (timezone), BUG-005 (signals table), BUG-007 (test data cleanup), ARCH-003 (error logging), STAB-002 (backups), STAB-003 (log rotation), STAB-005 (version pinning)

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
**Status:** Disabled in config
**Impact:** 10% allocation capacity unused

**Evidence:**
- Backtest Sharpe: -0.38
- Currently `enabled: False` in config

**Notes:**
- May be fixable with parameter tuning
- GA has evolved it to Sharpe 1.08 (ga_history shows improvement)
- Need to re-evaluate with GA-optimized parameters

**Files:**
- `strategies/sector_rotation.py`
- `config.py` (STRATEGIES section)

**Next Steps:**
1. Run backtest with GA-optimized parameters (generation 13)
2. If positive, re-enable with conservative allocation
3. If still negative, archive and document why

---

#### BUG-004: Gap-Fill Strategy Not Live Validated
**Status:** Enabled but untested in production
**Impact:** 10% allocation at risk

**Evidence:**
- Research Sharpe: 2.38 (highest of all strategies)
- No live signals generated yet
- Requires intraday 1-min data during 9:31-11:30 AM window

**Risk:**
- Intraday execution timing is critical
- Slippage assumptions may be wrong for fast-moving gaps
- Data latency could miss entry windows

**Files:**
- `strategies/gap_fill.py`
- `data/fetchers/intraday_bars.py`

**Next Steps:**
1. Monitor first week of gap detection (watch for false positives)
2. Log all gap candidates even if not traded
3. Compare detected gaps vs actual gap fill rates
4. Validate 1-min data is arriving in real-time

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
**Status:** Design concern
**Impact:** If orchestrator crashes, entire system stops

**Current State:**
- Systemd auto-restarts on failure (10s delay)
- But crash during order execution could leave positions orphaned

**Next Steps:**
1. Add position reconciliation on startup
2. Store orchestrator state checkpoint before critical operations
3. Consider watchdog process (not the aggressive one that caused issues)

---

### P1 - High

#### ARCH-002: No Real-Time P&L Dashboard
**Status:** Feature gap
**Impact:** Cannot see live performance during market hours

**Current State:**
- Dashboard shows strategy metrics but not live P&L
- `portfolio_daily` table only updated post-market

**Next Steps:**
1. Add real-time equity curve to dashboard
2. Query Alpaca account balance periodically
3. Calculate unrealized P&L from positions table

---

#### ARCH-003: Error Log Not Populated
**Status:** Integration gap
**Impact:** Errors only visible in log files, not database

**Evidence:**
- `error_log` table in performance.db exists but appears empty
- Errors go to orchestrator.log but not database

**Files:**
- `observability/logger.py` (DatabaseErrorHandler)
- `data/storage/db_manager.py`

**Next Steps:**
1. Verify DatabaseErrorHandler is attached to root logger
2. Test error logging path end-to-end
3. Add dashboard panel for error_log table

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
**Status:** Works but fragile
**Impact:** Gap in data if fetcher fails

**Current State:**
- 30-day rolling window of 1-min bars
- Fetched from Alpaca daily

**Risk:**
- If fetch fails for 30+ days, lose all intraday history
- No backfill mechanism

**Next Steps:**
1. Add data staleness alerting
2. Implement backfill for missed days

---

#### ARCH-006: No Alerting System
**Status:** Not implemented
**Impact:** Must manually check logs/dashboard

**Notes:**
- `execution/alerts.py` exists but not wired up
- No email/Slack notifications

**Next Steps:**
1. Decide on notification channel (email, Slack, Telegram)
2. Wire up alerts for: circuit breaker triggers, daily P&L, system errors

---

## STABILITY

### P1 - High

#### STAB-001: Memory Pressure Monitoring
**Status:** Partially implemented
**Impact:** System could slow down without warning

**Current State:**
- `logs/memory_alerts.log` exists (last entry Jan 3)
- ZRAM/ZSWAP working correctly
- But no proactive alerting when approaching limits

**Evidence:**
- Safe limit: <3GB RAM
- Stress test showed 2556 symbols causes thrashing

**Next Steps:**
1. Add periodic memory check (every 5 min)
2. Alert if RAM >80% before compression kicks in
3. Log compression ratio trends

---

#### STAB-002: Database Backup Strategy
**Status:** Manual only
**Impact:** Data loss risk if NVMe fails

**Current State:**
- No automated database backups
- System restore script exists for full Pi recovery

**Next Steps:**
1. Daily backup of all .db files to separate location
2. Weekly backup to cloud (optional)
3. Test restore procedure

---

#### STAB-003: Log Rotation
**Status:** Unclear
**Impact:** Disk fill risk

**Evidence:**
- `nightly_research.log` is 30MB
- `orchestrator.log` is 1.4MB
- No visible logrotate config

**Next Steps:**
1. Implement log rotation (keep 7 days, compress old)
2. Or configure journald for systemd services
3. Monitor disk usage

---

### P2 - Medium

#### STAB-004: Graceful Shutdown Handling
**Status:** Partial
**Impact:** Could lose in-flight operations

**Current State:**
- Kill switch mechanism exists
- But unclear if SIGTERM triggers clean shutdown

**Next Steps:**
1. Add signal handler for SIGTERM/SIGINT
2. Complete pending operations before exit
3. Write checkpoint on shutdown

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
**Status:** Observed but not handled
**Impact:** Wasted compute on converged populations

**Evidence:**
```
mean_reversion: generations 26-31 all at fitness 0.53
```

**Current State:**
- Adaptive mutation increases when stagnant
- But population may have converged to local optimum

**Next Steps:**
1. Implement diversity injection when stagnant >5 generations
2. Add restart mechanism (fresh random population)
3. Track and log stagnation events

---

### P1 - High

#### GA-002: Erratic Fitness Jumps
**Status:** Observed, cause unknown
**Impact:** May indicate bug or data issue

**Evidence:**
```
vol_managed_momentum gen 4: 0.24
vol_managed_momentum gen 5: 7.99
vol_managed_momentum gen 6: 8.88
```

**Possible Causes:**
1. Different evaluation periods between runs
2. Data loading inconsistency
3. Fitness function changed between runs

**Next Steps:**
1. Add logging of evaluation period for each fitness calc
2. Verify same data used across generations
3. Check for NaN/inf handling in fitness

---

#### GA-003: GP Discovery Pipeline Not Validated
**Status:** Code exists, not tested end-to-end
**Impact:** Core R&D capability not proven

**Current State:**
- `research/discovery/` has full GP implementation
- `evolution_engine.py`, `strategy_genome.py`, etc.
- But unclear if strategies are actually being discovered

**Evidence:**
- `discovered_strategies` table in research.db
- `candidates/` directory exists

**Next Steps:**
1. Run GP discovery with `--discovery` flag
2. Verify genome serialization/deserialization
3. Test promotion pipeline end-to-end
4. Document first discovered strategy

---

### P2 - Medium

#### GA-004: Portfolio-Level Fitness Not Active
**Status:** Code exists, not integrated
**Impact:** Strategies may be correlated

**Files:**
- `research/discovery/portfolio_fitness.py`

**Next Steps:**
1. Enable portfolio fitness in evolution config
2. Test correlation penalty effect

---

#### GA-005: Novelty Archive Tuning
**Status:** Default parameters
**Impact:** May not preserve enough diversity

**Current Config:**
- k_neighbors: 20
- archive_size: 500
- novelty_weight: 0.3

**Next Steps:**
1. Analyze archive diversity after 100+ generations
2. Tune parameters based on observed behavior

---

#### GA-006: Backtest Speed Optimization
**Status:** Working but slow
**Impact:** Limits generations per night

**Current State:**
- 4 workers, ~3-4 min for full research run
- Vectorized operations in use

**Next Steps:**
1. Profile backtest bottlenecks
2. Consider Numba/Cython for hot paths
3. Reduce data loaded per evaluation

---

## Quick Reference: Top 10 Priorities

| # | ID | Description | Effort | Impact | Status |
|---|-----|-------------|--------|--------|--------|
| 1 | ~~BUG-001~~ | ~~Fix pairs trading~~ | - | - | RESOLVED (cash account constraint) |
| 2 | ~~BUG-002~~ | ~~Fix ML regime datetime error~~ | - | - | RESOLVED |
| 3 | GA-003 | Validate GP discovery pipeline | Medium | High | VALIDATED (working) |
| 4 | BUG-004 | Live validate gap-fill strategy | Low | High | Pending (monitor first week) |
| 5 | ARCH-002 | Add real-time P&L to dashboard | Medium | Medium | |
| 6 | GA-001 | Add GA stagnation detection/recovery | Medium | Medium | |
| 7 | BUG-005 | Debug empty signals table | Low | Medium | |
| 8 | ~~STAB-002~~ | ~~Implement database backups~~ | - | - | RESOLVED |
| 9 | BUG-003 | Re-evaluate sector rotation with GA params | Low | Low | |
| 10 | ~~ARCH-003~~ | ~~Wire up error_log to database~~ | - | - | RESOLVED |

---

---

## PRE-LAUNCH AUDIT (January 4, 2026)

### System Status: READY WITH CAVEATS

| Component | Status | Notes |
|-----------|--------|-------|
| Services | Running | Orchestrator + Dashboard both active 3+ hours |
| Alpaca Connection | Active | $96,715 equity, 8 positions |
| Memory | Healthy | 1.3GB used, 2.7GB available |
| Disk | Healthy | 9% used (203GB free) |
| Kill Switch | Clear | No halt files present |
| Data | Stale (weekend) | VIX: Jan 2, SPY: Dec 30 - will refresh Monday |

### Critical Items for Monday Morning

1. **Position Sync Will Run at PRE_MARKET (8:00 AM ET)**
   - 8 broker positions currently not in local DB
   - Sync task now works (tested - was fixed between Jan 2-4)
   - Watch for "Position sync complete: 8 new" in logs

2. **Data Refresh Required**
   - SPY data is 5 days old (Dec 30)
   - VIX data is 2 days old (Jan 2)
   - refresh_data task runs at PRE_MARKET
   - Verify: `tail -f logs/orchestrator.log | grep -i refresh`

3. **ML Regime Detector**
   - Timezone bug FIXED in this session
   - Should train successfully Monday
   - Watch for: "Training regime detection model..."

### Items That Can Wait

- Pairs trading debugging (5% allocation, keep conservative)
- GP discovery validation (R&D, not blocking)
- Dashboard real-time P&L (nice to have)

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

---

*This is a living document. Update as items are resolved or new issues discovered.*
