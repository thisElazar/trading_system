# Tradebot Punchlist

**Created:** January 4, 2026
**Purpose:** Identified bugs and improvements for R&D focus
**Priority Framework:** P0 (blocks progress) → P1 (significant impact) → P2 (nice to have)

---

## Summary

| Category | P0 | P1 | P2 | Resolved | Total |
|----------|----|----|----|---------:|------:|
| Bugs | 1 | 2 | 1 | 4 | 8 |
| Architecture | 0 | 3 | 2 | 1 | 6 |
| Stability | 0 | 0 | 1 | 9 | 10 |
| Research/GA | 1 | 2 | 3 | 0 | 6 |
| GP Research Gaps | 2 | 4 | 5 | 0 | 11 |
| **Total** | **4** | **11** | **12** | **14** | **41** |

**Resolved Jan 4:** BUG-001 (pairs/cash account), BUG-002 (timezone), BUG-005 (signals table), BUG-007 (test data cleanup), ARCH-003 (error logging), STAB-002 (backups), STAB-003 (log rotation), STAB-005 (version pinning)

**Resolved Jan 8:** ARCH-001 (startup recovery), STAB-006 (DB thread safety), STAB-007 (TOCTOU race), STAB-008 (partial fills), STAB-009 (screen cache), STAB-010 (VIX timeout)

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

## GP RESEARCH GAPS (Literature Review - January 2026)

*Based on synthesis of academic literature and practitioner insights on genetic programming for autonomous trading strategy discovery. See `GP_RESEARCH_SYNTHESIS.md` for full research document.*

### P0 - Critical (Significant Research Gaps)

#### GP-007: Paper Trading Duration Too Short
**Status:** Configuration issue
**Impact:** Cannot distinguish skill from luck with current thresholds

**Research Basis:**
> "For swing trading generating 2-4 trades weekly, this translates to **6 months minimum** for preliminary validation (120 trades) and **12-18 months** for confident deployment."
> — Section 2: Promotion Pipelines

**Current Config:**
```python
# promotion_pipeline.py PromotionCriteria
min_paper_days: int = 14      # WAY too short
min_paper_trades: int = 10    # Not statistically significant
```

**Required Fix:**
```python
min_paper_days: int = 90       # 3 months minimum
min_paper_trades: int = 60     # Approaching statistical significance
```

**Files:** `research/discovery/promotion_pipeline.py`

**Rationale:** With only 14 days and 10 trades, you have ~10% of the statistical power needed. The research explicitly states a minimum of 120 trades for preliminary validation.

---

#### GP-008: CPCV Validation Not Implemented
**Status:** Major gap
**Impact:** High risk of promoting overfit strategies

**Research Basis:**
> "**Combinatorial Purged Cross-Validation (CPCV)**, developed by López de Prado, divides data into S subsets forming symmetric train/test combinations. With S=16, this generates 12,780 unique backtests, producing a distribution of performance metrics rather than single-point estimates."
> — Section 2: Promotion Pipelines
>
> "The **Probability of Backtest Overfitting** metric measures the probability that the in-sample optimal strategy underperforms the median out-of-sample. After testing only 7 strategy configurations, a researcher should expect to find a 2-year backtest with Sharpe >1.0 even when true out-of-sample Sharpe equals zero."
> — Section 2: Promotion Pipelines

**Current State:**
- Walk-forward validation exists
- Monte Carlo simulation exists
- But CPCV and PBO calculation are NOT implemented

**Required Implementation:**
New module: `research/validation/cpcv.py`

```python
def cpcv_backtest(strategy, data, n_splits=16, purge_days=5, embargo_pct=0.05):
    """
    Combinatorial Purged Cross-Validation (López de Prado, 2018).

    Args:
        n_splits: Number of folds (S=16 recommended → 12,780 combinations)
        purge_days: Days to remove between train/test to prevent leakage
        embargo_pct: Fraction of test data to skip after each train fold

    Returns:
        Distribution of performance metrics for PBO calculation
    """

def calculate_pbo(oos_performances: List[float]) -> float:
    """
    Probability of Backtest Overfitting.

    Returns probability that in-sample optimal underperforms OOS median.
    Reject strategies with PBO > 5%.
    """
```

**References:**
- López de Prado, "Advances in Financial Machine Learning" (2018), Chapter 12
- Bailey & López de Prado, "The Probability of Backtest Overfitting" (2014)
- López de Prado, "The Three Types of Backtests" (July 2024)

---

### P1 - High (Important Improvements)

#### GP-009: Add Calmar Ratio to Fitness
**Status:** Missing from fitness function
**Impact:** Not capturing swing trading risk appropriately

**Research Basis:**
> "The Calmar Ratio (CAGR/Maximum Drawdown) proves particularly relevant for swing trading where drawdown survival matters more than volatility smoothness."
> — Section 1: Fitness Functions
>
> "**Practical recommendation:** Use a weighted composite: `0.4×Sharpe + 0.35×Calmar + 0.15×(normalized_trade_count) + 0.10×(1/complexity)`."
> — Section 1: Fitness Functions

**Current Fitness (multi_objective.py):**
- Sortino ✅
- Max Drawdown ✅
- CVaR 95% ✅
- Novelty ✅
- Deflated Sharpe (informational) ✅

**Missing:** Calmar Ratio (CAGR / |MaxDD|)

**Fix:** Add to `multi_objective.py`:
```python
def calculate_calmar_ratio(result: BacktestResult) -> float:
    """Calmar = CAGR / abs(MaxDD). Critical for swing trading."""
    if result.max_drawdown_pct == 0:
        return 0.0
    cagr = result.annual_return if hasattr(result, 'annual_return') else 0.0
    return cagr / abs(result.max_drawdown_pct)
```

---

#### GP-010: HMM-Based Regime Detection
**Status:** Using RandomForest instead of HMM
**Impact:** Missing latent state modeling and transition probabilities

**Research Basis:**
> "For **regime detection**, Gaussian Hidden Markov Models with 2-3 states remain the most established approach."
> — Section 3: Regime-Adaptive Evolution
>
> "The **Statistical Jump Model** (Princeton, 2024) represents an emerging alternative that enhances regime persistence through jump penalties at state transitions, reducing annual turnover to approximately 44%."
> — Section 3: Regime-Adaptive Evolution

**Current State (`ml_regime_detector.py`):**
- Uses RandomForestClassifier / GradientBoostingClassifier
- Classifies based on features, doesn't model state transitions
- No persistence modeling

**Recommended Addition:**
```python
from hmmlearn import GaussianHMM

class HMMRegimeDetector:
    """2-3 state Gaussian HMM for regime detection."""

    def __init__(self, n_states=2):
        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=100
        )

    def detect_regime(self, features) -> Tuple[int, float]:
        """Returns (regime_id, probability)."""
        state_probs = self.model.predict_proba(features)
        regime = np.argmax(state_probs[-1])
        return regime, state_probs[-1, regime]
```

**References:**
- Nystrup et al. (2016), "Regime-switching strategies in global equity portfolios"
- Hamilton (1989), "A new approach to the economic analysis of nonstationary time series"

---

#### GP-011: Regime-Specialist Islands
**Status:** Islands vary by mutation/depth, not by regime
**Impact:** Not evolving regime-appropriate specialists

**Research Basis:**
> "Research strongly supports **evolving regime-specialist strategies** then combining them through ensemble switching, rather than forcing cross-regime robustness on individual strategies."
> — Section 3: Regime-Adaptive Evolution
>
> "Island model genetic algorithms provide natural architecture for regime-specialist evolution. The recommended configuration maintains four islands: low-volatility specialists with trend-following bias, high-volatility specialists with mean-reversion/defensive bias, transition specialists focusing on regime-change detection, and a generalist pool for cross-regime robust strategies."
> — Section 3: Regime-Adaptive Evolution
>
> "**Migration policy** should use low rates (2-5% every 50 generations) for within-regime islands to preserve specialization."
> — Section 3: Regime-Adaptive Evolution

**Current Config (`config.py`):**
```python
migration_rate: float = 0.15           # TOO HIGH (research: 2-5%)
migration_interval: int = 5            # TOO FREQUENT (research: 50 gens)
```

**Recommended Config:**
```python
@dataclass
class RegimeIslandConfig(IslandConfig):
    num_islands: int = 4
    migration_rate: float = 0.03       # 3% migration
    migration_interval: int = 50       # Every 50 generations
    island_roles: List[str] = field(default_factory=lambda: [
        "low_vol_specialist",      # Trend-following bias
        "high_vol_specialist",     # Mean-reversion/defensive bias
        "transition_specialist",   # Regime-change detection
        "generalist"               # Cross-regime robust
    ])
```

**Files:** `research/discovery/config.py`, `research/discovery/island_model.py`

---

#### GP-012: Novelty Pulsation for Plateaus
**Status:** Fixed novelty weight, no plateau-triggered exploration
**Impact:** May get stuck in local optima

**Research Basis:**
> "For **novelty-fitness weighting**, adaptive temporal switching outperforms fixed ratios. 'Novelty Pulsation' (Shahrzad et al., 2019) systematically alternates between novelty selection and local optimization: when the system hits fitness plateaus, it shifts to novelty exploration before returning to objective optimization. This approach shows order-of-magnitude faster convergence than fixed weighting in deceptive landscapes."
> — Section 5: Quality-Diversity

**Current State:**
- Fixed `novelty_weight: 0.3` in config
- `DiversityMonitor` triggers injection but doesn't shift selection pressure

**Recommended Addition to `evolution_engine.py`:**
```python
def _detect_plateau(self, window: int = 10, threshold: float = 0.05) -> bool:
    """Detect if evolution has stagnated."""
    if self.current_generation < window:
        return False
    recent_best = [self.generation_stats.get(g, {}).get('best_sortino', 0)
                   for g in range(self.current_generation - window, self.current_generation)]
    return (max(recent_best) - min(recent_best)) < threshold

def get_effective_novelty_weight(self) -> float:
    """Pulsate novelty weight based on plateau detection."""
    if self._detect_plateau():
        logger.info("Plateau detected - shifting to exploration (novelty=0.7)")
        return 0.7  # High exploration
    return self.config.novelty_weight  # Normal: 0.3
```

**Reference:** Shahrzad et al. (2019), "Novelty Pulsation: A Method for Maintaining Diversity in Multi-Objective Optimization"

---

### P2 - Medium (Nice to Have)

#### GP-013: Self-Adaptive Mutation Rates
**Status:** Island-level variation only
**Impact:** Not fully exploiting adaptive potential

**Research Basis:**
> "The GA should respond to decay through **self-adaptive parameter control**. Encode mutation and crossover rates within chromosomes, allowing parameters to evolve alongside strategy genes. Research shows augmented self-adaptive approaches outperform static parameter tuning by **65-584%** in solution diversity."
> — Section 6: Strategy Decay Detection

**Current:** Mutation rates vary by island but not per-individual.

**Recommended:** Add mutation_rate and crossover_rate fields to `StrategyGenome` and evolve them.

---

#### GP-014: Omega Ratio for Non-Normal Distributions
**Status:** Not implemented
**Impact:** Missing full distribution capture

**Research Basis:**
> "The Omega Ratio captures all moments of return distribution rather than just mean and variance, making it superior for the non-normal distributions typical in evolved strategies."
> — Section 1: Fitness Functions

**Implementation:**
```python
def calculate_omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """Omega = sum(returns > threshold) / sum(returns < threshold)"""
    gains = returns[returns > threshold].sum()
    losses = abs(returns[returns < threshold].sum())
    return gains / losses if losses > 0 else float('inf')
```

---

#### GP-015: 2-Day Lag Before Regime Changes
**Status:** Unknown
**Impact:** Potential whipsaws from noisy detection

**Research Basis:**
> "**Key implementation insight:** Include a 2-day lag before acting on regime changes to reduce whipsaws from noisy detection."
> — Section 3: Regime-Adaptive Evolution

**Check:** Verify if `ml_regime_detector.py` or ensemble switching has lag before propagating regime signals.

---

#### GP-016: Alpha Decay Detection System
**Status:** Partial (only live Sharpe monitored)
**Impact:** May not catch decay early enough

**Research Basis:**
> "Detection frameworks monitor **rolling Sharpe ratio** (36-month windows) for trend identification, **factor correlation** increases indicating crowding, and **slippage growth** suggesting capacity constraints."
> — Section 6: Strategy Decay Detection
>
> "Track Pearson correlation between strategy returns and known crowded factors—values above 0.6 signal crowding risk requiring retirement consideration."
> — Section 6: Strategy Decay Detection

**Current:** `promotion_pipeline.py` checks `live_sharpe < 0.3` for retirement.

**Needed:**
1. Rolling 36-month Sharpe calculation (not just current)
2. Correlation with momentum/value factors (>0.6 = crowding)
3. Slippage trend monitoring

---

#### GP-017: Behavioral Descriptor Enhancement
**Status:** 7D vector, missing some recommended dimensions
**Impact:** May not capture full behavioral diversity

**Research Basis:**
> "The recommended behavioral descriptor vector: `[Sharpe, MaxDD, Recovery_Time, Trade_Frequency, Market_Beta, Profit_Factor, Volatility, Win_Rate]`. This captures both performance characteristics and strategy mechanics."
> — Section 5: Quality-Diversity

**Current BehaviorVector (7D):**
- trade_frequency ✅
- avg_hold_period ✅
- long_short_ratio ✅
- return_autocorr ✅
- drawdown_depth ✅
- benchmark_corr (≈ Market_Beta) ✅
- signal_variance ✅

**Missing:** Recovery_Time, Profit_Factor, explicit Sharpe

**Note:** Current implementation is reasonable, this is a refinement.

---

### Quick Wins (Effort vs Impact)

| ID | Fix | Effort | Impact | Files |
|----|-----|--------|--------|-------|
| GP-007 | Change 2 numbers | 5 min | HIGH | promotion_pipeline.py |
| GP-009 | Add Calmar ratio | 30 min | MEDIUM | multi_objective.py |
| GP-011 | Reduce migration rate | 5 min | MEDIUM | config.py |
| GP-012 | Add plateau detection | 1 hour | MEDIUM | evolution_engine.py |
| GP-015 | Add regime change lag | 30 min | LOW | ml_regime_detector.py |

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
| 1 | **GP-007** | **Paper trading duration too short (14d → 90d)** | **5 min** | **HIGH** | **NEW** |
| 2 | **GP-008** | **Implement CPCV validation + PBO threshold** | **High** | **HIGH** | **NEW** |
| 3 | GA-003 | Validate GP discovery pipeline | Medium | High | VALIDATED (working) |
| 4 | BUG-004 | Live validate gap-fill strategy | Low | High | Pending (monitor first week) |
| 5 | GP-009 | Add Calmar ratio to fitness | Low | Medium | NEW |
| 6 | GP-011 | Reduce migration rate (15% → 3%) | 5 min | Medium | NEW |
| 7 | GP-012 | Add novelty pulsation for plateaus | 1 hour | Medium | NEW |
| 8 | ARCH-002 | Add real-time P&L to dashboard | Medium | Medium | |
| 9 | BUG-003 | Re-evaluate sector rotation with GA params | Low | Low | |
| 10 | GP-010 | Add HMM regime detection option | Medium | Medium | NEW |

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

---

*This is a living document. Update as items are resolved or new issues discovered.*
