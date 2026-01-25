# TradeBot Enhancement Roadmap 2026

**Based on:** State-of-the-Art Research Review (January 2026)
**Current State:** 22 days live, paper trading via Alpaca
**Platform:** Raspberry Pi 5 (8GB RAM)
**Updated:** January 25, 2026

---

## Executive Summary

Your system is **architecturally ahead** of much of the published research—you already have MAP-Elites, Novelty Search, NSGA-II multi-objective optimization, and behavioral vectors implemented. The research validates these choices.

Analysis of the existing 184 scalar GP strategies revealed critical issues:

| Finding | Value | Implication |
|---------|-------|-------------|
| Valid strategies | 97/184 (53%) | Half corrupted/incomplete |
| Unique entry patterns | 21 | Severe convergence |
| Degenerate patterns | 68% | Exploiting edge cases, not real alpha |
| Research finding | VGP >> Scalar GP | Must migrate before continuing |

**Decision:** Clean break. Archive scalar strategies, migrate to Vectorial GP.

---

## Phase 0: VGP Migration ✅ COMPLETE (Jan 23-25, 2026)

**Goal:** Replace scalar GP with Vectorial GP before any further evolution

**Completed Items:**
- ✅ Archived 184 scalar strategies to `discovered_strategies_v1_scalar`
- ✅ Implemented VGP primitives (~15 vector operators) in `gp_core.py`
- ✅ Created `VecType` for type-safe vector operations
- ✅ Updated `GenomeFactory` with `use_vgp=True` default
- ✅ Added VGP validation in `evolution_engine.py`
- ✅ Created migration script `vgp_migration.py` with rollback support
- ✅ Fixed degenerate strategy early-abort (3 consecutive signal limits → abort)
- ✅ Reduced data scope for Pi (100 symbols, 2 years)
- ✅ Fixed worker pool deadlocks (30s shutdown timeout, clashing pool detection)

**Key Files Changed:**
- `research/discovery/gp_core.py` - VGP primitives
- `research/discovery/strategy_genome.py` - `use_vgp` flag, `genome_version: 2`
- `research/discovery/evolution_engine.py` - VGP validation
- `research/discovery/strategy_compiler.py` - Degenerate abort logic
- `research/discovery/parallel_pool.py` - Deadlock-free shutdown
- `research/genetic/ga_parallel.py` - Unified lifecycle
- `research/genetic/persistent_optimizer.py` - Unified lifecycle

### 0.1 Archive Scalar Population

**Status:** ✅ DONE (Jan 23)
**Effort:** 1 hour

```bash
# Archive current state
sqlite3 db/research.db ".dump discovered_strategies" > backups/scalar_strategies_archive_20260122.sql
sqlite3 db/research.db "ALTER TABLE discovered_strategies RENAME TO discovered_strategies_v1_scalar"
```

**Preserve for historical reference:**
- `discovered_strategies` → `discovered_strategies_v1_scalar`
- Checkpoint files → `checkpoints/v1_scalar/`
- Document learnings (this analysis)

### 0.2 Implement VGP Primitives

**Status:** TO DO
**Effort:** 4-6 hours
**File:** `research/discovery/gp_core.py`

**New Type:**
```python
class VecType:
    """Vector type for VGP - represents a time series window"""
    pass
```

**Vector Primitives (indicators returning arrays):**
```python
# Lookback windows: 3, 5, 10 days
def _make_vec_close(lookback: int) -> Callable[[], np.ndarray]
def _make_vec_sma(period: int, lookback: int) -> Callable[[], np.ndarray]
def _make_vec_ema(period: int, lookback: int) -> Callable[[], np.ndarray]
def _make_vec_rsi(period: int, lookback: int) -> Callable[[], np.ndarray]
def _make_vec_volume(lookback: int) -> Callable[[], np.ndarray]
def _make_vec_returns(period: int, lookback: int) -> Callable[[], np.ndarray]
```

**Vector Operators:**
```python
# Aggregation: VecType → BoolType
def vec_all(vec: np.ndarray) -> bool      # All elements true
def vec_any(vec: np.ndarray) -> bool      # Any element true
def vec_majority(vec: np.ndarray) -> bool # >50% true

# Trend detection: VecType → BoolType
def vec_rising(vec: np.ndarray) -> bool   # Upward trend
def vec_falling(vec: np.ndarray) -> bool  # Downward trend

# Cross detection: (VecType, VecType) → BoolType
def vec_cross_above(v1, v2) -> bool       # v1 crossed above v2
def vec_cross_below(v1, v2) -> bool       # v1 crossed below v2

# Comparison: (VecType, FloatType) → VecType (boolean array)
def vec_lt(vec, threshold) -> np.ndarray  # Element-wise <
def vec_gt(vec, threshold) -> np.ndarray  # Element-wise >
```

**Constraints (prevent degenerate patterns):**
```python
# Remove or constrain problematic primitives
# - No raw const_N > price comparisons
# - Require vector context for entry/exit conditions
```

### 0.3 Create VGP Primitive Set

**Status:** ✅ DONE (Jan 23)
**Effort:** 2 hours
**File:** `research/discovery/gp_core.py`

```python
def create_vgp_primitive_set(config: PrimitiveConfig = None) -> gp.PrimitiveSetTyped:
    """
    Vectorial GP primitive set with temporal context.

    Key differences from scalar:
    - Indicators return vectors (last N values)
    - Aggregators collapse vectors to booleans
    - Cross/trend detectors provide temporal awareness
    """
    pset = gp.PrimitiveSetTyped("VGP_MAIN", [], BoolType)

    # Vector indicators (multiple lookbacks)
    for lookback in [3, 5, 10]:
        pset.addPrimitive(_make_vec_close(lookback), [], VecType, f'vec_close_{lookback}')
        pset.addPrimitive(_make_vec_rsi(14, lookback), [], VecType, f'vec_rsi_{lookback}')
        # ... etc

    # Aggregators
    pset.addPrimitive(vec_all, [VecType], BoolType, 'all')
    pset.addPrimitive(vec_any, [VecType], BoolType, 'any')
    # ... etc

    return pset
```

### 0.4 Update GenomeFactory

**Status:** ✅ DONE (Jan 23)
**Effort:** 1 hour
**File:** `research/discovery/strategy_genome.py`

```python
class GenomeFactory:
    def __init__(self, config=None, prim_config=None, use_vgp=True):
        # Default to VGP
        if use_vgp:
            self.bool_pset = create_vgp_primitive_set(prim_config)
        else:
            self.bool_pset = create_boolean_primitive_set(prim_config)  # Legacy
```

### 0.5 Reset Evolution State

**Status:** ✅ DONE (Jan 23)
**Effort:** 30 min

```sql
-- Create fresh tables for VGP
CREATE TABLE discovered_strategies_v2_vgp (
    -- Same schema as v1
);

-- Reset MAP-Elites grid
-- Reset novelty archive
-- Reset generation counter
```

### 0.6 Validate & Test

**Status:** ✅ DONE (Jan 23-24)
**Effort:** 2 hours

- [x] Generate random VGP trees, verify type safety
- [x] Compile and evaluate trees on test data
- [x] Run single generation, verify fitness evaluation works
- [x] Confirm backtest integration unchanged

### 0.7 Worker Lifecycle Hardening (Added Jan 25)

**Status:** ✅ DONE (Jan 25)

Fixed multiprocessing deadlocks discovered during overnight research:
- [x] Added 30s timeout to all pool shutdown methods
- [x] Added runtime detection for clashing pools (raises RuntimeError)
- [x] Added signal handling to all worker initializers
- [x] Documented single-pool constraint in class docstrings

### Phase 0 Success Criteria

- [x] Scalar strategies archived (not deleted)
- [x] VGP primitives implemented with type safety
- [x] GenomeFactory defaults to VGP
- [x] Single overnight evolution cycle completes successfully
- [x] New strategies show temporal patterns (not `const > price`)
- [x] Worker pools don't deadlock on shutdown

**Actual Effort:** ~15 hours across Jan 23-25

---

## Phase 1: Foundation Hardening (After VGP Migration → Feb 15)

**Goal:** Achieve operational stability and proper statistical validation

### 1.1 Achieve 24-Hour Uptime ✅ IN PROGRESS

**Status:** Current session started Jan 22
**Actions:** Monitor through overnight research, document any restarts

### 1.2 Implement Deflated Sharpe Ratio

**Priority:** CRITICAL
**Rationale:** With 184 strategies, expect Sharpe ~2.1 purely by luck (√(2×ln(184)))

**Implementation:**
```python
# research/validation/deflated_sharpe.py

def deflated_sharpe_ratio(sharpe: float, n_trials: int,
                          skewness: float, kurtosis: float,
                          track_record_length: int) -> float:
    """
    DSR = (SR - SR*) / σ(SR)

    Where SR* adjusts for multiple testing and non-normal returns
    """
    # Adjust for number of trials
    sr_star = expected_max_sharpe(n_trials)

    # Adjust for non-normality
    sr_std = sharpe_std(sharpe, skewness, kurtosis, track_record_length)

    return (sharpe - sr_star) / sr_std

def passes_dsr_threshold(sharpe: float, n_trials: int, **kwargs) -> bool:
    """t-stat must exceed 3.0 (not traditional 2.0)"""
    dsr = deflated_sharpe_ratio(sharpe, n_trials, **kwargs)
    return dsr > 3.0  # Harvey, Liu & Zhu threshold
```

**Files to modify:**
- `research/validation/cpcv.py` - Add DSR calculation
- `research/discovery/promotion_pipeline.py` - Add DSR gate
- `research/discovery/portfolio_fitness.py` - Track cumulative trials

**Acceptance Criteria:**
- [ ] DSR calculated for all 184 strategies
- [ ] Cumulative trials counter persisted across sessions
- [ ] Promotion requires DSR > 3.0
- [ ] Dashboard shows DSR alongside Sharpe

### 1.3 Add Benjamini-Hochberg FDR Correction

**Priority:** HIGH
**Rationale:** Less conservative than Bonferroni, better for strategy discovery

**Implementation:**
```python
# research/validation/multiple_testing.py

def benjamini_hochberg(p_values: List[float], fdr: float = 0.10) -> List[bool]:
    """
    Returns mask of which p-values pass FDR correction
    """
    sorted_indices = np.argsort(p_values)
    n = len(p_values)

    # BH critical values: (i/n) * FDR
    critical_values = [(i+1) / n * fdr for i in range(n)]

    # Find largest p-value <= its critical value
    significant = np.zeros(n, dtype=bool)
    for i, idx in enumerate(sorted_indices):
        if p_values[idx] <= critical_values[i]:
            significant[sorted_indices[:i+1]] = True

    return significant
```

**Acceptance Criteria:**
- [ ] All promoted strategies pass FDR at 0.10
- [ ] Batch validation runs weekly on full strategy pool

---

## Phase 2: Evolutionary Enhancements (Feb 15 → Mar 31)

**Goal:** Upgrade GP infrastructure based on research findings

### 2.1 Migrate to Vectorial GP (VGP)

**Priority:** HIGH
**Rationale:** Research shows VGP "always among best performers" while standard GP "always among worst"

**Current:** Scalar GP operating on single values
**Target:** Vector-based GP with lookback windows

**Conceptual Change:**
```python
# Current (scalar)
entry_signal = RSI(14) < 30 AND close < BB_lower(20)

# VGP (vectorial) - strategy sees context
entry_signal = VEC_RSI(14, lookback=5) < 30 AND  # Last 5 RSI values
               VEC_CROSS_BELOW(close, BB_lower, lookback=3)  # Cross event in last 3 bars
```

**Implementation Steps:**
1. Define vector primitives in `gp_core.py`:
   - `VEC_SMA(period, lookback)` → returns array
   - `VEC_MOMENTUM(period, lookback)` → returns array
   - `VEC_CROSS(series1, series2, lookback)` → boolean array
   - `VEC_SLOPE(series, lookback)` → trend direction

2. Add vector operators:
   - `ALL(vec)` → True if all elements True
   - `ANY(vec)` → True if any element True
   - `MAJORITY(vec)` → True if >50% True
   - `TREND(vec)` → 1 if increasing, -1 if decreasing

3. Update `strategy_genome.py` to handle vector trees

4. Modify fitness evaluation to process vector outputs

**Files to modify:**
- `research/discovery/gp_core.py` - Add vector primitives
- `research/discovery/strategy_genome.py` - Vector tree support
- `research/discovery/evolution_engine.py` - Vector evaluation

**Acceptance Criteria:**
- [ ] Vector primitives implemented with type safety
- [ ] Backward compatibility with existing strategies
- [ ] A/B test: VGP population vs standard GP population
- [ ] VGP outperforms standard GP over 30-day test

### 2.2 Implement Warm-Start Initialization

**Priority:** MEDIUM
**Rationale:** Reduces correlation among discovered factors from ~1.0 to ~0.6

**Implementation:**
```python
# research/discovery/warm_start.py

SEED_TEMPLATES = [
    # Mean reversion template
    "AND(RSI < {threshold}, close < BB_lower)",

    # Momentum template
    "AND(close > SMA_{period}, volume > SMA_volume)",

    # Breakout template
    "AND(close > HIGH_{lookback}, ATR > ATR_SMA)",
]

def warm_start_population(size: int, templates: List[str]) -> List[Individual]:
    """
    Initialize population with structural diversity:
    - 30% from templates with randomized parameters
    - 30% mutations of templates
    - 40% random (standard GP)
    """
```

**Acceptance Criteria:**
- [ ] Average correlation among top 20 strategies < 0.7
- [ ] Template library covers 5+ strategy archetypes

### 2.3 Enhance Behavioral Vectors (Already Strong)

**Current:** 6-dimensional behavior vectors
**Research:** 10-dimensional descriptors recommended

**Add dimensions:**
- Trade frequency (trades per month)
- Average holding period
- Regime preference (bull/bear/neutral performance)
- Drawdown recovery time

**Files to modify:**
- `research/discovery/novelty_search.py` - Extend behavior vector
- `research/discovery/diversity_metrics.py` - Add new metrics

---

## Phase 3: Regime Detection Hardening (Mar 1 → Apr 15)

**Goal:** Reduce whipsaws and improve regime classification

### 3.1 Implement Hysteresis-Based Regime Transitions

**Priority:** HIGH
**Rationale:** Prevents false regime flips during noisy periods

**Current:** Direct HMM state transitions
**Target:** Hysteresis bands with confirmation

**Implementation:**
```python
# research/ml_regime_detector.py

class HysteresisRegimeDetector:
    def __init__(self,
                 bull_entry=0.7,    # Enter bull if P(bull) > 0.7
                 bull_exit=0.3,     # Exit bull if P(bull) < 0.3
                 confirmation_days=2):
        self.thresholds = {
            'bull': {'entry': bull_entry, 'exit': bull_exit},
            'bear': {'entry': bull_exit, 'exit': bull_entry}
        }
        self.confirmation_days = confirmation_days
        self.pending_transition = None
        self.pending_count = 0

    def update(self, hmm_probabilities: dict) -> str:
        """
        Returns current regime, with hysteresis and confirmation
        """
        # Check if we should enter new regime
        # Require N consecutive days above/below threshold
```

**Acceptance Criteria:**
- [ ] Regime flip frequency reduced by 50%
- [ ] Backtested: hysteresis vs direct comparison
- [ ] Configurable thresholds per volatility level

### 3.2 Add Ensemble Voting with ML Classifier

**Priority:** MEDIUM
**Rationale:** Hybrid HMM + XGBoost achieves 78% accuracy

**Implementation:**
```python
# research/ensemble_regime_detector.py

class EnsembleRegimeDetector:
    def __init__(self):
        self.hmm = load_hmm_model()
        self.xgb = train_xgboost_classifier()  # On HMM features + technicals

    def predict(self, features: pd.DataFrame) -> str:
        hmm_pred = self.hmm.predict(features)
        xgb_pred = self.xgb.predict(features)

        # Voting: require agreement or use confidence weighting
        if hmm_pred == xgb_pred:
            return hmm_pred
        else:
            # Use higher confidence prediction
            return self._resolve_disagreement(hmm_pred, xgb_pred, features)
```

**Features to include (per State Street research):**
- Return-based: 1d, 5d, 20d, 60d returns
- Volatility: realized vol, VIX level, VIX term structure
- Technical: RSI, MACD, breadth indicators

**Acceptance Criteria:**
- [ ] Ensemble accuracy > 70% on held-out data
- [ ] Worst drawdown detection F1 > 0.75

### 3.3 Multi-Timeframe Confirmation

**Priority:** LOW
**Rationale:** Require daily + weekly agreement before regime action

**Implementation:**
- Daily HMM regime
- Weekly HMM regime (rolling 5-day windows)
- Only act on regime change if both timeframes agree

---

## Phase 4: Exit Logic Enhancement (Apr 1 → May 31)

**Goal:** Improve win rate from 24% to 40%+

### 4.1 Analyze Historical Exit Performance

**Priority:** HIGH (prerequisite for improvements)

**Actions:**
1. Query all 121 closed trades
2. Calculate: optimal exit time vs actual exit time
3. Identify patterns in premature/late exits
4. Segment by strategy, regime, holding period

**SQL Analysis:**
```sql
SELECT
    strategy,
    AVG(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as win_rate,
    AVG(pnl_percent) as avg_pnl_pct,
    AVG(julianday(exit_timestamp) - julianday(entry_timestamp)) as avg_hold_days,
    -- Compare exit price to peak price
    AVG((peak_price - exit_price) / peak_price) as missed_upside
FROM trades
WHERE status = 'CLOSED'
GROUP BY strategy;
```

### 4.2 Implement Trailing Stop Enhancement

**Priority:** MEDIUM
**Current:** Fixed TP/SL from strategy
**Target:** ATR-based trailing stops

**Implementation:**
```python
# execution/trailing_stop.py

class ATRTrailingStop:
    def __init__(self, atr_multiplier: float = 2.0):
        self.multiplier = atr_multiplier

    def calculate_stop(self, entry_price: float, current_price: float,
                       atr: float, side: str) -> float:
        """
        Trail stop at entry + (current - entry) - (ATR * multiplier)
        Never move stop backwards
        """
        if side == 'long':
            trail_stop = current_price - (atr * self.multiplier)
            return max(trail_stop, self.current_stop)
```

### 4.3 Research: Ensemble RL for Exits (Future)

**Priority:** LOW (research phase)
**Rationale:** FinRL contest shows ensemble RL achieves highest Sharpe

**Approach:**
1. Train offline on historical position data
2. Test in paper trading shadow mode
3. Compare to rule-based exits
4. Only deploy if statistically significant improvement

**Note:** RL requires periodic retraining—cannot be set-and-forget

---

## Phase 5: Portfolio Optimization (May 1 → Jun 30)

**Goal:** Improve strategy allocation and detect alpha decay

### 5.1 Implement Return-Adjusted HRP

**Priority:** MEDIUM
**Rationale:** RA-HRP achieves Sharpe 1.336 vs traditional HRP

**Current:** Equal allocation across strategies
**Target:** HRP with expected return integration

**Implementation:**
```python
# execution/portfolio_optimizer.py

def return_adjusted_hrp(returns: pd.DataFrame,
                        expected_returns: pd.Series) -> pd.Series:
    """
    1. Cluster strategies by correlation
    2. Apply inverse-variance weighting within clusters
    3. Adjust by expected return estimates
    4. Allocate across clusters by cluster variance
    """
    # Use scipy hierarchical clustering
    dist = correlDist(returns.corr())
    link = sch.linkage(dist, 'single')

    # Recursive bisection with return adjustment
    weights = getRecBipart(returns.cov(), link, expected_returns)
    return weights
```

**Acceptance Criteria:**
- [ ] Weekly rebalancing based on HRP weights
- [ ] Track improvement vs equal-weight baseline

### 5.2 Alpha Decay Detection System

**Priority:** MEDIUM
**Rationale:** Alpha decays 5.6% annually in US markets

**Current:** GP-016 has basic alpha decay detection
**Enhancement:** Add rolling IC degradation monitoring

**Implementation:**
```python
# research/discovery/alpha_decay.py

class AlphaDecayMonitor:
    def __init__(self, lookback_months: int = 6):
        self.lookback = lookback_months

    def check_decay(self, strategy_id: str) -> dict:
        """
        Returns decay metrics:
        - IC trend (should be stable or positive)
        - Sharpe degradation (flag if drops 50%+)
        - Correlation increase (crowding signal)
        """
        recent_ic = self._rolling_ic(strategy_id, months=3)
        historical_ic = self._rolling_ic(strategy_id, months=self.lookback)

        return {
            'ic_degradation': (historical_ic - recent_ic) / historical_ic,
            'sharpe_degradation': self._sharpe_trend(strategy_id),
            'correlation_increase': self._correlation_trend(strategy_id),
            'retire_recommended': self._should_retire(strategy_id)
        }
```

**Retirement Triggers:**
- Sustained Sharpe < 0.5 for 60 days
- IC drops below significance (p > 0.10)
- Correlation to existing strategies > 0.8

### 5.3 Factor Crowding Metrics (Future)

**Priority:** LOW
**Rationale:** 30% crowdedness threshold triggers reversals

**Metrics to track (MSCI approach):**
1. Valuation spread within factor
2. Pairwise correlation among factor holdings
3. Factor volatility vs historical
4. Recent factor reversal patterns

---

## Phase 6: Research Opportunities (Ongoing)

### 6.1 MAP-Elites Enhancement (You're Ahead!)

**Current State:** Already implemented in `research/discovery/map_elites.py`
**Research Finding:** No direct applications to trading found—you're pioneering this

**Enhancements:**
- Add more behavioral dimensions to grid
- Visualize MAP-Elites grid evolution over time
- Track which cells produce promoted strategies

### 6.2 Concept Drift Integration

**Priority:** MEDIUM
**Rationale:** Trigger re-evolution when market conditions shift

**Implementation:**
```python
# research/discovery/concept_drift.py

class ADWIN:
    """
    Adaptive Windowing for concept drift detection
    """
    def __init__(self, delta: float = 0.002):
        self.delta = delta

    def detect_drift(self, stream: List[float]) -> bool:
        """
        Returns True if statistical drift detected
        """
        # Maintain adaptive window
        # Compare subwindow means
        # Signal drift if difference exceeds threshold
```

**Integration:**
- Run drift detection on strategy returns nightly
- If drift detected, increase mutation rate for affected strategy type
- Log drift events to `performance.db:regime_log`

### 6.3 LLM-Augmented Signals (Future Research)

**Priority:** LOW (experimental)
**Rationale:** FinRL-DeepSeek shows promise for sentiment integration

**Approach:**
1. Research sentiment data sources (free tier options)
2. Prototype sentiment → signal mapping
3. Shadow test alongside existing strategies
4. Only integrate if additive (not correlated) alpha

---

## Implementation Timeline

```
         Jan        Feb        Mar        Apr        May        Jun
         ├──────────┼──────────┼──────────┼──────────┼──────────┤
Phase 0  ████░░░░░░
         VGP Migration (NOW)

Phase 1      ░░████████████░░░░░░
             DSR, FDR, Stability

Phase 2              ░░░░░░████████████░░░░░░
                     Warm-start, Diversity

Phase 3                        ░░░░████████████████░░░░
                               Hysteresis, Ensemble Regime

Phase 4                                  ░░░░████████████████
                                         Exit Analysis, Trailing Stops

Phase 5                                            ░░░░████████████
                                                   HRP, Alpha Decay

Ongoing  ────────────────────────────────────────────────────────────
         Ongoing: MAP-Elites, Drift Detection
```

---

## Success Metrics

### Phase 0 (End of January)

| Metric | Current | Target |
|--------|---------|--------|
| VGP primitives | 0 | Implemented |
| Scalar strategies archived | No | Yes |
| First VGP generation | 0 | Completed |
| VGP trees show temporal patterns | N/A | Verified |

### Q1 2026 (End of March)

| Metric | Current | Target |
|--------|---------|--------|
| Uptime | ~10 hours max | 24+ hours |
| Win Rate | 24% | 35% |
| DSR Implementation | None | Active |
| VGP Strategies Discovered | 0 | 50+ |
| Unique Entry Patterns | 21 (scalar) | 100+ (VGP) |

### Q2 2026 (End of June)

| Metric | Target |
|--------|--------|
| Win Rate | 40%+ |
| Paper Trading Days | 90+ (promotion eligible) |
| VGP Strategies Passing DSR | 10+ |
| HRP vs Equal Weight | HRP outperforms |

### Live Trading Prerequisites

Before transitioning to live trading:
- [ ] VGP migration complete
- [ ] 90 days paper trading complete
- [ ] 24-hour uptime achieved consistently
- [ ] 3+ strategies pass DSR threshold
- [ ] Win rate > 35%
- [ ] VGP migration complete
- [ ] Hysteresis regime detection active
- [ ] Alpha decay monitoring operational

---

## Quick Reference: Files to Modify by Phase

### Phase 0 (VGP Migration) ← CURRENT
- `research/discovery/gp_core.py` → Add VecType, vector primitives, vector operators
- `research/discovery/strategy_genome.py` → Add use_vgp flag to GenomeFactory
- `research/discovery/config.py` → Add VGP config (lookback periods)
- `db/research.db` → Archive v1 tables, create v2

### Phase 1 (Validation)
- `research/validation/cpcv.py` → Add DSR
- `research/validation/multiple_testing.py` → New file, BH correction
- `research/discovery/promotion_pipeline.py` → DSR gate

### Phase 2 (Warm-Start)
- `research/discovery/warm_start.py` → New file, template-based initialization

### Phase 3 (Regime)
- `research/ml_regime_detector.py` → Hysteresis
- `research/ensemble_regime_detector.py` → New file

### Phase 4 (Exits)
- `execution/trailing_stop.py` → New file
- `execution/position_manager.py` → Integrate trailing stops

### Phase 5 (Portfolio)
- `execution/portfolio_optimizer.py` → New file, HRP
- `research/discovery/alpha_decay.py` → Enhance existing

---

## Key Insight from Research

> "Most claimed research findings in financial economics are likely false due to lack of multiple testing control." — Bailey and López de Prado

Your 184 strategies are impressive, but without DSR correction, the best performers are likely statistical artifacts. **Phase 1 validation is not optional**—it's the foundation everything else builds on.

The good news: your architecture (MAP-Elites, Novelty Search, NSGA-II) is already designed to find *diverse* strategies, not just overfit ones. Adding proper statistical validation will separate genuine alpha from noise.

---

## Appendix A: Scalar GP Analysis (January 22, 2026)

Before migrating to VGP, we analyzed the existing 184 scalar GP strategies to extract learnings.

### Population Statistics

| Metric | Value |
|--------|-------|
| Total claimed | 184 |
| Valid JSON | 97 (53%) |
| Unique entry trees | 21 |
| Unique performance profiles | 115 |
| Degenerate entry patterns | 68% |
| Degenerate exit patterns | 62% |

### Indicator Usage in Entry Trees

| Indicator | Frequency | Notes |
|-----------|-----------|-------|
| `open()` | 77% | Mostly in degenerate `const_1 > open()` |
| `sma_` | 12% | Meaningful momentum signals |
| `low_` | 12% | Support level comparisons |
| `ema_` | 2% | Underutilized |
| `returns_` | 2% | Underutilized |

### Degenerate Patterns Identified

**Entry patterns (68% of population):**
```
gt(const_1, open())      # Only true for stocks < $1 (penny stocks)
gt(const_1, close())     # Same issue
lt(const_100, open())    # Only true for stocks > $100
not_(not_(true))         # Always true
```

**Exit patterns (62% of population):**
```
and_(not_(true), not_(true))           # Always false (never exit)
le(low(), const_100)                   # Always true for stocks < $100
or_(not_(true), ...)                   # First clause always false
ge(sma_200(), sma_200())               # Always true (x >= x)
```

### Meaningful Patterns Found

Despite the noise, a few genuine TA patterns emerged:

```python
# Momentum (SMA crossover)
gt(sma_20(), ema_12())

# Mean reversion (price below moving average)
gt(sma_10(), close())

# Trend following with support
or_(gt(close(), ema_50()), le(low(), high_10d()))

# Momentum confirmation
and_(ge(returns_5d(), returns_1d()), lt(low_5d(), abs(sma_5())))
```

### Learnings for VGP Design

1. **Constraint degenerate comparisons:** Prevent `const_N > price()` patterns in VGP
2. **Diversity mechanisms insufficient:** 21 unique trees from months of evolution indicates novelty search wasn't enough
3. **Exit trees need different treatment:** Maybe constrain to simpler patterns or use different primitive set
4. **Position/stop/target trees are noise:** Consider using fixed formulas (ATR-based) instead of evolved trees

### Decision

Given that:
- 68% of strategies exploit edge cases rather than market patterns
- Only ~15 meaningful patterns exist in 184 strategies
- Research shows scalar GP "always among worst performers"

**Archive the scalar population and start fresh with VGP.** The few meaningful patterns (SMA crossovers, mean reversion) are basic TA that VGP will rediscover naturally with temporal confirmation.

---

*Document created: January 22, 2026*
*Updated: January 22, 2026 (VGP migration plan added)*
*Next review: After Phase 0 completion*
