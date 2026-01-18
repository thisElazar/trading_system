# Variable Naming Standardization - Refactoring Plan

## Status: ✅ COMPLETE (January 2026)

| Phase | Status | Commits |
|-------|--------|---------|
| Phase 0: Setup | ✅ Done | - |
| Phase 1: Schema Definition | ✅ Done | `6988aff` |
| Phase 2: Signal Unification | ✅ Done | `395140e` |
| Phase 3: Database Migration | ✅ Done | `8660f33` |
| Phase 4: Strategy ID Docs | ✅ Done | In core/types.py |
| Phase 5: Python Variables | ⏭️ Deferred | Backward compat handles it |
| Phase 6: Verification | ✅ Done | All tests pass |

**What was migrated:**
- `signal_history` tables: strategy→strategy_id, signal_type→side, signal_strength→strength
- `trades` table: pnl_percent→pnl_pct
- Created `core/types.py` with canonical Signal, Side, SignalScore, EnsembleResult

**What was NOT migrated (intentional):**
- `signals` table: Uses signal_type='entry'/'exit' (different semantics)

---

## Executive Summary

This plan addresses variable naming inconsistencies across the trading system identified through deep audit. The goal is to establish canonical naming patterns and unify 7 Signal-related classes, standardize strategy identifiers, and align database schemas with Python code.

---

## Phase 0: Pre-Refactoring Setup

### 0.1 Create Safety Checkpoint
```bash
git add -A && git commit -m "Pre-variable-refactoring checkpoint"
```

### 0.2 Create `variable-standards` Skill
Location: `.claude/skills/variable-standards.md`

This skill documents canonical naming patterns for future reference and drift prevention.

---

## Phase 1: Canonical Schema Definition

### 1.1 Core Identifiers

| Concept | Canonical Name | Type | Values/Format |
|---------|---------------|------|---------------|
| Strategy identifier | `strategy_id` | `str` | Human name OR hex UUID (context-dependent) |
| Stock symbol | `symbol` | `str` | Uppercase ticker, e.g., "AAPL" |
| Signal unique ID | `signal_id` | `str` | UUID format |
| Trade unique ID | `trade_id` | `int` | Auto-increment |
| Position unique ID | `position_id` | `int` | Auto-increment |

### 1.2 Signal Fields

| Concept | Canonical Name | Type | Notes |
|---------|---------------|------|-------|
| Signal confidence | `strength` | `float` | 0.0-1.0, NOT confidence/conviction |
| Signal direction | `side` | `str` | "BUY", "SELL", "CLOSE" (uppercase) |
| Target price | `target_price` | `Optional[float]` | NOT take_profit/target |
| Stop loss | `stop_loss` | `Optional[float]` | Already consistent |
| Signal time | `signal_time` | `datetime` | When signal generated |

### 1.3 Position Fields

| Concept | Canonical Name | Type | Notes |
|---------|---------------|------|-------|
| Share count | `quantity` | `float` | NOT shares/qty |
| Entry price | `entry_price` | `float` | NOT open_price/cost_basis |
| Exit price | `exit_price` | `float` | NOT close_price |
| Current price | `current_price` | `float` | NOT price/last_price |
| Entry timestamp | `entry_time` | `datetime` | NOT opened_at/entry_date |
| Exit timestamp | `exit_time` | `datetime` | NOT closed_at/exit_date |

### 1.4 Performance Fields

| Concept | Canonical Name | Type | Notes |
|---------|---------------|------|-------|
| P&L dollars | `pnl` | `float` | Absolute dollar amount |
| P&L percentage | `pnl_pct` | `float` | NOT pnl_percent/profit_pct |
| Max drawdown % | `max_drawdown_pct` | `float` | NOT max_drawdown (ambiguous) |
| Sharpe ratio | `sharpe` | `float` | Already consistent |

### 1.5 Direction Values

| Old Values | Canonical Value | Context |
|------------|----------------|---------|
| `BUY`, `buy`, `long`, `LONG`, `entry` | `"BUY"` | Opening long position |
| `SELL`, `sell`, `short`, `SHORT` | `"SELL"` | Opening short OR closing long |
| `CLOSE`, `close`, `exit` | `"CLOSE"` | Explicitly closing position |

---

## Phase 2: Unified Signal Class (CRITICAL)

### 2.1 Problem
Currently 7 Signal-related classes with incompatible fields:
- `Signal` (strategies/base.py)
- `StoredSignal` (execution/signal_tracker.py)
- `EnsembleSignal` (execution/ensemble_coordinator.py)
- `AggregatedSignal` (execution/ensemble.py)
- `CandidateSignal` (execution/universe_scanner.py)
- `SignalRecord` (execution/signal_scoring.py)
- `SignalScore` (execution/signal_scoring.py)

### 2.2 Solution: Single Canonical Signal + Specialized Views

Create `core/types.py` with unified types:

```python
# /home/thiselazar/trading_system/core/types.py

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List

class Side(Enum):
    """Trade direction."""
    BUY = "BUY"
    SELL = "SELL"
    CLOSE = "CLOSE"

@dataclass
class Signal:
    """Canonical trading signal - single source of truth."""
    signal_id: str                              # UUID
    symbol: str                                 # Stock ticker
    strategy_id: str                            # Strategy that generated it
    side: Side                                  # Direction (enum)
    strength: float                             # 0.0-1.0 confidence
    price: float                                # Price at signal time
    signal_time: datetime                       # When generated

    # Optional fields
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    quantity: Optional[float] = None            # Suggested size

    # Context
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Lifecycle tracking
    status: str = "pending"                     # pending, executed, expired, cancelled
    expires_at: Optional[datetime] = None
    executed_at: Optional[datetime] = None

@dataclass
class SignalScore:
    """Scoring results for a signal (not a signal itself)."""
    signal_id: str
    win_probability: float
    expected_return: float
    risk_reward_ratio: float
    size_multiplier: float                      # 0-2x suggested adjustment
    factors: Dict[str, float] = field(default_factory=dict)

@dataclass
class EnsembleResult:
    """Aggregated output from ensemble (references signals, doesn't duplicate)."""
    symbol: str
    side: Side
    combined_strength: float                    # Weighted average of contributing signals
    contributing_signal_ids: List[str]          # References to Signal.signal_id
    final_quantity: float
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### 2.3 Files to Modify

| File | Current Class | Action |
|------|--------------|--------|
| `strategies/base.py:34-75` | `Signal` | Replace with import from `core/types.py` |
| `execution/signal_tracker.py:56-105` | `StoredSignal` | Remove, use `Signal` directly |
| `execution/ensemble_coordinator.py:52-62` | `EnsembleSignal` | Replace with `EnsembleResult` |
| `execution/ensemble.py:53-61` | `AggregatedSignal` | Remove, use `EnsembleResult` |
| `execution/universe_scanner.py:44-58` | `CandidateSignal` | Remove, use `Signal` with status="candidate" |
| `execution/signal_scoring.py:60-127` | `SignalRecord`, `SignalScore` | Keep `SignalScore`, remove `SignalRecord` |

### 2.4 Conversion Code to Remove/Update

| File | Lines | Current | Action |
|------|-------|---------|--------|
| `signal_tracker.py` | 77-91 | `from_strategy_signal()` | Remove (no longer needed) |
| `signal_tracker.py` | 93-105 | `to_strategy_signal()` | Remove (no longer needed) |
| `ensemble_coordinator.py` | 268-328 | Signal aggregation | Update to use `EnsembleResult` |
| `ensemble.py` | 86-193 | `ConflictResolver.resolve()` | Update to use `EnsembleResult` |

---

## Phase 3: Database Schema Alignment

### 3.1 trades.db Changes

```sql
-- Rename columns to match canonical names
ALTER TABLE trades RENAME COLUMN pnl_percent TO pnl_pct;
ALTER TABLE signals RENAME COLUMN confidence TO strength;
ALTER TABLE signals RENAME COLUMN take_profit TO target_price;

-- Standardize direction values
UPDATE trades SET side = UPPER(side);
UPDATE signals SET direction =
    CASE
        WHEN direction IN ('buy', 'long', 'entry') THEN 'BUY'
        WHEN direction IN ('sell', 'short') THEN 'SELL'
        WHEN direction IN ('close', 'exit') THEN 'CLOSE'
        ELSE UPPER(direction)
    END;
```

### 3.2 signal_scores.db Changes

```sql
-- Rename signal_strength to strength
-- This requires recreating the table (SQLite limitation)
CREATE TABLE signal_history_new (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id TEXT UNIQUE NOT NULL,
    symbol TEXT NOT NULL,
    strategy_id TEXT NOT NULL,          -- Was: strategy
    side TEXT NOT NULL,                 -- Was: signal_type
    strength REAL DEFAULT 0,            -- Was: signal_strength
    -- ... rest of columns
);
INSERT INTO signal_history_new SELECT ... FROM signal_history;
DROP TABLE signal_history;
ALTER TABLE signal_history_new RENAME TO signal_history;
```

### 3.3 shadow_trades Table (in trades.db)

```sql
-- Rename 'shares' to 'quantity' for consistency
ALTER TABLE shadow_trades RENAME COLUMN shares TO quantity;
```

---

## Phase 4: Strategy ID Clarification

### 4.1 Current State (Intentional Design)

Two parallel systems exist:
1. **GP-discovered strategies**: Use hex UUID (`00304020`) from `strategy_genome.py:60`
2. **Human-coded strategies**: Use descriptive names (`mean_reversion`) from class init

### 4.2 Documentation Required

Add to `core/types.py`:

```python
# Strategy ID Convention:
#
# GP-discovered strategies: 8-char hex UUID (e.g., "00304020")
#   - Generated at: research/discovery/strategy_genome.py:60
#   - Tracked in: promotion_pipeline.db, research.db
#   - When promoted to LIVE, hex ID becomes the trade strategy_id
#
# Human-coded strategies: Descriptive snake_case name (e.g., "mean_reversion")
#   - Defined in: strategies/*.py class __init__
#   - Registered directly with scheduler
#   - Never appear in promotion_pipeline.db
#
# Both types use strategy_id field - the value format indicates the source.
```

### 4.3 Rename Variables

| File | Line | Old | New |
|------|------|-----|-----|
| `signal_tracker.py` | 58 | `strategy_name` | `strategy_id` |
| `portfolio_fitness.py` | schema | `strategy_name` | `strategy_id` |
| `ensemble_coordinator.py` | 39 | `strat_id` | `strategy_id` |

---

## Phase 5: Python Code Standardization

### 5.1 Files Requiring Variable Renames

| File | Changes Required |
|------|-----------------|
| `execution/signal_tracker.py` | `strategy_name` → `strategy_id`, `confidence` → `strength`, `take_profit` → `target_price` |
| `execution/ensemble_coordinator.py` | `combined_strength` → keep (it's aggregated), `target` → `target_price`, `strat_id` → `strategy_id` |
| `execution/ensemble.py` | `direction` values to uppercase |
| `execution/universe_scanner.py` | `confidence` → `strength`, `take_profit` → `target_price` |
| `execution/shadow_trading.py` | `shares` → `quantity`, `entry_time` already correct |
| `execution/order_executor.py` | `qty` → `quantity` in output dicts |
| `research/backtester.py` | `entry_date` → `entry_time`, `exit_date` → `exit_time` |

### 5.2 Estimated Line Changes

| File | Estimated Changes |
|------|-------------------|
| `signal_tracker.py` | ~30 lines |
| `ensemble_coordinator.py` | ~20 lines |
| `ensemble.py` | ~15 lines |
| `universe_scanner.py` | ~10 lines |
| `shadow_trading.py` | ~25 lines |
| `order_executor.py` | ~10 lines |
| `backtester.py` | ~15 lines |
| **Total** | ~125 lines |

---

## Phase 6: Testing & Verification

### 6.1 Create Migration Test Script

```python
# scripts/verify_variable_migration.py

def test_signal_creation():
    """Verify Signal class works with canonical fields."""
    from core.types import Signal, Side
    sig = Signal(
        signal_id="test-001",
        symbol="AAPL",
        strategy_id="mean_reversion",
        side=Side.BUY,
        strength=0.85,
        price=150.0,
        signal_time=datetime.now()
    )
    assert sig.strength == 0.85
    assert sig.side == Side.BUY

def test_database_read():
    """Verify DB reads map to canonical names."""
    # Test that pnl_pct column is readable
    # Test that strength column is readable
    pass

def test_signal_flow():
    """Verify signal flows through entire pipeline."""
    # Strategy → Signal → Ensemble → Execution → Trade
    pass
```

### 6.2 Verification Commands

```bash
# Check no old field names remain
grep -r "\.confidence" --include="*.py" execution/ strategies/
grep -r "take_profit" --include="*.py" execution/ strategies/
grep -r "strategy_name" --include="*.py" execution/ research/
grep -r "pnl_percent" --include="*.py" .

# Check database schema matches
sqlite3 db/trades.db ".schema trades" | grep pnl_pct
sqlite3 db/signal_scores.db ".schema signal_history" | grep strength
```

---

## Execution Order

1. **Create skill** (`.claude/skills/variable-standards.md`) - Documents standards
2. **Create `core/types.py`** - Canonical type definitions
3. **Update Signal consumers** - One file at a time, test each
4. **Database migrations** - Run SQL scripts
5. **Update remaining Python** - Variable renames
6. **Run verification** - Ensure no regressions
7. **Commit** - With comprehensive message

---

## Rollback Plan

If issues occur:
1. `git checkout HEAD~1` to restore pre-refactoring state
2. Restore databases from `backups/pre_migration_*/`
3. Review failed tests to identify specific issues

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Signal conversion breaks | Medium | High | Test each file individually |
| Database migration fails | Low | High | Backup before, test on copy first |
| Import cycles with core/types.py | Medium | Medium | Careful import structure |
| Missed variable rename | Medium | Low | Grep verification |

---

## Success Criteria

- [ ] Single `Signal` class used throughout
- [ ] All `strength` fields (not confidence/conviction)
- [ ] All `strategy_id` fields (not strategy_name/strategy)
- [ ] All `pnl_pct` fields (not pnl_percent)
- [ ] All `quantity` fields (not shares/qty)
- [ ] Grep verification passes with no old field names
- [ ] All existing tests pass
- [ ] Signal flow test passes end-to-end
