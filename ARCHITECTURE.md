# Trading System Architecture

**Generated**: 2026-01-01
**Version**: Production Ready
**Status**: Paper Trading Ready (Error Tracking Enabled)

---

## Table of Contents

1. [System Hierarchy & Authority Chain](#1-system-hierarchy--authority-chain)
2. [Key Configuration Variables](#2-key-configuration-variables)
3. [Data Sources & Flow](#3-data-sources--flow)
4. [Strategy Implementations](#4-strategy-implementations)
5. [Database Schema](#5-database-schema)
6. [Risk Management Pipeline](#6-risk-management-pipeline)
7. [Observability & Error Tracking](#7-observability--error-tracking)
8. [Critical Issues & Recommendations](#8-critical-issues--recommendations)

---

## 1. System Hierarchy & Authority Chain

### Master Controller

```
DailyOrchestrator (daily_orchestrator.py)
├── Authority: SUPREME - controls all system operations
├── Execution: Phase-based task scheduling
├── Market States: PRE_MARKET → MARKET_OPEN → TRADING → MARKET_CLOSE → AFTER_HOURS → OVERNIGHT
└── Kill Switch: /killswitch/HALT file stops all operations
```

### Authority Chain (Top to Bottom)

| Level | Component | Authority | Can Override |
|-------|-----------|-----------|--------------|
| 1 | Kill Switch | Absolute halt | Everything |
| 2 | Circuit Breakers | Risk-based halt | Trading decisions |
| 3 | DailyOrchestrator | Task scheduling | Strategy execution |
| 4 | VIX Regime Detector | Position scaling | Individual positions |
| 5 | EnsembleCoordinator | Signal aggregation | Signal execution |
| 6 | Individual Strategies | Signal generation | Nothing |

### Orchestrator Task Registry

The orchestrator uses a task registry pattern (`TASK_REGISTRY` dict) mapping phase → task list:

```python
# Phase: PRE_MARKET (6:30 AM - 9:30 AM ET)
- validate_api_connectivity
- sync_positions_from_broker
- load_regime_detector
- run_premarket_analysis

# Phase: MARKET_OPEN (9:30 AM - 9:35 AM ET)
- check_opening_conditions
- execute_opening_orders

# Phase: TRADING (9:35 AM - 3:45 PM ET)
- monitor_positions
- process_intraday_signals
- check_circuit_breakers

# Phase: MARKET_CLOSE (3:45 PM - 4:00 PM ET)
- close_intraday_positions
- calculate_daily_pnl

# Phase: AFTER_HOURS (4:00 PM - 6:00 PM ET)
- run_daily_research
- update_watchlists
- run_strategy_optimization

# Phase: OVERNIGHT
- run_nightly_research_pipeline
- checkpoint_system_state
```

---

## 2. Key Configuration Variables

### Capital & Risk (`config.py`)

| Variable | Value | Source | Notes |
|----------|-------|--------|-------|
| `TOTAL_CAPITAL` | $97,000 | `env:TRADING_CAPITAL` or default | Paper trading allocation |
| `RISK_PER_TRADE` | 2% | Hardcoded | Max risk per position |
| `MAX_POSITION_SIZE` | $15,000 | Hardcoded | Absolute position limit |
| `MAX_POSITIONS` | 10 | Hardcoded | Concurrent position limit |
| `MAX_POSITION_PCT` | 5% | Hardcoded | Per-position % of portfolio |
| `CASH_BUFFER_PCT` | 5% | Hardcoded | Reserved cash |

### Circuit Breaker Thresholds

| Breaker | Threshold | Action |
|---------|-----------|--------|
| Daily Loss | 2% | Halt all trading |
| Drawdown | 15% | Reduce positions 50% |
| Rapid Loss | 1% in 15 min | Pause 30 minutes |
| Consecutive Losses | 5 losses | Pause strategy 4 hours |
| Strategy Loss | 5% of allocation | Disable strategy 24 hours |

### VIX Regime Scaling

| Regime | VIX Range | Position Scalar |
|--------|-----------|-----------------|
| Low | < 15 | 1.2x (scale up) |
| Normal | 15-25 | 1.0x |
| High | 25-35 | 0.7x (scale down) |
| Extreme | > 35 | 0.4x (defensive) |

### Strategy Allocations (Total: 100%)

| Strategy | Allocation | Status | Max Positions |
|----------|------------|--------|---------------|
| mean_reversion | 30% | ENABLED | 25 |
| relative_volume_breakout | 25% | ENABLED | 5 |
| gap_fill | 10% | ENABLED | 2 |
| vol_managed_momentum | 10% | ENABLED | 10 |
| vix_regime_rotation | 10% | ENABLED | 5 |
| pairs_trading | 5% | ENABLED | 14 |
| quality_smallcap_value | 5% | ENABLED | 30 |
| factor_momentum | 5% | ENABLED | 10 |
| sector_rotation | 10% | **DISABLED** | 10 |

**Verification**: Enabled allocations sum to **100%** (30+25+10+10+10+5+5+5)

---

## 3. Data Sources & Flow

### External Data Sources

```
┌─────────────────────────────────────────────────────────────────┐
│                      EXTERNAL DATA SOURCES                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Alpaca API  │  │  Yahoo Fin   │  │  Wikipedia   │          │
│  │  - Daily     │  │  - Extended  │  │  - S&P 500   │          │
│  │  - Intraday  │  │    history   │  │    list      │          │
│  │  - Execution │  │  - VIX proxy │  │              │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                    │
└─────────┼─────────────────┼─────────────────┼────────────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DATA LAYER (Parquet + SQLite)               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  data/historical/                                                │
│  ├── daily/           # Alpaca daily OHLCV (recent)             │
│  ├── daily_yahoo/     # Yahoo daily (10yr history)              │
│  ├── intraday_1min/   # Alpaca 1-min bars (30 days)             │
│  └── vix/             # VIX proxy data (^VIX)                   │
│                                                                  │
│  data/fundamentals/   # Market cap, sector, industry            │
│  data/reference/      # S&P 500 list, sector mappings           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow: Signal to Execution

```
┌─────────────────┐
│   Market Data   │
│  (Alpaca/Yahoo) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│  Data Loader    │────▶│  VIX Regime     │
│  (data_loader)  │     │  Detector       │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  Strategy       │────▶│  Regime-Based   │
│  Implementations│     │  Position Scale │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│         EnsembleCoordinator             │
│  - Aggregates signals from strategies   │
│  - Applies position sizing              │
│  - Resolves conflicts                   │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│         Circuit Breakers                │
│  - Daily loss check                     │
│  - Drawdown check                       │
│  - Rapid loss detection                 │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│         Order Execution                 │
│  - Alpaca API submission                │
│  - Fill tracking                        │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│         Database Persistence            │
│  - trades.db: signals, trades, orders   │
│  - performance.db: daily P&L, stats     │
└─────────────────────────────────────────┘
```

### Database Flow

| Database | Purpose | Writers | Readers |
|----------|---------|---------|---------|
| `trades.db` | Execution records | Order executor, Position manager | Performance tracker, Risk manager |
| `performance.db` | Strategy metrics | Daily orchestrator | Strategy optimizer, Risk manager |
| `research.db` | Backtest results | Research pipeline | Strategy selector |
| `pairs.db` | Cointegrated pairs | Pairs discovery | Pairs trading strategy |

---

## 4. Strategy Implementations

### Strategy Location Map

| Strategy | Implementation File | Entry Point |
|----------|---------------------|-------------|
| mean_reversion | `strategies/mean_reversion.py` | `MeanReversionStrategy.generate_signals()` |
| relative_volume_breakout | `strategies/relative_volume_breakout.py` | `RelativeVolumeBreakout.generate_signals()` |
| gap_fill | `strategies/intraday/gap_fill/strategy.py` | `GapFillStrategy.generate_signals()` |
| vol_managed_momentum | `strategies/vol_managed_momentum_v2.py` | `VolManagedMomentumV2.generate_signals()` |
| vix_regime_rotation | `strategies/vix_regime_rotation.py` | `VIXRegimeRotation.generate_signals()` |
| pairs_trading | `strategies/pairs_trading.py` | `PairsTradingStrategy.generate_signals()` |
| quality_smallcap_value | `strategies/quality_smallcap_value.py` | `QualitySmallcapValue.generate_signals()` |
| factor_momentum | `strategies/factor_momentum.py` | `FactorMomentum.generate_signals()` |
| sector_rotation | `strategies/sector_rotation.py` | `SectorRotation.generate_signals()` (DISABLED) |

### Strategy Completeness Audit

| Strategy | Signal Gen | Position Size | Stop Loss | Target | Risk Mgmt |
|----------|------------|---------------|-----------|--------|-----------|
| mean_reversion | ✅ | ✅ | ✅ | ✅ | ✅ |
| relative_volume_breakout | ✅ | ✅ | ✅ | ✅ | ✅ |
| gap_fill | ✅ | ✅ | ✅ | ✅ | ✅ |
| vol_managed_momentum | ✅ | ✅ | ✅ | ✅ | ✅ |
| vix_regime_rotation | ✅ | ✅ | ✅ | ✅ | ✅ |
| pairs_trading | ✅ | ✅ | ✅ | ✅ | ✅ |
| quality_smallcap_value | ✅ | ✅ | ✅ | ✅ | ✅ |
| factor_momentum | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## 5. Database Schema

### trades.db

```sql
-- Signal Generation
signals (
    id, timestamp, symbol, strategy, signal_type,  -- BUY/SELL/CLOSE
    strength, price, confidence, direction,
    stop_loss, take_profit, quantity,
    status,  -- pending/executed/expired
    executed, execution_id, expires_at, executed_at
)

-- Trade Lifecycle
trades (
    id, signal_id, timestamp, symbol, strategy, side,
    quantity, entry_price, stop_loss, target_price,
    exit_price, exit_timestamp, peak_price, trough_price,
    pnl, pnl_percent, status,  -- OPEN/CLOSED/CANCELLED
    exit_reason, commission, slippage
)

-- Current State
positions (
    symbol [PK], strategy, side, quantity,
    entry_price, entry_timestamp, stop_loss, target_price,
    current_price, unrealized_pnl, unrealized_pnl_pct,
    status  -- open/closed
)

-- Order Tracking
orders (
    id, trade_id, alpaca_order_id, symbol, side,
    order_type, quantity, limit_price, stop_price,
    status, filled_qty, filled_avg_price
)

-- Fill Records
executions (
    id, signal_id, order_id, symbol, direction,
    quantity, fill_price, commission, slippage, executed_at
)
```

### performance.db

```sql
-- Daily Strategy Metrics
strategy_daily (
    date, strategy, trades_opened, trades_closed,
    wins, losses, gross_pnl, net_pnl, commission,
    sharpe_20d, sharpe_60d, win_rate_30d,
    avg_win, avg_loss, profit_factor, max_drawdown, is_enabled
)

-- Portfolio-Level Metrics
portfolio_daily (
    date, equity, cash, positions_value, num_positions,
    exposure_pct, daily_pnl, daily_pnl_pct,
    cumulative_pnl, cumulative_pnl_pct,
    drawdown, drawdown_pct, high_water_mark,
    sharpe_ratio, sortino_ratio
)

-- Aggregate Strategy Stats
strategy_stats (
    strategy [PK], total_trades, winning_trades, losing_trades,
    total_pnl, avg_pnl, win_rate, profit_factor,
    sharpe_ratio, sortino_ratio, max_drawdown, is_enabled
)

-- Risk Events
circuit_breaker_state (
    breaker_type, triggered_at, expires_at,
    reason, action, target, is_active
)

-- VIX Regime Changes
regime_log (
    timestamp, vix_level, vix_regime, previous_regime,
    action_taken
)

-- Centralized Error Tracking (Dashboard Display)
error_log (
    id, timestamp, level,  -- ERROR/WARNING/CRITICAL
    logger_name, message, source_file, line_number,
    exception_type, exception_traceback,
    component,  -- execution, strategies, research, data, etc.
    is_resolved, resolved_at, resolved_by
)
```

---

## 6. Risk Management Pipeline

### Kill Switch Flow

```
┌─────────────────────────────────────────┐
│         Kill Switch Check               │
│  Location: /killswitch/HALT             │
│  Checked: Every orchestrator cycle      │
├─────────────────────────────────────────┤
│  If file exists:                        │
│  1. Log halt event                      │
│  2. Cancel all pending orders           │
│  3. Close all positions (optional)      │
│  4. Exit orchestrator                   │
└─────────────────────────────────────────┘
```

### Circuit Breaker Chain

```
Signal Generated
       │
       ▼
┌──────────────────┐
│ Daily Loss Check │ ──▶ If loss > 2%: HALT ALL
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Drawdown Check   │ ──▶ If DD > 15%: Scale positions 50%
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Rapid Loss Check │ ──▶ If 1% loss in 15min: Pause 30min
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Strategy Loss    │ ──▶ If strategy -5%: Disable 24hr
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Consecutive Loss │ ──▶ If 5 losses: Pause strategy 4hr
└────────┬─────────┘
         │
         ▼
   Execute Order
```

---

## 7. Observability & Error Tracking

### Centralized Error Logging

All system errors and warnings are captured in a centralized database for dashboard display and debugging.

```
┌─────────────────────────────────────────────────────────────────┐
│                    ERROR TRACKING FLOW                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │ Application  │────▶│ Logger       │────▶│ File Logs    │    │
│  │ Code         │     │ (WARNING+)   │     │ (*.log)      │    │
│  └──────────────┘     └──────┬───────┘     └──────────────┘    │
│                              │                                   │
│                              ▼                                   │
│                       ┌──────────────┐                          │
│                       │ Database     │                          │
│                       │ Handler      │                          │
│                       │ (async queue)│                          │
│                       └──────┬───────┘                          │
│                              │                                   │
│                              ▼                                   │
│                       ┌──────────────┐     ┌──────────────┐    │
│                       │ error_log    │────▶│ Dashboard    │    │
│                       │ (SQLite)     │     │ "System      │    │
│                       │              │     │  Health"     │    │
│                       └──────────────┘     └──────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Error Logging Implementation

| Component | File | Error Handling |
|-----------|------|----------------|
| **Signal Tracker** | `execution/signal_tracker.py` | All 10 DB methods wrapped with try-except, logs signal_id/symbol/price context |
| **Circuit Breaker** | `execution/circuit_breaker.py` | 8 silent `except: pass` replaced with proper error logging |
| **DB Manager** | `data/storage/db_manager.py` | Connection, schema init, and query methods protected |
| **Data Fetchers** | `data/fetchers/*.py` | Parquet save operations protected |
| **Order Executor** | `execution/order_executor.py` | Full context logging (symbol, qty, price, order_id) |

### Dashboard Integration

The dashboard (`observability/dashboard/app.py`) displays:

| Panel | Location | Auto-Refresh |
|-------|----------|--------------|
| **System Errors** | "System Health" section | 30 seconds |
| **Error Summary** | Count of critical/error/warning | 30 seconds |
| **Recent Alerts** | Alert history | 30 seconds |

### Error Log Fields

| Field | Description |
|-------|-------------|
| `timestamp` | When error occurred |
| `level` | ERROR, WARNING, or CRITICAL |
| `logger_name` | e.g., `trading_system.execution.signal_tracker` |
| `message` | Error description |
| `source_file` | Python file where error occurred |
| `line_number` | Line number in source file |
| `exception_type` | e.g., `sqlite3.OperationalError` |
| `exception_traceback` | Full stack trace |
| `component` | System component (execution, strategies, research) |
| `is_resolved` | Whether error has been addressed |

### Key Files

| File | Purpose |
|------|---------|
| `observability/logger.py` | Centralized logging with `DatabaseErrorHandler` |
| `observability/dashboard/app.py` | Dash dashboard with error display |
| `db/performance.db:error_log` | Error storage table |

---

## 8. Critical Issues & Recommendations

### HIGH PRIORITY - Pipeline Gaps

| # | Issue | Location | Impact | Status |
|---|-------|----------|--------|--------|
| 1 | **Signals not logged to DB** | `trades.db:signals` | No audit trail for signal generation | **FIXED** - `_process_intraday_signal()` now logs via `ExecutionTracker` |
| 2 | **Position sync not implemented** | `daily_orchestrator.py` | Local state can desync from broker | **FIXED** - Added `sync_positions_from_broker` task to PRE_MARKET phase |
| 3 | **Regime log not written** | `performance.db:regime_log` | No VIX regime change audit trail | **FIXED** - Added `_log_regime_change()` method |
| 4 | **Performance metrics not real-time** | `performance.db` | Only updated at EOD | Consider intraday snapshots |
| 5 | **Alpaca connectivity validation** | Pre-market phase | No explicit health check | Broker check done in `system_check` task |

### MEDIUM PRIORITY - Configuration Inconsistencies

| # | Issue | Files Affected | Status |
|---|-------|----------------|--------|
| 1 | Hardcoded `97000` instead of `TOTAL_CAPITAL` | `execution/ensemble.py`<br>`execution/ensemble_coordinator.py`<br>`strategies/intraday/*/config.py` | **FIXED** - Now uses `TOTAL_CAPITAL` from config |
| 2 | `portfolio_value` default in strategy configs | `strategies/intraday/*/config.py` | **FIXED** - Uses `TOTAL_CAPITAL` via `__post_init__` |
| 3 | Duplicate strategy name fields | `trades.db:positions.strategy` vs `strategy_name` | Low priority - cosmetic issue |

### LOW PRIORITY - Technical Debt

| # | Issue | Location | Notes |
|---|-------|----------|-------|
| 1 | VIX data is Yahoo proxy | `data/historical/vix/` | Acceptable for regime detection; real-time VIX feeds cost extra |
| 2 | Fundamentals not integrated | `data/fundamentals/` | Sector/industry data exists but not used in live trading |
| 3 | Multiple DataLoader implementations | `data_loader.py`, `data/data_loader.py` | Consolidate to single source |

### Data Inventory

| Category | Count | Location | Last Updated |
|----------|-------|----------|--------------|
| Historical Daily (symbols) | 2,556 | `data/historical/daily_yahoo/` | Auto-updated |
| Historical Intraday (symbols) | ~500 | `data/historical/intraday_1min/` | Rolling 30 days |
| Active Pairs | 7 | `db/pairs.db` | 2025-12-31 |
| GP Evolution Checkpoints | 4 | `research/discovery/checkpoints/` | Verified loading |
| Discovered Strategies | varies | `research/discovery/candidates/` | Ongoing |

---

## Appendix A: File Structure

```
trading_system/
├── config.py                    # Central configuration (SINGLE SOURCE)
├── daily_orchestrator.py        # Master controller
├── data_loader.py              # Data access layer
│
├── strategies/                  # Strategy implementations
│   ├── mean_reversion.py
│   ├── relative_volume_breakout.py
│   ├── vol_managed_momentum_v2.py
│   ├── vix_regime_rotation.py
│   ├── pairs_trading.py
│   ├── quality_smallcap_value.py
│   ├── factor_momentum.py
│   ├── sector_rotation.py       # DISABLED
│   └── intraday/
│       ├── gap_fill/
│       ├── orb/
│       └── vwap_reversion/
│
├── execution/                   # Order & position management
│   ├── ensemble_coordinator.py
│   ├── ensemble.py
│   ├── position_sizer.py
│   ├── adaptive_position_sizer.py
│   └── order_executor.py
│
├── research/                    # Research & optimization
│   ├── ml_regime_detector.py
│   ├── walk_forward_backtest.py
│   ├── adaptive_ga_params.py
│   └── discovery/              # GP strategy evolution
│       ├── gp_core.py
│       ├── checkpoints/
│       └── candidates/
│
├── db/                         # SQLite databases
│   ├── trades.db
│   ├── performance.db
│   ├── research.db
│   └── pairs.db
│
├── data/                       # Market data
│   ├── historical/
│   │   ├── daily_yahoo/
│   │   ├── intraday_1min/
│   │   └── vix/
│   ├── fundamentals/
│   └── reference/
│
├── observability/              # Monitoring & debugging
│   ├── logger.py              # Centralized logging + DatabaseErrorHandler
│   ├── dashboard/
│   │   └── app.py             # Dash dashboard (System Health panel)
│   └── alert_manager.py       # Alert routing
│
├── logs/                       # Application logs
├── killswitch/                 # Emergency halt mechanism
└── scripts/                    # Utilities & CLI tools
```

---

## Appendix B: Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TRADING_CAPITAL` | 97000 | Total portfolio value |
| `ALPACA_API_KEY` | (none) | Alpaca API key |
| `ALPACA_SECRET_KEY` | (none) | Alpaca secret key |
| `TRADING_SYSTEM_ROOT` | (auto-detect) | Override data root path |
| `LOG_LEVEL` | INFO | Logging verbosity |

---

## Appendix C: Deployment Checklist

### Pre-Deployment Verification

- [ ] All logs cleaned (fresh state)
- [ ] GP checkpoints load successfully (4/4 verified)
- [ ] ML Regime Detector model loads (fixed pickle issue)
- [ ] Adaptive GA params format correct (list of ParameterSpec)
- [ ] Strategy allocations sum to 100%
- [ ] Alpaca API credentials in `.env`
- [ ] Kill switch directory exists
- [ ] Circuit breaker thresholds reviewed

### Day-1 Monitoring

- [ ] Verify pre-market sync runs
- [ ] Confirm VIX regime detection works
- [ ] Check first signals logged to DB
- [ ] Monitor circuit breaker states
- [ ] Verify EOD P&L calculation
- [ ] Check dashboard "System Health" panel for errors
- [ ] Verify error_log table populating in performance.db
- [ ] Confirm no silent failures in logs

### Known Gaps (Accept or Fix Before Deployment)

- [x] Signal audit trail - **FIXED 2026-01-01**
- [x] Position broker sync - **FIXED 2026-01-01**
- [x] Hardcoded 97000 values - **FIXED 2026-01-01** (now uses `TOTAL_CAPITAL` from config)
- [x] Regime log writing - **FIXED 2026-01-01**
- [x] Centralized error tracking - **FIXED 2026-01-01** (errors/warnings logged to DB and dashboard)
- [x] Silent exception handling - **FIXED 2026-01-01** (8 circuit breaker `except: pass` blocks fixed)
- [x] Unprotected database operations - **FIXED 2026-01-01** (signal tracker, db manager)

---

*Document generated by system audit. Last updated: 2026-01-01*
