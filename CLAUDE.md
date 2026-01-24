# Trading System

Automated trading system on Raspberry Pi 5. Paper trading via Alpaca.

## Timezone Convention

**ALWAYS include timezone when discussing times.** The user is in Pacific Time (PT).

| Context | Timezone | Examples |
|---------|----------|----------|
| Market hours | Eastern (ET) | "Market opens 9:30 AM ET" |
| Database storage | UTC | research.db timestamps are UTC |
| User's local | Pacific (PT) | "It's 10:52 PM PT" |
| Cron/systemd timers | System local (PT) | Pi is set to America/Los_Angeles |

**Quick Reference:**
- Market open: 9:30 AM ET = 6:30 AM PT
- Market close: 4:00 PM ET = 1:00 PM PT
- Overnight research: Starts after 8:00 PM ET = 5:00 PM PT

**Code Policy (from `utils/timezone.py`):**
- Internal data: UTC-naive timestamps (no timezone info, interpreted as UTC)
- Display/market logic: Eastern time via `now_eastern()`, `is_market_hours()`
- Always use `from utils.timezone import ...` for new code

## Critical Safety Rules

1. **Never modify config.py without explicit approval** - affects live trading
2. **Never restart trading-orchestrator during market hours** (9:30-16:00 ET) without approval
3. **Kill switch**: `touch ~/trading_system/killswitch/HALT` stops all new orders
4. **Database backups**: Run before any DB schema changes

## System State

- **Orchestrator**: `systemctl status trading-orchestrator`
- **Positions**: `sqlite3 db/trades.db "SELECT symbol, direction, quantity, unrealized_pnl FROM positions WHERE status='open'"`
- **Phase**: Check orchestrator.log for current phase (PRE_MARKET, TRADING, POST_MARKET, OVERNIGHT, WEEKEND)

## Architecture Quick Reference

| Component | Entry Point | Purpose |
|-----------|-------------|---------|
| Orchestrator | daily_orchestrator.py | Phase management, task execution |
| Execution | execution/execution_manager.py | Signal routing, position limits |
| Research | run_nightly_research.py | GA/GP strategy optimization |
| Hardware | hardware/integration.py | LEDs, LCD, encoder |

## Common Issues

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| No trades executing | Circuit breaker triggered | Check `circuit_breaker_state` table |
| Orchestrator frozen | Memory pressure or deadlock | `sudo systemctl restart trading-orchestrator` |
| Position not exiting at target | Was using global 10% threshold | Fixed: now uses per-position TP/SL from DB |
| Pre-market refresh hung for hours | Alpaca API call with no timeout | Fixed: 30s timeout + 3 retries on all fetchers |
| LCD showing stale data | Orchestrator not updating | Check service status |
| Research not running | Wrong phase or disabled | Check ENABLE_* flags in config.py |
| Watchdog crash loop | Systemd `WatchdogSec` without `sd_notify` | Remove `WatchdogSec` from service file |
| "Nightly research failed" | Time boundary stop (expected) | Check if `stopped_early: True` - not a real failure |
| "Could not get open positions" | Wrong method call | Use `execution_tracker.db.get_open_positions()` |
| System reboot overnight | Memory pressure during research | LRU cache limits added, monitor with `free -h` |

## VGP Migration (Jan 23)

**Problem**: 68% of GP-discovered strategies exploited scalar edge cases like `const_1 > open()`.

**Solution**: Replaced scalar GP with Vectorial GP (VGP) primitives that express temporal patterns.

**Key Changes**:
- `gp_core.py`: Added `VecType`, ~15 vector primitives, `create_vgp_primitive_set()`
- `strategy_genome.py`: `use_vgp` flag (default True), `genome_version: 2` in serialization
- `config.py`: `vgp_lookbacks: [3, 5, 10]`, `vgp_enabled: True`
- `evolution_engine.py`: VGP validation (requires vector primitives in entry tree)
- `backtester_fast.py`: Added `trades`, `equity_curve` for compatibility

**Database**:
- Archived: `*_v1_scalar` tables (184 strategies preserved)
- Fresh v2 tables with `genome_version` field

**Example VGP Trees**:
```python
vec_cross_above(vec_ema_12_5, vec_sma_20_5)  # MA crossover
vec_rising(vec_rsi_5)                         # RSI trend
vec_converging(vec_vol_5(), vec_ret_1d_5())  # Volume/returns pattern
```

**Rollback**: `python research/discovery/vgp_migration.py --rollback`

## Recent Fixes (Jan 22)

- **Position exits**: Now use per-position `take_profit`/`stop_loss` prices from DB (not global 10% threshold)
- **Rapid gain scaler**: Disabled (was trimming positions prematurely)
- **API timeouts**: Alpaca fetchers have 30s request timeout, 60s per-symbol limit, 3 retries with backoff
- **Watchdog thresholds**: 98% memory (was 95%), 80% swap, 15-minute tolerance (was 5)

## Previous Fixes (Jan 14)

- **Watchdog service**: Removed `WatchdogSec=120` (script doesn't use sd_notify)
- **Research exit codes**: Time boundary stop now returns success=True
- **EOD refresh**: Added per-symbol and overall timeouts
- **Memory management**: LRU cache in shared_data.py, gc.collect() in EOD refresh

## Variable Naming Standards

Canonical types defined in `core/types.py` - import from there:
```python
from core.types import Signal, Side, SignalScore, EnsembleResult
```

**Key field names** (see `.claude/skills/variable-standards.md` for full reference):
| Concept | Use | NOT |
|---------|-----|-----|
| Strategy ID | `strategy_id` | strategy_name, strategy |
| Confidence | `strength` | confidence, conviction |
| Direction | `side` (Side enum) | signal_type, direction |
| P&L % | `pnl_pct` | pnl_percent, profit_pct |
| Share count | `quantity` | shares, qty |
| Target | `target_price` | take_profit, target |

**Backward compat**: Signal class accepts both old and new names in constructor.

## Skills Available

Detailed reference in `~/.claude/skills/tradebot-*/`:
- system-map, diagnostics, hardware, trading, research-pipeline, execution-flow

Project-level skills in `.claude/skills/`:
- variable-standards: Canonical naming conventions for signals, trades, positions
