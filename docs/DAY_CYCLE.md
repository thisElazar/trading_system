# Trading System Day Cycle

Complete reference for the daily operational cycle of the autonomous trading system.

**Last Updated:** 2026-01-18

---

## Overview

The system operates in distinct phases throughout the 24-hour cycle, each with specific tasks and check intervals. All times are **Eastern Time (ET)**.

```
08:00 ──► PRE_MARKET ──► 09:30 ──► MARKET_OPEN ──► 16:00 ──► POST_MARKET ──► 17:00 ──► EVENING ──► 21:30 ──► OVERNIGHT ──► 08:00
                              │
                              ├── INTRADAY_OPEN (09:30-09:35)
                              └── INTRADAY_ACTIVE (09:35-11:30)
```

---

## Phase Details

### PRE_MARKET (08:00 - 09:30 ET)
**Check Interval:** 60 seconds

Prepares the system for the trading day.

| Task | Description |
|------|-------------|
| `refresh_premarket_data` | Fetch daily bars for ~110 priority symbols (positions + core ETFs) |
| `refresh_intraday_data` | Download 5 days of minute bars for 42-symbol intraday universe |
| `refresh_data` | Update historical data cache |
| `system_check` | Verify broker connectivity, API keys, disk space |
| `sync_positions_from_broker` | Reconcile local DB with Alpaca positions |
| `review_positions` | Check overnight gaps on held positions |
| `cancel_stale_orders` | Cancel any unfilled orders from previous day |
| `update_regime_detection` | Calculate current VIX regime (low/normal/high/extreme) |
| `calculate_position_scalars` | Compute position sizing multipliers based on regime |
| `load_live_strategies` | Load GP-discovered strategies from promotion pipeline |

---

### INTRADAY_OPEN (09:30 - 09:35 ET)
**Check Interval:** 5 seconds (fast)

Critical first 5 minutes of market open.

| Task | Description |
|------|-------------|
| `start_intraday_stream` | Initialize real-time market data stream |
| `detect_gaps` | Identify gap-up/gap-down opportunities for gap_fill strategy |

---

### INTRADAY_ACTIVE (09:35 - 11:30 ET)
**Check Interval:** 10 seconds

Active intraday trading window.

| Task | Description |
|------|-------------|
| `monitor_intraday_positions` | Track intraday positions, check exit conditions |

---

### MARKET_OPEN (09:30 - 16:00 ET)
**Check Interval:** 30 seconds

*Runs parallel to intraday phases.* Core trading operations.

| Task | Description |
|------|-------------|
| `run_scheduler` | Execute strategies at scheduled times, generate signals |
| `monitor_positions` | Check position concentration (warn if >20% single position) |
| `check_risk_limits` | Circuit breakers, margin usage, daily loss limits |
| `score_pending_signals` | ML-based signal scoring for conviction/win probability |
| `process_shadow_trades` | Paper trading for strategies not yet graduated to live |

---

### POST_MARKET (16:00 - 17:00 ET)
**Check Interval:** 60 seconds

End-of-day reconciliation and reporting.

| Task | Description |
|------|-------------|
| `reconcile_positions` | Final sync with broker, verify all positions |
| `calculate_pnl` | Compute daily P&L, update performance metrics |
| `generate_daily_report` | Create summary of day's activity |
| `send_alerts` | Email/notification with daily summary |
| `update_ensemble_correlations` | Update strategy correlation matrix |
| `update_paper_metrics` | Update paper trading metrics in promotion pipeline |
| `update_live_metrics` | Update live trading metrics in promotion pipeline |
| `run_promotion_pipeline` | Check if paper strategies ready for live promotion |

---

### EVENING (17:00 - 21:30 ET)
**Check Interval:** 300 seconds (5 minutes)

Maintenance and data refresh.

| Task | Description |
|------|-------------|
| `refresh_eod_data` | Download end-of-day OHLCV data |
| `cleanup_logs` | Rotate and compress old log files |
| `backup_databases` | Backup SQLite databases |
| `cleanup_databases` | Vacuum and optimize databases |

---

### OVERNIGHT (21:30 - 08:00 ET)
**Check Interval:** 600 seconds (10 minutes)

Research and optimization.

| Task | Description |
|------|-------------|
| `run_nightly_research` | GA parameter optimization, strategy discovery, adaptive GA |
| `train_ml_regime_model` | Retrain regime detection ML model |

**Research Phases:**
1. **Parameter Optimization** - Evolve parameters of existing strategies (3 generations/night)
2. **Strategy Discovery** - GP-based discovery of novel strategy genomes
3. **Adaptive GA** - Regime-matched multi-scale testing

**Unified Scheduler Mode:** When `USE_UNIFIED_SCHEDULER=true`, overnight uses the same budget-aware task selection as weekends. Research budget is ~9 hours (10.5h total minus 1.5h prep reserve). If the next day is a holiday, the overnight window extends automatically.

---

### WEEKEND (Fri 16:00 - Sun 20:00 ET)
**Check Interval:** 1800 seconds (30 minutes)

Extended research window.

| Task | Description |
|------|-------------|
| `run_weekend_schedule` | Master dispatcher for extended research phases |

**Weekend Research:**
- Extended parameter optimization (more generations)
- Deep strategy discovery runs
- Full backtest validation of promoted strategies

#### Unified Scheduler Mode (when `USE_UNIFIED_SCHEDULER=true`)

When enabled, the Unified Scheduler replaces the rigid `WeekendSubPhase` state machine with dynamic budget-aware task selection:

| Budget Remaining | Tasks |
|------------------|-------|
| > 80% | Cleanup: weekly report, backup, vacuum |
| 20%-80% | Research: GA/GP optimization |
| 1.5h - 20% | Data refresh: index constituents, fundamentals |
| < 1.5h | Prep: validate strategies, verify readiness |

**Key Benefits:**
- Same logic works for weekends, overnight, and holidays
- Mid-week holidays get proportional research time (~22h, not 56h)
- Dashboard shows actual hours remaining instead of phase names

**Configuration:**
```bash
export USE_UNIFIED_SCHEDULER=true  # Enable
export USE_UNIFIED_SCHEDULER=false # Disable (rollback)
```

---

## Signal Flow During Market Hours

```
Strategy generates signal
         │
         ▼
┌─────────────────────┐
│  ExecutionManager   │
│  evaluate_signal()  │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Position Limits?    │──► At limit ──► Check Smart Override (85%+ conviction)
│ Global: 20          │                         │
│ Per-strategy: 8     │                         ▼
└─────────────────────┘               Check Rebalancing (15% better?)
         │                                      │
         ▼                                      ▼
    Under limit                    Override/Rebalance approved?
         │                                      │
         ▼                              Yes ◄───┴───► No = REJECT
┌─────────────────────┐
│  Signal Scoring     │
│  (conviction, win%) │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Route Decision     │
│  SHADOW vs LIVE     │
└─────────────────────┘
         │
         ▼
    Execute Order
```

---

## Hardware Integration

During operation, the system updates hardware status:

| LED | Meaning |
|-----|---------|
| **System (LED 1)** | Green = healthy, Yellow = warning, Red = error |
| **Trading (LED 2)** | Green = active trading, Blue = idle, Off = stopped |
| **Research (LED 3)** | Blue breathing = evolving, Green = complete, Off = idle |

**LCD Display Pages:**
- MARKET - SPY price, VIX, current phase
- TRADING - Portfolio value, daily P&L
- POSITIONS - List of open positions
- SYSTEM - RAM, CPU, uptime
- RESEARCH - Generation progress, best Sharpe

---

## Data Architecture

The system maintains two data universes refreshed on different schedules:

### Daily Bars (Full Universe)
**2,562 symbols | 627 MB | 10+ years history**

| Source | Symbols | Purpose |
|--------|---------|---------|
| Alpaca | 805 | Recent high-fidelity data |
| Yahoo | 2,556 | Extended historical depth |

**Refresh Schedule:**
- PRE_MARKET (8am): ~110 priority symbols (positions + core ETFs + sample)
- EVENING (5pm): Full universe EOD refresh

### Intraday/Minute Bars (Trading Universe)
**42 symbols | 5 MB | 30-day retention**

| Category | Symbols |
|----------|---------|
| Broad Market | SPY, QQQ, IWM, DIA |
| Sectors | XLF, XLE, XLK, XLV, XLI, XLP, XLU, XLY, XLC, XLB, XLRE |
| Mega-caps | AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, JPM, V, UNH |
| Volatility | VXX, UVXY, VIXY |
| Commodities | GLD, SLV, USO, UNG |
| Bonds | TLT, HYG, LQD |
| International | EEM, EFA, FXI |
| Thematic | ARKK, XBI, SMH, KWEB |

**Refresh Schedule:**
- PRE_MARKET (8am): Download last 5 trading days (~75 seconds)
- Configured in `config.py:INTRADAY_UNIVERSE`

---

## Key Configuration

```python
# config.py
MAX_POSITIONS = 20              # Global position limit
MAX_POSITION_SIZE = 15_000      # Max $ per position

# execution_manager.py
max_positions_per_strategy = 8  # Per-strategy limit
max_position_pct = 0.05         # 5% max per position
override_conviction_threshold = 0.85  # For smart override
```

---

## Monitoring

**Log Files:**
- `logs/orchestrator.log` - Main orchestrator activity
- `logs/nightly_research.log` - Research/GA progress
- `logs/execution.log` - Trade execution details

**Dashboard:**
- Web UI at `http://<pi-ip>:8050`
- Shows positions, P&L, research progress, system health

---

## System Watchdog

The system includes an automatic recovery watchdog that monitors health and reboots the Pi if the system becomes unresponsive.

### Two-Layer Protection

| Layer | What It Catches | Timeout | Action |
|-------|-----------------|---------|--------|
| **Software Watchdog** | Memory exhaustion, disk I/O failure, hung processes | 5 minutes | Graceful restart, then reboot |
| **Hardware Watchdog** | Kernel panic, complete system freeze | ~15 seconds | Hard reboot |

### Health Checks (every 30 seconds)

| Check | Threshold | Notes |
|-------|-----------|-------|
| Memory | < 95% used | Prevents OOM during research |
| Disk I/O | Can write/read | Catches NVMe issues |
| Load Average | < 8.0 | Pi 5 has 4 cores |
| Orchestrator | Process exists, not in D state | Ensures trading system responsive |

### Behavior

1. **System goes unhealthy** → Watchdog starts counting, logs warnings
2. **Unhealthy for 5 continuous minutes** → Triggers restart sequence
3. **Restart sequence:**
   - Stop `trading-orchestrator` service gracefully (30s timeout)
   - Execute `sudo reboot`
4. **Restart history** saved to `logs/watchdog_restarts.log`

### Service Management

```bash
# Check watchdog status
sudo systemctl status system-watchdog

# View live health checks
sudo journalctl -u system-watchdog -f

# Temporarily disable (for maintenance)
sudo systemctl stop system-watchdog

# Re-enable
sudo systemctl start system-watchdog
```

### Log Files

- `logs/watchdog.log` - Health check history
- `logs/watchdog_restarts.log` - Reboot events (survives restart)

### Configuration

Edit `scripts/system_watchdog.py` to adjust:
```python
CHECK_INTERVAL = 30          # Seconds between checks
UNHEALTHY_THRESHOLD = 300    # Seconds (5 min) before restart
MAX_MEMORY_PCT = 95          # Memory threshold
MAX_LOAD_AVG = 8.0           # Load average threshold
```

---

## Related Documentation

- [AUTONOMOUS_RESEARCH_ENGINE.md](AUTONOMOUS_RESEARCH_ENGINE.md) - Research system details
- [STRATEGY_PORTFOLIO_OVERVIEW.md](STRATEGY_PORTFOLIO_OVERVIEW.md) - Strategy descriptions
- [TRADEBOT_RD_OVERVIEW.md](TRADEBOT_RD_OVERVIEW.md) - R&D architecture
- [TASKSCHEDULER_DESIGN.md](TASKSCHEDULER_DESIGN.md) - TaskScheduler architecture and Unified Scheduler
- [TASKSCHEDULER_IMPLEMENTATION.md](TASKSCHEDULER_IMPLEMENTATION.md) - Implementation phases and commits
