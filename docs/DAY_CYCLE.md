# Trading System Day Cycle

Complete reference for the daily operational cycle of the autonomous trading system.

**Last Updated:** 2026-01-07

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
| `refresh_premarket_data` | Fetch pre-market quotes and overnight moves |
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

## Related Documentation

- [AUTONOMOUS_RESEARCH_ENGINE.md](AUTONOMOUS_RESEARCH_ENGINE.md) - Research system details
- [STRATEGY_PORTFOLIO_OVERVIEW.md](STRATEGY_PORTFOLIO_OVERVIEW.md) - Strategy descriptions
- [TRADEBOT_RD_OVERVIEW.md](TRADEBOT_RD_OVERVIEW.md) - R&D architecture
