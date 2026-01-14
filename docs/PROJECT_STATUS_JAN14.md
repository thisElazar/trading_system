# Trading System Project Status
**Date:** January 14, 2026
**System:** Raspberry Pi 5 running automated paper trading via Alpaca

---

## Executive Summary

Week 2 of operation. System stability significantly improved with fixes for research engine robustness, timeout handling, and memory management. All core systems operational. The punchlist remains 100% complete (41/41 items).

**Current State:**
- **Orchestrator:** Running (4h 58m uptime after morning reboot)
- **Phase:** MARKET_OPEN (trading active)
- **Open Positions:** 35 positions, +$51.42 unrealized P&L
- **GP Discovered Strategies:** 132 candidates in pipeline
- **GA Optimization:** 7 strategies actively evolving (174-255 generations each)

---

## Recent Changes (Jan 12-14)

### Commits This Period (15 commits)

| Commit | Description |
|--------|-------------|
| `fadf83c` | **Jan 14**: Stability fixes - EOD timeouts, LRU cache, watchdog fix |
| `d5430f4` | Fix position entry price drift from broker |
| `a4715d8` | Fix BehaviorVector and Position access errors |
| `fc7b87f` | Comprehensive research engine fixes documentation |
| `deaa3a8` | Log genome rejections at INFO level |
| `4c877ee` | Survivorship bias filters and research time boundaries |
| `6218f92` | Fix broken pipe and iloc indexing errors |
| `c67b448` | Pre-market research cleanup safeguard |
| `e487df3` | Connect GP Discovery to Live Trading execution |
| `f0dc96e` | Fix timezone migration issues |
| `4900f7d` | Complete timezone migration (22 files) |

### Issues Fixed Today (Jan 14)

| Issue | Root Cause | Fix |
|-------|------------|-----|
| **Watchdog crash loop** | `WatchdogSec=120` required `sd_notify()` calls script doesn't make | Removed from systemd service |
| **"Could not get open positions"** | Called `execution_tracker.get_open_positions()` instead of `.db.get_open_positions()` | Fixed method call |
| **Nightly research exit code 1** | Time boundary stop returned `success: False` | Now returns `success: True, stopped_early: True` |
| **EOD refresh hangs** | No timeout on symbol fetches | Added 15s/symbol, 30min overall timeouts |

### System Reboot Investigation

The Pi rebooted at 04:41 PT this morning. Investigation findings:
- Full system reboot (not just service crash)
- No OOM killer evidence in dmesg
- Likely cause: Hardware watchdog timeout during overnight research memory pressure
- Mitigation: Added LRU cache limits, gc.collect() calls, and timeout wrappers

---

## System Health

### Services Status
```
trading-orchestrator    active (running)  - 4h 58m uptime
trading-dashboard       active (running)
system-watchdog         active (running)  - Now stable (was crashing every 2 min)
```

### Resource Usage
- **Memory:** 54.7% (healthy)
- **Load:** 0.44 (healthy)
- **Disk:** Adequate

### Trading Status
- **Phase:** MARKET_OPEN
- **Open Positions:** 35
- **Unrealized P&L:** +$51.42
- **Circuit Breakers:** None active

---

## Research Pipeline Status

### GA Parameter Optimization

| Strategy | Generation | Best Sharpe |
|----------|------------|-------------|
| relative_volume_breakout | 196 | 2.59 |
| mean_reversion | 203 | 2.08 |
| factor_momentum | 222 | 1.41 |
| sector_rotation | 185 | 1.08 |
| quality_smallcap_value | 239 | 0.60 |
| vol_managed_momentum | 255 | 0.24 |
| vix_regime_rotation | 174 | 0.01 |

**Notes:**
- Diversity injection every 5 generations
- Hard reset at 50 generations stagnation
- Cross-session stagnation detection working

### GP Strategy Discovery
- **Discovered Strategies:** 132 candidates
- **Status:** Enabled (`ENABLE_STRATEGY_DISCOVERY = True`)
- **Last Run:** Reached generation 306 before time boundary

### Promotion Pipeline
- **Auto-validation:** Sortino >= 1.5 auto-promotes to VALIDATED
- **Paper Trading:** 90 days / 60 trades observation
- **Live Integration:** `strategy_loader.py` connects to scheduler

---

## Architecture Reference

### Key Files Modified Recently
| File | Purpose |
|------|---------|
| `daily_orchestrator.py` | Phase management, task execution, EOD refresh |
| `run_nightly_research.py` | Overnight optimization, time boundaries |
| `research/discovery/parallel_pool.py` | Worker pools with timeouts |
| `research/discovery/shared_data.py` | LRU-bounded data cache |
| `execution/strategy_loader.py` | GP → live trading bridge |

### Systemd Services
```
/etc/systemd/system/trading-orchestrator.service
/etc/systemd/system/trading-dashboard.service
/etc/systemd/system/system-watchdog.service  # Fixed today
/etc/systemd/system/trading-backup.timer
```

---

## Known Issues & Limitations

### Active
1. **VIX data:** Using Yahoo Finance fallback (Alpaca doesn't provide ^VIX)
2. **Intraday strategies:** Gap-fill disabled pending data stream fixes
3. **LCD occasional reset:** I2C recovery via 5-second button hold

### Resolved This Week
- Watchdog crash loop (124 restarts → stable)
- Research time boundary exit codes
- EOD refresh hangs
- Position access errors

---

## Quick Reference Commands

```bash
# Service status
systemctl status trading-orchestrator trading-dashboard system-watchdog

# Live logs
journalctl -u trading-orchestrator -f

# Open positions
sqlite3 ~/trading_system/db/trades.db \
  "SELECT symbol, direction, quantity, unrealized_pnl FROM positions WHERE status='open';"

# Research progress
sqlite3 ~/trading_system/db/research.db \
  "SELECT strategy, MAX(generation), MAX(best_fitness) FROM ga_history GROUP BY strategy;"

# GP discoveries
sqlite3 ~/trading_system/db/research.db \
  "SELECT COUNT(*) FROM discovered_strategies;"

# Kill switch
touch ~/trading_system/killswitch/HALT   # Stop all orders
rm ~/trading_system/killswitch/HALT      # Resume
```

---

## Documentation Index

| Document | Purpose | Last Updated |
|----------|---------|--------------|
| `CLAUDE.md` | Quick reference for Claude | Jan 11 |
| `docs/PUNCHLIST.md` | Issue tracking (100% complete) | Jan 11 |
| `docs/RESEARCH_ENGINE_FIXES_JAN2026.md` | Research fixes detail | Jan 13 |
| `docs/PROJECT_STATUS_JAN11.md` | Previous status | Jan 11 |
| `docs/TIMEZONE_MIGRATION.md` | Timezone handling guide | Jan 11 |
| `docs/DAY_CYCLE.md` | Trading day phases | Jan 9 |

---

## Next Considerations

Not urgent, but worth monitoring:
1. **Memory pressure during research** - LRU cache added, monitor effectiveness
2. **VIX data reliability** - Yahoo fallback working but not ideal
3. **GP → Live pipeline** - 132 candidates, none promoted to LIVE yet (expected - takes 90 days)

---

*Document generated: January 14, 2026, 09:30 PT*
