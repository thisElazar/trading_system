# TradeBot Genesis: From Broken SD to Market Ready

**Birthday: January 5, 2026**

---

## The Crash

On January 4th, 2026, while installing the Raspberry Pi 5 into its case, the microSD card was damaged. Weeks of development, configuration, and tuning—gone in an instant.

The system had been running smoothly:
- Genetic algorithm optimization finding profitable strategy parameters
- Shadow trading pipeline validating new strategies
- Nightly research cycles evolving the population
- Dashboard monitoring everything in real-time

Then silence.

---

## The Recovery

A fresh Raspbian installation on a new SD card. Node.js reinstalled to get Claude Code back online. But what else was missing?

### System Audit (Jan 4, 2026)

**What Survived** (on the NVMe):
- All trading system code (`/mnt/nvme/home/thiselazar/trading_system/`)
- Historical market data (1.4GB)
- GA populations with weeks of evolution
- 92 discovered strategies
- Database with performance metrics

**What Was Missing**:
| Package | Purpose | Status |
|---------|---------|--------|
| sqlite3 | Database CLI tool | Reinstalled |
| tree | Directory visualization | Reinstalled |
| vim | Full editor | Reinstalled |

### Backup Created

To prevent future disasters, a complete dependency backup was created on NVMe:

```
/mnt/nvme/system_backup/
├── installed_packages.txt    # 1,654 system packages
├── python_packages.txt       # 84 Python packages
├── requirements.txt          # Original requirements
├── node_version.txt          # Node v24.12.0
├── trading-*.service         # Systemd service files
└── restore_system.sh         # One-command restore script
```

Future recovery: `sudo bash /mnt/nvme/system_backup/restore_system.sh`

---

## The Improvements

With the system restored, we tackled a critical bug that had been holding back the genetic algorithm.

### The Stagnation Problem

The GA was stuck. Generation after generation showed zero improvement:

```
Generation 11: 27/40 rejected by constraints
Generation 12: 33/40 rejected by constraints
Generation 13: 31/40 rejected by constraints
Improvement: +0.0000
```

**Root Cause**: Walk-forward validation uses only 30% of data for out-of-sample testing. The constraint thresholds were calibrated for full datasets, causing 70-85% of offspring to be rejected. Elitism kept copying the same 2 individuals, and the population stagnated.

### The Fix: Anti-Stagnation Features

**1. OOS-Aware Constraint Scaling**
```python
# Before: Same thresholds for in-sample and out-of-sample
min_trades = 30  # Too high for 30% of data

# After: Scale by test ratio
if is_oos:
    min_trades_scaled = max(5, int(30 * 0.3))  # = 9 trades
```

**2. Soft Penalties Instead of Hard Rejection**
```python
# Before: Hard rejection
return 0.0, "Rejected"

# After: Small positive fitness maintains diversity
REJECTION_FITNESS = 0.01
return REJECTION_FITNESS, "Low fitness"
```

**3. Diversity Injection**
When >50% of population has low fitness, replace 20% with fresh random individuals:
```
Diversity injection: Replaced 8 low-fitness individuals (80% had low fitness)
```

**4. Adaptive Mutation**
- Base rate: 15%
- Increases 5% per stagnant generation
- Caps at 40%
- Resets to 15% when improvement found

### Results

Before:
```
Generation 1: 70% rejected, Improvement: +0.0000
Generation 2: 75% rejected, Improvement: +0.0000
Generation 3: 80% rejected, Improvement: +0.0000
Early stopping: No improvement
```

After:
```
Generation 1: NEW BEST = 0.2339
Generation 2: NEW BEST = 0.2397
Generation 3: Adaptive mutation: 20%
Generation 4: NEW BEST = 0.2409
Continuous improvement!
```

---

## System Architecture

### Hardware
- **Raspberry Pi 5** (8GB RAM)
- **NVMe SSD** (256GB) - Primary storage, survives SD failures
- **Network**: Local LAN + Alpaca API

### Trading Stack
```
┌─────────────────────────────────────────────────────────┐
│                    Daily Orchestrator                    │
│  (Phase-based scheduling: pre-market → post-market)     │
├─────────────────────────────────────────────────────────┤
│  Intraday Strategies    │    Swing Strategies           │
│  ├─ Gap Fill (SPY/QQQ)  │    ├─ Vol Managed Momentum    │
│  ├─ ORB                 │    ├─ Sector Rotation         │
│  └─ VWAP Reversion      │    ├─ Factor Momentum         │
│                         │    ├─ Mean Reversion          │
│                         │    └─ Quality SmallCap Value  │
├─────────────────────────────────────────────────────────┤
│  Execution Layer        │    Research Layer             │
│  ├─ Alpaca Broker       │    ├─ Genetic Optimizer       │
│  ├─ Shadow Trader       │    ├─ Walk-Forward Validation │
│  └─ Risk Manager        │    └─ Strategy Discovery      │
├─────────────────────────────────────────────────────────┤
│                      Data Layer                          │
│  ├─ Unified Data Loader (Yahoo + Alpaca)                │
│  ├─ Real-time Streaming (Alpaca WebSocket)              │
│  └─ SQLite Databases (trades, research, performance)    │
└─────────────────────────────────────────────────────────┘
```

### Services
| Service | Port | Purpose |
|---------|------|---------|
| trading-orchestrator | - | Phase-based task scheduling |
| trading-dashboard | 8050 | Real-time monitoring UI |

### Databases
| Database | Tables | Purpose |
|----------|--------|---------|
| trades.db | trades, positions, signals | Execution tracking |
| research.db | ga_populations, discoveries | Strategy evolution |
| performance.db | strategy_daily, regime_log | Performance analytics |
| pairs.db | pairs, spread_history | Pairs trading |

---

## Current State (Jan 5, 2026)

### Account Status
```
Broker: Alpaca (Paper Trading)
Account Status: ACTIVE
Equity: $96,715.13
Buying Power: $306,236.12
Pattern Day Trader: Yes
```

### Open Positions
| Symbol | Shares | Value | Unrealized P/L |
|--------|--------|-------|----------------|
| DHR | 29 | $6,681 | +$2 |
| IWM | 38 | $9,453 | -$30 |
| MS | 40 | $7,276 | +$66 |
| QQQ | 15 | $9,196 | -$92 |
| XEL | 80 | $5,974 | +$27 |
| XLF | 175 | $9,612 | -$62 |
| XLK | 66 | $9,523 | -$70 |
| XLY | 80 | $9,468 | -$204 |
| **Total** | | **$67,187** | **-$362** |

### GA Evolution Status
| Strategy | Generation | Best Fitness |
|----------|------------|--------------|
| relative_volume_breakout | 13 | 1.51 |
| factor_momentum | 25 | 1.41 |
| sector_rotation | 13 | 1.08 |
| quality_smallcap_value | 19 | 0.60 |
| mean_reversion | 31 | 0.53 |
| vol_managed_momentum | 4 | 0.24 |

---

## What Happens Tomorrow

### Market Open Schedule (ET)
| Time | Phase | Actions |
|------|-------|---------|
| 8:00 AM | Pre-Market | Data refresh, system check, sync positions |
| 9:30 AM | Intraday Open | Start streaming, detect gaps |
| 9:35 AM | Intraday Active | Monitor intraday positions |
| 9:35 AM+ | Market Open | Strategy scheduler, signal scoring |
| 4:00 PM | Post-Market | Reconcile, calculate P&L, reports |
| 5:00 PM | Evening | Cleanup, backups |
| 9:30 PM | Overnight | Nightly research, GA evolution |

### First Live Day Checklist
- [x] Services running (orchestrator, dashboard)
- [x] Alpaca connection verified
- [x] Positions synced from broker
- [x] Intraday strategies loaded (Gap Fill on SPY/QQQ)
- [x] GA anti-stagnation fixes deployed
- [x] Shadow trading pipeline tested
- [x] NVMe backup created

---

## Lessons Learned

1. **Always backup to NVMe** - SD cards fail, NVMe survives
2. **Document dependencies** - `restore_system.sh` saves hours
3. **Test constraint logic** - Walk-forward needs different thresholds
4. **Diversity matters** - Hard rejection kills genetic diversity
5. **Adaptive systems win** - Static parameters stagnate

---

## The Road Ahead

TradeBot is born. January 5, 2026.

Today marks the first full market day running autonomously on the Raspberry Pi 5. The genetic algorithms will continue evolving. The shadow trader will validate new strategies. The orchestrator will manage the daily rhythm of markets.

From a broken SD card to a self-improving trading system.

*The best systems are built from failures.*

---

**TradeBot v1.0**
*Raspberry Pi 5 | Alpaca Paper Trading | Genetic Optimization*

*"Evolve. Adapt. Trade."*
