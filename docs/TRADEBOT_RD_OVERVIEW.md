# Tradebot: A Research Platform for Evolutionary Trading Systems

**Status:** Research & Development Phase
**Timeline:** 3+ months before live capital consideration
**Focus Areas:** Genetic Algorithm Computation, Architectural Efficiency, Programming Stability
**Platform:** Raspberry Pi 5 (8GB RAM, NVMe SSD)
**Last Updated:** January 7, 2026

> Related documentation: [DAY_CYCLE.md](DAY_CYCLE.md) (operational schedule), [AUTONOMOUS_RESEARCH_ENGINE.md](AUTONOMOUS_RESEARCH_ENGINE.md) (GP details), [STRATEGY_PORTFOLIO_OVERVIEW.md](STRATEGY_PORTFOLIO_OVERVIEW.md) (strategy analysis)

---

## Executive Summary

Tradebot is not a trading system in the conventional sense—it is a **research platform** designed to explore the intersection of evolutionary computation and financial markets. The current deployment on a Raspberry Pi 5 serves as a constrained environment that forces architectural discipline: if the system can run efficiently on 8GB of RAM while evolving trading strategies overnight, it can scale anywhere.

The 3+ month R&D phase before any real capital is intentional. This time is devoted to:

1. **Validating the genetic algorithm infrastructure** - Can the system reliably discover, evaluate, and promote novel trading strategies?
2. **Stress-testing architectural decisions** - Does the phase-based orchestrator handle market hours correctly? Do circuit breakers trigger appropriately?
3. **Hardening the codebase** - Are there edge cases that cause crashes? Memory leaks? Database corruption?

Real trading is a downstream consequence of getting these fundamentals right.

---

## Part I: The Genetic Algorithm Engine

### Philosophy

The core hypothesis driving Tradebot is that **trading strategies can be discovered rather than invented**. Human traders bring biases, recency effects, and emotional attachment to ideas. A genetic algorithm brings none of these—it simply asks: "What patterns in historical data would have generated returns with acceptable risk?"

This is not curve-fitting. The system employs multiple defenses against overfitting:

- **Walk-forward validation**: Strategies are tested on data they've never seen
- **Deflated Sharpe Ratio**: Adjusts for multiple hypothesis testing
- **Novelty search**: Rewards behavioral diversity, not just performance
- **MAP-Elites**: Maintains a grid of strategies across different risk/return profiles
- **Complexity penalties**: Simpler strategies are preferred over complex ones

### The Genome Representation

Each evolved strategy is encoded as a **5-tree genome**:

```
┌─────────────────────────────────────────────────────────────────┐
│                      STRATEGY GENOME                            │
├─────────────────────────────────────────────────────────────────┤
│  entry_tree     │  Boolean GP tree → When to BUY                │
│  exit_tree      │  Boolean GP tree → When to SELL               │
│  position_tree  │  Float GP tree → Position size [0-1]          │
│  stop_loss_tree │  Float GP tree → Stop distance [%]            │
│  target_tree    │  Float GP tree → Profit target [%]            │
└─────────────────────────────────────────────────────────────────┘
```

These trees are built from primitives:
- **Operators:** AND, OR, NOT, IF-THEN, <, >, ==
- **Terminals:** SMA, EMA, RSI, Bollinger Bands, ATR, momentum indicators

A sample entry tree might evolve to: `AND(RSI < 30, price < BB_lower, volume > SMA_volume_20)` — encoding an oversold bounce strategy without any human suggesting it.

### Multi-Objective Optimization (NSGA-II)

Unlike simple genetic algorithms that optimize a single metric, Tradebot uses **NSGA-II** to simultaneously optimize 5 competing objectives:

| Objective | Direction | Rationale |
|-----------|-----------|-----------|
| Sortino Ratio | Maximize | Risk-adjusted return (penalizes downside) |
| Max Drawdown | Minimize | Capital preservation |
| Win Rate | Maximize | Consistency |
| Profit Factor | Maximize | Gross profit / gross loss |
| Complexity | Minimize | Parsimony pressure |

This produces a **Pareto front**—a set of non-dominated solutions where improving one objective necessarily worsens another. The promotion pipeline then selects strategies from this front based on portfolio-level criteria (correlation with existing strategies, regime coverage).

### Behavioral Diversity: Novelty Search + MAP-Elites

A pure fitness-driven GA tends to converge on a single "optimal" solution. In trading, this is dangerous—markets are non-stationary, and what works in one regime fails in another.

**Novelty Search** addresses this by rewarding strategies that *behave differently* from what's already in the population:

```
Behavior Vector = [annual_return, volatility, sharpe, max_dd, win_rate, profit_factor]

Novelty Score = average distance to k-nearest neighbors in behavior space
```

Strategies that explore new regions of behavior space are preserved even if their fitness is mediocre—they may prove valuable when regimes shift.

**MAP-Elites** complements this by maintaining a grid of "elite" strategies across different performance dimensions:

```
                    Max Drawdown
                    <10%    10-20%   20-30%   >30%
Annual Return ─────────────────────────────────────
    >30%      │  A  │   B   │   C   │   D   │
    20-30%    │  E  │   F   │   G   │   H   │
    10-20%    │  I  │   J   │   K   │   L   │
    <10%      │  M  │   N   │   O   │   P   │
              ─────────────────────────────────────
```

Each cell preserves the *best* strategy for that risk/return profile, ensuring the population maintains coverage across the entire solution space.

### Overnight Evolution Cycle

The GA runs overnight when markets are closed, utilizing all 4 CPU cores:

```
21:30 ET  ─→  Load checkpoint from research.db
          │
          ├─→  For each generation (3-5 per night):
          │      ├─ Tournament selection
          │      ├─ Crossover (80% rate)
          │      ├─ Mutation (20% base, adaptive up to 40%)
          │      ├─ Parallel fitness evaluation (4 workers)
          │      ├─ Non-dominated sorting (NSGA-II)
          │      ├─ Crowding distance assignment
          │      ├─ Novelty archive update
          │      └─ MAP-Elites grid update
          │
          ├─→  Checkpoint to research.db
          │
          └─→  Run promotion pipeline on top candidates
                    ├─ Walk-forward validation (60/40 split)
                    ├─ Out-of-sample Deflated Sharpe Ratio
                    ├─ Portfolio correlation check
                    └─ Promote if all criteria pass
          │
08:00 ET  ─→  Ready for market open
```

**Current Configuration (Pi-optimized):**
- Population size: 20-100 (adaptive)
- Generations per night: 3-5
- Max tree depth: 6 nodes
- Evaluation parallelism: 4 workers
- Checkpoint frequency: Every 10 generations

---

## Part II: Architectural Efficiency

### The Constrained Environment Advantage

Running on a Raspberry Pi 5 with 8GB RAM is a deliberate choice. It forces:

1. **Memory discipline**: No loading entire datasets into memory. Parquet files are read on-demand.
2. **Algorithmic efficiency**: The GA must evaluate strategies quickly. Vectorized operations are mandatory.
3. **Persistent state**: Everything important lives in SQLite. The system can restart from any point.
4. **Thermal awareness**: Long-running computations must not throttle the CPU.

If the system works here, it will work anywhere.

### Phase-Based Orchestration

The `daily_orchestrator.py` manages the system through distinct market phases:

```
┌────────────────────────────────────────────────────────────────────────┐
│                          TRADING DAY LIFECYCLE                         │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  08:00 ─────────────── PRE_MARKET ───────────────── 09:30              │
│        │  • Refresh overnight data                                     │
│        │  • System health check                                        │
│        │  • Sync positions from broker                                 │
│        │  • Update VIX regime detection                                │
│        │                                                               │
│  09:30 ─── INTRADAY_OPEN ─── 09:35                                     │
│        │  • Gap detection window                                       │
│        │  • Early entry signals                                        │
│        │                                                               │
│  09:35 ─────────────── INTRADAY_ACTIVE ───────────── 11:30             │
│        │  • Position monitoring                                        │
│        │  • Signal processing                                          │
│        │  • Intraday strategy execution                                │
│        │                                                               │
│  09:30 ─────────────── MARKET_OPEN (fallback) ─────── 16:00            │
│        │  • Strategy scheduler                                         │
│        │  • Continuous signal monitoring                               │
│        │                                                               │
│  16:00 ─────────────── POST_MARKET ─────────────────── 17:00           │
│        │  • Position reconciliation                                    │
│        │  • P&L calculation                                            │
│        │  • End-of-day position closing                                │
│        │                                                               │
│  17:00 ─────────────── EVENING ─────────────────────── 21:30           │
│        │  • Daily research (light)                                     │
│        │  • Watchlist updates                                          │
│        │  • Strategy parameter review                                  │
│        │                                                               │
│  21:30 ─────────────── OVERNIGHT ────────────────────── 08:00          │
│        │  • Genetic algorithm evolution                                │
│        │  • Strategy discovery (GP)                                    │
│        │  • Checkpoint system state                                    │
│        │                                                               │
│  WEEKEND (Fri 16:00 → Sun evening)                                     │
│        │  • Extended research cycles                                   │
│        │  • Full data refresh                                          │
│        │  • System validation                                          │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

Each phase has defined tasks that execute in sequence. The orchestrator is **idempotent**—if interrupted, it resumes from the correct phase on restart.

### Data Flow Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                           DATA PIPELINE                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   EXTERNAL SOURCES                                                   │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐                │
│   │ Alpaca  │  │ Yahoo   │  │ Yahoo   │  │Wikipedia│                │
│   │ (live)  │  │ (hist)  │  │ (VIX)   │  │ (SP500) │                │
│   └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘                │
│        │            │            │            │                      │
│        ▼            ▼            ▼            ▼                      │
│   ┌──────────────────────────────────────────────────────────┐      │
│   │              PARQUET FILES (1.4GB on NVMe)               │      │
│   │  ├─ daily/          (6 months, Alpaca)                   │      │
│   │  ├─ daily_yahoo/    (10 years, 2556 symbols)             │      │
│   │  ├─ intraday_1min/  (30 days rolling, ~500 symbols)      │      │
│   │  └─ vix/            (20+ years history)                  │      │
│   └────────────────────────┬─────────────────────────────────┘      │
│                            │                                         │
│                            ▼                                         │
│   ┌──────────────────────────────────────────────────────────┐      │
│   │              CachedDataManager (Memory Layer)            │      │
│   │  • LRU cache for frequently accessed symbols             │      │
│   │  • Lazy loading (only load what's needed)                │      │
│   │  • Automatic invalidation on data refresh                │      │
│   └────────────────────────┬─────────────────────────────────┘      │
│                            │                                         │
│                            ▼                                         │
│   ┌──────────────────────────────────────────────────────────┐      │
│   │              UnifiedDataLoader                           │      │
│   │  • Consolidates Alpaca + Yahoo data                      │      │
│   │  • Handles gaps and missing data                         │      │
│   │  • Provides consistent DataFrame interface               │      │
│   └────────────────────────┬─────────────────────────────────┘      │
│                            │                                         │
│                            ▼                                         │
│   ┌──────────────────────────────────────────────────────────┐      │
│   │              Technical Indicators                        │      │
│   │  • Bollinger Bands, RSI, MACD, ATR, SMA, EMA             │      │
│   │  • Computed on-demand, cached per symbol                 │      │
│   └────────────────────────┬─────────────────────────────────┘      │
│                            │                                         │
│                            ▼                                         │
│   ┌──────────────────────────────────────────────────────────┐      │
│   │              STRATEGIES (9 active)                       │      │
│   │  • Vol-Managed Momentum V2 (Sharpe 0.55)                 │      │
│   │  • Mean Reversion (Bollinger)                            │      │
│   │  • Gap-Fill (Intraday)                                   │      │
│   │  • Pairs Trading (debugging)                             │      │
│   │  • Relative Volume Breakout                              │      │
│   │  • Quality Small-Cap Value                               │      │
│   │  • Factor Momentum                                       │      │
│   │  • VIX Regime Rotation                                   │      │
│   │  • Sector Rotation (disabled)                            │      │
│   └──────────────────────────────────────────────────────────┘      │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Database Design (4 SQLite Databases)

The system uses purpose-separated databases to avoid lock contention:

| Database | Purpose | Size | Key Tables |
|----------|---------|------|------------|
| `trades.db` | Execution records | 176KB | signals, orders, executions, positions, trades |
| `performance.db` | Analytics | 132KB | strategy_daily, portfolio_daily, circuit_breaker_state, error_log |
| `research.db` | GA evolution | 1.1MB | ga_populations, evolution_checkpoints, strategy_candidates |
| `pairs.db` | Pairs trading | 40KB | cointegrated_pairs, pair_performance |

**Why SQLite?** It's embedded, requires no server, survives Pi reboots, and is surprisingly performant for this workload. The entire database layer fits in 5MB.

### Memory Architecture (Pi-Optimized)

```
┌────────────────────────────────────────────────────────────────────┐
│                      MEMORY HIERARCHY (8GB)                        │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  TARGET: <3GB active RAM to avoid swap pressure                    │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  Application Memory (~2.5GB typical)                        │  │
│  │  ├─ Python interpreter + venv                               │  │
│  │  ├─ Pandas DataFrames (100 symbols × 3 years max)           │  │
│  │  ├─ GA population (20-100 genomes)                          │  │
│  │  └─ SQLite page cache                                       │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                           │                                        │
│                           ▼ (overflow)                             │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  ZRAM (Compressed RAM swap, 1.4:1 ratio, zstd)              │  │
│  │  • First-tier overflow                                      │  │
│  │  • Minimal latency impact                                   │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                           │                                        │
│                           ▼ (overflow)                             │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  ZSWAP (Compressed swap cache)                              │  │
│  │  • Second-tier overflow                                     │  │
│  │  • Catches additional spills                                │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                           │                                        │
│                           ▼ (overflow)                             │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  NVMe Swap (Final fallback)                                 │  │
│  │  • Rarely hit in normal operation                           │  │
│  │  • 2GB swap partition                                       │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### Hardware Integration (Pi-Specific)

The system includes physical status indicators for at-a-glance monitoring:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HARDWARE STATUS PANEL                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────┐   ┌──────────────────────┐              │
│  │   RGB LED Array      │   │   LCD Display        │              │
│  │  ┌─────┬─────┬─────┐ │   │   (16x2 or 20x4)     │              │
│  │  │ SYS │ TRD │ RSH │ │   │                      │              │
│  │  │ 🟢  │ 🔵  │ 🔵  │ │   │  SPY: $478.25 ▲0.3% │              │
│  │  └─────┴─────┴─────┘ │   │  VIX: 14.2  NORMAL  │              │
│  └──────────────────────┘   └──────────────────────┘              │
│                                                                     │
│  LED Meanings:                                                      │
│  • System: Green=healthy, Yellow=warning, Red=error                │
│  • Trading: Green=active, Blue=idle, Off=halted                    │
│  • Research: Blue (breathing)=evolving, Green=complete             │
│                                                                     │
│  ┌──────────────────────┐                                          │
│  │   Rotary Encoder     │   Click: Scroll LCD pages                │
│  │       (input)        │   Hold 1-5s: Toggle backlight            │
│  │        ◎             │   Hold 5s+: Reset LCD                    │
│  └──────────────────────┘                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**LCD Display Pages:**
- MARKET: SPY price, VIX level, current phase
- TRADING: Portfolio value, daily P&L
- POSITIONS: Open positions list
- SYSTEM: RAM, CPU, uptime
- RESEARCH: Generation progress, best Sharpe (overnight only)

**Key Files:**
- `hardware/leds.py` - RGB LED controller with breathing effects
- `hardware/display.py` - LCD display manager
- `hardware/encoder.py` - Rotary encoder input handling
- `hardware/gpio_config.py` - Pin assignments

---

## Part III: Programming Stability

### Risk Management: The Circuit Breaker System

Five independent circuit breakers protect against runaway losses:

| Breaker | Trigger | Action | Duration |
|---------|---------|--------|----------|
| **Daily Loss** | Portfolio down >2% in a day | HALT all new orders | Until next day |
| **Drawdown** | Drawdown exceeds 15% | Reduce positions 50% | Until recovery |
| **Rapid Loss** | 1% loss in 15 minutes | PAUSE all strategies | 30 minutes |
| **Consecutive Losses** | 5 losses in a row | PAUSE triggering strategy | 4 hours |
| **Strategy Loss** | Strategy down 5% | DISABLE strategy | 24 hours |

**Kill Switch:** The ultimate failsafe. Creating `/killswitch/HALT` immediately:
- Cancels all pending orders
- Optionally closes all positions
- Stops all new signal generation

```bash
# Emergency stop
touch ~/trading_system/killswitch/HALT

# Graceful shutdown (complete pending orders, then stop)
touch ~/trading_system/killswitch/GRACEFUL
```

### VIX Regime Adaptation

The system automatically scales position sizes based on market volatility:

```
VIX < 15  (LOW)     → 1.2x position size  (favor momentum)
VIX 15-25 (NORMAL)  → 1.0x baseline
VIX 25-35 (HIGH)    → 0.7x reduction
VIX > 35  (EXTREME) → 0.4x defensive mode
```

Regime changes are logged to `performance.db:regime_log` for analysis.

### Error Handling & Recovery

All errors flow through a centralized logging system that persists to database:

```python
# Every ERROR/WARNING/CRITICAL is captured
performance.db:error_log
├─ timestamp
├─ level (ERROR/WARNING/CRITICAL)
├─ logger_name
├─ message
├─ source_file, line_number
├─ exception_type, exception_traceback
├─ component
├─ is_resolved, resolved_at, resolved_by
```

The dashboard displays unresolved errors in real-time, enabling rapid diagnosis.

### Signal Tracking & Audit Trail

Every trading signal is tracked through its complete lifecycle:

```
Signal Generated → Stored in trades.db:signals
       │
       ▼
Signal Submitted → Order created in trades.db:orders
       │
       ▼
Order Filled → Execution recorded in trades.db:executions
       │
       ▼
Position Tracked → Updated in trades.db:positions
       │
       ▼
Trade Closed → Finalized in trades.db:trades with P&L
```

This creates a complete audit trail for every trade, enabling forensic analysis of what went right or wrong.

### Checkpoint & Recovery

The system is designed to survive:
- Pi reboots (planned or unplanned)
- Power failures
- Process crashes
- Out-of-memory conditions

Recovery is enabled by:
1. **Database persistence**: All state lives in SQLite, not memory
2. **Idempotent operations**: Running a task twice produces the same result
3. **Phase-based orchestration**: Knows which phase to resume
4. **GA checkpointing**: Evolution continues from last saved generation

---

## Part IV: Current Strategy Portfolio

### Hand-Coded Strategies (9)

These strategies serve as baselines and generate signals while the GA discovers new approaches:

| Strategy | Type | Research Sharpe | Live Sharpe | Allocation | Status |
|----------|------|-----------------|-------------|------------|--------|
| Vol-Managed Momentum V2 | Momentum | 1.70 | 0.55 | 10% | Active |
| Mean Reversion (Bollinger) | Mean Reversion | 1.44 | 0.30 | 30% | Active |
| Gap-Fill | Intraday | 2.38 | N/A | 10% | Active |
| Pairs Trading | Mean Reversion | 2.30 | N/A | 0% | Disabled (cash) |
| Relative Volume Breakout | Breakout | 2.81 | 0.06 | 25% | Marginal |
| Quality Small-Cap Value | Value | 1.50 | 0.08 | 5% | Weak |
| Factor Momentum | Momentum | 0.84 | 0.42 | 5% | Moderate |
| VIX Regime Rotation | Regime | 0.73 | N/A | 10% | Active |
| Sector Rotation | Tactical | 0.73 | 0.30 | 0% | Disabled |

**Key Insight:** The gap between research Sharpe and live Sharpe is real. This is why we run paper trading before committing capital—to understand how execution, slippage, and regime changes affect performance.

### Regime Coverage

The strategies are designed to perform across different market conditions:

```
             LOW VIX        NORMAL VIX      HIGH VIX       EXTREME VIX
            (<15)          (15-25)         (25-35)        (>35)
            ────────────────────────────────────────────────────────
Momentum    ████████████   ██████████      ████            ██
Mean Rev    ██████         ████████████    ████████████    ████
Pairs       ████████       ██████████      ██████          ████
Breakout    ██████████     ████████        ██              ─
VIX Regime  ██████         ████████        ████████████    ████████████
```

### GA-Discovered Strategies

The genetic programming engine has produced candidates that are currently in the validation pipeline. Once promoted, they'll join the portfolio as "discovered" strategies, complementing the hand-coded ones.

---

## Part V: R&D Roadmap

### Current Phase: Foundation Validation (Months 1-3)

**Objective:** Prove the system is stable, accurate, and improvable.

**Key Metrics:**
- System uptime >99% during market hours
- No unexpected position entries or exits
- GA produces measurable improvements each week
- All circuit breakers tested and functional
- Error rate <1 per trading day

**Milestones:**
1. [ ] 30 consecutive trading days without intervention
2. [ ] GA discovers and promotes at least 1 new strategy
3. [ ] Paper trading P&L matches backtest within 20%
4. [ ] Complete stress test of all circuit breakers
5. [ ] Dashboard displays accurate real-time state

### Future Phase: Performance Optimization (Months 3-6)

**Objective:** Improve returns while maintaining stability.

**Focus Areas:**
- Tune GA parameters based on empirical results
- Fix pairs trading strategy
- Improve slippage modeling
- Add more granular regime detection
- Expand symbol universe carefully

### Distant Future: Capital Allocation (Month 6+)

Only after the above phases are complete, with documented evidence of:
- Positive paper trading P&L
- Stable system operation
- Understood edge cases
- Validated risk management

---

## Part VI: Key Files Reference

### Entry Points
- `daily_orchestrator.py` - Master controller (3000+ lines)
- `run_nightly_research.py` - GA evolution runner (2500+ lines)
- `config.py` - Central configuration (all parameters)

### Core Modules
- `strategies/` - 9 trading strategies
- `execution/` - Order management, position sizing, circuit breakers
- `research/discovery/` - GA/GP engine (20 modules)
- `data/` - Fetchers, indicators, storage
- `observability/` - Logging, dashboard

### Databases
- `db/trades.db` - Execution records
- `db/performance.db` - Analytics
- `db/research.db` - GA evolution
- `db/pairs.db` - Pairs trading

### Documentation
- `docs/ARCHITECTURE.md` - System architecture
- `docs/AUTONOMOUS_RESEARCH_ENGINE.md` - GA details
- `docs/STRATEGY_PORTFOLIO_OVERVIEW.md` - Strategy analysis

---

## Appendix: Quick Reference Commands

```bash
# System status
python daily_orchestrator.py --status

# Run single phase
python daily_orchestrator.py --once

# Start nightly research
python run_nightly_research.py --loop

# Emergency stop
touch ~/trading_system/killswitch/HALT

# View logs
journalctl -u trading-orchestrator -f

# Dashboard
http://<pi-ip>:8050

# Database inspection
sqlite3 db/trades.db ".tables"
sqlite3 db/performance.db "SELECT * FROM error_log ORDER BY timestamp DESC LIMIT 10"
sqlite3 db/research.db "SELECT strategy, generation, best_fitness FROM ga_history ORDER BY run_date DESC LIMIT 10"
```

---

*This document reflects the system state as of January 4, 2026. The focus remains on research and development—real capital deployment is not on the immediate horizon.*
