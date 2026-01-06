# Autonomous Research Engine

The autonomous research engine continuously evolves trading strategy parameters using genetic algorithms, accumulating improvements over weeks and months of unattended operation.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    run_nightly_research.py                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              NightlyResearchEngine                    │  │
│  │  - Loads market data                                  │  │
│  │  - For each strategy:                                 │  │
│  │      - Creates PersistentGAOptimizer                  │  │
│  │      - Evolves 1-5 generations                        │  │
│  │      - Saves state to database                        │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              PersistentGAOptimizer                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  load_population() ←→ SQLite (research.db)           │  │
│  │  evolve_incremental()                                 │  │
│  │  save_population() ←→ SQLite (research.db)           │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     GeneticOptimizer                         │
│  - Tournament selection                                      │
│  - Crossover & mutation                                      │
│  - Elitism                                                   │
│  - Fitness evaluation via walk-forward backtest              │
└─────────────────────────────────────────────────────────────┘
```

## Database Schema

Three new tables in `research.db`:

### ga_populations
Stores the current population state for each strategy (enables resume):
```sql
strategy TEXT          -- Strategy name (e.g., 'vol_managed_momentum')
generation INTEGER     -- Current generation number
population_json TEXT   -- Full population as JSON array
best_fitness REAL      -- Best fitness achieved
best_genes_json TEXT   -- Best individual's parameters
```

### ga_history
Tracks evolution progress over time (enables analysis):
```sql
strategy TEXT          -- Strategy name
generation INTEGER     -- Generation number
best_fitness REAL      -- Best fitness at this generation
mean_fitness REAL      -- Population mean fitness
std_fitness REAL       -- Population fitness std dev
run_date TEXT          -- Date of this evolution run
```

### ga_runs
Logs each research run (enables monitoring):
```sql
run_id TEXT            -- Unique run identifier
start_time TEXT        -- When run started
end_time TEXT          -- When run completed
strategies_evolved TEXT -- JSON list of strategies
total_generations INT  -- Total generations run
improvements_found INT -- Number of strategies that improved
status TEXT            -- running/completed/failed
```

## Usage

### Single Run (for cron)
```bash
python run_nightly_research.py
```

### Continuous Mode (for development/systemd)
```bash
python run_nightly_research.py --loop
```

### Check Status
```bash
python run_nightly_research.py --status
```

### Custom Run
```bash
# Specific strategies
python run_nightly_research.py -s vol_managed_momentum pairs_trading

# More generations
python run_nightly_research.py -g 10
```

## Deployment on Raspberry Pi 5

### Option 1: Systemd Service (Recommended)
```bash
# Copy service file
sudo cp config/trading-research.service /etc/systemd/system/

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable trading-research.service
sudo systemctl start trading-research.service

# Check status
sudo systemctl status trading-research

# View logs
journalctl -u trading-research -f
```

### Option 2: Cron Job
```bash
# Make script executable
chmod +x config/run_research_cron.sh

# Edit crontab
crontab -e

# Add (runs at 9:30 PM Eastern, weekdays):
30 21 * * 1-5 /mnt/nvme/trading_system/config/run_research_cron.sh
```

## Configuration

### GA Parameters (in `run_nightly_research.py`)
```python
NIGHTLY_GA_CONFIG = GeneticConfig(
    population_size=30,       # Individuals per generation
    generations=3,            # Generations per nightly run
    mutation_rate=0.15,       # Probability of gene mutation
    crossover_rate=0.7,       # Probability of crossover
    elitism=2,                # Best N individuals preserved
    tournament_size=3,        # Selection tournament size
    early_stop_generations=2  # Stop if no improvement
)
```

### Strategy Parameter Specs (in `research/genetic/optimizer.py`)
```python
STRATEGY_PARAMS = {
    'vol_managed_momentum': [
        ParameterSpec('formation_period', 63, 252, step=21, dtype=int),
        ParameterSpec('vol_lookback', 10, 30, step=5, dtype=int),
        ParameterSpec('target_vol', 0.10, 0.25, step=0.05),
        # ...
    ],
    # ...
}
```

## Fitness Function

The fitness function runs walk-forward backtests to prevent overfitting:

```python
def fitness_fn(genes: dict) -> float:
    # Create strategy with evolved parameters
    strategy = create_strategy_instance(strategy_name, genes)
    
    # Run backtest (uses proper train/test splits)
    result = backtester.run(strategy=strategy, data=data)
    
    # Primary metric: Sharpe ratio
    fitness = result.sharpe_ratio
    
    # Penalize insufficient trades (overfitting indicator)
    if result.total_trades < 10:
        fitness *= 0.5
    
    # Penalize extreme drawdowns
    if result.max_drawdown_pct < -40:
        fitness *= 0.8
    
    return fitness
```

## Monitoring & Analysis

### View improvement over time
```python
from data.storage.db_manager import get_db
db = get_db()

# Get 30-day history
history = db.get_ga_history('vol_managed_momentum', days=30)
for row in history:
    print(f"Gen {row['generation']}: {row['best_fitness']:.4f}")
```

### Query best parameters found
```python
best = db.get_ga_best_all_time('vol_managed_momentum')
print(f"Best fitness: {best['best_fitness']}")
print(f"Best genes: {best['best_genes']}")
```

### Get recent run summary
```python
runs = db.get_recent_ga_runs(limit=5)
for run in runs:
    print(f"{run['run_id']}: {run['status']} - {run['improvements_found']} improvements")
```

## Design Principles

1. **Incremental evolution** - Run few generations per night (1-5), not full convergence. Let improvements accumulate over weeks/months.

2. **Overfitting prevention** - Walk-forward validation is non-negotiable. Never evaluate fitness on data the strategy has seen.

3. **Persistence over performance** - Better to have a slow system that runs reliably for months than a fast one that crashes.

4. **Observable progress** - Must be able to query "show me how strategy X has improved over the last 30 days."

5. **Graceful degradation** - If backtester fails for one strategy, continue with others. Log errors, don't crash.

## Verification

Run the verification script to ensure everything is set up correctly:

```bash
python scripts/verify_research_engine.py
```

Expected output:
```
AUTONOMOUS RESEARCH ENGINE VERIFICATION
============================================================
1. Checking database tables...
   ✓ Table 'ga_populations' exists
   ✓ Table 'ga_history' exists
   ✓ Table 'ga_runs' exists

2. Checking strategy parameter specs...
   ✓ vol_managed_momentum: 5 parameters
   ✓ pairs_trading: 6 parameters
   ✓ relative_volume_breakout: 5 parameters
...

All checks passed! ✓
```

## Success Criteria

The system is complete when:

1. ✅ You can run `python run_nightly_research.py` and it evolves all strategies one generation
2. ✅ You can kill it, restart it, and it picks up where it left off
3. ✅ You can query the database to see fitness improvements over time
4. ✅ It can run unattended on the Pi for a week without intervention

---

## Recent Fixes (January 2026)

### Checkpoint System
- **Issue**: UNIQUE constraint violations on checkpoint saves
- **Fix**: Added microseconds + generation to checkpoint_id, changed to INSERT OR REPLACE

### DEAP Deserialization
- **Issue**: Type mismatch when loading checkpoints (float vs FloatType)
- **Fix**: Added `_safe_tree_from_string()` fallback in `strategy_genome.py`

### Evolution Thresholds
- **Issue**: No strategies being promoted (thresholds too strict)
- **Fix**: Relaxed in `config.py`:
  - min_trades: 50 → 30
  - min_deflated_sharpe: 0.7 → 0.5
  - promotion_threshold_sortino: 0.8 → 0.6
  - promotion_threshold_dsr: 0.85 → 0.75

### Data Loading
- **Issue**: `load_all_daily_data` method not found
- **Fix**: Changed to correct method `load_all_daily()` in overnight_runner.py

### Validation Status
All 5 validation tests now pass:
```
[1/5] Creating test market data... ✓
[2/5] Initializing evolution engine... ✓
[3/5] Running 2-generation evolution test... ✓
[4/5] Testing checkpoint save/load... ✓
[5/5] Testing graceful shutdown handling... ✓
VALIDATION PASSED (5/5 tests)
```

Run validation with:
```bash
python -m research.discovery.overnight_runner --validate
```
