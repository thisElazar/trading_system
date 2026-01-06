# Genetic Algorithm Module

This module provides evolutionary optimization for trading strategies.

## Relationship with `research/discovery/`

Both modules serve evolutionary strategy development but with different approaches:

- **`genetic/`** (this module): Parameter optimization for existing strategies
  - Optimizes parameters within defined ranges
  - Uses market period testing for regime robustness
  - Provides adaptive strategy management

- **`discovery/`**: Novel strategy discovery via genetic programming (GP)
  - Evolves entirely new trading rules
  - Uses GP trees for entry/exit logic
  - Island model for parallel evolution
  - MAP-Elites for behavioral diversity

## Key Components

### Core Optimization
- `optimizer.py` - Basic genetic algorithm
- `persistent_optimizer.py` - GA with database persistence
- `adaptive_optimizer.py` - Multi-scale regime-aware optimization

### Market Periods (shared with discovery)
- `market_periods.py` - Curated historical periods (COVID, 2008, bull runs)
- `regime_matching.py` - Match current conditions to historical
- `rapid_backtester.py` - Ultra-fast period testing

### Strategy Representation
- `strategy_genome.py` - Gene-based strategy encoding
- `adaptive_strategy_manager.py` - Real-time portfolio rebalancing

## Usage

```python
from research.genetic import AdaptiveGAOptimizer, MarketPeriodLibrary

# Optimize strategy parameters across market regimes
optimizer = AdaptiveGAOptimizer()
best_params = optimizer.optimize(strategy, data)
```

For novel strategy discovery, use `research/discovery/overnight_runner.py`.
