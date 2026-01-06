# Algorithmic Trading System

A research-backed algorithmic trading system designed for autonomous operation on Raspberry Pi 5.

## Overview

This system implements multiple trading strategies based on academic research, with proper walk-forward validation to prevent overfitting. The design prioritizes:

1. **Research-first approach**: Every strategy has documented academic backing with expected Sharpe ratios
2. **Risk management**: Position sizing, stop losses, and regime adaptation
3. **Transaction cost awareness**: Strategies are only deployed where costs don't eliminate the edge
4. **Walk-forward validation**: No peeking at future data during backtests

## Strategies (Tiered Implementation)

### Tier 1 - Core
| Strategy | Research Basis | Expected Sharpe | Status |
|----------|---------------|-----------------|--------|
| Vol-Managed Momentum | Barroso & Santa-Clara (2015) | 1.7 | ✅ Implemented |
| VIX Regime Rotation | Cross-sectional VIX research | 0.73 | ✅ Implemented |
| Gap-Fill Mean Reversion | S&P 500 gap study | 2.38 | ✅ Implemented |

### Tier 2 - Alpha
| Strategy | Research Basis | Expected Sharpe | Status |
|----------|---------------|-----------------|--------|
| Within-Industry Pairs | Federal Reserve research | 2.3-2.9 | ✅ Implemented (36 pairs) |
| Relative Volume Breakout | Stocks in Play study | 2.81 | ✅ Implemented |

### Tier 3 - Supplementary
| Strategy | Research Basis | Expected Sharpe | Status |
|----------|---------------|-----------------|--------|
| Quality Small-Cap Value | Fama-French + AQR | 0.8 | ✅ Implemented |
| Factor Momentum | Ehsani & Linnainmaa | 0.84 | ✅ Implemented |
| Mean Reversion | Statistical reversion | 1.0+ | ✅ Implemented |
| Sector Rotation | Sector momentum | 0.5+ | ✅ Implemented |

## Quick Start

### 1. Installation

```bash
# Clone/extract to your SSD
cd /path/to/your/ssd
unzip trading_system.zip

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.template .env

# Edit .env with your Alpaca API keys
# Get free paper trading keys at: https://alpaca.markets
```

### 3. Initialize Workspace

```bash
python scripts/init_workspace.py
```

This will:
- Create all required directories
- Initialize SQLite databases
- Create reference data files
- Validate API credentials
- Download initial historical data

### 4. Run Backtest

```bash
# Test the vol-managed momentum strategy
python strategies/vol_managed_momentum.py

# Run full backtest
python research/backtester.py
```

## Project Structure

```
trading_system/
├── config.py                 # Central configuration (EDIT DATA_ROOT HERE)
├── requirements.txt
├── .env.template
│
├── data/
│   ├── fetchers/            # Data download modules
│   │   ├── daily_bars.py    # Daily OHLCV data
│   │   └── vix.py           # VIX data and regime detection
│   ├── indicators/          # Technical indicators
│   │   └── technical.py     # All standard indicators
│   └── storage/             # Database management
│       └── db_manager.py    # SQLite operations
│
├── strategies/
│   ├── base.py              # Abstract strategy class
│   ├── vol_managed_momentum.py
│   └── vix_regime_rotation.py
│
├── execution/
│   ├── position_sizer.py    # Position sizing methods
│   └── order_executor.py    # Alpaca order execution
│
├── research/
│   └── backtester.py        # Walk-forward backtesting
│
├── observability/
│   └── logger.py            # Logging configuration
│
└── scripts/
    └── init_workspace.py    # First-run setup
```

## Data Storage

After initialization, data is stored in:

```
{DATA_ROOT}/
├── data/
│   ├── historical/daily/    # Parquet files per symbol
│   ├── historical/vix/      # VIX data
│   └── reference/           # Universe definitions (JSON)
├── db/
│   ├── trades.db           # Signals, trades, positions
│   ├── performance.db      # Strategy metrics
│   ├── research.db         # Backtest results
│   └── pairs.db            # Pairs trading data
├── research/               # Backtest outputs
└── logs/                   # Application logs
```

## Configuration

All configuration is in `config.py`. Key settings:

```python
# Capital & Risk
TOTAL_CAPITAL = 97_000
RISK_PER_TRADE = 0.02      # 2% max risk per position
MAX_POSITIONS = 10

# VIX Regimes
VIX_REGIMES = {
    "low": 15,      # VIX < 15: favor momentum
    "normal": 25,   # VIX 15-25: balanced
    "high": 40,     # VIX > 25: favor mean reversion
}

# Strategy Validation Thresholds
VALIDATION = {
    "vol_managed_momentum": {"min_sharpe": 1.0, "research_sharpe": 1.7},
    "vix_regime_rotation": {"min_sharpe": 0.4, "research_sharpe": 0.73},
}
```

## Migrating to Raspberry Pi

When ready to deploy to Pi:

1. Copy entire `trading_system/` folder to Pi's NVMe
2. Edit `config.py`:
   ```python
   # Change this line to point to Pi's NVMe mount
   DATA_ROOT = Path("/mnt/nvme/trading_system")
   ```
3. Run `python scripts/verify_migration.py` (coming in Phase 6)
4. Set up systemd service for auto-start

## Development Phases

- [x] **Phase 0**: Foundation (directories, databases, fetchers)
- [x] **Phase 1**: Core strategies + backtesting
- [x] **Phase 2**: Paper trading infrastructure (Alpaca)
- [x] **Phase 3**: Intraday capability (gap-fill)
- [x] **Phase 4**: Advanced strategies (pairs, volume breakout)
- [x] **Phase 5**: Research pipeline (overnight GA + GP discovery)
- [ ] **Phase 6**: Pi migration
- [ ] **Phase 7**: Hardening (monitoring, alerts)

## Autonomous Strategy Discovery

The system includes a genetic programming (GP) engine for autonomous strategy discovery:

```bash
# Validate the discovery system
python -m research.discovery.overnight_runner --validate

# Run evolution for 10 generations
python -m research.discovery.overnight_runner --generations 10

# Run overnight (8 hours)
python -m research.discovery.overnight_runner --hours 8
```

Features:
- **Checkpoint persistence**: Resumes from where it left off
- **Pareto optimization**: Multi-objective (return, risk, stability)
- **Novelty search**: Prevents convergence to local optima
- **Graceful shutdown**: Saves state on SIGTERM/SIGINT

See `docs/AUTONOMOUS_RESEARCH_ENGINE.md` for details.

## System Verification

Run the full test suite:

```bash
# Infrastructure tests (7 components)
python scripts/test_research_tools.py

# Evolution validation (5 tests)
python -m research.discovery.overnight_runner --validate
```

Expected output: All tests passing

## Key Principles

### From Research

1. **Strategy-ticker matching matters**: Mean reversion works on large-caps, not small-caps (transaction costs)
2. **Vol-managed momentum doubles Sharpe**: Scale inversely to realized volatility
3. **VIX regime adaptation adds 2-4% annually**: Different factors win in different regimes
4. **Within-industry pairs outperform cross-industry**: +0.82%/month vs -0.30%/month

### Implementation

1. **Walk-forward validation**: Never peek at future data
2. **Transaction costs first**: Model costs before celebrating backtests
3. **Auto-disable failing strategies**: Below 35% win rate after 20 trades
4. **Realistic expectations**: Expect 30-50% degradation from backtest to live

## API Reference

### Fetching Data

```python
from data.fetchers.daily_bars import DailyBarsFetcher

fetcher = DailyBarsFetcher()
df = fetcher.fetch_symbol("AAPL")
```

### Adding Indicators

```python
from data.indicators.technical import add_all_indicators

df = add_all_indicators(df)
# Adds: bb_upper, bb_lower, rsi, macd, atr, etc.
```

### Running Strategy

```python
from strategies.vol_managed_momentum import VolManagedMomentumStrategy

strategy = VolManagedMomentumStrategy()
signals = strategy.generate_signals(data, current_positions, vix_regime='normal')
```

### Backtesting

```python
from research.backtester import Backtester

backtester = Backtester(initial_capital=100000)
result = backtester.run(strategy, data)
print(f"Sharpe: {result.sharpe_ratio:.2f}")
```

### Position Sizing

```python
from execution.position_sizer import PositionSizer

sizer = PositionSizer(total_capital=100000)
size = sizer.calculate_atr_based("AAPL", price=175, atr=3.5)
print(f"Shares: {size.shares}, Risk: ${size.risk_amount:.2f}")
```

## License

Private/Personal Use

## Acknowledgments

Research basis from:
- Barroso & Santa-Clara (2015) - Momentum has its moments
- Jegadeesh & Titman (1993) - Returns to buying winners and selling losers
- Gatev, Goetzmann & Rouwenhorst (2006) - Pairs trading
- Federal Reserve research on within-industry mean reversion
