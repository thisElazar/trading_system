"""
Trading System Configuration
============================
THE ONLY PLACE PATHS AND CORE SETTINGS ARE DEFINED.

When migrating to Pi:
1. Change DATA_ROOT to point to NVMe mount
2. Run: python scripts/verify_migration.py
"""

from pathlib import Path
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Look for .env in the same directory as config.py
    _env_path = Path(__file__).parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
    else:
        # Also check parent directories
        load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, rely on system env vars

# ============================================================================
# ENVIRONMENT DETECTION
# ============================================================================

# Check for environment variable override (useful for testing)
_env_root = os.environ.get("TRADING_SYSTEM_ROOT")

if _env_root:
    DATA_ROOT = Path(_env_root)
elif Path("/Volumes/stashhead/trading_system").exists():
    # NVMe (primary - Pi deployment target)
    DATA_ROOT = Path("/Volumes/stashhead/trading_system")
elif Path("/Volumes/HotStorage/Pi/TradeBot/trading_system").exists():
    # Your Mac SSD (legacy)
    DATA_ROOT = Path("/Volumes/HotStorage/Pi/TradeBot/trading_system")
elif Path("D:/trading_system").exists():
    # Windows SSD
    DATA_ROOT = Path("D:/trading_system")
elif Path("/mnt/nvme/trading_system").exists():
    # Pi with NVMe
    DATA_ROOT = Path("/mnt/nvme/trading_system")
elif Path("/Volumes/TradingSSD/trading_system").exists():
    # macOS external SSD (alternate)
    DATA_ROOT = Path("/Volumes/TradingSSD/trading_system")
else:
    # Fallback to current directory (development)
    DATA_ROOT = Path(__file__).parent.resolve()

# ============================================================================
# DIRECTORY STRUCTURE
# ============================================================================

DIRS = {
    # Data storage
    "data_root":     DATA_ROOT / "data",
    "historical":    DATA_ROOT / "data" / "historical",
    "daily":         DATA_ROOT / "data" / "historical" / "daily",           # Alpaca daily (legacy)
    "daily_yahoo":   DATA_ROOT / "data" / "historical" / "daily_yahoo",     # Yahoo daily (extended history)
    "intraday":      DATA_ROOT / "data" / "historical" / "intraday",        # Alpaca intraday (legacy)
    "intraday_1min": DATA_ROOT / "data" / "historical" / "intraday_1min",   # Alpaca 1-min (full history)
    "vix":           DATA_ROOT / "data" / "historical" / "vix",
    "fundamentals":  DATA_ROOT / "data" / "fundamentals",
    "reference":     DATA_ROOT / "data" / "reference",
    
    # Databases
    "db":            DATA_ROOT / "db",
    
    # Research outputs
    "research":      DATA_ROOT / "research",
    "backtests":     DATA_ROOT / "research" / "backtests",
    "optimization":  DATA_ROOT / "research" / "optimization",
    "discovery":     DATA_ROOT / "research" / "discovery",
    "candidates":    DATA_ROOT / "research" / "discovery" / "candidates",
    
    # Logs and temp
    "logs":          DATA_ROOT / "logs",
    "temp":          DATA_ROOT / "temp",
    "cache":         DATA_ROOT / "temp" / "cache",

    # Circuit breaker
    "killswitch":    DATA_ROOT / "killswitch",
}

DATABASES = {
    "trades":            DIRS["db"] / "trades.db",
    "performance":       DIRS["db"] / "performance.db",
    "research":          DIRS["db"] / "research.db",
    "pairs":             DIRS["db"] / "pairs.db",
    # Consolidated from data/
    "promotion":         DIRS["db"] / "promotion_pipeline.db",
    "portfolio_fitness": DIRS["db"] / "portfolio_fitness.db",
    "volatility":        DIRS["db"] / "volatility.db",
    "signal_scores":     DIRS["db"] / "signal_scores.db",
}

# ============================================================================
# API CONFIGURATION
# ============================================================================

# Alpaca - load from environment or .env file
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_DATA_URL = "https://data.alpaca.markets"

# Telegram - for alerts
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ============================================================================
# TRADING PARAMETERS
# ============================================================================

# Capital & Risk
TOTAL_CAPITAL = int(os.environ.get("TRADING_CAPITAL", 97_000))
RISK_PER_TRADE = 0.02          # 2% max risk per position
MAX_POSITION_SIZE = 15_000     # Max $ per position
MAX_POSITIONS = 20             # Max concurrent positions (increased for paper trading research)
CASH_BUFFER_PCT = 0.05         # Keep 5% in cash

# Position Size Limits
MAX_POSITION_PCT = 0.05        # 5% max per position (down from 10%)

# Risk Controls
MAX_DAILY_LOSS_PCT = 0.02      # Pause trading if down 2% in a day
MAX_DRAWDOWN_PCT = 0.15        # Reduce position sizes if drawdown exceeds 15%

# Circuit Breaker Settings
CIRCUIT_BREAKER = {
    "daily_loss_pct": 0.02,           # Halt if down 2% in a day
    "drawdown_pct": 0.15,             # Reduce position sizes by 50% if drawdown exceeds 15%
    "rapid_loss_pct": 0.01,           # 1% loss in window = temporary halt
    "rapid_loss_window_min": 15,      # Window for rapid loss detection (minutes)
    "rapid_loss_pause_min": 30,       # Pause duration after rapid loss (minutes)
    "max_consecutive_losses": 5,      # Pause strategy after N consecutive losses
    "consecutive_loss_pause_hrs": 4,  # Hours to pause after consecutive losses
    "strategy_loss_pct": 0.05,        # Disable strategy at 5% loss of allocation
    "strategy_pause_hrs": 24,         # Hours to pause underperforming strategy
}

# Intraday Exit Monitoring
# Automatically closes positions when they hit take-profit or stop-loss thresholds
INTRADAY_EXIT_CONFIG = {
    "enabled": True,
    "take_profit_pct": 0.10,          # 10% gain triggers take-profit exit
    "stop_loss_pct": 0.08,            # 8% loss triggers stop-loss exit
    "check_interval_seconds": 30,      # How often to check (aligned with MARKET_OPEN phase)
    "use_market_orders": True,         # Use market orders for exits (faster fills)
    "log_checks": False,               # Log every check (noisy, for debugging)
}

# ============================================================================
# INTRADAY DATA UNIVERSE
# ============================================================================
# Symbols for which we download minute-bar data. Used by:
# - Gap-fill strategy (intraday mean reversion)
# - VWAP reversion strategy
# - Opening Range Breakout (ORB) strategy
# - Intraday research and pattern discovery

INTRADAY_UNIVERSE = {
    # Broad market ETFs - highest liquidity, tightest spreads
    "broad_market": ["SPY", "QQQ", "IWM", "DIA"],

    # Sector ETFs - sector-specific gaps and rotations
    "sectors": ["XLF", "XLE", "XLK", "XLV", "XLI", "XLP", "XLU", "XLY", "XLC", "XLB", "XLRE"],

    # Mega-cap stocks - extremely liquid, individual catalysts
    "mega_caps": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "UNH"],

    # Volatility products - for regime detection and hedging research
    "volatility": ["VXX", "UVXY", "VIXY"],

    # Commodities ETFs - macro exposure
    "commodities": ["GLD", "SLV", "USO", "UNG"],

    # Fixed income ETFs - rate sensitivity, risk-off signals
    "bonds": ["TLT", "HYG", "LQD"],

    # International ETFs - global market correlation
    "international": ["EEM", "EFA", "FXI"],

    # Thematic ETFs - high-beta, momentum plays
    "thematic": ["ARKK", "XBI", "SMH", "KWEB"],
}

# Flatten to single list for easy iteration
INTRADAY_SYMBOLS = []
for category in INTRADAY_UNIVERSE.values():
    INTRADAY_SYMBOLS.extend(category)

# Configuration for intraday data management
INTRADAY_DATA_CONFIG = {
    "retention_days": 30,              # Keep 30 days of minute bars
    "refresh_days": 5,                 # Download last 5 days on refresh
    "rate_limit_seconds": 0.25,        # Delay between API calls
    "max_parallel_downloads": 1,       # Sequential for now (rate limits)
}

# Price filters
MIN_STOCK_PRICE = 10
MAX_STOCK_PRICE = 500

# ============================================================================
# ADAPTIVE POSITION SIZING
# ============================================================================

# Adaptive sizing replaces fixed 2% risk with dynamic sizing based on:
# - Strategy performance (Sharpe ratio)
# - VIX regime (market volatility)
# - Current drawdown level
# - Win rate momentum
# - Kelly criterion

ADAPTIVE_SIZING = {
    # Strategy performance scaling
    "sharpe_scale_min": 0.5,        # Minimum scalar when Sharpe < 0
    "sharpe_scale_max": 1.5,        # Maximum scalar when Sharpe > 1
    "sharpe_neutral": 0.5,          # Sharpe ratio for 1.0x scaling
    "sharpe_lookback_days": 60,     # Days to calculate rolling Sharpe

    # VIX regime scaling
    "vix_low_scalar": 1.2,          # Scale up in low VIX (< 15)
    "vix_normal_scalar": 1.0,       # Normal sizing (15-25)
    "vix_high_scalar": 0.7,         # Scale down in high VIX (25-35)
    "vix_extreme_scalar": 0.4,      # Aggressive scale down (> 35)

    # Drawdown scaling (progressive)
    "drawdown_thresholds": [0.05, 0.10, 0.15, 0.20],  # 5%, 10%, 15%, 20%
    "drawdown_scalars": [1.0, 0.8, 0.6, 0.4, 0.25],   # Corresponding scalars

    # Win rate momentum
    "win_rate_lookback_short": 10,  # Recent trades
    "win_rate_lookback_long": 30,   # Baseline trades
    "win_rate_improvement_bonus": 0.2,  # +20% if improving
    "win_rate_decline_penalty": 0.15,   # -15% if declining

    # Kelly criterion
    "kelly_fraction": 0.25,         # Use 25% of full Kelly (conservative)
    "kelly_min_trades": 20,         # Minimum trades before Kelly kicks in

    # Combined scalar limits
    "combined_scalar_min": 0.25,    # Never go below 25% of base size
    "combined_scalar_max": 1.5,     # Never exceed 150% of base size

    # Logging
    "log_decisions": True,          # Log sizing decisions to DB
}

# ============================================================================
# MARKET CAP TIERS
# ============================================================================

MARKET_CAP_TIERS = {
    "mega":  50_000_000_000,   # $50B+
    "large": 10_000_000_000,   # $10-50B
    "mid":   2_000_000_000,    # $2-10B
    "small": 300_000_000,      # $300M-2B
    "micro": 0,                # < $300M (avoid)
}

# Transaction costs by tier (basis points, round-trip)
TRANSACTION_COSTS_BPS = {
    "mega":  30,    # 0.30%
    "large": 50,    # 0.50%
    "mid":   150,   # 1.50%
    "small": 300,   # 3.00%
    "micro": 500,   # 5.00% (don't trade these)
}

# ============================================================================
# VIX REGIME THRESHOLDS
# ============================================================================

VIX_REGIMES = {
    "low":      15,   # VIX < 15: low volatility
    "normal":   25,   # VIX 15-25: normal
    "high":     35,   # VIX 25-35: high volatility (optimized Dec 2025)
    "extreme":  35,   # VIX > 35: crisis mode (lowered from 40 for earlier defense)
}

# ============================================================================
# HMM REGIME DETECTION (GP-010)
# ============================================================================

# HMM-based regime detector learns transitions from historical data
# instead of using hardcoded VIX thresholds. Provides probabilistic
# regime assignment and N-day transition forecasts.
HMM_REGIME_CONFIG = {
    "enabled": True,                    # Enable HMM regime detection
    "n_states": 3,                      # Number of hidden states (regimes)
    "state_names": ["bull", "transition", "crisis"],
    "covariance_type": "full",          # Covariance type: full, diag, spherical
    "lookback_days": 756,               # Training window (3 years)
    "retrain_frequency": "monthly",     # How often to retrain model
    "model_path": str(DIRS["research"] / "models" / "hmm_regime.pkl"),

    # Fallback behavior when HMM unavailable
    "fallback_to_vix": True,           # Use VIX thresholds if HMM fails

    # Integration settings
    "confidence_threshold": 0.7,        # Min confidence to use HMM regime
    "blend_with_vix": True,            # Combine HMM + VIX for robust detection
    "blend_hmm_weight": 0.6,           # Weight for HMM in blended detection

    # GP-015: Regime change confirmation lag
    # Requires stable regime classification for N consecutive days before confirming
    # Prevents whipsaws from noisy single-day regime fluctuations
    "regime_confirmation_days": 2,      # Days of stable regime before confirming change
}

# ============================================================================
# STRATEGY CONFIGURATION
# ============================================================================

# Strategy enable/disable and allocation
# Allocations updated Jan 2026:
#   - mean_reversion: 35% (Sharpe 0.76, BEST performer - absorbed pairs_trading 5%)
#   - relative_volume_breakout: 25% (Sharpe 0.57)
#   - vix_regime_rotation: 10% (Sharpe 0.04 - kept for defensive/crisis insurance)
#   - vol_managed_momentum: 10% (V2 research-aligned)
#   - gap_fill: 10% (Research Sharpe 2.38, intraday)
#   - quality_smallcap_value: 5%
#   - factor_momentum: 5% (Ehsani & Linnainmaa 2022)
#   - pairs_trading: DISABLED (requires shorting - cash account is long-only)
#   - sector_rotation: DISABLED (Sharpe -0.38)
STRATEGIES = {
    "vol_managed_momentum": {
        "enabled": True,   # Enabled - using V2 (research-aligned) from vol_managed_momentum_v2.py
        "tier": 1,
        "allocation_pct": 0.10,
        "max_positions": 10,
        "rebalance_frequency": "monthly",
        "notes": "V2: 12-1 momentum with strategy-level vol scaling (Barroso & Santa-Clara 2015)",
    },
    "mean_reversion": {
        "enabled": True,   # BEST performer - Sharpe 0.76 (Dec 2025 backtest)
        "tier": 1,
        "allocation_pct": 0.35,  # Increased from 30% (absorbed pairs_trading 5%)
        "max_positions": 25,
        "rebalance_frequency": "monthly",
        "notes": "Within-industry short-term reversal (21-day)",
    },
    "vix_regime_rotation": {
        "enabled": True,   # Sharpe 0.04 - kept for defensive/crisis insurance
        "tier": 1,
        "allocation_pct": 0.10,  # Reduced from 15% to fund gap_fill
        "max_positions": 5,
        "rebalance_frequency": "event_driven",
    },
    "gap_fill": {
        "enabled": True,   # ENABLED - Research Sharpe 2.38 (HIGHEST), intraday infrastructure ready
        "tier": 1,
        "allocation_pct": 0.10,  # Reduced from 15% to fund vol_managed_momentum
        "max_positions": 2,
        "rebalance_frequency": "intraday",
        "notes": "Trades SPY/QQQ gap fills 9:31-11:30 AM; requires intraday data",
    },
    "pairs_trading": {
        "enabled": False,  # DISABLED Jan 2026 - requires shorting (margin account), cash account is long-only
        "tier": 1,
        "allocation_pct": 0.00,  # Reallocated to mean_reversion
        "max_positions": 14,  # 7 pairs (2 positions each)
        "rebalance_frequency": "daily",
        "notes": "Traditional pairs trading requires shorting one leg. Not viable with cash account.",
    },
    "relative_volume_breakout": {
        "enabled": True,   # Sharpe 0.57, 164% returns (Dec 2025 backtest)
        "tier": 1,
        "allocation_pct": 0.25,  # Reduced from 30% to fund gap_fill
        "max_positions": 5,
        "rebalance_frequency": "daily",  # 1-day hold
    },
    "sector_rotation": {
        # BUG-003: Re-enabled with GA-optimized params (Sharpe -0.38 -> 1.08)
        "enabled": True,
        "tier": 1,  # Upgraded from tier 2
        "allocation_pct": 0.10,
        "max_positions": 2,  # GA optimal: concentrate in top 2 sectors only
        "rebalance_frequency": "monthly",  # GA optimal: 28-day rebalance
        "notes": "GA-optimized: 105-day momentum, top 2 sectors, monthly rebalance (Sharpe 1.08)",
        # GA parameters (passed to strategy constructor)
        "params": {
            "momentum_period": 105,
            "top_n_sectors": 2,
            "rebalance_days": 28,
        },
    },
    "quality_smallcap_value": {
        "enabled": True,   # Fama-French + AQR Quality-Minus-Junk research (16.38% small-cap value premium)
        "tier": 1,
        "allocation_pct": 0.05,  # Reduced from 10% to fund gap_fill
        "max_positions": 30,
        "rebalance_frequency": "monthly",
        "notes": "Quality screens eliminate junky small-caps; uses price-based proxies for fundamentals",
    },
    "factor_momentum": {
        "enabled": True,   # Ehsani & Linnainmaa 2022 - factor momentum avoids crash risk
        "tier": 1,
        "allocation_pct": 0.05,  # Reduced from 10% to fund vol_managed_momentum
        "max_positions": 10,
        "rebalance_frequency": "monthly",
        "notes": "Trades sector ETFs based on factor momentum (value, growth, defensive, cyclical)",
    },
    "universe_scan": {
        "enabled": True,   # Full universe scanner - runs all enabled strategies on ~2,500 symbols
        "tier": 0,  # Meta-strategy that runs other strategies
        "allocation_pct": 0.0,  # No separate allocation (uses underlying strategy allocations)
        "max_positions": 0,  # Determined by underlying strategies
        "rebalance_frequency": "daily",
        "notes": "Two-phase scanner: Phase 1 scans batches of 128 symbols, Phase 2 validates candidates. ~3 min total.",
    },
}

# ============================================================================
# VALIDATION THRESHOLDS
# ============================================================================

# Backtest validation (strategy must achieve these to go live)
VALIDATION = {
    "vol_managed_momentum": {"min_sharpe": 1.0, "research_sharpe": 1.7},
    "mean_reversion":       {"min_sharpe": 0.4, "research_sharpe": 0.82},  # Fed research: 0.82%/mo within-industry
    "vix_regime_rotation":  {"min_sharpe": 0.4, "research_sharpe": 0.73},
    "gap_fill":             {"min_sharpe": 1.5, "research_sharpe": 2.38},
    "pairs_trading":        {"min_sharpe": 1.5, "research_sharpe": 2.5},
    "relative_volume_breakout": {"min_sharpe": 1.8, "research_sharpe": 2.81},
    "sector_rotation":      {"min_sharpe": 0.5, "research_sharpe": 0.73},
    "quality_smallcap_value": {"min_sharpe": 0.8, "research_sharpe": 1.2},  # Fama-French: 1.0-1.5 expected Sharpe
    "factor_momentum":      {"min_sharpe": 0.6, "research_sharpe": 0.84},  # Ehsani & Linnainmaa 2022
}

# Auto-disable thresholds (paper trading)
MIN_TRADES_FOR_EVAL = 20
MIN_WIN_RATE = 0.35              # Disable below 35% after MIN_TRADES
EMERGENCY_WIN_RATE = 0.20        # Disable immediately below 20% after 10 trades

# ============================================================================
# DATA FETCHING
# ============================================================================

# Historical data parameters
HISTORICAL_YEARS = 10            # Years of daily data (need long history for momentum)
INTRADAY_DAYS = 30               # Days of 1-min data to keep
BATCH_SIZE = 100                 # Symbols per API request

# Universe definitions
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
RUSSELL2000_PROXY = "IWM"        # Use IWM holdings as proxy

# Refresh schedules
WATCHLIST_REFRESH_DAYS = 7
PAIRS_RETEST_DAYS = 7
FUNDAMENTALS_REFRESH_DAYS = 90

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Log rotation
LOG_MAX_BYTES = 10_000_000       # 10 MB
LOG_BACKUP_COUNT = 5

# ============================================================================
# WEEKEND PHASE CONFIGURATION
# ============================================================================

# Weekend phase runs Friday 4 PM → Sunday evening
# All settings can be overridden from dashboard at runtime
WEEKEND_CONFIG = {
    # Research intensity - flexible based on needs
    "research": {
        "generations_default": 10,      # Can be 3-50 depending on focus
        "population_default": 30,       # Smaller (10-15) for focused, larger (50+) for broad
        "discovery_hours": 4.0,         # Max hours for GP discovery
        "discovery_enabled": True,      # Enable GP strategy discovery
        "adaptive_ga_enabled": True,    # Enable regime-matched testing
        "strategies": [],               # Empty = all enabled strategies
    },

    # Data scope - full universe available, strategies choose their focus
    "data": {
        "universe_scope": "full",       # "full" | "watchlist" | "custom"
        "fundamentals_refresh": True,
        "index_refresh": True,
        "pairs_recalc": True,
        "custom_symbols": [],           # For focused runs
    },

    # Maintenance tasks
    "maintenance": {
        "vacuum_databases": True,
        "log_retention_days": 30,
        "backup_enabled": True,
    },

    # Presets for quick configuration (dashboard buttons)
    "presets": {
        "quick": {
            "generations": 5,
            "population": 15,
            "discovery_enabled": False,
            "adaptive_ga_enabled": False,
        },
        "standard": {
            "generations": 10,
            "population": 30,
            "discovery_enabled": True,
            "adaptive_ga_enabled": False,
        },
        "deep": {
            "generations": 25,
            "population": 50,
            "discovery_enabled": True,
            "adaptive_ga_enabled": True,
        },
    },
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ensure_dirs():
    """Create all directories if they don't exist."""
    for name, path in DIRS.items():
        path.mkdir(parents=True, exist_ok=True)
    return True

def get_db_path(name: str) -> Path:
    """Get path to a specific database."""
    if name not in DATABASES:
        raise ValueError(f"Unknown database: {name}. Options: {list(DATABASES.keys())}")
    return DATABASES[name]

def get_dir(name: str) -> Path:
    """Get path to a specific directory."""
    if name not in DIRS:
        raise ValueError(f"Unknown directory: {name}. Options: {list(DIRS.keys())}")
    return DIRS[name]

def get_strategy_config(name: str) -> dict:
    """Get configuration for a specific strategy."""
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Options: {list(STRATEGIES.keys())}")
    return STRATEGIES[name]

def get_enabled_strategies() -> list:
    """Get list of currently enabled strategy names."""
    return [name for name, cfg in STRATEGIES.items() if cfg["enabled"]]

def get_transaction_cost(market_cap: float) -> float:
    """Get transaction cost in decimal (not bps) for a given market cap."""
    for tier, threshold in sorted(MARKET_CAP_TIERS.items(),
                                   key=lambda x: x[1], reverse=True):
        if market_cap >= threshold:
            return TRANSACTION_COSTS_BPS[tier] / 10000
    return TRANSACTION_COSTS_BPS["micro"] / 10000

def get_vix_regime(vix_level: float) -> str:
    """Classify VIX level into regime."""
    if vix_level < VIX_REGIMES["low"]:
        return "low"
    elif vix_level < VIX_REGIMES["normal"]:
        return "normal"
    elif vix_level < VIX_REGIMES["high"]:
        return "high"
    else:
        return "extreme"


# ============================================================================
# VALIDATION ON IMPORT
# ============================================================================

if __name__ == "__main__":
    print(f"Trading System Configuration")
    print(f"=" * 50)
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"Exists: {DATA_ROOT.exists()}")
    print()
    print("API Keys:")
    print(f"  ALPACA_API_KEY: {'✓ Set' if ALPACA_API_KEY else '✗ Not set'}")
    print(f"  ALPACA_SECRET_KEY: {'✓ Set' if ALPACA_SECRET_KEY else '✗ Not set'}")
    print()
    print("Directories:")
    for name, path in DIRS.items():
        status = "✓" if path.exists() else "✗"
        print(f"  {status} {name}: {path}")
    print()
    print("Enabled Strategies:")
    for name in get_enabled_strategies():
        cfg = STRATEGIES[name]
        print(f"  - {name} ({cfg['allocation_pct']*100:.0f}% allocation)")


# ============================================================================
# PERFORMANCE CONFIGURATION (Pi-Optimized)
# ============================================================================
# Centralized performance settings to prevent memory/CPU issues on Pi 5 (4GB)
# These settings are the SINGLE SOURCE OF TRUTH for performance tuning.
#
# History: After multiple crash investigations (Jan 2026), we found:
#   - 10 years of backtest data caused 2.2GB VIRT allocation
#   - 4 parallel workers duplicated data (4x memory)
#   - Sustained 100% CPU triggered watchdog/thermal shutdown
#
# Safe Pi limits discovered through testing:
#   - max_years: 1 (not 10) - prevents 2GB+ memory allocation
#   - max_symbols: 50 (not 2556) - load only what we need
#   - n_workers: 1 (not 4) - no data duplication
#   - parallel: False - sequential evaluation only

import platform

def _is_raspberry_pi() -> bool:
    """Detect if running on Raspberry Pi."""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            return 'Raspberry Pi' in f.read() or 'BCM' in f.read()
    except:
        return False

def _get_total_ram_gb() -> float:
    """Get total system RAM in GB."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024**3)
    except:
        return 4.0  # Assume constrained if we can't check

IS_RASPBERRY_PI = _is_raspberry_pi()
TOTAL_RAM_GB = _get_total_ram_gb()
IS_MEMORY_CONSTRAINED = TOTAL_RAM_GB < 8.0

# Performance profiles
PERFORMANCE_PROFILES = {
    'pi_safe': {
        # Memory management (optimized post-watchdog fix)
        # Increased from 100 to 200 for better generalization (batched eval handles it)
        'max_symbols': 200,
        'max_years': 3,
        'walk_forward': True,  # Validated stable on Pi
        
        # Parallelism (2 workers - prevents OOM on 4GB Pi)
        'parallel_enabled': True,
        'n_workers': 2,
        'use_persistent_pool': True,
        
        # GA settings (deeper exploration)
        'population_size': 20,
        'generations': 3,
        'early_stop_generations': 3,
        
        # CPU management (aggressive - watchdog disabled)
        'nice_level': 10,
        'sleep_between_evals': 0.02,  # 20ms cooldown
        
        # Discovery/Adaptive (reduced)
        'discovery_population': 20,
        'discovery_hours': 1.0,
        'adaptive_population': 20,
        'adaptive_islands': 2,
        'concurrent_strategies': 1,  # Sequential (parallel available)
    },
    'pi_balanced': {
        'walk_forward': True,  # Enable for more thorough validation
        # Memory management
        'max_symbols': 75,
        'max_years': 2,
        
        # Parallelism (2 workers - prevents OOM on 4GB Pi)
        'parallel_enabled': True,
        'n_workers': 2,
        'use_persistent_pool': True,
        
        # GA settings (moderate)
        'population_size': 10,
        'generations': 5,
        'early_stop_generations': 3,
        
        # CPU management
        'nice_level': 5,
        'sleep_between_evals': 0.02,
        
        # Discovery/Adaptive
        'discovery_population': 30,
        'discovery_hours': 2.0,
        'adaptive_population': 30,
        'adaptive_islands': 2,
        'concurrent_strategies': 1,  # Sequential (parallel available)
    },
    'workstation': {
        'walk_forward': True,  # Full validation on workstation
        # Memory management (generous)
        'max_symbols': 200,
        'max_years': 20,
        
        # Parallelism (enabled)
        'parallel_enabled': True,
        'n_workers': 4,
        'use_persistent_pool': True,
        
        # GA settings (full)
        'population_size': 30,
        'generations': 15,
        'early_stop_generations': 5,
        
        # CPU management
        'nice_level': 0,
        'sleep_between_evals': 0,
        
        # Discovery/Adaptive
        'discovery_population': 50,
        'discovery_hours': 4.0,
        'adaptive_population': 60,
        'adaptive_islands': 4,
    },
}

# Auto-select profile based on hardware
def _auto_select_profile() -> str:
    """Automatically select performance profile based on hardware."""
    if IS_RASPBERRY_PI:
        return 'pi_safe'
    elif IS_MEMORY_CONSTRAINED:
        return 'pi_balanced'
    else:
        return 'workstation'

# Allow override via environment variable
_profile_name = os.environ.get('TRADING_PERF_PROFILE', _auto_select_profile())
if _profile_name not in PERFORMANCE_PROFILES:
    _profile_name = _auto_select_profile()

# THE SINGLE SOURCE OF TRUTH
PERF = PERFORMANCE_PROFILES[_profile_name]
PERF_PROFILE_NAME = _profile_name

def get_perf(key: str, default=None):
    """Get a performance setting with optional default."""
    return PERF.get(key, default)

def set_perf_profile(profile_name: str):
    """Switch to a different performance profile at runtime."""
    global PERF, PERF_PROFILE_NAME
    if profile_name not in PERFORMANCE_PROFILES:
        raise ValueError(f"Unknown profile: {profile_name}. Options: {list(PERFORMANCE_PROFILES.keys())}")
    PERF = PERFORMANCE_PROFILES[profile_name]
    PERF_PROFILE_NAME = profile_name


# ============================================================================
# FEATURE FLAGS
# ============================================================================

# TaskScheduler (Option C) - Time-aware task scheduling with priorities
# Set to True to enable new scheduler, False for legacy phase-based execution
USE_TASK_SCHEDULER = os.environ.get('USE_TASK_SCHEDULER', 'false').lower() == 'true'

# When True, logs detailed scheduler decisions (ready tasks, blocking reasons)
TASK_SCHEDULER_DEBUG = os.environ.get('TASK_SCHEDULER_DEBUG', 'false').lower() == 'true'

# Unified Scheduler - Makes TaskScheduler the single authority for market status
# and operating mode. When True:
# - TaskScheduler determines operating mode (TRADING/RESEARCH/PREP)
# - Weekend/overnight/holiday handling uses dynamic budget-based task selection
# - get_current_phase() delegates to TaskScheduler for mode-aware decisions
#
# Migration strategy:
# - Week 1: Deploy with 'false' to verify no regressions
# - Week 2: Enable Friday evening for weekend testing
# - Week 3: Enable for overnight testing
# - Week 4: Full enablement
#
# Rollback: export USE_UNIFIED_SCHEDULER=false
USE_UNIFIED_SCHEDULER = os.environ.get('USE_UNIFIED_SCHEDULER', 'false').lower() == 'true'

