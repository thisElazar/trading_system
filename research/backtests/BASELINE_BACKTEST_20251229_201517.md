# Baseline Backtest Report

**Run ID:** `20251229_201517`
**Date:** 2025-12-29 20:16:17
**Mode:** Full
**Symbols:** 200

## Results Summary

| Strategy | Sharpe | Return | Max DD | Trades | Win Rate | Status |
|----------|--------|--------|--------|--------|----------|--------|
| vol_managed_momentum | - | - | - | - | - | ❌ ERROR |
| quality_small_cap_value | - | - | - | - | - | ❌ ERROR |
| factor_momentum | - | - | - | - | - | ❌ ERROR |
| pairs_trading | - | - | - | - | - | ❌ ERROR |
| relative_volume_breakout | - | - | - | - | - | ❌ ERROR |
| vix_regime_rotation | - | - | - | - | - | ❌ ERROR |
| sector_rotation | - | - | - | - | - | ❌ ERROR |
| mean_reversion | - | - | - | - | - | ❌ ERROR |

## Errors

### vol_managed_momentum
```
'BacktestResult' object has no attribute 'total_return_pct'
```

### quality_small_cap_value
```
Signal.__init__() missing 1 required positional argument: 'strategy'
```

### factor_momentum
```
Signal.__init__() missing 1 required positional argument: 'strategy'
```

### pairs_trading
```
PairsTradingStrategy.generate_signals() takes from 1 to 2 positional arguments but 4 were given
```

### relative_volume_breakout
```
RelativeVolumeBreakout.generate_signals() takes from 1 to 3 positional arguments but 4 were given
```

### vix_regime_rotation
```
'BacktestResult' object has no attribute 'total_return_pct'
```

### sector_rotation
```
'BacktestResult' object has no attribute 'total_return_pct'
```

### mean_reversion
```
'BacktestResult' object has no attribute 'total_return_pct'
```

## Rankings

### By Sharpe Ratio


## GA Readiness

**Working strategies:** 0/8

❌ Failed to run:
  - vol_managed_momentum
  - quality_small_cap_value
  - factor_momentum
  - pairs_trading
  - relative_volume_breakout
  - vix_regime_rotation
  - sector_rotation
  - mean_reversion
