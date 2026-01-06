# Baseline Backtest Report

**Run ID:** `20251229_223700`
**Date:** 2025-12-29 22:46:26
**Mode:** Full
**Symbols:** 201

## Results Summary

| Strategy | Sharpe | Return | Max DD | Trades | Win Rate | Status |
|----------|--------|--------|--------|--------|----------|--------|
| vol_managed_momentum | 0.17 | 13.2% | -35.9% | 3310 | 48.9% | ⚠️ |
| quality_small_cap_value | 0.21 | 20.3% | -40.8% | 2201 | 42.8% | ⚠️ |
| factor_momentum | 0.21 | 16.0% | -22.3% | 195 | 47.7% | ⚠️ |
| pairs_trading | -0.30 | -12.2% | -16.9% | 1088 | 44.1% | ❌ |
| relative_volume_breakout | 0.72 | 178.3% | -24.1% | 1943 | 74.3% | ✅ |
| vix_regime_rotation | 0.40 | 41.9% | -19.7% | 674 | 51.9% | ⚠️ |
| sector_rotation | -0.38 | -2.6% | -3.6% | 9 | 22.2% | ❌ |
| mean_reversion | 0.75 | 68.5% | -14.6% | 3381 | 58.9% | ✅ |

## Rankings

### By Sharpe Ratio

1. **mean_reversion**: 0.75
2. **relative_volume_breakout**: 0.72
3. **vix_regime_rotation**: 0.40
4. **factor_momentum**: 0.21
5. **quality_small_cap_value**: 0.21
6. **vol_managed_momentum**: 0.17
7. **pairs_trading**: -0.30
8. **sector_rotation**: -0.38

## GA Readiness

**Working strategies:** 8/8

✅ Ready for GA optimization:
  - vol_managed_momentum
  - quality_small_cap_value
  - factor_momentum
  - pairs_trading
  - relative_volume_breakout
  - vix_regime_rotation
  - sector_rotation
  - mean_reversion
