# Baseline Backtest Report

**Run ID:** `20251229_205840`
**Date:** 2025-12-29 21:07:35
**Mode:** Full
**Symbols:** 200

## Results Summary

| Strategy | Sharpe | Return | Max DD | Trades | Win Rate | Status |
|----------|--------|--------|--------|--------|----------|--------|
| vol_managed_momentum | 0.17 | 13.1% | -36.0% | 3285 | 49.0% | ⚠️ |
| quality_small_cap_value | 0.21 | 19.5% | -41.2% | 2199 | 42.8% | ⚠️ |
| factor_momentum | 0.26 | 21.2% | -18.9% | 194 | 47.4% | ⚠️ |
| pairs_trading | -0.38 | -17.4% | -20.7% | 2283 | 44.2% | ❌ |
| relative_volume_breakout | 0.72 | 178.3% | -24.1% | 1943 | 74.3% | ✅ |
| vix_regime_rotation | -0.04 | -0.3% | -3.4% | 5 | 40.0% | ❌ |
| sector_rotation | -0.37 | -2.4% | -3.4% | 8 | 25.0% | ❌ |
| mean_reversion | 0.75 | 68.5% | -14.6% | 3380 | 58.8% | ✅ |

## Rankings

### By Sharpe Ratio

1. **mean_reversion**: 0.75
2. **relative_volume_breakout**: 0.72
3. **factor_momentum**: 0.26
4. **quality_small_cap_value**: 0.21
5. **vol_managed_momentum**: 0.17
6. **vix_regime_rotation**: -0.04
7. **sector_rotation**: -0.37
8. **pairs_trading**: -0.38

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
