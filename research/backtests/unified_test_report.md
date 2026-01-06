# Unified Strategy Test Report

**Date:** 2025-12-29 09:13
**Symbols:** 581

## Results Summary

| Strategy | Sharpe | Annual Return | Trades | Win Rate | Meets Min | vs Research |
|----------|--------|---------------|--------|----------|-----------|-------------|
| relative_volume_breakout | 2.81 | 644.1% | 987 | 55.1% | ✓ | 100% |
| pairs_trading | 1.57 | 29.2% | 80 | 68.8% | ✓ | 63% |
| vix_regime_rotation | 0.50 | 0.6% | 5 | 80.0% | ✓ | 68% |
| vol_managed_momentum | 0.46 | 3.3% | 1430 | 50.6% | ✗ | 27% |
| sector_rotation | -0.04 | -0.2% | 56 | 46.4% | ✗ | -5% |

## Validation Status

- **vol_managed_momentum:** ❌ FAIL
  - Actual: 0.46, Min: 1.0, Research: 1.7
- **vix_regime_rotation:** ✅ PASS
  - Actual: 0.50, Min: 0.4, Research: 0.73
- **sector_rotation:** ❌ FAIL
  - Actual: -0.04, Min: 0.5, Research: 0.73
- **pairs_trading:** ✅ PASS
  - Actual: 1.57, Min: 1.5, Research: 2.5
- **relative_volume_breakout:** ✅ PASS
  - Actual: 2.81, Min: 1.8, Research: 2.81