# Parameter Optimization Report

**Strategy:** mean_reversion
**Timestamp:** 2025-12-30T02:36:55.534751

## Executive Summary

**Baseline Sharpe:** 0.50
**Optimized Sharpe:** 0.66
**Improvement:** 30.7%

**In-Sample Sharpe:** 0.66
**Out-of-Sample Sharpe:** 0.35
**Degradation:** 47.4%

⚠️ **CAUTION:** Moderate overfitting (30-50% degradation)

## Optimized Parameters

| Parameter | Baseline | Optimized | Change |
|-----------|----------|-----------|--------|
| lookback_period      | 21         | 14         | →      |
| bottom_percentile    | 0.2        | 0.25       | →      |
| max_stocks_per_sector | 5          | 5          | —      |
| stop_loss_pct        | -0.15      | -0.2       | →      |
| target_vol           | 0.15       | 0.15       | —      |

## Parameter Sensitivity

How much does each parameter affect performance?

| Parameter | Sensitivity | Classification |
|-----------|-------------|----------------|
| stop_loss_pct        |       0.128 | Unstable       |
| max_stocks_per_sector |       0.073 | Stable         |
| bottom_percentile    |       0.069 | Stable         |
| lookback_period      |       0.060 | Stable         |
| target_vol           |       0.029 | Stable         |

**Stable parameters:** lookback_period, bottom_percentile, max_stocks_per_sector, target_vol
**Unstable parameters:** stop_loss_pct

## Top Parameter Combinations

| Rank | Sharpe | Annual Return | Max DD | Parameters |
|------|--------|---------------|--------|------------|
|    1 |   0.64 |          4.7% |  -15.5% | lookback_period=14, bottom_percentile=0.25, max_stocks_per_sector=7, stop_loss_pct=-0.2, target_vol=0.2 |
|    2 |   0.62 |          4.6% |  -16.7% | lookback_period=21, bottom_percentile=0.25, max_stocks_per_sector=7, stop_loss_pct=-0.2, target_vol=0.15 |
|    3 |   0.54 |          3.9% |  -17.8% | lookback_period=21, bottom_percentile=0.25, max_stocks_per_sector=5, stop_loss_pct=-0.15, target_vol=0.1 |
|    4 |   0.53 |          3.9% |  -16.4% | lookback_period=30, bottom_percentile=0.25, max_stocks_per_sector=5, stop_loss_pct=-0.2, target_vol=0.2 |
|    5 |   0.53 |          4.0% |  -16.8% | lookback_period=14, bottom_percentile=0.15, max_stocks_per_sector=5, stop_loss_pct=-0.2, target_vol=0.15 |
|    6 |   0.52 |          4.0% |  -17.4% | lookback_period=14, bottom_percentile=0.1, max_stocks_per_sector=5, stop_loss_pct=-0.2, target_vol=0.1 |
|    7 |   0.51 |          3.8% |  -17.0% | lookback_period=14, bottom_percentile=0.15, max_stocks_per_sector=3, stop_loss_pct=-0.2, target_vol=0.1 |
|    8 |   0.49 |          3.5% |  -16.4% | lookback_period=30, bottom_percentile=0.25, max_stocks_per_sector=5, stop_loss_pct=-0.15, target_vol=0.2 |
|    9 |   0.49 |          3.6% |  -17.9% | lookback_period=30, bottom_percentile=0.2, max_stocks_per_sector=7, stop_loss_pct=-0.2, target_vol=0.15 |
|   10 |   0.48 |          3.6% |  -16.9% | lookback_period=14, bottom_percentile=0.2, max_stocks_per_sector=3, stop_loss_pct=-0.2, target_vol=0.1 |

## Recommendation

⚠️ **CONSIDER** - Moderate improvement but monitor for overfitting
