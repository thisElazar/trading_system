# Monte Carlo Robustness Analysis

**Strategy:** vix_regime_rotation
**Timestamp:** 2025-12-28T21:06:41.210872
**Simulations:** 100

## Executive Summary

**Median Sharpe Ratio:** 0.55
**68% Confidence Interval:** [0.23, 0.82]
**95% Confidence Interval:** [-0.15, 1.06]

**Probability of Sharpe > 1.0:** 8.0%
**Probability of Positive Returns:** 95.0%
**Probability of Max DD < 20%:** 100.0%

## Distribution Statistics

### Sharpe Ratio

- **Median:** 0.55
- **Mean:** 0.52
- **Std Dev:** 0.32
- **5th Percentile (Worst Case):** 0.01
- **95th Percentile:** 1.02

### Total Return

- **Median:** 5.5%
- **95% CI:** [-1.6%, 12.1%]
- **5th Percentile (Worst Case):** 0.1%

### Max Drawdown

- **Median:** 2.0%
- **95% CI:** [1.1%, 4.5%]
- **95th Percentile (Worst Case):** 3.6%

## Probability Analysis

- **P(Sharpe ≥ 0.4):** 65.0%
- **P(Sharpe ≥ 1.0):** 8.0%
- **P(Positive Returns):** 95.0%
- **P(Max DD < 20%):** 100.0%

## Transaction Cost Impact

| Cost Assumption | Sharpe Ratio |
|-----------------|--------------|
| optimistic      |         0.00 |
| realistic       |         0.00 |

**Degradation from costs:** 0.0%

## Interpretation

❌ **WEAK** - Strategy shows low probability of achieving target performance

✅ Even in worst-case scenarios, strategy maintains positive risk-adjusted returns
