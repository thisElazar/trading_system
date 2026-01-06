# Monte Carlo Robustness Analysis

**Strategy:** vix_regime_rotation
**Timestamp:** 2025-12-28T21:15:30.094642
**Simulations:** 100

## Executive Summary

**Median Sharpe Ratio:** 0.55
**68% Confidence Interval:** [0.19, 0.87]
**95% Confidence Interval:** [-0.02, 1.05]

**Probability of Sharpe > 1.0:** 8.0%
**Probability of Positive Returns:** 96.0%
**Probability of Max DD < 20%:** 100.0%

## Distribution Statistics

### Sharpe Ratio

- **Median:** 0.55
- **Mean:** 0.55
- **Std Dev:** 0.32
- **5th Percentile (Worst Case):** 0.05
- **95th Percentile:** 1.02

### Total Return

- **Median:** 5.4%
- **95% CI:** [-0.2%, 11.8%]
- **5th Percentile (Worst Case):** 0.4%

### Max Drawdown

- **Median:** 1.8%
- **95% CI:** [1.0%, 3.6%]
- **95th Percentile (Worst Case):** 3.2%

## Probability Analysis

- **P(Sharpe ≥ 0.4):** 74.0%
- **P(Sharpe ≥ 1.0):** 8.0%
- **P(Positive Returns):** 96.0%
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
