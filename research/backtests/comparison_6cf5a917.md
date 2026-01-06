# Strategy Comparison Report

**Run ID:** `6cf5a917`
**Timestamp:** 2025-12-29T10:43:56.788435
**Period:** earliest to latest
**Strategies Compared:** 3

## Executive Summary

**Best Overall:** mean_reversion
**Best High VIX:** None
**Best Low VIX:** None

## Overall Performance

| Strategy | Sharpe | Annual Return | Max DD | Win Rate | Total Trades |
|----------|--------|---------------|--------|----------|--------------|
| mean_reversion       |   0.50 |    3.6% |  -20.1% |  56.2% | 4510 |
| vol_managed_momentum |   0.32 |    3.3% |  -41.2% |  48.1% | 7910 |
| vix_regime_rotation  |   0.06 |    0.1% |   -3.3% |  40.0% |    5 |

## Statistical Significance

| Comparison | p-value | Significant | Effect Size |
|------------|---------|-------------|-------------|
| vol_managed_momentum_vs_mean_reversion |  0.9617 | ✗           | small       |
| vol_managed_momentum_vs_vix_regime_rotation |  0.3173 | ✗           | small       |
| mean_reversion_vs_vix_regime_rotation |  0.1196 | ✗           | small       |

## Transaction Cost Sensitivity

Sharpe ratio under different cost assumptions:

| Strategy | No Slippage | With Slippage | Degradation |
|----------|-------------|---------------|-------------|
| vol_managed_momentum |        0.71 |          0.32 |       54.9% |
| mean_reversion       |        0.78 |          0.50 |       35.8% |
| vix_regime_rotation  |        0.00 |          0.00 |        0.0% |

## Research Validation

Performance vs academic benchmarks:

| Strategy | Actual Sharpe | Research Sharpe | % of Target | Meets Threshold |
|----------|---------------|-----------------|-------------|-----------------|
| vol_managed_momentum |          0.32 |            1.70 |       18.9% | ✗               |
| mean_reversion       |          0.50 |            0.82 |       61.5% | ✓               |
| vix_regime_rotation  |          0.06 |            0.73 |        8.9% | ✗               |

## Detailed Metrics

### vol_managed_momentum

- **Total Return:** 38.74%
- **Annual Return:** 3.34%
- **Sharpe Ratio:** 0.32
- **Sortino Ratio:** 0.39
- **Max Drawdown:** -41.21%
- **Volatility:** 12.72%
- **Total Trades:** 7910
- **Win Rate:** 48.1%
- **Profit Factor:** 0.94
- **Avg Trade P&L:** $-4.20

### mean_reversion

- **Total Return:** 42.82%
- **Annual Return:** 3.64%
- **Sharpe Ratio:** 0.50
- **Sortino Ratio:** 0.65
- **Max Drawdown:** -20.11%
- **Volatility:** 7.68%
- **Total Trades:** 4510
- **Win Rate:** 56.2%
- **Profit Factor:** 1.06
- **Avg Trade P&L:** $3.17

### vix_regime_rotation

- **Total Return:** 0.54%
- **Annual Return:** 0.05%
- **Sharpe Ratio:** 0.06
- **Sortino Ratio:** 0.01
- **Max Drawdown:** -3.27%
- **Volatility:** 0.89%
- **Total Trades:** 5
- **Win Rate:** 40.0%
- **Profit Factor:** 1.21
- **Avg Trade P&L:** $99.98
