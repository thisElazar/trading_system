# Backtester Guide

This directory contains multiple backtester implementations optimized for different use cases.

## Primary Backtesters

### `backtester.py` - Production Backtester
**Use for**: Final validation, walk-forward testing, production strategy evaluation
- Full-featured with detailed metrics
- Walk-forward validation support
- Accurate transaction cost modeling
- Comprehensive equity curve and trade logging

### `backtester_fast.py` - GA/GP Optimization
**Use for**: Genetic algorithm fitness evaluation, rapid parameter search
- 5-10x faster than standard backtester
- Simplified execution model for speed
- Day sampling for faster iteration
- Suitable for evaluating thousands of strategies

## Specialized Backtesters

### `parallel_backtester.py`
Multi-process backtesting for batch evaluation of multiple strategies simultaneously.

### `unified_tester.py`
Unified interface that wraps multiple backtesters with consistent API.

### `genetic/rapid_backtester.py`
Ultra-fast backtester for short period testing (30-second runs).
Uses pre-sliced data caching and simplified position tracking.

## When to Use Which

| Scenario | Backtester |
|----------|------------|
| Final strategy validation | `backtester.py` |
| GA fitness evaluation | `backtester_fast.py` |
| GP discovery overnight | `backtester_fast.py` |
| Multi-strategy comparison | `parallel_backtester.py` |
| Quick period testing | `genetic/rapid_backtester.py` |

## Integration

The overnight discovery system (`discovery/overnight_runner.py`) uses `backtester_fast.py`
for efficiency. Promoted strategies should be re-validated with `backtester.py` before
going live.
