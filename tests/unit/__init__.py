"""
Unit Tests for the Trading System
=================================

Unit tests are:
- Fast (< 1 second each)
- Isolated (no external dependencies like APIs or databases)
- Deterministic (same input = same output)
- Independent (can run in any order)

Test Categories:
    - test_strategies.py - Strategy signal generation logic
    - test_backtester.py - Backtest engine calculations
    - test_risk.py - Risk management rules
    - test_signals.py - Signal validation and processing
    - test_indicators.py - Technical indicator calculations
    - test_order_execution.py - Order execution logic

Running Unit Tests:
    pytest tests/unit/ -m unit
    pytest tests/unit/ -v --tb=short
    pytest tests/unit/test_strategies.py -k "momentum"

All tests in this directory are automatically marked with @pytest.mark.unit
"""
