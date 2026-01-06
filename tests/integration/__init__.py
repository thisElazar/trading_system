"""
Integration Tests for the Trading System
========================================

Integration tests verify that multiple components work together correctly.
They may involve:
- Database operations (with test databases)
- Multiple strategy interactions
- End-to-end signal flow
- Backtest pipeline integration

Test Categories:
    - test_backtest_pipeline.py - Full backtest workflow
    - test_signal_flow.py - Signal generation to execution
    - test_database_operations.py - Database CRUD operations
    - test_data_pipeline.py - Data loading and processing

Running Integration Tests:
    pytest tests/integration/ -m integration
    pytest tests/integration/ -v --tb=long

Characteristics:
- Slower than unit tests (may take several seconds)
- May require test databases (automatically created)
- Test real component interactions
- Still avoid external APIs (use mocks)

All tests in this directory are automatically marked with @pytest.mark.integration
"""
