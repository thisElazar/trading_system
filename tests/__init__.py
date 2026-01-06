"""
Trading System Test Suite
=========================

Comprehensive pytest infrastructure for testing the trading system.

Directory Structure:
    tests/
    ├── __init__.py         # This file
    ├── conftest.py         # Shared fixtures
    ├── utils.py            # Test utilities
    ├── unit/               # Unit tests (fast, isolated)
    ├── integration/        # Integration tests
    ├── fixtures/           # Test data files
    └── mocks/              # Mock implementations

Usage:
    # Run all tests
    pytest

    # Run only unit tests
    pytest -m unit

    # Run with coverage
    pytest --cov=execution --cov=strategies --cov-report=html

    # Run excluding slow tests
    pytest -m "not slow"

    # Run critical path tests only
    pytest -m critical
"""

__version__ = "1.0.0"
