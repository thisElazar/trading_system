"""
Validation Module
=================
Advanced validation techniques for detecting overfit strategies.

Key components:
- CPCV: Combinatorial Purged Cross-Validation for backtest overfitting detection
"""

from .cpcv import (
    CPCVConfig,
    CPCVSplit,
    CPCVSplitResult,
    CPCVResult,
    generate_cpcv_splits,
    calculate_pbo,
    run_cpcv_validation,
)

__all__ = [
    "CPCVConfig",
    "CPCVSplit",
    "CPCVSplitResult",
    "CPCVResult",
    "generate_cpcv_splits",
    "calculate_pbo",
    "run_cpcv_validation",
]
