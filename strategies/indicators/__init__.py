"""
Simple Indicator Strategies
===========================
Single-symbol strategies based on technical indicators.

These serve dual purposes:
1. Visualization/exploration in the backtest tool
2. Building blocks (primitives) for GP evolution

These are NOT production strategies - they're intentionally simple
to allow manual tuning and understanding of indicator behavior.
"""

from strategies.indicators.ema_crossover import EMACrossoverStrategy
from strategies.indicators.rsi_reversal import RSIReversalStrategy
from strategies.indicators.macd_signal import MACDSignalStrategy

__all__ = [
    'EMACrossoverStrategy',
    'RSIReversalStrategy',
    'MACDSignalStrategy',
]

# Indicator strategy metadata
INDICATOR_STRATEGIES = {
    'ema_crossover': {
        'class': EMACrossoverStrategy,
        'single_symbol': True,
        'category': 'indicator',
        'description': 'Buy when fast EMA crosses above slow EMA',
        'params': ['fast_period', 'slow_period'],
    },
    'rsi_reversal': {
        'class': RSIReversalStrategy,
        'single_symbol': True,
        'category': 'indicator',
        'description': 'Buy oversold (RSI<30), sell overbought (RSI>70)',
        'params': ['rsi_period', 'oversold', 'overbought'],
    },
    'macd_signal': {
        'class': MACDSignalStrategy,
        'single_symbol': True,
        'category': 'indicator',
        'description': 'Trade MACD/signal line crossovers',
        'params': ['fast_period', 'slow_period', 'signal_period'],
    },
}
