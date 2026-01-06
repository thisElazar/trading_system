"""Debug why all strategies converge to Sortino=0.50"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from research.discovery.evolution_engine import EvolutionEngine
from research.discovery.config import EvolutionConfig
from research.discovery.multi_objective import calculate_sortino_ratio

def main():
    print("=" * 70)
    print("DEBUG: Why is Sortino converging to 0.50?")
    print("=" * 70)

    # Create test data
    np.random.seed(42)
    n_rows = 200
    dates = pd.date_range('2023-01-01', periods=n_rows, freq='D')
    
    test_data = {}
    for symbol in ['AAPL', 'MSFT', 'GOOGL']:
        close = 100 + np.cumsum(np.random.randn(n_rows) * 2)
        test_data[symbol] = pd.DataFrame({
            'open': close + np.random.randn(n_rows) * 0.5,
            'high': close + abs(np.random.randn(n_rows)) * 1.5,
            'low': close - abs(np.random.randn(n_rows)) * 1.5,
            'close': close,
            'volume': np.random.randint(1000000, 5000000, n_rows)
        }, index=dates)

    config = EvolutionConfig(population_size=10, max_tree_depth=4)
    engine = EvolutionEngine(config=config, use_fast_backtester=False)
    engine.load_data(data=test_data)
    engine.initialize_population()

    print("\nEvaluating 10 genomes and inspecting results...")
    print("-" * 70)

    for i, genome in enumerate(engine.population[:10]):
        fitness, behavior = engine.evaluate_genome(genome)
        result = genome.backtest_result
        
        # Get details
        equity = pd.Series(result.equity_curve) if result.equity_curve else pd.Series([100000])
        returns = equity.pct_change().dropna()
        
        print(f"\nGenome {i}: {genome.genome_id[:8]}")
        print(f"  Entry tree: {str(genome.entry_tree)[:50]}...")
        print(f"  Trades: {result.total_trades}")
        print(f"  Equity curve len: {len(equity)}")
        print(f"  Non-zero returns: {(returns != 0).sum()}/{len(returns)}")
        print(f"  Returns range: [{returns.min():.4f}, {returns.max():.4f}]")
        print(f"  Sortino (raw calc): {calculate_sortino_ratio(returns):.4f}")
        print(f"  Fitness sortino: {fitness.sortino:.4f}")
        print(f"  Fitness novelty: {fitness.novelty:.4f}")
        print(f"  Behavior: trade_freq={behavior.trade_frequency:.3f}, avg_hold={behavior.avg_holding_period:.1f}")

    # Test what happens with various return scenarios
    print("\n" + "=" * 70)
    print("Testing Sortino calculation edge cases:")
    print("-" * 70)
    
    # Flat (no trades)
    flat_returns = pd.Series([0.0] * 100)
    print(f"Flat (all zeros): {calculate_sortino_ratio(flat_returns):.4f}")
    
    # All positive
    pos_returns = pd.Series([0.01] * 100)
    print(f"All positive (0.01): {calculate_sortino_ratio(pos_returns):.4f}")
    
    # Mixed
    mixed = pd.Series(np.random.normal(0.001, 0.02, 100))
    print(f"Mixed normal: {calculate_sortino_ratio(mixed):.4f}")
    
    # Tiny positive
    tiny = pd.Series([0.0001] * 100)
    print(f"Tiny positive (0.0001): {calculate_sortino_ratio(tiny):.4f}")

if __name__ == "__main__":
    main()
