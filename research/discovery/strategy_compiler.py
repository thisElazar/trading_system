"""
Strategy Compiler
=================
Compiles GP genomes into executable strategy objects compatible
with the existing backtesting framework.
"""

import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path

import pandas as pd
from deap import gp

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategies.base import BaseStrategy, Signal, SignalType
from .strategy_genome import StrategyGenome, GenomeFactory
from .config import EvolutionConfig
from .gp_core import set_eval_data, clear_eval_data

logger = logging.getLogger(__name__)


class EvolvedStrategy(BaseStrategy):
    """
    Strategy generated from a GP genome.

    Wraps a StrategyGenome and provides the standard strategy interface
    expected by the backtester.
    """

    def __init__(self, genome: StrategyGenome, factory: GenomeFactory):
        """
        Initialize evolved strategy.

        Args:
            genome: The GP genome defining this strategy
            factory: GenomeFactory for compiling trees
        """
        super().__init__(name=f"evolved_{genome.genome_id}")
        self.genome = genome
        self.factory = factory

        # Compile trees once for efficiency
        self._entry_func = None
        self._exit_func = None
        self._position_func = None
        self._stop_func = None
        self._target_func = None

        self._compile_trees()

    def _compile_trees(self):
        """
        Store trees for lazy compilation.

        Note: We cannot pre-compile trees because GP trees with 0-arity
        primitives get evaluated during gp.compile(). Instead, we compile
        fresh each time with data context already set.
        """
        # Trees are stored in genome, nothing to pre-compile
        pass

    # Maximum signals per evaluation - prevents memory exhaustion from degenerate strategies
    # A strategy that triggers on > 10% of symbols is likely a bad genome
    MAX_SIGNALS_PER_EVAL = 50

    def generate_signals(self,
                         data: Dict[str, pd.DataFrame],
                         current_positions: List[str] = None,
                         vix_regime: str = None) -> List[Signal]:
        """
        Generate trading signals based on evolved GP rules.

        Args:
            data: Dict mapping symbol to DataFrame with OHLCV + indicators
            current_positions: List of symbols currently held
            vix_regime: Current VIX regime ('low', 'normal', 'high', 'extreme')

        Returns:
            List of Signal objects
        """
        signals = []
        current_positions = current_positions or []
        entry_count = 0  # Track how many symbols triggered entry

        for symbol, df in data.items():
            # Need enough history for indicators
            if len(df) < 50:
                continue

            # Safety: abort if we've generated too many signals (degenerate strategy)
            if len(signals) >= self.MAX_SIGNALS_PER_EVAL:
                logger.warning(
                    f"Strategy {self.genome.genome_id} hit signal limit ({self.MAX_SIGNALS_PER_EVAL}) - "
                    f"likely degenerate genome. Aborting signal generation."
                )
                break

            try:
                # Check if we should enter
                if symbol not in current_positions:
                    entry_signal = self._evaluate_entry(df)

                    if entry_signal:
                        entry_count += 1
                        price = float(df['close'].iloc[-1])
                        position_pct = self._evaluate_position(df)
                        stop_pct = self._evaluate_stop_loss(df)
                        target_pct = self._evaluate_target(df)

                        # Get timestamp safely
                        if hasattr(df.index[-1], 'to_pydatetime'):
                            ts = df.index[-1].to_pydatetime()
                        elif hasattr(df.index[-1], 'isoformat'):
                            ts = df.index[-1]
                        else:
                            ts = datetime.now()

                        signals.append(Signal(
                            timestamp=ts,
                            symbol=symbol,
                            strategy=self.name,
                            signal_type=SignalType.BUY,
                            strength=0.7,
                            price=price,
                            stop_loss=price * (1 - stop_pct),
                            target_price=price * (1 + target_pct),
                            position_size_pct=position_pct,
                            reason=f"GP entry: gen {self.genome.generation}",
                            metadata={
                                'genome_id': self.genome.genome_id,
                                'generation': self.genome.generation,
                                'stop_pct': stop_pct,
                                'target_pct': target_pct
                            }
                        ))

                # Check if we should exit existing position
                elif symbol in current_positions:
                    exit_signal = self._evaluate_exit(df)

                    if exit_signal:
                        if hasattr(df.index[-1], 'to_pydatetime'):
                            ts = df.index[-1].to_pydatetime()
                        elif hasattr(df.index[-1], 'isoformat'):
                            ts = df.index[-1]
                        else:
                            ts = datetime.now()

                        signals.append(Signal(
                            timestamp=ts,
                            symbol=symbol,
                            strategy=self.name,
                            signal_type=SignalType.CLOSE,
                            strength=1.0,
                            price=float(df['close'].iloc[-1]),
                            reason=f"GP exit: gen {self.genome.generation}",
                            metadata={
                                'genome_id': self.genome.genome_id,
                                'generation': self.genome.generation
                            }
                        ))

            except Exception as e:
                logger.debug(f"Signal generation failed for {symbol}: {e}")
                continue

        return signals

    def _compile_and_eval(self, tree, pset) -> Any:
        """
        Compile and evaluate a GP tree with data context set.

        Must be called with data context already set via set_eval_data().
        """
        try:
            result = gp.compile(tree, pset)
            # For 0-arity trees, compile returns the result directly
            if callable(result):
                return result()
            return result
        except Exception as e:
            logger.debug(f"Tree evaluation failed: {e}")
            return None

    def _evaluate_entry(self, df: pd.DataFrame) -> bool:
        """Evaluate entry condition."""
        try:
            set_eval_data(df)
            result = self._compile_and_eval(self.genome.entry_tree, self.factory.bool_pset)
            return bool(result) if result is not None else False
        except Exception:
            return False
        finally:
            clear_eval_data()

    def _evaluate_exit(self, df: pd.DataFrame) -> bool:
        """Evaluate exit condition."""
        try:
            set_eval_data(df)
            result = self._compile_and_eval(self.genome.exit_tree, self.factory.bool_pset)
            return bool(result) if result is not None else False
        except Exception:
            return False
        finally:
            clear_eval_data()

    def _evaluate_position(self, df: pd.DataFrame) -> float:
        """Evaluate position size (clamped to valid range)."""
        try:
            set_eval_data(df)
            result = self._compile_and_eval(self.genome.position_tree, self.factory.float_pset)
            if result is not None and not pd.isna(result):
                # Clamp to valid range
                return max(0.01, min(0.20, abs(float(result))))
            return 0.10
        except Exception:
            return 0.10
        finally:
            clear_eval_data()

    def _evaluate_stop_loss(self, df: pd.DataFrame) -> float:
        """Evaluate stop loss percentage (clamped to valid range)."""
        try:
            set_eval_data(df)
            result = self._compile_and_eval(self.genome.stop_loss_tree, self.factory.float_pset)
            if result is not None and not pd.isna(result):
                return max(0.01, min(0.15, abs(float(result))))
            return 0.05
        except Exception:
            return 0.05
        finally:
            clear_eval_data()

    def _evaluate_target(self, df: pd.DataFrame) -> float:
        """Evaluate target percentage (clamped to valid range)."""
        try:
            set_eval_data(df)
            result = self._compile_and_eval(self.genome.target_tree, self.factory.float_pset)
            if result is not None and not pd.isna(result):
                return max(0.02, min(0.30, abs(float(result))))
            return 0.10
        except Exception:
            return 0.10
        finally:
            clear_eval_data()


class StrategyCompiler:
    """
    Compiles GP genomes into executable strategies and evaluates them.

    Integrates with the existing backtesting framework.
    """

    def __init__(self, config: EvolutionConfig = None):
        """
        Initialize compiler.

        Args:
            config: Evolution configuration
        """
        self.config = config or EvolutionConfig()
        self.factory = GenomeFactory(config)

        # Cache compiled strategies with size limit to prevent memory leak
        self._strategy_cache: Dict[str, EvolvedStrategy] = {}
        self._cache_order: List[str] = []  # Track insertion order for LRU
        self._max_cache_size = 500  # Limit cache size

    def compile(self, genome: StrategyGenome) -> EvolvedStrategy:
        """
        Compile a genome into an executable strategy.

        Uses caching with LRU eviction to avoid recompilation and memory leaks.

        Args:
            genome: Genome to compile

        Returns:
            EvolvedStrategy instance
        """
        if genome.genome_id in self._strategy_cache:
            return self._strategy_cache[genome.genome_id]

        strategy = EvolvedStrategy(genome, self.factory)
        self._strategy_cache[genome.genome_id] = strategy
        self._cache_order.append(genome.genome_id)

        # Evict oldest entries if cache exceeds limit
        while len(self._strategy_cache) > self._max_cache_size:
            oldest_id = self._cache_order.pop(0)
            self._strategy_cache.pop(oldest_id, None)

        return strategy

    def clear_cache(self):
        """Clear the strategy cache."""
        self._strategy_cache.clear()
        self._cache_order.clear()

    def batch_compile(self, genomes: List[StrategyGenome]) -> List[EvolvedStrategy]:
        """
        Compile multiple genomes.

        Args:
            genomes: List of genomes to compile

        Returns:
            List of compiled strategies
        """
        return [self.compile(g) for g in genomes]


def save_strategy_to_file(genome: StrategyGenome, factory: GenomeFactory,
                          output_dir: Path) -> Path:
    """
    Save an evolved strategy as a Python file.

    Args:
        genome: Genome to save
        factory: GenomeFactory for tree serialization
        output_dir: Directory to save to

    Returns:
        Path to saved file
    """
    from .strategy_genome import generate_strategy_code

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"evolved_{genome.genome_id}.py"
    filepath = output_dir / filename

    code = generate_strategy_code(genome, factory)
    filepath.write_text(code)

    logger.info(f"Saved strategy to {filepath}")
    return filepath


if __name__ == "__main__":
    import logging
    import numpy as np

    logging.basicConfig(level=logging.INFO)

    print("Testing Strategy Compiler...")

    # Create factory and compiler
    factory = GenomeFactory()
    compiler = StrategyCompiler()

    # Create a random genome
    genome = factory.create_random_genome(generation=0)
    print(f"\nGenome: {genome}")
    print(f"  Entry tree: {str(genome.entry_tree)[:80]}...")

    # Compile to strategy
    strategy = compiler.compile(genome)
    print(f"\nCompiled strategy: {strategy.name}")

    # Create test data
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    test_data = {
        'AAPL': pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 150,
            'high': np.random.randn(100).cumsum() + 152,
            'low': np.random.randn(100).cumsum() + 148,
            'close': np.random.randn(100).cumsum() + 150,
            'volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates),
        'MSFT': pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 300,
            'high': np.random.randn(100).cumsum() + 302,
            'low': np.random.randn(100).cumsum() + 298,
            'close': np.random.randn(100).cumsum() + 300,
            'volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
    }

    # Generate signals
    signals = strategy.generate_signals(test_data, current_positions=[], vix_regime='normal')
    print(f"\nGenerated {len(signals)} signals:")
    for sig in signals[:3]:
        print(f"  {sig.symbol}: {sig.signal_type.value} @ ${sig.price:.2f}")

    # Test with existing position
    signals_with_pos = strategy.generate_signals(
        test_data,
        current_positions=['AAPL'],
        vix_regime='normal'
    )
    print(f"\nWith position in AAPL, generated {len(signals_with_pos)} signals")

    print("\nTest complete!")
