"""
Strategy Loader - Loads promoted strategies into execution.

Bridges the gap between:
- PromotionPipeline (database with LIVE strategies)
- StrategyScheduler (execution system)

This module enables discovered GP strategies to actually trade by:
1. Loading genome JSON from the promotion_pipeline database
2. Compiling genomes into executable EvolvedStrategy objects
3. Creating runner functions compatible with the scheduler
4. Registering strategies with the scheduler at their configured run times
"""

import logging
import json
from typing import Dict, Optional, Callable, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Import strategy compilation components
try:
    from research.discovery import (
        StrategyCompiler,
        GenomeFactory,
        EvolutionConfig,
        PI_CONFIG,
    )
    from research.discovery.promotion_pipeline import PromotionPipeline, StrategyStatus
    HAS_DISCOVERY = True
except ImportError as e:
    logger.warning(f"Discovery module not available: {e}")
    HAS_DISCOVERY = False
    StrategyCompiler = None
    GenomeFactory = None
    EvolutionConfig = None
    PI_CONFIG = None
    PromotionPipeline = None
    StrategyStatus = None


class StrategyLoader:
    """
    Loads promoted strategies from database into the execution scheduler.

    This class bridges the gap between the research/discovery pipeline
    (which produces genomes stored in the database) and the execution
    system (which needs runnable strategy objects).

    Usage:
        loader = StrategyLoader(promotion_pipeline, scheduler)
        loaded = loader.load_live_strategies()
        print(f"Loaded {loaded} discovered strategies")
    """

    def __init__(
        self,
        promotion_pipeline: 'PromotionPipeline',
        scheduler: 'StrategyScheduler',
        config: Optional['EvolutionConfig'] = None
    ):
        """
        Initialize the strategy loader.

        Args:
            promotion_pipeline: PromotionPipeline instance for DB access
            scheduler: StrategyScheduler instance for registration
            config: Optional EvolutionConfig (defaults to PI_CONFIG)
        """
        self.pipeline = promotion_pipeline
        self.scheduler = scheduler

        # Use Pi-optimized config by default
        self.config = config or (PI_CONFIG if HAS_DISCOVERY and PI_CONFIG else None)

        # Initialize compiler components
        self.factory: Optional[GenomeFactory] = None
        self.compiler: Optional[StrategyCompiler] = None

        if HAS_DISCOVERY and self.config:
            try:
                self.factory = GenomeFactory(self.config)
                self.compiler = StrategyCompiler(self.config)
                logger.info("Strategy loader initialized with compiler")
            except Exception as e:
                logger.error(f"Failed to initialize strategy compiler: {e}")

        # Track loaded strategies to avoid duplicates
        self._loaded_strategies: Dict[str, Any] = {}

        # Track failed loads to avoid retry spam
        self._failed_loads: Dict[str, datetime] = {}

    @property
    def available(self) -> bool:
        """Check if loader is ready to use."""
        return (
            HAS_DISCOVERY and
            self.factory is not None and
            self.compiler is not None and
            self.pipeline is not None and
            self.scheduler is not None
        )

    def load_live_strategies(self) -> int:
        """
        Load all LIVE strategies from database into scheduler.

        Returns:
            Number of strategies successfully loaded
        """
        if not self.available:
            logger.warning("Strategy loader not available")
            return 0

        try:
            # Get all strategies in LIVE status
            live_ids = self.pipeline.get_strategies_by_status(StrategyStatus.LIVE)

            if not live_ids:
                logger.debug("No LIVE strategies to load")
                return 0

            loaded = 0
            for strategy_id in live_ids:
                # Skip already loaded
                if strategy_id in self._loaded_strategies:
                    continue

                # Skip recently failed
                if strategy_id in self._failed_loads:
                    failed_at = self._failed_loads[strategy_id]
                    # Retry after 1 hour
                    if (datetime.now() - failed_at).seconds < 3600:
                        continue

                success = self._load_strategy(strategy_id)
                if success:
                    loaded += 1

            if loaded > 0:
                logger.info(f"Loaded {loaded} discovered strategies into scheduler")

            return loaded

        except Exception as e:
            logger.error(f"Failed to load live strategies: {e}")
            return 0

    def _load_strategy(self, strategy_id: str) -> bool:
        """
        Load a single strategy from database into scheduler.

        Args:
            strategy_id: Strategy ID to load

        Returns:
            True if successfully loaded
        """
        try:
            # Get strategy record from promotion pipeline
            record = self.pipeline.get_strategy_record(strategy_id)

            if not record:
                logger.warning(f"No record found for strategy {strategy_id}")
                self._failed_loads[strategy_id] = datetime.now()
                return False

            if not record.genome_json:
                logger.warning(f"No genome JSON for strategy {strategy_id}")
                self._failed_loads[strategy_id] = datetime.now()
                return False

            # Deserialize genome from JSON
            genome = self.factory.deserialize_genome(record.genome_json)

            if not genome:
                logger.warning(f"Failed to deserialize genome for {strategy_id}")
                self._failed_loads[strategy_id] = datetime.now()
                return False

            # Compile to executable strategy
            strategy = self.compiler.compile(genome)

            if not strategy:
                logger.warning(f"Failed to compile strategy {strategy_id}")
                self._failed_loads[strategy_id] = datetime.now()
                return False

            # Create runner function
            runner = self._create_runner(strategy_id, strategy)

            # Get run time from record (default to 10:00)
            run_time = getattr(record, 'run_time', None) or '10:00'

            # Register with scheduler
            self.scheduler.register_strategy(
                name=strategy_id,
                runner=runner,
                run_time=run_time,
                market_hours_only=True
            )

            # Track loaded strategy
            self._loaded_strategies[strategy_id] = {
                'strategy': strategy,
                'loaded_at': datetime.now(),
                'run_time': run_time,
                'genome_id': genome.genome_id if hasattr(genome, 'genome_id') else strategy_id
            }

            # Clear from failed if it was there
            if strategy_id in self._failed_loads:
                del self._failed_loads[strategy_id]

            logger.info(f"Loaded discovered strategy: {strategy_id} (run_time={run_time})")
            return True

        except Exception as e:
            logger.error(f"Failed to load strategy {strategy_id}: {e}")
            self._failed_loads[strategy_id] = datetime.now()
            return False

    def _create_runner(self, strategy_id: str, strategy: Any) -> Callable:
        """
        Create a runner function for the scheduler.

        The runner function must:
        - Take no arguments
        - Return List[Dict] of signals in scheduler format

        Args:
            strategy_id: Strategy identifier
            strategy: Compiled EvolvedStrategy instance

        Returns:
            Callable runner function
        """
        def runner() -> List[Dict]:
            """Execute discovered strategy and return signals."""
            try:
                # Import here to avoid circular imports
                from execution.scheduler import _get_strategy_data

                # Get market data and VIX regime
                data, vix_regime = _get_strategy_data()

                if not data:
                    return []

                # Get current positions (empty list for now, could query broker)
                current_positions = []

                # Generate signals from the compiled strategy
                signals = strategy.generate_signals(
                    data=data,
                    current_positions=current_positions,
                    vix_regime=vix_regime
                )

                if not signals:
                    return []

                # Convert Signal objects to dict format expected by scheduler
                return self._signals_to_dicts(signals, strategy_id)

            except Exception as e:
                logger.error(f"Error running discovered strategy {strategy_id}: {e}")
                return []

        return runner

    def _signals_to_dicts(self, signals: List, strategy_id: str) -> List[Dict]:
        """
        Convert Signal objects to dict format for scheduler.

        Args:
            signals: List of Signal objects
            strategy_id: Strategy that generated the signals

        Returns:
            List of signal dictionaries
        """
        result = []

        for sig in signals:
            try:
                # Handle both Signal objects and dicts
                if hasattr(sig, 'symbol'):
                    signal_dict = {
                        'symbol': sig.symbol,
                        'direction': 'long' if sig.signal_type.name in ('BUY', 'LONG') else 'short',
                        'price': sig.price,
                        'stop_loss': getattr(sig, 'stop_loss', 0),
                        'target': getattr(sig, 'target_price', 0),
                        'confidence': getattr(sig, 'strength', 0.5),
                        'metadata': {
                            'strategy': strategy_id,
                            'discovered': True,
                            'genome_id': getattr(sig, 'metadata', {}).get('genome_id', strategy_id),
                            'reason': getattr(sig, 'reason', 'GP-discovered signal'),
                        }
                    }
                else:
                    # Already a dict
                    signal_dict = sig
                    signal_dict['metadata'] = signal_dict.get('metadata', {})
                    signal_dict['metadata']['strategy'] = strategy_id
                    signal_dict['metadata']['discovered'] = True

                result.append(signal_dict)

            except Exception as e:
                logger.warning(f"Failed to convert signal: {e}")

        return result

    def unload_strategy(self, strategy_id: str) -> bool:
        """
        Remove a strategy from the scheduler (e.g., on retirement).

        Args:
            strategy_id: Strategy ID to unload

        Returns:
            True if successfully unloaded
        """
        if strategy_id in self._loaded_strategies:
            del self._loaded_strategies[strategy_id]
            logger.info(f"Unloaded strategy: {strategy_id}")

            # Note: StrategyScheduler doesn't have an unregister method
            # The strategy will remain registered but can be disabled via config
            # A future enhancement could add scheduler.unregister_strategy()

            return True

        return False

    def get_loaded_strategies(self) -> List[str]:
        """Get list of currently loaded strategy IDs."""
        return list(self._loaded_strategies.keys())

    def get_strategy_info(self, strategy_id: str) -> Optional[Dict]:
        """Get info about a loaded strategy."""
        return self._loaded_strategies.get(strategy_id)

    def reload_strategy(self, strategy_id: str) -> bool:
        """
        Reload a strategy (e.g., after parameter update).

        Args:
            strategy_id: Strategy ID to reload

        Returns:
            True if successfully reloaded
        """
        # Unload first
        self.unload_strategy(strategy_id)

        # Clear from failed loads to force retry
        if strategy_id in self._failed_loads:
            del self._failed_loads[strategy_id]

        # Reload
        return self._load_strategy(strategy_id)
