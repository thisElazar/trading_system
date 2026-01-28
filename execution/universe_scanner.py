"""
Universe Scanner
================
Scans the entire stock universe in memory-safe batches to find trading signals.

Two-phase approach:
1. Phase 1 (Broad Scan): Run strategies on all symbols in batches of ~256
   - Collect all signals with strength > threshold
   - Memory-safe: only one batch loaded at a time

2. Phase 2 (Validation): Re-run strategies on Phase 1 candidates
   - Double-check evaluation before passing signals through
   - Ensures signal quality

Performance (Pi 5, 4GB RAM):
- ~2,560 symbols in universe
- 10 batches × 12s = ~2 min for Phase 1
- ~1 min for Phase 2
- Total: ~3 min for full universe scan
"""

import logging
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
import gc
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.types import Side

logger = logging.getLogger(__name__)


@dataclass
class ScanConfig:
    """Configuration for universe scanning."""
    batch_size: int = 128  # Reduced from 256 for Pi 4GB safety (~300MB per batch)
    confidence_threshold: float = 0.6  # DEPRECATED: Use strength_threshold
    phase2_enabled: bool = True
    max_candidates: int = 200  # Max candidates to pass to Phase 2
    log_progress: bool = True
    aggressive_gc: bool = True  # Force garbage collection between batches

    @property
    def strength_threshold(self) -> float:
        """Canonical name for confidence_threshold."""
        return self.confidence_threshold


@dataclass
class CandidateSignal:
    """
    Signal candidate from Phase 1 scan.

    Uses old field names for backward compatibility.
    New code should use canonical names via properties.
    """
    symbol: str
    strategy: str                       # DEPRECATED: Use strategy_id
    direction: str                      # 'long' or 'short'
    confidence: float                   # DEPRECATED: Use strength
    price: float
    stop_loss: float = 0.0
    take_profit: float = 0.0            # DEPRECATED: Use target_price
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Phase 2 validation
    phase2_confidence: Optional[float] = None
    validated: bool = False

    # Canonical property aliases
    @property
    def strategy_id(self) -> str:
        return self.strategy

    @property
    def strength(self) -> float:
        return self.confidence

    @property
    def target_price(self) -> float:
        return self.take_profit

    @property
    def side(self) -> Side:
        dir_map = {'long': 'BUY', 'short': 'SELL'}
        return Side(dir_map.get(self.direction.lower(), self.direction.upper()))


class UniverseScanner:
    """
    Scans entire stock universe in batches to find trading opportunities.

    Usage:
        scanner = UniverseScanner()

        # Register strategies
        scanner.register_strategy('mean_reversion', MeanReversionStrategy())
        scanner.register_strategy('momentum', MomentumStrategy())

        # Run full scan
        signals = scanner.scan()

        # signals contains validated candidates ready for execution
    """

    def __init__(self, config: ScanConfig = None):
        self.config = config or ScanConfig()
        self.strategies: Dict[str, Any] = {}
        self._data_manager = None

    def register_strategy(self, name: str, strategy_instance):
        """Register a strategy for scanning."""
        self.strategies[name] = strategy_instance
        logger.debug(f"Registered strategy: {name}")

    def _get_data_manager(self):
        """Lazy-load data manager."""
        if self._data_manager is None:
            from data.cached_data_manager import CachedDataManager
            self._data_manager = CachedDataManager()
        return self._data_manager

    def _get_vix_regime(self) -> str:
        """Get current VIX regime."""
        try:
            dm = self._get_data_manager()
            vix = dm.get_vix()
            if vix < 15:
                return 'low'
            elif vix < 25:
                return 'normal'
            elif vix < 35:
                return 'high'
            else:
                return 'extreme'
        except Exception as e:
            logger.warning(f"Could not get VIX regime: {e}")
            return 'normal'

    def _split_batches(self, symbols: List[str]) -> List[List[str]]:
        """Split symbols into batches."""
        batch_size = self.config.batch_size
        return [
            symbols[i:i + batch_size]
            for i in range(0, len(symbols), batch_size)
        ]

    def _extract_signals(
        self,
        strategy_name: str,
        raw_signals: List[Any]
    ) -> List[CandidateSignal]:
        """Convert strategy signals to CandidateSignal objects."""
        candidates = []

        for sig in raw_signals:
            # Handle different signal formats
            if hasattr(sig, 'strength'):
                confidence = sig.strength
            elif hasattr(sig, 'confidence'):
                confidence = sig.confidence
            else:
                confidence = 0.5

            # Skip low confidence signals
            if confidence < self.config.confidence_threshold:
                continue

            # Extract signal details
            candidate = CandidateSignal(
                symbol=sig.symbol if hasattr(sig, 'symbol') else str(sig),
                strategy=strategy_name,
                direction='long' if getattr(sig, 'signal_type', None) in ['BUY', 'LONG'] or
                          str(getattr(sig, 'signal_type', '')).upper() in ['BUY', 'LONG'] else 'long',
                confidence=confidence,
                price=getattr(sig, 'price', 0.0),
                stop_loss=getattr(sig, 'stop_loss', 0.0) or 0.0,
                take_profit=getattr(sig, 'target_price', 0.0) or getattr(sig, 'target', 0.0) or 0.0,
                metadata=getattr(sig, 'metadata', {}) or {}
            )
            candidates.append(candidate)

        return candidates

    def _run_batch(
        self,
        symbols: List[str],
        vix_regime: str,
        batch_num: int,
        total_batches: int
    ) -> List[CandidateSignal]:
        """Run all strategies on a single batch of symbols."""
        dm = self._get_data_manager()

        # Load data for this batch
        start = time.time()
        dm.load_all(symbols)
        data = dm.get_bars_batch(symbols)
        load_time = time.time() - start

        if not data:
            logger.warning(f"Batch {batch_num}/{total_batches}: No data loaded")
            return []

        # Run all strategies on this batch
        candidates = []
        strategy_times = []

        for name, strategy in self.strategies.items():
            try:
                strat_start = time.time()
                signals = strategy.generate_signals(data=data, vix_regime=vix_regime)
                strat_time = time.time() - strat_start
                strategy_times.append(f"{name}={strat_time:.2f}s")

                if signals:
                    batch_candidates = self._extract_signals(name, signals)
                    candidates.extend(batch_candidates)

            except Exception as e:
                logger.error(f"Strategy {name} failed on batch {batch_num}: {e}")

        if self.config.log_progress:
            # Get memory usage
            try:
                import psutil
                mem_mb = psutil.Process().memory_info().rss / 1024 / 1024
                mem_str = f", mem={mem_mb:.0f}MB"
            except:
                mem_str = ""

            logger.info(
                f"Batch {batch_num}/{total_batches}: "
                f"{len(symbols)} symbols, {len(candidates)} candidates "
                f"(load={load_time:.1f}s, {', '.join(strategy_times)}{mem_str})"
            )

        return candidates

    def phase1_scan(self, symbols: List[str] = None) -> List[CandidateSignal]:
        """
        Phase 1: Broad scan of entire universe in batches.

        Args:
            symbols: Optional list of symbols. If None, uses full universe.

        Returns:
            List of candidate signals with confidence > threshold
        """
        dm = self._get_data_manager()

        # Get symbols if not provided
        if symbols is None:
            symbols = dm.get_available_symbols()
            # Filter out index symbols
            symbols = [s for s in symbols if not s.startswith('^')]

        logger.info(f"Phase 1: Scanning {len(symbols)} symbols in batches of {self.config.batch_size}")

        # Get VIX regime once for all batches
        vix_regime = self._get_vix_regime()
        logger.info(f"VIX regime: {vix_regime}")

        # Split into batches
        batches = self._split_batches(symbols)
        total_batches = len(batches)

        # Scan each batch
        all_candidates = []
        phase1_start = time.time()

        for i, batch_symbols in enumerate(batches, 1):
            batch_candidates = self._run_batch(
                batch_symbols,
                vix_regime,
                batch_num=i,
                total_batches=total_batches
            )
            all_candidates.extend(batch_candidates)

            # Clear cache and memory between batches
            if self.config.aggressive_gc:
                dm.clear_cache(batch_symbols)
                gc.collect()

        phase1_time = time.time() - phase1_start

        # Sort by confidence (highest first)
        all_candidates.sort(key=lambda x: x.confidence, reverse=True)

        logger.info(
            f"Phase 1 complete: {len(all_candidates)} candidates "
            f"from {len(symbols)} symbols in {phase1_time:.1f}s"
        )

        return all_candidates

    def phase2_validate(
        self,
        candidates: List[CandidateSignal]
    ) -> List[CandidateSignal]:
        """
        Phase 2: Re-run strategies on candidates for validation.

        Args:
            candidates: Candidate signals from Phase 1

        Returns:
            Validated candidates with phase2_confidence set
        """
        if not candidates:
            return []

        # Limit candidates to max
        if len(candidates) > self.config.max_candidates:
            candidates = candidates[:self.config.max_candidates]
            logger.info(f"Phase 2: Limited to top {self.config.max_candidates} candidates")

        # Get unique symbols from candidates
        symbols = list(set(c.symbol for c in candidates))

        logger.info(f"Phase 2: Validating {len(candidates)} candidates ({len(symbols)} unique symbols)")

        dm = self._get_data_manager()
        vix_regime = self._get_vix_regime()

        # Load data for candidate symbols
        phase2_start = time.time()
        dm.load_all(symbols)
        data = dm.get_bars_batch(symbols)

        if not data:
            logger.warning("Phase 2: No data loaded for candidates")
            return candidates

        # Re-run strategies and collect new signals
        phase2_signals: Dict[Tuple[str, str], float] = {}  # (symbol, strategy) -> confidence

        for name, strategy in self.strategies.items():
            try:
                signals = strategy.generate_signals(data=data, vix_regime=vix_regime)
                if signals:
                    for sig in signals:
                        symbol = sig.symbol if hasattr(sig, 'symbol') else str(sig)
                        confidence = getattr(sig, 'strength', getattr(sig, 'confidence', 0.5))
                        phase2_signals[(symbol, name)] = confidence
            except Exception as e:
                logger.error(f"Phase 2 strategy {name} failed: {e}")

        # Update candidates with Phase 2 results
        validated = []
        for candidate in candidates:
            key = (candidate.symbol, candidate.strategy)
            if key in phase2_signals:
                candidate.phase2_confidence = phase2_signals[key]
                candidate.validated = True
                # Only keep if still above threshold in Phase 2
                if candidate.phase2_confidence >= self.config.confidence_threshold:
                    validated.append(candidate)
            else:
                # Signal not reproduced in Phase 2 - skip it
                logger.debug(f"Signal {candidate.symbol}/{candidate.strategy} not reproduced in Phase 2")

        phase2_time = time.time() - phase2_start

        # Sort by Phase 2 confidence
        validated.sort(key=lambda x: x.phase2_confidence or 0, reverse=True)

        logger.info(
            f"Phase 2 complete: {len(validated)}/{len(candidates)} validated "
            f"in {phase2_time:.1f}s"
        )

        return validated

    def scan(self, symbols: List[str] = None) -> List[CandidateSignal]:
        """
        Run full two-phase universe scan.

        Args:
            symbols: Optional list of symbols. If None, uses full universe.

        Returns:
            List of validated candidate signals, sorted by confidence
        """
        if not self.strategies:
            logger.warning("No strategies registered for scanning")
            return []

        scan_start = time.time()
        logger.info(f"Starting universe scan with {len(self.strategies)} strategies")

        # Phase 1: Broad scan
        candidates = self.phase1_scan(symbols)

        if not candidates:
            logger.info("No candidates found in Phase 1")
            return []

        # Phase 2: Validation (if enabled)
        if self.config.phase2_enabled:
            validated = self.phase2_validate(candidates)
        else:
            validated = candidates
            for c in validated:
                c.validated = True
                c.phase2_confidence = c.confidence

        scan_time = time.time() - scan_start
        logger.info(
            f"Universe scan complete: {len(validated)} validated signals "
            f"in {scan_time:.1f}s ({scan_time/60:.1f} min)"
        )

        return validated

    def scan_strategy(
        self,
        strategy_name: str,
        symbols: List[str] = None
    ) -> List[CandidateSignal]:
        """
        Scan universe with a single strategy.

        Useful when strategies need to be run independently
        (e.g., different schedules).

        Args:
            strategy_name: Name of registered strategy to run
            symbols: Optional list of symbols

        Returns:
            Validated signals for this strategy
        """
        if strategy_name not in self.strategies:
            logger.error(f"Strategy {strategy_name} not registered")
            return []

        # Temporarily filter to just this strategy
        original_strategies = self.strategies.copy()
        self.strategies = {strategy_name: original_strategies[strategy_name]}

        try:
            results = self.scan(symbols)
        finally:
            self.strategies = original_strategies

        return results


def create_scanner_with_strategies() -> UniverseScanner:
    """
    Create a UniverseScanner with all enabled strategies registered.

    Returns:
        Configured UniverseScanner ready for scanning
    """
    from config import STRATEGIES

    scanner = UniverseScanner()

    # Import and register each enabled strategy
    # NOTE: gap_fill is NOT included here - it runs separately at 09:31 ET
    # via the scheduler (it has its own fixed universe of SPY/QQQ and ignores
    # batch data, which caused duplicate signals when included here)
    strategy_imports = {
        'mean_reversion': ('strategies.mean_reversion', 'MeanReversionStrategy'),
        'relative_volume_breakout': ('strategies.relative_volume_breakout', 'RelativeVolumeBreakout'),
        'vix_regime_rotation': ('strategies.vix_regime_rotation', 'VIXRegimeRotationStrategy'),
        'vol_managed_momentum': ('strategies.vol_managed_momentum_v2', 'VolManagedMomentumV2'),
        'quality_smallcap_value': ('strategies.quality_small_cap_value', 'QualitySmallCapValueStrategy'),
        'factor_momentum': ('strategies.factor_momentum', 'FactorMomentumStrategy'),
    }

    for name, (module, cls_name) in strategy_imports.items():
        if not STRATEGIES.get(name, {}).get('enabled', False):
            continue

        try:
            mod = __import__(module, fromlist=[cls_name])
            strategy_class = getattr(mod, cls_name)
            scanner.register_strategy(name, strategy_class())
            logger.info(f"Registered strategy: {name}")
        except Exception as e:
            logger.error(f"Failed to register strategy {name}: {e}")

    return scanner


if __name__ == "__main__":
    # Test the scanner
    import sys
    sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    )

    print("="*60)
    print("UNIVERSE SCANNER TEST")
    print("="*60)

    scanner = create_scanner_with_strategies()

    print(f"\nRegistered strategies: {list(scanner.strategies.keys())}")
    print(f"\nStarting full universe scan...")

    signals = scanner.scan()

    print(f"\n{'='*60}")
    print(f"RESULTS: {len(signals)} validated signals")
    print("="*60)

    for sig in signals[:20]:  # Show top 20
        print(f"  {sig.symbol:6} | {sig.strategy:25} | conf={sig.confidence:.2f} -> {sig.phase2_confidence:.2f}")
