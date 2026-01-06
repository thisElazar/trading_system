"""
Strategy Genome
===============
Creative DNA representation for novel strategy discovery.

Goes beyond simple parameter optimization to enable:
- Strategy archetype mixing (momentum + mean reversion hybrids)
- Signal combination and weighting
- Conditional logic evolution
- Multi-timeframe blending
- Risk management gene sequences

This enables the GA to discover genuinely novel strategies,
not just optimize parameters of existing ones.

Usage:
    from research.genetic.strategy_genome import StrategyGenome, GenomeFactory

    factory = GenomeFactory()

    # Create random genome
    genome = factory.create_random()

    # Create strategy from genome
    strategy = genome.to_strategy()

    # Crossover two genomes
    child = factory.crossover(parent1, parent2)
"""

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import numpy as np
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of trading signals."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    TREND_FOLLOWING = "trend_following"
    PAIRS = "pairs"
    REGIME = "regime"


class ConditionType(Enum):
    """Types of conditional logic."""
    ALWAYS = "always"              # Always active
    VIX_BELOW = "vix_below"        # Active when VIX below threshold
    VIX_ABOVE = "vix_above"        # Active when VIX above threshold
    TREND_UP = "trend_up"          # Active in uptrend
    TREND_DOWN = "trend_down"      # Active in downtrend
    HIGH_VOL = "high_vol"          # Active in high volatility
    LOW_VOL = "low_vol"            # Active in low volatility
    BREADTH_STRONG = "breadth_strong"  # Good market breadth
    BREADTH_WEAK = "breadth_weak"      # Weak breadth


class TimeframeType(Enum):
    """Timeframe for signals."""
    INTRADAY = "intraday"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class SignalGene:
    """
    Gene encoding a single signal source.

    Signals are building blocks that can be combined.
    """
    signal_type: SignalType
    weight: float                   # -1 to 1 (negative = contrarian)
    lookback_period: int            # Days for calculation
    threshold: float                # Entry threshold
    timeframe: TimeframeType

    # Additional parameters based on signal type
    params: Dict[str, float] = field(default_factory=dict)

    # Conditional activation
    condition: ConditionType = ConditionType.ALWAYS
    condition_threshold: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'signal_type': self.signal_type.value,
            'weight': self.weight,
            'lookback_period': self.lookback_period,
            'threshold': self.threshold,
            'timeframe': self.timeframe.value,
            'params': self.params,
            'condition': self.condition.value,
            'condition_threshold': self.condition_threshold,
        }

    @staticmethod
    def from_dict(d: Dict) -> 'SignalGene':
        return SignalGene(
            signal_type=SignalType(d['signal_type']),
            weight=d['weight'],
            lookback_period=d['lookback_period'],
            threshold=d['threshold'],
            timeframe=TimeframeType(d['timeframe']),
            params=d.get('params', {}),
            condition=ConditionType(d.get('condition', 'always')),
            condition_threshold=d.get('condition_threshold', 0.0),
        )


@dataclass
class RiskGene:
    """
    Gene encoding risk management rules.
    """
    # Position sizing
    base_position_pct: float        # Base position size (1-5%)
    vol_adjusted: bool              # Adjust for volatility?
    vol_target: float               # Target volatility if adjusted

    # Stop loss
    stop_loss_type: str             # "fixed", "atr", "trailing"
    stop_loss_value: float          # Percentage or ATR multiple
    stop_loss_buffer_days: int      # Days before stop activates

    # Take profit
    take_profit_type: str           # "fixed", "atr", "none"
    take_profit_value: float

    # Time-based
    max_hold_days: int
    time_stop_enabled: bool

    # Correlation/portfolio rules
    max_correlated_positions: int   # Max positions with correlation > 0.7
    max_sector_exposure: float      # Max % in single sector

    def to_dict(self) -> Dict[str, Any]:
        return {
            'base_position_pct': self.base_position_pct,
            'vol_adjusted': self.vol_adjusted,
            'vol_target': self.vol_target,
            'stop_loss_type': self.stop_loss_type,
            'stop_loss_value': self.stop_loss_value,
            'stop_loss_buffer_days': self.stop_loss_buffer_days,
            'take_profit_type': self.take_profit_type,
            'take_profit_value': self.take_profit_value,
            'max_hold_days': self.max_hold_days,
            'time_stop_enabled': self.time_stop_enabled,
            'max_correlated_positions': self.max_correlated_positions,
            'max_sector_exposure': self.max_sector_exposure,
        }


@dataclass
class FilterGene:
    """
    Gene encoding universe filters.
    """
    min_price: float
    max_price: float
    min_volume: float               # Minimum average volume
    min_market_cap: float           # Minimum market cap (billions)
    max_volatility: float           # Maximum volatility
    sector_filter: List[str]        # Allowed sectors (empty = all)
    exclude_recent_ipos: bool       # Exclude stocks < 1 year old
    require_options: bool           # Must have options

    def to_dict(self) -> Dict[str, Any]:
        return {
            'min_price': self.min_price,
            'max_price': self.max_price,
            'min_volume': self.min_volume,
            'min_market_cap': self.min_market_cap,
            'max_volatility': self.max_volatility,
            'sector_filter': self.sector_filter,
            'exclude_recent_ipos': self.exclude_recent_ipos,
            'require_options': self.require_options,
        }


@dataclass
class StrategyGenome:
    """
    Complete genome encoding a trading strategy.

    Combines multiple signal genes with risk and filter genes.
    """
    # Identity
    name: str = ""
    generation: int = 0
    created_at: datetime = field(default_factory=datetime.now)

    # Signal genes (can have 1-5 signals)
    signal_genes: List[SignalGene] = field(default_factory=list)

    # Signal combination method
    combination_method: str = "weighted_average"  # "weighted_average", "vote", "unanimous"
    min_signals_for_entry: int = 1

    # Risk genes
    risk_gene: Optional[RiskGene] = None

    # Filter genes
    filter_gene: Optional[FilterGene] = None

    # Regime adaptation
    regime_adjustments: Dict[str, float] = field(default_factory=dict)

    # Parent tracking
    parent_ids: List[str] = field(default_factory=list)

    # Fitness tracking
    fitness: float = 0.0
    period_scores: Dict[str, float] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Unique identifier."""
        components = [
            "_".join(s.signal_type.value[:3] for s in self.signal_genes),
            f"g{self.generation}",
            self.combination_method[:3],
        ]
        return "_".join(components)

    def get_archetype(self) -> str:
        """Get the dominant strategy archetype."""
        if not self.signal_genes:
            return "unknown"

        # Find dominant signal type by weight
        type_weights = {}
        for gene in self.signal_genes:
            t = gene.signal_type.value
            type_weights[t] = type_weights.get(t, 0) + abs(gene.weight)

        dominant = max(type_weights, key=type_weights.get)
        return dominant

    def is_hybrid(self) -> bool:
        """Check if strategy combines different archetypes."""
        types = set(g.signal_type for g in self.signal_genes)
        return len(types) > 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'id': self.id,
            'generation': self.generation,
            'archetype': self.get_archetype(),
            'is_hybrid': self.is_hybrid(),
            'signal_genes': [g.to_dict() for g in self.signal_genes],
            'combination_method': self.combination_method,
            'min_signals_for_entry': self.min_signals_for_entry,
            'risk_gene': self.risk_gene.to_dict() if self.risk_gene else None,
            'filter_gene': self.filter_gene.to_dict() if self.filter_gene else None,
            'regime_adjustments': self.regime_adjustments,
            'fitness': self.fitness,
        }

    def to_parameter_dict(self) -> Dict[str, float]:
        """Convert genome to flat parameter dictionary for compatibility."""
        params = {}

        if self.signal_genes:
            primary = self.signal_genes[0]
            params['lookback_period'] = float(primary.lookback_period)
            params['entry_threshold'] = primary.threshold
            params['signal_weight'] = primary.weight

            for key, val in primary.params.items():
                params[key] = val

        if self.risk_gene:
            params['position_size_pct'] = self.risk_gene.base_position_pct
            params['stop_loss_pct'] = self.risk_gene.stop_loss_value
            params['max_hold_days'] = float(self.risk_gene.max_hold_days)
            params['vol_target'] = self.risk_gene.vol_target

        return params


class GenomeFactory:
    """
    Factory for creating and evolving StrategyGenomes.
    """

    # Signal type templates
    SIGNAL_TEMPLATES = {
        SignalType.MOMENTUM: {
            'lookback_range': (20, 252),
            'threshold_range': (0.0, 0.3),
            'typical_params': ['skip_period', 'top_percentile'],
        },
        SignalType.MEAN_REVERSION: {
            'lookback_range': (5, 60),
            'threshold_range': (1.5, 3.0),
            'typical_params': ['entry_std', 'exit_std'],
        },
        SignalType.BREAKOUT: {
            'lookback_range': (10, 60),
            'threshold_range': (0.01, 0.05),
            'typical_params': ['min_volume_ratio', 'consolidation_days'],
        },
        SignalType.VOLATILITY: {
            'lookback_range': (10, 60),
            'threshold_range': (0.5, 2.0),
            'typical_params': ['vix_threshold', 'vol_expansion_ratio'],
        },
        SignalType.VOLUME: {
            'lookback_range': (5, 30),
            'threshold_range': (1.5, 3.0),
            'typical_params': ['relative_volume', 'price_confirmation'],
        },
        SignalType.TREND_FOLLOWING: {
            'lookback_range': (50, 200),
            'threshold_range': (0.0, 0.02),
            'typical_params': ['ma_type', 'filter_ma_period'],
        },
    }

    # Condition templates
    CONDITION_TEMPLATES = {
        ConditionType.VIX_BELOW: {'range': (12, 25)},
        ConditionType.VIX_ABOVE: {'range': (20, 40)},
        ConditionType.HIGH_VOL: {'range': (20, 35)},
        ConditionType.LOW_VOL: {'range': (10, 18)},
    }

    def __init__(self, seed: int = None):
        """Initialize factory."""
        if seed:
            random.seed(seed)
            np.random.seed(seed)

    def create_random_signal_gene(
        self,
        signal_type: SignalType = None,
        allow_conditions: bool = True
    ) -> SignalGene:
        """Create a random signal gene."""
        if signal_type is None:
            signal_type = random.choice(list(SignalType))

        template = self.SIGNAL_TEMPLATES.get(signal_type, {
            'lookback_range': (10, 100),
            'threshold_range': (0.5, 2.0),
            'typical_params': [],
        })

        lookback = random.randint(*template['lookback_range'])
        threshold = random.uniform(*template['threshold_range'])
        weight = random.uniform(-1, 1)
        timeframe = random.choice(list(TimeframeType))

        # Generate type-specific params
        params = {}
        for param_name in template.get('typical_params', []):
            params[param_name] = random.uniform(0.1, 2.0)

        # Condition
        condition = ConditionType.ALWAYS
        condition_threshold = 0.0

        if allow_conditions and random.random() < 0.3:
            condition = random.choice(list(ConditionType))
            if condition in self.CONDITION_TEMPLATES:
                thresh_range = self.CONDITION_TEMPLATES[condition]['range']
                condition_threshold = random.uniform(*thresh_range)

        return SignalGene(
            signal_type=signal_type,
            weight=weight,
            lookback_period=lookback,
            threshold=threshold,
            timeframe=timeframe,
            params=params,
            condition=condition,
            condition_threshold=condition_threshold,
        )

    def create_random_risk_gene(self) -> RiskGene:
        """Create a random risk management gene."""
        return RiskGene(
            base_position_pct=random.uniform(0.01, 0.05),
            vol_adjusted=random.random() < 0.6,
            vol_target=random.uniform(0.10, 0.20),
            stop_loss_type=random.choice(["fixed", "atr", "trailing"]),
            stop_loss_value=random.uniform(0.005, 0.03),
            stop_loss_buffer_days=random.randint(0, 3),
            take_profit_type=random.choice(["fixed", "atr", "none"]),
            take_profit_value=random.uniform(0.02, 0.10),
            max_hold_days=random.randint(1, 30),
            time_stop_enabled=random.random() < 0.5,
            max_correlated_positions=random.randint(2, 5),
            max_sector_exposure=random.uniform(0.20, 0.40),
        )

    def create_random_filter_gene(self) -> FilterGene:
        """Create a random filter gene."""
        return FilterGene(
            min_price=random.choice([5, 10, 15, 20]),
            max_price=random.choice([500, 1000, float('inf')]),
            min_volume=random.uniform(100000, 1000000),
            min_market_cap=random.choice([0.5, 1.0, 5.0, 10.0]),
            max_volatility=random.uniform(0.5, 1.5),
            sector_filter=[],  # Empty = all sectors
            exclude_recent_ipos=random.random() < 0.5,
            require_options=random.random() < 0.3,
        )

    def create_random(self, n_signals: int = None) -> StrategyGenome:
        """
        Create a completely random genome.

        Args:
            n_signals: Number of signal genes (1-4, random if None)
        """
        if n_signals is None:
            n_signals = random.randint(1, 3)

        # Create signal genes (ensuring some diversity)
        signal_genes = []
        used_types = set()

        for _ in range(n_signals):
            # Prefer diversity
            available_types = [t for t in SignalType if t not in used_types]
            if not available_types:
                available_types = list(SignalType)

            signal_type = random.choice(available_types)
            used_types.add(signal_type)

            gene = self.create_random_signal_gene(signal_type)
            signal_genes.append(gene)

        # Combination method
        combination = random.choice(["weighted_average", "vote", "unanimous"])
        min_signals = 1 if combination == "weighted_average" else max(1, n_signals // 2)

        # Regime adjustments
        regime_adjustments = {
            'risk_on': random.uniform(0.8, 1.2),
            'risk_off': random.uniform(0.5, 1.0),
            'crisis': random.uniform(0.1, 0.5),
            'transition': random.uniform(0.7, 1.0),
        }

        return StrategyGenome(
            signal_genes=signal_genes,
            combination_method=combination,
            min_signals_for_entry=min_signals,
            risk_gene=self.create_random_risk_gene(),
            filter_gene=self.create_random_filter_gene(),
            regime_adjustments=regime_adjustments,
            generation=0,
        )

    def create_from_archetype(self, archetype: str) -> StrategyGenome:
        """
        Create a genome based on a known strategy archetype.

        Provides a good starting point for evolution.
        """
        archetypes = {
            'momentum': {
                'signals': [SignalType.MOMENTUM],
                'combination': 'weighted_average',
            },
            'mean_reversion': {
                'signals': [SignalType.MEAN_REVERSION],
                'combination': 'weighted_average',
            },
            'momentum_volume': {
                'signals': [SignalType.MOMENTUM, SignalType.VOLUME],
                'combination': 'vote',
            },
            'trend_with_vol_filter': {
                'signals': [SignalType.TREND_FOLLOWING, SignalType.VOLATILITY],
                'combination': 'unanimous',
            },
            'mean_reversion_breakout': {
                'signals': [SignalType.MEAN_REVERSION, SignalType.BREAKOUT],
                'combination': 'weighted_average',
            },
        }

        config = archetypes.get(archetype, archetypes['momentum'])

        signal_genes = [
            self.create_random_signal_gene(st)
            for st in config['signals']
        ]

        return StrategyGenome(
            name=archetype,
            signal_genes=signal_genes,
            combination_method=config['combination'],
            min_signals_for_entry=1,
            risk_gene=self.create_random_risk_gene(),
            filter_gene=self.create_random_filter_gene(),
            generation=0,
        )

    def mutate_signal_gene(
        self,
        gene: SignalGene,
        mutation_strength: float = 0.1
    ) -> SignalGene:
        """Mutate a signal gene."""
        new_gene = SignalGene(
            signal_type=gene.signal_type,
            weight=gene.weight,
            lookback_period=gene.lookback_period,
            threshold=gene.threshold,
            timeframe=gene.timeframe,
            params=gene.params.copy(),
            condition=gene.condition,
            condition_threshold=gene.condition_threshold,
        )

        # Mutate each field with probability
        if random.random() < 0.2:
            new_gene.weight = np.clip(
                gene.weight + random.gauss(0, mutation_strength),
                -1, 1
            )

        if random.random() < 0.2:
            template = self.SIGNAL_TEMPLATES.get(gene.signal_type, {})
            lb_range = template.get('lookback_range', (10, 100))
            delta = int(random.gauss(0, 10 * mutation_strength))
            new_gene.lookback_period = int(np.clip(
                gene.lookback_period + delta,
                lb_range[0], lb_range[1]
            ))

        if random.random() < 0.2:
            new_gene.threshold = max(0, gene.threshold + random.gauss(0, mutation_strength))

        # Occasionally change condition
        if random.random() < 0.1:
            new_gene.condition = random.choice(list(ConditionType))

        return new_gene

    def mutate(
        self,
        genome: StrategyGenome,
        mutation_rate: float = 0.15,
        strength: float = 0.1
    ) -> StrategyGenome:
        """
        Mutate a genome.

        Can mutate:
        - Signal gene parameters
        - Add/remove signal genes
        - Change combination method
        - Risk gene parameters
        """
        new_signals = []

        for gene in genome.signal_genes:
            if random.random() < mutation_rate:
                new_signals.append(self.mutate_signal_gene(gene, strength))
            else:
                new_signals.append(gene)

        # Possibly add a new signal
        if len(new_signals) < 4 and random.random() < 0.1:
            new_signals.append(self.create_random_signal_gene())

        # Possibly remove a signal (keep at least 1)
        if len(new_signals) > 1 and random.random() < 0.1:
            idx = random.randint(0, len(new_signals) - 1)
            new_signals.pop(idx)

        # Mutate risk gene
        new_risk = genome.risk_gene
        if random.random() < mutation_rate and new_risk:
            new_risk = RiskGene(
                base_position_pct=np.clip(
                    new_risk.base_position_pct + random.gauss(0, 0.005),
                    0.01, 0.05
                ),
                vol_adjusted=new_risk.vol_adjusted if random.random() > 0.1 else not new_risk.vol_adjusted,
                vol_target=np.clip(
                    new_risk.vol_target + random.gauss(0, 0.02),
                    0.05, 0.30
                ),
                stop_loss_type=new_risk.stop_loss_type if random.random() > 0.1 else random.choice(["fixed", "atr", "trailing"]),
                stop_loss_value=np.clip(
                    new_risk.stop_loss_value + random.gauss(0, 0.005),
                    0.005, 0.05
                ),
                stop_loss_buffer_days=new_risk.stop_loss_buffer_days,
                take_profit_type=new_risk.take_profit_type,
                take_profit_value=np.clip(
                    new_risk.take_profit_value + random.gauss(0, 0.01),
                    0.01, 0.15
                ),
                max_hold_days=int(np.clip(
                    new_risk.max_hold_days + random.gauss(0, 3),
                    1, 60
                )),
                time_stop_enabled=new_risk.time_stop_enabled,
                max_correlated_positions=new_risk.max_correlated_positions,
                max_sector_exposure=new_risk.max_sector_exposure,
            )

        # Mutate combination method
        new_combination = genome.combination_method
        if random.random() < 0.05:
            new_combination = random.choice(["weighted_average", "vote", "unanimous"])

        return StrategyGenome(
            signal_genes=new_signals,
            combination_method=new_combination,
            min_signals_for_entry=genome.min_signals_for_entry,
            risk_gene=new_risk,
            filter_gene=genome.filter_gene,
            regime_adjustments=genome.regime_adjustments.copy(),
            generation=genome.generation + 1,
            parent_ids=[genome.id],
        )

    def crossover(
        self,
        parent1: StrategyGenome,
        parent2: StrategyGenome
    ) -> StrategyGenome:
        """
        Crossover two genomes.

        Signal genes are mixed, other genes taken from random parent.
        """
        # Mix signal genes
        all_signals = parent1.signal_genes + parent2.signal_genes
        random.shuffle(all_signals)

        # Take 1-3 signals, trying to get diversity
        n_signals = random.randint(1, min(3, len(all_signals)))
        new_signals = []
        used_types = set()

        for signal in all_signals:
            if len(new_signals) >= n_signals:
                break
            if signal.signal_type not in used_types:
                new_signals.append(signal)
                used_types.add(signal.signal_type)

        # Ensure at least one signal
        if not new_signals:
            new_signals = [all_signals[0]] if all_signals else [self.create_random_signal_gene()]

        # Take other genes from random parent
        risk_parent = random.choice([parent1, parent2])
        filter_parent = random.choice([parent1, parent2])
        combo_parent = random.choice([parent1, parent2])

        # Blend regime adjustments
        regime_adjustments = {}
        for key in ['risk_on', 'risk_off', 'crisis', 'transition']:
            val1 = parent1.regime_adjustments.get(key, 1.0)
            val2 = parent2.regime_adjustments.get(key, 1.0)
            regime_adjustments[key] = (val1 + val2) / 2

        return StrategyGenome(
            signal_genes=new_signals,
            combination_method=combo_parent.combination_method,
            min_signals_for_entry=combo_parent.min_signals_for_entry,
            risk_gene=risk_parent.risk_gene,
            filter_gene=filter_parent.filter_gene,
            regime_adjustments=regime_adjustments,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.id, parent2.id],
        )

    def creative_mutation(self, genome: StrategyGenome) -> StrategyGenome:
        """
        Apply creative mutation for novel strategy discovery.

        More aggressive than standard mutation - explores new territory.
        """
        mutation_type = random.choice([
            'signal_swap',      # Replace a signal with completely different type
            'invert_signal',    # Flip signal weight (contrarian)
            'add_condition',    # Add conditional activation
            'archetype_blend',  # Blend with a different archetype
        ])

        if mutation_type == 'signal_swap' and genome.signal_genes:
            new_signals = genome.signal_genes.copy()
            idx = random.randint(0, len(new_signals) - 1)

            # Pick a different signal type
            current_types = {g.signal_type for g in new_signals}
            available = [t for t in SignalType if t not in current_types]
            if available:
                new_type = random.choice(available)
                new_signals[idx] = self.create_random_signal_gene(new_type)

            return StrategyGenome(
                signal_genes=new_signals,
                combination_method=genome.combination_method,
                min_signals_for_entry=genome.min_signals_for_entry,
                risk_gene=genome.risk_gene,
                filter_gene=genome.filter_gene,
                regime_adjustments=genome.regime_adjustments,
                generation=genome.generation + 1,
                parent_ids=[genome.id],
            )

        elif mutation_type == 'invert_signal' and genome.signal_genes:
            new_signals = genome.signal_genes.copy()
            idx = random.randint(0, len(new_signals) - 1)
            old_gene = new_signals[idx]
            new_signals[idx] = SignalGene(
                signal_type=old_gene.signal_type,
                weight=-old_gene.weight,  # Invert!
                lookback_period=old_gene.lookback_period,
                threshold=old_gene.threshold,
                timeframe=old_gene.timeframe,
                params=old_gene.params,
                condition=old_gene.condition,
                condition_threshold=old_gene.condition_threshold,
            )

            return StrategyGenome(
                signal_genes=new_signals,
                combination_method=genome.combination_method,
                min_signals_for_entry=genome.min_signals_for_entry,
                risk_gene=genome.risk_gene,
                filter_gene=genome.filter_gene,
                regime_adjustments=genome.regime_adjustments,
                generation=genome.generation + 1,
                parent_ids=[genome.id],
            )

        elif mutation_type == 'add_condition' and genome.signal_genes:
            new_signals = genome.signal_genes.copy()
            idx = random.randint(0, len(new_signals) - 1)
            old_gene = new_signals[idx]

            new_condition = random.choice([c for c in ConditionType if c != ConditionType.ALWAYS])
            new_threshold = 0.0
            if new_condition in self.CONDITION_TEMPLATES:
                thresh_range = self.CONDITION_TEMPLATES[new_condition]['range']
                new_threshold = random.uniform(*thresh_range)

            new_signals[idx] = SignalGene(
                signal_type=old_gene.signal_type,
                weight=old_gene.weight,
                lookback_period=old_gene.lookback_period,
                threshold=old_gene.threshold,
                timeframe=old_gene.timeframe,
                params=old_gene.params,
                condition=new_condition,
                condition_threshold=new_threshold,
            )

            return StrategyGenome(
                signal_genes=new_signals,
                combination_method=genome.combination_method,
                min_signals_for_entry=genome.min_signals_for_entry,
                risk_gene=genome.risk_gene,
                filter_gene=genome.filter_gene,
                regime_adjustments=genome.regime_adjustments,
                generation=genome.generation + 1,
                parent_ids=[genome.id],
            )

        else:  # archetype_blend
            archetype = random.choice(['momentum', 'mean_reversion', 'trend_with_vol_filter'])
            other_genome = self.create_from_archetype(archetype)
            return self.crossover(genome, other_genome)


# =============================================================================
# CLI Demo
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 60)
    print("STRATEGY GENOME DEMO")
    print("=" * 60)

    factory = GenomeFactory()

    # Create random genome
    print("\nCreating random genome...")
    genome = factory.create_random(n_signals=2)
    print(f"  Archetype: {genome.get_archetype()}")
    print(f"  Is hybrid: {genome.is_hybrid()}")
    print(f"  Signals: {[g.signal_type.value for g in genome.signal_genes]}")
    print(f"  Combination: {genome.combination_method}")

    # Create from archetype
    print("\nCreating momentum genome...")
    momentum_genome = factory.create_from_archetype('momentum')
    print(f"  Signals: {[g.signal_type.value for g in momentum_genome.signal_genes]}")

    # Mutate
    print("\nMutating genome...")
    mutated = factory.mutate(genome, mutation_rate=0.3)
    print(f"  Original archetype: {genome.get_archetype()}")
    print(f"  Mutated archetype: {mutated.get_archetype()}")

    # Crossover
    print("\nCrossing genomes...")
    child = factory.crossover(genome, momentum_genome)
    print(f"  Child signals: {[g.signal_type.value for g in child.signal_genes]}")
    print(f"  Child is hybrid: {child.is_hybrid()}")

    # Creative mutation
    print("\nApplying creative mutation...")
    creative = factory.creative_mutation(genome)
    print(f"  Result signals: {[g.signal_type.value for g in creative.signal_genes]}")

    # Convert to parameters
    print("\nConverting to flat parameters...")
    params = genome.to_parameter_dict()
    print(f"  Parameters: {params}")

    print("\n" + "=" * 60)
