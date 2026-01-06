# Autonomous Strategy Discovery Engine - Architecture

## Executive Summary

This document outlines the architecture for transforming the existing parameter-based genetic algorithm optimizer into a full genetic programming (GP) strategy discovery engine capable of autonomous overnight operation.

**Key Transformation:**
- **Current:** Optimizes parameters of hand-coded strategies
- **Target:** Discovers entirely new trading logic through tree-based genetic programming

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Autonomous Strategy Discovery Engine             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐ │
│  │   GP Core   │───▶│  Evaluator  │───▶│  Multi-Objective Fitness │ │
│  │  (DEAP)     │    │  (Backtest) │    │  (NSGA-II + Novelty)     │ │
│  └─────────────┘    └─────────────┘    └─────────────────────────┘ │
│         │                                         │                 │
│         ▼                                         ▼                 │
│  ┌─────────────┐                        ┌─────────────────────────┐ │
│  │  Strategy   │                        │    Archive Manager      │ │
│  │  Compiler   │                        │  (Novelty + Pareto)     │ │
│  └─────────────┘                        └─────────────────────────┘ │
│         │                                         │                 │
│         ▼                                         ▼                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  Persistence Layer (SQLite)                  │   │
│  │  - Population state    - Evolution history   - Discoveries   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### 1. Genetic Programming Core (`gp_core.py`)

Uses DEAP (Distributed Evolutionary Algorithms in Python) for tree-based GP.

```python
# Genome Structure: Expression Trees for Trading Rules
@dataclass
class StrategyGenome:
    """Complete genome for an evolved strategy."""
    entry_tree: PrimitiveTree      # Boolean: when to enter
    exit_tree: PrimitiveTree       # Boolean: when to exit
    position_tree: PrimitiveTree   # Float: position size (0-1)
    stop_loss_tree: PrimitiveTree  # Float: stop loss % from entry
    target_tree: PrimitiveTree     # Float: target % from entry

    # Metadata
    generation: int
    genome_id: str
    parent_ids: List[str]
```

**Primitive Set (Function/Terminal Nodes):**

```python
# Terminal Nodes (Leaves)
terminals = {
    # Price data (returns float)
    'open': lambda data: data['open'].iloc[-1],
    'high': lambda data: data['high'].iloc[-1],
    'low': lambda data: data['low'].iloc[-1],
    'close': lambda data: data['close'].iloc[-1],
    'volume': lambda data: data['volume'].iloc[-1],

    # Technical indicators (returns float)
    'sma_5': lambda data: ta.trend.sma_indicator(data['close'], 5).iloc[-1],
    'sma_20': lambda data: ta.trend.sma_indicator(data['close'], 20).iloc[-1],
    'sma_50': lambda data: ta.trend.sma_indicator(data['close'], 50).iloc[-1],
    'ema_12': lambda data: ta.trend.ema_indicator(data['close'], 12).iloc[-1],
    'ema_26': lambda data: ta.trend.ema_indicator(data['close'], 26).iloc[-1],
    'rsi_14': lambda data: ta.momentum.rsi(data['close'], 14).iloc[-1],
    'macd': lambda data: ta.trend.macd_diff(data['close']).iloc[-1],
    'atr_14': lambda data: ta.volatility.average_true_range(data['high'], data['low'], data['close'], 14).iloc[-1],
    'bbands_upper': lambda data: ta.volatility.bollinger_hband(data['close'], 20).iloc[-1],
    'bbands_lower': lambda data: ta.volatility.bollinger_lband(data['close'], 20).iloc[-1],
    'volume_sma': lambda data: data['volume'].rolling(20).mean().iloc[-1],

    # Historical lookback (returns float)
    'high_5d': lambda data: data['high'].tail(5).max(),
    'low_5d': lambda data: data['low'].tail(5).min(),
    'high_20d': lambda data: data['high'].tail(20).max(),
    'low_20d': lambda data: data['low'].tail(20).min(),

    # Constants (ephemeral random constants - ERC)
    'const_small': lambda: random.uniform(0.001, 0.05),   # 0.1% - 5%
    'const_medium': lambda: random.uniform(0.05, 0.20),   # 5% - 20%
    'const_int': lambda: random.randint(5, 50),           # Integer periods
}

# Function Nodes
functions = {
    # Arithmetic (float, float) -> float
    'add': operator.add,
    'sub': operator.sub,
    'mul': operator.mul,
    'protected_div': lambda a, b: a / b if abs(b) > 1e-6 else 1.0,

    # Comparison (float, float) -> bool
    'gt': operator.gt,   # >
    'lt': operator.lt,   # <
    'ge': operator.ge,   # >=
    'le': operator.le,   # <=

    # Logical (bool, bool) -> bool
    'and_': operator.and_,
    'or_': operator.or_,
    'not_': operator.not_,

    # Conditional (bool, float, float) -> float
    'if_then_else': lambda cond, a, b: a if cond else b,

    # Unary (float) -> float
    'neg': operator.neg,
    'abs': abs,
    'log': lambda x: math.log(max(x, 1e-6)),
    'sqrt': lambda x: math.sqrt(max(x, 0)),
}
```

### 2. Strategy Compiler (`strategy_compiler.py`)

Transforms GP trees into executable strategy objects.

```python
class EvolvedStrategy(BaseStrategy):
    """Strategy generated from GP genome."""

    def __init__(self, genome: StrategyGenome):
        super().__init__(name=f"evolved_{genome.genome_id}")
        self.genome = genome
        self._entry_fn = compile_tree(genome.entry_tree)
        self._exit_fn = compile_tree(genome.exit_tree)
        self._position_fn = compile_tree(genome.position_tree)
        self._stop_fn = compile_tree(genome.stop_loss_tree)
        self._target_fn = compile_tree(genome.target_tree)

    def generate_signals(self, data, current_positions, vix_regime):
        signals = []
        for symbol, df in data.items():
            if len(df) < 50:  # Need enough history
                continue

            try:
                # Evaluate entry condition
                if self._entry_fn(df):
                    price = df['close'].iloc[-1]
                    position_pct = max(0.01, min(0.2, self._position_fn(df)))
                    stop_pct = max(0.01, min(0.15, self._stop_fn(df)))
                    target_pct = max(0.02, min(0.30, self._target_fn(df)))

                    signals.append(Signal(
                        timestamp=df.index[-1],
                        symbol=symbol,
                        strategy=self.name,
                        signal_type=SignalType.BUY,
                        strength=0.7,
                        price=price,
                        stop_loss=price * (1 - stop_pct),
                        target_price=price * (1 + target_pct),
                        position_size_pct=position_pct
                    ))
            except Exception:
                continue  # Skip malformed strategies

        return signals
```

### 3. Multi-Objective Fitness (`multi_objective.py`)

Implements NSGA-II with custom objectives.

```python
@dataclass
class FitnessVector:
    """Multi-objective fitness values."""
    sortino: float           # Risk-adjusted return (downside)
    max_drawdown: float      # Negative (minimize)
    cvar_95: float           # Conditional VaR (minimize)
    novelty: float           # Diversity score (maximize)
    deflated_sharpe: float   # Multiple-testing corrected

    # Derived
    trades: int              # For constraint checking

    def dominates(self, other: 'FitnessVector') -> bool:
        """Check if this solution dominates another (Pareto)."""
        dominated = False
        strictly_better = False

        # For sortino, novelty, deflated_sharpe: higher is better
        # For max_drawdown, cvar_95: lower (more negative) is worse

        objectives = [
            (self.sortino, other.sortino, True),           # maximize
            (self.max_drawdown, other.max_drawdown, False), # minimize (less negative)
            (self.cvar_95, other.cvar_95, False),          # minimize
            (self.novelty, other.novelty, True),           # maximize
        ]

        for mine, theirs, maximize in objectives:
            if maximize:
                if mine < theirs:
                    return False
                if mine > theirs:
                    strictly_better = True
            else:
                if mine > theirs:  # For minimization
                    return False
                if mine < theirs:
                    strictly_better = True

        return strictly_better


def calculate_fitness_vector(
    result: BacktestResult,
    novelty_archive: 'NoveltyArchive',
    total_trials: int
) -> FitnessVector:
    """Calculate multi-objective fitness vector."""

    # Sortino Ratio (downside risk-adjusted)
    sortino = result.sortino_ratio

    # Maximum Drawdown (as negative percentage)
    max_dd = result.max_drawdown_pct  # Already negative

    # CVaR at 95% (Expected Shortfall)
    returns = pd.Series(result.equity_curve).pct_change().dropna()
    var_95 = returns.quantile(0.05)
    cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95

    # Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
    # Corrects for multiple testing / selection bias
    sharpe = result.sharpe_ratio
    skew = returns.skew() if len(returns) > 0 else 0
    kurt = returns.kurtosis() if len(returns) > 0 else 0
    n = len(returns)

    # Standard error of Sharpe
    se_sharpe = np.sqrt((1 + 0.5 * sharpe**2 - skew * sharpe + (kurt - 3) / 4 * sharpe**2) / n) if n > 0 else 1

    # Expected max Sharpe under null hypothesis (multiple testing)
    expected_max_sharpe = (1 - np.euler_gamma) * norm.ppf(1 - 1 / total_trials) + \
                          np.euler_gamma * norm.ppf(1 - 1 / (total_trials * np.e))

    # Deflated Sharpe
    deflated = norm.cdf((sharpe - expected_max_sharpe) / se_sharpe)

    # Novelty Score (behavioral diversity)
    behavior = extract_behavior_vector(result)
    novelty = novelty_archive.calculate_novelty(behavior)

    return FitnessVector(
        sortino=sortino,
        max_drawdown=max_dd,
        cvar_95=cvar_95,
        novelty=novelty,
        deflated_sharpe=deflated,
        trades=result.total_trades
    )
```

### 4. Novelty Search (`novelty_search.py`)

Maintains behavioral diversity to prevent convergence.

```python
@dataclass
class BehaviorVector:
    """Characterizes strategy behavior for novelty comparison."""
    trade_frequency: float      # Trades per week
    avg_hold_period: float      # Days (log-normalized)
    long_short_ratio: float     # -1 to +1
    return_autocorr: float      # Lag-1 autocorrelation
    drawdown_depth: float       # Normalized 0-1
    benchmark_corr: float       # Correlation to SPY
    signal_variance: float      # Volatility of position changes

    def to_array(self) -> np.ndarray:
        return np.array([
            self.trade_frequency,
            self.avg_hold_period,
            self.long_short_ratio,
            self.return_autocorr,
            self.drawdown_depth,
            self.benchmark_corr,
            self.signal_variance
        ])


class NoveltyArchive:
    """Maintains archive of novel behaviors for diversity."""

    def __init__(self, k_neighbors: int = 20, archive_size: int = 500):
        self.k = k_neighbors
        self.max_size = archive_size
        self.archive: List[BehaviorVector] = []
        self.fitness_threshold = 0.0  # Minimum fitness to consider

    def calculate_novelty(self, behavior: BehaviorVector) -> float:
        """Calculate novelty as average distance to k-nearest neighbors."""
        if len(self.archive) < self.k:
            return 1.0  # Everything is novel when archive is small

        behavior_array = behavior.to_array()
        distances = []

        for archived in self.archive:
            dist = np.linalg.norm(behavior_array - archived.to_array())
            distances.append(dist)

        distances.sort()
        return np.mean(distances[:self.k])

    def maybe_add(self, behavior: BehaviorVector, novelty: float, fitness: float):
        """Probabilistically add to archive based on novelty."""
        if fitness < self.fitness_threshold:
            return

        # Add if archive not full
        if len(self.archive) < self.max_size:
            self.archive.append(behavior)
            return

        # Replace least novel if this is more novel
        novelties = [self.calculate_novelty(b) for b in self.archive]
        min_idx = np.argmin(novelties)

        if novelty > novelties[min_idx]:
            self.archive[min_idx] = behavior


def extract_behavior_vector(result: BacktestResult) -> BehaviorVector:
    """Extract behavioral characteristics from backtest result."""
    trades = result.trades
    equity_curve = pd.Series(result.equity_curve)

    # Trade frequency (trades per week over backtest period)
    days = len(equity_curve)
    weeks = days / 5
    trade_freq = len(trades) / max(weeks, 1)

    # Average holding period (log-normalized)
    if trades:
        hold_periods = []
        for t in trades:
            entry = pd.Timestamp(t['entry_date'])
            exit = pd.Timestamp(t['exit_date'])
            hold_periods.append((exit - entry).days)
        avg_hold = np.log1p(np.mean(hold_periods)) if hold_periods else 0
    else:
        avg_hold = 0

    # Long/short ratio (currently all long)
    long_short = 1.0  # TODO: Support shorts

    # Return autocorrelation
    returns = equity_curve.pct_change().dropna()
    autocorr = returns.autocorr(lag=1) if len(returns) > 1 else 0

    # Drawdown depth (normalized)
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    dd_depth = abs(drawdown.min())

    # Signal variance
    signal_var = returns.std() if len(returns) > 0 else 0

    return BehaviorVector(
        trade_frequency=trade_freq,
        avg_hold_period=avg_hold,
        long_short_ratio=long_short,
        return_autocorr=autocorr if not np.isnan(autocorr) else 0,
        drawdown_depth=min(dd_depth, 1.0),
        benchmark_corr=0.0,  # TODO: Calculate vs SPY
        signal_variance=signal_var
    )
```

### 5. NSGA-II Optimizer (`nsga2.py`)

Multi-objective evolutionary algorithm.

```python
class NSGA2Optimizer:
    """
    Non-dominated Sorting Genetic Algorithm II.

    Features:
    - Non-dominated sorting for Pareto ranking
    - Crowding distance for diversity
    - Binary tournament selection
    - Custom GP crossover and mutation
    """

    def __init__(
        self,
        population_size: int = 100,
        generations: int = 50,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.2
    ):
        self.pop_size = population_size
        self.generations = generations
        self.cx_rate = crossover_rate
        self.mut_rate = mutation_rate

        self.population: List[StrategyGenome] = []
        self.pareto_front: List[StrategyGenome] = []
        self.novelty_archive = NoveltyArchive()

    def non_dominated_sort(
        self,
        individuals: List[Tuple[StrategyGenome, FitnessVector]]
    ) -> List[List[int]]:
        """Sort population into non-dominated fronts."""
        n = len(individuals)
        domination_count = [0] * n
        dominated_by = [[] for _ in range(n)]
        fronts = [[]]

        for i in range(n):
            for j in range(i + 1, n):
                if individuals[i][1].dominates(individuals[j][1]):
                    dominated_by[i].append(j)
                    domination_count[j] += 1
                elif individuals[j][1].dominates(individuals[i][1]):
                    dominated_by[j].append(i)
                    domination_count[i] += 1

        # First front: individuals not dominated by anyone
        for i in range(n):
            if domination_count[i] == 0:
                fronts[0].append(i)

        # Build subsequent fronts
        current_front = 0
        while fronts[current_front]:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_by[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front += 1
            if next_front:
                fronts.append(next_front)

        return fronts[:-1] if fronts[-1] == [] else fronts

    def crowding_distance(
        self,
        front: List[int],
        fitness_vectors: List[FitnessVector]
    ) -> Dict[int, float]:
        """Calculate crowding distance for diversity preservation."""
        n = len(front)
        if n <= 2:
            return {i: float('inf') for i in front}

        distances = {i: 0.0 for i in front}
        objectives = ['sortino', 'max_drawdown', 'cvar_95', 'novelty']

        for obj in objectives:
            # Sort by objective
            sorted_front = sorted(front, key=lambda i: getattr(fitness_vectors[i], obj))

            # Boundary points get infinite distance
            distances[sorted_front[0]] = float('inf')
            distances[sorted_front[-1]] = float('inf')

            # Normalize range
            f_min = getattr(fitness_vectors[sorted_front[0]], obj)
            f_max = getattr(fitness_vectors[sorted_front[-1]], obj)

            if f_max - f_min > 0:
                for k in range(1, n - 1):
                    prev_val = getattr(fitness_vectors[sorted_front[k-1]], obj)
                    next_val = getattr(fitness_vectors[sorted_front[k+1]], obj)
                    distances[sorted_front[k]] += (next_val - prev_val) / (f_max - f_min)

        return distances

    def select_parents(
        self,
        population: List[StrategyGenome],
        fitness_vectors: List[FitnessVector],
        fronts: List[List[int]],
        crowding: Dict[int, float]
    ) -> List[StrategyGenome]:
        """Binary tournament selection based on rank and crowding."""
        selected = []

        # Assign ranks
        ranks = {}
        for rank, front in enumerate(fronts):
            for idx in front:
                ranks[idx] = rank

        for _ in range(self.pop_size):
            # Tournament
            i, j = random.sample(range(len(population)), 2)

            # Prefer lower rank (earlier front)
            if ranks[i] < ranks[j]:
                winner = i
            elif ranks[j] < ranks[i]:
                winner = j
            else:
                # Same rank: prefer higher crowding distance
                winner = i if crowding.get(i, 0) > crowding.get(j, 0) else j

            selected.append(population[winner])

        return selected
```

### 6. Island Model (`island_model.py`)

Parallel subpopulation evolution to prevent premature convergence.

```python
class IslandEvolutionEngine:
    """
    Island-based parallel evolution for diversity preservation.

    Features:
    - Multiple independent subpopulations (islands)
    - Periodic migration between islands
    - Per-island parameter variation (mutation rate, tree depth)
    - Global Pareto front tracking
    - Integrated diversity monitoring and MAP-Elites
    """

    def __init__(
        self,
        config: EvolutionConfig,
        island_config: IslandConfig,
        enable_diversity_monitor: bool = True,
        enable_map_elites: bool = True
    ):
        self.islands: List[Island] = []
        self.diversity_monitor = DiversityMonitor() if enable_diversity_monitor else None
        self.map_elites = create_default_grid() if enable_map_elites else None

    def evolve_generation(self):
        """Evolve all islands with migration and diversity checks."""
        # Evolve each island independently
        for island in self.islands:
            self.evolve_island(island)

        # Periodic migration (ring/random/full topology)
        if self.current_generation % self.island_config.migration_interval == 0:
            self.migrate()

        # Update MAP-Elites grid
        if self.map_elites:
            for island in self.islands:
                for genome in island.population:
                    self.map_elites.maybe_add(genome_id, fitness, behavior)

        # Check diversity and auto-intervene
        if self.diversity_monitor:
            should_intervene, reason = self.diversity_monitor.update(...)
            if should_intervene:
                self._inject_diversity_all_islands()


@dataclass
class IslandConfig:
    """Configuration for island-based evolution."""
    num_islands: int = 4                    # Number of subpopulations
    population_per_island: int = 20         # Individuals per island
    migration_interval: int = 5             # Generations between migrations
    migration_rate: float = 0.15            # Fraction to migrate
    topology: str = "ring"                  # ring, random, or full
    vary_mutation_rate: bool = True         # Each island gets different rate
    vary_tree_depth: bool = True            # Each island explores different depths
```

**Migration Topologies:**
- **Ring:** Island i sends to island (i+1) mod n - structured flow
- **Random:** Random pairs exchange migrants - unpredictable exploration
- **Full:** Best from each island goes to all others - maximum information flow

**Per-Island Variation:**
```python
# Islands explore different regions of the search space
Island 0: mutation=0.10, depth=3  # Conservative, simple strategies
Island 1: mutation=0.16, depth=4  # Moderate complexity
Island 2: mutation=0.22, depth=5  # Higher exploration
Island 3: mutation=0.28, depth=6  # Complex strategies
Island 4: mutation=0.34, depth=3  # High mutation, simple
Island 5: mutation=0.40, depth=4  # Maximum exploration
```

---

### 7. Diversity Metrics (`diversity_metrics.py`)

Comprehensive diversity monitoring with automatic convergence detection.

```python
@dataclass
class GenotypeMetrics:
    """Metrics describing genetic diversity of a population."""
    avg_tree_size: float          # Average nodes per tree
    tree_size_variance: float     # Variance in tree sizes
    avg_tree_depth: float         # Average tree depth
    operator_entropy: float       # Shannon entropy of operator usage (0-1)
    unique_operators: int         # Number of distinct operators
    avg_edit_distance: float      # Average tree edit distance
    unique_tree_ratio: float      # Fraction of unique tree structures
    genotype_diversity: float     # Overall diversity score (0-1)


@dataclass
class DiversityThresholds:
    """Thresholds for triggering diversity intervention."""
    min_genotype_diversity: float = 0.25      # Below this = inject diversity
    min_phenotype_diversity: float = 0.10     # Behavioral diversity minimum
    min_unique_tree_ratio: float = 0.20       # At least 20% unique trees
    min_operator_entropy: float = 0.30        # Operator usage entropy
    stagnation_generations: int = 10          # Generations without improvement
    injection_ratio: float = 0.30             # Replace 30% on intervention
    mutation_rate_boost: float = 2.0          # Multiply mutation rate
    boost_duration: int = 5                   # Generations to maintain boost


class DiversityMonitor:
    """
    Monitors population diversity and triggers automatic intervention.

    Tracks:
    - Genotype diversity (tree structure similarity, operator entropy)
    - Phenotype diversity (behavioral characterization)
    - Fitness stagnation (generations without improvement)

    Actions when thresholds breached:
    - Inject random individuals into population
    - Boost mutation rate temporarily
    - Log intervention for analysis
    """

    def update(self, generation, genomes, phenotype_diversity,
               archive_diversity, best_fitness) -> Tuple[bool, str]:
        """Check diversity and return (should_intervene, reason)."""

        # Calculate genotype metrics
        metrics = calculate_genotype_diversity(genomes)

        # Check all thresholds
        if metrics.genotype_diversity < self.thresholds.min_genotype_diversity:
            return True, f"genotype_div={metrics.genotype_diversity:.3f}"

        if phenotype_diversity < self.thresholds.min_phenotype_diversity:
            return True, f"phenotype_div={phenotype_diversity:.3f}"

        if self.generations_since_improvement >= self.thresholds.stagnation_generations:
            return True, f"stagnation={self.generations_since_improvement}gen"

        return False, "none"
```

**Key Metrics:**
- **Operator Entropy:** Shannon entropy of GP operator usage - low entropy means few operators dominate
- **Unique Tree Ratio:** Fraction of structurally unique trees - low ratio indicates convergence
- **Genotype Diversity:** Weighted combination of structural metrics
- **Phenotype Diversity:** Behavioral diversity from novelty search (7D behavior vector)

---

### 8. MAP-Elites Grid (`map_elites.py`)

Quality-Diversity optimization through behavioral space discretization.

```python
@dataclass
class MapElitesConfig:
    """Configuration for MAP-Elites grid."""
    dimensions: List[str] = ['trade_frequency', 'avg_hold_period', 'drawdown_depth']
    resolution: int = 10  # Bins per dimension (10^3 = 1000 cells)
    feature_ranges: Dict[str, Tuple[float, float]] = {
        'trade_frequency': (0.0, 10.0),     # Trades per week
        'avg_hold_period': (0.0, 30.0),     # Days
        'drawdown_depth': (0.0, 0.5),       # 0-50% drawdown
    }


class MAPElitesGrid:
    """
    Maintains a grid of elite solutions across behavioral feature space.

    Each cell represents a specific trading style (niche).
    Only the best-performing solution per niche is kept.
    Guarantees diversity across different behavioral profiles.

    Default: 3D grid with 10 bins = 1,000 behavioral niches
    """

    def maybe_add(self, genome_id, fitness, behavior, generation) -> bool:
        """
        Add solution if:
        1. Cell is empty, OR
        2. Solution has better fitness than current occupant
        """
        cell = self._behavior_to_cell(behavior)

        if cell not in self.grid or fitness > self.grid[cell].fitness:
            self.grid[cell] = EliteEntry(genome_id, fitness, behavior, cell)
            return True
        return False

    def get_coverage(self) -> float:
        """Fraction of cells occupied (0-1)."""
        return len(self.grid) / self.total_cells

    def sample_elites(self, n, method='uniform') -> List[EliteEntry]:
        """Sample elites for parent selection or analysis."""
        # Methods: 'uniform', 'quality' (fitness-weighted), 'sparse' (favor stable cells)
```

**Behavioral Dimensions:**
| Dimension | Range | Description |
|-----------|-------|-------------|
| trade_frequency | 0-10/week | Activity level |
| avg_hold_period | 0-30 days | Timeframe |
| drawdown_depth | 0-50% | Risk tolerance |

**Integration:**
- Updated every generation from all evaluated genomes
- Provides structured exploration of behavioral space
- Coverage metric indicates exploration breadth
- Elites can seed new populations or provide diverse parents

---

### 9. Continuous Evolution Engine (`evolution_engine.py`)

Orchestrates overnight autonomous operation.

```python
class EvolutionEngine:
    """
    Autonomous evolution engine for continuous strategy discovery.

    Features:
    - Checkpoint persistence every N generations
    - Regime change detection and adaptation
    - Progress monitoring and alerting
    - Graceful shutdown handling
    """

    def __init__(
        self,
        data_loader,
        backtester: Backtester,
        db: DatabaseManager,
        config: EvolutionConfig
    ):
        self.data_loader = data_loader
        self.backtester = backtester
        self.db = db
        self.config = config

        self.nsga2 = NSGA2Optimizer(
            population_size=config.population_size,
            generations=config.generations_per_session,
            crossover_rate=config.crossover_rate,
            mutation_rate=config.mutation_rate
        )

        self.current_generation = 0
        self.total_strategies_discovered = 0
        self.shutdown_requested = False

    def run_overnight(self, hours: float = 8.0):
        """
        Run evolution overnight for specified hours.

        Automatically:
        - Saves checkpoints every 10 generations
        - Adapts to regime changes
        - Handles errors gracefully
        - Reports progress
        """
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=hours)

        logger.info(f"Starting overnight evolution: {start_time} to {end_time}")

        # Load or initialize population
        self._load_checkpoint()

        # Load market data
        data = self.data_loader.load_all_symbols()
        vix_data = self.data_loader.load_vix()

        # Track current regime
        current_regime = self._detect_regime(vix_data)

        generation = 0
        while datetime.now() < end_time and not self.shutdown_requested:
            try:
                # Check for regime change
                new_regime = self._detect_regime(vix_data)
                if new_regime != current_regime:
                    logger.warning(f"Regime change detected: {current_regime} -> {new_regime}")
                    self._handle_regime_change(new_regime)
                    current_regime = new_regime

                # Evolve one generation
                self._evolve_generation(data, vix_data)
                generation += 1
                self.current_generation += 1

                # Checkpoint every 10 generations
                if generation % 10 == 0:
                    self._save_checkpoint()
                    self._log_progress()

                # Check for convergence
                if self._population_converged():
                    logger.info("Population converged, injecting diversity")
                    self._inject_diversity()

            except Exception as e:
                logger.error(f"Generation {generation} failed: {e}")
                self._save_checkpoint()  # Save before potential crash
                continue

        # Final save
        self._save_checkpoint()
        self._generate_report()

    def _handle_regime_change(self, new_regime: str):
        """Adapt population to regime change."""
        # Inject 30% random individuals
        inject_count = int(self.nsga2.pop_size * 0.3)
        for i in range(inject_count):
            new_genome = self._create_random_genome()
            # Replace lowest-ranked individuals
            self.nsga2.population[-(i+1)] = new_genome

        # Reset novelty archive partially
        self.nsga2.novelty_archive.archive = \
            self.nsga2.novelty_archive.archive[:len(self.nsga2.novelty_archive.archive)//2]

        logger.info(f"Injected {inject_count} random individuals for regime adaptation")

    def _population_converged(self) -> bool:
        """Check if population has converged (lost diversity)."""
        if len(self.nsga2.population) < 10:
            return False

        # Check behavioral diversity
        behaviors = [extract_behavior_vector(g.last_result)
                    for g in self.nsga2.population if hasattr(g, 'last_result')]

        if len(behaviors) < 5:
            return False

        # Calculate pairwise distances
        distances = []
        for i, b1 in enumerate(behaviors):
            for b2 in behaviors[i+1:]:
                dist = np.linalg.norm(b1.to_array() - b2.to_array())
                distances.append(dist)

        avg_distance = np.mean(distances) if distances else 0
        return avg_distance < 0.1  # Threshold for convergence

    def _save_checkpoint(self):
        """Save current evolution state to database."""
        checkpoint = {
            'generation': self.current_generation,
            'population': [self._serialize_genome(g) for g in self.nsga2.population],
            'pareto_front': [self._serialize_genome(g) for g in self.nsga2.pareto_front],
            'novelty_archive': self._serialize_archive(),
            'config': asdict(self.config)
        }

        self.db.save_evolution_checkpoint(
            checkpoint_id=f"evo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            data=checkpoint
        )

    def _load_checkpoint(self):
        """Load most recent checkpoint if available."""
        checkpoint = self.db.load_latest_evolution_checkpoint()

        if checkpoint:
            self.current_generation = checkpoint['generation']
            self.nsga2.population = [
                self._deserialize_genome(g) for g in checkpoint['population']
            ]
            self.nsga2.pareto_front = [
                self._deserialize_genome(g) for g in checkpoint['pareto_front']
            ]
            self._deserialize_archive(checkpoint['novelty_archive'])

            logger.info(f"Resumed from generation {self.current_generation}")
        else:
            self._initialize_population()
            logger.info("Starting fresh evolution")
```

---

## Database Schema Extensions

Add to `research.db`:

```sql
-- Evolution checkpoints for continuous operation
CREATE TABLE IF NOT EXISTS evolution_checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    checkpoint_id TEXT NOT NULL UNIQUE,
    generation INTEGER NOT NULL,
    population_json TEXT NOT NULL,        -- Serialized genomes
    pareto_front_json TEXT,               -- Current Pareto front
    novelty_archive_json TEXT,            -- Serialized archive
    config_json TEXT,                     -- Evolution config
    regime_state TEXT,                    -- Market regime at checkpoint
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Discovered strategies (promoted from evolution)
CREATE TABLE IF NOT EXISTS discovered_strategies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id TEXT NOT NULL UNIQUE,
    genome_json TEXT NOT NULL,            -- Full genome serialization
    generation_discovered INTEGER,

    -- Performance metrics (out-of-sample)
    oos_sharpe REAL,
    oos_sortino REAL,
    oos_max_drawdown REAL,
    oos_total_trades INTEGER,
    oos_win_rate REAL,

    -- Behavioral characteristics
    behavior_vector TEXT,                 -- JSON array
    novelty_score REAL,

    -- Status tracking
    status TEXT DEFAULT 'candidate',      -- candidate, validated, deployed, retired
    validation_date TEXT,
    deployment_date TEXT,
    retirement_date TEXT,
    retirement_reason TEXT,

    -- Generated code
    python_code TEXT,                     -- Generated strategy class

    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Evolution history for analysis
CREATE TABLE IF NOT EXISTS evolution_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    generation INTEGER NOT NULL,

    -- Population statistics
    pop_size INTEGER,
    pareto_front_size INTEGER,
    novelty_archive_size INTEGER,

    -- Fitness statistics
    best_sortino REAL,
    avg_sortino REAL,
    best_drawdown REAL,
    avg_novelty REAL,

    -- Diversity metrics
    behavior_diversity REAL,              -- Average pairwise distance
    genome_diversity REAL,                -- Tree structure diversity

    -- Discoveries this generation
    new_pareto_solutions INTEGER,
    strategies_validated INTEGER,

    regime TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_checkpoints_generation ON evolution_checkpoints(generation);
CREATE INDEX idx_discovered_status ON discovered_strategies(status);
CREATE INDEX idx_history_run ON evolution_history(run_id);
```

---

## Configuration

```python
@dataclass
class EvolutionConfig:
    """Configuration for autonomous evolution."""

    # Population
    population_size: int = 100
    generations_per_session: int = 50

    # Genetic operators
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2

    # Mutation type probabilities (must sum to 1.0)
    subtree_mutation_prob: float = 0.70
    point_mutation_prob: float = 0.15
    hoist_mutation_prob: float = 0.10
    shrink_mutation_prob: float = 0.05

    # Selection
    tournament_size: int = 3
    elitism: int = 5

    # Novelty search
    novelty_k_neighbors: int = 20
    novelty_archive_size: int = 500
    novelty_weight: float = 0.3  # Weight of novelty in selection

    # Fitness constraints
    min_trades: int = 50
    max_drawdown: float = -40.0  # Percentage
    min_deflated_sharpe: float = 0.7

    # Checkpointing
    checkpoint_frequency: int = 10  # Every N generations

    # Regime adaptation
    regime_change_injection: float = 0.3  # % of pop to replace

    # Tree constraints
    max_tree_depth: int = 6
    min_tree_depth: int = 2
    max_tree_size: int = 50  # Total nodes
```

---

## File Structure

```
research/discovery/
├── ARCHITECTURE.md          # This document
├── __init__.py              # Public API exports
├── config.py                # EvolutionConfig, IslandConfig
│
│   # Core GP Components
├── gp_core.py               # DEAP-based GP primitives
├── strategy_genome.py       # StrategyGenome dataclass
├── strategy_compiler.py     # EvolvedStrategy class
│
│   # Fitness & Selection
├── multi_objective.py       # FitnessVector, NSGA-II, calculate_fitness
├── novelty_search.py        # BehaviorVector, NoveltyArchive
├── portfolio_fitness.py     # Portfolio-level fitness evaluation
│
│   # Anti-Convergence (NEW)
├── island_model.py          # IslandEvolutionEngine, migration topologies
├── diversity_metrics.py     # GenotypeMetrics, DiversityMonitor, auto-intervention
├── map_elites.py            # MAPElitesGrid, quality-diversity optimization
│
│   # Evolution Engines
├── evolution_engine.py      # Single-population EvolutionEngine
├── overnight_runner.py      # Main entry point for overnight runs
│
│   # Database & Persistence
├── db_schema.py             # Database migrations
├── promotion_pipeline.py    # Strategy validation and promotion
│
├── utils/
│   ├── tree_serialization.py
│   ├── regime_detection.py
│   └── reporting.py
├── logs/                    # Evolution run logs
├── reports/                 # Generated reports
├── checkpoints/             # Evolution state checkpoints
└── tests/
    ├── test_gp_core.py
    ├── test_strategy_compiler.py
    ├── test_novelty.py
    ├── test_island_model.py
    └── test_diversity_metrics.py
```

---

## Integration with Existing System

### 1. Backtester Integration

The evolved strategies use the existing `Backtester` class:

```python
# In evolution_engine.py
def _evaluate_genome(self, genome: StrategyGenome, data, vix_data) -> FitnessVector:
    # Compile genome to strategy
    strategy = EvolvedStrategy(genome)

    # Use existing backtester
    result = self.backtester.run(
        strategy=strategy,
        data=data,
        vix_data=vix_data
    )

    # Calculate multi-objective fitness
    return calculate_fitness_vector(
        result=result,
        novelty_archive=self.nsga2.novelty_archive,
        total_trials=self.current_generation * self.nsga2.pop_size
    )
```

### 2. Strategy Registration

Promoted strategies are registered with the main system:

```python
# In strategy_compiler.py
def promote_to_production(genome: StrategyGenome, db: DatabaseManager):
    """Promote validated strategy to production candidates."""
    # Generate Python code
    code = generate_strategy_code(genome)

    # Save to database
    db.save_discovered_strategy(
        strategy_id=genome.genome_id,
        genome_json=serialize_genome(genome),
        python_code=code,
        status='validated'
    )

    # Write to file for review
    output_path = Path('research/discovery/candidates') / f'{genome.genome_id}.py'
    output_path.write_text(code)

    logger.info(f"Promoted strategy {genome.genome_id} to candidates")
```

### 3. Nightly Research Integration

Add to `run_nightly_research.py`:

```python
# After parameter optimization, run strategy discovery
if config.ENABLE_STRATEGY_DISCOVERY:
    from research.discovery.overnight_runner import run_discovery

    discovery_results = run_discovery(
        hours=config.DISCOVERY_HOURS,  # e.g., 4 hours
        db=db,
        data_loader=data_loader,
        backtester=backtester
    )

    logger.info(f"Discovery: {discovery_results['new_strategies']} new strategies found")
```

---

## Success Metrics

### Short-term (30 days)
- [ ] System runs autonomously overnight without intervention
- [ ] Discovers 20+ novel strategy variants
- [ ] Maintains population diversity (avg correlation < 0.5)
- [ ] At least 5 strategies pass DSR > 0.90 threshold

### Medium-term (90 days)
- [ ] 50+ strategies discovered with behavioral diversity
- [ ] 10+ strategies deployed to paper trading
- [ ] Ensemble portfolio Sharpe > 0.8
- [ ] System adapts to at least one regime change

### Long-term (6-12 months)
- [ ] Autonomous operation with minimal intervention
- [ ] Continuous strategy discovery and retirement
- [ ] Novel strategies outperform hand-coded on average
- [ ] Real capital deployment with 3-6% annual alpha target

---

## Next Steps

1. **Phase 1:** Implement GP core with DEAP
2. **Phase 2:** Build strategy compiler and evaluator
3. **Phase 3:** Implement NSGA-II multi-objective optimization
4. **Phase 4:** Add novelty search
5. **Phase 5:** Build continuous evolution infrastructure
6. **Phase 6:** Integration testing and overnight runs
7. **Phase 7:** Validation and production deployment
