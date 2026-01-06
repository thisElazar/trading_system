#!/usr/bin/env python3
"""
Weekend Research Coordinator
=============================
Extended research session manager for weekend optimization and discovery.

This coordinator:
1. Manages the extended weekend research window (Friday eve → Sunday)
2. Coordinates parameter optimization → discovery → adaptive GA
3. Writes progress state to a file for dashboard display
4. Supports graceful shutdown for future pause/resume
5. Handles configuration from WEEKEND_CONFIG or command line

Usage:
    # Run with default weekend config
    python weekend_research.py

    # Run with custom settings
    python weekend_research.py --generations 15 --population 40 --discovery

    # Run specific phase only
    python weekend_research.py --phase optimization
    python weekend_research.py --phase discovery
    python weekend_research.py --phase adaptive

    # Dry run (show what would be executed)
    python weekend_research.py --dry-run
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from enum import Enum

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from config import DIRS, WEEKEND_CONFIG, get_enabled_strategies

# Setup logging
LOG_DIR = DIRS.get('logs', Path('./logs'))
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'weekend_research.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('weekend_research')


class ResearchPhase(Enum):
    """Weekend research phases."""
    STARTING = "starting"
    OPTIMIZATION = "optimization"
    DISCOVERY = "discovery"
    ADAPTIVE = "adaptive"
    COMPLETE = "complete"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class ResearchProgress:
    """Tracks research progress for dashboard display."""
    phase: str = "starting"
    current_strategy: str = ""
    generation: int = 0
    total_generations: int = 0
    individual: int = 0
    population_size: int = 0
    best_fitness: float = 0.0
    strategies_completed: List[str] = field(default_factory=list)
    strategies_remaining: List[str] = field(default_factory=list)
    discoveries_found: int = 0
    started_at: Optional[str] = None
    last_update: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ResearchConfig:
    """Configuration for weekend research run."""
    # Parameter optimization
    generations: int = 10
    population: int = 30
    strategies: List[str] = field(default_factory=list)

    # Discovery
    discovery_enabled: bool = True
    discovery_hours: float = 4.0
    discovery_population: int = 50

    # Adaptive GA
    adaptive_enabled: bool = True
    adaptive_population: int = 60
    adaptive_islands: int = 4
    rapid_first: bool = False

    @classmethod
    def from_weekend_config(cls) -> 'ResearchConfig':
        """Create config from WEEKEND_CONFIG in config.py."""
        research = WEEKEND_CONFIG.get('research', {})
        return cls(
            generations=research.get('generations_default', 10),
            population=research.get('population_default', 30),
            strategies=research.get('strategies', []),
            discovery_enabled=research.get('discovery_enabled', True),
            discovery_hours=research.get('discovery_hours', 4.0),
            adaptive_enabled=research.get('adaptive_ga_enabled', True),
        )

    @classmethod
    def from_preset(cls, preset_name: str) -> 'ResearchConfig':
        """Create config from a preset (quick, standard, deep)."""
        presets = WEEKEND_CONFIG.get('presets', {})
        preset = presets.get(preset_name, presets.get('standard', {}))

        return cls(
            generations=preset.get('generations', 10),
            population=preset.get('population', 30),
            discovery_enabled=preset.get('discovery', True),
            discovery_hours=preset.get('discovery_hours', 4.0),
            adaptive_enabled=preset.get('adaptive', True),
        )


class WeekendResearchCoordinator:
    """
    Coordinates extended weekend research sessions.

    Manages the sequence of:
    1. Parameter optimization (GA evolution)
    2. Strategy discovery (GP-based)
    3. Adaptive GA (regime-matched testing)

    Writes progress to a state file for dashboard monitoring.
    """

    STATE_FILE = LOG_DIR / 'weekend_research_state.json'

    def __init__(self, config: ResearchConfig):
        self.config = config
        self.progress = ResearchProgress()
        self.shutdown_requested = False
        self._process: Optional[subprocess.Popen] = None

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

        # Get strategies to optimize
        if not self.config.strategies:
            self.config.strategies = list(get_enabled_strategies().keys())

        self.progress.strategies_remaining = list(self.config.strategies)
        self.progress.total_generations = self.config.generations
        self.progress.population_size = self.config.population

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
        if self._process and self._process.poll() is None:
            logger.info("Terminating running research process...")
            self._process.terminate()

    def _write_state(self):
        """Write current progress to state file for dashboard."""
        self.progress.last_update = datetime.now().isoformat()
        try:
            with open(self.STATE_FILE, 'w') as f:
                json.dump(self.progress.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write state file: {e}")

    def _read_state(self) -> Optional[ResearchProgress]:
        """Read previous state from file (for resume)."""
        try:
            if self.STATE_FILE.exists():
                with open(self.STATE_FILE, 'r') as f:
                    data = json.load(f)
                    return ResearchProgress(**data)
        except Exception as e:
            logger.warning(f"Failed to read state file: {e}")
        return None

    def _run_command(self, cmd: List[str], description: str) -> bool:
        """Run a command and stream output."""
        logger.info(f"Running: {description}")
        logger.info(f"Command: {' '.join(cmd)}")

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            # Stream output
            for line in self._process.stdout:
                line = line.rstrip()
                if line:
                    # Parse progress from research output
                    self._parse_progress(line)
                    print(line)  # Echo to our stdout

            self._process.wait()
            success = self._process.returncode == 0
            self._process = None

            if not success:
                logger.error(f"{description} failed with code {self._process.returncode}")

            return success

        except Exception as e:
            logger.error(f"Error running {description}: {e}")
            return False

    def _parse_progress(self, line: str):
        """Parse research output to update progress."""
        # Parse generation progress: [G1 5/10] or [Generation 5]
        if '[G' in line and '/' in line:
            try:
                # Format: [G1 5/10] Testing...
                parts = line.split(']')[0].split('[')[1]
                gen_part = parts.split()[0]  # G1
                ind_part = parts.split()[1]  # 5/10

                self.progress.generation = int(gen_part[1:])
                ind, total = ind_part.split('/')
                self.progress.individual = int(ind)
                self.progress.population_size = int(total)
                self._write_state()
            except:
                pass

        # Parse strategy start: Evolving: strategy_name
        if 'Evolving:' in line:
            try:
                strategy = line.split('Evolving:')[1].strip()
                self.progress.current_strategy = strategy
                self.progress.generation = 0
                self.progress.individual = 0
                self._write_state()
            except:
                pass

        # Parse best fitness: Best fitness: 45.67
        if 'Best fitness:' in line:
            try:
                fitness = float(line.split('Best fitness:')[1].split(',')[0].strip())
                self.progress.best_fitness = fitness
                self._write_state()
            except:
                pass

        # Parse strategy completion
        if 'saved to database' in line.lower() or 'completed' in line.lower():
            if self.progress.current_strategy and self.progress.current_strategy not in self.progress.strategies_completed:
                self.progress.strategies_completed.append(self.progress.current_strategy)
                if self.progress.current_strategy in self.progress.strategies_remaining:
                    self.progress.strategies_remaining.remove(self.progress.current_strategy)
                self._write_state()

        # Parse discovery findings
        if 'discovered' in line.lower() and 'strategy' in line.lower():
            self.progress.discoveries_found += 1
            self._write_state()

    def run_optimization(self) -> bool:
        """Run parameter optimization phase."""
        if self.shutdown_requested:
            return False

        logger.info("=" * 60)
        logger.info("PHASE 1: Parameter Optimization")
        logger.info("=" * 60)

        self.progress.phase = ResearchPhase.OPTIMIZATION.value
        self._write_state()

        # Build command
        python = sys.executable
        script = str(Path(__file__).parent / 'run_nightly_research.py')

        cmd = [
            python, script,
            '-g', str(self.config.generations),
            '-p', str(self.config.population),
        ]

        # Add specific strategies if configured
        if self.config.strategies:
            cmd.extend(['-s', ','.join(self.config.strategies)])

        return self._run_command(cmd, "Parameter Optimization")

    def run_discovery(self) -> bool:
        """Run strategy discovery phase."""
        if self.shutdown_requested:
            return False

        if not self.config.discovery_enabled:
            logger.info("Strategy discovery disabled, skipping")
            return True

        logger.info("=" * 60)
        logger.info("PHASE 2: Strategy Discovery")
        logger.info("=" * 60)

        self.progress.phase = ResearchPhase.DISCOVERY.value
        self.progress.current_strategy = "gp_discovery"
        self._write_state()

        python = sys.executable
        script = str(Path(__file__).parent / 'run_nightly_research.py')

        cmd = [
            python, script,
            '--discovery-only',
            '--discovery-hours', str(self.config.discovery_hours),
            '--discovery-pop', str(self.config.discovery_population),
        ]

        return self._run_command(cmd, "Strategy Discovery")

    def run_adaptive(self) -> bool:
        """Run adaptive GA phase."""
        if self.shutdown_requested:
            return False

        if not self.config.adaptive_enabled:
            logger.info("Adaptive GA disabled, skipping")
            return True

        logger.info("=" * 60)
        logger.info("PHASE 3: Adaptive GA (Regime-Matched Testing)")
        logger.info("=" * 60)

        self.progress.phase = ResearchPhase.ADAPTIVE.value
        self.progress.current_strategy = "adaptive_ga"
        self._write_state()

        python = sys.executable
        script = str(Path(__file__).parent / 'run_nightly_research.py')

        cmd = [
            python, script,
            '--adaptive-only',
            '--adaptive-pop', str(self.config.adaptive_population),
            '--adaptive-islands', str(self.config.adaptive_islands),
        ]

        if self.config.rapid_first:
            cmd.append('--rapid-first')

        return self._run_command(cmd, "Adaptive GA")

    def run(self, phase: Optional[str] = None) -> bool:
        """
        Run the full weekend research session.

        Args:
            phase: Optional specific phase to run (optimization, discovery, adaptive)
                   If None, runs all phases in sequence.

        Returns:
            True if all phases completed successfully
        """
        self.progress.started_at = datetime.now().isoformat()
        self._write_state()

        logger.info("=" * 70)
        logger.info("WEEKEND RESEARCH COORDINATOR")
        logger.info("=" * 70)
        logger.info(f"Started: {self.progress.started_at}")
        logger.info(f"Configuration:")
        logger.info(f"  Generations: {self.config.generations}")
        logger.info(f"  Population: {self.config.population}")
        logger.info(f"  Strategies: {', '.join(self.config.strategies)}")
        logger.info(f"  Discovery: {'enabled' if self.config.discovery_enabled else 'disabled'}")
        logger.info(f"  Adaptive GA: {'enabled' if self.config.adaptive_enabled else 'disabled'}")
        logger.info("=" * 70)

        success = True

        try:
            if phase is None or phase == 'optimization':
                if not self.run_optimization():
                    success = False
                    if self.shutdown_requested:
                        logger.info("Shutdown requested, stopping research")
                        self.progress.phase = ResearchPhase.PAUSED.value
                        self._write_state()
                        return False

            if phase is None or phase == 'discovery':
                if not self.run_discovery():
                    success = False

            if phase is None or phase == 'adaptive':
                if not self.run_adaptive():
                    success = False

            # Mark complete
            if success:
                self.progress.phase = ResearchPhase.COMPLETE.value
                logger.info("=" * 70)
                logger.info("WEEKEND RESEARCH COMPLETE")
                logger.info(f"Strategies optimized: {len(self.progress.strategies_completed)}")
                logger.info(f"Discoveries found: {self.progress.discoveries_found}")
                logger.info("=" * 70)
            else:
                self.progress.phase = ResearchPhase.ERROR.value
                logger.error("Weekend research completed with errors")

            self._write_state()
            return success

        except Exception as e:
            logger.error(f"Weekend research failed: {e}")
            self.progress.phase = ResearchPhase.ERROR.value
            self.progress.error = str(e)
            self._write_state()
            return False


def main():
    parser = argparse.ArgumentParser(
        description='Weekend Research Coordinator - Extended optimization and discovery'
    )

    # Config options
    parser.add_argument('-g', '--generations', type=int,
                        help='Number of GA generations (default: from config)')
    parser.add_argument('-p', '--population', type=int,
                        help='Population size (default: from config)')
    parser.add_argument('-s', '--strategies', type=str,
                        help='Comma-separated list of strategies to optimize')

    # Phase control
    parser.add_argument('--phase', choices=['optimization', 'discovery', 'adaptive'],
                        help='Run only a specific phase')

    # Discovery options
    parser.add_argument('--discovery', action='store_true',
                        help='Enable strategy discovery')
    parser.add_argument('--no-discovery', action='store_true',
                        help='Disable strategy discovery')
    parser.add_argument('--discovery-hours', type=float,
                        help='Max hours for discovery phase')

    # Adaptive GA options
    parser.add_argument('--adaptive', action='store_true',
                        help='Enable adaptive GA')
    parser.add_argument('--no-adaptive', action='store_true',
                        help='Disable adaptive GA')
    parser.add_argument('--rapid-first', action='store_true',
                        help='Run rapid testing before full adaptive GA')

    # Presets
    parser.add_argument('--preset', choices=['quick', 'standard', 'deep'],
                        help='Use a preset configuration')

    # Utility
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be executed without running')
    parser.add_argument('--status', action='store_true',
                        help='Show current research status and exit')

    args = parser.parse_args()

    # Status check
    if args.status:
        state_file = LOG_DIR / 'weekend_research_state.json'
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            print(json.dumps(state, indent=2))
        else:
            print("No weekend research state found")
        return

    # Build config
    if args.preset:
        config = ResearchConfig.from_preset(args.preset)
    else:
        config = ResearchConfig.from_weekend_config()

    # Override with command line args
    if args.generations:
        config.generations = args.generations
    if args.population:
        config.population = args.population
    if args.strategies:
        config.strategies = [s.strip() for s in args.strategies.split(',')]
    if args.discovery:
        config.discovery_enabled = True
    if args.no_discovery:
        config.discovery_enabled = False
    if args.discovery_hours:
        config.discovery_hours = args.discovery_hours
    if args.adaptive:
        config.adaptive_enabled = True
    if args.no_adaptive:
        config.adaptive_enabled = False
    if args.rapid_first:
        config.rapid_first = True

    # Dry run
    if args.dry_run:
        print("Weekend Research Configuration:")
        print(f"  Generations: {config.generations}")
        print(f"  Population: {config.population}")
        print(f"  Strategies: {config.strategies or 'all enabled'}")
        print(f"  Discovery: {config.discovery_enabled}")
        print(f"  Discovery Hours: {config.discovery_hours}")
        print(f"  Adaptive GA: {config.adaptive_enabled}")
        print(f"  Rapid First: {config.rapid_first}")
        print(f"\nPhase to run: {args.phase or 'all'}")
        return

    # Run coordinator
    coordinator = WeekendResearchCoordinator(config)
    success = coordinator.run(phase=args.phase)

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
