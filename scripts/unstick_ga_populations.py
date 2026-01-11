#!/usr/bin/env python3
"""
Unstick GA Populations - One-time reset for stuck strategies
=============================================================

This script identifies strategies that have been stuck (no fitness improvement)
for extended periods and resets their populations with fresh random individuals
while preserving the top elites.

Usage:
    # Dry run - show what would be reset
    python scripts/unstick_ga_populations.py --dry-run

    # Actually perform the reset
    python scripts/unstick_ga_populations.py

    # Reset specific strategy
    python scripts/unstick_ga_populations.py --strategy vol_managed_momentum

Created: 2026-01-11
Purpose: Break 8+ day stagnation in GA parameter optimization
"""

import argparse
import json
import logging
import random
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from research.genetic.optimizer import STRATEGY_PARAMS, ParameterSpec

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger('unstick_ga')

# Configuration
DB_PATH = Path(__file__).parent.parent / "db" / "research.db"
STAGNATION_THRESHOLD = 50  # Generations without improvement to trigger reset
ELITE_PRESERVE_RATIO = 0.20  # Keep top 20% of population
NEW_POPULATION_SIZE = 30  # Standard population size


def get_stagnation_stats(conn: sqlite3.Connection) -> List[Dict]:
    """
    Get stagnation statistics for all strategies.

    Returns list of dicts with strategy, generations_stuck, best_fitness, max_generation
    """
    cursor = conn.cursor()

    # For each strategy, find:
    # 1. Max generation ever reached
    # 2. Best fitness ever achieved
    # 3. When that best fitness was first achieved
    cursor.execute("""
        WITH strategy_bests AS (
            SELECT
                strategy,
                MAX(best_fitness) as best_ever,
                MAX(generation) as max_gen
            FROM ga_history
            WHERE generation >= 0  -- Exclude restart markers (-1)
            GROUP BY strategy
        ),
        first_best AS (
            SELECT
                h.strategy,
                MIN(h.generation) as first_best_gen
            FROM ga_history h
            JOIN strategy_bests b ON h.strategy = b.strategy AND h.best_fitness = b.best_ever
            WHERE h.generation >= 0
            GROUP BY h.strategy
        )
        SELECT
            b.strategy,
            b.max_gen - f.first_best_gen as generations_stuck,
            b.best_ever,
            b.max_gen,
            f.first_best_gen
        FROM strategy_bests b
        JOIN first_best f ON b.strategy = f.strategy
        ORDER BY generations_stuck DESC
    """)

    results = []
    for row in cursor.fetchall():
        results.append({
            'strategy': row[0],
            'generations_stuck': row[1],
            'best_fitness': row[2],
            'max_generation': row[3],
            'first_best_gen': row[4]
        })

    return results


def create_random_individual(specs: List[ParameterSpec], generation: int) -> Dict:
    """Create a random individual based on parameter specs."""
    genes = {}
    for spec in specs:
        if spec.step >= 1:
            # Integer parameter
            genes[spec.name] = float(random.randint(int(spec.min_val), int(spec.max_val)))
        else:
            # Float parameter
            n_steps = int((spec.max_val - spec.min_val) / spec.step)
            genes[spec.name] = spec.min_val + random.randint(0, n_steps) * spec.step

    return {
        'genes': genes,
        'fitness': 0.0,  # Will be evaluated next session
        'generation': generation
    }


def reset_strategy_population(
    conn: sqlite3.Connection,
    strategy: str,
    dry_run: bool = True
) -> Tuple[int, int]:
    """
    Reset a strategy's population with fresh random individuals.

    Args:
        conn: Database connection
        strategy: Strategy name
        dry_run: If True, don't actually modify database

    Returns:
        Tuple of (elites_preserved, new_individuals_created)
    """
    cursor = conn.cursor()

    # Load current population
    cursor.execute("""
        SELECT population_json, generation, best_fitness, best_genes_json
        FROM ga_populations
        WHERE strategy = ?
    """, (strategy,))

    row = cursor.fetchone()
    if not row:
        logger.warning(f"No population found for {strategy}")
        return 0, 0

    population_json, current_gen, best_fitness, best_genes_json = row
    current_population = json.loads(population_json)

    # Get parameter specs for this strategy
    if strategy not in STRATEGY_PARAMS:
        logger.warning(f"No parameter specs for {strategy}, skipping")
        return 0, 0

    specs = STRATEGY_PARAMS[strategy]

    # Sort population by fitness to identify elites
    sorted_pop = sorted(current_population, key=lambda x: x.get('fitness', 0), reverse=True)

    # Calculate how many elites to preserve
    n_elite = max(2, int(len(sorted_pop) * ELITE_PRESERVE_RATIO))
    elites = sorted_pop[:n_elite]

    # Create new random individuals
    n_new = NEW_POPULATION_SIZE - n_elite
    new_generation = current_gen + 1

    new_individuals = [
        create_random_individual(specs, new_generation)
        for _ in range(n_new)
    ]

    # Combine elites with new individuals
    new_population = elites + new_individuals

    if dry_run:
        logger.info(f"  [DRY RUN] Would reset {strategy}:")
        logger.info(f"    Current population: {len(current_population)} individuals")
        logger.info(f"    Preserving {n_elite} elites (best fitness: {elites[0].get('fitness', 0):.4f})")
        logger.info(f"    Creating {n_new} new random individuals")
        logger.info(f"    New generation: {new_generation}")
    else:
        # Update population in database
        cursor.execute("""
            UPDATE ga_populations
            SET population_json = ?,
                generation = ?,
                updated_at = ?
            WHERE strategy = ?
        """, (
            json.dumps(new_population),
            new_generation,
            datetime.now().isoformat(),
            strategy
        ))

        # Log restart event in history
        cursor.execute("""
            INSERT INTO ga_history (
                strategy, generation, best_fitness, mean_fitness, std_fitness,
                best_genes_json, generations_without_improvement, run_date, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            strategy,
            -1,  # Special marker for restart event
            best_fitness,
            0.0,
            0.0,
            best_genes_json,
            0,  # Reset stagnation counter
            datetime.now().strftime("%Y-%m-%d"),
            datetime.now().isoformat()
        ))

        conn.commit()

        logger.info(f"  RESET {strategy}:")
        logger.info(f"    Preserved {n_elite} elites, created {n_new} new individuals")
        logger.info(f"    Next generation: {new_generation}")

    return n_elite, n_new


def main():
    parser = argparse.ArgumentParser(
        description="Reset stuck GA populations to break stagnation"
    )
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        '--strategy', '-s',
        type=str,
        help="Reset only this specific strategy"
    )
    parser.add_argument(
        '--threshold', '-t',
        type=int,
        default=STAGNATION_THRESHOLD,
        help=f"Generations stuck threshold (default: {STAGNATION_THRESHOLD})"
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help="Reset even if not stuck (use with --strategy)"
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("GA Population Unstick Script")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made")

    # Connect to database
    conn = sqlite3.connect(str(DB_PATH))

    try:
        # Get stagnation stats
        stats = get_stagnation_stats(conn)

        if not stats:
            logger.info("No strategy history found in database")
            return

        logger.info("\nCurrent Stagnation Status:")
        logger.info("-" * 60)

        strategies_to_reset = []

        for s in stats:
            stuck_marker = "⚠️  STUCK" if s['generations_stuck'] >= args.threshold else "✓"
            logger.info(
                f"  {s['strategy']:30} | Gen {s['max_generation']:4} | "
                f"Stuck: {s['generations_stuck']:4} gens | "
                f"Best: {s['best_fitness']:.4f} | {stuck_marker}"
            )

            if args.strategy:
                if s['strategy'] == args.strategy:
                    if args.force or s['generations_stuck'] >= args.threshold:
                        strategies_to_reset.append(s['strategy'])
            elif s['generations_stuck'] >= args.threshold:
                strategies_to_reset.append(s['strategy'])

        logger.info("-" * 60)

        if not strategies_to_reset:
            if args.strategy:
                logger.info(f"\nStrategy '{args.strategy}' is not stuck (use --force to reset anyway)")
            else:
                logger.info(f"\nNo strategies stuck for >= {args.threshold} generations")
            return

        logger.info(f"\nStrategies to reset ({len(strategies_to_reset)}):")
        for s in strategies_to_reset:
            logger.info(f"  - {s}")

        if not args.dry_run:
            logger.info("\nPerforming reset...")

        total_elites = 0
        total_new = 0

        for strategy in strategies_to_reset:
            elites, new = reset_strategy_population(conn, strategy, dry_run=args.dry_run)
            total_elites += elites
            total_new += new

        logger.info("\n" + "=" * 60)
        if args.dry_run:
            logger.info(f"DRY RUN SUMMARY:")
            logger.info(f"  Would reset {len(strategies_to_reset)} strategies")
            logger.info(f"  Would preserve {total_elites} elite individuals")
            logger.info(f"  Would create {total_new} new random individuals")
            logger.info("\nRun without --dry-run to apply changes")
        else:
            logger.info(f"RESET COMPLETE:")
            logger.info(f"  Reset {len(strategies_to_reset)} strategies")
            logger.info(f"  Preserved {total_elites} elite individuals")
            logger.info(f"  Created {total_new} new random individuals")
            logger.info("\nNext nightly research run will evaluate new populations")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
