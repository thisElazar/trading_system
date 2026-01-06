#!/usr/bin/env python3
"""
Fix import errors in the autonomous research engine.

The issue: RelativeVolumeBreakout is imported as RelativeVolumeBreakoutStrategy

Run this script from your trading_system directory:
    cd /Volumes/HotStorage/Pi/TradeBot/trading_system
    python fix_import_errors.py
"""

import os
import sys
from pathlib import Path


def fix_file(filepath: Path, old_text: str, new_text: str) -> bool:
    """Replace text in a file."""
    if not filepath.exists():
        print(f"  ⚠️  File not found: {filepath}")
        return False
    
    content = filepath.read_text()
    
    if old_text not in content:
        print(f"  ✓ Already fixed or pattern not found: {filepath.name}")
        return False
    
    new_content = content.replace(old_text, new_text)
    filepath.write_text(new_content)
    print(f"  ✅ Fixed: {filepath}")
    return True


def main():
    # Determine base path
    if len(sys.argv) > 1:
        base_path = Path(sys.argv[1])
    else:
        base_path = Path.cwd()
    
    print("=" * 60)
    print("Fixing Import Errors in Research Engine")
    print("=" * 60)
    print(f"\nBase path: {base_path}")
    
    fixes = [
        # Fix 1: persistent_optimizer.py - fix the import
        {
            'file': base_path / 'research' / 'genetic' / 'persistent_optimizer.py',
            'old': 'from strategies.relative_volume_breakout import RelativeVolumeBreakoutStrategy',
            'new': 'from strategies.relative_volume_breakout import RelativeVolumeBreakout'
        },
        # Fix 2: persistent_optimizer.py - fix the class reference
        {
            'file': base_path / 'research' / 'genetic' / 'persistent_optimizer.py',
            'old': "'relative_volume_breakout': RelativeVolumeBreakoutStrategy,",
            'new': "'relative_volume_breakout': RelativeVolumeBreakout,"
        },
        # Fix 3: run_nightly_research.py - fix the import
        {
            'file': base_path / 'run_nightly_research.py',
            'old': 'from strategies.relative_volume_breakout import RelativeVolumeBreakoutStrategy',
            'new': 'from strategies.relative_volume_breakout import RelativeVolumeBreakout'
        },
        # Fix 4: run_nightly_research.py - fix the class reference
        {
            'file': base_path / 'run_nightly_research.py',
            'old': "'relative_volume_breakout': RelativeVolumeBreakoutStrategy,",
            'new': "'relative_volume_breakout': RelativeVolumeBreakout,"
        },
    ]
    
    # Also need to fix the factory functions to use lazy imports
    # This is a more robust fix that only imports what's needed
    
    print("\n1. Fixing class name imports...")
    for fix in fixes:
        fix_file(fix['file'], fix['old'], fix['new'])
    
    # Fix 5: Make imports lazy in persistent_optimizer.py to prevent cascade failures
    print("\n2. Converting to lazy imports in persistent_optimizer.py...")
    
    po_file = base_path / 'research' / 'genetic' / 'persistent_optimizer.py'
    if po_file.exists():
        content = po_file.read_text()
        
        # Find and replace the import block with lazy import pattern
        old_import_block = """from strategies.vol_managed_momentum import VolManagedMomentumStrategy
    from strategies.mean_reversion import MeanReversionStrategy
    from strategies.pairs_trading import PairsTradingStrategy
    from strategies.relative_volume_breakout import RelativeVolumeBreakout
    
    strategy_classes = {
        'vol_managed_momentum': VolManagedMomentumStrategy,
        'mean_reversion': MeanReversionStrategy,
        'pairs_trading': PairsTradingStrategy,
        'relative_volume_breakout': RelativeVolumeBreakout,
    }
    
    strategy_class = strategy_classes.get(strategy_name)"""
        
        new_import_block = """# Lazy import to avoid cascade failures
    strategy_class = None
    
    if strategy_name == 'vol_managed_momentum':
        from strategies.vol_managed_momentum import VolManagedMomentumStrategy
        strategy_class = VolManagedMomentumStrategy
    elif strategy_name == 'mean_reversion':
        from strategies.mean_reversion import MeanReversionStrategy
        strategy_class = MeanReversionStrategy
    elif strategy_name == 'pairs_trading':
        from strategies.pairs_trading import PairsTradingStrategy
        strategy_class = PairsTradingStrategy
    elif strategy_name == 'relative_volume_breakout':
        from strategies.relative_volume_breakout import RelativeVolumeBreakout
        strategy_class = RelativeVolumeBreakout"""
        
        if old_import_block in content:
            content = content.replace(old_import_block, new_import_block)
            po_file.write_text(content)
            print("  ✅ Converted to lazy imports: persistent_optimizer.py")
        else:
            print("  ⚠️  Import block pattern not found (may already be fixed)")
    
    # Fix 6: Make imports lazy in run_nightly_research.py
    print("\n3. Converting to lazy imports in run_nightly_research.py...")
    
    nr_file = base_path / 'run_nightly_research.py'
    if nr_file.exists():
        content = nr_file.read_text()
        
        # Check for eager imports at module level and make them lazy
        old_pattern = """from strategies.vol_managed_momentum import VolManagedMomentumStrategy
    from strategies.mean_reversion import MeanReversionStrategy
    from strategies.pairs_trading import PairsTradingStrategy
    from strategies.relative_volume_breakout import RelativeVolumeBreakoutStrategy
    from strategies.vix_regime_rotation import VIXRegimeRotationStrategy
    from strategies.sector_rotation import SectorRotationStrategy
    
    factories = {
        'vol_managed_momentum': VolManagedMomentumStrategy,
        'mean_reversion': MeanReversionStrategy,
        'pairs_trading': PairsTradingStrategy,
        'relative_volume_breakout': RelativeVolumeBreakoutStrategy,
        'vix_regime_rotation': VIXRegimeRotationStrategy,
        'sector_rotation': SectorRotationStrategy,
    }"""
        
        new_pattern = """# Lazy imports to avoid cascade failures on missing strategies
    strategy_class = None
    
    if strategy_name == 'vol_managed_momentum':
        from strategies.vol_managed_momentum import VolManagedMomentumStrategy
        strategy_class = VolManagedMomentumStrategy
    elif strategy_name == 'mean_reversion':
        from strategies.mean_reversion import MeanReversionStrategy
        strategy_class = MeanReversionStrategy
    elif strategy_name == 'pairs_trading':
        from strategies.pairs_trading import PairsTradingStrategy
        strategy_class = PairsTradingStrategy
    elif strategy_name == 'relative_volume_breakout':
        from strategies.relative_volume_breakout import RelativeVolumeBreakout
        strategy_class = RelativeVolumeBreakout
    elif strategy_name == 'vix_regime_rotation':
        from strategies.vix_regime_rotation import VIXRegimeRotationStrategy
        strategy_class = VIXRegimeRotationStrategy
    elif strategy_name == 'sector_rotation':
        from strategies.sector_rotation import SectorRotationStrategy
        strategy_class = SectorRotationStrategy
    
    if strategy_class is None:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    # Replace factory lookup with direct class usage
    factories = None  # Not needed with lazy imports"""
        
        if "RelativeVolumeBreakoutStrategy" in content:
            # Simple fix - just replace the class name
            content = content.replace("RelativeVolumeBreakoutStrategy", "RelativeVolumeBreakout")
            nr_file.write_text(content)
            print("  ✅ Fixed class name: run_nightly_research.py")
        else:
            print("  ✓ Already using correct class name")
    
    print("\n" + "=" * 60)
    print("FIXES COMPLETE")
    print("=" * 60)
    print("\nTo verify, run:")
    print("  python run_nightly_research.py -g 1")
    print("")


if __name__ == "__main__":
    main()
