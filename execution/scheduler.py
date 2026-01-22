"""
Strategy Scheduler
==================
Runs strategies at appropriate times automatically.

Schedule:
- Gap-fill: 9:31 AM ET (market open)
- Pairs: 10:00 AM ET (after open volatility)
- RV Breakout: 10:00 AM ET
- Position checks: Every 15 min during market hours
- EOD summary: 4:15 PM ET
"""

import time
import schedule
import threading
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Callable, Dict, List
import logging
import pytz

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import STRATEGIES
from execution.signal_tracker import SignalDatabase, ExecutionTracker

logger = logging.getLogger(__name__)

# Lazy imports for execution components (avoid circular imports)
_execution_manager = None
_broker = None
_shadow_trader = None

def _get_shadow_trader():
    """Lazy-load ShadowTrader for hybrid shadow + Alpaca execution."""
    global _shadow_trader
    if _shadow_trader is None:
        try:
            from execution.shadow_trading import ShadowTrader
            _shadow_trader = ShadowTrader()
            logger.info(f"ShadowTrader initialized with {len(_shadow_trader.strategies)} strategies")
        except Exception as e:
            logger.warning(f"Failed to initialize ShadowTrader: {e}")
            _shadow_trader = None
    return _shadow_trader

def _get_execution_manager():
    """Lazy-load ExecutionManager with ShadowTrader for hybrid execution."""
    global _execution_manager
    if _execution_manager is None:
        from execution.execution_manager import ExecutionManager
        shadow_trader = _get_shadow_trader()
        _execution_manager = ExecutionManager()
        # Inject shadow trader for graduation tracking
        if shadow_trader:
            _execution_manager.shadow_trader = shadow_trader
            logger.info("ExecutionManager initialized with ShadowTrader (hybrid mode)")
        else:
            logger.info("ExecutionManager initialized (shadow trader unavailable)")
    return _execution_manager

def _get_broker():
    """Lazy-load Alpaca broker."""
    global _broker
    if _broker is None:
        from execution.alpaca_connector import AlpacaConnector
        _broker = AlpacaConnector(paper=True)
        logger.info("Alpaca broker initialized for scheduler (paper mode)")
    return _broker

ET = pytz.timezone('America/New_York')
LOCAL_TZ = datetime.now().astimezone().tzinfo


def et_to_local_time(et_time_str: str) -> str:
    """
    Convert ET time string (HH:MM) to local timezone for schedule library.

    The schedule library uses datetime.now() (local time) to check if jobs
    should run, so we need to convert our ET times to local time.

    Args:
        et_time_str: Time in ET like "09:31" or "16:15"

    Returns:
        Time string in local timezone like "06:31" (for PST)
    """
    today = datetime.now(ET).date()
    hour, minute = map(int, et_time_str.split(':'))

    # Create datetime in ET
    et_dt = ET.localize(datetime(today.year, today.month, today.day, hour, minute))

    # Convert to local time
    local_dt = et_dt.astimezone(LOCAL_TZ)

    return local_dt.strftime('%H:%M')


class MarketHours:
    """Market hours utilities with holiday support."""

    MARKET_OPEN = dtime(9, 30)
    MARKET_CLOSE = dtime(16, 0)
    EARLY_CLOSE = dtime(13, 0)  # Early close days (day before holidays)

    # US Market Holidays by year (dates when market is CLOSED)
    HOLIDAYS = {
        2025: {
            (1, 1),   # New Year's Day
            (1, 20),  # MLK Day
            (2, 17),  # Presidents Day
            (4, 18),  # Good Friday
            (5, 26),  # Memorial Day
            (6, 19),  # Juneteenth
            (7, 4),   # Independence Day
            (9, 1),   # Labor Day
            (11, 27), # Thanksgiving
            (12, 25), # Christmas
        },
        2026: {
            (1, 1),   # New Year's Day
            (1, 19),  # MLK Day (3rd Monday)
            (2, 16),  # Presidents Day (3rd Monday)
            (4, 3),   # Good Friday
            (5, 25),  # Memorial Day (last Monday)
            (6, 19),  # Juneteenth
            (7, 3),   # Independence Day observed (July 4 is Saturday)
            (9, 7),   # Labor Day (1st Monday)
            (11, 26), # Thanksgiving (4th Thursday)
            (12, 25), # Christmas
        },
    }

    # Early close days (1 PM close) by year
    EARLY_CLOSE_DAYS = {
        2025: {
            (7, 3),   # Day before July 4th
            (11, 28), # Day after Thanksgiving
            (12, 24), # Christmas Eve
        },
        2026: {
            (7, 2),   # Day before July 4th observed
            (11, 27), # Day after Thanksgiving
            (12, 24), # Christmas Eve
        },
    }

    @staticmethod
    def is_holiday(date: datetime = None) -> bool:
        """Check if given date is a market holiday."""
        if date is None:
            date = datetime.now(ET)
        year_holidays = MarketHours.HOLIDAYS.get(date.year, set())
        return (date.month, date.day) in year_holidays

    @staticmethod
    def is_early_close(date: datetime = None) -> bool:
        """Check if given date is an early close day."""
        if date is None:
            date = datetime.now(ET)
        year_early_close = MarketHours.EARLY_CLOSE_DAYS.get(date.year, set())
        return (date.month, date.day) in year_early_close

    @staticmethod
    def get_close_time(date: datetime = None) -> dtime:
        """Get market close time for given date."""
        if MarketHours.is_early_close(date):
            return MarketHours.EARLY_CLOSE
        return MarketHours.MARKET_CLOSE

    @staticmethod
    def is_market_open() -> bool:
        """Check if market is currently open."""
        now = datetime.now(ET)

        # Weekend check
        if now.weekday() >= 5:
            return False

        # Holiday check
        if MarketHours.is_holiday(now):
            return False

        current_time = now.time()
        close_time = MarketHours.get_close_time(now)
        return MarketHours.MARKET_OPEN <= current_time <= close_time
    
    @staticmethod
    def time_to_open() -> float:
        """Seconds until market open."""
        now = datetime.now(ET)
        
        if now.weekday() >= 5:  # Weekend
            days_until_monday = 7 - now.weekday()
            next_open = now.replace(
                hour=9, minute=30, second=0, microsecond=0
            ) + timedelta(days=days_until_monday)
        elif now.time() >= MarketHours.MARKET_CLOSE:
            next_open = now.replace(
                hour=9, minute=30, second=0, microsecond=0
            ) + timedelta(days=1)
        elif now.time() < MarketHours.MARKET_OPEN:
            next_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        else:
            return 0  # Market is open
        
        return (next_open - now).total_seconds()


from datetime import timedelta


class StrategyScheduler:
    """
    Schedules and runs strategies at appropriate times.

    Usage:
        scheduler = StrategyScheduler()
        scheduler.register_strategy('gap_fill', gap_fill_runner, '09:31')
        scheduler.start()
    """

    def __init__(self, circuit_breaker=None):
        self.db = SignalDatabase()
        self.tracker = ExecutionTracker(self.db)
        self.strategies: Dict[str, dict] = {}
        self.running = False
        self._thread = None
        self.circuit_breaker = circuit_breaker
    
    def register_strategy(
        self, 
        name: str, 
        runner: Callable, 
        run_time: str,
        market_hours_only: bool = True
    ):
        """
        Register a strategy to run at a specific time.
        
        Args:
            name: Strategy name
            runner: Function to call (should return list of signals)
            run_time: Time to run in HH:MM format (ET)
            market_hours_only: Only run on trading days
        """
        self.strategies[name] = {
            'runner': runner,
            'run_time': run_time,
            'market_hours_only': market_hours_only,
            'last_run': None,
            'enabled': STRATEGIES.get(name, {}).get('enabled', False)
        }
        
        logger.info(f"Registered {name} to run at {run_time} ET")
    
    def _should_run(self, strategy_name: str) -> bool:
        """Check if strategy should run now."""
        strat = self.strategies.get(strategy_name)
        if not strat:
            return False
        
        if not strat['enabled']:
            return False
        
        if strat['market_hours_only']:
            now = datetime.now(ET)
            if now.weekday() >= 5:  # Weekend
                return False
        
        return True
    
    def _run_strategy(self, name: str):
        """Execute a strategy."""
        # CIRCUIT BREAKER CHECK
        if self.circuit_breaker:
            if not self.circuit_breaker.can_trade():
                logger.info(f"Skipping {name}: trading halted by circuit breaker")
                return

            if not self.circuit_breaker.can_run_strategy(name):
                logger.info(f"Skipping {name}: strategy paused by circuit breaker")
                return

        if not self._should_run(name):
            logger.debug(f"Skipping {name} (not enabled or market closed)")
            return

        strat = self.strategies[name]
        logger.info(f"Running {name}...")
        
        try:
            start = time.time()
            signals = strat['runner']()
            elapsed = time.time() - start
            
            strat['last_run'] = datetime.now()
            
            if signals:
                logger.info(f"{name} generated {len(signals)} signals in {elapsed:.1f}s")
                for sig in signals:
                    self._process_signal(name, sig)
            else:
                logger.info(f"{name} completed in {elapsed:.1f}s (no signals)")
                
        except Exception as e:
            logger.error(f"{name} failed: {e}", exc_info=True)
    
    def _process_signal(self, strategy_name: str, signal: dict):
        """Process a signal with hybrid Shadow + Alpaca execution.

        Flow:
        1. Extract signal details (for universe_scan, strategy is in metadata)
        2. Evaluate through ExecutionManager (position limits, conviction, etc.)
        3. ALWAYS record in shadow (for metrics tracking)
        4. If GRADUATED: Also submit to Alpaca for realistic paper trading
        5. Record in signal tracker for monitoring

        This hybrid approach:
        - Unproven strategies: Shadow only until graduation criteria met
        - Graduated strategies: Alpaca + Shadow (dual execution for comparison)
        """
        # For universe scan, use the underlying strategy name from metadata
        actual_strategy = signal.get('metadata', {}).get('strategy', strategy_name)
        symbol = signal.get('symbol')
        direction = signal.get('direction', 'long')
        price = signal.get('price', 0)
        stop_loss = signal.get('stop_loss', 0)
        take_profit = signal.get('target', 0)
        confidence = signal.get('confidence', 0.5)
        metadata = signal.get('metadata', {})

        # 1. Evaluate through ExecutionManager
        exec_manager = _get_execution_manager()
        decision = exec_manager.evaluate_signal(
            strategy_name=actual_strategy,
            symbol=symbol,
            direction=direction,
            signal_type='entry',
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            quantity=0,  # Let ExecutionManager calculate sizing
            confidence=confidence,
            context={'metadata': metadata}
        )

        if not decision.approved:
            logger.info(f"Signal REJECTED: {symbol} ({actual_strategy}) - {decision.rejection_reason}")
            return

        shares = decision.final_shares or 1
        is_graduated = decision.route == 'live'

        # 2. ALWAYS record in shadow for metrics tracking
        shadow_trader = _get_shadow_trader()
        shadow_trade_id = None
        if shadow_trader:
            shadow_trade_id = self._execute_shadow_trade(
                shadow_trader, actual_strategy, symbol, direction, price, shares
            )

        # 3. If GRADUATED: Also submit to Alpaca
        alpaca_order_id = None
        if is_graduated:
            broker = _get_broker()
            try:
                order = broker.submit_market_order(
                    symbol=symbol,
                    qty=shares,
                    side='buy' if direction == 'long' else 'sell'
                )
                if order:
                    alpaca_order_id = str(order.id)
                    logger.info(f"ALPACA Order: {direction} {shares} {symbol} @ market (order_id={alpaca_order_id})")
                else:
                    logger.error(f"Alpaca order returned None for {symbol}")
            except Exception as e:
                logger.error(f"Failed to submit Alpaca order for {symbol}: {e}")
        else:
            logger.info(f"SHADOW ONLY: {direction} {shares} {symbol} @ ${price:.2f} ({actual_strategy} not graduated)")

        # 4. Record in signal tracker for monitoring
        try:
            self.tracker.record_signal_and_execute(
                strategy_name=actual_strategy,
                symbol=symbol,
                direction=direction,
                entry_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                quantity=shares,
                confidence=confidence,
                metadata={
                    **metadata,
                    'alpaca_order_id': alpaca_order_id,
                    'shadow_trade_id': shadow_trade_id,
                    'execution_mode': 'hybrid_live' if is_graduated else 'shadow_only',
                    'execution_decision': {
                        'conviction': decision.conviction,
                        'win_probability': decision.win_probability,
                        'route': decision.route,
                        'is_graduated': is_graduated
                    }
                }
            )
        finally:
            # Decrement pending approval counter now that position is persisted
            # This must happen regardless of success/failure to prevent counter drift
            exec_manager.decrement_pending_approval(actual_strategy)

    def _execute_shadow_trade(self, shadow_trader, strategy: str, symbol: str,
                               direction: str, price: float, shares: int) -> str:
        """Execute a trade in shadow mode and return trade ID.

        Auto-registers strategy if not already in shadow trading.
        """
        try:
            # Auto-register strategy if needed
            if strategy not in shadow_trader.strategies:
                shadow_trader.add_strategy(
                    name=strategy,
                    initial_capital=10000.0,
                    min_trades=30,
                    min_win_rate=0.55,
                    min_profit_factor=1.5,
                    min_days=14
                )
                logger.info(f"Auto-registered strategy '{strategy}' for shadow trading")

            # Process the signal
            signal_type = 'buy' if direction == 'long' else 'sell'
            trade_id = shadow_trader.process_signal(
                strategy=strategy,
                symbol=symbol,
                signal_type=signal_type,
                price=price,
                shares=shares
            )

            if trade_id:
                logger.debug(f"Shadow trade recorded: {trade_id}")
            return trade_id

        except Exception as e:
            logger.warning(f"Failed to record shadow trade for {symbol}: {e}")
            return None
    
    def _check_positions(self):
        """Check open positions for stops/targets."""
        if not MarketHours.is_market_open():
            return
        
        # Get current prices (would come from live data feed)
        # For now, skip if no live data
        logger.debug("Position check (would fetch live prices)")
    
    def _daily_summary(self):
        """Generate end-of-day summary."""
        logger.info("Generating daily summary...")
        report = self.tracker.generate_report()
        print(report)
        
        # Update performance stats
        for name in self.strategies.keys():
            self.db.update_strategy_performance(name)
    
    def setup_schedule(self):
        """Configure the schedule based on registered strategies.

        NOTE: All times in docstrings and config are ET (Eastern Time),
        but the schedule library uses local time. We convert ET -> local
        using et_to_local_time() to ensure jobs run at the right time.
        """
        # Clear existing jobs
        schedule.clear()

        # Schedule each strategy (convert ET to local time)
        for name, strat in self.strategies.items():
            et_time = strat['run_time']
            local_time = et_to_local_time(et_time)
            schedule.every().day.at(local_time).do(
                self._run_strategy, name
            )
            logger.info(f"Scheduled {name} at {et_time} ET ({local_time} local)")

        # Position checks every 15 min during market hours (9:30 - 16:00 ET)
        for hour in range(9, 16):
            for minute in [0, 15, 30, 45]:
                if hour == 9 and minute < 30:
                    continue
                et_time_str = f"{hour:02d}:{minute:02d}"
                local_time = et_to_local_time(et_time_str)
                schedule.every().day.at(local_time).do(self._check_positions)

        # EOD summary at 16:15 ET
        schedule.every().day.at(et_to_local_time("16:15")).do(self._daily_summary)

        # Expire old signals at midnight (local time is fine here)
        schedule.every().day.at("00:00").do(self.db.expire_old_signals)
    
    def _run_loop(self):
        """Main scheduler loop."""
        while self.running:
            schedule.run_pending()
            time.sleep(1)
    
    def start(self, blocking: bool = False):
        """Start the scheduler."""
        self.setup_schedule()
        self.running = True
        
        logger.info("Scheduler started")
        logger.info(f"Registered strategies: {list(self.strategies.keys())}")
        
        if blocking:
            self._run_loop()
        else:
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
    
    def stop(self):
        """Stop the scheduler."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Scheduler stopped")
    
    def run_now(self, strategy_name: str):
        """Manually trigger a strategy run."""
        if strategy_name in self.strategies:
            self._run_strategy(strategy_name)
        else:
            logger.warning(f"Unknown strategy: {strategy_name}")

    def run_missed_strategies(self) -> list:
        """
        Run any strategies that should have run today but missed their window.

        This handles the case where the scheduler starts AFTER some strategies'
        scheduled times. The schedule library won't run past-due jobs, so we
        check and run them manually.

        Returns:
            List of strategy names that were run
        """
        now_et = datetime.now(ET)
        current_time_et = now_et.time()
        today_date = now_et.date()

        # Only run during market hours
        if not MarketHours.is_market_open():
            logger.debug("Market closed - skipping missed strategy check")
            return []

        ran = []
        for name, strat in self.strategies.items():
            if not strat['enabled']:
                continue

            # Parse scheduled time (ET)
            hour, minute = map(int, strat['run_time'].split(':'))
            scheduled_time = dtime(hour, minute)

            # Check if scheduled time has passed today
            if current_time_et > scheduled_time:
                # Check if already ran today
                last_run = strat.get('last_run')
                if last_run and last_run.date() == today_date:
                    logger.debug(f"{name}: Already ran today at {last_run.time()}")
                    continue

                logger.info(f"{name}: Missed scheduled time ({strat['run_time']} ET), running now...")
                self._run_strategy(name)
                ran.append(name)

        if ran:
            logger.info(f"Ran {len(ran)} missed strategies: {ran}")
        return ran

    def status(self) -> dict:
        """Get scheduler status."""
        return {
            'running': self.running,
            'market_open': MarketHours.is_market_open(),
            'strategies': {
                name: {
                    'enabled': s['enabled'],
                    'run_time': s['run_time'],
                    'last_run': s['last_run'].isoformat() if s['last_run'] else None
                }
                for name, s in self.strategies.items()
            },
            'pending_jobs': len(schedule.jobs),
            'open_positions': len(self.db.get_open_positions())
        }


# Strategy runner functions

def _get_strategy_data():
    """Fetch data for strategy runners using CachedDataManager."""
    import random
    from data.cached_data_manager import CachedDataManager
    from config import PERF

    dm = CachedDataManager()

    # Get available symbols
    symbols = dm.get_available_symbols()
    if not symbols:
        logger.warning("No symbols available")
        return {}, 'normal'

    # Limit symbols for memory (from config), but RANDOMLY sample
    # to get diverse symbols across the alphabet, not just "A" stocks
    max_symbols = PERF.get('max_symbols', 100)
    if len(symbols) > max_symbols:
        # Use consistent seed based on date so same symbols throughout day
        today_seed = int(datetime.now().strftime('%Y%m%d'))
        random.seed(today_seed)
        symbols = random.sample(symbols, max_symbols)
        random.seed()  # Reset to random state

    dm.load_all(symbols)

    # Build data dict
    data = dm.get_bars_batch(symbols)
    if not data:
        logger.warning("No data loaded")
        return {}, 'normal'

    # Get current VIX regime
    vix_regime = 'normal'
    try:
        current_vix = dm.get_vix()
        if current_vix < 15:
            vix_regime = 'low'
        elif current_vix < 25:
            vix_regime = 'normal'
        elif current_vix < 35:
            vix_regime = 'high'
        else:
            vix_regime = 'extreme'
        logger.info(f"VIX: {current_vix:.1f} -> regime: {vix_regime}")
    except Exception as e:
        logger.warning(f"Could not determine VIX regime: {e}")

    return data, vix_regime


def _signals_to_dicts(signals):
    """Convert Signal objects to dicts for scheduler."""
    return [
        {
            'symbol': s.symbol,
            'direction': 'long' if s.signal_type.name == 'BUY' else 'short',
            'price': s.price,
            'stop_loss': s.metadata.get('stop_loss', 0),
            'target': s.metadata.get('target', 0),
            'confidence': s.strength,
            'metadata': s.metadata
        }
        for s in signals
    ]


def run_gap_fill():
    """Gap-fill strategy runner."""
    from strategies.gap_fill import GapFillStrategy

    data, vix_regime = _get_strategy_data()
    if not data:
        return []

    strategy = GapFillStrategy()
    signals = strategy.generate_signals(data=data, vix_regime=vix_regime)
    return _signals_to_dicts(signals)


def run_mean_reversion():
    """Mean reversion strategy runner."""
    from strategies.mean_reversion import MeanReversionStrategy

    data, vix_regime = _get_strategy_data()
    if not data:
        return []

    strategy = MeanReversionStrategy()
    signals = strategy.generate_signals(data=data, vix_regime=vix_regime)
    return _signals_to_dicts(signals)


def run_vix_regime_rotation():
    """VIX regime rotation strategy runner."""
    from strategies.vix_regime_rotation import VIXRegimeRotationStrategy

    data, vix_regime = _get_strategy_data()
    if not data:
        return []

    strategy = VIXRegimeRotationStrategy()
    signals = strategy.generate_signals(data=data, vix_regime=vix_regime)
    return _signals_to_dicts(signals)


def run_vol_managed_momentum():
    """Volatility-managed momentum strategy runner (V2)."""
    from strategies.vol_managed_momentum_v2 import VolManagedMomentumV2

    data, vix_regime = _get_strategy_data()
    if not data:
        return []

    strategy = VolManagedMomentumV2()
    signals = strategy.generate_signals(data=data, vix_regime=vix_regime)
    return _signals_to_dicts(signals)


def run_quality_smallcap_value():
    """Quality small-cap value strategy runner."""
    from strategies.quality_small_cap_value import QualitySmallCapValueStrategy

    data, vix_regime = _get_strategy_data()
    if not data:
        return []

    strategy = QualitySmallCapValueStrategy()
    signals = strategy.generate_signals(data=data, vix_regime=vix_regime)
    return _signals_to_dicts(signals)


def run_factor_momentum():
    """Factor momentum strategy runner."""
    from strategies.factor_momentum import FactorMomentumStrategy

    data, vix_regime = _get_strategy_data()
    if not data:
        return []

    strategy = FactorMomentumStrategy()
    signals = strategy.generate_signals(data=data, vix_regime=vix_regime)
    return _signals_to_dicts(signals)


def run_rv_breakout():
    """RV Breakout strategy runner."""
    from strategies.relative_volume_breakout import RelativeVolumeBreakout

    data, vix_regime = _get_strategy_data()
    if not data:
        return []

    strategy = RelativeVolumeBreakout()
    signals = strategy.generate_signals(data=data, vix_regime=vix_regime)
    return _signals_to_dicts(signals)


def run_universe_scan():
    """
    Full universe scan using UniverseScanner.

    Scans ~2,500 symbols in batches, runs all enabled strategies,
    and returns validated signals. Takes ~2-3 minutes.

    This replaces individual strategy runners with a single comprehensive scan.
    """
    from execution.universe_scanner import create_scanner_with_strategies

    logger.info("Starting full universe scan...")

    try:
        scanner = create_scanner_with_strategies()
        validated_signals = scanner.scan()

        # Convert CandidateSignal objects to scheduler format
        results = []
        for sig in validated_signals:
            results.append({
                'symbol': sig.symbol,
                'direction': sig.direction,
                'price': sig.price,
                'stop_loss': sig.stop_loss,
                'target': sig.take_profit,
                'confidence': sig.phase2_confidence or sig.confidence,
                'metadata': {
                    **sig.metadata,
                    'strategy': sig.strategy,
                    'phase1_confidence': sig.confidence,
                    'validated': sig.validated,
                }
            })

        logger.info(f"Universe scan complete: {len(results)} validated signals")
        return results

    except Exception as e:
        logger.error(f"Universe scan failed: {e}", exc_info=True)
        return []


def create_default_scheduler(circuit_breaker=None, use_universe_scan: bool = True) -> StrategyScheduler:
    """Create scheduler with default strategy configuration.

    Args:
        circuit_breaker: Optional circuit breaker for trade limits
        use_universe_scan: If True (default), use full universe scanner at 10:00 ET.
                          If False, use individual strategy runners (legacy mode).

    Run times (ET):
    - 09:31: gap_fill (needs early gap detection - always separate)
    - 10:00: universe_scan (scans ~2,500 symbols with all strategies)
             OR individual strategies in legacy mode

    The universe scanner:
    - Scans entire market in memory-safe batches (~128 symbols/batch)
    - Runs all enabled strategies on full universe
    - Two-phase validation for signal quality
    - Takes ~3 minutes, generates validated signals
    """
    scheduler = StrategyScheduler(circuit_breaker=circuit_breaker)

    # Gap-fill always runs separately at market open (time-sensitive)
    scheduler.register_strategy('gap_fill', run_gap_fill, '09:31')

    if use_universe_scan:
        # Full universe scan at 10:00 ET - runs all enabled strategies
        # on ~2,500 symbols with two-phase validation
        scheduler.register_strategy('universe_scan', run_universe_scan, '10:00')
    else:
        # Legacy mode: individual strategy runners (100 symbols each)
        scheduler.register_strategy('mean_reversion', run_mean_reversion, '10:00')
        scheduler.register_strategy('relative_volume_breakout', run_rv_breakout, '10:00')
        scheduler.register_strategy('vix_regime_rotation', run_vix_regime_rotation, '10:30')
        scheduler.register_strategy('vol_managed_momentum', run_vol_managed_momentum, '10:30')
        scheduler.register_strategy('quality_smallcap_value', run_quality_smallcap_value, '11:00')
        scheduler.register_strategy('factor_momentum', run_factor_momentum, '11:00')

    return scheduler


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s'
    )
    
    print("="*60)
    print("STRATEGY SCHEDULER")
    print("="*60)
    
    scheduler = create_default_scheduler()
    
    print(f"\nStatus: {scheduler.status()}")
    print(f"\nMarket open: {MarketHours.is_market_open()}")
    
    if MarketHours.time_to_open() > 0:
        print(f"Time to open: {MarketHours.time_to_open()/3600:.1f} hours")
    
    print("\nTo start scheduler:")
    print("  scheduler.start(blocking=True)")
    print("\nTo run a strategy now:")
    print("  scheduler.run_now('pairs_trading')")
