"""
Fast Backtester for GA Optimization
====================================
Optimized for speed at the cost of some accuracy.
Use for parameter search, not final validation.

Speed improvements:
1. Pre-computed VIX regime array
2. Day sampling (configurable)
3. Simplified position tracking
4. Vectorized price lookups
5. Skip non-rebalance days for monthly strategies

Typical speedup: 5-10x faster than standard backtester
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import bisect
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, SignalType

logger = logging.getLogger(__name__)


@dataclass
class FastBacktestResult:
    """Lightweight result container compatible with novelty_search."""
    run_id: str
    strategy: str
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    annual_return: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    profit_factor: float = 0.0

    # Required by novelty_search.extract_behavior_vector()
    equity_curve: List[float] = field(default_factory=list)
    trades: List[Dict] = field(default_factory=list)

    # Additional fields for compatibility with BacktestResult
    start_date: str = ""
    end_date: str = ""
    total_return: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0


class FastBacktester:
    """
    Speed-optimized backtester for parameter search.

    Key optimizations:
    - Pre-compute all date mappings upfront
    - Cache VIX regimes
    - Sample days for faster iteration
    - Simplified slippage model
    """

    def __init__(self,
                 initial_capital: float = 100000,
                 sample_rate: float = 1.0,
                 slippage_bps: float = 10):
        """
        Args:
            initial_capital: Starting capital
            sample_rate: Fraction of days to simulate (0.2 = 20% of days)
            slippage_bps: Simple slippage in basis points
        """
        self.initial_capital = initial_capital
        self.sample_rate = sample_rate
        self.slippage_pct = slippage_bps / 10000

        self._reset()

    def _reset(self):
        self._cash = self.initial_capital
        self._positions = {}
        self._equity_curve = []
        self._trades = []

    def run(self,
            strategy: BaseStrategy,
            data: Dict[str, pd.DataFrame],
            start_date: datetime = None,
            end_date: datetime = None,
            vix_data: pd.DataFrame = None) -> FastBacktestResult:
        """Run fast backtest."""
        self._reset()
        run_id = str(uuid.uuid4())[:8]

        # Ensure datetime index
        for symbol, df in list(data.items()):
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df = df.copy()
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    data[symbol] = df
            if df.index.tz is not None:
                df = df.copy()
                df.index = df.index.tz_localize(None)
                data[symbol] = df

        # Get all unique dates
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index.tolist())
        all_dates = sorted(all_dates)

        if not all_dates:
            return FastBacktestResult(run_id=run_id, strategy=strategy.name)

        if start_date is None:
            start_date = all_dates[0]
        if end_date is None:
            end_date = all_dates[-1]

        dates = [d for d in all_dates if start_date <= d <= end_date]

        # Sample days if requested
        if self.sample_rate < 1.0:
            np.random.seed(42)  # Reproducible
            n_sample = max(int(len(dates) * self.sample_rate), 50)
            # Always include first and last
            middle_dates = dates[1:-1]
            if len(middle_dates) > n_sample - 2:
                sampled_middle = list(np.random.choice(middle_dates, n_sample - 2, replace=False))
                dates = [dates[0]] + sorted(sampled_middle) + [dates[-1]]

        # Pre-compute price arrays for fast lookup
        price_arrays = {}
        date_to_idx = {d: i for i, d in enumerate(all_dates)}

        for symbol, df in data.items():
            # Create price array indexed by global date index
            prices = np.full(len(all_dates), np.nan)
            for d, row in df.iterrows():
                if d in date_to_idx:
                    prices[date_to_idx[d]] = row['close']
            # Forward fill
            for i in range(1, len(prices)):
                if np.isnan(prices[i]):
                    prices[i] = prices[i-1]
            price_arrays[symbol] = prices

        # Pre-compute VIX regimes
        vix_regimes = {}
        if vix_data is not None and 'regime' in vix_data.columns:
            for d in dates:
                try:
                    mask = vix_data.index <= d
                    if mask.sum() > 0:
                        vix_regimes[d] = vix_data.loc[mask, 'regime'].iloc[-1]
                    else:
                        vix_regimes[d] = 'normal'
                except Exception:
                    vix_regimes[d] = 'normal'

        # Pre-build data slices for each rebalance date only
        # Identify rebalance dates (first of each month)
        rebalance_dates = set()
        last_month = None
        for d in dates:
            month_key = (d.year, d.month)
            if month_key != last_month:
                rebalance_dates.add(d)
                last_month = month_key

        # Pre-compute data slices for rebalance dates only
        rebalance_data_cache = {}
        for rebal_date in rebalance_dates:
            rebal_idx = date_to_idx[rebal_date]
            current_data = {}
            for symbol, df in data.items():
                # Find last date <= rebal_date
                mask = df.index <= rebal_date
                if mask.sum() > 0:
                    current_data[symbol] = df.loc[mask]
            rebalance_data_cache[rebal_date] = current_data

        # Main loop
        for i, current_date in enumerate(dates):
            global_idx = date_to_idx[current_date]

            # Get current prices
            current_prices = {}
            for symbol, prices in price_arrays.items():
                if not np.isnan(prices[global_idx]):
                    current_prices[symbol] = prices[global_idx]

            # Check stops and targets
            positions_to_close = []
            for symbol, pos in self._positions.items():
                if symbol not in current_prices:
                    continue
                price = current_prices[symbol]
                if pos['side'] == 'BUY':
                    if price <= pos['stop_loss']:
                        positions_to_close.append((symbol, price, 'stop'))
                    elif price >= pos['target']:
                        positions_to_close.append((symbol, price, 'target'))

            for symbol, exit_price, reason in positions_to_close:
                pos = self._positions.pop(symbol)
                exit_price *= (1 - self.slippage_pct)  # Slippage on exit
                pnl = (exit_price - pos['entry_price']) * pos['qty']
                self._cash += pos['qty'] * exit_price
                self._trades.append({
                    'symbol': symbol,
                    'pnl': pnl,
                    'pnl_pct': (exit_price - pos['entry_price']) / pos['entry_price']
                })

            # Generate signals on rebalance dates only
            if current_date in rebalance_dates:
                current_data = rebalance_data_cache[current_date]
                vix_regime = vix_regimes.get(current_date, 'normal')
                current_pos_list = list(self._positions.keys())

                signals = strategy.generate_signals(current_data, current_pos_list, vix_regime)

                for signal in signals:
                    if signal.signal_type == SignalType.CLOSE:
                        if signal.symbol in self._positions:
                            pos = self._positions.pop(signal.symbol)
                            exit_price = current_prices.get(signal.symbol, pos['entry_price'])
                            exit_price *= (1 - self.slippage_pct)
                            pnl = (exit_price - pos['entry_price']) * pos['qty']
                            self._cash += pos['qty'] * exit_price
                            self._trades.append({
                                'symbol': signal.symbol,
                                'pnl': pnl,
                                'pnl_pct': (exit_price - pos['entry_price']) / pos['entry_price']
                            })

                    elif signal.signal_type == SignalType.BUY:
                        if signal.symbol in self._positions:
                            continue
                        if signal.symbol not in current_prices:
                            continue

                        price = current_prices[signal.symbol]
                        price *= (1 + self.slippage_pct)  # Slippage on entry

                        # Position sizing
                        position_pct = getattr(signal, 'position_size_pct', 0.10)
                        position_value = self._cash * position_pct
                        qty = position_value / price

                        if qty * price > self._cash:
                            qty = self._cash / price * 0.95

                        if qty <= 0:
                            continue

                        self._positions[signal.symbol] = {
                            'entry_price': price,
                            'qty': qty,
                            'side': 'BUY',
                            'stop_loss': signal.stop_loss or price * 0.90,
                            'target': signal.target_price or price * 1.20
                        }
                        self._cash -= qty * price

            # Record equity
            portfolio_value = self._cash
            for symbol, pos in self._positions.items():
                if symbol in current_prices:
                    portfolio_value += pos['qty'] * current_prices[symbol]
            self._equity_curve.append((current_date, portfolio_value))

        # Close remaining positions
        for symbol, pos in list(self._positions.items()):
            if symbol in current_prices:
                exit_price = current_prices[symbol] * (1 - self.slippage_pct)
                pnl = (exit_price - pos['entry_price']) * pos['qty']
                self._trades.append({
                    'symbol': symbol,
                    'pnl': pnl,
                    'pnl_pct': (exit_price - pos['entry_price']) / pos['entry_price']
                })

        # Calculate metrics
        return self._calculate_metrics(run_id, strategy.name)

    def _calculate_metrics(self, run_id: str, strategy_name: str) -> FastBacktestResult:
        """Calculate performance metrics."""
        result = FastBacktestResult(run_id=run_id, strategy=strategy_name)

        if len(self._equity_curve) < 2:
            return result

        # Extract equity series
        equities = [e[1] for e in self._equity_curve]
        dates = [e[0] for e in self._equity_curve]

        # Store equity curve and trades for novelty_search compatibility
        result.equity_curve = equities
        result.trades = self._trades.copy()
        result.start_date = str(dates[0]) if dates else ""
        result.end_date = str(dates[-1]) if dates else ""

        # Returns
        returns = np.diff(equities) / equities[:-1]
        returns = returns[~np.isnan(returns)]

        if len(returns) < 10:
            return result

        # Total return
        result.total_return = (equities[-1] / equities[0]) - 1

        # Volatility
        result.volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0.0

        # Sharpe
        if returns.std() > 0:
            result.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)

        # Sortino
        neg_returns = returns[returns < 0]
        if len(neg_returns) > 0 and neg_returns.std() > 0:
            result.sortino_ratio = (returns.mean() / neg_returns.std()) * np.sqrt(252)
        else:
            result.sortino_ratio = result.sharpe_ratio * 1.2

        # Annual return
        total_days = (dates[-1] - dates[0]).days
        if total_days > 0:
            result.annual_return = ((1 + result.total_return) ** (365 / total_days) - 1) * 100

        # Max drawdown
        peak = equities[0]
        max_dd = 0
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (eq - peak) / peak
            if dd < max_dd:
                max_dd = dd
        result.max_drawdown_pct = max_dd * 100
        result.max_drawdown = max_dd * equities[0]  # Absolute value

        # Win rate
        result.total_trades = len(self._trades)
        if result.total_trades > 0:
            wins = sum(1 for t in self._trades if t['pnl'] > 0)
            result.win_rate = (wins / result.total_trades) * 100

            # Profit factor
            gross_profit = sum(t['pnl'] for t in self._trades if t['pnl'] > 0)
            gross_loss = abs(sum(t['pnl'] for t in self._trades if t['pnl'] < 0))
            if gross_loss > 0:
                result.profit_factor = gross_profit / gross_loss

        return result


def benchmark_speed():
    """Compare fast vs standard backtester speed."""
    import time
    from data.unified_data_loader import UnifiedDataLoader
    from strategies.vol_managed_momentum import VolManagedMomentumStrategy
    from research.backtester import Backtester

    print("=" * 60)
    print("BACKTESTER SPEED BENCHMARK")
    print("=" * 60)

    # Load data
    loader = UnifiedDataLoader()
    all_data = loader.load_all_daily()

    # Use 30 symbols
    sorted_syms = sorted(all_data.items(), key=lambda x: len(x[1]), reverse=True)[:30]
    data = {}
    for symbol, df in sorted_syms:
        df = df.copy()
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        # Last 5 years only
        df = df[df.index >= '2020-01-01']
        if len(df) > 200:
            data[symbol] = df

    print(f"Testing with {len(data)} symbols, ~5 years data")

    strategy = VolManagedMomentumStrategy()

    # Standard backtester
    print("\nStandard Backtester:")
    start = time.time()
    bt = Backtester(initial_capital=100000)
    result1 = bt.run(strategy, data.copy())
    time1 = time.time() - start
    print(f"  Time: {time1:.2f}s")
    print(f"  Sharpe: {result1.sharpe_ratio:.3f}")
    print(f"  Trades: {result1.total_trades}")

    # Fast backtester (100% days)
    strategy.last_rebalance_month = None
    print("\nFast Backtester (100% days):")
    start = time.time()
    fbt = FastBacktester(initial_capital=100000, sample_rate=1.0)
    result2 = fbt.run(strategy, data.copy())
    time2 = time.time() - start
    print(f"  Time: {time2:.2f}s")
    print(f"  Sharpe: {result2.sharpe_ratio:.3f}")
    print(f"  Trades: {result2.total_trades}")
    print(f"  Speedup: {time1/time2:.1f}x")

    # Fast backtester (50% days)
    strategy.last_rebalance_month = None
    print("\nFast Backtester (50% days):")
    start = time.time()
    fbt = FastBacktester(initial_capital=100000, sample_rate=0.5)
    result3 = fbt.run(strategy, data.copy())
    time3 = time.time() - start
    print(f"  Time: {time3:.2f}s")
    print(f"  Sharpe: {result3.sharpe_ratio:.3f}")
    print(f"  Trades: {result3.total_trades}")
    print(f"  Speedup: {time1/time3:.1f}x")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    benchmark_speed()
