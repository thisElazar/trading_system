"""
Opening Range Breakout (ORB) Backtest
=====================================
Backtest the ORB strategy using historical 1-minute data.

Tests the hypothesis: Breakouts from the first 30-minute range
continue in the breakout direction.
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.intraday.orb.config import ORBConfig

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Record of a completed trade."""
    symbol: str
    date: str
    direction: str
    entry_price: float
    entry_time: str
    or_high: float
    or_low: float
    or_range_pct: float
    exit_price: float
    exit_time: str
    exit_reason: str
    shares: int
    pnl: float
    pnl_pct: float
    hold_minutes: int


@dataclass
class BacktestResult:
    """Results from the backtest."""
    trades: List[Trade]
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    avg_pnl_pct: float = 0.0
    avg_hold_minutes: float = 0.0
    exits_by_reason: Dict[str, int] = field(default_factory=dict)
    long_trades: int = 0
    short_trades: int = 0
    long_win_rate: float = 0.0
    short_win_rate: float = 0.0
    # ORB-specific
    avg_or_range_pct: float = 0.0
    days_with_valid_range: int = 0
    days_with_breakout: int = 0


class ORBBacktester:
    """Backtests the Opening Range Breakout strategy."""

    def __init__(
        self,
        symbols: List[str] = None,
        config: ORBConfig = None,
        data_dir: Path = None
    ):
        self.symbols = symbols or ['SPY', 'QQQ', 'IWM', 'DIA']
        self.config = config or ORBConfig()
        self.data_dir = data_dir or Path(__file__).parent.parent / 'data' / 'historical'

    def _load_intraday_data(self, symbol: str, date_str: str) -> Optional[pd.DataFrame]:
        """Load 1-minute data for a specific date."""
        intraday_path = self.data_dir / 'intraday' / symbol / f'{date_str}.parquet'
        if not intraday_path.exists():
            return None
        return pd.read_parquet(intraday_path)

    def _get_available_dates(self, symbol: str) -> List[str]:
        """Get list of available trading dates."""
        intraday_dir = self.data_dir / 'intraday' / symbol
        if not intraday_dir.exists():
            return []
        return sorted([f.stem for f in intraday_dir.glob('*.parquet')])

    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate cumulative VWAP."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        cum_vol = df['volume'].cumsum()
        cum_pv = (typical_price * df['volume']).cumsum()
        vwap = cum_pv / cum_vol
        return vwap

    def _simulate_day(self, symbol: str, date_str: str, df: pd.DataFrame) -> tuple:
        """
        Simulate one trading day.
        Returns (trade or None, valid_range: bool, had_breakout: bool)
        """
        cfg = self.config

        # Handle timezone
        if df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_convert('America/New_York')

        # Need enough bars for opening range + trading
        if len(df) < cfg.range_minutes + 10:
            return None, False, False

        # Calculate opening range
        or_bars = df.iloc[:cfg.range_minutes]
        or_high = float(or_bars['high'].max())
        or_low = float(or_bars['low'].min())
        or_range_pct = (or_high - or_low) / or_low * 100 if or_low > 0 else 0
        or_range_dollars = or_high - or_low

        # Check if range is valid
        if or_range_pct < cfg.min_range_pct:
            logger.debug(f"{symbol} {date_str}: Range {or_range_pct:.2f}% too small")
            return None, False, False

        if or_range_pct > cfg.max_range_pct:
            logger.debug(f"{symbol} {date_str}: Range {or_range_pct:.2f}% too large")
            return None, False, False

        # Calculate VWAP for the full day
        df['vwap'] = self._calculate_vwap(df)
        df['rel_volume'] = df['volume'] / df['volume'].rolling(20).mean()

        # Breakout thresholds
        breakout_up = or_high * (1 + cfg.breakout_buffer_pct / 100)
        breakout_down = or_low * (1 - cfg.breakout_buffer_pct / 100)

        # Position tracking
        position = None
        entry_bar_idx = 0
        had_breakout = False

        for idx in range(cfg.entry_start_minute, min(len(df), cfg.force_exit_minute)):
            bar = df.iloc[idx]
            price = float(bar['close'])
            vwap = float(bar['vwap']) if not np.isnan(bar['vwap']) else price
            rel_vol = float(bar['rel_volume']) if not np.isnan(bar['rel_volume']) else 1.0

            # Manage existing position
            if position is not None:
                hold_minutes = idx - entry_bar_idx

                # Update trailing stop if active
                if cfg.use_trailing_stop and position.get('trailing_active'):
                    trail_distance = or_range_dollars * cfg.trailing_stop_pct
                    if position['direction'] == 'LONG':
                        new_stop = price - trail_distance
                        if new_stop > position['stop_loss']:
                            position['stop_loss'] = new_stop
                    else:
                        new_stop = price + trail_distance
                        if new_stop < position['stop_loss']:
                            position['stop_loss'] = new_stop

                # Check exits
                exit_reason = None

                # Stop loss
                if position['direction'] == 'LONG':
                    if price <= position['stop_loss']:
                        exit_reason = 'stop_loss'
                    elif price >= position['target']:
                        exit_reason = 'target'
                else:
                    if price >= position['stop_loss']:
                        exit_reason = 'stop_loss'
                    elif price <= position['target']:
                        exit_reason = 'target'

                # Activate trailing stop after 1x range profit
                if cfg.use_trailing_stop and not position.get('trailing_active'):
                    if position['direction'] == 'LONG':
                        profit = price - position['entry_price']
                        if profit >= or_range_dollars:
                            position['trailing_active'] = True
                    else:
                        profit = position['entry_price'] - price
                        if profit >= or_range_dollars:
                            position['trailing_active'] = True

                # Time limit
                if hold_minutes >= cfg.max_hold_minutes:
                    exit_reason = 'time_limit'

                # Force exit near close
                if idx >= cfg.force_exit_minute - 1:
                    exit_reason = 'eod_exit'

                if exit_reason:
                    if position['direction'] == 'LONG':
                        pnl = (price - position['entry_price']) * position['shares']
                    else:
                        pnl = (position['entry_price'] - price) * position['shares']

                    pnl_pct = pnl / (position['entry_price'] * position['shares']) * 100

                    trade = Trade(
                        symbol=symbol,
                        date=date_str,
                        direction=position['direction'],
                        entry_price=position['entry_price'],
                        entry_time=position['entry_time'],
                        or_high=or_high,
                        or_low=or_low,
                        or_range_pct=or_range_pct,
                        exit_price=price,
                        exit_time=str(bar.name.time())[:5] if hasattr(bar.name, 'time') else f'{idx}',
                        exit_reason=exit_reason,
                        shares=position['shares'],
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        hold_minutes=hold_minutes
                    )
                    return trade, True, True

                continue

            # Check for new entry (only one trade per day)
            if idx > cfg.entry_end_minute:
                continue

            # Volume confirmation
            if rel_vol < cfg.min_relative_volume:
                continue

            direction = None

            # Upside breakout
            if price > breakout_up:
                if cfg.require_vwap_alignment and price < vwap:
                    continue
                direction = 'LONG'
                stop_loss = or_low * (1 - cfg.stop_loss_buffer_pct / 100)
                target = price + (or_range_dollars * cfg.target_multiple)

            # Downside breakout
            elif not cfg.long_only and price < breakout_down:
                if cfg.require_vwap_alignment and price > vwap:
                    continue
                direction = 'SHORT'
                stop_loss = or_high * (1 + cfg.stop_loss_buffer_pct / 100)
                target = price - (or_range_dollars * cfg.target_multiple)

            if direction is None:
                continue

            had_breakout = True

            # Position sizing
            shares = int(cfg.max_position_value / price)
            if shares <= 0:
                continue

            position = {
                'direction': direction,
                'entry_price': price,
                'entry_time': str(bar.name.time())[:5] if hasattr(bar.name, 'time') else f'{idx}',
                'shares': shares,
                'stop_loss': stop_loss,
                'target': target,
                'trailing_active': False,
            }
            entry_bar_idx = idx

        # Force close any remaining position
        if position is not None:
            last_bar = df.iloc[-1]
            price = float(last_bar['close'])
            hold_minutes = len(df) - 1 - entry_bar_idx

            if position['direction'] == 'LONG':
                pnl = (price - position['entry_price']) * position['shares']
            else:
                pnl = (position['entry_price'] - price) * position['shares']

            pnl_pct = pnl / (position['entry_price'] * position['shares']) * 100

            trade = Trade(
                symbol=symbol,
                date=date_str,
                direction=position['direction'],
                entry_price=position['entry_price'],
                entry_time=position['entry_time'],
                or_high=or_high,
                or_low=or_low,
                or_range_pct=or_range_pct,
                exit_price=price,
                exit_time='EOD',
                exit_reason='market_close',
                shares=position['shares'],
                pnl=pnl,
                pnl_pct=pnl_pct,
                hold_minutes=hold_minutes
            )
            return trade, True, True

        return None, True, had_breakout

    def run(self, long_only: bool = None) -> BacktestResult:
        """Run the backtest."""
        if long_only is not None:
            self.config.long_only = long_only

        all_trades: List[Trade] = []
        days_with_valid_range = 0
        days_with_breakout = 0

        for symbol in self.symbols:
            dates = self._get_available_dates(symbol)
            logger.info(f"\n{symbol}: {len(dates)} trading days")

            for date_str in dates:
                df = self._load_intraday_data(symbol, date_str)
                if df is None or len(df) < 60:
                    continue

                trade, valid_range, had_breakout = self._simulate_day(symbol, date_str, df)

                if valid_range:
                    days_with_valid_range += 1
                if had_breakout:
                    days_with_breakout += 1
                if trade:
                    all_trades.append(trade)

        # Calculate results
        result = BacktestResult(trades=all_trades)
        result.total_trades = len(all_trades)
        result.days_with_valid_range = days_with_valid_range
        result.days_with_breakout = days_with_breakout

        if len(all_trades) > 0:
            result.winning_trades = sum(1 for t in all_trades if t.pnl > 0)
            result.losing_trades = sum(1 for t in all_trades if t.pnl <= 0)
            result.win_rate = result.winning_trades / len(all_trades) * 100
            result.total_pnl = sum(t.pnl for t in all_trades)
            result.avg_pnl = result.total_pnl / len(all_trades)
            result.avg_pnl_pct = sum(t.pnl_pct for t in all_trades) / len(all_trades)
            result.avg_hold_minutes = sum(t.hold_minutes for t in all_trades) / len(all_trades)
            result.avg_or_range_pct = sum(t.or_range_pct for t in all_trades) / len(all_trades)

            for trade in all_trades:
                reason = trade.exit_reason
                result.exits_by_reason[reason] = result.exits_by_reason.get(reason, 0) + 1

            long_trades = [t for t in all_trades if t.direction == 'LONG']
            short_trades = [t for t in all_trades if t.direction == 'SHORT']
            result.long_trades = len(long_trades)
            result.short_trades = len(short_trades)
            if long_trades:
                result.long_win_rate = sum(1 for t in long_trades if t.pnl > 0) / len(long_trades) * 100
            if short_trades:
                result.short_win_rate = sum(1 for t in short_trades if t.pnl > 0) / len(short_trades) * 100

        return result

    def print_results(self, result: BacktestResult):
        """Print formatted results."""
        print("\n" + "="*60)
        print("OPENING RANGE BREAKOUT (ORB) BACKTEST RESULTS")
        print("="*60)

        print(f"\nSymbols: {', '.join(self.symbols)}")
        print(f"Opening Range: First {self.config.range_minutes} minutes")
        print(f"Range Filter: {self.config.min_range_pct}% - {self.config.max_range_pct}%")
        print(f"Breakout Buffer: {self.config.breakout_buffer_pct}%")
        print(f"Target Multiple: {self.config.target_multiple}x range")
        print(f"VWAP Alignment: {self.config.require_vwap_alignment}")
        print(f"Long Only: {self.config.long_only}")

        print("\n" + "-"*40)
        print("OPPORTUNITY ANALYSIS")
        print("-"*40)
        print(f"Days with Valid Range: {result.days_with_valid_range}")
        print(f"Days with Breakout Signal: {result.days_with_breakout}")
        print(f"Trades Executed: {result.total_trades}")
        if result.total_trades > 0:
            print(f"Avg Opening Range: {result.avg_or_range_pct:.2f}%")

        print("\n" + "-"*40)
        print("PERFORMANCE SUMMARY")
        print("-"*40)
        print(f"Total Trades: {result.total_trades}")
        print(f"Winning Trades: {result.winning_trades}")
        print(f"Losing Trades: {result.losing_trades}")
        print(f"Win Rate: {result.win_rate:.1f}%")
        print(f"Total P&L: ${result.total_pnl:,.2f}")
        print(f"Avg P&L per Trade: ${result.avg_pnl:,.2f}")
        print(f"Avg P&L %: {result.avg_pnl_pct:.3f}%")
        print(f"Avg Hold Time: {result.avg_hold_minutes:.0f} minutes")

        print("\n" + "-"*40)
        print("BY EXIT REASON")
        print("-"*40)
        for reason, count in sorted(result.exits_by_reason.items()):
            pct = count / result.total_trades * 100 if result.total_trades > 0 else 0
            print(f"  {reason}: {count} ({pct:.1f}%)")

        print("\n" + "-"*40)
        print("BY DIRECTION")
        print("-"*40)
        print(f"Long Trades: {result.long_trades} (Win Rate: {result.long_win_rate:.1f}%)")
        print(f"Short Trades: {result.short_trades} (Win Rate: {result.short_win_rate:.1f}%)")

        if result.trades:
            print("\n" + "-"*40)
            print("SAMPLE TRADES (first 10)")
            print("-"*40)
            for trade in result.trades[:10]:
                icon = "+" if trade.pnl > 0 else ""
                print(
                    f"{trade.date} {trade.symbol} {trade.direction} | "
                    f"OR: ${trade.or_low:.2f}-${trade.or_high:.2f} ({trade.or_range_pct:.2f}%) | "
                    f"Entry: ${trade.entry_price:.2f} -> Exit: ${trade.exit_price:.2f} ({trade.exit_reason}) | "
                    f"P&L: {icon}${trade.pnl:.2f} | Hold: {trade.hold_minutes}m"
                )

        print("\n" + "="*60)


def main():
    """Run ORB backtest."""
    config = ORBConfig(
        symbols=['SPY', 'QQQ', 'IWM', 'DIA'],
        range_minutes=30,
        min_range_pct=0.15,
        max_range_pct=1.5,
        breakout_buffer_pct=0.02,
        min_relative_volume=1.5,
        require_vwap_alignment=True,
        target_multiple=1.5,
        use_trailing_stop=True,
        trailing_stop_pct=0.5,
        max_hold_minutes=240,
        long_only=True,
        portfolio_value=97000.0,
        max_position_pct=0.05
    )

    backtester = ORBBacktester(config=config)
    result = backtester.run()
    backtester.print_results(result)

    return result


if __name__ == "__main__":
    main()
