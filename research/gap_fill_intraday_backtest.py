"""
Gap-Fill Intraday Backtest
==========================
Backtest the gap-fill strategy using historical 1-minute data.

Tests the hypothesis: gaps 0.15%-0.60% fill 75% within 120 minutes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.intraday.gap_fill.detector import GapDetector, Gap
from strategies.intraday.gap_fill.config import GapFillConfig

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Record of a completed trade."""
    symbol: str
    date: str
    gap_pct: float
    gap_direction: str
    entry_price: float
    entry_time: str
    exit_price: float
    exit_time: str
    exit_reason: str
    shares: int
    pnl: float
    pnl_pct: float
    hold_minutes: int
    gap_filled_pct: float


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
    gaps_detected: int = 0
    gaps_traded: int = 0

    # By exit reason
    exits_by_reason: Dict[str, int] = field(default_factory=dict)

    # By direction
    long_trades: int = 0
    short_trades: int = 0
    long_win_rate: float = 0.0
    short_win_rate: float = 0.0


class GapFillBacktester:
    """
    Backtests the gap-fill strategy on historical 1-minute data.
    """

    def __init__(
        self,
        symbols: List[str] = None,
        config: GapFillConfig = None,
        data_dir: Path = None
    ):
        self.symbols = symbols or ['SPY', 'QQQ', 'IWM', 'DIA']
        self.config = config or GapFillConfig()
        self.data_dir = data_dir or Path(__file__).parent.parent / 'data' / 'historical'

        self.detector = GapDetector(
            min_gap_pct=self.config.min_gap_pct,
            max_gap_pct=self.config.max_gap_pct
        )

        # Load daily data for previous closes
        self.daily_data: Dict[str, pd.DataFrame] = {}
        self._load_daily_data()

    def _load_daily_data(self):
        """Load daily data for getting previous closes."""
        for symbol in self.symbols:
            daily_path = self.data_dir / 'daily' / f'{symbol}.parquet'
            if daily_path.exists():
                df = pd.read_parquet(daily_path)
                if 'timestamp' in df.columns:
                    df['date'] = pd.to_datetime(df['timestamp']).dt.date
                else:
                    df['date'] = pd.to_datetime(df.index).date
                self.daily_data[symbol] = df
                logger.info(f"Loaded daily data for {symbol}: {len(df)} days")

    def _get_previous_close(self, symbol: str, date: datetime.date) -> Optional[float]:
        """Get the previous day's close for a symbol."""
        if symbol not in self.daily_data:
            return None

        df = self.daily_data[symbol]
        # Find the most recent close before this date
        prev_days = df[df['date'] < date]
        if len(prev_days) == 0:
            return None

        return float(prev_days.iloc[-1]['close'])

    def _load_intraday_data(self, symbol: str, date_str: str) -> Optional[pd.DataFrame]:
        """Load 1-minute data for a specific date."""
        intraday_path = self.data_dir / 'intraday' / symbol / f'{date_str}.parquet'
        if not intraday_path.exists():
            return None

        df = pd.read_parquet(intraday_path)
        return df

    def _get_available_dates(self, symbol: str) -> List[str]:
        """Get list of available trading dates for a symbol."""
        intraday_dir = self.data_dir / 'intraday' / symbol
        if not intraday_dir.exists():
            return []

        dates = []
        for f in intraday_dir.glob('*.parquet'):
            dates.append(f.stem)

        return sorted(dates)

    def _simulate_day(
        self,
        symbol: str,
        date_str: str,
        df: pd.DataFrame,
        prev_close: float
    ) -> Optional[Trade]:
        """
        Simulate one trading day for gap-fill strategy.

        Returns Trade if a trade was executed, None otherwise.
        """
        # Get the opening bar (9:30 AM)
        try:
            # Handle timezone - data might be UTC
            if df.index.tz is not None:
                df_local = df.copy()
                df_local.index = df_local.index.tz_convert('America/New_York')
            else:
                df_local = df

            opening_bars = df_local.between_time('09:30', '09:31')
            if len(opening_bars) == 0:
                return None

            open_bar = opening_bars.iloc[0]
            open_price = float(open_bar['open'])

        except Exception as e:
            logger.debug(f"Error getting opening bar: {e}")
            return None

        # Detect gap
        gap = self.detector.detect_gap(
            symbol=symbol,
            previous_close=prev_close,
            open_price=open_price,
            timestamp=datetime.strptime(date_str, '%Y%m%d')
        )

        if gap is None:
            return None

        # Entry at first bar close (9:30)
        entry_price = float(open_bar['close'])
        entry_time = '09:30'

        # Determine direction
        if gap.gap_direction == 'down':
            # Gap down = go long (expecting bounce up)
            direction = 'long'
        else:
            # Gap up = go short (expecting pullback)
            direction = 'short'

        # Position size
        position_value = self.config.portfolio_value * self.config.max_position_pct
        shares = int(position_value / entry_price)

        if shares == 0:
            return None

        # Simulate through the day
        exit_price = None
        exit_time = None
        exit_reason = None
        hold_minutes = 0

        # Get bars from 9:31 to 11:30 (120 minute window)
        try:
            trading_bars = df_local.between_time('09:31', '11:30')
        except Exception:
            trading_bars = df.iloc[1:121]  # Fallback

        for idx, (ts, bar) in enumerate(trading_bars.iterrows()):
            current_price = float(bar['close'])
            hold_minutes = idx + 1

            # Check gap fill (75% threshold)
            fill_pct = gap.fill_percentage(current_price)

            if fill_pct >= self.config.fill_threshold:
                exit_price = current_price
                exit_time = str(ts.time())[:5] if hasattr(ts, 'time') else f'09:{31+idx}'
                exit_reason = f'gap_filled_{fill_pct*100:.0f}pct'
                break

            # Check stop loss
            if direction == 'long':
                pnl_pct = (current_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - current_price) / entry_price * 100

            if pnl_pct < -self.config.stop_loss_pct:
                exit_price = current_price
                exit_time = str(ts.time())[:5] if hasattr(ts, 'time') else f'09:{31+idx}'
                exit_reason = 'stop_loss'
                break

            # Check 120 minute time limit
            if hold_minutes >= 120:
                exit_price = current_price
                exit_time = '11:30'
                exit_reason = 'time_limit'
                break

        # If no exit yet, use last bar
        if exit_price is None:
            if len(trading_bars) > 0:
                exit_price = float(trading_bars.iloc[-1]['close'])
                exit_time = '11:30'
                exit_reason = 'time_limit'
                hold_minutes = len(trading_bars)
            else:
                return None

        # Calculate P&L
        if direction == 'long':
            pnl = (exit_price - entry_price) * shares
        else:
            pnl = (entry_price - exit_price) * shares

        pnl_pct = pnl / (entry_price * shares) * 100

        return Trade(
            symbol=symbol,
            date=date_str,
            gap_pct=gap.gap_pct,
            gap_direction=gap.gap_direction,
            entry_price=entry_price,
            entry_time=entry_time,
            exit_price=exit_price,
            exit_time=exit_time,
            exit_reason=exit_reason,
            shares=shares,
            pnl=pnl,
            pnl_pct=pnl_pct,
            hold_minutes=hold_minutes,
            gap_filled_pct=gap.fill_percentage(exit_price) * 100
        )

    def run(self) -> BacktestResult:
        """Run the backtest across all symbols and dates."""
        trades: List[Trade] = []
        gaps_detected = 0

        for symbol in self.symbols:
            dates = self._get_available_dates(symbol)
            logger.info(f"\n{symbol}: {len(dates)} trading days available")

            for date_str in dates:
                # Get previous close
                date = datetime.strptime(date_str, '%Y%m%d').date()
                prev_close = self._get_previous_close(symbol, date)

                if prev_close is None:
                    continue

                # Load intraday data
                df = self._load_intraday_data(symbol, date_str)
                if df is None or len(df) < 60:  # Need at least 60 minutes of data
                    continue

                # Simulate the day
                trade = self._simulate_day(symbol, date_str, df, prev_close)

                if trade is not None:
                    trades.append(trade)
                    gaps_detected += 1

        # Calculate results
        result = BacktestResult(trades=trades)
        result.gaps_detected = gaps_detected
        result.total_trades = len(trades)
        result.gaps_traded = len(trades)

        if len(trades) > 0:
            result.winning_trades = sum(1 for t in trades if t.pnl > 0)
            result.losing_trades = sum(1 for t in trades if t.pnl <= 0)
            result.win_rate = result.winning_trades / len(trades) * 100
            result.total_pnl = sum(t.pnl for t in trades)
            result.avg_pnl = result.total_pnl / len(trades)
            result.avg_pnl_pct = sum(t.pnl_pct for t in trades) / len(trades)
            result.avg_hold_minutes = sum(t.hold_minutes for t in trades) / len(trades)

            # By exit reason
            for trade in trades:
                reason = trade.exit_reason.split('_')[0] if '_' in trade.exit_reason else trade.exit_reason
                result.exits_by_reason[reason] = result.exits_by_reason.get(reason, 0) + 1

            # By direction
            long_trades = [t for t in trades if t.gap_direction == 'down']
            short_trades = [t for t in trades if t.gap_direction == 'up']
            result.long_trades = len(long_trades)
            result.short_trades = len(short_trades)
            if long_trades:
                result.long_win_rate = sum(1 for t in long_trades if t.pnl > 0) / len(long_trades) * 100
            if short_trades:
                result.short_win_rate = sum(1 for t in short_trades if t.pnl > 0) / len(short_trades) * 100

        return result

    def print_results(self, result: BacktestResult):
        """Print formatted backtest results."""
        print("\n" + "="*60)
        print("GAP-FILL INTRADAY BACKTEST RESULTS")
        print("="*60)

        print(f"\nSymbols: {', '.join(self.symbols)}")
        print(f"Gap Range: {self.config.min_gap_pct}% - {self.config.max_gap_pct}%")
        print(f"Fill Threshold: {self.config.fill_threshold*100}%")
        print(f"Max Hold: 120 minutes")
        print(f"Stop Loss: {self.config.stop_loss_pct}%")

        print("\n" + "-"*40)
        print("SUMMARY")
        print("-"*40)
        print(f"Total Trades: {result.total_trades}")
        print(f"Winning Trades: {result.winning_trades}")
        print(f"Losing Trades: {result.losing_trades}")
        print(f"Win Rate: {result.win_rate:.1f}%")
        print(f"Total P&L: ${result.total_pnl:,.2f}")
        print(f"Avg P&L per Trade: ${result.avg_pnl:,.2f}")
        print(f"Avg P&L %: {result.avg_pnl_pct:.2f}%")
        print(f"Avg Hold Time: {result.avg_hold_minutes:.0f} minutes")

        print("\n" + "-"*40)
        print("BY EXIT REASON")
        print("-"*40)
        for reason, count in sorted(result.exits_by_reason.items()):
            print(f"  {reason}: {count} ({count/result.total_trades*100:.1f}%)")

        print("\n" + "-"*40)
        print("BY DIRECTION")
        print("-"*40)
        print(f"Long Trades (gap down): {result.long_trades} (Win Rate: {result.long_win_rate:.1f}%)")
        print(f"Short Trades (gap up): {result.short_trades} (Win Rate: {result.short_win_rate:.1f}%)")

        if result.trades:
            print("\n" + "-"*40)
            print("INDIVIDUAL TRADES")
            print("-"*40)
            for trade in result.trades:
                direction = "LONG" if trade.gap_direction == 'down' else "SHORT"
                result_icon = "+" if trade.pnl > 0 else ""
                print(
                    f"{trade.date} {trade.symbol} {direction} | "
                    f"Gap: {trade.gap_pct:+.2f}% | "
                    f"Entry: ${trade.entry_price:.2f} | "
                    f"Exit: ${trade.exit_price:.2f} ({trade.exit_reason}) | "
                    f"P&L: {result_icon}${trade.pnl:.2f} ({result_icon}{trade.pnl_pct:.2f}%) | "
                    f"Hold: {trade.hold_minutes}m"
                )

        print("\n" + "="*60)


def main():
    """Run the gap-fill backtest."""
    # Create config
    config = GapFillConfig(
        symbols=['SPY', 'QQQ', 'IWM', 'DIA'],
        min_gap_pct=0.15,
        max_gap_pct=0.60,
        fill_threshold=0.75,
        stop_loss_pct=2.0,
        max_position_pct=0.05,
        portfolio_value=97000.0
    )

    # Run backtest
    backtester = GapFillBacktester(config=config)
    result = backtester.run()
    backtester.print_results(result)

    return result


if __name__ == "__main__":
    main()
