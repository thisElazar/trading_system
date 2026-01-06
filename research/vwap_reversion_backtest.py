"""
VWAP Mean Reversion Backtest
============================
Backtest the VWAP mean reversion strategy using historical 1-minute data.

Tests the hypothesis: Price deviating 1-2% from VWAP reverts back.
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

from strategies.intraday.vwap_reversion.config import VWAPReversionConfig

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
    entry_vwap_deviation: float
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


class VWAPReversionBacktester:
    """Backtests the VWAP mean reversion strategy."""

    def __init__(
        self,
        symbols: List[str] = None,
        config: VWAPReversionConfig = None,
        data_dir: Path = None
    ):
        self.symbols = symbols or ['SPY', 'QQQ', 'IWM', 'DIA']
        self.config = config or VWAPReversionConfig()
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

    def _calculate_rsi(self, closes: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _simulate_day(self, symbol: str, date_str: str, df: pd.DataFrame) -> List[Trade]:
        """Simulate one trading day."""
        trades = []
        cfg = self.config

        # Handle timezone
        if df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_convert('America/New_York')

        # Calculate indicators
        df['vwap'] = self._calculate_vwap(df)
        df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap'] * 100
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        df['rel_volume'] = df['volume'] / df['volume'].rolling(20).mean()

        # Position tracking
        position = None
        entry_bar_idx = 0

        for idx in range(cfg.start_trading_minute, min(len(df), cfg.force_exit_minute)):
            bar = df.iloc[idx]
            price = float(bar['close'])
            vwap = float(bar['vwap'])
            vwap_dev = float(bar['vwap_deviation'])
            rsi = float(bar['rsi']) if not np.isnan(bar['rsi']) else 50.0
            rel_vol = float(bar['rel_volume']) if not np.isnan(bar['rel_volume']) else 1.0

            # Manage existing position
            if position is not None:
                hold_minutes = idx - entry_bar_idx

                # Check exits
                exit_reason = None

                # Stop loss
                if position['direction'] == 'LONG':
                    if price <= position['stop_loss']:
                        exit_reason = 'stop_loss'
                    elif price >= vwap:  # VWAP cross
                        exit_reason = 'vwap_cross'
                else:  # SHORT
                    if price >= position['stop_loss']:
                        exit_reason = 'stop_loss'
                    elif price <= vwap:
                        exit_reason = 'vwap_cross'

                # Time limit
                if hold_minutes >= cfg.max_hold_minutes:
                    exit_reason = 'time_limit'

                # Force exit near close
                if idx >= cfg.force_exit_minute - 1:
                    exit_reason = 'eod_exit'

                # Partial reversion target
                if exit_reason is None:
                    entry_dev = position['entry_deviation']
                    if entry_dev != 0:
                        reversion_pct = 1 - abs(vwap_dev / entry_dev)
                        if reversion_pct >= cfg.reversion_target_pct:
                            exit_reason = 'target'

                if exit_reason:
                    # Calculate P&L
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
                        entry_vwap_deviation=position['entry_deviation'],
                        exit_price=price,
                        exit_time=str(bar.name.time())[:5] if hasattr(bar.name, 'time') else f'{idx}',
                        exit_reason=exit_reason,
                        shares=position['shares'],
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        hold_minutes=hold_minutes
                    )
                    trades.append(trade)
                    position = None
                continue

            # Check for new entry
            if idx > cfg.stop_trading_minute:
                continue

            abs_dev = abs(vwap_dev)
            if abs_dev < cfg.min_vwap_deviation_pct or abs_dev > cfg.max_vwap_deviation_pct:
                continue

            if rel_vol < cfg.min_relative_volume:
                continue

            # Determine direction
            direction = None
            if vwap_dev < -cfg.min_vwap_deviation_pct and rsi < cfg.rsi_oversold:
                direction = 'LONG'
            elif not cfg.long_only and vwap_dev > cfg.min_vwap_deviation_pct and rsi > cfg.rsi_overbought:
                direction = 'SHORT'

            if direction is None:
                continue

            # Calculate position
            shares = int(cfg.max_position_value / price)
            if shares <= 0:
                continue

            # Set stop loss
            if direction == 'LONG':
                stop_loss = price * (1 - cfg.stop_loss_pct / 100)
            else:
                stop_loss = price * (1 + cfg.stop_loss_pct / 100)

            position = {
                'direction': direction,
                'entry_price': price,
                'entry_time': str(bar.name.time())[:5] if hasattr(bar.name, 'time') else f'{idx}',
                'entry_deviation': vwap_dev,
                'shares': shares,
                'stop_loss': stop_loss,
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
                entry_vwap_deviation=position['entry_deviation'],
                exit_price=price,
                exit_time='EOD',
                exit_reason='market_close',
                shares=position['shares'],
                pnl=pnl,
                pnl_pct=pnl_pct,
                hold_minutes=hold_minutes
            )
            trades.append(trade)

        return trades

    def run(self, long_only: bool = None) -> BacktestResult:
        """Run the backtest."""
        if long_only is not None:
            self.config.long_only = long_only

        all_trades: List[Trade] = []

        for symbol in self.symbols:
            dates = self._get_available_dates(symbol)
            logger.info(f"\n{symbol}: {len(dates)} trading days")

            for date_str in dates:
                df = self._load_intraday_data(symbol, date_str)
                if df is None or len(df) < 60:
                    continue

                trades = self._simulate_day(symbol, date_str, df)
                all_trades.extend(trades)

        # Calculate results
        result = BacktestResult(trades=all_trades)
        result.total_trades = len(all_trades)

        if len(all_trades) > 0:
            result.winning_trades = sum(1 for t in all_trades if t.pnl > 0)
            result.losing_trades = sum(1 for t in all_trades if t.pnl <= 0)
            result.win_rate = result.winning_trades / len(all_trades) * 100
            result.total_pnl = sum(t.pnl for t in all_trades)
            result.avg_pnl = result.total_pnl / len(all_trades)
            result.avg_pnl_pct = sum(t.pnl_pct for t in all_trades) / len(all_trades)
            result.avg_hold_minutes = sum(t.hold_minutes for t in all_trades) / len(all_trades)

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
        print("VWAP MEAN REVERSION BACKTEST RESULTS")
        print("="*60)

        print(f"\nSymbols: {', '.join(self.symbols)}")
        print(f"VWAP Deviation Range: {self.config.min_vwap_deviation_pct}% - {self.config.max_vwap_deviation_pct}%")
        print(f"Reversion Target: {self.config.reversion_target_pct*100}%")
        print(f"Stop Loss: {self.config.stop_loss_pct}%")
        print(f"Long Only: {self.config.long_only}")

        print("\n" + "-"*40)
        print("SUMMARY")
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
                    f"Dev: {trade.entry_vwap_deviation:+.2f}% | "
                    f"${trade.entry_price:.2f} -> ${trade.exit_price:.2f} ({trade.exit_reason}) | "
                    f"P&L: {icon}${trade.pnl:.2f} | Hold: {trade.hold_minutes}m"
                )

        print("\n" + "="*60)


def main():
    """Run VWAP reversion backtest."""
    config = VWAPReversionConfig(
        symbols=['SPY', 'QQQ', 'IWM', 'DIA'],
        min_vwap_deviation_pct=1.0,
        max_vwap_deviation_pct=2.5,
        reversion_target_pct=0.75,
        stop_loss_pct=1.5,
        max_hold_minutes=120,
        long_only=True,
        portfolio_value=97000.0,
        max_position_pct=0.05
    )

    backtester = VWAPReversionBacktester(config=config)
    result = backtester.run()
    backtester.print_results(result)

    return result


if __name__ == "__main__":
    main()
