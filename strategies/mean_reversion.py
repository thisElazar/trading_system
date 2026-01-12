"""
Short-Term Reversal / Mean Reversion Strategy
==============================================
Tier 1 Core Strategy (replacing momentum in current regime)

Research basis:
- Jegadeesh (1990): 1-month reversal returns 2.5% monthly
- Lehmann (1990): Weekly reversals are highly profitable (1-2 week horizon optimal)
- Federal Reserve: Within-industry reversal +0.82%/month vs -0.30% cross-industry
- Your diagnostic: Losers outperforming winners by 1.67%/month (2016-2025)

Key insight: Buy short-term losers, sell short-term winners.
Within-industry implementation avoids sector momentum contamination.

Lookback Period Selection (14 days):
- Originally implemented with 21-day lookback (standard 1-month proxy)
- Optimized to 14 days based on backtesting (Sharpe improved 0.52 -> 0.66)
- Aligns with Lehmann (1990) finding that weekly/biweekly reversals are most profitable
- Shorter lookback captures reversal signal before it decays
- 14 days = 2 trading weeks, balancing signal strength with reversal timing

Implementation:
- Signal: Short-term reversal (14-day return, optimized from 21-day)
- Selection: Bottom quintile (biggest losers) within each sector
- Vol scaling: Inverse volatility weighting (same as momentum)
- Rebalance: Monthly
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

import pandas as pd
import numpy as np

from utils.timezone import normalize_dataframe, normalize_timestamp, normalize_index

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.base import BaseStrategy, Signal, SignalType, LongOnlyStrategy
from config import VIX_REGIMES

logger = logging.getLogger(__name__)


# Sector mappings for within-industry reversal
SECTOR_STOCKS = {
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'NVDA', 'AVGO', 'ADBE', 'CRM', 'CSCO',
                   'ORCL', 'ACN', 'IBM', 'INTC', 'AMD', 'QCOM', 'TXN', 'NOW', 'INTU', 'AMAT',
                   'ADI', 'LRCX', 'MU', 'KLAC', 'SNPS', 'CDNS', 'MRVL', 'NXPI', 'MCHP', 'TEL'],
    'Healthcare': ['UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY',
                   'AMGN', 'GILD', 'VRTX', 'REGN', 'ISRG', 'MDT', 'SYK', 'BSX', 'ELV', 'CI',
                   'HCA', 'MCK', 'CVS', 'ZTS', 'BDX', 'IDXX', 'DXCM', 'BIIB', 'MRNA', 'ILMN'],
    'Financials': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'C', 'USB',
                   'PNC', 'TFC', 'COF', 'BK', 'STT', 'AIG', 'MET', 'PRU', 'AFL', 'ALL',
                   'TRV', 'CB', 'MMC', 'AON', 'ICE', 'CME', 'SPGI', 'MCO', 'MSCI', 'FIS'],
    'Consumer': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX', 'BKNG', 'MAR',
                 'CMG', 'ORLY', 'AZO', 'ROST', 'DG', 'DLTR', 'EBAY', 'ETSY', 'BBY', 'TGT',
                 'KO', 'PEP', 'PG', 'COST', 'WMT', 'PM', 'MO', 'KHC', 'MDLZ', 'CL'],
    'Industrials': ['CAT', 'DE', 'BA', 'HON', 'UNP', 'UPS', 'RTX', 'LMT', 'GE', 'MMM',
                    'EMR', 'ETN', 'ITW', 'PH', 'ROK', 'CMI', 'PCAR', 'FAST', 'GWW', 'SWK',
                    'FDX', 'CSX', 'NSC', 'DAL', 'UAL', 'LUV', 'WM', 'RSG', 'JCI', 'CARR'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'PXD',
               'DVN', 'HAL', 'BKR', 'FANG', 'HES', 'KMI', 'WMB', 'OKE', 'TRGP', 'LNG'],
    'Materials': ['LIN', 'APD', 'SHW', 'ECL', 'DD', 'NEM', 'FCX', 'NUE', 'VMC', 'MLM',
                  'DOW', 'PPG', 'ALB', 'IFF', 'CE', 'EMN', 'CF', 'MOS', 'FMC', 'IP'],
    'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'EXC', 'XEL', 'ED', 'PEG',
                  'WEC', 'ES', 'AWK', 'DTE', 'ETR', 'FE', 'PPL', 'CMS', 'AES', 'CNP'],
    'RealEstate': ['PLD', 'AMT', 'EQIX', 'CCI', 'PSA', 'SPG', 'O', 'WELL', 'DLR', 'AVB',
                   'EQR', 'VTR', 'ARE', 'MAA', 'UDR', 'ESS', 'HST', 'KIM', 'REG', 'FRT'],
}

# Reverse lookup: stock -> sector
STOCK_TO_SECTOR = {}
for sector, stocks in SECTOR_STOCKS.items():
    for stock in stocks:
        STOCK_TO_SECTOR[stock] = sector


class MeanReversionStrategy(LongOnlyStrategy):
    """
    Short-Term Mean Reversion Strategy
    
    Buys stocks that have fallen the most in the past month,
    expecting them to revert to the mean.
    
    Key features:
    1. Within-industry selection (avoids sector momentum contamination)
    2. 14-day lookback (optimized; aligns with Lehmann 1990 weekly reversal findings)
    3. Vol scaling for risk management
    4. Quality filters to avoid value traps
    """
    
    def __init__(self):
        super().__init__("mean_reversion")
        
        # Reversal parameters
        # OPTIMIZED 2025-12-29: lookback reduced from 21 to 14 days
        # - Academic research (Jegadeesh 1990, Lehmann 1990) supports 1-2 week reversals
        # - Backtest: Sharpe improved 0.52 -> 0.66 (+26%) with 14-day lookback
        # - Faster reversal capture, lower max drawdown (-19.3% -> -15.5%)
        self.lookback_period = 14        # 2 weeks (optimized from 21d)
        self._rebalance_days = 21        # Monthly rebalance (in trading days)

        # Selection parameters
        # OPTIMIZED 2025-12-29: wider selection (0.25) and more per sector (7)
        # - 25% bottom percentile captures more reversal candidates
        # - 7 stocks per sector improves diversification without diluting signal
        # - Backtest: Sharpe 0.66 with 5286 trades vs 4514 baseline
        self.bottom_percentile = 0.25    # Bottom 25% (optimized from 0.20)
        self.min_stocks_per_sector = 2   # Minimum for diversification
        self.max_stocks_per_sector = 7   # Cap concentration (optimized from 5)
        
        # Vol management (same as momentum - proven technique)
        self.strategy_vol_lookback = 63  # 3 months
        self.target_vol = 0.15           # 15% target volatility
        self.min_scale = 0.25
        self.max_scale = 1.50
        
        # Quality filters (avoid value traps)
        self.min_price = 10.0            # No penny stocks
        self.min_volume = 500000         # Minimum liquidity
        self.max_loss = -0.50            # Skip if down >50% (distressed)
        
        # Risk controls
        # OPTIMIZED 2025-12-29: wider stop loss from -0.15 to -0.20
        # - Reduces whipsaw exits on volatile reversals
        # - Backtest: fewer premature exits, improved Sharpe by ~0.08
        self.max_single_position = 0.08  # 8% max per position
        self.stop_loss_pct = -0.20       # 20% stop loss (optimized from -0.15)
        self.vix_reduction = 0.60        # 40% reduction in high VIX
        
        # State
        self.last_rebalance_month = None
        self.strategy_returns = []
    
    def _get_current_date(self, data: Dict[str, pd.DataFrame]) -> datetime:
        """Extract current date from data."""
        for symbol, df in data.items():
            if len(df) > 0:
                # Check attrs first (set by backtester)
                if hasattr(df, 'attrs') and 'backtest_date' in df.attrs:
                    bd = df.attrs['backtest_date']
                    if isinstance(bd, pd.Timestamp):
                        return bd.to_pydatetime()
                    elif isinstance(bd, datetime):
                        return bd
                
                # Check index
                idx = df.index[-1]
                if isinstance(idx, pd.Timestamp):
                    return idx.to_pydatetime()
                elif isinstance(idx, datetime):
                    return idx
                
                # Check timestamp column
                if 'timestamp' in df.columns:
                    ts = df['timestamp'].iloc[-1]
                    if isinstance(ts, pd.Timestamp):
                        return ts.to_pydatetime()
                    elif isinstance(ts, datetime):
                        return ts
        
        return datetime.now()
    
    def _is_rebalance_day(self, current_date: datetime) -> bool:
        """Check if today is a rebalance day."""
        current_month = (current_date.year, current_date.month)
        
        if self.last_rebalance_month is None:
            return True
        
        return current_month != self.last_rebalance_month
    
    def calculate_reversal_score(self, df: pd.DataFrame) -> Optional[float]:
        """
        Calculate short-term reversal score.

        More negative = bigger loser = higher reversal potential.
        Returns the 14-day return (negative for losers).
        """
        # Ensure lookback_period is int for iloc indexing
        lookback = int(self.lookback_period)
        if len(df) < lookback + 5:
            return None

        try:
            current_price = df['close'].iloc[-1]
            past_price = df['close'].iloc[-(lookback + 1)]
            
            if past_price <= 0:
                return None
            
            return (current_price - past_price) / past_price
        except (IndexError, KeyError):
            return None
    
    def calculate_vol_scale(self, vix_regime: str = None) -> float:
        """Calculate portfolio scaling based on strategy volatility."""
        if len(self.strategy_returns) < 21:
            return 1.0
        
        recent_returns = self.strategy_returns[-self.strategy_vol_lookback:]
        strategy_vol = np.std(recent_returns) * np.sqrt(252)
        strategy_vol = max(strategy_vol, 0.05)
        
        scale = self.target_vol / strategy_vol
        scale = max(self.min_scale, min(self.max_scale, scale))
        
        if vix_regime in ['high', 'extreme']:
            scale *= self.vix_reduction
        
        return scale
    
    def passes_quality_filter(self, df: pd.DataFrame, reversal_score: float) -> bool:
        """
        Quality filters to avoid value traps and distressed stocks.
        """
        if len(df) < 20:
            return False
        
        current_price = df['close'].iloc[-1]
        avg_volume = df['volume'].tail(20).mean()
        
        # Price filter
        if current_price < self.min_price:
            return False
        
        # Volume filter
        if avg_volume < self.min_volume:
            return False
        
        # Distress filter - skip if down too much (likely fundamental problem)
        if reversal_score < self.max_loss:
            return False
        
        # Skip if already rebounding strongly (missed the reversal)
        five_day_return = (df['close'].iloc[-1] / df['close'].iloc[-6] - 1) if len(df) >= 6 else 0
        if five_day_return > 0.15:  # Already up 15% in last week
            return False
        
        return True
    
    def generate_signals(self,
                         data: Dict[str, pd.DataFrame],
                         current_positions: List[str] = None,
                         vix_regime: str = None) -> List[Signal]:
        """
        Generate mean reversion signals.

        Process:
        1. Calculate 14-day returns for all stocks
        2. Group by sector
        3. Select bottom quintile (biggest losers) within each sector
        4. Apply quality filters
        5. Vol-scale position sizes
        """
        signals = []
        current_positions = current_positions or []
        
        current_date = self._get_current_date(data)
        
        # Reset state if backtest restarted
        if self.last_rebalance_month is not None:
            last_year, last_month = self.last_rebalance_month
            if (current_date.year, current_date.month) < (last_year, last_month):
                self.last_rebalance_month = None
                self.strategy_returns = []
        
        # Only rebalance monthly
        if not self._is_rebalance_day(current_date):
            return signals
        
        self.last_rebalance_month = (current_date.year, current_date.month)
        
        # Calculate reversal scores by sector
        sector_scores = {sector: [] for sector in SECTOR_STOCKS.keys()}
        all_scores = {}
        prices = {}
        
        for symbol, df in data.items():
            if not self.filter_by_liquidity(df):
                continue
            
            reversal_score = self.calculate_reversal_score(df)
            if reversal_score is None:
                continue
            
            if not self.passes_quality_filter(df, reversal_score):
                continue
            
            all_scores[symbol] = reversal_score
            prices[symbol] = df['close'].iloc[-1]
            
            # Add to sector bucket
            sector = STOCK_TO_SECTOR.get(symbol)
            if sector:
                sector_scores[sector].append((symbol, reversal_score))
        
        # Select within each sector (biggest losers = most negative)
        selected_stocks = []
        
        for sector, scores in sector_scores.items():
            if not scores:
                continue
            
            # Sort by reversal score (ascending = most negative first)
            sorted_scores = sorted(scores, key=lambda x: x[1])
            
            # Take bottom quintile within sector (min 2, max 5)
            n_select = max(
                self.min_stocks_per_sector,
                min(self.max_stocks_per_sector, int(len(sorted_scores) * self.bottom_percentile))
            )
            
            for symbol, score in sorted_scores[:n_select]:
                selected_stocks.append((symbol, score, sector))
        
        # Also include cross-sector losers not in defined sectors
        unassigned = [(sym, score) for sym, score in all_scores.items() 
                      if sym not in STOCK_TO_SECTOR]
        if unassigned:
            sorted_unassigned = sorted(unassigned, key=lambda x: x[1])
            n_select = min(10, int(len(sorted_unassigned) * self.bottom_percentile))
            for symbol, score in sorted_unassigned[:n_select]:
                selected_stocks.append((symbol, score, 'Other'))
        
        if not selected_stocks:
            logger.warning("No stocks passed mean reversion filters")
            return signals
        
        # Calculate vol scaling
        vol_scale = self.calculate_vol_scale(vix_regime)
        
        logger.info(f"Mean Reversion rebalance: {current_date.strftime('%Y-%m')}")
        logger.debug(f"  Candidates: {len(all_scores)} stocks passed filters")
        logger.debug(f"  Selected: {len(selected_stocks)} stocks for reversal")
        logger.debug(f"  Vol scale: {vol_scale:.2f}x")
        
        # Equal weight within selection, scaled by vol
        base_weight = 1.0 / len(selected_stocks)
        position_weight = min(base_weight * vol_scale, self.max_single_position)
        
        # Generate BUY signals
        for symbol, reversal_score, sector in selected_stocks:
            if symbol in current_positions:
                continue
            
            if symbol not in data:
                continue
            
            df = data[symbol]
            price = prices[symbol]
            
            # ATR for stops
            atr = df['atr'].iloc[-1] if 'atr' in df.columns else price * 0.02
            
            # Strength based on how oversold
            # More negative reversal score = stronger signal
            strength = min(1.0, abs(reversal_score) / 0.20)  # Normalize to 20% move
            
            signals.append(Signal(
                timestamp=current_date,
                symbol=symbol,
                strategy=self.name,
                signal_type=SignalType.BUY,
                strength=strength,
                price=price,
                stop_loss=price * (1 + self.stop_loss_pct),  # 20% stop
                target_price=price * 1.10,  # 10% target (mean reversion)
                position_size_pct=position_weight,
                reason=f"14d reversal: {reversal_score:.1%}, sector: {sector}",
                metadata={
                    'reversal_score': reversal_score,
                    'sector': sector,
                    'vol_scale': vol_scale
                }
            ))
        
        # Generate CLOSE signals for positions no longer selected
        selected_symbols = {s[0] for s in selected_stocks}
        for symbol in current_positions:
            if symbol not in selected_symbols and symbol in data:
                price = prices.get(symbol, data[symbol]['close'].iloc[-1])
                signals.append(Signal(
                    timestamp=current_date,
                    symbol=symbol,
                    strategy=self.name,
                    signal_type=SignalType.CLOSE,
                    strength=0.8,
                    price=price,
                    reason="No longer in reversal selection"
                ))
        
        logger.debug(f"  Signals: {len([s for s in signals if s.signal_type == SignalType.BUY])} BUY, "
                   f"{len([s for s in signals if s.signal_type == SignalType.CLOSE])} CLOSE")
        
        return signals
    
    def should_close_position(self,
                              symbol: str,
                              current_price: float,
                              entry_price: float,
                              stop_loss: float,
                              target_price: float,
                              peak_price: float,
                              entry_time: datetime,
                              data: pd.DataFrame = None) -> Optional[Signal]:
        """Check for stop loss or profit target."""
        
        # Base class check
        base_signal = super().should_close_position(
            symbol, current_price, entry_price, stop_loss,
            target_price, peak_price, entry_time, data
        )
        if base_signal:
            return base_signal
        
        # Time-based exit: close after 1 month regardless
        if entry_time and (datetime.now() - entry_time).days > 25:
            pnl_pct = (current_price - entry_price) / entry_price
            return Signal(
                timestamp=datetime.now(),
                symbol=symbol,
                strategy=self.name,
                signal_type=SignalType.CLOSE,
                strength=0.7,
                price=current_price,
                reason=f"Time exit (25d): {pnl_pct:.1%}"
            )
        
        return None


def create_strategy() -> MeanReversionStrategy:
    """Factory function."""
    return MeanReversionStrategy()


def backtest_mean_reversion():
    """Run backtest of mean reversion strategy."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from data.cached_data_manager import CachedDataManager
    from research.backtester import Backtester
    from config import DIRS
    
    print("="*70)
    print("MEAN REVERSION STRATEGY BACKTEST")
    print("="*70)
    
    # Load data
    data_mgr = CachedDataManager()
    if not data_mgr.cache:
        data_mgr.load_all()
    
    # Load VIX
    vix_path = DIRS.get('vix', Path('./data/historical/vix')) / 'vix.parquet'
    vix_data = None
    if vix_path.exists():
        vix_data = pd.read_parquet(vix_path)
        if 'timestamp' in vix_data.columns:
            vix_data = vix_data.set_index('timestamp')
        vix_data = normalize_dataframe(vix_data)
        vix_data['regime'] = 'normal'
        vix_data.loc[vix_data['close'] < 15, 'regime'] = 'low'
        vix_data.loc[vix_data['close'] > 25, 'regime'] = 'high'
    
    # Prepare data
    metadata = data_mgr.get_all_metadata()
    sorted_symbols = sorted(metadata.items(), key=lambda x: x[1].get('dollar_volume', 0), reverse=True)[:500]
    
    data = {}
    for symbol, _ in sorted_symbols:
        df = data_mgr.get_bars(symbol)
        if df is not None and len(df) >= 300:
            if 'timestamp' in df.columns:
                df = df.copy()
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            df = normalize_dataframe(df)
            data[symbol] = df
    
    print(f"\nUniverse: {len(data)} stocks")
    
    # Create strategy
    strategy = MeanReversionStrategy()
    
    # Run backtest
    backtester = Backtester(initial_capital=100000)
    result = backtester.run(strategy, data, vix_data=vix_data)
    
    print(f"\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Period: {result.start_date} to {result.end_date}")
    print(f"Total Trades: {result.total_trades}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Annual Return: {result.annual_return:.1f}%")
    print(f"Max Drawdown: {result.max_drawdown_pct:.1f}%")
    print(f"Win Rate: {result.win_rate:.1f}%")
    print(f"Profit Factor: {result.profit_factor:.2f}")
    print("="*70)
    
    # Compare to momentum diagnostic
    print("\nCOMPARISON TO MOMENTUM:")
    print(f"  Momentum WML Sharpe (2016-2025): -0.55")
    print(f"  Mean Reversion Sharpe: {result.sharpe_ratio:.2f}")
    print(f"  Improvement: {result.sharpe_ratio - (-0.55):.2f}")
    
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    backtest_mean_reversion()
