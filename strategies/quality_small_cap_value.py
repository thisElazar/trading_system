"""
Quality Small-Cap Value Strategy
=================================
Tier 1 Core Strategy

Research basis: Fama-French, AQR Quality-Minus-Junk research
- Small-cap value premium: 16.38% annual returns vs 0.67% growth
- Quality screens eliminate "junky" small-caps (35% of Russell 2000 unprofitable)
- Expected Sharpe: 1.0-1.5 with proper quality filters

Key insight: The small-cap premium re-emerges when controlling for quality.
Raw small-cap strategies disappoint because small-caps are inherently
lower profitability, higher risk, and stagnant growth. Quality screens fix this.

Implementation:
- Universe: Russell 2000 / small-cap stocks ($300M-$2B market cap)
- Quality filters: ROA > 0, profit margin > 0, debt/equity < 1.0
- Value signal: Book-to-market ratio (top quartile)
- Liquidity filter: ADV > $5M, spread < 3%
- Position sizing: Equal-weight to avoid concentration
- Rebalance: Monthly

Transaction cost budget: 2-4% annually (small-cap spreads 1-3%)

FUNDAMENTALS DATA REQUIREMENT:
==============================
This strategy uses REAL fundamental data from yfinance for accurate quality metrics:
- ROA (Return on Assets) = Net Income / Total Assets
- Profit Margin = Net Income / Revenue
- Debt-to-Equity = Total Debt / Shareholder Equity
- Market Cap for small-cap filtering ($300M-$2B)

To download fundamentals data, run:
    python scripts/download_fundamentals.py

If fundamentals file is not found, the strategy falls back to price-based proxies
which are LESS ACCURATE. The fallback proxies use momentum and volatility as
rough approximations - they should only be used for testing, not live trading.

Fundamentals file location: data/fundamentals/fundamentals.parquet
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import logging

import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.base import BaseStrategy, Signal, SignalType, LongOnlyStrategy
from config import VIX_REGIMES

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Quality metrics for a stock."""
    symbol: str
    roa: float  # Return on Assets
    profit_margin: float  # Net profit margin
    debt_to_equity: float  # Leverage ratio
    book_to_market: float  # Value signal
    market_cap: float  # Size in millions
    avg_daily_volume: float  # Liquidity in dollars
    avg_share_volume: float = 0.0  # Average daily share volume (v2.0)
    passes_quality: bool = False
    passes_value: bool = False
    passes_liquidity: bool = False
    composite_score: float = 0.0


class QualitySmallCapValueStrategy(LongOnlyStrategy):
    """
    Quality Small-Cap Value Strategy

    Combines three proven factors:
    1. Size: Small-cap stocks ($300M-$2B market cap)
    2. Value: High book-to-market ratio
    3. Quality: Profitable, low leverage, positive margins

    Research basis: Fama-French, AQR Quality-Minus-Junk
    - Small-cap value premium: 16.38% annual returns vs 0.67% growth
    - Quality screens eliminate "junky" small-caps (35% of Russell 2000 unprofitable)
    - Expected Sharpe: 1.0-1.5 with proper quality filters

    Key insight: The small-cap premium re-emerges when controlling for quality.
    Raw small-cap strategies disappoint because small-caps are inherently
    lower profitability, higher risk, and stagnant growth. Quality screens fix this.

    Risk controls:
    - Max 4% in any single position (small-cap concentration risk)
    - 50% exposure reduction if VIX > 30
    - Mandatory liquidity screens (ADV > $3M)
    - Monthly rebalancing to control turnover

    Data Requirements:
    - Requires fundamentals data file: data/fundamentals/fundamentals.parquet
    - Run 'python scripts/download_fundamentals.py' to download
    - Falls back to price-based proxies if unavailable (less accurate)
    """

    # ==========================================================================
    # UNIVERSE CONSTRAINTS (Size Factor)
    # ==========================================================================
    # Small-cap: $300M-$2B market cap
    # Avoid micro-caps (<$300M) due to liquidity/manipulation risks
    # Avoid mid-caps (>$2B) as they don't have small-cap premium
    MIN_MARKET_CAP = 500   # $500M minimum (avoid micro-caps, was $300M)
    MAX_MARKET_CAP = 3000  # $3B maximum (include lower mid-caps, was $2B)

    # ==========================================================================
    # QUALITY THRESHOLDS (AQR Quality-Minus-Junk Research)
    # ==========================================================================
    # These thresholds filter out the "junk" that plagues small-cap indices
    # Research: 35% of Russell 2000 is unprofitable - we exclude them all
    # NOTE: D/E threshold relaxed from 0.6 to 3.0 - data shows most small-caps
    # have D/E > 2.0, and strict threshold was eliminating all candidates
    MIN_ROA = 0.03            # 3% min ROA (was 2% - v2.0 stricter)
    MIN_PROFIT_MARGIN = 0.03  # 3% min margin (was 2% - v2.0 stricter)
    MAX_DEBT_TO_EQUITY = 3.0  # 3.0x max D/E (was 0.6 - relaxed to allow candidates)

    # ==========================================================================
    # VALUE THRESHOLD (Fama-French HML Factor)
    # ==========================================================================
    # Top quartile by book-to-market = classic value screen
    # More aggressive percentile increases value tilt but reduces diversification
    VALUE_PERCENTILE = 0.25  # Top quartile by book-to-market

    # ==========================================================================
    # LIQUIDITY CONSTRAINTS (Transaction Cost Management)
    # ==========================================================================
    # Small-caps have wider spreads - budget 2-4% annually for transaction costs
    # ADV > $3M ensures we can enter/exit without excessive slippage
    MIN_ADV_DOLLARS = 10_000_000  # $10M min daily volume (was $3M - v2.0 slippage fix)
    MIN_SHARE_VOLUME = 500_000    # 500K shares min daily volume (v2.0 NEW)
    MAX_SPREAD_PCT = 0.015        # 1.5% max spread (was 2% - v2.0 stricter)
    MAX_PARTICIPATION_RATE = 0.05 # Don't trade more than 5% of ADV (v2.0 NEW)

    # ==========================================================================
    # POSITION SIZING (Diversification & Risk Management)
    # ==========================================================================
    # More positions = better diversification in small-cap space
    # Smaller position sizes = less concentration risk
    MAX_POSITIONS = 25           # 25 positions (was 40 - v2.0 fewer for better execution)
    MAX_SINGLE_POSITION = 0.05   # 5% max per stock (was 4% - v2.0)

    # ==========================================================================
    # VIX ADJUSTMENTS (Regime-Based Risk Management)
    # ==========================================================================
    # Small-caps suffer more in risk-off environments
    # Reduce exposure aggressively when VIX spikes
    # Threshold from central config (VIX_REGIMES)
    HIGH_VIX_THRESHOLD = VIX_REGIMES['normal']  # 25 - normal/high boundary
    HIGH_VIX_REDUCTION = 0.50    # 50% exposure if VIX > normal threshold
    
    # Fundamentals file path
    FUNDAMENTALS_PATH = Path(__file__).parent.parent / "data" / "fundamentals" / "fundamentals.parquet"

    def __init__(self):
        super().__init__("quality_smallcap_value")

        # Track rebalancing
        self.last_rebalance_month = None

        # Cache for fundamental data loaded from parquet file
        self._fundamentals_df: Optional[pd.DataFrame] = None
        self._using_real_fundamentals: bool = False

        # Load fundamentals on initialization
        self._load_fundamentals()

        # Current holdings for position tracking
        self._current_holdings: List[str] = []

    def _load_fundamentals(self) -> None:
        """
        Load fundamentals data from parquet file.

        Expected columns: symbol, market_cap, roa, profit_margin, debt_to_equity, roe

        If file not found, logs warning and strategy will fall back to price proxies.
        """
        if self.FUNDAMENTALS_PATH.exists():
            try:
                self._fundamentals_df = pd.read_parquet(self.FUNDAMENTALS_PATH)

                # Validate required columns
                required_cols = {'symbol', 'market_cap', 'roa', 'profit_margin', 'debt_to_equity'}
                available_cols = set(self._fundamentals_df.columns)
                missing_cols = required_cols - available_cols

                if missing_cols:
                    logger.warning(
                        f"Fundamentals file missing columns: {missing_cols}. "
                        f"Available: {available_cols}. Falling back to price proxies."
                    )
                    self._fundamentals_df = None
                    self._using_real_fundamentals = False
                else:
                    # Index by symbol for fast lookup
                    self._fundamentals_df = self._fundamentals_df.set_index('symbol')
                    self._using_real_fundamentals = True
                    logger.info(
                        f"Loaded fundamentals for {len(self._fundamentals_df)} symbols "
                        f"from {self.FUNDAMENTALS_PATH}"
                    )
            except Exception as e:
                logger.warning(f"Error loading fundamentals file: {e}. Falling back to price proxies.")
                self._fundamentals_df = None
                self._using_real_fundamentals = False
        else:
            logger.warning(
                f"Fundamentals file not found at {self.FUNDAMENTALS_PATH}. "
                f"Run 'python scripts/download_fundamentals.py' to download. "
                f"Falling back to LESS ACCURATE price-based proxies."
            )
            self._fundamentals_df = None
            self._using_real_fundamentals = False

    def _get_fundamentals_for_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Get fundamentals data for a specific symbol.

        Returns dict with keys: market_cap, roa, profit_margin, debt_to_equity, roe
        Returns None if symbol not found in fundamentals data.
        """
        if self._fundamentals_df is None:
            return None

        if symbol not in self._fundamentals_df.index:
            return None

        row = self._fundamentals_df.loc[symbol]
        return {
            'market_cap': row.get('market_cap'),
            'roa': row.get('roa'),
            'profit_margin': row.get('profit_margin'),
            'debt_to_equity': row.get('debt_to_equity'),
            'roe': row.get('roe'),
            'book_to_market': row.get('book_to_market'),  # Optional
        }
    
    def _is_rebalance_day(self, current_date: datetime) -> bool:
        """Check if today is a rebalance day (first trading day of month)."""
        current_month = (current_date.year, current_date.month)
        
        if self.last_rebalance_month is None:
            return True
        
        return current_month != self.last_rebalance_month
    
    def _calculate_quality_metrics(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        fundamental_data: Optional[Dict] = None
    ) -> Optional[QualityMetrics]:
        """
        Calculate quality metrics for a stock using REAL fundamental data.

        Primary data source: fundamentals.parquet file with real metrics from yfinance:
        - ROA (returnOnAssets): Net Income / Total Assets
        - Profit Margin (profitMargins): Net Income / Revenue
        - Debt-to-Equity (debtToEquity): Total Debt / Shareholder Equity
        - Market Cap: For small-cap filtering ($300M-$2B)

        Fallback (if fundamentals unavailable): Price-based proxies (LESS ACCURATE)
        - ROA proxy: 12-month momentum
        - Margin proxy: Sharpe ratio
        - Leverage proxy: Downside volatility ratio
        WARNING: These proxies are rough approximations for testing only!
        """
        if len(price_data) < 252:  # Need 1 year of data
            return None

        try:
            # Get price series
            if 'close' in price_data.columns:
                close = price_data['close']
            elif 'Close' in price_data.columns:
                close = price_data['Close']
            else:
                return None

            if 'volume' in price_data.columns:
                volume = price_data['volume']
            elif 'Volume' in price_data.columns:
                volume = price_data['Volume']
            else:
                volume = pd.Series([0] * len(price_data))

            # Calculate returns for momentum
            returns = close.pct_change().dropna()

            if len(returns) < 252:
                return None

            # Calculate momentum (always needed for composite score)
            momentum_12m = (close.iloc[-1] / close.iloc[-252]) - 1 if close.iloc[-252] > 0 else 0

            # Liquidity (always from price data)
            avg_dollar_volume = (close * volume).iloc[-20:].mean()
            adv = avg_dollar_volume
            
            # v2.0: Share volume for participation rate limiting
            avg_share_volume = volume.iloc[-20:].mean()

            # Value proxy (price-to-52-week-high inverse)
            high_52w = close.iloc[-252:].max()
            price_to_high = close.iloc[-1] / high_52w if high_52w > 0 else 1.0
            book_to_market_proxy = 1.0 - price_to_high

            # =========================================================
            # TRY TO GET REAL FUNDAMENTALS FIRST
            # =========================================================
            fundamentals = self._get_fundamentals_for_symbol(symbol)

            if fundamentals is not None:
                # Use REAL fundamental data from yfinance
                roa = fundamentals.get('roa')
                profit_margin = fundamentals.get('profit_margin')
                debt_to_equity = fundamentals.get('debt_to_equity')
                market_cap = fundamentals.get('market_cap')
                book_to_market = fundamentals.get('book_to_market', book_to_market_proxy)

                # Handle None/NaN values - skip stocks with missing fundamentals
                if roa is None or pd.isna(roa):
                    logger.debug(f"{symbol}: Missing ROA in fundamentals, skipping")
                    return None
                if profit_margin is None or pd.isna(profit_margin):
                    logger.debug(f"{symbol}: Missing profit_margin in fundamentals, skipping")
                    return None
                if debt_to_equity is None or pd.isna(debt_to_equity):
                    # Default to 0 if no debt (some companies have no debt)
                    debt_to_equity = 0.0
                if market_cap is None or pd.isna(market_cap):
                    logger.debug(f"{symbol}: Missing market_cap in fundamentals, skipping")
                    return None

                # Convert market cap from dollars to millions for comparison
                market_cap_millions = market_cap / 1_000_000

                # =========================================================
                # ENFORCE MARKET CAP FILTER ($300M - $2B)
                # =========================================================
                if market_cap_millions < self.MIN_MARKET_CAP:
                    logger.debug(
                        f"{symbol}: Market cap ${market_cap_millions:.0f}M below "
                        f"${self.MIN_MARKET_CAP}M minimum, filtering out"
                    )
                    return None
                if market_cap_millions > self.MAX_MARKET_CAP:
                    logger.debug(
                        f"{symbol}: Market cap ${market_cap_millions:.0f}M above "
                        f"${self.MAX_MARKET_CAP}M maximum, filtering out"
                    )
                    return None

                using_proxies = False
            else:
                # =========================================================
                # FALLBACK: Price-based proxies (LESS ACCURATE)
                # =========================================================
                if not hasattr(self, '_proxy_warning_logged'):
                    logger.warning(
                        f"Using INACCURATE price-based proxies for {symbol}. "
                        f"Run 'python scripts/download_fundamentals.py' for real data."
                    )
                    self._proxy_warning_logged = True

                # ROA Proxy: 12-month momentum (very rough approximation)
                roa = momentum_12m

                # Profit Margin Proxy: Sharpe-like ratio
                annual_return = returns.iloc[-252:].mean() * 252
                annual_vol = returns.iloc[-252:].std() * np.sqrt(252)
                profit_margin = annual_return / annual_vol if annual_vol > 0 else 0

                # Leverage Proxy: Downside volatility ratio
                downside_returns = returns[returns < 0]
                downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 20 else annual_vol
                debt_to_equity = downside_vol / annual_vol if annual_vol > 0 else 1.0

                # Market cap proxy (cannot enforce real filter without fundamentals)
                market_cap_millions = avg_dollar_volume * 100 / 1_000_000

                book_to_market = book_to_market_proxy
                using_proxies = True

            # Create metrics object
            metrics = QualityMetrics(
                symbol=symbol,
                roa=roa,
                profit_margin=profit_margin,
                debt_to_equity=debt_to_equity,
                book_to_market=book_to_market,
                market_cap=market_cap_millions,
                avg_daily_volume=adv,
                avg_share_volume=avg_share_volume  # v2.0
            )

            # Apply quality filters
            metrics.passes_quality = (
                roa > self.MIN_ROA and
                profit_margin > self.MIN_PROFIT_MARGIN and
                debt_to_equity < self.MAX_DEBT_TO_EQUITY
            )

            # v2.0: Enhanced liquidity check - both dollar volume AND share volume
            metrics.passes_liquidity = (
                adv > self.MIN_ADV_DOLLARS and
                avg_share_volume > self.MIN_SHARE_VOLUME
            )

            # Composite score for ranking (higher = better)
            # Research-optimized weights based on Fama-French and AQR findings:
            # - 45% value: Primary factor (book-to-market is the core signal)
            # - 35% quality: Critical for avoiding small-cap junk
            # - 20% momentum: Timing component (avoid value traps)
            quality_score = (roa + profit_margin) / 2
            metrics.composite_score = (
                0.45 * book_to_market +
                0.35 * quality_score +
                0.20 * momentum_12m
            )

            return metrics

        except Exception as e:
            logger.warning(f"Error calculating metrics for {symbol}: {e}")
            return None
    
    def _rank_and_select(
        self, 
        candidates: List[QualityMetrics],
        n_positions: int
    ) -> List[QualityMetrics]:
        """
        Rank candidates by composite score and select top N.
        
        Selection process:
        1. Filter for quality (ROA > 0, margins > 0, low leverage)
        2. Filter for liquidity (ADV > $5M)
        3. Rank by composite score (value + quality + momentum)
        4. Select top N (equal weight)
        """
        # Apply quality and liquidity filters
        qualified = [
            c for c in candidates 
            if c.passes_quality and c.passes_liquidity
        ]
        
        if not qualified:
            logger.warning("No stocks passed quality and liquidity filters")
            return []
        
        # Sort by composite score (descending)
        qualified.sort(key=lambda x: x.composite_score, reverse=True)
        
        # Apply value filter (top quartile by book-to-market)
        n_value = max(1, int(len(qualified) * self.VALUE_PERCENTILE))
        value_candidates = sorted(
            qualified, 
            key=lambda x: x.book_to_market, 
            reverse=True
        )[:n_value]
        
        # Re-sort by composite score
        value_candidates.sort(key=lambda x: x.composite_score, reverse=True)
        
        # Select top N
        selected = value_candidates[:n_positions]
        
        logger.info(
            f"Selected {len(selected)} stocks from {len(candidates)} candidates "
            f"({len(qualified)} passed filters)"
        )
        
        return selected
    
    def generate_signals(
        self, 
        data: Dict[str, pd.DataFrame],
        current_positions: List[str] = None,
        vix_regime: str = None
    ) -> List[Signal]:
        """
        Generate buy/sell signals based on quality small-cap value screen.
        
        Only generates signals on monthly rebalance days.
        """
        signals = []
        
        if not data:
            return signals
        
        # Get current date from data
        current_date = self._get_current_date(data)
        if current_date is None:
            return signals
        
        # Check if rebalance day
        if not self._is_rebalance_day(current_date):
            return signals
        
        logger.debug(f"Quality Small-Cap Value rebalance day: {current_date}")
        self.last_rebalance_month = (current_date.year, current_date.month)
        
        # VIX adjustment based on regime
        vix_multiplier = 1.0
        if vix_regime in ('high', 'extreme'):
            vix_multiplier = self.HIGH_VIX_REDUCTION
            logger.debug(f"High VIX regime ({vix_regime}), reducing exposure by {(1-vix_multiplier)*100:.0f}%")
        
        # Calculate metrics for all stocks
        candidates = []
        for symbol, df in data.items():
            metrics = self._calculate_quality_metrics(symbol, df)
            if metrics:
                candidates.append(metrics)
        
        if not candidates:
            logger.warning("No valid candidates for quality screening")
            return signals
        
        # Rank and select
        n_positions = int(self.MAX_POSITIONS * vix_multiplier)
        selected = self._rank_and_select(candidates, n_positions)
        
        if not selected:
            return signals
        
        # Calculate position sizes (equal weight)
        position_size = min(
            1.0 / len(selected),
            self.MAX_SINGLE_POSITION
        ) * vix_multiplier
        
        # Generate signals for selected stocks
        selected_symbols = {s.symbol for s in selected}
        current_holdings = set(current_positions) if current_positions else set()
        
        # Sell signals for stocks no longer selected
        for symbol in current_holdings - selected_symbols:
            if symbol in data:
                df = data[symbol]
                price = df['close'].iloc[-1] if 'close' in df.columns else df['Close'].iloc[-1]
                signals.append(Signal(
                    timestamp=current_date,
                    symbol=symbol,
                    strategy=self.name,
                    signal_type=SignalType.SELL,
                    strength=1.0,
                    price=price,
                    reason='no_longer_qualifies',
                    metadata={}
                ))
        
        # Buy signals for newly selected stocks
        for metrics in selected:
            if metrics.symbol not in current_holdings and metrics.symbol in data:
                df = data[metrics.symbol]
                price = df['close'].iloc[-1] if 'close' in df.columns else df['Close'].iloc[-1]
                signals.append(Signal(
                    timestamp=current_date,
                    symbol=metrics.symbol,
                    strategy=self.name,
                    signal_type=SignalType.BUY,
                    strength=position_size,
                    price=price,
                    reason='quality_value_select',
                    metadata={
                        'composite_score': metrics.composite_score,
                        'roa': metrics.roa,
                        'book_to_market': metrics.book_to_market,
                        'adv': metrics.avg_daily_volume
                    }
                ))
        
        logger.debug(f"Generated {len(signals)} signals ({len([s for s in signals if s.signal_type == SignalType.BUY])} buys)")
        return signals
    
    def _get_current_date(self, data: Dict[str, pd.DataFrame]) -> Optional[datetime]:
        """Extract current date from data."""
        for symbol, df in data.items():
            if len(df) > 0:
                if 'timestamp' in df.columns:
                    return pd.to_datetime(df['timestamp'].iloc[-1])
                elif isinstance(df.index, pd.DatetimeIndex):
                    return df.index[-1].to_pydatetime()
        return None
    
    def get_parameters(self) -> Dict:
        """Return current strategy parameters for optimization."""
        return {
            'min_roa': self.MIN_ROA,
            'min_profit_margin': self.MIN_PROFIT_MARGIN,
            'max_debt_to_equity': self.MAX_DEBT_TO_EQUITY,
            'value_percentile': self.VALUE_PERCENTILE,
            'max_positions': self.MAX_POSITIONS,
            'max_single_position': self.MAX_SINGLE_POSITION,
            'high_vix_threshold': self.HIGH_VIX_THRESHOLD,
            'high_vix_reduction': self.HIGH_VIX_REDUCTION,
        }
    
    def set_parameters(self, params: Dict) -> None:
        """Update strategy parameters (for genetic optimization)."""
        if 'min_roa' in params:
            self.MIN_ROA = params['min_roa']
        if 'min_profit_margin' in params:
            self.MIN_PROFIT_MARGIN = params['min_profit_margin']
        if 'max_debt_to_equity' in params:
            self.MAX_DEBT_TO_EQUITY = params['max_debt_to_equity']
        if 'value_percentile' in params:
            self.VALUE_PERCENTILE = params['value_percentile']
        if 'max_positions' in params:
            self.MAX_POSITIONS = int(params['max_positions'])
        if 'max_single_position' in params:
            self.MAX_SINGLE_POSITION = params['max_single_position']
        if 'high_vix_threshold' in params:
            self.HIGH_VIX_THRESHOLD = params['high_vix_threshold']
        if 'high_vix_reduction' in params:
            self.HIGH_VIX_REDUCTION = params['high_vix_reduction']


# Genetic algorithm parameter specs for optimization
# Ranges based on academic research and practical constraints
# NOTE: D/E range expanded to 1.0-5.0 to match actual small-cap data distributions
OPTIMIZATION_PARAMS = [
    {'name': 'min_roa', 'min_val': 0.0, 'max_val': 0.05, 'step': 0.01},           # 0-5% ROA threshold
    {'name': 'min_profit_margin', 'min_val': 0.0, 'max_val': 0.05, 'step': 0.01}, # 0-5% margin threshold
    {'name': 'max_debt_to_equity', 'min_val': 1.0, 'max_val': 5.0, 'step': 0.5},  # 1.0-5.0x D/E cap (expanded)
    {'name': 'value_percentile', 'min_val': 0.15, 'max_val': 0.35, 'step': 0.05}, # 15-35% value screen
    {'name': 'max_positions', 'min_val': 20, 'max_val': 50, 'step': 5},           # 20-50 positions
    {'name': 'max_single_position', 'min_val': 0.02, 'max_val': 0.05, 'step': 0.01}, # 2-5% max position
    {'name': 'high_vix_threshold', 'min_val': 20, 'max_val': 35, 'step': 5},      # VIX trigger
]
