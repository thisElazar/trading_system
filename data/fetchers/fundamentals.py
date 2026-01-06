"""
Fundamentals Data Fetcher
=========================
Fetches and stores fundamental data using yfinance.
Stores data in Parquet format for efficient reading.

Data includes:
- Valuation metrics: P/E, P/B, P/S, EV/EBITDA
- Profitability: ROE, ROA, profit margin, operating margin
- Quality indicators: Debt/Equity, current ratio, interest coverage
- Growth metrics: Revenue growth, earnings growth
- Dividend data: Yield, payout ratio
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
import time

import pandas as pd

# Import from parent
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DIRS, FUNDAMENTALS_REFRESH_DAYS

logger = logging.getLogger(__name__)

# Rate limiting configuration
DEFAULT_RATE_LIMIT_DELAY = 0.25
BATCH_DELAY = 1.0
MAX_RETRIES = 3
RETRY_DELAY = 5.0


class FundamentalsFetcher:
    """Fetches and manages fundamental data from yfinance."""

    METRIC_MAPPING = {
        "pe_ratio": ["trailingPE", "forwardPE"],
        "pb_ratio": ["priceToBook"],
        "ps_ratio": ["priceToSalesTrailing12Months"],
        "ev_to_ebitda": ["enterpriseToEbitda"],
        "peg_ratio": ["pegRatio"],
        "enterprise_value": ["enterpriseValue"],
        "market_cap": ["marketCap"],
        "roe": ["returnOnEquity"],
        "roa": ["returnOnAssets"],
        "profit_margin": ["profitMargins"],
        "operating_margin": ["operatingMargins"],
        "gross_margin": ["grossMargins"],
        "ebitda_margin": ["ebitdaMargins"],
        "debt_to_equity": ["debtToEquity"],
        "current_ratio": ["currentRatio"],
        "quick_ratio": ["quickRatio"],
        "interest_coverage": ["interestCoverage"],
        "total_debt": ["totalDebt"],
        "total_cash": ["totalCash"],
        "free_cashflow": ["freeCashflow"],
        "revenue_growth": ["revenueGrowth"],
        "earnings_growth": ["earningsGrowth"],
        "earnings_quarterly_growth": ["earningsQuarterlyGrowth"],
        "revenue_per_share": ["revenuePerShare"],
        "dividend_yield": ["dividendYield", "trailingAnnualDividendYield"],
        "dividend_rate": ["dividendRate", "trailingAnnualDividendRate"],
        "payout_ratio": ["payoutRatio"],
        "ex_dividend_date": ["exDividendDate"],
        "beta": ["beta"],
        "52_week_high": ["fiftyTwoWeekHigh"],
        "52_week_low": ["fiftyTwoWeekLow"],
        "50_day_average": ["fiftyDayAverage"],
        "200_day_average": ["twoHundredDayAverage"],
        "shares_outstanding": ["sharesOutstanding"],
        "float_shares": ["floatShares"],
        "shares_short": ["sharesShort"],
        "short_ratio": ["shortRatio"],
        "short_percent_of_float": ["shortPercentOfFloat"],
        "target_high_price": ["targetHighPrice"],
        "target_low_price": ["targetLowPrice"],
        "target_mean_price": ["targetMeanPrice"],
        "recommendation_mean": ["recommendationMean"],
        "recommendation_key": ["recommendationKey"],
        "number_of_analyst_opinions": ["numberOfAnalystOpinions"],
        "sector": ["sector"],
        "industry": ["industry"],
        "full_time_employees": ["fullTimeEmployees"],
    }

    def __init__(self, cache_days: int = None):
        self.data_dir = DIRS["fundamentals"]
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_days = cache_days or FUNDAMENTALS_REFRESH_DAYS
        self._yf = None

    def _ensure_yfinance(self):
        if self._yf is None:
            import yfinance as yf
            self._yf = yf
        return self._yf

    def get_parquet_path(self, symbol: str) -> Path:
        return self.data_dir / f"{symbol}.parquet"

    def _is_cache_valid(self, symbol: str) -> bool:
        parquet_path = self.get_parquet_path(symbol)
        if not parquet_path.exists():
            return False
        mtime = datetime.fromtimestamp(parquet_path.stat().st_mtime)
        age_days = (datetime.now() - mtime).days
        return age_days < self.cache_days

    def _extract_metrics(self, info: Dict[str, Any]) -> Dict[str, Any]:
        metrics = {}
        for metric_name, possible_keys in self.METRIC_MAPPING.items():
            value = None
            for key in possible_keys:
                if key in info and info[key] is not None:
                    value = info[key]
                    break
            metrics[metric_name] = value
        return metrics

    def _fetch_from_yfinance(self, symbol: str) -> Optional[Dict[str, Any]]:
        yf = self._ensure_yfinance()
        for attempt in range(MAX_RETRIES):
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                if not info or len(info) <= 1:
                    logger.warning(f"{symbol}: No data returned from yfinance")
                    return None
                metrics = self._extract_metrics(info)
                metrics["symbol"] = symbol
                metrics["fetch_timestamp"] = datetime.now().isoformat()
                metrics["source"] = "yfinance"
                return metrics
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    logger.warning(f"{symbol}: Attempt {attempt + 1} failed: {e}")
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(f"{symbol}: Failed after {MAX_RETRIES} attempts: {e}")
                    return None
        return None

    def fetch_symbol(self, symbol: str, force: bool = False) -> Optional[pd.DataFrame]:
        parquet_path = self.get_parquet_path(symbol)
        if not force and self._is_cache_valid(symbol):
            logger.debug(f"{symbol}: Using cached fundamentals")
            return self.load_symbol(symbol)
        metrics = self._fetch_from_yfinance(symbol)
        if metrics is None:
            if parquet_path.exists():
                logger.warning(f"{symbol}: Using stale cache after fetch failure")
                return self.load_symbol(symbol)
            return None
        df = pd.DataFrame([metrics])
        df["symbol"] = symbol
        df = df.set_index("symbol")
        try:
            df.to_parquet(parquet_path)
            logger.info(f"{symbol}: Saved fundamentals to {parquet_path.name}")
        except Exception as e:
            logger.error(f"{symbol}: Failed to save fundamentals to parquet: {e}")
        return df

    def fetch_symbols(self, symbols: List[str], force: bool = False,
                      delay: float = DEFAULT_RATE_LIMIT_DELAY) -> Dict[str, pd.DataFrame]:
        results = {}
        to_fetch = []
        for symbol in symbols:
            if force or not self._is_cache_valid(symbol):
                to_fetch.append(symbol)
            else:
                df = self.load_symbol(symbol)
                if df is not None:
                    results[symbol] = df
        logger.info(f"Fundamentals: {len(results)} cached, {len(to_fetch)} to fetch")
        for i, symbol in enumerate(to_fetch):
            try:
                df = self.fetch_symbol(symbol, force=True)
                if df is not None:
                    results[symbol] = df
                if (i + 1) % 50 == 0:
                    logger.info(f"Progress: {i + 1}/{len(to_fetch)} symbols fetched")
                if delay > 0 and i < len(to_fetch) - 1:
                    time.sleep(delay)
            except Exception as e:
                logger.error(f"{symbol}: Error - {e}")
                continue
        logger.info(f"Fetched fundamentals for {len(results)}/{len(symbols)} symbols")
        return results

    def fetch_batch(self, symbols: List[str], force: bool = False,
                    batch_size: int = 50) -> Dict[str, pd.DataFrame]:
        results = {}
        total_batches = (len(symbols) + batch_size - 1) // batch_size
        for batch_num, i in enumerate(range(0, len(symbols), batch_size)):
            batch = symbols[i:i + batch_size]
            logger.info(f"Processing batch {batch_num + 1}/{total_batches}")
            batch_results = self.fetch_symbols(batch, force=force)
            results.update(batch_results)
            if batch_num < total_batches - 1:
                time.sleep(BATCH_DELAY)
        return results

    def load_symbol(self, symbol: str) -> Optional[pd.DataFrame]:
        parquet_path = self.get_parquet_path(symbol)
        if not parquet_path.exists():
            return None
        try:
            return pd.read_parquet(parquet_path)
        except Exception as e:
            logger.error(f"{symbol}: Failed to load - {e}")
            return None

    def load_symbols(self, symbols: List[str]) -> pd.DataFrame:
        dfs = []
        for symbol in symbols:
            df = self.load_symbol(symbol)
            if df is not None:
                dfs.append(df)
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs)

    def get_available_symbols(self) -> List[str]:
        return [p.stem for p in self.data_dir.glob("*.parquet")]

    def get_cache_age(self, symbol: str) -> Optional[int]:
        parquet_path = self.get_parquet_path(symbol)
        if not parquet_path.exists():
            return None
        mtime = datetime.fromtimestamp(parquet_path.stat().st_mtime)
        return (datetime.now() - mtime).days

    def get_stale_symbols(self) -> List[str]:
        stale = []
        for symbol in self.get_available_symbols():
            age = self.get_cache_age(symbol)
            if age is not None and age >= self.cache_days:
                stale.append(symbol)
        return stale

    def cleanup_old_data(self, days: int = None) -> int:
        if days is None:
            days = self.cache_days * 2
        cutoff = datetime.now() - timedelta(days=days)
        removed = 0
        for parquet_path in self.data_dir.glob("*.parquet"):
            try:
                mtime = datetime.fromtimestamp(parquet_path.stat().st_mtime)
                if mtime < cutoff:
                    parquet_path.unlink()
                    removed += 1
            except Exception as e:
                logger.warning(f"Failed to remove {parquet_path}: {e}")
        if removed > 0:
            logger.info(f"Removed {removed} old fundamentals files")
        return removed

    def get_fundamentals_summary(self, symbols: List[str] = None) -> pd.DataFrame:
        if symbols is None:
            symbols = self.get_available_symbols()
        key_metrics = [
            "pe_ratio", "pb_ratio", "ps_ratio", "ev_to_ebitda",
            "roe", "roa", "profit_margin", "operating_margin",
            "debt_to_equity", "current_ratio",
            "revenue_growth", "earnings_growth",
            "dividend_yield", "payout_ratio",
            "market_cap", "beta"
        ]
        df = self.load_symbols(symbols)
        if df.empty:
            return pd.DataFrame()
        available_metrics = [m for m in key_metrics if m in df.columns]
        return df[available_metrics]


def get_fundamentals(symbol: str, force: bool = False) -> Optional[pd.DataFrame]:
    fetcher = FundamentalsFetcher()
    return fetcher.fetch_symbol(symbol, force=force)


def get_fundamentals_bulk(symbols: List[str], force: bool = False) -> Dict[str, pd.DataFrame]:
    fetcher = FundamentalsFetcher()
    return fetcher.fetch_batch(symbols, force=force)


def load_fundamentals(symbol: str) -> Optional[pd.DataFrame]:
    fetcher = FundamentalsFetcher()
    return fetcher.load_symbol(symbol)


def load_fundamentals_bulk(symbols: List[str]) -> pd.DataFrame:
    fetcher = FundamentalsFetcher()
    return fetcher.load_symbols(symbols)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch fundamentals from yfinance")
    parser.add_argument("symbols", nargs="*", default=["AAPL", "MSFT", "GOOGL"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--stale", action="store_true")
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    fetcher = FundamentalsFetcher()
    if args.list:
        symbols = fetcher.get_available_symbols()
        print(f"Cached fundamentals ({len(symbols)} symbols):")
        for symbol in sorted(symbols):
            age = fetcher.get_cache_age(symbol)
            status = "stale" if age >= fetcher.cache_days else "valid"
            print(f"  {symbol}: {age} days old ({status})")
    elif args.stale:
        stale = fetcher.get_stale_symbols()
        print(f"Stale symbols ({len(stale)}):")
        for symbol in sorted(stale):
            print(f"  {symbol}")
    elif args.cleanup:
        removed = fetcher.cleanup_old_data()
        print(f"Removed {removed} old files")
    elif args.summary:
        df = fetcher.get_fundamentals_summary()
        if df.empty:
            print("No cached fundamentals found")
        else:
            print(f"Fundamentals Summary ({len(df)} symbols):")
            print(df.to_string())
    else:
        print(f"Fetching fundamentals for: {", ".join(args.symbols)}")
        results = get_fundamentals_bulk(args.symbols, force=args.force)
        print(f"Results ({len(results)} symbols):")
        for symbol, df in results.items():
            pe = df.get("pe_ratio", [None]).values[0]
            roe = df.get("roe", [None]).values[0]
            div_yield = df.get("dividend_yield", [None]).values[0]
            pe_str = f"{pe:.2f}" if pe else "N/A"
            roe_str = f"{roe*100:.1f}%" if roe else "N/A"
            div_str = f"{div_yield*100:.2f}%" if div_yield else "N/A"
            print(f"  {symbol}: P/E={pe_str}, ROE={roe_str}, Div={div_str}")
