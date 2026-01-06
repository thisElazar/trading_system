#!/usr/bin/env python3
"""
Universe Downloader - Fast batch download of market data for all major indices

Downloads S&P 500, NASDAQ-100, Dow 30, and Russell 2000 constituents
Uses batch API requests (50 symbols per request) for speed
Saves to parquet files for efficient storage and loading

Usage:
    python universe_downloader.py                    # Download daily bars, 2 years
    python universe_downloader.py --years 5          # Download 5 years
    python universe_downloader.py --intraday         # Download 1-min bars (liquid only)
    python universe_downloader.py --index sp500      # Only S&P 500
"""

import os
import sys
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Third-party imports
try:
    import pandas as pd
    import numpy as np
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetAssetsRequest
    from alpaca.trading.enums import AssetStatus
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install alpaca-py pandas numpy")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from config
try:
    from config import (
        ALPACA_API_KEY, ALPACA_SECRET_KEY, 
        DIRS, BATCH_SIZE as CONFIG_BATCH_SIZE, HISTORICAL_YEARS
    )
    DATA_DIR = DIRS["historical"]
    REFERENCE_DIR = DIRS["reference"]
    BATCH_SIZE = CONFIG_BATCH_SIZE
except ImportError:
    # Fallback if config not available
    ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
    ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
    DATA_DIR = Path("./data/historical")
    REFERENCE_DIR = Path("./data/reference")
    BATCH_SIZE = 50
    HISTORICAL_YEARS = 2
RATE_LIMIT_DELAY = 0.3  # Seconds between batches (stay under 200/min)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# INDEX CONSTITUENTS
# ============================================================================

# These are approximate/representative - in production you'd fetch from a data source
# Dow 30 is fully contained in S&P 500, so we don't list separately

SP500_SAMPLE = [
    # Top 100 by market cap (representative sample)
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "TSLA", "BRK.B", "UNH",
    "JNJ", "XOM", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV",
    "LLY", "PEP", "KO", "COST", "AVGO", "WMT", "MCD", "CSCO", "TMO", "ACN",
    "ABT", "DHR", "CRM", "NKE", "NEE", "LIN", "TXN", "PM", "WFC", "ORCL",
    "AMD", "UPS", "QCOM", "UNP", "MS", "RTX", "LOW", "IBM", "HON", "GE",
    "INTC", "CAT", "SPGI", "BA", "AMGN", "BMY", "SBUX", "ELV", "PLD", "DE",
    "INTU", "ISRG", "MDT", "BLK", "AXP", "GILD", "ADI", "MDLZ", "SYK", "VRTX",
    "CVS", "REGN", "C", "BKNG", "CI", "TMUS", "PGR", "CB", "ZTS", "SCHW",
    "SO", "MO", "TJX", "DUK", "EOG", "BDX", "MMC", "CME", "CL", "AON",
    "SLB", "NOC", "APD", "ICE", "PNC", "ITW", "EQIX", "FDX", "ETN", "NSC",
    # Next 100
    "EMR", "HUM", "FCX", "GD", "MCK", "WM", "PSX", "SHW", "MPC", "GM",
    "ADP", "USB", "PXD", "CCI", "F", "AZO", "TT", "OXY", "D", "MCO",
    "AIG", "CTVA", "HCA", "TRV", "MNST", "PCAR", "JCI", "AEP", "PSA", "TEL",
    "NEM", "DG", "CTAS", "SRE", "CMG", "ROP", "BK", "KMB", "WELL", "APH",
    "MCHP", "GIS", "PAYX", "FTNT", "AJG", "MSCI", "MSI", "ROST", "O", "SPG",
    "CMI", "BIIB", "STZ", "EXC", "A", "AFL", "IQV", "ALL", "HSY", "DXCM",
    "NDAQ", "LHX", "ADM", "HLT", "KR", "IDXX", "YUM", "ODFL", "PRU", "GWW",
    "DD", "DHI", "CTSH", "MTD", "WMB", "AMP", "VRSK", "FAST", "HAL", "DOW",
    "KHC", "EW", "CEG", "OKE", "DLTR", "AME", "CPRT", "ED", "PPG", "XEL",
    "NUE", "DVN", "KEYS", "CBRE", "FTV", "ON", "GLW", "EFX", "RMD", "VICI",
    # Next 100
    "CSGP", "WST", "ANSS", "CDW", "EBAY", "WTW", "DAL", "FANG", "ALGN", "MPWR",
    "EXR", "IT", "VMC", "IR", "HPQ", "MLM", "AWK", "FE", "BAX", "ILMN",
    "ROK", "MKC", "URI", "LYB", "CHD", "GRMN", "CAH", "STE", "TTWO", "DTE",
    "PTC", "AVB", "ESS", "WAB", "XYL", "INVH", "LUV", "DRI", "CLX", "K",
    "HIG", "HOLX", "SYY", "CNP", "TSN", "NTRS", "CF", "WAT", "MOS", "PPL",
    "LVS", "TROW", "ARE", "DPZ", "FLT", "EXPE", "IFF", "JBHT", "IP", "PEAK",
    "TDY", "WRB", "AES", "COO", "BALL", "POOL", "ZBRA", "BRO", "EQR", "SBAC",
    "MGM", "TYL", "TER", "MAA", "AKAM", "SWK", "TECH", "EPAM", "J", "WDC",
    "TRMB", "CINF", "BBY", "STT", "LNT", "PKI", "RF", "EVRG", "VTR", "FMC",
    "HBAN", "CMS", "NVR", "SEDG", "PNR", "WY", "DGX", "MTB", "ATO", "KMX",
    # Remaining ~100
    "CFG", "KEY", "EMN", "TXT", "NTAP", "IEX", "ALB", "AOS", "IPG", "L",
    "CHRW", "FOXA", "FOX", "SJM", "AAP", "BWA", "RHI", "CRL", "HSIC", "NDSN",
    "XRAY", "PNW", "NWS", "NWSA", "HII", "FFIV", "FRT", "CBOE", "AIZ", "MKTX",
    "BEN", "TPR", "HRL", "TAP", "WYNN", "CZR", "DISH", "LKQ", "ETSY", "PAYC",
    "GNRC", "LUMN", "AAL", "PHM", "JKHY", "REG", "UDR", "KIM", "CPT", "HST",
    "ALLE", "RCL", "NCLH", "CCL", "UAL", "ALK", "MHK", "HAS", "PARA", "VNO",
]

NASDAQ100 = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "TSLA", "AVGO", "COST",
    "ASML", "PEP", "CSCO", "AZN", "NFLX", "AMD", "ADBE", "TMUS", "TXN", "INTC",
    "QCOM", "CMCSA", "AMGN", "INTU", "ISRG", "HON", "AMAT", "BKNG", "VRTX", "REGN",
    "ADP", "GILD", "SBUX", "ADI", "MDLZ", "MU", "LRCX", "PANW", "PYPL", "SNPS",
    "KLAC", "CDNS", "MAR", "CHTR", "MELI", "MNST", "CTAS", "CSX", "ORLY", "MRVL",
    "ABNB", "NXPI", "PCAR", "FTNT", "KDP", "WDAY", "DXCM", "KHC", "MCHP", "ADSK",
    "ROST", "PAYX", "LULU", "ODFL", "AEP", "CPRT", "CEG", "EXC", "IDXX", "VRSK",
    "ON", "EA", "CTSH", "BKR", "CSGP", "FAST", "GFS", "GEHC", "XEL", "FANG",
    "ANSS", "DDOG", "CDW", "BIIB", "EBAY", "ALGN", "TEAM", "ILMN", "ZS", "WBD",
    "ENPH", "TTD", "MRNA", "LCID", "SIRI", "JD", "RIVN", "PDD", "ZM", "CRWD",
]

# Russell 2000 - sample of 500 most liquid small caps
# In production, you'd get this from a data provider
RUSSELL2000_SAMPLE = [
    # This is a representative sample - the full list would be ~2000 symbols
    "ABCB", "ACIW", "ADNT", "AEIS", "AGCO", "AIN", "AIRC", "AJRD", "ALE", "ALKS",
    "ALRM", "ALSN", "AMBA", "AMKR", "AAN", "ANIK", "AOSL", "APPF", "APPS", "APYX",
    "ARCB", "AROC", "ARWR", "ASB", "ASGN", "ASIX", "ASPN", "ASTE", "ATEC", "ATEN",
    "ATI", "ATNI", "ATRO", "AVNT", "AVT", "AXNX", "AXS", "AZEK", "AZTA", "B",
    "BANC", "BANF", "BANR", "BBIO", "BCPC", "BCRX", "BDC", "BEAM", "BFAM", "BGS",
    "BHE", "BJRI", "BKE", "BL", "BLKB", "BNL", "BOH", "BOOT", "BOX", "BRC",
    "BRZE", "BSIG", "BTU", "CABO", "CAKE", "CALM", "CAMT", "CAR", "CARA", "CARG",
    "CARS", "CASA", "CATY", "CBRL", "CBT", "CBU", "CCB", "CCBG", "CCOI", "CCS",
    "CEIX", "CENT", "CENTA", "CENX", "CERS", "CEVA", "CG", "CGNT", "CHCO", "CHEF",
    "CHGG", "CHS", "CHUY", "CIEN", "CIVB", "CIR", "CIVI", "CKH", "CLBK", "CLF",
    "CLNE", "CLSK", "CLVR", "CMC", "CMBM", "CNMD", "CNNE", "CNO", "CNOB", "CNXC",
    "COHU", "COLL", "CONN", "COR", "CORZ", "COUR", "CPE", "CPF", "CPK", "CPRI",
    "CPRX", "CRC", "CRGY", "CRNC", "CRSR", "CRTO", "CRUS", "CRVL", "CSGS", "CSTL",
    "CSV", "CTBI", "CTOS", "CTS", "CUTR", "CVI", "CVLT", "CVR", "CW", "CWK",
    "CXDO", "CXW", "CYTK", "DAKT", "DAN", "DCO", "DCPH", "DDS", "DFH", "DGICA",
    "DIGI", "DIOD", "DK", "DLB", "DNOW", "DOC", "DOCN", "DORM", "DRH", "DRQ",
    "DV", "DXC", "DY", "EFSC", "EGP", "EGY", "ELAN", "ELF", "ENTA", "ENV",
    "ENVA", "EPRT", "EQBK", "ESE", "ESGR", "ESNT", "ESTE", "EVTC", "EXP", "EXPO",
    "EXTR", "FA", "FAF", "FARO", "FATE", "FBK", "FBNC", "FBRT", "FBP", "FCFS",
    "FCPT", "FDEF", "FDP", "FHB", "FHI", "FIBK", "FIVE", "FIVN", "FL", "FLIC",
    # ... continue with more symbols as needed
]

def get_all_tradeable_symbols():
    """Get all tradeable US equity symbols from Alpaca"""
    logger.info("Fetching all tradeable symbols from Alpaca...")
    
    try:
        trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
        assets = trading_client.get_all_assets(
            filter=GetAssetsRequest(status=AssetStatus.ACTIVE)
        )
        
        symbols = []
        for asset in assets:
            if (asset.tradable and 
                asset.asset_class == 'us_equity' and
                asset.symbol.isalpha() and  # Skip symbols with dots, numbers
                len(asset.symbol) <= 5):     # Skip long symbols (usually warrants)
                symbols.append(asset.symbol)
        
        logger.info(f"Found {len(symbols)} tradeable US equities")
        return symbols
    
    except Exception as e:
        logger.error(f"Failed to fetch assets: {e}")
        return []

def build_master_universe():
    """Combine all index constituents and deduplicate"""
    all_symbols = set()
    
    # Add index constituents
    all_symbols.update(SP500_SAMPLE)
    all_symbols.update(NASDAQ100)
    all_symbols.update(RUSSELL2000_SAMPLE)
    
    # Get all tradeable from Alpaca and intersect
    tradeable = set(get_all_tradeable_symbols())
    
    if tradeable:
        # Only keep symbols that are actually tradeable
        valid_symbols = all_symbols.intersection(tradeable)
        logger.info(f"Combined universe: {len(all_symbols)} symbols, {len(valid_symbols)} tradeable")
        return sorted(list(valid_symbols))
    else:
        # Fallback to hardcoded list
        logger.warning("Using hardcoded symbol list (couldn't verify tradeability)")
        return sorted(list(all_symbols))

# ============================================================================
# DATA DOWNLOADER
# ============================================================================

class UniverseDownloader:
    def __init__(self):
        self.data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        self.stats = {
            'total_symbols': 0,
            'successful': 0,
            'failed': 0,
            'bars_downloaded': 0,
            'start_time': None,
            'end_time': None
        }
    
    def download_daily_bars(self, symbols, years=2, output_dir=None):
        """
        Download daily bars for all symbols
        
        Args:
            symbols: List of stock symbols
            years: Number of years of history
            output_dir: Where to save parquet files
        """
        if output_dir is None:
            output_dir = DATA_DIR / "daily"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats['total_symbols'] = len(symbols)
        self.stats['start_time'] = datetime.now()
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        logger.info(f"Downloading {len(symbols)} symbols, {years} years of daily data")
        logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
        logger.info(f"Output directory: {output_dir}")
        
        # Process in batches
        batches = [symbols[i:i+BATCH_SIZE] for i in range(0, len(symbols), BATCH_SIZE)]
        
        for batch_num, batch in enumerate(batches):
            batch_start = time.time()
            
            try:
                request = StockBarsRequest(
                    symbol_or_symbols=batch,
                    timeframe=TimeFrame.Day,
                    start=start_date,
                    end=end_date
                )
                
                bars = self.data_client.get_stock_bars(request)
                
                if bars.df.empty:
                    logger.warning(f"Batch {batch_num+1}: No data returned")
                    self.stats['failed'] += len(batch)
                    continue
                
                # Split by symbol and save
                df = bars.df.reset_index()
                
                for symbol in batch:
                    symbol_df = df[df['symbol'] == symbol].copy()
                    
                    if symbol_df.empty:
                        self.stats['failed'] += 1
                        continue
                    
                    # Save to parquet
                    output_path = output_dir / f"{symbol}.parquet"
                    symbol_df.to_parquet(output_path, index=False)
                    
                    self.stats['successful'] += 1
                    self.stats['bars_downloaded'] += len(symbol_df)
                
                elapsed = time.time() - batch_start
                progress = (batch_num + 1) / len(batches) * 100
                logger.info(f"Batch {batch_num+1}/{len(batches)} ({progress:.1f}%) - {len(batch)} symbols in {elapsed:.1f}s")
                
            except Exception as e:
                logger.error(f"Batch {batch_num+1} failed: {e}")
                self.stats['failed'] += len(batch)
            
            # Rate limiting
            time.sleep(RATE_LIMIT_DELAY)
        
        self.stats['end_time'] = datetime.now()
        self._print_summary()
    
    def download_intraday_bars(self, symbols, days=504, output_dir=None):
        """
        Download 1-minute bars for liquid symbols
        
        Args:
            symbols: List of stock symbols (should be liquid ones)
            days: Number of trading days (~2 years = 504)
            output_dir: Where to save parquet files
        """
        if output_dir is None:
            output_dir = DATA_DIR / "intraday"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats['total_symbols'] = len(symbols)
        self.stats['start_time'] = datetime.now()
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(days * 1.5))  # Account for weekends
        
        logger.info(f"Downloading {len(symbols)} symbols, ~{days} days of 1-min data")
        logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
        
        # Intraday is much more data - use smaller batches
        intraday_batch_size = 10
        batches = [symbols[i:i+intraday_batch_size] for i in range(0, len(symbols), intraday_batch_size)]
        
        for batch_num, batch in enumerate(batches):
            batch_start = time.time()
            
            try:
                request = StockBarsRequest(
                    symbol_or_symbols=batch,
                    timeframe=TimeFrame.Minute,
                    start=start_date,
                    end=end_date
                )
                
                bars = self.data_client.get_stock_bars(request)
                
                if bars.df.empty:
                    logger.warning(f"Batch {batch_num+1}: No data returned")
                    self.stats['failed'] += len(batch)
                    continue
                
                df = bars.df.reset_index()
                
                for symbol in batch:
                    symbol_df = df[df['symbol'] == symbol].copy()
                    
                    if symbol_df.empty:
                        self.stats['failed'] += 1
                        continue
                    
                    output_path = output_dir / f"{symbol}_1min.parquet"
                    symbol_df.to_parquet(output_path, index=False)
                    
                    self.stats['successful'] += 1
                    self.stats['bars_downloaded'] += len(symbol_df)
                
                elapsed = time.time() - batch_start
                progress = (batch_num + 1) / len(batches) * 100
                logger.info(f"Batch {batch_num+1}/{len(batches)} ({progress:.1f}%) - {len(batch)} symbols in {elapsed:.1f}s")
                
            except Exception as e:
                logger.error(f"Batch {batch_num+1} failed: {e}")
                self.stats['failed'] += len(batch)
            
            # Longer delay for intraday to avoid rate limits
            time.sleep(RATE_LIMIT_DELAY * 2)
        
        self.stats['end_time'] = datetime.now()
        self._print_summary()
    
    def _print_summary(self):
        """Print download summary"""
        duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        print("\n" + "="*60)
        print("📊 DOWNLOAD SUMMARY")
        print("="*60)
        print(f"Total symbols:     {self.stats['total_symbols']}")
        print(f"Successful:        {self.stats['successful']}")
        print(f"Failed:            {self.stats['failed']}")
        print(f"Bars downloaded:   {self.stats['bars_downloaded']:,}")
        print(f"Duration:          {duration:.1f} seconds")
        print(f"Rate:              {self.stats['successful']/duration:.1f} symbols/sec")
        print("="*60)

def save_universe_reference(symbols, filepath=None):
    """Save the symbol list for reference"""
    if filepath is None:
        filepath = REFERENCE_DIR / "universe.json"
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    
    universe_data = {
        'generated_at': datetime.now().isoformat(),
        'count': len(symbols),
        'symbols': symbols
    }
    
    with open(filepath, 'w') as f:
        json.dump(universe_data, f, indent=2)
    
    logger.info(f"Saved universe reference to {filepath}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Download market data for trading universe')
    parser.add_argument('--years', type=int, default=2, help='Years of daily data to download')
    parser.add_argument('--intraday', action='store_true', help='Download 1-minute bars (liquid symbols only)')
    parser.add_argument('--intraday-days', type=int, default=504, help='Days of intraday data')
    parser.add_argument('--index', choices=['sp500', 'nasdaq', 'russell', 'all'], default='all',
                       help='Which index to download')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--liquid-only', type=int, default=0, 
                       help='Only download top N most liquid symbols for intraday')
    
    args = parser.parse_args()
    
    # Build symbol list based on index selection
    if args.index == 'sp500':
        symbols = sorted(list(set(SP500_SAMPLE)))
    elif args.index == 'nasdaq':
        symbols = sorted(list(set(NASDAQ100)))
    elif args.index == 'russell':
        symbols = sorted(list(set(RUSSELL2000_SAMPLE)))
    else:
        symbols = build_master_universe()
    
    print(f"\n🎯 Universe: {len(symbols)} symbols")
    print(f"📁 Output: {args.output or DATA_DIR}")
    
    # Save reference file
    save_universe_reference(symbols)
    
    # Download
    downloader = UniverseDownloader()
    
    output_dir = Path(args.output) if args.output else None
    
    # Always download daily bars
    downloader.download_daily_bars(symbols, years=args.years, output_dir=output_dir)
    
    # Optionally download intraday
    if args.intraday:
        # For intraday, use only the most liquid symbols
        if args.liquid_only > 0:
            liquid_symbols = symbols[:args.liquid_only]
        else:
            # Default: top 200 (mega + large cap)
            liquid_symbols = list(set(SP500_SAMPLE[:100] + NASDAQ100[:50]))
        
        print(f"\n📈 Downloading intraday for {len(liquid_symbols)} liquid symbols")
        
        intraday_dir = (Path(args.output) / "intraday") if args.output else None
        downloader.download_intraday_bars(
            liquid_symbols, 
            days=args.intraday_days,
            output_dir=intraday_dir
        )
    
    print("\n✅ Download complete!")
    print(f"Daily data: {DATA_DIR / 'daily'}")
    if args.intraday:
        print(f"Intraday data: {DATA_DIR / 'intraday'}")

if __name__ == "__main__":
    main()
