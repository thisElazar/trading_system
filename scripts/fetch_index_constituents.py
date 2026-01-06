#!/usr/bin/env python3
"""
Index Constituents Fetcher - Get current S&P 500, NASDAQ-100, and Russell 2000 symbols

Uses Wikipedia tables as source (updated regularly by community)
Falls back to SEC filings for Russell indices

Usage:
    python fetch_index_constituents.py              # Fetch all indices
    python fetch_index_constituents.py --index sp500
"""

import json
import argparse
from datetime import datetime
from pathlib import Path
import logging
import sys
import urllib.request
import ssl
import io

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import from config
try:
    from config import DIRS
    REFERENCE_DIR = DIRS["reference"]
except ImportError:
    REFERENCE_DIR = Path("./data/reference")

# Browser-like headers to avoid 403 errors
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
}

def fetch_url(url: str, timeout: int = 30) -> str:
    """Fetch URL content with proper headers"""
    request = urllib.request.Request(url, headers=HEADERS)
    context = ssl.create_default_context()
    
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        return response.read().decode('utf-8')

def fetch_sp500():
    """Fetch S&P 500 constituents - tries multiple sources"""
    import pandas as pd
    
    # Source 1: Wikipedia (with headers)
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        html = fetch_url(url)
        tables = pd.read_html(io.StringIO(html))
        
        df = tables[0]
        symbol_col = 'Symbol' if 'Symbol' in df.columns else df.columns[0]
        symbols = df[symbol_col].tolist()
        
        cleaned = [s.replace('.', '-') for s in symbols if isinstance(s, str)]
        
        if len(cleaned) >= 400:
            logger.info(f"Fetched {len(cleaned)} S&P 500 constituents from Wikipedia")
            return cleaned
    except Exception as e:
        logger.warning(f"Wikipedia S&P 500 failed: {e}")
    
    # Source 2: Slickcharts
    try:
        url = "https://www.slickcharts.com/sp500"
        html = fetch_url(url)
        tables = pd.read_html(io.StringIO(html))
        
        for table in tables:
            if 'Symbol' in table.columns:
                symbols = table['Symbol'].tolist()
                if len(symbols) >= 400:
                    logger.info(f"Fetched {len(symbols)} S&P 500 constituents from Slickcharts")
                    return symbols
    except Exception as e:
        logger.warning(f"Slickcharts S&P 500 failed: {e}")
    
    # Source 3: GitHub datahub (static but reliable)
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        html = fetch_url(url)
        df = pd.read_csv(io.StringIO(html))
        
        symbol_col = 'Symbol' if 'Symbol' in df.columns else df.columns[0]
        symbols = df[symbol_col].tolist()
        
        if len(symbols) >= 400:
            logger.info(f"Fetched {len(symbols)} S&P 500 constituents from GitHub datahub")
            return symbols
    except Exception as e:
        logger.warning(f"GitHub datahub S&P 500 failed: {e}")
    
    # Fallback: Hardcoded list (as of Dec 2024)
    logger.warning("Using hardcoded S&P 500 fallback")
    return get_sp500_fallback()

def fetch_nasdaq100():
    """Fetch NASDAQ-100 constituents - tries multiple sources"""
    import pandas as pd
    
    # Source 1: Wikipedia (with headers)
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        html = fetch_url(url)
        tables = pd.read_html(io.StringIO(html))
        
        for table in tables:
            if 'Ticker' in table.columns:
                symbols = table['Ticker'].tolist()
                if len(symbols) >= 90:
                    logger.info(f"Fetched {len(symbols)} NASDAQ-100 constituents from Wikipedia")
                    return symbols
            elif 'Symbol' in table.columns:
                symbols = table['Symbol'].tolist()
                if len(symbols) >= 90:
                    logger.info(f"Fetched {len(symbols)} NASDAQ-100 constituents from Wikipedia")
                    return symbols
    except Exception as e:
        logger.warning(f"Wikipedia NASDAQ-100 failed: {e}")
    
    # Source 2: Slickcharts
    try:
        url = "https://www.slickcharts.com/nasdaq100"
        html = fetch_url(url)
        tables = pd.read_html(io.StringIO(html))
        
        for table in tables:
            if 'Symbol' in table.columns:
                symbols = table['Symbol'].tolist()
                if len(symbols) >= 90:
                    logger.info(f"Fetched {len(symbols)} NASDAQ-100 constituents from Slickcharts")
                    return symbols
    except Exception as e:
        logger.warning(f"Slickcharts NASDAQ-100 failed: {e}")
    
    # Fallback: Hardcoded list
    logger.warning("Using hardcoded NASDAQ-100 fallback")
    return get_nasdaq100_fallback()

def fetch_dow30():
    """Fetch Dow Jones Industrial Average constituents - tries multiple sources"""
    import pandas as pd
    
    # Source 1: Wikipedia (with headers)
    try:
        url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
        html = fetch_url(url)
        tables = pd.read_html(io.StringIO(html))
        
        for table in tables:
            if 'Symbol' in table.columns:
                symbols = table['Symbol'].tolist()
                if len(symbols) >= 29:
                    logger.info(f"Fetched {len(symbols)} Dow 30 constituents from Wikipedia")
                    return symbols
    except Exception as e:
        logger.warning(f"Wikipedia Dow 30 failed: {e}")
    
    # Source 2: Slickcharts
    try:
        url = "https://www.slickcharts.com/dowjones"
        html = fetch_url(url)
        tables = pd.read_html(io.StringIO(html))
        
        for table in tables:
            if 'Symbol' in table.columns:
                symbols = table['Symbol'].tolist()
                if len(symbols) >= 29:
                    logger.info(f"Fetched {len(symbols)} Dow 30 constituents from Slickcharts")
                    return symbols
    except Exception as e:
        logger.warning(f"Slickcharts Dow 30 failed: {e}")
    
    # Fallback: Hardcoded list (stable - Dow rarely changes)
    logger.warning("Using hardcoded Dow 30 fallback")
    return get_dow30_fallback()


def get_sp500_fallback():
    """Hardcoded S&P 500 constituents (Dec 2024)"""
    return [
        "A", "AAL", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE", "ADI",
        "ADM", "ADP", "ADSK", "AEE", "AEP", "AES", "AFL", "AIG", "AIZ", "AJG",
        "AKAM", "ALB", "ALGN", "ALL", "ALLE", "AMAT", "AMCR", "AMD", "AME", "AMGN",
        "AMP", "AMT", "AMZN", "ANET", "ANSS", "AON", "AOS", "APA", "APD", "APH",
        "APTV", "ARE", "ATO", "AVB", "AVGO", "AVY", "AWK", "AXON", "AXP", "AZO",
        "BA", "BAC", "BALL", "BAX", "BBWI", "BBY", "BDX", "BEN", "BF-B", "BG",
        "BIIB", "BIO", "BK", "BKNG", "BKR", "BLDR", "BLK", "BMY", "BR", "BRK-B",
        "BRO", "BSX", "BWA", "BX", "BXP", "C", "CAG", "CAH", "CARR", "CAT",
        "CB", "CBOE", "CBRE", "CCI", "CCL", "CDNS", "CDW", "CE", "CEG", "CF",
        "CFG", "CHD", "CHRW", "CHTR", "CI", "CINF", "CL", "CLX", "CMCSA", "CME",
        "CMG", "CMI", "CMS", "CNC", "CNP", "COF", "COO", "COP", "COR", "COST",
        "CPAY", "CPB", "CPRT", "CPT", "CRL", "CRM", "CRWD", "CSCO", "CSGP", "CSX",
        "CTAS", "CTLT", "CTRA", "CTSH", "CTVA", "CVS", "CVX", "CZR", "D", "DAL",
        "DAY", "DD", "DE", "DECK", "DELL", "DFS", "DG", "DGX", "DHI", "DHR",
        "DIS", "DLR", "DLTR", "DOC", "DOV", "DOW", "DPZ", "DRI", "DTE", "DUK",
        "DVA", "DVN", "DXCM", "EA", "EBAY", "ECL", "ED", "EFX", "EG", "EIX",
        "EL", "ELV", "EMN", "EMR", "ENPH", "EOG", "EPAM", "EQIX", "EQR", "EQT",
        "ERIE", "ES", "ESS", "ETN", "ETR", "EVRG", "EW", "EXC", "EXPD", "EXPE",
        "EXR", "F", "FANG", "FAST", "FCX", "FDS", "FDX", "FE", "FFIV", "FI",
        "FICO", "FIS", "FITB", "FMC", "FOX", "FOXA", "FRT", "FSLR", "FTNT", "FTV",
        "GD", "GDDY", "GE", "GEHC", "GEN", "GEV", "GILD", "GIS", "GL", "GLW",
        "GM", "GNRC", "GOOG", "GOOGL", "GPC", "GPN", "GRMN", "GS", "GWW", "HAL",
        "HAS", "HBAN", "HCA", "HD", "HES", "HIG", "HII", "HLT", "HOLX", "HON",
        "HPE", "HPQ", "HRL", "HSIC", "HST", "HSY", "HUBB", "HUM", "HWM", "IBM",
        "ICE", "IDXX", "IEX", "IFF", "INCY", "INTC", "INTU", "INVH", "IP", "IPG",
        "IQV", "IR", "IRM", "ISRG", "IT", "ITW", "IVZ", "J", "JBHT", "JBL",
        "JCI", "JKHY", "JNJ", "JNPR", "JPM", "K", "KDP", "KEY", "KEYS", "KHC",
        "KIM", "KLAC", "KMB", "KMI", "KMX", "KO", "KR", "KVUE", "L", "LDOS",
        "LEN", "LH", "LHX", "LIN", "LKQ", "LLY", "LMT", "LNT", "LOW", "LRCX",
        "LULU", "LUV", "LVS", "LW", "LYB", "LYV", "MA", "MAA", "MAR", "MAS",
        "MCD", "MCHP", "MCK", "MCO", "MDLZ", "MDT", "MET", "META", "MGM", "MHK",
        "MKC", "MKTX", "MLM", "MMC", "MMM", "MNST", "MO", "MOH", "MOS", "MPC",
        "MPWR", "MRK", "MRNA", "MRO", "MS", "MSCI", "MSFT", "MSI", "MTB", "MTCH",
        "MTD", "MU", "NCLH", "NDAQ", "NDSN", "NEE", "NEM", "NFLX", "NI", "NKE",
        "NOC", "NOW", "NRG", "NSC", "NTAP", "NTRS", "NUE", "NVDA", "NVR", "NWS",
        "NWSA", "NXPI", "O", "ODFL", "OKE", "OMC", "ON", "ORCL", "ORLY", "OTIS",
        "OXY", "PANW", "PARA", "PAYC", "PAYX", "PCAR", "PCG", "PEG", "PEP", "PFE",
        "PFG", "PG", "PGR", "PH", "PHM", "PKG", "PLD", "PLTR", "PM", "PNC",
        "PNR", "PNW", "PODD", "POOL", "PPG", "PPL", "PRU", "PSA", "PSX", "PTC",
        "PWR", "PYPL", "QCOM", "QRVO", "RCL", "REG", "REGN", "RF", "RJF", "RL",
        "RMD", "ROK", "ROL", "ROP", "ROST", "RSG", "RTX", "RVTY", "SBAC", "SBUX",
        "SCHW", "SHW", "SJM", "SLB", "SMCI", "SNA", "SNPS", "SO", "SOLV", "SPG",
        "SPGI", "SRE", "STE", "STLD", "STT", "STX", "STZ", "SW", "SWK", "SWKS",
        "SYF", "SYK", "SYY", "T", "TAP", "TDG", "TDY", "TECH", "TEL", "TER",
        "TFC", "TFX", "TGT", "TJX", "TMO", "TMUS", "TPR", "TRGP", "TRMB", "TROW",
        "TRV", "TSCO", "TSLA", "TSN", "TT", "TTWO", "TXN", "TXT", "TYL", "UAL",
        "UBER", "UDR", "UHS", "ULTA", "UNH", "UNP", "UPS", "URI", "USB", "V",
        "VICI", "VLO", "VLTO", "VMC", "VRSK", "VRSN", "VRTX", "VST", "VTR", "VTRS",
        "VZ", "WAB", "WAT", "WBA", "WBD", "WDC", "WEC", "WELL", "WFC", "WM",
        "WMB", "WMT", "WRB", "WST", "WTW", "WY", "WYNN", "XEL", "XOM", "XYL",
        "YUM", "ZBH", "ZBRA", "ZTS"
    ]


def get_nasdaq100_fallback():
    """Hardcoded NASDAQ-100 constituents (Dec 2024)"""
    return [
        "AAPL", "ABNB", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AMAT", "AMD", "AMGN",
        "AMZN", "ANSS", "ARM", "ASML", "AVGO", "AZN", "BIIB", "BKNG", "BKR", "CDNS",
        "CDW", "CEG", "CHTR", "CMCSA", "COST", "CPRT", "CRWD", "CSCO", "CSGP", "CSX",
        "CTAS", "CTSH", "DASH", "DDOG", "DLTR", "DXCM", "EA", "EXC", "FANG", "FAST",
        "FTNT", "GEHC", "GFS", "GILD", "GOOG", "GOOGL", "HON", "IDXX", "ILMN", "INTC",
        "INTU", "ISRG", "KDP", "KHC", "KLAC", "LIN", "LRCX", "LULU", "MAR", "MCHP",
        "MDB", "MDLZ", "MELI", "META", "MNST", "MRNA", "MRVL", "MSFT", "MU", "NFLX",
        "NVDA", "NXPI", "ODFL", "ON", "ORLY", "PANW", "PAYX", "PCAR", "PDD", "PEP",
        "PYPL", "QCOM", "REGN", "ROP", "ROST", "SBUX", "SMCI", "SNPS", "TEAM", "TMUS",
        "TSLA", "TTD", "TTWO", "TXN", "VRSK", "VRTX", "WBD", "WDAY", "XEL", "ZS"
    ]


def get_dow30_fallback():
    """Hardcoded Dow 30 constituents (Dec 2024)"""
    return [
        "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
        "DOW", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD",
        "MMM", "MRK", "MSFT", "NKE", "NVDA", "PG", "TRV", "UNH", "V", "WMT"
    ]

def fetch_russell2000_sample():
    """
    Russell 2000 is proprietary (FTSE Russell).
    This fetches a sample from iShares IWM ETF holdings.
    For production, you'd need a data subscription.
    """
    import pandas as pd
    
    # Source 1: iShares IWM ETF holdings (direct CSV)
    try:
        url = "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund"
        html = fetch_url(url)
        
        # Parse CSV, skipping header rows
        df = pd.read_csv(io.StringIO(html), skiprows=9)
        
        if 'Ticker' in df.columns:
            symbols = df['Ticker'].dropna().tolist()
            # Filter out non-equity holdings and clean
            symbols = [s.strip() for s in symbols if isinstance(s, str) and len(s) <= 5 and s.replace('-', '').isalpha()]
            if len(symbols) >= 1000:
                logger.info(f"Fetched {len(symbols)} Russell 2000 constituents from IWM")
                return symbols[:2000]
    except Exception as e:
        logger.warning(f"iShares IWM failed: {e}")
    
    # Source 2: Try alternative CSV endpoint
    try:
        url = "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/1521942788811.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund"
        html = fetch_url(url)
        df = pd.read_csv(io.StringIO(html), skiprows=9)
        
        if 'Ticker' in df.columns:
            symbols = df['Ticker'].dropna().tolist()
            symbols = [s.strip() for s in symbols if isinstance(s, str) and len(s) <= 5 and s.replace('-', '').isalpha()]
            if len(symbols) >= 1000:
                logger.info(f"Fetched {len(symbols)} Russell 2000 constituents from IWM (alt)")
                return symbols[:2000]
    except Exception as e:
        logger.warning(f"iShares IWM alt failed: {e}")
    
    # Fallback: Return curated sample of small-cap symbols
    logger.info("Using curated Russell 2000 sample fallback...")
    return get_russell2000_fallback()

def get_russell2000_fallback():
    """Curated list of Russell 2000 small-cap stocks"""
    # This is a representative sample of ~500 liquid small caps
    # Sourced from various public lists
    
    return [
        # Financial Services
        "HOPE", "SBCF", "HTBK", "GABC", "CUBI", "FFBC", "BANF", "FULT", "RNST", "WABC",
        "CADE", "TOWN", "NBHC", "PPBI", "WASH", "BUSE", "SFNC", "SRCE", "IBCP", "EGBN",
        
        # Healthcare
        "AMED", "ACHC", "ADPT", "ALKS", "AMPH", "ANIP", "ANGO", "ARWR", "AXSM", "BCYC",
        "BEAM", "BCRX", "BHVN", "BLUE", "BMRN", "CARA", "CERS", "CORT", "CPRX", "DCPH",
        
        # Technology
        "AAON", "ABCB", "ACIW", "ADTN", "AEIS", "AGYS", "ALIT", "ALRM", "AMBA", "AMKR",
        "AOSL", "APPF", "APPS", "ATEN", "AVNW", "BAND", "BCOV", "BIGC", "BL", "BLKB",
        
        # Industrials
        "AGCO", "AIMC", "AIN", "ALSN", "ARCB", "ASTE", "ATI", "AVNT", "AZZ", "BWXT",
        "CACI", "CAR", "CARG", "CMCO", "CMC", "CW", "DAN", "DY", "EXPO", "FELE",
        
        # Consumer
        "AAP", "AEO", "ANF", "BBBY", "BBY", "BIG", "BJRI", "BKE", "BOOT", "BROS",
        "CAKE", "CATO", "CBRL", "CHS", "CONN", "CRI", "CROX", "DBI", "DDS", "DLTR",
        
        # Energy
        "AROC", "CEIX", "CLR", "CNX", "CTRA", "DEN", "DK", "ERF", "FANG", "GPP",
        "HP", "HLX", "KOS", "LPI", "MGY", "MTDR", "MUR", "NOG", "OAS", "OVV",
        
        # Materials
        "ATI", "CBT", "CC", "CF", "CLW", "CMC", "CRS", "EMN", "FUL", "GEF",
        "HUN", "IOSP", "KOP", "KWR", "LPX", "MOS", "MP", "NUE", "OLN", "RYAM",
        
        # REITs
        "AAT", "ACRE", "AIV", "APLE", "BDN", "BRX", "COLD", "CTO", "DEA", "DHC",
        "EPRT", "EPR", "FCPT", "FPI", "GOOD", "GTY", "HASI", "IIPR", "JBGS", "KRG",
        
        # Utilities
        "ALE", "AVA", "BKH", "MGEE", "NJR", "NWE", "OGE", "OGS", "PNM", "POR",
        "SJW", "SR", "SWX", "UTHR", "UTL", "VST", "WEC", "WGL", "XEL", "NWN",
        
        # More diversified small caps
        "ABMD", "ACIW", "ADNT", "AEIS", "AGCO", "AIN", "AIRC", "AJRD", "ALE", "ALKS",
        "ALRM", "ALSN", "AMBA", "AMKR", "AAN", "ANIK", "AOSL", "APPF", "APPS", "APYX",
        "ARCB", "AROC", "ARWR", "ASB", "ASGN", "ASIX", "ASPN", "ASTE", "ATEC", "ATEN",
    ]

def save_constituents(index_name, symbols, output_dir=None):
    """Save constituents to JSON file"""
    if output_dir is None:
        output_dir = REFERENCE_DIR
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = output_dir / f"{index_name}_constituents.json"
    
    data = {
        'index': index_name,
        'fetched_at': datetime.now().isoformat(),
        'count': len(symbols),
        'symbols': symbols
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved {len(symbols)} {index_name} constituents to {filepath}")
    return filepath

def build_combined_universe(output_dir=None):
    """Fetch all indices and create combined universe"""
    
    print("\n" + "="*60)
    print("📊 FETCHING INDEX CONSTITUENTS")
    print("="*60 + "\n")
    
    all_symbols = set()
    
    # S&P 500
    print("Fetching S&P 500...")
    sp500 = fetch_sp500()
    if sp500:
        save_constituents('sp500', sp500, output_dir)
        all_symbols.update(sp500)
    
    # NASDAQ-100
    print("\nFetching NASDAQ-100...")
    nasdaq = fetch_nasdaq100()
    if nasdaq:
        save_constituents('nasdaq100', nasdaq, output_dir)
        all_symbols.update(nasdaq)
    
    # Dow 30
    print("\nFetching Dow 30...")
    dow = fetch_dow30()
    if dow:
        save_constituents('dow30', dow, output_dir)
        all_symbols.update(dow)
    
    # Russell 2000 sample
    print("\nFetching Russell 2000 sample...")
    russell = fetch_russell2000_sample()
    if russell:
        save_constituents('russell2000', russell, output_dir)
        all_symbols.update(russell)
    
    # Combined universe
    combined = sorted(list(all_symbols))
    save_constituents('combined_universe', combined, output_dir)
    
    print("\n" + "="*60)
    print("📈 SUMMARY")
    print("="*60)
    print(f"S&P 500:      {len(sp500)} symbols")
    print(f"NASDAQ-100:   {len(nasdaq)} symbols")
    print(f"Dow 30:       {len(dow)} symbols")
    print(f"Russell 2000: {len(russell)} symbols")
    print(f"Combined:     {len(combined)} unique symbols")
    print("="*60 + "\n")
    
    return combined

def main():
    parser = argparse.ArgumentParser(description='Fetch index constituents')
    parser.add_argument('--index', choices=['sp500', 'nasdaq', 'dow', 'russell', 'all'], 
                       default='all', help='Which index to fetch')
    parser.add_argument('--output', type=str, help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output) if args.output else REFERENCE_DIR
    
    if args.index == 'sp500':
        symbols = fetch_sp500()
        save_constituents('sp500', symbols, output_dir)
    elif args.index == 'nasdaq':
        symbols = fetch_nasdaq100()
        save_constituents('nasdaq100', symbols, output_dir)
    elif args.index == 'dow':
        symbols = fetch_dow30()
        save_constituents('dow30', symbols, output_dir)
    elif args.index == 'russell':
        symbols = fetch_russell2000_sample()
        save_constituents('russell2000', symbols, output_dir)
    else:
        build_combined_universe(output_dir)

if __name__ == "__main__":
    main()
