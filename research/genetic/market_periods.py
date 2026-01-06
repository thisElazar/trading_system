"""
Market Period Library
=====================
Curated collection of specific market periods for targeted backtesting.

Enables:
- Rapid 30-second tests on specific conditions (COVID crash, 2008, etc.)
- Regime-specific strategy validation
- Current market matching to historical periods
- Multi-environment fitness evaluation

Usage:
    from research.genetic.market_periods import MarketPeriodLibrary, MarketPeriod

    library = MarketPeriodLibrary()

    # Get periods matching current conditions
    similar = library.find_similar_periods(current_regime_state)

    # Get specific period
    covid_crash = library.get_period("covid_crash")
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class PeriodType(Enum):
    """Categories of market periods."""
    CRISIS = "crisis"               # Sharp drawdowns, panic
    HIGH_VOL = "high_vol"           # Elevated volatility
    LOW_VOL = "low_vol"             # Compressed volatility
    BULL_RUN = "bull_run"           # Steady uptrend
    BEAR_MARKET = "bear_market"     # Extended downtrend
    RECOVERY = "recovery"           # Post-crash recovery
    SIDEWAYS = "sideways"           # Range-bound
    SECTOR_ROTATION = "rotation"    # Leadership shifts


@dataclass
class MarketPeriod:
    """Definition of a specific market period."""
    name: str
    start_date: date
    end_date: date
    period_type: PeriodType
    description: str

    # Characteristics for matching
    avg_vix: float                  # Average VIX during period
    vix_range: Tuple[float, float]  # (min, max) VIX
    avg_daily_range: float          # Average daily range %
    trend_direction: float          # -1 to 1 (bear to bull)
    correlation_regime: float       # 0-1 (diversified to correlated)
    sector_leadership: str          # "cyclical", "defensive", "mixed"

    # Performance expectations (for validation)
    spy_return: float               # SPY total return during period

    # Difficulty rating for strategies
    difficulty: float = 0.5         # 0-1, higher = harder

    # Tags for filtering
    tags: List[str] = field(default_factory=list)

    @property
    def duration_days(self) -> int:
        return (self.end_date - self.start_date).days

    @property
    def is_short_period(self) -> bool:
        """Suitable for rapid testing (< 60 trading days)."""
        return self.duration_days <= 90

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'period_type': self.period_type.value,
            'description': self.description,
            'avg_vix': self.avg_vix,
            'vix_range': self.vix_range,
            'trend_direction': self.trend_direction,
            'spy_return': self.spy_return,
            'duration_days': self.duration_days,
            'difficulty': self.difficulty,
            'tags': self.tags,
        }


# ============================================================================
# CURATED MARKET PERIODS
# ============================================================================

MARKET_PERIODS = {
    # =========================================================================
    # CRISIS PERIODS
    # =========================================================================
    "covid_crash": MarketPeriod(
        name="covid_crash",
        start_date=date(2020, 2, 19),
        end_date=date(2020, 3, 23),
        period_type=PeriodType.CRISIS,
        description="COVID-19 pandemic crash - fastest 30% decline in history",
        avg_vix=57.0,
        vix_range=(14.0, 82.7),
        avg_daily_range=4.5,
        trend_direction=-1.0,
        correlation_regime=0.95,
        sector_leadership="defensive",
        spy_return=-34.0,
        difficulty=0.95,
        tags=["pandemic", "liquidity_crisis", "correlation_spike", "vix_extreme"]
    ),

    "gfc_crash_2008": MarketPeriod(
        name="gfc_crash_2008",
        start_date=date(2008, 9, 15),
        end_date=date(2008, 11, 20),
        period_type=PeriodType.CRISIS,
        description="Global Financial Crisis - Lehman collapse phase",
        avg_vix=62.0,
        vix_range=(25.0, 80.9),
        avg_daily_range=4.2,
        trend_direction=-1.0,
        correlation_regime=0.92,
        sector_leadership="defensive",
        spy_return=-42.0,
        difficulty=0.98,
        tags=["financial_crisis", "deleveraging", "credit_freeze"]
    ),

    "gfc_bottom_2009": MarketPeriod(
        name="gfc_bottom_2009",
        start_date=date(2009, 1, 2),
        end_date=date(2009, 3, 9),
        period_type=PeriodType.CRISIS,
        description="GFC capitulation - final leg down to March 2009 bottom",
        avg_vix=45.0,
        vix_range=(38.0, 52.0),
        avg_daily_range=3.5,
        trend_direction=-0.8,
        correlation_regime=0.88,
        sector_leadership="defensive",
        spy_return=-25.0,
        difficulty=0.90,
        tags=["capitulation", "bottom_formation"]
    ),

    "flash_crash_2010": MarketPeriod(
        name="flash_crash_2010",
        start_date=date(2010, 5, 3),
        end_date=date(2010, 5, 25),
        period_type=PeriodType.CRISIS,
        description="Flash Crash - intraday 1000pt Dow drop and aftermath",
        avg_vix=32.0,
        vix_range=(22.0, 48.0),
        avg_daily_range=2.8,
        trend_direction=-0.6,
        correlation_regime=0.82,
        sector_leadership="mixed",
        spy_return=-8.0,
        difficulty=0.75,
        tags=["flash_crash", "liquidity_event", "quick_recovery"]
    ),

    "euro_crisis_2011": MarketPeriod(
        name="euro_crisis_2011",
        start_date=date(2011, 7, 22),
        end_date=date(2011, 10, 4),
        period_type=PeriodType.CRISIS,
        description="European debt crisis / US downgrade",
        avg_vix=35.0,
        vix_range=(17.0, 48.0),
        avg_daily_range=2.5,
        trend_direction=-0.7,
        correlation_regime=0.85,
        sector_leadership="defensive",
        spy_return=-19.0,
        difficulty=0.80,
        tags=["sovereign_debt", "contagion", "downgrade"]
    ),

    "china_deval_2015": MarketPeriod(
        name="china_deval_2015",
        start_date=date(2015, 8, 18),
        end_date=date(2015, 9, 29),
        period_type=PeriodType.CRISIS,
        description="China devaluation / global growth fears",
        avg_vix=28.0,
        vix_range=(14.0, 53.0),
        avg_daily_range=2.3,
        trend_direction=-0.5,
        correlation_regime=0.78,
        sector_leadership="defensive",
        spy_return=-12.0,
        difficulty=0.70,
        tags=["china", "currency", "em_contagion"]
    ),

    "volpocalypse_2018": MarketPeriod(
        name="volpocalypse_2018",
        start_date=date(2018, 1, 29),
        end_date=date(2018, 2, 9),
        period_type=PeriodType.CRISIS,
        description="VIX short squeeze / XIV collapse",
        avg_vix=30.0,
        vix_range=(11.0, 50.0),
        avg_daily_range=2.5,
        trend_direction=-0.6,
        correlation_regime=0.80,
        sector_leadership="mixed",
        spy_return=-10.0,
        difficulty=0.75,
        tags=["vol_squeeze", "systematic_unwind", "short_vol_blowup"]
    ),

    "q4_2018_selloff": MarketPeriod(
        name="q4_2018_selloff",
        start_date=date(2018, 10, 1),
        end_date=date(2018, 12, 24),
        period_type=PeriodType.CRISIS,
        description="Q4 2018 selloff - Fed tightening fears",
        avg_vix=22.0,
        vix_range=(12.0, 36.0),
        avg_daily_range=1.8,
        trend_direction=-0.7,
        correlation_regime=0.75,
        sector_leadership="defensive",
        spy_return=-19.8,
        difficulty=0.72,
        tags=["fed_tightening", "rate_fears", "christmas_eve"]
    ),

    "tariff_tantrum_2019": MarketPeriod(
        name="tariff_tantrum_2019",
        start_date=date(2019, 5, 1),
        end_date=date(2019, 6, 3),
        period_type=PeriodType.HIGH_VOL,
        description="Trade war escalation selloff",
        avg_vix=18.0,
        vix_range=(12.0, 23.0),
        avg_daily_range=1.3,
        trend_direction=-0.4,
        correlation_regime=0.65,
        sector_leadership="defensive",
        spy_return=-6.8,
        difficulty=0.55,
        tags=["trade_war", "tariffs", "geopolitical"]
    ),

    "svb_crisis_2023": MarketPeriod(
        name="svb_crisis_2023",
        start_date=date(2023, 3, 8),
        end_date=date(2023, 3, 24),
        period_type=PeriodType.CRISIS,
        description="Silicon Valley Bank collapse - regional bank crisis",
        avg_vix=24.0,
        vix_range=(18.0, 31.0),
        avg_daily_range=1.8,
        trend_direction=-0.3,
        correlation_regime=0.72,
        sector_leadership="defensive",
        spy_return=-3.0,
        difficulty=0.60,
        tags=["bank_crisis", "rates", "sector_specific"]
    ),

    # =========================================================================
    # RECOVERY PERIODS
    # =========================================================================
    "covid_recovery_2020": MarketPeriod(
        name="covid_recovery_2020",
        start_date=date(2020, 3, 23),
        end_date=date(2020, 6, 8),
        period_type=PeriodType.RECOVERY,
        description="Post-COVID crash V-shaped recovery",
        avg_vix=35.0,
        vix_range=(24.0, 66.0),
        avg_daily_range=2.5,
        trend_direction=0.9,
        correlation_regime=0.70,
        sector_leadership="cyclical",
        spy_return=44.5,
        difficulty=0.40,
        tags=["v_recovery", "fed_intervention", "growth_leadership"]
    ),

    "gfc_recovery_2009": MarketPeriod(
        name="gfc_recovery_2009",
        start_date=date(2009, 3, 9),
        end_date=date(2009, 6, 12),
        period_type=PeriodType.RECOVERY,
        description="Post-GFC recovery - March 2009 bottom",
        avg_vix=35.0,
        vix_range=(27.0, 50.0),
        avg_daily_range=2.2,
        trend_direction=0.85,
        correlation_regime=0.68,
        sector_leadership="cyclical",
        spy_return=40.0,
        difficulty=0.45,
        tags=["recovery", "qe", "bottom_fishing"]
    ),

    "q1_2019_recovery": MarketPeriod(
        name="q1_2019_recovery",
        start_date=date(2018, 12, 26),
        end_date=date(2019, 4, 30),
        period_type=PeriodType.RECOVERY,
        description="Q1 2019 recovery from Christmas Eve low",
        avg_vix=16.0,
        vix_range=(12.0, 25.0),
        avg_daily_range=1.0,
        trend_direction=0.8,
        correlation_regime=0.55,
        sector_leadership="cyclical",
        spy_return=25.0,
        difficulty=0.35,
        tags=["recovery", "fed_pivot", "momentum"]
    ),

    # =========================================================================
    # BULL RUNS
    # =========================================================================
    "2017_low_vol_bull": MarketPeriod(
        name="2017_low_vol_bull",
        start_date=date(2017, 1, 1),
        end_date=date(2017, 12, 31),
        period_type=PeriodType.BULL_RUN,
        description="2017 - historic low volatility bull market",
        avg_vix=11.0,
        vix_range=(9.0, 16.0),
        avg_daily_range=0.5,
        trend_direction=0.9,
        correlation_regime=0.45,
        sector_leadership="cyclical",
        spy_return=21.8,
        difficulty=0.25,
        tags=["low_vol", "goldilocks", "momentum"]
    ),

    "2013_taper_bull": MarketPeriod(
        name="2013_taper_bull",
        start_date=date(2013, 1, 1),
        end_date=date(2013, 12, 31),
        period_type=PeriodType.BULL_RUN,
        description="2013 bull run despite taper tantrum",
        avg_vix=14.0,
        vix_range=(11.0, 21.0),
        avg_daily_range=0.7,
        trend_direction=0.85,
        correlation_regime=0.50,
        sector_leadership="cyclical",
        spy_return=32.4,
        difficulty=0.30,
        tags=["qe", "momentum", "strong_trend"]
    ),

    "2021_meme_bull": MarketPeriod(
        name="2021_meme_bull",
        start_date=date(2021, 1, 1),
        end_date=date(2021, 11, 8),
        period_type=PeriodType.BULL_RUN,
        description="2021 meme stock / retail mania bull",
        avg_vix=18.0,
        vix_range=(15.0, 37.0),
        avg_daily_range=1.0,
        trend_direction=0.75,
        correlation_regime=0.55,
        sector_leadership="cyclical",
        spy_return=27.0,
        difficulty=0.35,
        tags=["retail_mania", "meme_stocks", "speculation"]
    ),

    "post_election_2016": MarketPeriod(
        name="post_election_2016",
        start_date=date(2016, 11, 9),
        end_date=date(2017, 3, 1),
        period_type=PeriodType.BULL_RUN,
        description="Trump election rally - rotation to value/cyclicals",
        avg_vix=12.0,
        vix_range=(10.0, 15.0),
        avg_daily_range=0.6,
        trend_direction=0.8,
        correlation_regime=0.48,
        sector_leadership="cyclical",
        spy_return=12.0,
        difficulty=0.30,
        tags=["election", "rotation", "reflation"]
    ),

    "2023_ai_bull": MarketPeriod(
        name="2023_ai_bull",
        start_date=date(2023, 1, 1),
        end_date=date(2023, 7, 31),
        period_type=PeriodType.BULL_RUN,
        description="2023 AI/tech-led rally - narrow breadth",
        avg_vix=16.0,
        vix_range=(12.0, 26.0),
        avg_daily_range=0.9,
        trend_direction=0.7,
        correlation_regime=0.52,
        sector_leadership="cyclical",
        spy_return=20.0,
        difficulty=0.40,
        tags=["ai_mania", "narrow_breadth", "mega_cap"]
    ),

    # =========================================================================
    # BEAR MARKETS
    # =========================================================================
    "2022_bear_market": MarketPeriod(
        name="2022_bear_market",
        start_date=date(2022, 1, 3),
        end_date=date(2022, 10, 12),
        period_type=PeriodType.BEAR_MARKET,
        description="2022 inflation / rate hike bear market",
        avg_vix=26.0,
        vix_range=(17.0, 37.0),
        avg_daily_range=1.5,
        trend_direction=-0.6,
        correlation_regime=0.72,
        sector_leadership="defensive",
        spy_return=-25.0,
        difficulty=0.75,
        tags=["inflation", "rate_hikes", "growth_to_value"]
    ),

    "2000_2002_tech_bust": MarketPeriod(
        name="2000_2002_tech_bust",
        start_date=date(2000, 3, 24),
        end_date=date(2002, 10, 9),
        period_type=PeriodType.BEAR_MARKET,
        description="Dot-com bust - extended bear market",
        avg_vix=28.0,
        vix_range=(16.0, 45.0),
        avg_daily_range=1.8,
        trend_direction=-0.5,
        correlation_regime=0.60,
        sector_leadership="defensive",
        spy_return=-49.0,
        difficulty=0.85,
        tags=["tech_bust", "valuation_reset", "extended_bear"]
    ),

    # =========================================================================
    # HIGH VOLATILITY PERIODS
    # =========================================================================
    "aug_2015_vix_spike": MarketPeriod(
        name="aug_2015_vix_spike",
        start_date=date(2015, 8, 17),
        end_date=date(2015, 9, 4),
        period_type=PeriodType.HIGH_VOL,
        description="August 2015 VIX spike - China fears",
        avg_vix=32.0,
        vix_range=(13.0, 53.0),
        avg_daily_range=2.8,
        trend_direction=-0.4,
        correlation_regime=0.80,
        sector_leadership="defensive",
        spy_return=-11.0,
        difficulty=0.70,
        tags=["vix_spike", "china", "quick_selloff"]
    ),

    "omicron_scare_2021": MarketPeriod(
        name="omicron_scare_2021",
        start_date=date(2021, 11, 24),
        end_date=date(2021, 12, 20),
        period_type=PeriodType.HIGH_VOL,
        description="Omicron variant scare",
        avg_vix=24.0,
        vix_range=(17.0, 31.0),
        avg_daily_range=1.5,
        trend_direction=-0.2,
        correlation_regime=0.68,
        sector_leadership="mixed",
        spy_return=-5.0,
        difficulty=0.50,
        tags=["covid_variant", "quick_recovery", "dip_buying"]
    ),

    # =========================================================================
    # LOW VOLATILITY PERIODS
    # =========================================================================
    "summer_2014_grind": MarketPeriod(
        name="summer_2014_grind",
        start_date=date(2014, 5, 1),
        end_date=date(2014, 9, 18),
        period_type=PeriodType.LOW_VOL,
        description="Summer 2014 low vol grind higher",
        avg_vix=12.0,
        vix_range=(10.0, 17.0),
        avg_daily_range=0.5,
        trend_direction=0.5,
        correlation_regime=0.42,
        sector_leadership="mixed",
        spy_return=8.0,
        difficulty=0.30,
        tags=["low_vol", "grind", "mean_reversion_friendly"]
    ),

    "2019_h2_calm": MarketPeriod(
        name="2019_h2_calm",
        start_date=date(2019, 6, 4),
        end_date=date(2019, 12, 31),
        period_type=PeriodType.LOW_VOL,
        description="H2 2019 calm rally",
        avg_vix=14.0,
        vix_range=(11.0, 21.0),
        avg_daily_range=0.7,
        trend_direction=0.6,
        correlation_regime=0.48,
        sector_leadership="mixed",
        spy_return=17.0,
        difficulty=0.30,
        tags=["low_vol", "fed_cuts", "momentum"]
    ),

    # =========================================================================
    # SIDEWAYS / CHOPPY
    # =========================================================================
    "2015_sideways": MarketPeriod(
        name="2015_sideways",
        start_date=date(2015, 1, 1),
        end_date=date(2015, 8, 17),
        period_type=PeriodType.SIDEWAYS,
        description="2015 sideways market before August selloff",
        avg_vix=14.0,
        vix_range=(11.0, 22.0),
        avg_daily_range=0.8,
        trend_direction=0.1,
        correlation_regime=0.52,
        sector_leadership="mixed",
        spy_return=1.0,
        difficulty=0.55,
        tags=["sideways", "choppy", "range_bound"]
    ),

    "2018_h1_chop": MarketPeriod(
        name="2018_h1_chop",
        start_date=date(2018, 2, 12),
        end_date=date(2018, 9, 30),
        period_type=PeriodType.SIDEWAYS,
        description="2018 H1 post-volpocalypse chop",
        avg_vix=15.0,
        vix_range=(9.0, 26.0),
        avg_daily_range=0.9,
        trend_direction=0.2,
        correlation_regime=0.55,
        sector_leadership="mixed",
        spy_return=10.0,
        difficulty=0.50,
        tags=["sideways", "choppy", "volatility_compression"]
    ),

    # =========================================================================
    # YEAR-SPECIFIC PERIODS (for annual testing)
    # =========================================================================
    "year_2014": MarketPeriod(
        name="year_2014",
        start_date=date(2014, 1, 2),
        end_date=date(2014, 12, 31),
        period_type=PeriodType.BULL_RUN,
        description="Full year 2014",
        avg_vix=14.0,
        vix_range=(10.0, 26.0),
        avg_daily_range=0.7,
        trend_direction=0.55,
        correlation_regime=0.50,
        sector_leadership="mixed",
        spy_return=13.7,
        difficulty=0.40,
        tags=["full_year", "qe_taper"]
    ),

    "year_2015": MarketPeriod(
        name="year_2015",
        start_date=date(2015, 1, 2),
        end_date=date(2015, 12, 31),
        period_type=PeriodType.SIDEWAYS,
        description="Full year 2015",
        avg_vix=16.5,
        vix_range=(10.0, 53.0),
        avg_daily_range=0.9,
        trend_direction=0.05,
        correlation_regime=0.55,
        sector_leadership="mixed",
        spy_return=1.4,
        difficulty=0.60,
        tags=["full_year", "china_deval", "flat"]
    ),

    "year_2016": MarketPeriod(
        name="year_2016",
        start_date=date(2016, 1, 4),
        end_date=date(2016, 12, 30),
        period_type=PeriodType.BULL_RUN,
        description="Full year 2016",
        avg_vix=15.5,
        vix_range=(11.0, 28.0),
        avg_daily_range=0.8,
        trend_direction=0.45,
        correlation_regime=0.52,
        sector_leadership="mixed",
        spy_return=12.0,
        difficulty=0.45,
        tags=["full_year", "election", "brexit"]
    ),

    "year_2017": MarketPeriod(
        name="year_2017",
        start_date=date(2017, 1, 3),
        end_date=date(2017, 12, 29),
        period_type=PeriodType.BULL_RUN,
        description="Full year 2017",
        avg_vix=11.0,
        vix_range=(9.0, 16.0),
        avg_daily_range=0.5,
        trend_direction=0.9,
        correlation_regime=0.45,
        sector_leadership="cyclical",
        spy_return=21.8,
        difficulty=0.25,
        tags=["full_year", "low_vol", "goldilocks"]
    ),

    "year_2018": MarketPeriod(
        name="year_2018",
        start_date=date(2018, 1, 2),
        end_date=date(2018, 12, 31),
        period_type=PeriodType.SIDEWAYS,
        description="Full year 2018",
        avg_vix=17.0,
        vix_range=(9.0, 50.0),
        avg_daily_range=1.0,
        trend_direction=-0.2,
        correlation_regime=0.58,
        sector_leadership="mixed",
        spy_return=-4.4,
        difficulty=0.65,
        tags=["full_year", "rate_hikes", "q4_selloff"]
    ),

    "year_2019": MarketPeriod(
        name="year_2019",
        start_date=date(2019, 1, 2),
        end_date=date(2019, 12, 31),
        period_type=PeriodType.BULL_RUN,
        description="Full year 2019",
        avg_vix=15.5,
        vix_range=(11.0, 25.0),
        avg_daily_range=0.8,
        trend_direction=0.8,
        correlation_regime=0.50,
        sector_leadership="cyclical",
        spy_return=31.5,
        difficulty=0.35,
        tags=["full_year", "fed_pivot", "trade_war"]
    ),

    "year_2020": MarketPeriod(
        name="year_2020",
        start_date=date(2020, 1, 2),
        end_date=date(2020, 12, 31),
        period_type=PeriodType.HIGH_VOL,
        description="Full year 2020",
        avg_vix=29.0,
        vix_range=(12.0, 82.7),
        avg_daily_range=1.8,
        trend_direction=0.4,
        correlation_regime=0.65,
        sector_leadership="mixed",
        spy_return=18.4,
        difficulty=0.70,
        tags=["full_year", "covid", "v_recovery"]
    ),

    "year_2021": MarketPeriod(
        name="year_2021",
        start_date=date(2021, 1, 4),
        end_date=date(2021, 12, 31),
        period_type=PeriodType.BULL_RUN,
        description="Full year 2021",
        avg_vix=19.5,
        vix_range=(15.0, 37.0),
        avg_daily_range=1.0,
        trend_direction=0.7,
        correlation_regime=0.55,
        sector_leadership="cyclical",
        spy_return=28.7,
        difficulty=0.40,
        tags=["full_year", "meme_stocks", "reopening"]
    ),

    "year_2022": MarketPeriod(
        name="year_2022",
        start_date=date(2022, 1, 3),
        end_date=date(2022, 12, 30),
        period_type=PeriodType.BEAR_MARKET,
        description="Full year 2022",
        avg_vix=25.5,
        vix_range=(16.0, 37.0),
        avg_daily_range=1.4,
        trend_direction=-0.55,
        correlation_regime=0.68,
        sector_leadership="defensive",
        spy_return=-18.1,
        difficulty=0.75,
        tags=["full_year", "inflation", "rate_hikes"]
    ),

    "year_2023": MarketPeriod(
        name="year_2023",
        start_date=date(2023, 1, 3),
        end_date=date(2023, 12, 29),
        period_type=PeriodType.BULL_RUN,
        description="Full year 2023",
        avg_vix=17.0,
        vix_range=(12.0, 31.0),
        avg_daily_range=0.9,
        trend_direction=0.6,
        correlation_regime=0.52,
        sector_leadership="cyclical",
        spy_return=26.3,
        difficulty=0.45,
        tags=["full_year", "ai_rally", "soft_landing"]
    ),
}


class MarketPeriodLibrary:
    """
    Library of curated market periods for targeted backtesting.

    Enables:
    - Rapid testing on specific market conditions
    - Finding periods similar to current market
    - Multi-regime strategy validation
    """

    def __init__(self):
        self.periods = MARKET_PERIODS.copy()
        logger.info(f"MarketPeriodLibrary initialized with {len(self.periods)} periods")

    def get_period(self, name: str) -> Optional[MarketPeriod]:
        """Get a specific period by name."""
        return self.periods.get(name)

    def get_all_periods(self) -> List[MarketPeriod]:
        """Get all periods."""
        return list(self.periods.values())

    def get_periods_by_type(self, period_type: PeriodType) -> List[MarketPeriod]:
        """Get all periods of a specific type."""
        return [p for p in self.periods.values() if p.period_type == period_type]

    def get_short_periods(self, max_days: int = 90) -> List[MarketPeriod]:
        """Get periods suitable for rapid testing."""
        return [p for p in self.periods.values() if p.duration_days <= max_days]

    def get_periods_by_tag(self, tag: str) -> List[MarketPeriod]:
        """Get periods containing a specific tag."""
        return [p for p in self.periods.values() if tag in p.tags]

    def get_crisis_periods(self) -> List[MarketPeriod]:
        """Get all crisis periods for stress testing."""
        return self.get_periods_by_type(PeriodType.CRISIS)

    def get_year_periods(self) -> List[MarketPeriod]:
        """Get full year periods for long-term validation."""
        return [p for p in self.periods.values() if "full_year" in p.tags]

    def find_similar_periods(
        self,
        current_vix: float,
        current_trend: float = 0.0,
        current_correlation: float = 0.5,
        top_n: int = 5
    ) -> List[Tuple[MarketPeriod, float]]:
        """
        Find periods most similar to current market conditions.

        Args:
            current_vix: Current VIX level
            current_trend: Current trend direction (-1 to 1)
            current_correlation: Current correlation regime (0 to 1)
            top_n: Number of similar periods to return

        Returns:
            List of (period, similarity_score) tuples, sorted by similarity
        """
        similarities = []

        for period in self.periods.values():
            # Calculate similarity based on multiple factors

            # VIX similarity (0-1, 1 = identical)
            vix_diff = abs(current_vix - period.avg_vix)
            vix_sim = max(0, 1 - vix_diff / 40)  # Normalize by 40 VIX points

            # Trend similarity
            trend_diff = abs(current_trend - period.trend_direction)
            trend_sim = max(0, 1 - trend_diff / 2)  # Scale is -1 to 1

            # Correlation similarity
            corr_diff = abs(current_correlation - period.correlation_regime)
            corr_sim = max(0, 1 - corr_diff)

            # Weighted combination
            similarity = (
                vix_sim * 0.4 +
                trend_sim * 0.35 +
                corr_sim * 0.25
            )

            similarities.append((period, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: -x[1])

        return similarities[:top_n]

    def find_periods_by_vix_range(
        self,
        min_vix: float = 0,
        max_vix: float = 100
    ) -> List[MarketPeriod]:
        """Find periods where average VIX falls within range."""
        return [
            p for p in self.periods.values()
            if min_vix <= p.avg_vix <= max_vix
        ]

    def get_diverse_test_set(self, n_periods: int = 6) -> List[MarketPeriod]:
        """
        Get a diverse set of periods for comprehensive testing.

        Ensures representation from different market regimes.
        """
        selected = []

        # Get one from each major type
        type_priority = [
            PeriodType.CRISIS,
            PeriodType.BEAR_MARKET,
            PeriodType.RECOVERY,
            PeriodType.BULL_RUN,
            PeriodType.HIGH_VOL,
            PeriodType.SIDEWAYS,
        ]

        for period_type in type_priority:
            periods = self.get_periods_by_type(period_type)
            if periods and len(selected) < n_periods:
                # Pick the one with highest difficulty (most challenging)
                best = max(periods, key=lambda p: p.difficulty)
                selected.append(best)

        return selected

    def get_rapid_test_suite(self) -> Dict[str, List[MarketPeriod]]:
        """
        Get periods organized for rapid multi-environment testing.

        Returns dict with:
        - 'crisis': Short crisis periods
        - 'recovery': Recovery periods
        - 'bull': Bull market periods
        - 'bear': Bear market periods
        - 'sideways': Choppy periods
        """
        return {
            'crisis': [
                self.periods['covid_crash'],
                self.periods['volpocalypse_2018'],
                self.periods['flash_crash_2010'],
            ],
            'recovery': [
                self.periods['covid_recovery_2020'],
                self.periods['q1_2019_recovery'],
            ],
            'bull': [
                self.periods['2017_low_vol_bull'],
                self.periods['post_election_2016'],
            ],
            'bear': [
                self.periods['q4_2018_selloff'],
                self.periods['2022_bear_market'],
            ],
            'sideways': [
                self.periods['2018_h1_chop'],
                self.periods['2015_sideways'],
            ],
        }

    def get_period_data_range(self, period: MarketPeriod) -> Tuple[datetime, datetime]:
        """Get datetime range for a period."""
        return (
            datetime.combine(period.start_date, datetime.min.time()),
            datetime.combine(period.end_date, datetime.max.time())
        )

    def filter_data_to_period(
        self,
        data: pd.DataFrame,
        period: MarketPeriod,
        date_column: str = 'timestamp'
    ) -> pd.DataFrame:
        """
        Filter a DataFrame to only include data within a period.

        Args:
            data: DataFrame with a date/timestamp column
            period: MarketPeriod to filter to
            date_column: Name of the date column

        Returns:
            Filtered DataFrame
        """
        if date_column not in data.columns:
            # Try index
            if isinstance(data.index, pd.DatetimeIndex):
                mask = (data.index.date >= period.start_date) & (data.index.date <= period.end_date)
                return data[mask]
            logger.warning(f"Could not find date column '{date_column}'")
            return data

        # Convert to date for comparison
        dates = pd.to_datetime(data[date_column]).dt.date
        mask = (dates >= period.start_date) & (dates <= period.end_date)
        return data[mask]

    def print_period_summary(self, period_name: str = None):
        """Print summary of one or all periods."""
        if period_name:
            periods = [self.get_period(period_name)]
            if periods[0] is None:
                print(f"Period '{period_name}' not found")
                return
        else:
            periods = self.get_all_periods()

        print("\n" + "=" * 80)
        print("MARKET PERIOD LIBRARY")
        print("=" * 80)

        for period in sorted(periods, key=lambda p: p.start_date):
            print(f"\n{period.name.upper()}")
            print("-" * 40)
            print(f"  Type: {period.period_type.value}")
            print(f"  Dates: {period.start_date} to {period.end_date} ({period.duration_days} days)")
            print(f"  Description: {period.description}")
            print(f"  VIX: avg {period.avg_vix:.0f} (range {period.vix_range[0]:.0f}-{period.vix_range[1]:.0f})")
            print(f"  SPY Return: {period.spy_return:+.1f}%")
            print(f"  Trend: {period.trend_direction:+.2f} | Correlation: {period.correlation_regime:.2f}")
            print(f"  Difficulty: {period.difficulty:.2f}")
            print(f"  Tags: {', '.join(period.tags)}")


# =============================================================================
# CLI Demo
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    library = MarketPeriodLibrary()

    print("\n" + "=" * 60)
    print("MARKET PERIOD LIBRARY DEMO")
    print("=" * 60)

    # Show stats
    print(f"\nTotal periods: {len(library.get_all_periods())}")
    print(f"Crisis periods: {len(library.get_crisis_periods())}")
    print(f"Short periods (rapid testing): {len(library.get_short_periods())}")
    print(f"Full year periods: {len(library.get_year_periods())}")

    # Find similar to current conditions (hypothetical)
    print("\n" + "-" * 40)
    print("Finding periods similar to current: VIX=18, trend=0.3, corr=0.55")
    similar = library.find_similar_periods(18, 0.3, 0.55)
    for period, score in similar:
        print(f"  {period.name:25s} similarity: {score:.2f}")

    # Get rapid test suite
    print("\n" + "-" * 40)
    print("Rapid test suite:")
    suite = library.get_rapid_test_suite()
    for category, periods in suite.items():
        print(f"  {category.upper()}: {[p.name for p in periods]}")

    # Show a few key periods
    print("\n" + "-" * 40)
    library.print_period_summary("covid_crash")
    library.print_period_summary("2017_low_vol_bull")
