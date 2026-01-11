"""
Screen Controller for Trading System

Multi-page display system controlled by rotary encoder.
Uses simple subprocess-based GPIO polling for reliability.

Encoder controls:
- Rotate: Change page
- Click: Scroll positions / refresh

Data Persistence:
- Screen data is cached to disk on updates
- On restart, loads cached data immediately (marked as stale)
- Provides instant display while system initializes
"""

import json
import os
import random
import sqlite3
import subprocess
import threading
import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, Any, List
from enum import Enum, auto

import psutil

from .display import LCDDisplay, get_display_manager
from .gpio_config import LCD_ADDR, GPIO_CHIP, ENCODER_PINS

# Cache file location
SCREEN_CACHE_FILE = Path(__file__).parent.parent / "db" / "screen_cache.json"

# Import LED interface for feedback (uses orchestrator if authority, else client)
try:
    from .led_authority import get_led_interface
    HAS_LEDS = True
except ImportError:
    get_led_interface = None
    HAS_LEDS = False


class DisplayPage(Enum):
    """Available display pages."""
    MARKET = auto()       # SPY, VIX, phase info (default)
    TRADING = auto()      # Portfolio value, P&L
    POSITIONS = auto()    # Active positions list
    SYSTEM = auto()       # System health info
    RESEARCH = auto()     # Research progress
    SCREENSAVER = auto()  # Animated screensaver


class ScreenController:
    """
    Controls LCD display with encoder navigation.
    Uses direct subprocess GPIO polling for reliability.
    """

    PAGE_ORDER = [
        DisplayPage.MARKET,     # Default page
        DisplayPage.TRADING,
        DisplayPage.POSITIONS,
        DisplayPage.SYSTEM,
        DisplayPage.RESEARCH,
        DisplayPage.SCREENSAVER,
    ]

    PAGE_TITLES = {
        DisplayPage.MARKET: "MARKET",
        DisplayPage.TRADING: "TRADING",
        DisplayPage.POSITIONS: "POSITIONS",
        DisplayPage.SYSTEM: "SYSTEM",
        DisplayPage.RESEARCH: "RESEARCH",
        DisplayPage.SCREENSAVER: "SCREENSAVER",
    }

    def __init__(self, single_screen_mode: bool = True):
        """Initialize screen controller."""
        self.single_screen_mode = single_screen_mode
        self._display_manager = get_display_manager()

        # Encoder pins
        enc_cfg = ENCODER_PINS.get('main', {})
        self._clk_pin = enc_cfg.get('clk', 22)
        self._dt_pin = enc_cfg.get('dt', 17)
        self._sw_pin = enc_cfg.get('sw', 27)

        # Current state
        self._current_page_idx = 0
        self._current_page = self.PAGE_ORDER[0]
        self._position_scroll_idx = 0

        # Encoder state tracking
        self._last_clk = 1
        self._last_sw = 1
        self._last_rotate_time: float = 0
        self._debounce_ms: float = 150  # Minimum ms between page changes
        self._button_press_time: float = 0  # For long-press detection
        self._backlight_on: bool = True
        self._reset_feedback_given: bool = False  # LED blink given at 5s threshold
        self._reset_hold_threshold: float = 5.0  # Seconds to hold for reset

        # LED interface for feedback (orchestrator if authority, else client)
        self._leds = None
        if HAS_LEDS:
            try:
                self._leds = get_led_interface()
            except Exception:
                pass

        # Screensaver animation
        self._screensaver_frame: int = 0

        # Price tracking for directional arrows
        self._prev_spy: float = 0
        self._prev_vix: float = 0
        # Persistent arrow state (only changes on actual price movement)
        self._last_spy_arrow: str = '-'
        self._last_vix_arrow: str = '-'

        # Game of Life state (20x8 grid using half-blocks)
        self._gol_grid: List[List[int]] = []
        self._gol_prev_grid: List[List[int]] = []
        self._gol_stale_count: int = 0
        self._gol_chars_initialized: bool = False

        # Data cache (updated by orchestrator)
        # Try to load from persistent cache first for instant display on restart
        self._data_is_stale: bool = False
        self._data: Dict[str, Any] = self._load_cache() or {
            'portfolio_value': 0,
            'daily_pnl': 0,
            'daily_pnl_pct': 0,
            'position_count': 0,
            'positions': [],
            'cash': 0,
            'cash_pct': 0,
            'spy_price': 0,
            'vix': 0,
            'phase': 'unknown',
            'phase_time_remaining': '',
            'memory_pct': 0,
            'zram_pct': 0,
            'cpu_pct': 0,
            'uptime': '',
            'research_status': 'IDLE',
            'research_generation': 0,
            'research_max_gen': 100,
            'research_best_sharpe': 0,
            'research_eta': 0,
        }

        self._lock = threading.Lock()
        self._running = False
        self._encoder_thread: Optional[threading.Thread] = None
        self._display_thread: Optional[threading.Thread] = None

        # System stats - fetched locally by screen controller (independent of orchestrator)
        self._system_stats: Dict[str, Any] = {
            'memory_pct': 0,
            'zram_pct': 0,
            'cpu_pct': 0,
            'cpu_temp': 0,
            'load_avg': 0,
        }
        self._last_system_stats_fetch: float = 0
        self._system_stats_interval: float = 5.0  # Fetch every 5 seconds

        # Research stats - fetched locally from research.db (independent of orchestrator)
        self._research_stats: Dict[str, Any] = {
            'status': 'IDLE',           # IDLE, EVOLVING, STALLED, DONE
            'strategy': '',             # Current strategy being evolved
            'generation': 0,            # Current generation for that strategy
            'best_sharpe': 0.0,         # Best sharpe for that strategy
            'last_update': None,        # Timestamp of last ga_history write
            # Summary stats for page 2
            'total_generations': 0,     # Sum of max generations across all strategies
            'best_strategy': '',        # Strategy with highest sharpe
            'best_strategy_gen': 0,     # Generation count for best strategy
            'best_strategy_sharpe': 0.0,# Sharpe for best strategy
            'strategy_count': 0,        # Number of strategies being evolved
        }
        self._last_research_stats_fetch: float = 0
        self._research_stats_interval_active: float = 5.0   # Fetch every 5s when research running
        self._research_stats_interval_idle: float = 60.0    # Fetch every 60s when idle
        self._research_stale_threshold_ga: float = 300.0    # 5 minutes for GA history (fast updates)
        self._research_stale_threshold_gp: float = 7200.0   # 2 hours for GP checkpoints (slow updates)
        self._research_db_path: Path = Path(__file__).parent.parent / "db" / "research.db"
        self._research_page_idx: int = 0                    # 0 = live progress, 1 = summary

        # Trading data freshness tracking
        self._trading_data_updated_at: float = 0  # time.time() when orchestrator last pushed
        self._orchestrator_start_time: float = time.time()  # Track controller start for uptime

        # Get the primary display
        self._screen = self._display_manager.trading

    @property
    def screen_available(self) -> bool:
        """Check if screen is available."""
        return self._screen.available

    def _load_cache(self) -> Optional[Dict[str, Any]]:
        """Load cached screen data from disk for instant display on restart."""
        try:
            if SCREEN_CACHE_FILE.exists():
                with open(SCREEN_CACHE_FILE, 'r') as f:
                    cached = json.load(f)

                # Check cache age - if older than 24 hours, ignore
                cached_at = cached.get('_cached_at', '')
                if cached_at:
                    try:
                        cache_time = datetime.fromisoformat(cached_at)
                        age_hours = (datetime.now() - cache_time).total_seconds() / 3600
                        if age_hours > 24:
                            print(f"[ScreenController] Cache too old ({age_hours:.1f}h), starting fresh")
                            return None
                    except (ValueError, TypeError):
                        pass

                # Mark data as stale (will show indicator until fresh data arrives)
                self._data_is_stale = True
                print(f"[ScreenController] Loaded cached data from {SCREEN_CACHE_FILE}")
                return cached
        except Exception as e:
            print(f"[ScreenController] Failed to load cache: {e}")
        return None

    def _save_cache(self) -> None:
        """Save current screen data to disk for persistence across restarts."""
        try:
            # Ensure directory exists
            SCREEN_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

            # Add timestamp
            cache_data = self._data.copy()
            cache_data['_cached_at'] = datetime.now().isoformat()

            with open(SCREEN_CACHE_FILE, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)

        except Exception as e:
            # Don't log every time - too noisy
            pass

    def _read_encoder_gpio(self) -> tuple:
        """Read encoder GPIO pins using subprocess."""
        try:
            result = subprocess.run(
                ['gpioget', '-c', GPIO_CHIP, '--bias=pull-up',
                 str(self._clk_pin), str(self._dt_pin), str(self._sw_pin)],
                capture_output=True, text=True, timeout=1
            )
            states = {}
            for part in result.stdout.strip().split():
                if '=' in part:
                    pin_str, val = part.replace('"', '').split('=')
                    states[int(pin_str)] = 0 if val == 'inactive' else 1
            return (
                states.get(self._clk_pin, 1),
                states.get(self._dt_pin, 1),
                states.get(self._sw_pin, 1)
            )
        except Exception:
            return (1, 1, 1)

    def _encoder_poll_loop(self) -> None:
        """Poll encoder GPIO and handle events."""
        while self._running:
            try:
                clk, dt, sw = self._read_encoder_gpio()
                now = time.time()

                # Rotation detection (on CLK falling edge)
                if self._last_clk == 1 and clk == 0:
                    now_ms = now * 1000
                    if now_ms - self._last_rotate_time >= self._debounce_ms:
                        self._last_rotate_time = now_ms
                        with self._lock:
                            old_page = self._current_page
                            if dt == 1:  # CW
                                self._current_page_idx = (self._current_page_idx + 1) % len(self.PAGE_ORDER)
                            else:  # CCW
                                self._current_page_idx = (self._current_page_idx - 1) % len(self.PAGE_ORDER)
                            self._current_page = self.PAGE_ORDER[self._current_page_idx]
                            self._position_scroll_idx = 0

                            # Reset display when leaving screensaver (clears custom chars)
                            if old_page == DisplayPage.SCREENSAVER and self._current_page != DisplayPage.SCREENSAVER:
                                self._screen.clear()
                                self._gol_chars_initialized = False

                        self._render_current_page()

                # Button press detection (falling edge - button down)
                if self._last_sw == 1 and sw == 0:
                    self._button_press_time = now
                    self._reset_feedback_given = False

                # While button is held, check for 5-second threshold
                if sw == 0 and self._button_press_time > 0:
                    hold_duration = now - self._button_press_time
                    # Give LED feedback at 5-second threshold (once)
                    if hold_duration >= self._reset_hold_threshold and not self._reset_feedback_given:
                        self._reset_feedback_given = True
                        # Blink LEDs to confirm reset threshold reached
                        if self._leds:
                            try:
                                self._leds.flash_all('cyan', times=3, interval=0.1)
                            except Exception:
                                pass

                # Button release detection
                if self._last_sw == 0 and sw == 1:
                    press_duration = now - self._button_press_time
                    if press_duration >= self._reset_hold_threshold:
                        # 5+ second hold: Reset screen
                        self._reset_screen()
                    elif press_duration >= 1.0:
                        # 1-5 second hold: Toggle backlight
                        self._toggle_backlight()
                    else:
                        # Short click
                        self._on_click()

                self._last_clk = clk
                self._last_sw = sw

            except Exception:
                pass

            time.sleep(0.008)  # ~125Hz polling

    def _toggle_backlight(self) -> None:
        """Toggle LCD backlight on/off."""
        self._backlight_on = not self._backlight_on
        if hasattr(self._screen, '_lcd') and self._screen._lcd:
            try:
                if self._backlight_on:
                    self._screen._lcd.backlight_enabled = True
                else:
                    self._screen._lcd.backlight_enabled = False
            except Exception:
                pass

    def _reset_screen(self) -> None:
        """Reset the LCD screen by reinitializing it."""
        print("[ScreenController] Screen reset requested via button hold")

        # Flash LEDs green to confirm reset is happening
        if self._leds:
            try:
                self._leds.flash_all('green', times=2, interval=0.15)
            except Exception:
                pass

        # Reinitialize the LCD
        success = self._screen.reinit()

        if success:
            # Reinitialize custom chars for screensaver
            self._gol_chars_initialized = False
            self._backlight_on = True
            # Render current page
            self._render_current_page()
            print("[ScreenController] Screen reset successful")
        else:
            # Flash red to indicate failure
            if self._leds:
                try:
                    self._leds.flash_all('red', times=3, interval=0.2)
                except Exception:
                    pass
            print("[ScreenController] Screen reset failed")

    def _on_click(self) -> None:
        """Handle encoder click - scroll positions, toggle research pages, or reset screensaver."""
        with self._lock:
            if self._current_page == DisplayPage.POSITIONS:
                # Scroll through positions
                positions = self._data.get('positions', [])
                if positions:
                    self._position_scroll_idx = (self._position_scroll_idx + 1) % max(1, len(positions) - 2)
            elif self._current_page == DisplayPage.RESEARCH:
                # Toggle between live progress (0) and summary (1)
                self._research_page_idx = (self._research_page_idx + 1) % 2
            elif self._current_page == DisplayPage.SCREENSAVER:
                # Reset Game of Life with new random pattern
                self._gol_grid = []
                self._gol_stale_count = 99  # Force reinit on next render

        self._render_current_page()

    def update_data(self, data: Dict[str, Any]) -> None:
        """
        Update cached data from orchestrator.

        Call this periodically with fresh data.
        Saves to disk cache for persistence across restarts.

        Note: System stats (memory_pct, cpu_pct, etc.) are now fetched
        independently by the screen controller. The orchestrator can still
        push them, but they will be overwritten by fresh local values.
        """
        with self._lock:
            self._data.update(data)
            # Clear stale flag now that we have fresh data
            self._data_is_stale = False
            # Track when trading data was last received
            self._trading_data_updated_at = time.time()

        # Save to disk cache (throttled - only every few updates)
        # We do this outside the lock to avoid blocking
        self._save_cache()

    def _render_current_page(self) -> None:
        """Render the current page to the display."""
        if not self._screen.available:
            return

        with self._lock:
            page = self._current_page
            data = self._data.copy()

        if page == DisplayPage.TRADING:
            self._render_trading(data)
        elif page == DisplayPage.MARKET:
            self._render_market(data)
        elif page == DisplayPage.POSITIONS:
            self._render_positions(data)
        elif page == DisplayPage.SYSTEM:
            self._render_system(data)
        elif page == DisplayPage.RESEARCH:
            self._render_research(data)
        elif page == DisplayPage.SCREENSAVER:
            self._render_screensaver()

    def _render_trading(self, data: Dict[str, Any]) -> None:
        """Render trading overview page."""
        pv = data.get('portfolio_value', 0)
        pnl = data.get('daily_pnl', 0)
        pnl_pct = data.get('daily_pnl_pct', 0)
        positions = data.get('position_count', 0)
        cash_pct = data.get('cash_pct', 0)

        pnl_sign = '+' if pnl >= 0 else ''
        pct_sign = '+' if pnl_pct >= 0 else ''

        # Show data age if stale (from cache or orchestrator not updating)
        data_age = self._get_trading_data_age_str()
        if data_age == "live":
            header = "[TRADING]"
        elif data_age == "no data":
            header = "[TRADING] waiting..."
        else:
            header = f"[TRADING] {data_age}"

        lines = [
            header,
            f"${pv:,.0f}",
            f"Today: {pnl_sign}${pnl:,.0f} ({pct_sign}{pnl_pct:.1f}%)",
            f"Pos: {positions}  Cash: {cash_pct:.0f}%"
        ]
        self._screen.write_all(lines)

    def _render_market(self, data: Dict[str, Any]) -> None:
        """Render market status page."""
        spy = data.get('spy_price', 0)
        vix = data.get('vix', 0)
        phase = data.get('phase', 'unknown')
        time_remaining = data.get('phase_time_remaining', '--')

        # Shorten phase names for header
        phase_short = {
            'pre_market': 'PRE-MKT',
            'market_open': 'OPEN',
            'intraday_open': 'INTRA',
            'intraday_active': 'INTRA',
            'post_market': 'POST',
            'evening': 'EVENING',
            'overnight': 'OVERNIGHT',
            'weekend': 'WEEKEND',
        }.get(phase, phase[:8].upper())

        now = datetime.now().strftime('%H:%M:%S')

        # Determine directional arrows (^ up, v down, - flat)
        # Arrows PERSIST - only change when price actually moves in a new direction
        # Dash only shown if no previous movement recorded
        if self._prev_spy > 0 and spy > 0:
            if spy > self._prev_spy:
                self._last_spy_arrow = '^'
            elif spy < self._prev_spy:
                self._last_spy_arrow = 'v'
            # If equal, keep previous arrow (don't reset to dash)
        spy_arrow = self._last_spy_arrow

        if self._prev_vix > 0 and vix > 0:
            if vix > self._prev_vix:
                self._last_vix_arrow = '^'
            elif vix < self._prev_vix:
                self._last_vix_arrow = 'v'
            # If equal, keep previous arrow (don't reset to dash)
        vix_arrow = self._last_vix_arrow

        # Update previous values for next comparison
        if spy > 0:
            self._prev_spy = spy
        if vix > 0:
            self._prev_vix = vix

        # Header always shows phase and countdown to next phase
        header = f"{phase_short} ({time_remaining})"

        # Show staleness indicator on time line if data is old
        data_age = self._get_trading_data_age_str()
        if data_age in ("live", "no data"):
            time_line = f"Time: {now}"
        else:
            time_line = f"{now} [{data_age}]"

        lines = [
            header.center(20),
            f"SPY: ${spy:.2f} {spy_arrow}",
            f"VIX: {vix:.1f} {vix_arrow}",
            time_line
        ]
        self._screen.write_all(lines)

    def _render_positions(self, data: Dict[str, Any]) -> None:
        """Render positions list page."""
        positions = data.get('positions', [])
        scroll_idx = self._position_scroll_idx

        # Show data age if stale
        data_age = self._get_trading_data_age_str()
        if data_age == "live":
            header = "[POSITIONS]"
        elif data_age == "no data":
            header = "[POSITIONS] waiting..."
        else:
            header = f"[POSITIONS] {data_age}"
        lines = [header]

        if not positions:
            lines.append("No open positions")
            lines.append("")
            lines.append("Click to refresh")
        else:
            # Show up to 3 positions starting at scroll_idx
            visible = positions[scroll_idx:scroll_idx + 3]
            for pos in visible:
                symbol = pos.get('symbol', '???')[:6]
                qty = pos.get('qty', 0)
                pnl = pos.get('unrealized_pnl', 0)
                pnl_sign = '+' if pnl >= 0 else ''
                lines.append(f"{symbol:6} {qty:>4} {pnl_sign}${pnl:,.0f}")

            # Fill remaining lines
            while len(lines) < 4:
                if len(positions) > 3:
                    lines.append(f"  [{scroll_idx+1}-{min(scroll_idx+3, len(positions))}/{len(positions)}] Click:more")
                else:
                    lines.append("")

        self._screen.write_all(lines)

    def _render_system(self, data: Dict[str, Any]) -> None:
        """Render system status page.

        Uses locally-fetched system stats (always fresh) rather than
        orchestrator-pushed data. This ensures the SYSTEM page updates
        every 5 seconds regardless of orchestrator check interval.
        """
        # Use fresh local stats (fetched by _fetch_system_stats)
        with self._lock:
            stats = self._system_stats.copy()

        mem_pct = stats.get('memory_pct', 0)
        zram_pct = stats.get('zram_pct', 0)
        cpu_pct = stats.get('cpu_pct', 0)
        uptime = stats.get('uptime', '--')
        cpu_temp = stats.get('cpu_temp', 0)
        load_avg = stats.get('load_avg', 0)

        # System page is always fresh (locally fetched), no stale indicator needed
        lines = [
            "[SYSTEM]",
            f"RAM:{mem_pct:.0f}%  ZRAM:{zram_pct:.0f}%",
            f"CPU: {cpu_pct:.0f}%  Up: {uptime}",
            f"Temp:{cpu_temp:.0f}C  Load:{load_avg:.1f}"
        ]
        self._screen.write_all(lines)

    def _render_research(self, data: Dict[str, Any]) -> None:
        """Render research status page.

        Two pages (click to toggle):
        - Page 0: Live progress (current strategy being evolved)
        - Page 1: Summary (total gens, best strategy overall)

        Uses locally-fetched research stats from research.db (always fresh).
        """
        # Use fresh local stats (fetched by _fetch_research_stats)
        with self._lock:
            stats = self._research_stats.copy()
            page_idx = self._research_page_idx

        if page_idx == 1:
            # Page 2: Summary stats
            self._render_research_summary(stats)
        else:
            # Page 1: Live progress (default)
            self._render_research_live(stats)

    def _render_research_live(self, stats: Dict[str, Any]) -> None:
        """Render research live progress page (page 1)."""
        status = stats.get('status', 'IDLE')
        strategy = stats.get('strategy', '')
        gen = stats.get('generation', 0)
        sharpe = stats.get('best_sharpe', 0.0)

        # Determine display based on status
        if status == 'IDLE':
            lines = [
                "[RESEARCH 1/2]",
                "Status: IDLE",
                "",
                "Click for summary"
            ]
        elif status == 'STARTING':
            lines = [
                "[RESEARCH 1/2]",
                "Status: STARTING",
                "",
                "Initializing..."
            ]
        elif status == 'STALLED':
            lines = [
                "[RESEARCH 1/2]",
                "Status: STALLED",
                "Process not responding",
                "Click for summary"
            ]
        elif status == 'DONE':
            # Show best result from completed run
            if strategy:
                lines = [
                    "[RESEARCH 1/2]",
                    "COMPLETE",
                    f"Best: {strategy}",
                    f"Gen:{gen} Sharpe:{sharpe:.2f}"
                ]
            else:
                lines = [
                    "[RESEARCH 1/2]",
                    "Status: COMPLETE",
                    f"Generations: {gen}",
                    f"Best Sharpe: {sharpe:.2f}"
                ]
        else:
            # EVOLVING - show live progress (updates every 5 seconds)
            if strategy:
                lines = [
                    "[RESEARCH 1/2]",
                    f"NOW: {strategy}",
                    f"Generation: {gen}",
                    f"Sharpe: {sharpe:.2f}"
                ]
            else:
                lines = [
                    "[RESEARCH 1/2]",
                    "EVOLVING...",
                    f"Generation: {gen}",
                    f"Sharpe: {sharpe:.2f}"
                ]

        self._screen.write_all(lines)

    def _render_research_summary(self, stats: Dict[str, Any]) -> None:
        """Render research summary page (page 2)."""
        total_gens = stats.get('total_generations', 0)
        strategy_count = stats.get('strategy_count', 0)
        best_strategy = stats.get('best_strategy', '')
        best_gen = stats.get('best_strategy_gen', 0)
        best_sharpe = stats.get('best_strategy_sharpe', 0.0)

        if strategy_count == 0:
            lines = [
                "[RESEARCH 2/2]",
                "No research data",
                "",
                "Click for live view"
            ]
        else:
            lines = [
                "[RESEARCH 2/2]",
                f"Total: {total_gens} gens/{strategy_count} strats",
                f"Best: {best_strategy}",
                f"Gen:{best_gen} Sharpe:{best_sharpe:.2f}"
            ]

        self._screen.write_all(lines)

    def _render_screensaver(self) -> None:
        """Render Conway's Game of Life on 20x8 high-res grid using half-block characters."""
        # High resolution: 20 cols x 8 rows (2 virtual rows per LCD row)
        rows, cols = 8, 20

        # Ensure custom characters are loaded for half-block rendering (once per session)
        if not self._gol_chars_initialized:
            if hasattr(self._screen, '_init_custom_chars'):
                self._screen._init_custom_chars()
            self._gol_chars_initialized = True

        # Initialize grid if empty or needs reset
        if not self._gol_grid or len(self._gol_grid) != rows or self._gol_stale_count > 15:
            # Reinitialize custom chars on grid reset (in case display was cleared)
            if hasattr(self._screen, '_init_custom_chars'):
                self._screen._init_custom_chars()
            self._gol_grid = self._gol_init_grid(rows, cols)
            self._gol_prev_grid = []
            self._gol_stale_count = 0

        # Calculate next generation
        new_grid = self._gol_next_gen(self._gol_grid, rows, cols)

        # Check for stagnation (same as current or oscillating with prev)
        if new_grid == self._gol_grid or new_grid == self._gol_prev_grid:
            self._gol_stale_count += 1
        else:
            self._gol_stale_count = 0

        # Check for extinction
        if sum(sum(row) for row in new_grid) == 0:
            self._gol_stale_count = 99  # Force reinit

        # Update state
        self._gol_prev_grid = self._gol_grid
        self._gol_grid = new_grid

        # Render to display using half-block characters
        # Custom chars: 0=upper half, 1=lower half, 2=full block, space=empty
        lines = []
        for lcd_row in range(4):
            top_row = self._gol_grid[lcd_row * 2]
            bot_row = self._gol_grid[lcd_row * 2 + 1]
            line = ''
            for c in range(cols):
                top = top_row[c]
                bot = bot_row[c]
                if top and bot:
                    line += chr(2)   # Full block
                elif top:
                    line += chr(0)   # Upper half
                elif bot:
                    line += chr(1)   # Lower half
                else:
                    line += ' '      # Empty
            lines.append(line)

        self._screen.write_all(lines)

    def _gol_init_grid(self, rows: int, cols: int) -> List[List[int]]:
        """Initialize Game of Life grid with random cells (~35% density)."""
        return [[1 if random.random() < 0.35 else 0 for _ in range(cols)] for _ in range(rows)]

    def _gol_next_gen(self, grid: List[List[int]], rows: int, cols: int) -> List[List[int]]:
        """Calculate next generation using standard Game of Life rules."""
        new_grid = [[0] * cols for _ in range(rows)]

        for r in range(rows):
            for c in range(cols):
                # Count neighbors (toroidal wrapping)
                neighbors = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = (r + dr) % rows, (c + dc) % cols
                        neighbors += grid[nr][nc]

                # Apply rules: B3/S23
                if grid[r][c]:
                    # Alive: survives with 2-3 neighbors
                    new_grid[r][c] = 1 if neighbors in (2, 3) else 0
                else:
                    # Dead: born with exactly 3 neighbors
                    new_grid[r][c] = 1 if neighbors == 3 else 0

        return new_grid

    def _fetch_system_stats(self) -> None:
        """
        Fetch system stats locally (independent of orchestrator).

        This runs every 5 seconds regardless of orchestrator state,
        ensuring the SYSTEM page always shows fresh data.
        """
        now = time.time()
        if now - self._last_system_stats_fetch < self._system_stats_interval:
            return

        self._last_system_stats_fetch = now

        try:
            # Memory usage
            mem = psutil.virtual_memory()
            memory_pct = mem.percent

            # ZRAM/swap usage
            zram_pct = 0
            try:
                swap = psutil.swap_memory()
                if swap.total > 0:
                    zram_pct = swap.percent
            except Exception:
                pass

            # CPU usage (non-blocking, uses delta since last call)
            cpu_pct = psutil.cpu_percent(interval=None)

            # Load average (1-minute)
            load_avg = os.getloadavg()[0]

            # CPU temperature (Pi-specific)
            cpu_temp = 0
            try:
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    cpu_temp = int(f.read().strip()) / 1000.0
            except Exception:
                pass

            # Calculate uptime from controller start
            uptime_secs = now - self._orchestrator_start_time
            if uptime_secs >= 86400:  # Days
                uptime_str = f"{int(uptime_secs // 86400)}d {int((uptime_secs % 86400) // 3600)}h"
            elif uptime_secs >= 3600:  # Hours
                uptime_str = f"{int(uptime_secs // 3600)}h {int((uptime_secs % 3600) // 60)}m"
            else:
                uptime_str = f"{int(uptime_secs // 60)}m"

            # Update stats (thread-safe)
            with self._lock:
                self._system_stats.update({
                    'memory_pct': memory_pct,
                    'zram_pct': zram_pct,
                    'cpu_pct': cpu_pct,
                    'cpu_temp': cpu_temp,
                    'load_avg': load_avg,
                    'uptime': uptime_str,
                })

        except Exception:
            pass  # Silently fail - don't crash the display loop

    def _fetch_research_stats(self) -> None:
        """
        Fetch research progress from research.db (independent of orchestrator).

        Uses adaptive interval:
        - Every 5 seconds when research is actively running
        - Every 60 seconds when idle (just checking for status changes)

        Detects stalled research:
        - If status='running' but no ga_history writes in 5 minutes, shows STALLED

        Shows current activity:
        - Which strategy is currently being evolved
        - What generation it's on
        - Its current best Sharpe
        """
        now = time.time()

        # Adaptive interval based on current status
        current_status = self._research_stats.get('status', 'IDLE')
        interval = (self._research_stats_interval_active
                   if current_status == 'EVOLVING'
                   else self._research_stats_interval_idle)

        if now - self._last_research_stats_fetch < interval:
            return

        self._last_research_stats_fetch = now

        try:
            if not self._research_db_path.exists():
                return

            conn = sqlite3.connect(str(self._research_db_path), timeout=1.0)
            conn.row_factory = sqlite3.Row

            try:
                # Check for actively running research first
                cursor = conn.execute("""
                    SELECT run_id, status, planned_generations, total_generations
                    FROM ga_runs
                    WHERE status = 'running'
                    ORDER BY start_time DESC
                    LIMIT 1
                """)
                run_row = cursor.fetchone()

                if run_row:
                    # Database says research is running - verify with recent activity
                    today = date.today().isoformat()

                    # Get the MOST RECENT ga_history entry (current strategy being evolved)
                    # Use generation DESC as tiebreaker when multiple gens have same timestamp
                    cursor = conn.execute("""
                        SELECT strategy, generation, best_fitness, created_at
                        FROM ga_history
                        WHERE run_date = ?
                        ORDER BY created_at DESC, generation DESC
                        LIMIT 1
                    """, (today,))
                    latest = cursor.fetchone()

                    if latest and latest['created_at']:
                        # Parse timestamp and check staleness
                        # Database stores UTC timestamps, so compare with UTC now
                        try:
                            last_write = datetime.fromisoformat(latest['created_at'])
                            age_seconds = (datetime.utcnow() - last_write).total_seconds()
                        except (ValueError, TypeError):
                            age_seconds = 9999  # Treat parse errors as stale

                        if age_seconds < self._research_stale_threshold_ga:
                            # Actively evolving - show current strategy progress
                            # Shorten strategy name for display
                            strategy_name = latest['strategy'] or 'unknown'
                            short_name = self._shorten_strategy_name(strategy_name)

                            with self._lock:
                                self._research_stats.update({
                                    'status': 'EVOLVING',
                                    'strategy': short_name,
                                    'generation': latest['generation'] or 0,
                                    'best_sharpe': round(latest['best_fitness'] or 0, 2),
                                    'last_update': latest['created_at'],
                                })
                        else:
                            # GA history is stale - check if GP discovery phase is active
                            # First check lightweight progress file (real-time, every generation)
                            discovery_active = False
                            gp_progress_file = Path(__file__).parent.parent / "run" / "gp_progress.json"
                            try:
                                if gp_progress_file.exists():
                                    with open(gp_progress_file, 'r') as f:
                                        progress = json.load(f)
                                    # Check if progress file is fresh (< 5 min)
                                    prog_time = datetime.fromisoformat(progress['timestamp'])
                                    prog_age = (datetime.now() - prog_time).total_seconds()
                                    if prog_age < 300:  # 5 minutes
                                        discovery_active = True
                                        with self._lock:
                                            self._research_stats.update({
                                                'status': 'EVOLVING',
                                                'strategy': 'GP-Discovery',
                                                'generation': progress.get('generation', 0),
                                                'best_sharpe': progress.get('best_sortino', 0.0),
                                                'last_update': progress['timestamp'],
                                            })
                            except Exception:
                                pass  # File might not exist or be invalid

                            # Fall back to evolution_checkpoints table (less frequent updates)
                            if not discovery_active:
                                try:
                                    cursor = conn.execute("""
                                        SELECT generation, created_at
                                        FROM evolution_checkpoints
                                        ORDER BY created_at DESC
                                        LIMIT 1
                                    """)
                                    checkpoint_row = cursor.fetchone()
                                    if checkpoint_row and checkpoint_row['created_at']:
                                        cp_write = datetime.fromisoformat(checkpoint_row['created_at'])
                                        cp_age = (datetime.utcnow() - cp_write).total_seconds()
                                        # Use longer threshold for GP checkpoints (2 hours)
                                        if cp_age < self._research_stale_threshold_gp:
                                            discovery_active = True
                                            # Get best Sortino from discovered_strategies
                                            best_sortino = 0.0
                                            try:
                                                best_cursor = conn.execute("""
                                                    SELECT MAX(oos_sortino) FROM discovered_strategies
                                                """)
                                                best_row = best_cursor.fetchone()
                                                if best_row and best_row[0]:
                                                    best_sortino = best_row[0]
                                            except Exception:
                                                pass
                                            with self._lock:
                                                self._research_stats.update({
                                                    'status': 'EVOLVING',
                                                    'strategy': 'GP-Discovery',
                                                    'generation': checkpoint_row['generation'] or 0,
                                                    'best_sharpe': round(best_sortino, 2),
                                                    'last_update': checkpoint_row['created_at'],
                                                })
                                except Exception:
                                    pass  # Table might not exist

                            if not discovery_active:
                                # Check if research process is actively computing
                                # (high CPU = computing, not stalled)
                                research_computing = False
                                try:
                                    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent']):
                                        cmdline = proc.info.get('cmdline') or []
                                        if any('run_nightly_research.py' in arg for arg in cmdline):
                                            # Get CPU over short interval
                                            cpu = proc.cpu_percent(interval=0.1)
                                            if cpu > 5.0:  # >5% CPU = actively computing
                                                research_computing = True
                                                break
                                except Exception:
                                    pass

                                if research_computing:
                                    # Show as computing (between phases or slow evaluation)
                                    with self._lock:
                                        self._research_stats.update({
                                            'status': 'EVOLVING',
                                            'strategy': 'Computing...',
                                            'generation': latest['generation'] or 0,
                                            'best_sharpe': round(latest['best_fitness'] or 0, 2),
                                            'last_update': latest['created_at'],
                                        })
                                else:
                                    # Neither active DB writes nor CPU - truly stalled
                                    with self._lock:
                                        self._research_stats.update({
                                            'status': 'STALLED',
                                            'strategy': '',
                                            'generation': 0,
                                            'best_sharpe': 0.0,
                                            'last_update': latest['created_at'],
                                        })
                    else:
                        # No ga_history entries yet - research just started
                        with self._lock:
                            self._research_stats.update({
                                'status': 'STARTING',
                                'strategy': '',
                                'generation': 0,
                                'best_sharpe': 0.0,
                                'last_update': None,
                            })
                else:
                    # No running research - check for most recent completed
                    cursor = conn.execute("""
                        SELECT run_id, status, planned_generations, total_generations
                        FROM ga_runs
                        ORDER BY start_time DESC
                        LIMIT 1
                    """)
                    run_row = cursor.fetchone()

                    if run_row and run_row['status'] == 'completed':
                        # Get best result from today's runs
                        today = date.today().isoformat()
                        cursor = conn.execute("""
                            SELECT strategy, generation, best_fitness
                            FROM ga_history
                            WHERE run_date = ?
                            ORDER BY best_fitness DESC
                            LIMIT 1
                        """, (today,))
                        best_row = cursor.fetchone()

                        if best_row:
                            short_name = self._shorten_strategy_name(best_row['strategy'] or '')
                            with self._lock:
                                self._research_stats.update({
                                    'status': 'DONE',
                                    'strategy': short_name,
                                    'generation': best_row['generation'] or 0,
                                    'best_sharpe': round(best_row['best_fitness'] or 0, 2),
                                    'last_update': None,
                                })
                        else:
                            with self._lock:
                                self._research_stats.update({
                                    'status': 'DONE',
                                    'strategy': '',
                                    'generation': run_row['total_generations'] or 0,
                                    'best_sharpe': 0.0,
                                    'last_update': None,
                                })
                    else:
                        # No research data or failed/interrupted
                        with self._lock:
                            self._research_stats.update({
                                'status': 'IDLE',
                                'strategy': '',
                                'generation': 0,
                                'best_sharpe': 0.0,
                                'last_update': None,
                            })

                # Always fetch summary stats for page 2 (regardless of status)
                today = date.today().isoformat()
                cursor = conn.execute("""
                    SELECT strategy,
                           MAX(generation) as max_gen,
                           MAX(best_fitness) as best_sharpe
                    FROM ga_history
                    WHERE run_date = ?
                    GROUP BY strategy
                    ORDER BY best_sharpe DESC
                """, (today,))
                rows = cursor.fetchall()

                if rows:
                    # Calculate totals
                    total_gens = sum(r['max_gen'] or 0 for r in rows)
                    strategy_count = len(rows)

                    # Best strategy is first row (ordered by sharpe DESC)
                    best = rows[0]
                    best_name = self._shorten_strategy_name(best['strategy'] or '')

                    with self._lock:
                        self._research_stats.update({
                            'total_generations': total_gens,
                            'strategy_count': strategy_count,
                            'best_strategy': best_name,
                            'best_strategy_gen': best['max_gen'] or 0,
                            'best_strategy_sharpe': round(best['best_sharpe'] or 0, 2),
                        })
                else:
                    with self._lock:
                        self._research_stats.update({
                            'total_generations': 0,
                            'strategy_count': 0,
                            'best_strategy': '',
                            'best_strategy_gen': 0,
                            'best_strategy_sharpe': 0.0,
                        })

            finally:
                conn.close()

        except Exception:
            pass  # Silently fail - don't crash the display loop

    def _shorten_strategy_name(self, name: str) -> str:
        """Shorten strategy name for LCD display (max ~12 chars)."""
        # Common abbreviations
        abbreviations = {
            'relative_volume_breakout': 'rel_vol_brk',
            'factor_momentum': 'factor_mom',
            'sector_rotation': 'sector_rot',
            'quality_smallcap_value': 'qual_smcap',
            'mean_reversion': 'mean_rev',
            'vol_managed_momentum': 'vol_mgd_mom',
            'vix_regime_rotation': 'vix_regime',
            'pairs_trading': 'pairs',
            'gap_fill': 'gap_fill',
        }
        return abbreviations.get(name, name[:12])

    def _get_trading_data_age_str(self) -> str:
        """Get human-readable age of trading data from orchestrator."""
        if self._trading_data_updated_at == 0:
            return "no data"

        age_secs = time.time() - self._trading_data_updated_at
        if age_secs < 60:
            return "live"
        elif age_secs < 3600:
            return f"{int(age_secs // 60)}m ago"
        elif age_secs < 86400:
            return f"{int(age_secs // 3600)}h ago"
        else:
            return f"{int(age_secs // 86400)}d ago"

    def _display_update_loop(self) -> None:
        """Background display update loop.

        Fetches local data independently of orchestrator:
        - System stats: every 5 seconds (always fresh)
        - Research stats: every 5 seconds when active, 60 seconds when idle

        Renders the current page every 1 second.
        """
        while self._running:
            # Fetch fresh system stats (self-throttled to every 5 seconds)
            self._fetch_system_stats()

            # Fetch research stats (adaptive: 5s when active, 60s when idle)
            self._fetch_research_stats()

            # Render current page
            self._render_current_page()

            time.sleep(1.0)  # Update display every second

    def start(self) -> None:
        """Start the screen controller with encoder polling."""
        if self._running:
            return

        if not self._screen.available:
            print("[ScreenController] No screen available")
            return

        self._running = True

        # Start encoder polling thread
        self._encoder_thread = threading.Thread(target=self._encoder_poll_loop, daemon=True)
        self._encoder_thread.start()

        # Start display update thread
        self._display_thread = threading.Thread(target=self._display_update_loop, daemon=True)
        self._display_thread.start()

        # Show initial page
        self._render_current_page()
        print("[ScreenController] Started with encoder polling")

    def stop(self) -> None:
        """Stop the screen controller."""
        self._running = False
        if self._encoder_thread:
            self._encoder_thread.join(timeout=2)
            self._encoder_thread = None
        if self._display_thread:
            self._display_thread.join(timeout=2)
            self._display_thread = None
        self._screen.clear()

    def show_message(self, message: str, duration: float = 2.0) -> None:
        """Show a temporary message."""
        self._screen.show_message(message, duration)
        if self._running:
            self._render_current_page()


# Singleton instance
_screen_controller: Optional[ScreenController] = None


def get_screen_controller() -> ScreenController:
    """Get the singleton screen controller instance."""
    global _screen_controller
    if _screen_controller is None:
        _screen_controller = ScreenController(single_screen_mode=True)
    return _screen_controller
