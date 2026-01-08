"""
Screen Controller for Trading System

Multi-page display system controlled by rotary encoder.
Uses simple subprocess-based GPIO polling for reliability.

Encoder controls:
- Rotate: Change page
- Click: Scroll positions / refresh
"""

import random
import subprocess
import threading
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum, auto

from .display import LCDDisplay, get_display_manager
from .gpio_config import LCD_TRADING_ADDR, GPIO_CHIP, ENCODER_PINS

# Import LED controller for feedback
try:
    from .leds import get_led_controller
    HAS_LEDS = True
except ImportError:
    get_led_controller = None
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
        enc_cfg = ENCODER_PINS.get('trading', {})
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

        # LED controller for feedback
        self._leds = None
        if HAS_LEDS:
            try:
                self._leds = get_led_controller()
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
        self._data: Dict[str, Any] = {
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

        # Get the primary display
        self._screen = self._display_manager.trading

    @property
    def screen_available(self) -> bool:
        """Check if screen is available."""
        return self._screen.available

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
        """Handle encoder click - scroll positions or reset screensaver."""
        with self._lock:
            if self._current_page == DisplayPage.POSITIONS:
                # Scroll through positions
                positions = self._data.get('positions', [])
                if positions:
                    self._position_scroll_idx = (self._position_scroll_idx + 1) % max(1, len(positions) - 2)
            elif self._current_page == DisplayPage.SCREENSAVER:
                # Reset Game of Life with new random pattern
                self._gol_grid = []
                self._gol_stale_count = 99  # Force reinit on next render

        self._render_current_page()

    def update_data(self, data: Dict[str, Any]) -> None:
        """
        Update cached data from orchestrator.

        Call this periodically with fresh data.
        """
        with self._lock:
            self._data.update(data)

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

        lines = [
            "[TRADING]",
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

        # Header shows phase and time remaining
        header = f"{phase_short} ({time_remaining})"

        lines = [
            header.center(20),
            f"SPY: ${spy:.2f} {spy_arrow}",
            f"VIX: {vix:.1f} {vix_arrow}",
            f"Time: {now}"
        ]
        self._screen.write_all(lines)

    def _render_positions(self, data: Dict[str, Any]) -> None:
        """Render positions list page."""
        positions = data.get('positions', [])
        scroll_idx = self._position_scroll_idx

        lines = ["[POSITIONS]"]

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
        """Render system status page."""
        mem_pct = data.get('memory_pct', 0)
        zram_pct = data.get('zram_pct', 0)
        cpu_pct = data.get('cpu_pct', 0)
        uptime = data.get('uptime', '--')
        phase = data.get('phase', 'unknown')

        lines = [
            "[SYSTEM]",
            f"RAM:{mem_pct:.0f}%  ZRAM:{zram_pct:.0f}%",
            f"CPU: {cpu_pct:.0f}%  Up: {uptime}",
            f"Phase: {phase[:12]}"
        ]
        self._screen.write_all(lines)

    def _render_research(self, data: Dict[str, Any]) -> None:
        """Render research status page."""
        status = data.get('research_status', 'IDLE')
        gen = data.get('research_generation', 0)
        max_gen = data.get('research_max_gen', 100)
        sharpe = data.get('research_best_sharpe', 0)
        eta = data.get('research_eta', 0)

        # Progress bar
        progress = int((gen / max_gen) * 14) if max_gen > 0 else 0
        bar = '#' * progress + '.' * (14 - progress)

        eta_str = f"{eta // 60}h{eta % 60:02d}m" if eta >= 60 else f"{eta}m"

        lines = [
            "[RESEARCH]",
            f"Status: {status}",
            f"[{bar}]",
            f"Gen:{gen}/{max_gen} SR:{sharpe:.2f}"
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

    def _display_update_loop(self) -> None:
        """Background display update loop."""
        while self._running:
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
