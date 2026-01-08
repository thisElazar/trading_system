"""
LCD Display Controller for Trading System

Controls 20x4 I2C character LCD displays:
- Screen 1 (0x27): Trading information
- Screen 2 (0x25): Research/evolution status

Uses RPLCD library for LCD control.
"""

import threading
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

try:
    from RPLCD.i2c import CharLCD
    RPLCD_AVAILABLE = True
except ImportError:
    RPLCD_AVAILABLE = False
    CharLCD = None

from .gpio_config import (
    I2C_BUS, LCD_TRADING_ADDR, LCD_RESEARCH_ADDR, LCD_COLS, LCD_ROWS
)


@dataclass
class DisplayContent:
    """Content to display on a screen."""
    lines: List[str]
    scroll_line: Optional[int] = None  # Line index to scroll (0-3)
    scroll_text: Optional[str] = None  # Full text to scroll


class LCDDisplay:
    """Controller for a single 20x4 I2C LCD."""

    def __init__(self, address: int, name: str = "LCD"):
        self.address = address
        self.name = name
        self.cols = LCD_COLS
        self.rows = LCD_ROWS
        self._lcd: Optional[CharLCD] = None
        self._lock = threading.Lock()
        self._scroll_thread: Optional[threading.Thread] = None
        self._scroll_stop = threading.Event()
        self._available = False

        self._init_lcd()

    def _init_lcd(self) -> bool:
        """Initialize the LCD. Returns True if successful."""
        if not RPLCD_AVAILABLE:
            print(f"[{self.name}] RPLCD library not available")
            return False

        try:
            self._lcd = CharLCD(
                i2c_expander='PCF8574',
                address=self.address,
                port=I2C_BUS,
                cols=self.cols,
                rows=self.rows,
                dotsize=8
            )
            self._lcd.clear()
            self._init_custom_chars()
            self._available = True
            return True
        except Exception as e:
            print(f"[{self.name}] Failed to initialize LCD at 0x{self.address:02X}: {e}")
            self._available = False
            return False

    def _init_custom_chars(self) -> None:
        """Initialize custom characters for high-res graphics."""
        if not self._lcd:
            return

        try:
            # Custom char 0: Upper half block (top 4 rows filled)
            self._lcd.create_char(0, [0b11111, 0b11111, 0b11111, 0b11111,
                                      0b00000, 0b00000, 0b00000, 0b00000])
            # Custom char 1: Lower half block (bottom 4 rows filled)
            self._lcd.create_char(1, [0b00000, 0b00000, 0b00000, 0b00000,
                                      0b11111, 0b11111, 0b11111, 0b11111])
            # Custom char 2: Full block
            self._lcd.create_char(2, [0b11111, 0b11111, 0b11111, 0b11111,
                                      0b11111, 0b11111, 0b11111, 0b11111])
        except Exception:
            pass  # Custom chars are optional

    @property
    def available(self) -> bool:
        """Check if LCD is available."""
        return self._available

    def clear(self) -> None:
        """Clear the display."""
        if not self._available:
            return
        with self._lock:
            try:
                self._lcd.clear()
            except Exception:
                pass

    def write_line(self, line: int, text: str, center: bool = False) -> None:
        """
        Write text to a specific line (0-3).

        Args:
            line: Line number (0-3)
            text: Text to display (will be truncated/padded to 20 chars)
            center: Center the text on the line
        """
        if not self._available or line < 0 or line >= self.rows:
            return

        # Truncate or pad to exactly 20 characters
        text = str(text)[:self.cols]
        if center:
            text = text.center(self.cols)
        else:
            text = text.ljust(self.cols)

        with self._lock:
            try:
                self._lcd.cursor_pos = (line, 0)
                self._lcd.write_string(text)
            except Exception:
                pass

    def write_all(self, lines: List[str], center: bool = False) -> None:
        """
        Write up to 4 lines to the display.

        Args:
            lines: List of strings for each line
            center: Center all lines
        """
        for i, text in enumerate(lines[:self.rows]):
            self.write_line(i, text, center)

    def write_content(self, content: DisplayContent) -> None:
        """Write DisplayContent to the screen."""
        self.write_all(content.lines)
        if content.scroll_line is not None and content.scroll_text:
            self.start_scroll(content.scroll_line, content.scroll_text)

    def start_scroll(self, line: int, text: str, interval: float = 0.3) -> None:
        """
        Start scrolling text on a specific line.

        Args:
            line: Line number (0-3)
            text: Full text to scroll
            interval: Time between scroll steps
        """
        self.stop_scroll()

        if len(text) <= self.cols:
            self.write_line(line, text)
            return

        self._scroll_stop = threading.Event()
        padded = text + "   " + text  # Seamless loop

        def scroll_loop():
            offset = 0
            while not self._scroll_stop.is_set():
                visible = padded[offset:offset + self.cols]
                self.write_line(line, visible)
                offset = (offset + 1) % (len(text) + 3)
                self._scroll_stop.wait(interval)

        self._scroll_thread = threading.Thread(target=scroll_loop, daemon=True)
        self._scroll_thread.start()

    def stop_scroll(self) -> None:
        """Stop any active scrolling."""
        if self._scroll_thread and self._scroll_thread.is_alive():
            self._scroll_stop.set()
            self._scroll_thread.join(timeout=1)
            self._scroll_thread = None

    def show_message(self, message: str, duration: float = 2.0) -> None:
        """Show a centered message briefly."""
        self.clear()
        lines = message.split('\n')[:self.rows]
        # Center vertically
        start_line = (self.rows - len(lines)) // 2
        for i, line in enumerate(lines):
            self.write_line(start_line + i, line, center=True)
        if duration > 0:
            time.sleep(duration)

    def shutdown(self) -> None:
        """Clean shutdown."""
        self.stop_scroll()
        if self._available:
            try:
                self.clear()
                self.write_line(1, "System Shutdown", center=True)
            except Exception:
                pass


class DisplayManager:
    """Manages both LCD displays."""

    def __init__(self):
        self.trading = LCDDisplay(LCD_TRADING_ADDR, "Trading")
        self.research = LCDDisplay(LCD_RESEARCH_ADDR, "Research")
        self._update_thread: Optional[threading.Thread] = None
        self._update_stop = threading.Event()

    @property
    def trading_available(self) -> bool:
        return self.trading.available

    @property
    def research_available(self) -> bool:
        return self.research.available

    def show_startup(self) -> None:
        """Show startup message on available displays."""
        if self.trading.available:
            self.trading.clear()
            self.trading.write_all([
                "=== TRADING SYS ===",
                "",
                "   Initializing...",
                ""
            ])

        if self.research.available:
            self.research.clear()
            self.research.write_all([
                "=== RESEARCH ENG ===",
                "",
                "   Initializing...",
                ""
            ])

    def update_trading(self, data: Dict[str, Any]) -> None:
        """
        Update trading display with current data.

        Expected data keys:
            - portfolio_value: float
            - daily_pnl: float
            - daily_pnl_pct: float
            - position_count: int
            - cash_pct: float
            - spy_price: float
            - vix: float
        """
        if not self.trading.available:
            return

        pv = data.get('portfolio_value', 0)
        pnl = data.get('daily_pnl', 0)
        pnl_pct = data.get('daily_pnl_pct', 0)
        positions = data.get('position_count', 0)
        cash = data.get('cash_pct', 0)
        spy = data.get('spy_price', 0)
        vix = data.get('vix', 0)

        pnl_sign = '+' if pnl >= 0 else ''

        lines = [
            f"${pv:,.0f}  {pnl_sign}{pnl_pct:.2f}%",
            f"Today:{pnl_sign}${pnl:,.0f}",
            f"Pos:{positions}  Cash:{cash:.0f}%",
            f"SPY:{spy:.2f}  VIX:{vix:.1f}"
        ]
        self.trading.write_all(lines)

    def update_research(self, data: Dict[str, Any]) -> None:
        """
        Update research display with current data.

        Expected data keys:
            - status: str ('EVOLVING', 'IDLE', 'COMPLETE')
            - generation: int
            - max_generation: int
            - best_sharpe: float
            - best_drawdown: float
            - eta_minutes: int
            - population: int
        """
        if not self.research.available:
            return

        status = data.get('status', 'IDLE')
        gen = data.get('generation', 0)
        max_gen = data.get('max_generation', 100)
        sharpe = data.get('best_sharpe', 0)
        dd = data.get('best_drawdown', 0)
        eta = data.get('eta_minutes', 0)
        pop = data.get('population', 0)

        # Progress bar
        progress = int((gen / max_gen) * 16) if max_gen > 0 else 0
        bar = '#' * progress + '.' * (16 - progress)

        eta_str = f"{eta // 60}h{eta % 60:02d}m" if eta >= 60 else f"{eta}m"

        lines = [
            f"{status} Gen {gen}/{max_gen}",
            f"[{bar}]",
            f"SR:{sharpe:.2f} DD:{dd:.0f}%",
            f"ETA:{eta_str}  Pop:{pop}"
        ]
        self.research.write_all(lines)

    def shutdown(self) -> None:
        """Clean shutdown of all displays."""
        self._update_stop.set()
        if self._update_thread:
            self._update_thread.join(timeout=2)
        self.trading.shutdown()
        self.research.shutdown()


# Singleton instance
_display_manager: Optional[DisplayManager] = None


def get_display_manager() -> DisplayManager:
    """Get the singleton display manager instance."""
    global _display_manager
    if _display_manager is None:
        _display_manager = DisplayManager()
    return _display_manager
