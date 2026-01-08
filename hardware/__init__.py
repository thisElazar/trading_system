"""
Trading System Hardware Interface Module

Provides control for physical hardware components:
- 2x 20x4 I2C LCD displays (trading & research status)
- 3x RGB LEDs (system health, trading status, research status)
- 2x Rotary encoders with push buttons (UI navigation)

Usage:
    from hardware import HardwareManager

    # Initialize all hardware
    hw = HardwareManager()
    hw.startup()

    # Update displays
    hw.displays.update_trading({
        'portfolio_value': 100000,
        'daily_pnl': 500,
        'daily_pnl_pct': 0.5,
        'position_count': 4,
        'cash_pct': 25,
        'spy_price': 602.14,
        'vix': 14.2
    })

    # Set LED status
    hw.leds.set_status('system', 'healthy')
    hw.leds.set_status('trading', 'active')

    # Register encoder callbacks
    hw.encoders.on_click('trading', lambda e, ev, pos: print(f"Clicked!"))

    # Clean shutdown
    hw.shutdown()
"""

from .gpio_config import (
    GPIO_CHIP,
    I2C_BUS,
    LCD_TRADING_ADDR,
    LCD_RESEARCH_ADDR,
    LED_PINS,
    ENCODER_PINS,
    COLORS,
    STATUS_COLORS,
)

from .leds import LEDController, get_led_controller
from .display import DisplayManager, LCDDisplay, get_display_manager
from .encoders import EncoderHandler, EncoderEvent, get_encoder_handler


class HardwareManager:
    """
    Unified manager for all hardware components.

    Provides a single interface to control LEDs, displays, and encoders.
    """

    def __init__(self, auto_start: bool = False):
        """
        Initialize hardware manager.

        Args:
            auto_start: If True, automatically start encoder polling
        """
        self.leds = get_led_controller()
        self.displays = get_display_manager()
        self.encoders = get_encoder_handler()

        self._started = False
        if auto_start:
            self.startup()

    @property
    def trading_display(self) -> LCDDisplay:
        """Get the trading display."""
        return self.displays.trading

    @property
    def research_display(self) -> LCDDisplay:
        """Get the research display."""
        return self.displays.research

    def startup(self) -> None:
        """
        Initialize and start all hardware.

        Runs startup sequences and begins encoder polling.
        """
        if self._started:
            return

        # Show startup on displays
        self.displays.show_startup()

        # LED startup sequence
        self.leds.startup_sequence()

        # Start encoder polling
        self.encoders.start()

        # Set initial LED states
        self.leds.set_status('system', 'healthy')
        self.leds.set_status('trading', 'idle')
        self.leds.set_status('research', 'offline')

        self._started = True

    def shutdown(self) -> None:
        """Clean shutdown of all hardware."""
        self.encoders.stop()
        self.leds.shutdown()
        self.displays.shutdown()
        self._started = False

    def set_system_status(self, status: str) -> None:
        """Set system health LED status."""
        self.leds.set_status('system', status)

    def set_trading_status(self, status: str) -> None:
        """Set trading LED status."""
        self.leds.set_status('trading', status)

    def set_research_status(self, status: str) -> None:
        """Set research LED status."""
        self.leds.set_status('research', status)

    def flash_alert(self, color: str = 'red', times: int = 3) -> None:
        """Flash all LEDs as an alert."""
        self.leds.flash_all(color, times)

    def update_trading_display(self, data: dict) -> None:
        """Update trading display with portfolio data."""
        self.displays.update_trading(data)

    def update_research_display(self, data: dict) -> None:
        """Update research display with evolution data."""
        self.displays.update_research(data)

    def __enter__(self):
        self.startup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


# Convenience function
def get_hardware_manager(auto_start: bool = True) -> HardwareManager:
    """Get a hardware manager instance."""
    return HardwareManager(auto_start=auto_start)


__all__ = [
    # Main manager
    'HardwareManager',
    'get_hardware_manager',

    # Individual components
    'LEDController',
    'get_led_controller',
    'DisplayManager',
    'LCDDisplay',
    'get_display_manager',
    'EncoderHandler',
    'EncoderEvent',
    'get_encoder_handler',

    # Config
    'GPIO_CHIP',
    'LED_PINS',
    'ENCODER_PINS',
    'COLORS',
    'STATUS_COLORS',
]
