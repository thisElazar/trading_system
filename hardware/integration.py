"""
Hardware Integration for Trading System Orchestrator

Provides simple hooks to update hardware status based on system state.
Import this module in daily_orchestrator.py to enable hardware feedback.

Usage:
    from hardware.integration import HardwareStatus

    hw = HardwareStatus()
    hw.set_phase('market_open')       # Updates LEDs based on phase
    hw.set_research_active(True)      # Starts breathing on research LED
    hw.set_error('connection failed') # Flash error, set system LED red
    hw.update_display(data)           # Update LCD with trading data
"""

import sys
import logging
import threading
import time
from datetime import datetime
from typing import Optional, Dict, Any

# Ensure lgpio is available
sys.path.insert(0, '/usr/lib/python3/dist-packages')

logger = logging.getLogger(__name__)

# Try to import hardware components
try:
    from .leds import get_led_controller, LEDController
    from .display import LCDDisplay
    from .screen_controller import get_screen_controller, ScreenController
    from .gpio_config import LCD_TRADING_ADDR
    HARDWARE_AVAILABLE = True
except Exception as e:
    logger.warning(f"Hardware module not available: {e}")
    HARDWARE_AVAILABLE = False
    LEDController = None
    LCDDisplay = None
    ScreenController = None


class HardwareStatus:
    """
    High-level interface for updating hardware based on system status.

    Handles graceful degradation if hardware isn't available.
    """

    # Map market phases to LED states
    PHASE_LED_MAP = {
        'pre_market': {'system': 'healthy', 'trading': 'pending'},
        'market_open': {'system': 'healthy', 'trading': 'active'},
        'intraday_open': {'system': 'healthy', 'trading': 'active'},
        'intraday_active': {'system': 'healthy', 'trading': 'active'},
        'post_market': {'system': 'healthy', 'trading': 'idle'},
        'evening': {'system': 'healthy', 'trading': 'idle'},
        'overnight': {'system': 'healthy', 'trading': 'stopped'},
        'weekend': {'system': 'healthy', 'trading': 'stopped'},
    }

    def __init__(self):
        self._leds: Optional[LEDController] = None
        self._screen_controller: Optional[ScreenController] = None
        self._research_active = False
        self._current_phase = None
        self._last_display_data: Dict[str, Any] = {}

        if HARDWARE_AVAILABLE:
            try:
                self._leds = get_led_controller()
                logger.info("LED controller initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize LEDs: {e}")

            try:
                self._screen_controller = get_screen_controller()
                if self._screen_controller.screen_available:
                    logger.info("Screen controller initialized")
                else:
                    logger.warning("Screen not available")
                    self._screen_controller = None
            except Exception as e:
                logger.warning(f"Failed to initialize screen controller: {e}")
                self._screen_controller = None

    @property
    def available(self) -> bool:
        """Check if hardware is available."""
        return self._leds is not None

    def startup(self) -> None:
        """Run hardware startup sequence."""
        logger.info("Running hardware startup sequence...")

        if self._leds:
            try:
                self._leds.startup_sequence()
                self._leds.set_status('system', 'healthy')
                self._leds.set_status('trading', 'idle')
                self._leds.set_status('research', 'offline')
                logger.info("LED startup complete")
            except Exception as e:
                logger.error(f"LED startup failed: {e}")

        if self._screen_controller:
            try:
                if not self._screen_controller.screen_available:
                    logger.warning("Screen not available - skipping screen startup")
                    return

                # Reset/reinitialize screen on startup to ensure clean state
                # This fixes LED issues that occur without a manual reset
                logger.info("Reinitializing screen for clean startup...")
                self._screen_controller._screen.reinit()
                time.sleep(0.3)

                # Show startup message briefly
                self._screen_controller._screen.clear()
                self._screen_controller._screen.write_all([
                    "==== TradeBot ====",
                    "",
                    "   Starting...",
                    ""
                ])
                time.sleep(1)

                # Start screen controller (enables encoder navigation)
                self._screen_controller.start()
                logger.info("Screen controller started with encoder navigation")
            except Exception as e:
                logger.error(f"Screen controller startup failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())
        else:
            logger.warning("No screen controller available - LCD display disabled")

    def shutdown(self) -> None:
        """Clean hardware shutdown."""
        if self._screen_controller:
            try:
                self._screen_controller.stop()
                self._screen_controller._screen.clear()
                self._screen_controller._screen.write_line(1, "System Shutdown", center=True)
            except Exception:
                pass

        if self._leds:
            try:
                self._leds.shutdown()
            except Exception:
                pass

    def set_phase(self, phase: str) -> None:
        """
        Update LEDs based on market phase.

        Args:
            phase: Phase name (e.g., 'market_open', 'overnight')
        """
        if not self._leds:
            return

        self._current_phase = phase
        led_states = self.PHASE_LED_MAP.get(phase, {})

        try:
            if 'system' in led_states:
                self._leds.set_status('system', led_states['system'])
            if 'trading' in led_states:
                self._leds.set_status('trading', led_states['trading'])
            logger.debug(f"Hardware phase set to: {phase}")
        except Exception as e:
            logger.error(f"Failed to set phase LEDs: {e}")

    def set_research_active(self, active: bool) -> None:
        """
        Set research LED state.

        Args:
            active: True to start breathing, False to turn off
        """
        if not self._leds:
            return

        self._research_active = active

        try:
            if active:
                self._leds.breathe('research', 'blue', period=3.0, min_brightness=0.25)
                logger.info("Research LED breathing started")
            else:
                self._leds.set_status('research', 'offline')
                logger.info("Research LED stopped")
        except Exception as e:
            logger.error(f"Failed to set research LED: {e}")

    def set_research_complete(self) -> None:
        """Set research LED to complete (solid green)."""
        if not self._leds:
            return
        try:
            self._leds.set_status('research', 'complete')
        except Exception as e:
            logger.error(f"Failed to set research complete: {e}")

    def set_error(self, message: str = "") -> None:
        """
        Indicate an error condition.

        Flashes all LEDs red and sets system LED to error state.
        """
        if not self._leds:
            return
        try:
            self._leds.flash_all('red', times=3, interval=0.15)
            self._leds.set_status('system', 'error')
            logger.warning(f"Hardware error indicated: {message}")
        except Exception as e:
            logger.error(f"Failed to indicate error: {e}")

    def set_warning(self) -> None:
        """Set system LED to warning state."""
        if not self._leds:
            return
        try:
            self._leds.set_status('system', 'warning')
        except Exception as e:
            logger.error(f"Failed to set warning: {e}")

    def clear_error(self) -> None:
        """Clear error state and restore normal LEDs."""
        if not self._leds:
            return
        try:
            self._leds.set_status('system', 'healthy')
            # Restore phase-appropriate trading LED
            if self._current_phase:
                self.set_phase(self._current_phase)
        except Exception as e:
            logger.error(f"Failed to clear error: {e}")

    def update_display(self, data: Dict[str, Any]) -> None:
        """
        Update display data. Screen controller handles rendering.

        Args:
            data: Dict with keys like portfolio_value, daily_pnl, phase,
                  research_status, research_generation, etc.
        """
        self._last_display_data.update(data)

        # Update screen controller with new data
        if self._screen_controller:
            try:
                self._screen_controller.update_data(data)
            except Exception as e:
                logger.debug(f"Display update error: {e}")


# Singleton instance
_hardware_status: Optional[HardwareStatus] = None


def get_hardware_status() -> HardwareStatus:
    """Get the singleton hardware status instance."""
    global _hardware_status
    if _hardware_status is None:
        _hardware_status = HardwareStatus()
    return _hardware_status
