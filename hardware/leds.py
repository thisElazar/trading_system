"""
LED Controller for Trading System Status Indicators

Controls 3 RGB LEDs for system status:
- LED 1 (System): Overall system health
- LED 2 (Trading): Trading engine status
- LED 3 (Research): Research/evolution status

Uses gpiod for Pi 5 compatibility.
"""

import subprocess
import sys
import threading
import time
import math
from typing import Optional, Dict, List

# Ensure lgpio from system packages is available (not in venv)
sys.path.insert(0, '/usr/lib/python3/dist-packages')

try:
    import lgpio
    LGPIO_AVAILABLE = True
except ImportError:
    LGPIO_AVAILABLE = False
    lgpio = None

from .gpio_config import (
    GPIO_CHIP, LED_PINS, ALL_LED_PINS, COLORS, STATUS_COLORS
)


class LEDController:
    """Controls RGB status LEDs via GPIO."""

    def __init__(self):
        self._lock = threading.Lock()
        self._current_proc: Optional[subprocess.Popen] = None
        self._blink_threads: Dict[str, threading.Event] = {}
        self._breathe_threads: Dict[str, threading.Event] = {}
        self._breathe_pins: set = set()  # Pins currently controlled by lgpio breathing
        self._current_colors: Dict[str, str] = {
            'system': 'off',
            'trading': 'off',
            'research': 'off',
        }

    def _set_pins(self, pins_on: List[int]) -> None:
        """Set GPIO pins. Pins in list are LOW (on), others HIGH (off).

        Excludes pins that are being controlled by breathing threads.
        """
        with self._lock:
            if self._current_proc:
                self._current_proc.terminate()
                try:
                    self._current_proc.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    self._current_proc.kill()

            # Exclude pins controlled by breathing (lgpio handles those)
            controlled_pins = [p for p in ALL_LED_PINS if p not in self._breathe_pins]
            if not controlled_pins:
                self._current_proc = None
                return

            settings = [f'{p}={0 if p in pins_on else 1}' for p in controlled_pins]
            self._current_proc = subprocess.Popen(
                ['gpioset', '-c', GPIO_CHIP] + settings,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

    def _get_pins_for_led(self, led: str, color: str) -> List[int]:
        """Get list of pins to turn on for a given LED and color."""
        if led not in LED_PINS:
            return []
        if color not in COLORS:
            return []

        led_config = LED_PINS[led]
        color_channels = COLORS[color]
        return [led_config[ch] for ch in color_channels if ch in led_config]

    def _compute_all_pins_on(self) -> List[int]:
        """Compute all pins that should be on based on current colors."""
        pins_on = []
        for led, color in self._current_colors.items():
            pins_on.extend(self._get_pins_for_led(led, color))
        return pins_on

    def set_color(self, led: str, color: str) -> None:
        """
        Set an LED to a specific color.

        Args:
            led: 'system', 'trading', or 'research'
            color: 'off', 'red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'white'
        """
        # Stop any blinking/breathing for this LED
        self._stop_blink(led)
        self._stop_breathe(led)

        self._current_colors[led] = color
        self._set_pins(self._compute_all_pins_on())

    def set_status(self, led: str, status: str) -> None:
        """
        Set an LED based on status name.

        Args:
            led: 'system', 'trading', or 'research'
            status: Status name from STATUS_COLORS (e.g., 'healthy', 'active', 'evolving')
        """
        color = STATUS_COLORS.get(status, 'off')
        self.set_color(led, color)

    def set_all(self, color: str) -> None:
        """Set all LEDs to the same color."""
        for led in ['system', 'trading', 'research']:
            self._current_colors[led] = color
        self._stop_all_blinks()
        self._set_pins(self._compute_all_pins_on())

    def all_off(self) -> None:
        """Turn off all LEDs."""
        self.set_all('off')

    def all_on(self, color: str = 'white') -> None:
        """Turn on all LEDs to specified color."""
        self.set_all(color)

    def _stop_blink(self, led: str) -> None:
        """Stop blinking for a specific LED."""
        if led in self._blink_threads:
            self._blink_threads[led].set()
            del self._blink_threads[led]

    def _stop_breathe(self, led: str) -> None:
        """Stop breathing for a specific LED."""
        if led in self._breathe_threads:
            self._breathe_threads[led].set()
            del self._breathe_threads[led]

    def _stop_all_blinks(self) -> None:
        """Stop all blinking and breathing."""
        for event in self._blink_threads.values():
            event.set()
        self._blink_threads.clear()
        for event in self._breathe_threads.values():
            event.set()
        self._breathe_threads.clear()

    def blink(self, led: str, color: str, interval: float = 0.5, count: int = 0) -> None:
        """
        Blink an LED.

        Args:
            led: 'system', 'trading', or 'research'
            color: Color to blink
            interval: Time between on/off states in seconds
            count: Number of blinks (0 = indefinite until stopped)
        """
        self._stop_blink(led)

        stop_event = threading.Event()
        self._blink_threads[led] = stop_event

        def blink_loop():
            blinks = 0
            while not stop_event.is_set():
                self._current_colors[led] = color
                self._set_pins(self._compute_all_pins_on())
                if stop_event.wait(interval):
                    break

                self._current_colors[led] = 'off'
                self._set_pins(self._compute_all_pins_on())
                if stop_event.wait(interval):
                    break

                blinks += 1
                if count > 0 and blinks >= count:
                    break

            # Restore to off after blinking
            if led in self._blink_threads:
                del self._blink_threads[led]

        thread = threading.Thread(target=blink_loop, daemon=True)
        thread.start()

    def breathe(self, led: str, color: str = 'blue', period: float = 3.0, min_brightness: float = 0.2) -> None:
        """
        Create a smooth breathing/pulsing effect using PWM.

        Uses lgpio for hardware PWM control on supported pins (GPIO 12).
        Falls back to color cycling for other pins.

        Args:
            led: 'system', 'trading', or 'research'
            color: Color to breathe (uses first color channel for PWM)
            period: Time for one full breath cycle (seconds)
            min_brightness: Minimum brightness level (0.0-1.0), default 0.2
        """
        self._stop_blink(led)
        self._stop_breathe(led)

        stop_event = threading.Event()
        self._breathe_threads[led] = stop_event

        # Get the pin for the requested color channel
        pins = self._get_pins_for_led(led, color)
        if not pins:
            return

        pin = pins[0]

        # Use PWM breathing if lgpio is available
        if LGPIO_AVAILABLE:
            # Register pin as breathing-controlled and restart gpioset to release it
            self._breathe_pins.add(pin)
            self._set_pins(self._compute_all_pins_on())

            def breathe_loop():
                h = None
                try:
                    h = lgpio.gpiochip_open(0)
                    lgpio.gpio_claim_output(h, pin)

                    start = time.time()
                    while not stop_event.is_set():
                        t = time.time() - start
                        # Sine wave for smooth breathing (0 to 1)
                        raw = (math.sin(t * (2 * math.pi / period)) + 1) / 2
                        # Scale to min_brightness -> 1.0 range
                        brightness = min_brightness + raw * (1 - min_brightness)

                        # Software PWM - rapid toggle
                        if brightness <= 0.01:
                            lgpio.gpio_write(h, pin, 1)  # off
                            if stop_event.wait(0.01):
                                break
                        elif brightness >= 0.99:
                            lgpio.gpio_write(h, pin, 0)  # full on
                            if stop_event.wait(0.01):
                                break
                        else:
                            # PWM cycle
                            on_time = 0.001 * brightness
                            off_time = 0.001 * (1 - brightness)
                            lgpio.gpio_write(h, pin, 0)  # on
                            time.sleep(on_time)
                            lgpio.gpio_write(h, pin, 1)  # off
                            time.sleep(off_time)

                    # Cleanup - turn off
                    lgpio.gpio_write(h, pin, 1)
                except Exception as e:
                    print(f"[LEDs] Breathe error: {e}")
                finally:
                    if h is not None:
                        try:
                            lgpio.gpiochip_close(h)
                        except:
                            pass
                    # Unregister pin and let gpioset take over again
                    self._breathe_pins.discard(pin)
                    if led in self._breathe_threads:
                        del self._breathe_threads[led]
                    # Restart gpioset to reclaim this pin
                    self._set_pins(self._compute_all_pins_on())

            thread = threading.Thread(target=breathe_loop, daemon=True)
            thread.start()
        else:
            # Fallback to simple blink if no lgpio
            self.blink(led, color, interval=period/2)

    def flash_all(self, color: str = 'white', times: int = 3, interval: float = 0.1) -> None:
        """Flash all LEDs synchronously (blocking)."""
        self._stop_all_blinks()
        for _ in range(times):
            self.set_all(color)
            time.sleep(interval)
            self.all_off()
            time.sleep(interval)

    def startup_sequence(self) -> None:
        """Run a startup LED sequence to confirm hardware is working."""
        # Quick color cycle
        for color in ['red', 'green', 'blue', 'white']:
            self.set_all(color)
            time.sleep(0.2)
        self.all_off()
        time.sleep(0.1)
        # Flash green for success
        self.flash_all('green', times=2, interval=0.15)

    def shutdown(self) -> None:
        """Clean shutdown - stop all blinks and turn off LEDs."""
        self._stop_all_blinks()
        self.all_off()
        if self._current_proc:
            self._current_proc.terminate()
            self._current_proc = None

    def __del__(self):
        self.shutdown()


# Singleton instance for easy import
_led_controller: Optional[LEDController] = None


def get_led_controller() -> LEDController:
    """Get the singleton LED controller instance."""
    global _led_controller
    if _led_controller is None:
        _led_controller = LEDController()
    return _led_controller
