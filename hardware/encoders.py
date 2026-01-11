"""
Rotary Encoder Input Handler for Trading System

Handles rotary encoder with push button for screen navigation.

Uses gpiod for Pi 5 compatibility with polling-based reading.
"""

import subprocess
import threading
import time
from typing import Optional, Callable, Dict, List
from dataclasses import dataclass
from enum import Enum

try:
    import gpiod
    GPIOD_AVAILABLE = True
except ImportError:
    GPIOD_AVAILABLE = False
    gpiod = None

from .gpio_config import GPIO_CHIP, ENCODER_PINS, ALL_ENCODER_PINS


class EncoderEvent(Enum):
    """Types of encoder events."""
    ROTATE_CW = "rotate_cw"      # Clockwise rotation
    ROTATE_CCW = "rotate_ccw"    # Counter-clockwise rotation
    CLICK = "click"              # Button press
    LONG_PRESS = "long_press"    # Button held > 1 second


@dataclass
class EncoderState:
    """Current state of an encoder."""
    position: int = 0
    button_pressed: bool = False
    last_event: Optional[EncoderEvent] = None


class EncoderHandler:
    """Handles rotary encoder input with callbacks."""

    def __init__(self, poll_interval: float = 0.005):
        """
        Initialize encoder handler.

        Args:
            poll_interval: How often to poll GPIO state (seconds)
        """
        self.poll_interval = poll_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # State tracking
        self._states: Dict[str, EncoderState] = {
            'main': EncoderState(),
        }
        self._last_gpio: Dict[int, int] = {}
        self._button_press_time: Dict[str, float] = {}

        # Callbacks: Dict[encoder_name, Dict[event_type, List[callbacks]]]
        self._callbacks: Dict[str, Dict[EncoderEvent, List[Callable]]] = {
            'main': {e: [] for e in EncoderEvent},
        }

        # GPIO line request
        self._lines = None
        self._chip = None

    def _init_gpio(self) -> bool:
        """Initialize GPIO lines."""
        if not GPIOD_AVAILABLE:
            print("[Encoders] gpiod library not available, falling back to subprocess")
            return True  # Will use subprocess fallback

        try:
            self._chip = gpiod.Chip(GPIO_CHIP)
            config = gpiod.LineSettings(
                direction=gpiod.line.Direction.INPUT,
                bias=gpiod.line.Bias.PULL_UP
            )
            self._lines = self._chip.request_lines(
                consumer="encoder_handler",
                config={pin: config for pin in ALL_ENCODER_PINS}
            )
            return True
        except Exception as e:
            print(f"[Encoders] Failed to initialize gpiod: {e}, using subprocess fallback")
            self._lines = None
            return True

    def _read_pins_gpiod(self) -> Dict[int, int]:
        """Read all encoder pins using gpiod."""
        if not self._lines:
            return self._read_pins_subprocess()

        try:
            values = self._lines.get_values(ALL_ENCODER_PINS)
            return dict(zip(ALL_ENCODER_PINS, values))
        except Exception:
            return self._read_pins_subprocess()

    def _read_pins_subprocess(self) -> Dict[int, int]:
        """Read all encoder pins using gpioget subprocess."""
        try:
            result = subprocess.run(
                ['gpioget', '-c', GPIO_CHIP, '--bias=pull-up'] +
                [str(p) for p in ALL_ENCODER_PINS],
                capture_output=True, text=True, timeout=1
            )
            states = {}
            for part in result.stdout.strip().split():
                if '=' in part:
                    pin_str, val = part.replace('"', '').split('=')
                    states[int(pin_str)] = 1 if val == 'active' else 0
            return states
        except Exception:
            return {}

    def _read_pins(self) -> Dict[int, int]:
        """Read all encoder pins."""
        if self._lines:
            return self._read_pins_gpiod()
        return self._read_pins_subprocess()

    def _process_encoder(self, name: str, pins: Dict[str, int], gpio_state: Dict[int, int]) -> None:
        """Process state changes for one encoder."""
        clk_pin = pins['clk']
        dt_pin = pins['dt']
        sw_pin = pins['sw']

        clk = gpio_state.get(clk_pin, 1)
        dt = gpio_state.get(dt_pin, 1)
        sw = gpio_state.get(sw_pin, 1)

        last_clk = self._last_gpio.get(clk_pin, 1)
        last_sw = self._last_gpio.get(sw_pin, 1)

        state = self._states[name]

        # Rotation detection (on CLK falling edge)
        if last_clk == 1 and clk == 0:
            if dt == 1:
                state.position += 1
                state.last_event = EncoderEvent.ROTATE_CW
                self._fire_callbacks(name, EncoderEvent.ROTATE_CW, state.position)
            else:
                state.position -= 1
                state.last_event = EncoderEvent.ROTATE_CCW
                self._fire_callbacks(name, EncoderEvent.ROTATE_CCW, state.position)

        # Button detection
        if last_sw == 1 and sw == 0:
            # Button pressed
            self._button_press_time[name] = time.time()
            state.button_pressed = True

        if last_sw == 0 and sw == 1 and state.button_pressed:
            # Button released
            state.button_pressed = False
            press_duration = time.time() - self._button_press_time.get(name, time.time())

            if press_duration > 1.0:
                state.last_event = EncoderEvent.LONG_PRESS
                self._fire_callbacks(name, EncoderEvent.LONG_PRESS, state.position)
            else:
                state.last_event = EncoderEvent.CLICK
                self._fire_callbacks(name, EncoderEvent.CLICK, state.position)

    def _fire_callbacks(self, encoder: str, event: EncoderEvent, position: int) -> None:
        """Fire all registered callbacks for an event."""
        for callback in self._callbacks[encoder][event]:
            try:
                callback(encoder, event, position)
            except Exception as e:
                print(f"[Encoders] Callback error: {e}")

    def _poll_loop(self) -> None:
        """Main polling loop."""
        while self._running:
            gpio_state = self._read_pins()

            if gpio_state:
                for name, pins in ENCODER_PINS.items():
                    self._process_encoder(name, pins, gpio_state)
                self._last_gpio = gpio_state

            time.sleep(self.poll_interval)

    def register_callback(
        self,
        encoder: str,
        event: EncoderEvent,
        callback: Callable[[str, EncoderEvent, int], None]
    ) -> None:
        """
        Register a callback for encoder events.

        Args:
            encoder: 'main' (single encoder)
            event: EncoderEvent type to listen for
            callback: Function(encoder_name, event, position) to call
        """
        if encoder in self._callbacks and event in self._callbacks[encoder]:
            self._callbacks[encoder][event].append(callback)

    def on_rotate(self, encoder: str, callback: Callable[[str, EncoderEvent, int], None]) -> None:
        """Register callback for any rotation (CW or CCW)."""
        self.register_callback(encoder, EncoderEvent.ROTATE_CW, callback)
        self.register_callback(encoder, EncoderEvent.ROTATE_CCW, callback)

    def on_click(self, encoder: str, callback: Callable[[str, EncoderEvent, int], None]) -> None:
        """Register callback for button click."""
        self.register_callback(encoder, EncoderEvent.CLICK, callback)

    def on_long_press(self, encoder: str, callback: Callable[[str, EncoderEvent, int], None]) -> None:
        """Register callback for long button press."""
        self.register_callback(encoder, EncoderEvent.LONG_PRESS, callback)

    def get_position(self, encoder: str) -> int:
        """Get current position of an encoder."""
        return self._states.get(encoder, EncoderState()).position

    def reset_position(self, encoder: str, value: int = 0) -> None:
        """Reset encoder position to a value."""
        if encoder in self._states:
            self._states[encoder].position = value

    def start(self) -> None:
        """Start the encoder polling thread."""
        if self._running:
            return

        self._init_gpio()
        self._running = True
        self._last_gpio = self._read_pins()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the encoder polling thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

        if self._lines:
            try:
                self._lines.release()
            except Exception:
                pass
            self._lines = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# Singleton instance
_encoder_handler: Optional[EncoderHandler] = None


def get_encoder_handler() -> EncoderHandler:
    """Get the singleton encoder handler instance."""
    global _encoder_handler
    if _encoder_handler is None:
        _encoder_handler = EncoderHandler()
    return _encoder_handler
