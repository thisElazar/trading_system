"""
LED Authority System - Single point of control for all LED operations.

Architecture:
- LEDOrchestrator: Runs in main process, owns the hardware, polls for requests
- LEDClient: Lightweight file-based API for other processes to request LED states

Other processes never touch GPIO directly - they write request files that the
orchestrator picks up. This prevents race conditions and ensures clean state
management.

Trading system stability > LED functionality. All LED operations are non-blocking
and failure-tolerant.
"""

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Request directory - using run/ for ephemeral data
REQUEST_DIR = Path(__file__).parent.parent / "run" / "led_requests"
FLASH_REQUEST_FILE = REQUEST_DIR / "flash.req"


class LEDClient:
    """
    Lightweight LED client for non-authority processes.

    Writes request files instead of touching GPIO. Fire-and-forget -
    failures are logged but never block the caller.
    """

    def __init__(self):
        self._ensure_request_dir()

    def _ensure_request_dir(self) -> None:
        """Create request directory if needed."""
        try:
            REQUEST_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.debug(f"Could not create LED request dir: {e}")

    def _write_request(self, led: str, request: Dict[str, Any]) -> bool:
        """Write a request file. Returns True on success."""
        try:
            request_file = REQUEST_DIR / f"{led}.json"
            # Atomic write via temp file
            temp_file = request_file.with_suffix('.tmp')
            temp_file.write_text(json.dumps(request))
            temp_file.rename(request_file)
            return True
        except Exception as e:
            logger.debug(f"LED request write failed ({led}): {e}")
            return False

    def set_color(self, led: str, color: str) -> None:
        """Request LED to be set to a solid color."""
        self._write_request(led, {
            "state": "solid",
            "color": color,
        })

    def set_status(self, led: str, status: str) -> None:
        """Request LED to be set based on status name."""
        # Map status to color (duplicated from gpio_config to avoid import)
        status_colors = {
            'healthy': 'green', 'warning': 'yellow', 'error': 'red',
            'active': 'green', 'idle': 'blue', 'stopped': 'off', 'pending': 'yellow',
            'evolving': 'blue', 'complete': 'green', 'failed': 'red',
            'paused': 'yellow', 'offline': 'off',
        }
        color = status_colors.get(status, 'off')
        self.set_color(led, color)

    def breathe(self, led: str, color: str = 'blue', period: float = 3.0,
                min_brightness: float = 0.2) -> None:
        """Request LED to breathe/pulse."""
        self._write_request(led, {
            "state": "breathing",
            "color": color,
            "period": period,
            "min_brightness": min_brightness,
        })

    def blink(self, led: str, color: str, interval: float = 0.5, count: int = 0) -> None:
        """Request LED to blink."""
        self._write_request(led, {
            "state": "blink",
            "color": color,
            "interval": interval,
            "count": count,
        })

    def flash_all(self, color: str = 'white', times: int = 3, interval: float = 0.1) -> None:
        """Request a flash-all operation."""
        try:
            request = {
                "color": color,
                "times": times,
                "interval": interval,
                "ts": time.time(),  # Timestamp to detect new requests
            }
            temp_file = FLASH_REQUEST_FILE.with_suffix('.tmp')
            temp_file.write_text(json.dumps(request))
            temp_file.rename(FLASH_REQUEST_FILE)
        except Exception as e:
            logger.debug(f"Flash request write failed: {e}")

    def all_off(self) -> None:
        """Request all LEDs off."""
        for led in ['system', 'trading', 'research']:
            self.set_color(led, 'off')

    # No-op methods for API compatibility
    def startup_sequence(self) -> None:
        """No-op - orchestrator handles startup."""
        pass

    def shutdown(self) -> None:
        """No-op - orchestrator handles shutdown."""
        pass


class LEDOrchestrator:
    """
    Single authority for LED hardware control.

    Runs in the main orchestrator process. Polls request files and
    translates them to actual LED operations. Maintains canonical
    state that flash_all can save/restore properly.
    """

    def __init__(self, poll_interval: float = 0.1):
        from .leds import LEDController

        self._controller = LEDController()
        self._poll_interval = poll_interval
        self._running = False
        self._poll_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Track what we've applied to avoid redundant operations
        self._applied_states: Dict[str, Dict] = {}
        self._last_flash_ts: float = 0

        # Ensure request directory exists
        REQUEST_DIR.mkdir(parents=True, exist_ok=True)

        logger.info("LEDOrchestrator initialized as authority")

    @property
    def controller(self) -> 'LEDController':
        """Direct access to controller for local operations."""
        return self._controller

    def start(self) -> None:
        """Start the request polling loop."""
        if self._running:
            return

        self._running = True
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        logger.info("LEDOrchestrator polling started")

    def stop(self) -> None:
        """Stop polling and clean up."""
        self._running = False
        if self._poll_thread:
            self._poll_thread.join(timeout=1.0)
        self._controller.shutdown()
        logger.info("LEDOrchestrator stopped")

    def _poll_loop(self) -> None:
        """Main polling loop - reads request files and applies changes."""
        while self._running:
            try:
                self._process_requests()
            except Exception as e:
                logger.debug(f"LED poll error: {e}")

            time.sleep(self._poll_interval)

    def _process_requests(self) -> None:
        """Process all pending request files."""
        # Check for flash request first (highest priority)
        self._check_flash_request()

        # Process per-LED requests
        for led in ['system', 'trading', 'research']:
            self._process_led_request(led)

    def _check_flash_request(self) -> None:
        """Check and process flash request if new."""
        try:
            if not FLASH_REQUEST_FILE.exists():
                return

            request = json.loads(FLASH_REQUEST_FILE.read_text())
            ts = request.get('ts', 0)

            # Only process if newer than last processed
            if ts > self._last_flash_ts:
                self._last_flash_ts = ts

                # Delete request file before processing (so we don't re-process on slow flash)
                FLASH_REQUEST_FILE.unlink(missing_ok=True)

                # Execute flash
                with self._lock:
                    self._controller.flash_all(
                        color=request.get('color', 'white'),
                        times=request.get('times', 3),
                        interval=request.get('interval', 0.1),
                    )

                    # After flash, reapply current states
                    # (flash_all handles this internally but we ensure it)

        except json.JSONDecodeError:
            # Corrupted file - delete it
            FLASH_REQUEST_FILE.unlink(missing_ok=True)
        except Exception as e:
            logger.debug(f"Flash request error: {e}")

    def _process_led_request(self, led: str) -> None:
        """Process request file for a single LED."""
        request_file = REQUEST_DIR / f"{led}.json"

        try:
            if not request_file.exists():
                return

            request = json.loads(request_file.read_text())

            # Check if state changed
            if request == self._applied_states.get(led):
                return

            # Apply the new state
            state = request.get('state', 'solid')
            color = request.get('color', 'off')

            with self._lock:
                if state == 'solid':
                    self._controller.set_color(led, color)
                elif state == 'breathing':
                    self._controller.breathe(
                        led, color,
                        period=request.get('period', 3.0),
                        min_brightness=request.get('min_brightness', 0.2),
                    )
                elif state == 'blink':
                    self._controller.blink(
                        led, color,
                        interval=request.get('interval', 0.5),
                        count=request.get('count', 0),
                    )
                elif state == 'off':
                    self._controller.set_color(led, 'off')

            self._applied_states[led] = request
            logger.debug(f"LED {led} -> {state}:{color}")

        except json.JSONDecodeError:
            # Corrupted - delete
            request_file.unlink(missing_ok=True)
        except Exception as e:
            logger.debug(f"LED request error ({led}): {e}")

    # Pass-through methods for direct local use
    def set_color(self, led: str, color: str) -> None:
        """Directly set LED color (for local/authority use)."""
        with self._lock:
            self._controller.set_color(led, color)
            self._applied_states[led] = {"state": "solid", "color": color}

    def set_status(self, led: str, status: str) -> None:
        """Directly set LED status (for local/authority use)."""
        with self._lock:
            self._controller.set_status(led, status)

    def breathe(self, led: str, color: str = 'blue', period: float = 3.0,
                min_brightness: float = 0.2) -> None:
        """Start breathing animation (for local/authority use)."""
        with self._lock:
            self._controller.breathe(led, color, period, min_brightness)
            self._applied_states[led] = {
                "state": "breathing", "color": color,
                "period": period, "min_brightness": min_brightness,
            }

    def blink(self, led: str, color: str, interval: float = 0.5, count: int = 0) -> None:
        """Start blinking (for local/authority use)."""
        with self._lock:
            self._controller.blink(led, color, interval, count)

    def flash_all(self, color: str = 'white', times: int = 3, interval: float = 0.1) -> None:
        """Flash all LEDs (for local/authority use)."""
        with self._lock:
            self._controller.flash_all(color, times, interval)

    def startup_sequence(self) -> None:
        """Run startup sequence."""
        with self._lock:
            self._controller.startup_sequence()

    def shutdown(self) -> None:
        """Shutdown LEDs and stop polling."""
        self.stop()


# Module-level state
_orchestrator: Optional[LEDOrchestrator] = None
_is_authority: bool = False


def init_as_authority(poll_interval: float = 0.1) -> LEDOrchestrator:
    """
    Initialize this process as the LED authority.

    Call this once in the main orchestrator process.
    Returns the LEDOrchestrator instance.
    """
    global _orchestrator, _is_authority

    if _orchestrator is not None:
        return _orchestrator

    _orchestrator = LEDOrchestrator(poll_interval)
    _is_authority = True
    _orchestrator.start()

    return _orchestrator


def get_led_interface():
    """
    Get the appropriate LED interface for this process.

    Returns LEDOrchestrator if this is the authority process,
    otherwise returns LEDClient.
    """
    global _orchestrator, _is_authority

    if _is_authority and _orchestrator is not None:
        return _orchestrator

    # Not authority - return lightweight client
    return LEDClient()


def cleanup_request_files() -> None:
    """Clean up stale request files on startup."""
    try:
        for f in REQUEST_DIR.glob("*.json"):
            f.unlink()
        FLASH_REQUEST_FILE.unlink(missing_ok=True)
    except Exception:
        pass
