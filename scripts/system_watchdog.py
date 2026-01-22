#!/usr/bin/env python3
"""
System Watchdog for Trading System
==================================
Monitors system health and triggers restart if frozen for extended period.

Two-layer protection:
1. Software watchdog - 15 minute tolerance, graceful restart
2. Hardware watchdog - 60 second backup for kernel freezes

Health checks:
- Memory usage < 98% (raised from 95% for research workloads)
- Swap usage < 80% (new - critical indicator of memory exhaustion)
- Disk I/O working
- Orchestrator process responsive
- System load reasonable for Pi 5

Critical failure triggers restart only when:
- BOTH memory AND swap are over threshold (memory pressure is real)
- OR disk I/O fails

This prevents false restarts during research when memory spikes
but swap is still available.

Usage:
    Run as systemd service (system-watchdog.service)
"""

import os
import sys
import time
import logging
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple

# Configuration
CHECK_INTERVAL = 30          # Seconds between health checks
UNHEALTHY_THRESHOLD = 900    # Seconds (15 minutes) before triggering restart
MAX_MEMORY_PCT = 98          # Memory usage threshold (raised for research workloads)
MAX_SWAP_PCT = 80            # Swap usage threshold - critical if swap is exhausted
MAX_LOAD_AVG = 8.0           # Load average threshold (Pi 5 has 4 cores)
WATCHDOG_TIMEOUT = 60        # Hardware watchdog timeout (seconds)

# Paths
STATE_FILE = Path("/tmp/watchdog_state.json")
WATCHDOG_DEVICE = "/dev/watchdog"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers with immediate flush
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

file_handler = logging.FileHandler('/home/thiselazar/trading_system/logs/watchdog.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

logger.addHandler(stream_handler)
logger.addHandler(file_handler)

# Force unbuffered output for systemd
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


class SystemWatchdog:
    """Monitors system health and manages hardware watchdog."""

    def __init__(self):
        self.unhealthy_since: Optional[datetime] = None
        self.watchdog_fd = None
        self.pet_hardware_watchdog = True

    def open_hardware_watchdog(self) -> bool:
        """Open hardware watchdog device."""
        try:
            # Opening the watchdog starts it - must pet regularly or system reboots
            self.watchdog_fd = os.open(WATCHDOG_DEVICE, os.O_WRONLY)
            logger.info(f"Hardware watchdog opened (timeout: {WATCHDOG_TIMEOUT}s)")
            return True
        except PermissionError:
            logger.warning("No permission to open watchdog device (run as root)")
            return False
        except OSError as e:
            if e.errno == 16:  # Device busy - kernel is managing it
                logger.info("Hardware watchdog managed by kernel (this is fine)")
            else:
                logger.warning(f"Could not open hardware watchdog: {e}")
            return False
        except Exception as e:
            logger.warning(f"Could not open hardware watchdog: {e}")
            return False

    def pet_watchdog(self):
        """Pet the hardware watchdog to prevent reboot."""
        if self.watchdog_fd is not None:
            try:
                os.write(self.watchdog_fd, b'1')
            except Exception as e:
                logger.error(f"Failed to pet watchdog: {e}")

    def close_watchdog_safely(self):
        """Close watchdog with magic close to disable it."""
        if self.watchdog_fd is not None:
            try:
                # Write 'V' to disable watchdog on close (if supported)
                os.write(self.watchdog_fd, b'V')
                os.close(self.watchdog_fd)
                logger.info("Hardware watchdog disabled and closed")
            except Exception as e:
                logger.error(f"Error closing watchdog: {e}")

    def check_memory(self) -> Tuple[bool, float, dict]:
        """Check memory usage with detailed breakdown."""
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        meminfo[parts[0].rstrip(':')] = int(parts[1])

            total = meminfo.get('MemTotal', 1)
            available = meminfo.get('MemAvailable', 0)
            used_pct = ((total - available) / total) * 100

            # Detailed breakdown for logging
            details = {
                'total_mb': total // 1024,
                'available_mb': available // 1024,
                'used_pct': used_pct,
                'buffers_mb': meminfo.get('Buffers', 0) // 1024,
                'cached_mb': meminfo.get('Cached', 0) // 1024,
            }

            healthy = used_pct < MAX_MEMORY_PCT
            return healthy, used_pct, details
        except Exception as e:
            logger.error(f"Memory check failed: {e}")
            return False, 100.0, {}

    def check_swap(self) -> Tuple[bool, float, dict]:
        """Check swap usage - critical indicator of memory pressure."""
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        meminfo[parts[0].rstrip(':')] = int(parts[1])

            swap_total = meminfo.get('SwapTotal', 0)
            swap_free = meminfo.get('SwapFree', 0)

            if swap_total == 0:
                # No swap configured - not a failure condition
                return True, 0.0, {'total_mb': 0, 'used_mb': 0, 'used_pct': 0}

            swap_used = swap_total - swap_free
            swap_used_pct = (swap_used / swap_total) * 100

            details = {
                'total_mb': swap_total // 1024,
                'used_mb': swap_used // 1024,
                'free_mb': swap_free // 1024,
                'used_pct': swap_used_pct,
            }

            # Swap usage above threshold indicates severe memory pressure
            healthy = swap_used_pct < MAX_SWAP_PCT
            return healthy, swap_used_pct, details
        except Exception as e:
            logger.error(f"Swap check failed: {e}")
            return False, 100.0, {}

    def check_load(self) -> Tuple[bool, float]:
        """Check system load average."""
        try:
            load_1, load_5, load_15 = os.getloadavg()
            # Use 5-minute average for stability
            healthy = load_5 < MAX_LOAD_AVG
            return healthy, load_5
        except Exception as e:
            logger.error(f"Load check failed: {e}")
            return False, 99.0

    def check_orchestrator(self) -> Tuple[bool, str]:
        """Check if orchestrator process is running and responsive."""
        try:
            # Check if process exists
            result = subprocess.run(
                ['pgrep', '-f', 'daily_orchestrator'],
                capture_output=True,
                timeout=10
            )

            if result.returncode != 0:
                return False, "not running"

            pid = result.stdout.decode().strip().split('\n')[0]

            # Check if process is responsive (not in D state)
            with open(f'/proc/{pid}/stat', 'r') as f:
                stat = f.read().split()
                state = stat[2] if len(stat) > 2 else '?'

            if state == 'D':
                return False, f"uninterruptible sleep (pid {pid})"

            return True, f"running (pid {pid}, state {state})"

        except subprocess.TimeoutExpired:
            return False, "pgrep timed out"
        except Exception as e:
            return False, f"check failed: {e}"

    def check_disk_io(self) -> Tuple[bool, str]:
        """Check if disk I/O is working."""
        try:
            test_file = Path("/tmp/watchdog_io_test")
            start = time.time()

            # Write test
            test_file.write_text(f"test {datetime.now()}")

            # Read test
            _ = test_file.read_text()

            # Cleanup
            test_file.unlink()

            elapsed = time.time() - start
            if elapsed > 5.0:
                return False, f"slow ({elapsed:.1f}s)"

            return True, f"ok ({elapsed:.2f}s)"

        except Exception as e:
            return False, f"failed: {e}"

    def run_health_checks(self) -> Tuple[bool, dict]:
        """Run all health checks and return overall status."""
        results = {}

        # Memory check (with detailed breakdown)
        mem_ok, mem_pct, mem_details = self.check_memory()
        results['memory'] = {
            'healthy': mem_ok,
            'value': f"{mem_pct:.1f}%",
            'details': mem_details
        }

        # Swap check (critical indicator of memory pressure)
        swap_ok, swap_pct, swap_details = self.check_swap()
        results['swap'] = {
            'healthy': swap_ok,
            'value': f"{swap_pct:.1f}%",
            'details': swap_details
        }

        # Load check
        load_ok, load_avg = self.check_load()
        results['load'] = {'healthy': load_ok, 'value': f"{load_avg:.2f}"}

        # Orchestrator check
        orch_ok, orch_status = self.check_orchestrator()
        results['orchestrator'] = {'healthy': orch_ok, 'value': orch_status}

        # Disk I/O check
        io_ok, io_status = self.check_disk_io()
        results['disk_io'] = {'healthy': io_ok, 'value': io_status}

        # Overall health - critical checks only
        # Memory alone isn't critical if swap has headroom
        # Critical failure = (memory high AND swap high) OR disk I/O failure
        memory_critical = not mem_ok and not swap_ok  # Both memory AND swap stressed
        overall_healthy = not memory_critical and io_ok

        # Log detailed memory state when under pressure
        if not mem_ok or not swap_ok:
            logger.warning(
                f"Memory pressure detected: RAM {mem_pct:.1f}% "
                f"({mem_details.get('available_mb', 0)}MB free), "
                f"Swap {swap_pct:.1f}% ({swap_details.get('free_mb', 0)}MB free)"
            )

        return overall_healthy, results

    def trigger_restart(self, reason: str):
        """Trigger system restart."""
        logger.critical(f"TRIGGERING RESTART: {reason}")

        try:
            # Log to file that will survive reboot
            restart_log = Path("/home/thiselazar/trading_system/logs/watchdog_restarts.log")
            with open(restart_log, 'a') as f:
                f.write(f"{datetime.now().isoformat()} - Restart triggered: {reason}\n")
        except:
            pass

        try:
            # Try graceful shutdown first
            logger.info("Attempting graceful shutdown...")
            subprocess.run(['sudo', 'systemctl', 'stop', 'trading-orchestrator'], timeout=30)
        except:
            pass

        # Reboot
        logger.info("Initiating system reboot...")
        subprocess.run(['sudo', 'reboot'])

    def run(self):
        """Main watchdog loop."""
        logger.info("=" * 60)
        logger.info("System watchdog starting...")
        logger.info(f"Configuration:")
        logger.info(f"  - Check interval: {CHECK_INTERVAL}s")
        logger.info(f"  - Unhealthy threshold: {UNHEALTHY_THRESHOLD}s ({UNHEALTHY_THRESHOLD/60:.0f} minutes)")
        logger.info(f"  - Max memory: {MAX_MEMORY_PCT}%")
        logger.info(f"  - Max swap: {MAX_SWAP_PCT}%")
        logger.info(f"  - Max load avg: {MAX_LOAD_AVG}")
        logger.info("=" * 60)

        # Try to open hardware watchdog
        hw_watchdog_enabled = self.open_hardware_watchdog()
        check_count = 0
        INFO_LOG_INTERVAL = 10  # Log INFO every 10 checks (5 minutes at 30s interval)
        DETAIL_LOG_INTERVAL = 60  # Log detailed memory info every 30 minutes

        try:
            while True:
                healthy, results = self.run_health_checks()
                check_count += 1

                # Build status message
                status_parts = []
                for check, data in results.items():
                    status = "OK" if data['healthy'] else "FAIL"
                    status_parts.append(f"{check}:{status}({data['value']})")

                if healthy:
                    if self.unhealthy_since is not None:
                        duration = datetime.now() - self.unhealthy_since
                        logger.info(f"System RECOVERED after {duration.seconds}s unhealthy")
                        # Log detailed state on recovery
                        self._log_detailed_state(results)
                    self.unhealthy_since = None

                    # Pet hardware watchdog
                    if hw_watchdog_enabled:
                        self.pet_watchdog()

                    # Log INFO periodically when healthy (including first check)
                    if check_count == 1 or check_count % INFO_LOG_INTERVAL == 0:
                        logger.info(f"Health OK (check #{check_count}): {' | '.join(status_parts)}")

                    # Log detailed memory/swap state periodically
                    if check_count == 1 or check_count % DETAIL_LOG_INTERVAL == 0:
                        self._log_detailed_state(results)

                    # Flush file handler
                    for handler in logger.handlers:
                        handler.flush()
                else:
                    now = datetime.now()
                    if self.unhealthy_since is None:
                        self.unhealthy_since = now
                        logger.warning(f"System UNHEALTHY: {' | '.join(status_parts)}")
                        # Log detailed state when first becoming unhealthy
                        self._log_detailed_state(results)

                    unhealthy_duration = (now - self.unhealthy_since).total_seconds()
                    remaining = UNHEALTHY_THRESHOLD - unhealthy_duration
                    logger.warning(
                        f"Unhealthy for {unhealthy_duration:.0f}s / {UNHEALTHY_THRESHOLD}s "
                        f"({remaining:.0f}s until restart): {' | '.join(status_parts)}"
                    )

                    # Still pet hardware watchdog to prevent premature reboot
                    # Only stop petting if we're about to intentionally restart
                    if unhealthy_duration < UNHEALTHY_THRESHOLD - 30:
                        if hw_watchdog_enabled:
                            self.pet_watchdog()

                    if unhealthy_duration >= UNHEALTHY_THRESHOLD:
                        failed_checks = [k for k, v in results.items() if not v['healthy']]
                        # Log comprehensive state before restart
                        logger.critical("PRE-RESTART STATE DUMP:")
                        self._log_detailed_state(results)
                        self.trigger_restart(
                            f"Unhealthy for {unhealthy_duration:.0f}s: {failed_checks} | "
                            f"mem={results['memory']['value']} swap={results['swap']['value']}"
                        )

                    # Flush on every unhealthy check
                    for handler in logger.handlers:
                        handler.flush()

                time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            logger.info("Watchdog stopped by user")
        finally:
            self.close_watchdog_safely()

    def _log_detailed_state(self, results: dict):
        """Log detailed memory and system state for debugging."""
        mem = results.get('memory', {}).get('details', {})
        swap = results.get('swap', {}).get('details', {})

        logger.info(
            f"Memory detail: {mem.get('used_pct', 0):.1f}% used, "
            f"{mem.get('available_mb', 0)}MB available / {mem.get('total_mb', 0)}MB total, "
            f"buffers={mem.get('buffers_mb', 0)}MB, cached={mem.get('cached_mb', 0)}MB"
        )
        logger.info(
            f"Swap detail: {swap.get('used_pct', 0):.1f}% used, "
            f"{swap.get('used_mb', 0)}MB used / {swap.get('total_mb', 0)}MB total, "
            f"{swap.get('free_mb', 0)}MB free"
        )


def main():
    watchdog = SystemWatchdog()
    watchdog.run()


if __name__ == "__main__":
    main()
