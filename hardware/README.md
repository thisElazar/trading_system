# Hardware Integration Module

Physical interface for the trading system running on Raspberry Pi 5. Provides real-time status feedback via LEDs, LCD display, and rotary encoder input.

## Hardware Components

| Component | Interface | Purpose |
|-----------|-----------|---------|
| 20x4 LCD (x2) | I2C (0x27, 0x25) | Trading info & research status displays |
| RGB LEDs (x3) | GPIO | System/trading/research status indicators |
| Rotary Encoder | GPIO | Display navigation and control |

---

## Architecture

```
DailyOrchestrator
       |
       v
HardwareStatus (integration.py)
       |
       +---> LEDController (leds.py)
       |         - GPIO control via gpioset subprocess
       |         - Supports static, blinking, breathing modes
       |
       +---> ScreenController (screen_controller.py)
                 - Multi-page display system
                 - Encoder polling for navigation
                 - Renders data from orchestrator
                 |
                 v
           LCDDisplay (display.py)
                 - I2C LCD driver via RPLCD
                 - Custom character support
```

### Design Principles

1. **Complete Isolation** - Hardware failures never affect trading logic
2. **Graceful Degradation** - System runs fine without any hardware connected
3. **Daemon Threads** - All hardware runs in background threads that auto-terminate
4. **Wrapped Exceptions** - Every hardware call is in try/except blocks

---

## Display System

### Pages (Navigate with Encoder Rotation)

| Page | Content |
|------|---------|
| **MARKET** | Phase + time remaining, SPY price with trend arrow, VIX with trend arrow, current time |
| **TRADING** | Portfolio value, daily P&L ($ and %), position count, cash percentage |
| **POSITIONS** | Scrollable list of open positions (symbol, qty, unrealized P&L) |
| **SYSTEM** | RAM %, ZRAM %, CPU %, uptime, current phase |
| **RESEARCH** | Status, progress bar, generation/max, best Sharpe ratio |
| **SCREENSAVER** | Conway's Game of Life (20x8 high-resolution) |

### Encoder Controls

| Action | Function |
|--------|----------|
| Rotate CW | Next page |
| Rotate CCW | Previous page |
| Short click | Scroll positions (on POSITIONS page) |
| Long press (1s) | Toggle backlight on/off |

### Data Flow

The orchestrator pushes data to the display every 30 seconds:

```python
display_data = {
    'phase': 'overnight',
    'portfolio_value': 100000.00,
    'daily_pnl': 1234.56,
    'daily_pnl_pct': 1.23,
    'position_count': 5,
    'positions': [{'symbol': 'AAPL', 'qty': 10, 'unrealized_pnl': 500}, ...],
    'cash_pct': 45.5,
    'spy_price': 590.25,
    'vix': 15.3,
    'memory_pct': 42,
    'zram_pct': 25,
    'cpu_pct': 12,
    'uptime': '2h 30m',
    'phase_time_remaining': '5h 12m',
    'research_status': 'EVOLVING',
    'research_generation': 50,
    'research_max_gen': 100,
    'research_best_sharpe': 1.85,
}
```

### Trend Arrows

SPY and VIX display directional arrows based on price movement:
- `^` - Price increased since last update
- `v` - Price decreased since last update
- `-` - Price unchanged or first reading

### Custom LCD Characters

The display uses custom characters for high-resolution graphics:

| Char Code | Pattern | Use |
|-----------|---------|-----|
| `chr(0)` | Upper half block | Game of Life, progress bars |
| `chr(1)` | Lower half block | Game of Life |
| `chr(2)` | Full block | Game of Life, filled sections |

---

## LED Status System

See [LED_STATUS_CODES.md](LED_STATUS_CODES.md) for complete LED reference.

### Quick Reference

| LED | Green | Yellow | Blue | Red | Off |
|-----|-------|--------|------|-----|-----|
| System | Healthy | Warning | - | Error | Offline |
| Trading | Active | Pre-market | Post-market | - | Overnight |
| Research | Complete | Paused | Breathing=Active | Failed | Idle |

---

## File Reference

| File | Purpose |
|------|---------|
| `__init__.py` | HardwareManager unified interface |
| `integration.py` | HardwareStatus - orchestrator bridge |
| `screen_controller.py` | Multi-page display, encoder handling, Game of Life |
| `display.py` | LCDDisplay I2C driver, custom characters |
| `leds.py` | LEDController with static/blink/breathe modes |
| `encoders.py` | EncoderHandler GPIO polling |
| `gpio_config.py` | Pin assignments, I2C addresses, color maps |

---

## GPIO Pin Assignments

### LEDs (Common Anode - LOW=ON)

| LED | Red | Green | Blue |
|-----|-----|-------|------|
| System | GPIO 23 | GPIO 24 | GPIO 25 |
| Trading | GPIO 26 | GPIO 16 | GPIO 20 |
| Research | GPIO 19 | GPIO 21 | GPIO 12 |

### Encoder

| Function | GPIO |
|----------|------|
| CLK | GPIO 22 |
| DT | GPIO 17 |
| SW (Button) | GPIO 27 |

### I2C

| Device | Address | Bus |
|--------|---------|-----|
| LCD Trading | 0x27 | 1 |
| LCD Research | 0x25 | 1 |

---

## Performance Impact

| Component | Overhead | Notes |
|-----------|----------|-------|
| Encoder polling | ~1-2% CPU | 125Hz polling in daemon thread |
| Display updates | ~50-100ms | I2C write every 1 second |
| LED control | Negligible | Subprocess calls only on state change |
| Game of Life | Microseconds | 160 cells, simple neighbor counting |

---

## Troubleshooting

### Display shows zeros/no data
- Check orchestrator is running: `sudo systemctl status trading-orchestrator`
- During research, display updates every 30 seconds (not blocked)

### Encoder not responding
- Check GPIO permissions
- Verify pins in `gpio_config.py` match wiring

### LCD not initializing
- Check I2C: `sudo i2cdetect -y 1`
- Should see devices at 0x27 (trading) and 0x25 (research)

### Custom characters not showing
- LCD may need power cycle after code update
- Restart orchestrator: `sudo systemctl restart trading-orchestrator`

---

## Development Notes

### Adding a New Display Page

1. Add page to `DisplayPage` enum in `screen_controller.py`
2. Add to `PAGE_ORDER` list
3. Add title to `PAGE_TITLES` dict
4. Create `_render_newpage(self, data)` method
5. Add case to `_render_current_page()`

### Adding New Data Fields

1. Add field collection in `_update_hardware_display()` in `daily_orchestrator.py`
2. Add to `display_data` dict
3. Add default value in `ScreenController._data`
4. Use in render method

### Testing Hardware Independently

```python
from hardware.integration import get_hardware_status

hw = get_hardware_status()
hw.startup()
hw.update_display({
    'spy_price': 590.25,
    'vix': 15.3,
    'phase': 'market_open',
    # ... other fields
})
```

---

## Known Gaps & Roadmap

### 1. Research Progress Reporting (Priority: High)

**Current State:** Research page shows static values (generation=0, sharpe=0) because the research subprocess runs independently and doesn't communicate progress back.

**Planned Solution:** Progress file approach

```
/home/thiselazar/trading_system/logs/research_progress.json
{
    "status": "EVOLVING",
    "phase": "param_optimization",
    "current_generation": 45,
    "max_generation": 100,
    "best_sharpe": 1.85,
    "current_strategy": "mean_reversion",
    "eta_minutes": 32,
    "updated_at": "2026-01-06T01:15:00"
}
```

**Implementation:**
1. Research script (`run_nightly_research.py`) writes progress file periodically
2. Orchestrator reads file during `_update_hardware_display()`
3. Graceful handling if file missing or stale (>5 min old)

---

### 2. CPU Temperature Display (Priority: Medium)

**Current State:** SYSTEM page shows RAM, ZRAM, CPU%, but no temperature.

**Planned Solution:** Add CPU temp reading

```python
# In _update_hardware_display():
cpu_temp = 0
try:
    with open('/sys/class/thermal/thermal_zone0/temp') as f:
        cpu_temp = int(f.read()) / 1000  # Convert millidegrees to degrees
except Exception:
    pass
```

**Display Format:** `CPU: 45C 12%` or add 4th line: `Temp: 45C`

---

### 3. Trade Execution Alerts (Priority: Low)

**Current State:** No visual feedback when trades execute.

**Planned Solution:** Temporary message overlay system

**Implementation:**
1. Add `show_alert(message, duration)` to ScreenController
2. Hook into ExecutionManager trade callbacks
3. Flash LED + show message: `BUY AAPL 10 @ $185.50`
4. Auto-dismiss after 5 seconds, return to current page

---

### 4. Error/Warning Display (Priority: Low)

**Current State:** Errors only visible in logs.

**Planned Solution:** Recent errors page or alert overlay

**Implementation:**
1. New ALERTS page showing last 3-5 warnings/errors
2. Or overlay system that interrupts current page briefly
3. Could use existing `DatabaseErrorHandler` as data source

---

## Implementation Status

| Feature | Status | Notes |
|---------|--------|-------|
| Multi-page display | Done | 6 pages with encoder navigation |
| Real-time data updates | Done | 30-second refresh during all phases |
| Trend arrows (SPY/VIX) | Done | Directional indicators |
| RAM + ZRAM display | Done | Both memory types shown |
| High-res Game of Life | Done | 20x8 resolution screensaver |
| Research progress | Planned | Needs progress file implementation |
| CPU temperature | Planned | Easy add to SYSTEM page |
| Trade alerts | Planned | Needs execution hook |
| Error display | Planned | Nice-to-have |
