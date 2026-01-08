# LED Status Codes Reference

## Hardware Overview

Three RGB LEDs provide real-time system status:

| LED | Position | Purpose |
|-----|----------|---------|
| 1 | Left | System health |
| 2 | Middle | Trading engine |
| 3 | Right | Research engine |

---

## LED 1: System Health

| Color | Status | Meaning |
|-------|--------|---------|
| Green | Healthy | System running normally |
| Yellow | Warning | Non-critical issue detected |
| Red | Error | Critical error - check logs |
| Off | Offline | System not running |

---

## LED 2: Trading Engine

| Color | Status | Meaning |
|-------|--------|---------|
| Green | Active | Market open, actively trading |
| Yellow | Pending | Pre-market, warming up |
| Blue | Idle | Post-market, system ready but not trading |
| Off | Stopped | Overnight/weekend, trading disabled |

---

## LED 3: Research Engine

| Color | Status | Meaning |
|-------|--------|---------|
| Breathing Blue | Evolving | Nightly research running (GP evolution, parameter optimization) |
| Solid Green | Complete | Research finished successfully |
| Red | Failed | Research encountered an error |
| Yellow | Paused | Research paused |
| Off | Offline | No research activity |

---

## Market Phase LED States

| Phase | Time (ET) | System | Trading | Research |
|-------|-----------|--------|---------|----------|
| Pre-market | 8:00-9:30 | Green | Yellow | Off |
| Market Open | 9:30-16:00 | Green | Green | Off |
| Post-market | 16:00-17:00 | Green | Blue | Off |
| Evening | 17:00-21:30 | Green | Blue | Off |
| Overnight | 21:30-8:00 | Green | Off | Breathing Blue* |
| Weekend | Fri 16:00 - Sun 20:00 | Green | Off | Breathing Blue* |

*Research LED breathes during nightly optimization runs

---

## Special Patterns

| Pattern | Meaning |
|---------|---------|
| All flash red (3x) | Error detected |
| Startup sequence (R-G-B-W cycle) | System boot |
| All off | System stopped or hardware disconnected |

---

## Troubleshooting

**No LEDs lit:**
- Check orchestrator service: `sudo systemctl status trading-orchestrator`
- Check hardware connection

**Research LED not breathing:**
- Nightly research may have already completed
- Check: `journalctl -u trading-orchestrator -n 50`

**Trading LED red during overnight:**
- This is a bug - should be off. Check gpio_config.py STATUS_COLORS

---

## GPIO Pin Reference

| LED | Red | Green | Blue |
|-----|-----|-------|------|
| System (1) | GPIO 23 | GPIO 24 | GPIO 25 |
| Trading (2) | GPIO 26 | GPIO 16 | GPIO 20 |
| Research (3) | GPIO 19 | GPIO 21 | GPIO 12 |

Note: Common anode LEDs - LOW = ON, HIGH = OFF
