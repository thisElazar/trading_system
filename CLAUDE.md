# Trading System

Automated trading system on Raspberry Pi 5. Paper trading via Alpaca.

## Critical Safety Rules

1. **Never modify config.py without explicit approval** - affects live trading
2. **Never restart trading-orchestrator during market hours** (9:30-16:00 ET) without approval
3. **Kill switch**: `touch ~/trading_system/killswitch/HALT` stops all new orders
4. **Database backups**: Run before any DB schema changes

## System State

- **Orchestrator**: `systemctl status trading-orchestrator`
- **Positions**: `sqlite3 db/trades.db "SELECT symbol, direction, quantity, unrealized_pnl FROM positions WHERE status='open'"`
- **Phase**: Check orchestrator.log for current phase (PRE_MARKET, TRADING, POST_MARKET, OVERNIGHT, WEEKEND)

## Architecture Quick Reference

| Component | Entry Point | Purpose |
|-----------|-------------|---------|
| Orchestrator | daily_orchestrator.py | Phase management, task execution |
| Execution | execution/execution_manager.py | Signal routing, position limits |
| Research | run_nightly_research.py | GA/GP strategy optimization |
| Hardware | hardware/integration.py | LEDs, LCD, encoder |

## Common Issues

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| No trades executing | Circuit breaker triggered | Check `circuit_breaker_state` table |
| Orchestrator frozen | Memory pressure or deadlock | `sudo systemctl restart trading-orchestrator` |
| LCD showing stale data | Orchestrator not updating | Check service status |
| Research not running | Wrong phase or disabled | Check ENABLE_* flags in config.py |

## Skills Available

Detailed reference in `~/.claude/skills/tradebot-*/`:
- system-map, diagnostics, hardware, trading, research-pipeline, execution-flow
