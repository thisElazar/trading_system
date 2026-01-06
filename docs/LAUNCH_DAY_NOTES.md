# Launch Day Notes - January 5, 2026

## Quick Health Check Commands

```bash
# Services
systemctl status trading-orchestrator
systemctl status trading-dashboard

# Logs (live tail)
tail -f /home/thiselazar/trading_system/logs/orchestrator.log

# Alpaca account
cd /home/thiselazar/trading_system && source venv/bin/activate
python3 -c "from alpaca.trading.client import TradingClient; c = TradingClient('PKKOBJDY4UKSHT6GYNZ3HMZAND', 'EBYZFLkzrXg4J3Pp9mwdjyuvHmsNuD3DJzXdrnCVPo64', paper=True); a = c.get_account(); print(f'Equity: \${float(a.equity):,.2f}')"

# Dashboard
# http://<pi-ip>:8050
```

## Files Modified Today (Jan 4)

| File | Changes |
|------|---------|
| `research/genetic/persistent_optimizer.py` | OOS scaling, soft penalties, diversity injection, adaptive mutation |
| `run_nightly_research.py` | Updated constraint function to match |

## Key Thresholds

- **REJECTION_FITNESS**: 0.01 (was 0.0)
- **Diversity injection**: Triggers when >50% have low fitness
- **Adaptive mutation**: 15% base, +5% per stagnant gen, max 40%
- **Early stop extended**: +2 gens when mutation elevated

## Watch For Tomorrow

1. **9:30 AM ET**: Gap detection on SPY/QQQ
2. **Orchestrator logs**: Should show phase transitions
3. **GA improvements**: Nightly research should find improvements now

## If Something Goes Wrong

```bash
# Restart services
sudo systemctl restart trading-orchestrator
sudo systemctl restart trading-dashboard

# Check for errors
journalctl -u trading-orchestrator -n 50

# Full system restore (if SD dies again)
sudo bash /mnt/nvme/system_backup/restore_system.sh
```

---

*Good luck, TradeBot. See you at market open.*
