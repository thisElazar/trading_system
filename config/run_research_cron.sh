#!/bin/bash
# Cron job for nightly research
# Add to crontab: crontab -e
#
# Run at 9:30 PM Eastern (after market close) every weekday:
# 30 21 * * 1-5 /mnt/nvme/trading_system/config/run_research_cron.sh
#
# Or for testing, run every 6 hours:
# 0 */6 * * * /mnt/nvme/trading_system/config/run_research_cron.sh

# Configuration
TRADING_ROOT="${TRADING_SYSTEM_ROOT:-/mnt/nvme/trading_system}"
LOG_DIR="$TRADING_ROOT/logs"
PYTHON="${PYTHON_PATH:-/usr/bin/python3}"

# Create log directory if needed
mkdir -p "$LOG_DIR"

# Timestamp for logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/cron_research_$TIMESTAMP.log"

# Change to trading directory
cd "$TRADING_ROOT" || exit 1

# Run the research script
echo "Starting nightly research at $(date)" >> "$LOG_FILE"
"$PYTHON" run_nightly_research.py >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

echo "Completed at $(date) with exit code $EXIT_CODE" >> "$LOG_FILE"

# Cleanup old logs (keep last 30 days)
find "$LOG_DIR" -name "cron_research_*.log" -mtime +30 -delete

exit $EXIT_CODE
