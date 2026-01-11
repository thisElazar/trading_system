"""
Alert System
============
Notifications for signals, executions, and system events.

Supports:
- Console logging
- File logging
- Webhook (Slack, Discord, etc.)
- Email (optional)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
from enum import Enum
from dataclasses import dataclass
import urllib.request
import urllib.error

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DIRS

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    SIGNAL = "signal"
    EXECUTION = "execution"
    POSITION_OPEN = "position_open"
    POSITION_CLOSE = "position_close"
    STOP_LOSS = "stop_loss"
    TARGET_HIT = "target_hit"
    STRATEGY_ERROR = "strategy_error"
    SYSTEM = "system"
    PERFORMANCE = "performance"


@dataclass
class Alert:
    """Alert message."""
    alert_type: AlertType
    level: AlertLevel
    title: str
    message: str
    data: Optional[dict] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        return {
            'type': self.alert_type.value,
            'level': self.level.value,
            'title': self.title,
            'message': self.message,
            'data': self.data,
            'timestamp': self.timestamp
        }
    
    def format_console(self) -> str:
        """Format for console output."""
        icon = {
            AlertLevel.DEBUG: "🔍",
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨"
        }.get(self.level, "•")
        
        return f"{icon} [{self.alert_type.value.upper()}] {self.title}\n   {self.message}"
    
    def format_slack(self) -> dict:
        """Format for Slack webhook."""
        color = {
            AlertLevel.DEBUG: "#808080",
            AlertLevel.INFO: "#36a64f",
            AlertLevel.WARNING: "#ff9900",
            AlertLevel.ERROR: "#ff0000",
            AlertLevel.CRITICAL: "#8b0000"
        }.get(self.level, "#808080")
        
        return {
            "attachments": [{
                "color": color,
                "title": self.title,
                "text": self.message,
                "fields": [
                    {"title": "Type", "value": self.alert_type.value, "short": True},
                    {"title": "Level", "value": self.level.value, "short": True}
                ],
                "ts": datetime.fromisoformat(self.timestamp).timestamp()
            }]
        }


class AlertHandler:
    """Base class for alert handlers."""
    
    def __init__(self, min_level: AlertLevel = AlertLevel.INFO):
        self.min_level = min_level
        self._level_order = [AlertLevel.DEBUG, AlertLevel.INFO, AlertLevel.WARNING, 
                           AlertLevel.ERROR, AlertLevel.CRITICAL]
    
    def should_handle(self, alert: Alert) -> bool:
        """Check if this handler should process the alert."""
        return self._level_order.index(alert.level) >= self._level_order.index(self.min_level)
    
    def handle(self, alert: Alert):
        """Process an alert. Override in subclasses."""
        raise NotImplementedError


class ConsoleHandler(AlertHandler):
    """Print alerts to console."""
    
    def handle(self, alert: Alert):
        if not self.should_handle(alert):
            return
        print(alert.format_console())


class FileHandler(AlertHandler):
    """Write alerts to file."""
    
    def __init__(self, log_path: Path = None, min_level: AlertLevel = AlertLevel.INFO):
        super().__init__(min_level)
        self.log_path = log_path or (DIRS.get('logs', Path('./logs')) / 'alerts.log')
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def handle(self, alert: Alert):
        if not self.should_handle(alert):
            return
        
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(alert.to_dict()) + '\n')


class WebhookHandler(AlertHandler):
    """Send alerts to webhook (Slack, Discord, etc.)."""
    
    def __init__(
        self, 
        webhook_url: str, 
        min_level: AlertLevel = AlertLevel.WARNING,
        format_type: str = 'slack'
    ):
        super().__init__(min_level)
        self.webhook_url = webhook_url
        self.format_type = format_type
    
    def handle(self, alert: Alert):
        if not self.should_handle(alert):
            return
        
        if not self.webhook_url:
            return
        
        try:
            if self.format_type == 'slack':
                payload = alert.format_slack()
            else:
                payload = alert.to_dict()
            
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status != 200:
                    logger.warning(f"Webhook returned {response.status}")
                    
        except urllib.error.URLError as e:
            logger.warning(f"Webhook failed: {e}")
        except Exception as e:
            logger.warning(f"Webhook error: {e}")


class TelegramHandler(AlertHandler):
    """Send alerts via Telegram Bot API."""

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        min_level: AlertLevel = AlertLevel.WARNING
    ):
        super().__init__(min_level)
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    def _format_message(self, alert: Alert) -> str:
        """Format alert for Telegram (Markdown)."""
        icon = {
            AlertLevel.DEBUG: "🔍",
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨"
        }.get(alert.level, "•")

        # Escape markdown special chars in message
        message = alert.message.replace('_', '\\_').replace('*', '\\*')
        title = alert.title.replace('_', '\\_').replace('*', '\\*')

        return f"{icon} *{title}*\n{message}"

    def handle(self, alert: Alert):
        if not self.should_handle(alert):
            return

        if not self.bot_token or not self.chat_id:
            return

        try:
            payload = {
                'chat_id': self.chat_id,
                'text': self._format_message(alert),
                'parse_mode': 'Markdown'
            }

            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                self.api_url,
                data=data,
                headers={'Content-Type': 'application/json'}
            )

            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status != 200:
                    logger.warning(f"Telegram returned {response.status}")

        except urllib.error.URLError as e:
            logger.warning(f"Telegram failed: {e}")
        except Exception as e:
            logger.warning(f"Telegram error: {e}")


class AlertManager:
    """
    Central alert management.

    Usage:
        alerts = AlertManager()
        alerts.add_handler(ConsoleHandler())
        alerts.add_handler(WebhookHandler(slack_url))
        alerts.add_handler(TelegramHandler(bot_token, chat_id))

        alerts.signal("AAPL", "long", 150.0, "pairs_trading")
        alerts.error("Strategy failed", exception)
    """
    
    def __init__(self):
        self.handlers: List[AlertHandler] = []
        self.history: List[Alert] = []
        self.max_history = 1000
    
    def add_handler(self, handler: AlertHandler):
        """Add an alert handler."""
        self.handlers.append(handler)
    
    def _send(self, alert: Alert):
        """Send alert to all handlers."""
        self.history.append(alert)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        for handler in self.handlers:
            try:
                handler.handle(alert)
            except Exception as e:
                logger.error(f"Handler failed: {e}")
    
    # Convenience methods
    def signal(
        self, 
        symbol: str, 
        direction: str, 
        price: float, 
        strategy: str,
        **kwargs
    ):
        """Alert for new signal."""
        alert = Alert(
            alert_type=AlertType.SIGNAL,
            level=AlertLevel.INFO,
            title=f"Signal: {direction.upper()} {symbol}",
            message=f"{strategy} generated {direction} signal at ${price:.2f}",
            data={'symbol': symbol, 'direction': direction, 'price': price, 
                  'strategy': strategy, **kwargs}
        )
        self._send(alert)
    
    def execution(
        self, 
        symbol: str, 
        direction: str, 
        quantity: int,
        fill_price: float,
        slippage: float = 0
    ):
        """Alert for execution."""
        alert = Alert(
            alert_type=AlertType.EXECUTION,
            level=AlertLevel.INFO,
            title=f"Executed: {direction.upper()} {quantity} {symbol}",
            message=f"Filled at ${fill_price:.2f} (slippage: ${slippage:.2f})",
            data={'symbol': symbol, 'direction': direction, 'quantity': quantity,
                  'fill_price': fill_price, 'slippage': slippage}
        )
        self._send(alert)
    
    def position_opened(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        strategy: str
    ):
        """Alert for position opened."""
        alert = Alert(
            alert_type=AlertType.POSITION_OPEN,
            level=AlertLevel.INFO,
            title=f"Position Opened: {direction.upper()} {symbol}",
            message=f"Entry: ${entry_price:.2f} | Stop: ${stop_loss:.2f} | Strategy: {strategy}",
            data={'symbol': symbol, 'direction': direction, 'entry_price': entry_price,
                  'stop_loss': stop_loss, 'strategy': strategy}
        )
        self._send(alert)
    
    def position_closed(
        self,
        symbol: str,
        direction: str,
        exit_price: float,
        pnl_pct: float,
        reason: str
    ):
        """Alert for position closed."""
        level = AlertLevel.INFO if pnl_pct >= 0 else AlertLevel.WARNING
        icon = "✅" if pnl_pct >= 0 else "❌"
        
        alert = Alert(
            alert_type=AlertType.POSITION_CLOSE,
            level=level,
            title=f"{icon} Position Closed: {symbol}",
            message=f"Exit: ${exit_price:.2f} | P&L: {pnl_pct:+.2f}% | Reason: {reason}",
            data={'symbol': symbol, 'direction': direction, 'exit_price': exit_price,
                  'pnl_pct': pnl_pct, 'reason': reason}
        )
        self._send(alert)
    
    def stop_hit(self, symbol: str, stop_price: float, loss_pct: float):
        """Alert for stop loss hit."""
        alert = Alert(
            alert_type=AlertType.STOP_LOSS,
            level=AlertLevel.WARNING,
            title=f"🛑 Stop Loss: {symbol}",
            message=f"Hit stop at ${stop_price:.2f} | Loss: {loss_pct:.2f}%",
            data={'symbol': symbol, 'stop_price': stop_price, 'loss_pct': loss_pct}
        )
        self._send(alert)
    
    def target_hit(self, symbol: str, target_price: float, gain_pct: float):
        """Alert for target hit."""
        alert = Alert(
            alert_type=AlertType.TARGET_HIT,
            level=AlertLevel.INFO,
            title=f"🎯 Target Hit: {symbol}",
            message=f"Hit target at ${target_price:.2f} | Gain: {gain_pct:.2f}%",
            data={'symbol': symbol, 'target_price': target_price, 'gain_pct': gain_pct}
        )
        self._send(alert)
    
    def error(self, title: str, error: Exception = None, strategy: str = None):
        """Alert for error."""
        message = str(error) if error else "Unknown error"
        if strategy:
            message = f"[{strategy}] {message}"
        
        alert = Alert(
            alert_type=AlertType.STRATEGY_ERROR,
            level=AlertLevel.ERROR,
            title=title,
            message=message,
            data={'strategy': strategy, 'error_type': type(error).__name__ if error else None}
        )
        self._send(alert)
    
    def critical(self, title: str, message: str):
        """Alert for critical system issue."""
        alert = Alert(
            alert_type=AlertType.SYSTEM,
            level=AlertLevel.CRITICAL,
            title=f"🚨 {title}",
            message=message
        )
        self._send(alert)
    
    def performance(self, strategy: str, stats: dict):
        """Alert for performance update."""
        win_rate = stats.get('win_rate', 0)
        total_pnl = stats.get('total_pnl', 0)
        
        level = AlertLevel.INFO if total_pnl >= 0 else AlertLevel.WARNING
        
        alert = Alert(
            alert_type=AlertType.PERFORMANCE,
            level=level,
            title=f"Performance: {strategy}",
            message=f"Win Rate: {win_rate:.1%} | Total P&L: {total_pnl:.2f}%",
            data=stats
        )
        self._send(alert)
    
    def daily_summary(self, stats: dict):
        """Send daily summary."""
        open_positions = stats.get('open_positions', 0)
        today_pnl = stats.get('today_pnl', 0)
        total_signals = stats.get('total_signals', 0)

        alert = Alert(
            alert_type=AlertType.SYSTEM,
            level=AlertLevel.INFO,
            title="📊 Daily Summary",
            message=f"Signals: {total_signals} | Open Positions: {open_positions} | Today P&L: {today_pnl:.2f}%",
            data=stats
        )
        self._send(alert)

    def send_alert(self, message: str, level: str = "info", title: str = None):
        """
        Send a generic alert message.

        Args:
            message: Alert message text
            level: Alert level ('debug', 'info', 'warning', 'error', 'critical')
            title: Optional title (defaults to level-based title)
        """
        level_map = {
            'debug': AlertLevel.DEBUG,
            'info': AlertLevel.INFO,
            'warning': AlertLevel.WARNING,
            'error': AlertLevel.ERROR,
            'critical': AlertLevel.CRITICAL,
        }
        alert_level = level_map.get(level.lower(), AlertLevel.INFO)

        if title is None:
            title = f"System {level.capitalize()}"

        alert = Alert(
            alert_type=AlertType.SYSTEM,
            level=alert_level,
            title=title,
            message=message
        )
        self._send(alert)

    def get_recent(self, count: int = 20, level: AlertLevel = None) -> List[Alert]:
        """Get recent alerts."""
        alerts = self.history[-count:]
        if level:
            level_order = [AlertLevel.DEBUG, AlertLevel.INFO, AlertLevel.WARNING, 
                         AlertLevel.ERROR, AlertLevel.CRITICAL]
            min_idx = level_order.index(level)
            alerts = [a for a in alerts if level_order.index(a.level) >= min_idx]
        return alerts


def create_alert_manager(
    console: bool = True,
    file: bool = True,
    webhook_url: str = None,
    telegram_token: str = None,
    telegram_chat_id: str = None
) -> AlertManager:
    """Create alert manager with common handlers."""
    manager = AlertManager()

    if console:
        manager.add_handler(ConsoleHandler(min_level=AlertLevel.INFO))

    if file:
        manager.add_handler(FileHandler(min_level=AlertLevel.DEBUG))

    if webhook_url:
        manager.add_handler(WebhookHandler(webhook_url, min_level=AlertLevel.WARNING))

    if telegram_token and telegram_chat_id:
        manager.add_handler(TelegramHandler(telegram_token, telegram_chat_id, min_level=AlertLevel.WARNING))

    return manager


# Global instance
_alerts: Optional[AlertManager] = None

def get_alerts() -> AlertManager:
    """Get or create global alert manager."""
    global _alerts
    if _alerts is None:
        _alerts = create_alert_manager()
    return _alerts


if __name__ == "__main__":
    # Demo
    alerts = create_alert_manager(console=True, file=True)
    
    print("="*60)
    print("ALERT SYSTEM DEMO")
    print("="*60 + "\n")
    
    alerts.signal("AAPL", "long", 250.0, "pairs_trading", zscore=-2.1)
    alerts.execution("AAPL", "long", 10, 250.05, slippage=0.05)
    alerts.position_opened("AAPL", "long", 250.05, 245.0, "pairs_trading")
    alerts.target_hit("AAPL", 260.0, 3.98)
    alerts.position_closed("AAPL", "long", 260.0, 3.98, "target")
    alerts.error("Strategy failed", ValueError("Invalid data"), "gap_fill")
    
    print("\n" + "="*60)
    print("Recent alerts saved to logs/alerts.log")
