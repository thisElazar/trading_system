"""
Dashboard Module
================
Web-based monitoring dashboard using Dash + Plotly.

Features:
- Real-time position monitoring
- Account summary (equity, cash, buying power)
- Equity curve chart
- Strategy performance comparison
- Recent alerts
- System status (market phase, next phase)

Usage:
    python observability/dashboard/app.py

Then open http://localhost:5000 in your browser.
"""

from .app import app

__all__ = ['app']
