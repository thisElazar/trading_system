"""
Health check endpoint for monitoring tools.
Add to dashboard by importing and calling register_health_endpoint(app)
"""
import json
import time
import psutil
from flask import Response

def register_health_endpoint(dash_app):
    """Register /health endpoint on the Flask server underlying Dash."""
    
    @dash_app.server.route("/health")
    def health_check():
        start = time.time()
        health = {
            "status": "healthy",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "checks": {}
        }
        
        # Memory check
        mem = psutil.virtual_memory()
        health["checks"]["memory"] = {
            "available_mb": round(mem.available / 1024 / 1024),
            "percent_used": mem.percent,
            "status": "ok" if mem.available > 300 * 1024 * 1024 else "warning"
        }
        
        # CPU check
        health["checks"]["cpu"] = {
            "percent": psutil.cpu_percent(interval=0.1),
            "load_avg": list(psutil.getloadavg()),
            "status": "ok"
        }
        
        # Disk check
        disk = psutil.disk_usage("/")
        health["checks"]["disk"] = {
            "free_gb": round(disk.free / 1024 / 1024 / 1024, 1),
            "percent_used": disk.percent,
            "status": "ok" if disk.percent < 90 else "warning"
        }
        
        # Temperature check (Pi specific)
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                temp_c = int(f.read().strip()) / 1000
                health["checks"]["temperature"] = {
                    "celsius": temp_c,
                    "status": "ok" if temp_c < 70 else "warning" if temp_c < 80 else "critical"
                }
        except:
            health["checks"]["temperature"] = {"status": "unavailable"}
        
        # Alpaca connection check
        try:
            from execution.alpaca_connector import AlpacaConnector
            connector = AlpacaConnector(paper=True)
            account = connector.get_account()
            health["checks"]["alpaca"] = {
                "connected": True,
                "portfolio_value": float(account.portfolio_value),
                "status": "ok"
            }
        except Exception as e:
            health["checks"]["alpaca"] = {
                "connected": False,
                "error": str(e),
                "status": "critical"
            }
        
        # Trading service check
        try:
            import subprocess
            result = subprocess.run(
                ["systemctl", "is-active", "trading-system"],
                capture_output=True, text=True, timeout=5
            )
            health["checks"]["trading_service"] = {
                "active": result.stdout.strip() == "active",
                "status": "ok" if result.stdout.strip() == "active" else "critical"
            }
        except:
            health["checks"]["trading_service"] = {"status": "unavailable"}
        
        # Overall status
        statuses = [c.get("status", "ok") for c in health["checks"].values()]
        if "critical" in statuses:
            health["status"] = "unhealthy"
        elif "warning" in statuses:
            health["status"] = "degraded"
        
        health["response_time_ms"] = round((time.time() - start) * 1000, 1)
        
        return Response(
            json.dumps(health, indent=2),
            mimetype="application/json",
            status=200 if health["status"] == "healthy" else 503
        )
    
    return dash_app
