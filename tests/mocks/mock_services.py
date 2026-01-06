"""
Mock External Services for testing.

Provides mock implementations of external services and APIs
with configurable behavior, latency injection, and failure simulation.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from unittest.mock import MagicMock
import random


# =============================================================================
# Network Failure Simulation
# =============================================================================

class FailureMode(Enum):
    """Types of simulated failures."""
    NONE = 'none'
    TIMEOUT = 'timeout'
    CONNECTION_ERROR = 'connection_error'
    HTTP_500 = 'http_500'
    HTTP_429_RATE_LIMIT = 'rate_limit'
    HTTP_401_UNAUTHORIZED = 'unauthorized'
    HTTP_403_FORBIDDEN = 'forbidden'
    HTTP_404_NOT_FOUND = 'not_found'
    PARTIAL_FAILURE = 'partial'  # Some calls fail randomly
    INTERMITTENT = 'intermittent'  # Alternating success/failure


class NetworkError(Exception):
    """Network-related error."""
    def __init__(self, message: str, status_code: int = None):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class TimeoutError(NetworkError):
    """Request timeout error."""
    def __init__(self, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds
        super().__init__(f"Request timed out after {timeout_seconds}s", 408)


class RateLimitError(NetworkError):
    """Rate limit exceeded error."""
    def __init__(self, retry_after: int = 60):
        self.retry_after = retry_after
        super().__init__(f"Rate limited. Retry after {retry_after}s", 429)


class ConnectionError(NetworkError):
    """Connection failed error."""
    def __init__(self, host: str = "unknown"):
        self.host = host
        super().__init__(f"Failed to connect to {host}", None)


# =============================================================================
# Latency Injection
# =============================================================================

@dataclass
class LatencyConfig:
    """Configuration for latency injection."""
    min_ms: int = 0
    max_ms: int = 0
    jitter_pct: float = 0.0  # Random variation as percentage
    spike_probability: float = 0.0  # Probability of latency spike
    spike_multiplier: float = 10.0  # How much to multiply latency during spike

    def get_delay(self) -> float:
        """Get delay in seconds."""
        base_delay = random.uniform(self.min_ms, self.max_ms) / 1000.0

        # Add jitter
        if self.jitter_pct > 0:
            jitter = base_delay * random.uniform(-self.jitter_pct, self.jitter_pct)
            base_delay += jitter

        # Check for spike
        if self.spike_probability > 0 and random.random() < self.spike_probability:
            base_delay *= self.spike_multiplier

        return max(0, base_delay)


class LatencyInjector:
    """Injects configurable latency into operations."""

    def __init__(self, config: LatencyConfig = None):
        self.config = config or LatencyConfig()

    def inject(self) -> None:
        """Inject latency (blocking)."""
        delay = self.config.get_delay()
        if delay > 0:
            time.sleep(delay)

    async def inject_async(self) -> None:
        """Inject latency (async)."""
        delay = self.config.get_delay()
        if delay > 0:
            await asyncio.sleep(delay)


# =============================================================================
# Mock HTTP Client
# =============================================================================

@dataclass
class MockHTTPResponse:
    """Mock HTTP response."""
    status_code: int = 200
    content: bytes = b'{}'
    headers: Dict[str, str] = field(default_factory=dict)
    elapsed_ms: float = 0

    @property
    def text(self) -> str:
        return self.content.decode('utf-8')

    @property
    def json(self) -> Any:
        import json
        return json.loads(self.text)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise NetworkError(f"HTTP {self.status_code}", self.status_code)


class MockHTTPClient:
    """
    Mock HTTP client for testing external API calls.

    Provides:
    - Configurable responses
    - Failure injection
    - Latency simulation
    - Request tracking
    """

    def __init__(
        self,
        default_response: MockHTTPResponse = None,
        latency_config: LatencyConfig = None,
    ):
        self._responses: Dict[str, MockHTTPResponse] = {}
        self._default_response = default_response or MockHTTPResponse()
        self._failure_mode = FailureMode.NONE
        self._failure_rate = 0.0
        self._latency = LatencyInjector(latency_config)
        self._request_history: List[Dict] = []
        self._call_count = 0

    def set_response(
        self,
        url_pattern: str,
        response: MockHTTPResponse,
    ) -> None:
        """Set response for a specific URL pattern."""
        self._responses[url_pattern] = response

    def set_json_response(
        self,
        url_pattern: str,
        data: Any,
        status_code: int = 200,
    ) -> None:
        """Set JSON response for a URL pattern."""
        import json
        self._responses[url_pattern] = MockHTTPResponse(
            status_code=status_code,
            content=json.dumps(data).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
        )

    def set_failure_mode(
        self,
        mode: FailureMode,
        failure_rate: float = 1.0,
    ) -> None:
        """Configure failure behavior."""
        self._failure_mode = mode
        self._failure_rate = failure_rate

    def set_latency(self, min_ms: int = 0, max_ms: int = 100) -> None:
        """Configure latency."""
        self._latency.config = LatencyConfig(min_ms=min_ms, max_ms=max_ms)

    def _should_fail(self) -> bool:
        """Check if this request should fail."""
        if self._failure_mode == FailureMode.NONE:
            return False
        if self._failure_mode == FailureMode.INTERMITTENT:
            self._call_count += 1
            return self._call_count % 2 == 0
        return random.random() < self._failure_rate

    def _apply_failure(self) -> None:
        """Apply configured failure mode."""
        if self._failure_mode == FailureMode.TIMEOUT:
            raise TimeoutError(30.0)
        elif self._failure_mode == FailureMode.CONNECTION_ERROR:
            raise ConnectionError("api.example.com")
        elif self._failure_mode == FailureMode.HTTP_500:
            raise NetworkError("Internal Server Error", 500)
        elif self._failure_mode == FailureMode.HTTP_429_RATE_LIMIT:
            raise RateLimitError(60)
        elif self._failure_mode == FailureMode.HTTP_401_UNAUTHORIZED:
            raise NetworkError("Unauthorized", 401)
        elif self._failure_mode == FailureMode.HTTP_403_FORBIDDEN:
            raise NetworkError("Forbidden", 403)
        elif self._failure_mode == FailureMode.HTTP_404_NOT_FOUND:
            raise NetworkError("Not Found", 404)
        elif self._failure_mode in [FailureMode.PARTIAL_FAILURE, FailureMode.INTERMITTENT]:
            raise NetworkError("Simulated partial failure", 500)

    def get(self, url: str, **kwargs) -> MockHTTPResponse:
        """Mock GET request."""
        return self._request('GET', url, **kwargs)

    def post(self, url: str, **kwargs) -> MockHTTPResponse:
        """Mock POST request."""
        return self._request('POST', url, **kwargs)

    def put(self, url: str, **kwargs) -> MockHTTPResponse:
        """Mock PUT request."""
        return self._request('PUT', url, **kwargs)

    def delete(self, url: str, **kwargs) -> MockHTTPResponse:
        """Mock DELETE request."""
        return self._request('DELETE', url, **kwargs)

    def _request(self, method: str, url: str, **kwargs) -> MockHTTPResponse:
        """Execute mock request."""
        start_time = time.time()

        # Track request
        self._request_history.append({
            'method': method,
            'url': url,
            'timestamp': datetime.now(),
            'kwargs': kwargs,
        })

        # Inject latency
        self._latency.inject()

        # Check for failure
        if self._should_fail():
            self._apply_failure()

        # Find matching response
        for pattern, response in self._responses.items():
            if pattern in url:
                response.elapsed_ms = (time.time() - start_time) * 1000
                return response

        # Return default response
        self._default_response.elapsed_ms = (time.time() - start_time) * 1000
        return self._default_response

    def get_request_history(self) -> List[Dict]:
        """Get history of requests."""
        return self._request_history.copy()

    def clear_history(self) -> None:
        """Clear request history."""
        self._request_history.clear()

    def reset(self) -> None:
        """Reset all configuration."""
        self._responses.clear()
        self._failure_mode = FailureMode.NONE
        self._failure_rate = 0.0
        self._request_history.clear()
        self._call_count = 0


# =============================================================================
# Mock Email/Notification Service
# =============================================================================

@dataclass
class MockEmail:
    """Mock email message."""
    to: str
    subject: str
    body: str
    html_body: Optional[str] = None
    sent_at: datetime = field(default_factory=datetime.now)


class MockEmailService:
    """
    Mock email service for testing notifications.

    Captures sent emails for verification.
    """

    def __init__(self):
        self._sent_emails: List[MockEmail] = []
        self._should_fail = False
        self._failure_message = ""

    def send(
        self,
        to: str,
        subject: str,
        body: str,
        html_body: str = None,
    ) -> bool:
        """Send a mock email."""
        if self._should_fail:
            raise Exception(self._failure_message or "Email delivery failed")

        self._sent_emails.append(MockEmail(
            to=to,
            subject=subject,
            body=body,
            html_body=html_body,
        ))
        return True

    def get_sent_emails(self) -> List[MockEmail]:
        """Get all sent emails."""
        return self._sent_emails.copy()

    def get_emails_to(self, recipient: str) -> List[MockEmail]:
        """Get emails sent to specific recipient."""
        return [e for e in self._sent_emails if e.to == recipient]

    def get_emails_with_subject(self, subject_contains: str) -> List[MockEmail]:
        """Get emails with subject containing string."""
        return [e for e in self._sent_emails if subject_contains in e.subject]

    def clear(self) -> None:
        """Clear sent emails."""
        self._sent_emails.clear()

    def set_failure(self, should_fail: bool, message: str = "") -> None:
        """Configure failure behavior."""
        self._should_fail = should_fail
        self._failure_message = message


# =============================================================================
# Mock Slack/Discord Webhook
# =============================================================================

@dataclass
class MockWebhookMessage:
    """Mock webhook message."""
    channel: str
    message: str
    attachments: List[Dict] = field(default_factory=list)
    sent_at: datetime = field(default_factory=datetime.now)


class MockWebhookService:
    """Mock webhook service for testing alerts."""

    def __init__(self):
        self._sent_messages: List[MockWebhookMessage] = []
        self._should_fail = False

    def send(
        self,
        channel: str,
        message: str,
        attachments: List[Dict] = None,
    ) -> bool:
        """Send a mock webhook message."""
        if self._should_fail:
            raise NetworkError("Webhook delivery failed", 500)

        self._sent_messages.append(MockWebhookMessage(
            channel=channel,
            message=message,
            attachments=attachments or [],
        ))
        return True

    def get_messages(self) -> List[MockWebhookMessage]:
        """Get all sent messages."""
        return self._sent_messages.copy()

    def get_messages_to_channel(self, channel: str) -> List[MockWebhookMessage]:
        """Get messages sent to specific channel."""
        return [m for m in self._sent_messages if m.channel == channel]

    def clear(self) -> None:
        """Clear sent messages."""
        self._sent_messages.clear()

    def set_failure(self, should_fail: bool) -> None:
        """Configure failure behavior."""
        self._should_fail = should_fail


# =============================================================================
# Mock Time Provider
# =============================================================================

class MockTimeProvider:
    """
    Mock time provider for testing time-sensitive logic.

    Allows controlling the "current time" for tests.
    """

    def __init__(self, initial_time: datetime = None):
        self._frozen_time = initial_time
        self._time_offset = timedelta(0)

    def now(self) -> datetime:
        """Get current (mock) time."""
        if self._frozen_time:
            return self._frozen_time + self._time_offset
        return datetime.now() + self._time_offset

    def freeze(self, time: datetime) -> None:
        """Freeze time at specific moment."""
        self._frozen_time = time

    def unfreeze(self) -> None:
        """Unfreeze time."""
        self._frozen_time = None
        self._time_offset = timedelta(0)

    def advance(self, delta: timedelta) -> None:
        """Advance time by specified amount."""
        self._time_offset += delta

    def advance_days(self, days: int) -> None:
        """Advance time by days."""
        self.advance(timedelta(days=days))

    def advance_hours(self, hours: int) -> None:
        """Advance time by hours."""
        self.advance(timedelta(hours=hours))

    def advance_minutes(self, minutes: int) -> None:
        """Advance time by minutes."""
        self.advance(timedelta(minutes=minutes))

    def is_market_hours(self) -> bool:
        """Check if current mock time is during market hours."""
        now = self.now()
        # Market hours: 9:30 AM - 4:00 PM ET, Mon-Fri
        if now.weekday() >= 5:  # Weekend
            return False
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now <= market_close


# =============================================================================
# Mock Cache Service
# =============================================================================

class MockCacheService:
    """Mock cache service for testing caching logic."""

    def __init__(self):
        self._cache: Dict[str, Dict] = {}  # key -> {value, expires_at}
        self._should_fail = False

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if self._should_fail:
            raise Exception("Cache service unavailable")

        if key not in self._cache:
            return None

        entry = self._cache[key]
        if entry['expires_at'] and datetime.now() > entry['expires_at']:
            del self._cache[key]
            return None

        return entry['value']

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: int = None,
    ) -> bool:
        """Set value in cache."""
        if self._should_fail:
            raise Exception("Cache service unavailable")

        expires_at = None
        if ttl_seconds:
            expires_at = datetime.now() + timedelta(seconds=ttl_seconds)

        self._cache[key] = {
            'value': value,
            'expires_at': expires_at,
        }
        return True

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all cache."""
        self._cache.clear()

    def set_failure(self, should_fail: bool) -> None:
        """Configure failure behavior."""
        self._should_fail = should_fail

    def get_all_keys(self) -> List[str]:
        """Get all cache keys."""
        return list(self._cache.keys())


# =============================================================================
# Mock Metrics/Monitoring Service
# =============================================================================

@dataclass
class MockMetric:
    """Mock metric data point."""
    name: str
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class MockMetricsService:
    """Mock metrics service for testing monitoring."""

    def __init__(self):
        self._metrics: List[MockMetric] = []
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}

    def gauge(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Record a gauge metric."""
        self._gauges[name] = value
        self._metrics.append(MockMetric(
            name=name,
            value=value,
            tags=tags or {},
        ))

    def counter(self, name: str, value: int = 1, tags: Dict[str, str] = None) -> None:
        """Increment a counter."""
        self._counters[name] = self._counters.get(name, 0) + value
        self._metrics.append(MockMetric(
            name=name,
            value=float(self._counters[name]),
            tags=tags or {},
        ))

    def timing(self, name: str, value_ms: float, tags: Dict[str, str] = None) -> None:
        """Record a timing metric."""
        self._metrics.append(MockMetric(
            name=name,
            value=value_ms,
            tags=tags or {},
        ))

    def get_gauge(self, name: str) -> Optional[float]:
        """Get current gauge value."""
        return self._gauges.get(name)

    def get_counter(self, name: str) -> int:
        """Get current counter value."""
        return self._counters.get(name, 0)

    def get_metrics(self, name: str = None) -> List[MockMetric]:
        """Get recorded metrics, optionally filtered by name."""
        if name:
            return [m for m in self._metrics if m.name == name]
        return self._metrics.copy()

    def clear(self) -> None:
        """Clear all metrics."""
        self._metrics.clear()
        self._counters.clear()
        self._gauges.clear()


# =============================================================================
# Service Container (for dependency injection in tests)
# =============================================================================

class MockServiceContainer:
    """
    Container for all mock services.

    Provides a single point of access to all mock services for tests.
    """

    def __init__(self):
        self.http = MockHTTPClient()
        self.email = MockEmailService()
        self.webhook = MockWebhookService()
        self.time = MockTimeProvider()
        self.cache = MockCacheService()
        self.metrics = MockMetricsService()

    def reset_all(self) -> None:
        """Reset all services to initial state."""
        self.http.reset()
        self.email.clear()
        self.webhook.clear()
        self.time.unfreeze()
        self.cache.clear()
        self.metrics.clear()

    def set_all_failures(self, should_fail: bool) -> None:
        """Set all services to fail or succeed."""
        if should_fail:
            self.http.set_failure_mode(FailureMode.HTTP_500)
        else:
            self.http.set_failure_mode(FailureMode.NONE)
        self.email.set_failure(should_fail)
        self.webhook.set_failure(should_fail)
        self.cache.set_failure(should_fail)


# =============================================================================
# Factory Functions
# =============================================================================

def create_mock_http_client(
    latency_ms: int = 0,
    failure_mode: FailureMode = FailureMode.NONE,
    failure_rate: float = 1.0,
) -> MockHTTPClient:
    """Factory function for mock HTTP client."""
    client = MockHTTPClient(
        latency_config=LatencyConfig(min_ms=latency_ms, max_ms=latency_ms),
    )
    client.set_failure_mode(failure_mode, failure_rate)
    return client


def create_service_container() -> MockServiceContainer:
    """Factory function for service container."""
    return MockServiceContainer()
