#!/usr/bin/env python3
"""
Intervention System
===================
Allows human oversight and intervention in the autonomous trading system.

Key Features:
- Configurable checkpoints (review points) in the workflow
- Multiple intervention modes: auto-approve, require-approval, notify-only
- File-based approval mechanism (touch files to approve/reject)
- Alert integration for notifications
- Timeout handling with configurable default actions
- Decision logging for audit trail

Usage:
    # In code:
    intervention = InterventionManager()

    # Request approval before applying GA results
    if intervention.request_approval(
        checkpoint="apply_ga_results",
        context={"strategy": "momentum", "improvement": 0.05},
        timeout_minutes=30
    ):
        apply_results()
    else:
        skip_results()

    # CLI:
    python -m orchestration.intervention --approve ga_results
    python -m orchestration.intervention --reject ga_results --reason "Too aggressive"
    python -m orchestration.intervention --status
"""

import json
import logging
import os
import time
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
import hashlib

logger = logging.getLogger(__name__)


class InterventionMode(Enum):
    """How the system handles intervention points."""
    AUTONOMOUS = "autonomous"          # Never pause, log decisions only
    NOTIFY_ONLY = "notify_only"        # Send alerts but don't wait
    APPROVAL_REQUIRED = "approval_required"  # Wait for approval
    REVIEW_RECOMMENDED = "review_recommended"  # Wait with timeout, then proceed


class InterventionResult(Enum):
    """Result of an intervention request."""
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT_PROCEED = "timeout_proceed"
    TIMEOUT_ABORT = "timeout_abort"
    SKIPPED = "skipped"  # Autonomous mode


class CheckpointPriority(Enum):
    """Priority levels for checkpoints."""
    LOW = "low"           # Informational only
    MEDIUM = "medium"     # Review recommended
    HIGH = "high"         # Approval strongly recommended
    CRITICAL = "critical" # Approval required


@dataclass
class InterventionRequest:
    """A pending intervention request."""
    request_id: str
    checkpoint: str
    priority: CheckpointPriority
    context: Dict[str, Any]
    created_at: datetime
    expires_at: datetime
    default_action: str  # "approve" or "reject"
    status: str = "pending"  # pending, approved, rejected, expired
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None  # "user", "timeout", "system"
    rejection_reason: Optional[str] = None


@dataclass
class InterventionConfig:
    """Configuration for the intervention system."""
    mode: InterventionMode = InterventionMode.REVIEW_RECOMMENDED
    default_timeout_minutes: int = 30
    default_action_on_timeout: str = "approve"  # "approve" or "reject"
    alert_on_pending: bool = True
    alert_on_timeout: bool = True
    log_all_decisions: bool = True
    approval_dir: Path = field(default_factory=lambda: Path("./intervention"))

    # Checkpoint-specific overrides
    checkpoint_config: Dict[str, Dict[str, Any]] = field(default_factory=dict)


# Default checkpoint configurations
DEFAULT_CHECKPOINTS = {
    # Nightly research checkpoints
    "pre_research": {
        "priority": CheckpointPriority.LOW,
        "description": "Before starting nightly research",
        "default_timeout_minutes": 5,
        "default_action": "approve",
    },
    "apply_ga_results": {
        "priority": CheckpointPriority.MEDIUM,
        "description": "Before applying GA optimization results",
        "default_timeout_minutes": 30,
        "default_action": "approve",
    },
    "apply_discovered_strategy": {
        "priority": CheckpointPriority.HIGH,
        "description": "Before adding a newly discovered strategy",
        "default_timeout_minutes": 60,
        "default_action": "reject",  # Conservative default
    },
    "regime_change_rebalance": {
        "priority": CheckpointPriority.MEDIUM,
        "description": "Before rebalancing due to regime change",
        "default_timeout_minutes": 15,
        "default_action": "approve",
    },

    # Trading checkpoints
    "large_position_change": {
        "priority": CheckpointPriority.HIGH,
        "description": "Before making a large position change (>10% of portfolio)",
        "default_timeout_minutes": 15,
        "default_action": "reject",
    },
    "new_strategy_activation": {
        "priority": CheckpointPriority.HIGH,
        "description": "Before activating a new strategy in live trading",
        "default_timeout_minutes": 60,
        "default_action": "reject",
    },
    "risk_limit_override": {
        "priority": CheckpointPriority.CRITICAL,
        "description": "Before overriding risk limits",
        "default_timeout_minutes": 30,
        "default_action": "reject",
    },

    # System checkpoints
    "config_change": {
        "priority": CheckpointPriority.MEDIUM,
        "description": "Before applying configuration changes",
        "default_timeout_minutes": 15,
        "default_action": "approve",
    },
    "emergency_shutdown": {
        "priority": CheckpointPriority.CRITICAL,
        "description": "Before emergency system shutdown",
        "default_timeout_minutes": 5,
        "default_action": "approve",  # Safety first
    },
}


class InterventionManager:
    """
    Manages human intervention points in the trading system.

    The manager provides checkpoints where the system can pause
    and wait for human approval before proceeding.
    """

    def __init__(self, config: Optional[InterventionConfig] = None):
        """
        Initialize the intervention manager.

        Args:
            config: Intervention configuration. If None, uses defaults.
        """
        self.config = config or InterventionConfig()
        self.pending_requests: Dict[str, InterventionRequest] = {}
        self.decision_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

        # Ensure approval directory exists
        self.config.approval_dir.mkdir(parents=True, exist_ok=True)

        # Try to import alert manager
        try:
            from execution.alerts import AlertManager
            self._alert_manager = AlertManager()
        except ImportError:
            self._alert_manager = None
            logger.warning("AlertManager not available, notifications disabled")

        # Load checkpoint configs
        self._checkpoint_configs = {**DEFAULT_CHECKPOINTS}
        self._checkpoint_configs.update(self.config.checkpoint_config)

        logger.info(f"InterventionManager initialized (mode={self.config.mode.value})")

    def request_approval(
        self,
        checkpoint: str,
        context: Optional[Dict[str, Any]] = None,
        timeout_minutes: Optional[int] = None,
        priority: Optional[CheckpointPriority] = None,
        callback: Optional[Callable[[InterventionResult], None]] = None,
    ) -> bool:
        """
        Request approval at a checkpoint.

        Args:
            checkpoint: Name of the checkpoint
            context: Additional context for the decision
            timeout_minutes: Override default timeout
            priority: Override default priority
            callback: Optional callback when decision is made

        Returns:
            True if approved, False if rejected
        """
        # Get checkpoint config
        cp_config = self._checkpoint_configs.get(checkpoint, {})

        # Determine priority
        if priority is None:
            priority = cp_config.get("priority", CheckpointPriority.MEDIUM)

        # Determine timeout
        if timeout_minutes is None:
            timeout_minutes = cp_config.get(
                "default_timeout_minutes",
                self.config.default_timeout_minutes
            )

        # Determine default action
        default_action = cp_config.get(
            "default_action",
            self.config.default_action_on_timeout
        )

        # Handle autonomous mode
        if self.config.mode == InterventionMode.AUTONOMOUS:
            result = InterventionResult.SKIPPED
            self._log_decision(checkpoint, context, result, "autonomous_mode")
            return True

        # Create request
        request_id = self._generate_request_id(checkpoint)
        now = datetime.now()

        request = InterventionRequest(
            request_id=request_id,
            checkpoint=checkpoint,
            priority=priority,
            context=context or {},
            created_at=now,
            expires_at=now + timedelta(minutes=timeout_minutes),
            default_action=default_action,
        )

        # Store request
        with self._lock:
            self.pending_requests[request_id] = request

        # Create approval files
        self._create_approval_files(request)

        # Send notification
        if self.config.alert_on_pending:
            self._send_notification(request, "pending")

        # Handle notify-only mode
        if self.config.mode == InterventionMode.NOTIFY_ONLY:
            result = InterventionResult.SKIPPED
            self._log_decision(checkpoint, context, result, "notify_only_mode")
            return True

        # Wait for approval
        result = self._wait_for_decision(request)

        # Cleanup
        self._cleanup_approval_files(request)

        with self._lock:
            if request_id in self.pending_requests:
                del self.pending_requests[request_id]

        # Execute callback
        if callback:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Callback error: {e}")

        # Log decision
        self._log_decision(checkpoint, context, result, request.resolved_by)

        return result in (InterventionResult.APPROVED, InterventionResult.TIMEOUT_PROCEED)

    def approve(self, request_id: str, approved_by: str = "user") -> bool:
        """
        Approve a pending request.

        Args:
            request_id: The request ID to approve
            approved_by: Who approved (user, system, etc.)

        Returns:
            True if approval was recorded
        """
        with self._lock:
            if request_id not in self.pending_requests:
                # Try to find by checkpoint name
                for rid, req in self.pending_requests.items():
                    if req.checkpoint == request_id:
                        request_id = rid
                        break
                else:
                    logger.warning(f"Request {request_id} not found")
                    return False

            request = self.pending_requests[request_id]
            request.status = "approved"
            request.resolved_at = datetime.now()
            request.resolved_by = approved_by

            # Write approval file
            approval_file = self.config.approval_dir / f"{request_id}.approved"
            approval_file.write_text(json.dumps({
                "approved_by": approved_by,
                "timestamp": datetime.now().isoformat()
            }))

            logger.info(f"Request {request_id} approved by {approved_by}")
            return True

    def reject(
        self,
        request_id: str,
        reason: str = "",
        rejected_by: str = "user"
    ) -> bool:
        """
        Reject a pending request.

        Args:
            request_id: The request ID to reject
            reason: Reason for rejection
            rejected_by: Who rejected

        Returns:
            True if rejection was recorded
        """
        with self._lock:
            if request_id not in self.pending_requests:
                # Try to find by checkpoint name
                for rid, req in self.pending_requests.items():
                    if req.checkpoint == request_id:
                        request_id = rid
                        break
                else:
                    logger.warning(f"Request {request_id} not found")
                    return False

            request = self.pending_requests[request_id]
            request.status = "rejected"
            request.resolved_at = datetime.now()
            request.resolved_by = rejected_by
            request.rejection_reason = reason

            # Write rejection file
            rejection_file = self.config.approval_dir / f"{request_id}.rejected"
            rejection_file.write_text(json.dumps({
                "rejected_by": rejected_by,
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            }))

            logger.info(f"Request {request_id} rejected by {rejected_by}: {reason}")
            return True

    def get_pending_requests(self) -> List[InterventionRequest]:
        """Get all pending requests."""
        with self._lock:
            return list(self.pending_requests.values())

    def get_decision_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent decision history."""
        return self.decision_history[-limit:]

    def set_mode(self, mode: InterventionMode):
        """Change the intervention mode."""
        old_mode = self.config.mode
        self.config.mode = mode
        logger.info(f"Intervention mode changed: {old_mode.value} -> {mode.value}")

    def _generate_request_id(self, checkpoint: str) -> str:
        """Generate a unique request ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_input = f"{checkpoint}_{timestamp}_{os.getpid()}"
        short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"{checkpoint}_{timestamp}_{short_hash}"

    def _create_approval_files(self, request: InterventionRequest):
        """Create files for file-based approval."""
        # Create a pending file with request details
        pending_file = self.config.approval_dir / f"{request.request_id}.pending"
        pending_file.write_text(json.dumps({
            "request_id": request.request_id,
            "checkpoint": request.checkpoint,
            "priority": request.priority.value,
            "context": request.context,
            "created_at": request.created_at.isoformat(),
            "expires_at": request.expires_at.isoformat(),
            "default_action": request.default_action,
            "instructions": (
                f"To APPROVE: touch {request.request_id}.approved\n"
                f"To REJECT: touch {request.request_id}.rejected\n"
                f"Or use: python -m orchestration.intervention --approve {request.checkpoint}\n"
                f"Timeout at: {request.expires_at.isoformat()}"
            )
        }, indent=2))

        logger.info(f"Created approval request: {pending_file}")

    def _cleanup_approval_files(self, request: InterventionRequest):
        """Remove approval files after decision."""
        for suffix in [".pending", ".approved", ".rejected"]:
            file_path = self.config.approval_dir / f"{request.request_id}{suffix}"
            if file_path.exists():
                file_path.unlink()

    def _wait_for_decision(self, request: InterventionRequest) -> InterventionResult:
        """Wait for a decision on the request."""
        check_interval = 5  # seconds

        while datetime.now() < request.expires_at:
            # Check for approval file
            approval_file = self.config.approval_dir / f"{request.request_id}.approved"
            if approval_file.exists():
                request.status = "approved"
                request.resolved_at = datetime.now()
                request.resolved_by = "user"
                return InterventionResult.APPROVED

            # Check for rejection file
            rejection_file = self.config.approval_dir / f"{request.request_id}.rejected"
            if rejection_file.exists():
                request.status = "rejected"
                request.resolved_at = datetime.now()
                request.resolved_by = "user"
                try:
                    data = json.loads(rejection_file.read_text())
                    request.rejection_reason = data.get("reason", "")
                except (json.JSONDecodeError, IOError):
                    pass  # Use empty reason if file can't be parsed
                return InterventionResult.REJECTED

            # Check if request was resolved via API
            with self._lock:
                if request.status != "pending":
                    if request.status == "approved":
                        return InterventionResult.APPROVED
                    else:
                        return InterventionResult.REJECTED

            time.sleep(check_interval)

        # Timeout reached
        request.status = "expired"
        request.resolved_at = datetime.now()
        request.resolved_by = "timeout"

        if self.config.alert_on_timeout:
            self._send_notification(request, "timeout")

        if request.default_action == "approve":
            logger.warning(f"Request {request.request_id} timed out, proceeding (default)")
            return InterventionResult.TIMEOUT_PROCEED
        else:
            logger.warning(f"Request {request.request_id} timed out, aborting (default)")
            return InterventionResult.TIMEOUT_ABORT

    def _send_notification(self, request: InterventionRequest, event: str):
        """Send notification about intervention request."""
        if self._alert_manager is None:
            return

        cp_config = self._checkpoint_configs.get(request.checkpoint, {})
        description = cp_config.get("description", request.checkpoint)

        if event == "pending":
            message = (
                f"INTERVENTION REQUIRED: {description}\n"
                f"Priority: {request.priority.value.upper()}\n"
                f"Checkpoint: {request.checkpoint}\n"
                f"Context: {json.dumps(request.context, indent=2)}\n"
                f"Expires: {request.expires_at.strftime('%H:%M:%S')}\n"
                f"Default: {request.default_action}\n\n"
                f"Approve: python -m orchestration.intervention --approve {request.checkpoint}\n"
                f"Reject: python -m orchestration.intervention --reject {request.checkpoint}"
            )
            level = "warning" if request.priority in (CheckpointPriority.HIGH, CheckpointPriority.CRITICAL) else "info"
        else:  # timeout
            message = (
                f"INTERVENTION TIMEOUT: {description}\n"
                f"Action taken: {request.default_action}\n"
                f"Checkpoint: {request.checkpoint}"
            )
            level = "warning"

        try:
            self._alert_manager.send_alert(message, level=level)
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")

    def _log_decision(
        self,
        checkpoint: str,
        context: Optional[Dict[str, Any]],
        result: InterventionResult,
        resolved_by: Optional[str]
    ):
        """Log a decision for audit trail."""
        if not self.config.log_all_decisions:
            return

        entry = {
            "timestamp": datetime.now().isoformat(),
            "checkpoint": checkpoint,
            "context": context or {},
            "result": result.value,
            "resolved_by": resolved_by,
        }

        self.decision_history.append(entry)

        # Also log to file
        log_file = self.config.approval_dir / "decision_log.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        logger.info(f"Decision logged: {checkpoint} -> {result.value}")

    def print_status(self):
        """Print current intervention status."""
        print("\n" + "=" * 60)
        print("INTERVENTION SYSTEM STATUS")
        print("=" * 60)
        print(f"Mode: {self.config.mode.value}")
        print(f"Approval Directory: {self.config.approval_dir}")
        print(f"Default Timeout: {self.config.default_timeout_minutes} minutes")
        print(f"Default Action: {self.config.default_action_on_timeout}")

        pending = self.get_pending_requests()
        if pending:
            print(f"\nPending Requests ({len(pending)}):")
            print("-" * 40)
            for req in pending:
                time_left = req.expires_at - datetime.now()
                print(f"  [{req.priority.value.upper()}] {req.checkpoint}")
                print(f"    ID: {req.request_id}")
                print(f"    Expires in: {time_left}")
                print(f"    Default: {req.default_action}")
                if req.context:
                    print(f"    Context: {json.dumps(req.context)}")
        else:
            print("\nNo pending requests.")

        recent = self.get_decision_history(10)
        if recent:
            print(f"\nRecent Decisions ({len(recent)}):")
            print("-" * 40)
            for dec in recent[-5:]:
                print(f"  {dec['timestamp'][:19]} | {dec['checkpoint']} | {dec['result']} | {dec['resolved_by']}")

        print("=" * 60 + "\n")


def main():
    """CLI for intervention management."""
    import argparse

    parser = argparse.ArgumentParser(description="Intervention System CLI")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--approve", type=str, help="Approve a pending request")
    parser.add_argument("--reject", type=str, help="Reject a pending request")
    parser.add_argument("--reason", type=str, default="", help="Reason for rejection")
    parser.add_argument("--list", action="store_true", help="List pending requests")
    parser.add_argument("--mode", type=str,
                        choices=["autonomous", "notify_only", "approval_required", "review_recommended"],
                        help="Set intervention mode")
    parser.add_argument("--history", action="store_true", help="Show decision history")

    args = parser.parse_args()

    manager = InterventionManager()

    if args.status or (not any([args.approve, args.reject, args.list, args.mode, args.history])):
        manager.print_status()

    if args.approve:
        if manager.approve(args.approve):
            print(f"Approved: {args.approve}")
        else:
            print(f"Failed to approve: {args.approve}")

    if args.reject:
        if manager.reject(args.reject, reason=args.reason):
            print(f"Rejected: {args.reject}")
        else:
            print(f"Failed to reject: {args.reject}")

    if args.list:
        pending = manager.get_pending_requests()
        if pending:
            for req in pending:
                print(f"{req.request_id}: {req.checkpoint} ({req.priority.value})")
        else:
            print("No pending requests")

    if args.mode:
        mode_map = {
            "autonomous": InterventionMode.AUTONOMOUS,
            "notify_only": InterventionMode.NOTIFY_ONLY,
            "approval_required": InterventionMode.APPROVAL_REQUIRED,
            "review_recommended": InterventionMode.REVIEW_RECOMMENDED,
        }
        manager.set_mode(mode_map[args.mode])
        print(f"Mode set to: {args.mode}")

    if args.history:
        history = manager.get_decision_history()
        for entry in history:
            print(f"{entry['timestamp']}: {entry['checkpoint']} -> {entry['result']}")


if __name__ == "__main__":
    main()
