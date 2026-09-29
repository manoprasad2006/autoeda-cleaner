"""Human-in-the-Loop (HITL) approval models and operations."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import uuid


class ApprovalStatus:
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EDITED = "edited"
    EXPIRED = "expired"


class RiskLevel:
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ApprovalRequest:
    id: str
    workflow_id: str
    proposal_id: str
    agent_name: str
    action_type: str
    description: str
    affected_columns: list[str]
    rows_affected: int
    risk_level: str
    status: str  # pending, approved, rejected, edited, expired
    reviewer: str | None = None
    reviewer_note: str | None = None
    edited_parameters: dict | None = None
    created_at: str = ""
    resolved_at: str | None = None

    @classmethod
    def create(
        cls,
        workflow_id: str,
        proposal_id: str,
        agent_name: str,
        action_type: str,
        description: str,
        affected_columns: list[str],
        rows_affected: int,
        risk_level: str = RiskLevel.LOW,
    ) -> ApprovalRequest:
        now = datetime.now(timezone.utc).isoformat()
        return cls(
            id=f"appr-{uuid.uuid4().hex[:8]}",
            workflow_id=workflow_id,
            proposal_id=proposal_id,
            agent_name=agent_name,
            action_type=action_type,
            description=description,
            affected_columns=list(affected_columns),
            rows_affected=rows_affected,
            risk_level=risk_level,
            status=ApprovalStatus.PENDING,
            created_at=now,
        )


def create_approval_request(
    workflow_id: str,
    proposal_id: str,
    agent_name: str,
    action_type: str,
    description: str,
    affected_columns: list[str],
    rows_affected: int,
    risk_level: str = RiskLevel.LOW,
) -> ApprovalRequest:
    return ApprovalRequest.create(
        workflow_id=workflow_id,
        proposal_id=proposal_id,
        agent_name=agent_name,
        action_type=action_type,
        description=description,
        affected_columns=affected_columns,
        rows_affected=rows_affected,
        risk_level=risk_level,
    )


def approve_request(
    approval: ApprovalRequest,
    reviewer: str = "human_user",
    note: str | None = None,
) -> ApprovalRequest:
    if approval.status != ApprovalStatus.PENDING:
        raise ValueError(
            f"Cannot approve request in status '{approval.status}'. Must be '{ApprovalStatus.PENDING}'."
        )
    approval.status = ApprovalStatus.APPROVED
    approval.reviewer = reviewer
    approval.reviewer_note = note
    approval.resolved_at = datetime.now(timezone.utc).isoformat()
    return approval


def reject_request(
    approval: ApprovalRequest,
    reviewer: str = "human_user",
    note: str | None = None,
) -> ApprovalRequest:
    if approval.status != ApprovalStatus.PENDING:
        raise ValueError(
            f"Cannot reject request in status '{approval.status}'. Must be '{ApprovalStatus.PENDING}'."
        )
    approval.status = ApprovalStatus.REJECTED
    approval.reviewer = reviewer
    approval.reviewer_note = note
    approval.resolved_at = datetime.now(timezone.utc).isoformat()
    return approval


def edit_and_approve_request(
    approval: ApprovalRequest,
    edited_parameters: dict,
    reviewer: str = "human_user",
    note: str | None = None,
) -> ApprovalRequest:
    if approval.status != ApprovalStatus.PENDING:
        raise ValueError(
            f"Cannot edit & approve request in status '{approval.status}'. Must be '{ApprovalStatus.PENDING}'."
        )
    approval.status = ApprovalStatus.EDITED
    approval.edited_parameters = edited_parameters
    approval.reviewer = reviewer
    approval.reviewer_note = note
    approval.resolved_at = datetime.now(timezone.utc).isoformat()
    return approval


def expire_request(
    approval: ApprovalRequest,
    reason: str = "Superseded or timeout",
) -> ApprovalRequest:
    if approval.status == ApprovalStatus.PENDING:
        approval.status = ApprovalStatus.EXPIRED
        approval.reviewer_note = reason
        approval.resolved_at = datetime.now(timezone.utc).isoformat()
    return approval


def get_pending_requests(approvals: list[ApprovalRequest]) -> list[ApprovalRequest]:
    return [a for a in approvals if a.status == ApprovalStatus.PENDING]


def get_approval_history(approvals: list[ApprovalRequest]) -> list[ApprovalRequest]:
    return [a for a in approvals if a.status != ApprovalStatus.PENDING]
