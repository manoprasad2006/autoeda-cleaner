"""Audit event data models and standard event types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid


class ActorType:
    AGENT = "agent"
    HUMAN = "human"
    SYSTEM = "system"


class EventType:
    DATASET_IMPORTED = "Dataset imported"
    INTAKE_COMPLETED = "Intake completed"
    QUALITY_ANALYSIS_COMPLETED = "Quality analysis completed"
    PROPOSAL_CREATED = "Proposal created"
    APPROVAL_REQUESTED = "Approval requested"
    PROPOSAL_APPROVED = "Proposal approved"
    PROPOSAL_REJECTED = "Proposal rejected"
    PROPOSAL_EDITED = "Proposal edited"
    APPROVAL_EXPIRED = "Approval expired"
    CLEANING_EXECUTED = "Cleaning executed"
    CLEANING_ROLLED_BACK = "Cleaning rolled back"
    EDA_COMPLETED = "EDA completed"
    VISUALIZATION_RECOMMENDED = "Visualization recommended"
    ML_READINESS_COMPLETED = "ML readiness assessed"
    AI_INSIGHT_GENERATED = "AI insight generated"
    AI_OUTPUT_REVIEWED = "AI output reviewed"
    REPORT_GENERATED = "Report generated"
    EXPORT_APPROVED = "Export approved"
    EXPORT_COMPLETED = "Export completed"
    AGENT_FAILURE = "Agent failure"
    STAGE_TRANSITION = "Stage transitioned"


@dataclass
class AuditEvent:
    id: str
    workflow_id: str
    timestamp: str
    actor_type: str  # agent, human, system
    actor_name: str
    event_type: str
    message: str
    proposal_id: str | None = None
    approval_id: str | None = None
    metadata: dict = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        workflow_id: str,
        actor_type: str,
        actor_name: str,
        event_type: str,
        message: str,
        proposal_id: str | None = None,
        approval_id: str | None = None,
        metadata: dict | None = None,
    ) -> AuditEvent:
        return cls(
            id=str(uuid.uuid4())[:8],
            workflow_id=workflow_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            actor_type=actor_type,
            actor_name=actor_name,
            event_type=event_type,
            message=message,
            proposal_id=proposal_id,
            approval_id=approval_id,
            metadata=metadata or {},
        )
