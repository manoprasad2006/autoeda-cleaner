"""Shared workflow state with strict immutability and versioning."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid

import pandas as pd

from workflow.approvals import ApprovalRequest
from workflow.errors import RollbackError
from workflow.events import AuditEvent, ActorType, EventType


class WorkflowStatus:
    IDLE = "idle"
    RUNNING = "running"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"


class WorkflowStage:
    INTAKE = "intake"
    QUALITY = "quality"
    CLEANING_PLAN = "cleaning_plan"
    WAITING_FOR_CLEANING_APPROVAL = "waiting_for_cleaning_approval"
    CLEANING_EXECUTION = "cleaning_execution"
    VISUALIZATION_AND_EDA = "visualization_and_eda"
    ML_READINESS = "ml_readiness"
    INSIGHT_GENERATION = "insight_generation"
    WAITING_FOR_INSIGHT_REVIEW = "waiting_for_insight_review"
    REPORT_GENERATION = "report_generation"
    WAITING_FOR_EXPORT_APPROVAL = "waiting_for_export_approval"
    COMPLETED = "completed"


STAGE_ORDER: list[str] = [
    WorkflowStage.INTAKE,
    WorkflowStage.QUALITY,
    WorkflowStage.CLEANING_PLAN,
    WorkflowStage.WAITING_FOR_CLEANING_APPROVAL,
    WorkflowStage.CLEANING_EXECUTION,
    WorkflowStage.VISUALIZATION_AND_EDA,
    WorkflowStage.ML_READINESS,
    WorkflowStage.INSIGHT_GENERATION,
    WorkflowStage.WAITING_FOR_INSIGHT_REVIEW,
    WorkflowStage.REPORT_GENERATION,
    WorkflowStage.WAITING_FOR_EXPORT_APPROVAL,
    WorkflowStage.COMPLETED,
]


@dataclass
class DatasetVersion:
    version_id: str
    parent_version_id: str | None
    created_by: str
    created_at: str
    change_summary: str
    dataframe: pd.DataFrame


@dataclass
class Proposal:
    id: str
    agent_name: str
    action_type: str
    title: str
    description: str
    affected_columns: list[str]
    rows_affected: int
    risk_level: str
    parameters: dict
    before_summary: str
    expected_after_summary: str
    requires_approval: bool = True
    status: str = (
        "pending"  # pending, approved, rejected, edited, executed, rolled_back
    )
    created_at: str = ""
    executed_at: str | None = None

    @classmethod
    def create(
        cls,
        agent_name: str,
        action_type: str,
        title: str,
        description: str,
        affected_columns: list[str],
        rows_affected: int,
        risk_level: str,
        parameters: dict,
        before_summary: str,
        expected_after_summary: str,
        requires_approval: bool = True,
    ) -> Proposal:
        now = datetime.now(timezone.utc).isoformat()
        return cls(
            id=f"prop-{uuid.uuid4().hex[:8]}",
            agent_name=agent_name,
            action_type=action_type,
            title=title,
            description=description,
            affected_columns=list(affected_columns),
            rows_affected=rows_affected,
            risk_level=risk_level,
            parameters=dict(parameters),
            before_summary=before_summary,
            expected_after_summary=expected_after_summary,
            requires_approval=requires_approval,
            status="pending",
            created_at=now,
        )


@dataclass
class AgentRun:
    run_id: str
    agent_name: str
    stage: str
    started_at: str
    completed_at: str | None = None
    success: bool = True
    message: str = ""
    error: str | None = None


@dataclass
class WorkflowState:
    workflow_id: str
    dataset_id: str
    dataset_filename: str
    original_dataset: pd.DataFrame
    active_dataset: pd.DataFrame
    dataset_metadata: Any = None

    current_stage: str = WorkflowStage.INTAKE
    status: str = WorkflowStatus.IDLE

    agent_runs: list[AgentRun] = field(default_factory=list)
    proposals: list[Proposal] = field(default_factory=list)
    approvals: list[ApprovalRequest] = field(default_factory=list)
    audit_events: list[AuditEvent] = field(default_factory=list)

    quality_result: Any = None
    profile_result: Any = None
    eda_result: Any = None
    ml_readiness_result: Any = None

    generated_insights: str | None = None
    generated_report: str | None = None

    created_at: str = ""
    updated_at: str = ""

    versions: list[DatasetVersion] = field(default_factory=list)
    active_version_id: str = "v0"
    candidate_dataset: pd.DataFrame | None = None
    step_count: int = 0
    max_steps: int = 50
    error_message: str | None = None

    # Transient storage for visual recommendations & intake findings
    visualization_recommendations: list[dict] = field(default_factory=list)
    intake_findings: list[dict] = field(default_factory=list)
    intake_warnings: list[str] = field(default_factory=list)
    ai_insights_structured: list[dict] = field(default_factory=list)

    @classmethod
    def create_initial(
        cls,
        df: pd.DataFrame,
        filename: str,
        metadata: Any = None,
        workflow_id: str | None = None,
    ) -> WorkflowState:
        now = datetime.now(timezone.utc).isoformat()
        w_id = workflow_id or f"wf-{uuid.uuid4().hex[:8]}"
        d_id = f"ds-{uuid.uuid4().hex[:8]}"

        # Always take immutable deep copies of the initial dataframe
        orig_copy = df.copy(deep=True)
        active_copy = df.copy(deep=True)

        initial_version = DatasetVersion(
            version_id="v0",
            parent_version_id=None,
            created_by="system",
            created_at=now,
            change_summary="Initial dataset loaded",
            dataframe=active_copy.copy(deep=True),
        )

        state = cls(
            workflow_id=w_id,
            dataset_id=d_id,
            dataset_filename=filename,
            original_dataset=orig_copy,
            active_dataset=active_copy,
            dataset_metadata=metadata,
            current_stage=WorkflowStage.INTAKE,
            status=WorkflowStatus.IDLE,
            created_at=now,
            updated_at=now,
            versions=[initial_version],
            active_version_id="v0",
        )

        state.record_audit(
            actor_type=ActorType.SYSTEM,
            actor_name="IntakeSystem",
            event_type=EventType.DATASET_IMPORTED,
            message=f"Dataset '{filename}' imported ({len(df)} rows, {len(df.columns)} columns).",
            metadata={"filename": filename, "rows": len(df), "cols": len(df.columns)},
        )
        return state

    def record_audit(
        self,
        actor_type: str,
        actor_name: str,
        event_type: str,
        message: str,
        proposal_id: str | None = None,
        approval_id: str | None = None,
        metadata: dict | None = None,
    ) -> AuditEvent:
        event = AuditEvent.create(
            workflow_id=self.workflow_id,
            actor_type=actor_type,
            actor_name=actor_name,
            event_type=event_type,
            message=message,
            proposal_id=proposal_id,
            approval_id=approval_id,
            metadata=metadata or {},
        )
        self.audit_events.append(event)
        self.updated_at = datetime.now(timezone.utc).isoformat()
        return event

    def add_proposal(self, proposal: Proposal) -> ApprovalRequest:
        self.proposals.append(proposal)
        approval = ApprovalRequest.create(
            workflow_id=self.workflow_id,
            proposal_id=proposal.id,
            agent_name=proposal.agent_name,
            action_type=proposal.action_type,
            description=proposal.description,
            affected_columns=proposal.affected_columns,
            rows_affected=proposal.rows_affected,
            risk_level=proposal.risk_level,
        )
        self.approvals.append(approval)
        self.record_audit(
            actor_type=ActorType.AGENT,
            actor_name=proposal.agent_name,
            event_type=EventType.PROPOSAL_CREATED,
            message=f"Created proposal '{proposal.title}' (Risk: {proposal.risk_level.upper()})",
            proposal_id=proposal.id,
            approval_id=approval.id,
            metadata={
                "action_type": proposal.action_type,
                "params": proposal.parameters,
            },
        )
        self.record_audit(
            actor_type=ActorType.SYSTEM,
            actor_name="ApprovalSystem",
            event_type=EventType.APPROVAL_REQUESTED,
            message=f"Approval requested for proposal '{proposal.title}'",
            proposal_id=proposal.id,
            approval_id=approval.id,
        )
        self.updated_at = datetime.now(timezone.utc).isoformat()
        return approval

    def get_proposal(self, proposal_id: str) -> Proposal | None:
        for p in self.proposals:
            if p.id == proposal_id:
                return p
        return None

    def get_approval(self, approval_id: str) -> ApprovalRequest | None:
        for a in self.approvals:
            if a.id == approval_id:
                return a
        return None

    def get_approval_for_proposal(self, proposal_id: str) -> ApprovalRequest | None:
        for a in self.approvals:
            if a.proposal_id == proposal_id:
                return a
        return None

    def commit_version(
        self,
        new_df: pd.DataFrame,
        created_by: str,
        change_summary: str,
    ) -> DatasetVersion:
        """Create a new dataset version and make it the active dataset.
        The original_dataset remains strictly immutable."""
        now = datetime.now(timezone.utc).isoformat()
        next_ver_id = f"v{len(self.versions)}"
        version = DatasetVersion(
            version_id=next_ver_id,
            parent_version_id=self.active_version_id,
            created_by=created_by,
            created_at=now,
            change_summary=change_summary,
            dataframe=new_df.copy(deep=True),
        )
        self.versions.append(version)
        self.active_version_id = next_ver_id
        self.active_dataset = new_df.copy(deep=True)
        self.candidate_dataset = None
        self.updated_at = now
        return version

    def can_rollback(self) -> bool:
        current_version = self.get_version(self.active_version_id)
        return (
            current_version is not None
            and current_version.parent_version_id is not None
        )

    def get_version(self, version_id: str) -> DatasetVersion | None:
        for v in self.versions:
            if v.version_id == version_id:
                return v
        return None

    def rollback(self, reviewer: str = "human_user") -> DatasetVersion:
        """Roll back active dataset to parent version."""
        current_version = self.get_version(self.active_version_id)
        if not current_version or not current_version.parent_version_id:
            raise RollbackError("Cannot rollback initial dataset version.")

        parent_version = self.get_version(current_version.parent_version_id)
        if not parent_version:
            raise RollbackError(
                f"Parent version '{current_version.parent_version_id}' not found."
            )

        # Create a new version representing the rollback
        now = datetime.now(timezone.utc).isoformat()
        rollback_ver_id = f"v{len(self.versions)}"
        summary = (
            f"Rollback from {self.active_version_id} to {parent_version.version_id}"
        )
        rb_version = DatasetVersion(
            version_id=rollback_ver_id,
            parent_version_id=self.active_version_id,
            created_by=reviewer,
            created_at=now,
            change_summary=summary,
            dataframe=parent_version.dataframe.copy(deep=True),
        )
        self.versions.append(rb_version)
        self.active_version_id = rollback_ver_id
        self.active_dataset = parent_version.dataframe.copy(deep=True)
        self.candidate_dataset = None
        self.updated_at = now

        self.record_audit(
            actor_type=ActorType.HUMAN,
            actor_name=reviewer,
            event_type=EventType.CLEANING_ROLLED_BACK,
            message=summary,
            metadata={"restored_version": parent_version.version_id},
        )
        return rb_version
