"""Audit log tracking and reporting utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
import pandas as pd

from workflow.events import AuditEvent


@dataclass
class AuditTrail:
    events: list[AuditEvent] = field(default_factory=list)

    def record(
        self,
        workflow_id: str,
        actor_type: str,
        actor_name: str,
        event_type: str,
        message: str,
        proposal_id: str | None = None,
        approval_id: str | None = None,
        metadata: dict | None = None,
    ) -> AuditEvent:
        event = AuditEvent.create(
            workflow_id=workflow_id,
            actor_type=actor_type,
            actor_name=actor_name,
            event_type=event_type,
            message=message,
            proposal_id=proposal_id,
            approval_id=approval_id,
            metadata=metadata or {},
        )
        self.events.append(event)
        return event

    def add(self, event: AuditEvent) -> None:
        self.events.append(event)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.events:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "actor_type",
                    "actor_name",
                    "event_type",
                    "message",
                    "proposal_id",
                    "approval_id",
                ]
            )
        return pd.DataFrame(
            [
                {
                    "timestamp": e.timestamp,
                    "actor_type": e.actor_type,
                    "actor_name": e.actor_name,
                    "event_type": e.event_type,
                    "message": e.message,
                    "proposal_id": e.proposal_id or "-",
                    "approval_id": e.approval_id or "-",
                }
                for e in self.events
            ]
        )

    def get_by_proposal(self, proposal_id: str) -> list[AuditEvent]:
        return [e for e in self.events if e.proposal_id == proposal_id]

    def get_by_actor(self, actor_name: str) -> list[AuditEvent]:
        return [e for e in self.events if e.actor_name == actor_name]

    def recent(self, n: int = 10) -> list[AuditEvent]:
        return self.events[-n:]
