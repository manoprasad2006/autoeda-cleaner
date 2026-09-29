"""Schemas for agent results and proposals."""

from __future__ import annotations

from dataclasses import dataclass, field

from workflow.events import AuditEvent
from workflow.state import Proposal


@dataclass
class AgentResult:
    agent_name: str
    success: bool
    message: str
    proposals: list[Proposal] = field(default_factory=list)
    events: list[AuditEvent] = field(default_factory=list)
    next_stage: str | None = None
    error: str | None = None
