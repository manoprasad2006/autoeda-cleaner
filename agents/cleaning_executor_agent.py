"""Cleaning Executor Agent: Governed execution of approved cleaning proposals."""

from __future__ import annotations

from agents.base import BaseAgent
from agents.schemas import AgentResult
from tools.registry import registry
from workflow.approvals import ApprovalStatus
from workflow.events import ActorType, EventType
from workflow.state import WorkflowStage, WorkflowState


class CleaningExecutorAgent(BaseAgent):
    name = "CleaningExecutorAgent"
    description = "Executes only human-approved proposals using registered deterministic tools, preserving original data."

    def run(self, state: WorkflowState) -> AgentResult:
        executed_count = 0
        execution_messages: list[str] = []

        # Iterate over approvals to find approved or edited items
        for approval in state.approvals:
            if approval.status in (ApprovalStatus.APPROVED, ApprovalStatus.EDITED):
                proposal = state.get_proposal(approval.proposal_id)
                if proposal and proposal.status != "executed":
                    # Execute proposal through registered tool
                    _, diff_msg = registry.execute(
                        "execute_cleaning",
                        state=state,
                        proposal_id=proposal.id,
                        reviewer=approval.reviewer or "human_user",
                    )
                    executed_count += 1
                    execution_messages.append(f"✓ {proposal.title}: {diff_msg}")

        # If any cleaning executed, refresh profile and quality score
        if executed_count > 0:
            state.profile_result = registry.execute(
                "profile_dataset", state.active_dataset
            )
            state.quality_result = registry.execute(
                "assess_quality", state.active_dataset, state.profile_result
            )

        event = state.record_audit(
            actor_type=ActorType.AGENT,
            actor_name=self.name,
            event_type=EventType.CLEANING_EXECUTED,
            message=f"Executed {executed_count} approved proposal(s).",
            metadata={"executed_count": executed_count, "details": execution_messages},
        )

        return AgentResult(
            agent_name=self.name,
            success=True,
            message=f"Execution completed for {executed_count} approved proposal(s).",
            events=[event],
            next_stage=WorkflowStage.VISUALIZATION_AND_EDA,
        )
