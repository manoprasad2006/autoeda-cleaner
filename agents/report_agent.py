"""Report Agent: Compiles verified intelligence into governed report proposals."""
from __future__ import annotations

from agents.base import BaseAgent
from agents.schemas import AgentResult
from tools.registry import registry
from workflow.approvals import RiskLevel
from workflow.events import ActorType, EventType
from workflow.state import Proposal, WorkflowStage, WorkflowState


class ReportAgent(BaseAgent):
    name = "ReportAgent"
    description = "Compiles approved findings, active dataset, and audit history into an export report proposal."

    def run(self, state: WorkflowState) -> AgentResult:
        # Generate the HTML report content via registered tool
        report_html = registry.execute("generate_report", state=state)
        state.generated_report = report_html

        # Formulate proposal for exporting report and dataset package
        p_export = Proposal.create(
            agent_name=self.name,
            action_type="export_dataset",
            title="Approve Report & Dataset Export",
            description=f"Approve compilation and export of approved active version ({state.active_version_id}) and governance report.",
            affected_columns=[],
            rows_affected=len(state.active_dataset),
            risk_level=RiskLevel.MEDIUM,
            parameters={"format": "zip", "version_id": state.active_version_id},
            before_summary="Report generated in staging memory.",
            expected_after_summary=f"Package active dataset '{state.dataset_filename}' ({len(state.active_dataset)} rows) and audit log for download.",
            requires_approval=True,
        )
        state.add_proposal(p_export)

        event = state.record_audit(
            actor_type=ActorType.AGENT,
            actor_name=self.name,
            event_type=EventType.REPORT_GENERATED,
            message="Intelligence report compiled. Awaiting human export approval.",
            metadata={"active_version": state.active_version_id, "rows": len(state.active_dataset)},
        )

        return AgentResult(
            agent_name=self.name,
            success=True,
            message="Report compiled. Paused for human export approval.",
            proposals=[p_export],
            events=[event],
            next_stage=WorkflowStage.WAITING_FOR_EXPORT_APPROVAL,
        )
