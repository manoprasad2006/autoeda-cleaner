"""Insight Agent: Generates structured business hypotheses and AI summaries."""

from __future__ import annotations

from agents.base import BaseAgent
from agents.schemas import AgentResult
from tools.registry import registry
from utils.config import load_settings
from workflow.approvals import RiskLevel
from workflow.events import ActorType, EventType
from workflow.state import Proposal, WorkflowStage, WorkflowState


class InsightAgent(BaseAgent):
    name = "InsightAgent"
    description = "Generates structured hypotheses from summary statistics, avoiding causal claims and code execution."

    def run(self, state: WorkflowState) -> AgentResult:
        settings = load_settings()
        api_key = settings.gemini_api_key
        model = settings.gemini_model or "gemini-2.5-flash"

        # Generate summary and business insights using registered tools
        summary = registry.execute(
            "generate_ai_summary",
            state=state,
            api_key=api_key,
            model=model,
        )
        insights = registry.execute(
            "generate_business_insights",
            state=state,
            api_key=api_key,
            model=model,
        )

        state.generated_insights = summary
        state.ai_insights_structured = insights

        # Formulate proposal for human review of AI outputs
        p_review = Proposal.create(
            agent_name=self.name,
            action_type="review_ai_insights",
            title="Review AI Insights & Hypotheses",
            description="Review AI-generated hypotheses before including in executive reporting.",
            affected_columns=[],
            rows_affected=0,
            risk_level=RiskLevel.LOW,
            parameters={"insight_count": len(insights)},
            before_summary="Generated unverified hypotheses from summary statistics.",
            expected_after_summary="User verifies and approves insights for reporting.",
            requires_approval=True,
        )
        state.add_proposal(p_review)

        event = state.record_audit(
            actor_type=ActorType.AGENT,
            actor_name=self.name,
            event_type=EventType.AI_INSIGHT_GENERATED,
            message=f"Generated {len(insights)} hypothesis item(s). Human review required.",
            metadata={"insight_count": len(insights), "has_api_key": bool(api_key)},
        )

        return AgentResult(
            agent_name=self.name,
            success=True,
            message=f"Generated {len(insights)} structured insight(s). Paused for human review.",
            proposals=[p_review],
            events=[event],
            next_stage=WorkflowStage.WAITING_FOR_INSIGHT_REVIEW,
        )
