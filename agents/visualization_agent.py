"""Visualization Agent: Recommends semantically sound charts and aggregations."""

from __future__ import annotations

from agents.base import BaseAgent
from agents.schemas import AgentResult
from tools.registry import registry
from workflow.events import ActorType, EventType
from workflow.state import WorkflowStage, WorkflowState


class VisualizationAgent(BaseAgent):
    name = "VisualizationAgent"
    description = "Inspects profile and EDA results, recommends semantically sound chart types and aggregations."

    def run(self, state: WorkflowState) -> AgentResult:
        df = state.active_dataset
        if state.profile_result is None:
            state.profile_result = registry.execute("profile_dataset", df)
        profile = state.profile_result

        # Run EDA if not already present
        if state.eda_result is None:
            state.eda_result = registry.execute(
                "run_eda",
                df=df,
                numerical_columns=profile.numerical_columns,
                categorical_columns=profile.categorical_columns,
            )
        eda = state.eda_result

        # Recommend visualizations through registered tool
        recommendations = registry.execute(
            "recommend_visualizations",
            df=df,
            profile=profile,
            eda=eda,
        )
        state.visualization_recommendations = recommendations

        event = state.record_audit(
            actor_type=ActorType.AGENT,
            actor_name=self.name,
            event_type=EventType.VISUALIZATION_RECOMMENDED,
            message=f"Generated {len(recommendations)} semantically validated chart recommendation(s).",
            metadata={"recommendation_count": len(recommendations)},
        )

        return AgentResult(
            agent_name=self.name,
            success=True,
            message=f"Recommended {len(recommendations)} charts with semantic aggregation safety.",
            events=[event],
            next_stage=WorkflowStage.ML_READINESS,
        )
