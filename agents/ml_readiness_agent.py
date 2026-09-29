"""ML Readiness Agent: Diagnoses machine learning feasibility without automatic model training."""

from __future__ import annotations

from agents.base import BaseAgent
from agents.schemas import AgentResult
from tools.registry import registry
from workflow.events import ActorType, EventType
from workflow.state import WorkflowStage, WorkflowState


class MLReadinessAgent(BaseAgent):
    name = "MLReadinessAgent"
    description = "Evaluates data suitability for machine learning, detecting leakage, imbalance, and encoding needs."

    def run(self, state: WorkflowState) -> AgentResult:
        df = state.active_dataset
        if state.profile_result is None:
            state.profile_result = registry.execute("profile_dataset", df=df)
        profile = state.profile_result

        if state.quality_result is None:
            state.quality_result = registry.execute(
                "assess_quality", df=df, profile=profile
            )
        quality = state.quality_result

        if state.eda_result is None:
            state.eda_result = registry.execute(
                "run_eda",
                df=df,
                numerical_columns=profile.numerical_columns,
                categorical_columns=profile.categorical_columns,
            )
        eda = state.eda_result

        # Run ML readiness evaluation via registered tool
        ml_result = registry.execute(
            "assess_ml_readiness",
            df=df,
            profile=profile,
            quality=quality,
            eda=eda,
            target_column=None,
        )
        state.ml_readiness_result = ml_result

        event = state.record_audit(
            actor_type=ActorType.AGENT,
            actor_name=self.name,
            event_type=EventType.ML_READINESS_COMPLETED,
            message=f"ML readiness assessed: score {ml_result.readiness_score:.1f}/100. Inferred task: {ml_result.inferred_task}.",
            metadata={
                "readiness_score": ml_result.readiness_score,
                "inferred_task": ml_result.inferred_task,
                "issue_count": len(ml_result.issues),
            },
        )

        return AgentResult(
            agent_name=self.name,
            success=True,
            message=f"ML readiness diagnosis complete (Score: {ml_result.readiness_score:.1f}/100). No models trained.",
            events=[event],
            next_stage=WorkflowStage.INSIGHT_GENERATION,
        )
