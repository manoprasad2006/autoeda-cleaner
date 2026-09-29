"""Supervisor Agent: Orchestrates the multi-agent lifecycle and enforces governance checkpoints."""

from __future__ import annotations

from datetime import datetime, timezone
import uuid

from agents.base import BaseAgent
from agents.cleaning_executor_agent import CleaningExecutorAgent
from agents.cleaning_planner_agent import CleaningPlannerAgent
from agents.insight_agent import InsightAgent
from agents.intake_agent import IntakeAgent
from agents.ml_readiness_agent import MLReadinessAgent
from agents.quality_agent import QualityAgent
from agents.report_agent import ReportAgent
from agents.schemas import AgentResult
from agents.visualization_agent import VisualizationAgent
from workflow.approvals import ApprovalStatus
from workflow.errors import (
    InvalidStageTransitionError,
    MaxStepsExceededError,
)
from workflow.events import ActorType, EventType
from workflow.state import (
    AgentRun,
    WorkflowStage,
    WorkflowState,
    WorkflowStatus,
)

# Allowed direct transitions between workflow stages
ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    WorkflowStage.INTAKE: {WorkflowStage.QUALITY, WorkflowStage.INTAKE},
    WorkflowStage.QUALITY: {WorkflowStage.CLEANING_PLAN, WorkflowStage.QUALITY},
    WorkflowStage.CLEANING_PLAN: {
        WorkflowStage.WAITING_FOR_CLEANING_APPROVAL,
        WorkflowStage.VISUALIZATION_AND_EDA,
        WorkflowStage.CLEANING_PLAN,
    },
    WorkflowStage.WAITING_FOR_CLEANING_APPROVAL: {
        WorkflowStage.CLEANING_EXECUTION,
        WorkflowStage.VISUALIZATION_AND_EDA,
        WorkflowStage.WAITING_FOR_CLEANING_APPROVAL,
    },
    WorkflowStage.CLEANING_EXECUTION: {
        WorkflowStage.VISUALIZATION_AND_EDA,
        WorkflowStage.CLEANING_EXECUTION,
    },
    WorkflowStage.VISUALIZATION_AND_EDA: {
        WorkflowStage.ML_READINESS,
        WorkflowStage.VISUALIZATION_AND_EDA,
    },
    WorkflowStage.ML_READINESS: {
        WorkflowStage.INSIGHT_GENERATION,
        WorkflowStage.ML_READINESS,
    },
    WorkflowStage.INSIGHT_GENERATION: {
        WorkflowStage.WAITING_FOR_INSIGHT_REVIEW,
        WorkflowStage.WAITING_FOR_INSIGHT_REVIEW,
        WorkflowStage.INSIGHT_GENERATION,
    },
    WorkflowStage.WAITING_FOR_INSIGHT_REVIEW: {
        WorkflowStage.REPORT_GENERATION,
        WorkflowStage.WAITING_FOR_INSIGHT_REVIEW,
    },
    WorkflowStage.REPORT_GENERATION: {
        WorkflowStage.WAITING_FOR_EXPORT_APPROVAL,
        WorkflowStage.REPORT_GENERATION,
    },
    WorkflowStage.WAITING_FOR_EXPORT_APPROVAL: {
        WorkflowStage.COMPLETED,
        WorkflowStage.WAITING_FOR_EXPORT_APPROVAL,
    },
    WorkflowStage.COMPLETED: {WorkflowStage.COMPLETED},
}


class SupervisorAgent(BaseAgent):
    name = "SupervisorAgent"
    description = "Coordinates specialized agents, enforces stage transitions, and halts at Human-in-the-Loop checkpoints."

    def __init__(self) -> None:
        self.intake_agent = IntakeAgent()
        self.quality_agent = QualityAgent()
        self.cleaning_planner = CleaningPlannerAgent()
        self.cleaning_executor = CleaningExecutorAgent()
        self.visualization_agent = VisualizationAgent()
        self.ml_readiness_agent = MLReadinessAgent()
        self.insight_agent = InsightAgent()
        self.report_agent = ReportAgent()

    def run(self, state: WorkflowState) -> AgentResult:
        """Runs the workflow until the next HITL checkpoint, completion, or error."""
        return self.run_until_checkpoint(state)

    def transition_stage(self, state: WorkflowState, new_stage: str) -> None:
        """Validate and apply stage transition."""
        allowed = ALLOWED_TRANSITIONS.get(state.current_stage, set())
        if new_stage not in allowed:
            raise InvalidStageTransitionError(
                f"Illegal transition from '{state.current_stage}' to '{new_stage}'. Allowed: {allowed}"
            )
        old_stage = state.current_stage
        state.current_stage = new_stage
        state.record_audit(
            actor_type=ActorType.SYSTEM,
            actor_name=self.name,
            event_type=EventType.STAGE_TRANSITION,
            message=f"Stage transitioned from '{old_stage}' to '{new_stage}'.",
            metadata={"from_stage": old_stage, "to_stage": new_stage},
        )

    def step(self, state: WorkflowState) -> AgentResult:
        """Executes a single workflow step."""
        state.step_count += 1
        if state.step_count > state.max_steps:
            state.status = WorkflowStatus.FAILED
            state.error_message = f"Max execution steps ({state.max_steps}) exceeded."
            raise MaxStepsExceededError(state.error_message)

        stage = state.current_stage

        # Dispatch based on current stage
        if stage == WorkflowStage.INTAKE:
            return self._execute_agent(self.intake_agent, state, WorkflowStage.QUALITY)

        elif stage == WorkflowStage.QUALITY:
            return self._execute_agent(
                self.quality_agent, state, WorkflowStage.CLEANING_PLAN
            )

        elif stage == WorkflowStage.CLEANING_PLAN:
            result = self._execute_agent(self.cleaning_planner, state, None)
            if state.proposals:
                # Cleaning proposals exist; transition to waiting for cleaning approval
                self.transition_stage(
                    state, WorkflowStage.WAITING_FOR_CLEANING_APPROVAL
                )
                state.status = WorkflowStatus.WAITING_FOR_APPROVAL
            else:
                self.transition_stage(state, WorkflowStage.VISUALIZATION_AND_EDA)
            return result

        elif stage == WorkflowStage.WAITING_FOR_CLEANING_APPROVAL:
            # Check if any cleaning proposals are still pending
            cleaning_actions = {
                "remove_duplicates",
                "trim_text",
                "normalize_case",
                "fill_numeric_missing",
                "fill_categorical_missing",
                "cap_outliers",
                "convert_dates",
                "drop_column",
                "protect_column",
            }
            pending_cleaning = [
                a
                for a in state.approvals
                if a.action_type in cleaning_actions
                and a.status == ApprovalStatus.PENDING
            ]
            if pending_cleaning:
                state.status = WorkflowStatus.WAITING_FOR_APPROVAL
                return AgentResult(
                    agent_name=self.name,
                    success=True,
                    message=f"Waiting for human review of {len(pending_cleaning)} cleaning proposal(s).",
                    next_stage=WorkflowStage.WAITING_FOR_CLEANING_APPROVAL,
                )
            # All decisions recorded; proceed to execution
            self.transition_stage(state, WorkflowStage.CLEANING_EXECUTION)
            return self.step(state)

        elif stage == WorkflowStage.CLEANING_EXECUTION:
            return self._execute_agent(
                self.cleaning_executor, state, WorkflowStage.VISUALIZATION_AND_EDA
            )

        elif stage == WorkflowStage.VISUALIZATION_AND_EDA:
            return self._execute_agent(
                self.visualization_agent, state, WorkflowStage.ML_READINESS
            )

        elif stage == WorkflowStage.ML_READINESS:
            return self._execute_agent(
                self.ml_readiness_agent, state, WorkflowStage.INSIGHT_GENERATION
            )

        elif stage == WorkflowStage.INSIGHT_GENERATION:
            result = self._execute_agent(self.insight_agent, state, None)
            self.transition_stage(state, WorkflowStage.WAITING_FOR_INSIGHT_REVIEW)
            state.status = WorkflowStatus.WAITING_FOR_APPROVAL
            return result

        elif stage == WorkflowStage.WAITING_FOR_INSIGHT_REVIEW:
            insight_approvals = [
                a
                for a in state.approvals
                if a.action_type == "review_ai_insights"
                and a.status == ApprovalStatus.PENDING
            ]
            if insight_approvals:
                state.status = WorkflowStatus.WAITING_FOR_APPROVAL
                return AgentResult(
                    agent_name=self.name,
                    success=True,
                    message="Waiting for human review of AI-generated insights.",
                    next_stage=WorkflowStage.WAITING_FOR_INSIGHT_REVIEW,
                )
            self.transition_stage(state, WorkflowStage.REPORT_GENERATION)
            return self.step(state)

        elif stage == WorkflowStage.REPORT_GENERATION:
            result = self._execute_agent(self.report_agent, state, None)
            self.transition_stage(state, WorkflowStage.WAITING_FOR_EXPORT_APPROVAL)
            state.status = WorkflowStatus.WAITING_FOR_APPROVAL
            return result

        elif stage == WorkflowStage.WAITING_FOR_EXPORT_APPROVAL:
            export_approvals = [
                a
                for a in state.approvals
                if a.action_type == "export_dataset"
                and a.status == ApprovalStatus.PENDING
            ]
            if export_approvals:
                state.status = WorkflowStatus.WAITING_FOR_APPROVAL
                return AgentResult(
                    agent_name=self.name,
                    success=True,
                    message="Waiting for human export authorization.",
                    next_stage=WorkflowStage.WAITING_FOR_EXPORT_APPROVAL,
                )
            self.transition_stage(state, WorkflowStage.COMPLETED)
            state.status = WorkflowStatus.COMPLETED
            return AgentResult(
                agent_name=self.name,
                success=True,
                message="Workflow completed successfully with full governance.",
                next_stage=WorkflowStage.COMPLETED,
            )

        elif stage == WorkflowStage.COMPLETED:
            state.status = WorkflowStatus.COMPLETED
            return AgentResult(
                agent_name=self.name,
                success=True,
                message="Workflow is already completed.",
                next_stage=WorkflowStage.COMPLETED,
            )

        else:
            raise InvalidStageTransitionError(f"Unknown workflow stage '{stage}'.")

    def _execute_agent(
        self,
        agent: BaseAgent,
        state: WorkflowState,
        next_stage: str | None,
    ) -> AgentResult:
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        start_time = datetime.now(timezone.utc).isoformat()
        state.status = WorkflowStatus.RUNNING

        run_record = AgentRun(
            run_id=run_id,
            agent_name=agent.name,
            stage=state.current_stage,
            started_at=start_time,
        )
        state.agent_runs.append(run_record)

        try:
            result = agent.run(state)
            run_record.completed_at = datetime.now(timezone.utc).isoformat()
            run_record.success = result.success
            run_record.message = result.message

            if not result.success:
                state.status = WorkflowStatus.FAILED
                state.error_message = (
                    result.error or "Agent failed without explicit error."
                )
                state.record_audit(
                    actor_type=ActorType.AGENT,
                    actor_name=agent.name,
                    event_type=EventType.AGENT_FAILURE,
                    message=f"Agent '{agent.name}' failed: {state.error_message}",
                )
                return result

            # Progress to next stage if requested
            target_stage = next_stage or result.next_stage
            if target_stage and target_stage != state.current_stage:
                self.transition_stage(state, target_stage)

            return result

        except Exception as exc:
            run_record.completed_at = datetime.now(timezone.utc).isoformat()
            run_record.success = False
            run_record.error = str(exc)
            state.status = WorkflowStatus.FAILED
            state.error_message = str(exc)
            state.record_audit(
                actor_type=ActorType.AGENT,
                actor_name=agent.name,
                event_type=EventType.AGENT_FAILURE,
                message=f"Agent '{agent.name}' encountered exception: {exc}",
            )
            return AgentResult(
                agent_name=agent.name,
                success=False,
                message=f"Error executing {agent.name}: {exc}",
                error=str(exc),
            )

    def run_until_checkpoint(self, state: WorkflowState) -> AgentResult:
        """Executes steps sequentially until pausing at a human approval checkpoint or completion."""
        last_result = None
        while state.status not in (
            WorkflowStatus.WAITING_FOR_APPROVAL,
            WorkflowStatus.COMPLETED,
            WorkflowStatus.FAILED,
        ):
            last_result = self.step(state)
            if state.current_stage in (
                WorkflowStage.WAITING_FOR_CLEANING_APPROVAL,
                WorkflowStage.WAITING_FOR_INSIGHT_REVIEW,
                WorkflowStage.WAITING_FOR_EXPORT_APPROVAL,
            ):
                break
        return last_result or AgentResult(
            agent_name=self.name,
            success=True,
            message=f"Workflow paused at '{state.current_stage}'. Status: {state.status}.",
            next_stage=state.current_stage,
        )

    def resume_workflow(self, state: WorkflowState) -> AgentResult:
        """Resumes workflow after human decisions have been made on pending requests."""
        if state.status == WorkflowStatus.WAITING_FOR_APPROVAL:
            state.status = WorkflowStatus.RUNNING
        return self.run_until_checkpoint(state)
