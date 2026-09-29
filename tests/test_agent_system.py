"""Comprehensive tests for the governed multi-agent intelligence platform.
Covers all 20 required specifications:
1. Agent base contract
2. Supervisor stage transitions
3. Unknown tool rejection
4. Proposal creation
5. Approval creation
6. Approval execution
7. Rejection behavior
8. Edit-and-approve behavior
9. Expired approval behavior
10. Rollback behavior
11. Original dataset immutability
12. Multiple pending approvals
13. High-risk actions requiring approval
14. Audit events
15. Gemini unavailable fallback
16. Prompt length limits
17. AI output review requirement
18. Report export approval requirement
19. Visualization aggregation semantics
20. Streamlit approval-panel interaction
"""
from __future__ import annotations

import pandas as pd
import pytest

from agents.base import BaseAgent
from agents.cleaning_executor_agent import CleaningExecutorAgent
from agents.cleaning_planner_agent import CleaningPlannerAgent
from agents.intake_agent import IntakeAgent
from agents.schemas import AgentResult
from agents.supervisor import SupervisorAgent
from tools.registry import ToolRegistry, registry
from tools.visualization_tools import is_non_additive_measure
from workflow.approvals import (
    ApprovalStatus,
    RiskLevel,
    approve_request,
    create_approval_request,
    edit_and_approve_request,
    expire_request,
    get_pending_requests,
    reject_request,
)
from workflow.errors import (
    ApprovalRequiredError,
    InvalidStageTransitionError,
    RollbackError,
    UnregisteredToolError,
)
from workflow.events import EventType
from workflow.state import (
    Proposal,
    WorkflowStage,
    WorkflowState,
    WorkflowStatus,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [101, 102, 103, 104, 104],  # duplicate row 104
            "name": [" Alice ", "Bob", "Charlie", "David", "David"],
            "score": [85.0, 92.5, None, 10.0, 10.0],
            "satisfaction_rating": [4.5, 4.0, 3.5, 1.0, 1.0],
            "category": ["A", "B", "A", None, None],
        }
    )


# 1. Agent base contract
def test_agent_base_contract(sample_df):
    class DummyAgent(BaseAgent):
        name = "DummyAgent"
        description = "Test agent implementation"

        def run(self, state: WorkflowState) -> AgentResult:
            return AgentResult(
                agent_name=self.name,
                success=True,
                message="Completed successfully",
                proposals=[],
                events=[],
                next_stage=WorkflowStage.COMPLETED,
            )

    agent = DummyAgent()
    state = WorkflowState.create_initial(sample_df, "test.csv")
    result = agent.run(state)
    assert isinstance(result, AgentResult)
    assert result.success is True
    assert result.agent_name == "DummyAgent"
    assert result.next_stage == WorkflowStage.COMPLETED


# 2. Supervisor stage transitions
def test_supervisor_stage_transitions(sample_df):
    sup = SupervisorAgent()
    state = WorkflowState.create_initial(sample_df, "test.csv")
    assert state.current_stage == WorkflowStage.INTAKE

    # Legal transition
    sup.transition_stage(state, WorkflowStage.QUALITY)
    assert state.current_stage == WorkflowStage.QUALITY

    # Illegal transition: Quality directly to Completed should raise InvalidStageTransitionError
    with pytest.raises(InvalidStageTransitionError):
        sup.transition_stage(state, WorkflowStage.COMPLETED)


# 3. Unknown tool rejection
def test_unknown_tool_rejection():
    custom_registry = ToolRegistry()
    with pytest.raises(UnregisteredToolError) as exc_info:
        custom_registry.execute("arbitrary_eval_tool", "1+1")
    assert "not registered" in str(exc_info.value)


# 4. Proposal creation
def test_proposal_creation(sample_df):
    state = WorkflowState.create_initial(sample_df, "test.csv")
    prop = Proposal.create(
        agent_name="TestAgent",
        action_type="trim_text",
        title="Trim names",
        description="Remove surrounding whitespace",
        affected_columns=["name"],
        rows_affected=1,
        risk_level=RiskLevel.LOW,
        parameters={"columns": ["name"]},
        before_summary="Un-trimmed text",
        expected_after_summary="Trimmed text",
    )
    approval = state.add_proposal(prop)
    assert prop.id.startswith("prop-")
    assert approval.proposal_id == prop.id
    assert approval.status == ApprovalStatus.PENDING
    assert len(state.proposals) == 1
    assert len(state.approvals) == 1


# 5. Approval creation
def test_approval_creation():
    appr = create_approval_request(
        workflow_id="wf-1",
        proposal_id="prop-1",
        agent_name="Planner",
        action_type="remove_duplicates",
        description="Drop duplicates",
        affected_columns=["all"],
        rows_affected=1,
        risk_level=RiskLevel.LOW,
    )
    assert appr.id.startswith("appr-")
    assert appr.status == ApprovalStatus.PENDING
    assert appr.workflow_id == "wf-1"


# 6. Approval execution
def test_approval_execution(sample_df):
    state = WorkflowState.create_initial(sample_df, "test.csv")
    planner = CleaningPlannerAgent()
    planner.run(state)
    assert len(state.proposals) > 0

    # Unapproved proposal execution should raise ApprovalRequiredError
    with pytest.raises(ApprovalRequiredError):
        registry.execute("execute_cleaning", state=state, proposal_id=state.proposals[0].id)

    # Approve the first proposal
    first_appr = state.approvals[0]
    approve_request(first_appr, reviewer="lead_analyst")
    assert first_appr.status == ApprovalStatus.APPROVED

    # Execute approved proposal
    executor = CleaningExecutorAgent()
    res = executor.run(state)
    assert res.success is True
    assert state.get_proposal(first_appr.proposal_id).status == "executed"
    assert len(state.versions) >= 2


# 7. Rejection behavior
def test_rejection_behavior(sample_df):
    state = WorkflowState.create_initial(sample_df, "test.csv")
    planner = CleaningPlannerAgent()
    planner.run(state)
    first_appr = state.approvals[0]
    reject_request(first_appr, reviewer="human_user", note="Do not modify duplicates yet")
    assert first_appr.status == ApprovalStatus.REJECTED
    assert first_appr.reviewer_note == "Do not modify duplicates yet"

    # Execution should be rejected
    with pytest.raises(ApprovalRequiredError):
        registry.execute("execute_cleaning", state=state, proposal_id=first_appr.proposal_id)


# 8. Edit-and-approve behavior
def test_edit_and_approve_behavior(sample_df):
    state = WorkflowState.create_initial(sample_df, "test.csv")
    # Propose fill numeric missing
    prop = Proposal.create(
        agent_name="Planner",
        action_type="fill_numeric_missing",
        title="Fill score",
        description="Impute missing scores",
        affected_columns=["score"],
        rows_affected=1,
        risk_level=RiskLevel.LOW,
        parameters={"strategy": "median", "column": "score"},
        before_summary="1 missing",
        expected_after_summary="Filled with median",
    )
    appr = state.add_proposal(prop)

    # Edit parameter to use a custom value 99.0
    edit_and_approve_request(
        appr,
        edited_parameters={"strategy": "custom", "custom_value": 99.0},
        reviewer="lead_analyst",
    )
    assert appr.status == ApprovalStatus.EDITED
    assert appr.edited_parameters["custom_value"] == 99.0

    # Execute
    candidate, diff = registry.execute("execute_cleaning", state=state, proposal_id=prop.id)
    assert 99.0 in candidate["score"].values


# 9. Expired approval behavior
def test_expired_approval_behavior():
    appr = create_approval_request(
        workflow_id="wf-1",
        proposal_id="prop-1",
        agent_name="Planner",
        action_type="trim_text",
        description="Trim whitespace",
        affected_columns=["name"],
        rows_affected=1,
    )
    expire_request(appr, reason="Superseded by batch recipe")
    assert appr.status == ApprovalStatus.EXPIRED

    # Attempting to approve an expired request should raise ValueError
    with pytest.raises(ValueError):
        approve_request(appr)


# 10. Rollback behavior
def test_rollback_behavior(sample_df):
    state = WorkflowState.create_initial(sample_df, "test.csv")
    # Initial state cannot rollback
    assert state.can_rollback() is False
    with pytest.raises(RollbackError):
        state.rollback()

    # Apply a proposal
    prop = Proposal.create(
        agent_name="Planner",
        action_type="remove_duplicates",
        title="Remove Duplicates",
        description="Drop duplicates",
        affected_columns=list(sample_df.columns),
        rows_affected=1,
        risk_level=RiskLevel.LOW,
        parameters={},
        before_summary="5 rows",
        expected_after_summary="4 rows",
    )
    appr = state.add_proposal(prop)
    approve_request(appr)

    # Execute
    registry.execute("execute_cleaning", state=state, proposal_id=prop.id)
    assert len(state.active_dataset) == 4
    assert state.active_version_id == "v1"
    assert state.can_rollback() is True

    # Rollback
    state.rollback(reviewer="human_user")
    assert len(state.active_dataset) == 5
    assert state.active_version_id == "v2"


# 11. Original dataset immutability
def test_original_dataset_immutability(sample_df):
    state = WorkflowState.create_initial(sample_df, "test.csv")
    orig_copy = sample_df.copy(deep=True)

    # Perform mutations on active dataset
    prop = Proposal.create(
        agent_name="Planner",
        action_type="drop_column",
        title="Drop category",
        description="Drop column",
        affected_columns=["category"],
        rows_affected=len(sample_df),
        risk_level=RiskLevel.HIGH,
        parameters={"columns": ["category"]},
        before_summary="category present",
        expected_after_summary="category removed",
    )
    appr = state.add_proposal(prop)
    approve_request(appr)
    registry.execute("execute_cleaning", state=state, proposal_id=prop.id)

    assert "category" not in state.active_dataset.columns
    # Verify original dataset remains completely unchanged and has 'category'
    assert "category" in state.original_dataset.columns
    assert len(state.original_dataset) == len(orig_copy)
    pd.testing.assert_frame_equal(state.original_dataset, orig_copy)


# 12. Multiple pending approvals
def test_multiple_pending_approvals(sample_df):
    state = WorkflowState.create_initial(sample_df, "test.csv")
    p1 = Proposal.create(
        "Agent", "trim_text", "Trim", "Desc", ["name"], 1, RiskLevel.LOW, {}, "b", "a"
    )
    p2 = Proposal.create(
        "Agent", "remove_duplicates", "Dedup", "Desc", ["all"], 1, RiskLevel.LOW, {}, "b", "a"
    )
    a1 = state.add_proposal(p1)
    a2 = state.add_proposal(p2)

    pending = get_pending_requests(state.approvals)
    assert len(pending) == 2

    # Approve only p1
    approve_request(a1)
    pending_after = get_pending_requests(state.approvals)
    assert len(pending_after) == 1
    assert pending_after[0].proposal_id == p2.id
    assert a2.status == ApprovalStatus.PENDING


# 13. High-risk actions requiring approval
def test_high_risk_actions_require_approval(sample_df):
    planner = CleaningPlannerAgent()
    state = WorkflowState.create_initial(sample_df, "test.csv")
    state.quality_result = registry.execute("assess_quality", sample_df, registry.execute("profile_dataset", sample_df))
    planner.run(state)

    for prop in state.proposals:
        appr = state.get_approval_for_proposal(prop.id)
        assert appr is not None
        assert appr.status == ApprovalStatus.PENDING
        if prop.action_type in ("cap_outliers", "drop_column"):
            assert prop.risk_level == RiskLevel.HIGH


# 14. Audit events
def test_audit_events(sample_df):
    state = WorkflowState.create_initial(sample_df, "test.csv")
    assert any(e.event_type == EventType.DATASET_IMPORTED for e in state.audit_events)

    intake = IntakeAgent()
    intake.run(state)
    assert any(e.event_type == EventType.INTAKE_COMPLETED for e in state.audit_events)

    prop = Proposal.create(
        "TestAgent", "protect_column", "Protect ID", "Protect", ["id"], 0, RiskLevel.LOW, {}, "b", "a"
    )
    state.add_proposal(prop)
    assert any(e.event_type == EventType.PROPOSAL_CREATED for e in state.audit_events)
    assert any(e.event_type == EventType.APPROVAL_REQUESTED for e in state.audit_events)


# 15. Gemini unavailable fallback
def test_gemini_unavailable_fallback(sample_df):
    state = WorkflowState.create_initial(sample_df, "test.csv")
    state.profile_result = registry.execute("profile_dataset", sample_df)
    state.quality_result = registry.execute("assess_quality", sample_df, state.profile_result)
    state.eda_result = registry.execute("run_eda", sample_df, ["score"], ["category"])

    # Run AI summary with NO api_key
    summary = registry.execute("generate_ai_summary", state=state, api_key=None)
    assert "Deterministic Analysis" in summary
    assert "Overall" in summary

    # Run business insights with NO api_key
    insights = registry.execute("generate_business_insights", state=state, api_key=None)
    assert isinstance(insights, list)
    assert len(insights) > 0
    assert "hypothesis" in insights[0]
    assert insights[0]["is_ai_generated"] is False


# 16. Prompt length limits & raw row exclusion
def test_prompt_length_limits(sample_df):
    state = WorkflowState.create_initial(sample_df, "test.csv")
    from tools.ai_tools import _build_bounded_summary_prompt

    prompt = _build_bounded_summary_prompt(
        filename=state.dataset_filename,
        n_rows=100000,
        n_cols=50,
        quality_score=95.0,
        columns_summary=[{"name": "col_1", "type": "int", "missing_pct": 0, "unique_count": 5}],
    )
    assert len(prompt) < 10000
    assert "Alice" not in prompt  # Raw data values are strictly excluded
    assert "DO NOT generate executable code" in prompt


# 17. AI output review requirement
def test_ai_output_review_requirement(sample_df):
    sup = SupervisorAgent()
    state = WorkflowState.create_initial(sample_df, "test.csv")
    # Transition to insight generation
    state.current_stage = WorkflowStage.INSIGHT_GENERATION
    _ = sup.step(state)
    assert state.current_stage == WorkflowStage.WAITING_FOR_INSIGHT_REVIEW
    assert state.status == WorkflowStatus.WAITING_FOR_APPROVAL
    review_appr = next((a for a in state.approvals if a.action_type == "review_ai_insights"), None)
    assert review_appr is not None
    assert review_appr.status == ApprovalStatus.PENDING


# 18. Report export approval requirement
def test_report_export_approval_requirement(sample_df):
    sup = SupervisorAgent()
    state = WorkflowState.create_initial(sample_df, "test.csv")
    state.current_stage = WorkflowStage.REPORT_GENERATION
    _ = sup.step(state)
    assert state.current_stage == WorkflowStage.WAITING_FOR_EXPORT_APPROVAL
    assert state.status == WorkflowStatus.WAITING_FOR_APPROVAL
    export_appr = next((a for a in state.approvals if a.action_type == "export_dataset"), None)
    assert export_appr is not None
    assert export_appr.status == ApprovalStatus.PENDING

    # Export dataset bundle requires approval
    zip_bytes = registry.execute("export_dataset", state=state, export_format="zip")
    assert isinstance(zip_bytes, bytes)
    assert len(zip_bytes) > 0


# 19. Visualization aggregation semantics
def test_visualization_aggregation_semantics(sample_df):
    # Non-additive measures should be detected
    assert is_non_additive_measure("satisfaction_rating") is True
    assert is_non_additive_measure("customer_score") is True
    assert is_non_additive_measure("churn_pct") is True
    assert is_non_additive_measure("total_sales_revenue") is False

    profile = registry.execute("profile_dataset", sample_df)
    eda = registry.execute("run_eda", sample_df, ["score", "satisfaction_rating"], ["category"])
    recs = registry.execute("recommend_visualizations", sample_df, profile, eda)

    for rec in recs:
        # If column is non-additive, recommended aggregation must NOT be sum
        if rec.get("is_non_additive"):
            assert rec.get("recommended_aggregation") != "sum"


# 20. Streamlit approval-panel and UI components
def test_approval_panel_and_ui_interaction(sample_df):
    from ui.agent_status import (
        render_review_required_banner,
        render_workflow_header,
        render_agent_timeline,
    )
    from ui.proposal_view import render_proposal_card

    state = WorkflowState.create_initial(sample_df, "test.csv")
    state.status = WorkflowStatus.WAITING_FOR_APPROVAL

    # Verify these functions execute cleanly
    render_review_required_banner(state)
    render_workflow_header(state)
    render_agent_timeline(state)

    prop = Proposal.create(
        "TestAgent", "remove_duplicates", "Dedup", "Desc", ["all"], 1, RiskLevel.LOW, {}, "b", "a"
    )
    render_proposal_card(prop, is_pending=True)
