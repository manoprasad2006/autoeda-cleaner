"""Agent Orchestration and Governance Hub page."""

from __future__ import annotations

import streamlit as st

from agents.supervisor import SupervisorAgent
from studio.common import heading, require_data
from ui.agent_status import (
    render_agent_timeline,
    render_review_required_banner,
    render_workflow_header,
)
import importlib
from ui.approval_panel import render_approval_panel
from ui.audit_panel import render_audit_panel
import utils.session as session_utils
from workflow.state import WorkflowState, WorkflowStatus

if not hasattr(session_utils, "get_workflow_state"):
    importlib.reload(session_utils)

get_workflow_state = getattr(
    session_utils,
    "get_workflow_state",
    lambda: st.session_state.get("workflow_state"),
)
set_workflow_state = getattr(
    session_utils,
    "set_workflow_state",
    lambda state: st.session_state.__setitem__("workflow_state", state),
)

heading(
    "00 / Governance",
    "Agent Orchestration Hub",
    "Governed multi-agent data intelligence platform with strict Human-in-the-Loop approval.",
)

df = require_data()
state: WorkflowState | None = get_workflow_state()

if state is None:
    filename = st.session_state.get("dataset_filename", "Dataset")
    metadata = st.session_state.get("dataset_metadata")
    state = WorkflowState.create_initial(df, filename=filename, metadata=metadata)
    set_workflow_state(state)

# Render Status Header & Timeline
render_review_required_banner(state)
render_workflow_header(state)
st.write("")
render_agent_timeline(state)
st.divider()

# Orchestrator Controls
ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 1.5, 1.5, 2])

supervisor = SupervisorAgent()

with ctrl1:
    if state.status == WorkflowStatus.WAITING_FOR_APPROVAL:
        btn_label = "▶ Resume After Review"
        btn_type = "primary"
    elif state.status == WorkflowStatus.COMPLETED:
        btn_label = "✓ Workflow Completed"
        btn_type = "secondary"
    else:
        btn_label = "⚡ Run Autonomous Pipeline"
        btn_type = "primary"

    if st.button(
        btn_label, type=btn_type, disabled=state.status == WorkflowStatus.COMPLETED
    ):
        with st.spinner("Agents coordinating..."):
            if state.status == WorkflowStatus.WAITING_FOR_APPROVAL:
                supervisor.resume_workflow(state)
            else:
                supervisor.run(state)
        set_workflow_state(state)
        st.rerun()

with ctrl2:
    if st.button("⏭ Step Once", disabled=state.status == WorkflowStatus.COMPLETED):
        with st.spinner("Executing next step..."):
            supervisor.step(state)
        set_workflow_state(state)
        st.rerun()

with ctrl3:
    if st.button("🔄 Reset Workflow"):
        filename = state.dataset_filename
        metadata = state.dataset_metadata
        new_state = WorkflowState.create_initial(
            state.original_dataset, filename, metadata
        )
        set_workflow_state(new_state)
        st.success("Workflow reset to original immutable dataset.")
        st.rerun()

with ctrl4:
    st.caption(
        f"Original dataset: **{state.dataset_filename}** ({len(state.original_dataset):,} rows)"
    )
    st.caption(
        f"Active version: **{state.active_version_id}** ({len(state.active_dataset):,} rows)"
    )
    if state.active_version_id != "v0":
        from modules.exports import csv_bytes

        st.download_button(
            label=f"📥 Download {state.active_version_id} CSV",
            data=csv_bytes(state.active_dataset),
            file_name=f"processed_{state.active_version_id}_{state.dataset_filename}.csv",
            mime="text/csv",
            key="btn_quick_download_active_csv",
        )

# Main Hub Tabs
tab_approvals, tab_findings, tab_audit = st.tabs(
    [
        "🛡️ Governance & Approvals",
        "🔍 Intelligence & Findings",
        "📜 Full Audit Trail & Rollback",
    ]
)

with tab_approvals:
    render_approval_panel(state, on_action_applied=lambda: set_workflow_state(state))

with tab_findings:
    st.subheader("Agent Intelligence Summary")
    f_col1, f_col2 = st.columns(2)

    with f_col1:
        with st.container(border=True):
            st.markdown("#### 📥 Intake Findings")
            if state.intake_findings:
                for item in state.intake_findings:
                    st.markdown(f"- **[{item.get('category')}]** {item.get('detail')}")
            else:
                st.caption("No structural intake findings recorded yet.")

        with st.container(border=True):
            st.markdown("#### 🎯 ML Readiness Diagnosis")
            if state.ml_readiness_result:
                ml = state.ml_readiness_result
                st.metric("ML Readiness Score", f"{ml.readiness_score:.1f}/100")
                st.caption(f"Inferred Task: **{ml.inferred_task}**")
                for iss in ml.issues:
                    st.warning(
                        f"**[{iss.severity.upper()}] {iss.category}:** {iss.description}"
                    )
            else:
                st.caption("ML readiness agent has not run yet.")

    with f_col2:
        with st.container(border=True):
            st.markdown("#### 📊 Quality Score Breakdown")
            if state.quality_result:
                q = state.quality_result
                st.metric("Quality Score", f"{q.overall_score:.1f}/100")
                st.write(f"- Completeness: **{q.completeness_score:.1f}%**")
                st.write(f"- Duplicate score: **{q.duplicate_score:.1f}%**")
                st.write(f"- Consistency score: **{q.consistency_score:.1f}%**")
                st.write(f"- Validity score: **{q.validity_score:.1f}%**")
            else:
                st.caption("Quality agent has not run yet.")

        with st.container(border=True):
            st.markdown("#### 💡 AI Business Insights")
            if state.ai_insights_structured:
                for ins in state.ai_insights_structured:
                    st.markdown(f"**Hypothesis:** {ins.get('hypothesis')}")
                    st.caption(
                        f"Evidence: {ins.get('evidence')} | Confidence: {ins.get('confidence')}"
                    )
                    st.caption(f"_{ins.get('disclaimer')}_")
                    st.divider()
            elif state.generated_insights:
                st.markdown(state.generated_insights)
            else:
                st.caption("Insight agent has not run yet.")

with tab_audit:
    render_audit_panel(state)
