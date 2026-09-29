"""Agent status, stage timeline, and notification banner components."""
from __future__ import annotations

import streamlit as st
from workflow.state import WorkflowStage, WorkflowState, WorkflowStatus


STAGE_LABELS: dict[str, str] = {
    WorkflowStage.INTAKE: "Data Intake",
    WorkflowStage.QUALITY: "Quality Scoring",
    WorkflowStage.CLEANING_PLAN: "Cleaning Planning",
    WorkflowStage.WAITING_FOR_CLEANING_APPROVAL: "Cleaning Approval Checkpoint",
    WorkflowStage.CLEANING_EXECUTION: "Cleaning Execution",
    WorkflowStage.VISUALIZATION_AND_EDA: "Visualization & EDA",
    WorkflowStage.ML_READINESS: "ML Readiness Assessment",
    WorkflowStage.INSIGHT_GENERATION: "AI Insights Generation",
    WorkflowStage.WAITING_FOR_INSIGHT_REVIEW: "AI Review Checkpoint",
    WorkflowStage.REPORT_GENERATION: "Report Compilation",
    WorkflowStage.WAITING_FOR_EXPORT_APPROVAL: "Export Authorization Checkpoint",
    WorkflowStage.COMPLETED: "Workflow Completed",
}

STATUS_BADGES: dict[str, tuple[str, str]] = {
    WorkflowStatus.IDLE: ("○ Idle", "#64748b"),
    WorkflowStatus.RUNNING: ("⚡ Running", "#0284c7"),
    WorkflowStatus.WAITING_FOR_APPROVAL: ("⏳ Waiting for Approval", "#d97706"),
    WorkflowStatus.COMPLETED: ("✓ Completed", "#16a34a"),
    WorkflowStatus.REJECTED: ("✕ Rejected", "#dc2626"),
    WorkflowStatus.FAILED: ("⚠ Failed", "#dc2626"),
}


def render_review_required_banner(state: WorkflowState) -> None:
    """Render a prominent banner when human input is required."""
    if state.status == WorkflowStatus.WAITING_FOR_APPROVAL:
        pending_count = len([a for a in state.approvals if a.status == "pending"])
        st.warning(
            f"**Review Required:** The autonomous workflow is currently paused at **{STAGE_LABELS.get(state.current_stage, state.current_stage)}**.\n\n"
            f"There are **{pending_count} pending decision(s)** requiring your explicit authorization before proceeding.",
            icon="⚠️",
        )


def render_workflow_header(state: WorkflowState) -> None:
    """Render the active stage, status badge, and version indicator."""
    badge_label, badge_color = STATUS_BADGES.get(
        state.status, (state.status.title(), "#64748b")
    )
    c1, c2, c3, c4 = st.columns([2, 1.5, 1.5, 1])

    with c1:
        st.caption("GOVERNED AGENT WORKFLOW")
        st.subheader(STAGE_LABELS.get(state.current_stage, state.current_stage))

    with c2:
        st.caption("STATUS")
        st.markdown(
            f'<span style="background:{badge_color}1a;color:{badge_color};padding:4px 10px;border-radius:9999px;font-weight:600;font-size:13px;border:1px solid {badge_color}40;">{badge_label}</span>',
            unsafe_allow_html=True,
        )

    with c3:
        st.caption("ACTIVE VERSION")
        st.markdown(f"**{state.active_version_id}** ({len(state.versions)} version(s) tracked)")

    with c4:
        st.caption("STEPS")
        st.markdown(f"**{state.step_count}** / {state.max_steps}")


def render_agent_timeline(state: WorkflowState) -> None:
    """Render the activity timeline of completed, running, and pending agents."""
    st.markdown("#### Agent Activity Timeline")

    # Define the sequence of agents and stages
    timeline_items = [
        ("Data Intake Agent", WorkflowStage.INTAKE),
        ("Data Quality Agent", WorkflowStage.QUALITY),
        ("Cleaning Planner Agent", WorkflowStage.CLEANING_PLAN),
        ("Cleaning Approval Checkpoint", WorkflowStage.WAITING_FOR_CLEANING_APPROVAL),
        ("Cleaning Executor Agent", WorkflowStage.CLEANING_EXECUTION),
        ("Visualization Agent", WorkflowStage.VISUALIZATION_AND_EDA),
        ("ML Readiness Agent", WorkflowStage.ML_READINESS),
        ("Insight Agent", WorkflowStage.INSIGHT_GENERATION),
        ("Insight Review Checkpoint", WorkflowStage.WAITING_FOR_INSIGHT_REVIEW),
        ("Report Agent", WorkflowStage.REPORT_GENERATION),
        ("Export Authorization Checkpoint", WorkflowStage.WAITING_FOR_EXPORT_APPROVAL),
    ]

    current_idx = -1
    for i, (_, stage) in enumerate(timeline_items):
        if state.current_stage == stage:
            current_idx = i
            break
    if state.current_stage == WorkflowStage.COMPLETED:
        current_idx = len(timeline_items)

    nodes = []
    for i, (name, stage) in enumerate(timeline_items):
        if i < current_idx:
            icon = "✓"
            color = "#16a34a"
            status = "Completed"
        elif i == current_idx:
            if state.status == WorkflowStatus.WAITING_FOR_APPROVAL:
                icon = "⏳"
                color = "#d97706"
                status = "Awaiting Decision"
            elif state.status == WorkflowStatus.FAILED:
                icon = "✕"
                color = "#dc2626"
                status = "Failed"
            else:
                icon = "⚡"
                color = "#0284c7"
                status = "Active"
        else:
            icon = "○"
            color = "#64748b"
            status = "Pending"

        short_name = name.replace(" Agent", "").replace(" Checkpoint", "")
        nodes.append(
            f"""<div style="flex:0 0 110px;min-width:110px;text-align:center;padding:10px 6px;border-top:3px solid {color};background:rgba(18,28,46,0.85);border:1px solid rgba(41,53,78,0.7);border-top:3px solid {color};border-radius:8px;">
            <div style="font-size:16px;color:{color};font-weight:bold;">{icon}</div>
            <div style="font-size:11px;font-weight:600;color:#e2e8f0;margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="{name}">{short_name}</div>
            <div style="font-size:10px;color:#94a3b8;margin-top:1px;">{status}</div>
            </div>"""
        )

    st.markdown(
        f"""<div style="display:flex;gap:8px;overflow-x:auto;-webkit-overflow-scrolling:touch;padding-bottom:8px;scrollbar-width:thin;">
        {''.join(nodes)}
        </div>""",
        unsafe_allow_html=True,
    )
