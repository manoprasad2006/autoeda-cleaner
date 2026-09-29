"""Human-in-the-Loop approval panel with granular review and risk confirmation."""
from __future__ import annotations

import json
from typing import Callable
import streamlit as st

from ui.proposal_view import render_proposal_card
from workflow.approvals import (
    ApprovalRequest,
    ApprovalStatus,
    RiskLevel,
    approve_request,
    edit_and_approve_request,
    reject_request,
)
from workflow.state import WorkflowState


def render_approval_panel(
    state: WorkflowState,
    on_action_applied: Callable[[], None] | None = None,
) -> None:
    """Renders all pending and resolved approval requests."""
    pending_approvals = [a for a in state.approvals if a.status == ApprovalStatus.PENDING]
    resolved_approvals = [a for a in state.approvals if a.status != ApprovalStatus.PENDING]

    st.subheader(f"Governance & Approval Decisions ({len(pending_approvals)} Pending)")

    if not pending_approvals and not resolved_approvals:
        st.info("No approval requests have been submitted yet. Run the agent workflow to inspect proposals.")
        return

    if pending_approvals:
        st.markdown(
            "Review each proposal below. Consequential transformations are **blocked** until you explicitly approve or reject them. "
            "High-risk proposals require explicit acknowledgment."
        )

        selected_for_batch: list[ApprovalRequest] = []

        for approval in pending_approvals:
            proposal = state.get_proposal(approval.proposal_id)
            if not proposal:
                continue

            with st.container(border=True):
                # Render proposal details card
                render_proposal_card(proposal, is_pending=True)

                is_high_risk = proposal.risk_level == RiskLevel.HIGH

                # High risk confirmation guard
                confirmed_high_risk = True
                if is_high_risk:
                    confirmed_high_risk = st.checkbox(
                        f"⚠️ I understand that '{proposal.title}' carries HIGH risk (potential data destruction/loss).",
                        key=f"confirm_high_risk_{approval.id}",
                    )

                # Batch selection checkbox
                include_in_batch = st.checkbox(
                    "Select for batch approval",
                    key=f"batch_sel_{approval.id}",
                    disabled=is_high_risk and not confirmed_high_risk,
                )
                if include_in_batch:
                    selected_for_batch.append(approval)

                # Interactive Decision Columns
                c_appr, c_rej, c_edit = st.columns([1.5, 1.5, 2])

                with c_appr:
                    can_approve = not is_high_risk or confirmed_high_risk
                    if st.button(
                        f"✓ Approve ({proposal.title[:18]}...)",
                        key=f"btn_appr_{approval.id}",
                        type="primary",
                        disabled=not can_approve,
                    ):
                        approve_request(approval, reviewer="human_user")
                        st.success(f"Approved: {proposal.title}")
                        if on_action_applied:
                            on_action_applied()
                        st.rerun()

                with c_rej:
                    reject_note = st.text_input(
                        "Rejection note (optional)",
                        key=f"note_rej_{approval.id}",
                        placeholder="Reason for rejection...",
                        label_visibility="collapsed",
                    )
                    if st.button(
                        "✕ Reject",
                        key=f"btn_rej_{approval.id}",
                    ):
                        reject_request(approval, reviewer="human_user", note=reject_note)
                        st.warning(f"Rejected: {proposal.title}")
                        if on_action_applied:
                            on_action_applied()
                        st.rerun()

                with c_edit:
                    with st.expander("✏️ Edit Parameters & Approve"):
                        edited_json_str = st.text_area(
                            "Parameters (JSON)",
                            value=json.dumps(proposal.parameters, indent=2),
                            key=f"json_edit_{approval.id}",
                            height=100,
                        )
                        if st.button("Save & Approve", key=f"btn_edit_{approval.id}"):
                            try:
                                parsed_params = json.loads(edited_json_str)
                                edit_and_approve_request(
                                    approval,
                                    edited_parameters=parsed_params,
                                    reviewer="human_user",
                                    note="Approved with custom parameter edits",
                                )
                                st.success(f"Edited and approved: {proposal.title}")
                                if on_action_applied:
                                    on_action_applied()
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Invalid JSON parameters: {exc}")

        # Batch approval section
        if selected_for_batch:
            st.divider()
            b1, b2 = st.columns([2, 2])
            with b1:
                st.write(f"**{len(selected_for_batch)} proposal(s) selected for approval.**")
            with b2:
                if st.button(
                    f"✓ Apply {len(selected_for_batch)} Selected Approvals",
                    type="primary",
                    key="btn_apply_selected_batch",
                ):
                    for appr in selected_for_batch:
                        approve_request(appr, reviewer="human_user")
                    st.success(f"Applied approvals for {len(selected_for_batch)} proposal(s).")
                    if on_action_applied:
                        on_action_applied()
                    st.rerun()

    # Resolved approvals history
    if resolved_approvals:
        with st.expander(f"📜 Resolved Approvals History ({len(resolved_approvals)})"):
            for appr in resolved_approvals:
                prop = state.get_proposal(appr.proposal_id)
                status_color = "#16a34a" if appr.status in ("approved", "edited") else "#dc2626"
                st.markdown(
                    f"**{prop.title if prop else appr.proposal_id}** — "
                    f'<span style="color:{status_color};font-weight:bold;">{appr.status.upper()}</span> '
                    f"by `{appr.reviewer or 'user'}` at `{appr.resolved_at}`",
                    unsafe_allow_html=True,
                )
                if appr.reviewer_note:
                    st.caption(f"Note: {appr.reviewer_note}")
