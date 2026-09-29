"""Audit log display and version rollback panel."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from workflow.state import WorkflowState


def render_audit_panel(state: WorkflowState) -> None:
    """Renders the complete audit log and rollback controls."""
    st.subheader(f"System Audit Trail ({len(state.audit_events)} Events)")

    # Rollback controls
    if state.can_rollback():
        with st.container(border=True):
            r1, r2 = st.columns([3, 1])
            with r1:
                st.markdown(
                    f"**Rollback Available:** Active dataset is version **{state.active_version_id}**. "
                    "You can roll back to the parent version to undo the last execution."
                )
            with r2:
                if st.button(
                    "↩ Rollback to Parent", type="secondary", key="btn_rollback_version"
                ):
                    try:
                        rolled_back = state.rollback(reviewer="human_user")
                        st.success(
                            f"Rolled back active dataset to {rolled_back.version_id}."
                        )
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Rollback failed: {exc}")

    if not state.audit_events:
        st.caption("No audit events recorded yet.")
        return

    # Filter controls
    f_actor, f_type = st.columns(2)
    actors = ["All"] + sorted(list({e.actor_name for e in state.audit_events}))
    event_types = ["All"] + sorted(list({e.event_type for e in state.audit_events}))

    selected_actor = f_actor.selectbox(
        "Filter by Actor", actors, index=0, key="audit_filter_actor"
    )
    selected_type = f_type.selectbox(
        "Filter by Event Type", event_types, index=0, key="audit_filter_type"
    )

    filtered_events = state.audit_events
    if selected_actor != "All":
        filtered_events = [e for e in filtered_events if e.actor_name == selected_actor]
    if selected_type != "All":
        filtered_events = [e for e in filtered_events if e.event_type == selected_type]

    # Convert to dataframe for table view
    df_events = pd.DataFrame(
        [
            {
                "Time (UTC)": e.timestamp[11:19],
                "Actor": f"{e.actor_name} ({e.actor_type})",
                "Event Type": e.event_type,
                "Message": e.message,
                "Proposal ID": e.proposal_id or "-",
                "Approval ID": e.approval_id or "-",
            }
            for e in reversed(filtered_events)
        ]
    )

    st.dataframe(df_events, use_container_width=True, hide_index=True)

    # Download audit CSV
    csv_bytes = df_events.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Audit Trail (CSV)",
        data=csv_bytes,
        file_name=f"audit_trail_{state.dataset_filename}.csv",
        mime="text/csv",
        key="btn_download_audit_csv",
    )
