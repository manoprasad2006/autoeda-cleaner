"""Proposal visualization component showing before/after diffs and risk level."""
from __future__ import annotations

import streamlit as st
from workflow.approvals import RiskLevel
from workflow.state import Proposal


RISK_COLORS = {
    RiskLevel.LOW: ("#16a34a", "#dcfce7"),
    RiskLevel.MEDIUM: ("#d97706", "#fef3c7"),
    RiskLevel.HIGH: ("#dc2626", "#fee2e2"),
}


def render_proposal_card(proposal: Proposal, is_pending: bool = True) -> None:
    """Renders a structured, transparent proposal card."""
    text_color, bg_color = RISK_COLORS.get(proposal.risk_level, ("#475569", "#f1f5f9"))

    with st.container(border=True):
        top_c1, top_c2 = st.columns([3, 1])
        with top_c1:
            st.markdown(f"### {proposal.title}")
            st.caption(f"Proposed by **{proposal.agent_name}** · Action: `{proposal.action_type}` · ID: `{proposal.id}`")

        with top_c2:
            st.markdown(
                f'<div style="text-align:right;"><span style="background:{bg_color};color:{text_color};padding:4px 10px;border-radius:9999px;font-size:12px;font-weight:700;border:1px solid {text_color}40;">RISK: {proposal.risk_level.upper()}</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown(f"**Description:** {proposal.description}")

        # Metrics row
        m1, m2, m3 = st.columns(3)
        m1.metric("Rows Affected", f"{proposal.rows_affected:,}")
        m2.metric("Columns Affected", f"{len(proposal.affected_columns)}")
        m3.metric("Requires Approval", "Yes" if proposal.requires_approval else "No")

        if proposal.affected_columns:
            st.markdown(f"**Target Columns:** `{', '.join(proposal.affected_columns)}`")

        # Before vs After diff preview
        c_before, c_after = st.columns(2)
        with c_before:
            with st.container(border=True):
                st.markdown("**Current State (Before):**")
                st.markdown(f"_{proposal.before_summary}_")
        with c_after:
            with st.container(border=True):
                st.markdown("**Projected State (Expected After):**")
                st.markdown(f"_{proposal.expected_after_summary}_")

        with st.expander("ℹ️ Why am I seeing this?"):
            if proposal.action_type == "remove_duplicates":
                st.write(
                    "The Data Quality Agent identified exact duplicate rows in the dataset. "
                    "Removing duplicates avoids artificial weight amplification and statistical bias during analysis."
                )
            elif proposal.action_type == "trim_text":
                st.write(
                    "Leading and trailing whitespace in string columns frequently creates artificial categories "
                    "(e.g., ' Yes' vs 'Yes'). Trimming text ensures consistent categorization."
                )
            elif "missing" in proposal.action_type:
                st.write(
                    "Missing values hinder statistical analysis and prevent machine learning training. "
                    "Imputation replaces missing values with robust summary statistics (median or mode) while preserving column shape."
                )
            elif proposal.action_type == "cap_outliers":
                st.write(
                    "Extreme numerical values can heavily distort mean, variance, and regression models. "
                    "IQR capping restricts extreme observations to 1.5 times the interquartile range."
                )
            elif proposal.action_type == "drop_column":
                st.write(
                    "The column contains 100% missing values or zero information variance. "
                    "Dropping uninformative columns simplifies schemas and saves memory."
                )
            elif proposal.action_type == "protect_column":
                st.write(
                    "This column was detected as an identifier (ID/Key) or modeling target. "
                    "Protecting it prevents automated cell mutations from destroying data integrity."
                )
            elif proposal.action_type == "review_ai_insights":
                st.write(
                    "The Insight Agent generated analytical hypotheses from statistical summaries. "
                    "To prevent hallucination or misinterpretation, human review is required before report export."
                )
            elif proposal.action_type == "export_dataset":
                st.write(
                    "The Report Agent compiled the active dataset and audit log. "
                    "Explicit user authorization ensures data is never shared or exported without consent."
                )
            else:
                st.write("This proposal was formulated by the autonomous planning agent to address detected data quality issues.")
