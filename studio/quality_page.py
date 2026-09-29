from dataclasses import asdict
import pandas as pd
import plotly.express as px
import streamlit as st
from studio.common import heading, require_data, analyze, chart, completeness, COLORS
from modules.cleaner import CleaningOptions, auto_clean
from utils.session import (
    get_dataset,
    set_cleaned_dataset,
    restore_original,
    has_cleaned_dataset,
    get_cleaning_log,
)

heading(
    "03 / Prepare",
    "Clean with confidence.",
    "Choose a recipe, inspect its impact, then apply it. Your original data remains available.",
)
df = require_data()
original = get_dataset()
p, q, e = analyze(df)
a, b, c = st.columns(3)
a.metric("Quality score", f"{q.overall_score:.1f}/100")
b.metric("Missing cells", f"{q.missing_cell_count:,}")
c.metric("Duplicate rows", f"{q.duplicate_row_count:,}")
health, recipe, audit = st.tabs(["Quality map", "Cleaning recipe", "Applied changes"])
with health:
    a, b = st.columns(2)
    with a:
        st.subheader("Missingness by column")
        chart(
            px.bar(
                completeness(df).head(25),
                x="Missing %",
                y="Column",
                orientation="h",
                color_discrete_sequence=COLORS,
            )
        )
    with b:
        st.subheader("Outlier signals")
        if q.outlier_columns:
            chart(
                px.bar(
                    pd.DataFrame(
                        q.outlier_columns.items(), columns=["Column", "Values"]
                    ),
                    x="Column",
                    y="Values",
                    color_discrete_sequence=[COLORS[2]],
                )
            )
        else:
            st.success("No IQR outliers detected in numeric columns.")
        st.caption(
            "Outliers can be valid observations. The score weights completeness 40%, duplicates 25%, type consistency 20%, and outlier checks 15%."
        )
    if q.inconsistent_columns:
        st.warning("Mixed Python value types: " + ", ".join(q.inconsistent_columns))
with recipe:
    st.caption(
        "Each preview starts from the original dataset. Identifier-like columns are protected by default. Protect your prediction target as well."
    )
    st.info(
        "A preview changes nothing. Apply only after reviewing the operations and row count."
    )
    a, b = st.columns(2)
    trim = a.checkbox("Trim surrounding whitespace", True)
    dedup = a.checkbox("Remove exact duplicate rows", True)
    case = a.checkbox("Normalize text to lowercase", False)
    numeric = b.selectbox("Fill missing numeric values", ["median", "mean", "leave"])
    cat = b.checkbox("Fill missing categories with the mode", False)
    cap = b.checkbox("Cap continuous numeric outliers", False)
    threshold = st.slider(
        "Only fill columns with missingness below this percentage", 0, 100, 40
    )
    defaults = [
        c
        for c in original.columns
        if c.lower() == "id" or c.lower().endswith("_id") or c.lower().endswith("id")
    ]
    protected = st.multiselect(
        "Protected columns — no cell edits", list(original.columns), default=defaults
    )
    st.caption(
        "Entirely empty columns remain empty. Outlier capping skips columns with 15 or fewer distinct values. Duplicate removal is a row operation."
    )
    options = CleaningOptions(
        dedup, trim, case, numeric, cat, cap, threshold / 100, tuple(protected)
    )
    signature = repr(options)
    if st.button("Preview cleaning", type="primary"):
        with st.spinner("Building a reversible preview…"):
            candidate, log = auto_clean(original, options)
            st.session_state["cleaning_preview"] = (
                signature,
                candidate,
                log,
                asdict(options),
            )
    preview = st.session_state.get("cleaning_preview")
    if preview and preview[0] == signature:
        _, candidate, log, config = preview
        before = analyze(original)[1]
        after = analyze(candidate)[1]
        a, b, c = st.columns(3)
        a.metric(
            "Quality after",
            f"{after.overall_score:.1f}",
            f"{after.overall_score - before.overall_score:.1f} points",
        )
        b.metric("Rows retained", f"{len(candidate):,}")
        c.metric("Operations", len(log.actions))
        st.dataframe(log.to_dataframe(), hide_index=True, width="stretch")
        st.dataframe(candidate.head(10), hide_index=True, width="stretch")
        if st.button("Apply this preview", type="primary"):
            set_cleaned_dataset(candidate, log)
            st.session_state["applied_recipe"] = config
            st.rerun()
    elif preview:
        st.info("Recipe changed. Generate a fresh preview before applying.")
with audit:
    log = get_cleaning_log()
    wf_state = st.session_state.get("workflow_state")
    if has_cleaned_dataset():
        st.success("A cleaned version is active across your workspace.")
        if (
            log is not None
            and hasattr(log, "to_dataframe")
            and not log.to_dataframe().empty
        ):
            st.dataframe(log.to_dataframe(), hide_index=True, width="stretch")
        elif wf_state and getattr(wf_state, "audit_events", None):
            cleaning_events = [
                e
                for e in wf_state.audit_events
                if e.event_type in ("Cleaning executed", "Cleaning rolled back")
            ]
            if cleaning_events:
                st.caption("Agent Governance Execution History:")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Time (UTC)": e.timestamp[11:19],
                                "Actor": f"{e.actor_name} ({e.actor_type})",
                                "Event": e.event_type,
                                "Details": e.message,
                            }
                            for e in reversed(cleaning_events)
                        ]
                    ),
                    hide_index=True,
                    width="stretch",
                )
            else:
                st.info("A cleaned version is active via agent governance.")
        else:
            st.info("A cleaned dataset version is currently active.")

        if st.button("Restore original dataset"):
            restore_original()
            st.rerun()
    else:
        st.info("No cleaning recipe has been applied yet.")
