import pandas as pd
import plotly.express as px
import streamlit as st
from studio.common import hero, analyze, chart, demo_data, COLORS
from modules.loader import get_metadata
from utils.session import get_active_dataset, set_dataset, has_cleaned_dataset

hero(
    "Your data, in focus",
    "Make every dataset tell a story.",
    "Understand what you have. Improve what matters. Turn your next question into a clear, shareable insight.",
)
df = get_active_dataset()
if df is None:
    left, right = st.columns([1, 3])
    if left.button("Explore sample workspace", type="primary", width="stretch"):
        data = demo_data()
        set_dataset(
            data,
            get_metadata(data, "Commerce demo · synthetic", 0),
            "Commerce demo · synthetic",
        )
        st.rerun()
    right.page_link(
        "studio/import_page.py",
        label="Bring your own dataset →",
        icon=":material/upload_file:",
    )
    st.caption(
        "The sample is synthetic commerce data. No external connection is needed."
    )
    st.markdown("### A clear path from file to findings")
    for col, (number, title, body) in zip(
        st.columns(4),
        [
            (
                "01",
                "Bring it together",
                "Upload a spreadsheet or explore a realistic sample.",
            ),
            (
                "02",
                "Build confidence",
                "Inspect quality, preview fixes, and keep the original.",
            ),
            (
                "03",
                "Find the signal",
                "Explore distributions, relationships, and time trends.",
            ),
            (
                "04",
                "Share the story",
                "Export a clean dataset, audit trail, and analysis report.",
            ),
        ],
    ):
        col.markdown(
            f'<div class="step"><small>{number}</small><h3>{title}</h3><p>{body}</p></div>',
            unsafe_allow_html=True,
        )
    st.stop()
p, q, e = analyze(df)
st.caption(
    f"{st.session_state['dataset_filename']} · {'Cleaned' if has_cleaned_dataset() else 'Original'} · Full-dataset statistics"
)

wf_state = st.session_state.get("workflow_state")
if wf_state:
    pending_count = len(
        [
            a
            for a in getattr(wf_state, "approvals", [])
            if getattr(a, "status", "") == "pending"
        ]
    )
    ov_c1, ov_c2 = st.columns([3, 1])
    with ov_c1:
        if pending_count > 0:
            st.warning(
                f"⚠️ **Governance Notice:** {pending_count} proposal(s) awaiting your decision in the Agent Governance Hub."
            )
        else:
            st.info(
                f"✦ **Governed Multi-Agent Pipeline:** Active version is **{wf_state.active_version_id}** ({wf_state.current_stage})."
            )
    with ov_c2:
        st.page_link(
            "studio/agent_page.py",
            label="Open Agent Hub →",
            icon=":material/smart_toy:",
        )
kpis = [
    ("Records", f"{len(df):,}"),
    ("Columns", str(len(df.columns))),
    ("Health signal", f"{q.overall_score:.1f}/100"),
    ("Missing cells", f"{q.missing_cell_count:,}"),
]
for row in (kpis[:2], kpis[2:]):
    for col, (label, value) in zip(st.columns(2), row):
        col.metric(label, value)
st.write("")
a, b = st.columns([1.5, 1])
with a:
    with st.container(border=True):
        st.subheader("The shape of your data")
        nums = [
            c
            for c in p.numerical_columns
            if df[c].nunique() > 15 and "id" not in c.lower()
        ]
        if nums:
            metric = st.selectbox("Measure", nums)
            if p.date_columns:
                date = p.date_columns[0]
                additive = any(
                    token in metric.lower()
                    for token in (
                        "revenue",
                        "sales",
                        "cost",
                        "amount",
                        "units",
                        "count",
                    )
                )
                aggregation = "sum" if additive else "mean"
                trend = (
                    df.set_index(date)[metric]
                    .resample("MS")
                    .agg(aggregation)
                    .reset_index()
                )
                st.caption(
                    f"Monthly {aggregation} of {metric}. Use Chart studio for another aggregation."
                )
                chart(
                    px.area(
                        trend,
                        x=date,
                        y=metric,
                        title=f"Monthly {aggregation} · {metric}",
                        color_discrete_sequence=COLORS,
                    )
                )
            else:
                chart(
                    px.histogram(
                        df,
                        x=metric,
                        nbins=35,
                        color_discrete_sequence=COLORS,
                        title=f"Distribution of {metric}",
                    )
                )
        else:
            st.dataframe(df.head(8), hide_index=True, width="stretch")
with b:
    with st.container(border=True):
        st.subheader("Quality at a glance")
        scores = pd.DataFrame(
            {
                "Dimension": [
                    "Completeness",
                    "Uniqueness",
                    "Type consistency",
                    "Outlier check",
                ],
                "Score": [
                    q.completeness_score,
                    q.duplicate_score,
                    q.consistency_score,
                    q.validity_score,
                ],
            }
        )
        fig = px.bar(
            scores,
            x="Score",
            y="Dimension",
            orientation="h",
            text="Score",
            color_discrete_sequence=COLORS,
        )
        fig.update_xaxes(range=[0, 105])
        chart(fig)
        st.caption(
            "Rule-based health signal, not a guarantee of correctness. Review the dimensions and domain assumptions before applying fixes."
        )
a, b = st.columns([1.5, 1])
with a:
    st.subheader("Where to focus next")
    if q.missing_cell_count:
        st.warning(
            f"{q.missing_cell_count:,} missing cells across {sum(c.missing_count > 0 for c in p.columns)} columns."
        )
    if p.duplicate_rows:
        st.info(f"{p.duplicate_rows:,} exact duplicate rows are ready for review.")
    if p.high_cardinality_columns:
        st.info(
            "Review possible identifiers: " + ", ".join(p.high_cardinality_columns[:5])
        )
    if not q.missing_cell_count and not p.duplicate_rows:
        st.success("No missing cells or exact duplicates. Explore relationships next.")
    st.page_link(
        "studio/quality_page.py", label="Review quality & prepare a cleaning recipe →"
    )
with b:
    st.subheader("Next steps")
    st.page_link(
        "studio/explorer_page.py",
        label="Inspect columns and filter records",
        icon=":material/table_chart:",
    )
    st.page_link(
        "studio/charts_page.py",
        label="Build a visual story",
        icon=":material/monitoring:",
    )
    st.page_link(
        "studio/report_page.py",
        label="Export your findings",
        icon=":material/ios_share:",
    )
