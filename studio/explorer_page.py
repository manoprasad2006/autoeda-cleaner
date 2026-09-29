import pandas as pd
import plotly.express as px
import streamlit as st
from studio.common import heading, require_data, analyze, chart, COLORS
from modules.loader import get_metadata
from utils.session import set_dataset

heading(
    "02 / Understand",
    "Meet your dataset.",
    "Inspect individual columns, find records, and explicitly convert text dates before exploring trends.",
)
df = require_data()
p, q, e = analyze(df)
rows, columns = st.tabs(["Record explorer", "Column intelligence"])
with rows:
    a, b = st.columns([2, 1])
    query = a.text_input(
        "Search records", placeholder="Search any column…", max_chars=200
    )
    shown = b.multiselect(
        "Visible columns", list(df.columns), default=list(df.columns[:12])
    )
    filtered = df
    if query:
        mask = pd.Series(False, index=df.index)
        for col in df.columns:
            mask |= (
                df[col]
                .astype("string")
                .str.contains(query, case=False, regex=False, na=False)
            )
        filtered = df.loc[mask]
    st.caption(
        f"{len(filtered):,} matching rows. Displaying up to 1,000; exports include all matches."
    )
    st.dataframe(
        filtered[shown].head(1000), width="stretch", hide_index=True, height=420
    )
    from modules.exports import csv_bytes

    st.download_button(
        "Export matching rows",
        csv_bytes(filtered[shown]),
        "filtered_data.csv",
        "text/csv",
        disabled=not shown,
    )
with columns:
    a, b = st.columns([1, 2])
    selected = a.selectbox("Inspect a column", list(df.columns))
    series = df[selected]
    a.metric("Missing", f"{series.isna().mean() * 100:.1f}%")
    a.metric("Distinct values", f"{series.nunique():,}")
    a.caption(f"Storage type: {series.dtype}")
    with b:
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(
            series
        ):
            chart(
                px.histogram(df, x=selected, nbins=40, color_discrete_sequence=COLORS)
            )
            st.dataframe(series.describe().to_frame("Value").T, width="stretch")
        else:
            counts = (
                series.astype("string")
                .fillna("(missing)")
                .value_counts()
                .head(15)
                .rename_axis("Value")
                .reset_index(name="Records")
            )
            chart(
                px.bar(
                    counts,
                    x="Records",
                    y="Value",
                    orientation="h",
                    color_discrete_sequence=COLORS,
                )
            )
    st.divider()
    st.subheader("Schema overview")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Column": c.name,
                    "Type": c.dtype,
                    "Missing %": c.missing_pct,
                    "Distinct": c.unique_count,
                    "Constant": c.is_constant,
                    "Many categories": c.is_high_cardinality,
                }
                for c in p.columns
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    with st.expander("Convert a text column to dates"):
        st.caption(
            "This creates a new original version and clears previous cleaning and AI results. Invalid nonempty values block conversion."
        )
        date_col = st.selectbox("Date column", list(df.columns), key="date_conversion")
        fmt = st.text_input(
            "Date format",
            value="%Y-%m-%d",
            help="For example %d/%m/%Y or %Y-%m-%d %H:%M:%S",
        )
        if st.button("Validate and convert dates"):
            try:
                candidate = df.copy()
                candidate[date_col] = pd.to_datetime(
                    candidate[date_col], format=fmt, errors="raise"
                )
                name = st.session_state["dataset_filename"]
                set_dataset(candidate, get_metadata(candidate, name, 0), name)
                st.rerun()
            except (ValueError, TypeError) as exc:
                st.error(
                    f"Conversion failed. Check the format and column values. {exc}"
                )
