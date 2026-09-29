import pandas as pd
import plotly.express as px
import streamlit as st
from studio.common import heading, require_data, analyze, chart, COLORS

heading(
    "04 / Discover",
    "Find the signal. Shape the story.",
    "Build an interactive chart, compare groups, and follow changes over time.",
)
df = require_data()
p, q, e = analyze(df)
a, b = st.columns([1, 3])
with a:
    with st.container(border=True):
        kind = st.selectbox(
            "Visualization",
            [
                "Distribution",
                "Relationship",
                "Category comparison",
                "Box plot",
                "Correlation",
                "Time trend",
            ],
        )
        numeric = [
            c for c in p.numerical_columns if not pd.api.types.is_bool_dtype(df[c])
        ]
        cats = p.categorical_columns
        filter_col = st.selectbox("Filter by category", ["No filter"] + cats)
        data = df
        if filter_col != "No filter":
            values = sorted(df[filter_col].dropna().astype(str).unique().tolist())
            selected = st.multiselect(
                "Include values", values[:500], default=values[:500]
            )
            data = df[df[filter_col].astype(str).isin(selected)]
            if len(values) > 500:
                st.caption("Showing the first 500 filter values.")
        st.caption(f"{len(data):,} records in this view")
        title = st.text_input(
            "Chart title", placeholder="Give your chart a story", max_chars=120
        )
        st.caption("Filters only affect this chart. They do not modify your dataset.")
with b:
    if data.empty:
        st.info("No rows match your filters. Select another category.")
        st.stop()
    fig = None
    if (
        kind
        in ["Distribution", "Relationship", "Box plot", "Correlation", "Time trend"]
        and not numeric
    ):
        st.info("This chart needs numeric columns. Try Category comparison.")
        st.stop()
    if kind == "Distribution":
        x = st.selectbox("Measure", numeric)
        bins = st.slider("Number of bins", 10, 100, 35)
        fig = px.histogram(
            data, x=x, nbins=bins, color_discrete_sequence=COLORS, marginal="box"
        )
    elif kind == "Relationship":
        if len(numeric) < 2:
            st.info("Choose a dataset with at least two numeric columns.")
            st.stop()
        xcol, ycol, ccol = st.columns(3)
        x = xcol.selectbox("X axis", numeric)
        y = ycol.selectbox("Y axis", numeric, index=1)
        color = ccol.selectbox("Color", ["None"] + cats)
        plot = data.sample(min(5000, len(data)), random_state=42)
        st.caption(
            f"Plotting {len(plot):,} sampled rows; axis selection does not change your data."
        )
        fig = px.scatter(
            plot,
            x=x,
            y=y,
            color=None if color == "None" else color,
            opacity=0.65,
            color_discrete_sequence=COLORS,
        )
    elif kind == "Category comparison":
        if not cats:
            st.info("This chart needs a categorical column.")
            st.stop()
        a1, a2, a3 = st.columns(3)
        group = a1.selectbox("Group", cats)
        agg = a2.selectbox("Aggregation", ["Count", "Sum", "Mean", "Median"])
        measure = (
            a3.selectbox("Measure", numeric) if agg != "Count" and numeric else None
        )
        if agg != "Count" and measure is None:
            st.info("A numeric column is required for this aggregation.")
            st.stop()
        if agg == "Count":
            series = data.groupby(group, dropna=False).size()
        else:
            series = data.groupby(group, dropna=False)[measure].agg(agg.lower())
        grouped = (
            series.sort_values(ascending=False).head(20).rename("Value").reset_index()
        )
        fig = px.bar(grouped, x=group, y="Value", color_discrete_sequence=COLORS)
        st.caption("Top 20 groups by the selected aggregation.")
    elif kind == "Box plot":
        x = st.selectbox("Measure", numeric)
        group = st.selectbox("Group by", ["None"] + cats)
        plot = data
        if group != "None":
            top = data[group].value_counts().head(20).index
            plot = data[data[group].isin(top)]
        fig = px.box(
            plot,
            y=x,
            x=None if group == "None" else group,
            color_discrete_sequence=COLORS,
            points=False,
        )
        st.caption("Up to 20 groups; all matching rows are included in each box.")
    elif kind == "Correlation":
        selected = st.multiselect("Numeric columns", numeric, default=numeric[:12])
        method = st.radio("Method", ["pearson", "spearman"], horizontal=True)
        if len(selected) < 2:
            st.info("Select at least two columns.")
            st.stop()
        if len(selected) > 40:
            st.info("Select up to 40 columns for a readable correlation map.")
            st.stop()
        fig = px.imshow(
            data[selected].corr(method=method),
            zmin=-1,
            zmax=1,
            text_auto=".2f",
            color_continuous_scale=["#ED91B3", "#172139", "#66E0C2"],
        )
        st.caption("Correlation measures association; it does not establish causation.")
    elif kind == "Time trend":
        if not p.date_columns:
            st.info("Convert a date column in Data explorer to enable time trends.")
            st.stop()
        a1, a2, a3 = st.columns(3)
        date = a1.selectbox("Date", p.date_columns)
        measure = a2.selectbox("Measure", numeric)
        freq = a3.selectbox("Interval", ["Day", "Week", "Month"])
        agg = st.radio("Aggregation", ["sum", "mean", "count"], horizontal=True)
        series = (
            data.set_index(date)[measure]
            .resample({"Day": "D", "Week": "W", "Month": "MS"}[freq])
            .agg(agg)
        )
        fig = px.area(
            series.rename(measure).reset_index(),
            x=date,
            y=measure,
            color_discrete_sequence=COLORS,
        )
    if fig is not None:
        fig.update_layout(title=title or kind)
        chart(fig)
        st.download_button(
            "Download interactive chart",
            fig.to_html(include_plotlyjs=True).encode(),
            "chart.html",
            "text/html",
        )
