from __future__ import annotations
from html import escape
import numpy as np
import pandas as pd
import streamlit as st
from modules.profiler import profile_dataset
from modules.quality import assess_quality
from modules.eda import run_eda
from utils.session import get_active_dataset, has_cleaned_dataset

COLORS = ["#66E0C2", "#8896FF", "#F7BC72", "#EF87AE", "#65BDEB"]


def style():
    st.markdown(
        """<style>
    .stApp {background:radial-gradient(ellipse at 90% 0%,#182646 0%,#0b1020 48%)}
    .block-container{max-width:1480px;padding-top:1.5rem;padding-bottom:3rem}
    h1,h2,h3{letter-spacing:-.025em} h1{font-size:2.45rem!important}
    [data-testid="stSidebar"]{background:#0d1425;border-right:1px solid #26324b}
    [data-testid="stMetric"]{background:#131e31;border:1px solid #29354e;border-radius:14px;padding:17px}
    [data-testid="stMetricLabel"]{color:#a9b5cc}
    [data-testid="stMetricValue"]{font-weight:650;letter-spacing:-.04em}
    [data-testid="stVerticalBlockBorderWrapper"]>div{border-radius:16px!important}
    .hero{border:1px solid #334464;background:linear-gradient(120deg,#192a43,#172039 65%,#243854);border-radius:18px;padding:30px;margin-bottom:20px;position:relative;overflow:hidden}
    .hero p{color:#b9c5db;max-width:700px;font-size:1.05rem}
    .hero h1{max-width:900px;margin:0 0 12px}
    .step{border:1px solid #293650;border-radius:16px;padding:22px;min-height:155px;background:#121c2e}
    .step small{color:#66e0c2;font-weight:700}.step h3{margin:12px 0 8px}.step p{color:#9baac4;font-size:.9rem}
    .muted{color:#98a8c3;font-size:.88rem}
    div.stButton>button,div.stDownloadButton>button{border-radius:10px;min-height:42px}
    @media(max-width:700px){.hero{padding:22px 20px}.hero p{font-size:.95rem}h1{font-size:2rem!important}.block-container{padding-top:.75rem}[data-testid="stMetric"]{padding:14px}}
    </style>""",
        unsafe_allow_html=True,
    )


def hero(label, title, description):
    st.markdown(
        f'<div class="hero"><h1>{escape(title)}</h1><p>{escape(description)}</p><span class="muted">{escape(label)} · IntelliData Studio</span></div>',
        unsafe_allow_html=True,
    )


def heading(label, title, description):
    st.title(title)
    st.caption(f"{label} · {description}")


def chart(fig, key=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=COLORS,
        font=dict(family="Arial", color="#C4D0E6", size=12),
        margin=dict(l=12, r=12, t=45, b=15),
        height=360,
        legend=dict(orientation="h", y=-0.2),
    )
    fig.update_xaxes(gridcolor="#26314a", zerolinecolor="#26314a")
    fig.update_yaxes(gridcolor="#26314a", zerolinecolor="#26314a")
    st.plotly_chart(fig, width="stretch", key=key, config={"displaylogo": False})


def require_data():
    df = get_active_dataset()
    if df is None:
        st.info("Start by importing a dataset or opening the sample workspace.")
        st.page_link(
            "studio/import_page.py",
            label="Open data import",
            icon=":material/upload_file:",
        )
        st.stop()
    if len(df.select_dtypes(include="number").columns) > 100:
        st.caption(
            "Statistical correlation summaries use the first 100 numeric columns. Chart studio lets you choose a smaller set."
        )
    st.caption(
        f"{st.session_state.get('dataset_filename', 'Dataset')} · {'Cleaned version' if has_cleaned_dataset() else 'Original version'} · {len(df):,} rows"
    )
    return df


@st.cache_data(max_entries=8, ttl=1800, show_spinner=False)
def analyze(df):
    profile = profile_dataset(df)
    quality = assess_quality(df, profile)
    eda = run_eda(df, profile.numerical_columns[:100], profile.categorical_columns)
    return profile, quality, eda


def demo_data():
    rng = np.random.default_rng(42)
    n = 1800
    region = rng.choice(
        ["North America", "Europe", "Asia Pacific", "Latin America"],
        n,
        p=[0.38, 0.29, 0.23, 0.10],
    )
    channel = rng.choice(["Organic", "Paid search", "Partner", "Direct"], n)
    revenue = np.round(rng.lognormal(5.5, 0.7, n), 2)
    df = pd.DataFrame(
        {
            "order_id": np.arange(10001, 10001 + n),
            "order_date": pd.date_range("2025-01-01", periods=n, freq="8h"),
            "region": region,
            "channel": channel,
            "segment": rng.choice(
                ["Enterprise", "Growth", "Starter"], n, p=[0.20, 0.35, 0.45]
            ),
            "revenue": revenue,
            "cost": np.round(revenue * rng.uniform(0.35, 0.80, n), 2),
            "satisfaction": np.round(rng.uniform(2.5, 5, n), 1),
            "delivery_days": rng.integers(1, 12, n),
        }
    )
    df.loc[rng.choice(n, 65, False), "satisfaction"] = np.nan
    df.loc[rng.choice(n, 24, False), "region"] = None
    return pd.concat([df, df.iloc[:18]], ignore_index=True)


def completeness(df):
    return pd.DataFrame(
        {"Column": df.columns, "Missing %": (df.isna().mean() * 100).values}
    ).sort_values("Missing %", ascending=False)
