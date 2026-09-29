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
    /* Global App Canvas */
    .stApp {
        background: radial-gradient(ellipse at 90% 0%, #182646 0%, #0b1020 48%);
        background-attachment: fixed;
    }

    /* Responsive Main Container */
    .block-container {
        max-width: 1480px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    @media (max-width: 992px) {
        .block-container {
            padding-left: 1.25rem !important;
            padding-right: 1.25rem !important;
            padding-top: 1.2rem !important;
        }
    }
    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.85rem !important;
            padding-right: 0.85rem !important;
            padding-top: 0.85rem !important;
            padding-bottom: 2.5rem !important;
            max-width: 100% !important;
        }
    }

    /* Headings and Typography */
    h1, h2, h3 {
        letter-spacing: -0.025em;
        word-break: break-word;
    }
    h1 { font-size: 2.35rem !important; }
    @media (max-width: 768px) {
        h1 { font-size: 1.65rem !important; line-height: 1.25 !important; }
        h2 { font-size: 1.35rem !important; line-height: 1.3 !important; }
        h3 { font-size: 1.15rem !important; line-height: 1.35 !important; }
    }
    @media (max-width: 480px) {
        h1 { font-size: 1.45rem !important; }
        h2 { font-size: 1.2rem !important; }
        h3 { font-size: 1.05rem !important; }
    }

    /* Hero Component */
    .hero {
        border: 1px solid #334464;
        background: linear-gradient(120deg, #192a43, #172039 65%, #243854);
        border-radius: 18px;
        padding: 28px;
        margin-bottom: 20px;
        position: relative;
        overflow: hidden;
    }
    .hero p { color: #b9c5db; max-width: 700px; font-size: 1.05rem; }
    .hero h1 { max-width: 900px; margin: 0 0 12px; }
    @media (max-width: 768px) {
        .hero {
            padding: 18px 16px !important;
            border-radius: 14px !important;
            margin-bottom: 16px !important;
        }
        .hero h1 { font-size: 1.55rem !important; margin: 0 0 8px !important; }
        .hero p { font-size: 0.88rem !important; line-height: 1.45 !important; }
    }

    /* Cards & Step Guides */
    .step {
        border: 1px solid #293650;
        border-radius: 16px;
        padding: 20px;
        min-height: 140px;
        background: #121c2e;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
    }
    .step small { color: #66e0c2; font-weight: 700; font-size: 0.78rem; text-transform: uppercase; }
    .step h3 { margin: 10px 0 6px; font-size: 1.18rem; }
    .step p { color: #9baac4; font-size: 0.88rem; line-height: 1.4; margin: 0; }
    @media (max-width: 768px) {
        .step {
            min-height: auto !important;
            padding: 14px !important;
            border-radius: 12px !important;
        }
        .step h3 { margin: 6px 0 4px !important; font-size: 1.05rem !important; }
        .step p { font-size: 0.82rem !important; }
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: #131e31;
        border: 1px solid #29354e;
        border-radius: 14px;
        padding: 16px;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    [data-testid="stMetricLabel"] { color: #a9b5cc; font-size: 0.88rem; }
    [data-testid="stMetricValue"] { font-weight: 650; letter-spacing: -0.04em; }
    @media (max-width: 768px) {
        [data-testid="stMetric"] {
            padding: 10px 12px !important;
            border-radius: 10px !important;
        }
        [data-testid="stMetricLabel"] { font-size: 0.78rem !important; }
        [data-testid="stMetricValue"] { font-size: 1.35rem !important; }
    }

    /* Responsive Column Grids */
    @media (max-width: 768px) {
        /* Multi-column layouts wrap gracefully on mobile */
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 10px !important;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="column"] {
            min-width: calc(50% - 6px) !important;
            flex: 1 1 calc(50% - 6px) !important;
        }
    }
    @media (max-width: 520px) {
        /* On small phones, stack full width for buttons and cards */
        [data-testid="stHorizontalBlock"] > [data-testid="column"] {
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }
        /* Allow metric pairs to remain side-by-side on mobile phones */
        [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > [data-testid="column"] {
            min-width: calc(50% - 6px) !important;
            flex: 1 1 calc(50% - 6px) !important;
        }
    }

    /* Mobile DataFrames and Tables */
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        width: 100% !important;
        max-width: 100% !important;
        overflow-x: auto !important;
        -webkit-overflow-scrolling: touch !important;
        border-radius: 12px;
    }

    /* Scrollable Mobile Tabs */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        overflow-x: auto !important;
        flex-wrap: nowrap !important;
        -webkit-overflow-scrolling: touch !important;
        scrollbar-width: none !important;
        gap: 4px !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab-list"]::-webkit-scrollbar {
        display: none;
    }
    [data-testid="stTabs"] button[role="tab"] {
        white-space: nowrap !important;
        padding: 8px 14px !important;
        font-size: 0.88rem !important;
    }

    /* Touch-Friendly Form Elements */
    div.stButton > button, div.stDownloadButton > button {
        border-radius: 10px;
        min-height: 44px !important;
        font-size: 0.92rem !important;
        touch-action: manipulation;
    }
    div[data-baseweb="select"] {
        min-height: 42px !important;
    }
    input[type="text"], input[type="number"] {
        min-height: 42px !important;
        font-size: 0.95rem !important;
    }

    /* Mobile Sidebar Styling */
    [data-testid="stSidebar"] {
        background: #0d1425;
        border-right: 1px solid #26324b;
    }
    @media (max-width: 768px) {
        [data-testid="stSidebar"] {
            width: min(85vw, 320px) !important;
            box-shadow: 4px 0 24px rgba(0, 0, 0, 0.75) !important;
        }
        [data-testid="stSidebarCollapseButton"] {
            background: rgba(20, 31, 53, 0.9) !important;
            border-radius: 8px !important;
            border: 1px solid #29354e !important;
        }
    }

    .muted { color: #98a8c3; font-size: 0.88rem; }
    [data-testid="stVerticalBlockBorderWrapper"] > div { border-radius: 16px !important; }
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
        font=dict(family="Arial", color="#C4D0E6", size=11),
        margin=dict(l=8, r=8, t=36, b=12),
        height=320,
        autosize=True,
        legend=dict(orientation="h", y=-0.25, x=0.0),
    )
    fig.update_xaxes(gridcolor="#26314a", zerolinecolor="#26314a")
    fig.update_yaxes(gridcolor="#26314a", zerolinecolor="#26314a")
    st.plotly_chart(
        fig, width="stretch", key=key, config={"displaylogo": False, "responsive": True}
    )


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
