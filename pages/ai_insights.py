import pandas as pd
import streamlit as st

from modules.ai.business_insights import generate_business_insights
from modules.ai.client import GeminiClient
from modules.ai.executive_summary import generate_executive_summary
from modules.ai.feature_engineering import generate_feature_engineering_suggestions
from modules.eda import run_eda
from modules.profiler import profile_dataset
from modules.quality import assess_quality
from utils.config import load_settings
from utils.logger import get_logger
from utils.session import (
    get_active_dataset,
    get_business_insights,
    get_cleaning_log,
    get_executive_summary,
    get_feature_engineering,
    has_business_insights,
    has_cleaned_dataset,
    has_dataset,
    has_executive_summary,
    has_feature_engineering,
    init_session_state,
    set_business_insights,
    set_executive_summary,
    set_feature_engineering,
)

settings = load_settings()
logger = get_logger("pages.ai_insights", settings.logs_dir, settings.log_level)
init_session_state()

st.title("AI Insights")

if not has_dataset():
    st.info("Upload a dataset first on the **Upload** page.")
    st.stop()

if not has_cleaned_dataset():
    st.info(
        "Run automatic cleaning first on the **Cleaning** page — the "
        "executive summary describes what was found and fixed, so it "
        "needs a completed cleaning pass to work from."
    )
    st.stop()

df = get_active_dataset()
log = get_cleaning_log()


@st.cache_data(show_spinner="Analyzing dataset...")
def _cached_pipeline(data: pd.DataFrame):
    profile = profile_dataset(data)
    quality = assess_quality(data, profile)
    eda = run_eda(data, profile.numerical_columns, profile.categorical_columns)
    return profile, quality, eda


profile, quality, eda = _cached_pipeline(df)

st.subheader("Executive Summary")

if not settings.gemini_api_key:
    st.info(
        "Add a GEMINI_API_KEY in your .env file to generate an AI-written "
        "executive summary."
    )
else:
    if st.button("Generate Executive Summary", type="primary"):
        with st.spinner("Writing executive summary..."):
            client = GeminiClient(
                api_key=settings.gemini_api_key, model=settings.gemini_model
            )
            summary = generate_executive_summary(client, profile, quality, log, eda)
            set_executive_summary(summary)
        logger.info("Executive summary generated | length=%d", len(summary))
        st.rerun()

    if has_executive_summary():
        st.markdown(get_executive_summary())

st.subheader("Business Insights")

if not settings.gemini_api_key:
    st.info(
        "Add a GEMINI_API_KEY in your .env file to generate AI business "
        "insights."
    )
else:
    if st.button("Generate Business Insights", type="primary"):
        with st.spinner("Identifying business insights..."):
            client = GeminiClient(
                api_key=settings.gemini_api_key, model=settings.gemini_model
            )
            insights = generate_business_insights(client, quality, eda)
            set_business_insights(insights)
        logger.info("Business insights generated | length=%d", len(insights))
        st.rerun()

    if has_business_insights():
        st.markdown(get_business_insights())

st.subheader("Feature Engineering Suggestions")

if not settings.gemini_api_key:
    st.info(
        "Add a GEMINI_API_KEY in your .env file to generate AI feature "
        "engineering suggestions."
    )
else:
    if st.button("Generate Feature Suggestions", type="primary"):
        with st.spinner("Identifying feature engineering opportunities..."):
            client = GeminiClient(
                api_key=settings.gemini_api_key, model=settings.gemini_model
            )
            suggestions = generate_feature_engineering_suggestions(client, profile, eda)
            set_feature_engineering(suggestions)
        logger.info("Feature suggestions generated | length=%d", len(suggestions))
        st.rerun()

    if has_feature_engineering():
        st.markdown(get_feature_engineering())