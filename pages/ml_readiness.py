import pandas as pd
import streamlit as st

from modules.ai.client import GeminiClient
from modules.ai.ml_recommendation import generate_ml_recommendations
from modules.eda import run_eda
from modules.ml_readiness import assess_ml_readiness
from modules.profiler import profile_dataset
from modules.quality import assess_quality
from utils.config import load_settings
from utils.logger import get_logger
from utils.session import (
    get_active_dataset,
    get_ml_recommendations,
    has_dataset,
    has_ml_recommendations,
    init_session_state,
    set_ml_recommendations,
)

settings = load_settings()
logger = get_logger("pages.ml_readiness", settings.logs_dir, settings.log_level)
init_session_state()

st.title("ML Readiness")

if not has_dataset():
    st.info("Upload a dataset first on the **Upload** page.")
    st.stop()

df = get_active_dataset()


@st.cache_data(show_spinner="Analyzing dataset...")
def _cached_pipeline(data: pd.DataFrame):
    profile = profile_dataset(data)
    quality = assess_quality(data, profile)
    eda = run_eda(data, profile.numerical_columns, profile.categorical_columns)
    return profile, quality, eda


profile, quality, eda = _cached_pipeline(df)

target_options = ["(none - unsupervised)"] + list(df.columns)
target_choice = st.selectbox(
    "Target column (optional — what are you trying to predict?)",
    target_options,
)
target_column = None if target_choice == "(none - unsupervised)" else target_choice


@st.cache_data(show_spinner="Assessing ML readiness...")
def _cached_readiness(data: pd.DataFrame, _profile, _quality, _eda, target: str | None):
    return assess_ml_readiness(data, _profile, _quality, _eda, target_column=target)


readiness = _cached_readiness(df, profile, quality, eda, target_column)

st.metric("Readiness Score", f"{readiness.readiness_score}%")
st.caption(f"Inferred task: **{readiness.inferred_task}**")

st.subheader("Issues Found")
if readiness.issues:
    for issue in readiness.issues:
        icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}[issue.severity]
        with st.expander(f"{icon} {issue.category}"):
            st.write(issue.description)
            st.caption(f"Affected columns: {', '.join(issue.affected_columns)}")
else:
    st.success("No readiness issues found.")

st.subheader("Algorithm Recommendations")

if not settings.gemini_api_key:
    st.info(
        "Add a GEMINI_API_KEY in your .env file to generate AI algorithm "
        "recommendations."
    )
else:
    if st.button("Generate Recommendations", type="primary"):
        with st.spinner("Recommending algorithms..."):
            client = GeminiClient(
                api_key=settings.gemini_api_key, model=settings.gemini_model
            )
            recs = generate_ml_recommendations(client, profile, eda, readiness)
            set_ml_recommendations(recs)
        logger.info("ML recommendations generated | target=%s", target_column)
        st.rerun()

    if has_ml_recommendations():
        st.markdown(get_ml_recommendations())