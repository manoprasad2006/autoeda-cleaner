import pandas as pd
import streamlit as st

from modules.ai.cleaning_explainer import action_key, explain_all_actions
from modules.ai.client import GeminiClient
from modules.cleaner import auto_clean
from modules.profiler import profile_dataset
from modules.quality import assess_quality
from utils.config import load_settings
from utils.logger import get_logger
from utils.session import (
    get_cleaning_explanations,
    get_cleaning_log,
    get_dataset,
    has_cleaned_dataset,
    has_cleaning_explanations,
    has_dataset,
    init_session_state,
    set_cleaned_dataset,
    set_cleaning_explanations,
)

settings = load_settings()
logger = get_logger("pages.cleaning", settings.logs_dir, settings.log_level)
init_session_state()

st.title("Intelligent Data Cleaning")

if not has_dataset():
    st.info("Upload a dataset first on the **Upload** page.")
    st.stop()

df = get_dataset()

if st.button("Run Automatic Cleaning", type="primary"):
    with st.spinner("Cleaning dataset..."):
        cleaned_df, log = auto_clean(df)
        set_cleaned_dataset(cleaned_df, log)
    logger.info(
        "Cleaning complete | actions=%d | rows %d -> %d",
        len(log.actions),
        len(df),
        len(cleaned_df),
    )
    st.rerun()

if not has_cleaned_dataset():
    st.info("Click the button above to run automatic cleaning.")
    st.stop()

cleaned_df = st.session_state["cleaned_dataset"]
log = get_cleaning_log()

st.success(f"Cleaning complete — {len(log.actions)} action(s) taken.")


@st.cache_data(show_spinner="Comparing quality scores...")
def _score(data: pd.DataFrame):
    profile = profile_dataset(data)
    return assess_quality(data, profile)


quality_before = _score(df)
quality_after = _score(cleaned_df)

st.subheader("Quality Score: Before vs After")

col1, col2, col3 = st.columns(3)
col1.metric("Before", f"{quality_before.overall_score}%")
col2.metric("After", f"{quality_after.overall_score}%")
col3.metric(
    "Improvement",
    f"{round(quality_after.overall_score - quality_before.overall_score, 2)}%",
)

st.subheader("Rows: Before vs After")
col1, col2 = st.columns(2)
col1.metric("Rows before", f"{len(df):,}")
col2.metric("Rows after", f"{len(cleaned_df):,}")

st.subheader("Cleaning Log")

log_df = log.to_dataframe()

if log_df.empty:
    st.info("No cleaning actions were necessary — the dataset was already clean.")
else:
    st.dataframe(
        log_df[["column", "issue", "method", "rows_affected", "reason"]],
        width="stretch",
        hide_index=True,
    )

st.subheader("AI-Generated Explanations")

if not settings.gemini_api_key:
    st.info(
        "Add a GEMINI_API_KEY in your .env file to get plain-language, "
        "business-friendly explanations of each cleaning decision."
    )
elif log_df.empty:
    pass  # nothing to explain if the dataset needed no cleaning
else:
    if st.button("Generate AI Explanations"):
        with st.spinner("Asking Gemini to explain each cleaning decision..."):
            client = GeminiClient(
                api_key=settings.gemini_api_key, model=settings.gemini_model
            )
            explanations = explain_all_actions(client, log.actions)
            set_cleaning_explanations(explanations)
        st.rerun()

    if has_cleaning_explanations():
        explanations = get_cleaning_explanations()
        for i, action in enumerate(log.actions):
            with st.expander(f"{action.column} — {action.issue}"):
                st.write(explanations.get(action_key(action, i), action.reason))

st.subheader("Download Cleaned Dataset")
csv_bytes = cleaned_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download as CSV",
    data=csv_bytes,
    file_name="cleaned_dataset.csv",
    mime="text/csv",
)