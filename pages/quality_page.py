import pandas as pd
import streamlit as st

from modules.profiler import profile_dataset
from modules.quality import assess_quality
from utils.config import load_settings
from utils.logger import get_logger
from utils.session import get_dataset, has_dataset, init_session_state

settings = load_settings()
logger = get_logger("pages.quality", settings.logs_dir, settings.log_level)
init_session_state()

st.title("Data Quality Assessment")

if not has_dataset():
    st.info("Upload a dataset first on the **Upload** page.")
    st.stop()

@st.cache_data(show_spinner="Assessing data quality...")
def _cached_quality(df: pd.DataFrame):
    profile = profile_dataset(df)
    return assess_quality(df, profile)


df = get_dataset()
quality = _cached_quality(df)

logger.info("Quality assessed | overall_score=%.1f", quality.overall_score)

def _status_label(score: float) -> tuple[str, str]:
    if score >= 90:
        return "Excellent", "success"
    if score >= 75:
        return "Good", "info"
    if score >= 60:
        return "Fair", "warning"
    return "Needs Attention", "error"

label, status = _status_label(quality.overall_score)

st.metric("Overall Data Quality Score", f"{quality.overall_score}%")
getattr(st, status)(f"**{label}** — {quality.overall_score}% overall quality")

st.subheader("Score Breakdown")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Completeness", f"{quality.completeness_score}%")
col2.metric("Duplicate Score", f"{quality.duplicate_score}%")
col3.metric("Consistency", f"{quality.consistency_score}%")
col4.metric("Validity", f"{quality.validity_score}%")

st.subheader("Score Breakdown")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Completeness", f"{quality.completeness_score}%")
col2.metric("Duplicate Score", f"{quality.duplicate_score}%")
col3.metric("Consistency", f"{quality.consistency_score}%")
col4.metric("Validity", f"{quality.validity_score}%")

st.subheader("Issues Found")

issues_found = False

if quality.missing_cell_count > 0:
    issues_found = True
    st.warning(f"**{quality.missing_cell_count} missing values** across the dataset.")

if quality.duplicate_row_count > 0:
    issues_found = True
    st.warning(f"**{quality.duplicate_row_count} duplicate rows** detected.")

if quality.inconsistent_columns:
    issues_found = True
    st.warning(
        f"**{len(quality.inconsistent_columns)} columns with mixed data types**: "
        f"{', '.join(quality.inconsistent_columns)}"
    )

if quality.outlier_columns:
    issues_found = True
    total_outliers = sum(quality.outlier_columns.values())
    outlier_detail = ", ".join(
        f"{col} ({count})" for col, count in quality.outlier_columns.items()
    )
    st.warning(f"**{total_outliers} outliers** detected: {outlier_detail}")

if not issues_found:
    st.success("No data quality issues detected. This dataset is clean.")