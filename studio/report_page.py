import io
import streamlit as st
from studio.common import heading, require_data
from modules.exports import csv_bytes, report_html, export_bundle
from utils.session import get_cleaning_log

heading(
    "06 / Share",
    "Package your next decision.",
    "Take your data, quality assessment, and cleaning history with you.",
)
df = require_data()
name = st.session_state["dataset_filename"]
log = get_cleaning_log()
a, b, c = st.columns(3)
with a:
    with st.container(border=True):
        st.subheader("Clean, portable data")
        st.write(
            "Export the active version for your next analysis. Text cells that resemble spreadsheet formulas are escaped in CSV."
        )
        st.download_button(
            "Download CSV", csv_bytes(df), "dataset.csv", "text/csv", width="stretch"
        )
        try:
            data = io.BytesIO()
            df.to_parquet(data, index=False)
            st.download_button(
                "Download Parquet",
                data.getvalue(),
                "dataset.parquet",
                "application/octet-stream",
                width="stretch",
            )
        except (ValueError, TypeError, ImportError):
            st.caption("Parquet export is unavailable for these column types.")
with b:
    with st.container(border=True):
        st.subheader("Analysis report")
        st.write(
            "A standalone HTML report with quality metrics, column profiles, statistics, and the cleaning audit. Open in a browser or print to PDF."
        )
        st.download_button(
            "Download HTML report",
            report_html(df, name, log),
            "intellidata_report.html",
            "text/html",
            width="stretch",
        )
with c:
    with st.container(border=True):
        st.subheader("The complete handoff")
        st.write(
            "One ZIP with the active dataset, report, machine-readable quality scores, applied recipe, and audit trail."
        )
        st.download_button(
            "Download workspace bundle",
            export_bundle(df, name, log, st.session_state.get("applied_recipe")),
            "intellidata_workspace.zip",
            "application/zip",
            type="primary",
            width="stretch",
        )
st.info(
    "Exports describe the active dataset. Workspace data and chat are held in the current server session; download your work before closing it."
)
