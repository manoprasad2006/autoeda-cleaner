import io
import pandas as pd
import streamlit as st
from studio.common import heading, demo_data
from modules.loader import load_dataset, validate_file, get_metadata
from utils.config import load_settings
from utils.session import set_dataset

heading(
    "01 / Connect",
    "Bring your data into focus.",
    "Import one table, check the preview, and open your workspace. Replacing a dataset clears its previous results.",
)
settings = load_settings()
a, b = st.columns([1.7, 1])
with a:
    with st.container(border=True):
        st.subheader("Import a dataset")
        upload = st.file_uploader(
            "Drop a file here",
            type=["csv", "xlsx", "json", "parquet"],
            help=f"Up to {settings.max_upload_mb} MB, 500,000 rows and 500 columns.",
        )
        delimiter = st.selectbox(
            "CSV separator", ["Auto / comma", "Semicolon", "Tab", "Pipe"]
        )
        st.caption(
            "Excel imports the first sheet. JSON must contain a flat table. Data stays in this server session unless you explicitly use AI."
        )
        if upload is not None:
            try:
                validate_file(upload.name, upload.size, settings.max_upload_mb)
                raw = upload.getvalue()
                if upload.name.lower().endswith(".csv") and delimiter != "Auto / comma":
                    separators = {"Semicolon": ";", "Tab": "\t", "Pipe": "|"}
                    parsed = pd.read_csv(io.BytesIO(raw), sep=separators[delimiter])
                    data = load_dataset(
                        io.BytesIO(parsed.to_csv(index=False).encode()), "parsed.csv"
                    )
                else:
                    data = load_dataset(io.BytesIO(raw), upload.name)
                if data.attrs.get("import_warning"):
                    st.warning(data.attrs["import_warning"])
                st.caption(f"{len(data):,} rows · {len(data.columns)} columns")
                st.dataframe(data.head(8), hide_index=True, width="stretch")
                if st.button("Open this dataset", type="primary"):
                    set_dataset(
                        data, get_metadata(data, upload.name, upload.size), upload.name
                    )
                    st.switch_page("studio/overview_page.py")
            except Exception as exc:
                st.error(f"Import could not finish: {exc}")
with b:
    with st.container(border=True):
        st.subheader("Try it with a story")
        st.write(
            "A synthetic commerce dataset with orders, regions, acquisition channels, revenue, costs, and a few realistic quality issues."
        )
        if st.button("Load commerce demo", width="stretch"):
            data = demo_data()
            set_dataset(
                data,
                get_metadata(data, "Commerce demo · synthetic", 0),
                "Commerce demo · synthetic",
            )
            st.switch_page("studio/overview_page.py")
        st.divider()
        st.markdown(
            "**Your workflow**\n\n1. Inspect columns and data types\n2. Preview a cleaning recipe\n3. Explore patterns and trends\n4. Export data and a report"
        )
