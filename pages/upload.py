import streamlit as st
from modules.loader import (
    EmptyDatasetError,
    FileTooLargeError,
    SUPPORTED_EXTENSIONS,
    UnsupportedFileTypeError,
    get_metadata,
    load_dataset,
    validate_file,
)
from utils.config import load_settings
from utils.logger import get_logger
from utils.session import get_dataset, has_dataset, init_session_state, set_dataset

settings = load_settings()
logger = get_logger("pages.upload", settings.logs_dir, settings.log_level)

init_session_state()

st.title("Upload Dataset")
st.caption(
    f"Supported formats: {', '.join(ext.lstrip('.').upper() for ext in SUPPORTED_EXTENSIONS)} "
    f"· Max size: {settings.max_upload_mb}MB"
)

uploaded_file = st.file_uploader(
    "Choose a dataset file",
    type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
    accept_multiple_files=False,
)

if uploaded_file is not None:
    try:
        validate_file(uploaded_file.name, uploaded_file.size, settings.max_upload_mb)
        df = load_dataset(uploaded_file, uploaded_file.name)
        metadata = get_metadata(df, uploaded_file.name, uploaded_file.size)

        set_dataset(df, metadata, uploaded_file.name)

        logger.info(
            "Dataset loaded | file=%s rows=%d cols=%d",
            uploaded_file.name,
            metadata.n_rows,
            metadata.n_columns,
        )

    except (UnsupportedFileTypeError, FileTooLargeError, EmptyDatasetError) as exc:
        st.error(str(exc))
        logger.warning("Upload rejected | file=%s | reason=%s", uploaded_file.name, exc)
        st.stop()

    except Exception as exc:  # noqa: BLE001
        st.error(
            "Couldn't read this file — it may be corrupted or not "
            "actually formatted as its extension suggests."
        )
        logger.error("Unexpected load failure | file=%s", uploaded_file.name, exc_info=exc)
        st.stop()


if has_dataset():
    df = get_dataset()
    metadata = st.session_state["dataset_metadata"]
    st.success(f"Loaded **{metadata.filename}**")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{metadata.n_rows:,}")
    col2.metric("Columns", metadata.n_columns)
    col3.metric("File size", f"{metadata.file_size_mb} MB")
    col4.metric("In-memory size", f"{metadata.memory_usage_mb} MB")

    with st.expander("Column data types"):
        st.write(metadata.dtype_counts)

    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width="stretch")

else:
    st.info("Upload a file above to get started.")