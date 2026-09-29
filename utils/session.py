"""
Centralized Streamlit session_state access.

Why this exists: st.session_state is a global dict shared across every
page of a multipage Streamlit app. Without a single place defining what
keys exist, it's easy to typo a key name in one page and silently break
another page that reads it. Every later phase (cleaning log, chat
history, EDA results) adds keys here rather than sprinkling raw string
literals through the codebase.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

# --- Key names (single source of truth) ---
DATASET = "dataset"  # pd.DataFrame — the working dataset
DATASET_METADATA = "dataset_metadata"  # DatasetMetadata — see modules/loader.py
DATASET_FILENAME = "dataset_filename"  # str — original uploaded filename
CLEANED_DATASET = "cleaned_dataset"  # pd.DataFrame — output of modules/cleaner.py
CLEANING_LOG = "cleaning_log"  # CleaningLog — see modules/cleaner.py
CLEANING_EXPLANATIONS = (
    "cleaning_explanations"  # dict[str, str] — see modules/ai/cleaning_explainer.py
)
WORKFLOW_STATE = "workflow_state"  # WorkflowState — governed multi-agent state


def init_session_state() -> None:
    """
    Ensure every key this app relies on exists in session_state, even
    before a file is uploaded. Call once at the top of every page —
    it's a no-op after the first call (setdefault doesn't overwrite).

    This avoids `KeyError` / `AttributeError` scattered across pages
    from checking `if "dataset" in st.session_state` everywhere.
    """
    st.session_state.setdefault(DATASET, None)
    st.session_state.setdefault(DATASET_METADATA, None)
    st.session_state.setdefault(DATASET_FILENAME, None)
    st.session_state.setdefault(CLEANED_DATASET, None)
    st.session_state.setdefault(CLEANING_LOG, None)
    st.session_state.setdefault(CLEANING_EXPLANATIONS, None)
    st.session_state.setdefault(WORKFLOW_STATE, None)


def set_dataset(df: pd.DataFrame, metadata: Any, filename: str) -> None:
    """Store a newly loaded dataset and its metadata in session_state."""
    clear_derived()
    st.session_state["dataset_version"] = st.session_state.get("dataset_version", 0) + 1
    st.session_state[DATASET] = df.copy(deep=True)
    st.session_state[DATASET_METADATA] = metadata
    st.session_state[DATASET_FILENAME] = filename
    from workflow.state import WorkflowState

    st.session_state[WORKFLOW_STATE] = WorkflowState.create_initial(
        df=df, filename=filename, metadata=metadata
    )
    _log_event("dataset_loaded", df)


def has_dataset() -> bool:
    """True once a dataset has been successfully uploaded this session."""
    return st.session_state.get(DATASET) is not None


def get_dataset() -> pd.DataFrame | None:
    return st.session_state.get(DATASET)


def set_cleaned_dataset(df: pd.DataFrame, log: Any) -> None:
    """Store the cleaned dataset and its cleaning log."""
    clear_derived(keep_cleaning=True)
    st.session_state["dataset_version"] = st.session_state.get("dataset_version", 0) + 1
    st.session_state[CLEANED_DATASET] = df.copy(deep=True)
    st.session_state[CLEANING_LOG] = log
    _log_event("cleaning_applied", df)


def has_cleaned_dataset() -> bool:
    """True once auto_clean() has run this session."""
    return st.session_state.get(CLEANED_DATASET) is not None


def get_active_dataset() -> pd.DataFrame | None:
    """Returns the cleaned dataset if cleaning has run, otherwise the
    original upload. Every phase from here on (EDA, visualization, ML)
    should call this instead of get_dataset(), so they automatically
    work on cleaned data once it exists, without each page needing to
    know or check whether cleaning has happened yet."""
    if has_cleaned_dataset():
        return st.session_state.get(CLEANED_DATASET)
    return get_dataset()


def get_cleaning_log() -> Any:
    return st.session_state.get(CLEANING_LOG)


def set_cleaning_explanations(explanations: dict) -> None:
    st.session_state[CLEANING_EXPLANATIONS] = explanations


def get_cleaning_explanations() -> dict | None:
    return st.session_state.get(CLEANING_EXPLANATIONS)


def has_cleaning_explanations() -> bool:
    return st.session_state.get(CLEANING_EXPLANATIONS) is not None


ML_RECOMMENDATIONS = "ml_recommendations"


def set_ml_recommendations(recs: str) -> None:
    st.session_state[ML_RECOMMENDATIONS] = recs


def get_ml_recommendations() -> str | None:
    return st.session_state.get(ML_RECOMMENDATIONS)


def has_ml_recommendations() -> bool:
    return st.session_state.get(ML_RECOMMENDATIONS) is not None


BUSINESS_INSIGHTS = "business_insights"


def set_business_insights(insights: str) -> None:
    st.session_state[BUSINESS_INSIGHTS] = insights


def get_business_insights() -> str | None:
    return st.session_state.get(BUSINESS_INSIGHTS)


def has_business_insights() -> bool:
    return st.session_state.get(BUSINESS_INSIGHTS) is not None


CHAT_SESSION = "chat_session"


def set_chat_session(session: any) -> None:
    st.session_state[CHAT_SESSION] = session


def get_chat_session() -> any:
    return st.session_state.get(CHAT_SESSION)


def has_chat_session() -> bool:
    return st.session_state.get(CHAT_SESSION) is not None


EXECUTIVE_SUMMARY = "executive_summary"


def set_executive_summary(summary: str) -> None:
    st.session_state[EXECUTIVE_SUMMARY] = summary


def get_executive_summary() -> str | None:
    return st.session_state.get(EXECUTIVE_SUMMARY)


def has_executive_summary() -> bool:
    return st.session_state.get(EXECUTIVE_SUMMARY) is not None


FEATURE_ENGINEERING = "feature_engineering"


def set_feature_engineering(fe: str) -> None:
    st.session_state[FEATURE_ENGINEERING] = fe


def get_feature_engineering() -> str | None:
    return st.session_state.get(FEATURE_ENGINEERING)


def has_feature_engineering() -> bool:
    return st.session_state.get(FEATURE_ENGINEERING) is not None


def get_workflow_state() -> Any | None:
    return st.session_state.get(WORKFLOW_STATE)


def set_workflow_state(state: Any) -> None:
    st.session_state[WORKFLOW_STATE] = state
    if state is not None and getattr(state, "active_dataset", None) is not None:
        if getattr(state, "active_version_id", "v0") != "v0":
            st.session_state[CLEANED_DATASET] = state.active_dataset.copy(deep=True)
            from modules.cleaner import CleaningLog

            if st.session_state.get(CLEANING_LOG) is None:
                log = CleaningLog()
                for e in getattr(state, "audit_events", []):
                    if e.event_type == "Cleaning executed":
                        log.add(
                            column="multiple",
                            issue="governed cleaning",
                            method=e.actor_name,
                            reason=e.message,
                            rows_affected=0,
                            before_summary="v0",
                            after_summary=getattr(state, "active_version_id", "v1"),
                        )
                st.session_state[CLEANING_LOG] = log


def has_workflow_state() -> bool:
    return st.session_state.get(WORKFLOW_STATE) is not None


def clear_derived(keep_cleaning: bool = False) -> None:
    keys = [
        "cleaning_explanations",
        "ml_recommendations",
        "business_insights",
        "executive_summary",
        "feature_engineering",
        "chat_session",
        "cleaning_preview",
        "ai_output",
        "ai_consent",
        "applied_recipe",
    ]
    if not keep_cleaning:
        keys += [CLEANED_DATASET, CLEANING_LOG, WORKFLOW_STATE]
    for key in keys:
        st.session_state.pop(key, None)


def restore_original() -> None:
    orig = get_dataset()
    meta = st.session_state.get(DATASET_METADATA)
    fname = st.session_state.get(DATASET_FILENAME, "Dataset")
    clear_derived()
    st.session_state["dataset_version"] = st.session_state.get("dataset_version", 0) + 1
    if orig is not None:
        from workflow.state import WorkflowState

        st.session_state[WORKFLOW_STATE] = WorkflowState.create_initial(
            orig, fname, meta
        )


def _log_event(event: str, df: pd.DataFrame) -> None:
    from utils.config import load_settings
    from utils.logger import get_logger

    settings = load_settings()
    get_logger("workspace", settings.logs_dir, settings.log_level).info(
        "%s | rows=%d columns=%d", event, len(df), len(df.columns)
    )


__all__ = [
    "BUSINESS_INSIGHTS",
    "CHAT_SESSION",
    "CLEANED_DATASET",
    "CLEANING_EXPLANATIONS",
    "CLEANING_LOG",
    "DATASET",
    "DATASET_FILENAME",
    "DATASET_METADATA",
    "EXECUTIVE_SUMMARY",
    "FEATURE_ENGINEERING",
    "ML_RECOMMENDATIONS",
    "WORKFLOW_STATE",
    "clear_derived",
    "get_active_dataset",
    "get_business_insights",
    "get_chat_session",
    "get_cleaning_explanations",
    "get_cleaning_log",
    "get_dataset",
    "get_executive_summary",
    "get_feature_engineering",
    "get_ml_recommendations",
    "get_workflow_state",
    "has_business_insights",
    "has_chat_session",
    "has_cleaned_dataset",
    "has_cleaning_explanations",
    "has_dataset",
    "has_executive_summary",
    "has_feature_engineering",
    "has_ml_recommendations",
    "has_workflow_state",
    "init_session_state",
    "restore_original",
    "set_business_insights",
    "set_chat_session",
    "set_cleaned_dataset",
    "set_cleaning_explanations",
    "set_dataset",
    "set_executive_summary",
    "set_feature_engineering",
    "set_ml_recommendations",
    "set_workflow_state",
]
