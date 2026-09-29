"""Deterministic cleaning tools with governance and rollback."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import numpy as np
import pandas as pd

from modules.cleaner import CleaningOptions, auto_clean
from modules.quality import iqr_bounds
from tools.registry import registry
from workflow.approvals import ApprovalStatus
from workflow.errors import ApprovalRequiredError
from workflow.events import ActorType, EventType


@registry.register(
    name="preview_cleaning",
    description="Simulates cleaning transformations on candidate copy without altering active or original dataset.",
    requires_approval=False,
    destructive=False,
)
def preview_cleaning_tool(
    df: pd.DataFrame,
    options: CleaningOptions | None = None,
) -> tuple[pd.DataFrame, Any]:
    return auto_clean(df, options)


@registry.register(
    name="execute_cleaning",
    description="Executes an explicitly approved cleaning proposal against the dataset and creates a new version.",
    requires_approval=True,
    destructive=True,
)
def execute_cleaning_tool(
    state: Any,
    proposal_id: str,
    reviewer: str = "human_user",
) -> tuple[pd.DataFrame, str]:
    proposal = state.get_proposal(proposal_id)
    if not proposal:
        raise ValueError(f"Proposal '{proposal_id}' not found.")

    approval = state.get_approval_for_proposal(proposal_id)
    if not approval or approval.status not in (ApprovalStatus.APPROVED, ApprovalStatus.EDITED):
        status_name = approval.status if approval else "no approval record"
        raise ApprovalRequiredError(
            f"Cannot execute proposal '{proposal_id}': status is '{status_name}'. Must be approved."
        )

    # Use edited parameters if present, otherwise proposal parameters
    params = approval.edited_parameters if approval.status == ApprovalStatus.EDITED else proposal.parameters

    # Start with a candidate copy of the active dataset
    candidate = state.active_dataset.copy(deep=True)
    action_type = proposal.action_type
    affected_cols = proposal.affected_columns
    diff_summary = ""

    if action_type == "remove_duplicates":
        before_count = len(candidate)
        subset = params.get("subset", None)
        candidate = candidate.drop_duplicates(subset=subset).reset_index(drop=True)
        dropped = before_count - len(candidate)
        diff_summary = f"Removed {dropped} duplicate row(s). Dataset rows: {before_count} -> {len(candidate)}"

    elif action_type == "trim_text":
        total_trimmed = 0
        for col in affected_cols:
            if col in candidate.columns and candidate[col].dtype == "object":
                before = candidate[col]
                after = before.map(lambda x: x.strip() if isinstance(x, str) else x)
                changed = int((before.notna() & before.ne(after)).sum())
                total_trimmed += changed
                candidate[col] = after
        diff_summary = f"Trimmed whitespace in columns {affected_cols}. {total_trimmed} cell(s) modified."

    elif action_type == "normalize_case":
        case_type = params.get("case", "lower")
        total_normalized = 0
        for col in affected_cols:
            if col in candidate.columns and candidate[col].dtype == "object":
                before = candidate[col]
                if case_type == "upper":
                    after = before.map(lambda x: x.upper() if isinstance(x, str) else x)
                else:
                    after = before.map(lambda x: x.lower() if isinstance(x, str) else x)
                changed = int((before.notna() & before.ne(after)).sum())
                total_normalized += changed
                candidate[col] = after
        diff_summary = f"Normalized case ({case_type}) in columns {affected_cols}. {total_normalized} cell(s) modified."

    elif action_type == "fill_numeric_missing":
        strategy = params.get("strategy", "median")
        custom_val = params.get("custom_value", None)
        filled_count = 0
        for col in affected_cols:
            if col in candidate.columns:
                series = candidate[col]
                missing = int(series.isna().sum())
                if missing:
                    if custom_val is not None:
                        fill_val = custom_val
                    else:
                        finite = series.replace([np.inf, -np.inf], np.nan).dropna()
                        fill_val = finite.median() if strategy == "median" else finite.mean()
                    candidate[col] = series.astype("Float64").fillna(fill_val)
                    filled_count += missing
        diff_summary = f"Filled {filled_count} missing numeric cells in {affected_cols} using strategy '{strategy}'."

    elif action_type == "fill_categorical_missing":
        custom_val = params.get("custom_value", None)
        filled_count = 0
        for col in affected_cols:
            if col in candidate.columns:
                series = candidate[col]
                missing = int(series.isna().sum())
                if missing:
                    if custom_val is not None:
                        fill_val = custom_val
                    else:
                        modes = series.mode()
                        fill_val = modes.iloc[0] if not modes.empty else "Missing"
                    candidate[col] = series.fillna(fill_val)
                    filled_count += missing
        diff_summary = f"Filled {filled_count} missing categorical cells in {affected_cols}."

    elif action_type == "cap_outliers":
        total_capped = 0
        for col in affected_cols:
            if col in candidate.columns:
                series = candidate[col]
                bounds = iqr_bounds(series.replace([np.inf, -np.inf], np.nan))
                if bounds:
                    lo, hi = bounds
                    mask = (series < lo) | (series > hi)
                    count = int(mask.sum())
                    if count:
                        candidate[col] = series.astype("Float64").clip(lo, hi)
                        total_capped += count
        diff_summary = f"Capped {total_capped} outlier values in {affected_cols} using 1.5*IQR bounds."

    elif action_type == "convert_dates":
        for col in affected_cols:
            if col in candidate.columns:
                candidate[col] = pd.to_datetime(candidate[col], errors="coerce")
        diff_summary = f"Converted columns {affected_cols} to datetime."

    elif action_type == "drop_column":
        existing = [c for c in affected_cols if c in candidate.columns]
        candidate = candidate.drop(columns=existing)
        diff_summary = f"Dropped columns: {existing}."

    elif action_type == "protect_column":
        diff_summary = f"Marked columns {affected_cols} as protected from modifications."

    else:
        raise ValueError(f"Unknown action_type '{action_type}'.")

    # Mark proposal executed
    proposal.status = "executed"
    proposal.executed_at = datetime.now(timezone.utc).isoformat()

    # Commit candidate dataset as the new active version
    version = state.commit_version(
        new_df=candidate,
        created_by=f"CleaningExecutor ({reviewer})",
        change_summary=f"Executed proposal '{proposal.title}': {diff_summary}",
    )

    state.record_audit(
        actor_type=ActorType.AGENT,
        actor_name="CleaningExecutorAgent",
        event_type=EventType.CLEANING_EXECUTED,
        message=f"Executed approved proposal '{proposal.title}'. New active version {version.version_id}. {diff_summary}",
        proposal_id=proposal.id,
        approval_id=approval.id,
        metadata={"diff_summary": diff_summary, "version_id": version.version_id},
    )

    return candidate, diff_summary


@registry.register(
    name="rollback_cleaning",
    description="Rolls back the active dataset to its previous version.",
    requires_approval=True,
    destructive=True,
)
def rollback_cleaning_tool(state: Any, reviewer: str = "human_user") -> Any:
    return state.rollback(reviewer=reviewer)
