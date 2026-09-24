from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from modules.quality import count_outliers_iqr,  iqr_bounds

import pandas as pd


@dataclass(frozen=True)
class CleaningAction:
    """A single record of one cleaning operation performed on one column
    (or the whole dataset, for row-level operations like deduplication)."""

    timestamp: str
    column: str
    issue: str
    method: str
    reason: str
    rows_affected: int
    before_summary: str
    after_summary: str

@dataclass
class CleaningLog:
    """Collects CleaningAction entries as cleaning operations run."""

    actions: list[CleaningAction] = field(default_factory=list)

    def add(
        self,
        column: str,
        issue: str,
        method: str,
        reason: str,
        rows_affected: int,
        before_summary: str,
        after_summary: str,
    ) -> None:
        self.actions.append(
            CleaningAction(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                column=column,
                issue=issue,
                method=method,
                reason=reason,
                rows_affected=rows_affected,
                before_summary=before_summary,
                after_summary=after_summary,
            )
        )

    def to_dataframe(self) -> pd.DataFrame:
        if not self.actions:
            return pd.DataFrame(
                columns=[
                    "timestamp", "column", "issue", "method",
                    "reason", "rows_affected", "before_summary", "after_summary",
                ]
            )
        return pd.DataFrame([vars(a) for a in self.actions])


#handling missing values
def handle_missing_values(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        if missing_count == 0:
            continue

        before_summary = f"{missing_count} missing values"

        if pd.api.types.is_numeric_dtype(df[col]):
            has_outliers = count_outliers_iqr(df[col]) > 0
            if has_outliers:
                fill_value = df[col].median()
                method = "median imputation"
                reason = (
                    "Column contains outliers; median is more robust to "
                    "extreme values than mean, so it preserves the "
                    "underlying distribution better."
                )
            else:
                fill_value = df[col].mean()
                method = "mean imputation"
                reason = (
                    "Column has no significant outliers, so mean "
                    "imputation preserves the distribution accurately."
                )
            df[col] = df[col].fillna(fill_value)
            after_summary = f"filled with {method.split()[0]} = {round(fill_value, 2)}"

        else:
            mode_values = df[col].mode(dropna=True)
            fill_value = mode_values.iloc[0] if not mode_values.empty else "Unknown"
            method = "mode imputation"
            reason = (
                "Categorical column; the most frequent value is the "
                "statistically safest fill for text/category data."
            )
            df[col] = df[col].fillna(fill_value)
            after_summary = f"filled with mode = '{fill_value}'"

        log.add(
            column=col,
            issue="missing values",
            method=method,
            reason=reason,
            rows_affected=missing_count,
            before_summary=before_summary,
            after_summary=after_summary,
        )

    return df

#remove duplications
def remove_duplicates(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count == 0:
        return df

    before_rows = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after_rows = len(df)

    log.add(
        column="(all columns)",
        issue="duplicate rows",
        method="drop_duplicates (keep first occurrence)",
        reason=(
            "Exact duplicate rows add no new information and can bias "
            "statistics and model training by over-weighting repeated "
            "records."
        ),
        rows_affected=duplicate_count,
        before_summary=f"{before_rows} rows",
        after_summary=f"{after_rows} rows",
    )

    return df


#string cleaning: whitespace and case normalization
def clean_string_columns(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        series = df[col]
        if not pd.api.types.is_string_dtype(series):
            continue

        non_null = series.dropna()
        if non_null.empty or not non_null.map(lambda v: isinstance(v, str)).all():
            continue

        stripped = series.str.strip()
        whitespace_changed = int((stripped != series).fillna(False).sum())

        if whitespace_changed > 0:
            df[col] = stripped
            log.add(
                column=col,
                issue="inconsistent whitespace",
                method="strip leading/trailing whitespace",
                reason=(
                    "Extra whitespace causes identical-looking values "
                    "(e.g. 'India' vs 'India ') to be treated as different "
                    "categories, silently fragmenting what should be one group."
                ),
                rows_affected=whitespace_changed,
                before_summary=f"{whitespace_changed} values had extra whitespace",
                after_summary="whitespace trimmed",
            )

        current = df[col]
        current_non_null = current.dropna()
        if current_non_null.empty:
            continue

        unique_before = current_non_null.nunique()
        unique_after_lower = current_non_null.str.lower().nunique()

        if unique_after_lower < unique_before:
            rows_changed = int((current != current.str.lower()).fillna(False).sum())
            df[col] = current.str.lower()
            log.add(
                column=col,
                issue="inconsistent casing",
                method="lowercase normalization",
                reason=(
                    f"Lowercasing reduced distinct values from {unique_before} "
                    f"to {unique_after_lower}, meaning casing differences (e.g. "
                    f"'India' vs 'india') were fragmenting what should be one "
                    f"category into several."
                ),
                rows_affected=rows_changed,
                before_summary=f"{unique_before} distinct values",
                after_summary=f"{unique_after_lower} distinct values",
            )

    return df


#treaating outliers 
def treat_outliers(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    df = df.copy()

    for col in df.select_dtypes(include="number").columns:
        bounds = iqr_bounds(df[col])
        if bounds is None:
            continue

        lower_bound, upper_bound = bounds
        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_count = int(outlier_mask.sum())

        if outlier_count == 0:
            continue

        before_summary = f"{outlier_count} values outside [{round(lower_bound, 2)}, {round(upper_bound, 2)}]"
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        log.add(
            column=col,
            issue="outliers",
            method="IQR capping (winsorization)",
            reason=(
                f"Values were capped to the range "
                f"[{round(lower_bound, 2)}, {round(upper_bound, 2)}] rather than "
                f"removed, preserving row count while preventing extreme "
                f"values from distorting means, variances, and downstream "
                f"ML models."
            ),
            rows_affected=outlier_count,
            before_summary=before_summary,
            after_summary=f"values capped to bounds",
        )

    return df



#auto clean 
def auto_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningLog]:
    log = CleaningLog()

    df = remove_duplicates(df, log)
    df = clean_string_columns(df, log)
    df = handle_missing_values(df, log)
    df = treat_outliers(df, log)

    return df, log