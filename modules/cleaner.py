from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from modules.quality import iqr_bounds


@dataclass(frozen=True)
class CleaningAction:
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
    actions: list[CleaningAction] = field(default_factory=list)

    def add(
        self,
        column,
        issue,
        method,
        reason,
        rows_affected,
        before_summary,
        after_summary,
    ):
        self.actions.append(
            CleaningAction(
                datetime.now(timezone.utc).isoformat(),
                column,
                issue,
                method,
                reason,
                int(rows_affected),
                before_summary,
                after_summary,
            )
        )

    def to_dataframe(self):
        return pd.DataFrame(
            [vars(a) for a in self.actions],
            columns=list(CleaningAction.__dataclass_fields__),
        )


@dataclass(frozen=True)
class CleaningOptions:
    deduplicate: bool = True
    trim_text: bool = True
    normalize_case: bool = False
    numeric_fill: str = "median"
    categorical_fill: bool = False
    cap_outliers: bool = False
    max_missing_fraction: float = 0.4
    protected_columns: tuple[str, ...] = ()


def auto_clean(df: pd.DataFrame, options: CleaningOptions | None = None):
    """Build a candidate copy. Never mutate the input or guess absent values.

    Protected columns are excluded from cell edits. Deduplication remains a
    row operation. All-null and heavily missing columns remain unchanged.
    """
    options = options or CleaningOptions()
    if options.numeric_fill not in ("leave", "mean", "median"):
        raise ValueError("Unknown numeric fill strategy")
    if not 0 <= options.max_missing_fraction <= 1:
        raise ValueError("Missing threshold must be between 0 and 1")
    result = df.copy(deep=True)
    log = CleaningLog()
    for col in result.columns:
        if col in options.protected_columns:
            continue
        series = result[col]
        nonnull = series.dropna()
        if nonnull.empty:
            continue
        if nonnull.map(lambda x: isinstance(x, str)).all():
            for enabled, transform, issue, method in [
                (
                    options.trim_text,
                    lambda s: s.str.strip(),
                    "whitespace",
                    "trim whitespace",
                ),
                (
                    options.normalize_case,
                    lambda s: s.str.lower(),
                    "casing",
                    "lowercase",
                ),
            ]:
                if enabled:
                    before = result[col]
                    after = transform(before)
                    count = (before.notna() & before.ne(after)).fillna(False).sum()
                    if count:
                        result[col] = after
                        log.add(
                            col,
                            issue,
                            method,
                            "Text normalization selected in the cleaning recipe.",
                            count,
                            "Original text",
                            "Normalized text",
                        )
        series = result[col]
        missing = int(series.isna().sum())
        numeric = pd.api.types.is_numeric_dtype(
            series
        ) and not pd.api.types.is_bool_dtype(series)
        if missing and missing / max(len(result), 1) <= options.max_missing_fraction:
            fill = None
            if numeric and options.numeric_fill != "leave":
                finite = series.replace([np.inf, -np.inf], np.nan).dropna()
                if len(finite):
                    fill = (
                        finite.median()
                        if options.numeric_fill == "median"
                        else finite.mean()
                    )
                    result[col] = series.astype("Float64").fillna(fill)
            elif options.categorical_fill and not pd.api.types.is_datetime64_any_dtype(
                series
            ):
                modes = series.mode()
                if not modes.empty:
                    fill = modes.iloc[0]
                    result[col] = series.fillna(fill)
            if fill is not None:
                log.add(
                    col,
                    "missing values",
                    f"{options.numeric_fill if numeric else 'mode'} imputation",
                    "Selected fill strategy; imputation can change distributions.",
                    missing,
                    f"{missing} missing",
                    f"Filled with {fill}",
                )
        if options.cap_outliers and numeric and series.nunique() > 15:
            bounds = iqr_bounds(result[col].replace([np.inf, -np.inf], np.nan))
            if bounds:
                lo, hi = bounds
                mask = (result[col] < lo) | (result[col] > hi)
                count = int(mask.sum())
                if count:
                    result[col] = result[col].astype("Float64").clip(lo, hi)
                    log.add(
                        col,
                        "outliers",
                        "IQR capping",
                        "Opt-in capping of continuous values; valid extremes may be affected.",
                        count,
                        f"{count} outside bounds",
                        f"Capped to [{lo:.3g}, {hi:.3g}]",
                    )
    if options.deduplicate:
        count = int(result.duplicated().sum())
        if count:
            before = len(result)
            result = result.drop_duplicates()
            log.add(
                "(all columns)",
                "duplicate rows",
                "keep first occurrence",
                "Remove exact duplicates after normalization.",
                count,
                f"{before} rows",
                f"{len(result)} rows",
            )
    return result.reset_index(drop=True), log
