from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from modules.eda import EDAResult
from modules.profiler import DatasetProfile
from modules.quality import QualityScore


@dataclass(frozen=True)
class ReadinessIssue:
    category: str
    severity: str  # "high" | "medium" | "low"
    description: str
    affected_columns: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MLReadinessResult:
    readiness_score: float
    issues: list[ReadinessIssue]
    target_column: str | None
    inferred_task: str  # "classification" | "regression" | "clustering (no target specified)"


_SEVERITY_PENALTY = {"high": 15, "medium": 8, "low": 3}


def _check_missing_values(profile: DatasetProfile) -> ReadinessIssue | None:
    cols_with_missing = [c.name for c in profile.columns if c.missing_count > 0]
    if not cols_with_missing:
        return None
    return ReadinessIssue(
        category="Missing Values",
        severity="high",
        description=(
            f"{len(cols_with_missing)} column(s) still contain missing values. "
            f"Most ML algorithms cannot train on missing data directly."
        ),
        affected_columns=cols_with_missing,
    )


def _check_encoding_needed(profile: DatasetProfile) -> ReadinessIssue | None:
    if not profile.categorical_columns:
        return None
    return ReadinessIssue(
        category="Encoding Needed",
        severity="medium",
        description=(
            f"{len(profile.categorical_columns)} categorical column(s) need "
            f"encoding (e.g. one-hot or label encoding) before most ML "
            f"algorithms can use them."
        ),
        affected_columns=profile.categorical_columns,
    )


def _check_scaling_needed(eda: EDAResult) -> ReadinessIssue | None:
    if len(eda.numeric_stats) < 2:
        return None
    ranges = [(s.name, s.max - s.min) for s in eda.numeric_stats if s.max > s.min]
    if len(ranges) < 2:
        return None

    max_range = max(r for _, r in ranges)
    min_range = min(r for _, r in ranges)
    if min_range == 0 or max_range / min_range < 100:
        return None

    return ReadinessIssue(
        category="Feature Scaling Needed",
        severity="medium",
        description=(
            f"Numeric columns have very different scales (largest range is "
            f"{round(max_range / min_range)}x the smallest). Distance-based "
            f"and gradient-based algorithms will be biased toward "
            f"large-scale features unless scaled."
        ),
        affected_columns=[name for name, _ in ranges],
    )

def _check_multicollinearity(eda: EDAResult) -> ReadinessIssue | None:
    """Near-duplicate numeric features (correlation >= 0.9) can destabilize
    linear models and waste model capacity on redundant information."""
    near_duplicates = [p for p in eda.correlation_pairs if abs(p.correlation) >= 0.9]
    if not near_duplicates:
        return None

    affected = sorted({col for p in near_duplicates for col in (p.column_a, p.column_b)})
    pair_descriptions = ", ".join(
        f"{p.column_a}/{p.column_b} ({p.correlation})" for p in near_duplicates
    )

    return ReadinessIssue(
        category="Multicollinearity",
        severity="medium",
        description=(
            f"{len(near_duplicates)} pair(s) of numeric columns are nearly "
            f"redundant with each other: {pair_descriptions}. Consider "
            f"dropping one from each pair for linear models."
        ),
        affected_columns=affected,
    )


def _check_outliers(quality: QualityScore) -> ReadinessIssue | None:
    if not quality.outlier_columns:
        return None
    return ReadinessIssue(
        category="Outliers",
        severity="low",
        description=(
            f"{len(quality.outlier_columns)} column(s) still contain outlier "
            f"values. Already addressed by the cleaning step if it was run, "
            f"but worth confirming outlier-sensitive models (e.g. linear "
            f"regression) aren't skewed by remaining extremes."
        ),
        affected_columns=list(quality.outlier_columns.keys()),
    )


def _check_unusable_columns(profile: DatasetProfile) -> ReadinessIssue | None:
    unusable = profile.constant_columns + profile.high_cardinality_columns
    if not unusable:
        return None
    return ReadinessIssue(
        category="Unusable Columns",
        severity="medium",
        description=(
            f"{len(unusable)} column(s) should likely be dropped or "
            f"specially encoded before training: constant columns carry no "
            f"signal, and high-cardinality columns will explode one-hot "
            f"encoding into too many features."
        ),
        affected_columns=unusable,
    )


def _infer_task_type(df: pd.DataFrame, target_column: str) -> str:
    """Best-effort guess at whether the target looks like a classification
    or regression problem, based purely on its data type and cardinality."""
    series = df[target_column].dropna()
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > 15:
        return "regression"
    return "classification"


def _check_class_imbalance(df: pd.DataFrame, target_column: str) -> ReadinessIssue | None:
    series = df[target_column].dropna()
    if series.nunique() < 2 or series.nunique() > 15:
        return None  # not a plausible classification target

    counts = series.value_counts()
    majority_pct = counts.iloc[0] / len(series) * 100

    if majority_pct < 75:
        return None

    return ReadinessIssue(
        category="Class Imbalance",
        severity="high" if majority_pct >= 90 else "medium",
        description=(
            f"The target column '{target_column}' is imbalanced: the "
            f"majority class ('{counts.index[0]}') makes up "
            f"{round(majority_pct, 1)}% of rows. Accuracy alone will be "
            f"misleading; consider precision/recall/F1 or resampling."
        ),
        affected_columns=[target_column],
    )


def _check_target_leakage(eda: EDAResult, target_column: str) -> ReadinessIssue | None:
    """A feature correlated almost perfectly (>= 0.98) with the target is
    suspicious -- it may BE the target in disguise (e.g. a column
    computed from the target itself), which would make a model look
    artificially perfect during testing and fail in the real world."""
    suspicious = [
        p for p in eda.correlation_pairs
        if target_column in (p.column_a, p.column_b) and abs(p.correlation) >= 0.98
    ]
    if not suspicious:
        return None

    other_cols = [
        p.column_b if p.column_a == target_column else p.column_a for p in suspicious
    ]
    return ReadinessIssue(
        category="Possible Target Leakage",
        severity="high",
        description=(
            f"{', '.join(other_cols)} correlate almost perfectly "
            f"(>= 0.98) with the target '{target_column}'. Verify these "
            f"aren't derived from the target itself, which would leak "
            f"the answer into training."
        ),
        affected_columns=other_cols,
    )

def assess_ml_readiness(
    df: pd.DataFrame,
    profile: DatasetProfile,
    quality: QualityScore,
    eda: EDAResult,
    target_column: str | None = None,
) -> MLReadinessResult:
    issues: list[ReadinessIssue] = []

    for check in (
        _check_missing_values(profile),
        _check_encoding_needed(profile),
        _check_scaling_needed(eda),
        _check_multicollinearity(eda),
        _check_outliers(quality),
        _check_unusable_columns(profile),
    ):
        if check is not None:
            issues.append(check)

    if target_column is not None and target_column in df.columns:
        inferred_task = _infer_task_type(df, target_column)
        if inferred_task == "classification":
            imbalance_issue = _check_class_imbalance(df, target_column)
            if imbalance_issue is not None:
                issues.append(imbalance_issue)

        leakage_issue = _check_target_leakage(eda, target_column)
        if leakage_issue is not None:
            issues.append(leakage_issue)
    else:
        inferred_task = "clustering (no target specified)"

    readiness_score = 100.0
    for issue in issues:
        readiness_score -= _SEVERITY_PENALTY[issue.severity]
    readiness_score = max(0.0, min(100.0, readiness_score))

    return MLReadinessResult(
        readiness_score=round(readiness_score, 2),
        issues=issues,
        target_column=target_column,
        inferred_task=inferred_task,
    )