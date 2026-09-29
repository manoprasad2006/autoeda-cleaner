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
    inferred_task: (
        str  # "classification" | "regression" | "clustering (no target specified)"
    )


_SEVERITY_PENALTY = {"high": 15, "medium": 8, "low": 3}


def _check_missing_values(
    profile: DatasetProfile, exclude: str | None = None
) -> ReadinessIssue | None:
    cols_with_missing = [
        c.name for c in profile.columns if c.missing_count > 0 and c.name != exclude
    ]
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


def _check_encoding_needed(
    profile: DatasetProfile, exclude: str | None = None
) -> ReadinessIssue | None:
    categorical_features = [c for c in profile.categorical_columns if c != exclude]
    if not categorical_features:
        return None
    return ReadinessIssue(
        category="Encoding Needed",
        severity="medium",
        description=(
            f"{len(categorical_features)} categorical column(s) need "
            f"encoding (e.g. one-hot or label encoding) before most ML "
            f"algorithms can use them."
        ),
        affected_columns=categorical_features,
    )


def _check_scaling_needed(
    eda: EDAResult, exclude: str | None = None
) -> ReadinessIssue | None:
    numeric_features = [s for s in eda.numeric_stats if s.name != exclude]
    if len(numeric_features) < 2:
        return None
    ranges = [(s.name, s.max - s.min) for s in numeric_features if s.max > s.min]
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


def _check_multicollinearity(
    eda: EDAResult, exclude: str | None = None
) -> ReadinessIssue | None:
    near_duplicates = [
        p
        for p in eda.correlation_pairs
        if abs(p.correlation) >= 0.9 and exclude not in (p.column_a, p.column_b)
    ]
    if not near_duplicates:
        return None

    affected = sorted(
        {col for p in near_duplicates for col in (p.column_a, p.column_b)}
    )
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


def _check_unusable_columns(
    profile: DatasetProfile, exclude: str | None = None
) -> ReadinessIssue | None:
    unusable = [
        c
        for c in (profile.constant_columns + profile.high_cardinality_columns)
        if c != exclude
    ]
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
    series = df[target_column].dropna()
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > 15:
        return "regression"
    return "classification"


def _check_class_imbalance(
    df: pd.DataFrame, target_column: str
) -> ReadinessIssue | None:
    series = df[target_column].dropna()
    if series.nunique() < 2 or series.nunique() > 15:
        return None

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


def _check_target_leakage(
    df: pd.DataFrame, eda: EDAResult, target_column: str
) -> ReadinessIssue | None:
    target_series = df[target_column]

    if pd.api.types.is_numeric_dtype(target_series):
        suspicious_pairs = [
            p
            for p in eda.correlation_pairs
            if target_column in (p.column_a, p.column_b) and abs(p.correlation) >= 0.98
        ]
        other_cols = [
            p.column_b if p.column_a == target_column else p.column_a
            for p in suspicious_pairs
        ]
    else:
        unique_values = target_series.dropna().unique()
        if len(unique_values) != 2:
            return None  # not binary -- see docstring on why we skip rather than guess

        encoded_target = target_series.map({unique_values[0]: 0, unique_values[1]: 1})
        other_cols = []
        for stat in eda.numeric_stats:
            if stat.name == target_column:
                continue
            corr = df[stat.name].corr(encoded_target)
            if pd.notna(corr) and abs(corr) >= 0.98:
                other_cols.append(stat.name)

    if not other_cols:
        return None

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
    if target_column is not None and target_column not in df.columns:
        raise ValueError("Target column is not in the dataset")
    if target_column is not None:
        target = df[target_column]
        if target.isna().any():
            issues.append(
                ReadinessIssue(
                    "Missing target labels",
                    "high",
                    "Target values are missing. Exclude unlabeled rows or obtain labels before supervised training.",
                    [target_column],
                )
            )
        if target.nunique() < 2:
            issues.append(
                ReadinessIssue(
                    "Insufficient target variation",
                    "high",
                    "The target needs at least two distinct observed values.",
                    [target_column],
                )
            )

    for check in (
        _check_missing_values(profile, exclude=target_column),
        _check_encoding_needed(profile, exclude=target_column),
        _check_scaling_needed(eda, exclude=target_column),
        _check_multicollinearity(eda, exclude=target_column),
        _check_outliers(quality),
        _check_unusable_columns(profile, exclude=target_column),
    ):
        if check is not None:
            issues.append(check)

    if target_column is not None and target_column in df.columns:
        inferred_task = _infer_task_type(df, target_column)
        if inferred_task == "classification":
            imbalance_issue = _check_class_imbalance(df, target_column)
            if imbalance_issue is not None:
                issues.append(imbalance_issue)

        leakage_issue = _check_target_leakage(df, eda, target_column)
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
