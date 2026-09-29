from __future__ import annotations

from modules.ai.client import GeminiClient
from modules.ai.prompt_templates import build_prompt, truncate_for_prompt
from modules.eda import EDAResult
from modules.ml_readiness import MLReadinessResult
from modules.profiler import DatasetProfile

_INSTRUCTION = (
    "Below is an ML readiness assessment for a dataset, including the "
    "inferred task type and any data quality issues found. Recommend "
    "2-3 specific, well-known algorithms suited to this task and this "
    "dataset's characteristics (size, dimensionality, issues found). "
    "For each: give the algorithm name, one sentence on why it fits "
    "this data, one real trade-off or limitation, and the evaluation "
    "metric(s) most appropriate given any issues noted (e.g. avoid "
    "accuracy alone if class imbalance was flagged). Format as a "
    "markdown bullet list, grouped by algorithm."
)


def _gather_context(
    profile: DatasetProfile, eda: EDAResult, readiness: MLReadinessResult
) -> str:
    lines = [
        f"Inferred task: {readiness.inferred_task}",
        f"Target column: {readiness.target_column or '(none specified)'}",
        f"Readiness score: {readiness.readiness_score}%",
        f"Dataset size: {profile.n_rows} rows, "
        f"{len(profile.numerical_columns) + len(profile.categorical_columns)} usable features.",
    ]

    if readiness.issues:
        lines.append("Readiness issues found:")
        for issue in readiness.issues:
            lines.append(
                f"  - [{issue.severity}] {issue.category}: {issue.description}"
            )
    else:
        lines.append("No readiness issues found.")

    return "\n".join(lines)


def build_ml_recommendation_prompt(
    profile: DatasetProfile, eda: EDAResult, readiness: MLReadinessResult
) -> str:
    context = _gather_context(profile, eda, readiness)
    return build_prompt(_INSTRUCTION, truncate_for_prompt(context))


_TASK_RECOMMENDATIONS = {
    "classification": [
        ("Logistic Regression", "a simple, interpretable baseline for classification"),
        (
            "Random Forest Classifier",
            "handles non-linear relationships and mixed feature types well",
        ),
    ],
    "regression": [
        ("Linear Regression", "a simple, interpretable baseline for regression"),
        (
            "Random Forest Regressor",
            "handles non-linear relationships without heavy feature engineering",
        ),
    ],
    "clustering (no target specified)": [
        (
            "K-Means",
            "a standard starting point for grouping similar rows without labels",
        ),
        (
            "DBSCAN",
            "useful if clusters are irregularly shaped or outliers should be isolated",
        ),
    ],
}


def _fallback_recommendations(readiness: MLReadinessResult) -> str:
    algorithms = _TASK_RECOMMENDATIONS.get(readiness.inferred_task, [])
    has_imbalance = any(i.category == "Class Imbalance" for i in readiness.issues)
    has_multicollinearity = any(
        i.category == "Multicollinearity" for i in readiness.issues
    )

    bullets = []
    for name, reason in algorithms:
        bullets.append(f"- **{name}** — {reason}.")

    if readiness.inferred_task == "classification":
        metric = (
            "precision, recall, and F1 (not accuracy alone, since class "
            "imbalance was detected)"
            if has_imbalance
            else "accuracy, with precision/recall as a secondary check"
        )
        bullets.append(f"- Suggested evaluation: {metric}.")
    elif readiness.inferred_task == "regression":
        bullets.append("- Suggested evaluation: RMSE and R².")

    if has_multicollinearity:
        bullets.append(
            "- Note: multicollinearity was detected — tree-based models "
            "(Random Forest) are more robust to this than linear models."
        )

    return "\n".join(bullets)


def generate_ml_recommendations(
    client: GeminiClient,
    profile: DatasetProfile,
    eda: EDAResult,
    readiness: MLReadinessResult,
) -> str:
    prompt = build_ml_recommendation_prompt(profile, eda, readiness)
    response = client.generate(prompt)

    if response.success:
        return response.text

    return _fallback_recommendations(readiness)
