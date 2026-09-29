"""Visualization recommendation tools with semantic aggregation safety."""

from __future__ import annotations

from typing import Any
import pandas as pd

from modules.eda import EDAResult
from modules.profiler import DatasetProfile
from tools.registry import registry

# Keywords indicating non-additive measures where summing is semantically invalid
NON_ADDITIVE_KEYWORDS = {
    "score",
    "rating",
    "pct",
    "percent",
    "percentage",
    "ratio",
    "rate",
    "rank",
    "temperature",
    "temp",
    "prob",
    "probability",
    "index",
    "id",
    "satisfaction",
    "nps",
    "age",
    "gpa",
}


def is_non_additive_measure(column_name: str) -> bool:
    """Return True if the column name implies an intensity, ratio, rating, or score
    that should NEVER be summed."""
    col_lower = column_name.lower().replace("_", " ").replace("-", " ")
    tokens = col_lower.split()
    return any(
        kw in tokens or any(kw in token for token in tokens)
        for kw in NON_ADDITIVE_KEYWORDS
    )


@registry.register(
    name="recommend_visualizations",
    description="Recommends semantically sound charts and aggregations based on profile and EDA results.",
    requires_approval=False,
    destructive=False,
)
def recommend_visualizations_tool(
    df: pd.DataFrame,
    profile: DatasetProfile,
    eda: EDAResult | None = None,
) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []

    # 1. Distribution recommendations for numeric columns
    for col in profile.numerical_columns[:4]:
        non_additive = is_non_additive_measure(col)
        recs_type = "histogram"
        reason = (
            f"Numeric column '{col}' is best understood via its frequency distribution. "
            + (
                "Note: This measure is non-additive (e.g., score/rate); avoid summing."
                if non_additive
                else ""
            )
        )
        recommendations.append(
            {
                "chart_type": recs_type,
                "title": f"Distribution of {col}",
                "x_column": col,
                "y_column": None,
                "group_by": None,
                "recommended_aggregation": "distribution",
                "is_non_additive": non_additive,
                "reason": reason,
            }
        )

    # 2. Categorical breakdown for low-to-medium cardinality
    for col in profile.categorical_columns:
        matching_col_profile = next((c for c in profile.columns if c.name == col), None)
        if matching_col_profile and 2 <= matching_col_profile.unique_count <= 15:
            recommendations.append(
                {
                    "chart_type": "bar",
                    "title": f"Count by {col}",
                    "x_column": col,
                    "y_column": None,
                    "group_by": None,
                    "recommended_aggregation": "count",
                    "is_non_additive": False,
                    "reason": f"Categorical column '{col}' has {matching_col_profile.unique_count} distinct categories, ideal for frequency bar comparison.",
                }
            )
            if len(recommendations) >= 6:
                break

    # 3. Numeric grouped by Categorical (Box plot / Bar)
    if profile.categorical_columns and profile.numerical_columns:
        cat_col = profile.categorical_columns[0]
        num_col = profile.numerical_columns[0]
        non_additive = is_non_additive_measure(num_col)
        agg = "mean" if non_additive else "sum"
        reason = f"Comparing {num_col} across {cat_col}. " + (
            f"Using '{agg}' because '{num_col}' is an intensity/rate/score and cannot be summed."
            if non_additive
            else f"Using '{agg}' aggregation."
        )
        recommendations.append(
            {
                "chart_type": "box",
                "title": f"{num_col} by {cat_col}",
                "x_column": cat_col,
                "y_column": num_col,
                "group_by": cat_col,
                "recommended_aggregation": agg,
                "is_non_additive": non_additive,
                "reason": reason,
            }
        )

    # 4. Correlation scatter plots if EDA found strong/moderate pairs
    if eda and eda.correlation_pairs:
        for pair in eda.correlation_pairs[:2]:
            recommendations.append(
                {
                    "chart_type": "scatter",
                    "title": f"{pair.column_b} vs {pair.column_a} (Corr: {pair.correlation:+.2f})",
                    "x_column": pair.column_a,
                    "y_column": pair.column_b,
                    "group_by": None,
                    "recommended_aggregation": "none",
                    "is_non_additive": is_non_additive_measure(pair.column_a)
                    or is_non_additive_measure(pair.column_b),
                    "reason": f"High correlation ({pair.correlation:+.2f}) detected between '{pair.column_a}' and '{pair.column_b}'. Inspect relationship directly.",
                }
            )

    return recommendations
