from __future__ import annotations

from modules.ai.client import GeminiClient
from modules.ai.prompt_templates import build_prompt, truncate_for_prompt
from modules.eda import EDAResult
from modules.quality import QualityScore

_INSTRUCTION = (
    "Below are statistical findings from a cleaned dataset: correlations "
    "between numeric columns, column distributions, and categorical "
    "breakdowns. Identify 3-5 business-relevant insights a stakeholder "
    "would care about — for example: notable relationships between "
    "metrics, columns with unusual concentration or spread, or dominant "
    "categories. For each insight, state the finding and one plausible "
    "business interpretation. Be clear that interpretations are "
    "hypotheses to investigate, not confirmed conclusions. Format as a "
    "markdown bullet list."
)


def _gather_context(quality: QualityScore, eda: EDAResult) -> str:
    lines = []

    if eda.correlation_pairs:
        lines.append("Correlations found:")
        for pair in eda.correlation_pairs:
            lines.append(
                f"  - {pair.column_a} & {pair.column_b}: "
                f"{pair.correlation} ({pair.strength})"
            )
    else:
        lines.append("No strong correlations were found between numeric columns.")

    if eda.numeric_stats:
        lines.append("Numeric column shapes:")
        for stat in eda.numeric_stats:
            skew_note = (
                "right-skewed (long tail of high values)" if stat.skewness > 1
                else "left-skewed (long tail of low values)" if stat.skewness < -1
                else "roughly symmetric"
            )
            lines.append(f"  - {stat.name}: {skew_note} (skewness={stat.skewness})")

    if eda.categorical_stats:
        lines.append("Dominant categories:")
        for stat in eda.categorical_stats:
            lines.append(
                f"  - {stat.name}: '{stat.top_value}' makes up "
                f"{stat.top_value_pct}% of values"
            )

    if quality.outlier_columns:
        lines.append("Columns with notable outliers:")
        for col, count in quality.outlier_columns.items():
            lines.append(f"  - {col}: {count} outlier values")

    return "\n".join(lines)


def build_business_insights_prompt(quality: QualityScore, eda: EDAResult) -> str:
    context = _gather_context(quality, eda)
    return build_prompt(_INSTRUCTION, truncate_for_prompt(context))

def _fallback_insights(quality: QualityScore, eda: EDAResult) -> str:
    """Algorithmically assembled insights used if the AI call fails.

    Unlike executive_summary.py's fallback, this one is honest about a
    real limitation: we can restate WHAT was found (a correlation
    exists, a category dominates), but genuine business interpretation
    -- WHY it might matter, what to investigate -- requires reasoning
    we can't produce without the AI. The fallback states facts plainly
    rather than inventing a business narrative it can't actually back."""
    bullets = []

    for pair in eda.correlation_pairs[:3]:
        bullets.append(
            f"- **{pair.column_a} and {pair.column_b}** show a "
            f"{pair.strength} relationship (correlation: {pair.correlation}). "
            f"Worth investigating whether one influences the other."
        )

    for stat in eda.categorical_stats[:3]:
        if stat.top_value_pct >= 50:
            bullets.append(
                f"- **{stat.name}** is dominated by '{stat.top_value}' "
                f"({stat.top_value_pct}% of records) — worth checking "
                f"whether this concentration is expected."
            )

    if quality.outlier_columns:
        outlier_summary = ", ".join(
            f"{col} ({count})" for col, count in list(quality.outlier_columns.items())[:3]
        )
        bullets.append(f"- Outliers were found in: {outlier_summary}.")

    if not bullets:
        return "No notable patterns were found in this dataset."

    return "\n".join(bullets)


def generate_business_insights(client: GeminiClient, quality: QualityScore, eda: EDAResult) -> str:
    prompt = build_business_insights_prompt(quality, eda)
    response = client.generate(prompt)

    if response.success:
        return response.text

    return _fallback_insights(quality, eda)