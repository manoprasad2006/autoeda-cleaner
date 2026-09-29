from __future__ import annotations

from modules.ai.client import GeminiClient
from modules.ai.prompt_templates import build_prompt, truncate_for_prompt
from modules.cleaner import CleaningLog
from modules.eda import EDAResult
from modules.profiler import DatasetProfile
from modules.quality import QualityScore

_INSTRUCTION = (
    "Below is a structured summary of a dataset that has been profiled, "
    "quality-scored, cleaned, and analyzed. Write a concise executive "
    "summary (4-6 sentences) covering: what this dataset contains, its "
    "overall data quality and what was fixed, the most important "
    "patterns found, and one clear recommendation for what to do next. "
    "Write for a business stakeholder who has not seen the raw data."
)


def _gather_context(
    profile: DatasetProfile,
    quality: QualityScore,
    log: CleaningLog,
    eda: EDAResult,
) -> str:
    lines = [
        f"Dataset: {profile.n_rows} rows, {profile.n_columns} columns "
        f"({profile.memory_usage_mb} MB in memory).",
        f"Column types: {len(profile.numerical_columns)} numeric, "
        f"{len(profile.categorical_columns)} categorical, "
        f"{len(profile.date_columns)} date.",
        f"Overall data quality score: {quality.overall_score}% "
        f"(completeness {quality.completeness_score}%, "
        f"duplicates {quality.duplicate_score}%, "
        f"consistency {quality.consistency_score}%, "
        f"validity {quality.validity_score}%).",
        f"Cleaning actions taken: {len(log.actions)}.",
    ]

    for action in log.actions:
        lines.append(f"  - {action.column}: {action.issue} -> {action.method}")

    if eda.correlation_pairs:
        lines.append("Notable correlations:")
        for pair in eda.correlation_pairs[:5]:
            lines.append(
                f"  - {pair.column_a} & {pair.column_b}: "
                f"{pair.correlation} ({pair.strength})"
            )

    if eda.numeric_stats:
        lines.append("Key numeric columns:")
        for stat in eda.numeric_stats[:5]:
            lines.append(
                f"  - {stat.name}: mean={stat.mean}, median={stat.median}, "
                f"skewness={stat.skewness}"
            )

    return "\n".join(lines)


def build_executive_summary_prompt(
    profile: DatasetProfile,
    quality: QualityScore,
    log: CleaningLog,
    eda: EDAResult,
) -> str:
    context = _gather_context(profile, quality, log, eda)
    return build_prompt(_INSTRUCTION, truncate_for_prompt(context))


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    """Return the correctly pluralized word for `count`."""
    if count == 1:
        return singular
    return plural if plural is not None else f"{singular}s"


def _fallback_summary(
    profile: DatasetProfile,
    quality: QualityScore,
    log: CleaningLog,
) -> str:
    """Algorithmically assembled summary used if the AI call fails."""
    quality_word = (
        "excellent"
        if quality.overall_score >= 90
        else "good"
        if quality.overall_score >= 75
        else "fair"
        if quality.overall_score >= 60
        else "poor"
    )

    sentences = [
        f"This dataset contains {profile.n_rows:,} "
        f"{_plural(profile.n_rows, 'row')} and {profile.n_columns} "
        f"{_plural(profile.n_columns, 'column')}.",
        f"Overall data quality was {quality_word} "
        f"({quality.overall_score}%) for the active dataset.",
    ]

    if log.actions:
        sentences.append(
            f"{len(log.actions)} automated cleaning "
            f"{_plural(len(log.actions), 'action')} "
            f"{_plural(len(log.actions), 'was', 'were')} applied, "
            f"addressing issues including "
            f"{', '.join(sorted({a.issue for a in log.actions}))}."
        )
    else:
        sentences.append("No cleaning was necessary; the dataset was already clean.")

    return " ".join(sentences)


def generate_executive_summary(
    client: GeminiClient,
    profile: DatasetProfile,
    quality: QualityScore,
    log: CleaningLog,
    eda: EDAResult,
) -> str:
    prompt = build_executive_summary_prompt(profile, quality, log, eda)
    response = client.generate(prompt)

    if response.success:
        return response.text

    return _fallback_summary(profile, quality, log)
