"""Governed AI tools with privacy boundaries, prompt sanitization, and deterministic fallbacks."""

from __future__ import annotations

import json
from typing import Any
import pandas as pd

from modules.ai.client import GeminiClient
from modules.ml_readiness import (
    assess_ml_readiness as _assess_ml_readiness,
    MLReadinessResult,
)
from tools.registry import registry


def _sanitize_untrusted_text(text: str, max_chars: int = 200) -> str:
    """Sanitize strings coming from dataset column names or values to prevent prompt injection."""
    cleaned = text.replace("\r", " ").replace("\n", " ").replace("`", "'")
    return cleaned[:max_chars].strip()


def _build_bounded_summary_prompt(
    filename: str,
    n_rows: int,
    n_cols: int,
    quality_score: float,
    columns_summary: list[dict[str, Any]],
) -> str:
    """Build a prompt containing only statistical metadata, strictly zero raw dataset rows."""
    cols_repr = json.dumps(columns_summary[:30], indent=2)
    return f"""You are a data intelligence assistant.
Analyze this dataset metadata (DO NOT assume causation; label hypotheses as hypotheses).
DO NOT generate executable code.
Strictly adhere to the provided schema and statistics.

Dataset: {_sanitize_untrusted_text(filename)}
Rows: {n_rows}
Columns: {n_cols}
Data Quality Score: {quality_score:.1f}/100

Column Profiles:
{cols_repr}

Provide:
1. Executive Summary (2-3 sentences)
2. Primary Data Distribution Observations
3. 2-3 Actionable Business Hypotheses (explicitly labelled as hypotheses)
"""


def _deterministic_summary_fallback(
    filename: str,
    n_rows: int,
    n_cols: int,
    quality_score: float,
    missing_pct: float,
) -> str:
    """Deterministic fallback when Gemini is unavailable or not configured."""
    return f"""### Statistical Dataset Summary (Deterministic Analysis)

*Dataset:* **{filename}**
*Dimensions:* **{n_rows:,}** rows × **{n_cols}** columns
*Data Quality Score:* **{quality_score:.1f}/100**
*Missing Data:* Overall **{missing_pct:.1f}%** missing rate across dataset cells.

**Key Findings:**
- The dataset was evaluated locally without external AI transmission.
- Quality score reflects completeness, duplicate rate, data-type consistency, and IQR outliers.
- All subsequent cleaning and modeling should address identified missing values and feature scales.

*Note: AI reasoning was unavailable; deterministic heuristic summary applied.*"""


def _deterministic_insights_fallback(
    numerical_cols: list[str],
    categorical_cols: list[str],
    correlations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    insights = []
    if correlations:
        for c in correlations[:3]:
            insights.append(
                {
                    "hypothesis": f"Observed correlation ({c.get('correlation', 0):+.2f}) between {c.get('column_a')} and {c.get('column_b')}.",
                    "evidence": f"Pearson r = {c.get('correlation', 0):+.2f} ({c.get('strength', 'moderate')} association).",
                    "confidence": "Medium",
                    "disclaimer": "Correlation does not imply causation. Further experimental verification required.",
                    "is_ai_generated": False,
                }
            )
    else:
        insights.append(
            {
                "hypothesis": f"Dataset contains {len(numerical_cols)} numerical and {len(categorical_cols)} categorical attributes.",
                "evidence": f"Identified {len(numerical_cols) + len(categorical_cols)} total features.",
                "confidence": "High",
                "disclaimer": "Descriptive heuristic baseline. No external AI inference used.",
                "is_ai_generated": False,
            }
        )
    return insights


@registry.register(
    name="generate_ai_summary",
    description="Generates an executive summary using bounded statistical metadata (zero raw rows sent).",
    requires_approval=False,
    destructive=False,
)
def generate_ai_summary_tool(
    state: Any,
    api_key: str | None = None,
    model: str = "gemini-2.5-flash",
) -> str:
    df = state.active_dataset
    profile = state.profile_result
    quality = state.quality_result

    q_score = quality.overall_score if quality else 80.0
    missing_pct = (
        (quality.missing_cell_count / (len(df) * len(df.columns)) * 100)
        if (quality and len(df) and len(df.columns))
        else 0.0
    )

    if not api_key:
        return _deterministic_summary_fallback(
            state.dataset_filename, len(df), len(df.columns), q_score, missing_pct
        )

    try:
        client = GeminiClient(api_key=api_key, model=model)
        col_summaries = []
        if profile:
            for col in profile.columns[:30]:
                col_summaries.append(
                    {
                        "name": _sanitize_untrusted_text(col.name, 40),
                        "type": col.dtype,
                        "missing_pct": col.missing_pct,
                        "unique_count": col.unique_count,
                    }
                )

        prompt = _build_bounded_summary_prompt(
            filename=state.dataset_filename,
            n_rows=len(df),
            n_cols=len(df.columns),
            quality_score=q_score,
            columns_summary=col_summaries,
        )
        res = client.generate(prompt, max_retries=2)
        if res.success and res.text:
            return f"✦ **AI-Generated Summary** (Human review required before report publication):\n\n{res.text}"
        return _deterministic_summary_fallback(
            state.dataset_filename, len(df), len(df.columns), q_score, missing_pct
        )
    except Exception:
        # Never leak secrets or raw exceptions; fall back gracefully
        return _deterministic_summary_fallback(
            state.dataset_filename, len(df), len(df.columns), q_score, missing_pct
        )


@registry.register(
    name="generate_business_insights",
    description="Generates structured business hypotheses and insights from summary statistics.",
    requires_approval=False,
    destructive=False,
)
def generate_business_insights_tool(
    state: Any,
    api_key: str | None = None,
    model: str = "gemini-2.5-flash",
) -> list[dict[str, Any]]:
    eda = state.eda_result
    profile = state.profile_result

    num_cols = profile.numerical_columns if profile else []
    cat_cols = profile.categorical_columns if profile else []
    corr_dicts = (
        [
            {
                "column_a": p.column_a,
                "column_b": p.column_b,
                "correlation": p.correlation,
                "strength": p.strength,
            }
            for p in eda.correlation_pairs
        ]
        if eda
        else []
    )

    fallback = _deterministic_insights_fallback(num_cols, cat_cols, corr_dicts)

    if not api_key:
        return fallback

    try:
        client = GeminiClient(api_key=api_key, model=model)
        prompt = f"""You are a senior business intelligence analyst.
Analyze these statistical correlations (NO raw data rows provided):
{json.dumps(corr_dicts[:10], indent=2)}

Output ONLY valid JSON list of objects matching this schema:
[
  {{
    "hypothesis": "Hypothesis description...",
    "evidence": "Observed statistical evidence...",
    "confidence": "High" | "Medium" | "Low",
    "disclaimer": "Explicit note: correlation is not causation."
  }}
]
"""
        res = client.generate(prompt, max_retries=2)
        if res.success and res.text:
            cleaned_text = res.text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[3:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            data = json.loads(cleaned_text.strip())
            if isinstance(data, list) and len(data) > 0:
                for item in data:
                    item["is_ai_generated"] = True
                return data
        return fallback
    except Exception:
        return fallback


@registry.register(
    name="assess_ml_readiness",
    description="Evaluates dataset suitability for machine learning algorithms.",
    requires_approval=False,
    destructive=False,
)
def assess_ml_readiness_tool(
    df: pd.DataFrame,
    profile: Any,
    quality: Any,
    eda: Any,
    target_column: str | None = None,
) -> MLReadinessResult:
    return _assess_ml_readiness(
        df=df,
        profile=profile,
        quality=quality,
        eda=eda,
        target_column=target_column,
    )
