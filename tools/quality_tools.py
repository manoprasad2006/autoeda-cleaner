"""Data quality scoring tools."""
from __future__ import annotations

import pandas as pd
from modules.profiler import DatasetProfile
from modules.quality import assess_quality as _assess_quality, QualityScore
from tools.registry import registry


@registry.register(
    name="assess_quality",
    description="Computes completeness, uniqueness, consistency, and validity quality scores.",
    requires_approval=False,
    destructive=False,
)
def assess_quality_tool(df: pd.DataFrame, profile: DatasetProfile) -> QualityScore:
    return _assess_quality(df, profile)
