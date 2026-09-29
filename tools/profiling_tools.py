"""Profiling and exploratory data analysis tools."""

from __future__ import annotations

import pandas as pd
from modules.profiler import profile_dataset as _profile_dataset, DatasetProfile
from modules.eda import run_eda as _run_eda, EDAResult
from tools.registry import registry


@registry.register(
    name="profile_dataset",
    description="Profiles dataset columns, types, missingness, and cardinality without modifying data.",
    requires_approval=False,
    destructive=False,
)
def profile_dataset_tool(df: pd.DataFrame) -> DatasetProfile:
    return _profile_dataset(df)


@registry.register(
    name="run_eda",
    description="Performs exploratory data analysis computing statistics and correlations.",
    requires_approval=False,
    destructive=False,
)
def run_eda_tool(
    df: pd.DataFrame,
    numerical_columns: list[str],
    categorical_columns: list[str],
) -> EDAResult:
    return _run_eda(df, numerical_columns, categorical_columns)
