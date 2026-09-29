from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    dtype: str
    missing_count: int
    missing_pct: float
    unique_count: int
    is_constant: bool
    is_high_cardinality: bool


@dataclass(frozen=True)
class DatasetProfile:
    n_rows: int
    n_columns: int
    memory_usage_mb: float
    duplicate_rows: int
    duplicate_pct: float
    numerical_columns: list[str]
    categorical_columns: list[str]
    date_columns: list[str]
    constant_columns: list[str]
    high_cardinality_columns: list[str]
    columns: list[ColumnProfile] = field(default_factory=list)


def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    n_rows, n_columns = df.shape
    memory_usage_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 3)
    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = round((duplicate_rows / n_rows * 100) if n_rows else 0.0, 2)

    columns: list[ColumnProfile] = []
    numerical_columns: list[str] = []
    categorical_columns: list[str] = []
    date_columns: list[str] = []
    constant_columns: list[str] = []
    high_cardinality_columns: list[str] = []

    for col in df.columns:
        series = df[col]

        missing_count = int(series.isna().sum())
        missing_pct = round((missing_count / n_rows * 100) if n_rows else 0.0, 2)
        unique_count = int(series.nunique(dropna=True))
        is_constant = unique_count <= 1
        is_high_cardinality = unique_count > 50 and (
            unique_count / n_rows > 0.5 if n_rows else False
        )

        dtype_str = str(series.dtype)

        is_date = pd.api.types.is_datetime64_any_dtype(series)
        is_numeric = pd.api.types.is_numeric_dtype(
            series
        ) and not pd.api.types.is_bool_dtype(series)

        if is_date:
            date_columns.append(col)
        elif is_numeric:
            numerical_columns.append(col)
        else:
            categorical_columns.append(col)

        is_high_cardinality = (
            not is_date
            and not is_numeric
            and unique_count > 50
            and (unique_count / n_rows > 0.5 if n_rows else False)
        )

        if is_constant:
            constant_columns.append(col)
        if is_high_cardinality:
            high_cardinality_columns.append(col)

        columns.append(
            ColumnProfile(
                name=col,
                dtype=dtype_str,
                missing_count=missing_count,
                missing_pct=missing_pct,
                unique_count=unique_count,
                is_constant=is_constant,
                is_high_cardinality=is_high_cardinality,
            )
        )
    return DatasetProfile(
        n_rows=n_rows,
        n_columns=n_columns,
        memory_usage_mb=memory_usage_mb,
        duplicate_rows=duplicate_rows,
        duplicate_pct=duplicate_pct,
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        date_columns=date_columns,
        constant_columns=constant_columns,
        high_cardinality_columns=high_cardinality_columns,
        columns=columns,
    )
