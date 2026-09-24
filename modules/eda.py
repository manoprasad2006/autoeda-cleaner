from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


#numerical column analysis
@dataclass(frozen=True)
class NumericColumnStats:
    name: str
    mean: float
    median: float
    std: float
    min: float
    max: float
    q1: float
    q3: float
    skewness: float


def analyze_numeric_columns(df: pd.DataFrame, numerical_columns: list[str]) -> list[NumericColumnStats]:
    stats = []
    for col in numerical_columns:
        series = df[col].dropna()
        if series.empty:
            continue

        stats.append(
            NumericColumnStats(
                name=col,
                mean=round(float(series.mean()), 3),
                median=round(float(series.median()), 3),
                std=round(float(series.std()), 3),
                min=round(float(series.min()), 3),
                max=round(float(series.max()), 3),
                q1=round(float(series.quantile(0.25)), 3),
                q3=round(float(series.quantile(0.75)), 3),
                skewness=round(float(series.skew()), 3),
            )
        )
    return stats


@dataclass(frozen=True)
class CorrelationPair:
    column_a: str
    column_b: str
    correlation: float
    strength: str


@dataclass(frozen=True)
class EDAResult:
    numeric_stats: list[NumericColumnStats]
    categorical_stats: list[CategoricalColumnStats]
    correlation_pairs: list[CorrelationPair]
    correlation_matrix: pd.DataFrame


def run_eda(df: pd.DataFrame, numerical_columns: list[str], categorical_columns: list[str]) -> EDAResult:
    numeric_stats = analyze_numeric_columns(df, numerical_columns)
    categorical_stats = analyze_categorical_columns(df, categorical_columns)

    correlation_pairs = []
    correlation_matrix = pd.DataFrame()

    if len(numerical_columns) >= 2:
        # Compute correlation matrix, dropping non-numeric data if any leaked
        correlation_matrix = df[numerical_columns].corr()
        for i in range(len(numerical_columns)):
            for j in range(i + 1, len(numerical_columns)):
                col_a = numerical_columns[i]
                col_b = numerical_columns[j]
                
                # Check if columns exist in the correlation matrix (in case corr() dropped them)
                if col_a in correlation_matrix.columns and col_b in correlation_matrix.columns:
                    corr_val = correlation_matrix.loc[col_a, col_b]
                    if pd.notna(corr_val) and abs(corr_val) >= 0.5:
                        strength = "Strong" if abs(corr_val) >= 0.7 else "Moderate"
                        correlation_pairs.append(
                            CorrelationPair(
                                column_a=col_a,
                                column_b=col_b,
                                correlation=round(float(corr_val), 3),
                                strength=strength
                            )
                        )
                        
    return EDAResult(
        numeric_stats=numeric_stats,
        categorical_stats=categorical_stats,
        correlation_pairs=correlation_pairs,
        correlation_matrix=correlation_matrix
    )


#categorical column analysis
@dataclass(frozen=True)
class CategoricalColumnStats:
    name: str
    unique_count: int
    top_value: str
    top_value_count: int
    top_value_pct: float
    value_counts: dict[str, int] = field(default_factory=dict)


def analyze_categorical_columns(
    df: pd.DataFrame, categorical_columns: list[str], top_n: int = 10
) -> list[CategoricalColumnStats]:
    stats = []
    for col in categorical_columns:
        series = df[col].dropna()
        if series.empty:
            continue

        counts = series.value_counts().head(top_n)
        total = len(series)
        top_value = str(counts.index[0])
        top_value_count = int(counts.iloc[0])

        stats.append(
            CategoricalColumnStats(
                name=col,
                unique_count=int(series.nunique()),
                top_value=top_value,
                top_value_count=top_value_count,
                top_value_pct=round(top_value_count / total * 100, 2),
                value_counts={str(k): int(v) for k, v in counts.items()},
            )
        )
    return stats