import io
import zipfile
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from modules.cleaner import auto_clean, CleaningOptions
from modules.loader import (
    load_dataset,
    validate_file,
    FileTooLargeError,
    UnsupportedFileTypeError,
    EmptyDatasetError,
)
from modules.profiler import profile_dataset
from modules.quality import assess_quality
from modules.eda import run_eda
from modules.ml_readiness import assess_ml_readiness
from modules.exports import csv_bytes, report_html, export_bundle


def test_quality_dimensions():
    df = pd.DataFrame(
        {
            "n": [1, 2, 3, 4, 5, 6, 7, 100, None],
            "mixed": pd.Series(
                ["a", 1, "b", "c", "d", "e", "f", "g", "h"], dtype=object
            ),
        }
    )
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    score = assess_quality(df, profile_dataset(df))
    assert score.duplicate_row_count == 1
    assert score.missing_cell_count == 1
    assert score.inconsistent_columns == ["mixed"]
    assert score.outlier_columns["n"] == 1
    assert 0 <= score.overall_score < 100


def test_clean_quality_and_boolean_columns():
    df = pd.DataFrame({"n": range(10), "flag": [True, False] * 5})
    assert assess_quality(df, profile_dataset(df)).overall_score == 100


def test_empty_numeric_and_nullable_columns():
    df = pd.DataFrame(
        {
            "empty": [np.nan] * 4,
            "number": pd.Series([1, None, 2, 3], dtype="Int64"),
            "category": pd.Series(["x", None, "x", "y"], dtype="category"),
        }
    )
    clean, log = auto_clean(df, CleaningOptions(categorical_fill=True))
    assert clean["empty"].isna().all()
    assert clean["number"].isna().sum() == 0
    assert clean["category"].isna().sum() == 0
    assert all(a.column != "empty" for a in log.actions)


def test_cleaning_is_reversible_and_conservative():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "text": [" A ", "a", None, "b"],
            "n": [1, 2, None, 999],
            "sparse": [None, None, None, "x"],
        }
    )
    original = df.copy(deep=True)
    clean, log = auto_clean(df, CleaningOptions(protected_columns=("id",)))
    assert_frame_equal(df, original)
    assert clean["n"].iloc[-1] == 999
    assert clean["text"].iloc[0] == "A"
    assert clean["text"].isna().sum() == 1
    assert clean["sparse"].isna().sum() == 3
    assert [a.rows_affected for a in log.actions if a.issue == "whitespace"] == [1]


def test_deduplication_happens_after_normalization():
    df = pd.DataFrame({"x": [" A ", "A"]})
    clean, log = auto_clean(df)
    assert len(clean) == 1
    assert log.actions[-1].issue == "duplicate rows"


def test_protected_target_is_not_imputed():
    df = pd.DataFrame({"target": [1, None, 0], "x": [1, 2, None]})
    clean, _ = auto_clean(df, CleaningOptions(protected_columns=("target",)))
    assert clean.target.isna().sum() == 1
    assert clean.x.isna().sum() == 0


def test_upload_contracts():
    with pytest.raises(FileTooLargeError):
        validate_file("x.csv", 51 * 1024 * 1024, 50)
    with pytest.raises(UnsupportedFileTypeError):
        load_dataset(io.BytesIO(b"x"), "x.exe")
    with pytest.raises(EmptyDatasetError):
        load_dataset(io.BytesIO(b"a,b\n"), "x.csv")
    with pytest.raises(ValueError, match="Nested"):
        load_dataset(io.BytesIO(b'[{"x":{"a":1}}]'), "x.json")
    df = load_dataset(io.BytesIO(b"a,b\n1,2\n"), "x.csv")
    assert df.shape == (1, 2)


def test_missing_target_is_reported():
    df = pd.DataFrame({"target": [1, None, 0, 1], "x": [1, 2, 3, 4]})
    p = profile_dataset(df)
    readiness = assess_ml_readiness(
        df,
        p,
        assess_quality(df, p),
        run_eda(df, p.numerical_columns, p.categorical_columns),
        "target",
    )
    assert any(i.category == "Missing target labels" for i in readiness.issues)


def test_export_escapes_html_and_spreadsheet_formulas():
    df = pd.DataFrame({"<script>": ["=1+1", "normal"], "n": [-1, 2]})
    assert "'=1+1" in csv_bytes(df).decode("utf-8-sig")
    report = report_html(df, "<script>alert(1)</script>")
    assert "<script>" not in report
    assert "&lt;script&gt;" in report
    with zipfile.ZipFile(io.BytesIO(export_bundle(df, "test"))) as archive:
        assert {"dataset.csv", "report.html", "quality.json", "recipe.json"} <= set(
            archive.namelist()
        )


def test_all_null_and_boolean_dataset_analysis():
    df = pd.DataFrame({"empty": [np.nan] * 4, "flag": [True, False, True, False]})
    p = profile_dataset(df)
    eda = run_eda(df, p.numerical_columns, p.categorical_columns)
    assert p.categorical_columns == ["flag"]
    assert not eda.numeric_stats
    assert eda.categorical_stats[0].unique_count == 2


def test_opt_in_outlier_capping_preserves_protected_column():
    df = pd.DataFrame(
        {"x": list(range(20)) + [10000], "target": list(range(20)) + [10000]}
    )
    clean, log = auto_clean(
        df, CleaningOptions(cap_outliers=True, protected_columns=("target",))
    )
    assert clean.x.max() < 10000
    assert clean.target.max() == 10000
    assert any(a.issue == "outliers" for a in log.actions)


def test_empty_dataset_quality_is_not_perfect():
    df = pd.DataFrame({"x": []})
    assert assess_quality(df, profile_dataset(df)).overall_score == 0
