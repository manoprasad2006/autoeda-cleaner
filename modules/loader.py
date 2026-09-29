from __future__ import annotations
import io
from dataclasses import dataclass
from typing import Callable
import pandas as pd


def _robust_csv_reader(f: io.BytesIO) -> pd.DataFrame:
    try:
        return pd.read_csv(f)
    except Exception:
        f.seek(0)
        import csv

        text = f.read().decode("utf-8", errors="replace")

        rows = []
        for row in csv.reader(io.StringIO(text)):
            if len(row) == 1 and "|" in row[0]:
                row = [x.strip() for x in row[0].split("|")]
            # Strip extra commas from the end if they are empty
            while row and row[-1].strip() == "":
                row.pop()
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        max_len = max(len(r) for r in rows)
        header = [x.strip() for x in rows[0]]
        # Fill in missing header names
        for i in range(len(header), max_len):
            header.append(f"Unnamed_{i}")

        data = []
        for r in rows[1:]:
            r_clean = [x.strip() if isinstance(x, str) else x for x in r]
            r_clean.extend([None] * (max_len - len(r_clean)))
            data.append(r_clean)

        recovered = pd.DataFrame(data, columns=header)
        recovered.attrs["import_warning"] = (
            "CSV recovery was used. Check row alignment and data types before analysis."
        )
        return recovered


_READERS: dict[str, Callable[[io.BytesIO], pd.DataFrame]] = {
    ".csv": _robust_csv_reader,
    ".xlsx": pd.read_excel,
    ".json": pd.read_json,
    ".parquet": pd.read_parquet,
}

SUPPORTED_EXTENSIONS = tuple(_READERS.keys())


class UnsupportedFileTypeError(ValueError):
    """Raised when the uploaded file's extension isn't one we support."""


class FileTooLargeError(ValueError):
    """Raised when the uploaded file exceeds the configured size limit."""


class EmptyDatasetError(ValueError):
    """Raised when the file parses successfully but contains zero rows."""


@dataclass(frozen=True)
class DatasetMetadata:
    filename: str
    file_size_mb: float
    n_rows: int
    n_columns: int
    memory_usage_mb: float
    column_names: list[str]
    dtype_counts: dict[str, int]


def _extension_of(filename: str) -> str:
    if "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[-1].lower()


def validate_file(filename: str, file_size_bytes: int, max_upload_mb: int) -> None:
    extension = _extension_of(filename)
    if extension not in _READERS:
        raise UnsupportedFileTypeError(
            f"'{extension or 'unknown'}' is not supported. "
            f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    size_mb = file_size_bytes / (1024 * 1024)
    if size_mb > max_upload_mb:
        raise FileTooLargeError(
            f"File is {size_mb:.1f}MB, which exceeds the {max_upload_mb}MB limit."
        )


def load_dataset(file_obj: io.BytesIO, filename: str) -> pd.DataFrame:
    extension = _extension_of(filename)
    validate_file(filename, 0, 50)
    file_obj.seek(0)
    reader = _READERS[extension]
    df = reader(file_obj)

    if df.shape[0] == 0:
        raise EmptyDatasetError("The file parsed successfully but contains zero rows.")
    if df.shape[1] > 500:
        raise ValueError("Datasets are limited to 500 columns.")
    if len(df) > 500_000:
        raise ValueError("Datasets are limited to 500,000 rows.")
    if df.memory_usage(deep=True).sum() > 256 * 1024 * 1024:
        raise ValueError("Parsed data exceeds the 256 MB in-memory dataset limit.")
    df.columns = [str(c) for c in df.columns]
    if not df.columns.is_unique:
        raise ValueError("Column names must be unique.")
    if any(
        df[c].dropna().map(lambda v: isinstance(v, (list, dict, set))).any()
        for c in df.select_dtypes(include="object")
    ):
        raise ValueError(
            "Nested values are not supported. Flatten the data before uploading."
        )
    return df


def get_metadata(
    df: pd.DataFrame, filename: str, file_size_bytes: int
) -> DatasetMetadata:
    dtype_counts: dict[str, int] = {}
    for dtype in df.dtypes:
        key = str(dtype)
        dtype_counts[key] = dtype_counts.get(key, 0) + 1

    return DatasetMetadata(
        filename=filename,
        file_size_mb=round(file_size_bytes / (1024 * 1024), 3),
        n_rows=df.shape[0],
        n_columns=df.shape[1],
        memory_usage_mb=round(df.memory_usage(deep=True).sum() / (1024 * 1024), 3),
        column_names=list(df.columns),
        dtype_counts=dtype_counts,
    )
