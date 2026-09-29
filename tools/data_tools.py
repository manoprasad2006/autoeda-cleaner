"""Data loading and validation tools."""

from __future__ import annotations

import io
import pandas as pd

from modules.loader import (
    load_dataset as _load_dataset,
    validate_file,
    get_metadata,
    DatasetMetadata,
)
from tools.registry import registry


@registry.register(
    name="validate_dataset",
    description="Validates dataset filename extension and file size limit.",
    requires_approval=False,
    destructive=False,
)
def validate_dataset_tool(
    filename: str,
    file_size_bytes: int,
    max_upload_mb: int = 100,
) -> bool:
    validate_file(filename, file_size_bytes, max_upload_mb)
    return True


@registry.register(
    name="load_dataset",
    description="Loads a dataset from file bytes into a pandas DataFrame.",
    requires_approval=False,
    destructive=False,
)
def load_dataset_tool(
    file_source: io.BytesIO | bytes,
    filename: str,
    file_size_bytes: int = 0,
    max_upload_mb: int = 100,
) -> tuple[pd.DataFrame, DatasetMetadata]:
    if isinstance(file_source, bytes):
        buffer = io.BytesIO(file_source)
        size = file_size_bytes or len(file_source)
    else:
        buffer = file_source
        size = file_size_bytes

    validate_file(filename, size, max_upload_mb)
    df = _load_dataset(buffer, filename)
    metadata = get_metadata(df, filename, size)
    return df, metadata
