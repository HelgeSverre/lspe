"""Optional Parquet projections for immutable JSONL scientific artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a deterministic columnar projection when the analysis extra is installed."""

    if not rows:
        return
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("Parquet artifacts require `uv sync --extra analysis`.") from error
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="zstd")


def write_jsonl_parquet(source: Path, target: Path) -> None:
    """Project an existing JSONL artifact without altering its append-only source."""

    if not source.is_file():
        return
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    write_parquet(target, rows)
