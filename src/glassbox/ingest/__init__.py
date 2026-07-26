"""Dataset ingestion into ``features.*``.

Ingest is deliberately dumb: parse, derive stable identifiers, append. It does
*not* split, sample, or transform features. Anything clever that happened here
would be invisible to the audit trail — the snapshot would record the result
without recording the reasoning — so all of it belongs downstream where it can be
pinned to a model version.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc

from ..catalog import glassbox_root, load_catalog
from ..schemas import CREDIT_APPLICATIONS, EVAL_HOLDOUT
from ..snapshots import DataSnapshot, capture_snapshot
from ..writer import append_arrow
from . import adult

__all__ = ["adult", "ingest_adult", "IngestResult"]


@dataclass
class IngestResult:
    dataset: str
    ingest_batch_id: str
    rows_written: int
    train_snapshot: DataSnapshot
    eval_snapshot: DataSnapshot
    split_counts: dict[str, int]


def ingest_adult(root: Path | None = None) -> IngestResult:
    """Load Adult into the feature table and capture the train/eval data versions.

    The eval holdout is copied into its own table and frozen there, rather than
    left as a filter over the live table. Phase 5 compares fairness metrics across
    model versions, and if the eval set could drift between comparisons the
    decomposition would be partly measuring the eval set rather than the model.
    """
    root = root or glassbox_root()
    catalog = load_catalog(root)

    batch_id = str(uuid.uuid4())
    path = adult.download_adult(root)
    table = adult.parse_adult(path, ingest_batch_id=batch_id)

    written = append_arrow(catalog, CREDIT_APPLICATIONS, table)

    split_col = table["split"].to_pylist()
    split_counts = {s: split_col.count(s) for s in ("train", "eval", "holdout")}

    # Freeze the eval holdout once. Re-running ingest must not append a second
    # copy, or the eval snapshot silently changes identity under every model that
    # was already compared against it.
    eval_table = catalog.load_table(EVAL_HOLDOUT.identifier)
    if eval_table.current_snapshot() is None:
        mask = pc.equal(table["split"], pa.scalar("eval"))
        append_arrow(catalog, EVAL_HOLDOUT, table.filter(mask))

    train_snap = capture_snapshot(catalog, CREDIT_APPLICATIONS, split="train")
    eval_snap = capture_snapshot(catalog, EVAL_HOLDOUT)

    return IngestResult(
        dataset="adult",
        ingest_batch_id=batch_id,
        rows_written=written,
        train_snapshot=train_snap,
        eval_snapshot=eval_snap,
        split_counts=split_counts,
    )
