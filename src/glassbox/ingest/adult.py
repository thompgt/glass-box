"""Ingest UCI Adult Census Income into ``features.credit_applications``.

Adult was chosen over German Credit as the primary dataset. German Credit has
1,000 rows, which puts a demographic-parity difference computed on its holdout
inside its own sampling noise — and Phase 5 needs to *attribute* a fairness
regression to a cause, which is impossible when the regression is noise. Adult's
48,842 rows make the signal real. German Credit is retained as a second dataset
to demonstrate the pipeline is not dataset-specific.

Two synthesized columns need justification, since neither exists in the raw data:

``subject_id``
    Adult has no subject identifier. We derive one from the content of the raw
    row (see :func:`subject_id_for`), so that re-ingesting the same file
    reproduces byte-identical IDs — which is a precondition for reproducibility
    meaning anything at all.

``as_of_ts``
    Adult has no timestamps, but the explanation endpoint must report the
    training snapshot's date range, and Iceberg's ``day()`` partitioning needs a
    real temporal spread. Derived deterministically from ``subject_id``, so it is
    stable across ingests and independent of when the ingest ran.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import urllib.request
from collections import Counter
from pathlib import Path

import pyarrow as pa

from ..digest import row_digest
from ..schemas import CREDIT_APPLICATIONS
from ..writer import arrow_schema_for, records_to_arrow

# Mirrors, tried in order. The UCI site has moved once already; a local cache in
# data/raw/ means this only has to work the first time.
ADULT_URLS = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    "https://raw.githubusercontent.com/uci-ml-repo/data/main/adult/adult.data",
)

RAW_COLUMNS = (
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
)

INT_COLUMNS = frozenset(
    {"age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"}
)

# as_of_ts is spread deterministically across this window.
EPOCH = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)
WINDOW_DAYS = 730


def data_dir(root: Path) -> Path:
    return root / "data" / "raw"


def download_adult(root: Path) -> Path:
    """Fetch adult.data into ``data/raw/``, or return the cached copy."""
    dest = data_dir(root) / "adult.data"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)

    errors = []
    for url in ADULT_URLS:
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310
                payload = resp.read()
            if payload:
                dest.write_bytes(payload)
                return dest
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{url}: {exc}")

    raise RuntimeError(
        "Could not download adult.data from any mirror. Place the file manually "
        f"at {dest} and re-run.\n" + "\n".join(errors)
    )


def subject_id_for(raw_line: str, occurrence: int) -> str:
    """Content-derived, stable, collision-safe subject identifier.

    Adult contains exact duplicate rows, so hashing the line alone would assign
    one identifier to several people — which would silently corrupt both the
    erasure contamination report and training-set membership. The occurrence
    index disambiguates them, and because it is assigned in file order it is
    reproducible for a given input file.
    """
    payload = f"{raw_line}#{occurrence}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def as_of_ts_for(subject_id: str) -> dt.datetime:
    """Deterministic pseudo-timestamp derived from the subject id.

    Derived from the id rather than assigned at ingest time so that two ingests
    of the same file produce the same content digest — an ingest-time clock would
    make every re-ingest a different data version.
    """
    offset_days = int(subject_id[8:12], 16) % WINDOW_DAYS
    offset_secs = int(subject_id[12:16], 16) % 86_400
    return EPOCH + dt.timedelta(days=offset_days, seconds=offset_secs)


def split_for(subject_id: str) -> str:
    """Hash-based split assignment: stable, and independent of row order.

    A random split with a seed would still depend on the order rows were read in,
    which Iceberg does not guarantee. Hashing the subject means a subject lands in
    the same split forever, no matter how the data is scanned or re-ingested.
    """
    bucket = int(subject_id[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "eval"
    return "holdout"


def _clean(value: str) -> str | None:
    value = value.strip()
    if value in ("", "?"):
        return None
    return value


def parse_adult(path: Path, ingest_batch_id: str) -> pa.Table:
    """Parse adult.data into an Arrow table matching the Iceberg feature schema."""
    seen: Counter[str] = Counter()
    records: list[dict] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("|"):  # adult.test's header comment
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(RAW_COLUMNS):
            continue

        occurrence = seen[line]
        seen[line] += 1
        subject_id = subject_id_for(line, occurrence)

        row: dict = {"subject_id": subject_id}
        for col, raw in zip(RAW_COLUMNS, parts, strict=True):
            if col == "income":
                # adult.test suffixes labels with a period; normalize both files.
                row["label"] = 1 if raw.rstrip(".").strip() == ">50K" else 0
            elif col in INT_COLUMNS:
                cleaned = _clean(raw)
                row[col] = int(cleaned) if cleaned is not None else None
            else:
                row[col] = _clean(raw)

        row["as_of_ts"] = as_of_ts_for(subject_id)
        row["split"] = split_for(subject_id)
        row["ingest_batch_id"] = ingest_batch_id
        # Digest of the content only — excludes ingest metadata so that the same
        # source row ingested in two different batches digests identically.
        row["source_row_digest"] = row_digest(
            {k: v for k, v in row.items() if k not in ("ingest_batch_id", "source_row_digest")}
        )
        records.append(row)

    if not records:
        raise ValueError(f"no parseable rows in {path}")

    return _to_arrow(records)


def arrow_schema() -> pa.Schema:
    """Arrow schema matching ``features.credit_applications`` exactly.

    Derived from the Iceberg schema rather than hand-written. Phase 0 measured
    that PyIceberg rejects both ``timestamp[ns]`` (pandas' default) and
    ``large_string`` (Polars' default) on write, and a hand-written schema is one
    more thing that can silently drift from the table it targets.
    """
    return arrow_schema_for(CREDIT_APPLICATIONS)


def _to_arrow(records: list[dict]) -> pa.Table:
    return records_to_arrow(CREDIT_APPLICATIONS, records)
