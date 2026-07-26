"""Canonical serialization and content digests.

Two rules govern this module, and both exist because the audit trail's
credibility rests on digests meaning what they claim to mean:

1. **Never hash a pickle.** ``pickle``/``joblib`` output embeds memo ordering,
   protocol version, and fully-qualified module paths, so a NumPy patch bump
   changes the bytes without changing the model. A digest that flips for reasons
   unrelated to the thing being digested is worse than no digest. Model digests
   go through a per-class canonical dump (see :mod:`glassbox.train.digest_model`).

2. **Content digests are order-independent.** A data version's identity must not
   depend on the order Iceberg happened to hand back its data files, which is not
   stable. We digest each row independently and then digest the *sorted* set of
   row digests. Two scans of the same snapshot therefore agree regardless of file
   ordering, chunking, or parallelism.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pyarrow as pa

__all__ = [
    "canonical_json",
    "sha256_hex",
    "row_digest",
    "content_digest",
    "digest_arrow_table",
    "env_digest",
]


def _default(obj: Any) -> Any:
    """Deterministic fallback encoder for types json doesn't handle natively."""
    if isinstance(obj, (dt.datetime, dt.date, dt.time)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, bytes):
        return obj.hex()
    # numpy scalars expose .item(); avoid importing numpy just for isinstance
    item = getattr(obj, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"cannot canonically serialize {type(obj)!r}")


def canonical_json(obj: Any) -> str:
    """JSON with sorted keys, no insignificant whitespace, and ASCII escaping.

    ``sort_keys`` is what makes this canonical: two dicts that are equal produce
    identical bytes regardless of insertion order.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=_default,
    )


def sha256_hex(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def row_digest(row: dict[str, Any]) -> str:
    """Digest of a single logical row. Stable across column order."""
    return sha256_hex(canonical_json(row))


def content_digest(row_digests: list[str]) -> str:
    """Digest of a set of rows, independent of the order they were read in.

    Sorting before hashing is the whole point — see the module docstring. It also
    means a content digest is a genuine set identity: it cannot distinguish two
    scans that returned the same rows in different orders, which is exactly the
    equivalence we want for "is this the same data version?".
    """
    joined = "\n".join(sorted(row_digests))
    return sha256_hex(joined)


def digest_arrow_table(
    table: pa.Table,
    *,
    exclude: tuple[str, ...] = (),
) -> tuple[str, list[str]]:
    """Return ``(content_digest, per_row_digests)`` for an Arrow table.

    ``exclude`` drops columns that are metadata about the row rather than part of
    its content — notably ``source_row_digest`` itself, which would otherwise be
    self-referential.
    """
    keep = [name for name in table.column_names if name not in exclude]
    projected = table.select(keep).combine_chunks()

    # to_pylist() materializes the whole table. At portfolio scale (~50k rows)
    # that is a second or two and vastly simpler than buffer-level hashing, which
    # would have to account for chunk boundaries and offset buffers.
    digests = [row_digest(row) for row in projected.to_pylist()]
    return content_digest(digests), digests


def env_digest(lock_path: Path | None = None) -> str:
    """Digest of the pinned environment.

    Reproduction refuses to run when this differs from what was recorded at
    training time: retraining under a different NumPy is a different experiment,
    and reporting a digest mismatch as "not reproducible" would be a lie.
    """
    if lock_path is None:
        lock_path = Path(__file__).resolve().parents[2] / "requirements.lock"
    if not lock_path.exists():
        return "unlocked"
    return sha256_hex(lock_path.read_bytes())
