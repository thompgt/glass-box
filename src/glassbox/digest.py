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
import os
import warnings
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
    "find_lockfile",
    "UNLOCKED",
    "UnlockedEnvironmentWarning",
]

LOCKFILE_NAME = "requirements.lock"
LOCKFILE_ENV = "GLASSBOX_LOCKFILE"

# Recorded when no lockfile can be found. A sentinel rather than a failure so a
# wheel install can still train — but see :func:`env_digest` for why it is loud.
UNLOCKED = "unlocked"


class UnlockedEnvironmentWarning(UserWarning):
    """No lockfile was found, so environment drift cannot be detected."""


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


def find_lockfile() -> Path | None:
    """Locate ``requirements.lock``, or ``None`` if this install has no lock.

    Searched in order of how deliberate the answer is: an explicit environment
    override, the repository root (an editable install or a checkout), then
    alongside the package itself (a copy shipped with a wheel).
    """
    override = os.environ.get(LOCKFILE_ENV)
    if override:
        path = Path(override).expanduser()
        return path if path.exists() else None

    here = Path(__file__).resolve()
    for candidate in (here.parents[2] / LOCKFILE_NAME, here.parent / LOCKFILE_NAME):
        if candidate.exists():
            return candidate
    return None


def env_digest(lock_path: Path | None = None) -> str:
    """Digest of the pinned environment.

    Reproduction refuses to run when this differs from what was recorded at
    training time: retraining under a different NumPy is a different experiment,
    and reporting a digest mismatch as "not reproducible" would be a lie.

    The absent-lockfile case is the one worth being noisy about. Returning a
    constant sentinel makes the guard *vacuous*: every model records
    ``"unlocked"``, reproduction compares that to itself, and the check passes
    unconditionally while dependency ranges are free to move underneath it. A
    guard that always passes is worse than no guard, because it is reported as
    having been checked. So the sentinel stays — a wheel with no lock must still
    be able to train — but it announces itself, and
    :func:`glassbox.train.reproduce.retrain_from_provenance` refuses to treat it
    as agreement under ``strict_env``.
    """
    if lock_path is None:
        lock_path = find_lockfile()
    if lock_path is None or not lock_path.exists():
        warnings.warn(
            f"no {LOCKFILE_NAME} found: recording env_digest={UNLOCKED!r}. The "
            f"environment-drift guard cannot detect anything in this state — every "
            f"model version will agree with every other regardless of what is "
            f"installed. Generate one with "
            f"`python -m pip freeze --exclude-editable > {LOCKFILE_NAME}`, or point "
            f"{LOCKFILE_ENV} at an existing lock.",
            UnlockedEnvironmentWarning,
            stacklevel=2,
        )
        return UNLOCKED
    return sha256_hex(lock_path.read_bytes())
