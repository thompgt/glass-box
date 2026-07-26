"""The durable write-ahead spool between a served decision and its audit row.

An Iceberg commit per request is not an option. Every commit writes a new
metadata file and a new snapshot, so a table taking one commit per prediction
accumulates metadata faster than data, and the commit itself costs
tens-to-hundreds of milliseconds — an order of magnitude more than scoring and
explaining the row. Batching is therefore forced.

But batching in memory would mean a crash loses decisions that were already
*served*. For an audit trail that is the one unacceptable failure: an
unrecorded decision is indistinguishable from a decision that never happened,
and the subject it was served to knows otherwise. So the spool is a write-ahead
log on local disk, fsynced before the response is returned, and drained into
Iceberg in batches.

Ordering inside a flush mirrors :mod:`glassbox.train.registry`: **attributions
first, predictions last.** The prediction row is the commit point. Every read
path starts from ``audit.predictions``, so attributions with no prediction row
are invisible garbage — harmless. The reverse ordering would publish a
prediction whose attributions do not exist, which is an integrity failure
indistinguishable from tampering.

Replay is idempotent in both directions:

* A segment that was fully committed but not deleted (crash after the Iceberg
  commit, before the unlink) is skipped envelope-by-envelope, because its
  prediction ids are already in ``audit.predictions``.
* A segment that crashed *between* the two commits leaves orphan attributions.
  Those are deleted before the retry, so replay cannot double-write them. This
  only runs for segments recovered from a previous process — a segment this
  process just claimed cannot have orphans, and the delete is expensive enough
  (copy-on-write file rewrites) to be worth not paying on the hot path.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..schemas import ATTRIBUTIONS, PREDICTIONS
from ..writer import append_records

SPOOL_DIRNAME = "spool"
PENDING_SUFFIX = ".jsonl"
# A segment being drained is renamed rather than read in place, so that a
# concurrent writer appending to the live segment cannot have its rows consumed
# and then unlinked underneath it.
CLAIMED_SUFFIX = ".flushing"

# Timestamp columns, which JSON cannot represent and Iceberg requires as real
# timestamptz values. Listed explicitly rather than sniffed, because a
# type-sniffing round trip would silently convert any string that happens to
# parse as a date.
_PREDICTION_TS_FIELDS = ("prediction_ts",)
_ATTRIBUTION_TS_FIELDS = ("prediction_ts",)


@dataclass
class SpoolEnvelope:
    """One served decision and the attributions that explain it, kept together.

    They are written and committed as a unit because a prediction whose
    attributions were lost is exactly the artifact this system exists to make
    impossible.
    """

    prediction: dict[str, Any]
    attributions: list[dict[str, Any]]

    @property
    def prediction_id(self) -> str:
        return self.prediction["prediction_id"]


@dataclass
class FlushResult:
    predictions_written: int = 0
    attributions_written: int = 0
    envelopes_skipped: int = 0
    segments_drained: int = 0
    orphans_cleaned: int = 0

    def __bool__(self) -> bool:
        return bool(self.predictions_written or self.envelopes_skipped)


@dataclass
class Spool:
    """A write-ahead log of served decisions awaiting their Iceberg commit."""

    root: Path
    batch_size: int = 200
    _segment: Path | None = field(default=None, init=False, repr=False)
    _appended: int = field(default=0, init=False, repr=False)
    # Serializes appends and segment rotation. Two requests interleaving their
    # writes would tear both lines, and a rotation racing an append would drop
    # one into a segment that is already being drained.
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @property
    def directory(self) -> Path:
        return Path(self.root) / SPOOL_DIRNAME

    # ------------------------------------------------------------- writing ----

    def append(self, envelope: SpoolEnvelope) -> Path:
        """Durably record a decision. Returns the segment it landed in.

        Returns only after ``fsync``. The response to the caller must not be sent
        before this returns, or the system can serve a decision it has no record
        of — which is the failure mode the whole spool exists to prevent.
        """
        line = json.dumps(_encode(envelope), separators=(",", ":"), sort_keys=True)

        with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            segment = self._current_segment()

            with open(segment, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())

            self._appended += 1
            if self._appended >= self.batch_size:
                self._segment = None
                self._appended = 0
        return segment

    def _current_segment(self) -> Path:
        if self._segment is None:
            # Timestamp-prefixed so that a plain sort of the directory is
            # chronological: the audit trail's commit order should match the
            # order decisions were served, and a bare uuid4 would randomize it.
            stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%f")
            self._segment = self.directory / f"{stamp}-{uuid.uuid4().hex[:8]}{PENDING_SUFFIX}"
        return self._segment

    def pending(self) -> list[Path]:
        if not self.directory.exists():
            return []
        return sorted(
            p
            for p in self.directory.iterdir()
            if p.suffix in (PENDING_SUFFIX, CLAIMED_SUFFIX)
        )

    def pending_count(self) -> int:
        """Decisions recorded on disk but not yet committed to Iceberg."""
        return sum(
            sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
            for p in self.pending()
        )

    # ------------------------------------------------------------ draining ----

    def flush(self, catalog) -> FlushResult:
        """Drain every pending segment into Iceberg.

        Safe to call at any time, from a shutdown hook, a background task, or a
        CLI command. Segments are drained oldest-first so the audit trail's
        commit order matches the order decisions were served.
        """
        result = FlushResult()
        for segment in self.pending():
            self._drain_segment(catalog, segment, result)
        return result

    def _drain_segment(self, catalog, segment: Path, result: FlushResult) -> None:
        recovered = segment.suffix == CLAIMED_SUFFIX

        with self._lock:
            if segment == self._segment:
                # Retire the live segment before renaming it, so an append racing
                # this flush opens a new file rather than writing rows into a
                # segment that is about to be drained and unlinked.
                self._segment = None
                self._appended = 0
            claimed = segment if recovered else self._claim(segment)

        if claimed is None:
            # Another drainer took it. Not an error — flush is idempotent.
            return

        envelopes = _read_segment(claimed)
        if not envelopes:
            claimed.unlink(missing_ok=True)
            result.segments_drained += 1
            return

        already = _existing_prediction_ids(catalog, [e.prediction_id for e in envelopes])
        fresh = [e for e in envelopes if e.prediction_id not in already]
        result.envelopes_skipped += len(envelopes) - len(fresh)

        if fresh and recovered:
            # This segment was interrupted mid-flush by a previous process, so
            # its attributions may already be committed without their prediction
            # rows. Clear them so the retry cannot double-write.
            result.orphans_cleaned += _delete_attributions(
                catalog, [e.prediction_id for e in fresh]
            )

        if fresh:
            attributions = [a for e in fresh for a in e.attributions]
            # Attributions first: the prediction row is the commit point.
            result.attributions_written += append_records(catalog, ATTRIBUTIONS, attributions)
            result.predictions_written += append_records(
                catalog, PREDICTIONS, [e.prediction for e in fresh]
            )

        claimed.unlink(missing_ok=True)
        result.segments_drained += 1

    def _claim(self, segment: Path) -> Path | None:
        claimed = segment.with_suffix(CLAIMED_SUFFIX)
        try:
            segment.rename(claimed)
        except (FileNotFoundError, PermissionError, OSError):
            return None
        return claimed


# ------------------------------------------------------------ encoding ----

def _encode(envelope: SpoolEnvelope) -> dict[str, Any]:
    return {
        "prediction": _isoformat(envelope.prediction, _PREDICTION_TS_FIELDS),
        "attributions": [_isoformat(a, _ATTRIBUTION_TS_FIELDS) for a in envelope.attributions],
    }


def _decode(payload: dict[str, Any]) -> SpoolEnvelope:
    return SpoolEnvelope(
        prediction=_parse_ts(payload["prediction"], _PREDICTION_TS_FIELDS),
        attributions=[_parse_ts(a, _ATTRIBUTION_TS_FIELDS) for a in payload["attributions"]],
    )


def _isoformat(record: dict[str, Any], ts_fields: tuple[str, ...]) -> dict[str, Any]:
    out = dict(record)
    for name in ts_fields:
        value = out.get(name)
        if isinstance(value, dt.datetime):
            out[name] = value.isoformat()
    return out


def _parse_ts(record: dict[str, Any], ts_fields: tuple[str, ...]) -> dict[str, Any]:
    out = dict(record)
    for name in ts_fields:
        value = out.get(name)
        if isinstance(value, str):
            out[name] = dt.datetime.fromisoformat(value)
    return out


def _read_segment(segment: Path) -> list[SpoolEnvelope]:
    """Parse a segment, tolerating a torn final line.

    A crash mid-``write`` can leave a partial last line. That decision was never
    acknowledged to a caller — the response is sent only after fsync returns — so
    dropping it loses nothing, whereas refusing to parse the segment would strand
    every complete decision ahead of it.
    """
    envelopes = []
    for line in segment.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            envelopes.append(_decode(json.loads(line)))
        except (json.JSONDecodeError, KeyError):
            continue
    return envelopes


# ------------------------------------------------------------- reading ----

def _existing_prediction_ids(catalog, prediction_ids: list[str]) -> set[str]:
    from pyiceberg.expressions import In

    if not prediction_ids:
        return set()
    table = catalog.load_table(PREDICTIONS.identifier)
    scan = table.scan(
        row_filter=In("prediction_id", set(prediction_ids)),
        selected_fields=("prediction_id",),
    )
    return set(scan.to_arrow()["prediction_id"].to_pylist())


def _delete_attributions(catalog, prediction_ids: list[str]) -> int:
    from pyiceberg.expressions import In

    if not prediction_ids:
        return 0
    table = catalog.load_table(ATTRIBUTIONS.identifier)
    predicate = In("prediction_id", set(prediction_ids))
    existing = table.scan(row_filter=predicate, selected_fields=("prediction_id",)).to_arrow()
    if existing.num_rows == 0:
        return 0
    table.delete(predicate)
    return existing.num_rows
