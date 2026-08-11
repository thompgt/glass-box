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
import itertools
import json
import os
import threading
import uuid
import warnings
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
# Damaged segments are copied here rather than deleted. See Spool._quarantine.
CORRUPT_DIRNAME = "corrupt"
CORRUPT_SUFFIX = ".corrupt"

# Timestamp columns, which JSON cannot represent and Iceberg requires as real
# timestamptz values. Listed explicitly rather than sniffed, because a
# type-sniffing round trip would silently convert any string that happens to
# parse as a date.
_PREDICTION_TS_FIELDS = ("prediction_ts",)
_ATTRIBUTION_TS_FIELDS = ("prediction_ts",)

# Breaks ties between segments opened within the same clock tick. The wall clock
# alone is not enough: ``datetime.now()`` resolves to roughly a millisecond on
# Windows, and a server under load rotates segments faster than that, so two
# segments routinely carry an identical timestamp. Sorting them by the random
# uuid suffix would then commit them out of order.
_sequence = itertools.count()


def _stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%f")


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


class CorruptSegmentWarning(UserWarning):
    """A spool segment held an unreadable line that was not its last.

    The final line of a segment can be torn by a crash mid-write and is
    tolerated. An earlier one cannot be explained that way: the append that wrote
    it returned, so its decision was served.
    """


@dataclass
class FlushResult:
    predictions_written: int = 0
    attributions_written: int = 0
    envelopes_skipped: int = 0
    segments_drained: int = 0
    orphans_cleaned: int = 0
    segments_quarantined: int = 0
    corrupt_lines: int = 0

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
    # Serializes whole flushes. The append lock is not enough: it is released the
    # moment a segment is claimed, so two flushes could each claim a *different*
    # segment and interleave their Iceberg commits — and, worse, a flush entering
    # while another is mid-drain used to see the in-flight ``.flushing`` file and
    # drain it a second time. One flush at a time per process.
    _flush_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    # Segments a *previous* process claimed and never finished. Determined once,
    # at construction, because that is the only moment at which a ``.flushing``
    # file is unambiguously stale: after this object exists, every such file is
    # either one of these or one this process is draining right now.
    _recoverable: set[Path] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        self._recoverable = self._scan_recoverable()

    @property
    def directory(self) -> Path:
        return Path(self.root) / SPOOL_DIRNAME

    def _scan_recoverable(self) -> set[Path]:
        """Claimed-but-undrained segments left behind by a previous process."""
        if not self.directory.exists():
            return set()
        return {p for p in self.directory.iterdir() if p.suffix == CLAIMED_SUFFIX}

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
            # The sequence number orders segments the clock cannot separate; the
            # uuid only keeps two processes sharing a root from colliding, and is
            # last precisely because it carries no order.
            stamp = _stamp()
            seq = next(_sequence)
            self._segment = (
                self.directory / f"{stamp}-{seq:08d}-{uuid.uuid4().hex[:8]}{PENDING_SUFFIX}"
            )
        return self._segment

    def pending(self) -> list[Path]:
        """Segments awaiting a commit: unclaimed ones, plus crash leftovers.

        A ``.flushing`` segment is included **only** when it was already on disk
        when this Spool was constructed. Returning every ``.flushing`` file would
        hand a caller a segment another flush is actively draining, and both
        drainers would then miss each other's existence check and append the same
        audit rows twice — the exact duplication this module exists to prevent.
        """
        if not self.directory.exists():
            return []
        live = [p for p in self.directory.iterdir() if p.suffix == PENDING_SUFFIX]
        stale = [p for p in self._recoverable if p.exists()]
        return sorted(live + stale)

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
        with self._flush_lock:
            result = FlushResult()
            for segment in self.pending():
                self._drain_segment(catalog, segment, result)
            return result

    def _drain_segment(self, catalog, segment: Path, result: FlushResult) -> None:
        with self._lock:
            # Claiming a recovered segment is removing it from the recovery set:
            # its ``.flushing`` name is already the claim, and taking it out here
            # means a second drainer cannot also treat it as recoverable.
            recovered = segment in self._recoverable
            self._recoverable.discard(segment)
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

        envelopes, corrupt_lines = _read_segment(claimed)
        if corrupt_lines:
            # Every line but the last was written by a completed, fsynced append
            # whose response has already reached a caller. One of them being
            # unreadable is not an expected condition — it is disk or filesystem
            # damage — and the old code stepped over it, discarding an
            # acknowledged decision with nothing anywhere recording that it
            # existed. Keep the bytes, say so, and still commit what parsed.
            result.corrupt_lines += len(corrupt_lines)
            result.segments_quarantined += 1
            self._quarantine(claimed, corrupt_lines)

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

    def _quarantine(self, segment: Path, corrupt_lines: list[int]) -> Path:
        """Copy a damaged segment aside, and say loudly that it happened.

        A copy rather than a move: the parseable envelopes are still committed
        from the original in the normal way, so the audit trail is as complete as
        the bytes allow, and the quarantined file is evidence — the only remaining
        record of the lines that could not be read.
        """
        quarantine_dir = self.directory / CORRUPT_DIRNAME
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        destination = quarantine_dir / f"{segment.name}.{_stamp()}{CORRUPT_SUFFIX}"
        destination.write_bytes(segment.read_bytes())

        warnings.warn(
            f"spool segment {segment.name} has {len(corrupt_lines)} unreadable "
            f"line(s) at {corrupt_lines} that are not the final line, so they were "
            f"fully written and their decisions were already served. Every line "
            f"that parsed has been committed; the segment is preserved at "
            f"{destination} as the only remaining record of the rest.",
            CorruptSegmentWarning,
            stacklevel=2,
        )
        return destination

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


def _read_segment(segment: Path) -> tuple[list[SpoolEnvelope], list[int]]:
    """Parse a segment. Returns ``(envelopes, corrupt_line_numbers)``.

    A crash mid-``write`` can leave a partial **last** line. That decision was
    never acknowledged to a caller — the response is sent only after fsync
    returns — so dropping it loses nothing, whereas refusing to parse the segment
    would strand every complete decision ahead of it.

    That reasoning covers exactly one line, and it used to be applied to all of
    them. Any *earlier* unreadable line is a different event: it was written by
    an append that returned, so a caller was told a decision had been recorded.
    Silently skipping it deletes that decision from the only place it exists.
    Those line numbers come back to the caller, which preserves the file and
    warns, rather than being swallowed here.
    """
    envelopes: list[SpoolEnvelope] = []
    corrupt: list[int] = []
    lines = segment.read_text(encoding="utf-8").splitlines()
    last = len(lines) - 1

    for number, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            envelopes.append(_decode(json.loads(line)))
        except (json.JSONDecodeError, KeyError, TypeError):
            if number != last:
                corrupt.append(number + 1)  # 1-based, for a human reading a file
    return envelopes, corrupt


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
