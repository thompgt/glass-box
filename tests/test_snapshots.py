"""Data-version capture, idempotency, and time travel."""

from __future__ import annotations

from pathlib import Path

import pytest

from glassbox.digest import digest_arrow_table
from glassbox.ingest import adult, ingest_adult
from glassbox.schemas import CREDIT_APPLICATIONS, DATA_SNAPSHOTS, EVAL_HOLDOUT
from glassbox.snapshots import (
    DIGEST_EXCLUDE,
    capture_snapshot,
    get_snapshot,
    materialize,
    sorted_scan,
)
from glassbox.writer import append_arrow


@pytest.fixture
def ingested(catalog, adult_file: Path, gb_root: Path):
    return ingest_adult(gb_root)


def test_ingest_writes_rows_and_captures_two_data_versions(ingested, catalog):
    assert ingested.rows_written == 302  # 300 synthetic + 2 duplicates
    assert sum(ingested.split_counts.values()) == 302

    snaps = catalog.load_table(DATA_SNAPSHOTS.identifier).scan().to_arrow()
    assert snaps.num_rows == 2

    train = get_snapshot(catalog, ingested.train_snapshot.data_snapshot_uuid)
    assert train["row_count"] == ingested.split_counts["train"]
    assert train["filter_expr"] == "split == 'train'"
    assert train["min_as_of_ts"] is not None and train["max_as_of_ts"] is not None


def test_capture_is_idempotent(ingested, catalog):
    """Capturing the same data version twice must not mint a second identity."""
    before = catalog.load_table(DATA_SNAPSHOTS.identifier).scan().to_arrow().num_rows

    again = capture_snapshot(catalog, CREDIT_APPLICATIONS, split="train")

    after = catalog.load_table(DATA_SNAPSHOTS.identifier).scan().to_arrow().num_rows
    assert again.data_snapshot_uuid == ingested.train_snapshot.data_snapshot_uuid
    assert again.content_digest == ingested.train_snapshot.content_digest
    assert after == before, "re-capture must not append a duplicate row"


def test_capture_uuid_distinguishes_filters(ingested, catalog):
    unfiltered = capture_snapshot(catalog, CREDIT_APPLICATIONS)
    assert unfiltered.data_snapshot_uuid != ingested.train_snapshot.data_snapshot_uuid
    assert unfiltered.row_count > ingested.train_snapshot.row_count


def test_sorted_scan_is_deterministic(ingested, catalog):
    """The sort is what keeps model training reproducible across scans."""
    a = sorted_scan(catalog, CREDIT_APPLICATIONS)["subject_id"].to_pylist()
    b = sorted_scan(catalog, CREDIT_APPLICATIONS)["subject_id"].to_pylist()
    assert a == b == sorted(a)


def test_materialize_returns_the_pinned_rows_after_new_data_lands(
    ingested, catalog, adult_file: Path, gb_root: Path
):
    """A captured data version must keep meaning the same rows as the table grows.

    This is the property the whole audit trail depends on: a model pinned to a
    snapshot can be retrained from exactly the rows it saw, not from whatever the
    table happens to hold later.
    """
    pinned = get_snapshot(catalog, ingested.train_snapshot.data_snapshot_uuid)
    pinned_rows = materialize(catalog, pinned, CREDIT_APPLICATIONS).num_rows

    # Land a second batch of different rows.
    extra_path = gb_root / "data" / "raw" / "extra.data"
    extra_path.write_text(
        "\n".join(
            line.replace("Bachelors", "Doctorate")
            for line in adult_file.read_text(encoding="utf-8").splitlines()[:50]
        )
        + "\n",
        encoding="utf-8",
    )
    append_arrow(
        catalog, CREDIT_APPLICATIONS, adult.parse_adult(extra_path, ingest_batch_id="second")
    )

    live_rows = sorted_scan(catalog, CREDIT_APPLICATIONS).num_rows
    assert live_rows > 302

    # The pinned version is unmoved.
    assert materialize(catalog, pinned, CREDIT_APPLICATIONS).num_rows == pinned_rows

    # And a fresh capture is a genuinely different data version.
    fresh = capture_snapshot(catalog, CREDIT_APPLICATIONS, split="train")
    assert fresh.data_snapshot_uuid != ingested.train_snapshot.data_snapshot_uuid
    assert fresh.content_digest != ingested.train_snapshot.content_digest


def test_capture_digests_the_snapshot_it_pins_not_the_live_table(
    ingested, catalog, adult_file: Path, gb_root: Path, monkeypatch
):
    """A capture must describe the id it records, even if the table moves mid-capture.

    ``capture_snapshot`` reads ``current_snapshot()`` and then scans. If the scan
    is unpinned, a commit landing in that window is digested into a record that
    names an earlier snapshot id — so reproduction, which re-reads by id, sees a
    digest mismatch and reports data it can prove is untouched as tampered with.

    The concurrent commit is injected inside the scan so the window is hit every
    run rather than once in a thousand.
    """
    import glassbox.snapshots as snapshots_module

    extra_path = gb_root / "data" / "raw" / "extra.data"
    extra_path.write_text(
        "\n".join(
            line.replace("Bachelors", "Doctorate")
            for line in adult_file.read_text(encoding="utf-8").splitlines()[:50]
        )
        + "\n",
        encoding="utf-8",
    )

    real_scan = snapshots_module.sorted_scan
    intruded = False

    def scan_after_a_concurrent_commit(catalog, td, **kwargs):
        nonlocal intruded
        if not intruded:
            intruded = True
            append_arrow(
                catalog,
                CREDIT_APPLICATIONS,
                adult.parse_adult(extra_path, ingest_batch_id="concurrent"),
            )
        return real_scan(catalog, td, **kwargs)

    monkeypatch.setattr(snapshots_module, "sorted_scan", scan_after_a_concurrent_commit)
    snap = capture_snapshot(catalog, CREDIT_APPLICATIONS, split="train")
    monkeypatch.undo()

    assert intruded, "the concurrent commit never ran"
    # The record describes the snapshot it names: re-reading by that id and
    # digesting again must reproduce the recorded digest exactly.
    replayed = materialize(catalog, snap, CREDIT_APPLICATIONS)
    replayed_digest, _ = digest_arrow_table(replayed, exclude=DIGEST_EXCLUDE)
    assert replayed_digest == snap.content_digest
    assert replayed.num_rows == snap.row_count
    # And it is still the pre-intrusion data version, unchanged by the commit.
    assert snap.data_snapshot_uuid == ingested.train_snapshot.data_snapshot_uuid
    assert snap.content_digest == ingested.train_snapshot.content_digest


def test_eval_holdout_is_frozen_across_reingest(ingested, catalog, gb_root: Path):
    """Re-running ingest must not change the eval set's identity.

    Every cross-version fairness comparison is anchored to this snapshot; if it
    moved, the comparisons would silently stop being comparable.
    """
    before = ingested.eval_snapshot
    again = ingest_adult(gb_root)

    assert again.eval_snapshot.data_snapshot_uuid == before.data_snapshot_uuid
    assert again.eval_snapshot.content_digest == before.content_digest
    assert catalog.load_table(EVAL_HOLDOUT.identifier).scan().to_arrow().num_rows == (
        before.row_count
    )


def test_reingest_is_idempotent(ingested, catalog, gb_root: Path):
    """Re-running ingest must not duplicate subjects.

    PyIceberg does not enforce Iceberg identifier fields on write, so nothing at
    the storage layer prevents a second row per subject. A duplicated subject_id
    would make erasure contamination and training membership ambiguous.
    """
    again = ingest_adult(gb_root)
    assert again.rows_written == 0

    live = sorted_scan(catalog, CREDIT_APPLICATIONS)
    ids = live["subject_id"].to_pylist()
    assert len(ids) == len(set(ids)) == 302

    # Same rows in, same data version out.
    assert again.train_snapshot.data_snapshot_uuid == ingested.train_snapshot.data_snapshot_uuid
    assert again.train_snapshot.content_digest == ingested.train_snapshot.content_digest
