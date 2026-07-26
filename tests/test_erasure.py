"""Capability #4 end to end: erasing a subject, and what has to survive it.

The assertion that matters most is not "the rows are gone from the table" — a
copy-on-write delete satisfies that while leaving every byte on disk, readable by
time travel. It is :func:`test_no_surviving_parquet_file_contains_the_subject`,
which opens every Parquet file in the feature warehouse and looks.

Paired with it is the opposite claim: ``audit.training_membership`` still names
the subject afterwards. That is not a leak, it is the capability. Erasure removes
someone from the training corpus; it must not remove the record that they were
once in it, or the next erasure request has no way to answer "which models saw
me?" — and that question is asked precisely when the data is already gone.
"""

from __future__ import annotations

import pytest

from glassbox.erasure import (
    COMPLETED,
    ErasureError,
    contamination_report,
    erase_subject,
    erasure_requests_for,
    membership_of,
)
from glassbox.ingest import ingest_adult
from glassbox.schemas import (
    CREDIT_APPLICATIONS,
    DATA_SNAPSHOTS,
    EVAL_HOLDOUT,
    SNAPSHOT_TOMBSTONES,
    TRAINING_MEMBERSHIP,
)
from glassbox.snapshots import SnapshotTombstonedError
from glassbox.train import train_baseline
from glassbox.train.reproduce import retrain_from_provenance

pytestmark = pytest.mark.slow


@pytest.fixture
def trained(catalog, adult_file, gb_root):
    ingest_adult(gb_root)
    return train_baseline(root=gb_root)


@pytest.fixture
def subject(catalog, trained) -> str:
    """A subject that really did train the model."""
    rows = (
        catalog.load_table(TRAINING_MEMBERSHIP.identifier)
        .scan(selected_fields=("subject_id", "role"))
        .to_arrow()
        .to_pylist()
    )
    return next(r["subject_id"] for r in rows if r["role"] == "train")


def feature_parquet_files(gb_root):
    """Every Parquet file under the feature namespace, found by walking the disk.

    Deliberately not asking Iceberg which files exist. The whole question is
    whether bytes Iceberg has stopped tracking are still lying about, so the check
    has to be made independently of the thing under test.
    """
    return list((gb_root / "warehouse" / "features").rglob("*.parquet"))


def parquet_holds(paths, subject_id: str) -> bool:
    import pyarrow.parquet as pq

    for path in paths:
        try:
            column = pq.read_table(path, columns=["subject_id"])["subject_id"].to_pylist()
        except Exception:  # noqa: BLE001 — a file without the column cannot hold them
            continue
        if subject_id in column:
            return True
    return False


def rows_for(catalog, td, subject_id: str) -> int:
    from pyiceberg.expressions import EqualTo

    table = catalog.load_table(td.identifier)
    return table.scan(row_filter=EqualTo("subject_id", subject_id)).to_arrow().num_rows


# ------------------------------------------------------------ the deletion ----

def test_the_live_rows_are_gone(catalog, trained, subject):
    result = erase_subject(catalog, subject)

    assert result.status == COMPLETED
    assert result.live_rows_deleted >= 1
    assert rows_for(catalog, CREDIT_APPLICATIONS, subject) == 0
    assert rows_for(catalog, EVAL_HOLDOUT, subject) == 0


def test_no_surviving_parquet_file_contains_the_subject(catalog, trained, subject, gb_root):
    """The claim a copy-on-write delete does not make on its own.

    Phase 0 measured expire_snapshots as metadata-only, so dropping the snapshots
    leaves the Parquet in place. Without the unlink step this assertion fails
    while every table-level check still passes.
    """
    assert parquet_holds(feature_parquet_files(gb_root), subject), "fixture is not testing anything"

    erase_subject(catalog, subject)

    assert not parquet_holds(feature_parquet_files(gb_root), subject)


def test_leaving_the_files_in_place_would_have_left_the_subject_on_disk(
    catalog, trained, subject, gb_root
):
    """The inverse of the test above, so it is clear which step does the work."""
    erase_subject(catalog, subject, unlink_files=False)

    assert parquet_holds(feature_parquet_files(gb_root), subject)


def test_time_travel_cannot_reach_the_subject_afterwards(catalog, trained, subject):
    """Deleting the rows without expiring the history leaves them one scan away."""
    erase_subject(catalog, subject)

    table = catalog.load_table(CREDIT_APPLICATIONS.identifier)
    from pyiceberg.expressions import EqualTo

    for snapshot in table.snapshots():
        found = table.scan(
            row_filter=EqualTo("subject_id", subject),
            selected_fields=("subject_id",),
            snapshot_id=snapshot.snapshot_id,
        ).to_arrow()
        assert found.num_rows == 0, f"snapshot {snapshot.snapshot_id} still has them"


def test_other_subjects_are_untouched(catalog, trained, subject, gb_root):
    """Bucketing bounds the blast radius; it must not have widened it to everyone."""
    before = catalog.load_table(CREDIT_APPLICATIONS.identifier).scan().to_arrow().num_rows

    erase_subject(catalog, subject)

    after = catalog.load_table(CREDIT_APPLICATIONS.identifier).scan().to_arrow().num_rows
    assert after == before - 1
    assert feature_parquet_files(gb_root), "every data file was deleted"


# ------------------------------------------------------ what has to survive ----

def test_the_subject_can_still_be_traced_to_the_models_that_saw_them(catalog, trained, subject):
    """The capability. Asked after the data is gone, because that is when it is asked."""
    before = contamination_report(catalog, subject).model_version_ids
    assert before == [trained.model_version.model_version_id]

    erase_subject(catalog, subject)

    after = contamination_report(catalog, subject)
    assert after.model_version_ids == before
    assert after.live_row_count == 0, "erased from the corpus"
    assert membership_of(catalog, subject), "but not from the record of having been in it"


def test_a_tombstone_records_what_was_destroyed(catalog, trained, subject):
    """After expiry the data version is unknowable; the tombstone is what is left."""
    live = (
        catalog.load_table(DATA_SNAPSHOTS.identifier)
        .scan()
        .to_arrow()
        .to_pylist()
    )
    by_uuid = {r["data_snapshot_uuid"]: r for r in live}

    result = erase_subject(catalog, subject)
    assert result.snapshots_tombstoned

    tombstones = (
        catalog.load_table(SNAPSHOT_TOMBSTONES.identifier).scan().to_arrow().to_pylist()
    )
    assert {t["data_snapshot_uuid"] for t in tombstones} == set(result.snapshots_tombstoned)

    for tombstone in tombstones:
        original = by_uuid[tombstone["data_snapshot_uuid"]]
        assert tombstone["row_count_at_expiry"] == original["row_count"]
        assert tombstone["content_digest"] == original["content_digest"]
        assert tombstone["erasure_request_id"] == result.request_id
        assert trained.model_version.model_version_id in tombstone["pinned_model_version_ids"]
        assert tombstone["schema_json"]


def test_the_model_stays_attributable_but_stops_being_reproducible(catalog, trained, subject):
    """Provenance outlives erasure; reproducibility does not, and says so."""
    model_version_id = trained.model_version.model_version_id
    erase_subject(catalog, subject)

    report = contamination_report(catalog, subject)
    contamination = next(c for c in report.contaminated if c.model_version_id == model_version_id)
    assert contamination.registered
    assert contamination.recipe == "baseline-logreg"
    assert not contamination.reproducible

    with pytest.raises(SnapshotTombstonedError):
        retrain_from_provenance(model_version_id, strict_env=False)


def test_an_eval_split_subject_does_not_tombstone_the_training_data_version(
    catalog, trained, gb_root
):
    """A snapshot pinned to split=='train' does not contain an eval-split subject,
    and tombstoning it would destroy reproducibility to satisfy an erasure that
    never touched its rows."""
    rows = (
        catalog.load_table(EVAL_HOLDOUT.identifier)
        .scan(selected_fields=("subject_id",))
        .to_arrow()
        .to_pylist()
    )
    eval_subject = rows[0]["subject_id"]

    result = erase_subject(catalog, eval_subject)

    tombstoned = set(result.snapshots_tombstoned)
    assert trained.model_version.eval_snapshot_uuid in tombstoned
    assert trained.model_version.data_snapshot_uuid not in tombstoned


# ---------------------------------------------------------- the request row ----

def test_the_request_is_on_the_record_before_anything_is_destroyed(
    catalog, trained, subject, monkeypatch
):
    """A request row that deleted nothing is a harmless orphan. Data destroyed with
    no record of who asked or why is its own compliance failure."""
    import glassbox.erasure.execute as module

    def explode(*args, **kwargs):
        raise RuntimeError("disk gave out")

    monkeypatch.setattr(module, "_delete_subject_rows", explode)

    with pytest.raises(ErasureError, match="failed partway"):
        erase_subject(catalog, subject)

    requests = erasure_requests_for(catalog, subject)
    assert len(requests) == 1
    assert requests[0]["status"] == "failed"
    assert requests[0]["contaminated_model_ids"] == [trained.model_version.model_version_id]
    assert "disk gave out" in requests[0]["resolution_json"]


def test_the_completed_request_records_what_it_cost(catalog, trained, subject):
    result = erase_subject(catalog, subject)

    row = erasure_requests_for(catalog, subject)[0]
    assert row["status"] == COMPLETED
    assert row["completed_at"] is not None
    assert row["live_rows_deleted"] == result.live_rows_deleted
    assert row["contaminated_model_count"] == 1
    assert row["contaminated_model_ids"] == [trained.model_version.model_version_id]
    assert str(result.data_files_unlinked) in row["resolution_json"]


def test_erasing_someone_who_was_never_in_the_data_is_a_valid_request(catalog, trained):
    """'You are not in our data' is an answer, and it belongs on the record."""
    result = erase_subject(catalog, "never-heard-of-them")

    assert result.status == COMPLETED
    assert result.live_rows_deleted == 0
    assert result.contaminated_model_ids == []
    assert result.snapshots_tombstoned == []
    assert len(erasure_requests_for(catalog, "never-heard-of-them")) == 1


# --------------------------------------------------------------- the policy ----

def test_a_contaminated_model_is_not_retired_by_default(catalog, trained, subject):
    """Whether a model must come out of service is a legal judgement, and burying
    it inside a delete function is how it stops being made deliberately."""
    result = erase_subject(catalog, subject)

    assert result.models_retired == []
    assert contamination_report(catalog, subject).serving


def test_retirement_happens_when_a_caller_actually_decides_it(catalog, trained, subject):
    result = erase_subject(catalog, subject, retire_contaminated=True)

    assert result.models_retired == [trained.model_version.model_version_id]

    report = contamination_report(catalog, subject)
    assert report.serving == []
    assert report.model_version_ids == [trained.model_version.model_version_id], (
        "retired, not forgotten"
    )
