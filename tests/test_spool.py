"""The write-ahead spool, and the crash windows it exists to survive.

The interesting tests here are not "does a flush write rows". They are the three
ways a process can die around a flush, each of which has a different correct
outcome:

* died before the flush          -> the decision is still committed
* died between the two commits   -> retried without duplicating attributions
* died after the commits         -> retried without duplicating anything

A spool that only passes the happy path is a batching optimization wearing an
audit trail's clothes.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from glassbox.schemas import ATTRIBUTIONS, PREDICTIONS
from glassbox.serve import Spool, SpoolEnvelope
from glassbox.writer import append_records

TS = dt.datetime(2026, 3, 14, 15, 9, 26, tzinfo=dt.UTC)


def envelope(prediction_id: str, *, n_features: int = 3) -> SpoolEnvelope:
    prediction = {
        "prediction_id": prediction_id,
        "prediction_ts": TS,
        "model_version_id": "mv-1",
        "subject_id": f"subject-{prediction_id}",
        "input_json": '{"age":39}',
        "input_digest": "d" * 64,
        "score": 0.25,
        "threshold": 0.5,
        "decision": "deny",
        "explainer_status": "complete",
        "latency_ms": 4.0,
        "shap_ms": 1.0,
    }
    attributions = [
        {
            "prediction_id": prediction_id,
            "prediction_ts": TS,
            "model_version_id": "mv-1",
            "feature_name": f"num__f{i}",
            "feature_value_str": str(i),
            "shap_value": 0.1 * i,
            "base_value": -0.7,
            "abs_rank": i + 1,
            "explainer_type": "linear",
        }
        for i in range(n_features)
    ]
    return SpoolEnvelope(prediction=prediction, attributions=attributions)


def rows(catalog, table_def) -> list[dict]:
    return catalog.load_table(table_def.identifier).scan().to_arrow().to_pylist()


@pytest.fixture
def spool(gb_root):
    return Spool(root=gb_root, batch_size=100)


# ------------------------------------------------------------ durability ----

def test_a_decision_is_on_disk_before_it_is_in_iceberg(spool, catalog):
    """The response must never outrun the record of what was decided."""
    spool.append(envelope("p1"))

    assert spool.pending_count() == 1
    assert rows(catalog, PREDICTIONS) == []

    result = spool.flush(catalog)

    assert result.predictions_written == 1
    assert result.attributions_written == 3
    assert spool.pending_count() == 0


def test_a_spooled_decision_survives_losing_the_process(gb_root, catalog):
    """Died before the flush: a brand-new Spool over the same root finds it."""
    Spool(root=gb_root).append(envelope("p1"))

    reborn = Spool(root=gb_root)
    assert reborn.pending_count() == 1
    assert reborn.flush(catalog).predictions_written == 1

    assert [r["prediction_id"] for r in rows(catalog, PREDICTIONS)] == ["p1"]


def test_timestamps_survive_the_json_round_trip_as_timestamps(spool, catalog):
    """JSON has no timestamp type, and Iceberg will not accept a string for one."""
    spool.append(envelope("p1"))
    spool.flush(catalog)

    committed = rows(catalog, PREDICTIONS)[0]
    assert committed["prediction_ts"] == TS
    assert rows(catalog, ATTRIBUTIONS)[0]["prediction_ts"] == TS


# ------------------------------------------------------------- ordering ----

def test_attributions_are_committed_before_the_prediction_citing_them(spool, catalog, monkeypatch):
    """The prediction row is the commit point, so it must be written last.

    The reverse order would publish, however briefly, a prediction whose
    attributions do not exist — a state indistinguishable from tampering to
    anything reading the table.
    """
    written: list[str] = []
    import glassbox.serve.spool as spool_module

    real = spool_module.append_records

    def spy(catalog, td, records):
        written.append(td.name)
        return real(catalog, td, records)

    monkeypatch.setattr(spool_module, "append_records", spy)
    spool.append(envelope("p1"))
    spool.flush(catalog)

    assert written == [ATTRIBUTIONS.name, PREDICTIONS.name]


def test_segments_drain_in_the_order_decisions_were_served(gb_root, catalog, monkeypatch):
    """Commit order, not scan order.

    A scan makes no ordering promise across data files, so reading the table back
    would test PyIceberg's file layout rather than the spool. What the spool owes
    is that it *commits* oldest-segment-first, which is what the audit trail's
    snapshot history then records.
    """
    import glassbox.serve.spool as spool_module

    committed: list[str] = []
    real = spool_module.append_records

    def spy(catalog, td, records):
        if td.name == PREDICTIONS.name:
            committed.extend(r["prediction_id"] for r in records)
        return real(catalog, td, records)

    monkeypatch.setattr(spool_module, "append_records", spy)

    spool = Spool(root=gb_root, batch_size=1)
    for i in range(3):
        spool.append(envelope(f"p{i}"))

    assert len(spool.pending()) == 3
    assert spool.pending() == sorted(spool.pending()), "segment names must sort chronologically"

    spool.flush(catalog)
    assert committed == ["p0", "p1", "p2"]


# -------------------------------------------------------- crash recovery ----

def test_replaying_a_fully_committed_segment_does_not_duplicate_it(spool, catalog):
    """Died after both commits, before the unlink."""
    spool.append(envelope("p1"))
    segment = spool.pending()[0]
    surviving = segment.read_text(encoding="utf-8")

    spool.flush(catalog)
    # The crash: the segment is still on disk, already committed.
    (spool.directory / segment.name).write_text(surviving, encoding="utf-8")

    result = Spool(root=spool.root).flush(catalog)

    assert result.envelopes_skipped == 1
    assert result.predictions_written == 0
    assert len(rows(catalog, PREDICTIONS)) == 1
    assert len(rows(catalog, ATTRIBUTIONS)) == 3


def test_replaying_a_half_committed_segment_does_not_duplicate_attributions(spool, catalog):
    """Died between the two commits — the window that actually corrupts data.

    Attributions are committed, the prediction is not. Retrying naively would
    append a second copy of every attribution, and an audit query summing
    contributions would then double-count that decision.
    """
    env = envelope("p1")
    append_records(catalog, ATTRIBUTIONS, env.attributions)  # the orphaned half

    spool.append(env)
    # Mark the segment as one a previous process had claimed and not finished.
    segment = spool.pending()[0]
    segment.rename(segment.with_suffix(".flushing"))

    result = Spool(root=spool.root).flush(catalog)

    assert result.orphans_cleaned == 3
    assert len(rows(catalog, ATTRIBUTIONS)) == 3, "attributions were double-written"
    assert len(rows(catalog, PREDICTIONS)) == 1


def test_a_torn_final_line_does_not_strand_the_decisions_ahead_of_it(spool, catalog):
    """A partial line was never acknowledged, so dropping it loses nothing."""
    spool.append(envelope("p1"))
    spool.append(envelope("p2"))

    segment = spool.pending()[0]
    with open(segment, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"prediction": {"prediction_id": "p3"}})[:40])

    result = spool.flush(catalog)

    assert result.predictions_written == 2
    assert {r["prediction_id"] for r in rows(catalog, PREDICTIONS)} == {"p1", "p2"}


# --------------------------------------------------------------- batching ----

def test_a_full_batch_rotates_to_a_new_segment(gb_root):
    spool = Spool(root=gb_root, batch_size=2)
    for i in range(5):
        spool.append(envelope(f"p{i}"))

    assert len(spool.pending()) == 3
    assert spool.pending_count() == 5


def test_flushing_an_empty_spool_is_a_no_op(spool, catalog):
    result = spool.flush(catalog)

    assert not result
    assert result.segments_drained == 0
    assert rows(catalog, PREDICTIONS) == []
