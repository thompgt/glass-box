"""The properties everything else rests on.

If any assertion here fails, no downstream provenance claim means anything: a
subject id that is not stable makes erasure contamination wrong, and a content
digest that depends on scan order makes reproducibility flap for reasons
unrelated to the model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from glassbox.digest import canonical_json, content_digest, digest_arrow_table, row_digest
from glassbox.ingest import adult
from glassbox.schemas import ALL_TABLES, CREDIT_APPLICATIONS
from glassbox.writer import arrow_schema_for

from .conftest import DUPLICATE_ROW

# --------------------------------------------------------------- digests ----

def test_canonical_json_is_key_order_independent():
    a = {"b": 1, "a": 2, "c": {"z": 1, "y": 2}}
    b = {"c": {"y": 2, "z": 1}, "a": 2, "b": 1}
    assert canonical_json(a) == canonical_json(b)
    assert row_digest(a) == row_digest(b)


def test_content_digest_is_row_order_independent():
    """Iceberg gives no ordering guarantee across data files.

    A data version's identity must therefore be a property of the *set* of rows,
    not of the order a particular scan happened to return them in.
    """
    digests = [row_digest({"subject_id": f"s{i}", "v": i}) for i in range(50)]
    assert content_digest(digests) == content_digest(list(reversed(digests)))
    assert content_digest(digests) != content_digest(digests[:-1])


def test_content_digest_detects_a_single_changed_value():
    base = [row_digest({"subject_id": f"s{i}", "v": i}) for i in range(50)]
    changed = list(base)
    changed[10] = row_digest({"subject_id": "s10", "v": 999})
    assert content_digest(base) != content_digest(changed)


def test_digest_arrow_table_excludes_ingest_metadata(adult_file: Path):
    """The same source rows ingested in two batches are the same data version."""
    t1 = adult.parse_adult(adult_file, ingest_batch_id="batch-one")
    t2 = adult.parse_adult(adult_file, ingest_batch_id="batch-two")

    d1, _ = digest_arrow_table(t1, exclude=("ingest_batch_id", "source_row_digest"))
    d2, _ = digest_arrow_table(t2, exclude=("ingest_batch_id", "source_row_digest"))
    assert d1 == d2

    # Without the exclusion the batch id would leak into the identity.
    d1_all, _ = digest_arrow_table(t1)
    d2_all, _ = digest_arrow_table(t2)
    assert d1_all != d2_all


# ------------------------------------------------------------ subject ids ----

def test_subject_id_is_stable_across_reparses(adult_file: Path):
    a = adult.parse_adult(adult_file, ingest_batch_id="x")["subject_id"].to_pylist()
    b = adult.parse_adult(adult_file, ingest_batch_id="y")["subject_id"].to_pylist()
    assert a == b


def test_duplicate_rows_get_distinct_subject_ids(adult_file: Path):
    """Adult contains byte-identical rows for what must be treated as distinct people.

    Hashing the row alone would collapse them into one identity, which would make
    an erasure contamination report silently wrong: deleting one "subject" would
    claim to have erased several.
    """
    ids = adult.parse_adult(adult_file, ingest_batch_id="x")["subject_id"].to_pylist()
    assert len(ids) == len(set(ids)), "subject ids must be unique"

    first = adult.subject_id_for(DUPLICATE_ROW, 0)
    second = adult.subject_id_for(DUPLICATE_ROW, 1)
    assert first != second
    assert {first, second} <= set(ids)


def test_split_assignment_is_a_function_of_subject_id(adult_file: Path):
    table = adult.parse_adult(adult_file, ingest_batch_id="x")
    for sid, split in zip(
        table["subject_id"].to_pylist(), table["split"].to_pylist(), strict=True
    ):
        assert adult.split_for(sid) == split
    assert set(table["split"].to_pylist()) <= {"train", "eval", "holdout"}


def test_as_of_ts_does_not_depend_on_ingest_time(adult_file: Path):
    """A clock-derived timestamp would make every re-ingest a new data version."""
    table = adult.parse_adult(adult_file, ingest_batch_id="x")
    for sid, ts in zip(
        table["subject_id"].to_pylist(), table["as_of_ts"].to_pylist(), strict=True
    ):
        assert adult.as_of_ts_for(sid) == ts


# ----------------------------------------------------------------- schema ----

@pytest.mark.parametrize("td", ALL_TABLES, ids=lambda td: td.name)
def test_every_table_has_a_derivable_arrow_schema(td):
    """Guards the Phase 0 finding that PyIceberg validates Arrow schemas strictly."""
    schema = arrow_schema_for(td)
    assert len(schema) == len(td.schema.fields)
    assert [f.name for f in schema] == [f.name for f in td.schema.fields]


def test_ingest_arrow_schema_matches_the_feature_table():
    assert adult.arrow_schema() == arrow_schema_for(CREDIT_APPLICATIONS)
