"""Every Iceberg table in the system, defined in one place.

All tables are declared up front — including the ones that stay empty until the
fairness and erasure phases — because adding a column to a populated Iceberg
table means a schema evolution plus a backfill, and PyIceberg's evolution path
has edge cases around required fields. Declaring early costs nothing; evolving
later costs a day.

Two partition decisions are load-bearing and are documented at their definitions:

* ``features.*`` is bucketed on ``subject_id`` so that erasing one subject
  rewrites ~1/16 of data files rather than all of them. PyIceberg deletes are
  copy-on-write (it rewrites whole data files rather than writing merge-on-read
  delete files), so the partition layout *is* the erasure cost model.
* ``audit.predictions`` and ``audit.attributions`` are partitioned by
  ``day(prediction_ts)`` so time-scoped audit queries prune, and sorted by
  ``prediction_id`` so a single-prediction lookup touches one file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.table.sorting import SortField, SortOrder
from pyiceberg.transforms import BucketTransform, DayTransform, IdentityTransform
from pyiceberg.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    ListType,
    LongType,
    NestedField,
    StringType,
    TimestamptzType,
)

from .catalog import AUDIT_NS, FEATURES_NS

# Number of buckets for subject_id. Sets the erasure blast radius: one deletion
# rewrites roughly 1/SUBJECT_BUCKETS of the feature table's data files.
SUBJECT_BUCKETS = 16


@dataclass(frozen=True)
class TableDef:
    identifier: tuple[str, str]
    schema: Schema
    partition_spec: PartitionSpec = field(default_factory=PartitionSpec)
    sort_order: SortOrder = field(default_factory=SortOrder)
    properties: dict[str, str] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return ".".join(self.identifier)


DEFAULT_PROPERTIES = {
    # v2 is required for any future row-level delete work and is the sane floor.
    "format-version": "2",
    "write.parquet.compression-codec": "zstd",
}


# --------------------------------------------------------------------------- #
# features.*
# --------------------------------------------------------------------------- #

def _credit_application_schema() -> Schema:
    """Feature schema, shared by the live table and the frozen eval holdout.

    The two tables use an identical schema deliberately: the eval set must be
    scannable by exactly the same code path as training data, or the fairness
    comparison is measuring the pipeline rather than the model.
    """
    return Schema(
        NestedField(1, "subject_id", StringType(), required=True),
        NestedField(2, "as_of_ts", TimestamptzType(), required=True),
        NestedField(3, "ingest_batch_id", StringType(), required=True),
        NestedField(4, "source_row_digest", StringType(), required=True),
        NestedField(5, "split", StringType(), required=True),
        NestedField(6, "age", IntegerType(), required=False),
        NestedField(7, "workclass", StringType(), required=False),
        NestedField(8, "fnlwgt", IntegerType(), required=False),
        NestedField(9, "education", StringType(), required=False),
        NestedField(10, "education_num", IntegerType(), required=False),
        NestedField(11, "marital_status", StringType(), required=False),
        NestedField(12, "occupation", StringType(), required=False),
        NestedField(13, "relationship", StringType(), required=False),
        NestedField(14, "race", StringType(), required=False),
        NestedField(15, "sex", StringType(), required=False),
        NestedField(16, "capital_gain", IntegerType(), required=False),
        NestedField(17, "capital_loss", IntegerType(), required=False),
        NestedField(18, "hours_per_week", IntegerType(), required=False),
        NestedField(19, "native_country", StringType(), required=False),
        NestedField(20, "label", IntegerType(), required=True),
        identifier_field_ids=[1],
    )


_SUBJECT_BUCKET_SPEC = PartitionSpec(
    PartitionField(
        source_id=1,
        field_id=1000,
        transform=BucketTransform(SUBJECT_BUCKETS),
        name="subject_bucket",
    )
)

_SUBJECT_SORT = SortOrder(SortField(source_id=1, transform=IdentityTransform()), order_id=1)

CREDIT_APPLICATIONS = TableDef(
    identifier=(FEATURES_NS, "credit_applications"),
    schema=_credit_application_schema(),
    partition_spec=_SUBJECT_BUCKET_SPEC,
    sort_order=_SUBJECT_SORT,
    properties=DEFAULT_PROPERTIES,
)

EVAL_HOLDOUT = TableDef(
    identifier=(FEATURES_NS, "eval_holdout"),
    schema=_credit_application_schema(),
    partition_spec=_SUBJECT_BUCKET_SPEC,
    sort_order=_SUBJECT_SORT,
    properties=DEFAULT_PROPERTIES,
)


# --------------------------------------------------------------------------- #
# audit.*
# --------------------------------------------------------------------------- #

DATA_SNAPSHOTS = TableDef(
    identifier=(AUDIT_NS, "data_snapshots"),
    schema=Schema(
        NestedField(1, "data_snapshot_uuid", StringType(), required=True),
        NestedField(2, "table_identifier", StringType(), required=True),
        NestedField(3, "iceberg_snapshot_id", LongType(), required=True),
        NestedField(4, "captured_at", TimestamptzType(), required=True),
        NestedField(5, "filter_expr", StringType(), required=False),
        NestedField(6, "row_count", LongType(), required=True),
        NestedField(7, "min_as_of_ts", TimestamptzType(), required=False),
        NestedField(8, "max_as_of_ts", TimestamptzType(), required=False),
        NestedField(9, "content_digest", StringType(), required=True),
        NestedField(10, "iceberg_schema_id", IntegerType(), required=False),
        NestedField(11, "partition_spec_id", IntegerType(), required=False),
        # live | pinned | tombstoned
        NestedField(12, "status", StringType(), required=True),
        NestedField(13, "expired_at", TimestamptzType(), required=False),
        identifier_field_ids=[1],
    ),
    properties=DEFAULT_PROPERTIES,
)

MODEL_VERSIONS = TableDef(
    identifier=(AUDIT_NS, "model_versions"),
    schema=Schema(
        NestedField(1, "model_version_id", StringType(), required=True),
        NestedField(2, "registered_name", StringType(), required=False),
        NestedField(3, "mlflow_run_id", StringType(), required=True),
        NestedField(4, "mlflow_model_version", StringType(), required=False),
        NestedField(5, "data_snapshot_uuid", StringType(), required=True),
        NestedField(6, "eval_snapshot_uuid", StringType(), required=True),
        NestedField(7, "trained_at", TimestamptzType(), required=True),
        NestedField(8, "estimator_class", StringType(), required=True),
        NestedField(9, "hyperparams_json", StringType(), required=True),
        NestedField(10, "automl_config_digest", StringType(), required=True),
        NestedField(11, "seed", IntegerType(), required=True),
        # sha256 of the canonical model dump — never of the pickle. See digest.py.
        NestedField(12, "artifact_digest", StringType(), required=True),
        NestedField(13, "env_digest", StringType(), required=True),
        NestedField(14, "code_git_sha", StringType(), required=True),
        NestedField(15, "metrics_json", StringType(), required=False),
        # active | retired | counterfactual | tombstoned
        NestedField(16, "status", StringType(), required=True),
        NestedField(17, "retired_at", TimestamptzType(), required=False),
        NestedField(18, "retire_reason", StringType(), required=False),
        identifier_field_ids=[1],
    ),
    properties=DEFAULT_PROPERTIES,
)

# Materialized at training time and never reconstructed. After a snapshot is
# expired you *cannot* recover membership by time travel, because the rows are
# gone — which is precisely the moment an erasure request needs the answer.
TRAINING_MEMBERSHIP = TableDef(
    identifier=(AUDIT_NS, "training_membership"),
    schema=Schema(
        NestedField(1, "model_version_id", StringType(), required=True),
        NestedField(2, "subject_id", StringType(), required=True),
        NestedField(3, "role", StringType(), required=True),  # train | eval
        NestedField(4, "row_digest", StringType(), required=False),
    ),
    partition_spec=PartitionSpec(
        PartitionField(
            source_id=1, field_id=1000, transform=IdentityTransform(), name="model_version_id"
        ),
        PartitionField(
            source_id=2,
            field_id=1001,
            transform=BucketTransform(SUBJECT_BUCKETS),
            name="subject_bucket",
        ),
    ),
    properties=DEFAULT_PROPERTIES,
)

PREDICTIONS = TableDef(
    identifier=(AUDIT_NS, "predictions"),
    schema=Schema(
        NestedField(1, "prediction_id", StringType(), required=True),
        NestedField(2, "prediction_ts", TimestamptzType(), required=True),
        NestedField(3, "model_version_id", StringType(), required=True),
        NestedField(4, "subject_id", StringType(), required=False),
        NestedField(5, "input_json", StringType(), required=True),
        NestedField(6, "input_digest", StringType(), required=True),
        NestedField(7, "score", DoubleType(), required=True),
        NestedField(8, "threshold", DoubleType(), required=True),
        NestedField(9, "decision", StringType(), required=True),
        # pending | complete | failed
        NestedField(10, "explainer_status", StringType(), required=True),
        NestedField(11, "latency_ms", DoubleType(), required=False),
        NestedField(12, "shap_ms", DoubleType(), required=False),
        identifier_field_ids=[1],
    ),
    partition_spec=PartitionSpec(
        PartitionField(source_id=2, field_id=1000, transform=DayTransform(), name="prediction_day")
    ),
    sort_order=SortOrder(SortField(source_id=1, transform=IdentityTransform()), order_id=1),
    properties=DEFAULT_PROPERTIES,
)

# Tall layout: one row per (prediction, feature). Chosen over a
# map<string,double> column because every interesting audit question is an
# aggregate over features ("which features most often drive denials for group X")
# and tall rows answer those in plain SQL with no unnesting. The ~14x row
# amplification is irrelevant here — Parquet dictionary-encodes feature_name to
# almost nothing.
ATTRIBUTIONS = TableDef(
    identifier=(AUDIT_NS, "attributions"),
    schema=Schema(
        NestedField(1, "prediction_id", StringType(), required=True),
        # Duplicated from predictions purely so this table can prune on day().
        NestedField(2, "prediction_ts", TimestamptzType(), required=True),
        NestedField(3, "model_version_id", StringType(), required=True),
        NestedField(4, "feature_name", StringType(), required=True),
        NestedField(5, "feature_value_str", StringType(), required=False),
        NestedField(6, "shap_value", DoubleType(), required=True),
        NestedField(7, "base_value", DoubleType(), required=True),
        NestedField(8, "abs_rank", IntegerType(), required=True),
        NestedField(9, "explainer_type", StringType(), required=True),  # tree | linear
    ),
    partition_spec=PartitionSpec(
        PartitionField(source_id=2, field_id=1000, transform=DayTransform(), name="prediction_day")
    ),
    sort_order=SortOrder(SortField(source_id=1, transform=IdentityTransform()), order_id=1),
    properties=DEFAULT_PROPERTIES,
)

FAIRNESS_EVALUATIONS = TableDef(
    identifier=(AUDIT_NS, "fairness_evaluations"),
    schema=Schema(
        NestedField(1, "evaluation_id", StringType(), required=True),
        NestedField(2, "model_version_id", StringType(), required=True),
        NestedField(3, "eval_snapshot_uuid", StringType(), required=True),
        NestedField(4, "sensitive_attribute", StringType(), required=True),
        NestedField(5, "group_value", StringType(), required=False),
        NestedField(6, "metric_name", StringType(), required=True),
        NestedField(7, "metric_value", DoubleType(), required=True),
        NestedField(8, "n", LongType(), required=True),
        NestedField(9, "evaluated_at", TimestamptzType(), required=True),
    ),
    partition_spec=PartitionSpec(
        PartitionField(
            source_id=2, field_id=1000, transform=IdentityTransform(), name="model_version_id"
        )
    ),
    properties=DEFAULT_PROPERTIES,
)

FAIRNESS_DECOMPOSITIONS = TableDef(
    identifier=(AUDIT_NS, "fairness_decompositions"),
    schema=Schema(
        NestedField(1, "decomposition_id", StringType(), required=True),
        NestedField(2, "baseline_model_version_id", StringType(), required=True),
        NestedField(3, "candidate_model_version_id", StringType(), required=True),
        NestedField(4, "counterfactual_a_prime_id", StringType(), required=True),
        NestedField(5, "counterfactual_b_prime_id", StringType(), required=True),
        NestedField(6, "eval_snapshot_uuid", StringType(), required=True),
        NestedField(7, "sensitive_attribute", StringType(), required=True),
        NestedField(8, "metric_name", StringType(), required=True),
        NestedField(9, "total_delta", DoubleType(), required=True),
        NestedField(10, "model_effect", DoubleType(), required=True),
        NestedField(11, "data_effect", DoubleType(), required=True),
        NestedField(12, "method", StringType(), required=True),
        # Set false when the eval snapshot this was computed against has since
        # been invalidated by an erasure. The row is kept, not deleted: a
        # comparison that was made and later invalidated is itself audit history.
        NestedField(13, "comparable", BooleanType(), required=True),
        NestedField(14, "computed_at", TimestamptzType(), required=True),
        identifier_field_ids=[1],
    ),
    properties=DEFAULT_PROPERTIES,
)

ERASURE_REQUESTS = TableDef(
    identifier=(AUDIT_NS, "erasure_requests"),
    schema=Schema(
        NestedField(1, "request_id", StringType(), required=True),
        NestedField(2, "subject_id", StringType(), required=True),
        NestedField(3, "requested_at", TimestamptzType(), required=True),
        NestedField(4, "completed_at", TimestamptzType(), required=False),
        NestedField(5, "status", StringType(), required=True),
        NestedField(6, "live_rows_deleted", LongType(), required=False),
        NestedField(7, "contaminated_model_count", IntegerType(), required=False),
        NestedField(
            8,
            "contaminated_model_ids",
            ListType(element_id=100, element_type=StringType(), element_required=False),
            required=False,
        ),
        NestedField(9, "resolution_json", StringType(), required=False),
        identifier_field_ids=[1],
    ),
    properties=DEFAULT_PROPERTIES,
)

# Written *before* physical expiry. This is the mechanism by which provenance
# survives erasure: after the snapshot is gone you can no longer re-execute the
# training run, but you retain an immutable statement of exactly what was
# destroyed, when, why, and which models depended on it.
SNAPSHOT_TOMBSTONES = TableDef(
    identifier=(AUDIT_NS, "snapshot_tombstones"),
    schema=Schema(
        NestedField(1, "data_snapshot_uuid", StringType(), required=True),
        NestedField(2, "iceberg_snapshot_id", LongType(), required=True),
        NestedField(3, "table_identifier", StringType(), required=True),
        NestedField(4, "expired_at", TimestamptzType(), required=True),
        NestedField(5, "reason", StringType(), required=True),
        NestedField(6, "row_count_at_expiry", LongType(), required=True),
        NestedField(7, "content_digest", StringType(), required=True),
        NestedField(8, "schema_json", StringType(), required=True),
        NestedField(
            9,
            "pinned_model_version_ids",
            ListType(element_id=100, element_type=StringType(), element_required=False),
            required=False,
        ),
        NestedField(10, "erasure_request_id", StringType(), required=False),
        identifier_field_ids=[1],
    ),
    properties=DEFAULT_PROPERTIES,
)


ALL_TABLES: tuple[TableDef, ...] = (
    CREDIT_APPLICATIONS,
    EVAL_HOLDOUT,
    DATA_SNAPSHOTS,
    MODEL_VERSIONS,
    TRAINING_MEMBERSHIP,
    PREDICTIONS,
    ATTRIBUTIONS,
    FAIRNESS_EVALUATIONS,
    FAIRNESS_DECOMPOSITIONS,
    ERASURE_REQUESTS,
    SNAPSHOT_TOMBSTONES,
)

BY_NAME: dict[str, TableDef] = {t.name: t for t in ALL_TABLES}
