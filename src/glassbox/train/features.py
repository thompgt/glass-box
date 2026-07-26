"""Feature matrix construction.

Everything here has to be a pure function of the data, with no dependence on row
order, iteration order, or wall-clock time. Two places where that is easy to get
wrong and hard to notice:

* ``OneHotEncoder`` derives its categories from the training data. Its output
  column order therefore depends on the *values present*, not on their order, but
  only because we pass explicitly sorted categories rather than letting it infer
  them from a possibly-unsorted scan.
* Missing values in Adult are ``?``, which we map to a real ``"__missing__"``
  category instead of dropping the row. Dropping would make the training set a
  function of null placement, and would quietly change which subjects appear in
  ``training_membership`` — which is the erasure contamination index.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = (
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
)

CATEGORICAL_FEATURES = (
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
)

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "label"
MISSING = "__missing__"

# Columns that must never reach the model. subject_id is an identifier and would
# let the model memorize individuals; the rest are provenance bookkeeping.
EXCLUDED = ("subject_id", "as_of_ts", "ingest_batch_id", "source_row_digest", "split", "label")


def to_frame(table: pa.Table):
    """Arrow -> pandas, with nulls resolved deterministically.

    Timestamps are converted with ``timestamp_as_object`` off and coerced to
    microseconds elsewhere; here we only touch feature columns, so no ns/us
    ambiguity reaches pandas.
    """
    import pandas as pd

    df = table.select([c for c in FEATURE_COLUMNS if c in table.column_names]).to_pandas()

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(MISSING).astype(str)
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            # Adult has no true numeric nulls, but a median fill would make the
            # matrix depend on the training set in a way the digest wouldn't see.
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("float64")

    return df[list(FEATURE_COLUMNS)]


def target_of(table: pa.Table) -> np.ndarray:
    return np.asarray(table[TARGET].to_pylist(), dtype=np.int64)


def subject_ids_of(table: pa.Table) -> list[str]:
    return table["subject_id"].to_pylist()


def build_preprocessor(train_table: pa.Table) -> ColumnTransformer:
    """Preprocessor with explicitly sorted categories.

    Passing ``categories=`` rather than letting the encoder infer them removes the
    last dependence on scan order: inferred categories come out in order of first
    appearance in some scikit-learn versions, which would make the encoded column
    layout — and therefore the coefficients, and therefore the digest — depend on
    how Iceberg happened to lay out data files.
    """
    df = to_frame(train_table)
    categories = [np.array(sorted(df[col].unique()), dtype=object) for col in CATEGORICAL_FEATURES]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(NUMERIC_FEATURES)),
            (
                "cat",
                OneHotEncoder(
                    categories=categories,
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float64,
                ),
                list(CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def build_pipeline(train_table: pa.Table, estimator) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor(train_table)), ("model", estimator)])
