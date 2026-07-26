"""Retrain a registered model from its recorded provenance and compare digests.

This is capability #1, and the definition of "bit-exact" it implements is
deliberate. Three things could be meant:

1. Byte-identical ``pickle`` output — not achievable, and not worth chasing. See
   :mod:`glassbox.train.canonical`.
2. Byte-identical **canonical model dump** — the coefficients, intercept, class
   order and encoder feature names. Achievable, and what ``artifact_digest``
   records.
3. Identical predictions on a fixed probe set.

We assert **(2)**, and assert **(3)** as well rather than instead. (2) alone
could in principle pass while some un-digested part of the pipeline differed;
(3) alone is too weak, since two genuinely different models can agree on a small
probe. Requiring both closes the gap in each direction.

Reproduction refuses to run under a different environment digest. Retraining
under a different NumPy is a different experiment, and reporting the resulting
mismatch as "not reproducible" would be a false claim about the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..catalog import glassbox_root, load_catalog
from ..digest import canonical_json, env_digest
from ..schemas import CREDIT_APPLICATIONS, EVAL_HOLDOUT
from ..snapshots import get_snapshot, materialize
from . import features as F
from .baseline import RECIPE as BASELINE_RECIPE
from .baseline import fit_baseline
from .canonical import model_digest
from .registry import get_model_version, load_model_checked

PROBE_ROWS = 256


class EnvironmentDriftError(RuntimeError):
    """The environment differs from the one the model was trained in."""


class UnknownRecipeError(RuntimeError):
    """No trainer is registered for the recipe recorded on this model version."""


@dataclass
class ReproductionResult:
    model_version_id: str
    recipe: str
    recorded_digest: str
    reproduced_digest: str
    data_snapshot_uuid: str
    recorded_content_digest: str
    reproduced_content_digest: str
    predictions_match: bool
    probe_rows: int
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def digests_match(self) -> bool:
        return self.recorded_digest == self.reproduced_digest

    @property
    def data_matches(self) -> bool:
        return self.recorded_content_digest == self.reproduced_content_digest

    @property
    def reproduced(self) -> bool:
        return self.digests_match and self.data_matches and self.predictions_match


# Recipe name -> fitter. Reproduction dispatches on the recorded recipe rather
# than on estimator_class, because two different recipes can produce the same
# estimator class and re-running the wrong one would report a false mismatch.
FITTERS = {BASELINE_RECIPE: fit_baseline}


def retrain_from_provenance(
    model_version_id: str,
    *,
    root: Path | None = None,
    strict_env: bool = True,
) -> ReproductionResult:
    root = root or glassbox_root()
    catalog = load_catalog(root)

    record = get_model_version(catalog, model_version_id)
    if record is None:
        raise ValueError(f"no audit.model_versions row for {model_version_id}")

    current_env = env_digest()
    if strict_env and record["env_digest"] != current_env:
        raise EnvironmentDriftError(
            f"model {model_version_id} was trained under env_digest "
            f"{record['env_digest']}, current environment is {current_env}. "
            f"Retraining here would be a different experiment; pass "
            f"strict_env=False to proceed anyway and report the result as advisory."
        )

    fitter = FITTERS.get(record["recipe"])
    if fitter is None:
        raise UnknownRecipeError(
            f"model {model_version_id} records recipe {record['recipe']!r}, which has "
            f"no registered trainer (known: {sorted(FITTERS)})"
        )

    train_snap = get_snapshot(catalog, record["data_snapshot_uuid"])
    eval_snap = get_snapshot(catalog, record["eval_snapshot_uuid"])
    if train_snap is None or eval_snap is None:
        raise ValueError(f"model {model_version_id} references a missing data snapshot")

    # materialize() raises SnapshotTombstonedError if the data version was
    # expired to satisfy an erasure — a designed refusal, not a crash.
    hyperparams = _decode_hyperparams(record["hyperparams_json"])
    fit = fitter(catalog, train_snap, eval_snap, seed=record["seed"], hyperparams=hyperparams)

    reproduced_digest = model_digest(fit.model)
    reproduced_content = _content_digest_of(catalog, train_snap)

    original, _ = load_model_checked(catalog, model_version_id, root)
    predictions_match, probe_rows = _predictions_agree(original, fit.model, fit.eval_table)

    return ReproductionResult(
        model_version_id=model_version_id,
        recipe=record["recipe"],
        recorded_digest=record["artifact_digest"],
        reproduced_digest=reproduced_digest,
        data_snapshot_uuid=record["data_snapshot_uuid"],
        recorded_content_digest=train_snap["content_digest"],
        reproduced_content_digest=reproduced_content,
        predictions_match=predictions_match,
        probe_rows=probe_rows,
        metrics=fit.metrics,
    )


def _decode_hyperparams(raw: str) -> dict[str, Any]:
    import json

    params = json.loads(raw)
    # canonical_json round-trips through JSON, so the values come back as the
    # types they were written with; nothing to coerce, but assert the shape.
    if not isinstance(params, dict):
        raise ValueError(f"hyperparams_json is not an object: {raw!r}")
    return params


def _content_digest_of(catalog, snap: dict) -> str:
    """Re-digest the pinned rows.

    Recomputed rather than trusted so that a reproduction failure can distinguish
    "the data moved under me" from "the fit is nondeterministic" — two problems
    with entirely different fixes.
    """
    from ..digest import digest_arrow_table
    from ..snapshots import DIGEST_EXCLUDE

    is_training_table = snap["table_identifier"] == CREDIT_APPLICATIONS.name
    td = CREDIT_APPLICATIONS if is_training_table else EVAL_HOLDOUT
    table = materialize(catalog, snap, td)
    digest, _ = digest_arrow_table(table, exclude=DIGEST_EXCLUDE)
    return digest


def _predictions_agree(original, reproduced, eval_table) -> tuple[bool, int]:
    probe = eval_table.slice(0, min(PROBE_ROWS, eval_table.num_rows))
    X = F.to_frame(probe)
    a = original.predict_proba(X)[:, 1]
    b = reproduced.predict_proba(X)[:, 1]
    return bool(np.array_equal(a, b)), int(probe.num_rows)


def summary(result: ReproductionResult) -> str:
    return canonical_json(
        {
            "model_version_id": result.model_version_id,
            "recipe": result.recipe,
            "reproduced": result.reproduced,
            "digests_match": result.digests_match,
            "data_matches": result.data_matches,
            "predictions_match": result.predictions_match,
        }
    )
