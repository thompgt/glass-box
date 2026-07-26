"""Attribute a fairness change to the model change or the data change.

Two model versions differ in two ways at once. Someone changed the training
configuration, and the training data moved underneath them. Observing that
demographic parity went from 0.11 to 0.19 does not say which of those did it, and
the remedies are opposite: revert the configuration, or go and look at what
changed in the data. Guessing wrong costs a release cycle.

So the change is decomposed by training the two counterfactuals that hold one
factor fixed. Writing ``A = (config_A, data_A)`` for the baseline and
``B = (config_B, data_B)`` for the candidate::

    A' = (config_A, data_B)     the baseline's configuration, on the new data
    B' = (config_B, data_A)     the candidate's configuration, on the old data

All four are evaluated on the *same* frozen eval snapshot, and the effects are
the average of the two ways round the square::

    model_effect = ½[(M(B') - M(A)) + (M(B)  - M(A'))]
    data_effect  = ½[(M(A') - M(A)) + (M(B)  - M(B'))]

Averaging both paths rather than picking one is the whole point. Along a single
path the two effects are order-dependent — changing the data first and then the
configuration attributes any interaction between them entirely to whichever went
second — and there is no principled reason to prefer one order. The symmetric
average is the two-player Shapley value, and it is the unique attribution that is
order-independent and exactly exhausts the total:

    model_effect + data_effect = M(B) - M(A)

That identity is asserted on every decomposition rather than assumed, because a
decomposition whose parts do not sum to the whole is not a decomposition.

**The counterfactuals are registered model versions, not scratch fits.** They are
written to ``audit.model_versions`` with ``status='counterfactual'``, so the
decomposition cites four model version IDs and anyone can reload any of them,
re-run the evaluation, and check the arithmetic. A decomposition derived from
models that no longer exist would be exactly the unverifiable claim this project
exists to avoid.

**Counterfactuals are reused when they already exist**, which is sound only
because capability #1 holds: training the same recipe on the same pinned data
with the same seed is bit-exact, so a matching registered model version *is* the
counterfactual rather than merely resembling it. In the common case where only
the data changed, ``A'`` is B and ``B'`` is A, and the decomposition trains
nothing at all.
"""

from __future__ import annotations

import datetime as dt
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..digest import canonical_json
from ..schemas import FAIRNESS_DECOMPOSITIONS, MODEL_VERSIONS
from ..train.baseline import RECIPE as BASELINE_RECIPE
from ..train.baseline import train_baseline
from ..train.registry import get_model_version
from ..writer import append_records
from .evaluate import (
    DEFAULT_SENSITIVE_ATTRIBUTE,
    DEFAULT_THRESHOLD,
    FairnessEvaluation,
    evaluate_fairness,
)
from .metrics import MIN_GROUP_N

DECOMPOSITION_NAMESPACE = uuid.UUID("b47e1d90-5c62-5a1f-9e84-3f70c6b28d15")

METHOD = "shapley-2factor"

# The counterfactual half of a decomposition has to *register* a model version,
# not merely fit one, so this maps recipes to registering trainers rather than
# reusing reproduce.FITTERS. Adding a recipe here is what makes it comparable.
TRAINERS = {BASELINE_RECIPE: train_baseline}

# Slack allowed on ``model_effect + data_effect == total_delta``. This is pure
# floating-point rounding over four numbers in [0, 1]; anything larger is a bug in
# the decomposition, not an accumulation. It is roughly ten orders of magnitude
# below the smallest fairness difference anyone would act on.
CLOSURE_TOLERANCE = 1e-12

# Counterfactual model versions are the artifacts of an analysis, not candidates
# for deployment, and nothing should serve one by accident.
COUNTERFACTUAL_STATUS = "counterfactual"


class UncomparableModelsError(ValueError):
    """The two versions cannot be placed on a common footing."""


class DecompositionClosureError(RuntimeError):
    """The parts do not sum to the whole."""


@dataclass
class FairnessDecomposition:
    """One metric's change, split into a model effect and a data effect."""

    decomposition_id: str
    baseline_model_version_id: str
    candidate_model_version_id: str
    counterfactual_a_prime_id: str
    counterfactual_b_prime_id: str
    eval_snapshot_uuid: str
    sensitive_attribute: str
    metric_name: str
    total_delta: float
    model_effect: float
    data_effect: float
    decision_threshold: float
    min_group_n: int = MIN_GROUP_N
    method: str = METHOD
    comparable: bool = True
    computed_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.UTC))
    # The four measured values. Not persisted here because they are already rows
    # in audit.fairness_evaluations, keyed by the four model version IDs above —
    # duplicating them would create a second place for them to disagree.
    measured: dict[str, float] = field(default_factory=dict, repr=False)

    @property
    def regressed(self) -> bool:
        """Differences are unsigned, so a worsening is an increase. See metrics."""
        return self.total_delta > 0

    @property
    def attributed_to(self) -> str:
        """Whichever effect moved the metric further, or 'neither' if nothing did."""
        if abs(self.model_effect) == abs(self.data_effect) == 0.0:
            return "neither"
        return "model" if abs(self.model_effect) >= abs(self.data_effect) else "data"

    def to_record(self) -> dict[str, Any]:
        return {
            "decomposition_id": self.decomposition_id,
            "baseline_model_version_id": self.baseline_model_version_id,
            "candidate_model_version_id": self.candidate_model_version_id,
            "counterfactual_a_prime_id": self.counterfactual_a_prime_id,
            "counterfactual_b_prime_id": self.counterfactual_b_prime_id,
            "eval_snapshot_uuid": self.eval_snapshot_uuid,
            "sensitive_attribute": self.sensitive_attribute,
            "metric_name": self.metric_name,
            "total_delta": float(self.total_delta),
            "model_effect": float(self.model_effect),
            "data_effect": float(self.data_effect),
            "method": self.method,
            "comparable": self.comparable,
            "computed_at": self.computed_at,
            "decision_threshold": float(self.decision_threshold),
            "min_group_n": int(self.min_group_n),
        }


@dataclass
class RegressionAnalysis:
    """Everything the decomposition of one model pair produced."""

    baseline_model_version_id: str
    candidate_model_version_id: str
    counterfactual_a_prime_id: str
    counterfactual_b_prime_id: str
    eval_snapshot_uuid: str
    sensitive_attribute: str
    decision_threshold: float
    min_group_n: int
    evaluations: dict[str, FairnessEvaluation]
    decompositions: list[FairnessDecomposition]

    @property
    def model_version_ids(self) -> tuple[str, str, str, str]:
        return (
            self.baseline_model_version_id,
            self.candidate_model_version_id,
            self.counterfactual_a_prime_id,
            self.counterfactual_b_prime_id,
        )

    @property
    def trained_counterfactuals(self) -> int:
        """How many of the four corners were not already on the record."""
        return len(set(self.model_version_ids)) - 2

    def for_metric(self, name: str) -> FairnessDecomposition:
        for d in self.decompositions:
            if d.metric_name == name:
                return d
        raise KeyError(
            f"{name} was not decomposed: it is not defined for all four of the "
            f"models compared. Three of the four terms is not a decomposition. "
            f"A common cause is that no two groups reached min_group_n="
            f"{self.min_group_n} on this eval set. "
            f"Available: {sorted(d.metric_name for d in self.decompositions)}"
        )

    @property
    def regressions(self) -> list[FairnessDecomposition]:
        return [d for d in self.decompositions if d.regressed]


def decompose_regression(
    catalog,
    baseline_model_version_id: str,
    candidate_model_version_id: str,
    *,
    sensitive_attribute: str = DEFAULT_SENSITIVE_ATTRIBUTE,
    eval_snapshot_uuid: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    min_group_n: int = MIN_GROUP_N,
    root: Path | None = None,
    persist: bool = True,
) -> RegressionAnalysis:
    """Split the fairness change between two model versions into its two causes."""
    if baseline_model_version_id == candidate_model_version_id:
        raise UncomparableModelsError(
            "baseline and candidate are the same model version; the decomposition "
            "would be a table of zeros claiming an analysis had happened"
        )

    a = _require_model(catalog, baseline_model_version_id, "baseline")
    b = _require_model(catalog, candidate_model_version_id, "candidate")
    eval_snapshot_uuid = _common_eval_snapshot(a, b, eval_snapshot_uuid)
    _require_trainers(a, b)

    # A' = the baseline's configuration on the candidate's data, and vice versa.
    a_prime = _model_for(catalog, config=a, data_snapshot_uuid=b["data_snapshot_uuid"],
                         eval_snapshot_uuid=eval_snapshot_uuid, root=root)
    b_prime = _model_for(catalog, config=b, data_snapshot_uuid=a["data_snapshot_uuid"],
                         eval_snapshot_uuid=eval_snapshot_uuid, root=root)

    evaluations = {
        role: evaluate_fairness(
            catalog,
            model_version_id,
            sensitive_attribute=sensitive_attribute,
            eval_snapshot_uuid=eval_snapshot_uuid,
            threshold=threshold,
            min_group_n=min_group_n,
            root=root,
            persist=persist,
        )
        for role, model_version_id in (
            ("A", baseline_model_version_id),
            ("B", candidate_model_version_id),
            ("A_prime", a_prime),
            ("B_prime", b_prime),
        )
    }

    decompositions = [
        _decompose_metric(
            metric,
            evaluations,
            baseline_model_version_id=baseline_model_version_id,
            candidate_model_version_id=candidate_model_version_id,
            a_prime=a_prime,
            b_prime=b_prime,
            eval_snapshot_uuid=eval_snapshot_uuid,
            sensitive_attribute=sensitive_attribute,
            threshold=threshold,
            min_group_n=min_group_n,
        )
        for metric in _metrics_defined_for_all(evaluations)
    ]

    if persist and decompositions:
        _persist(catalog, decompositions)

    return RegressionAnalysis(
        baseline_model_version_id=baseline_model_version_id,
        candidate_model_version_id=candidate_model_version_id,
        counterfactual_a_prime_id=a_prime,
        counterfactual_b_prime_id=b_prime,
        eval_snapshot_uuid=eval_snapshot_uuid,
        sensitive_attribute=sensitive_attribute,
        decision_threshold=float(threshold),
        min_group_n=int(min_group_n),
        evaluations=evaluations,
        decompositions=decompositions,
    )


# ------------------------------------------------------------- the square ----

def _decompose_metric(
    metric: str,
    evaluations: dict[str, FairnessEvaluation],
    *,
    baseline_model_version_id: str,
    candidate_model_version_id: str,
    a_prime: str,
    b_prime: str,
    eval_snapshot_uuid: str,
    sensitive_attribute: str,
    threshold: float,
    min_group_n: int,
) -> FairnessDecomposition:
    m_a = evaluations["A"].metric(metric)
    m_b = evaluations["B"].metric(metric)
    m_a_prime = evaluations["A_prime"].metric(metric)
    m_b_prime = evaluations["B_prime"].metric(metric)

    total = m_b - m_a
    # Both ways round the square, averaged. Along one path the interaction between
    # the two changes is attributed entirely to whichever was applied second.
    model_effect = ((m_b_prime - m_a) + (m_b - m_a_prime)) / 2
    data_effect = ((m_a_prime - m_a) + (m_b - m_b_prime)) / 2

    residual = total - (model_effect + data_effect)
    if abs(residual) > CLOSURE_TOLERANCE:
        raise DecompositionClosureError(
            f"{metric} decomposition does not close: model_effect {model_effect!r} + "
            f"data_effect {data_effect!r} leaves {residual!r} of total_delta "
            f"{total!r} unattributed"
        )

    return FairnessDecomposition(
        decomposition_id=decomposition_id(
            baseline_model_version_id,
            candidate_model_version_id,
            eval_snapshot_uuid,
            sensitive_attribute,
            metric,
            threshold,
            min_group_n,
        ),
        baseline_model_version_id=baseline_model_version_id,
        candidate_model_version_id=candidate_model_version_id,
        counterfactual_a_prime_id=a_prime,
        counterfactual_b_prime_id=b_prime,
        eval_snapshot_uuid=eval_snapshot_uuid,
        sensitive_attribute=sensitive_attribute,
        metric_name=metric,
        total_delta=total,
        model_effect=model_effect,
        data_effect=data_effect,
        decision_threshold=float(threshold),
        min_group_n=int(min_group_n),
        measured={"A": m_a, "B": m_b, "A_prime": m_a_prime, "B_prime": m_b_prime},
    )


def _metrics_defined_for_all(evaluations: dict[str, FairnessEvaluation]) -> list[str]:
    """Only metrics all four corners define.

    A metric missing from one corner cannot be decomposed — three of the four
    terms is not a decomposition — and substituting anything for the absent value
    would invent the effect being reported.
    """
    common: set[str] | None = None
    for evaluation in evaluations.values():
        names = set(evaluation.differences)
        common = names if common is None else common & names
    return sorted(common or set())


def decomposition_id(
    baseline: str,
    candidate: str,
    eval_snapshot_uuid: str,
    sensitive_attribute: str,
    metric_name: str,
    threshold: float,
    min_group_n: int = MIN_GROUP_N,
) -> str:
    key = "|".join(
        [
            baseline,
            candidate,
            eval_snapshot_uuid,
            sensitive_attribute,
            metric_name,
            repr(float(threshold)),
            str(int(min_group_n)),
            METHOD,
        ]
    )
    return str(uuid.uuid5(DECOMPOSITION_NAMESPACE, key))


# --------------------------------------------------------- counterfactuals ----

def _model_for(
    catalog,
    *,
    config: dict[str, Any],
    data_snapshot_uuid: str,
    eval_snapshot_uuid: str,
    root: Path | None,
) -> str:
    """The model version for ``config`` trained on ``data_snapshot_uuid``.

    Reuses an existing registered version when one matches, which is sound because
    training is bit-exact for a fixed (recipe, hyperparams, seed, data): retraining
    would produce the same artifact digest, so the existing row *is* the
    counterfactual. In the common case where only the data changed, this finds the
    baseline and candidate themselves and trains nothing.
    """
    hyperparams = json.loads(config["hyperparams_json"])
    existing = _find_matching(
        catalog,
        recipe=config["recipe"],
        hyperparams=hyperparams,
        seed=config["seed"],
        data_snapshot_uuid=data_snapshot_uuid,
        eval_snapshot_uuid=eval_snapshot_uuid,
    )
    if existing is not None:
        return existing

    trainer = TRAINERS[config["recipe"]]
    result = trainer(
        data_snapshot_uuid=data_snapshot_uuid,
        eval_snapshot_uuid=eval_snapshot_uuid,
        seed=config["seed"],
        hyperparams=hyperparams,
        status=COUNTERFACTUAL_STATUS,
        root=root,
    )
    return result.model_version.model_version_id


def _find_matching(
    catalog,
    *,
    recipe: str,
    hyperparams: dict[str, Any],
    seed: int,
    data_snapshot_uuid: str,
    eval_snapshot_uuid: str,
) -> str | None:
    from pyiceberg.expressions import And, EqualTo

    table = catalog.load_table(MODEL_VERSIONS.identifier)
    rows = (
        table.scan(
            row_filter=And(
                EqualTo("data_snapshot_uuid", data_snapshot_uuid),
                EqualTo("eval_snapshot_uuid", eval_snapshot_uuid),
            )
        )
        .to_arrow()
        .to_pylist()
    )

    wanted = canonical_json(hyperparams)
    matches = [
        r
        for r in rows
        if r["recipe"] == recipe
        and r["seed"] == seed
        and r["status"] != "tombstoned"
        and canonical_json(json.loads(r["hyperparams_json"])) == wanted
    ]
    if not matches:
        return None
    # Deterministic choice, so two runs of the same decomposition cite the same
    # counterfactual and therefore produce the same decomposition_id.
    matches.sort(key=lambda r: (r["trained_at"], r["model_version_id"]))
    return matches[0]["model_version_id"]


# ------------------------------------------------------------- validation ----

def _require_model(catalog, model_version_id: str, role: str) -> dict[str, Any]:
    record = get_model_version(catalog, model_version_id)
    if record is None:
        raise UncomparableModelsError(
            f"no audit.model_versions row for the {role} {model_version_id}"
        )
    return record


def _common_eval_snapshot(a: dict, b: dict, explicit: str | None) -> str:
    """The one eval set both models are measured on.

    If the two versions name different eval snapshots and the caller has not
    chosen, this refuses. Picking one silently would put part of the difference
    between the eval sets into ``total_delta``, and the decomposition would then
    attribute an eval-set change to the model or the data — a confident wrong
    answer, which is worse than no answer.
    """
    if explicit:
        return explicit
    if a["eval_snapshot_uuid"] != b["eval_snapshot_uuid"]:
        raise UncomparableModelsError(
            f"baseline was evaluated on {a['eval_snapshot_uuid']} and candidate on "
            f"{b['eval_snapshot_uuid']}. Comparing them would fold the difference "
            f"between the two eval sets into the result; pass eval_snapshot_uuid "
            f"explicitly to choose one."
        )
    return a["eval_snapshot_uuid"]


def _require_trainers(a: dict, b: dict) -> None:
    unknown = sorted({a["recipe"], b["recipe"]} - set(TRAINERS))
    if unknown:
        raise UncomparableModelsError(
            f"no counterfactual trainer registered for recipe(s) {unknown}. The "
            f"decomposition has to retrain each configuration on the other's data, "
            f"so a recipe that cannot be re-run cannot be decomposed. "
            f"Known: {sorted(TRAINERS)}"
        )


def _persist(catalog, decompositions: list[FairnessDecomposition]) -> int:
    """Append decompositions that are not already recorded.

    Unlike an evaluation, a decomposition is not re-verified on rewrite: it is a
    pure function of four evaluation rows, and those are already protected against
    changing underneath us by :func:`evaluate._persist`.
    """
    from pyiceberg.expressions import In

    ids = {d.decomposition_id for d in decompositions}
    table = catalog.load_table(FAIRNESS_DECOMPOSITIONS.identifier)
    known = set(
        table.scan(
            row_filter=In("decomposition_id", ids), selected_fields=("decomposition_id",)
        )
        .to_arrow()["decomposition_id"]
        .to_pylist()
    )
    fresh = [d.to_record() for d in decompositions if d.decomposition_id not in known]
    return append_records(catalog, FAIRNESS_DECOMPOSITIONS, fresh)
