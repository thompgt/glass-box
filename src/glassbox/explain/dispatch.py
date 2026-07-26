"""Choosing an explainer for a model class.

The search space is constrained to model families that have an **exact** SHAP
explainer: ``TreeExplainer`` for the tree families and ``LinearExplainer`` for L2
logistic regression. There is no model-agnostic fallback, and that omission is
the design.

``KernelExplainer`` and ``PermutationExplainer`` would let any estimator through,
but they turn every attribution from an exact Shapley value into a sampled
approximation carrying its own variance — while the whole claim of this system is
"this is *the* attribution behind that decision". An approximate attribution with
an unreported confidence interval is a weaker artifact than a slightly less
accurate model with an exact one. So an unrecognized estimator raises
:class:`UnexplainableModelError` at registration time rather than degrading
silently at audit time.

One numeric subtlety that is easy to get wrong: SHAP values are not always in the
same space as ``predict_proba``. ``LinearExplainer`` on a logistic regression and
``TreeExplainer`` on a LightGBM/XGBoost binary booster both return **log-odds**,
so additivity has to be checked against ``logit(score)``. But ``TreeExplainer`` on
a scikit-learn forest returns **probability** space, because a forest's raw output
*is* the averaged class probability — checking that one against ``logit(score)``
would report a correct explanation as broken.

So the link function is not a global constant; it is a property of the
(explainer, model family) pair, resolved by :func:`link_of` and carried on the
:class:`Explanation` itself. :func:`reconciles` reads it from there rather than
assuming.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..train.canonical import feature_names_of, final_estimator


class UnexplainableModelError(TypeError):
    """No exact explainer is defined for this model class."""


class AdditivityError(RuntimeError):
    """The attributions do not sum to the score they claim to explain.

    Raised rather than logged. An explanation that does not reconcile is not a
    weaker explanation, it is a false one — it asserts a decomposition of a
    number it does not actually decompose, and serving it would put a claim into
    the audit trail that the audit trail cannot support.
    """


@dataclass
class Attribution:
    feature_name: str
    feature_value_str: str | None
    shap_value: float
    abs_rank: int


@dataclass
class Explanation:
    base_value: float
    attributions: list[Attribution]
    explainer_type: str
    # "logit" | "identity" — the space these values live in, so that a consumer
    # can reconcile them against a probability without knowing the model family.
    link: str = "logit"

    def top(self, k: int) -> list[Attribution]:
        return self.attributions[:k]

    def residual(self, k: int) -> float:
        """Sum of the attributions outside the top ``k``.

        Returned alongside the truncated list so the explanation still reconciles
        to the score. Without it, a top-10 view of a 100-column encoded matrix
        silently fails to add up, and an explanation that does not reconcile is
        not an explanation.
        """
        return float(sum(a.shap_value for a in self.attributions[k:]))


def explainer_for(model):
    """Return ``(explainer, explainer_type)`` for a fitted pipeline or estimator."""
    import shap

    estimator = final_estimator(model)
    module = type(estimator).__module__

    if hasattr(estimator, "coef_") and hasattr(estimator, "intercept_"):
        # LinearExplainer needs the background distribution in *encoded* space,
        # which the pipeline's preprocessor defines. Handled by the caller, which
        # passes the transformed background set.
        return shap.LinearExplainer, "linear"

    if module.startswith(("lightgbm", "xgboost", "sklearn.ensemble", "sklearn.tree")):
        return shap.TreeExplainer, "tree"

    raise UnexplainableModelError(
        f"no exact SHAP explainer for {module}.{type(estimator).__name__}; "
        f"the search space is deliberately constrained to exactly-explainable "
        f"model families (see glassbox.explain.dispatch)"
    )


def link_for_module(module: str, explainer_type: str) -> str:
    """The output space of an (explainer, estimator module) pair.

    ``"logit"`` means the attributions sum to ``logit(predict_proba)``;
    ``"identity"`` means they sum to ``predict_proba`` directly. Getting this
    wrong does not produce a slightly-off check — it produces a check that fails
    on every correct explanation, so it is resolved explicitly rather than
    assumed.

    Keyed on the module *name* rather than a live estimator so that the audit
    read path, which has only the recorded ``estimator_class`` string and no
    model in memory, resolves the link through this same rule. Two copies of this
    logic would eventually disagree, and the disagreement would look like a
    corrupt audit trail rather than like a bug.
    """
    if explainer_type == "linear":
        return "logit"
    if module.startswith(("sklearn.ensemble", "sklearn.tree")):
        # A forest's raw output is the averaged class probability, so
        # TreeExplainer's values are already in probability space.
        return "identity"
    # LightGBM and XGBoost binary boosters sum raw margins: log-odds.
    return "logit"


def link_of(model, explainer_type: str) -> str:
    """:func:`link_for_module` for a live model."""
    return link_for_module(type(final_estimator(model)).__module__, explainer_type)


def encoded_feature_names(model) -> list[str]:
    return feature_names_of(model)


def logit(p: float | np.ndarray):
    """Inverse of the logistic link, with clipping so 0 and 1 do not blow up."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def expected_total(explanation: Explanation, score: float) -> float:
    """The value ``base + sum(attributions)`` must equal, in the explanation's space."""
    return float(logit(score)) if explanation.link == "logit" else float(score)


def reconciles(
    explanation: Explanation, score: float, k: int, *, tol: float = 1e-6
) -> bool:
    """Does base + top-k + residual equal the model's output for this row?

    This is the additivity property that makes a SHAP explanation an explanation
    rather than a ranked list of vaguely relevant numbers. ``k`` exists because
    the served view is truncated: the residual carries the tail, so a top-10 view
    of a 100-column encoded matrix still adds up.
    """
    total = explanation.base_value + sum(a.shap_value for a in explanation.top(k))
    total += explanation.residual(k)
    return bool(abs(total - expected_total(explanation, score)) < tol)
