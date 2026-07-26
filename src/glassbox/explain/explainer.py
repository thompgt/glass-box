"""Computing per-prediction attributions.

Attributions are computed **synchronously at inference time**, not reconstructed
later from replayed inputs. The distinction matters for the audit claim: an
attribution recomputed afterwards is a statement about a model, whereas the thing
a right-to-explanation request asks for is a statement about *that decision*. On
a linear or tree model over ~100 encoded columns this costs single-digit
milliseconds, so there is no reason to accept the weaker artifact.

The expensive part of the write path is not SHAP — it is the Iceberg commit. See
:mod:`glassbox.serve.spool`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..train.canonical import final_estimator
from ..train.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS
from .dispatch import (
    AdditivityError,
    Attribution,
    Explanation,
    expected_total,
    explainer_for,
    link_of,
)

# Loose enough to absorb float64 accumulation over ~100 encoded columns, tight
# enough that a genuine space or link mismatch (which is O(1), not O(1e-9)) is
# caught rather than tolerated.
ADDITIVITY_TOL = 1e-6


class PredictionExplainer:
    """A fitted model plus its explainer, held together.

    Built once per loaded model version and reused across requests: constructing
    a SHAP explainer re-derives the background distribution, which is far more
    expensive than explaining a single row.
    """

    def __init__(self, model, background: pd.DataFrame):
        self.model = model
        self.preprocess = model.named_steps["preprocess"]
        self.estimator = final_estimator(model)

        explainer_cls, self.explainer_type = explainer_for(model)
        self.link = link_of(model, self.explainer_type)
        encoded_background = self.preprocess.transform(background)

        if self.explainer_type == "linear":
            # LinearExplainer takes (coef, intercept) with a masker built from the
            # encoded background — the model here is the *final* estimator, since
            # the pipeline's preprocessing has already been applied.
            self.explainer = explainer_cls(self.estimator, encoded_background)
        else:
            self.explainer = explainer_cls(self.estimator, encoded_background)

        self.feature_names = list(self.preprocess.get_feature_names_out())

    def explain(self, row: pd.DataFrame, score: float | None = None) -> Explanation:
        """Explain a single-row frame of raw (unencoded) features.

        ``score`` is the model's positive-class probability for this row. Pass it
        when the caller has already computed it — the serving path has — to avoid
        a second ``predict_proba``; it is only used to verify additivity.
        """
        encoded = self.preprocess.transform(row)
        values, base = self._shap_values(encoded)

        raw = row.iloc[0].to_dict()
        # Stable ordering: ties on |shap| resolve by encoded column position
        # rather than by whatever argsort happens to do, so two explanations of
        # the same row always assign the same abs_rank to the same feature.
        order = sorted(range(len(values)), key=lambda i: (-abs(values[i]), i))

        attributions = [
            Attribution(
                feature_name=self.feature_names[i],
                feature_value_str=_source_value(self.feature_names[i], raw),
                shap_value=float(values[i]),
                abs_rank=rank + 1,
            )
            for rank, i in enumerate(order)
        ]
        explanation = Explanation(
            base_value=float(base),
            attributions=attributions,
            explainer_type=self.explainer_type,
            link=self.link,
        )

        if score is None:
            score = float(self.model.predict_proba(row)[0, 1])
        self._assert_additive(explanation, score)
        return explanation

    def _assert_additive(self, explanation: Explanation, score: float) -> None:
        """Fail closed if the attributions do not sum to the score.

        Checked on every explanation rather than once in a test. The failure this
        guards against is not a coding slip that a test would have caught — it is
        a SHAP version or model family whose output space differs from what
        :func:`link_of` declares, which appears only at runtime and would
        otherwise write a decomposition that does not decompose anything into the
        audit trail.
        """
        total = explanation.base_value + sum(a.shap_value for a in explanation.attributions)
        want = expected_total(explanation, score)
        if abs(total - want) >= ADDITIVITY_TOL:
            raise AdditivityError(
                f"attributions do not reconcile: base + sum(shap) = {total!r} but the "
                f"model scored {score!r} ({explanation.link} space -> {want!r}); "
                f"explainer_type={explanation.explainer_type}. Refusing to record an "
                f"explanation that does not explain the score."
            )

    def _shap_values(self, encoded) -> tuple[np.ndarray, float]:
        """Normalize SHAP's several output shapes to (n_features,) for class 1.

        SHAP returns a list of two arrays, a 2-D array, or a 3-D array depending
        on version and model family. Normalizing here means the persistence layer
        and the additivity check only ever see one shape.
        """
        out = self.explainer(encoded) if callable(self.explainer) else None
        if out is not None and hasattr(out, "values"):
            values = np.asarray(out.values)
            base = np.asarray(out.base_values)
        else:
            values = np.asarray(self.explainer.shap_values(encoded))
            base = np.asarray(self.explainer.expected_value)

        values = np.squeeze(values)
        if values.ndim == 2:
            # (n_features, n_classes) for a binary tree model -> positive class
            values = values[:, -1]
        base = np.squeeze(base)
        if base.ndim >= 1:
            base = base.reshape(-1)[-1]

        return values.astype(np.float64), float(base)


def _source_value(encoded_name: str, raw: dict) -> str | None:
    """Map an encoded column name back to the raw feature value it came from.

    ``ColumnTransformer`` emits names like ``cat__occupation_Tech-support`` and
    ``num__age``. The audit table stores the *source* value, because
    "occupation was Tech-support" is what a data subject can act on, while
    "cat__occupation_Tech-support = 1.0" is not.
    """
    name = encoded_name.split("__", 1)[-1]

    if name in FEATURE_COLUMNS:
        value = raw.get(name)
        return None if value is None else str(value)

    for col in CATEGORICAL_FEATURES:
        if name.startswith(f"{col}_"):
            return None if raw.get(col) is None else str(raw[col])

    return None


def build_background(train_frame: pd.DataFrame, n: int = 200, seed: int = 0) -> pd.DataFrame:
    """A deterministic background sample for the explainer's baseline.

    Sampled with a fixed seed and taken from the *training* rows the model
    actually saw, so that two loads of the same model version produce the same
    base value — otherwise every restart would silently change the attributions
    the audit trail reports.
    """
    if len(train_frame) <= n:
        return train_frame.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(train_frame), size=n, replace=False))
    return train_frame.iloc[idx].reset_index(drop=True)
