"""Canonical model serialization — the basis of the reproducibility claim.

**Never hash the pickle.** ``pickle``/``joblib`` bytes embed memo ordering,
protocol version, and fully-qualified module paths, so a NumPy or scikit-learn
patch release changes them while the model is unchanged. A digest that flips for
reasons unrelated to the thing being digested is worse than no digest: it turns a
provenance claim into noise, and the first time it flaps you stop trusting it.

What we hash instead is a canonical description of the *decision function* — the
learned parameters plus everything that maps a raw feature vector onto them. For
a linear model that is the coefficients, the intercept, the class order, and the
encoder's feature names, all in a fixed order. Two models with the same canonical
bytes make the same prediction for every input, which is the property the audit
trail actually needs.

Adding an estimator family means adding a handler here. There is no generic
fallback on purpose: silently digesting a model we do not understand would
produce a digest that means nothing, and :class:`UnsupportedModelError` at
training time is far cheaper than discovering that during an audit.
"""

from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline

from ..digest import canonical_json, sha256_hex


class UnsupportedModelError(TypeError):
    """Raised when no canonical serialization is defined for a model class."""


def _f64(array) -> str:
    """Deterministic hex encoding of a float array.

    ``tobytes()`` on an explicitly float64 C-ordered array is byte-stable for
    identical values across platforms, unlike ``repr()`` which has gone through
    formatting changes.
    """
    arr = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    return arr.tobytes().hex()


def _linear_payload(estimator, feature_names: list[str]) -> dict:
    return {
        "kind": "linear",
        "class": type(estimator).__name__,
        "feature_names": feature_names,
        "classes": [str(c) for c in getattr(estimator, "classes_", [])],
        "coef": _f64(estimator.coef_),
        "coef_shape": list(np.shape(estimator.coef_)),
        "intercept": _f64(estimator.intercept_),
    }


def _tree_payload(estimator) -> dict:
    """LightGBM / XGBoost canonical dumps.

    Both expose a text or binary model dump that is stable for a given trained
    model. Used from Phase 4 onward; defined here so the dispatch table has one
    home.
    """
    module = type(estimator).__module__

    if module.startswith("lightgbm"):
        booster = getattr(estimator, "booster_", None) or estimator
        return {
            "kind": "lightgbm",
            "class": type(estimator).__name__,
            "dump": booster.model_to_string(),
        }

    if module.startswith("xgboost"):
        booster = estimator.get_booster()
        return {
            "kind": "xgboost",
            "class": type(estimator).__name__,
            # UBJSON rather than the pickle path; save_raw is the documented
            # stable serialization.
            "dump": bytes(booster.save_raw("ubj")).hex(),
        }

    raise UnsupportedModelError(
        f"no canonical dump defined for {module}.{type(estimator).__name__}"
    )


def feature_names_of(model) -> list[str]:
    """Output feature names of a fitted pipeline's preprocessing stage."""
    if isinstance(model, Pipeline) and "preprocess" in model.named_steps:
        return [str(n) for n in model.named_steps["preprocess"].get_feature_names_out()]
    names = getattr(model, "feature_names_in_", None)
    return [str(n) for n in names] if names is not None else []


def final_estimator(model):
    return model.steps[-1][1] if isinstance(model, Pipeline) else model


def canonical_payload(model) -> dict:
    """A JSON-serializable, order-stable description of the decision function."""
    estimator = final_estimator(model)
    names = feature_names_of(model)

    if hasattr(estimator, "coef_") and hasattr(estimator, "intercept_"):
        return _linear_payload(estimator, names)

    payload = _tree_payload(estimator)
    payload["feature_names"] = names
    return payload


def canonical_bytes(model) -> bytes:
    return canonical_json(canonical_payload(model)).encode("utf-8")


def model_digest(model) -> str:
    """SHA-256 of the canonical dump. This is what ``artifact_digest`` records."""
    return sha256_hex(canonical_bytes(model))
