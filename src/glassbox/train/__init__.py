"""Training, model registration, and the reproducibility harness."""

from .baseline import train_baseline
from .canonical import UnsupportedModelError, model_digest
from .registry import ProvenanceIntegrityError, get_model_version, load_model_checked, register

__all__ = [
    "train_baseline",
    "model_digest",
    "UnsupportedModelError",
    "register",
    "get_model_version",
    "load_model_checked",
    "ProvenanceIntegrityError",
]
