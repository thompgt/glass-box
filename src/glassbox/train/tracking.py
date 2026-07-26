"""Local MLflow configuration.

MLflow is a blob store and an experiment scratchpad here, not a system of record.
Its tracking DB is mutable and its registry stage transitions are destructive, so
nothing that has to be provable is allowed to live only in MLflow — see
``audit.model_versions`` for the authoritative copy of everything below.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

import mlflow

from ..catalog import glassbox_root

EXPERIMENT = "glassbox"


def configure_mlflow(root: Path | None = None) -> str:
    """Point MLflow at the local SQLite store and return the experiment id."""
    root = root or glassbox_root()
    tracking_db = root / "mlflow.db"
    artifacts = root / "mlruns"
    artifacts.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"sqlite:///{tracking_db.as_posix()}")

    client = mlflow.tracking.MlflowClient()
    existing = client.get_experiment_by_name(EXPERIMENT)
    if existing is None:
        return client.create_experiment(EXPERIMENT, artifact_location=artifacts.as_uri())
    return existing.experiment_id


@lru_cache(maxsize=1)
def code_git_sha() -> str:
    """Current commit, suffixed ``-dirty`` when the tree has uncommitted changes.

    The suffix matters: a model trained from an uncommitted working tree is not
    reproducible from the repository, and recording a bare SHA would claim
    otherwise.
    """
    root = glassbox_root()
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"
