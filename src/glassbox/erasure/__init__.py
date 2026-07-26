"""Capability #4: erasure, and the provenance that has to survive it.

Two halves that answer different questions:

:mod:`.report`
    "Which models saw this person?" Read-only, and answerable *after* their data
    is gone, because ``audit.training_membership`` is materialized at training
    time rather than reconstructed from snapshots that erasure destroys.

:mod:`.execute`
    Actually removing them — live rows, snapshot history, and the Parquet on
    disk — while writing down what was destroyed before destroying it.
"""

from __future__ import annotations

from .report import (
    Contamination,
    ContaminationReport,
    contamination_report,
    decisions_about,
    live_rows_for,
    membership_of,
)

__all__ = [
    "Contamination",
    "ContaminationReport",
    "contamination_report",
    "decisions_about",
    "live_rows_for",
    "membership_of",
]
