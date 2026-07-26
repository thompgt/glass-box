# Glass Box

**A provenance-complete ML audit trail.** Every prediction traces back to the
exact model version, the exact training-data version, and the per-feature
attribution behind that specific decision.

Domain is credit scoring (UCI Adult Census Income), chosen because fairness and
right-to-explanation obligations are real there — which makes the provenance
layer load-bearing rather than decorative.

Runs entirely locally: SQLite Iceberg catalog, local-filesystem warehouse, local
MLflow. No cloud dependencies.

---

## The invariant

> Iceberg `audit.*` is the sole system of record for **provenance claims**.
> MLflow is a content-addressed blob store and an experiment scratchpad.
> A provenance claim is true **iff** an Iceberg row asserts it **and** the
> referenced MLflow artifact's SHA-256 matches the digest recorded in that row.

If they disagree, the serving path returns **503
`PROVENANCE_INTEGRITY_FAILURE`**. It will not serve a prediction it cannot
attribute.

---

## The four capabilities

| # | Capability | Status |
|---|---|---|
| 1 | **Bit-exact reproducibility** — retrain from a recorded data version, assert an identical model artifact | planned |
| 2 | **Right-to-explanation endpoint** — prediction ID → decision, signed attributions, model version, hyperparams, training-snapshot row count and date range | planned |
| 3 | **Fairness regression detection** across model versions, attributing any regression to the model change or the data change | planned |
| 4 | **Erasure contamination report** — subject → every model whose training snapshot contained them | planned |

---

## Status

Phase 0 (Iceberg literacy spike). Nothing works end to end yet.

See [`docs/`](docs/) for design notes and [`docs/pyiceberg-notes.md`](docs/pyiceberg-notes.md)
for measured PyIceberg write-side behaviour on the pinned version.

---

## Setup

Requires **Python 3.12** (not 3.13 — PyIceberg/SHAP/FLAML wheel coverage).

```bash
py -3.12 -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install -e ".[dev]"
```

Optional extras are installed per phase to keep early resolution fast:

```bash
pip install -e ".[dev,explain]"    # Phase 3 onward — SHAP
pip install -e ".[dev,automl]"     # Phase 4 onward — FLAML, LightGBM, XGBoost
pip install -e ".[dev,fairness]"   # Phase 5 onward — fairlearn
```

## Usage

```bash
glassbox init          # bootstrap catalog, namespaces, and all audit tables
```

More commands land as phases complete.

---

## Design decisions

Recorded as they are made, each with the alternative that was rejected:

- **Python 3.12 over 3.13** — dependency wheel coverage, not a language need.
- **Adult Census over German Credit as the primary dataset** — German Credit's
  1,000 rows make a fairness regression statistically indistinguishable from
  sampling noise, and you cannot attribute noise to a cause. German Credit is
  retained as a second dataset to prove the pipeline is dataset-agnostic.

## License

MIT
