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
| 1 | **Bit-exact reproducibility** — retrain from a recorded data version, assert an identical model artifact | **working** |
| 2 | **Right-to-explanation endpoint** — prediction ID → decision, signed attributions, model version, hyperparams, training-snapshot row count and date range | **working** |
| 3 | **Fairness regression detection** across model versions, attributing any regression to the model change or the data change | **working** |
| 4 | **Erasure contamination report** — subject → every model whose training snapshot contained them | **working** |

---

## Status

**All four capabilities work end to end**, on the baseline logistic-regression
recipe. `glassbox init → ingest → train → predict → explain` runs locally,
`glassbox reproduce <model-version>` retrains from the recorded data version and
compares digests, `glassbox compare A B` attributes a fairness change to its
cause, and `glassbox erase <subject>` removes a subject while keeping the record
that they were once there.

Phase 0's PyIceberg measurements still hold — 13/13 probes.

Run the probes yourself: `python scripts/phase0_spike.py` regenerates
[`docs/pyiceberg-notes.md`](docs/pyiceberg-notes.md).

Four findings that shaped the code:

| Finding | Consequence |
|---|---|
| PyIceberg's default `PyArrowFileIO` **cannot address local files on Windows** — the `C:` drive letter is parsed as a URI scheme | `FsspecFileIO` is pinned explicitly in `catalog.py` |
| `expire_snapshots` removes **metadata only**; orphaned Parquet files stay on disk | GDPR erasure cannot rely on it alone and must unlink data files itself |
| A single-row delete rewrote **1 of 4** data files on a `bucket(4)` table | Confirms bucketing `features.*` on `subject_id` bounds erasure cost |
| `timestamp[ns]` (pandas' default) and `large_string` (Polars' default) are both **rejected** on write | Every write derives its Arrow schema from the **live table's** Iceberg schema (`writer.py`) rather than hand-rolling one — or trusting the declaration, whose nested field IDs `create_table` renumbers |

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
```

## Usage

```bash
glassbox init                      # catalog, namespaces, and all 11 audit tables
glassbox ingest adult              # load the dataset, capture its data versions
glassbox train                     # fit, register, materialize training membership
glassbox predict <model-version> \
  --features '{"age": 39, "workclass": "Private", ...}'
glassbox explain <prediction-id>   # why that decision was made
```

`glassbox serve <model-version>` exposes the same two operations over HTTP:

```
POST /predictions                  -> {"prediction_id": ..., "decision": ...}
GET  /explanations/{prediction_id} -> the full audit answer
```

The explanation is assembled from `audit.*` alone — the read path has no access
to the model. That is the point: an attribution recomputed from a reloaded model
is a statement about the model as it is now, while the question being asked is
about the decision as it was made.

Fairness and erasure:

```bash
glassbox fairness <model-version> --attribute sex   # group metrics on the frozen eval set
glassbox compare <baseline> <candidate>             # attribute a change to model or data
glassbox contamination <subject-id>                 # which models trained on this person
glassbox erase <subject-id>                         # remove them; irreversibly
```

Other commands: `glassbox reproduce <model-version>` (retrain and compare
digests), `glassbox flush` (drain spooled predictions into the audit tables),
`glassbox tables` (row counts).

### What an explanation contains

```
DENY  score 0.3616  threshold 0.5
  model_version_id   bef1a742-...  (baseline-logreg)
  artifact_digest    0552e17ba0061e65...
  hyperparams        {'C': 1.0, 'max_iter': 1000, 'solver': 'lbfgs', 'tol': 1e-06}
  trained on         216 rows, 2024-01-10 .. 2025-12-26

  # | about you              |    shap | pushed
  1 | occupation = Tech-supp | -0.2755 | toward denial
  2 | sex = Male             | +0.2026 | toward approval
  3 | sex is not Female      | +0.2026 | toward approval
  ...
base -0.6725 + shown + residual +0.2425 = score in logit space   reconciles
```

Two details that are load-bearing rather than cosmetic:

- **The residual.** The list is truncated, so the tail is carried explicitly and
  the total still sums to the score. A top-5 view of a 33-column encoded matrix
  that silently fails to add up is not an explanation.
- **"sex is not Female".** One-hot columns for categories the subject is *not*
  in still carry attribution — a zero differs from the training population's
  average, and that difference moves the score. Dropping them would break
  additivity; labelling them `sex_Female = Male` would assert something false.

### What a fairness comparison says

```
baseline  A  8ee8d824-...      candidate B  e59a9a8e-...
  A' (A's config, B's data)  e59a9a8e-...
  B' (B's config, A's data)  8ee8d824-...
  0 counterfactual(s) trained

  metric              A        B      total     model     data    blame
  demographic_parity  0.0309   0.3505  +0.3196   +0.0000  +0.3196   data
  equal_opportunity   0.0500   0.3250  +0.2750   +0.0000  +0.2750   data
  equalized_odds      0.0500   0.3684  +0.3184   +0.0000  +0.3184   data
  accuracy            0.2372   0.3300  +0.0928   +0.0000  +0.0928   data
```

Retraining the same recipe on a batch of new rows where `sex` determines the
label: demographic parity widens elevenfold, and the decomposition puts every
bit of it on the data and none on the model, which is the right answer and a
checkable one. `model + data = total` holds exactly — asserted on every
decomposition, not assumed.

Getting there requires training the two counterfactuals that hold one factor
fixed: `A'` is the baseline's configuration on the candidate's data, `B'` the
reverse. The effects are the average of both ways round the square, because along
a single path any interaction between the two changes is attributed entirely to
whichever was applied second, and there is no principled reason to prefer an
order. The symmetric average is the two-player Shapley value.

Both counterfactuals are registered model versions with `status='counterfactual'`,
so the decomposition cites four model IDs and anyone can reload each and re-check
the arithmetic. They are reused when they already exist — sound only because
capability 1 holds, since bit-exact training means a matching row *is* the
counterfactual rather than merely resembling it. In the run above only the data
changed, so `A'` is `B`, `B'` is `A`, and nothing was trained.

### What erasure destroys, and what it must not

`expire_snapshots` is metadata-only — Phase 0 measured `files_removed=False` — and
a PyIceberg delete is copy-on-write, so the rows leave the current snapshot while
every byte stays on disk and one time-travel scan away. Erasure therefore deletes
the live rows, expires every snapshot that still contains them, **and unlinks the
Parquet files no surviving snapshot references.** The test for this opens every
file in the feature warehouse and looks; without the unlink step it fails while
every table-level assertion still passes.

What survives is `audit.training_membership`. That is not a leak, it is the
capability: "which models saw me?" has to stay answerable *after* the data is
gone, which is precisely when it is asked. The affected models stay fully
attributable — recipe, hyperparameters, digests, the tombstone recording what
their training data contained — and become permanently unreproducible. Provenance
outlives erasure; reproducibility does not, and the system says which.

Two boundaries worth stating plainly:

- **Contaminated models are not retired automatically.** Whether a model trained
  on erased data must leave service is a legal judgement that differs by
  jurisdiction. It is reported; `--retire` applies it when someone has decided.
- **Decision records are not deleted.** `audit.predictions` holds decisions served
  about the subject, including their inputs. Erasing those would destroy the
  evidence a subject needs to contest a decision made about them, so the report
  lists them and leaves the call to an operator.

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
