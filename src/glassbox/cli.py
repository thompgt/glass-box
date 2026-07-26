"""Glass Box command line."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from .bootstrap import bootstrap
from .catalog import catalog_db_path, glassbox_root, warehouse_path

app = typer.Typer(
    name="glassbox",
    help="Provenance-complete ML audit trail.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command("init")
def init_cmd() -> None:
    """Bootstrap the Iceberg catalog, namespaces, and all audit tables."""
    root = glassbox_root()
    console.print(f"[dim]root      [/dim] {root}")
    console.print(f"[dim]warehouse [/dim] {warehouse_path(root)}")
    console.print(f"[dim]catalog   [/dim] {catalog_db_path(root)}")

    result = bootstrap(root)

    table = Table(show_header=True, header_style="bold")
    table.add_column("table")
    table.add_column("status")
    for name in result.created:
        table.add_row(name, "[green]created[/green]")
    for name in result.existing:
        table.add_row(name, "[dim]exists[/dim]")
    console.print(table)
    console.print(f"[bold]{result.total}[/bold] tables present.")


@app.command("ingest")
def ingest_cmd(
    dataset: str = typer.Argument("adult", help="Dataset to ingest: adult"),
) -> None:
    """Ingest a dataset into features.* and capture its data versions."""
    if dataset != "adult":
        raise typer.BadParameter(f"unknown dataset {dataset!r} (only 'adult' so far)")

    from .ingest import ingest_adult

    result = ingest_adult()

    console.print(
        f"[green]ingested[/green] {result.rows_written} rows  batch={result.ingest_batch_id}"
    )
    console.print(f"  splits: {result.split_counts}")

    for label, snap in (("train", result.train_snapshot), ("eval", result.eval_snapshot)):
        console.print(f"\n[bold]{label} data version[/bold]")
        console.print(f"  data_snapshot_uuid  {snap.data_snapshot_uuid}")
        console.print(f"  iceberg_snapshot_id {snap.iceberg_snapshot_id}")
        console.print(f"  rows                {snap.row_count}")
        console.print(f"  content_digest      {snap.content_digest[:32]}...")
        console.print(f"  as_of range         {snap.min_as_of_ts} .. {snap.max_as_of_ts}")


@app.command("train")
def train_cmd(
    seed: int = typer.Option(42, help="Training seed."),
) -> None:
    """Train the baseline model and register its provenance record."""
    from .train import train_baseline

    result = train_baseline(seed=seed)
    mv = result.model_version

    console.print(f"[green]trained[/green] {mv.estimator_class}")
    console.print(f"  model_version_id   {mv.model_version_id}")
    console.print(f"  data_snapshot      {mv.data_snapshot_uuid}")
    console.print(f"  artifact_digest    {mv.artifact_digest[:32]}...")
    console.print(f"  rows               {result.train_rows} train / {result.eval_rows} eval")
    for name, value in sorted(mv.metrics.items()):
        console.print(f"  {name:<18} {value:.4f}")
    console.print(f"\nServe it: [bold]glassbox serve {mv.model_version_id}[/bold]")


@app.command("reproduce")
def reproduce_cmd(
    model_version_id: str = typer.Argument(..., help="Model version to retrain and compare."),
    strict_env: bool = typer.Option(
        True, help="Refuse to compare across a changed environment."
    ),
) -> None:
    """Retrain from a recorded data version and compare the artifact digests."""
    from .train.reproduce import retrain_from_provenance

    result = retrain_from_provenance(model_version_id, strict_env=strict_env)

    verdict = "[green]reproduced[/green]" if result.reproduced else "[red]DIVERGED[/red]"
    console.print(f"{verdict} {model_version_id}")
    console.print(f"  recorded   {result.recorded_digest}")
    console.print(f"  reproduced {result.reproduced_digest}")
    console.print(
        f"  predictions match on {result.probe_rows} probe rows: {result.predictions_match}"
    )

    if not result.reproduced:
        raise typer.Exit(code=1)


@app.command("serve")
def serve_cmd(
    model_version_id: str = typer.Argument(..., help="Model version to serve."),
    host: str = typer.Option("127.0.0.1", help="Bind address."),
    port: int = typer.Option(8000, help="Bind port."),
    threshold: float = typer.Option(0.5, help="Score at or above which a decision approves."),
) -> None:
    """Serve predictions and their explanations over HTTP."""
    import uvicorn

    from .serve.app import create_app

    console.print(f"[green]serving[/green] {model_version_id} on http://{host}:{port}")
    console.print(f"  POST http://{host}:{port}/predictions")
    console.print(f"  GET  http://{host}:{port}/explanations/{{prediction_id}}")

    uvicorn.run(
        create_app(model_version_id=model_version_id, threshold=threshold),
        host=host,
        port=port,
    )


@app.command("predict")
def predict_cmd(
    model_version_id: str = typer.Argument(..., help="Model version to decide with."),
    features: str = typer.Option(..., "--features", "-f", help="Applicant features as JSON."),
    subject_id: str = typer.Option(None, help="Optional data-subject identifier."),
    threshold: float = typer.Option(0.5, help="Score at or above which a decision approves."),
) -> None:
    """Score one applicant and commit the decision to the audit trail.

    The same code path the HTTP service uses, spool included — so a decision made
    here is as explainable as one made under load, and ``glassbox explain`` reads
    it back the same way.
    """
    import json

    from .catalog import load_catalog
    from .serve.service import PredictionService

    root = glassbox_root()
    catalog = load_catalog(root)
    service = PredictionService.load(catalog, model_version_id, root, threshold=threshold)

    outcome = service.predict(json.loads(features), subject_id=subject_id)
    service.flush()

    colour = "green" if outcome.decision == "approve" else "red"
    console.print(f"[{colour}]{outcome.decision.upper()}[/{colour}]  score {outcome.score:.4f}")
    console.print(f"  prediction_id  {outcome.prediction_id}")
    console.print(f"\nExplain it: [bold]glassbox explain {outcome.prediction_id}[/bold]")


@app.command("explain")
def explain_cmd(
    prediction_id: str = typer.Argument(..., help="Prediction to explain."),
    top_k: int = typer.Option(10, help="How many attributions to show."),
) -> None:
    """Answer 'why was this decided?' for one recorded prediction."""
    from .catalog import load_catalog
    from .serve.explanations import ExplanationNotFoundError, explanation_for

    try:
        report = explanation_for(load_catalog(), prediction_id, top_k=top_k)
    except ExplanationNotFoundError:
        console.print(f"[red]no recorded prediction[/red] {prediction_id}")
        raise typer.Exit(code=1) from None

    console.print(f"[bold]{report.decision.upper()}[/bold]  score {report.score:.4f}  "
                  f"threshold {report.threshold}")
    console.print(f"  prediction_ts      {report.prediction_ts}")
    console.print(f"  model_version_id   {report.model_version_id}  ({report.recipe})")
    console.print(f"  artifact_digest    {report.artifact_digest[:32]}...")
    console.print(f"  hyperparams        {report.hyperparams}")

    low, high = report.training_as_of_range
    console.print(
        f"  trained on         {report.training_row_count} rows, {low} .. {high}"
    )

    table = Table(show_header=True, header_style="bold", title="\nattributions")
    table.add_column("#", justify="right")
    table.add_column("about you")
    table.add_column("shap", justify="right")
    table.add_column("pushed")

    for a in report.attributions:
        colour = "green" if a.shap_value > 0 else "red"
        # Dimmed when the column describes a category the subject is not in. It
        # still moved the score, so it is shown; it is just not a fact about them.
        statement = a.statement if a.applies else f"[dim]{a.statement}[/dim]"
        table.add_row(
            str(a.abs_rank),
            statement,
            f"[{colour}]{a.shap_value:+.4f}[/{colour}]",
            a.direction,
        )
    console.print(table)

    console.print(
        f"base {report.base_value:+.4f}  + shown  + residual {report.residual:+.4f}  "
        f"= score in {report.link} space   "
        + ("[green]reconciles[/green]" if report.reconciles else "[red]DOES NOT RECONCILE[/red]")
    )
    console.print(
        f"[dim]{len(report.attributions)} of {report.total_feature_count} features shown; "
        f"the residual carries the rest so the total still sums to the score.[/dim]"
    )


@app.command("flush")
def flush_cmd() -> None:
    """Drain any spooled predictions into the audit tables."""
    from .catalog import load_catalog
    from .serve.spool import Spool

    spool = Spool(root=glassbox_root())
    pending = spool.pending_count()
    result = spool.flush(load_catalog())

    console.print(
        f"[green]flushed[/green] {result.predictions_written} predictions and "
        f"{result.attributions_written} attributions from {pending} spooled "
        f"({result.envelopes_skipped} already committed)"
    )


@app.command("fairness")
def fairness_cmd(
    model_version_id: str = typer.Argument(..., help="Model version to evaluate."),
    attribute: str = typer.Option("sex", help="Sensitive attribute to group by."),
    threshold: float = typer.Option(0.5, help="Score at or above which a decision approves."),
    min_group_n: int = typer.Option(
        30, help="Groups smaller than this are measured but left out of the differences."
    ),
) -> None:
    """Measure one model version's group fairness on its frozen eval snapshot."""
    from .catalog import load_catalog
    from .fairness import evaluate_fairness

    root = glassbox_root()
    result = evaluate_fairness(
        load_catalog(root),
        model_version_id,
        sensitive_attribute=attribute,
        threshold=threshold,
        min_group_n=min_group_n,
        root=root,
    )

    console.print(f"[bold]{model_version_id}[/bold]  grouped by {attribute}")
    console.print(f"  eval snapshot      {result.eval_snapshot_uuid}")
    console.print(f"  decision threshold {result.decision_threshold}")

    table = Table(show_header=True, header_style="bold", title="\nby group")
    table.add_column(attribute)
    table.add_column("n", justify="right")
    table.add_column("selected", justify="right")
    table.add_column("base rate", justify="right")
    table.add_column("TPR", justify="right")
    table.add_column("FPR", justify="right")

    for group in result.groups:
        # Dimmed when the group is below the cutoff: its rates are real, but they
        # are not evidence, and they took no part in the differences below.
        style = "" if group.n >= result.min_group_n else "dim "
        table.add_row(
            f"[{style}]{group.group_value}[/{style}]" if style else group.group_value,
            str(group.n),
            f"{group.selection_rate:.3f}",
            f"{group.positive_label_rate:.3f}",
            _rate(group.true_positive_rate),
            _rate(group.false_positive_rate),
        )
    console.print(table)

    if not result.differences:
        console.print(
            f"[yellow]no across-group difference is defined[/yellow] — fewer than two "
            f"groups reached min_group_n={result.min_group_n} with a defined rate. "
            f"Quoting one anyway would be quoting sampling noise."
        )
        return

    console.print(f"\n[bold]across groups[/bold]  (over {result.comparable_n} subjects)")
    for name, value in sorted(result.differences.items()):
        console.print(f"  {name:<34} {value:+.4f}")


@app.command("compare")
def compare_cmd(
    baseline: str = typer.Argument(..., help="Model version to compare against."),
    candidate: str = typer.Argument(..., help="Model version under scrutiny."),
    attribute: str = typer.Option("sex", help="Sensitive attribute to group by."),
    threshold: float = typer.Option(0.5, help="Score at or above which a decision approves."),
    min_group_n: int = typer.Option(30, help="Minimum group size for a difference metric."),
) -> None:
    """Attribute a fairness change between two model versions to its cause.

    Trains the two counterfactuals that hold one factor fixed, so the answer is
    "the data did it" or "the model did it" rather than "something did it".
    """
    from .catalog import load_catalog
    from .fairness import decompose_regression

    root = glassbox_root()
    console.print("[dim]training any counterfactual not already on record...[/dim]")
    analysis = decompose_regression(
        load_catalog(root),
        baseline,
        candidate,
        sensitive_attribute=attribute,
        threshold=threshold,
        min_group_n=min_group_n,
        root=root,
    )

    console.print(f"[bold]baseline [/bold] A  {analysis.baseline_model_version_id}")
    console.print(f"[bold]candidate[/bold] B  {analysis.candidate_model_version_id}")
    console.print(f"  A' (A's config, B's data)  {analysis.counterfactual_a_prime_id}")
    console.print(f"  B' (B's config, A's data)  {analysis.counterfactual_b_prime_id}")
    console.print(f"  {analysis.trained_counterfactuals} counterfactual(s) trained")

    if not analysis.decompositions:
        console.print(
            f"\n[yellow]nothing to decompose[/yellow] — no metric is defined for all "
            f"four models at min_group_n={analysis.min_group_n}."
        )
        return

    table = Table(show_header=True, header_style="bold", title="\nattributed change")
    table.add_column("metric")
    table.add_column("A", justify="right")
    table.add_column("B", justify="right")
    table.add_column("total", justify="right")
    table.add_column("model", justify="right")
    table.add_column("data", justify="right")
    table.add_column("blame")

    for d in analysis.decompositions:
        colour = "red" if d.regressed else "green"
        table.add_row(
            d.metric_name.replace("_difference", ""),
            f"{d.measured['A']:.4f}",
            f"{d.measured['B']:.4f}",
            f"[{colour}]{d.total_delta:+.4f}[/{colour}]",
            f"{d.model_effect:+.4f}",
            f"{d.data_effect:+.4f}",
            d.attributed_to,
        )
    console.print(table)
    console.print(
        "[dim]model + data = total, exactly. A positive total is a widening gap, "
        "because the differences are unsigned max-min spreads.[/dim]"
    )


@app.command("contamination")
def contamination_cmd(
    subject_id: str = typer.Argument(..., help="Data subject to trace."),
) -> None:
    """Every model whose training snapshot contained this subject."""
    from .catalog import load_catalog
    from .erasure import contamination_report

    report = contamination_report(load_catalog(), subject_id)

    if not report.known:
        console.print(f"[green]not in the trail[/green] {subject_id}")
        return

    console.print(f"[bold]{subject_id}[/bold]")
    console.print(f"  live feature rows  {report.live_row_count}  {report.live_rows}")
    console.print(f"  decisions served   {len(report.decisions)}")

    table = Table(show_header=True, header_style="bold", title="\ncontaminated models")
    table.add_column("model_version_id")
    table.add_column("role")
    table.add_column("status")
    table.add_column("reproducible")

    for c in report.contaminated:
        status = c.status if c.registered else "[yellow]never committed[/yellow]"
        table.add_row(
            c.model_version_id,
            c.role,
            status,
            "yes" if c.reproducible else "[dim]no — data erased[/dim]",
        )
    console.print(table)

    if report.orphaned:
        console.print(
            "[dim]'never committed' means training died between writing membership "
            "and writing the model version. That over-reports on purpose.[/dim]"
        )
    if report.serving:
        console.print(
            f"\n[yellow]{len(report.serving)} contaminated model(s) are still "
            f"active.[/yellow] Whether they must be retired is a policy decision."
        )


@app.command("erase")
def erase_cmd(
    subject_id: str = typer.Argument(..., help="Data subject to erase."),
    retire: bool = typer.Option(False, help="Also retire every contaminated active model."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Erase a subject: live rows, snapshot history, and the Parquet on disk.

    Irreversible, and deliberately so — a reversible erasure is not an erasure.
    Every model trained on this subject stops being reproducible, permanently.
    """
    from .catalog import load_catalog
    from .erasure import contamination_report, erase_subject

    root = glassbox_root()
    catalog = load_catalog(root)
    report = contamination_report(catalog, subject_id)

    console.print(f"[bold]{subject_id}[/bold]")
    console.print(f"  live feature rows   {report.live_row_count}")
    console.print(f"  contaminated models {len(report.model_version_ids)}")
    console.print(
        "[red]This destroys the training snapshots those models were built from. "
        "They stay fully attributable and become permanently unreproducible.[/red]"
    )
    if not yes:
        typer.confirm("Proceed?", abort=True)

    result = erase_subject(catalog, subject_id, retire_contaminated=retire, root=root)

    console.print(f"\n[green]{result.status}[/green]  request {result.request_id}")
    console.print(f"  live rows deleted       {result.live_rows_deleted}")
    console.print(f"  data versions tombstoned {len(result.snapshots_tombstoned)}")
    console.print(f"  iceberg snapshots expired {result.iceberg_snapshots_expired}")
    console.print(f"  parquet files unlinked  {result.data_files_unlinked}")
    console.print(f"  decompositions voided   {result.decompositions_invalidated}")
    console.print(f"  models retired          {len(result.models_retired)}")
    console.print(
        f"\n[dim]audit.training_membership still records that {subject_id} trained "
        f"these models. That is the point: 'which models saw me?' has to stay "
        f"answerable after the data is gone.[/dim]"
    )


def _rate(value: float) -> str:
    """An undefined rate is shown as absent, never as zero."""
    import math

    return "[dim]n/a[/dim]" if math.isnan(value) else f"{value:.3f}"


@app.command("tables")
def tables_cmd() -> None:
    """List tables currently in the catalog with their row counts."""
    from .catalog import load_catalog
    from .schemas import ALL_TABLES

    catalog = load_catalog()
    table = Table(show_header=True, header_style="bold")
    table.add_column("table")
    table.add_column("rows", justify="right")
    table.add_column("snapshots", justify="right")

    for td in ALL_TABLES:
        try:
            t = catalog.load_table(td.identifier)
        except Exception:
            table.add_row(td.name, "[red]missing[/red]", "-")
            continue
        rows = t.scan().to_arrow().num_rows
        table.add_row(td.name, str(rows), str(len(list(t.snapshots()))))

    console.print(table)


if __name__ == "__main__":
    app()
