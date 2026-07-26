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
