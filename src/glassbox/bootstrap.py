"""Create the catalog, namespaces, and every table declared in :mod:`schemas`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pyiceberg.exceptions import TableAlreadyExistsError

from .catalog import ensure_namespaces, load_catalog
from .schemas import ALL_TABLES, TableDef


@dataclass
class BootstrapResult:
    created: list[str]
    existing: list[str]

    @property
    def total(self) -> int:
        return len(self.created) + len(self.existing)


def create_table(catalog, td: TableDef) -> bool:
    """Create one table. Returns True if created, False if it already existed."""
    try:
        catalog.create_table(
            identifier=td.identifier,
            schema=td.schema,
            partition_spec=td.partition_spec,
            sort_order=td.sort_order,
            properties=dict(td.properties),
        )
        return True
    except TableAlreadyExistsError:
        return False


def bootstrap(root: Path | None = None) -> BootstrapResult:
    """Idempotent. Safe to run against an existing warehouse."""
    catalog = load_catalog(root)
    ensure_namespaces(catalog)

    created, existing = [], []
    for td in ALL_TABLES:
        (created if create_table(catalog, td) else existing).append(td.name)

    return BootstrapResult(created=created, existing=existing)
