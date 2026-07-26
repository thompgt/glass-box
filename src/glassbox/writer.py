"""The single Arrow-construction path for every Iceberg write.

PyIceberg validates incoming Arrow schemas strictly. The two mismatches that bite
in practice are ``timestamp[ns]`` (what pandas produces) against Iceberg's
microsecond ``timestamptz``, and ``large_string`` (what Polars produces) against
``string``. Both surface as an opaque schema-compatibility error at commit time,
far from the code that built the table.

Rather than hand-writing an Arrow schema per table and keeping it in sync, every
write derives its Arrow schema from an Iceberg schema.

**Writes derive it from the live table, not from the :class:`TableDef`.** The
declaration in :mod:`glassbox.schemas` is a request; the table is the fact, and
for nested types the two differ. ``create_table`` reassigns field IDs depth-first
as it creates the table, so ``ListType(element_id=100)`` in the declaration
becomes element ID 10 or 11 in the table that actually exists. Building Arrow
against the declared ID produces a file schema whose list element PyIceberg then
cannot find in the requested schema, and the write dies inside a projection
visitor with ``Could not find field with id: 10`` — nowhere near the list column
that caused it. Flat tables never notice, because their top-level IDs happen to
survive renumbering unchanged.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
from pyiceberg.io.pyarrow import schema_to_pyarrow
from pyiceberg.schema import Schema

from .schemas import TableDef


def arrow_schema_for(td: TableDef) -> pa.Schema:
    """The Arrow schema for a table *as declared*.

    For reading and for constructing data before a table exists. Writes should go
    through :func:`append_records` or :func:`append_arrow`, which resolve the
    schema from the live table instead — see the module docstring.
    """
    return schema_to_pyarrow(td.schema)


def records_to_arrow(td: TableDef, records: list[dict[str, Any]]) -> pa.Table:
    """Build an Arrow table for ``td``'s declared schema from a list of row dicts.

    Missing keys become nulls, which the Iceberg schema will reject for required
    fields — deliberately, since a required field silently defaulting is how
    audit rows end up meaningless.
    """
    return _records_to_arrow(td.schema, records)


def coerce(td: TableDef, table: pa.Table) -> pa.Table:
    """Cast an existing Arrow table to ``td``'s declared schema, reordering columns."""
    return _coerce(td.schema, table)


def append_records(catalog, td: TableDef, records: list[dict[str, Any]]) -> int:
    """Append row dicts to ``td``. Returns the number of rows written."""
    if not records:
        return 0
    table = catalog.load_table(td.identifier)
    table.append(_records_to_arrow(table.schema(), records))
    return len(records)


def append_arrow(catalog, td: TableDef, data: pa.Table) -> int:
    """Append an Arrow table to ``td``, coercing it to the table's schema first."""
    if data.num_rows == 0:
        return 0
    table = catalog.load_table(td.identifier)
    table.append(_coerce(table.schema(), data))
    return data.num_rows


# ---------------------------------------------------------------- internals ----

def _records_to_arrow(schema: Schema, records: list[dict[str, Any]]) -> pa.Table:
    arrow_schema = schema_to_pyarrow(schema)
    columns = {
        f.name: pa.array([r.get(f.name) for r in records], type=f.type) for f in arrow_schema
    }
    return pa.table(columns, schema=arrow_schema)


def _coerce(schema: Schema, table: pa.Table) -> pa.Table:
    arrow_schema = schema_to_pyarrow(schema)
    columns = {}
    for f in arrow_schema:
        if f.name not in table.column_names:
            columns[f.name] = pa.nulls(table.num_rows, type=f.type)
        else:
            columns[f.name] = table[f.name].combine_chunks().cast(f.type)
    return pa.table(columns, schema=arrow_schema)
