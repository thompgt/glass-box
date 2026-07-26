"""The single Arrow-construction path for every Iceberg write.

PyIceberg validates incoming Arrow schemas strictly. The two mismatches that bite
in practice are ``timestamp[ns]`` (what pandas produces) against Iceberg's
microsecond ``timestamptz``, and ``large_string`` (what Polars produces) against
``string``. Both surface as an opaque schema-compatibility error at commit time,
far from the code that built the table.

Rather than hand-writing an Arrow schema per table and keeping it in sync, every
write derives its Arrow schema *from* the Iceberg schema via
:func:`arrow_schema_for`. A hand-written schema can drift from the table it
targets; a derived one cannot.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
from pyiceberg.io.pyarrow import schema_to_pyarrow

from .schemas import TableDef


def arrow_schema_for(td: TableDef) -> pa.Schema:
    """The exact Arrow schema PyIceberg expects when writing to ``td``."""
    return schema_to_pyarrow(td.schema)


def records_to_arrow(td: TableDef, records: list[dict[str, Any]]) -> pa.Table:
    """Build an Arrow table for ``td`` from a list of row dicts.

    Missing keys become nulls, which the Iceberg schema will reject for required
    fields — deliberately, since a required field silently defaulting is how
    audit rows end up meaningless.
    """
    schema = arrow_schema_for(td)
    columns = {
        f.name: pa.array([r.get(f.name) for r in records], type=f.type) for f in schema
    }
    return pa.table(columns, schema=schema)


def coerce(td: TableDef, table: pa.Table) -> pa.Table:
    """Cast an existing Arrow table to ``td``'s expected schema, reordering columns."""
    schema = arrow_schema_for(td)
    columns = {}
    for f in schema:
        if f.name not in table.column_names:
            columns[f.name] = pa.nulls(table.num_rows, type=f.type)
        else:
            columns[f.name] = table[f.name].combine_chunks().cast(f.type)
    return pa.table(columns, schema=schema)


def append_records(catalog, td: TableDef, records: list[dict[str, Any]]) -> int:
    """Append row dicts to ``td``. Returns the number of rows written."""
    if not records:
        return 0
    table = catalog.load_table(td.identifier)
    table.append(records_to_arrow(td, records))
    return len(records)


def append_arrow(catalog, td: TableDef, data: pa.Table) -> int:
    """Append an Arrow table to ``td``, coercing it to the expected schema first."""
    if data.num_rows == 0:
        return 0
    table = catalog.load_table(td.identifier)
    table.append(coerce(td, data))
    return data.num_rows
