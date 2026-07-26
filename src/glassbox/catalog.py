"""Iceberg catalog bootstrap.

Everything is local: a SQLite catalog DB and a filesystem warehouse, both under
a single root directory. The root is resolved once, here, so no other module has
to know where state lives.

We construct :class:`SqlCatalog` directly rather than going through
``pyiceberg.catalog.load_catalog`` + a ``.pyiceberg.yaml`` file. Config-file
discovery searches the home directory and environment, which makes "clone the
repo and run it" depend on machine state that is not in the repo. Passing
properties explicitly keeps the whole configuration in version control.

**Windows note (measured in Phase 0, see docs/pyiceberg-notes.md).** PyIceberg's
default ``PyArrowFileIO`` parses a warehouse location by URI scheme, and on
Windows the drive letter in ``C:/Users/...`` is read *as* the scheme — producing
either ``Unrecognized filesystem type in URI: c`` or a mangled ``/C:/Users/...``
path that fails with ``WinError 123``. Every form of a local absolute path fails
this way. ``FsspecFileIO`` handles ``file://`` URIs correctly on Windows, so it is
pinned here rather than left to inference.
"""

from __future__ import annotations

import os
from pathlib import Path

from pyiceberg.catalog.sql import SqlCatalog

CATALOG_NAME = "glassbox"

# See the module docstring: PyArrowFileIO cannot address local files on Windows.
FILE_IO_IMPL = "pyiceberg.io.fsspec.FsspecFileIO"

FEATURES_NS = "features"
AUDIT_NS = "audit"
NAMESPACES = (FEATURES_NS, AUDIT_NS)


def glassbox_root() -> Path:
    """Directory holding all local state (warehouse, catalog DB, spool, mlruns).

    Override with ``GLASSBOX_ROOT``; tests point this at a tmp_path so they never
    touch the developer's warehouse.
    """
    env = os.environ.get("GLASSBOX_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def warehouse_path(root: Path | None = None) -> Path:
    return (root or glassbox_root()) / "warehouse"


def catalog_db_path(root: Path | None = None) -> Path:
    return (root or glassbox_root()) / "catalog.db"


def warehouse_uri(root: Path | None = None) -> str:
    """A ``file://`` URI for the warehouse.

    ``Path.as_uri()`` is used rather than string concatenation because on Windows
    the drive letter and backslashes need real URI encoding
    (``file:///C:/Users/...``), and hand-built URIs silently produce a warehouse
    directory named ``C:`` relative to the CWD.
    """
    return warehouse_path(root).as_uri()


def load_catalog(root: Path | None = None) -> SqlCatalog:
    """Open (creating if needed) the local SQLite-backed Iceberg catalog."""
    root = root or glassbox_root()
    warehouse_path(root).mkdir(parents=True, exist_ok=True)

    return SqlCatalog(
        CATALOG_NAME,
        **{
            "uri": f"sqlite:///{catalog_db_path(root).as_posix()}",
            "warehouse": warehouse_uri(root),
            "py-io-impl": FILE_IO_IMPL,
        },
    )


def ensure_namespaces(catalog: SqlCatalog) -> None:
    existing = {".".join(ns) for ns in catalog.list_namespaces()}
    for ns in NAMESPACES:
        if ns not in existing:
            catalog.create_namespace(ns)
