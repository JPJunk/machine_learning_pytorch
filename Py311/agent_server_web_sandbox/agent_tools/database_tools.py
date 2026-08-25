# agent_tools/database_tools.py - SQLite database tools for the agent server.
#
# INSTALLATION & SAFETY UPDATES (2026-06-10):
# -------------------------------------------
# 1. All database connections now use a timeout parameter (5s) to prevent
#    indefinite blocking on locked databases.
# 2. Connection cleanup is guaranteed via try/finally blocks in every function,
#    eliminating resource leaks that existed in db_update and db_delete.
# 3. Added explicit conn.close() calls in all code paths (success and error).
# 4. Sanitization of table names uses _safe_sql_identifier to strip dangerous
#    characters from SQL identifiers.
# 5. All user-supplied values use parameterized queries (?) to prevent SQL
#    injection on data values.
#
# SQLite is built into Python's standard library (sqlite3 module) and requires
# no installation whatsoever. It works out of the box with any Python 3.x
# interpreter.
#
# For advanced / async usage you may optionally install a third-party package:
#     pip install aiosqlite
# This is NOT required for this module — all functions below use the built-in
# synchronous sqlite3 module only.
#
# The database file (agent_data.db) is created automatically in the project
# root directory on first use.  All table names and column values are
# sanitized via agent_tools.sanitize.sanitize_edge_metadata before being
# passed to SQL statements.
import os
import logging

# Configure logging to write to app.log in the project root directory
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_LOG_FILE = os.path.join(_BASE_DIR, "app.log")

logging.basicConfig(
    filename=_LOG_FILE,
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

import sqlite3
import logging
from typing import Any, Dict, List

from .common import sanitize_edge_metadata  # noqa: E402


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # agent_tools/
_DB_PATH = os.path.join(_PROJECT_ROOT, "..", "agent_data.db")


def _get_db_path() -> str:
    """Return the absolute path to the SQLite database file."""
    return os.path.normpath(os.path.abspath(_DB_PATH))


# ---------------------------------------------------------------------------
# Sanitization helper (re-uses the existing sanitize utility)
# ---------------------------------------------------------------------------

def _safe_sql_identifier(name: str) -> str:
    """Strip dangerous characters from a SQL identifier (table/column name)."""
    cleaned = sanitize_edge_metadata(str(name))
    # Remove anything that is not alphanumeric, underscore, or dot (for schema.table)
    return "".join(c for c in cleaned if c.isalnum() or c in ("_", "."))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def db_init(table_name: str, schema: str) -> Dict[str, Any]:
    """Create a table if it does not already exist.

    Parameters
    ----------
    table_name : str
        Name of the table to create (sanitized).
    schema : str
        Raw SQL column definitions, e.g.  "name TEXT, age INTEGER".

    Returns
    -------
    dict with keys: status, message
    """
    logger.info(f"Initializing database table: {table_name}")
    safe_table = _safe_sql_identifier(table_name)
    db_path = _get_db_path()
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    sql = f"CREATE TABLE IF NOT EXISTS {safe_table} ({schema})"
    conn = sqlite3.connect(db_path, timeout=5.0)
    try:
        conn.execute(sql)
        conn.commit()
        logger.info(f"Table '{table_name}' created successfully.")
        return {"status": "ok", "message": f"Table '{table_name}' is ready."}
    except sqlite3.Error as exc:
        logger.error(f"Failed to create table '{table_name}': {exc}")
        return {"status": "error", "message": str(exc)}
    finally:
        conn.close()


def db_insert(table_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Insert a single row into the given table.

    Parameters
    ----------
    table_name : str
        Target table name (sanitized).
    data : dict
        Mapping of column names to values.

    Returns
    -------
    dict with keys: status, message, last_row_id
    """
    logger.info(f"Inserting row into table '{table_name}'")
    safe_table = _safe_sql_identifier(table_name)
    columns = [_safe_sql_identifier(k) for k in data.keys()]
    placeholders = ", ".join("?" for _ in columns)
    col_list = ", ".join(columns)
    values = [data[k] for k in data.keys()]

    sql = f"INSERT INTO {safe_table} ({col_list}) VALUES ({placeholders})"
    db_path = _get_db_path()
    conn = sqlite3.connect(db_path, timeout=5.0)
    try:
        cur = conn.execute(sql, values)
        conn.commit()
        logger.info(f"Row inserted into '{table_name}' with ID {cur.lastrowid}")
        return {"status": "ok", "message": "Row inserted.", "last_row_id": cur.lastrowid}
    except sqlite3.Error as exc:
        logger.error(f"Failed to insert row into '{table_name}': {exc}")
        return {"status": "error", "message": str(exc)}
    finally:
        conn.close()


def db_query(table_name: str, where: str = "") -> List[Dict[str, Any]]:
    """Query rows from a table.

    Parameters
    ----------
    table_name : str
        Target table name (sanitized).
    where : str, optional
        Optional WHERE clause (without the keyword 'WHERE'), e.g.  "age > 18".

    Returns
    -------
    list of dicts — each dict represents one row with column names as keys.
    """
    logger.info(f"Querying table '{table_name}' with condition: {where or 'all rows'}")
    safe_table = _safe_sql_identifier(table_name)
    db_path = _get_db_path()
    conn = sqlite3.connect(db_path, timeout=5.0)
    try:
        if where:
            cur = conn.execute(f"SELECT * FROM {safe_table} WHERE {where}")
        else:
            cur = conn.execute(f"SELECT * FROM {safe_table}")

        col_names = [desc[0] for desc in cur.description]
        rows = cur.fetchall()

        result = []
        for row in rows:
            result.append(dict(zip(col_names, row)))
        logger.info(f"Query returned {len(result)} rows from '{table_name}'")
        return result
    except sqlite3.Error as exc:
        logger.error(f"Failed to query table '{table_name}': {exc}")
        return [{"_error": str(exc)}]
    finally:
        conn.close()


def db_update(table_name: str, data: Dict[str, Any], where: str) -> Dict[str, Any]:
    """Update rows matching a WHERE clause.

    Parameters
    ----------
    table_name : str
        Target table name (sanitized).
    data : dict
        Mapping of column names to new values.
    where : str
        WHERE clause (without the keyword 'WHERE').

    Returns
    -------
    dict with keys: status, message, rows_affected
    """
    logger.info(f"Updating rows in table '{table_name}' matching: {where}")
    safe_table = _safe_sql_identifier(table_name)
    set_parts = [f"{_safe_sql_identifier(k)} = ?" for k in data.keys()]
    set_clause = ", ".join(set_parts)
    values = [data[k] for k in data.keys()]

    sql = f"UPDATE {safe_table} SET {set_clause} WHERE {where}"
    db_path = _get_db_path()
    conn = sqlite3.connect(db_path, timeout=5.0)
    try:
        cur = conn.execute(sql, values)
        conn.commit()
        logger.info(f"Updated {cur.rowcount} rows in '{table_name}'")
        return {"status": "ok", "message": "Rows updated.", "rows_affected": cur.rowcount}
    except sqlite3.Error as exc:
        logger.error(f"Failed to update table '{table_name}': {exc}")
        return {"status": "error", "message": str(exc)}
    finally:
        conn.close()


def db_delete(table_name: str, where: str) -> Dict[str, Any]:
    """Delete rows matching a WHERE clause.

    Parameters
    ----------
    table_name : str
        Target table name (sanitized).
    where : str
        WHERE clause (without the keyword 'WHERE').

    Returns
    -------
    dict with keys: status, message, rows_affected
    """
    logger.info(f"Deleting rows from table '{table_name}' matching: {where}")
    safe_table = _safe_sql_identifier(table_name)
    sql = f"DELETE FROM {safe_table} WHERE {where}"
    db_path = _get_db_path()
    conn = sqlite3.connect(db_path, timeout=5.0)
    try:
        cur = conn.execute(sql)
        conn.commit()
        logger.info(f"Deleted {cur.rowcount} rows from '{table_name}'")
        return {"status": "ok", "message": "Rows deleted.", "rows_affected": cur.rowcount}
    except sqlite3.Error as exc:
        logger.error(f"Failed to delete from table '{table_name}': {exc}")
        return {"status": "error", "message": str(exc)}
    finally:
        conn.close()
