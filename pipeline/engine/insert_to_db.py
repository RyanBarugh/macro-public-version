from __future__ import annotations

"""
insert_to_db.py
===============
Database write operations for the macro pipeline.

Single connection pattern: receives `conn` from the orchestrator,
never opens its own connection. Matches TFF/Oanda pipeline pattern.

Uses sql.Identifier for schema references (hardening).
"""

from datetime import datetime, timezone

import pandas as pd
from psycopg2 import sql
from psycopg2.extras import execute_values

from .logger import get_logger

logger = get_logger(__name__)

ALLOWED_SCHEMAS = frozenset({"macro", "equity", "prices"})


def _validate_schema(schema: str) -> None:
    if schema not in ALLOWED_SCHEMAS:
        raise ValueError(f"Schema '{schema}' not in allowed schemas: {sorted(ALLOWED_SCHEMAS)}")


def upsert_series_data(
    df: pd.DataFrame,
    conn,
    schema: str = "macro",
    insert_only: bool = False,
) -> None:
    """
    Generic upsert into {schema}.series_data.
    Works for any provider, any country, any indicator.
    Expects df columns: series_id, time, value.
    Tracks revisions via previous_value and revised_at.

    insert_only=True  : ON CONFLICT DO NOTHING — never overwrites existing rows.
                        Use for preliminary sources (e.g. METI) where a later
                        authoritative source should win.
    insert_only=False : ON CONFLICT DO UPDATE — overwrites value, saves old value
                        to previous_value / revised_at for revision tracking.
    """
    _validate_schema(schema)

    required = {"series_id", "time", "value"}
    missing = required - set(df.columns)
    if missing:
        logger.error("DB upsert aborted: missing columns=%s", sorted(missing))
        raise ValueError(f"Missing required columns for DB upsert: {sorted(missing)}")

    now = datetime.now(timezone.utc)

    base_rows = df[["series_id", "time", "value"]].itertuples(index=False, name=None)
    rows = [(sid, t, v, now) for (sid, t, v) in base_rows]

    if not rows:
        logger.info("DB upsert skipped: no rows for %s.series_data", schema)
        return

    logger.info("DB upsert start: table=%s.series_data rows=%d", schema, len(rows))

    if insert_only:
        upsert_sql = sql.SQL("""
            INSERT INTO {}.{} (series_id, time, value, updated_at)
            VALUES %s
            ON CONFLICT (series_id, time) DO NOTHING
        """).format(sql.Identifier(schema), sql.Identifier("series_data"))
    else:
        upsert_sql = sql.SQL("""
            INSERT INTO {}.{} (series_id, time, value, updated_at)
            VALUES %s
            ON CONFLICT (series_id, time)
            DO UPDATE SET
                previous_value = CASE
                    WHEN {}.{}.value IS DISTINCT FROM EXCLUDED.value
                    THEN {}.{}.value
                    ELSE {}.{}.previous_value
                END,
                revised_at = CASE
                    WHEN {}.{}.value IS DISTINCT FROM EXCLUDED.value
                    THEN EXCLUDED.updated_at
                    ELSE {}.{}.revised_at
                END,
                value      = EXCLUDED.value,
                updated_at = CASE
                    WHEN {}.{}.value IS DISTINCT FROM EXCLUDED.value
                    THEN EXCLUDED.updated_at
                    ELSE {}.{}.updated_at
                END
        """).format(
            sql.Identifier(schema), sql.Identifier("series_data"),
            sql.Identifier(schema), sql.Identifier("series_data"),
            sql.Identifier(schema), sql.Identifier("series_data"),
            sql.Identifier(schema), sql.Identifier("series_data"),
            sql.Identifier(schema), sql.Identifier("series_data"),
            sql.Identifier(schema), sql.Identifier("series_data"),
            sql.Identifier(schema), sql.Identifier("series_data"),
            sql.Identifier(schema), sql.Identifier("series_data"),
        )

    try:
        with conn.cursor() as cur:
            execute_values(cur, upsert_sql, rows, page_size=1000)
        conn.commit()
        logger.info("DB upsert complete: table=%s.series_data rows=%d", schema, len(rows))
    except Exception:
        conn.rollback()
        logger.exception("DB upsert failed: table=%s.series_data", schema)
        raise


def upsert_us_retail_sales_derived(
    df: pd.DataFrame,
    conn,
    schema: str = "macro",
) -> None:
    """
    Inserts/updates rows into {schema}.us_retail_sales_derived.
    """
    _validate_schema(schema)

    required = {
        "time",
        "total_level",
        "ex_autos_level",
        "control_level",
        "total_mom_pct",
        "total_yoy_pct",
        "ex_autos_mom_pct",
        "control_mom_pct",
        "total_mom_te",
        "ex_autos_mom_te",
        "control_mom_te",
        "total_yoy_te",
    }
    missing = required - set(df.columns)
    if missing:
        logger.error("Derived DB upsert aborted: missing columns=%s", sorted(missing))
        raise ValueError(f"Missing required columns for derived DB upsert: {sorted(missing)}")

    now = datetime.now(timezone.utc)

    cols = [
        "time",
        "total_level",
        "ex_autos_level",
        "control_level",
        "total_mom_pct",
        "total_yoy_pct",
        "ex_autos_mom_pct",
        "control_mom_pct",
        "total_mom_te",
        "ex_autos_mom_te",
        "control_mom_te",
        "total_yoy_te",
    ]

    base_rows = df[cols].itertuples(index=False, name=None)
    rows = [tuple(r) + (now,) for r in base_rows]

    if not rows:
        logger.info("Derived DB upsert skipped: no rows for %s.us_retail_sales_derived", schema)
        return

    logger.info("Derived DB upsert start: table=%s.us_retail_sales_derived rows=%d", schema, len(rows))

    upsert_sql = sql.SQL("""
        INSERT INTO {}.{} (
            time,
            total_level, ex_autos_level, control_level,
            total_mom_pct, total_yoy_pct,
            ex_autos_mom_pct, control_mom_pct,
            total_mom_te, ex_autos_mom_te, control_mom_te,
            total_yoy_te,
            updated_at
        )
        VALUES %s
        ON CONFLICT (time)
        DO UPDATE SET
            total_level = EXCLUDED.total_level,
            ex_autos_level = EXCLUDED.ex_autos_level,
            control_level = EXCLUDED.control_level,
            total_mom_pct = EXCLUDED.total_mom_pct,
            total_yoy_pct = EXCLUDED.total_yoy_pct,
            ex_autos_mom_pct = EXCLUDED.ex_autos_mom_pct,
            control_mom_pct = EXCLUDED.control_mom_pct,
            total_mom_te = EXCLUDED.total_mom_te,
            ex_autos_mom_te = EXCLUDED.ex_autos_mom_te,
            control_mom_te = EXCLUDED.control_mom_te,
            total_yoy_te = EXCLUDED.total_yoy_te,
            updated_at = EXCLUDED.updated_at
    """).format(sql.Identifier(schema), sql.Identifier("us_retail_sales_derived"))

    try:
        with conn.cursor() as cur:
            execute_values(cur, upsert_sql, rows, page_size=1000)
        conn.commit()
        logger.info("Derived DB upsert complete: table=%s.us_retail_sales_derived rows=%d", schema, len(rows))
    except Exception:
        conn.rollback()
        logger.exception("Derived DB upsert failed: table=%s.us_retail_sales_derived", schema)
        raise