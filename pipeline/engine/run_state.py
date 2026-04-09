from __future__ import annotations

"""
run_state.py
============
Audit trail for pipeline runs: STARTED → SUCCESS / FAILED / SKIPPED.

Single connection pattern: receives `conn` from the orchestrator.
Matches TFF/Oanda pipeline pattern exactly.

Supports multiple pipelines via RunStateConfig:
    - macro: macro_run_state in macro schema
    - equity: equity_run_state in equity schema
"""

from dataclasses import dataclass

from psycopg2 import sql

from .logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RunStateConfig:
    pipeline_name: str = "macro"
    table_name: str    = "macro_run_state"


ALLOWED_SCHEMAS = frozenset({"macro", "equity"})


def _validate_schema(schema: str) -> None:
    if schema not in ALLOWED_SCHEMAS:
        raise ValueError(f"Schema '{schema}' not in allowed schemas: {sorted(ALLOWED_SCHEMAS)}")


def ensure_run_state_table(conn, schema: str = "macro", cfg: RunStateConfig = RunStateConfig()) -> None:
    _validate_schema(schema)
    stmt = sql.SQL("""
        CREATE TABLE IF NOT EXISTS {}.{} (
            run_id      TEXT PRIMARY KEY,
            pipeline    TEXT NOT NULL,
            run_type    TEXT NOT NULL,
            status      TEXT NOT NULL DEFAULT 'STARTED',
            start_ts    TIMESTAMPTZ NOT NULL DEFAULT now(),
            end_ts      TIMESTAMPTZ,
            fail_stage  TEXT,
            fail_reason TEXT
        )
    """).format(sql.Identifier(schema), sql.Identifier(cfg.table_name))
    with conn.cursor() as cur:
        cur.execute(stmt)
    conn.commit()


def insert_started(*, conn, schema: str, run_id: str, run_type: str, cfg: RunStateConfig = RunStateConfig()) -> None:
    _validate_schema(schema)
    stmt = sql.SQL("""
        INSERT INTO {}.{} (run_id, pipeline, run_type, status)
        VALUES (%s, %s, %s, 'STARTED')
        ON CONFLICT (run_id) DO UPDATE
        SET pipeline = EXCLUDED.pipeline, run_type = EXCLUDED.run_type,
            status = 'STARTED', start_ts = now(), end_ts = null,
            fail_stage = null, fail_reason = null
    """).format(sql.Identifier(schema), sql.Identifier(cfg.table_name))
    with conn.cursor() as cur:
        cur.execute(stmt, (run_id, cfg.pipeline_name, run_type))
    conn.commit()


def mark_success(*, conn, schema: str, run_id: str, cfg: RunStateConfig = RunStateConfig()) -> None:
    _validate_schema(schema)
    stmt = sql.SQL("UPDATE {}.{} SET status = 'SUCCESS', end_ts = now() WHERE run_id = %s").format(
        sql.Identifier(schema), sql.Identifier(cfg.table_name))
    with conn.cursor() as cur:
        cur.execute(stmt, (run_id,))
    conn.commit()


def mark_skipped(*, conn, schema: str, run_id: str, reason: str, cfg: RunStateConfig = RunStateConfig()) -> None:
    _validate_schema(schema)
    stmt = sql.SQL("UPDATE {}.{} SET status = 'SKIPPED', end_ts = now(), fail_reason = %s WHERE run_id = %s").format(
        sql.Identifier(schema), sql.Identifier(cfg.table_name))
    with conn.cursor() as cur:
        cur.execute(stmt, ((reason or "")[:2000], run_id))
    conn.commit()


def mark_failed(*, conn, schema: str, run_id: str, fail_stage: str, fail_reason: str, cfg: RunStateConfig = RunStateConfig()) -> None:
    _validate_schema(schema)
    stmt = sql.SQL("""
        UPDATE {}.{} SET status = 'FAILED', end_ts = now(), fail_stage = %s, fail_reason = %s
        WHERE run_id = %s
    """).format(sql.Identifier(schema), sql.Identifier(cfg.table_name))
    with conn.cursor() as cur:
        cur.execute(stmt, (fail_stage, (fail_reason or "")[:2000], run_id))
    conn.commit()