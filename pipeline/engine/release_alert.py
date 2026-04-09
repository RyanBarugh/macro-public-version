from __future__ import annotations

"""
release_alert.py
================
Detects new macro data releases and sends an email alert with:
  - Which publication just dropped (e.g. "Employment Situation")
  - Currency + source agency
  - New values vs previous values for each updated series
  - Signal score changes (growth_score moved from 0.3 to 0.5)

Called from core.py after a successful run — checks which series had
their value change during this run window.

Detection logic:
  - Any row in series_data where revised_at falls within the last 10 min
    AND previous_value IS NOT NULL = a revision just landed.
  - Any row where updated_at falls within the last 10 min AND
    previous_value IS NULL = new data point.
  - Group updated series by publication_name from publication_lags table.
  - For each affected currency, diff the composite signal scores.

No external calendar needed — the 5-minute polling cadence means
the alert fires ~5-10 minutes after publication.
"""

from datetime import datetime, timezone, timedelta

import pandas as pd

from .logger import get_logger
from .email_alerts import send_email_alert

logger = get_logger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# DETECTION — which series updated in this run?
# ═════════════════════════════════════════════════════════════════════════════

def detect_new_releases(conn, schema: str, lookback_minutes: int = 10) -> list[dict]:
    """
    Find series that received new data or revisions recently.

    Uses a fixed 10-minute window so it works from any code path
    (incremental, derived-only) without timing races.
    """
    sql = f"""
        SELECT s.series_id, s.time, s.value, s.previous_value,
               s.updated_at, s.revised_at,
               p.currency, p.publication_name, p.source_agency
        FROM {schema}.series_data s
        JOIN {schema}.publication_lags p ON s.series_id = p.series_id
        WHERE s.updated_at >= NOW() - INTERVAL '{lookback_minutes} minutes'
           OR s.revised_at >= NOW() - INTERVAL '{lookback_minutes} minutes'
        ORDER BY p.publication_name, p.currency, s.series_id, s.time DESC
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]

    if not rows:
        return []

    updates = [dict(zip(cols, r)) for r in rows]
    for u in updates:
        u["is_revision"] = u["previous_value"] is not None

    # Deduplicate: keep only the most recent observation per series
    seen = set()
    deduped = []
    for u in updates:
        if u["series_id"] not in seen:
            seen.add(u["series_id"])
            deduped.append(u)

    return deduped


def group_by_release(updates: list[dict]) -> dict[str, list[dict]]:
    """
    Group updated series by publication_name.

    Returns: {"Employment Situation": [updates...], "CPI": [updates...]}
    """
    groups: dict[str, list[dict]] = {}
    for u in updates:
        key = u.get("publication_name") or u["series_id"]
        groups.setdefault(key, []).append(u)
    return groups


# ═════════════════════════════════════════════════════════════════════════════
# SIGNAL DIFF — what changed in the scores?
# ═════════════════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════════════════════
# SIGNAL LADDER — imported from email_alerts (shared)
# ═════════════════════════════════════════════════════════════════════════════
# query_pair_ladder and format_pair_ladder_html live in email_alerts.py


# ═════════════════════════════════════════════════════════════════════════════
# EMAIL FORMATTING
# ═════════════════════════════════════════════════════════════════════════════

def _delta_str(val: float | None) -> str:
    """Format a delta value with color and sign."""
    if val is None:
        return '<span style="color:#999">new</span>'
    color = "#059669" if val > 0 else "#dc2626" if val < 0 else "#999"
    sign = "+" if val > 0 else ""
    return f'<span style="color:{color};font-weight:600">{sign}{val:.3f}</span>'


def _fmt_num(val: float) -> str:
    """Format a number nicely — commas for large, decimals for small."""
    abs_val = abs(val)
    if abs_val >= 1_000_000:
        return f"{val:,.0f}"
    elif abs_val >= 100:
        return f"{val:,.1f}"
    elif abs_val >= 1:
        return f"{val:.2f}"
    else:
        return f"{val:.4f}"


def _value_str(val: float, prev: float | None) -> str:
    """Format a series value with previous value comparison."""
    if prev is not None:
        return f'<span style="font-weight:600">{_fmt_num(val)}</span> <span style="color:#999;font-size:11px">(was {_fmt_num(prev)})</span>'
    return f'<span style="font-weight:600">{_fmt_num(val)}</span> <span style="color:#999;font-size:11px">(new)</span>'


def format_release_alert(
    releases: dict[str, list[dict]],
    pair_ladder: list[dict],
) -> tuple[str, str] | None:
    """
    Format the release alert email with pair signal ladder.
    Returns (subject, body_html) or None if nothing to report.
    """
    if not releases:
        return None

    from .email_alerts import _wrap, _header, _footer, _info_cell, format_pair_ladder_html

    release_names = sorted(releases.keys())
    n_series = sum(len(v) for v in releases.values())
    affected_ccys = sorted(set(
        u["currency"].lower() for updates in releases.values() for u in updates
        if u.get("currency")
    ))

    subject = f"Macro release — {', '.join(release_names[:3])}" + (
        f" +{len(release_names) - 3} more" if len(release_names) > 3 else ""
    )

    # ── Release detail rows ───────────────────────────────────────
    release_rows = ""
    for pub_name, updates in sorted(releases.items()):
        ccy = updates[0].get("currency", "?").upper()
        agency = updates[0].get("source_agency", "")

        series_rows = ""
        for u in updates[:8]:
            sid = u["series_id"]
            display_id = sid.replace("us_", "").replace("_sa", "").replace("_", " ").title()
            val = u["value"]
            prev = u.get("previous_value")
            series_rows += (
                f'<tr>'
                f'<td style="padding:4px 0;font-size:12px;color:#444">{display_id}</td>'
                f'<td style="padding:4px 0;font-size:12px;text-align:right">{_value_str(val, prev)}</td>'
                f'</tr>'
            )

        if len(updates) > 8:
            series_rows += (
                f'<tr><td colspan="2" style="padding:4px 0;font-size:11px;color:#999">'
                f'...and {len(updates) - 8} more series</td></tr>'
            )

        release_rows += f"""
<tr><td style="padding:0 28px 16px">
    <div style="border:1px solid #e8e8e6;border-radius:8px;overflow:hidden">
        <div style="background:#fafaf9;padding:10px 14px;border-bottom:1px solid #e8e8e6">
            <span style="font-size:14px;font-weight:600;color:#111">{pub_name}</span>
            <span style="font-size:11px;padding:2px 8px;background:#eff6ff;color:#1e40af;border-radius:10px;margin-left:8px">{ccy}</span>
            <span style="font-size:11px;color:#999;margin-left:6px">{agency}</span>
        </div>
        <div style="padding:8px 14px">
            <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%">
            {series_rows}
            </table>
        </div>
    </div>
</td></tr>"""

    # ── Pair signal ladder (with deltas for release alerts) ─────
    ladder_rows = format_pair_ladder_html(pair_ladder, show_delta=True) if pair_ladder else ""

    timestamp = datetime.now(timezone.utc).strftime("%A %d %B %Y · %H:%M UTC")

    body = _wrap(f"""
{_header("Macro data release")}
<tr><td style="padding:24px 28px 0">
    <table role="presentation" cellpadding="0" cellspacing="0" border="0">
    <tr><td style="background-color:#eff6ff;padding:5px 14px;border-radius:20px">
        <span style="font-size:13px;font-weight:600;color:#1e40af">&#9679; {n_series} series updated across {len(releases)} release{'s' if len(releases) != 1 else ''}</span>
    </td></tr>
    </table>
</td></tr>
<tr><td style="padding:16px 28px 16px">
    <table role="presentation" cellpadding="0" cellspacing="6" border="0" width="100%">
    <tr>
        {_info_cell("Releases", ", ".join(release_names[:4]))}
        {_info_cell("Currencies", ", ".join(c.upper() for c in affected_ccys))}
    </tr>
    </table>
</td></tr>
{release_rows}
{ladder_rows}
{_footer()}""")

    return subject, body


# ═════════════════════════════════════════════════════════════════════════════
# DEDUP — prevent duplicate alert emails across runs
# ═════════════════════════════════════════════════════════════════════════════

def _ensure_alert_log_table(conn, schema: str) -> None:
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {schema}.release_alerts_log (
                series_id   TEXT NOT NULL,
                time        DATE NOT NULL,
                alerted_at  TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (series_id, time)
            )
        """)
    conn.commit()


def _filter_already_alerted(conn, schema: str, updates: list[dict]) -> list[dict]:
    """Remove updates we've already sent an alert for."""
    if not updates:
        return []
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT series_id, time FROM {schema}.release_alerts_log "
            f"WHERE alerted_at >= NOW() - INTERVAL '60 minutes'"
        )
        already = {(r[0], r[1]) for r in cur.fetchall()}
    return [u for u in updates if (u["series_id"], u["time"]) not in already]


def _mark_alerted(conn, schema: str, updates: list[dict]) -> None:
    """Record that we sent alerts for these series+time combos."""
    if not updates:
        return
    from psycopg2.extras import execute_values
    rows = [(u["series_id"], u["time"]) for u in updates]
    with conn.cursor() as cur:
        execute_values(
            cur,
            f"INSERT INTO {schema}.release_alerts_log (series_id, time) "
            f"VALUES %s ON CONFLICT DO NOTHING",
            rows,
        )
    conn.commit()


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC API — called from core.py after successful derived stage
# ═════════════════════════════════════════════════════════════════════════════

def check_and_send_release_alert(conn, schema: str) -> bool:
    """
    Detect new releases, get pair signal ladder, send alert if anything landed.
    Deduplicates across runs so the same data point only triggers one email.

    Returns True if an alert was sent, False if no new data.
    """
    _ensure_alert_log_table(conn, schema)

    updates = detect_new_releases(conn, schema)
    updates = _filter_already_alerted(conn, schema, updates)

    if not updates:
        logger.info("No new data releases detected in this run.")
        return False

    releases = group_by_release(updates)
    affected_ccys = list(set(
        u["currency"].lower() for u in updates if u.get("currency")
    ))

    logger.info(
        "New data detected: %d releases, %d series, currencies=%s",
        len(releases), len(updates), affected_ccys,
    )

    from .email_alerts import query_pair_ladder
    pair_ladder = query_pair_ladder(conn, schema)

    result = format_release_alert(releases, pair_ladder)
    if result is None:
        return False

    subject, body = result
    try:
        send_email_alert(subject=subject, body_html=body)
        _mark_alerted(conn, schema, updates)
        return True
    except Exception:
        logger.exception("Failed to send release alert email")
        return False