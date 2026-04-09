from __future__ import annotations

"""
email_alerts.py
===============
HTML email alerting for the macro pipeline.

Two email types:
  1. Failure email — sent IMMEDIATELY when a run crashes or hits DERIVED_GATE.
     Called from core.py except block.
  2. Daily digest — sent ONCE per day by a separate scheduled invocation.
     Queries macro_run_state for the last 24h and summarises all runs.

Table-based layout for bulletproof rendering across all email clients.
Matches TFF/Oanda pipeline email style.

Env vars required:
    SMTP_USER, SMTP_PASS, SENDER_EMAIL, RECEIVER_EMAIL
"""

import smtplib
import os
from datetime import datetime, timezone
from email.message import EmailMessage
from urllib.parse import quote

from .logger import get_logger

logger = get_logger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# SEND
# ═════════════════════════════════════════════════════════════════════════════

def send_email_alert(subject: str, body_html: str) -> None:
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    sender_email = os.getenv("SENDER_EMAIL", "alerts@edgeflow.com")
    receiver_email = os.getenv("RECEIVER_EMAIL")

    if not all([smtp_user, smtp_pass, sender_email, receiver_email]):
        logger.warning("Email alerting skipped — missing SMTP credentials in env")
        return

    msg = EmailMessage()
    msg["From"] = f"EdgeFlow <{sender_email}>"
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.set_content("This email requires an HTML-capable email client.")
    msg.add_alternative(body_html, subtype="html")

    try:
        with smtplib.SMTP_SSL("in-v3.mailjet.com", 465) as smtp:
            smtp.login(smtp_user, smtp_pass)
            smtp.send_message(msg)
        logger.info("Alert email sent: %s", subject)
    except Exception as e:
        logger.error("Failed to send alert email: %s", e, exc_info=True)


# ═════════════════════════════════════════════════════════════════════════════
# SHARED HTML HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%A %d %B %Y · %H:%M UTC")


def _cloudwatch_logs_url() -> str | None:
    region = os.getenv("AWS_REGION", "eu-west-2")
    function_name = os.getenv("AWS_LAMBDA_FUNCTION_NAME")
    if not function_name:
        return None
    log_group = quote(quote(f"/aws/lambda/{function_name}", safe=""), safe="")
    return (
        f"https://{region}.console.aws.amazon.com/cloudwatch/home"
        f"?region={region}"
        f"#logsV2:log-groups/log-group/{log_group}"
    )


def _metric_cell(value: str, label: str, color: str = "#111111") -> str:
    return f"""
    <td align="center" style="padding:12px 8px;background:#fafaf9;border-radius:8px">
        <div style="font-size:26px;font-weight:700;color:{color};line-height:1">{value}</div>
        <div style="font-size:11px;color:#999999;margin-top:4px">{label}</div>
    </td>"""


def _info_cell(label: str, value: str) -> str:
    return f"""
        <td style="background-color:#fafaf9;border-radius:8px;padding:12px 14px;width:50%">
            <div style="font-size:11px;color:#999999">{label}</div>
            <div style="font-size:14px;font-weight:600;color:#333333">{value}</div>
        </td>"""


def _wrap(content: str) -> str:
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"></head>
<body style="margin:0;padding:0;background-color:#f4f4f2;font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif">
<table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%" style="background-color:#f4f4f2">
<tr><td align="center" style="padding:24px 16px">
<table role="presentation" cellpadding="0" cellspacing="0" border="0" width="520" style="max-width:520px;width:100%;background-color:#ffffff;border-radius:12px;overflow:hidden;border:1px solid #e8e8e6">
{content}
</table>
</td></tr>
</table>
</body></html>"""


def _header(title: str = "Macro pipeline report") -> str:
    return f"""
<tr><td style="background-color:#111214;padding:28px 28px 24px">
    <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%">
    <tr><td>
        <div style="font-size:11px;letter-spacing:0.08em;color:#777777;margin-bottom:12px">&#9670; EDGEFLOW</div>
        <div style="font-size:20px;font-weight:600;color:#ffffff;margin-bottom:4px">{title}</div>
        <div style="font-size:12px;color:#666666">{_timestamp()}</div>
    </td></tr>
    </table>
</td></tr>"""


def _footer() -> str:
    cw_url = _cloudwatch_logs_url()
    if cw_url:
        logs_cell = (
            f'<a href="{cw_url}" style="font-size:11px;color:#2563eb;text-decoration:none">'
            f'View logs in CloudWatch &#8599;</a>'
        )
    else:
        logs_cell = '<span style="font-size:11px;color:#bbbbbb">Logs in console</span>'

    return f"""
<tr><td style="padding:14px 28px;border-top:1px solid #f0f0ee">
    <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%">
    <tr>
        <td style="font-size:11px;color:#bbbbbb">Macro pipeline</td>
        <td align="right">{logs_cell}</td>
    </tr>
    </table>
</td></tr>"""


# ═════════════════════════════════════════════════════════════════════════════
# 1. FAILURE EMAIL — sent immediately from core.py except block
# ═════════════════════════════════════════════════════════════════════════════

def format_failure_email(
    e: Exception,
    run_type: str,
    fail_stage: str = "PIPELINE",
    series_fetched: int = 0,
    series_total: int = 0,
    required_failures: list[str] | None = None,
) -> tuple[str, str]:

    error_type = type(e).__name__
    error_msg = str(e)[:500]
    subject = f"Macro FAILED — {fail_stage}: {error_type}"

    # Required failures detail
    required_row = ""
    if required_failures:
        items = ", ".join(sorted(required_failures))
        required_row = f"""
<tr><td style="padding:0 28px 20px">
    <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%" style="background-color:#fef2f2;border:1px solid #fecaca;border-radius:8px">
    <tr><td style="padding:14px 16px">
        <div style="font-size:12px;font-weight:600;color:#991b1b;margin-bottom:4px">Required series missing ({len(required_failures)})</div>
        <div style="font-size:11px;color:#7f1d1d;font-family:Consolas,monospace;line-height:1.5">{items}</div>
    </td></tr>
    </table>
</td></tr>"""

    # Progress before crash
    progress_row = ""
    if series_total > 0:
        progress_row = f"""
<tr><td style="padding:0 28px 20px">
    <table role="presentation" cellpadding="0" cellspacing="6" border="0" width="100%">
    <tr>
        {_info_cell("Series before crash", f"{series_fetched}/{series_total}")}
        {_info_cell("Failed stage", fail_stage)}
    </tr>
    </table>
</td></tr>"""

    body = _wrap(f"""
{_header("Macro pipeline alert")}
<tr><td style="padding:24px 28px 0">
    <table role="presentation" cellpadding="0" cellspacing="0" border="0">
    <tr><td style="background-color:#fef2f2;padding:5px 14px;border-radius:20px">
        <span style="font-size:13px;font-weight:600;color:#dc2626">&#10007; Pipeline failed</span>
    </td></tr>
    </table>
</td></tr>
<tr><td style="padding:20px 28px">
    <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%" style="background-color:#fef2f2;border:1px solid #fecaca;border-radius:8px">
    <tr><td style="padding:16px 18px">
        <div style="font-size:12px;font-weight:600;color:#991b1b;margin-bottom:8px">&#9888; {error_type}</div>
        <div style="font-size:12px;color:#7f1d1d;font-family:Consolas,'Courier New',monospace;line-height:1.5;background-color:#fdf0f0;border-radius:6px;padding:10px 12px">{error_msg}</div>
    </td></tr>
    </table>
</td></tr>
<tr><td style="padding:0 28px 20px;padding-top:16px">
    <table role="presentation" cellpadding="0" cellspacing="6" border="0" width="100%">
    <tr>
        {_info_cell("Run type", run_type.title())}
        {_info_cell("Failed stage", fail_stage)}
    </tr>
    </table>
</td></tr>
{required_row}
{progress_row}
<tr><td style="padding:0 28px 24px">
    <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%" style="background-color:#fffbeb;border:1px solid #fde68a;border-radius:8px">
    <tr><td style="padding:14px 16px">
        <div style="font-size:12px;font-weight:600;color:#92400e;margin-bottom:4px">What to check</div>
        <div style="font-size:13px;color:#78350f;line-height:1.7">1. Are external APIs reachable? (FRED, BLS, EODHD)<br>2. Is the DB connection alive?<br>3. Check CloudWatch logs for the full traceback<br>4. If DERIVED_GATE: which required series failed?</div>
    </td></tr>
    </table>
</td></tr>
{_footer()}""")

    return subject, body


# ═════════════════════════════════════════════════════════════════════════════
# 2. DAILY DIGEST — sent once per day by separate scheduled invocation
# ═════════════════════════════════════════════════════════════════════════════

def query_daily_stats(conn, schema: str = "macro") -> dict:
    """
    Query macro_run_state for the last 24h and return summary stats.
    """
    sql = f"""
        SELECT run_id, status, run_type, start_ts, end_ts,
               fail_stage, fail_reason
        FROM {schema}.macro_run_state
        WHERE start_ts >= now() - interval '24 hours'
        ORDER BY start_ts DESC
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]

    runs = [dict(zip(cols, r)) for r in rows]

    succeeded = [r for r in runs if r["status"] == "SUCCESS"]
    failed = [r for r in runs if r["status"] == "FAILED"]
    skipped = [r for r in runs if r["status"] == "SKIPPED"]

    return {
        "total_runs": len(runs),
        "succeeded": len(succeeded),
        "failed": len(failed),
        "skipped": len(skipped),
        "latest_run_status": runs[0]["status"] if runs else "none",
        "latest_run_ts": runs[0]["end_ts"].strftime("%H:%M UTC") if runs and runs[0]["end_ts"] else runs[0]["start_ts"].strftime("%H:%M UTC") if runs else "none",
        "last_success_ts": succeeded[0]["end_ts"].strftime("%H:%M UTC") if succeeded else "none",
        "last_fail_ts": failed[0]["start_ts"].strftime("%H:%M UTC") if failed else None,
        "last_fail_stage": failed[0]["fail_stage"] if failed else None,
        "last_fail_reason": (failed[0]["fail_reason"] or "")[:200] if failed else None,
        "failures": [
            {
                "run_id": r["run_id"],
                "fail_stage": r["fail_stage"],
                "fail_reason": (r["fail_reason"] or "")[:100],
                "start_ts": r["start_ts"].strftime("%H:%M"),
            }
            for r in failed
        ],
    }


def query_series_freshness(conn, schema: str = "macro") -> dict:
    """
    Check how many distinct series were updated in the last 24h.
    """
    sql = f"""
        SELECT COUNT(DISTINCT series_id) as updated,
               MAX(updated_at) as latest
        FROM {schema}.series_data
        WHERE updated_at >= now() - interval '24 hours'
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        row = cur.fetchone()

    return {
        "series_updated": row[0] or 0,
        "latest_update": row[1].strftime("%H:%M UTC") if row[1] else "none",
    }


# ═════════════════════════════════════════════════════════════════════════════
# SHARED: PAIR SIGNAL LADDER — used by both digest and release alert
# ═════════════════════════════════════════════════════════════════════════════

def query_pair_ladder(conn, schema: str = "macro") -> list[dict]:
    """
    Get the latest pair signal for all 28 G10 pairs, sorted by value descending.
    Also gets the previous value to compute delta.

    Returns: [{"pair": "AUDCAD", "value": 1.783, "delta": 0.02}, ...]
    """
    sql = f"""
        WITH latest AS (
            SELECT series_id, value, time,
                   ROW_NUMBER() OVER (PARTITION BY series_id ORDER BY time DESC) as rn
            FROM {schema}.composite_signals
            WHERE currency = 'pair'
        )
        SELECT
            a.series_id,
            a.value as current_val,
            b.value as prev_val
        FROM latest a
        LEFT JOIN latest b ON a.series_id = b.series_id AND b.rn = 2
        WHERE a.rn = 1
        ORDER BY a.value DESC
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()

    if not rows:
        return []

    ladder = []
    for series_id, current_val, prev_val in rows:
        pair = series_id.replace("_signal", "").upper()
        current = round(float(current_val), 3)
        delta = round(current - float(prev_val), 3) if prev_val is not None else None
        ladder.append({"pair": pair, "value": current, "delta": delta})

    return ladder


def format_pair_ladder_html(pair_ladder: list[dict], show_delta: bool = False) -> str:
    """
    Render pair signal ladder as HTML table rows.
    Shared by digest and release alert emails.
    """
    if not pair_ladder:
        return ""

    max_abs = max(abs(p["value"]) for p in pair_ladder) if pair_ladder else 1

    pair_cells = ""
    for p in pair_ladder:
        val = p["value"]
        delta = p.get("delta")
        color = "#059669" if val > 0 else "#dc2626" if val < 0 else "#999"
        sign = "+" if val > 0 else ""
        bar_pct = min(abs(val) / max_abs * 100, 100) if max_abs > 0 else 0
        bar_color = "#d1fae5" if val > 0 else "#fee2e2"
        bar_align = "left" if val > 0 else "right"

        # Delta column
        delta_cell = ""
        if show_delta and delta is not None and abs(delta) > 0.001:
            d_color = "#059669" if delta > 0 else "#dc2626"
            d_sign = "+" if delta > 0 else ""
            delta_cell = f'<td style="padding:2px 0;font-size:10px;color:{d_color};width:45px;text-align:right">{d_sign}{delta:.3f}</td>'
        elif show_delta:
            delta_cell = '<td style="padding:2px 0;width:45px"></td>'

        pair_cells += (
            f'<tr>'
            f'<td style="padding:2px 0;font-size:12px;font-weight:600;width:70px">{p["pair"]}</td>'
            f'<td style="padding:2px 4px;font-size:12px;color:{color};width:55px;text-align:right">{sign}{val:.3f}</td>'
            f'{delta_cell}'
            f'<td style="padding:2px 0">'
            f'<div style="width:100%;height:14px;position:relative">'
            f'<div style="position:absolute;{bar_align}:0;top:0;height:14px;width:{bar_pct:.0f}%;background:{bar_color};border-radius:2px"></div>'
            f'</div>'
            f'</td>'
            f'</tr>'
        )

    return f"""
<tr><td style="padding:0 28px 16px">
    <div style="border:1px solid #e8e8e6;border-radius:8px;overflow:hidden">
        <div style="background:#fafaf9;padding:10px 14px;border-bottom:1px solid #e8e8e6">
            <span style="font-size:13px;font-weight:600;color:#111">Pair signal ladder</span>
            <span style="font-size:11px;color:#999;margin-left:8px">Strong long ▲ to strong short ▼</span>
        </div>
        <div style="padding:8px 14px">
            <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%">
            {pair_cells}
            </table>
        </div>
    </div>
</td></tr>"""


def format_daily_digest(stats: dict, freshness: dict, pair_ladder: list[dict] | None = None) -> tuple[str, str]:
    """
    Build the daily digest email from run stats and series freshness.
    """
    total = stats["total_runs"]
    ok = stats["succeeded"]
    failed = stats["failed"]
    skipped = stats["skipped"]
    series_updated = freshness["series_updated"]

    # Latest run determines the headline status
    latest_status = stats.get("latest_run_status", "none")
    latest_ts = stats.get("latest_run_ts", "?")
    latest_ok = latest_status == "SUCCESS"

    # Headline badge: is the pipeline healthy RIGHT NOW?
    if latest_ok and failed == 0:
        status_label = "All systems operational"
        status_color = "#059669"
        status_bg = "#ecfdf5"
        status_icon = "&#10003;"
    elif latest_ok and failed > 0:
        status_label = f"Recovered — {failed} earlier failure{'s' if failed != 1 else ''} today"
        status_color = "#d97706"
        status_bg = "#fffbeb"
        status_icon = "&#10003;"
    else:
        status_label = f"Latest run {latest_status} at {latest_ts}"
        status_color = "#dc2626"
        status_bg = "#fef2f2"
        status_icon = "&#10007;"

    ok_color = "#059669" if ok == total else "#d97706"

    subject = (
        f"Macro daily digest — {ok}/{total} runs OK"
        + (f" · {series_updated} series updated" if series_updated else "")
        + (f" · {failed} failed earlier" if failed and latest_ok else "")
        + (f" · {failed} FAILED" if failed and not latest_ok else "")
    )

    # Failure detail rows
    failure_row = ""
    if stats["failures"]:
        failure_items = ""
        for f in stats["failures"][:5]:
            failure_items += (
                f'<div style="padding:6px 0;border-bottom:1px solid #fef2f2">'
                f'<span style="font-size:11px;color:#999">{f["start_ts"]}</span> '
                f'<span style="font-size:12px;font-weight:600;color:#991b1b">{f["fail_stage"]}</span> '
                f'<span style="font-size:11px;color:#7f1d1d">{f["fail_reason"]}</span>'
                f'</div>'
            )
        failure_row = f"""
<tr><td style="padding:0 28px 20px">
    <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%" style="background-color:#fef2f2;border:1px solid #fecaca;border-radius:8px">
    <tr><td style="padding:14px 16px">
        <div style="font-size:12px;font-weight:600;color:#991b1b;margin-bottom:8px">Failed runs</div>
        {failure_items}
    </td></tr>
    </table>
</td></tr>"""

    # Pair signal ladder
    ladder_row = format_pair_ladder_html(pair_ladder) if pair_ladder else ""

    body = _wrap(f"""
{_header("Macro daily digest")}
<tr><td style="padding:24px 28px 0">
    <table role="presentation" cellpadding="0" cellspacing="0" border="0">
    <tr><td style="background-color:{status_bg};padding:5px 14px;border-radius:20px">
        <span style="font-size:13px;font-weight:600;color:{status_color}">{status_icon} {status_label}</span>
    </td></tr>
    </table>
</td></tr>
<tr><td style="padding:20px 28px">
    <table role="presentation" cellpadding="0" cellspacing="6" border="0" width="100%">
    <tr>
        {_metric_cell(f"{ok}/{total}", "runs succeeded", ok_color)}
        {_metric_cell(str(series_updated), "series updated")}
        {_metric_cell(str(skipped), "skipped")}
    </tr>
    </table>
</td></tr>
<tr><td style="padding:0 28px 20px">
    <table role="presentation" cellpadding="0" cellspacing="6" border="0" width="100%">
    <tr>
        {_info_cell("Last success", stats["last_success_ts"])}
        {_info_cell("Latest data", freshness["latest_update"])}
    </tr>
    </table>
</td></tr>
{failure_row}
{ladder_row}
{_footer()}""")

    return subject, body


def send_daily_digest(conn, schema: str = "macro") -> None:
    """
    Query run stats + series freshness + pair ladder, format and send the daily digest.
    """
    try:
        stats = query_daily_stats(conn, schema)
        freshness = query_series_freshness(conn, schema)
        pair_ladder = query_pair_ladder(conn, schema)
        subject, body = format_daily_digest(stats, freshness, pair_ladder)
        send_email_alert(subject=subject, body_html=body)
    except Exception:
        logger.exception("Failed to send daily digest")
        raise