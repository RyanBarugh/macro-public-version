"""
equities/email_alerts.py
=========================

Equity-specific email formatting for pipeline alerts.

Uses send_email_alert() from engine/email_alerts.py for actual SMTP delivery.
This module only handles equity-specific HTML formatting.

Matches engine/email_alerts.py pattern:
  - format_failure_email() returns (subject, body_html) tuple
  - Table-based HTML layout for email client compatibility
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..engine.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED HTML HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%A %d %B %Y · %H:%M UTC")


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


def _header(title: str = "Equity pipeline report") -> str:
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


def _info_cell(label: str, value: str) -> str:
    return f"""
        <td style="background-color:#fafaf9;border-radius:8px;padding:12px 14px;width:50%">
            <div style="font-size:11px;color:#999999">{label}</div>
            <div style="font-size:14px;font-weight:600;color:#333333">{value}</div>
        </td>"""


def _footer() -> str:
    return """
<tr><td style="padding:14px 28px;border-top:1px solid #f0f0ee">
    <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%">
    <tr>
        <td style="font-size:11px;color:#bbbbbb">Equity pipeline</td>
    </tr>
    </table>
</td></tr>"""


# ═══════════════════════════════════════════════════════════════════════════════
# FAILURE EMAIL
# ═══════════════════════════════════════════════════════════════════════════════

def format_failure_email(
    e: Exception,
    run_type: str,
    fail_stage: str = "PIPELINE",
    tickers_fetched: int = 0,
    tickers_total: int = 0,
    tickers_failed: int = 0,
) -> tuple[str, str]:
    """
    Format a failure email for the equity pipeline.
    Returns (subject, body_html) tuple.
    """
    error_type = type(e).__name__
    error_msg = str(e)[:500]
    subject = f"Equity FAILED — {fail_stage}: {error_type}"

    progress_row = ""
    if tickers_total > 0:
        progress_row = f"""
<tr><td style="padding:0 28px 20px">
    <table role="presentation" cellpadding="0" cellspacing="6" border="0" width="100%">
    <tr>
        {_info_cell("Tickers fetched", f"{tickers_fetched}/{tickers_total}")}
        {_info_cell("Failed", str(tickers_failed))}
    </tr>
    </table>
</td></tr>"""

    body = _wrap(f"""
{_header("Equity pipeline alert")}
<tr><td style="padding:24px 28px 0">
    <table role="presentation" cellpadding="0" cellspacing="0" border="0">
    <tr><td style="background-color:#fef2f2;padding:5px 14px;border-radius:20px">
        <span style="font-size:13px;font-weight:600;color:#dc2626">&#10007; Pipeline failed</span>
    </td></tr>
    </table>
</td></tr>
<tr><td style="padding:20px 28px">
    <table role="presentation" cellpadding="0" cellspacing="6" border="0" width="100%">
    <tr>
        {_info_cell("Run type", run_type)}
        {_info_cell("Failed stage", fail_stage)}
    </tr>
    </table>
</td></tr>
<tr><td style="padding:0 28px 20px">
    <div style="background-color:#fafaf9;border-radius:8px;padding:14px 16px">
        <div style="font-size:11px;color:#999999;margin-bottom:4px">ERROR</div>
        <div style="font-size:13px;color:#333333;font-family:Consolas,monospace;line-height:1.5;word-break:break-word">{error_type}: {error_msg}</div>
    </div>
</td></tr>
{progress_row}
{_footer()}
""")

    return subject, body


# ═══════════════════════════════════════════════════════════════════════════════
# SUCCESS EMAIL — sent after each successful run
# ═══════════════════════════════════════════════════════════════════════════════

def format_success_email(
    run_type: str,
    tickers_fetched: int = 0,
    tickers_failed: int = 0,
    tickers_skipped: int = 0,
    rows_written: int = 0,
    latest_price_date: str | None = None,
    derived_completed: str = "0/0",
    duration_s: float = 0,
    failed_symbols: list[str] | None = None,
) -> tuple[str, str]:
    """
    Format a success email for the equity pipeline.
    Returns (subject, body_html) tuple.
    """
    total = tickers_fetched + tickers_failed + tickers_skipped

    # Headline — is the data fresh?
    if latest_price_date:
        subject = f"Equity OK — {tickers_fetched}/{total} tickers · data to {latest_price_date}"
        status_label = "Pipeline healthy"
        status_color = "#059669"
        status_bg = "#ecfdf5"
        status_icon = "&#10003;"
    else:
        subject = f"Equity OK — {tickers_fetched}/{total} tickers · no new data"
        status_label = "Pipeline ran — no new prices"
        status_color = "#d97706"
        status_bg = "#fffbeb"
        status_icon = "&#10003;"

    # Date freshness indicator
    date_color = "#059669"
    date_label = latest_price_date or "none"

    # Warning row if failures or skips
    warning_row = ""
    if tickers_failed > 0 or tickers_skipped > 0:
        items = []
        if tickers_failed > 0:
            items.append(f"{tickers_failed} tickers failed to fetch")
        if tickers_skipped > 0:
            items.append(f"{tickers_skipped} skipped (timeout)")
        warning_text = " · ".join(items)

        # Show failed ticker names (cap at 30 to avoid huge emails)
        symbols_detail = ""
        if failed_symbols:
            shown = failed_symbols[:30]
            symbols_str = ", ".join(shown)
            if len(failed_symbols) > 30:
                symbols_str += f" … +{len(failed_symbols) - 30} more"
            symbols_detail = f"""
        <div style="font-size:11px;color:#78350f;font-family:Consolas,monospace;margin-top:6px;line-height:1.5">{symbols_str}</div>"""

        warning_row = f"""
<tr><td style="padding:0 28px 20px">
    <div style="background-color:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:12px 16px">
        <div style="font-size:12px;font-weight:600;color:#92400e">{warning_text}</div>{symbols_detail}
    </div>
</td></tr>"""

    body = _wrap(f"""
{_header("Equity pipeline report")}
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
        <td align="center" style="padding:12px 8px;background:#fafaf9;border-radius:8px">
            <div style="font-size:26px;font-weight:700;color:{date_color};line-height:1">{date_label}</div>
            <div style="font-size:11px;color:#999999;margin-top:4px">latest price date</div>
        </td>
        <td align="center" style="padding:12px 8px;background:#fafaf9;border-radius:8px">
            <div style="font-size:26px;font-weight:700;color:#111111;line-height:1">{tickers_fetched}/{total}</div>
            <div style="font-size:11px;color:#999999;margin-top:4px">tickers fetched</div>
        </td>
        <td align="center" style="padding:12px 8px;background:#fafaf9;border-radius:8px">
            <div style="font-size:26px;font-weight:700;color:#111111;line-height:1">{duration_s}s</div>
            <div style="font-size:11px;color:#999999;margin-top:4px">duration</div>
        </td>
    </tr>
    </table>
</td></tr>
<tr><td style="padding:0 28px 20px">
    <table role="presentation" cellpadding="0" cellspacing="6" border="0" width="100%">
    <tr>
        {_info_cell("Prices written", f"{rows_written:,} rows")}
        {_info_cell("Derived", derived_completed)}
    </tr>
    </table>
</td></tr>
{warning_row}
{_footer()}
""")

    return subject, body