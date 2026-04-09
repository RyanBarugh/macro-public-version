from __future__ import annotations

"""
logger.py
=========
Logging configuration for the macro pipeline.

Lambda:  stdout only → CloudWatch Logs captures everything automatically.
Local:   stdout + rotating file handler for offline debugging.

All handlers get the SecretScrubFilter so sensitive values never reach output.

No S3 log upload — CloudWatch is the single log destination in Lambda.
Matches TFF/Oanda pipeline pattern exactly.
"""

import logging
import os
import re
import sys


def _is_lambda() -> bool:
    return bool(os.getenv("AWS_LAMBDA_FUNCTION_NAME"))


# ── Log scrubbing ─────────────────────────────────────────────────────────────

class SecretScrubFilter(logging.Filter):
    """
    Redacts sensitive values from log messages before they reach CloudWatch.

    Catches:
      - Bearer tokens:             "Bearer abc123..."  → "Bearer ***REDACTED***"
      - API key/token assignments:  "api_token=xyz"    → "api_token=***REDACTED***"
      - Password assignments:       "password=xyz"     → "password=***REDACTED***"
      - PostgreSQL URIs:            "postgresql://user:pass@host" → "postgresql://user:***REDACTED***@host"
    """
    _PATTERNS = [
        (re.compile(r"(Bearer\s+)\S+", re.IGNORECASE), r"\1***REDACTED***"),
        (
            re.compile(
                r'(?i)(api[_-]?key|api[_-]?token|token|password|secret|authorization)'
                r"""(\s*[:=]\s*)['"]?[\w\-\.]+['"]?"""
            ),
            r"\1\2***REDACTED***",
        ),
        (re.compile(r"(postgresql://\w+:)[^@]+(@)"), r"\1***REDACTED***\2"),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            for pattern, replacement in self._PATTERNS:
                record.msg = pattern.sub(replacement, record.msg)
        if hasattr(record, "args") and record.args:
            try:
                formatted = record.msg % record.args
                for pattern, replacement in self._PATTERNS:
                    formatted = pattern.sub(replacement, formatted)
                record.msg = formatted
                record.args = None
            except (TypeError, ValueError):
                pass
        return True


# ── Configure logging ─────────────────────────────────────────────────────────

def configure_logging(log_level: int = logging.INFO) -> None:
    """
    Configure logging for the macro pipeline.

    Called ONCE at the top of run_pipeline() — no other file calls this.
    """
    root = logging.getLogger()
    root.setLevel(log_level)

    if root.hasHandlers():
        root.handlers.clear()

    scrub_filter = SecretScrubFilter()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    console.addFilter(scrub_filter)
    root.addHandler(console)

    if not _is_lambda():
        try:
            from logging.handlers import RotatingFileHandler

            log_file = "logs/macro_pipeline.log"
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            file_handler = RotatingFileHandler(
                filename=log_file, maxBytes=5_000_000,
                backupCount=5, encoding="utf-8",
            )
            file_handler.setFormatter(fmt)
            file_handler.addFilter(scrub_filter)
            root.addHandler(file_handler)
        except Exception as e:
            root.warning("Failed to initialise file logging: %s", e)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)