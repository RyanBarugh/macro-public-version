"""
lambda_function_equity.py — AWS Lambda entry point for the equity pipeline.

Deploy as a SEPARATE Lambda function from the macro pipeline.
Schedule via EventBridge: once daily after US market close (e.g. 22:00 UTC).

EventBridge rule examples:
    Daily:          {"run_type": "incremental"}
    Backfill:       {"run_type": "backfill"}
    Derived only:   {"action": "derived"}
    Prices only:    {"run_type": "incremental", "skip_derived": true}
"""
from pipeline.equities.core import lambda_handler


def handler(event, context):
    return lambda_handler(event, context)