"""
lambda_function.py — AWS Lambda entry point for the macro pipeline.
"""
from pipeline.engine.core import lambda_handler


def handler(event, context):
    return lambda_handler(event, context)