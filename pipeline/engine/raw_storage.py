import json
import boto3
from datetime import datetime

from .config import load_configuration
from .logger import get_logger

logger = get_logger(__name__)


def upload_raw_to_s3(series_id: str, payload: dict, run_type: str, start: str) -> None:
    """
    Upload raw API response to S3 under:

    <prefix>raw/<series_id>/<run_type>/<start>.json

    run_type must be either:
    - "incremental"
    - "backfill"
    """
    
    if run_type not in ("incremental", "backfill", "reconcile"):
        raise ValueError(f"Invalid run_type: {run_type}")

    logger.info("Uploading raw series_id=%s run_type=%s start=%s to S3", series_id, run_type, start)

    cfg = load_configuration()
    s3 = boto3.client("s3", region_name=cfg.aws_region)

    key = (
        f"{cfg.prefix}raw/"
        f"{series_id}/"
        f"{run_type}/"
        f"{start}.json"
    )

    s3.put_object(
        Bucket=cfg.bucket,
        Key=key,
        Body=json.dumps(payload).encode("utf-8"),
        ContentType="application/json",
    )

    logger.info("Uploaded raw to s3://%s/%s", cfg.bucket, key)