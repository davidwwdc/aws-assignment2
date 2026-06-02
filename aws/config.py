"""
Configuration for the decoupled AWS version.

Expected environment variables:
    AWS_REGION
    S3_BUCKET_NAME
    FIXED_SIZE_QUEUE_URL
    PARAGRAPH_AWARE_QUEUE_URL
    DATABASE_URL

DATABASE_URL example:
    postgresql://app_user:password@my-rds-instance.xxxxxx.us-east-1.rds.amazonaws.com:5432/pdf_rag
"""

from __future__ import annotations

import os


AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "")
FIXED_SIZE_QUEUE_URL = os.getenv("FIXED_SIZE_QUEUE_URL", "")
PARAGRAPH_AWARE_QUEUE_URL = os.getenv("PARAGRAPH_AWARE_QUEUE_URL", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")

ALLOWED_EXTENSIONS = {"pdf"}
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "5"))
SQS_WAIT_TIME_SECONDS = int(os.getenv("SQS_WAIT_TIME_SECONDS", "20"))
SQS_VISIBILITY_TIMEOUT = int(os.getenv("SQS_VISIBILITY_TIMEOUT", "300"))

STRATEGIES = {
    "fixed_size": "Fixed-size chunking",
    "paragraph_aware": "Paragraph-aware chunking",
}


def require_config() -> None:
    missing = [
        name
        for name, value in {
            "S3_BUCKET_NAME": S3_BUCKET_NAME,
            "FIXED_SIZE_QUEUE_URL": FIXED_SIZE_QUEUE_URL,
            "PARAGRAPH_AWARE_QUEUE_URL": PARAGRAPH_AWARE_QUEUE_URL,
            "DATABASE_URL": DATABASE_URL,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


def require_web_config() -> None:
    missing = [
        name
        for name, value in {
            "S3_BUCKET_NAME": S3_BUCKET_NAME,
            "DATABASE_URL": DATABASE_URL,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


def require_worker_config() -> None:
    require_config()
