from __future__ import annotations

import boto3
from botocore.config import Config

from config import AWS_REGION


AWS_CLIENT_CONFIG = Config(
    connect_timeout=10,
    read_timeout=60,
    retries={"max_attempts": 3, "mode": "standard"},
)

s3 = boto3.client("s3", region_name=AWS_REGION, config=AWS_CLIENT_CONFIG)
sqs = boto3.client("sqs", region_name=AWS_REGION, config=AWS_CLIENT_CONFIG)
