from __future__ import annotations

import json
import os
import time
from pathlib import Path
from urllib.parse import unquote_plus

from psycopg.errors import DeadlockDetected

from aws_clients import sqs
from config import S3_BUCKET_NAME, SQS_VISIBILITY_TIMEOUT, SQS_WAIT_TIME_SECONDS
from db import get_conn, get_or_create_document_for_s3_object, update_document_status
from rag import chunk_text, download_pdf_from_s3, extract_text_from_pdf


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] pid={os.getpid()} {message}", flush=True)


def filename_from_s3_key(s3_key: str) -> str:
    return s3_key.rsplit("/", 1)[-1] or s3_key


def parse_s3_events_from_sqs_body(body: str) -> list[dict[str, str]]:
    # Parse the SQS body which may be:
    # - a raw S3 event JSON
    # - an SNS notification JSON with a stringified "Message" field
    # - double-encoded JSON (string containing JSON string)
    try:
        payload = json.loads(body)
    except Exception:
        # If body isn't valid JSON, return no events
        return []

    # Unwrap repeated JSON encoding (e.g. '"{...}"')
    while isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            break

    # If this is an SNS envelope, extract the Message field which may itself
    # be a JSON string or already a dict.
    if isinstance(payload, dict) and "Message" in payload:
        message = payload["Message"]
        if isinstance(message, str):
            try:
                payload = json.loads(message)
            except Exception:
                # leave payload as the raw message string (unlikely to contain Records)
                payload = message
        else:
            payload = message

    # Now extract S3 Records if present
    records = []
    if isinstance(payload, dict):
        records = payload.get("Records", []) or []
    elif isinstance(payload, list):
        records = payload

    events: list[dict[str, str]] = []
    for record in records:
        if not isinstance(record, dict) or "s3" not in record:
            continue

        bucket = record["s3"]["bucket"]["name"]
        key = unquote_plus(record["s3"]["object"]["key"])
        events.append({"s3_bucket": bucket, "s3_key": key})

    return events


def process_s3_object_with_retries(strategy: str, s3_bucket: str, s3_key: str) -> None:
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            process_s3_object(strategy=strategy, s3_bucket=s3_bucket, s3_key=s3_key)
            return
        except DeadlockDetected:
            if attempt == max_attempts:
                raise
            sleep_seconds = attempt * 2
            log(
                f"{strategy}: deadlock while processing s3://{s3_bucket}/{s3_key}; "
                f"retrying in {sleep_seconds} seconds"
            )
            time.sleep(sleep_seconds)


def process_s3_object(strategy: str, s3_bucket: str, s3_key: str) -> None:
    pdf_path: Path | None = None
    document_id: int | None = None

    log(f"{strategy}: loading document for s3://{s3_bucket}/{s3_key}")
    with get_conn() as conn:
        with conn.transaction():
            document_id = get_or_create_document_for_s3_object(
                conn,
                s3_bucket=s3_bucket,
                s3_key=s3_key,
                filename=filename_from_s3_key(s3_key),
            )

        run = conn.execute(
            """
            SELECT run_id
            FROM processing_runs
            WHERE document_id = %s AND strategy = %s
            """,
            (document_id, strategy),
        ).fetchone()

        if run is None:
            raise ValueError(f"Processing run not found for document {document_id}, {strategy}")

        run_id = run["run_id"]
        conn.execute(
            """
            UPDATE processing_runs
            SET status = 'processing',
                started_at = CURRENT_TIMESTAMP,
                error_message = NULL
            WHERE run_id = %s
            """,
            (run_id,),
        )
        update_document_status(conn, document_id)
        conn.commit()

    try:
        start_time = time.time()
        log(f"{strategy}: downloading s3://{s3_bucket}/{s3_key}")
        pdf_path = download_pdf_from_s3(s3_bucket, s3_key)
        log(f"{strategy}: downloaded PDF to {pdf_path}")

        log(f"{strategy}: extracting text from PDF")
        text = extract_text_from_pdf(pdf_path)
        log(f"{strategy}: extracted {len(text)} characters")

        log(f"{strategy}: creating chunks")
        chunks = chunk_text(text, strategy)
        log(f"{strategy}: created {len(chunks)} chunks")

        elapsed = time.time() - start_time
        average_length = sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0

        log(f"{strategy}: writing chunks and status to database")
        with get_conn() as conn:
            with conn.transaction():
                conn.execute("DELETE FROM chunks WHERE run_id = %s", (run_id,))
                for index, chunk in enumerate(chunks):
                    conn.execute(
                        """
                        INSERT INTO chunks (run_id, chunk_index, chunk_text, char_count)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (run_id, index, chunk, len(chunk)),
                    )

                conn.execute(
                    """
                    UPDATE processing_runs
                    SET status = 'completed',
                        chunk_count = %s,
                        average_chunk_length = %s,
                        processing_time_seconds = %s,
                        completed_at = CURRENT_TIMESTAMP,
                        error_message = NULL
                    WHERE run_id = %s
                    """,
                    (len(chunks), average_length, elapsed, run_id),
                )
                update_document_status(conn, document_id)

        log(f"{strategy}: completed document_id={document_id} in {elapsed:.3f} seconds")

    except Exception as exc:
        log(f"{strategy}: failed document_id={document_id}: {exc}")
        if document_id is not None:
            with get_conn() as conn:
                with conn.transaction():
                    conn.execute(
                        """
                        UPDATE processing_runs
                        SET status = 'failed',
                            error_message = %s,
                            completed_at = CURRENT_TIMESTAMP
                        WHERE document_id = %s AND strategy = %s
                        """,
                        (str(exc), document_id, strategy),
                    )
                    update_document_status(conn, document_id)
        raise

    finally:
        if pdf_path is not None and pdf_path.exists():
            pdf_path.unlink()


def run_worker(strategy: str, queue_url: str) -> None:
    if not S3_BUCKET_NAME:
        raise RuntimeError("S3_BUCKET_NAME is not configured")
    if not queue_url:
        raise RuntimeError(f"Queue URL is not configured for {strategy}")

    queue_name = queue_url.rsplit("/", 1)[-1]
    log(f"Starting {strategy} worker on queue {queue_name}")
    while True:
        log(f"{strategy}: waiting for SQS messages")
        response = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=SQS_WAIT_TIME_SECONDS,
            VisibilityTimeout=SQS_VISIBILITY_TIMEOUT,
        )

        for message in response.get("Messages", []):
            receipt_handle = message["ReceiptHandle"]
            log(f"{strategy}: received SQS message {message.get('MessageId')}")

            try:
                events = parse_s3_events_from_sqs_body(message["Body"])
                if not events:
                    log(f"{strategy}: no S3 object records in message {message.get('MessageId')}")

                for event in events:
                    process_s3_object_with_retries(
                        strategy=strategy,
                        s3_bucket=event.get("s3_bucket", S3_BUCKET_NAME),
                        s3_key=event["s3_key"],
                    )
            except Exception as exc:
                log(f"{strategy}: leaving failed message {message.get('MessageId')} on queue: {exc}")
                continue

            sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
            log(f"{strategy}: deleted SQS message {message.get('MessageId')}")
