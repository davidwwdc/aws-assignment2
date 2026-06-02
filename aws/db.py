from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import psycopg
from psycopg.rows import dict_row

from config import DATABASE_URL
from config import STRATEGIES


BASE_DIR = Path(__file__).resolve().parent


@contextmanager
def get_conn() -> Iterator[psycopg.Connection[dict[str, Any]]]:
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not configured")

    with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
        yield conn


def init_db() -> None:
    with open(BASE_DIR / "schema.sql", encoding="utf-8") as schema_file:
        schema_sql = schema_file.read()

    with get_conn() as conn:
        conn.execute(schema_sql)
        conn.commit()


def update_document_status(conn: psycopg.Connection[dict[str, Any]], document_id: int) -> None:
    runs = conn.execute(
        """
        SELECT status
        FROM processing_runs
        WHERE document_id = %s
        """,
        (document_id,),
    ).fetchall()

    statuses = [run["status"] for run in runs]
    if statuses and all(status == "completed" for status in statuses):
        overall_status = "ready"
    elif any(status == "failed" for status in statuses):
        overall_status = "failed"
    elif any(status == "processing" for status in statuses):
        overall_status = "processing"
    else:
        overall_status = "uploaded"

    conn.execute(
        """
        UPDATE documents
        SET overall_status = %s
        WHERE document_id = %s
        """,
        (overall_status, document_id),
    )


def get_or_create_document_for_s3_object(
    conn: psycopg.Connection[dict[str, Any]],
    s3_bucket: str,
    s3_key: str,
    filename: str,
) -> int:
    conn.execute(
        "SELECT pg_advisory_xact_lock(hashtext(%s))",
        (f"{s3_bucket}/{s3_key}",),
    )

    document = conn.execute(
        """
        INSERT INTO documents (filename, s3_bucket, s3_key, overall_status)
        VALUES (%s, %s, %s, 'uploaded')
        ON CONFLICT (s3_bucket, s3_key)
        DO UPDATE SET filename = documents.filename
        RETURNING document_id
        """,
        (filename, s3_bucket, s3_key),
    ).fetchone()
    document_id = int(document["document_id"])

    for strategy in STRATEGIES:
        conn.execute(
            """
            INSERT INTO processing_runs (document_id, strategy, status)
            VALUES (%s, %s, 'pending')
            ON CONFLICT (document_id, strategy) DO NOTHING
            """,
            (document_id, strategy),
        )

    return document_id
