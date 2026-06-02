"""
Decoupled AWS PDF RAG Comparison Web App 

This version separates the upload/web tier from PDF processing:
- Web app tier uploads PDFs to S3 and stores metadata in PostgreSQL.
- Standalone workers read queue message, download the PDF from S3, and write results to Postgres.

Run web after setting environment variables:
    pip install -r requirements.txt
    python app.py

Run workers separately:
    python worker_fixed_size.py
    python worker_paragraph_aware.py
"""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from flask import Flask, redirect, render_template_string, request, url_for
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from aws_clients import s3
from config import (
    MAX_FILE_SIZE_MB,
    S3_BUCKET_NAME,
    STRATEGIES,
    require_web_config,
)
from db import get_conn, get_or_create_document_for_s3_object, init_db
from rag import allowed_file, retrieve_top_chunks


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_MB * 1024 * 1024


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


def load_document(document_id: int) -> dict[str, Any] | None:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM documents WHERE document_id = %s", (document_id,)
        ).fetchone()


def load_processing_runs(document_id: int) -> list[dict[str, Any]]:
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT *
            FROM processing_runs
            WHERE document_id = %s
            ORDER BY strategy
            """,
            (document_id,),
        ).fetchall()


def load_chunks(run_id: int) -> list[dict[str, Any]]:
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT *
            FROM chunks
            WHERE run_id = %s
            ORDER BY chunk_index
            """,
            (run_id,),
        ).fetchall()


def ready_for_comparison(runs: list[dict[str, Any]]) -> bool:
    statuses = {run["strategy"]: run["status"] for run in runs}
    return all(statuses.get(strategy) == "completed" for strategy in STRATEGIES)


def save_retrieval_results(
    document_id: int,
    query_text: str,
    results_by_strategy: dict[str, list[dict[str, Any]]],
) -> int:
    with get_conn() as conn:
        with conn.transaction():
            query = conn.execute(
                """
                INSERT INTO retrieval_queries (document_id, query_text)
                VALUES (%s, %s)
                RETURNING query_id
                """,
                (document_id, query_text),
            ).fetchone()
            query_id = int(query["query_id"])

            for strategy, results in results_by_strategy.items():
                run = conn.execute(
                    """
                    SELECT run_id
                    FROM processing_runs
                    WHERE document_id = %s AND strategy = %s
                    """,
                    (document_id, strategy),
                ).fetchone()

                if run is None:
                    continue

                for result in results:
                    conn.execute(
                        """
                        INSERT INTO retrieval_results
                            (query_id, run_id, rank, chunk_index, score, chunk_text)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            query_id,
                            run["run_id"],
                            result["rank"],
                            result["chunk_index"],
                            result["score"],
                            result["chunk_text"],
                        ),
                    )

    return query_id


@app.get("/")
def index():
    with get_conn() as conn:
        documents = conn.execute(
            """
            SELECT
                d.document_id,
                d.filename,
                d.overall_status,
                d.created_at,
                COUNT(pr.run_id) AS total_runs,
                SUM(CASE WHEN pr.status = 'completed' THEN 1 ELSE 0 END) AS completed_runs,
                SUM(CASE WHEN pr.status = 'failed' THEN 1 ELSE 0 END) AS failed_runs
            FROM documents d
            LEFT JOIN processing_runs pr ON d.document_id = pr.document_id
            GROUP BY d.document_id, d.filename, d.overall_status, d.created_at
            ORDER BY d.created_at DESC
            """
        ).fetchall()

    uploaded = request.args.get("uploaded") == "1"
    has_unfinished_documents = any(
        document["completed_runs"] != document["total_runs"]
        or document["overall_status"] in {"uploaded", "processing"}
        for document in documents
    )

    return render_template_string(
        INDEX_TEMPLATE,
        documents=documents,
        uploaded=uploaded,
        auto_refresh=uploaded or has_unfinished_documents,
    )


@app.post("/upload")
def upload():
    uploaded_file: FileStorage | None = request.files.get("pdf")

    if uploaded_file is None or uploaded_file.filename == "":
        return "No PDF file selected", 400

    if not allowed_file(uploaded_file.filename):
        return "Only PDF files are supported", 400

    original_filename = secure_filename(uploaded_file.filename)
    s3_key = f"uploads/{int(time.time())}_{uuid4().hex}_{original_filename}"

    s3.upload_fileobj(
        uploaded_file,
        S3_BUCKET_NAME,
        s3_key,
        ExtraArgs={"ContentType": "application/pdf"},
    )

    with get_conn() as conn:
        with conn.transaction():
            document_id = get_or_create_document_for_s3_object(
                conn,
                s3_bucket=S3_BUCKET_NAME,
                s3_key=s3_key,
                filename=original_filename,
            )

    return redirect(url_for("document_detail", document_id=document_id))


@app.get("/documents/<int:document_id>")
def document_detail(document_id: int):
    document = load_document(document_id)
    if document is None:
        return "Document not found", 404

    runs = load_processing_runs(document_id)
    run_details = []

    for run in runs:
        chunks = load_chunks(run["run_id"])
        run_details.append(
            {
                "run": run,
                "display_name": STRATEGIES.get(run["strategy"], run["strategy"]),
                "sample_chunks": chunks[:2],
            }
        )

    return render_template_string(
        DETAIL_TEMPLATE,
        document=document,
        runs=run_details,
        ready=ready_for_comparison(runs),
        query=None,
        results_by_strategy=None,
        strategy_names=STRATEGIES,
    )


@app.post("/documents/<int:document_id>/query")
def query_document(document_id: int):
    document = load_document(document_id)
    if document is None:
        return "Document not found", 404

    query_text = request.form.get("query", "").strip()
    if not query_text:
        return redirect(url_for("document_detail", document_id=document_id))

    runs = load_processing_runs(document_id)
    if not ready_for_comparison(runs):
        return "Both processing runs must be completed before comparison", 400

    results_by_strategy: dict[str, list[dict[str, Any]]] = {}
    run_details = []

    for run in runs:
        chunks = load_chunks(run["run_id"])
        results = retrieve_top_chunks(chunks, query_text, top_k=3)
        results_by_strategy[run["strategy"]] = results
        run_details.append(
            {
                "run": run,
                "display_name": STRATEGIES.get(run["strategy"], run["strategy"]),
                "sample_chunks": chunks[:2],
            }
        )

    save_retrieval_results(document_id, query_text, results_by_strategy)

    return render_template_string(
        DETAIL_TEMPLATE,
        document=document,
        runs=run_details,
        ready=True,
        query=query_text,
        results_by_strategy=results_by_strategy,
        strategy_names=STRATEGIES,
    )


@app.post("/reset")
def reset():
    with get_conn() as conn:
        with conn.transaction():
            conn.execute("DELETE FROM retrieval_results")
            conn.execute("DELETE FROM retrieval_queries")
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM processing_runs")
            conn.execute("DELETE FROM documents")

    return redirect(url_for("index"))


BASE_STYLE = """
<style>
    body {
        font-family: Arial, sans-serif;
        margin: 2rem auto;
        max-width: 1100px;
        line-height: 1.5;
        color: #222;
    }
    h1, h2, h3 { color: #111; }
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 1rem 0;
    }
    th, td {
        border: 1px solid #ddd;
        padding: 0.6rem;
        vertical-align: top;
    }
    th { background: #f4f4f4; text-align: left; }
    .card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        background: #fafafa;
    }
    .status {
        font-weight: bold;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        background: #eee;
    }
    .completed, .ready { background: #d9f7d9; }
    .failed { background: #ffd8d8; }
    .processing { background: #fff2cc; }
    .pending, .uploaded { background: #e8e8ff; }
    .query-box {
        display: flex;
        gap: 0.5rem;
        margin: 1rem 0;
    }
    input[type="text"] {
        flex: 1;
        padding: 0.5rem;
    }
    button {
        padding: 0.5rem 0.8rem;
        cursor: pointer;
    }
    pre {
        background: #f3f3f3;
        border: 1px solid #ddd;
        padding: 0.8rem;
        overflow-x: auto;
        white-space: pre-wrap;
    }
    .grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
    }
</style>
"""

INDEX_TEMPLATE = (
    BASE_STYLE
    + """
{% if auto_refresh %}
<meta http-equiv="refresh" content="3">
{% endif %}

<h1>AWS Event-Driven PDF RAG Comparison App</h1>

<div class="card">
    <h2>Upload a PDF</h2>
    <p>
        The web tier stores the PDF in S3. S3 object-created notifications
        deliver processing events to the worker queues.
    </p>
    <form method="post" action="/upload" enctype="multipart/form-data">
        <input type="file" name="pdf" accept="application/pdf" required>
        <button type="submit">Upload to S3</button>
    </form>
</div>

<h2>Uploaded Documents</h2>

{% if uploaded %}
<p>The PDF was uploaded to S3. It will appear below after an S3 event reaches a worker.</p>
{% endif %}

{% if documents %}
<table>
    <thead>
        <tr>
            <th>ID</th>
            <th>Filename</th>
            <th>Status</th>
            <th>Processing runs</th>
            <th>Uploaded</th>
            <th>Action</th>
        </tr>
    </thead>
    <tbody>
        {% for doc in documents %}
        <tr>
            <td>{{ doc.document_id }}</td>
            <td>{{ doc.filename }}</td>
            <td><span class="status {{ doc.overall_status }}">{{ doc.overall_status }}</span></td>
            <td>{{ doc.completed_runs or 0 }}/{{ doc.total_runs or 0 }} completed</td>
            <td>{{ doc.created_at }}</td>
            <td><a href="/documents/{{ doc.document_id }}">Open</a></td>
        </tr>
        {% endfor %}
    </tbody>
</table>
{% else %}
<p>No documents uploaded yet.</p>
{% endif %}

<form method="post" action="/reset" onsubmit="return confirm('Delete all database rows? Uploaded S3 objects are not deleted by this demo reset.')">
    <button type="submit">Reset Database Rows</button>
</form>
"""
)

DETAIL_TEMPLATE = (
    BASE_STYLE
    + """
{% if not ready %}
<meta http-equiv="refresh" content="3">
{% endif %}

<p><a href="/">Back to documents</a></p>

<h1>Document {{ document.document_id }}: {{ document.filename }}</h1>

<div class="card">
    <p><strong>Status:</strong> <span class="status {{ document.overall_status }}">{{ document.overall_status }}</span></p>
    <p><strong>S3 bucket:</strong> {{ document.s3_bucket }}</p>
    <p><strong>S3 key:</strong> {{ document.s3_key }}</p>
    <p><strong>Uploaded:</strong> {{ document.created_at }}</p>
</div>

<h2>Processing Runs</h2>
<table>
    <thead>
        <tr>
            <th>Strategy</th>
            <th>Status</th>
            <th>Chunks</th>
            <th>Average length</th>
            <th>Processing time</th>
            <th>Error</th>
        </tr>
    </thead>
    <tbody>
        {% for item in runs %}
        {% set run = item.run %}
        <tr>
            <td>{{ item.display_name }}</td>
            <td><span class="status {{ run.status }}">{{ run.status }}</span></td>
            <td>{{ run.chunk_count or '-' }}</td>
            <td>{{ '%.1f'|format(run.average_chunk_length) if run.average_chunk_length else '-' }}</td>
            <td>{{ '%.3f'|format(run.processing_time_seconds) if run.processing_time_seconds else '-' }} sec</td>
            <td>{{ run.error_message or '' }}</td>
        </tr>
        {% endfor %}
    </tbody>
</table>

<h2>Sample Chunks</h2>
<div class="grid">
    {% for item in runs %}
    <div class="card">
        <h3>{{ item.display_name }}</h3>
        {% if item.sample_chunks %}
            {% for chunk in item.sample_chunks %}
            <p><strong>Chunk {{ chunk.chunk_index }}</strong> - {{ chunk.char_count }} characters</p>
            <pre>{{ chunk.chunk_text[:900] }}{% if chunk.chunk_text|length > 900 %}...{% endif %}</pre>
            {% endfor %}
        {% else %}
            <p>No chunks available.</p>
        {% endif %}
    </div>
    {% endfor %}
</div>

<h2>Query Comparison</h2>
{% if ready %}
<form method="post" action="/documents/{{ document.document_id }}/query" class="query-box">
    <input type="text" name="query" placeholder="Example: What does the document say about cloud computing?" value="{{ query or '' }}" required>
    <button type="submit">Compare Retrieval</button>
</form>
{% else %}
<p>Query comparison is available only after both processing runs are completed.</p>
{% endif %}

{% if query and results_by_strategy %}
<h3>Query: {{ query }}</h3>
<div class="grid">
    {% for strategy, results in results_by_strategy.items() %}
    <div class="card">
        <h3>{{ strategy_names[strategy] }}</h3>
        {% for result in results %}
        <p>
            <strong>Rank {{ result.rank }}</strong>,
            chunk {{ result.chunk_index }},
            score {{ '%.4f'|format(result.score) }}
        </p>
        <pre>{{ result.chunk_text[:1200] }}{% if result.chunk_text|length > 1200 %}...{% endif %}</pre>
        {% endfor %}
    </div>
    {% endfor %}
</div>
{% endif %}
"""
)


if __name__ == "__main__":
    require_web_config()
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)
