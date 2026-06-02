"""
Monolithic PDF RAG Comparison Web App

This starter application intentionally keeps everything in one Python web app:
- PDF upload
- PDF text extraction
- two chunking strategies
- chunk/statistics storage
- query-time retrieval comparison

Local setup:
    python
     
       -m venv venv
    source venv/bin/activate
    pip install flask pypdf scikit-learn
    python app.py

Then open:
    http://127.0.0.1:5000
"""

from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flask import Flask, g, redirect, render_template_string, request, url_for
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
DATABASE_PATH = BASE_DIR / "app.db"

ALLOWED_EXTENSIONS = {"pdf"}
MAX_FILE_SIZE_MB = 5

STRATEGIES = {
    "fixed_size": "Fixed-size chunking",
    "paragraph_aware": "Paragraph-aware chunking",
}

app = Flask(__name__)
app.config["UPLOAD_DIR"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_MB * 1024 * 1024


# -----------------------------------------------------------------------------
# Database helpers
# -----------------------------------------------------------------------------


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        g.db = conn
    return g.db


@app.teardown_appcontext
def close_db(_exception: BaseException | None) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db() -> None:
    UPLOAD_DIR.mkdir(exist_ok=True)

    conn = sqlite3.connect(DATABASE_PATH)
    cur = conn.cursor()

    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
            document_id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            storage_path TEXT NOT NULL,
            overall_status TEXT NOT NULL DEFAULT 'uploaded',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS processing_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            strategy TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            chunk_count INTEGER,
            average_chunk_length REAL,
            processing_time_seconds REAL,
            error_message TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            started_at TEXT,
            completed_at TEXT,
            UNIQUE (document_id, strategy),
            FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            char_count INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (run_id, chunk_index),
            FOREIGN KEY (run_id) REFERENCES processing_runs(run_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS retrieval_queries (
            query_id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            query_text TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS retrieval_results (
            result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id INTEGER NOT NULL,
            run_id INTEGER NOT NULL,
            rank INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            score REAL NOT NULL,
            chunk_text TEXT NOT NULL,
            FOREIGN KEY (query_id) REFERENCES retrieval_queries(query_id) ON DELETE CASCADE,
            FOREIGN KEY (run_id) REFERENCES processing_runs(run_id) ON DELETE CASCADE
        );
        """
    )

    conn.commit()
    conn.close()


# -----------------------------------------------------------------------------
# PDF and RAG logic
# -----------------------------------------------------------------------------


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages: list[str] = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        pages.append(f"# Page {page_number}\n\n{page_text.strip()}")

    return "\n\n".join(pages).strip()


def fixed_size_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def paragraph_aware_chunks(text: str, max_chars: int = 1200) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= max_chars:
            current = f"{current}\n\n{paragraph}" if current else paragraph
        else:
            if current:
                chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)

    return chunks


def chunk_text(text: str, strategy: str) -> list[str]:
    if strategy == "fixed_size":
        return fixed_size_chunks(text)
    if strategy == "paragraph_aware":
        return paragraph_aware_chunks(text)
    raise ValueError(f"Unknown strategy: {strategy}")


def retrieve_top_chunks(chunks: list[sqlite3.Row], query: str, top_k: int = 3) -> list[dict[str, Any]]:
    texts = [chunk["chunk_text"] for chunk in chunks]

    if not texts:
        return []

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).flatten()

    ranked_indices = scores.argsort()[::-1][:top_k]

    results: list[dict[str, Any]] = []
    for rank, index in enumerate(ranked_indices, start=1):
        chunk = chunks[int(index)]
        results.append(
            {
                "rank": rank,
                "chunk_index": chunk["chunk_index"],
                "score": float(scores[int(index)]),
                "chunk_text": chunk["chunk_text"],
            }
        )

    return results


# -----------------------------------------------------------------------------
# Monolithic processing
# -----------------------------------------------------------------------------


def create_document_record(filename: str, storage_path: Path) -> int:
    db = get_db()
    cur = db.execute(
        """
        INSERT INTO documents (filename, storage_path, overall_status)
        VALUES (?, ?, 'uploaded')
        """,
        (filename, str(storage_path)),
    )
    document_id = int(cur.lastrowid)

    for strategy in STRATEGIES:
        db.execute(
            """
            INSERT INTO processing_runs (document_id, strategy, status)
            VALUES (?, ?, 'pending')
            """,
            (document_id, strategy),
        )

    db.commit()
    return document_id


def process_document_synchronously(document_id: int) -> None:
    """
    This is intentionally monolithic.

    In the refactored cloud architecture, this function issplit into:
    - event publication after upload
    - worker A consuming fixed-size queue
    - worker B consuming paragraph-aware queue
    - RDS updates from each worker
    """
    db = get_db()
    document = db.execute(
        "SELECT * FROM documents WHERE document_id = ?", (document_id,)
    ).fetchone()

    if document is None:
        raise ValueError(f"Document {document_id} not found")

    pdf_path = Path(document["storage_path"])

    try:
        text = extract_text_from_pdf(pdf_path)

        for strategy in STRATEGIES:
            run = db.execute(
                """
                SELECT * FROM processing_runs
                WHERE document_id = ? AND strategy = ?
                """,
                (document_id, strategy),
            ).fetchone()

            if run is None:
                raise ValueError(f"Processing run not found for {document_id}, {strategy}")

            start_time = time.time()

            db.execute(
                """
                UPDATE processing_runs
                SET status = 'processing', started_at = CURRENT_TIMESTAMP
                WHERE run_id = ?
                """,
                (run["run_id"],),
            )
            db.commit()

            chunks = chunk_text(text, strategy)

            db.execute("DELETE FROM chunks WHERE run_id = ?", (run["run_id"],))
            for index, chunk in enumerate(chunks):
                db.execute(
                    """
                    INSERT INTO chunks (run_id, chunk_index, chunk_text, char_count)
                    VALUES (?, ?, ?, ?)
                    """,
                    (run["run_id"], index, chunk, len(chunk)),
                )

            elapsed = time.time() - start_time
            average_length = sum(len(c) for c in chunks) / len(chunks) if chunks else 0

            db.execute(
                """
                UPDATE processing_runs
                SET status = 'completed',
                    chunk_count = ?,
                    average_chunk_length = ?,
                    processing_time_seconds = ?,
                    completed_at = CURRENT_TIMESTAMP,
                    error_message = NULL
                WHERE run_id = ?
                """,
                (len(chunks), average_length, elapsed, run["run_id"]),
            )

        db.execute(
            """
            UPDATE documents
            SET overall_status = 'ready'
            WHERE document_id = ?
            """,
            (document_id,),
        )
        db.commit()

    except Exception as exc:
        db.execute(
            """
            UPDATE documents
            SET overall_status = 'failed'
            WHERE document_id = ?
            """,
            (document_id,),
        )
        db.execute(
            """
            UPDATE processing_runs
            SET status = 'failed',
                error_message = ?,
                completed_at = CURRENT_TIMESTAMP
            WHERE document_id = ?
              AND status != 'completed'
            """,
            (str(exc), document_id),
        )
        db.commit()
        raise


# -----------------------------------------------------------------------------
# Query helpers
# -----------------------------------------------------------------------------


def load_document(document_id: int) -> sqlite3.Row | None:
    return get_db().execute(
        "SELECT * FROM documents WHERE document_id = ?", (document_id,)
    ).fetchone()


def load_processing_runs(document_id: int) -> list[sqlite3.Row]:
    return list(
        get_db().execute(
            """
            SELECT * FROM processing_runs
            WHERE document_id = ?
            ORDER BY strategy
            """,
            (document_id,),
        ).fetchall()
    )


def load_chunks(run_id: int) -> list[sqlite3.Row]:
    return list(
        get_db().execute(
            """
            SELECT * FROM chunks
            WHERE run_id = ?
            ORDER BY chunk_index
            """,
            (run_id,),
        ).fetchall()
    )


def ready_for_comparison(runs: list[sqlite3.Row]) -> bool:
    statuses = {run["strategy"]: run["status"] for run in runs}
    return all(statuses.get(strategy) == "completed" for strategy in STRATEGIES)


def save_retrieval_results(
    document_id: int,
    query_text: str,
    results_by_strategy: dict[str, list[dict[str, Any]]],
) -> int:
    db = get_db()
    cur = db.execute(
        """
        INSERT INTO retrieval_queries (document_id, query_text)
        VALUES (?, ?)
        """,
        (document_id, query_text),
    )
    query_id = int(cur.lastrowid)

    for strategy, results in results_by_strategy.items():
        run = db.execute(
            """
            SELECT run_id FROM processing_runs
            WHERE document_id = ? AND strategy = ?
            """,
            (document_id, strategy),
        ).fetchone()

        if run is None:
            continue

        for result in results:
            db.execute(
                """
                INSERT INTO retrieval_results
                    (query_id, run_id, rank, chunk_index, score, chunk_text)
                VALUES (?, ?, ?, ?, ?, ?)
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

    db.commit()
    return query_id


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------


@app.get("/")
def index():
    documents = get_db().execute(
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

    return render_template_string(INDEX_TEMPLATE, documents=documents)


@app.post("/upload")
def upload():
    uploaded_file: FileStorage | None = request.files.get("pdf")

    if uploaded_file is None or uploaded_file.filename == "":
        return "No PDF file selected", 400

    if not allowed_file(uploaded_file.filename):
        return "Only PDF files are supported", 400

    original_filename = secure_filename(uploaded_file.filename)
    timestamp = int(time.time())
    stored_filename = f"{timestamp}_{original_filename}"
    storage_path = app.config["UPLOAD_DIR"] / stored_filename
    uploaded_file.save(storage_path)

    document_id = create_document_record(original_filename, storage_path)

    # Monolithic behaviour: process immediately during the upload request.
    # This is exactly what students will later refactor away.
    process_document_synchronously(document_id)

    return redirect(url_for("document_detail", document_id=document_id))


@app.get("/documents/<int:document_id>")
def document_detail(document_id: int):
    document = load_document(document_id)
    if document is None:
        return "Document not found", 404

    runs = load_processing_runs(document_id)
    run_details = []
    sample_chunks_by_strategy: dict[str, list[sqlite3.Row]] = {}

    for run in runs:
        chunks = load_chunks(run["run_id"])
        sample_chunks_by_strategy[run["strategy"]] = chunks[:2]
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
    db = get_db()
    db.executescript(
        """
        DELETE FROM retrieval_results;
        DELETE FROM retrieval_queries;
        DELETE FROM chunks;
        DELETE FROM processing_runs;
        DELETE FROM documents;
        """
    )
    db.commit()

    for file_path in UPLOAD_DIR.glob("*"):
        if file_path.is_file():
            file_path.unlink()

    return redirect(url_for("index"))


# -----------------------------------------------------------------------------
# Templates
# -----------------------------------------------------------------------------


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
    .completed { background: #d9f7d9; }
    .failed { background: #ffd8d8; }
    .processing { background: #fff2cc; }
    .pending { background: #e8e8ff; }
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
<h1>Monolithic PDF RAG Comparison App</h1>

<div class="card">
    <h2>Upload a PDF</h2>
    <p>
        This monolithic version processes the PDF immediately inside the upload request.
        In the cloud assignment, this synchronous processing can be refactored into
        SNS, SQS, and separate worker services.
    </p>
    <form method="post" action="/upload" enctype="multipart/form-data">
        <input type="file" name="pdf" accept="application/pdf" required>
        <button type="submit">Upload and Process</button>
    </form>
</div>

<h2>Uploaded Documents</h2>

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

<form method="post" action="/reset" onsubmit="return confirm('Delete all local documents and database rows?')">
    <button type="submit">Reset Local Demo Data</button>
</form>
"""
)

DETAIL_TEMPLATE = (
    BASE_STYLE
    + """
<p><a href="/">← Back to documents</a></p>

<h1>Document {{ document.document_id }}: {{ document.filename }}</h1>

<div class="card">
    <p><strong>Status:</strong> <span class="status {{ document.overall_status }}">{{ document.overall_status }}</span></p>
    <p><strong>Stored file:</strong> {{ document.storage_path }}</p>
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
            <p><strong>Chunk {{ chunk.chunk_index }}</strong> — {{ chunk.char_count }} characters</p>
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
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)
