from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from aws_clients import s3


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() == "pdf"


def download_pdf_from_s3(bucket: str, key: str) -> Path:
    temp_file = NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_path = Path(temp_file.name)
    temp_file.close()
    s3.download_file(bucket, key, str(temp_path))
    return temp_path


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


def retrieve_top_chunks(chunks: list[dict[str, Any]], query: str, top_k: int = 3) -> list[dict[str, Any]]:
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
