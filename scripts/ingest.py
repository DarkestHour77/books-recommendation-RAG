#!/usr/bin/env python3
"""
Ingest books and reviews from goodreads_db into book_embeddings (pgvector).

Usage:
    python scripts/ingest.py            # idempotent — skips already-ingested chunks
    python scripts/ingest.py --truncate # drop all rows first, then re-embed everything
"""

import argparse
import logging
import os
import sys

import psycopg2
import psycopg2.extras
import tiktoken
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBED_BATCH = 100
DB_BATCH = 500
MAX_WORKERS = 8

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
enc = tiktoken.get_encoding("cl100k_base")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)


def truncate_to_tokens(text: str, max_tokens: int = 8000) -> str:
    ids = enc.encode(text)
    return enc.decode(ids[:max_tokens]) if len(ids) > max_tokens else text


def book_chunk_text(row: dict) -> str:
    return (
        f"{row['title'] or ''} by {row['author'] or ''}. "
        f"{row['description'] or ''} "
        f"Genres: {row['genres'] or ''}"
    ).strip()


def review_chunk_text(row: dict) -> str:
    return (row["review_text"] or "").strip()


@retry(wait=wait_exponential(min=1, max=30), stop=stop_after_attempt(6))
def embed_batch(texts: list[str]) -> list[list[float]]:
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def chunks_for_book(row: dict) -> list[dict]:
    text = book_chunk_text(row)
    if not text:
        return []
    parts = splitter.split_text(text)
    records = []
    for i, part in enumerate(parts):
        records.append(
            {
                "record_type": "book",
                "work_id": row["work_id"],
                "review_id": None,
                "chunk_index": i,
                "chunk_text": truncate_to_tokens(part),
                "title": row["title"],
                "author": row["author"],
                "isbn": row["isbn"],
                "isbn13": row["isbn13"],
                "original_publication_year": row["original_publication_year"],
                "num_pages": row["num_pages"],
                "image_url": row["image_url"],
                "avg_rating": row["avg_rating"],
                "ratings_count": row["ratings_count"],
                "user_id": None,
                "started_at": None,
                "read_at": None,
                "date_added": None,
                "rating": None,
            }
        )
    return records


def chunks_for_review(row: dict) -> list[dict]:
    text = review_chunk_text(row)
    if not text:
        return []
    parts = splitter.split_text(text)
    records = []
    for i, part in enumerate(parts):
        records.append(
            {
                "record_type": "review",
                "work_id": row["work_id"],
                "review_id": row["review_id"],
                "chunk_index": i,
                "chunk_text": truncate_to_tokens(part),
                "title": row["title"],
                "author": row["author"],
                "isbn": row["isbn"],
                "isbn13": row["isbn13"],
                "original_publication_year": row["original_publication_year"],
                "num_pages": row["num_pages"],
                "image_url": row["image_url"],
                "avg_rating": row["avg_rating"],
                "ratings_count": row["ratings_count"],
                "user_id": row["user_id"],
                "started_at": row["started_at"],
                "read_at": row["read_at"],
                "date_added": row["date_added"],
                "rating": row["rating"],
            }
        )
    return records


INSERT_SQL = """
INSERT INTO book_embeddings (
    record_type, work_id, review_id, chunk_index, chunk_text, embedding,
    title, author, isbn, isbn13, original_publication_year, num_pages,
    image_url, avg_rating, ratings_count, user_id, started_at, read_at,
    date_added, rating
) VALUES %s
ON CONFLICT (record_type, work_id, review_id, chunk_index) DO NOTHING
"""

COLS = [
    "record_type", "work_id", "review_id", "chunk_index", "chunk_text", "embedding",
    "title", "author", "isbn", "isbn13", "original_publication_year", "num_pages",
    "image_url", "avg_rating", "ratings_count", "user_id", "started_at", "read_at",
    "date_added", "rating",
]


def flush(cur, conn, pending: list[dict]) -> int:
    if not pending:
        return 0

    texts = [r["chunk_text"] for r in pending]
    batches = [texts[i : i + EMBED_BATCH] for i in range(0, len(texts), EMBED_BATCH)]

    vectors: list[list[float]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(embed_batch, b): idx for idx, b in enumerate(batches)}
        results = {}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
        for idx in range(len(batches)):
            vectors.extend(results[idx])

    rows = [tuple(r[c] if c != "embedding" else vectors[i] for c in COLS)
            for i, r in enumerate(pending)]
    execute_values(cur, INSERT_SQL, rows)
    conn.commit()
    return len(rows)


def create_schema(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS book_embeddings (
                id                        BIGSERIAL PRIMARY KEY,
                record_type               TEXT NOT NULL,
                work_id                   BIGINT NOT NULL,
                review_id                 VARCHAR(50),
                chunk_index               INT NOT NULL,
                chunk_text                TEXT NOT NULL,
                embedding                 VECTOR(1536),
                title                     TEXT,
                author                    TEXT,
                isbn                      VARCHAR(20),
                isbn13                    VARCHAR(20),
                original_publication_year INT,
                num_pages                 INT,
                image_url                 TEXT,
                avg_rating                FLOAT,
                ratings_count             INT,
                user_id                   VARCHAR(50),
                started_at                TIMESTAMP,
                read_at                   TIMESTAMP,
                date_added                TIMESTAMP,
                rating                    INT,
                UNIQUE (record_type, work_id, review_id, chunk_index)
            );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS be_rating_idx ON book_embeddings (rating);")
        cur.execute("CREATE INDEX IF NOT EXISTS be_year_idx ON book_embeddings (original_publication_year);")
        cur.execute("CREATE INDEX IF NOT EXISTS be_pages_idx ON book_embeddings (num_pages);")
        cur.execute("CREATE INDEX IF NOT EXISTS be_author_idx ON book_embeddings (author);")
        cur.execute("CREATE INDEX IF NOT EXISTS be_type_idx ON book_embeddings (record_type);")
        conn.commit()
    log.info("Schema ready.")


def build_hnsw_index(conn):
    log.info("Building HNSW index — this may take several minutes...")
    with conn.cursor() as cur:
        cur.execute(
            "CREATE INDEX IF NOT EXISTS be_embedding_hnsw ON book_embeddings "
            "USING hnsw (embedding vector_cosine_ops);"
        )
        conn.commit()
    log.info("HNSW index built.")


def ingest_books(conn) -> int:
    total = 0
    pending: list[dict] = []

    with conn.cursor(name="books_cursor", cursor_factory=psycopg2.extras.RealDictCursor) as read_cur, \
         conn.cursor() as write_cur:
        read_cur.itersize = 500
        read_cur.execute("""
            SELECT work_id, original_title AS title, author, description, genres,
                   isbn, isbn13, original_publication_year, num_pages,
                   image_url, avg_rating, ratings_count
            FROM books
        """)
        count_cur = conn.cursor()
        count_cur.execute("SELECT COUNT(*) FROM books")
        total_books = count_cur.fetchone()[0]
        count_cur.close()

        for row in tqdm(read_cur, total=total_books, desc="books"):
            pending.extend(chunks_for_book(row))
            if len(pending) >= DB_BATCH:
                total += flush(write_cur, conn, pending)
                pending = []
                log.info("books: %d chunks upserted so far", total)

        if pending:
            total += flush(write_cur, conn, pending)

    return total


def ingest_reviews(conn) -> int:
    total = 0
    pending: list[dict] = []

    with conn.cursor(name="reviews_cursor", cursor_factory=psycopg2.extras.RealDictCursor) as read_cur, \
         conn.cursor() as write_cur:
        read_cur.itersize = 500
        read_cur.execute("""
            SELECT r.review_id, r.work_id, r.user_id, r.review_text,
                   r.started_at, r.read_at, r.date_added, r.rating,
                   b.original_title AS title, b.author, b.isbn, b.isbn13,
                   b.original_publication_year, b.num_pages, b.image_url,
                   b.avg_rating, b.ratings_count
            FROM reviews r
            JOIN books b ON b.work_id = r.work_id
            WHERE r.review_text IS NOT NULL AND length(r.review_text) > 0
        """)
        count_cur = conn.cursor()
        count_cur.execute(
            "SELECT COUNT(*) FROM reviews WHERE review_text IS NOT NULL AND length(review_text) > 0"
        )
        total_reviews = count_cur.fetchone()[0]
        count_cur.close()

        for row in tqdm(read_cur, total=total_reviews, desc="reviews"):
            pending.extend(chunks_for_review(row))
            if len(pending) >= DB_BATCH:
                total += flush(write_cur, conn, pending)
                pending = []
                log.info("reviews: %d chunks upserted so far", total)

        if pending:
            total += flush(write_cur, conn, pending)

    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--truncate", action="store_true",
                        help="Drop all existing embeddings before re-ingesting")
    args = parser.parse_args()

    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    register_vector(conn)

    create_schema(conn)

    if args.truncate:
        log.info("--truncate: dropping all rows from book_embeddings")
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE book_embeddings;")
            conn.commit()

    log.info("Ingesting book records...")
    book_chunks = ingest_books(conn)
    log.info("Book chunks upserted: %d", book_chunks)

    log.info("Ingesting review records...")
    review_chunks = ingest_reviews(conn)
    log.info("Review chunks upserted: %d", review_chunks)

    build_hnsw_index(conn)
    conn.close()
    log.info("Ingest complete. Total chunks: %d", book_chunks + review_chunks)


if __name__ == "__main__":
    main()
