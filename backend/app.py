"""
FastAPI backend for the Book Recommender.
Exposes POST /api/search as an SSE (Server-Sent Events) streaming endpoint.
"""

import json
import logging
import os
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import psycopg2.extras
from psycopg2.pool import SimpleConnectionPool
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.query import run_query, setup_conn

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="Book Recommender API")

# Allow the Vite dev server and any localhost origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── DB connection pool ────────────────────────────────────────────────────────

_pool: SimpleConnectionPool | None = None


def get_pool() -> SimpleConnectionPool:
    global _pool
    if _pool is None:
        _pool = SimpleConnectionPool(minconn=1, maxconn=8, dsn=os.getenv("DATABASE_URL"))
    return _pool


def _get_conn():
    conn = get_pool().getconn()
    setup_conn(conn)
    return conn


def _release_conn(conn):
    get_pool().putconn(conn)


# ── Request schema ────────────────────────────────────────────────────────────


class SearchRequest(BaseModel):
    query: str


# ── SSE helpers ───────────────────────────────────────────────────────────────


def _sse_event(payload: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(payload)}\n\n"


# ── Endpoint ──────────────────────────────────────────────────────────────────


@app.post("/api/search")
async def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    def event_generator():
        conn = _get_conn()
        try:
            stream, sources = run_query(conn, req.query.strip())

            # Stream LLM tokens
            token_count = 0
            for chunk in stream:
                # Some providers send metadata chunks with empty choices — skip them
                if not chunk.choices:
                    continue
                token = chunk.choices[0].delta.content or ""
                if token:
                    token_count += 1
                    yield _sse_event({"type": "token", "content": token})

            if token_count == 0:
                log.warning("LLM stream produced no tokens for query: %r", req.query)

            # Serialisable sources (convert non-JSON types)
            safe_sources = []
            for src in sources:
                s = {}
                for k, v in src.items():
                    if isinstance(v, (str, int, float, bool, type(None))):
                        s[k] = v
                    else:
                        s[k] = str(v)
                safe_sources.append(s)

            yield _sse_event({"type": "sources", "data": safe_sources})
            yield _sse_event({"type": "done", "empty": token_count == 0})
        except Exception as exc:
            log.exception("Error during search")
            yield _sse_event({"type": "error", "message": str(exc)})
        finally:
            _release_conn(conn)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/book/{work_id}")
async def get_book(work_id: int):
    """Return full book details from the books table."""
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT work_id, original_title AS title, author, description, genres,
                       isbn, isbn13, original_publication_year, num_pages,
                       image_url, avg_rating, ratings_count, reviews_count,
                       text_reviews_count,
                       star5_ratings, star4_ratings, star3_ratings,
                       star2_ratings, star1_ratings
                FROM books
                WHERE work_id = %s
                """,
                (work_id,),
            )
            row = cur.fetchone()
    finally:
        _release_conn(conn)

    if row is None:
        raise HTTPException(status_code=404, detail="Book not found")

    result = {}
    for k, v in row.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            result[k] = v
        else:
            result[k] = str(v)
    return result


@app.get("/health")
def health():
    return {"status": "ok"}
