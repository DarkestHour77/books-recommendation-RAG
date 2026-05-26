"""
Query pipeline: rewrite + filter → embed → pgvector → rerank → LLM.
Importable by app.py; can also be called directly for debugging.
"""

import json
import logging
import os

import cohere
import numpy as np
import psycopg2.extras
from dotenv import load_dotenv
from openai import OpenAI
from pgvector.psycopg2 import register_vector
from spellchecker import SpellChecker

load_dotenv()

log = logging.getLogger(__name__)

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "perplexity/pplx-embed-v1-0.6b")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
COHERE_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0")
VECTOR_FETCH_BOOKS = 100
VECTOR_FETCH_REVIEWS = 100
RERANK_TOP_N = 5

embed_client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
)
llm_client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
)
cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
spell = SpellChecker()


def setup_conn(conn):
    """Ensure the vector extension exists and register the type. Call after every psycopg2.connect()."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
    register_vector(conn)


def spell_correct(query: str) -> str:
    tokens = query.split()
    corrected = []
    for tok in tokens:
        clean = tok.strip(".,!?\"'")
        if clean[:1].isupper() or any(c.isdigit() for c in clean) or clean.lower() in spell:
            corrected.append(tok)
        else:
            fixed = spell.correction(clean)
            corrected.append(tok.replace(clean, fixed) if fixed else tok)
    return " ".join(corrected)


def rewrite_and_extract(user_query: str) -> tuple[str, dict]:
    """Single LLM call: returns (rewritten_query, filters_dict)."""
    prompt = f"""You are a query preprocessor for a book recommendation engine.
Return JSON only, no commentary, with this exact shape:
{{
  "rewritten_query": "<concise, retrieval-friendly rewrite of the user query>",
  "filters": {{
    "rating": <int 1-5 or null>,
    "genre": <string or null>,
    "year_min": <int or null>,
    "year_max": <int or null>,
    "max_pages": <int or null>,
    "author": <string or null>
  }}
}}

Query: {user_query}"""

    resp = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = resp.choices[0].message.content or ""
    # Strip markdown code fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[-2] if cleaned.count("```") >= 2 else cleaned
        cleaned = cleaned.lstrip("json").strip()
    # Extract first {...} block in case of leading/trailing text
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end != -1:
        cleaned = cleaned[start:end + 1]
    try:
        parsed = json.loads(cleaned)
        rewritten = parsed.get("rewritten_query") or user_query
        filters = parsed.get("filters") or {}
    except (json.JSONDecodeError, KeyError, ValueError):
        log.warning("LLM returned malformed JSON for rewrite/extract; using raw query. Raw: %s", raw[:200])
        rewritten, filters = user_query, {}
    return rewritten, filters


def embed_query(text: str) -> list[float]:
    resp = embed_client.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding


def vector_search(conn, query_embedding: list[float], filters: dict) -> list[dict]:
    conditions = ["1=1"]
    params: dict = {"query_vector": np.array(query_embedding)}

    if filters.get("rating"):
        conditions.append("rating >= %(rating)s")
        params["rating"] = int(filters["rating"])
    if filters.get("year_min"):
        conditions.append("original_publication_year >= %(year_min)s")
        params["year_min"] = int(filters["year_min"])
    if filters.get("year_max"):
        conditions.append("original_publication_year <= %(year_max)s")
        params["year_max"] = int(filters["year_max"])
    if filters.get("max_pages"):
        conditions.append("num_pages <= %(max_pages)s")
        params["max_pages"] = int(filters["max_pages"])
    if filters.get("author"):
        conditions.append("author ILIKE %(author)s")
        params["author"] = f"%{filters['author']}%"

    where_clause = " AND ".join(conditions)
    base_sql = """
        SELECT chunk_text, record_type, work_id, review_id,
               title, author, isbn, avg_rating, image_url,
               original_publication_year, rating
        FROM book_embeddings
        WHERE record_type = %(record_type)s AND {where}
        ORDER BY embedding <=> %(query_vector)s
        LIMIT %(limit)s
    """

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        try:
            cur.execute("SAVEPOINT before_hnsw_hint;")
            cur.execute("SET LOCAL hnsw.iterative_scan = 'relaxed_order';")
        except Exception:
            cur.execute("ROLLBACK TO SAVEPOINT before_hnsw_hint;")

        results = []
        for rtype, limit in [("book", VECTOR_FETCH_BOOKS), ("review", VECTOR_FETCH_REVIEWS)]:
            cur.execute(
                base_sql.format(where=where_clause),
                {**params, "record_type": rtype, "limit": limit},
            )
            results.extend(dict(r) for r in cur.fetchall())
        return results


def rerank(user_query: str, candidates: list[dict], top_n: int = RERANK_TOP_N) -> list[dict]:
    if not candidates:
        return []
    docs = [c["chunk_text"] for c in candidates]
    results = cohere_client.rerank(
        model=COHERE_MODEL,
        query=user_query,
        documents=docs,
        top_n=min(top_n, len(docs)),
    )
    return [candidates[r.index] for r in results.results]


def filter_mentioned_books(
    user_query: str, chunks: list[dict], target_n: int = RERANK_TOP_N
) -> list[dict]:
    """Ask the LLM which books in `chunks` are already referenced in the user query
    and should therefore be excluded from recommendations.

    `chunks` is expected to be a ranked pool larger than `target_n` so that when
    books are excluded the gap is filled by the next-best candidates already present
    in the pool.  The function always returns exactly `target_n` results (or fewer
    only when the pool itself is smaller than `target_n`).

    If the LLM call fails, the first `target_n` chunks are returned unchanged.
    """
    if not chunks:
        return chunks

    titles = [c.get("title", "") for c in chunks]
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(titles))

    prompt = f"""You are a filter for a book recommendation engine.

The user asked: "{user_query}"

The following books have been retrieved as candidates (ranked best-first):
{numbered}

Which of these books is the user already referring to or asking about in their query?
These books should NOT be recommended back to them.

Return JSON only, no commentary, with this exact shape:
{{
  "exclude_indices": [<1-based index of book to exclude>, ...]
}}

If no books should be excluded, return: {{"exclude_indices": []}}"""

    try:
        resp = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = resp.choices[0].message.content or ""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[-2] if cleaned.count("```") >= 2 else cleaned
            cleaned = cleaned.lstrip("json").strip()
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1:
            cleaned = cleaned[start:end + 1]
        parsed = json.loads(cleaned)
        exclude = set(int(i) - 1 for i in parsed.get("exclude_indices", []))
        log.info("LLM decided to exclude book indices (0-based): %s", exclude)
    except Exception as exc:
        log.warning("filter_mentioned_books LLM call failed (%s); skipping filter.", exc)
        return chunks[:target_n]

    # Walk the ranked pool in order; skip excluded books, keep until we hit target_n.
    kept = []
    for i, c in enumerate(chunks):
        if i in exclude:
            log.info("Excluding mentioned book: %s", c.get("title"))
            continue
        kept.append(c)
        if len(kept) == target_n:
            break

    if not kept:
        log.warning("LLM excluded all books in pool; falling back to top-%d.", target_n)
        return chunks[:target_n]

    return kept


def build_rag_prompt(user_query: str, chunks: list[dict]) -> list[dict]:
    context_parts = []
    for i, c in enumerate(chunks):
        if c["record_type"] == "book":
            source_label = "Book Info"
            header = ""
        else:
            source_label = "Reader Review"
            header = f"[Review of: {c.get('title', '?')} by {c.get('author', '?')}]\n"
        context_parts.append(f"[{i+1}] [{source_label}] {header}{c['chunk_text']}")
    context = "\n\n".join(context_parts)
    return [
        {
            "role": "system",
            "content": (
                "You are a helpful book recommendation assistant. "
                "Answer only using the provided context. "
                "For every book in the context (numbered [1], [2], …) write exactly one section. "
                "Begin each section with a bold heading on its own line in this format: "
                "**[number]. Title by Author** "
                "then write a short recommendation paragraph for that book. "
                "Present the sections in the same numbered order as the context. "
                "Do not skip any book. "
                "If a book does not fit the query, still include it and briefly explain why it might not match."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {user_query}",
        },
    ]


def run_query(conn, user_query: str) -> tuple[object, list[dict]]:
    """
    Returns (stream, top_chunks).
    stream: openai streaming completion — iterate with:
        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
    top_chunks: list of source metadata dicts for attribution cards.
    """
    cleaned = spell_correct(user_query)
    rewritten, filters = rewrite_and_extract(cleaned)
    log.info("Rewritten query: %s | Filters: %s", rewritten, filters)

    query_vec = embed_query(rewritten)
    candidates = vector_search(conn, query_vec, filters)
    log.info("Vector search returned %d candidates", len(candidates))

    # Keep only the highest-ranked chunk per book so Cohere sees diverse results
    seen: dict = {}
    for c in candidates:
        wid = c["work_id"]
        if wid not in seen:
            seen[wid] = c
    candidates = list(seen.values())
    log.info("After dedup: %d unique books", len(candidates))

    # Rerank a larger pool (2× target) so we have backfills ready if
    # filter_mentioned_books strips books the user already referenced.
    rerank_fetch = RERANK_TOP_N * 2
    ranked_pool = rerank(user_query, candidates, top_n=rerank_fetch)
    log.info("Reranked to %d chunks (extended pool)", len(ranked_pool))

    top_chunks = filter_mentioned_books(user_query, ranked_pool, target_n=RERANK_TOP_N)
    log.info("After mention-filter: %d chunks", len(top_chunks))

    messages = build_rag_prompt(user_query, top_chunks)
    stream = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        stream=True,
        temperature=0.3,
    )
    return stream, top_chunks


if __name__ == "__main__":
    import sys
    from psycopg2.pool import SimpleConnectionPool

    logging.basicConfig(level=logging.INFO)
    pool = SimpleConnectionPool(1, 2, dsn=os.getenv("DATABASE_URL"))
    conn = pool.getconn()
    setup_conn(conn)

    q = " ".join(sys.argv[1:]) or "good science fiction novels"
    stream, chunks = run_query(conn, q)
    print("\n--- Answer ---")
    for part in stream:
        print(part.choices[0].delta.content or "", end="", flush=True)
    print("\n\n--- Sources ---")
    for c in chunks:
        print(f"  {c['title']} by {c['author']} | rating={c['avg_rating']}")

    pool.putconn(conn)
