"""
Query pipeline: rewrite + filter → embed → pgvector → rerank → LLM.
Importable by app.py; can also be called directly for debugging.
"""

import json
import logging
import os

import cohere
import psycopg2.extras
from dotenv import load_dotenv
from openai import OpenAI
from pgvector.psycopg2 import register_vector
from spellchecker import SpellChecker

load_dotenv()

log = logging.getLogger(__name__)

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "minimax/minimax-m2.5:free")
COHERE_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0")
VECTOR_FETCH = 200
RERANK_TOP_N = 5

embed_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
llm_client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
)
cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
spell = SpellChecker()


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
    try:
        parsed = json.loads(raw)
        rewritten = parsed.get("rewritten_query") or user_query
        filters = parsed.get("filters") or {}
    except (json.JSONDecodeError, KeyError, ValueError):
        log.warning("LLM returned malformed JSON for rewrite/extract; using raw query")
        rewritten, filters = user_query, {}
    return rewritten, filters


def embed_query(text: str) -> list[float]:
    resp = embed_client.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding


def vector_search(conn, query_embedding: list[float], filters: dict) -> list[dict]:
    conditions = ["1=1"]
    params: dict = {"query_vector": query_embedding}

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
    sql = f"""
        SELECT chunk_text, record_type, work_id, review_id,
               title, author, isbn, avg_rating, image_url,
               original_publication_year, rating
        FROM book_embeddings
        WHERE {where_clause}
        ORDER BY embedding <=> %(query_vector)s
        LIMIT {VECTOR_FETCH}
    """

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        try:
            cur.execute("SET LOCAL hnsw.iterative_scan = 'relaxed_order';")
        except Exception:
            pass
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


def rerank(user_query: str, candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []
    docs = [c["chunk_text"] for c in candidates]
    results = cohere_client.rerank(
        model=COHERE_MODEL,
        query=user_query,
        documents=docs,
        top_n=min(RERANK_TOP_N, len(docs)),
    )
    return [candidates[r.index] for r in results.results]


def build_rag_prompt(user_query: str, chunks: list[dict]) -> list[dict]:
    context = "\n\n".join(
        f"[{i+1}] {c['chunk_text']}" for i, c in enumerate(chunks)
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a helpful book recommendation assistant. "
                "Answer only using the provided context. "
                "If the answer isn't in the context, say you don't know."
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

    top_chunks = rerank(user_query, candidates)
    log.info("Reranked to %d chunks", len(top_chunks))

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
    register_vector(conn)

    q = " ".join(sys.argv[1:]) or "good science fiction novels"
    stream, chunks = run_query(conn, q)
    print("\n--- Answer ---")
    for part in stream:
        print(part.choices[0].delta.content or "", end="", flush=True)
    print("\n\n--- Sources ---")
    for c in chunks:
        print(f"  {c['title']} by {c['author']} | rating={c['avg_rating']}")

    pool.putconn(conn)
