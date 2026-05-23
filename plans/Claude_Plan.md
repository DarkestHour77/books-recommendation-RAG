# Book Recommendation RAG System — Implementation Plan

## Context

A Goodreads-based RAG system using PostgreSQL (`goodreads_db`) as the sole data store for both
raw book/review data and vector embeddings (via `pgvector`). A Streamlit UI is the user-facing
interface. All AI calls go through environment variables defined in `.env`.

---

## Technology Decisions (locked)

| Concern | Choice |
|---|---|
| Source data | PostgreSQL `goodreads_db` (`books` + `reviews` tables) |
| Vector store | `pgvector` extension on same `goodreads_db` |
| Embedding model | `text-embedding-3-small` (OpenAI, 1536 dims) |
| LLM | `minimax/minimax-m2.5:free` via OpenRouter |
| Reranker | Cohere Rerank API |
| Spell checker | `pyspellchecker` |
| UI | Streamlit |

---

## Architecture: Single PostgreSQL Store

```
goodreads_db
├── books             (source table — work-level metadata)
├── reviews           (source table — per-user reviews, FK to books.work_id)
└── book_embeddings   (pgvector table — chunks + vectors + metadata)
```

### `book_embeddings` DDL

Two **record types** live in this table:
- `record_type = 'book'` — one (or few) chunks per work, embedding `title + author + description + genres`
- `record_type = 'review'` — one or more chunks per review, embedding `review_text`

This avoids duplicating book description across every review of the same work and prevents
review vectors from being diluted by boilerplate metadata.

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE book_embeddings (
    id                        BIGSERIAL PRIMARY KEY,
    record_type               TEXT NOT NULL,    -- 'book' | 'review'
    work_id                   BIGINT NOT NULL,  -- always present; joins to books.work_id
    review_id                 VARCHAR(50),      -- NULL for record_type='book'
    chunk_index               INT NOT NULL,     -- position of this chunk within the record
    chunk_text                TEXT NOT NULL,
    embedding                 VECTOR(1536),     -- text-embedding-3-small output
    -- metadata for filtering and display (denormalised from books/reviews)
    title                     TEXT,
    author                    TEXT,
    isbn                      VARCHAR(20),
    isbn13                    VARCHAR(20),
    original_publication_year INT,
    num_pages                 INT,
    image_url                 TEXT,
    avg_rating                FLOAT,
    ratings_count             INT,
    user_id                   VARCHAR(50),      -- NULL for record_type='book'
    started_at                TIMESTAMP,
    read_at                   TIMESTAMP,
    date_added                TIMESTAMP,
    rating                    INT,              -- NULL for record_type='book'
    UNIQUE (record_type, work_id, review_id, chunk_index)
);

-- Create the HNSW index AFTER bulk ingestion, not before — incremental index
-- updates during insert are significantly slower than a single post-load build.
-- Run this once ingest.py finishes:
CREATE INDEX ON book_embeddings USING hnsw (embedding vector_cosine_ops);

-- B-tree indexes on filter columns. HNSW only accelerates the vector ORDER BY;
-- the WHERE clause is evaluated separately and benefits from regular indexes.
CREATE INDEX ON book_embeddings (rating);
CREATE INDEX ON book_embeddings (original_publication_year);
CREATE INDEX ON book_embeddings (num_pages);
CREATE INDEX ON book_embeddings (author);
CREATE INDEX ON book_embeddings (record_type);
```

### Cost / volume estimate

Rough sizing before you commit to a full run:
- Goodreads dump is ~500 MB compressed (~2–3 GB uncompressed); on the order of ~2M reviews and
  ~500k unique works after dedup. Numbers below are illustrative — measure on your actual data.
- text-embedding-3-small is $0.02 / 1M tokens. At ~250 tokens per review chunk × 2M reviews ≈
  500M tokens ≈ **~$10** for review embeddings, plus a few cents for book chunks.
- A serial ingest at ~100 chunks/request and ~1 req/s would take **>5 hours**. Parallelise
  (see Step 4) to bring this to tens of minutes.
- Cohere Rerank: only runs at query time, top-20 docs per query. Negligible cost for personal use.

---

## Ingestion Pipeline (`scripts/ingest.py`)

### Step 1 — Stream book + review rows from `goodreads_db`

The source schema has two tables: `books` (work-level) and `reviews` (per-user). Ingest each
record type with its own streaming query.

Use a server-side cursor so the full table is never loaded into memory.
Use `RealDictCursor` so rows are accessible by column name, not position index.
Call `load_dotenv()` at the top of the script before reading any env vars.
Register the `VECTOR` type so Python lists serialise correctly:

```python
import os
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(os.getenv("DATABASE_URL"))
register_vector(conn)   # required — without this, inserting a list as VECTOR(1536) fails

# Stream books
with conn.cursor(name="books_cursor", cursor_factory=psycopg2.extras.RealDictCursor) as cur:
    cur.itersize = 500
    cur.execute("""
        SELECT work_id, original_title AS title, author, description, genres,
               isbn, isbn13, original_publication_year, num_pages,
               image_url, avg_rating, ratings_count
        FROM books
    """)
    for row in cur:
        process_book(row)

# Stream reviews joined to their book for denormalised metadata
with conn.cursor(name="reviews_cursor", cursor_factory=psycopg2.extras.RealDictCursor) as cur:
    cur.itersize = 500
    cur.execute("""
        SELECT r.review_id, r.work_id, r.user_id, r.review_text,
               r.started_at, r.read_at, r.date_added, r.rating,
               b.original_title AS title, b.author, b.isbn, b.isbn13,
               b.original_publication_year, b.num_pages, b.image_url,
               b.avg_rating, b.ratings_count
        FROM reviews r
        JOIN books b ON b.work_id = r.work_id
        WHERE r.review_text IS NOT NULL AND length(r.review_text) > 0
    """)
    for row in cur:
        process_review(row)
```

**Deduplication / re-ingest:** Default behaviour is idempotent upsert via
`INSERT ... ON CONFLICT (record_type, work_id, review_id, chunk_index) DO NOTHING`. To force a
full re-embed, pass `--truncate` to drop existing rows before insert.

### Step 2 — Build chunk text per record

**Book records** — concatenate book-level semantic fields only:
```python
def book_chunk_text(row):
    return (
        f"{row['title'] or ''} by {row['author'] or ''}. "
        f"{row['description'] or ''} "
        f"Genres: {row['genres'] or ''}"
    ).strip()
```

**Review records** — embed the review text on its own (book metadata still travels in the
denormalised columns, but isn't part of the embedded text):
```python
def review_chunk_text(row):
    return (row['review_text'] or '').strip()
```

Skip rows where the resulting chunk text is empty.

### Step 3 — Split long records into sub-chunks

Use `RecursiveCharacterTextSplitter` with 10-20% overlap. Every sub-chunk from the same record
inherits the same `(record_type, work_id, review_id)` plus a `chunk_index` for position:

```python
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_text(text)
for i, chunk in enumerate(chunks):
    store(chunk, chunk_index=i, record_type=record_type,
          work_id=row['work_id'], review_id=row.get('review_id'), ...)
```

**Token-length guard.** `text-embedding-3-small` rejects inputs longer than 8191 tokens.
Char-based splitting is approximate; before sending to the embeddings API, truncate any chunk
that exceeds the token limit:
```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
def truncate(text, max_tokens=8000):
    ids = enc.encode(text)
    return enc.decode(ids[:max_tokens]) if len(ids) > max_tokens else text
```

### Step 4 — Embed each chunk (batched + parallel)

Call `text-embedding-3-small` for every chunk. Batch up to 100 chunks per API call and run
multiple batches concurrently — a serial loop over millions of reviews takes hours.

```python
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(wait=wait_exponential(min=1, max=30), stop=stop_after_attempt(6))
def embed_batch(texts):
    resp = openai_client.embeddings.create(model=os.getenv("EMBEDDING_MODEL"), input=texts)
    return [d.embedding for d in resp.data]

with ThreadPoolExecutor(max_workers=8) as pool:
    for vectors in pool.map(embed_batch, chunk_batches):
        ...
```

The same embedding model **must** be used at query time.

### Step 5 — Upsert into `book_embeddings`

Write each embedded chunk (text + vector + all metadata columns) using
`execute_values` for bulk insert. Commit in batches of ~500 rows. Use `tqdm` for progress
and the `logging` module (not `print`) for batch/error reporting.

```python
import logging
from tqdm import tqdm
from psycopg2.extras import execute_values

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

for batch in tqdm(batches, desc="upserting embeddings"):
    execute_values(cur, "INSERT INTO book_embeddings (...) VALUES %s "
                        "ON CONFLICT (record_type, work_id, review_id, chunk_index) DO NOTHING",
                   batch)
    conn.commit()
```

---

## Query Pipeline (`scripts/query.py` / Streamlit app)

### Step 1 — Streamlit UI

`app.py` at the project root runs the Streamlit interface:
- Text input box for the user's query
- Results section showing the streamed LLM answer + source cards (title, author, avg_rating, image)

Call `load_dotenv()` at the top of `app.py`. Use a **psycopg2 connection pool** rather than a
single shared connection — Streamlit can serve concurrent reruns/users, and a single connection
will serialise them or break under contention. Cache the pool with `st.cache_resource`:

```python
from dotenv import load_dotenv
from psycopg2.pool import SimpleConnectionPool
from pgvector.psycopg2 import register_vector

load_dotenv()

@st.cache_resource
def get_db_pool():
    return SimpleConnectionPool(minconn=1, maxconn=8, dsn=os.getenv("DATABASE_URL"))

def with_conn():
    pool = get_db_pool()
    conn = pool.getconn()
    register_vector(conn)
    try:
        yield conn
    finally:
        pool.putconn(conn)

@st.cache_resource
def get_spell_checker():
    from spellchecker import SpellChecker
    return SpellChecker()
```

Two OpenAI clients are needed — one for embeddings (default OpenAI base URL) and one for the
LLM (OpenRouter base URL):
```python
from openai import OpenAI
embed_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
llm_client = OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"),
                    base_url=os.getenv("OPENROUTER_BASE_URL"))
```

### Step 2 — Clean and spell-correct the query

`pyspellchecker` will "correct" proper nouns (author names, book titles) — only apply it to
lowercase tokens and skip anything capitalised or containing digits:

```python
def correct(query, spell):
    out = []
    for tok in query.split():
        if tok[:1].isupper() or any(c.isdigit() for c in tok) or tok.lower() in spell:
            out.append(tok)
        else:
            out.append(spell.correction(tok) or tok)
    return " ".join(out)
```

### Step 3 — Rewrite + extract filters in one LLM call

Combining query rewrite and filter extraction into a single structured-output call halves
query latency vs. doing two separate round-trips:

```python
prompt = f"""
You are a query preprocessor for a book recommendation engine.
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

Query: {user_query}
"""
```

Wrap the JSON parse in a try/except — fall back to using the original query with empty filters
if the LLM returns malformed output:
```python
import json
try:
    parsed = json.loads(llm_response)
    rewritten = parsed["rewritten_query"]
    filters = parsed.get("filters", {})
except (json.JSONDecodeError, KeyError, ValueError):
    rewritten, filters = user_query, {}
```

### Step 4 — Embed the rewritten query

Call `text-embedding-3-small` with the rewritten query (same model as ingestion).

### Step 5 — pgvector search with metadata filters

Run HNSW approximate nearest-neighbour search, combining semantic similarity with hard filters.

**HNSW + WHERE caveat.** pgvector's HNSW index does post-filtering by default — a query like
`WHERE rating >= 4 ORDER BY embedding <=> ... LIMIT 20` returns the top-20 by vector distance
*then* drops rows that fail the WHERE, so you can end up with fewer than 20 hits. Two
mitigations:
1. Enable iterative scan (pgvector ≥ 0.8): `SET LOCAL hnsw.iterative_scan = 'relaxed_order';`
2. Over-fetch — request `LIMIT 200` and trim after the WHERE/rerank.

Build the `WHERE` clause dynamically. Only add a clause when the filter value is non-null,
otherwise the query excludes all rows where that column is NULL:

```python
conditions = ["1=1"]
params = {"query_vector": query_embedding}

if filters.get("rating"):
    conditions.append("rating >= %(rating)s")
    params["rating"] = filters["rating"]
if filters.get("year_min"):
    conditions.append("original_publication_year >= %(year_min)s")
    params["year_min"] = filters["year_min"]
if filters.get("year_max"):
    conditions.append("original_publication_year <= %(year_max)s")
    params["year_max"] = filters["year_max"]
if filters.get("max_pages"):
    conditions.append("num_pages <= %(max_pages)s")
    params["max_pages"] = filters["max_pages"]
if filters.get("author"):
    conditions.append("author ILIKE %(author)s")
    params["author"] = f"%{filters['author']}%"
# genre filter — substring match on the denormalised genres text isn't in the schema yet;
# if you want it, denormalise books.genres into book_embeddings or do a JOIN back to books.

where_clause = " AND ".join(conditions)
sql = f"""
    SET LOCAL hnsw.iterative_scan = 'relaxed_order';
    SELECT chunk_text, record_type, work_id, review_id, title, author, isbn,
           avg_rating, image_url, original_publication_year
    FROM book_embeddings
    WHERE {where_clause}
    ORDER BY embedding <=> %(query_vector)s
    LIMIT 200
"""
cur.execute(sql, params)
```

### Step 6 — Rerank with Cohere

Send the top-N candidates + original query to the Cohere Rerank API. Trim to the top 3-5:
```python
import cohere
co = cohere.Client(os.getenv("COHERE_API_KEY"))
results = co.rerank(model=os.getenv("COHERE_RERANK_MODEL"),
                    query=user_query, documents=chunks, top_n=5)
```

### Step 7 — LLM synthesis (streamed)

Assemble the RAG prompt and call the LLM via OpenRouter. Stream the response to Streamlit so
the user sees tokens as they arrive:

```
System: You are a helpful book recommendation assistant. Answer only using the provided
        context. If the answer isn't in the context, say you don't know.

Context:
  [chunk 1 text]
  [chunk 2 text]
  ...

Question: {user_query}
```

```python
stream = llm_client.chat.completions.create(
    model=os.getenv("LLM_MODEL"), messages=[...], stream=True,
)
st.write_stream((c.choices[0].delta.content or "" for c in stream))
```

### Step 8 — Display response in Streamlit

Show the streamed LLM answer plus source attribution cards for each reranked chunk:
`title`, `author`, `work_id`, `avg_rating`, `image_url` (rendered with `st.image`).

---

## Evaluation harness

Keep a tiny `eval/eval_queries.yaml` checked into the repo with a handful of queries and
expected `work_id`s. A `scripts/eval.py` runs each query end-to-end and reports recall@5 vs.
the expected set. This is enough to catch regressions when tuning chunk size, top-k, or
rerank parameters.

```yaml
- query: "books like Project Hail Mary"
  expect_work_ids: [12345, 67890]
- query: "short sci-fi novels under 300 pages"
  expect_work_ids: [...]
```

---

## Missing Infrastructure to Create

### `requirements.txt`
```
langchain
langchain-community
langchain-text-splitters
openai                  # text-embedding-3-small + OpenRouter-compatible client
cohere                  # Rerank API
psycopg2-binary
pgvector                # registers VECTOR type with psycopg2; no separate vector DB needed
pyspellchecker
python-dotenv
streamlit
tiktoken                # token-aware truncation before embedding
tenacity                # retry/backoff for API calls
tqdm                    # ingest progress bar
```

### `.env.example`
```
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/goodreads_db

# Embedding (OpenAI)
OPENAI_API_KEY=
EMBEDDING_MODEL=text-embedding-3-small

# LLM via OpenRouter
OPENROUTER_API_KEY=
LLM_MODEL=minimax/minimax-m2.5:free
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Reranker (Cohere)
COHERE_API_KEY=
COHERE_RERANK_MODEL=rerank-english-v3.0
```

### File structure
```
book-recomendation/
├── app.py               # Streamlit UI entry point
├── .env
├── .env.example
├── requirements.txt
├── PLAN.md
├── assests/
│   └── (raw data files — reference only, not read at runtime)
├── eval/
│   └── eval_queries.yaml
└── scripts/
    ├── ingest.py        # DB stream → chunk → embed → upsert book_embeddings
    ├── query.py         # rewrite+extract → embed → pgvector → rerank → LLM
    └── eval.py          # run eval_queries.yaml and report recall@k
```

---

## What Was Already Correct in the Original Plan

- Chunk text vs metadata split — well-defined
- HNSW index — now via `pgvector`, not a separate DB
- Query rewriting prompt — solid (now merged with filter extraction)
- RAG prompt structure (system + context + question)
- Source attribution in the response
- Metadata filtering (semantic search + hard SQL constraints)
- Same embedding model at ingest and query time
