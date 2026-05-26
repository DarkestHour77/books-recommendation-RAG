"""
Evaluate recall@k against eval/eval_queries.yaml.

Usage:
    python scripts/eval.py                  # default recall@5
    python scripts/eval.py --top-k 3
"""

import argparse
import logging
import os
import sys

import yaml
from dotenv import load_dotenv
from psycopg2.pool import SimpleConnectionPool

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BACKEND_DIR)
from scripts.query import run_query, setup_conn

load_dotenv(os.path.join(_BACKEND_DIR, '.env'))
logging.basicConfig(level=logging.WARNING)

EVAL_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "eval", "eval_queries.yaml")


def recall_at_k(expected: list, retrieved: list, k: int) -> float:
    if not expected:
        return 0.0
    top_k_ids = {str(c["work_id"]) for c in retrieved[:k]}
    hits = sum(1 for e in expected if str(e) in top_k_ids)
    return hits / len(expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    with open(EVAL_FILE) as f:
        cases = yaml.safe_load(f)

    pool = SimpleConnectionPool(1, 2, dsn=os.getenv("DATABASE_URL"))
    conn = pool.getconn()
    setup_conn(conn)

    scores = []
    for case in cases:
        query = case["query"]
        expected = case.get("expect_work_ids") or []
        _, chunks = run_query(conn, query)
        score = recall_at_k(expected, chunks, args.top_k)
        scores.append(score)
        status = f"{score:.2f}" if expected else "N/A (no expected ids)"
        print(f"  recall@{args.top_k}={status:6}  query={query!r}")

    if any(case.get("expect_work_ids") for case in cases):
        filled = [s for case, s in zip(cases, scores) if case.get("expect_work_ids")]
        print(f"\nMean recall@{args.top_k}: {sum(filled)/len(filled):.3f}")
    else:
        print("\nNo expected_work_ids set — fill in eval/eval_queries.yaml to get scores.")

    pool.putconn(conn)


if __name__ == "__main__":
    main()
