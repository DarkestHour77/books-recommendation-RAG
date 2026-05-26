#!/usr/bin/env python3
"""
Test that the configured embedding model is reachable via OpenRouter
and print the output vector dimension.

Usage:
    python scripts/test_embedding.py
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

model = os.getenv("EMBEDDING_MODEL", "perplexity/pplx-embed-v1-0.6b")
api_key = os.getenv("OPENROUTER_API_KEY")
base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

if not api_key:
    raise SystemExit("OPENROUTER_API_KEY is not set in .env")

client = OpenAI(api_key=api_key, base_url=base_url)

print(f"Model  : {model}")
print(f"API URL: {base_url}")
print("Sending test embedding request...")

resp = client.embeddings.create(model=model, input=["hello world"])
vector = resp.data[0].embedding

print(f"Dimension: {len(vector)}")
print(f"First 5 values: {vector[:5]}")
print("\nUpdate ingest.py line 188 to: VECTOR(%d)" % len(vector))
