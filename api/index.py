"""
Vercel serverless function entry point.
Vercel requires Python functions to live inside the `api/` directory.
This file re-exports the FastAPI `app` from the backend package.
"""

import sys
import os

# Use abspath so this works whether __file__ is relative or absolute.
_backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend")
sys.path.insert(0, os.path.normpath(_backend_dir))

from app import app  # noqa: F401  — Vercel detects the ASGI `app` variable
