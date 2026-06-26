"""
Supervisor entry point. Re-exports the FastAPI app from /app/app.py
so that the platform-managed supervisor can run it as `server:app`
from the /app/backend directory.
"""
import sys
import os

# Make /app importable
sys.path.insert(0, "/app")
os.chdir("/app")

from app import app  # noqa: F401  (re-exported for uvicorn)
