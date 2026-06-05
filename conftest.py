"""Pytest configuration. Adds repo root to PYTHONPATH so `api`/`models` import."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
