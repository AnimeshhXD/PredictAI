"""
app/streamlit_app.py
====================
Entrypoint wrapper redirecting to root app.py for Streamlit execution.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import app

if __name__ == "__main__":
    app.main()
