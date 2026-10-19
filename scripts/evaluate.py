"""
scripts/evaluate.py
===================
Entrypoint wrapper redirecting to run_experiments.py for evaluating baselines,
PredictAI, and ablations.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is importable
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from scripts.run_experiments import main

if __name__ == "__main__":
    main()
