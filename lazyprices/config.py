"""Paths and constants shared across modules."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("LAZYPRICES_ROOT", Path(__file__).resolve().parent.parent))
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

LUCA_CACHE = Path(
    os.environ.get("LUCA_CACHE", "/home/queipo/luca-ai-lab/data/cache/corpus_facts")
)

SEC_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT", "Ignacio Queipo de Llano i.queipodellano@iese.net"
)
SEC_MAX_RPS = 8.0

# Fiscal years covered by the 10-K download. 2007 is downloaded only so that
# the FY2008 filing has a predecessor to be compared with.
FIRST_FISCAL_YEAR = 2007
LAST_FISCAL_YEAR = 2025

BENCHMARK = "SPY"
