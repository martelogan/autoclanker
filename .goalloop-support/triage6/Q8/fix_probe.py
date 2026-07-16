"""Probe: with csv.field_size_limit lifted, does Python match Rust byte-for-byte?"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path("/Users/logan_martel/Projects/autoclanker_bootstrap_v5/clankerprof-v2-audit")
sys.path.insert(0, str(ROOT))

csv.field_size_limit(sys.maxsize)  # simulate the proposed fix

from clankerprof.cli import main  # noqa: E402

Q8 = ROOT / ".goalloop-support" / "triage6" / "Q8"
rc = main(
    [
        "targets",
        "--profile",
        str(Q8 / "profile.pb"),
        "--target",
        "parent",
        "--runtime",
        "ruby",
        "--core-classes",
        str(Q8 / "big.csv"),
    ]
)
print(f"exit: {rc}", file=sys.stderr)
