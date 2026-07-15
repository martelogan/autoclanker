"""Build J5 reproduction inputs: negative-valued GC and collapse-only profiles."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/Users/logan_martel/Projects/autoclanker_bootstrap_v5/clankerprof-v2-audit")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fixtures.pprof_builder import PprofFixtureBuilder  # noqa: E402

OUT = ROOT / ".goalloop-support" / "triage3" / "J5"

# 1. One (marking) leaf frame valued -10.
builder = PprofFixtureBuilder.create()
marking = builder.location(builder.function("(marking)", ""))
builder.sample((marking,), -10)
(OUT / "gc_neg.pb").write_bytes(builder.encode())

# 1b. Positive control: same shape valued +10.
builder = PprofFixtureBuilder.create()
marking = builder.location(builder.function("(marking)", ""))
builder.sample((marking,), 10)
(OUT / "gc_pos.pb").write_bytes(builder.encode())

# 2. One ordinary /srv frame valued -10 (to collapse fully).
builder = PprofFixtureBuilder.create()
leaf = builder.location(builder.function("Leaf", "/srv/app/leaf.py"))
builder.sample((leaf,), -10)
(OUT / "collapse_neg.pb").write_bytes(builder.encode())

# 2b. Positive control valued +10.
builder = PprofFixtureBuilder.create()
leaf = builder.location(builder.function("Leaf", "/srv/app/leaf.py"))
builder.sample((leaf,), 10)
(OUT / "collapse_pos.pb").write_bytes(builder.encode())

print("wrote fixtures under", OUT)
