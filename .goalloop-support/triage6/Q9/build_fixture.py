"""Build the Q9 reproduction: two valid signed-i64 samples under target `parent`."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/Users/logan_martel/Projects/autoclanker_bootstrap_v5/clankerprof-v2-audit")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fixtures.pprof_builder import PprofFixtureBuilder  # noqa: E402

OUT = ROOT / ".goalloop-support" / "triage6" / "Q9"

A_VALUE = 5000000000000000000
B_VALUE = 5717268054699998221
TOTAL = A_VALUE + B_VALUE

builder = PprofFixtureBuilder.create()
loc_parent = builder.location(builder.function("parent", "/srv/app/parent.py"))
loc_a = builder.location(builder.function("leaf_a", "/srv/app/a.py"))
loc_b = builder.location(builder.function("leaf_b", "/srv/app/b.py"))
# leaf-to-root stacks: leaf first, parent last
builder.sample((loc_a, loc_parent), A_VALUE)
builder.sample((loc_b, loc_parent), B_VALUE)
(OUT / "profile.pb").write_bytes(builder.encode())
print("wrote", OUT / "profile.pb")
print("values fit i64:", A_VALUE <= 2**63 - 1 and B_VALUE <= 2**63 - 1)
print("total:", TOTAL, "> i64::MAX:", TOTAL > 2**63 - 1, "<= u64::MAX:", TOTAL <= 2**64 - 1)
