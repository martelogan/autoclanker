"""Build the Q4 reproduction profile: scope T with LeafA=10ns (cost kind A) and LeafB=5ns (cost kind B)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/Users/logan_martel/Projects/autoclanker_bootstrap_v5/clankerprof-v2-audit")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fixtures.pprof_builder import PprofFixtureBuilder  # noqa: E402

OUT = ROOT / ".goalloop-support" / "triage6" / "Q4"

builder = PprofFixtureBuilder.create()
leaf_a = builder.location(builder.function("LeafA", "/srv/app/leaf_a.py"))
leaf_b = builder.location(builder.function("LeafB", "/srv/app/leaf_b.py"))
target = builder.location(builder.function("T", "/srv/app/t.py"))
builder.sample((leaf_a, target), 10)
builder.sample((leaf_b, target), 5)
(OUT / "profile.pb").write_bytes(builder.encode())
print("wrote", OUT / "profile.pb")
