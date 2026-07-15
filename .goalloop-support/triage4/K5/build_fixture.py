"""Build the K5 reproduction: one -10 ns sample Leaf->T (negative scope total)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/Users/logan_martel/Projects/autoclanker_bootstrap_v5/clankerprof-v2-audit")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fixtures.pprof_builder import PprofFixtureBuilder  # noqa: E402

OUT = ROOT / ".goalloop-support" / "triage4" / "K5"

builder = PprofFixtureBuilder.create()
leaf = builder.location(builder.function("Leaf", "/srv/app/leaf.py"))
target = builder.location(builder.function("T", "/srv/app/t.py"))
builder.sample((leaf, target), -10)
(OUT / "profile.pb").write_bytes(builder.encode())
print("wrote", OUT / "profile.pb")
