"""Build the K9 reproduction inputs: T -> Array#map (<cfunc>) profile."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/Users/logan_martel/Projects/autoclanker_bootstrap_v5/clankerprof-v2-audit")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fixtures.pprof_builder import PprofFixtureBuilder  # noqa: E402

OUT = ROOT / ".goalloop-support" / "triage4" / "K9"

builder = PprofFixtureBuilder.create()
leaf = builder.location(builder.function("Array#map", "<cfunc>"))
target = builder.location(builder.function("T", "/srv/app/t.py"))
builder.sample((leaf, target), 7)
(OUT / "profile.pb").write_bytes(builder.encode())
print("wrote", OUT / "profile.pb")
