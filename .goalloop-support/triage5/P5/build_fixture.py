"""Build the P5 reproduction inputs: a one-stack 10-ns profile."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/Users/logan_martel/Projects/autoclanker_bootstrap_v5/clankerprof-v2-audit")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fixtures.pprof_builder import PprofFixtureBuilder  # noqa: E402

OUT = ROOT / ".goalloop-support" / "triage5" / "P5"

builder = PprofFixtureBuilder.create()
leaf = builder.location(builder.function("Leaf", "/srv/app/leaf.py"))
builder.sample((leaf,), 10)
(OUT / "profile.pb").write_bytes(builder.encode())
print("wrote", OUT / "profile.pb")
