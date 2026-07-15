"""Build the G7 reproduction: scope T with Pos=10 and Neg=-5 leaf samples."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/Users/logan_martel/Projects/autoclanker_bootstrap_v5/clankerprof-v2-audit")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fixtures.pprof_builder import PprofFixtureBuilder  # noqa: E402

OUT = ROOT / ".goalloop-support" / "triage2" / "G7"

builder = PprofFixtureBuilder.create()
target = builder.location(builder.function("T", "/srv/app/t.py"))
pos = builder.location(builder.function("Pos", "/srv/app/pos.py"))
neg = builder.location(builder.function("Neg", "/srv/app/neg.py"))
builder.sample((pos, target), 10)
builder.sample((neg, target), -5)
(OUT / "profile.pb").write_bytes(builder.encode())
print("wrote", OUT / "profile.pb")
