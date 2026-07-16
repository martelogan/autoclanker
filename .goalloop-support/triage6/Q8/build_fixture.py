"""Build Q8 reproduction inputs: one-sample profile (child<-parent, 1 ns) and a big-field CSV."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/Users/logan_martel/Projects/autoclanker_bootstrap_v5/clankerprof-v2-audit")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fixtures.pprof_builder import PprofFixtureBuilder  # noqa: E402

OUT = ROOT / ".goalloop-support" / "triage6" / "Q8"

builder = PprofFixtureBuilder.create()
leaf = builder.location(builder.function("child", "/srv/app/child.rb"))
parent = builder.location(builder.function("parent", "/srv/app/parent.rb"))
builder.sample((leaf, parent), 1)
(OUT / "profile.pb").write_bytes(builder.encode())
print("wrote", OUT / "profile.pb")

# One CSV record whose single field is 131073 chars: one over csv.field_size_limit().
big = "A" * 131073
(OUT / "big.csv").write_text(big + "\n", encoding="utf-8")
print("wrote big.csv, field length", len(big))
