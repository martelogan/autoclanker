"""Build the Q10 reproduction profile: two samples summing to 400 with a repr-tie pct."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/Users/logan_martel/Projects/autoclanker_bootstrap_v5/clankerprof-v2-audit")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fixtures.pprof_builder import PprofFixtureBuilder  # noqa: E402

OUT = ROOT / ".goalloop-support" / "triage6" / "Q10"

builder = PprofFixtureBuilder.create()
pos = builder.location(builder.function("PosLeaf", "/srv/app/pos.py"))
neg = builder.location(builder.function("NegLeaf", "/srv/app/neg.py"))
builder.sample((pos,), 5268900186379573)
builder.sample((neg,), -5268900186379173)
(OUT / "profile.pb").write_bytes(builder.encode())
print("wrote", OUT / "profile.pb")
