"""Build the Q2 reproduction: /a +10, /b -10, /c +5; name:match selects only a and b."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/Users/logan_martel/Projects/autoclanker_bootstrap_v5/clankerprof-v2-audit")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fixtures.pprof_builder import PprofFixtureBuilder  # noqa: E402

OUT = ROOT / ".goalloop-support" / "triage6" / "Q2"

builder = PprofFixtureBuilder.create()
loc_a = builder.location(builder.function("match_a", "/a/one.rb"))
loc_b = builder.location(builder.function("match_b", "/b/two.rb"))
loc_c = builder.location(builder.function("other_c", "/c/three.rb"))
builder.sample((loc_a,), 10)
builder.sample((loc_b,), -10)
builder.sample((loc_c,), 5)
(OUT / "profile.pb").write_bytes(builder.encode())
print("wrote", OUT / "profile.pb")
