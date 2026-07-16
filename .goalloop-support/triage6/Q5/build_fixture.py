"""Build the Q5 reproduction input: one sample under /a/x worth 7 ns."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/Users/logan_martel/Projects/autoclanker_bootstrap_v5/clankerprof-v2-audit")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fixtures.pprof_builder import PprofFixtureBuilder  # noqa: E402

OUT = ROOT / ".goalloop-support" / "triage6" / "Q5"

builder = PprofFixtureBuilder.create()
leaf = builder.location(builder.function("work", "/a/x/work.py"))
builder.sample((leaf,), 7)
(OUT / "profile.pb").write_bytes(builder.encode())
print("wrote", OUT / "profile.pb")
