"""Build the G11 reproduction: one fact at /gems/bar/file.rb."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/Users/logan_martel/Projects/autoclanker_bootstrap_v5/clankerprof-v2-audit")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fixtures.pprof_builder import PprofFixtureBuilder  # noqa: E402

OUT = ROOT / ".goalloop-support" / "triage2" / "G11"

builder = PprofFixtureBuilder.create()
leaf = builder.location(builder.function("Work", "/gems/bar/file.rb"))
root = builder.location(builder.function("Main", "/srv/app/main.rb"))
builder.sample((leaf, root), 7)
(OUT / "profile.pb").write_bytes(builder.encode())
print("wrote", OUT / "profile.pb")
