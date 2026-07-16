"""Build the P10 reproduction: native leaf 'Weird\\nClass#run' under Parent, plus the multiline-quoted core-class CSV."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/Users/logan_martel/Projects/autoclanker_bootstrap_v5/clankerprof-v2-audit")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fixtures.pprof_builder import PprofFixtureBuilder  # noqa: E402

OUT = ROOT / ".goalloop-support" / "triage5" / "P10"

builder = PprofFixtureBuilder.create()
leaf = builder.location(builder.function("Weird\nClass#run", "<cfunc>"))
parent = builder.location(builder.function("Parent", "/srv/app/parent.rb"))
builder.sample((leaf, parent), 10)
(OUT / "profile.pb").write_bytes(builder.encode())
print("wrote", OUT / "profile.pb")

(OUT / "core.csv").write_bytes(b'"Weird\nClass"\n')
print("wrote", OUT / "core.csv")

# Sanity: what does Python csv.reader make of it?
import csv

with (OUT / "core.csv").open(newline="", encoding="utf-8") as handle:
    print("python csv.reader rows:", [row for row in csv.reader(handle)])
