"""Build G9 reproduction inputs: unicode filename profile + tiny-percentage profile."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/Users/logan_martel/Projects/autoclanker_bootstrap_v5/clankerprof-v2-audit")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fixtures.pprof_builder import PprofFixtureBuilder  # noqa: E402

OUT = ROOT / ".goalloop-support" / "triage2" / "G9"

# Case 1: unicode filename /café/x
b = PprofFixtureBuilder.create()
leaf = b.location(b.function("Leaf", "/café/x"))
target = b.location(b.function("T", "/srv/app/t.py"))
b.sample((leaf, target), 7)
(OUT / "unicode.pb").write_bytes(b.encode())

# Case 2: categories valued 1 and 99999999 -> pct 1e-06 for the small one
b2 = PprofFixtureBuilder.create()
leaf_a = b2.location(b2.function("LeafA", "/srv/a/leaf.py"))
leaf_b = b2.location(b2.function("LeafB", "/srv/b/leaf.py"))
t2 = b2.location(b2.function("T", "/srv/app/t.py"))
b2.sample((leaf_a, t2), 1)
b2.sample((leaf_b, t2), 99999999)
(OUT / "floats.pb").write_bytes(b2.encode())
(OUT / "floats-config.json").write_text(
    json.dumps({"T": {"A": "path:srv/a", "B": "path:srv/b"}}), encoding="utf-8"
)
print("wrote fixtures")
