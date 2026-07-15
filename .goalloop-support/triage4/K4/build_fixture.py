"""Build K4 reproduction: target T with +110 (Positive) and -10 (Negative) categories."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/Users/logan_martel/Projects/autoclanker_bootstrap_v5/clankerprof-v2-audit")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fixtures.pprof_builder import PprofFixtureBuilder  # noqa: E402

OUT = ROOT / ".goalloop-support" / "triage4" / "K4"

builder = PprofFixtureBuilder.create()
target = builder.location(builder.function("T", "/srv/app/t.py"))
pos_leaf = builder.location(builder.function("PosLeaf", "/srv/app/pos.py"))
neg_leaf = builder.location(builder.function("NegLeaf", "/srv/app/neg.py"))
builder.sample((pos_leaf, target), 110)
builder.sample((neg_leaf, target), -10)
(OUT / "profile.pb").write_bytes(builder.encode())

(OUT / "config.json").write_text(
    json.dumps({"T": {"Positive": "pos.py", "Negative": "neg.py"}}),
    encoding="utf-8",
)
(OUT / "attrs_finite.json").write_text('{"huge":{"T":1.7e308}}', encoding="utf-8")
(OUT / "attrs_1e309.json").write_text('{"huge":{"T":1e309}}', encoding="utf-8")

# Base fixture without negatives for the 1e309 sub-claim (pct <= 100).
base = PprofFixtureBuilder.create()
b_target = base.location(base.function("T", "/srv/app/t.py"))
b_leaf = base.location(base.function("PosLeaf", "/srv/app/pos.py"))
base.sample((b_leaf, b_target), 100)
(OUT / "profile_base.pb").write_bytes(base.encode())

print("wrote fixtures to", OUT)
