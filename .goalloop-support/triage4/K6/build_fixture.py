"""Build the K6 reproduction inputs: parent T with Positive=110 ns and Negative=-10 ns."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/Users/logan_martel/Projects/autoclanker_bootstrap_v5/clankerprof-v2-audit")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fixtures.pprof_builder import PprofFixtureBuilder  # noqa: E402

OUT = ROOT / ".goalloop-support" / "triage4" / "K6"

builder = PprofFixtureBuilder.create()
pos_leaf = builder.location(builder.function("PosLeaf", "app/pos/leaf.py"))
neg_leaf = builder.location(builder.function("NegLeaf", "app/neg/leaf.py"))
target = builder.location(builder.function("T", "app/t.py"))
builder.sample((pos_leaf, target), 110)
builder.sample((neg_leaf, target), -10)
(OUT / "profile.pb").write_bytes(builder.encode())

config = {
    "T": {
        "Positive": "app/pos/**",
        "Negative": "app/neg/**",
    }
}
(OUT / "target_config.json").write_text(json.dumps(config, indent=2) + "\n")
print("wrote", OUT / "profile.pb", "and", OUT / "target_config.json")
