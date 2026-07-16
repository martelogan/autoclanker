"""Build P2 repro inputs: four one-sample Leaf->T profiles + configs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/Users/logan_martel/Projects/autoclanker_bootstrap_v5/clankerprof-v2-audit")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fixtures.pprof_builder import PprofFixtureBuilder  # noqa: E402

OUT = ROOT / ".goalloop-support" / "triage5" / "P2"

NL = chr(10)
CASES = {
    "anchor": ("/app/x" + NL, "regex:x$"),
    "casefold": ("/" + chr(0x130), "regex:(?i)i"),
    "wordclass": ("/" + chr(0x301), "regex:" + chr(92) + "w"),
    "spaceclass": ("/" + chr(0x1F), "regex:" + chr(92) + "s"),
}

for name, (filename, pattern) in CASES.items():
    builder = PprofFixtureBuilder.create()
    leaf = builder.location(builder.function("Leaf", filename))
    target = builder.location(builder.function("T", "/srv/app/t.py"))
    builder.sample((leaf, target), 7)
    (OUT / (name + ".pb")).write_bytes(builder.encode())
    config = {"T": {"Matched": [pattern]}}
    (OUT / (name + ".config.json")).write_text(
        json.dumps(config, ensure_ascii=False), encoding="utf-8"
    )
    print("wrote", name, ascii(filename), repr(pattern))
