"""Build the F9 reproduction fixture: leaf at /app/a.rb under parent T."""

from __future__ import annotations

import json
from pathlib import Path

from tests.fixtures.pprof_builder import PprofFixtureBuilder

OUT = Path(__file__).resolve().parent

builder = PprofFixtureBuilder.create()
parent = builder.location(builder.function("T", "/app/t.rb"))
leaf = builder.location(builder.function("Leaf", "/app/a.rb"))
builder.sample((leaf, parent), 10_000_000)

(OUT / "profile.pb").write_bytes(builder.encode())
(OUT / "targets.json").write_text(
    json.dumps({"T": {"Application": "path:/app/[ab].rb"}}), encoding="utf-8"
)
print("wrote", OUT / "profile.pb", "and", OUT / "targets.json")
