"""Build a tiny scopes profile + config for the F7 compare repro."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/Users/logan_martel/Projects/autoclanker_bootstrap_v5/clankerprof-v2-audit")
sys.path.insert(0, str(ROOT))

from tests.fixtures.pprof_builder import PprofFixtureBuilder

OUT = ROOT / ".goalloop-support" / "triage" / "F7"

builder = PprofFixtureBuilder.create()
request = builder.location(
    builder.function("RequestHandler#render_response", "/app/http/request.rb")
)
component = builder.location(
    builder.function("ComponentRenderer#render", "/app/components/card.rb")
)
builder.sample((component, request), 10_000_000)
(OUT / "profile.pb").write_bytes(builder.encode())

(OUT / "scopes.toml").write_text(
    """
[cost_kind]
"Components" = "path:app/components/**"

[[scope]]
label = "x"
match = "name_eq:RequestHandler#render_response"
""",
    encoding="utf-8",
)
print("wrote", OUT / "profile.pb", OUT / "scopes.toml")
