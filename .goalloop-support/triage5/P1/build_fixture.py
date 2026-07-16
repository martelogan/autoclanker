"""Build the P1 reproduction profile: one 70-ns stack
CacheClient#get -> TelemetryWrapper#call -> ComponentRenderer#render -> RequestHandler#render_response
(mirrors tests/test_clankerprof.py::_descendant_attribute_profile_bytes, value scaled to 70 ns)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/Users/logan_martel/Projects/autoclanker_bootstrap_v5/clankerprof-v2-audit")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fixtures.pprof_builder import PprofFixtureBuilder  # noqa: E402

OUT = ROOT / ".goalloop-support" / "triage5" / "P1"

builder = PprofFixtureBuilder.create()
request = builder.location(
    builder.function("RequestHandler#render_response", "/app/http/request.rb")
)
component = builder.location(
    builder.function("ComponentRenderer#render", "/app/components/card.rb")
)
wrapper = builder.location(
    builder.function("TelemetryWrapper#call", "/app/lib/telemetry.rb")
)
cache = builder.location(
    builder.function("CacheClient#get", "/vendor/cache-client-1.2.3/lib/client.rb")
)
builder.sample((cache, wrapper, component, request), 70)
(OUT / "profile.pb").write_bytes(builder.encode())
print("wrote", OUT / "profile.pb")
