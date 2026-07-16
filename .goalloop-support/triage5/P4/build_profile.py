"""Build the P4 repro profile: two equal 10-ns samples, owners ZOwner then AOwner."""

from __future__ import annotations

import sys
from pathlib import Path

WORKTREE = Path("/Users/logan_martel/Projects/autoclanker_bootstrap_v5/clankerprof-v2-audit")
sys.path.insert(0, str(WORKTREE))

from tests.fixtures.pprof_builder import PprofFixtureBuilder  # noqa: E402

builder = PprofFixtureBuilder.create()
z_owner = builder.location(builder.function("ZOwner", "app/rendering/owner.rb"))
a_owner = builder.location(builder.function("AOwner", "app/rendering/owner.rb"))
handler = builder.location(
    builder.function("RequestHandler#render_response", "app/http/handler.rb")
)
# Leaf-first stacks: owner below the scope frame; ZOwner encountered first.
builder.sample((z_owner, handler), 10)
builder.sample((a_owner, handler), 10)

out = Path(__file__).parent / "p4.pb"
out.write_bytes(builder.encode())
print(f"wrote {out}")
