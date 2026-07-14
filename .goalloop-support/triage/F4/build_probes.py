"""Build the three F4 probe artifacts from a real exported v2 facts artifact."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path("/Users/logan_martel/Projects/autoclanker_bootstrap_v5/clankerprof-v2-audit")
OUT = ROOT / ".goalloop-support" / "triage" / "F4"
sys.path.insert(0, str(ROOT))

from clankerprof.facts import dumps_sample_facts  # noqa: E402
from clankerprof.model import Profile  # noqa: E402
from clankerprof.proto import decode_profile_bytes  # noqa: E402
from tests.fixtures.pprof_builder import PprofFixtureBuilder  # noqa: E402

builder = PprofFixtureBuilder.create()
loc = builder.location(builder.function("Target#render", "/app/target.rb"))
builder.sample((loc,), 7)
profile_bytes = builder.encode()
(OUT / "base.pb").write_bytes(profile_bytes)

profile: Profile = decode_profile_bytes(profile_bytes)
base = json.loads(dumps_sample_facts(profile.to_sample_facts()))
(OUT / "base_facts.json").write_text(json.dumps(base, sort_keys=True) + "\n")

BIG_ID = 9223372036854775808  # 2**63 = i64::MAX + 1, valid pprof uint64 ID
I64_MAX = 9223372036854775807

# Probe 1: valid uint64 location/function ID above i64::MAX, sample value 7.
p1 = copy.deepcopy(base)
p1["frames"][0][0] = BIG_ID
p1["frames"][0][1] = BIG_ID
p1["samples"][0]["location_ids"] = [BIG_ID]
(OUT / "probe1_bigid.json").write_text(json.dumps(p1, sort_keys=True) + "\n")

# Probe 2: non-integral sample value 7.9.
p2 = copy.deepcopy(base)
p2["samples"][0]["values"] = [7.9]
p2["summary"]["total_primary_value"] = 7  # what Python reconstructs after int()
(OUT / "probe2_float.json").write_text(json.dumps(p2, sort_keys=True) + "\n")

# Probe 3: two samples each carrying a valid signed-int64 max value.
p3 = copy.deepcopy(base)
sample0 = p3["samples"][0]
sample0["values"] = [I64_MAX]
sample1 = copy.deepcopy(sample0)
sample1["sample_index"] = 1
p3["samples"] = [sample0, sample1]
p3["summary"]["sample_count"] = 2
p3["summary"]["non_empty_sample_count"] = 2
p3["summary"]["total_primary_value"] = 2 * I64_MAX  # 18446744073709551614
(OUT / "probe3_overflow.json").write_text(json.dumps(p3, sort_keys=True) + "\n")

print("base:", json.dumps(base, sort_keys=True))
