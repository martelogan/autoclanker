import json
import subprocess
import sys

sys.path.insert(0, ".")
from tests.fixtures.pprof_builder import PprofFixtureBuilder

d = ".goalloop-support/triage7/S2"
b = PprofFixtureBuilder.create(
    sample_types=(("samples", "count"), ("cpu", "nanoseconds")),
    default_sample_type="cpu",
)
leaf = b.location(b.function("Leaf", "/app/leaf.py"))
b.sample((leaf,), (1, 100))
open(f"{d}/profile.pb", "wb").write(b.encode())
subprocess.run(
    [
        "uv", "run", "clankerprof", "facts",
        "--profile", f"{d}/profile.pb",
        "--output", f"{d}/facts.json",
    ],
    check=True,
)
facts = json.load(open(f"{d}/facts.json"))
print("exported profile meta:", facts["profile"], "summary:", facts["summary"])
facts["profile"]["primary_value_index"] = 99
# keep summary consistent with the values[0] fallback (total 1)
facts["summary"]["total_primary_value"] = 1
json.dump(facts, open(f"{d}/facts_bad.json", "w"))
