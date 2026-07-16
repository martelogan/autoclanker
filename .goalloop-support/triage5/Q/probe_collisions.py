"""Probe the NUL frame-key and arrow caller-pair collisions (R5-10)."""

import json
import subprocess

NUL = "\x00"

facts = {
    "schema_version": "clankerprof.sample_facts.v2",
    "tool": "clankerprof_facts",
    "profile": {
        "value_types": [],
        "period_type": None,
        "period": 0,
        "default_sample_type": "",
        "primary_value_index": 0,
    },
    "summary": {
        "sample_count": 2,
        "empty_sample_count": 0,
        "non_empty_sample_count": 2,
        "total_primary_value": 30,
    },
    "strings": ["A", f"B{NUL}C", f"A{NUL}B", "C"],
    "frames": [[1, 1, 0, 1, 0, False], [2, 2, 2, 3, 0, False]],
    "samples": [
        {"sample_index": 0, "values": [10], "location_ids": [1], "stack": [0]},
        {"sample_index": 1, "values": [20], "location_ids": [2], "stack": [1]},
    ],
}
p = ".goalloop-support/triage5/Q/nul-collision.json"
json.dump(facts, open(p, "w"))
py = subprocess.run(
    ["uv", "run", "clankerprof", "slices", "--facts", p],
    capture_output=True,
    text=True,
)
rs = subprocess.run(
    [
        "cargo", "run", "-q", "-p", "clankerprof-core", "--bin", "clankerprof-rs",
        "--", "slices", "--facts", p,
    ],
    capture_output=True,
    text=True,
)
frames = [
    (f["function"], f["filename"], f["time_ns"])
    for s in json.loads(py.stdout)["slices"]
    for f in s["frames"]
]
print("py frame rows:", len(frames), frames)
print("identical:", py.stdout == rs.stdout, "| py", py.returncode, "| rs", rs.returncode)

# Arrow caller/leaf pair collision via scopes: (caller="A -> B", leaf="C") vs
# (caller="A", leaf="B -> C").
facts2 = {
    **facts,
    "strings": ["C", "/x", "A -> B", "B -> C", "A", "T", "/t"],
    "frames": [
        [1, 1, 0, 1, 0, False],
        [2, 2, 2, 1, 0, False],
        [3, 3, 3, 1, 0, False],
        [4, 4, 4, 1, 0, False],
        [5, 5, 5, 6, 0, False],
    ],
    "summary": {
        "sample_count": 2,
        "empty_sample_count": 0,
        "non_empty_sample_count": 2,
        "total_primary_value": 30,
    },
    "samples": [
        {"sample_index": 0, "values": [10], "location_ids": [1, 2, 5], "stack": [0, 1, 4]},
        {"sample_index": 1, "values": [20], "location_ids": [3, 4, 5], "stack": [3, 2, 4]},
    ],
}
p2 = ".goalloop-support/triage5/Q/arrow-collision.json"
json.dump(facts2, open(p2, "w"))
cfg = ".goalloop-support/triage5/Q/arrow-scopes.yml"
open(cfg, "w").write('cost_kind:\n  Work: "name:C"\nscope:\n  - function: T\n')
py2 = subprocess.run(
    ["uv", "run", "clankerprof", "scopes", "--facts", p2, "--config", cfg],
    capture_output=True,
    text=True,
)
rs2 = subprocess.run(
    [
        "cargo", "run", "-q", "-p", "clankerprof-core", "--bin", "clankerprof-rs",
        "--", "scopes", "--facts", p2, "--config", cfg,
    ],
    capture_output=True,
    text=True,
)
if py2.returncode == 0:
    payload = json.loads(py2.stdout)
    pairs = [
        (pair["pair"], pair["samples"], pair["time_ns"])
        for b in payload["boundaries"]
        for bucket in b["buckets"]
        for cat in bucket["categories"]
        for pair in cat.get("caller_leaf_pairs", [])
    ]
    print("py pair rows:", len(pairs), pairs)
else:
    print("py scopes stderr:", py2.stderr.strip()[:200])
print("identical:", py2.stdout == rs2.stdout, "| py", py2.returncode, "| rs", rs2.returncode)
