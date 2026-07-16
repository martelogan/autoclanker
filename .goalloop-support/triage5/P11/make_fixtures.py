import json

NUL = chr(0)


def base(strings, frames, samples):
    return {
        "schema_version": "clankerprof.sample_facts.v2",
        "tool": "clankerprof_facts",
        "profile": {
            "value_types": [{"type": "cpu", "unit": "nanoseconds"}],
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
        "strings": strings,
        "frames": frames,
        "samples": samples,
    }


slices = base(
    ["A", "B" + NUL + "C", "A" + NUL + "B", "C"],
    [
        [1, 1, 0, 1, 5, False],
        [2, 2, 2, 3, 7, False],
    ],
    [
        {"sample_index": 0, "values": [10], "location_ids": [1], "stack": [0]},
        {"sample_index": 1, "values": [20], "location_ids": [2], "stack": [1]},
    ],
)

scopes = base(
    ["C", "app/x.py", "A -> B", "app/y.py", "B -> C", "A"],
    [
        [1, 1, 0, 1, 3, False],
        [2, 2, 2, 3, 9, False],
        [3, 3, 4, 1, 4, False],
        [4, 4, 5, 3, 11, False],
    ],
    [
        {"sample_index": 0, "values": [10], "location_ids": [1, 2], "stack": [0, 1]},
        {"sample_index": 1, "values": [20], "location_ids": [3, 4], "stack": [2, 3]},
    ],
)

# default ensure_ascii=True keeps the NUL escaped in the file text
with open("slices_facts.json", "w") as f:
    json.dump(slices, f, indent=2)
    f.write("\n")
with open("scopes_facts.json", "w") as f:
    json.dump(scopes, f, indent=2)
    f.write("\n")
print("slices strings:", repr(json.load(open("slices_facts.json"))["strings"]))
print("scopes strings:", repr(json.load(open("scopes_facts.json"))["strings"]))
