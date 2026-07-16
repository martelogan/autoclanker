"""Build valid facts-v2 fixtures for triage9 reproductions."""

import json
import sys


def build(frames_spec, samples_spec, sample_indexes=None):
    # frames_spec: list of (name, filename) leaf-to-root order irrelevant here
    strings = []

    def intern(s):
        if s in strings:
            return strings.index(s)
        strings.append(s)
        return len(strings) - 1

    frames = []
    for i, (name, filename) in enumerate(frames_spec):
        frames.append([i + 1, i + 1, intern(name), intern(filename), 1, False])

    samples = []
    total = 0
    empty = 0
    for i, (stack, value) in enumerate(samples_spec):
        idx = sample_indexes[i] if sample_indexes else i
        samples.append(
            {
                "sample_index": idx,
                "values": [value],
                "location_ids": [frames[f][0] for f in stack],
                "stack": stack,
            }
        )
        total += value
        if not stack:
            empty += 1

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
            "sample_count": len(samples),
            "empty_sample_count": empty,
            "non_empty_sample_count": len(samples) - empty,
            "total_primary_value": total,
        },
        "strings": strings,
        "frames": frames,
        "samples": samples,
    }


if __name__ == "__main__":
    spec = json.load(sys.stdin)
    payload = build(
        [tuple(f) for f in spec["frames"]],
        [(s[0], s[1]) for s in spec["samples"]],
        spec.get("sample_indexes"),
    )
    json.dump(payload, sys.stdout, indent=1)
