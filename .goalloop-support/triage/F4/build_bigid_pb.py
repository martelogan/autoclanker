"""Encode a raw pprof profile whose location/function IDs are 2**63 (valid uint64)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/Users/logan_martel/Projects/autoclanker_bootstrap_v5/clankerprof-v2-audit")
OUT = ROOT / ".goalloop-support" / "triage" / "F4"
sys.path.insert(0, str(ROOT))

from tests.fixtures.pprof_builder import (  # noqa: E402
    _field_bytes,
    _field_string,
    _field_varint,
)

BIG_ID = 9223372036854775808  # 2**63

payload = bytearray()
# string_table: index 0 must be "", then names.
strings = ["", "cpu", "nanoseconds", "Target#render", "/app/target.rb"]
# sample_type: ValueType{type=cpu, unit=nanoseconds}
payload.extend(_field_bytes(1, _field_varint(1, 1) + _field_varint(2, 2)))
# sample: location_id=BIG_ID, value=7
payload.extend(_field_bytes(2, _field_varint(1, BIG_ID) + _field_varint(2, 7)))
# location: id=BIG_ID, line{function_id=BIG_ID, line=1}
line_msg = _field_varint(1, BIG_ID) + _field_varint(2, 1)
payload.extend(_field_bytes(4, _field_varint(1, BIG_ID) + _field_bytes(4, line_msg)))
# function: id=BIG_ID, name=3, filename=4
payload.extend(
    _field_bytes(5, _field_varint(1, BIG_ID) + _field_varint(2, 3) + _field_varint(4, 4))
)
for value in strings:
    payload.extend(_field_string(6, value))

(OUT / "bigid.pb").write_bytes(bytes(payload))
print("wrote", OUT / "bigid.pb", len(payload), "bytes")
