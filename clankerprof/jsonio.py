"""Strict JSON input parsing shared by clankerprof input surfaces.

Python's ``json`` module accepts the non-standard ``Infinity``/``-Infinity``/
``NaN`` tokens that RFC 8259 forbids and that the Rust port (serde_json)
rejects at parse time. Every clankerprof JSON input goes through this loader
so both implementations fail closed on the same inputs.
"""

from __future__ import annotations

import json

from typing import Any


def _reject_constant(token: str) -> Any:
    raise ValueError(f"JSON input must not contain the non-finite token {token!r}.")


def parse_strict_json(text: str) -> Any:
    return json.loads(text, parse_constant=_reject_constant)
