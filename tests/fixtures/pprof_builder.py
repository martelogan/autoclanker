from __future__ import annotations

import gzip

from dataclasses import dataclass


def _varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("This fixture encoder only supports non-negative varints.")
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _key(field: int, wire_type: int) -> bytes:
    return _varint((field << 3) | wire_type)


def _field_varint(field: int, value: int) -> bytes:
    return _key(field, 0) + _varint(value)


def _field_bytes(field: int, payload: bytes) -> bytes:
    return _key(field, 2) + _varint(len(payload)) + payload


def _field_string(field: int, value: str) -> bytes:
    return _field_bytes(field, value.encode("utf-8"))


@dataclass(slots=True)
class PprofFixtureBuilder:
    strings: list[str]
    functions: list[tuple[int, int, int]]
    locations: list[tuple[int, ...]]
    samples: list[tuple[tuple[int, ...], int]]

    @classmethod
    def create(cls) -> PprofFixtureBuilder:
        return cls(strings=[""], functions=[], locations=[], samples=[])

    def string(self, value: str) -> int:
        self.strings.append(value)
        return len(self.strings) - 1

    def function(self, name: str, filename: str) -> int:
        function_id = len(self.functions) + 1
        self.functions.append((function_id, self.string(name), self.string(filename)))
        return function_id

    def location(self, function_id: int) -> int:
        location_id = len(self.locations) + 1
        self.locations.append((function_id,))
        return location_id

    def inline_location(self, function_ids: tuple[int, ...]) -> int:
        location_id = len(self.locations) + 1
        self.locations.append(function_ids)
        return location_id

    def sample(self, location_ids: tuple[int, ...], value: int) -> None:
        self.samples.append((location_ids, value))

    def encode(self, *, gzipped: bool = False) -> bytes:
        payload = bytearray()
        payload.extend(
            _field_bytes(
                1,
                _field_varint(1, self.string("cpu"))
                + _field_varint(2, self.string("nanoseconds")),
            )
        )
        for location_ids, value in self.samples:
            sample_payload = bytearray()
            for location_id in location_ids:
                sample_payload.extend(_field_varint(1, location_id))
            sample_payload.extend(_field_varint(2, value))
            payload.extend(_field_bytes(2, bytes(sample_payload)))
        for location_id, function_ids in enumerate(self.locations, start=1):
            line_payload = bytearray()
            for function_id in function_ids:
                line_payload.extend(
                    _field_bytes(4, _field_varint(1, function_id) + _field_varint(2, 1))
                )
            payload.extend(
                _field_bytes(
                    4,
                    _field_varint(1, location_id) + bytes(line_payload),
                )
            )
        for function_id, name_index, filename_index in self.functions:
            payload.extend(
                _field_bytes(
                    5,
                    _field_varint(1, function_id)
                    + _field_varint(2, name_index)
                    + _field_varint(4, filename_index),
                )
            )
        for value in self.strings:
            payload.extend(_field_string(6, value))
        data = bytes(payload)
        return gzip.compress(data) if gzipped else data
