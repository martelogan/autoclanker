from __future__ import annotations

import gzip

from dataclasses import dataclass, field

_U64_MASK = (1 << 64) - 1


def _varint(value: int) -> bytes:
    if value < 0:
        value &= _U64_MASK
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


def _default_sample_types() -> list[tuple[str, str]]:
    return [("cpu", "nanoseconds")]


@dataclass(slots=True)
class PprofFixtureBuilder:
    strings: list[str]
    functions: list[tuple[int, int, int]]
    locations: list[tuple[tuple[tuple[int, int], ...], bool]]
    samples: list[tuple[tuple[int, ...], tuple[int, ...]]]
    sample_types: list[tuple[str, str]] = field(default_factory=_default_sample_types)
    default_sample_type: str | None = None
    period_type: tuple[str, str] | None = None
    period: int = 0

    @classmethod
    def create(
        cls,
        *,
        sample_types: tuple[tuple[str, str], ...] = (("cpu", "nanoseconds"),),
        default_sample_type: str | None = None,
        period_type: tuple[str, str] | None = None,
        period: int = 0,
    ) -> PprofFixtureBuilder:
        return cls(
            strings=[""],
            functions=[],
            locations=[],
            samples=[],
            sample_types=list(sample_types),
            default_sample_type=default_sample_type,
            period_type=period_type,
            period=period,
        )

    def string(self, value: str) -> int:
        self.strings.append(value)
        return len(self.strings) - 1

    def _intern(self, value: str) -> int:
        try:
            return self.strings.index(value)
        except ValueError:
            return self.string(value)

    def function(self, name: str, filename: str) -> int:
        function_id = len(self.functions) + 1
        self.functions.append((function_id, self.string(name), self.string(filename)))
        return function_id

    def location(self, function_id: int, line: int = 1) -> int:
        location_id = len(self.locations) + 1
        self.locations.append((((function_id, line),), False))
        return location_id

    def inline_location(
        self,
        function_ids: tuple[int, ...],
        lines: tuple[int, ...] | None = None,
        *,
        folded: bool = False,
    ) -> int:
        location_id = len(self.locations) + 1
        line_values = lines if lines is not None else tuple(1 for _ in function_ids)
        self.locations.append(
            (tuple(zip(function_ids, line_values, strict=True)), folded)
        )
        return location_id

    def folded_location(self, function_id: int, line: int = 1) -> int:
        location_id = len(self.locations) + 1
        self.locations.append((((function_id, line),), True))
        return location_id

    def sample(
        self,
        location_ids: tuple[int, ...],
        value: int | tuple[int, ...],
    ) -> None:
        values = (value,) if isinstance(value, int) else tuple(value)
        self.samples.append((location_ids, values))

    def encode(self, *, gzipped: bool = False, packed_samples: bool = False) -> bytes:
        payload = bytearray()
        for type_name, unit in self.sample_types:
            payload.extend(
                _field_bytes(
                    1,
                    _field_varint(1, self._intern(type_name))
                    + _field_varint(2, self._intern(unit)),
                )
            )
        for location_ids, values in self.samples:
            sample_payload = bytearray()
            if packed_samples:
                sample_payload.extend(
                    _field_bytes(
                        1,
                        b"".join(_varint(location_id) for location_id in location_ids),
                    )
                )
                sample_payload.extend(
                    _field_bytes(2, b"".join(_varint(value) for value in values))
                )
            else:
                for location_id in location_ids:
                    sample_payload.extend(_field_varint(1, location_id))
                for value in values:
                    sample_payload.extend(_field_varint(2, value))
            payload.extend(_field_bytes(2, bytes(sample_payload)))
        for location_id, (function_lines, is_folded) in enumerate(
            self.locations,
            start=1,
        ):
            line_payload = bytearray()
            for function_id, line in function_lines:
                line_payload.extend(
                    _field_bytes(
                        4,
                        _field_varint(1, function_id) + _field_varint(2, line),
                    )
                )
            location_payload = (
                _field_varint(1, location_id)
                + bytes(line_payload)
                + (_field_varint(5, 1) if is_folded else b"")
            )
            payload.extend(
                _field_bytes(
                    4,
                    location_payload,
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
        if self.period_type is not None:
            type_name, unit = self.period_type
            payload.extend(
                _field_bytes(
                    11,
                    _field_varint(1, self._intern(type_name))
                    + _field_varint(2, self._intern(unit)),
                )
            )
        if self.period:
            payload.extend(_field_varint(12, self.period))
        if self.default_sample_type is not None:
            payload.extend(_field_varint(14, self._intern(self.default_sample_type)))
        for value in self.strings:
            payload.extend(_field_string(6, value))
        data = bytes(payload)
        return gzip.compress(data) if gzipped else data


# Public aliases for tests that hand-encode raw pprof messages directly.
field_bytes = _field_bytes
field_string = _field_string
field_varint = _field_varint
