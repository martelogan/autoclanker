from __future__ import annotations

import gzip
import zlib

from dataclasses import dataclass, field, replace

from clankerprof.model import (
    Function,
    Location,
    Profile,
    Sample,
    ValueType,
    select_primary_value_index,
)


class PprofDecodeError(ValueError):
    """Raised when profile bytes are not a supported pprof protobuf profile."""


_U64_MASK = (1 << 64) - 1
_INT64_SIGN_BIT = 1 << 63


def _to_signed64(value: int) -> int:
    """Reinterpret an unsigned varint as a two's-complement int64."""
    value &= _U64_MASK
    return value - (1 << 64) if value & _INT64_SIGN_BIT else value


@dataclass(slots=True)
class _Reader:
    data: bytes
    pos: int = 0

    def eof(self) -> bool:
        return self.pos >= len(self.data)

    def read_byte(self) -> int:
        if self.pos >= len(self.data):
            raise PprofDecodeError("Unexpected end of protobuf stream.")
        value = self.data[self.pos]
        self.pos += 1
        return value

    def read_varint(self) -> int:
        shift = 0
        result = 0
        while True:
            byte = self.read_byte()
            result |= (byte & 0x7F) << shift
            if byte < 0x80:
                return result
            shift += 7
            if shift > 70:
                raise PprofDecodeError("Invalid protobuf varint.")

    def read_key(self) -> tuple[int, int]:
        key = self.read_varint()
        return key >> 3, key & 0x07

    def read_length_delimited(self) -> bytes:
        length = self.read_varint()
        end = self.pos + length
        if end > len(self.data):
            raise PprofDecodeError("Length-delimited field extends beyond stream.")
        payload = self.data[self.pos : end]
        self.pos = end
        return payload

    def skip(self, wire_type: int) -> None:
        if wire_type == 0:
            self.read_varint()
            return
        if wire_type == 1:
            self.pos += 8
            return
        if wire_type == 2:
            self.read_length_delimited()
            return
        if wire_type == 5:
            self.pos += 4
            return
        raise PprofDecodeError(f"Unsupported protobuf wire type: {wire_type}")


def _decode_text_index(strings: list[str], index: int) -> str:
    if index < 0 or index >= len(strings):
        return ""
    return strings[index]


def _read_packed_varints(payload: bytes) -> tuple[int, ...]:
    reader = _Reader(payload)
    values: list[int] = []
    while not reader.eof():
        values.append(reader.read_varint())
    return tuple(values)


@dataclass(slots=True)
class _RawValueType:
    type_index: int = 0
    unit_index: int = 0


def _parse_value_type(payload: bytes) -> _RawValueType:
    reader = _Reader(payload)
    result = _RawValueType()
    while not reader.eof():
        field, wire = reader.read_key()
        if wire != 0:
            reader.skip(wire)
            continue
        value = _to_signed64(reader.read_varint())
        if field == 1:
            result.type_index = value
        elif field == 2:
            result.unit_index = value
    return result


@dataclass(slots=True)
class _RawFunction:
    function_id: int = 0
    name: int = 0
    system_name: int = 0
    filename: int = 0
    start_line: int = 0


@dataclass(slots=True)
class _RawLine:
    function_id: int = 0
    line: int = 0


def _raw_lines() -> list[_RawLine]:
    return []


@dataclass(slots=True)
class _RawLocation:
    location_id: int = 0
    lines: list[_RawLine] = field(default_factory=_raw_lines)
    is_folded: bool = False


def _parse_function(payload: bytes) -> _RawFunction:
    reader = _Reader(payload)
    result = _RawFunction()
    while not reader.eof():
        field, wire = reader.read_key()
        if wire != 0:
            reader.skip(wire)
            continue
        value = reader.read_varint()
        if field == 1:
            result.function_id = value
        elif field == 2:
            result.name = value
        elif field == 3:
            result.system_name = value
        elif field == 4:
            result.filename = value
        elif field == 5:
            result.start_line = _to_signed64(value)
    return result


def _parse_line(payload: bytes) -> _RawLine:
    reader = _Reader(payload)
    result = _RawLine()
    while not reader.eof():
        field, wire = reader.read_key()
        if wire != 0:
            reader.skip(wire)
            continue
        value = reader.read_varint()
        if field == 1:
            result.function_id = value
        elif field == 2:
            result.line = _to_signed64(value)
    return result


def _parse_location(payload: bytes) -> _RawLocation:
    reader = _Reader(payload)
    result = _RawLocation()
    while not reader.eof():
        field, wire = reader.read_key()
        if field == 1 and wire == 0:
            result.location_id = reader.read_varint()
        elif field == 4 and wire == 2:
            result.lines.append(_parse_line(reader.read_length_delimited()))
        elif field == 5 and wire == 0:
            result.is_folded = bool(reader.read_varint())
        else:
            reader.skip(wire)
    return result


def _parse_sample(payload: bytes) -> Sample:
    reader = _Reader(payload)
    location_ids: list[int] = []
    values: list[int] = []
    while not reader.eof():
        field, wire = reader.read_key()
        if field == 1:
            if wire == 0:
                location_ids.append(reader.read_varint())
            elif wire == 2:
                location_ids.extend(
                    _read_packed_varints(reader.read_length_delimited())
                )
            else:
                reader.skip(wire)
        elif field == 2:
            if wire == 0:
                values.append(_to_signed64(reader.read_varint()))
            elif wire == 2:
                values.extend(
                    _to_signed64(value)
                    for value in _read_packed_varints(reader.read_length_delimited())
                )
            else:
                reader.skip(wire)
        else:
            reader.skip(wire)
    return Sample(location_ids=tuple(location_ids), values=tuple(values))


def decode_profile_bytes(data: bytes) -> Profile:
    """Decode raw or gzipped pprof protobuf bytes into the typed profile model."""
    try:
        payload = gzip.decompress(data)
    except (EOFError, zlib.error) as exc:
        raise PprofDecodeError(f"Truncated or corrupt gzip profile: {exc}") from exc
    except OSError:
        payload = data

    reader = _Reader(payload)
    strings: list[str] = []
    raw_sample_types: list[_RawValueType] = []
    raw_period_type: _RawValueType | None = None
    period = 0
    default_sample_type_index = 0
    raw_functions: list[_RawFunction] = []
    raw_locations: list[_RawLocation] = []
    samples: list[Sample] = []

    while not reader.eof():
        field, wire = reader.read_key()
        if field == 1 and wire == 2:
            raw_sample_types.append(_parse_value_type(reader.read_length_delimited()))
        elif field == 2 and wire == 2:
            samples.append(_parse_sample(reader.read_length_delimited()))
        elif field == 4 and wire == 2:
            raw_locations.append(_parse_location(reader.read_length_delimited()))
        elif field == 5 and wire == 2:
            raw_functions.append(_parse_function(reader.read_length_delimited()))
        elif field == 6 and wire == 2:
            strings.append(
                reader.read_length_delimited().decode("utf-8", errors="replace")
            )
        elif field == 11 and wire == 2:
            raw_period_type = _parse_value_type(reader.read_length_delimited())
        elif field == 12 and wire == 0:
            period = _to_signed64(reader.read_varint())
        elif field == 14 and wire == 0:
            default_sample_type_index = _to_signed64(reader.read_varint())
        else:
            reader.skip(wire)

    functions = {
        item.function_id: Function(
            function_id=item.function_id,
            name=_decode_text_index(strings, item.name),
            system_name=_decode_text_index(strings, item.system_name),
            filename=_decode_text_index(strings, item.filename),
            start_line=item.start_line,
        )
        for item in raw_functions
        if item.function_id > 0
    }
    locations = {
        item.location_id: Location(
            location_id=item.location_id,
            lines=tuple((line.function_id, line.line) for line in item.lines),
            is_folded=item.is_folded,
        )
        for item in raw_locations
        if item.location_id > 0
    }

    sample_types = tuple(
        ValueType(
            type_name=_decode_text_index(strings, raw.type_index),
            unit=_decode_text_index(strings, raw.unit_index),
        )
        for raw in raw_sample_types
    )
    period_type = (
        ValueType(
            type_name=_decode_text_index(strings, raw_period_type.type_index),
            unit=_decode_text_index(strings, raw_period_type.unit_index),
        )
        if raw_period_type is not None
        else None
    )
    default_sample_type = _decode_text_index(strings, default_sample_type_index)
    primary_value_index = select_primary_value_index(sample_types, default_sample_type)
    if primary_value_index != 0:
        samples = [
            replace(sample, primary_index=primary_value_index) for sample in samples
        ]

    return Profile(
        string_table=tuple(strings),
        functions=functions,
        locations=locations,
        samples=tuple(samples),
        sample_types=sample_types,
        period_type=period_type,
        period=period,
        default_sample_type=default_sample_type,
        primary_value_index=primary_value_index,
    )


def load_profile(path: str) -> Profile:
    with open(path, "rb") as handle:
        return decode_profile_bytes(handle.read())
