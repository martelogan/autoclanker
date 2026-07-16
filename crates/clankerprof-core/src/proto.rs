use crate::model::{
    select_primary_value_index, Function, Location, Profile, Sample, TimeNs, ValueType,
};
use flate2::bufread::GzDecoder;
use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use std::fs;
use std::io::Read;
use std::path::Path;

#[derive(Debug)]
pub enum PprofDecodeError {
    Io(std::io::Error),
    InvalidProtobuf(String),
}

impl Display for PprofDecodeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "{error}"),
            Self::InvalidProtobuf(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for PprofDecodeError {}

impl From<std::io::Error> for PprofDecodeError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

type DecodeResult<T> = Result<T, PprofDecodeError>;

#[derive(Debug)]
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn eof(&self) -> bool {
        self.pos >= self.data.len()
    }

    fn read_byte(&mut self) -> DecodeResult<u8> {
        let Some(value) = self.data.get(self.pos).copied() else {
            return Err(PprofDecodeError::InvalidProtobuf(
                "Unexpected end of protobuf stream.".to_string(),
            ));
        };
        self.pos += 1;
        Ok(value)
    }

    fn read_varint(&mut self) -> DecodeResult<u64> {
        let mut shift = 0u32;
        let mut result = 0u64;
        loop {
            let byte = self.read_byte()?;
            // Bits past 63 are dropped, matching protobuf's 64-bit varint
            // wrap; the shift guard below caps varints at 10 bytes so this
            // can never overflow-shift.
            result |= u64::from(byte & 0x7f).wrapping_shl(shift);
            if byte < 0x80 {
                return Ok(result);
            }
            shift += 7;
            if shift >= 70 {
                return Err(PprofDecodeError::InvalidProtobuf(
                    "Invalid protobuf varint.".to_string(),
                ));
            }
        }
    }

    fn read_key(&mut self) -> DecodeResult<(u64, u8)> {
        let key = self.read_varint()?;
        let field = key >> 3;
        // Field number 0 is illegal protobuf; shared by every message parser.
        if field == 0 {
            return Err(PprofDecodeError::InvalidProtobuf(
                "Illegal protobuf field number 0.".to_string(),
            ));
        }
        // Tags must fit uint32, capping field numbers at 2^29 - 1; conformant
        // serializers cannot emit anything above it, so it is malformed input.
        if field > 0x1FFF_FFFF {
            return Err(PprofDecodeError::InvalidProtobuf(format!(
                "Illegal protobuf field number {field}."
            )));
        }
        Ok((field, (key & 0x07) as u8))
    }

    fn read_length_delimited(&mut self) -> DecodeResult<&'a [u8]> {
        let length = self.read_varint()? as usize;
        let end = self.pos.checked_add(length).ok_or_else(|| {
            PprofDecodeError::InvalidProtobuf(
                "Length-delimited field extends beyond stream.".to_string(),
            )
        })?;
        if end > self.data.len() {
            return Err(PprofDecodeError::InvalidProtobuf(
                "Length-delimited field extends beyond stream.".to_string(),
            ));
        }
        let payload = &self.data[self.pos..end];
        self.pos = end;
        Ok(payload)
    }

    fn skip(&mut self, wire_type: u8, field: u64) -> DecodeResult<()> {
        self.skip_with_depth(wire_type, field, 0)
    }

    fn skip_with_depth(&mut self, wire_type: u8, field: u64, depth: u32) -> DecodeResult<()> {
        match wire_type {
            0 => {
                self.read_varint()?;
                Ok(())
            }
            1 => self.advance(8),
            2 => {
                self.read_length_delimited()?;
                Ok(())
            }
            // Deprecated-but-legal group: skip balanced nested fields until
            // the matching end-group key. Truncation/imbalance still errors.
            3 => {
                if depth >= 128 {
                    return Err(PprofDecodeError::InvalidProtobuf(
                        "Protobuf group nesting exceeds the supported depth.".to_string(),
                    ));
                }
                loop {
                    let (inner_field, inner_wire) = self.read_key()?;
                    if inner_wire == 4 {
                        if inner_field != field {
                            return Err(PprofDecodeError::InvalidProtobuf(
                                "Mismatched protobuf group end.".to_string(),
                            ));
                        }
                        return Ok(());
                    }
                    self.skip_with_depth(inner_wire, inner_field, depth + 1)?;
                }
            }
            4 => Err(PprofDecodeError::InvalidProtobuf(
                "Unmatched protobuf group end.".to_string(),
            )),
            5 => self.advance(4),
            _ => Err(PprofDecodeError::InvalidProtobuf(format!(
                "Unsupported protobuf wire type: {wire_type}"
            ))),
        }
    }

    fn advance(&mut self, count: usize) -> DecodeResult<()> {
        let end = self.pos.checked_add(count).ok_or_else(|| {
            PprofDecodeError::InvalidProtobuf("Skip extends beyond stream.".to_string())
        })?;
        if end > self.data.len() {
            return Err(PprofDecodeError::InvalidProtobuf(
                "Skip extends beyond stream.".to_string(),
            ));
        }
        self.pos = end;
        Ok(())
    }
}

#[derive(Debug, Default)]
struct RawValueType {
    type_index: i64,
    unit_index: i64,
}

fn parse_value_type(payload: &[u8]) -> DecodeResult<RawValueType> {
    let mut reader = Reader::new(payload);
    let mut result = RawValueType::default();
    while !reader.eof() {
        let (field, wire) = reader.read_key()?;
        if wire != 0 {
            reader.skip(wire, field)?;
            continue;
        }
        let value = int64_from_varint(reader.read_varint()?);
        match field {
            1 => result.type_index = value,
            2 => result.unit_index = value,
            _ => {}
        }
    }
    Ok(result)
}

fn string_at_signed(strings: &[String], index: i64) -> String {
    usize::try_from(index)
        .ok()
        .and_then(|position| strings.get(position))
        .cloned()
        .unwrap_or_default()
}

#[derive(Debug, Default)]
struct RawFunction {
    function_id: u64,
    name: usize,
    system_name: usize,
    filename: usize,
    start_line: i64,
}

#[derive(Debug, Default)]
struct RawLine {
    function_id: u64,
    line: i64,
}

#[derive(Debug, Default)]
struct RawLocation {
    location_id: u64,
    lines: Vec<RawLine>,
    is_folded: bool,
}

pub fn load_profile(path: impl AsRef<Path>) -> DecodeResult<Profile> {
    decode_profile_bytes(&fs::read(path)?)
}

pub fn decode_profile_bytes(data: &[u8]) -> DecodeResult<Profile> {
    let payload = maybe_gzip_decompress(data)?;
    let mut reader = Reader::new(&payload);
    let mut strings = Vec::new();
    let mut raw_sample_types = Vec::new();
    let mut period_type_bytes: Option<Vec<u8>> = None;
    let mut period = 0i64;
    let mut default_sample_type_index = 0i64;
    let mut raw_functions = Vec::new();
    let mut raw_locations = Vec::new();
    let mut samples = Vec::new();

    while !reader.eof() {
        let (field, wire) = reader.read_key()?;
        match (field, wire) {
            (1, 2) => raw_sample_types.push(parse_value_type(reader.read_length_delimited()?)?),
            (2, 2) => samples.push(parse_sample(reader.read_length_delimited()?)?),
            (4, 2) => raw_locations.push(parse_location(reader.read_length_delimited()?)?),
            (5, 2) => raw_functions.push(parse_function(reader.read_length_delimited()?)?),
            (6, 2) => strings.push(decode_utf8_strict(reader.read_length_delimited()?)?),
            // Singular embedded-message fields merge across occurrences in
            // protobuf; concatenating the payloads is exactly that merge for
            // a scalar-field submessage (later set fields win on re-parse).
            (11, 2) => period_type_bytes
                .get_or_insert_with(Vec::new)
                .extend_from_slice(reader.read_length_delimited()?),
            (12, 0) => period = int64_from_varint(reader.read_varint()?),
            (14, 0) => default_sample_type_index = int64_from_varint(reader.read_varint()?),
            _ => reader.skip(wire, field)?,
        }
    }

    let functions = raw_functions
        .into_iter()
        .filter(|item| item.function_id > 0)
        .map(|item| {
            (
                item.function_id,
                Function {
                    function_id: item.function_id,
                    name: string_at(&strings, item.name),
                    system_name: string_at(&strings, item.system_name),
                    filename: string_at(&strings, item.filename),
                    start_line: item.start_line,
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    let locations = raw_locations
        .into_iter()
        .filter(|item| item.location_id > 0)
        .map(|item| {
            (
                item.location_id,
                Location {
                    location_id: item.location_id,
                    lines: item
                        .lines
                        .into_iter()
                        .map(|line| (line.function_id, line.line))
                        .collect(),
                    is_folded: item.is_folded,
                },
            )
        })
        .collect::<BTreeMap<_, _>>();

    let sample_types: Vec<ValueType> = raw_sample_types
        .into_iter()
        .map(|raw| ValueType {
            type_name: string_at_signed(&strings, raw.type_index),
            unit: string_at_signed(&strings, raw.unit_index),
        })
        .collect();
    let raw_period_type = match period_type_bytes {
        Some(bytes) => Some(parse_value_type(&bytes)?),
        None => None,
    };
    let period_type = raw_period_type.map(|raw| ValueType {
        type_name: string_at_signed(&strings, raw.type_index),
        unit: string_at_signed(&strings, raw.unit_index),
    });
    let default_sample_type = string_at_signed(&strings, default_sample_type_index);
    let primary_value_index = select_primary_value_index(&sample_types, &default_sample_type);
    for sample in &mut samples {
        sample.primary_index = primary_value_index;
    }

    Ok(Profile {
        string_table: strings,
        functions,
        locations,
        samples,
        sample_types,
        period_type,
        period,
        default_sample_type,
        primary_value_index,
    })
}

fn maybe_gzip_decompress(data: &[u8]) -> DecodeResult<Vec<u8>> {
    if !data.starts_with(&[0x1f, 0x8b]) {
        return Ok(data.to_vec());
    }
    // Mirror CPython's gzip module member by member: RFC 1952 allows a stream
    // of members, and zero padding after any member is consumed (gzip FAQ #8,
    // exactly like CPython's _read_eof). Non-zero trailing bytes make CPython
    // raise BadGzipFile, which decode_profile_bytes treats as "not gzip after
    // all" and re-parses the original bytes as raw protobuf — so this returns
    // the raw data for that case instead of erroring.
    let mut rest: &[u8] = data;
    let mut payload = Vec::new();
    loop {
        let mut decoder = GzDecoder::new(rest);
        decoder.read_to_end(&mut payload)?;
        rest = decoder.into_inner();
        while let Some((&0, tail)) = rest.split_first() {
            rest = tail;
        }
        if rest.is_empty() {
            return Ok(payload);
        }
        if !rest.starts_with(&[0x1f, 0x8b]) {
            return Ok(data.to_vec());
        }
    }
}

fn parse_function(payload: &[u8]) -> DecodeResult<RawFunction> {
    let mut reader = Reader::new(payload);
    let mut result = RawFunction::default();
    while !reader.eof() {
        let (field, wire) = reader.read_key()?;
        if wire != 0 {
            reader.skip(wire, field)?;
            continue;
        }
        let value = reader.read_varint()?;
        match field {
            1 => result.function_id = value,
            2 => result.name = value as usize,
            3 => result.system_name = value as usize,
            4 => result.filename = value as usize,
            5 => result.start_line = int64_from_varint(value),
            _ => {}
        }
    }
    Ok(result)
}

fn parse_line(payload: &[u8]) -> DecodeResult<RawLine> {
    let mut reader = Reader::new(payload);
    let mut result = RawLine::default();
    while !reader.eof() {
        let (field, wire) = reader.read_key()?;
        if wire != 0 {
            reader.skip(wire, field)?;
            continue;
        }
        let value = reader.read_varint()?;
        match field {
            1 => result.function_id = value,
            2 => result.line = int64_from_varint(value),
            _ => {}
        }
    }
    Ok(result)
}

fn parse_location(payload: &[u8]) -> DecodeResult<RawLocation> {
    let mut reader = Reader::new(payload);
    let mut result = RawLocation::default();
    while !reader.eof() {
        let (field, wire) = reader.read_key()?;
        match (field, wire) {
            (1, 0) => result.location_id = reader.read_varint()?,
            (4, 2) => result
                .lines
                .push(parse_line(reader.read_length_delimited()?)?),
            (5, 0) => result.is_folded = reader.read_varint()? != 0,
            _ => reader.skip(wire, field)?,
        }
    }
    Ok(result)
}

fn parse_sample(payload: &[u8]) -> DecodeResult<Sample> {
    let mut reader = Reader::new(payload);
    let mut location_ids = Vec::new();
    let mut values = Vec::new();
    while !reader.eof() {
        let (field, wire) = reader.read_key()?;
        match field {
            1 => match wire {
                0 => location_ids.push(reader.read_varint()?),
                2 => location_ids.extend(read_packed_varints(reader.read_length_delimited()?)?),
                _ => reader.skip(wire, field)?,
            },
            2 => match wire {
                0 => values.push(TimeNs::from(int64_from_varint(reader.read_varint()?))),
                2 => values.extend(
                    read_packed_varints(reader.read_length_delimited()?)?
                        .into_iter()
                        .map(|raw| TimeNs::from(int64_from_varint(raw))),
                ),
                _ => reader.skip(wire, field)?,
            },
            _ => reader.skip(wire, field)?,
        }
    }
    Ok(Sample {
        location_ids,
        values,
        primary_index: 0,
    })
}

fn read_packed_varints(payload: &[u8]) -> DecodeResult<Vec<u64>> {
    let mut reader = Reader::new(payload);
    let mut values = Vec::new();
    while !reader.eof() {
        values.push(reader.read_varint()?);
    }
    Ok(values)
}

fn decode_utf8_strict(data: &[u8]) -> DecodeResult<String> {
    // Python's strict "utf-8" codec and String::from_utf8 accept exactly the
    // same byte set (well-formed UTF-8, surrogates rejected in both).
    String::from_utf8(data.to_vec()).map_err(|_| {
        PprofDecodeError::InvalidProtobuf("Invalid UTF-8 in pprof string table.".to_string())
    })
}

fn string_at(strings: &[String], index: usize) -> String {
    strings.get(index).cloned().unwrap_or_default()
}

fn int64_from_varint(value: u64) -> i64 {
    value as i64
}

#[cfg(test)]
mod gzip_tests {
    use super::maybe_gzip_decompress;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;

    fn gzip_member(payload: &[u8]) -> Vec<u8> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(payload).expect("write member");
        encoder.finish().expect("finish member")
    }

    #[test]
    fn concatenated_gzip_members_all_decode() {
        // RFC 1952 allows a stream of members; Python's gzip.decompress
        // reads them all, so a single-member decode silently drops data.
        let mut stream = gzip_member(&[0x32, 0x00]);
        stream.extend(gzip_member(&[0x12, 0x02, 0x10, 0x07]));
        let decoded = maybe_gzip_decompress(&stream).expect("decode");
        assert_eq!(decoded, vec![0x32, 0x00, 0x12, 0x02, 0x10, 0x07]);
    }

    #[test]
    fn truncated_gzip_still_errors() {
        let mut stream = gzip_member(&[0x32, 0x00]);
        stream.truncate(stream.len() - 5);
        assert!(maybe_gzip_decompress(&stream).is_err());
    }

    #[test]
    fn zero_padded_gzip_members_decode() {
        // Zero padding after any member is conventional (gzip FAQ #8) and
        // consumed by CPython's gzip module — after the final member and
        // between members alike.
        let mut trailing = gzip_member(&[0x32, 0x00]);
        trailing.extend([0u8; 16]);
        assert_eq!(
            maybe_gzip_decompress(&trailing).expect("trailing padding"),
            vec![0x32, 0x00]
        );
        let mut between = gzip_member(&[0x32, 0x00]);
        between.extend([0u8; 8]);
        between.extend(gzip_member(&[0x12, 0x02, 0x10, 0x07]));
        assert_eq!(
            maybe_gzip_decompress(&between).expect("padding between members"),
            vec![0x32, 0x00, 0x12, 0x02, 0x10, 0x07]
        );
    }

    #[test]
    fn nonzero_trailing_garbage_falls_back_to_raw_bytes() {
        // CPython raises BadGzipFile on non-zero trailing bytes and
        // decode_profile_bytes then re-parses the original bytes as raw
        // protobuf; the decoder mirrors that by returning the input.
        let mut stream = gzip_member(&[0x32, 0x00]);
        stream.extend([0x01, 0x02, 0x03]);
        assert_eq!(maybe_gzip_decompress(&stream).expect("fallback"), stream);
    }
}

#[cfg(test)]
mod group_skip_tests {
    use super::*;

    fn base_profile() -> Vec<u8> {
        // string table [""], one sample value 7
        vec![0x32, 0x00, 0x12, 0x02, 0x10, 0x07]
    }

    #[test]
    fn balanced_unknown_groups_are_skipped() {
        // field 15: SGROUP (0x7b) ... EGROUP (0x7c); empty and nested forms.
        let mut empty = vec![0x7b, 0x7c];
        empty.extend(base_profile());
        let profile = decode_profile_bytes(&empty).expect("empty group skips");
        assert_eq!(profile.samples.len(), 1);

        // field 17 group nested inside the field 15 group.
        let mut nested = vec![0x7b, 0x8b, 0x01, 0x8c, 0x01, 0x7c];
        nested.extend(base_profile());
        let profile = decode_profile_bytes(&nested).expect("nested group skips");
        assert_eq!(profile.samples.len(), 1);
    }

    #[test]
    fn over_maximum_field_numbers_are_decode_errors() {
        // Field number 2^29 is one above protobuf's tag maximum.
        let over = decode_profile_bytes(&[0x80, 0x80, 0x80, 0x80, 0x10, 0x00])
            .expect_err("field 2^29 is illegal");
        assert_eq!(over.to_string(), "Illegal protobuf field number 536870912.");
        // The maximum legal field number (2^29 - 1) still skips as unknown.
        let profile = decode_profile_bytes(&[0xf8, 0xff, 0xff, 0xff, 0x0f, 0x00])
            .expect("max legal field skips");
        assert_eq!(profile.samples.len(), 0);
    }

    #[test]
    fn unbalanced_groups_are_decode_errors() {
        let truncated = decode_profile_bytes(&[0x7b]).expect_err("truncated group");
        assert_eq!(truncated.to_string(), "Unexpected end of protobuf stream.");

        let stray = decode_profile_bytes(&[0x7c]).expect_err("stray group end");
        assert_eq!(stray.to_string(), "Unmatched protobuf group end.");

        // field 15 group closed by a field 16 end key (0x84 0x01).
        let mismatched =
            decode_profile_bytes(&[0x7b, 0x84, 0x01]).expect_err("mismatched group end");
        assert_eq!(mismatched.to_string(), "Mismatched protobuf group end.");

        // 200 nested opens exceed the 128-level cap before any end key.
        let mut deep = vec![0x7b; 200];
        deep.push(0x7c);
        let capped = decode_profile_bytes(&deep).expect_err("depth cap");
        assert_eq!(
            capped.to_string(),
            "Protobuf group nesting exceeds the supported depth."
        );
    }

    #[test]
    fn repeated_period_type_occurrences_merge() {
        // strings ["", "cpu", "nanoseconds"], then period_type split across
        // two field-11 occurrences: {type: 1} and {unit: 2}.
        let bytes = [
            0x32, 0x00, 0x32, 0x03, b'c', b'p', b'u', 0x32, 0x0b, b'n', b'a', b'n', b'o', b's',
            b'e', b'c', b'o', b'n', b'd', b's', 0x5a, 0x02, 0x08, 0x01, 0x5a, 0x02, 0x10, 0x02,
        ];
        let profile = decode_profile_bytes(&bytes).expect("merged period_type");
        let period_type = profile.period_type.expect("period_type present");
        assert_eq!(period_type.type_name, "cpu");
        assert_eq!(period_type.unit, "nanoseconds");
    }
}
