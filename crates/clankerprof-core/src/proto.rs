use crate::model::{select_primary_value_index, Function, Location, Profile, Sample, ValueType};
use flate2::read::GzDecoder;
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
        Ok((key >> 3, (key & 0x07) as u8))
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

    fn skip(&mut self, wire_type: u8) -> DecodeResult<()> {
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
            reader.skip(wire)?;
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
    let mut raw_period_type: Option<RawValueType> = None;
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
            (6, 2) => strings.push(decode_utf8_lossy(reader.read_length_delimited()?)),
            (11, 2) => raw_period_type = Some(parse_value_type(reader.read_length_delimited()?)?),
            (12, 0) => period = int64_from_varint(reader.read_varint()?),
            (14, 0) => default_sample_type_index = int64_from_varint(reader.read_varint()?),
            _ => reader.skip(wire)?,
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
    if data.starts_with(&[0x1f, 0x8b]) {
        let mut decoder = GzDecoder::new(data);
        let mut payload = Vec::new();
        decoder.read_to_end(&mut payload)?;
        return Ok(payload);
    }
    Ok(data.to_vec())
}

fn parse_function(payload: &[u8]) -> DecodeResult<RawFunction> {
    let mut reader = Reader::new(payload);
    let mut result = RawFunction::default();
    while !reader.eof() {
        let (field, wire) = reader.read_key()?;
        if wire != 0 {
            reader.skip(wire)?;
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
            reader.skip(wire)?;
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
            _ => reader.skip(wire)?,
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
                _ => reader.skip(wire)?,
            },
            2 => match wire {
                0 => values.push(int64_from_varint(reader.read_varint()?)),
                2 => values.extend(
                    read_packed_varints(reader.read_length_delimited()?)?
                        .into_iter()
                        .map(int64_from_varint),
                ),
                _ => reader.skip(wire)?,
            },
            _ => reader.skip(wire)?,
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

fn decode_utf8_lossy(data: &[u8]) -> String {
    String::from_utf8_lossy(data).into_owned()
}

fn string_at(strings: &[String], index: usize) -> String {
    strings.get(index).cloned().unwrap_or_default()
}

fn int64_from_varint(value: u64) -> i64 {
    value as i64
}
