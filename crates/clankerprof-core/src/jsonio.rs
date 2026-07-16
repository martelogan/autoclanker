//! Strict JSON input parsing shared by clankerprof input surfaces.
//!
//! `serde_json::from_str::<Value>` silently keeps the last duplicate object
//! member, so the same multiset of members can change meaning with ordering —
//! the JSON-member-level twin of the order-dependent gate bypass the
//! duplicate-row rule kills, and of the YAML duplicate-key rule. Every Rust
//! JSON input surface routes through this module so duplicate member names
//! are validation errors in both implementations, never silent last-wins.
//! The `at line N column M` suffix serde appends is engine-specific detail,
//! exactly like the YAML duplicate-key contract.

use serde::de::{self, DeserializeSeed, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Number, Value};
use std::fmt;

struct StrictJson;

impl<'de> DeserializeSeed<'de> for StrictJson {
    type Value = Value;

    fn deserialize<D>(self, deserializer: D) -> Result<Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(StrictJsonVisitor)
    }
}

struct StrictJsonVisitor;

impl<'de> Visitor<'de> for StrictJsonVisitor {
    type Value = Value;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("any JSON value")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Value, E> {
        Ok(Value::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Value, E> {
        Ok(Value::Number(value.into()))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Value, E> {
        Ok(Value::Number(value.into()))
    }

    fn visit_f64<E: de::Error>(self, value: f64) -> Result<Value, E> {
        // The JSON grammar cannot produce non-finite floats, so this branch
        // never fires from parsing; it exists to avoid a lossy fallback.
        Number::from_f64(value)
            .map(Value::Number)
            .ok_or_else(|| de::Error::custom("non-finite JSON number"))
    }

    fn visit_str<E: de::Error>(self, value: &str) -> Result<Value, E> {
        Ok(Value::String(value.to_string()))
    }

    fn visit_string<E>(self, value: String) -> Result<Value, E> {
        Ok(Value::String(value))
    }

    fn visit_unit<E>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }

    fn visit_seq<A>(self, mut access: A) -> Result<Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut items = Vec::new();
        while let Some(item) = access.next_element_seed(StrictJson)? {
            items.push(item);
        }
        Ok(Value::Array(items))
    }

    fn visit_map<A>(self, mut access: A) -> Result<Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut map = Map::new();
        while let Some(key) = access.next_key::<String>()? {
            if map.contains_key(&key) {
                return Err(de::Error::custom(format!(
                    "duplicate entry with key \"{key}\""
                )));
            }
            map.insert(key, access.next_value_seed(StrictJson)?);
        }
        Ok(Value::Object(map))
    }
}

/// Parse JSON into a `Value`, rejecting duplicate object member names.
pub fn parse_strict_json(text: &str) -> Result<Value, String> {
    let mut deserializer = serde_json::Deserializer::from_str(text);
    let value = StrictJson
        .deserialize(&mut deserializer)
        .map_err(|error| error.to_string())?;
    deserializer.end().map_err(|error| error.to_string())?;
    Ok(value)
}

/// Duplicate-member check for surfaces that need a different parse shape
/// (the order-preserving target-config parse): only duplicate-key errors
/// surface here; other syntax errors defer to the caller's own parse and its
/// established messages.
pub fn ensure_no_duplicate_keys(text: &str) -> Result<(), String> {
    match parse_strict_json(text) {
        Err(error) if error.contains("duplicate entry with key") => Err(error),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_member_names_are_rejected_at_any_depth() {
        let error = parse_strict_json("{\"a\": 1, \"a\": 2}").expect_err("top-level duplicate");
        assert!(error.contains("duplicate entry with key \"a\""), "{error}");
        let error = parse_strict_json("{\"row\": [{\"pct\": 30.0, \"pct\": 10.0}]}")
            .expect_err("nested duplicate");
        assert!(
            error.contains("duplicate entry with key \"pct\""),
            "{error}"
        );
    }

    #[test]
    fn unique_members_parse_identically_to_serde_json() {
        let text = "{\"b\": [1, 2.5, null, true, \"x\"], \"a\": {\"k\": -3}}";
        let strict = parse_strict_json(text).expect("valid JSON parses");
        let plain: Value = serde_json::from_str(text).expect("valid JSON parses");
        assert_eq!(strict, plain);
    }

    #[test]
    fn ensure_no_duplicate_keys_only_surfaces_duplicates() {
        assert_eq!(ensure_no_duplicate_keys("{\"a\": 1}"), Ok(()));
        // Syntax errors defer to the caller's own parse and messages.
        assert_eq!(ensure_no_duplicate_keys("{not json"), Ok(()));
        let error =
            ensure_no_duplicate_keys("{\"a\": 1, \"a\": 2}").expect_err("duplicate surfaces");
        assert!(error.contains("duplicate entry with key \"a\""), "{error}");
    }

    #[test]
    fn trailing_garbage_is_still_a_parse_error() {
        let error = parse_strict_json("{\"a\": 1} extra").expect_err("trailing garbage");
        assert!(error.contains("trailing"), "{error}");
    }
}
