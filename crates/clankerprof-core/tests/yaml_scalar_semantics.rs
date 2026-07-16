//! Empirical typing of plain YAML scalars in value position under serde_yaml.
//!
//! Python's strict loader (clankerprof/jsonio.py) mirrors this table so both
//! implementations type identical config text identically; the assertions pin
//! serde_yaml's behavior so an engine upgrade that shifts scalar resolution
//! fails loudly here instead of silently diverging from Python. The Python
//! mirror of this table lives in tests/test_clankerprof.py
//! (test_clankerprof_strict_yaml_scalars_match_serde_yaml).

fn probe(scalar: &str) -> String {
    let doc = format!("v: {scalar}");
    match serde_yaml::from_str::<serde_yaml::Value>(&doc) {
        Err(_) => "parse-error".to_string(),
        Ok(value) => match &value["v"] {
            serde_yaml::Value::Null => "null".to_string(),
            serde_yaml::Value::Bool(b) => format!("bool:{b}"),
            serde_yaml::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    format!("int:{i}")
                } else if let Some(u) = n.as_u64() {
                    format!("int:{u}")
                } else {
                    let f = n.as_f64().unwrap();
                    if f.is_nan() {
                        "float:nan".to_string()
                    } else {
                        format!("float:{f:e}")
                    }
                }
            }
            serde_yaml::Value::String(s) => format!("str:{s}"),
            other => format!("other:{other:?}"),
        },
    }
}

/// One row per scalar: (plain scalar text, expected typing).
///
/// Shared with the Python mirror test — keep the two tables identical.
pub const SCALAR_TABLE: &[(&str, &str)] = &[
    // decimal integers (leading zeros demote to string; +/- signs allowed)
    ("12", "int:12"),
    ("-7", "int:-7"),
    ("+12", "int:12"),
    ("0", "int:0"),
    ("-0", "int:0"),
    ("+0", "int:0"),
    ("007", "str:007"),
    ("017", "str:017"),
    ("00", "str:00"),
    ("-007", "str:-007"),
    ("9223372036854775807", "int:9223372036854775807"),
    ("-9223372036854775808", "int:-9223372036854775808"),
    ("18446744073709551615", "int:18446744073709551615"),
    ("18446744073709551616", "parse-error"),
    ("-9223372036854775809", "parse-error"),
    // prefixed integers (signed, no underscores)
    ("0x1F", "int:31"),
    ("-0x1F", "int:-31"),
    ("+0x1F", "int:31"),
    ("0x1_F", "str:0x1_F"),
    ("-0x8000000000000000", "int:-9223372036854775808"),
    ("0x10000000000000000", "parse-error"),
    ("0o17", "int:15"),
    ("-0o17", "int:-15"),
    ("0o8", "str:0o8"),
    ("0o1_7", "str:0o1_7"),
    ("0b101", "int:5"),
    ("-0b101", "int:-5"),
    ("0b2", "str:0b2"),
    ("0b1_01", "str:0b1_01"),
    // YAML 1.1-only integer forms are plain strings
    ("1_0", "str:1_0"),
    ("1__0", "str:1__0"),
    ("1_0_0", "str:1_0_0"),
    ("1:2:3", "str:1:2:3"),
    ("60:1", "str:60:1"),
    // floats (dot or exponent required; unsigned exponents allowed)
    ("3.14", "float:3.14e0"),
    ("-2.5", "float:-2.5e0"),
    ("1.", "float:1e0"),
    (".5", "float:5e-1"),
    ("+.5", "float:5e-1"),
    ("-.5", "float:-5e-1"),
    ("+0.5", "float:5e-1"),
    (".0", "float:0e0"),
    ("0.", "float:0e0"),
    ("00.5", "float:5e-1"),
    ("007.5", "float:7.5e0"),
    ("1e2", "float:1e2"),
    ("1E2", "float:1e2"),
    ("1e+2", "float:1e2"),
    ("1e-2", "float:1e-2"),
    ("+1e2", "float:1e2"),
    ("-1e2", "float:-1e2"),
    ("01e2", "float:1e2"),
    ("12e03", "float:1.2e4"),
    ("1.5e10", "float:1.5e10"),
    ("1.5E+10", "float:1.5e10"),
    ("5.e3", "float:5e3"),
    (".5e3", "float:5e2"),
    ("0e0", "float:0e0"),
    ("0.0e0", "float:0e0"),
    ("-0.0", "float:-0e0"),
    // YAML 1.1-only float forms are plain strings
    ("1_0.5", "str:1_0.5"),
    ("1.5_5", "str:1.5_5"),
    ("1:2:3.5", "str:1:2:3.5"),
    // non-finite spellings (NaN forms unsigned-only); overflow stays string
    (".inf", "float:inf"),
    (".Inf", "float:inf"),
    (".INF", "float:inf"),
    ("-.inf", "float:-inf"),
    ("+.inf", "float:inf"),
    (".nan", "float:nan"),
    (".NaN", "float:nan"),
    (".NAN", "float:nan"),
    ("-.nan", "str:-.nan"),
    ("+.nan", "str:+.nan"),
    ("inf", "str:inf"),
    ("nan", "str:nan"),
    ("Infinity", "str:Infinity"),
    ("1e309", "str:1e309"),
    ("-1e309", "str:-1e309"),
    ("1e400", "str:1e400"),
    // booleans: true/false spellings only, never yes/no/on/off
    ("true", "bool:true"),
    ("True", "bool:true"),
    ("TRUE", "bool:true"),
    ("false", "bool:false"),
    ("False", "bool:false"),
    ("FALSE", "bool:false"),
    ("yes", "str:yes"),
    ("Yes", "str:Yes"),
    ("YES", "str:YES"),
    ("no", "str:no"),
    ("on", "str:on"),
    ("off", "str:off"),
    ("Off", "str:Off"),
    ("y", "str:y"),
    ("N", "str:N"),
    // nulls
    ("~", "null"),
    ("null", "null"),
    ("Null", "null"),
    ("NULL", "null"),
    ("", "null"),
    // misc plain strings
    ("2026-01-01", "str:2026-01-01"),
    ("=", "str:="),
    (".", "str:."),
    (".e5", "str:.e5"),
];

#[test]
fn serde_yaml_scalar_typing_matches_pinned_table() {
    let mut failures = Vec::new();
    for (scalar, expected) in SCALAR_TABLE {
        let actual = probe(scalar);
        if actual != *expected {
            failures.push(format!("{scalar:?}: expected {expected}, got {actual}"));
        }
    }
    assert!(
        failures.is_empty(),
        "scalar typing drifted:\n{}",
        failures.join("\n")
    );
}

#[test]
fn out_of_range_integer_error_names_the_decimal_value_and_width() {
    let positive = serde_yaml::from_str::<serde_yaml::Value>("v: 18446744073709551616")
        .unwrap_err()
        .to_string();
    assert!(
        positive.contains("invalid type: integer `18446744073709551616` as u128"),
        "unexpected error text: {positive}"
    );
    let negative = serde_yaml::from_str::<serde_yaml::Value>("v: -9223372036854775809")
        .unwrap_err()
        .to_string();
    assert!(
        negative.contains("invalid type: integer `-9223372036854775809` as i128"),
        "unexpected error text: {negative}"
    );
    let hex = serde_yaml::from_str::<serde_yaml::Value>("v: 0x10000000000000000")
        .unwrap_err()
        .to_string();
    assert!(
        hex.contains("invalid type: integer `18446744073709551616` as u128"),
        "unexpected error text: {hex}"
    );
}
