//! Python `json.dumps`-compatible serialization.
//!
//! Non-facts artifacts, receipts, and error envelopes are contracted to be
//! byte-identical to the Python reference, which emits them through
//! `json.dumps(payload, sort_keys=True, ...)` with default `ensure_ascii=True`
//! and CPython `repr` float spelling. serde_json's writer differs on exactly
//! those two axes (raw UTF-8, ryu exponent spelling), so this module renders
//! `serde_json::Value` trees the way CPython does. Facts artifacts are the
//! deliberate exception: Python writes them with `ensure_ascii=False`, and the
//! existing facts writer already matches byte-for-byte.

use serde_json::Value;

/// `json.dumps(value, indent=2, sort_keys=True)` equivalent.
pub fn dumps_pretty(value: &Value) -> String {
    let mut output = String::new();
    write_value(&mut output, value, Some(2), 0);
    output
}

/// `json.dumps(value, sort_keys=True)` equivalent (default separators).
pub fn dumps_compact(value: &Value) -> String {
    let mut output = String::new();
    write_value(&mut output, value, None, 0);
    output
}

fn write_value(output: &mut String, value: &Value, indent: Option<usize>, depth: usize) {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(true) => output.push_str("true"),
        Value::Bool(false) => output.push_str("false"),
        Value::Number(number) => {
            if let Some(float) = number
                .as_f64()
                .filter(|_| !number.is_i64() && !number.is_u64())
            {
                output.push_str(&format_pyfloat(float));
            } else {
                output.push_str(&number.to_string());
            }
        }
        Value::String(text) => write_string(output, text),
        Value::Array(items) => {
            if items.is_empty() {
                output.push_str("[]");
                return;
            }
            output.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                    if indent.is_none() {
                        output.push(' ');
                    }
                }
                push_newline_indent(output, indent, depth + 1);
                write_value(output, item, indent, depth + 1);
            }
            push_newline_indent(output, indent, depth);
            output.push(']');
        }
        Value::Object(entries) => {
            if entries.is_empty() {
                output.push_str("{}");
                return;
            }
            // serde_json's default Map is BTreeMap-backed; UTF-8 byte order
            // equals code-point order, so iteration matches sort_keys=True.
            output.push('{');
            for (index, (key, item)) in entries.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                    if indent.is_none() {
                        output.push(' ');
                    }
                }
                push_newline_indent(output, indent, depth + 1);
                write_string(output, key);
                output.push_str(": ");
                write_value(output, item, indent, depth + 1);
            }
            push_newline_indent(output, indent, depth);
            output.push('}');
        }
    }
}

fn push_newline_indent(output: &mut String, indent: Option<usize>, depth: usize) {
    if let Some(width) = indent {
        output.push('\n');
        for _ in 0..depth * width {
            output.push(' ');
        }
    }
}

/// `json.dumps` string escaping with `ensure_ascii=True`: everything outside
/// the printable ASCII range becomes `\uXXXX` (surrogate pairs for astral
/// code points), matching CPython's ESCAPE_ASCII table.
fn write_string(output: &mut String, text: &str) {
    output.push('"');
    for character in text.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            '\u{8}' => output.push_str("\\b"),
            '\u{c}' => output.push_str("\\f"),
            ' '..='~' => output.push(character),
            _ => {
                let code_point = character as u32;
                if code_point > 0xFFFF {
                    let reduced = code_point - 0x10000;
                    let high = 0xD800 + (reduced >> 10);
                    let low = 0xDC00 + (reduced & 0x3FF);
                    output.push_str(&format!("\\u{high:04x}\\u{low:04x}"));
                } else {
                    output.push_str(&format!("\\u{code_point:04x}"));
                }
            }
        }
    }
    output.push('"');
}

/// CPython `repr(float)` spelling: shortest round-trip digits, fixed notation
/// for decimal exponents in [-4, 16), otherwise scientific with a mandatory
/// exponent sign and at least two exponent digits (`1e-06`, `1e+21`), and a
/// trailing `.0` on integral fixed-notation values.
pub fn format_pyfloat(value: f64) -> String {
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0.0".to_string()
        } else {
            "0.0".to_string()
        };
    }
    // {:e} yields shortest round-trip digits like CPython repr, but the two
    // differ on exact decimal midpoints (see repair_midpoint_tie).
    let scientific = format!("{value:e}");
    let (mantissa, exponent) = scientific
        .split_once('e')
        .expect("{:e} always contains an exponent");
    let (sign, mantissa) = match mantissa.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", mantissa),
    };
    let digits: String = mantissa.chars().filter(|ch| *ch != '.').collect();
    let digits = repair_midpoint_tie(value, digits);
    let exponent: i32 = exponent.parse().expect("{:e} exponent is an integer");
    if (-4..16).contains(&exponent) {
        let mut body = String::new();
        if exponent < 0 {
            body.push_str("0.");
            for _ in 0..(-exponent - 1) {
                body.push('0');
            }
            body.push_str(&digits);
        } else {
            let point = (exponent + 1) as usize;
            if point >= digits.len() {
                body.push_str(&digits);
                for _ in 0..(point - digits.len()) {
                    body.push('0');
                }
                body.push_str(".0");
            } else {
                body.push_str(&digits[..point]);
                body.push('.');
                body.push_str(&digits[point..]);
            }
        }
        format!("{sign}{body}")
    } else {
        let mantissa_text = if digits.len() == 1 {
            digits
        } else {
            format!("{}.{}", &digits[..1], &digits[1..])
        };
        format!("{sign}{mantissa_text}e{:+03}", exponent)
    }
}

/// CPython (Gay dtoa mode 0) rounds an exact decimal midpoint between the two
/// shortest round-trip candidates half-to-even, while Rust's flt2dec rounds it
/// away from zero. When the upper candidate ends in an odd digit the spellings
/// diverge (e.g. 1317225046594893.25: Python "…893.2", Rust "…893.3"), so
/// detect the exact tie and step down to the even lower candidate.
fn repair_midpoint_tie(value: f64, digits: String) -> String {
    let last = *digits.as_bytes().last().expect("shortest digits non-empty");
    if last == b'0' || (last - b'0') % 2 == 0 {
        // Already even (half-to-even agrees), and a tie rounded up to a
        // trailing zero would have a shorter spelling flt2dec would have
        // chosen instead, so '0' never needs a borrow-decrement.
        return digits;
    }
    // The midpoint between candidates D(q-1) and D(q) has significant digits
    // D, then q-1, then a single 5. An f64's exact decimal expansion has at
    // most 767 significant digits, so if the correctly rounded 781-digit
    // expansion equals midpoint-then-zeros, the value IS the midpoint. A
    // shortest spelling that rounded up across a power of ten cannot be an
    // odd-digit tie (its upper candidate would end in 0), so positional
    // comparison against the exact expansion is alignment-safe.
    let exact = format!("{:.780e}", value.abs());
    let (mantissa, _) = exact.split_once('e').expect("{:e} contains an exponent");
    let expansion: Vec<u8> = mantissa.bytes().filter(|byte| *byte != b'.').collect();
    let head = &digits.as_bytes()[..digits.len() - 1];
    let is_tie = expansion.len() > head.len() + 2
        && expansion.starts_with(head)
        && expansion[head.len()] == last - 1
        && expansion[head.len() + 1] == b'5'
        && expansion[head.len() + 2..].iter().all(|byte| *byte == b'0');
    if !is_tie {
        return digits;
    }
    let mut bytes = digits.into_bytes();
    *bytes.last_mut().expect("digits non-empty") -= 1;
    String::from_utf8(bytes).expect("digits are ASCII")
}

#[cfg(test)]
mod tests {
    use super::{dumps_compact, dumps_pretty, format_pyfloat};
    use serde_json::json;

    #[test]
    fn pyfloat_matches_cpython_repr_grammar() {
        // Expectations are CPython repr() outputs.
        let table: &[(f64, &str)] = &[
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (0.1, "0.1"),
            (1.0, "1.0"),
            (-2.5, "-2.5"),
            (100.0, "100.0"),
            (1e-4, "0.0001"),
            (1e-5, "1e-05"),
            (1e-6, "1e-06"),
            (1.2345678901234567e-8, "1.2345678901234567e-08"),
            (7.123456789012345e22, "7.123456789012345e+22"),
            (1234.5, "1234.5"),
            (1e15, "1000000000000000.0"),
            (9.9e15, "9900000000000000.0"),
            (1e16, "1e+16"),
            (1.5e16, "1.5e+16"),
            (1e21, "1e+21"),
            (1.7976931348623157e308, "1.7976931348623157e+308"),
            (5e-324, "5e-324"),
            (2.220446049250313e-16, "2.220446049250313e-16"),
            (0.30000000000000004, "0.30000000000000004"),
            (33.333333333333336, "33.333333333333336"),
            (-1e-07, "-1e-07"),
            // Exact decimal midpoints: CPython rounds half-to-even where
            // flt2dec rounds away from zero (repair_midpoint_tie).
            (1317225046594893.25, "1317225046594893.2"),
            (-1317225046594793.25, "-1317225046594793.2"),
            // Control: an even-rounding tie needs no repair.
            (1317225046594893.75, "1317225046594893.8"),
            // Controls: odd last digits that are NOT ties stay untouched.
            (0.1, "0.1"),
            (2.5e-09, "2.5e-09"),
        ];
        for (value, expected) in table {
            assert_eq!(&format_pyfloat(*value), expected, "repr({value:?})");
        }
    }

    #[test]
    fn strings_escape_like_ensure_ascii() {
        let payload =
            json!({"path": "/caf\u{e9}/x", "emoji": "a\u{1F600}b", "ctl": "x\u{1f}\u{7f}y"});
        assert_eq!(
            dumps_compact(&payload),
            "{\"ctl\": \"x\\u001f\\u007fy\", \"emoji\": \"a\\ud83d\\ude00b\", \"path\": \"/caf\\u00e9/x\"}"
        );
    }

    #[test]
    fn pretty_layout_matches_json_dumps_indent_2() {
        let payload = json!({"b": [1, {"y": 1e-6}], "a": {}, "c": []});
        assert_eq!(
            dumps_pretty(&payload),
            "{\n  \"a\": {},\n  \"b\": [\n    1,\n    {\n      \"y\": 1e-06\n    }\n  ],\n  \"c\": []\n}"
        );
    }
}
