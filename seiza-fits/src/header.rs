//! FITS header card value parsing.

use fitsio_pure::value::Value;

/// A typed FITS header value.
#[derive(Debug, Clone, PartialEq)]
pub enum HeaderValue {
    Logical(bool),
    Integer(i64),
    Float(f64),
    String(String),
    /// Unparseable value, or an undefined value represented by an empty string.
    /// A quoted empty FITS string is instead `String(String::new())`.
    Raw(String),
}

impl HeaderValue {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Integer(v) => Some(*v as f64),
            Self::Float(v) => Some(*v),
            Self::String(s) => s.trim().parse().ok(),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Integer(v) => Some(*v),
            Self::Float(v) => Some(*v as i64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Logical(v) => Some(*v),
            _ => None,
        }
    }
}

/// Parse the value part of a header card (everything after `= `),
/// handling quoted strings with `''` escapes, trailing `/ comment`s,
/// logicals, integers, and floats (including FORTRAN `D` exponents).
/// Anything else, including an undefined value, is kept as `Raw` text.
pub fn parse_header_value(raw: &str) -> HeaderValue {
    match fitsio_pure::value::parse_value(raw.trim_start().as_bytes()) {
        Some((value, _)) if !matches!(value, Value::ComplexInt(..) | Value::ComplexFloat(..)) => {
            header_value(&value)
        }
        _ => HeaderValue::Raw(raw.split('/').next().unwrap_or("").trim().to_string()),
    }
}

/// Convert a value parsed by fitsio-pure. Complex values are kept as `Raw`
/// text and an undefined value as `Raw("")`.
pub(crate) fn header_value(value: &Value) -> HeaderValue {
    match value {
        Value::Logical(v) => HeaderValue::Logical(*v),
        Value::Integer(v) => HeaderValue::Integer(*v),
        Value::Float(v) => HeaderValue::Float(*v),
        Value::String(v) => HeaderValue::String(v.clone()),
        Value::ComplexInt(re, im) => HeaderValue::Raw(format!("({re}, {im})")),
        Value::ComplexFloat(re, im) => HeaderValue::Raw(format!("({re}, {im})")),
        Value::Undefined => HeaderValue::Raw(String::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_card_value_forms() {
        assert_eq!(
            parse_header_value("                   T"),
            HeaderValue::Logical(true)
        );
        assert_eq!(
            parse_header_value("                  16 / bits"),
            HeaderValue::Integer(16)
        );
        assert_eq!(
            parse_header_value("              32768.0 / offset"),
            HeaderValue::Float(32768.0)
        );
        assert_eq!(
            parse_header_value("  -1.0D-3 / fortran"),
            HeaderValue::Float(-0.001)
        );
        assert_eq!(
            parse_header_value("'ZWO ASI2600MM Pro' / camera"),
            HeaderValue::String("ZWO ASI2600MM Pro".to_string())
        );
        assert_eq!(
            parse_header_value("'it''s quoted'"),
            HeaderValue::String("it's quoted".to_string())
        );
    }

    #[test]
    fn distinguishes_undefined_values_from_empty_strings() {
        for raw in ["", "                    ", "        / no filter"] {
            assert_eq!(parse_header_value(raw), HeaderValue::Raw(String::new()));
        }
        for raw in ["''", "'        ' / empty string"] {
            assert_eq!(parse_header_value(raw), HeaderValue::String(String::new()));
        }
    }
}
