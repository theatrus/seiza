//! FITS header card value parsing.

/// A typed FITS header value.
#[derive(Debug, Clone, PartialEq)]
pub enum HeaderValue {
    Logical(bool),
    Integer(i64),
    Float(f64),
    String(String),
    /// Unparseable or empty value; the raw text is kept
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
pub fn parse_header_value(raw: &str) -> HeaderValue {
    let trimmed = raw.trim_start();

    // Quoted string: find the closing quote, honoring '' escapes
    if let Some(rest) = trimmed.strip_prefix('\'') {
        let mut value = String::new();
        let mut chars = rest.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '\'' {
                if chars.peek() == Some(&'\'') {
                    chars.next();
                    value.push('\'');
                } else {
                    break;
                }
            } else {
                value.push(c);
            }
        }
        return HeaderValue::String(value.trim_end().to_string());
    }

    // Strip the comment
    let value = trimmed.split('/').next().unwrap_or("").trim();
    match value {
        "T" => return HeaderValue::Logical(true),
        "F" => return HeaderValue::Logical(false),
        "" => return HeaderValue::Raw(String::new()),
        _ => {}
    }
    if let Ok(v) = value.parse::<i64>() {
        return HeaderValue::Integer(v);
    }
    // FORTRAN-style exponents use D instead of E
    let normalized = value.replace(['D', 'd'], "E");
    if let Ok(v) = normalized.parse::<f64>() {
        return HeaderValue::Float(v);
    }
    HeaderValue::Raw(value.to_string())
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
}
