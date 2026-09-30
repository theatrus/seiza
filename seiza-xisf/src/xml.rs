//! The XISF header as a tree of elements, kept whole so that a later write
//! can carry every element this crate does not interpret.

use crate::XisfError;
use quick_xml::escape::escape;
use quick_xml::events::{BytesStart, Event};
use quick_xml::{Reader, XmlVersion};
use std::collections::BTreeMap;
use std::fmt::Write as _;

/// An XML element of an XISF header, with everything inside it.
///
/// Character data is kept as one string per element. XISF elements hold
/// either character data or child elements, and white space between child
/// elements is not significant, so nothing that matters is lost.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct XisfElement {
    /// The element name as written, with any namespace prefix.
    pub name: String,
    /// Attributes by name as written, with any namespace prefix.
    pub attributes: BTreeMap<String, String>,
    pub text: String,
    pub children: Vec<XisfElement>,
    /// The stored bytes of an attached or external data block this element
    /// locates, loaded so the block can travel with the element. Inline and
    /// embedded blocks stay in `text` or in a `Data` child.
    pub block: Option<Vec<u8>>,
}

impl XisfElement {
    /// The name without its namespace prefix.
    pub fn local_name(&self) -> &str {
        self.name.rsplit(':').next().unwrap_or(&self.name)
    }

    pub fn attribute(&self, name: &str) -> Option<&str> {
        self.attributes.get(name).map(String::as_str)
    }

    /// The first child element with a given local name.
    pub fn child(&self, local_name: &str) -> Option<&XisfElement> {
        self.children
            .iter()
            .find(|child| child.local_name() == local_name)
    }

    /// Whether the element locates an attached or external data block,
    /// whose bytes live outside the header.
    pub(crate) fn has_outside_block(&self) -> bool {
        self.attribute("location").is_some_and(|location| {
            let location = location.trim();
            location.starts_with("attachment:")
                || location.starts_with("url(")
                || location.starts_with("path(")
        })
    }

    /// Serialize the element. `location` replaces the `location` attribute
    /// of elements whose loaded blocks were given new positions, in document
    /// order.
    pub(crate) fn write(&self, xml: &mut String, locations: &mut impl Iterator<Item = String>) {
        let _ = write!(xml, "<{}", self.name);
        let relocated = if self.block.is_some() {
            locations.next()
        } else {
            None
        };
        for (name, value) in &self.attributes {
            let value = match (&relocated, name.as_str()) {
                (Some(location), "location") => location.as_str(),
                _ => value.as_str(),
            };
            let _ = write!(xml, " {name}=\"{}\"", escape(value));
        }
        // White space between child elements is not significant.
        let text = if self.children.is_empty() || !self.text.trim().is_empty() {
            self.text.as_str()
        } else {
            ""
        };
        if text.is_empty() && self.children.is_empty() {
            xml.push_str("/>");
            return;
        }
        xml.push('>');
        xml.push_str(&escape(text));
        for child in &self.children {
            child.write(xml, locations);
        }
        let _ = write!(xml, "</{}>", self.name);
    }

    /// Visit this element and its descendants in document order.
    pub(crate) fn visit_mut(&mut self, visit: &mut impl FnMut(&mut XisfElement)) {
        visit(self);
        for child in &mut self.children {
            child.visit_mut(visit);
        }
    }
}

/// Parse an XISF header into the tree of its `xisf` root element. Other
/// top-level elements, such as a detached XML signature, are dropped.
pub(crate) fn parse_tree(xml: &[u8]) -> Result<XisfElement, XisfError> {
    let mut reader = Reader::from_reader(xml);
    reader.config_mut().trim_text(false);
    let mut buffer = Vec::new();
    let mut stack = Vec::<XisfElement>::new();
    let mut root = None;
    let mut finish = |element: XisfElement, stack: &mut Vec<XisfElement>| match stack.last_mut() {
        Some(parent) => parent.children.push(element),
        None if root.is_none() && element.local_name() == "xisf" => root = Some(element),
        None => {}
    };

    loop {
        match reader
            .read_event_into(&mut buffer)
            .map_err(|error| XisfError::Malformed(format!("invalid XML header: {error}")))?
        {
            Event::Start(start) => stack.push(element(&reader, &start)?),
            Event::Empty(start) => {
                let element = element(&reader, &start)?;
                finish(element, &mut stack);
            }
            Event::End(_) => {
                if let Some(element) = stack.pop() {
                    finish(element, &mut stack);
                }
            }
            Event::Text(text) => {
                let text = text
                    .decode()
                    .map_err(|error| XisfError::Malformed(format!("invalid XML text: {error}")))?;
                if let Some(element) = stack.last_mut() {
                    element.text.push_str(&text);
                }
            }
            Event::CData(text) => {
                let text = text
                    .decode()
                    .map_err(|error| XisfError::Malformed(format!("invalid XML text: {error}")))?;
                if let Some(element) = stack.last_mut() {
                    element.text.push_str(&text);
                }
            }
            Event::GeneralRef(reference) => {
                let invalid = |error: &dyn std::fmt::Display| {
                    XisfError::Malformed(format!("invalid XML reference: {error}"))
                };
                let text = match reference
                    .resolve_char_ref()
                    .map_err(|error| invalid(&error))?
                {
                    Some(character) => character.to_string(),
                    None => {
                        let name = reference.decode().map_err(|error| invalid(&error))?;
                        quick_xml::escape::resolve_predefined_entity(&name)
                            .ok_or_else(|| {
                                XisfError::Malformed(format!("undefined XML entity &{name};"))
                            })?
                            .to_string()
                    }
                };
                if let Some(element) = stack.last_mut() {
                    element.text.push_str(&text);
                }
            }
            Event::Eof => break,
            Event::DocType(_) => {
                return Err(XisfError::Unsupported("XML document types".into()));
            }
            _ => {}
        }
        buffer.clear();
    }
    let root = root.ok_or_else(|| XisfError::Malformed("missing xisf root element".into()))?;
    if root.attribute("version").map(str::trim) != Some("1.0") {
        return Err(XisfError::Unsupported(format!(
            "XISF version {:?}",
            root.attribute("version")
        )));
    }
    Ok(root)
}

fn element(reader: &Reader<&[u8]>, start: &BytesStart<'_>) -> Result<XisfElement, XisfError> {
    let name = std::str::from_utf8(start.name().as_ref())
        .map_err(|_| XisfError::Malformed("non-UTF-8 XML element name".into()))?
        .to_string();
    let mut attributes = BTreeMap::new();
    for attribute in start.attributes() {
        let attribute = attribute
            .map_err(|error| XisfError::Malformed(format!("invalid XML attribute: {error}")))?;
        let key = std::str::from_utf8(attribute.key.as_ref())
            .map_err(|_| XisfError::Malformed("non-UTF-8 XML attribute name".into()))?;
        let value = attribute
            .decoded_and_normalized_value(XmlVersion::Implicit1_0, reader.decoder())
            .map_err(|error| XisfError::Malformed(format!("invalid XML attribute: {error}")))?;
        attributes.insert(key.to_string(), value.into_owned());
    }
    Ok(XisfElement {
        name,
        attributes,
        ..XisfElement::default()
    })
}
