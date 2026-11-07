//! The XISF header as a tree of elements, kept whole so that a later write
//! can carry every element this crate does not interpret.

use crate::XisfError;
use quick_xml::escape::escape;
use quick_xml::events::{BytesStart, Event};
use quick_xml::{Reader, XmlVersion};
use std::collections::BTreeMap;
use std::fmt::Write as _;

/// Bounds the trees that metadata cloning, traversal and serialization
/// recurse over. Includes the root element and empty leaf elements.
pub(crate) const MAX_ELEMENT_DEPTH: usize = 128;

fn check_depth(depth: usize) -> Result<(), XisfError> {
    if depth > MAX_ELEMENT_DEPTH {
        return Err(XisfError::Malformed(format!(
            "XML element nesting exceeds {MAX_ELEMENT_DEPTH} levels"
        )));
    }
    Ok(())
}

/// Check caller-built metadata before any recursive clone or traversal.
/// `parent_depth` includes the root and any Image or Metadata wrapper.
pub(crate) fn validate_depth(
    elements: &[XisfElement],
    parent_depth: usize,
) -> Result<(), XisfError> {
    if elements.is_empty() {
        return Ok(());
    }
    let mut stack = vec![(elements.iter(), parent_depth)];
    while let Some((siblings, parent_depth)) = stack.last_mut() {
        if let Some(element) = siblings.next() {
            let depth = *parent_depth + 1;
            check_depth(depth)?;
            if !element.children.is_empty() {
                stack.push((element.children.iter(), depth));
            }
        } else {
            stack.pop();
        }
    }
    Ok(())
}

/// An XML element of an XISF header, with everything inside it.
///
/// Character data is kept as one string per element. XISF elements hold
/// either character data or child elements, and white space between child
/// elements is not significant, so nothing that matters is lost.
///
/// Readers and writers accept at most 128 nested elements, including the
/// `xisf` root and any enclosing `Image` or `Metadata` element. A writer
/// refuses deeper caller-built metadata before cloning or serializing it.
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
            let _ = write!(xml, " {name}=\"{}\"", escape_attribute(value));
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

/// Escape an attribute value. Besides the markup characters, tabs and line
/// breaks become character references, since a reader's attribute value
/// normalization would otherwise turn them into spaces.
pub(crate) fn escape_attribute(value: &str) -> String {
    escape(value)
        .replace('\t', "&#9;")
        .replace('\n', "&#10;")
        .replace('\r', "&#13;")
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
            Event::Start(start) => {
                check_depth(stack.len() + 1)?;
                stack.push(element(&start)?);
            }
            Event::Empty(start) => {
                check_depth(stack.len() + 1)?;
                let element = element(&start)?;
                finish(element, &mut stack);
            }
            Event::End(_) => {
                if let Some(element) = stack.pop() {
                    finish(element, &mut stack);
                }
            }
            // The reader has already checked the text is UTF-8; like the
            // previous decode(), into_inner() leaves line ends as written.
            Event::Text(text) => {
                let text = text.into_inner();
                if let Some(element) = stack.last_mut() {
                    element.text.push_str(&text);
                }
            }
            Event::CData(text) => {
                let text = text.into_inner();
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
                        let name = reference.into_inner();
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
            Event::Eof => {
                if !stack.is_empty() {
                    return Err(XisfError::Malformed(
                        "unclosed XML element at end of header".into(),
                    ));
                }
                break;
            }
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

// quick-xml checks names are UTF-8 while reading.
fn element(start: &BytesStart<'_>) -> Result<XisfElement, XisfError> {
    let name = start.name().as_ref().to_string();
    let mut attributes = BTreeMap::new();
    for attribute in start.attributes() {
        let attribute = attribute
            .map_err(|error| XisfError::Malformed(format!("invalid XML attribute: {error}")))?;
        let key: &str = attribute.key.as_ref();
        let value = attribute
            .normalized_value(XmlVersion::Implicit1_0)
            .map_err(|error| XisfError::Malformed(format!("invalid XML attribute: {error}")))?;
        attributes.insert(key.to_string(), value.into_owned());
    }
    Ok(XisfElement {
        name,
        attributes,
        ..XisfElement::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nested(depth: usize, empty_leaf: bool) -> String {
        let parents = depth - if empty_leaf { 2 } else { 1 };
        format!(
            "<xisf version=\"1.0\">{}{}{}</xisf>",
            "<Extension>".repeat(parents),
            if empty_leaf { "<Extension/>" } else { "" },
            "</Extension>".repeat(parents),
        )
    }

    #[test]
    fn nesting_boundary_includes_empty_leaf_elements() {
        for empty_leaf in [false, true] {
            let root = parse_tree(nested(MAX_ELEMENT_DEPTH, empty_leaf).as_bytes()).unwrap();
            validate_depth(&root.children, 1).unwrap();
            let mut depth = 1;
            let mut element = &root;
            while let Some(child) = element.children.first() {
                depth += 1;
                element = child;
            }
            assert_eq!(depth, MAX_ELEMENT_DEPTH);
            assert!(matches!(
                parse_tree(nested(MAX_ELEMENT_DEPTH + 1, empty_leaf).as_bytes()),
                Err(XisfError::Malformed(_))
            ));
            assert!(matches!(
                validate_depth(&root.children, 2),
                Err(XisfError::Malformed(_))
            ));
        }
    }

    #[test]
    fn many_siblings_are_not_deep_nesting() {
        let xml = format!(
            "<xisf version=\"1.0\">{}</xisf>",
            "<Extension/>".repeat(10000)
        );
        let root = parse_tree(xml.as_bytes()).unwrap();
        assert_eq!(root.children.len(), 10000);
        validate_depth(&root.children, 1).unwrap();
    }

    #[test]
    fn unfinished_elements_are_refused_even_after_a_complete_root() {
        for xml in [
            "<xisf version=\"1.0\"><Extension>",
            "<xisf version=\"1.0\"/><Signature><SignedInfo>",
            "<Extension><xisf version=\"1.0\"/>",
        ] {
            assert!(matches!(
                parse_tree(xml.as_bytes()),
                Err(XisfError::Malformed(_))
            ));
        }
        parse_tree(b"<xisf version=\"1.0\"/><Signature><SignedInfo/></Signature>").unwrap();
    }

    #[test]
    fn supported_nested_tree_survives_clone_and_serialization() {
        let root = parse_tree(nested(MAX_ELEMENT_DEPTH, true).as_bytes()).unwrap();
        let copy = root.clone();
        let mut xml = String::new();
        copy.write(&mut xml, &mut std::iter::empty());
        assert_eq!(parse_tree(xml.as_bytes()).unwrap(), root);
    }
}
