//! The property lists the host's answers carry, as `=pls` parameters: binary property lists
//! (`bplist00`), which CoreMedia's `FigCreateCFPropertyListFromData` reads as
//! `CFPropertyListCreateWithData` does, and which the guest's VideoToolbox writes its own in.
//!
//! The writer makes one object for each value, no two shared, numbered in the order a walk from
//! the top value meets them, a container before what it holds: the format's readers take any
//! order and need no object shared. Integers take the fewest bytes the format reads back as the
//! same value: 1, 2 or 4 unsigned bytes for a value below 2^32, 8 bytes, signed, for any other.
//! A string all of whose characters are ASCII is written as ASCII, any other as UTF-16.

/// A property list value: the kinds the host's answers hold.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Value {
    /// A boolean.
    Boolean(bool),
    /// An integer.
    Integer(i64),
    /// A string.
    String(String),
    /// An array.
    Array(Vec<Value>),
    /// A dictionary, its keys in the order given, each once.
    Dictionary(Vec<(String, Value)>),
}

impl Value {
    /// A string value.
    pub fn string(value: &str) -> Self {
        Self::String(value.to_owned())
    }

    /// A dictionary of `entries`.
    pub fn dictionary<const N: usize>(entries: [(&str, Value); N]) -> Self {
        Self::Dictionary(
            entries
                .into_iter()
                .map(|(key, value)| (key.to_owned(), value))
                .collect(),
        )
    }
}

/// One object of the list, its children by number.
enum Object<'a> {
    Boolean(bool),
    Integer(i64),
    String(&'a str),
    Array(Vec<u64>),
    Dictionary(Vec<u64>, Vec<u64>),
}

/// Numbers `value` and everything it holds as objects from `objects.len()`, in the order of the
/// module doc, and returns its number.
fn flatten<'a>(value: &'a Value, objects: &mut Vec<Object<'a>>) -> u64 {
    let number = objects.len() as u64;
    match value {
        Value::Boolean(b) => objects.push(Object::Boolean(*b)),
        Value::Integer(i) => objects.push(Object::Integer(*i)),
        Value::String(s) => objects.push(Object::String(s)),
        Value::Array(values) => {
            objects.push(Object::Array(Vec::new()));
            let children = values.iter().map(|v| flatten(v, objects)).collect();
            if let Some(Object::Array(slot)) = objects.get_mut(number as usize) {
                *slot = children;
            }
        }
        Value::Dictionary(entries) => {
            objects.push(Object::Dictionary(Vec::new(), Vec::new()));
            let keys: Vec<u64> = entries
                .iter()
                .map(|(key, _)| {
                    let n = objects.len() as u64;
                    objects.push(Object::String(key));
                    n
                })
                .collect();
            let values = entries.iter().map(|(_, v)| flatten(v, objects)).collect();
            if let Some(Object::Dictionary(k, v)) = objects.get_mut(number as usize) {
                *k = keys;
                *v = values;
            }
        }
    }
    number
}

/// The fewest bytes, of 1, 2, 4 and 8, that hold `value` unsigned.
fn width(value: u64) -> u8 {
    match value {
        0..=0xff => 1,
        0x100..=0xffff => 2,
        0x1_0000..=0xffff_ffff => 4,
        _ => 8,
    }
}

/// Appends `value`'s low `bytes` bytes, big-endian.
fn put(out: &mut Vec<u8>, value: u64, bytes: u8) {
    let all = value.to_be_bytes();
    out.extend_from_slice(all.get(8 - usize::from(bytes)..).unwrap_or(&all));
}

/// Appends an integer object.
fn put_integer(out: &mut Vec<u8>, value: i64) {
    if value < 0 {
        out.push(0x13);
        out.extend_from_slice(&value.to_be_bytes());
        return;
    }
    let value = value.unsigned_abs();
    let bytes = width(value);
    out.push(0x10 | bytes.trailing_zeros() as u8);
    put(out, value, bytes);
}

/// Appends a marker of kind `kind` (its high nibble) for `count` items: in the marker where under
/// 15, else after it as an integer object.
fn put_count(out: &mut Vec<u8>, kind: u8, count: usize) {
    if count < 15 {
        out.push(kind | count as u8);
    } else {
        out.push(kind | 0xf);
        put_integer(out, count as i64);
    }
}

/// `value` as a binary property list.
pub fn encode(value: &Value) -> Vec<u8> {
    let mut objects = Vec::new();
    flatten(value, &mut objects);
    let count = objects.len() as u64;
    let reference = width(count.saturating_sub(1));
    let mut out = b"bplist00".to_vec();
    let mut offsets = Vec::with_capacity(objects.len());
    for object in &objects {
        offsets.push(out.len() as u64);
        match object {
            Object::Boolean(b) => out.push(if *b { 0x09 } else { 0x08 }),
            Object::Integer(i) => put_integer(&mut out, *i),
            Object::String(s) if s.is_ascii() => {
                put_count(&mut out, 0x50, s.len());
                out.extend_from_slice(s.as_bytes());
            }
            Object::String(s) => {
                let units: Vec<u16> = s.encode_utf16().collect();
                put_count(&mut out, 0x60, units.len());
                for unit in units {
                    out.extend_from_slice(&unit.to_be_bytes());
                }
            }
            Object::Array(children) => {
                put_count(&mut out, 0xa0, children.len());
                for &child in children {
                    put(&mut out, child, reference);
                }
            }
            Object::Dictionary(keys, values) => {
                put_count(&mut out, 0xd0, keys.len());
                for &child in keys.iter().chain(values) {
                    put(&mut out, child, reference);
                }
            }
        }
    }
    let table = out.len() as u64;
    let offset = width(table);
    for &at in &offsets {
        put(&mut out, at, offset);
    }
    // The trailer: 6 bytes unused, the offsets' and references' widths, the object count, the
    // top object's number and the offset table's offset.
    out.extend_from_slice(&[0; 6]);
    out.push(offset);
    out.push(reference);
    out.extend_from_slice(&count.to_be_bytes());
    out.extend_from_slice(&0u64.to_be_bytes());
    out.extend_from_slice(&table.to_be_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dictionary of every kind, as the writer lays it out: read back by Python's `plistlib`
    /// and checked by `plutil -lint` on macOS 27.0.1 (2026-11-16), which read it as written.
    #[test]
    fn a_dictionary_of_every_kind_is_laid_out_as_read_back() {
        let value = Value::dictionary([
            ("a", Value::Boolean(true)),
            ("n", Value::Integer(400)),
            (
                "l",
                Value::Array(vec![Value::Integer(-1), Value::string("x")]),
            ),
        ]);
        let bytes = encode(&value);
        let mut expected = b"bplist00".to_vec();
        // 0: the dictionary, keys 1 to 3, values 4, 5, 6.
        expected.extend_from_slice(&[0xd3, 1, 2, 3, 4, 5, 6]);
        expected.extend_from_slice(b"\x51a\x51n\x51l");
        expected.extend_from_slice(&[0x09, 0x11, 0x01, 0x90]);
        // 6: the array of 7 and 8.
        expected.extend_from_slice(&[0xa2, 7, 8]);
        expected.push(0x13);
        expected.extend_from_slice(&(-1i64).to_be_bytes());
        expected.extend_from_slice(b"\x51x");
        let table = expected.len() as u8;
        expected.extend_from_slice(&[8, 15, 17, 19, 21, 22, 25, 28, 37]);
        expected.extend_from_slice(&[0, 0, 0, 0, 0, 0, 1, 1]);
        expected.extend_from_slice(&9u64.to_be_bytes());
        expected.extend_from_slice(&0u64.to_be_bytes());
        expected.extend_from_slice(&u64::from(table).to_be_bytes());
        assert_eq!(bytes, expected);
    }

    #[test]
    fn integers_take_the_fewest_bytes_that_read_back_the_same() {
        let one = |i| {
            let mut out = Vec::new();
            put_integer(&mut out, i);
            out
        };
        assert_eq!(one(0), [0x10, 0]);
        assert_eq!(one(0xff), [0x10, 0xff]);
        assert_eq!(one(0x100), [0x11, 1, 0]);
        assert_eq!(one(0x1_0000), [0x12, 0, 1, 0, 0]);
        assert_eq!(one(0x1_0000_0000), [0x13, 0, 0, 0, 1, 0, 0, 0, 0]);
        assert_eq!(
            one(-2),
            [0x13, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfe]
        );
    }

    #[test]
    fn long_strings_and_arrays_count_after_their_marker() {
        let long = "abcdefghijklmnop";
        let bytes = encode(&Value::string(long));
        assert_eq!(&bytes[8..11], &[0x5f, 0x10, 16]);
        assert_eq!(&bytes[11..27], long.as_bytes());
        let wide = encode(&Value::string("é"));
        assert_eq!(&wide[8..11], &[0x61, 0x00, 0xe9]);
    }

    #[test]
    fn many_objects_take_wider_references() {
        let values = (0..300).map(Value::Integer).collect();
        let bytes = encode(&Value::Array(values));
        // The array's marker, its count as a 2-byte integer, then 2-byte references.
        assert_eq!(&bytes[8..14], &[0xaf, 0x11, 0x01, 0x2c, 0x00, 0x01]);
        let trailer = &bytes[bytes.len() - 32..];
        assert_eq!(trailer[7], 2);
        assert_eq!(u64::from_be_bytes(trailer[8..16].try_into().unwrap()), 301);
    }
}
