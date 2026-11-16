//! Binary property lists (`bplist00`), the form CoreMedia writes and reads the `=pls` parameters
//! and a sample buffer's leaves in ([`encode`], [`decode`]).
//!
//! The writer makes one object for each value, no two shared, numbered in the order a walk from
//! the top value meets them, a container before what it holds: the format's readers take any
//! order and need no object shared. Integers take the fewest bytes the format reads back as the
//! same value: 1, 2 or 4 unsigned bytes for a value below 2^32, 8 bytes, signed, for any other.
//! A string all of whose characters are ASCII is written as ASCII, any other as UTF-16.
//!
//! The reader takes the guest's bytes, and holds what they decode to within bounds of ours: the
//! objects one list may have and the values and bytes it may decode to ([`MAX_VALUES`],
//! [`MAX_BYTES`]), and how deep its containers may nest ([`MAX_DEPTH`]). A list names its objects
//! by number, so a few hundred bytes that name one array twice at each of 64 levels would
//! otherwise decode to 2^64 values; each value is spent from the budget as it is made, so none
//! past the bounds is ever made.

use crate::paravirt::WireError;

/// A property list value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// A boolean.
    Boolean(bool),
    /// An integer.
    Integer(i64),
    /// A real number.
    Real(f64),
    /// A date: seconds from 2001-01-01 00:00:00 UTC.
    Date(f64),
    /// Bytes.
    Data(Vec<u8>),
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

    /// The value of a dictionary's entry `key`; `None` for any other value.
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Self::Dictionary(entries) => entries
                .iter()
                .find(|(found, _)| found == key)
                .map(|(_, value)| value),
            _ => None,
        }
    }

    /// An integer's value; `None` for any other value.
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            Self::Integer(i) => Some(*i),
            _ => None,
        }
    }

    /// A boolean's value; `None` for any other value.
    pub fn as_boolean(&self) -> Option<bool> {
        match self {
            Self::Boolean(b) => Some(*b),
            _ => None,
        }
    }

    /// A string's value; `None` for any other value.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }
}

/// One object of the list, its children by number.
enum Object<'a> {
    Boolean(bool),
    Integer(i64),
    Real(f64),
    Date(f64),
    Data(&'a [u8]),
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
        Value::Real(r) => objects.push(Object::Real(*r)),
        Value::Date(d) => objects.push(Object::Date(*d)),
        Value::Data(d) => objects.push(Object::Data(d)),
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
            Object::Real(r) => {
                out.push(0x23);
                out.extend_from_slice(&r.to_be_bytes());
            }
            Object::Date(d) => {
                out.push(0x33);
                out.extend_from_slice(&d.to_be_bytes());
            }
            Object::Data(d) => {
                put_count(&mut out, 0x40, d.len());
                out.extend_from_slice(d);
            }
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

/// The most objects a list the host reads may have, and the most values it may decode to.
pub const MAX_VALUES: u64 = 1 << 16;
/// The most bytes of strings and data a list the host reads may decode to.
pub const MAX_BYTES: u64 = 16 << 20;
/// The deepest a list's arrays and dictionaries may nest, the top one level one.
pub const MAX_DEPTH: u32 = 32;

/// The format's trailer: 6 bytes unused, the offsets' and references' widths, the object count,
/// the top object's number and the offset table's offset.
const TRAILER_SIZE: usize = 32;

/// What lists may still decode to ([`decode_within`]): one list's bounds, or what several the
/// guest sent together share.
pub(crate) struct Budget {
    values: u64,
    bytes: u64,
}

impl Budget {
    /// The bounds of one list ([`MAX_VALUES`], [`MAX_BYTES`]).
    pub(crate) fn new() -> Self {
        Self {
            values: MAX_VALUES,
            bytes: MAX_BYTES,
        }
    }

    /// Spends `values` values and `bytes` bytes, or refuses past the bounds.
    pub(crate) fn spend(&mut self, values: u64, bytes: u64) -> Result<(), WireError> {
        self.values = self.values.checked_sub(values).ok_or(WireError::Malformed(
            "a property list past the host's bound",
        ))?;
        self.bytes = self.bytes.checked_sub(bytes).ok_or(WireError::Malformed(
            "a property list past the host's bound",
        ))?;
        Ok(())
    }
}

/// A list's objects, as its trailer lays them out.
struct Objects<'a> {
    bytes: &'a [u8],
    offset_width: u8,
    reference_width: u8,
    count: u64,
    table: usize,
}

/// The big-endian unsigned integer of `width` bytes at `at`.
fn uint(bytes: &[u8], at: usize, width: u8) -> Option<u64> {
    let field = bytes.get(at..at.checked_add(usize::from(width))?)?;
    Some(field.iter().fold(0u64, |v, &b| (v << 8) | u64::from(b)))
}

impl<'a> Objects<'a> {
    /// Where object `number` starts: past the magic and before the offset table.
    fn start(&self, number: u64) -> Result<usize, WireError> {
        let malformed = WireError::Malformed("a property list object");
        if number >= self.count {
            return Err(malformed);
        }
        let entry = usize::try_from(number)
            .ok()
            .and_then(|n| n.checked_mul(usize::from(self.offset_width)))
            .and_then(|n| n.checked_add(self.table))
            .ok_or(WireError::Malformed("a property list object"))?;
        let start = uint(self.bytes, entry, self.offset_width)
            .and_then(|at| usize::try_from(at).ok())
            .ok_or(WireError::Malformed("a property list object"))?;
        if start < 8 || start >= self.table {
            return Err(malformed);
        }
        Ok(start)
    }

    /// The count of the object whose marker is at `at` and whose low nibble is `nibble`, and
    /// where its contents start: in the marker under 15, else an integer object after it.
    fn count(&self, at: usize, nibble: u8) -> Result<(u64, usize), WireError> {
        if nibble < 15 {
            return Ok((u64::from(nibble), at + 1));
        }
        let malformed = WireError::Malformed("a property list count");
        let marker = *self.bytes.get(at + 1).ok_or(malformed.clone())?;
        if marker & 0xf0 != 0x10 || marker & 0x0f > 3 {
            return Err(malformed);
        }
        let width = 1u8 << (marker & 0x0f);
        let count = uint(self.bytes, at + 2, width).ok_or(malformed)?;
        Ok((count, at + 2 + usize::from(width)))
    }

    /// `length` bytes from `at`, which must end before the offset table.
    fn span(&self, at: usize, length: u64) -> Result<&'a [u8], WireError> {
        let end = usize::try_from(length)
            .ok()
            .and_then(|n| at.checked_add(n))
            .filter(|&end| end <= self.table)
            .ok_or(WireError::Malformed(
                "a property list object past its table",
            ))?;
        self.bytes.get(at..end).ok_or(WireError::Malformed(
            "a property list object past its table",
        ))
    }

    /// The references of a container of `count` children from `at`.
    fn references(&self, at: usize, count: u64) -> Result<Vec<u64>, WireError> {
        let width = u64::from(self.reference_width);
        let span = self.span(at, count.saturating_mul(width))?;
        Ok(span
            .chunks(usize::from(self.reference_width))
            .map(|r| r.iter().fold(0u64, |v, &b| (v << 8) | u64::from(b)))
            .collect())
    }

    /// Object `number`, at nesting `depth`, spent from `budget`.
    fn value(&self, number: u64, depth: u32, budget: &mut Budget) -> Result<Value, WireError> {
        let at = self.start(number)?;
        let marker = *self
            .bytes
            .get(at)
            .ok_or(WireError::Malformed("a property list object"))?;
        let nibble = marker & 0x0f;
        budget.spend(1, 0)?;
        match marker >> 4 {
            0x0 => match marker {
                0x08 => Ok(Value::Boolean(false)),
                0x09 => Ok(Value::Boolean(true)),
                _ => Err(WireError::Malformed("a property list marker")),
            },
            0x1 if nibble <= 3 => {
                let width = 1u8 << nibble;
                let raw = uint(self.bytes, at + 1, width)
                    .filter(|_| at + 1 + usize::from(width) <= self.table)
                    .ok_or(WireError::Malformed("a property list integer"))?;
                // 1, 2 and 4 bytes are unsigned, 8 signed.
                Ok(Value::Integer(i64::from_be_bytes(raw.to_be_bytes())))
            }
            0x2 if nibble == 2 || nibble == 3 => {
                let span = self.span(at + 1, 1 << nibble)?;
                Ok(Value::Real(match <[u8; 4]>::try_from(span) {
                    Ok(four) => f64::from(f32::from_be_bytes(four)),
                    Err(_) => f64::from_be_bytes(
                        span.try_into()
                            .map_err(|_| WireError::Malformed("a property list real"))?,
                    ),
                }))
            }
            0x3 if marker == 0x33 => {
                let span = self.span(at + 1, 8)?;
                Ok(Value::Date(f64::from_be_bytes(span.try_into().map_err(
                    |_| WireError::Malformed("a property list date"),
                )?)))
            }
            0x4 => {
                let (length, from) = self.count(at, nibble)?;
                budget.spend(0, length)?;
                Ok(Value::Data(self.span(from, length)?.to_vec()))
            }
            0x5 => {
                let (length, from) = self.count(at, nibble)?;
                budget.spend(0, length)?;
                let span = self.span(from, length)?;
                if !span.is_ascii() {
                    return Err(WireError::Malformed("a property list ASCII string"));
                }
                Ok(Value::String(String::from_utf8_lossy(span).into_owned()))
            }
            0x6 => {
                let (units, from) = self.count(at, nibble)?;
                budget.spend(0, units.saturating_mul(2))?;
                let span = self.span(from, units.saturating_mul(2))?;
                let units: Vec<u16> = span
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|u| u16::from_be_bytes(*u))
                    .collect();
                String::from_utf16(&units)
                    .map(Value::String)
                    .map_err(|_| WireError::Malformed("a property list UTF-16 string"))
            }
            0xa => {
                let depth = deeper(depth)?;
                let (count, from) = self.count(at, nibble)?;
                let children = self.references(from, count)?;
                children
                    .into_iter()
                    .map(|child| self.value(child, depth, budget))
                    .collect::<Result<_, _>>()
                    .map(Value::Array)
            }
            0xd => {
                let depth = deeper(depth)?;
                let (count, from) = self.count(at, nibble)?;
                let references = self.references(from, count.saturating_mul(2))?;
                let (keys, values) = references.split_at(references.len() / 2);
                let mut entries: Vec<(String, Value)> = Vec::with_capacity(keys.len());
                for (&key, &value) in keys.iter().zip(values) {
                    let Value::String(key) = self.value(key, depth, budget)? else {
                        return Err(WireError::Malformed("a property list key"));
                    };
                    let value = self.value(value, depth, budget)?;
                    // A repeated key keeps its first value, as CoreFoundation's reader does.
                    if !entries.iter().any(|(found, _)| *found == key) {
                        entries.push((key, value));
                    }
                }
                Ok(Value::Dictionary(entries))
            }
            _ => Err(WireError::Malformed("a property list marker")),
        }
    }
}

/// `depth` one level deeper, or malformed past [`MAX_DEPTH`].
fn deeper(depth: u32) -> Result<u32, WireError> {
    Some(depth + 1)
        .filter(|&d| d <= MAX_DEPTH)
        .ok_or(WireError::Malformed("a property list nested too deep"))
}

/// The value of the binary property list `bytes`, within the module's bounds. NULL (no bytes) is
/// the caller's to tell apart.
pub fn decode(bytes: &[u8]) -> Result<Value, WireError> {
    decode_within(bytes, &mut Budget::new())
}

/// [`decode`], spending what the list decodes to from `budget`.
pub(crate) fn decode_within(bytes: &[u8], budget: &mut Budget) -> Result<Value, WireError> {
    let malformed = || WireError::Malformed("not a binary property list");
    if bytes.len() < 8 + TRAILER_SIZE || !bytes.starts_with(b"bplist00") {
        return Err(malformed());
    }
    let trailer = bytes
        .get(bytes.len() - TRAILER_SIZE..)
        .ok_or_else(malformed)?;
    let field = |at| uint(trailer, at, 8).ok_or_else(malformed);
    let offset_width = *trailer.get(6).ok_or_else(malformed)?;
    let reference_width = *trailer.get(7).ok_or_else(malformed)?;
    let (count, top, table) = (field(8)?, field(16)?, field(24)?);
    let table = usize::try_from(table).map_err(|_| malformed())?;
    let table_end = usize::try_from(count)
        .ok()
        .and_then(|n| n.checked_mul(usize::from(offset_width)))
        .and_then(|n| n.checked_add(table));
    if !(1..=8).contains(&offset_width)
        || !(1..=8).contains(&reference_width)
        || count == 0
        || count > MAX_VALUES
        || table < 8
        || table_end.is_none_or(|end| end > bytes.len() - TRAILER_SIZE)
    {
        return Err(malformed());
    }
    let objects = Objects {
        bytes,
        offset_width,
        reference_width,
        count,
        table,
    };
    objects.value(top, 0, budget)
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

    #[test]
    fn what_the_writer_writes_reads_back() -> Result<(), WireError> {
        let value = Value::dictionary([
            ("a", Value::Boolean(true)),
            ("n", Value::Integer(400)),
            ("m", Value::Integer(-7)),
            ("r", Value::Real(0.5)),
            ("t", Value::Date(1.25)),
            ("d", Value::Data(vec![0, 1, 2])),
            ("w", Value::string("é")),
            ("l", Value::Array((0..20).map(Value::Integer).collect())),
        ]);
        assert_eq!(decode(&encode(&value))?, value);
        Ok(())
    }

    /// A list written by hand whose dictionary names its one string twice, as a key and a value,
    /// with a 4-byte real, a repeated key and an 8-byte top object number.
    #[test]
    fn a_hand_written_list_with_shared_objects_reads() -> Result<(), WireError> {
        let mut bytes = b"bplist00".to_vec();
        // 0: {k: k, k: 2.0}, 1: "k", 2: 2.0 as a 4-byte real.
        bytes.extend_from_slice(&[0xd2, 1, 1, 1, 2]);
        bytes.extend_from_slice(&[0x51, b'k']);
        bytes.push(0x22);
        bytes.extend_from_slice(&2.0f32.to_be_bytes());
        let table = bytes.len() as u64;
        bytes.extend_from_slice(&[8, 13, 15]);
        bytes.extend_from_slice(&[0, 0, 0, 0, 0, 0, 1, 1]);
        bytes.extend_from_slice(&3u64.to_be_bytes());
        bytes.extend_from_slice(&0u64.to_be_bytes());
        bytes.extend_from_slice(&table.to_be_bytes());
        assert_eq!(
            decode(&bytes)?,
            Value::Dictionary(vec![("k".to_owned(), Value::string("k"))])
        );
        Ok(())
    }

    /// An array that names one other array twice at each of 40 levels: 2^40 values, refused at
    /// the depth bound before any is made past it.
    #[test]
    fn a_list_past_the_bounds_is_refused() {
        let levels = 40u8;
        let mut bytes = b"bplist00".to_vec();
        let mut offsets = Vec::new();
        for level in 0..levels {
            offsets.push(bytes.len() as u8);
            bytes.extend_from_slice(&[0xa2, level + 1, level + 1]);
        }
        offsets.push(bytes.len() as u8);
        bytes.push(0x09);
        let table = bytes.len() as u64;
        bytes.extend_from_slice(&offsets);
        bytes.extend_from_slice(&[0, 0, 0, 0, 0, 0, 1, 1]);
        bytes.extend_from_slice(&u64::from(levels + 1).to_be_bytes());
        bytes.extend_from_slice(&0u64.to_be_bytes());
        bytes.extend_from_slice(&table.to_be_bytes());
        assert!(decode(&bytes).is_err());
    }

    #[test]
    fn bytes_that_are_no_binary_list_are_malformed() {
        assert!(decode(b"").is_err());
        assert!(decode(b"<?xml version=\"1.0\"?>").is_err());
        let mut bytes = encode(&Value::Integer(1));
        // The top object's number past the count.
        let at = bytes.len() - 16;
        bytes[at + 7] = 9;
        assert!(decode(&bytes).is_err());
        // A container that names itself.
        let mut looped = b"bplist00".to_vec();
        looped.extend_from_slice(&[0xa1, 0]);
        looped.push(8);
        looped.extend_from_slice(&[0, 0, 0, 0, 0, 0, 1, 1]);
        looped.extend_from_slice(&1u64.to_be_bytes());
        looped.extend_from_slice(&0u64.to_be_bytes());
        looped.extend_from_slice(&10u64.to_be_bytes());
        assert!(decode(&looped).is_err());
    }
}
