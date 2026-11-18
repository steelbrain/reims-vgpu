//! The replay stream's bytes, format version 2 (metal2vulkan's `replay-v2-spec.md`): its header,
//! its three record kinds (a name, a blob, a call) and the values a call carries, written by
//! [`Stream`] and read back by [`read`].
//!
//! - **The file:** [`HEADER`] (`M2VRSTRM`, the version and the flags, [`FLAGS`], each as a
//!   little-endian `u32`), then records: a kind byte, the payload's length as a varint, the
//!   payload. Varints are unsigned LEB128; a signed value is zigzag-encoded first.
//! - **Flags** say what a stream promises beyond the version's records. Bit 0,
//!   [`FLAG_ANNOUNCED_BYTES_RECORDED`]: at each `didModifyRange:`, the stream's last
//!   `m2v.write` of each 4 KiB chunk of the range holds the guest's bytes, so a player pushes the
//!   recorded bytes there; one reading a stream without it pushes its own read-back. A reader
//!   checks the magic and the version alone ([`read`]), so a stream with the bit reads as before.
//! - **Names** are interned: the n-th name record defines name id n, from 1, before the first
//!   record that uses it. A selector and a map key are name ids.
//! - **Blobs** are named by the SHA-256 of their bytes, each written once per stream before its
//!   first use ([`Stream::blob`]); a blob whose bytes live outside the stream (a library in the
//!   blob directory) is named by its hash alone, with no blob record.
//!
//! A stream split into segments starts each with the header ([`Stream::header`]); names and
//! blobs carry across segments, so the writer keeps its tables across them. The recorder starts
//! each segment after the first with the identity record of each device then live
//! (`m2v.deviceIdentity`, on the device, answering its `name` and `hasUnifiedMemory`), which
//! also follows the record that made each device: the header itself stays the magic, the
//! version and [`FLAGS`].
//!
//! A no-copy buffer's bytes arrive as writes, but where the device keeps two copies of a
//! `Managed` buffer (`hasUnifiedMemory` NO), a no-copy `Managed` one's record carries those its
//! create copied into the device's copy, from its first byte: the descriptor's `contents` of
//! `newBufferWithDescriptor:`, the bytes before its `zeroFrom` (nil where that is 0), and the
//! first argument of `newBufferWithBytesNoCopy:length:options:deallocator:`, all its bytes, each
//! a blob where long, as a copied buffer's `contents` are.
//!
//! The commands the CPU encodes into an indirect command buffer are call records on the buffer,
//! `indirectRenderCommandAtIndex:` or `indirectComputeCommandAtIndex:`, whose arguments are
//! `[index, commands]` (the spec's section 8): the command's index, and a list of the commands
//! sent to it, in order, each a map of one pair whose name is the command's selector
//! (`MTLIndirectRenderCommand`'s or `MTLIndirectComputeCommand`'s) and whose value is the list of
//! its arguments, as an encoder's command of that selector carries them: objects by their ids
//! (nil for none), sizes and regions as maps, a command of no arguments an empty list.
//!
//! A texture of the recording platform's own (the window's cursor glyph), which takes no
//! resource ID and no range of Metal's space, is a `newTextureWithDescriptor:` on the device as
//! any other, naming a descriptor whose `m2v.textureDescriptor` record's map ends with the key
//! `outside`, `true` ([`super::OUTSIDE`]): a player places the texture outside the guest's
//! texture indices, so the guest's textures after it keep their indices. No other descriptor
//! carries the key.

use std::collections::{HashMap, HashSet};

use sha2::{Digest as _, Sha256};

/// The format's version.
pub const VERSION: u32 = 2;

/// Header flag bit 0: at each `didModifyRange:` record of a buffer, for each 4 KiB chunk of the
/// range (of the target its writes name: the buffer, or the block it lies in), the last
/// `m2v.write` before it that carries the chunk holds the bytes the CPU's copy held at the call,
/// bytes the GPU wrote and the stream took without a write included. So a player pushes its
/// recorded bytes at an announce, never its own read-back of a buffer the GPU wrote.
pub const FLAG_ANNOUNCED_BYTES_RECORDED: u32 = 1;

/// The flags every stream the recorder writes carries in its header.
pub const FLAGS: u32 = FLAG_ANNOUNCED_BYTES_RECORDED;

/// The file's first 16 bytes: the magic, the version and the flags.
pub const HEADER: [u8; 16] = {
    let mut h = [0u8; 16];
    let magic = *b"M2VRSTRM";
    let mut i = 0;
    while i < 8 {
        h[i] = magic[i];
        i += 1;
    }
    let v = VERSION.to_le_bytes();
    h[8] = v[0];
    h[9] = v[1];
    h[10] = v[2];
    h[11] = v[3];
    let f = FLAGS.to_le_bytes();
    h[12] = f[0];
    h[13] = f[1];
    h[14] = f[2];
    h[15] = f[3];
    h
};

/// Record kind 1: a name.
pub const KIND_NAME: u8 = 1;
/// Record kind 2: a blob (its hash, then its bytes).
pub const KIND_BLOB: u8 = 2;
/// Record kind 3: a call.
pub const KIND_CALL: u8 = 3;

/// A SHA-256 hash.
pub type Hash = [u8; 32];

/// The SHA-256 of `data`.
pub fn sha256(data: &[u8]) -> Hash {
    Sha256::digest(data).into()
}

/// One value of a call's arguments or answers: the format's tags 0 to 13.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Tag 0: nil, or an argument not recorded.
    Nil,
    /// Tags 1 and 2.
    Bool(bool),
    /// Tag 3: an unsigned integer, a raw enum or option set, a resource ID.
    Uint(u64),
    /// Tag 4: a signed integer.
    Sint(i64),
    /// Tag 5: a float.
    F32(f32),
    /// Tag 6: a double.
    F64(f64),
    /// Tag 7: an object, by its stream id; 0, which no object is, a property set to nil.
    Object(u64),
    /// Tag 8: a GPU address, which the player relocates.
    Address(u64),
    /// Tag 9: a string.
    Str(String),
    /// Tag 10: inline bytes.
    Bytes(Vec<u8>),
    /// Tag 11: bytes by content, the blob's hash.
    Blob(Hash),
    /// Tag 12: a struct or C array, in order.
    List(Vec<Value>),
    /// Tag 13: a descriptor or struct by property name.
    Map(Vec<(String, Value)>),
    /// Host memory an argument names: its address and length. The recorder reads or drops it
    /// before a call is written; written alone, it is nil.
    Host(usize, u64),
    /// An argument's `Debug` text, its objects already their stream ids: the recorder's writer
    /// reads it into its value ([`super::debug_value`]) before the call is written; written
    /// alone, it is nil.
    Debug(String),
}

impl Value {
    /// Calls `f` with each object this value names, nested ones included.
    pub fn each_object(&self, f: &mut impl FnMut(u64)) {
        match self {
            Value::Object(id) => f(*id),
            Value::List(items) => items.iter().for_each(|v| v.each_object(f)),
            Value::Map(pairs) => pairs.iter().for_each(|(_, v)| v.each_object(f)),
            _ => {}
        }
    }
}

/// One call record, with its selector and answer names as text.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Call {
    /// Metal's selector, a C function's name, or an `m2v.` name.
    pub selector: String,
    /// The receiver's id, 0 for none.
    pub receiver: u64,
    /// The id of the object the call made, 0 for none.
    pub result: u64,
    /// One value per argument, in order.
    pub args: Vec<Value>,
    /// What the call answered that the program could see, by name.
    pub answers: Vec<(String, Value)>,
}

/// Appends `v` as an unsigned LEB128 varint.
pub fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// `v` zigzag-encoded.
pub fn zigzag(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

/// The signed value of a zigzag-encoded `v`.
pub fn unzigzag(v: u64) -> i64 {
    ((v >> 1) as i64) ^ -((v & 1) as i64)
}

/// The writer's side of a stream: the bytes written since the last [`Stream::take`], the names
/// and blobs defined so far and the call records written, which carry across segments.
#[derive(Debug, Default)]
pub struct Stream {
    out: Vec<u8>,
    names: HashMap<String, u64>,
    blobs: HashSet<Hash>,
    scratch: Vec<u8>,
    calls: u64,
}

impl Stream {
    /// A stream with nothing written yet: the caller writes [`Stream::header`] first.
    pub fn new() -> Stream {
        Stream::default()
    }

    /// Writes the header, which starts the stream and each of its segments.
    pub fn header(&mut self) {
        self.out.extend_from_slice(&HEADER);
    }

    /// The bytes written since the last take.
    pub fn take(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.out)
    }

    /// How many bytes wait to be taken.
    pub fn pending(&self) -> usize {
        self.out.len()
    }

    /// How many call records it wrote, across its segments: the next call's position in the
    /// stream, from 0, as a player counts the calls it played.
    pub fn calls(&self) -> u64 {
        self.calls
    }

    fn record(out: &mut Vec<u8>, kind: u8, payload: &[u8]) {
        out.push(kind);
        put_varint(out, payload.len() as u64);
        out.extend_from_slice(payload);
    }

    /// The id of `name`, written as a name record the first time.
    pub fn name(&mut self, name: &str) -> u64 {
        if let Some(&id) = self.names.get(name) {
            return id;
        }
        let id = self.names.len() as u64 + 1;
        self.names.insert(name.to_owned(), id);
        Self::record(&mut self.out, KIND_NAME, name.as_bytes());
        id
    }

    /// The hash of `data`, written as a blob record the first time the stream sees it.
    pub fn blob(&mut self, data: &[u8]) -> Hash {
        let hash = sha256(data);
        self.blob_hashed(hash, data);
        hash
    }

    /// Writes `data`, whose hash is `hash`, as a blob record unless the stream holds it; whether
    /// it was written.
    pub fn blob_hashed(&mut self, hash: Hash, data: &[u8]) -> bool {
        if !self.blobs.insert(hash) {
            return false;
        }
        let mut payload = Vec::with_capacity(32 + data.len());
        payload.extend_from_slice(&hash);
        payload.extend_from_slice(data);
        Self::record(&mut self.out, KIND_BLOB, &payload);
        true
    }

    /// Whether the stream wrote the blob of `hash` already.
    pub fn holds_blob(&self, hash: &Hash) -> bool {
        self.blobs.contains(hash)
    }

    /// Writes `call` as a call record, after the name records its names need; the record's
    /// length.
    pub fn call(&mut self, call: &Call) -> usize {
        let mut p = std::mem::take(&mut self.scratch);
        p.clear();
        let selector = self.name(&call.selector);
        put_varint(&mut p, selector);
        put_varint(&mut p, call.receiver);
        put_varint(&mut p, call.result);
        put_varint(&mut p, call.args.len() as u64);
        for v in &call.args {
            self.value(&mut p, v);
        }
        put_varint(&mut p, call.answers.len() as u64);
        for (name, v) in &call.answers {
            let id = self.name(name);
            put_varint(&mut p, id);
            self.value(&mut p, v);
        }
        let before = self.out.len();
        Self::record(&mut self.out, KIND_CALL, &p);
        self.scratch = p;
        self.calls += 1;
        self.out.len() - before
    }

    fn value(&mut self, p: &mut Vec<u8>, v: &Value) {
        match v {
            Value::Nil | Value::Host(..) | Value::Debug(_) => p.push(0),
            Value::Bool(false) => p.push(1),
            Value::Bool(true) => p.push(2),
            Value::Uint(u) => {
                p.push(3);
                put_varint(p, *u);
            }
            Value::Sint(s) => {
                p.push(4);
                put_varint(p, zigzag(*s));
            }
            Value::F32(f) => {
                p.push(5);
                p.extend_from_slice(&f.to_le_bytes());
            }
            Value::F64(f) => {
                p.push(6);
                p.extend_from_slice(&f.to_le_bytes());
            }
            Value::Object(id) => {
                p.push(7);
                put_varint(p, *id);
            }
            Value::Address(a) => {
                p.push(8);
                put_varint(p, *a);
            }
            Value::Str(s) => {
                p.push(9);
                put_varint(p, s.len() as u64);
                p.extend_from_slice(s.as_bytes());
            }
            Value::Bytes(b) => {
                p.push(10);
                put_varint(p, b.len() as u64);
                p.extend_from_slice(b);
            }
            Value::Blob(h) => {
                p.push(11);
                p.extend_from_slice(h);
            }
            Value::List(items) => {
                p.push(12);
                put_varint(p, items.len() as u64);
                for item in items {
                    self.value(p, item);
                }
            }
            Value::Map(pairs) => {
                p.push(13);
                put_varint(p, pairs.len() as u64);
                for (name, item) in pairs {
                    let id = self.name(name);
                    put_varint(p, id);
                    self.value(p, item);
                }
            }
        }
    }
}

/// One record read back, its names resolved.
#[derive(Debug, Clone, PartialEq)]
pub enum Record {
    /// A name record: the name it defines.
    Name(String),
    /// A blob record: its hash and bytes.
    Blob(Hash, Vec<u8>),
    /// A call record.
    Call(Call),
}

/// Why a stream did not read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadError {
    /// The file does not start with [`HEADER`]'s magic and version.
    Header,
    /// A record of a kind the format has no kind of, at this byte.
    UnknownKind(u8, usize),
    /// A record's payload does not read as its kind's, at this byte.
    Malformed(usize),
}

/// The records read, and whether a record was cut short at the end (a truncated tail, which
/// stops the read and leaves every record before it).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Read {
    /// The records, in order.
    pub records: Vec<Record>,
    /// Whether the last record was cut short.
    pub truncated: bool,
}

struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Cursor<'_> {
    fn byte(&mut self) -> Option<u8> {
        let b = *self.bytes.get(self.at)?;
        self.at += 1;
        Some(b)
    }

    fn varint(&mut self) -> Option<u64> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let b = self.byte()?;
            v |= u64::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                return Some(v);
            }
        }
        None
    }

    fn take(&mut self, n: usize) -> Option<&[u8]> {
        let end = self.at.checked_add(n)?;
        let s = self.bytes.get(self.at..end)?;
        self.at = end;
        Some(s)
    }
}

/// Reads a whole stream, or a segment: the header, then each record. A segment of a stream
/// whose names or blobs an earlier segment defined reads with that segment's names given in
/// `names` (the ids in order from 1), which grows as names are read.
pub fn read(bytes: &[u8], names: &mut Vec<String>) -> Result<Read, ReadError> {
    if bytes.get(..12) != Some(&HEADER[..12]) {
        return Err(ReadError::Header);
    }
    let mut c = Cursor { bytes, at: 16 };
    let mut out = Read::default();
    while c.at < bytes.len() {
        let start = c.at;
        let Some(kind) = c.byte() else { break };
        let Some(len) = c.varint() else {
            out.truncated = true;
            break;
        };
        let Some(payload) = usize::try_from(len).ok().and_then(|n| c.take(n)) else {
            out.truncated = true;
            break;
        };
        let record = match kind {
            KIND_NAME => {
                let name =
                    String::from_utf8(payload.to_vec()).map_err(|_| ReadError::Malformed(start))?;
                names.push(name.clone());
                Record::Name(name)
            }
            KIND_BLOB => {
                let hash: Hash = payload
                    .get(..32)
                    .and_then(|h| h.try_into().ok())
                    .ok_or(ReadError::Malformed(start))?;
                Record::Blob(hash, payload.get(32..).unwrap_or_default().to_vec())
            }
            KIND_CALL => {
                Record::Call(read_call(payload, names).ok_or(ReadError::Malformed(start))?)
            }
            other => return Err(ReadError::UnknownKind(other, start)),
        };
        out.records.push(record);
    }
    Ok(out)
}

fn name_of(names: &[String], id: u64) -> Option<String> {
    names
        .get(usize::try_from(id).ok()?.checked_sub(1)?)
        .cloned()
}

fn read_call(payload: &[u8], names: &[String]) -> Option<Call> {
    let mut c = Cursor {
        bytes: payload,
        at: 0,
    };
    let selector = name_of(names, c.varint()?)?;
    let receiver = c.varint()?;
    let result = c.varint()?;
    let n = c.varint()?;
    let mut args = Vec::new();
    for _ in 0..n {
        args.push(read_value(&mut c, names)?);
    }
    let n = c.varint()?;
    let mut answers = Vec::new();
    for _ in 0..n {
        let name = name_of(names, c.varint()?)?;
        answers.push((name, read_value(&mut c, names)?));
    }
    (c.at == payload.len()).then_some(Call {
        selector,
        receiver,
        result,
        args,
        answers,
    })
}

fn read_value(c: &mut Cursor<'_>, names: &[String]) -> Option<Value> {
    Some(match c.byte()? {
        0 => Value::Nil,
        1 => Value::Bool(false),
        2 => Value::Bool(true),
        3 => Value::Uint(c.varint()?),
        4 => Value::Sint(unzigzag(c.varint()?)),
        5 => Value::F32(f32::from_le_bytes(c.take(4)?.try_into().ok()?)),
        6 => Value::F64(f64::from_le_bytes(c.take(8)?.try_into().ok()?)),
        7 => Value::Object(c.varint()?),
        8 => Value::Address(c.varint()?),
        9 => {
            let n = usize::try_from(c.varint()?).ok()?;
            Value::Str(String::from_utf8(c.take(n)?.to_vec()).ok()?)
        }
        10 => {
            let n = usize::try_from(c.varint()?).ok()?;
            Value::Bytes(c.take(n)?.to_vec())
        }
        11 => Value::Blob(c.take(32)?.try_into().ok()?),
        12 => {
            let n = c.varint()?;
            let mut items = Vec::new();
            for _ in 0..n {
                items.push(read_value(c, names)?);
            }
            Value::List(items)
        }
        13 => {
            let n = c.varint()?;
            let mut pairs = Vec::new();
            for _ in 0..n {
                let name = name_of(names, c.varint()?)?;
                pairs.push((name, read_value(c, names)?));
            }
            Value::Map(pairs)
        }
        _ => return None,
    })
}

/// What a stream's segments hold, as text: its records and bytes by kind, its calls by selector,
/// its answers by name, the chunks its writes carry, and whether its last segment ends in a
/// truncated tail.
pub fn summary(segments: &[Vec<u8>]) -> Result<String, ReadError> {
    use std::collections::BTreeMap;
    use std::fmt::Write as _;
    let mut names = Vec::new();
    let mut kinds = [0u64; 4];
    let mut blob_bytes = 0u64;
    let mut calls: BTreeMap<String, u64> = BTreeMap::new();
    let mut answers: BTreeMap<String, u64> = BTreeMap::new();
    let mut write_pairs = 0u64;
    let mut truncated = false;
    let mut total = 0u64;
    for seg in segments {
        total += seg.len() as u64;
        let r = read(seg, &mut names)?;
        truncated = r.truncated;
        for rec in r.records {
            match rec {
                Record::Name(_) => kinds[1] += 1,
                Record::Blob(_, data) => {
                    kinds[2] += 1;
                    blob_bytes += data.len() as u64;
                }
                Record::Call(c) => {
                    kinds[3] += 1;
                    if c.selector == "m2v.write" {
                        write_pairs += c.args.len() as u64 / 2;
                    }
                    for (name, _) in c.answers {
                        *answers.entry(name).or_insert(0) += 1;
                    }
                    *calls.entry(c.selector).or_insert(0) += 1;
                }
            }
        }
    }
    let mut out = String::new();
    let _ = writeln!(out, "segments {} bytes {total}", segments.len());
    let _ = writeln!(
        out,
        "names {} blobs {} blob_bytes {blob_bytes}",
        kinds[1], kinds[2]
    );
    let _ = writeln!(
        out,
        "calls {} write_chunks {write_pairs} truncated {truncated}",
        kinds[3]
    );
    for (sel, n) in calls {
        let _ = writeln!(out, "call {n} {sel}");
    }
    for (name, n) in answers {
        let _ = writeln!(out, "answer {n} {name}");
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reads the stream whose segments `REIMS_VGPU_RECORD_READ` names (`DIR/PID`, the segments
    /// `DIR/PID.NNNN.m2vr`) and prints its summary: a check of a recording on the host that made
    /// it, run by hand.
    #[test]
    #[ignore = "reads a recording named by REIMS_VGPU_RECORD_READ, on the host that made it"]
    fn read_a_recording() {
        let base = std::env::var("REIMS_VGPU_RECORD_READ").unwrap();
        let mut segments = Vec::new();
        for n in 0.. {
            match std::fs::read(format!("{base}.{n:04}.m2vr")) {
                Ok(bytes) => segments.push(bytes),
                Err(_) => break,
            }
        }
        assert!(!segments.is_empty(), "no segment at {base}.0000.m2vr");
        println!("{}", summary(&segments).unwrap());
    }

    /// A summary counts each record kind and each selector.
    #[test]
    fn a_summary_counts_kinds_and_selectors() {
        let mut s = Stream::new();
        s.header();
        let h = s.blob(&[1; 80]);
        s.call(&Call {
            selector: "m2v.write".to_owned(),
            receiver: 3,
            args: vec![Value::Uint(0), Value::Blob(h)],
            ..Call::default()
        });
        s.call(&Call {
            selector: "commit".to_owned(),
            receiver: 4,
            answers: vec![("gpuAddress".to_owned(), Value::Address(1 << 46))],
            ..Call::default()
        });
        let text = summary(&[s.take()]).unwrap();
        assert!(text.contains("names 3 blobs 1 blob_bytes 80\n"));
        assert!(text.contains("calls 2 write_chunks 1 truncated false\n"));
        assert!(text.contains("call 1 commit\n"));
        assert!(text.contains("answer 1 gpuAddress\n"));
    }

    fn every_value() -> Vec<Value> {
        vec![
            Value::Nil,
            Value::Bool(false),
            Value::Bool(true),
            Value::Uint(0),
            Value::Uint(u64::MAX),
            Value::Sint(-1),
            Value::Sint(i64::MIN),
            Value::F32(1.5),
            Value::F64(-0.25),
            Value::Object(7),
            Value::Address(1 << 46),
            Value::Str("BGRA8Unorm".to_owned()),
            Value::Bytes(vec![1, 2, 3]),
            Value::Blob([9; 32]),
            Value::List(vec![Value::Uint(1), Value::List(vec![])]),
            Value::Map(vec![
                ("width".to_owned(), Value::Uint(64)),
                ("texture".to_owned(), Value::Object(3)),
            ]),
        ]
    }

    /// The header is the magic, version 2 and the flags, bit 0 set: the stream holds the guest's
    /// bytes at each announce
    #[test]
    fn the_header_is_the_magic_version_and_flags() {
        assert_eq!(&HEADER[..8], b"M2VRSTRM");
        assert_eq!(HEADER[8..12], 2u32.to_le_bytes());
        assert_eq!(HEADER[12..], 1u32.to_le_bytes());
        assert_eq!(FLAG_ANNOUNCED_BYTES_RECORDED, 1, "bit 0");
    }

    /// Varints and zigzag round-trip at their edges
    #[test]
    fn varints_and_zigzag_round_trip() {
        for v in [0, 1, 127, 128, 300, u64::from(u32::MAX), u64::MAX] {
            let mut out = Vec::new();
            put_varint(&mut out, v);
            let mut c = Cursor { bytes: &out, at: 0 };
            assert_eq!(c.varint(), Some(v));
            assert_eq!(c.at, out.len());
        }
        for v in [0, 1, -1, i64::MAX, i64::MIN] {
            assert_eq!(unzigzag(zigzag(v)), v);
        }
        assert_eq!(zigzag(-1), 1);
        assert_eq!(zigzag(1), 2);
    }

    /// Each record kind, and every value tag inside a call, reads back as it was written; names
    /// are defined once, before their first use
    #[test]
    fn each_record_kind_round_trips() {
        let mut s = Stream::new();
        s.header();
        let hash = s.blob(&[5; 100]);
        assert!(!s.blob_hashed(hash, &[5; 100]), "a blob is written once");
        let call = Call {
            selector: "newBufferWithLength:options:".to_owned(),
            receiver: 1,
            result: 2,
            args: every_value(),
            answers: vec![
                ("gpuAddress".to_owned(), Value::Address(1 << 46)),
                ("gpuLength".to_owned(), Value::Uint(4096)),
            ],
        };
        s.call(&call);
        let again = Call {
            selector: "release".to_owned(),
            receiver: 2,
            ..Call::default()
        };
        s.call(&again);
        let bytes = s.take();
        let mut names = Vec::new();
        let read = read(&bytes, &mut names).unwrap();
        assert!(!read.truncated);
        let calls: Vec<_> = read
            .records
            .iter()
            .filter_map(|r| match r {
                Record::Call(c) => Some(c.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(calls, [call, again]);
        assert_eq!(read.records[0], Record::Blob(hash, vec![5; 100]));
        assert_eq!(
            read.records[1],
            Record::Name("newBufferWithLength:options:".to_owned())
        );
        let defined = read
            .records
            .iter()
            .filter(|r| matches!(r, Record::Name(_)))
            .count();
        assert_eq!(defined, names.len());
        assert_eq!(names.iter().filter(|n| *n == "width").count(), 1);
    }

    /// A record cut short at the end is a truncated tail, and every record before it reads
    #[test]
    fn a_cut_record_is_a_truncated_tail() {
        let mut s = Stream::new();
        s.header();
        s.call(&Call {
            selector: "commit".to_owned(),
            receiver: 4,
            ..Call::default()
        });
        s.call(&Call {
            selector: "commit".to_owned(),
            receiver: 5,
            args: vec![Value::Bytes(vec![0; 40])],
            ..Call::default()
        });
        let whole = s.take();
        let cut = &whole[..whole.len() - 10];
        let read = read(cut, &mut Vec::new()).unwrap();
        assert!(read.truncated);
        assert_eq!(read.records.len(), 2, "the name and the first call");
        let Record::Call(first) = &read.records[1] else {
            panic!("a call")
        };
        assert_eq!(first.receiver, 4);
    }

    /// An unknown record kind is a typed error naming the kind and where it lies
    #[test]
    fn an_unknown_kind_is_a_typed_error() {
        let mut bytes = HEADER.to_vec();
        bytes.extend_from_slice(&[9, 0]);
        assert_eq!(
            read(&bytes, &mut Vec::new()),
            Err(ReadError::UnknownKind(9, 16))
        );
        assert_eq!(
            read(b"not a stream....", &mut Vec::new()),
            Err(ReadError::Header)
        );
    }

    /// A host range is written as nil
    #[test]
    fn a_host_range_is_written_as_nil() {
        let mut s = Stream::new();
        s.header();
        s.call(&Call {
            selector: "m2v.x".to_owned(),
            args: vec![Value::Host(0x1000, 64)],
            ..Call::default()
        });
        let bytes = s.take();
        let read = read(&bytes, &mut Vec::new()).unwrap();
        let Record::Call(c) = &read.records[1] else {
            panic!("a call")
        };
        assert_eq!(c.args, [Value::Nil]);
    }
}
