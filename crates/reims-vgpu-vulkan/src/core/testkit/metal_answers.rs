//! Metal's answers, recorded from the Metal platform on an M3 Max under macOS 27.0.1 (26A434),
//! 2026-11-07, before that platform was deleted: the tests that compared this library's answers
//! with Metal's check against them. They are Metal's, and can never be recorded again: never
//! rewrite them from this library's own output.
//!
//! A test checks its answer against Metal's under a key unique in the test ([`check`],
//! [`check_bytes`], [`matches`], [`matches_bytes`], [`near`], [`answer`]), through the macros
//! ([`crate::metal_check`] and the rest), which name the test's source file. A macro given no key
//! keys the answer by its place among the test's answers given none, `#0` first. A test's answers
//! are a text file next to that source, `metal_answers/<source file stem>.<test>.txt`: a header
//! line, then a line an answer, `<key>\t<value>`. The value is one of:
//! - the answer's `Debug` form in full;
//! - `bytes <length> <FNV-1a 128 digest> <first 32 bytes in hex>`: a byte buffer, or an answer's
//!   raw bytes ([`Raw`]), so a change of a `Debug` form never touches it;
//! - `hex <length> <every byte in hex>`: a byte buffer a test compares within a tolerance, where
//!   this library's answer on the Mac (MoltenVK) was not Metal's;
//! - `xz <length> <offset>`: the same, longer than [`HEX_LIMIT`] bytes, at `offset` in the bytes
//!   that the test's `.xz` file next to its text file holds.

// A test harness reports failure by panicking (17.4: test-only code); a malformed file of answers
// is such a failure.
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::unreachable,
    clippy::as_conversions
)]

use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::path::{Path, PathBuf};

use crate::core::sync::Mutex;

/// The longest byte buffer a file of answers holds as `hex`: a longer one is in its `.xz` file.
pub const HEX_LIMIT: usize = 256;

/// A file of answers, read: its answers by key, and its `.xz` file's bytes once read.
struct Answers {
    file: PathBuf,
    held: HashMap<String, String>,
    packed: Option<Vec<u8>>,
}

/// The files the checks have read in this process.
static READ: Mutex<Vec<Answers>> = Mutex::new(Vec::new());

/// How many answers given no key each file has had checked in this process.
static NEXT: Mutex<Vec<(PathBuf, usize)>> = Mutex::new(Vec::new());

/// The key of the next answer given no key in `source`'s test file: `#0`, `#1`, ...
pub fn next_key(source: &str, manifest: &str) -> String {
    let file = file_of(source, manifest);
    let mut next = NEXT.lock();
    let n = match next.iter_mut().find(|(f, _)| *f == file) {
        Some((_, n)) => n,
        None => {
            next.push((file, 0));
            &mut next.last_mut().unwrap_or_else(|| unreachable!()).1
        }
    };
    let key = format!("#{n}");
    *n += 1;
    key
}

/// The running test's name: the last part of its thread's name, which libtest and nextest give
/// it.
fn test_name() -> String {
    let thread = std::thread::current();
    let name = thread.name().unwrap_or_default();
    let last = name.rsplit("::").next().unwrap_or_default();
    assert!(
        !last.is_empty() && last != "main",
        "a Metal answer off a test's own thread ({name:?})"
    );
    last.to_owned()
}

/// The file of the running test's answers, next to `source` (a test's `file!()`), which lies
/// under an ancestor of `manifest` (its crate's `CARGO_MANIFEST_DIR`).
fn file_of(source: &str, manifest: &str) -> PathBuf {
    let src = Path::new(source);
    let path = if src.is_absolute() {
        src.to_path_buf()
    } else {
        let base = Path::new(manifest)
            .ancestors()
            .find(|a| a.join(src).exists())
            .unwrap_or_else(|| panic!("{source} under no ancestor of {manifest}"));
        base.join(src)
    };
    let stem = path.file_stem().unwrap_or_default().to_string_lossy();
    let dir = path
        .parent()
        .unwrap_or(Path::new("."))
        .join("metal_answers");
    dir.join(format!("{stem}.{}.txt", test_name()))
}

/// `key` as a line holds it: tabs and line ends become spaces.
fn key_text(key: &dyn Display) -> String {
    key.to_string().replace(['\t', '\n', '\r'], " ")
}

/// FNV-1a over `bytes`, 128 bits.
fn fnv128(bytes: &[u8]) -> u128 {
    let mut h: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
    for &b in bytes {
        h ^= u128::from(b);
        h = h.wrapping_mul(0x0000_0000_0100_0000_0000_0000_0000_013b);
    }
    h
}

/// A byte buffer's value: its length, digest and first 32 bytes.
fn bytes_value(bytes: &[u8]) -> String {
    let first: String = bytes.iter().take(32).map(|b| format!("{b:02x}")).collect();
    format!("bytes {} {:032x} {first}", bytes.len(), fnv128(bytes))
}

/// An answer's `Debug` form.
fn debug_value(answer: &dyn Debug) -> String {
    format!("{answer:?}")
}

/// The index in [`READ`] of `file`, read at its first use.
fn read(read: &mut Vec<Answers>, file: PathBuf) -> usize {
    if let Some(at) = read.iter().position(|a| a.file == file) {
        return at;
    }
    let text = std::fs::read_to_string(&file)
        .unwrap_or_else(|e| panic!("no Metal answers at {}: {e}", file.display()));
    let held = (text.lines().skip(1))
        .filter_map(|l| l.split_once('\t'))
        .map(|(k, v)| (k.to_owned(), v.to_owned()))
        .collect();
    read.push(Answers {
        file,
        held,
        packed: None,
    });
    read.len() - 1
}

/// The value recorded under `key` in `file`.
fn recorded(file: PathBuf, key: &str) -> String {
    let mut answers = READ.lock();
    let at = read(&mut answers, file);
    let a = &answers[at];
    (a.held.get(key))
        .cloned()
        .unwrap_or_else(|| panic!("{}: no Metal answer under {key:?}", a.file.display()))
}

/// `length` bytes at `offset` of `file`'s `.xz` file, decoded at its first use.
fn packed(file: PathBuf, length: usize, offset: usize) -> Vec<u8> {
    let mut answers = READ.lock();
    let at = read(&mut answers, file);
    let a = &mut answers[at];
    if a.packed.is_none() {
        let xz = a.file.with_extension("xz");
        let input = std::fs::read(&xz)
            .unwrap_or_else(|e| panic!("no Metal answers at {}: {e}", xz.display()));
        let mut bytes = Vec::new();
        lzma_rs::xz_decompress(&mut input.as_slice(), &mut bytes)
            .unwrap_or_else(|e| panic!("{}: {e:?}", xz.display()));
        a.packed = Some(bytes);
    }
    let bytes = a.packed.as_deref().unwrap_or_default();
    (bytes.get(offset..offset + length))
        .unwrap_or_else(|| panic!("{}: no {length} bytes at {offset}", a.file.display()))
        .to_vec()
}

/// The bytes of Metal's answer under `key` where it was recorded in full (`hex` or `xz`); `None`
/// where it was recorded by its digest (`bytes`).
pub fn full(source: &str, manifest: &str, key: &dyn Display) -> Option<Vec<u8>> {
    let file = file_of(source, manifest);
    let value = recorded(file.clone(), &key_text(key));
    let mut parts = value.splitn(3, ' ');
    let (form, length, rest) = (parts.next(), parts.next(), parts.next().unwrap_or_default());
    let length: usize = (length.and_then(|l| l.parse().ok()))
        .unwrap_or_else(|| panic!("{key}: not a byte buffer's ({value})"));
    let bytes: Vec<u8> = match form {
        Some("bytes") => return None,
        Some("hex") => (0..rest.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&rest[i..i + 2], 16).unwrap())
            .collect(),
        Some("xz") => packed(file, length, rest.parse().unwrap()),
        _ => panic!("{key}: not a byte buffer's ({value})"),
    };
    assert_eq!(length, bytes.len(), "{key}");
    Some(bytes)
}

/// The bytes of Metal's answer under `key`, for a test to compare `ours` with within its
/// tolerance: `ours` where they are Metal's, else Metal's recorded in full. Panics where `ours`
/// differ from an answer recorded by its digest alone.
pub fn near(source: &str, manifest: &str, key: &dyn Display, ours: &[u8]) -> Vec<u8> {
    let value = recorded(file_of(source, manifest), &key_text(key));
    if value == bytes_value(ours) {
        return ours.to_vec();
    }
    full(source, manifest, key)
        .unwrap_or_else(|| panic!("{key}: ours {}, Metal's {value}", bytes_value(ours)))
}

/// Metal's answer under `key`, a byte buffer, for a test to compare `ours` with: [`near`] of
/// `ours`, or where `ours` is none (this platform's run died), the answer recorded in full, else
/// none.
pub fn answer(source: &str, manifest: &str, key: &dyn Display, ours: Option<&[u8]>) -> Vec<u8> {
    match ours {
        Some(ours) => near(source, manifest, key, ours),
        None => full(source, manifest, key).unwrap_or_default(),
    }
}

/// The `Debug` form of Metal's answer recorded under `key`: an input a test took from Metal.
pub fn value(source: &str, manifest: &str, key: &dyn Display) -> String {
    recorded(file_of(source, manifest), &key_text(key))
}

/// Whether `ours` is Metal's answer under `key`.
pub fn matches(source: &str, manifest: &str, key: &dyn Display, ours: &dyn Debug) -> bool {
    recorded(file_of(source, manifest), &key_text(key)) == debug_value(ours)
}

/// Whether `ours`, a byte buffer, is Metal's answer under `key`.
pub fn matches_bytes(source: &str, manifest: &str, key: &dyn Display, ours: &[u8]) -> bool {
    recorded(file_of(source, manifest), &key_text(key)) == bytes_value(ours)
}

/// Panics unless `ours` is Metal's answer under `key` ([`matches`]).
pub fn check(source: &str, manifest: &str, key: &dyn Display, ours: &dyn Debug) {
    let (theirs, ours) = (
        recorded(file_of(source, manifest), &key_text(key)),
        debug_value(ours),
    );
    assert!(ours == theirs, "{key}: ours {ours}, Metal's {theirs}");
}

/// Panics unless `ours`, a byte buffer, is Metal's answer under `key` ([`matches_bytes`]).
pub fn check_bytes(source: &str, manifest: &str, key: &dyn Display, ours: &[u8]) {
    let (theirs, ours) = (
        recorded(file_of(source, manifest), &key_text(key)),
        bytes_value(ours),
    );
    assert!(ours == theirs, "{key}: ours {ours}, Metal's {theirs}");
}

/// An answer's raw bytes, which a byte buffer's value digests ([`check_bytes`]): each integer,
/// float (its bits) and length little-endian, in order. A change of a `Debug` form never changes
/// them.
pub trait Raw {
    /// Appends the raw bytes to `out`.
    fn raw(&self, out: &mut Vec<u8>);

    /// The raw bytes.
    fn raw_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.raw(&mut out);
        out
    }
}

macro_rules! raw_le {
    ($($t:ty),*) => {$(
        impl Raw for $t {
            fn raw(&self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }
        }
    )*};
}
raw_le!(u8, u16, u32, u64, u128, i8, i16, i32, i64, i128);

impl Raw for usize {
    fn raw(&self, out: &mut Vec<u8>) {
        (*self as u64).raw(out);
    }
}

impl Raw for isize {
    fn raw(&self, out: &mut Vec<u8>) {
        (*self as i64).raw(out);
    }
}

impl Raw for bool {
    fn raw(&self, out: &mut Vec<u8>) {
        out.push(u8::from(*self));
    }
}

impl Raw for f32 {
    fn raw(&self, out: &mut Vec<u8>) {
        self.to_bits().raw(out);
    }
}

impl Raw for f64 {
    fn raw(&self, out: &mut Vec<u8>) {
        self.to_bits().raw(out);
    }
}

impl Raw for str {
    fn raw(&self, out: &mut Vec<u8>) {
        self.len().raw(out);
        out.extend_from_slice(self.as_bytes());
    }
}

impl Raw for String {
    fn raw(&self, out: &mut Vec<u8>) {
        self.as_str().raw(out);
    }
}

impl<T: Raw + ?Sized> Raw for &T {
    fn raw(&self, out: &mut Vec<u8>) {
        (**self).raw(out);
    }
}

impl<T: Raw> Raw for [T] {
    fn raw(&self, out: &mut Vec<u8>) {
        self.len().raw(out);
        for t in self {
            t.raw(out);
        }
    }
}

impl<T: Raw, const N: usize> Raw for [T; N] {
    fn raw(&self, out: &mut Vec<u8>) {
        for t in self {
            t.raw(out);
        }
    }
}

impl<T: Raw> Raw for Vec<T> {
    fn raw(&self, out: &mut Vec<u8>) {
        self.as_slice().raw(out);
    }
}

impl<T: Raw> Raw for Option<T> {
    fn raw(&self, out: &mut Vec<u8>) {
        match self {
            None => out.push(0),
            Some(t) => {
                out.push(1);
                t.raw(out);
            }
        }
    }
}

impl<T: Raw, E: Raw> Raw for Result<T, E> {
    fn raw(&self, out: &mut Vec<u8>) {
        match self {
            Ok(t) => {
                out.push(0);
                t.raw(out);
            }
            Err(e) => {
                out.push(1);
                e.raw(out);
            }
        }
    }
}

impl Raw for () {
    fn raw(&self, _: &mut Vec<u8>) {}
}

macro_rules! raw_tuple {
    ($(($($n:tt $t:ident),*)),*) => {$(
        impl<$($t: Raw),*> Raw for ($($t,)*) {
            fn raw(&self, out: &mut Vec<u8>) {
                $(self.$n.raw(out);)*
            }
        }
    )*};
}
raw_tuple!(
    (0 A),
    (0 A, 1 B),
    (0 A, 1 B, 2 C),
    (0 A, 1 B, 2 C, 3 D),
    (0 A, 1 B, 2 C, 3 D, 4 E),
    (0 A, 1 B, 2 C, 3 D, 4 E, 5 F),
    (0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G),
    (0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G, 7 H)
);

impl Raw for crate::core::platform::types::TextureLevelInfo {
    fn raw(&self, out: &mut Vec<u8>) {
        for v in [
            self.field_0,
            self.field_8,
            self.field_10,
            self.field_18,
            self.field_20,
            self.field_28,
            self.field_30,
            self.aligned_height,
            self.field_40,
            self.bytes_per_pixel,
            self.bytes_per_row,
            self.field_58,
            self.field_60,
        ] {
            v.raw(out);
        }
    }
}

impl Raw for crate::core::platform::types::PixelFormatInfo {
    fn raw(&self, out: &mut Vec<u8>) {
        self.name.raw(out);
        for v in [
            self.flags,
            self.cast_class,
            self.block_bytes,
            self.block_width,
            self.block_height,
            self.block_depth,
        ] {
            v.raw(out);
        }
    }
}

/// [`check_bytes`] of an answer's raw bytes ([`Raw`]) from a test's source file.
#[macro_export]
macro_rules! metal_check_raw {
    ($ours:expr $(,)?) => {
        $crate::metal_check_bytes!(&$crate::core::testkit::metal_answers::Raw::raw_bytes(
            &$ours
        ))
    };
    ($key:expr, $ours:expr $(,)?) => {
        $crate::metal_check_bytes!(
            $key,
            &$crate::core::testkit::metal_answers::Raw::raw_bytes(&$ours)
        )
    };
}

/// [`matches_bytes`] of an answer's raw bytes ([`Raw`]) from a test's source file.
#[macro_export]
macro_rules! metal_matches_raw {
    ($ours:expr $(,)?) => {
        $crate::metal_matches_bytes!(&$crate::core::testkit::metal_answers::Raw::raw_bytes(
            &$ours
        ))
    };
    ($key:expr, $ours:expr $(,)?) => {
        $crate::metal_matches_bytes!(
            $key,
            &$crate::core::testkit::metal_answers::Raw::raw_bytes(&$ours)
        )
    };
}

/// [`near`] from a test's source file.
#[macro_export]
macro_rules! metal_near {
    ($key:expr, $ours:expr $(,)?) => {
        $crate::core::testkit::metal_answers::near(
            file!(),
            env!("CARGO_MANIFEST_DIR"),
            &$key,
            $ours,
        )
    };
}

/// [`answer`] from a test's source file.
#[macro_export]
macro_rules! metal_answer {
    ($key:expr, $ours:expr $(,)?) => {
        $crate::core::testkit::metal_answers::answer(
            file!(),
            env!("CARGO_MANIFEST_DIR"),
            &$key,
            $ours,
        )
    };
}

/// [`full`] from a test's source file.
#[macro_export]
macro_rules! metal_full {
    ($key:expr $(,)?) => {
        $crate::core::testkit::metal_answers::full(file!(), env!("CARGO_MANIFEST_DIR"), &$key)
    };
}

/// [`value`] from a test's source file.
#[macro_export]
macro_rules! metal_value {
    ($key:expr $(,)?) => {
        $crate::core::testkit::metal_answers::value(file!(), env!("CARGO_MANIFEST_DIR"), &$key)
    };
}

/// [`check`] from a test's source file.
#[macro_export]
macro_rules! metal_check {
    ($ours:expr $(,)?) => {
        $crate::core::testkit::metal_answers::check(
            file!(),
            env!("CARGO_MANIFEST_DIR"),
            &$crate::core::testkit::metal_answers::next_key(file!(), env!("CARGO_MANIFEST_DIR")),
            &$ours,
        )
    };
    ($key:expr, $ours:expr $(,)?) => {
        $crate::core::testkit::metal_answers::check(
            file!(),
            env!("CARGO_MANIFEST_DIR"),
            &$key,
            &$ours,
        )
    };
}

/// [`check_bytes`] from a test's source file.
#[macro_export]
macro_rules! metal_check_bytes {
    ($ours:expr $(,)?) => {
        $crate::core::testkit::metal_answers::check_bytes(
            file!(),
            env!("CARGO_MANIFEST_DIR"),
            &$crate::core::testkit::metal_answers::next_key(file!(), env!("CARGO_MANIFEST_DIR")),
            $ours,
        )
    };
    ($key:expr, $ours:expr $(,)?) => {
        $crate::core::testkit::metal_answers::check_bytes(
            file!(),
            env!("CARGO_MANIFEST_DIR"),
            &$key,
            $ours,
        )
    };
}

/// [`matches`] from a test's source file.
#[macro_export]
macro_rules! metal_matches {
    ($ours:expr $(,)?) => {
        $crate::core::testkit::metal_answers::matches(
            file!(),
            env!("CARGO_MANIFEST_DIR"),
            &$crate::core::testkit::metal_answers::next_key(file!(), env!("CARGO_MANIFEST_DIR")),
            &$ours,
        )
    };
    ($key:expr, $ours:expr $(,)?) => {
        $crate::core::testkit::metal_answers::matches(
            file!(),
            env!("CARGO_MANIFEST_DIR"),
            &$key,
            &$ours,
        )
    };
}

/// [`matches_bytes`] from a test's source file.
#[macro_export]
macro_rules! metal_matches_bytes {
    ($ours:expr $(,)?) => {
        $crate::core::testkit::metal_answers::matches_bytes(
            file!(),
            env!("CARGO_MANIFEST_DIR"),
            &$crate::core::testkit::metal_answers::next_key(file!(), env!("CARGO_MANIFEST_DIR")),
            $ours,
        )
    };
    ($key:expr, $ours:expr $(,)?) => {
        $crate::core::testkit::metal_answers::matches_bytes(
            file!(),
            env!("CARGO_MANIFEST_DIR"),
            &$key,
            $ours,
        )
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The digest is FNV-1a 128's, and a byte buffer's value names its length, digest and first
    /// bytes
    #[test]
    fn a_byte_buffer_is_its_length_digest_and_first_bytes() {
        assert_eq!(fnv128(b""), 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d);
        assert_eq!(fnv128(b"a"), 0xd228_cb69_6f1a_8caf_7891_2b70_4e4a_8964);
        assert_eq!(
            bytes_value(&[1, 2, 0xff]),
            format!("bytes 3 {:032x} 0102ff", fnv128(&[1, 2, 0xff]))
        );
    }
}
