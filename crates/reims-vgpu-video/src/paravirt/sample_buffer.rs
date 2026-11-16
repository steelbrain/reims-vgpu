//! Sample buffers as `=sbf` parameters carry them: an `sbuf` atom whose children hold the timing,
//! the data, the sizes, the format description where it changed, and the attachments, as
//! CoreMedia's serializer writes them.
//!
//! The atom headers are big-endian; the scalars inside them are the guest's native order,
//! little-endian on arm64 and x86-64 alike. A property list leaf (`nmbv`, `bulv`, `strk`, `datv`,
//! `dtev`) holds a binary property list of one value. A dictionary's keys are a known key's index
//! (`idxk`, a native u16, [`known_keys`]) or a `strk` string.
//!
//! Every value is spent from one budget for the whole serialization ([`plist::Budget`]): each leaf
//! is a property list of its own, and a serialization of many leaves, each within one list's
//! bounds, would otherwise decode to far more than its bytes.

use crate::core::fourcc::FourCc;
use crate::paravirt::WireError;
use crate::paravirt::atom::{Atom, AtomWriter, siblings};
use crate::paravirt::known_keys;
use crate::paravirt::message::MediaTime;
use crate::paravirt::plist::{self, Budget, Value};

const SAMPLE_BUFFER: FourCc = FourCc::new(*b"sbuf");
const OUTPUT_PRESENTATION_TIME: FourCc = FourCc::new(*b"opts");
const TIMING: FourCc = FourCc::new(*b"stia");
const DATA: FourCc = FourCc::new(*b"sdat");
const SAMPLE_COUNT: FourCc = FourCc::new(*b"nsmp");
const SIZES: FourCc = FourCc::new(*b"ssiz");
const FORMAT_DESCRIPTION: FourCc = FourCc::new(*b"fdsc");
const PROPAGATED_ATTACHMENTS: FourCc = FourCc::new(*b"satt");
const OTHER_ATTACHMENTS: FourCc = FourCc::new(*b"sat2");
const SAMPLE_ATTACHMENTS: FourCc = FourCc::new(*b"sary");

const MEDIA_TYPE: FourCc = FourCc::new(*b"mdia");
const DIMENSIONS: FourCc = FourCc::new(*b"vdim");
const CODEC: FourCc = FourCc::new(*b"codc");
const EXTENSIONS: FourCc = FourCc::new(*b"extn");

const DICTIONARY: FourCc = FourCc::new(*b"dict");
const ENTRY: FourCc = FourCc::new(*b"keyv");
const KEY_INDEX: FourCc = FourCc::new(*b"idxk");
const ARRAY: FourCc = FourCc::new(*b"aray");
const NULL: FourCc = FourCc::new(*b"null");
const NUMBER: FourCc = FourCc::new(*b"nmbv");
const BOOLEAN: FourCc = FourCc::new(*b"bulv");
const STRING: FourCc = FourCc::new(*b"strk");
const BYTES: FourCc = FourCc::new(*b"datv");
const DATE: FourCc = FourCc::new(*b"dtev");
const COLOR_SPACE: FourCc = FourCc::new(*b"clrs");
const URL: FourCc = FourCc::new(*b"urlv");

/// The media type of video.
pub const MEDIA_TYPE_VIDEO: FourCc = FourCc::new(*b"vide");
/// The media type of an auxiliary picture.
pub const MEDIA_TYPE_AUXILIARY_PICTURE: FourCc = FourCc::new(*b"auxv");
/// The media type of a picture.
pub const MEDIA_TYPE_PICTURE: FourCc = FourCc::new(*b"pict");

/// The size of one sample's timing entry: its duration, presentation and decode time stamps.
pub const TIMING_SIZE: usize = 72;

/// The deepest a serialization's dictionaries, arrays and format descriptions nest, the
/// attachment atoms' own dictionary or array level one: ours, as the reader recurses once a level.
/// A sample's attachments nest a few levels.
pub const MAX_NESTING: u32 = 32;

/// A value in a serialized dictionary or array.
#[derive(Clone, Debug, PartialEq)]
pub enum SerializedValue {
    /// `dict`: entries in the order they were written.
    Dictionary(Vec<(String, SerializedValue)>),
    /// `aray`.
    Array(Vec<SerializedValue>),
    /// A property list leaf: a number, boolean, string, data or date.
    PropertyList(Value),
    /// `null`: kCFNull.
    Null,
    /// `clrs`: a CGColorSpace, as its property list.
    ColorSpace(Value),
    /// `urlv`: a URL, as its absolute string.
    Url(String),
    /// `fdsc`: a format description.
    FormatDescription(Box<SerializedFormatDescription>),
}

/// The entry `key` of a serialized dictionary.
pub fn entry<'a>(
    entries: &'a [(String, SerializedValue)],
    key: &str,
) -> Option<&'a SerializedValue> {
    entries
        .iter()
        .find(|(found, _)| found == key)
        .map(|(_, value)| value)
}

/// A format description as the serialization carries it: its media type, dimensions, codec and
/// extensions.
#[derive(Clone, Debug, PartialEq)]
pub struct SerializedFormatDescription {
    /// The media type, `vide` for video.
    pub media_type: FourCc,
    /// The codec type.
    pub codec: FourCc,
    /// The width.
    pub width: i32,
    /// The height.
    pub height: i32,
    /// The extensions dictionary, such as `SampleDescriptionExtensionAtoms` = {`avcC` = data}.
    pub extensions: Option<Vec<(String, SerializedValue)>>,
}

impl SerializedFormatDescription {
    /// The value of the extension `key`.
    pub fn extension(&self, key: &str) -> Option<&SerializedValue> {
        entry(self.extensions.as_deref()?, key)
    }

    /// The atom `kind` of the `SampleDescriptionExtensionAtoms` extension, such as `avcC`.
    pub fn sample_description_atom(&self, kind: &str) -> Option<&[u8]> {
        let SerializedValue::Dictionary(atoms) =
            self.extension("SampleDescriptionExtensionAtoms")?
        else {
            return None;
        };
        match entry(atoms, kind)? {
            SerializedValue::PropertyList(Value::Data(bytes)) => Some(bytes),
            _ => None,
        }
    }
}

/// One sample's timing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SampleTiming {
    /// The duration.
    pub duration: MediaTime,
    /// The presentation time stamp.
    pub presentation: MediaTime,
    /// The decode time stamp.
    pub decode: MediaTime,
}

impl SampleTiming {
    fn from_native(bytes: &[u8]) -> Option<Self> {
        Some(Self {
            duration: MediaTime::from_native(bytes.get(0..24)?)?,
            presentation: MediaTime::from_native(bytes.get(24..48)?)?,
            decode: MediaTime::from_native(bytes.get(48..72)?)?,
        })
    }

    /// The timing's 72 native bytes, as a `stia` entry holds it.
    fn to_native(self) -> [u8; TIMING_SIZE] {
        let mut bytes = [0; TIMING_SIZE];
        bytes[..24].copy_from_slice(&self.duration.to_native());
        bytes[24..48].copy_from_slice(&self.presentation.to_native());
        bytes[48..].copy_from_slice(&self.decode.to_native());
        bytes
    }
}

/// A sample buffer read from a serialization.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct SampleBuffer {
    /// The number of samples: `nsmp`, or 1 where there is data and no count.
    pub sample_count: u32,
    /// The timing entries.
    pub timings: Vec<SampleTiming>,
    /// The sample sizes.
    pub sizes: Vec<u32>,
    /// The sample data.
    pub data: Option<Vec<u8>>,
    /// The output presentation time stamp, where one was set.
    pub output_presentation_time: Option<MediaTime>,
    /// The format description, where this serialization carries one. The writer leaves it out
    /// where it equals the last one it wrote, so a reader keeps the last one it read.
    pub format_description: Option<SerializedFormatDescription>,
    /// `satt`: the attachments that propagate.
    pub propagated_attachments: Option<Vec<(String, SerializedValue)>>,
    /// `sat2`: the attachments that do not.
    pub other_attachments: Option<Vec<(String, SerializedValue)>>,
    /// `sary`: each sample's attachments.
    pub sample_attachments: Option<Vec<SerializedValue>>,
}

fn native_u32(atom: &Atom<'_>) -> Result<u32, WireError> {
    atom.payload()
        .get(..4)
        .and_then(|bytes| <[u8; 4]>::try_from(bytes).ok())
        .map(u32::from_le_bytes)
        .ok_or(WireError::Malformed("short atom"))
}

fn native_i32(bytes: &[u8]) -> Option<i32> {
    Some(i32::from_le_bytes(bytes.get(..4)?.try_into().ok()?))
}

/// The children of `atom`; an atom with no payload has none.
fn children<'a>(atom: &Atom<'a>) -> impl Iterator<Item = Result<Atom<'a>, WireError>> + use<'a> {
    siblings(atom.payload())
}

/// The first child of `atom`, which must be of type `kind`.
fn only_child<'a>(atom: &Atom<'a>, kind: FourCc) -> Result<Option<Atom<'a>>, WireError> {
    match children(atom).next().transpose()? {
        Some(child) if child.kind == kind => Ok(Some(child)),
        Some(_) => Err(WireError::Malformed("unexpected child atom")),
        None => Ok(None),
    }
}

/// `depth` one level deeper, or malformed past [`MAX_NESTING`].
fn deeper(depth: u32) -> Result<u32, WireError> {
    Some(depth + 1)
        .filter(|&d| d <= MAX_NESTING)
        .ok_or(WireError::Malformed("sample buffer values nest too deep"))
}

/// Reads a value by its atom type, `depth` the level of the container it is in, spending it from
/// `budget`.
fn read_value(
    atom: &Atom<'_>,
    depth: u32,
    budget: &mut Budget,
) -> Result<SerializedValue, WireError> {
    match atom.kind {
        NULL => {
            budget.spend(1, 0)?;
            Ok(SerializedValue::Null)
        }
        DICTIONARY => Ok(SerializedValue::Dictionary(read_dictionary(
            atom,
            deeper(depth)?,
            budget,
        )?)),
        ARRAY => Ok(SerializedValue::Array(read_array(
            atom,
            deeper(depth)?,
            budget,
        )?)),
        NUMBER | BOOLEAN | BYTES | DATE | STRING => Ok(SerializedValue::PropertyList(
            plist::decode_within(atom.payload(), budget)?,
        )),
        COLOR_SPACE => Ok(SerializedValue::ColorSpace(plist::decode_within(
            atom.payload(),
            budget,
        )?)),
        URL => match plist::decode_within(atom.payload(), budget)? {
            Value::String(url) => Ok(SerializedValue::Url(url)),
            _ => Err(WireError::Malformed("URL is not a string")),
        },
        FORMAT_DESCRIPTION => Ok(SerializedValue::FormatDescription(Box::new(
            read_format_description(atom, deeper(depth)?, budget)?,
        ))),
        _ => Err(WireError::Malformed("unknown value atom")),
    }
}

/// Reads a `dict` at nesting `depth`: `keyv` children only, each a key then a value. A repeated
/// key keeps its first value, as `CFDictionaryAddValue` does. The dictionary and each key and
/// value are spent from `budget`.
fn read_dictionary(
    atom: &Atom<'_>,
    depth: u32,
    budget: &mut Budget,
) -> Result<Vec<(String, SerializedValue)>, WireError> {
    budget.spend(1, 0)?;
    let mut entries: Vec<(String, SerializedValue)> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for child in children(atom) {
        let child = child?;
        if child.kind != ENTRY {
            return Err(WireError::Malformed("dictionary child is not an entry"));
        }
        let mut parts = children(&child);
        let key_atom = parts
            .next()
            .transpose()?
            .ok_or(WireError::Malformed("entry without a key"))?;
        let key = match key_atom.kind {
            KEY_INDEX => {
                budget.spend(1, 0)?;
                let index = key_atom
                    .payload()
                    .try_into()
                    .map(u16::from_le_bytes)
                    .map_err(|_| WireError::Malformed("key index length"))?;
                known_keys::key_at(index)
                    .ok_or(WireError::Malformed("key index"))?
                    .to_owned()
            }
            STRING => match plist::decode_within(key_atom.payload(), budget)? {
                Value::String(key) => key,
                _ => return Err(WireError::Malformed("key is not a string")),
            },
            _ => return Err(WireError::Malformed("entry key atom")),
        };
        let value_atom = parts
            .next()
            .transpose()?
            .ok_or(WireError::Malformed("entry without a value"))?;
        let value = read_value(&value_atom, depth, budget)?;
        if seen.insert(key.clone()) {
            entries.push((key, value));
        }
    }
    Ok(entries)
}

/// Reads an `aray` at nesting `depth`, skipping children with no payload. The array and each value
/// are spent from `budget`.
fn read_array(
    atom: &Atom<'_>,
    depth: u32,
    budget: &mut Budget,
) -> Result<Vec<SerializedValue>, WireError> {
    budget.spend(1, 0)?;
    let mut values = Vec::new();
    for child in children(atom) {
        let child = child?;
        if !child.payload().is_empty() {
            values.push(read_value(&child, depth, budget)?);
        }
    }
    Ok(values)
}

/// Reads an `fdsc` at nesting `depth`: the last `mdia`, `vdim` and `codc` win, unknown children
/// are skipped, and only video-like media types make a description. It and its extensions are
/// spent from `budget`.
fn read_format_description(
    atom: &Atom<'_>,
    depth: u32,
    budget: &mut Budget,
) -> Result<SerializedFormatDescription, WireError> {
    budget.spend(1, 0)?;
    let mut media_type = FourCc(0);
    let mut dimensions = (0, 0);
    let mut codec = FourCc(0);
    let mut extensions = None;
    for child in children(atom) {
        let child = child?;
        match child.kind {
            MEDIA_TYPE => media_type = FourCc(native_u32(&child)?),
            DIMENSIONS => {
                let payload = child.payload();
                dimensions = (
                    native_i32(payload).ok_or(WireError::Malformed("short atom"))?,
                    payload
                        .get(4..)
                        .and_then(native_i32)
                        .ok_or(WireError::Malformed("short atom"))?,
                );
            }
            CODEC => codec = FourCc(native_u32(&child)?),
            EXTENSIONS => {
                extensions = match only_child(&child, DICTIONARY)? {
                    Some(dictionary) => Some(read_dictionary(&dictionary, deeper(depth)?, budget)?),
                    None => None,
                };
            }
            _ => {}
        }
    }
    if ![
        MEDIA_TYPE_VIDEO,
        MEDIA_TYPE_AUXILIARY_PICTURE,
        MEDIA_TYPE_PICTURE,
    ]
    .contains(&media_type)
    {
        return Err(WireError::Malformed("format description media type"));
    }
    Ok(SerializedFormatDescription {
        media_type,
        codec,
        width: dimensions.0,
        height: dimensions.1,
        extensions,
    })
}

impl SampleBuffer {
    /// Reads a serialization.
    ///
    /// A second `stia`, `ssiz` or `fdsc` is malformed, as is an `ssiz` whose length is not a
    /// multiple of 4, and atoms the reader does not know are skipped. The counts must make a
    /// sample buffer: no timing entry, one, or one per sample; no size, one, or one per sample;
    /// and as many sample attachment dictionaries as samples.
    pub fn parse(serialization: &[u8]) -> Result<Self, WireError> {
        let mut budget = Budget::new();
        let root = siblings(serialization)
            .next()
            .transpose()?
            .filter(|root| root.kind == SAMPLE_BUFFER)
            .ok_or(WireError::Malformed("not a sample buffer"))?;
        let mut buffer = Self::default();
        let mut have_timings = false;
        let mut have_sizes = false;
        for child in children(&root) {
            let child = child?;
            let payload = child.payload();
            match child.kind {
                TIMING => {
                    if have_timings {
                        return Err(WireError::Malformed("second timing atom"));
                    }
                    have_timings = true;
                    buffer.timings = payload
                        .as_chunks::<TIMING_SIZE>()
                        .0
                        .iter()
                        .filter_map(|timing| SampleTiming::from_native(timing))
                        .collect();
                }
                SIZES => {
                    if have_sizes {
                        return Err(WireError::Malformed("second sizes atom"));
                    }
                    if payload.len() % 4 != 0 {
                        return Err(WireError::Malformed("sizes length"));
                    }
                    have_sizes = true;
                    buffer.sizes = payload
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|size| u32::from_le_bytes(*size))
                        .collect();
                }
                DATA => {
                    buffer.data = Some(payload.to_vec());
                    if buffer.sample_count == 0 {
                        buffer.sample_count = 1;
                    }
                }
                SAMPLE_COUNT => buffer.sample_count = native_u32(&child)?,
                OUTPUT_PRESENTATION_TIME => {
                    buffer.output_presentation_time = Some(
                        MediaTime::from_native(payload)
                            .ok_or(WireError::Malformed("short atom"))?,
                    );
                }
                FORMAT_DESCRIPTION => {
                    if buffer.format_description.is_some() {
                        return Err(WireError::Malformed("second format description"));
                    }
                    buffer.format_description =
                        Some(read_format_description(&child, 1, &mut budget)?);
                }
                PROPAGATED_ATTACHMENTS | OTHER_ATTACHMENTS => {
                    if let Some(dictionary) = only_child(&child, DICTIONARY)? {
                        let entries = Some(read_dictionary(&dictionary, 1, &mut budget)?);
                        if child.kind == PROPAGATED_ATTACHMENTS {
                            buffer.propagated_attachments = entries;
                        } else {
                            buffer.other_attachments = entries;
                        }
                    }
                }
                SAMPLE_ATTACHMENTS => {
                    if let Some(array) = only_child(&child, ARRAY)? {
                        // No element with a payload makes no array, and no sample attachments.
                        let attachments = read_array(&array, 1, &mut budget)?;
                        buffer.sample_attachments =
                            Some(attachments).filter(|attachments| !attachments.is_empty());
                    }
                }
                _ => {}
            }
        }
        buffer.check_counts()?;
        Ok(buffer)
    }

    /// The counts `CMSampleBufferCreateReady` and the attachment copy accept.
    fn check_counts(&self) -> Result<(), WireError> {
        let samples = usize::try_from(self.sample_count).unwrap_or(usize::MAX);
        let fits = |entries: usize| entries == 0 || entries == 1 || entries == samples;
        if !fits(self.timings.len()) {
            return Err(WireError::Malformed("timing entries"));
        }
        if samples != 0 && !fits(self.sizes.len()) {
            return Err(WireError::Malformed("size entries"));
        }
        if let Some(attachments) = &self.sample_attachments
            && attachments.len() != samples
        {
            return Err(WireError::Malformed("sample attachment entries"));
        }
        Ok(())
    }

    /// The first sample's attachment `key`, where it has one.
    pub fn sample_attachment(&self, key: &str) -> Option<&SerializedValue> {
        match self.sample_attachments.as_deref()?.first()? {
            SerializedValue::Dictionary(entries) => entry(entries, key),
            _ => None,
        }
    }
}

impl SampleBuffer {
    /// The buffer's serialization, as CoreMedia's writer orders it: the output time stamp where
    /// valid, the timings, the data, the count where nonzero and the sizes, the format
    /// description, then the three attachment atoms. Empty dictionaries and arrays are left out,
    /// as the writer leaves them out; [`Self::parse`] reads it back.
    pub fn serialize(&self) -> Vec<u8> {
        let mut writer = AtomWriter::new();
        writer.begin(SAMPLE_BUFFER);
        if let Some(time) = self.output_presentation_time.filter(|time| time.is_valid()) {
            writer.atom(OUTPUT_PRESENTATION_TIME, &time.to_native());
        }
        if !self.timings.is_empty() {
            let timings: Vec<u8> = self.timings.iter().flat_map(|t| t.to_native()).collect();
            writer.atom(TIMING, &timings);
        }
        if let Some(data) = &self.data {
            writer.atom(DATA, data);
        }
        if self.sample_count != 0 {
            writer.atom(SAMPLE_COUNT, &self.sample_count.to_le_bytes());
        }
        if !self.sizes.is_empty() {
            let sizes: Vec<u8> = (self.sizes.iter()).flat_map(|n| n.to_le_bytes()).collect();
            writer.atom(SIZES, &sizes);
        }
        if let Some(description) = &self.format_description {
            write_format_description(&mut writer, description);
        }
        for (kind, attachments) in [
            (PROPAGATED_ATTACHMENTS, &self.propagated_attachments),
            (OTHER_ATTACHMENTS, &self.other_attachments),
        ] {
            if let Some(attachments) = attachments {
                writer.begin(kind);
                write_dictionary(&mut writer, attachments, false);
                writer.end();
            }
        }
        if let Some(attachments) = &self.sample_attachments {
            writer.begin(SAMPLE_ATTACHMENTS);
            write_array(&mut writer, attachments);
            writer.end();
        }
        writer.end();
        writer.finish()
    }
}

/// Writes a property list leaf as an atom of `kind`.
fn write_property_list(writer: &mut AtomWriter, kind: FourCc, value: &Value) {
    writer.atom(kind, &plist::encode(value));
}

/// Writes a value by its type; `keep_empty` keeps an empty dictionary.
fn write_value(writer: &mut AtomWriter, value: &SerializedValue, keep_empty: bool) {
    match value {
        SerializedValue::Dictionary(entries) => write_dictionary(writer, entries, keep_empty),
        SerializedValue::Array(values) => write_array(writer, values),
        SerializedValue::PropertyList(leaf) => {
            let kind = match leaf {
                Value::Boolean(_) => BOOLEAN,
                Value::String(_) => STRING,
                Value::Data(_) => BYTES,
                Value::Date(_) => DATE,
                _ => NUMBER,
            };
            write_property_list(writer, kind, leaf);
        }
        SerializedValue::Null => {
            writer.atom(NULL, &[]);
        }
        SerializedValue::ColorSpace(leaf) => write_property_list(writer, COLOR_SPACE, leaf),
        SerializedValue::Url(url) => {
            write_property_list(writer, URL, &Value::String(url.clone()));
        }
        SerializedValue::FormatDescription(description) => {
            write_format_description(writer, description);
        }
    }
}

/// Writes a `dict`, or nothing where it is empty and `keep_empty` is false; its entries pass
/// `keep_empty` on to their values. A key the table knows is written by its index.
fn write_dictionary(
    writer: &mut AtomWriter,
    entries: &[(String, SerializedValue)],
    keep_empty: bool,
) {
    if entries.is_empty() && !keep_empty {
        return;
    }
    writer.begin(DICTIONARY);
    for (key, value) in entries {
        writer.begin(ENTRY);
        match known_keys::index_of(key) {
            Some(index) => {
                writer.atom(KEY_INDEX, &index.to_le_bytes());
            }
            None => write_property_list(writer, STRING, &Value::String(key.clone())),
        }
        write_value(writer, value, keep_empty);
        writer.end();
    }
    writer.end();
}

/// Writes an `aray`, or nothing where it is empty; its elements keep their empty dictionaries.
fn write_array(writer: &mut AtomWriter, values: &[SerializedValue]) {
    if values.is_empty() {
        return;
    }
    writer.begin(ARRAY);
    for value in values {
        write_value(writer, value, true);
    }
    writer.end();
}

/// Writes an `fdsc`.
fn write_format_description(writer: &mut AtomWriter, description: &SerializedFormatDescription) {
    writer.begin(FORMAT_DESCRIPTION);
    writer.atom(MEDIA_TYPE, &description.media_type.0.to_le_bytes());
    let mut dimensions = [0; 8];
    dimensions[..4].copy_from_slice(&description.width.to_le_bytes());
    dimensions[4..].copy_from_slice(&description.height.to_le_bytes());
    writer.atom(DIMENSIONS, &dimensions);
    writer.atom(CODEC, &description.codec.0.to_le_bytes());
    if let Some(extensions) = &description.extensions {
        writer.begin(EXTENSIONS);
        write_dictionary(writer, extensions, false);
        writer.end();
    }
    writer.end();
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::paravirt::atom::AtomWriter;

    /// A leaf of `kind` holding `value`'s binary property list.
    fn leaf(writer: &mut AtomWriter, kind: FourCc, value: &Value) {
        writer.atom(kind, &plist::encode(value));
    }

    /// An `avc1` format description of `width` by `height` whose extensions hold `record` as
    /// `SampleDescriptionExtensionAtoms` = {`avcC` = record}, its outer key by its index and its
    /// inner one as a string, as the serializer writes them.
    pub(crate) fn format_description(
        writer: &mut AtomWriter,
        width: i32,
        height: i32,
        record: &[u8],
    ) {
        writer.begin(FORMAT_DESCRIPTION);
        writer.atom(MEDIA_TYPE, &u32::from_be_bytes(*b"vide").to_le_bytes());
        let mut dimensions = width.to_le_bytes().to_vec();
        dimensions.extend_from_slice(&height.to_le_bytes());
        writer.atom(DIMENSIONS, &dimensions);
        writer.atom(CODEC, &u32::from_be_bytes(*b"avc1").to_le_bytes());
        writer.begin(EXTENSIONS).begin(DICTIONARY).begin(ENTRY);
        // SampleDescriptionExtensionAtoms by its index.
        writer.atom(KEY_INDEX, &49u16.to_le_bytes());
        writer.begin(DICTIONARY).begin(ENTRY);
        leaf(writer, STRING, &Value::string("avcC"));
        leaf(writer, BYTES, &Value::Data(record.to_vec()));
        writer.end().end().end().end().end().end();
    }

    /// A one-sample serialization of `data`, with a format description of `record` where given
    /// and the sample's attachments {`NotSync` = true} where `not_sync`.
    pub(crate) fn one_sample(data: &[u8], record: Option<&[u8]>, not_sync: bool) -> Vec<u8> {
        let mut writer = AtomWriter::new();
        writer.begin(SAMPLE_BUFFER);
        let timing = [0u8; TIMING_SIZE];
        writer.atom(TIMING, &timing);
        writer.atom(DATA, data);
        writer.atom(SAMPLE_COUNT, &1u32.to_le_bytes());
        writer.atom(SIZES, &(data.len() as u32).to_le_bytes());
        if let Some(record) = record {
            format_description(&mut writer, 64, 48, record);
        }
        if not_sync {
            writer
                .begin(SAMPLE_ATTACHMENTS)
                .begin(ARRAY)
                .begin(DICTIONARY)
                .begin(ENTRY);
            writer.atom(KEY_INDEX, &0u16.to_le_bytes());
            leaf(&mut writer, BOOLEAN, &Value::Boolean(true));
            writer.end().end().end().end();
        }
        writer.finish()
    }

    #[test]
    fn a_hand_written_sample_reads() -> Result<(), WireError> {
        assert_eq!(
            known_keys::key_at(49),
            Some("SampleDescriptionExtensionAtoms")
        );
        let record = [1, 0x64, 0, 0x0b, 0xff, 0xe0, 0];
        let sample =
            SampleBuffer::parse(&one_sample(&[0, 0, 0, 2, 0x65, 0x88], Some(&record), true))?;
        assert_eq!(sample.sample_count, 1);
        assert_eq!(sample.sizes, [6]);
        assert_eq!(sample.data.as_deref(), Some(&[0, 0, 0, 2, 0x65, 0x88][..]));
        assert_eq!(sample.timings.len(), 1);
        let format = sample
            .format_description
            .as_ref()
            .expect("a format description");
        assert_eq!(
            (format.codec, format.width, format.height),
            (FourCc::new(*b"avc1"), 64, 48)
        );
        assert_eq!(format.sample_description_atom("avcC"), Some(&record[..]));
        assert_eq!(
            sample.sample_attachment("NotSync"),
            Some(&SerializedValue::PropertyList(Value::Boolean(true)))
        );
        Ok(())
    }

    /// The writer names every key the table knows by its index, `avcC` among them, where the
    /// hand-written sample spells it: what it writes reads back as the same sample.
    #[test]
    fn a_sample_read_and_written_reads_back_the_same() -> Result<(), WireError> {
        let record = [1, 0x64, 0, 0x0b, 0xff, 0xe0, 0];
        let sample =
            SampleBuffer::parse(&one_sample(&[0, 0, 0, 2, 0x65, 0x88], Some(&record), true))?;
        assert_eq!(SampleBuffer::parse(&sample.serialize())?, sample);
        Ok(())
    }

    #[test]
    fn an_encoded_sample_written_reads_back() -> Result<(), WireError> {
        let time = |value| MediaTime {
            value,
            timescale: 30,
            flags: MediaTime::FLAG_VALID,
            epoch: 0,
        };
        let no = SerializedValue::PropertyList(Value::Boolean(false));
        let sample = SampleBuffer {
            sample_count: 1,
            timings: vec![SampleTiming {
                duration: time(1),
                presentation: time(4),
                decode: time(4),
            }],
            sizes: vec![6],
            data: Some(vec![0, 0, 0, 2, 0x65, 0x88]),
            output_presentation_time: Some(time(4)),
            sample_attachments: Some(vec![SerializedValue::Dictionary(vec![
                ("DependsOnOthers".to_owned(), no.clone()),
                ("EarlierDisplayTimesAllowed".to_owned(), no),
            ])]),
            ..SampleBuffer::default()
        };
        assert_eq!(SampleBuffer::parse(&sample.serialize())?, sample);
        Ok(())
    }

    #[test]
    fn data_without_a_count_is_one_sample() -> Result<(), WireError> {
        let mut writer = AtomWriter::new();
        writer.begin(SAMPLE_BUFFER).atom(DATA, &[1, 2, 3]);
        let sample = SampleBuffer::parse(&writer.finish())?;
        assert_eq!(sample.sample_count, 1);
        assert_eq!(sample.data.as_deref(), Some(&[1, 2, 3][..]));
        Ok(())
    }

    #[test]
    fn the_root_must_be_a_sample_buffer() {
        let mut writer = AtomWriter::new();
        writer.atom(DATA, &[1]);
        assert_eq!(
            SampleBuffer::parse(&writer.finish()),
            Err(WireError::Malformed("not a sample buffer"))
        );
    }

    #[test]
    fn a_second_timing_atom_is_malformed() {
        let mut writer = AtomWriter::new();
        writer.begin(SAMPLE_BUFFER);
        writer.atom(TIMING, &[0; TIMING_SIZE]);
        writer.atom(TIMING, &[0; TIMING_SIZE]);
        assert_eq!(
            SampleBuffer::parse(&writer.finish()),
            Err(WireError::Malformed("second timing atom"))
        );
    }

    #[test]
    fn values_past_the_nesting_bound_are_malformed() {
        let mut writer = AtomWriter::new();
        writer.begin(SAMPLE_BUFFER).begin(SAMPLE_ATTACHMENTS);
        for _ in 0..=MAX_NESTING {
            writer.begin(ARRAY);
        }
        writer.atom(NULL, &[0]);
        assert_eq!(
            SampleBuffer::parse(&writer.finish()),
            Err(WireError::Malformed("sample buffer values nest too deep"))
        );
    }
}
