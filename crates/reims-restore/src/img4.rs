//! Image4, the DER containers of Apple's secure boot: an IM4P holds a payload (a kernelcache, a
//! device tree, a trust cache), an IM4M is a manifest (a ticket TSS signs), and an IMG4 is a
//! payload stitched to the manifest that authorizes it, the form a Mac's Preboot volume holds its
//! boot objects in.
//!
//! The forms, as the tickets and objects this crate handles hold them:
//! - IM4P: `SEQUENCE { IA5String "IM4P", IA5String type, IA5String version, OCTET STRING
//!   payload, [OCTET STRING keybags], [SEQUENCE { INTEGER algorithm, INTEGER size }] }`.
//! - IM4M: `SEQUENCE { IA5String "IM4M", INTEGER 0, SET { MANB }, OCTET STRING signature,
//!   SEQUENCE { Certificate... } }`. The signature covers the `SET { MANB }` as encoded.
//! - MANB: `[PRIVATE 'MANB'] SEQUENCE { IA5String "MANB", SET { MANP, object... } }`, where MANP
//!   holds the manifest's own properties and each object (`krnl`, `dtre`, ...) the properties of
//!   one payload, its digest `DGST` among them.
//! - A property, or an object: `[PRIVATE 'abcd'] SEQUENCE { IA5String "abcd", value }`, its tag
//!   number the four characters read as a big-endian u32.
//! - IMG4: `SEQUENCE { IA5String "IMG4", IM4P, [0] { IM4M } }`.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail, ensure};
use sha2::{Digest, Sha384};

/// SHA-384, Image4's digest: of an IM4P (an object's `DGST`) and of a manifest (the name of the
/// boot directory it authorizes).
pub fn sha384(b: &[u8]) -> [u8; 48] {
    Sha384::digest(b).into()
}

/// An IM4P: a payload with its four-character type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Im4p {
    /// Its type, such as `krnl`.
    pub tag: [u8; 4],
    /// Its version string, such as the build of the kernel it holds.
    pub version: String,
    /// The payload as stored: compressed (LZFSE or LZSS) where the producer compressed it.
    pub payload: Vec<u8>,
    /// The compression the IM4P records for the payload, where it records one.
    pub compression: Option<Compression>,
    /// The keybags of an encrypted payload, as encoded; none for a plain one.
    pub keybags: Option<Vec<u8>>,
}

/// The compression an IM4P records: `SEQUENCE { INTEGER algorithm, INTEGER size }`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Compression {
    /// The algorithm: 1 is LZFSE.
    pub algorithm: u64,
    /// The payload's size once decompressed.
    pub size: u64,
}

/// Parses an IM4P.
pub fn parse_im4p(der: &[u8]) -> Result<Im4p> {
    let top = read_whole(der).context("IM4P")?;
    ensure!(
        top.is(Class::Universal, true, SEQUENCE),
        "an IM4P is not a SEQUENCE"
    );
    let mut it = Children::new(top.value);
    let magic = it.next_req("IM4P magic")?;
    ensure!(
        ia5(&magic)? == "IM4P",
        "not an IM4P (magic {:?})",
        ia5(&magic)?
    );
    let tag = it.next_req("IM4P type")?;
    let tag = ia5(&tag)?;
    let tag: [u8; 4] = tag
        .as_bytes()
        .try_into()
        .with_context(|| format!("IM4P type {tag:?} is not four characters"))?;
    let version = ia5(&it.next_req("IM4P version")?)?.to_owned();
    let payload = it.next_req("IM4P payload")?;
    ensure!(
        payload.is(Class::Universal, false, OCTET_STRING),
        "the IM4P's payload is not an OCTET STRING"
    );
    let mut im4p = Im4p {
        tag,
        version,
        payload: payload.value.to_vec(),
        compression: None,
        keybags: None,
    };
    while let Some(e) = it.next()? {
        if e.is(Class::Universal, false, OCTET_STRING)
            && im4p.keybags.is_none()
            && im4p.compression.is_none()
        {
            im4p.keybags = Some(e.value.to_vec());
        } else if e.is(Class::Universal, true, SEQUENCE) && im4p.compression.is_none() {
            let mut c = Children::new(e.value);
            let algorithm = uint(&c.next_req("compression algorithm")?)?;
            let size = uint(&c.next_req("decompressed size")?)?;
            ensure!(
                c.next()?.is_none(),
                "the IM4P's compression info has more than two elements"
            );
            im4p.compression = Some(Compression { algorithm, size });
        } else if e.class == Class::Context {
            // The payload's own properties (PAYP), which nothing here reads.
        } else {
            bail!(
                "the IM4P has an unexpected element (tag {} of class {:?})",
                e.tag,
                e.class
            );
        }
    }
    Ok(im4p)
}

/// An IMG4 of `im4p` and `im4m`, both kept byte for byte: `SEQUENCE { IA5String "IMG4", im4p,
/// [0] { im4m } }`, the layout of the boot objects Virtualization.framework's restore writes.
pub fn stitch_img4(im4p: &[u8], im4m: &[u8]) -> Vec<u8> {
    let mut manifest = Vec::new();
    put(&mut manifest, CONTEXT_CONSTRUCTED, 0, im4m);
    seq(&[&ia5_der("IMG4"), im4p, &manifest])
}

/// An IMG4's IM4P and IM4M (where it has one), as encoded.
pub fn split_img4(der: &[u8]) -> Result<(&[u8], Option<&[u8]>)> {
    let top = read_whole(der).context("IMG4")?;
    ensure!(
        top.is(Class::Universal, true, SEQUENCE),
        "an IMG4 is not a SEQUENCE"
    );
    let mut it = Children::new(top.value);
    ensure!(ia5(&it.next_req("IMG4 magic")?)? == "IMG4", "not an IMG4");
    let im4p = it.next_req("IMG4's IM4P")?;
    let mut im4m = None;
    while let Some(e) = it.next()? {
        if e.is(Class::Context, true, 0) {
            let inner = read_whole(e.value).context("the IMG4's manifest")?;
            im4m = Some(inner.raw);
        }
    }
    Ok((im4p.raw, im4m))
}

/// A sealed volume's root hashes, as an IPSW's root hash IM4P (`isys` for the system volume,
/// `csys` for a BaseSystem) carries them: a 16-byte header, then three 32-byte hashes, each
/// followed by 32 zero bytes, 208 bytes in all, on every release from 14.6.1 to 27.2.1 **[ipsw]**.
/// The first is the volume's root hash; a BaseSystem's other two are zero, and what a system
/// volume's other two are is not known yet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RootHash {
    pub hashes: [[u8; 32]; 3],
}

/// The header every root hash payload starts with **[ipsw]**.
const ROOT_HASH_HEADER: [u8; 16] = [2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0x20, 0, 0, 0];

impl RootHash {
    /// A payload's length.
    pub const LEN: usize = 16 + 3 * 64;

    /// Reads a root hash IM4P's payload.
    pub fn parse(payload: &[u8]) -> Result<Self> {
        ensure!(
            payload.len() == Self::LEN,
            "a root hash payload of {} bytes, not {}",
            payload.len(),
            Self::LEN
        );
        ensure!(
            payload[..16] == ROOT_HASH_HEADER,
            "a root hash payload whose header is {:02x?}",
            &payload[..16]
        );
        let mut hashes = [[0u8; 32]; 3];
        for (n, h) in hashes.iter_mut().enumerate() {
            let at = 16 + 64 * n;
            h.copy_from_slice(&payload[at..at + 32]);
            ensure!(
                payload[at + 32..at + 64].iter().all(|b| *b == 0),
                "root hash {n} isn't followed by zeros"
            );
        }
        Ok(Self { hashes })
    }

    /// The payload.
    pub fn payload(&self) -> Vec<u8> {
        let mut out = ROOT_HASH_HEADER.to_vec();
        for h in &self.hashes {
            out.extend_from_slice(h);
            out.extend_from_slice(&[0; 32]);
        }
        out
    }

    /// The system volume's root hash IM4P: tag `isys`, version `0`, as the IPSW's are
    /// **[ipsw]**.
    pub fn system_im4p(&self) -> Vec<u8> {
        encode_im4p(b"isys", "0", &self.payload())
    }

    /// The name of the sealed volume's snapshot: `com.apple.os.update-` and the root hash in
    /// upper-case hex **[vz]**.
    pub fn snapshot_name(&self) -> String {
        let hex: String = self.hashes[0].iter().map(|b| format!("{b:02X}")).collect();
        format!("com.apple.os.update-{hex}")
    }
}

/// The IM4P of a LocalPolicy: type `lpol`, version `1.0`, a payload of one zero byte. A
/// LocalPolicy is this IM4P stitched to the manifest that authorizes it, whose `lpol` digest is
/// this IM4P's SHA-384.
pub const LPOL_IM4P: [u8; 22] = [
    0x30, 0x14, 0x16, 0x04, b'I', b'M', b'4', b'P', 0x16, 0x04, b'l', b'p', b'o', b'l', 0x16, 0x03,
    b'1', b'.', b'0', 0x04, 0x01, 0x00,
];

/// The object a build manifest's component becomes in a ticket, for the components of the virtual
/// Mac's identities (matched by digest on 27.2.1's tickets) and the LocalPolicy.
pub fn component_tag(component: &str) -> Option<&'static str> {
    Some(match component {
        "Ap,BaseSystemTrustCache" => "bstc",
        "Ap,LocalPolicy" => "lpol",
        "Ap,SystemVolumeCanonicalMetadata" => "msys",
        "Ap,rOSLogo1" => "rlg1",
        "Ap,rOSLogo2" => "rlg2",
        "AppleLogo" => "logo",
        "BaseSystem" => "bsys",
        "BaseSystemVolume" => "csys",
        "DeviceTree" => "dtre",
        "KernelCache" => "krnl",
        "LLB" => "illb",
        "OS" => "rosi",
        "RecoveryMode" => "recm",
        "RestoreDeviceTree" => "rdtr",
        "RestoreKernelCache" => "rkrn",
        "RestoreLogo" => "rlgo",
        "RestoreRamDisk" => "rdsk",
        "RestoreTrustCache" => "rtsc",
        "StaticTrustCache" => "trst",
        "SystemVolume" => "isys",
        "iBEC" => "ibec",
        "iBSS" => "ibss",
        "iBoot" => "ibot",
        "Cryptex1,SystemOS" => "csos",
        "Cryptex1,SystemVolume" => "cssy",
        "Cryptex1,SystemTrustCache" => "trcs",
        "Cryptex1,AppOS" => "caos",
        "Cryptex1,AppVolume" => "casy",
        "Cryptex1,AppTrustCache" => "trca",
        "Cryptex1,RosettaOS" => "cros",
        "Cryptex1,RosettaVolume" => "crsy",
        "Cryptex1,RosettaTrustCache" => "trcr",
        _ => return None,
    })
}

/// The components whose IM4P a restore retags before stitching: the restore OS's copies of
/// payloads the IPSW stores once under the installed OS's type (the device tree's IM4P is `dtre`;
/// as `RestoreDeviceTree` it becomes `rdtr`). The build manifest's digest for such a component is
/// that of the retagged IM4P.
const RETAGGED: [&str; 4] = [
    "RestoreDeviceTree",
    "RestoreKernelCache",
    "RestoreLogo",
    "RestoreTrustCache",
];

/// `im4p` with its type replaced by `tag`, every other byte kept.
pub fn retag_im4p(im4p: &[u8], tag: [u8; 4]) -> Result<Vec<u8>> {
    let top = read_whole(im4p).context("IM4P")?;
    ensure!(
        top.is(Class::Universal, true, SEQUENCE),
        "an IM4P is not a SEQUENCE"
    );
    let mut it = Children::new(top.value);
    ensure!(ia5(&it.next_req("IM4P magic")?)? == "IM4P", "not an IM4P");
    let old = it.next_req("IM4P type")?;
    ensure!(
        ia5(&old)?.len() == 4,
        "the IM4P's type is not four characters"
    );
    let at = old.value.as_ptr() as usize - im4p.as_ptr() as usize;
    let mut out = im4p.to_vec();
    out[at..at + 4].copy_from_slice(&tag);
    Ok(out)
}

/// A component's IM4P personalized by a ticket, as a restore stitches the objects it installs:
/// retagged where the component is a restore OS's copy (`RestoreKernelCache` becomes `rkrn`, and
/// so on), then stitched to `im4m`. Fails unless the ticket's object for the component holds the
/// SHA-384 of the IM4P stitched.
pub fn personalize(component: &str, im4p: &[u8], im4m: &[u8]) -> Result<Vec<u8>> {
    let tag = component_tag(component);
    let im4p = match tag {
        Some(t) if RETAGGED.contains(&component) => {
            retag_im4p(im4p, t.as_bytes().try_into().expect("tags are four bytes"))?
        }
        _ => {
            parse_im4p(im4p).with_context(|| format!("{component}'s IM4P"))?;
            im4p.to_vec()
        }
    };
    let ticket = parse_im4m(im4m).context("the ticket")?;
    let digest = sha384(&im4p);
    let covered = match tag {
        Some(t) => {
            ticket
                .objects
                .get(t)
                .and_then(|o| o.get("DGST"))
                .and_then(Prop::as_data)
                == Some(&digest[..])
        }
        None => ticket
            .objects
            .values()
            .any(|o| o.get("DGST").and_then(Prop::as_data) == Some(&digest[..])),
    };
    ensure!(
        covered,
        "the ticket doesn't authorize this IM4P as {component} (no object holds its digest)"
    );
    Ok(stitch_img4(&im4p, im4m))
}

/// A manifest property's value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Prop {
    /// An INTEGER, such as `ECID` or `CHIP`.
    Int(u64),
    /// A BOOLEAN, such as `CPRO` or `EPRO`.
    Bool(bool),
    /// An OCTET STRING, such as `DGST`, `BNCH` or `love` (whose bytes are text).
    Data(Vec<u8>),
    /// An IA5String or UTF8String.
    Str(String),
}

impl Prop {
    /// The integer, if it is one.
    pub fn as_int(&self) -> Option<u64> {
        match self {
            Prop::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// The boolean, if it is one.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Prop::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// The bytes, if it is data.
    pub fn as_data(&self) -> Option<&[u8]> {
        match self {
            Prop::Data(v) => Some(v),
            _ => None,
        }
    }
}

/// A parsed IM4M.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Im4m {
    /// MANP: the manifest's own properties, by four-character name.
    pub props: BTreeMap<String, Prop>,
    /// Every other entry of MANB: the objects (`krnl`, `dtre`, ...) and their properties.
    pub objects: BTreeMap<String, BTreeMap<String, Prop>>,
    /// The `SET { MANB }` as encoded: the bytes the signature covers.
    pub signed: Vec<u8>,
    /// The signature, as stored; empty where the manifest has none.
    pub signature: Vec<u8>,
    /// The certificates of its chain, each as encoded, in the order stored.
    pub certs: Vec<Vec<u8>>,
}

/// Parses an IM4M.
pub fn parse_im4m(der: &[u8]) -> Result<Im4m> {
    let top = read_whole(der).context("IM4M")?;
    ensure!(
        top.is(Class::Universal, true, SEQUENCE),
        "an IM4M is not a SEQUENCE"
    );
    let mut it = Children::new(top.value);
    ensure!(ia5(&it.next_req("IM4M magic")?)? == "IM4M", "not an IM4M");
    let version = uint(&it.next_req("IM4M version")?)?;
    ensure!(version == 0, "IM4M version {version}, not 0");
    let body = it.next_req("IM4M body")?;
    ensure!(
        body.is(Class::Universal, true, SET),
        "the IM4M's body is not a SET"
    );
    let mut im4m = Im4m {
        props: BTreeMap::new(),
        objects: BTreeMap::new(),
        signed: body.raw.to_vec(),
        signature: Vec::new(),
        certs: Vec::new(),
    };

    let mut manbs = Children::new(body.value);
    let manb = manbs.next_req("MANB")?;
    ensure!(
        manbs.next()?.is_none(),
        "the IM4M's body holds more than MANB"
    );
    let entries = named(&manb, "MANB")?;
    ensure!(
        entries.is(Class::Universal, true, SET),
        "MANB's entries are not a SET"
    );
    let mut it_entries = Children::new(entries.value);
    while let Some(entry) = it_entries.next()? {
        let name = fourcc_of(&entry)?;
        let props =
            props_of(&named(&entry, &name)?).with_context(|| format!("MANB entry {name}"))?;
        if name == "MANP" {
            ensure!(im4m.props.is_empty(), "MANB holds MANP twice");
            im4m.props = props;
        } else {
            ensure!(
                im4m.objects.insert(name.clone(), props).is_none(),
                "MANB holds {name} twice"
            );
        }
    }

    if let Some(sig) = it.next()? {
        ensure!(
            sig.is(Class::Universal, false, OCTET_STRING),
            "the IM4M's signature is not an OCTET STRING"
        );
        im4m.signature = sig.value.to_vec();
        if let Some(chain) = it.next()? {
            ensure!(
                chain.is(Class::Universal, true, SEQUENCE),
                "the IM4M's certificate chain is not a SEQUENCE"
            );
            let mut certs = Children::new(chain.value);
            while let Some(c) = certs.next()? {
                im4m.certs.push(c.raw.to_vec());
            }
        }
    }
    ensure!(
        it.next()?.is_none(),
        "the IM4M has elements past its certificate chain"
    );
    Ok(im4m)
}

/// A MANB `SET` with `props` as MANP and each of `objects`: the bytes a manifest's signature
/// covers. DER orders a SET's elements by their encoding, which for four-character names is their
/// byte order.
pub fn encode_manb(
    props: &BTreeMap<String, Prop>,
    objects: &BTreeMap<String, BTreeMap<String, Prop>>,
) -> Vec<u8> {
    let mut entries = vec![named_entry("MANP", &encode_props(props))];
    entries.extend(
        objects
            .iter()
            .map(|(name, p)| named_entry(name, &encode_props(p))),
    );
    let manb = named_entry("MANB", &set_of(entries));
    set_of(vec![manb])
}

/// An IM4M of an encoded MANB `SET`, its signature, and its certificate chain (a `SEQUENCE` of
/// certificates, as encoded).
pub fn encode_im4m(manb_set: &[u8], signature: &[u8], chain: &[u8]) -> Vec<u8> {
    let mut sig = Vec::new();
    put(&mut sig, UNIVERSAL, OCTET_STRING, signature);
    seq(&[&ia5_der("IM4M"), &uint_der(0), manb_set, &sig, chain])
}

/// An IM4P of `tag`, `version` and `payload`, with no compression or keybags.
pub fn encode_im4p(tag: &[u8; 4], version: &str, payload: &[u8]) -> Vec<u8> {
    let tag = std::str::from_utf8(tag).unwrap_or("????");
    let mut data = Vec::new();
    put(&mut data, UNIVERSAL, OCTET_STRING, payload);
    seq(&[&ia5_der("IM4P"), &ia5_der(tag), &ia5_der(version), &data])
}

fn encode_props(props: &BTreeMap<String, Prop>) -> Vec<u8> {
    set_of(
        props
            .iter()
            .map(|(name, value)| named_entry(name, &prop_der(value)))
            .collect(),
    )
}

/// `[PRIVATE 'name'] SEQUENCE { IA5String name, value }`.
fn named_entry(name: &str, value: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    put(
        &mut out,
        PRIVATE_CONSTRUCTED,
        fourcc_tag(name),
        &seq(&[&ia5_der(name), value]),
    );
    out
}

fn prop_der(p: &Prop) -> Vec<u8> {
    let mut out = Vec::new();
    match p {
        Prop::Int(v) => return uint_der(*v),
        Prop::Bool(b) => put(&mut out, UNIVERSAL, BOOLEAN, &[if *b { 0xff } else { 0 }]),
        Prop::Data(d) => put(&mut out, UNIVERSAL, OCTET_STRING, d),
        Prop::Str(s) => return ia5_der(s),
    }
    out
}

/// A four-character name as a tag number: its bytes, big-endian.
fn fourcc_tag(name: &str) -> u64 {
    name.bytes().fold(0u64, |acc, b| acc << 8 | u64::from(b))
}

/// The name of a `[PRIVATE 'name'] SEQUENCE { IA5String name, ... }`, checked against its tag.
fn fourcc_of(e: &Tlv<'_>) -> Result<String> {
    ensure!(
        e.class == Class::Private && e.constructed,
        "a manifest entry is not a constructed PRIVATE element"
    );
    let inner = read_whole(e.value)?;
    ensure!(
        inner.is(Class::Universal, true, SEQUENCE),
        "a manifest entry does not hold a SEQUENCE"
    );
    let name = ia5(&Children::new(inner.value).next_req("entry name")?)?.to_owned();
    ensure!(
        name.len() == 4 && fourcc_tag(&name) == e.tag,
        "manifest entry {name:?} has tag {:#x}",
        e.tag
    );
    Ok(name)
}

/// The value of the entry `[PRIVATE 'name'] SEQUENCE { IA5String name, value }`.
fn named<'a>(e: &Tlv<'a>, name: &str) -> Result<Tlv<'a>> {
    ensure!(fourcc_of(e)? == name, "expected {name}");
    let inner = read_whole(e.value)?;
    let mut it = Children::new(inner.value);
    it.next_req("entry name")?;
    let value = it.next_req("entry value")?;
    ensure!(
        it.next()?.is_none(),
        "entry {name} has more than a name and a value"
    );
    Ok(value)
}

/// The properties in a `SET` of entries.
fn props_of(set: &Tlv<'_>) -> Result<BTreeMap<String, Prop>> {
    ensure!(
        set.is(Class::Universal, true, SET),
        "properties are not a SET"
    );
    let mut props = BTreeMap::new();
    let mut it = Children::new(set.value);
    while let Some(e) = it.next()? {
        let name = fourcc_of(&e)?;
        let v = named(&e, &name)?;
        let value = match (v.class, v.constructed, v.tag) {
            (Class::Universal, false, BOOLEAN) => {
                ensure!(
                    v.value.len() == 1,
                    "property {name}: a BOOLEAN of {} bytes",
                    v.value.len()
                );
                Prop::Bool(v.value[0] != 0)
            }
            (Class::Universal, false, INTEGER) => {
                Prop::Int(uint(&v).with_context(|| format!("property {name}"))?)
            }
            (Class::Universal, false, OCTET_STRING) => Prop::Data(v.value.to_vec()),
            (Class::Universal, false, IA5_STRING | UTF8_STRING) => Prop::Str(
                std::str::from_utf8(v.value)
                    .with_context(|| format!("property {name}: not text"))?
                    .to_owned(),
            ),
            _ => bail!(
                "property {name} has a value of tag {} (class {:?}), which manifests don't use",
                v.tag,
                v.class
            ),
        };
        ensure!(
            props.insert(name.clone(), value).is_none(),
            "property {name} appears twice"
        );
    }
    Ok(props)
}

// DER, as far as Image4 and the certificates of its chains use it.

pub(crate) const BOOLEAN: u64 = 1;
pub(crate) const INTEGER: u64 = 2;
pub(crate) const OCTET_STRING: u64 = 4;
pub(crate) const UTF8_STRING: u64 = 12;
pub(crate) const SEQUENCE: u64 = 16;
pub(crate) const SET: u64 = 17;
pub(crate) const IA5_STRING: u64 = 22;

/// The identifier octet's class and constructed bits.
pub(crate) const UNIVERSAL: u8 = 0x00;
pub(crate) const UNIVERSAL_CONSTRUCTED: u8 = 0x20;
pub(crate) const CONTEXT_CONSTRUCTED: u8 = 0xa0;
pub(crate) const PRIVATE_CONSTRUCTED: u8 = 0xe0;

/// An element's class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Class {
    Universal,
    Application,
    Context,
    Private,
}

/// One DER element.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Tlv<'a> {
    pub class: Class,
    pub constructed: bool,
    pub tag: u64,
    /// Its contents.
    pub value: &'a [u8],
    /// Its whole encoding: identifier, length and contents.
    pub raw: &'a [u8],
}

impl Tlv<'_> {
    fn is(&self, class: Class, constructed: bool, tag: u64) -> bool {
        self.class == class && self.constructed == constructed && self.tag == tag
    }
}

/// Reads the element at the front of `input`; returns it and what follows it.
pub(crate) fn read_tlv(input: &[u8]) -> Result<(Tlv<'_>, &[u8])> {
    let mut i = 0;
    let first = *input
        .first()
        .context("DER: an element expected, the input ended")?;
    i += 1;
    let class = match first >> 6 {
        0 => Class::Universal,
        1 => Class::Application,
        2 => Class::Context,
        _ => Class::Private,
    };
    let constructed = first & 0x20 != 0;
    let mut tag = u64::from(first & 0x1f);
    if tag == 0x1f {
        tag = 0;
        loop {
            let b = *input.get(i).context("DER: the input ended inside a tag")?;
            i += 1;
            ensure!(tag >> 57 == 0, "DER: a tag number past 64 bits");
            tag = tag << 7 | u64::from(b & 0x7f);
            if b & 0x80 == 0 {
                break;
            }
        }
    }
    let l = *input
        .get(i)
        .context("DER: the input ended before a length")?;
    i += 1;
    let len = if l < 0x80 {
        usize::from(l)
    } else {
        let n = usize::from(l & 0x7f);
        ensure!(n != 0, "DER: an indefinite length");
        ensure!(n <= 4, "DER: a length of {n} bytes");
        let bytes = input
            .get(i..i + n)
            .context("DER: the input ended inside a length")?;
        i += n;
        bytes
            .iter()
            .fold(0usize, |acc, &b| acc << 8 | usize::from(b))
    };
    let end = i
        .checked_add(len)
        .filter(|&e| e <= input.len())
        .with_context(|| {
            format!(
                "DER: an element of {len} bytes where {} remain",
                input.len() - i.min(input.len())
            )
        })?;
    Ok((
        Tlv {
            class,
            constructed,
            tag,
            value: &input[i..end],
            raw: &input[..end],
        },
        &input[end..],
    ))
}

/// Reads an element that must span all of `input`.
pub(crate) fn read_whole(input: &[u8]) -> Result<Tlv<'_>> {
    let (e, rest) = read_tlv(input)?;
    ensure!(
        rest.is_empty(),
        "DER: {} bytes past the element's end",
        rest.len()
    );
    Ok(e)
}

/// The elements of a constructed element's contents, in order.
pub(crate) struct Children<'a> {
    rest: &'a [u8],
}

impl<'a> Children<'a> {
    pub(crate) fn new(contents: &'a [u8]) -> Self {
        Self { rest: contents }
    }

    /// The next element, or `None` at the end.
    #[allow(clippy::should_implement_trait)]
    pub(crate) fn next(&mut self) -> Result<Option<Tlv<'a>>> {
        if self.rest.is_empty() {
            return Ok(None);
        }
        let (e, rest) = read_tlv(self.rest)?;
        self.rest = rest;
        Ok(Some(e))
    }

    /// The next element, which must exist.
    pub(crate) fn next_req(&mut self, what: &str) -> Result<Tlv<'a>> {
        self.next()?.with_context(|| format!("DER: {what} missing"))
    }
}

/// The text of an IA5String.
pub(crate) fn ia5<'a>(e: &Tlv<'a>) -> Result<&'a str> {
    ensure!(
        e.is(Class::Universal, false, IA5_STRING),
        "DER: an IA5String expected, tag {} found",
        e.tag
    );
    ensure!(e.value.is_ascii(), "DER: an IA5String that is not ASCII");
    Ok(std::str::from_utf8(e.value)?)
}

/// A non-negative INTEGER of at most 64 bits.
pub(crate) fn uint(e: &Tlv<'_>) -> Result<u64> {
    ensure!(
        e.is(Class::Universal, false, INTEGER),
        "DER: an INTEGER expected, tag {} found",
        e.tag
    );
    let v = e.value;
    ensure!(!v.is_empty(), "DER: an empty INTEGER");
    ensure!(v[0] & 0x80 == 0, "DER: a negative INTEGER");
    let v = if v.len() > 1 && v[0] == 0 { &v[1..] } else { v };
    ensure!(v.len() <= 8, "DER: an INTEGER past 64 bits");
    Ok(v.iter().fold(0u64, |acc, &b| acc << 8 | u64::from(b)))
}

/// Appends an element of `class_bits` (class and constructed bits), `tag` and `value`.
pub(crate) fn put(out: &mut Vec<u8>, class_bits: u8, tag: u64, value: &[u8]) {
    if tag < 0x1f {
        out.push(class_bits | tag as u8);
    } else {
        out.push(class_bits | 0x1f);
        let groups = (64 - tag.leading_zeros()).div_ceil(7);
        for g in (0..groups).rev() {
            let b = (tag >> (7 * g)) as u8 & 0x7f;
            out.push(if g == 0 { b } else { b | 0x80 });
        }
    }
    let len = value.len();
    if len < 0x80 {
        out.push(len as u8);
    } else {
        let bytes = len.to_be_bytes();
        let skip = bytes.iter().take_while(|&&b| b == 0).count();
        out.push(0x80 | (bytes.len() - skip) as u8);
        out.extend_from_slice(&bytes[skip..]);
    }
    out.extend_from_slice(value);
}

/// `SEQUENCE { parts... }`.
pub(crate) fn seq(parts: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::new();
    put(&mut out, UNIVERSAL_CONSTRUCTED, SEQUENCE, &parts.concat());
    out
}

/// `SET { elements... }`, the elements in DER's order: by their encodings.
pub(crate) fn set_of(mut elements: Vec<Vec<u8>>) -> Vec<u8> {
    elements.sort();
    let mut out = Vec::new();
    put(&mut out, UNIVERSAL_CONSTRUCTED, SET, &elements.concat());
    out
}

/// An IA5String.
pub(crate) fn ia5_der(s: &str) -> Vec<u8> {
    let mut out = Vec::new();
    put(&mut out, UNIVERSAL, IA5_STRING, s.as_bytes());
    out
}

/// A non-negative INTEGER in its shortest form.
pub(crate) fn uint_der(v: u64) -> Vec<u8> {
    let bytes = v.to_be_bytes();
    let skip = bytes.iter().take_while(|&&b| b == 0).count().min(7);
    let mut value = Vec::with_capacity(9);
    if bytes[skip] & 0x80 != 0 {
        value.push(0);
    }
    value.extend_from_slice(&bytes[skip..]);
    let mut out = Vec::new();
    put(&mut out, UNIVERSAL, INTEGER, &value);
    out
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn a_root_hash_payload_reads_back() {
        let r = RootHash {
            hashes: [[0xab; 32], [1; 32], [0; 32]],
        };
        let p = r.payload();
        assert_eq!(p.len(), RootHash::LEN);
        assert_eq!(&p[..16], &ROOT_HASH_HEADER);
        assert_eq!(RootHash::parse(&p).unwrap(), r);
        let im4p = parse_im4p(&r.system_im4p()).unwrap();
        assert_eq!(&im4p.tag, b"isys");
        assert_eq!(im4p.payload, p);
        assert!(r.snapshot_name().starts_with("com.apple.os.update-ABABAB"));
        assert_eq!(r.snapshot_name().len(), 20 + 64);
        let mut bad = p.clone();
        bad[16 + 40] = 1;
        assert!(RootHash::parse(&bad).is_err());
        assert!(RootHash::parse(&p[..200]).is_err());
    }

    /// A manifest written from scratch, shaped like a TSS ticket: MANP and two objects.
    pub fn synthetic_ticket() -> (
        BTreeMap<String, Prop>,
        BTreeMap<String, BTreeMap<String, Prop>>,
    ) {
        let mut props = BTreeMap::new();
        props.insert("ECID".to_owned(), Prop::Int(0x8123_4567_89ab_cdef));
        props.insert("CHIP".to_owned(), Prop::Int(0xfe00));
        props.insert("BORD".to_owned(), Prop::Int(0x20));
        props.insert("CPRO".to_owned(), Prop::Bool(true));
        props.insert("BNCH".to_owned(), Prop::Data(vec![7; 32]));
        props.insert("love".to_owned(), Prop::Data(b"99.1.0.0.0,0".to_vec()));
        let mut objects = BTreeMap::new();
        for (name, d) in [("krnl", 1u8), ("dtre", 2)] {
            let mut o = BTreeMap::new();
            o.insert("DGST".to_owned(), Prop::Data(vec![d; 48]));
            o.insert("EPRO".to_owned(), Prop::Bool(true));
            objects.insert(name.to_owned(), o);
        }
        (props, objects)
    }

    #[test]
    fn manifests_round_trip() {
        let (props, objects) = synthetic_ticket();
        let manb = encode_manb(&props, &objects);
        let chain = seq(&[&seq(&[&uint_der(1)]), &seq(&[&uint_der(2)])]);
        let der = encode_im4m(&manb, &[0x55; 64], &chain);
        let m = parse_im4m(&der).unwrap();
        assert_eq!(m.props, props);
        assert_eq!(m.objects, objects);
        assert_eq!(m.signed, manb);
        assert_eq!(m.signature, vec![0x55; 64]);
        assert_eq!(m.certs, vec![seq(&[&uint_der(1)]), seq(&[&uint_der(2)])]);
        assert_eq!(m.props["ECID"].as_int(), Some(0x8123_4567_89ab_cdef));
        assert_eq!(m.objects["krnl"]["DGST"].as_data(), Some(&[1u8; 48][..]));
    }

    #[test]
    fn manifest_entries_are_in_ders_order() {
        let (props, objects) = synthetic_ticket();
        let m = encode_manb(&props, &objects);
        // MANP, then the objects by name; MANP's properties by name, upper case first.
        let text: Vec<u8> = m.iter().copied().filter(u8::is_ascii_alphabetic).collect();
        let text = String::from_utf8(text).unwrap();
        let at = |s: &str| text.find(s).unwrap();
        assert!(at("MANP") < at("dtre") && at("dtre") < at("krnl"));
        assert!(at("BNCH") < at("BORD") && at("BORD") < at("CHIP") && at("ECID") < at("love"));
    }

    #[test]
    fn private_tags_encode_their_four_characters() {
        // 'MANP' = 0x4d414e50, in base 128: 04 6a 05 1c 50.
        let e = named_entry("MANP", &uint_der(1));
        assert_eq!(&e[..6], &[0xff, 0x84, 0xea, 0x85, 0x9c, 0x50]);
        let (t, rest) = read_tlv(&e).unwrap();
        assert!(rest.is_empty());
        assert_eq!(
            (t.class, t.constructed, t.tag),
            (Class::Private, true, 0x4d41_4e50)
        );
    }

    #[test]
    fn integers_take_their_shortest_form() {
        assert_eq!(uint_der(0), [2, 1, 0]);
        assert_eq!(uint_der(0x20), [2, 1, 0x20]);
        assert_eq!(uint_der(0xfe00), [2, 3, 0, 0xfe, 0]);
        assert_eq!(
            uint_der(u64::MAX),
            [2, 9, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]
        );
        for v in [0, 1, 0x7f, 0x80, 0xfe00, 1 << 63, u64::MAX] {
            assert_eq!(uint(&read_whole(&uint_der(v)).unwrap()).unwrap(), v);
        }
        assert!(
            uint(&read_whole(&[2, 1, 0x80]).unwrap()).is_err(),
            "negative"
        );
    }

    #[test]
    fn long_lengths_round_trip() {
        let payload = vec![0xab; 70_000];
        let im4p = encode_im4p(b"krnl", "1.0", &payload);
        assert_eq!(&im4p[..4], &[0x30, 0x83, 0x01, 0x11]);
        let p = parse_im4p(&im4p).unwrap();
        assert_eq!(p.tag, *b"krnl");
        assert_eq!(p.version, "1.0");
        assert_eq!(p.payload, payload);
        assert_eq!(p.compression, None);
    }

    #[test]
    fn im4p_compression_and_keybags_are_read() {
        let mut data = Vec::new();
        put(&mut data, UNIVERSAL, OCTET_STRING, b"bvx2....");
        let mut kbag = Vec::new();
        put(&mut kbag, UNIVERSAL, OCTET_STRING, &[9; 10]);
        let compression = seq(&[&uint_der(1), &uint_der(123_456)]);
        let der = seq(&[
            &ia5_der("IM4P"),
            &ia5_der("rkrn"),
            &ia5_der("v2"),
            &data,
            &kbag,
            &compression,
        ]);
        let p = parse_im4p(&der).unwrap();
        assert_eq!(p.payload, b"bvx2....");
        assert_eq!(p.keybags.as_deref(), Some(&[9u8; 10][..]));
        assert_eq!(
            p.compression,
            Some(Compression {
                algorithm: 1,
                size: 123_456
            })
        );
    }

    #[test]
    fn bad_im4ps_are_refused() {
        let mut data = Vec::new();
        put(&mut data, UNIVERSAL, OCTET_STRING, b"x");
        let good = seq(&[&ia5_der("IM4P"), &ia5_der("krnl"), &ia5_der("1"), &data]);
        assert!(parse_im4p(&good).is_ok());
        assert!(parse_im4p(&good[..good.len() - 1]).is_err(), "truncated");
        assert!(
            parse_im4p(&seq(&[
                &ia5_der("IM4M"),
                &ia5_der("krnl"),
                &ia5_der("1"),
                &data
            ]))
            .is_err()
        );
        assert!(
            parse_im4p(&seq(&[
                &ia5_der("IM4P"),
                &ia5_der("krnl!"),
                &ia5_der("1"),
                &data
            ]))
            .is_err()
        );
        let mut trailing = good.clone();
        trailing.push(0);
        assert!(parse_im4p(&trailing).is_err(), "bytes past the end");
        assert!(
            parse_im4p(&[0x30, 0x80, 0, 0]).is_err(),
            "indefinite length"
        );
    }

    #[test]
    fn stitching_keeps_both_parts_verbatim() {
        let im4p = encode_im4p(b"lpol", "1.0", &[0]);
        let (props, objects) = synthetic_ticket();
        let im4m = encode_im4m(&encode_manb(&props, &objects), &[1; 8], &seq(&[]));
        let img4 = stitch_img4(&im4p, &im4m);
        // SEQUENCE { "IMG4", im4p, [0] { im4m } }.
        let top = read_whole(&img4).unwrap();
        let mut it = Children::new(top.value);
        assert_eq!(ia5(&it.next().unwrap().unwrap()).unwrap(), "IMG4");
        assert_eq!(it.next().unwrap().unwrap().raw, &im4p[..]);
        let wrapped = it.next().unwrap().unwrap();
        assert_eq!(
            (wrapped.class, wrapped.constructed, wrapped.tag),
            (Class::Context, true, 0)
        );
        assert_eq!(wrapped.value, &im4m[..]);
        assert_eq!(split_img4(&img4).unwrap(), (&im4p[..], Some(&im4m[..])));
        assert_eq!(
            &im4p,
            &[
                0x30, 0x14, 0x16, 4, b'I', b'M', b'4', b'P', 0x16, 4, b'l', b'p', b'o', b'l', 0x16,
                3, b'1', b'.', b'0', 4, 1, 0
            ]
        );
    }

    #[test]
    fn manifests_with_bad_entries_are_refused() {
        // An entry whose tag doesn't match its name.
        let mut wrong = Vec::new();
        put(
            &mut wrong,
            PRIVATE_CONSTRUCTED,
            fourcc_tag("ECID"),
            &seq(&[&ia5_der("CHIP"), &uint_der(1)]),
        );
        let manp = named_entry("MANP", &set_of(vec![wrong]));
        let manb = set_of(vec![named_entry("MANB", &set_of(vec![manp]))]);
        assert!(parse_im4m(&encode_im4m(&manb, &[], &seq(&[]))).is_err());
        // A duplicated property.
        let p = named_entry("ECID", &uint_der(1));
        let mut dup = Vec::new();
        put(
            &mut dup,
            UNIVERSAL_CONSTRUCTED,
            SET,
            &[p.clone(), p].concat(),
        );
        let manb = set_of(vec![named_entry(
            "MANB",
            &set_of(vec![named_entry("MANP", &dup)]),
        )]);
        assert!(parse_im4m(&encode_im4m(&manb, &[], &seq(&[]))).is_err());
    }

    #[test]
    fn the_local_policys_im4p_is_lpol_1_0_with_one_zero_byte() {
        assert_eq!(LPOL_IM4P.to_vec(), encode_im4p(b"lpol", "1.0", &[0]));
    }

    /// A ticket whose objects hold the digests of `im4ps`, by tag.
    fn ticket_of(im4ps: &[(&str, &[u8])]) -> Vec<u8> {
        let mut objects = BTreeMap::new();
        for (tag, im4p) in im4ps {
            let mut o = BTreeMap::new();
            o.insert("DGST".to_owned(), Prop::Data(sha384(im4p).to_vec()));
            objects.insert((*tag).to_owned(), o);
        }
        encode_im4m(&encode_manb(&BTreeMap::new(), &objects), &[1; 8], &seq(&[]))
    }

    #[test]
    fn restore_copies_are_retagged_before_stitching() {
        let dt = encode_im4p(b"dtre", "EmbeddedDeviceTrees-1", b"payload");
        let rdtr = retag_im4p(&dt, *b"rdtr").unwrap();
        assert_eq!(rdtr.len(), dt.len());
        assert_eq!(parse_im4p(&rdtr).unwrap().tag, *b"rdtr");
        assert_eq!(parse_im4p(&rdtr).unwrap().payload, b"payload");

        let ticket = ticket_of(&[("dtre", &dt), ("rdtr", &rdtr)]);
        let img4 = personalize("RestoreDeviceTree", &dt, &ticket).unwrap();
        assert_eq!(split_img4(&img4).unwrap(), (&rdtr[..], Some(&ticket[..])));
        let img4 = personalize("DeviceTree", &dt, &ticket).unwrap();
        assert_eq!(split_img4(&img4).unwrap().0, &dt[..]);
        // A ticket that doesn't cover the payload.
        let other = ticket_of(&[("dtre", &rdtr)]);
        assert!(personalize("DeviceTree", &dt, &other).is_err());
        assert!(personalize("RestoreDeviceTree", &dt, &ticket_of(&[("dtre", &dt)])).is_err());
        // A component the table doesn't name is found by digest.
        let new = encode_im4p(b"newt", "1", b"x");
        assert!(personalize("NewThing", &new, &ticket_of(&[("newt", &new)])).is_ok());
    }

    #[test]
    fn sha384_is_sha384() {
        // FIPS 180-2's "abc".
        let d = sha384(b"abc");
        assert_eq!(d[..8], [0xcb, 0x00, 0x75, 0x3f, 0x45, 0xa3, 0x5e, 0x8b]);
        assert_eq!(d[32..40], [0x80, 0x86, 0x07, 0x2b, 0xa1, 0xe7, 0xcc, 0x23]);
    }
}
