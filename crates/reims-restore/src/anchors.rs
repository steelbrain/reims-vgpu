//! The LocalPolicy's trust anchors, swapped for the project's (docs/design.md, "The
//! LocalPolicy"). A LocalPolicy's manifest carries a chain, leaf and intermediate, which the
//! guest's kernel (AppleImage4) anchors on the img4 test root embedded in kexts of its boot
//! kernel collection (the boot policy kext and AppleImage4, and on 27 Image4 too); the boot
//! policy kext also holds the intermediate and leaf it expects, and a copy of the leaf's public
//! key. The restore builds the project's own chain — root, intermediate and leaf, each exactly
//! as long as the guest's certificate it stands in for, since the kexts keep the anchors as one
//! run of DER walked by length — signs the LocalPolicies with its leaf, and has Ambrosia replace
//! the guest's anchors with it in memory at boot (`Kernel > Patch` entries of its configuration,
//! one per anchor in each kext). The kernel's other roots, which anchor Cryptex1 tickets and
//! firmware, are left as they are.
//!
//! Everything taken from the guest is read from the IPSW's own kernel collection at install
//! time ([`crate::kernelcache`]): the certificates, the key, and each certificate's img4
//! extension (OID 1.2.840.113635.100.6.1.15), which the project's carry over verbatim. They are
//! found by their shape, not by their bytes: the one chain, a self-signed root and two
//! certificates under it that carry the extension, in the boot policy kext's constants; and the
//! root in each other entry that holds it. Every count a patch relies on is checked, so a release
//! that holds them otherwise fails the restore rather than the boot.

use anyhow::{Context, Result, bail, ensure};
use p384::ecdsa::SigningKey;
use p384::ecdsa::signature::Signer as _;
use sha2::Digest as _;

use crate::img4::{
    BOOLEAN, CONTEXT_CONSTRUCTED, Children, Class, INTEGER, OCTET_STRING, SEQUENCE, SET, Tlv,
    UNIVERSAL, UNIVERSAL_CONSTRUCTED, UTF8_STRING, put, read_tlv, read_whole, seq,
};
use crate::kernelcache::FileSet;

/// The kext whose boot policy checks a LocalPolicy, which holds the whole chain.
pub const BOOT_POLICY: &str = "com.apple.security.AppleVPBootPolicy";
/// The kext that verifies a LocalPolicy's chain against the img4 test root, which it embeds.
pub const APPLE_IMAGE4: &str = "com.apple.security.AppleImage4";

const BIT_STRING: u64 = 3;
const OBJECT_IDENTIFIER: u64 = 6;
const UTC_TIME: u64 = 23;

/// 1.2.840.113635.100.6.1.15, the img4 extension.
const OID_IMG4: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x63, 0x64, 0x06, 0x01, 0x0f];
/// 2.5.29.19, 2.5.29.15, 2.5.29.14, 2.5.29.35: basic constraints, key usage, subject and
/// authority key identifiers.
const OID_BASIC_CONSTRAINTS: &[u8] = &[0x55, 0x1d, 0x13];
const OID_KEY_USAGE: &[u8] = &[0x55, 0x1d, 0x0f];
const OID_SKI: &[u8] = &[0x55, 0x1d, 0x0e];
const OID_AKI: &[u8] = &[0x55, 0x1d, 0x23];
/// A private extension that takes up the length the project's certificates differ by: a
/// UUID-derived OID (2.25.<uuid>, ITU-T X.667) of the project's, never critical, so a verifier
/// passes over it.
const OID_FILLER: &[u8] = &[
    0x69, 0x83, 0xc8, 0x84, 0xbc, 0x94, 0xe5, 0x8c, 0xda, 0x92, 0xed, 0xab, 0xe7, 0x92, 0xaf, 0xd5,
    0xf8, 0x9a, 0xed, 0x68,
];
/// 2.5.4.10 and 2.5.4.3: organization and common name.
const OID_O: &[u8] = &[0x55, 0x04, 0x0a];
const OID_CN: &[u8] = &[0x55, 0x04, 0x03];
/// 1.2.840.10045.2.1, 1.3.132.0.34, 1.2.840.10045.4.3.3: an EC key, P-384, ECDSA with
/// SHA-384.
const OID_EC_KEY: &[u8] = &[0x2a, 0x86, 0x48, 0xce, 0x3d, 0x02, 0x01];
const OID_P384: &[u8] = &[0x2b, 0x81, 0x04, 0x00, 0x22];
const OID_ECDSA_SHA384: &[u8] = &[0x2a, 0x86, 0x48, 0xce, 0x3d, 0x04, 0x03, 0x03];

/// An uncompressed P-384 point: 0x04, then X and Y.
const POINT_LEN: usize = 97;

/// The parts of a certificate the swap reads.
struct Cert<'a> {
    raw: &'a [u8],
    issuer: &'a [u8],
    subject: &'a [u8],
    /// The subject's public key, an uncompressed point.
    point: &'a [u8],
    /// Each extension's OID, whether it is critical, and its value.
    extensions: Vec<(&'a [u8], bool, &'a [u8])>,
}

impl<'a> Cert<'a> {
    fn extension(&self, oid: &[u8]) -> Option<(bool, &'a [u8])> {
        self.extensions
            .iter()
            .find(|(o, _, _)| *o == oid)
            .map(|&(_, c, v)| (c, v))
    }
}

fn is(e: &Tlv<'_>, class: Class, constructed: bool, tag: u64) -> bool {
    e.class == class && e.constructed == constructed && e.tag == tag
}

/// `raw` read as an X.509 certificate, or `None` where it is not one.
fn parse_cert(raw: &[u8]) -> Option<Cert<'_>> {
    let top = read_whole(raw).ok()?;
    if !is(&top, Class::Universal, true, SEQUENCE) {
        return None;
    }
    let mut c = Children::new(top.value);
    let tbs = c.next().ok()??;
    let alg = c.next().ok()??;
    let sig = c.next().ok()??;
    if !is(&tbs, Class::Universal, true, SEQUENCE)
        || !is(&alg, Class::Universal, true, SEQUENCE)
        || !is(&sig, Class::Universal, false, BIT_STRING)
        || c.next().ok()?.is_some()
    {
        return None;
    }
    let mut t = Children::new(tbs.value);
    let version = t.next().ok()??;
    if !is(&version, Class::Context, true, 0) {
        return None;
    }
    let serial = t.next().ok()??;
    let _sig_alg = t.next().ok()??;
    let issuer = t.next().ok()??;
    let _validity = t.next().ok()??;
    let subject = t.next().ok()??;
    let spki = t.next().ok()??;
    if !is(&serial, Class::Universal, false, INTEGER)
        || !is(&issuer, Class::Universal, true, SEQUENCE)
        || !is(&subject, Class::Universal, true, SEQUENCE)
        || !is(&spki, Class::Universal, true, SEQUENCE)
    {
        return None;
    }
    let mut s = Children::new(spki.value);
    let _key_alg = s.next().ok()??;
    let key = s.next().ok()??;
    if !is(&key, Class::Universal, false, BIT_STRING) || key.value.first() != Some(&0) {
        return None;
    }
    let mut extensions = Vec::new();
    while let Some(e) = t.next().ok()? {
        if !is(&e, Class::Context, true, 3) {
            continue;
        }
        let list = read_whole(e.value).ok()?;
        let mut l = Children::new(list.value);
        while let Some(x) = l.next().ok()? {
            let mut f = Children::new(x.value);
            let oid = f.next().ok()??;
            let mut v = f.next().ok()??;
            let mut critical = false;
            if is(&v, Class::Universal, false, BOOLEAN) {
                critical = v.value == [0xff];
                v = f.next().ok()??;
            }
            if !is(&oid, Class::Universal, false, OBJECT_IDENTIFIER)
                || !is(&v, Class::Universal, false, OCTET_STRING)
            {
                return None;
            }
            extensions.push((oid.value, critical, v.value));
        }
    }
    Some(Cert {
        raw,
        issuer: issuer.raw,
        subject: subject.raw,
        point: &key.value[1..],
        extensions,
    })
}

/// Every certificate in `b`, at the offsets where one starts, scanning past each found.
fn certs_in(b: &[u8]) -> Vec<Cert<'_>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i + 4 <= b.len() {
        if b[i] == 0x30
            && b[i + 1] == 0x82
            && let Ok((e, _)) = read_tlv(&b[i..])
            && let Some(c) = parse_cert(e.raw)
        {
            i += c.raw.len();
            out.push(c);
            continue;
        }
        i += 1;
    }
    out
}

/// How often `needle` occurs in `hay`, overlaps counted.
fn count(hay: &[u8], needle: &[u8]) -> usize {
    if needle.is_empty() || hay.len() < needle.len() {
        return 0;
    }
    hay.windows(needle.len()).filter(|w| *w == needle).count()
}

/// The guest's anchors, as its kernel collection holds them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Guest {
    pub root: Vec<u8>,
    pub intermediate: Vec<u8>,
    pub leaf: Vec<u8>,
    /// The leaf's public key, which the boot policy kext also keeps on its own.
    pub point: Vec<u8>,
    /// The intermediate's and the leaf's img4 extension values.
    pub intermediate_img4: Vec<u8>,
    pub leaf_img4: Vec<u8>,
    /// The entries other than the boot policy kext that embed the root, in the collection's
    /// order: [`APPLE_IMAGE4`] among them.
    pub root_kexts: Vec<String>,
}

/// Each kext's segments that Ambrosia patches: all but `__LINKEDIT`, which the whole collection
/// shares.
fn patched_segments<'a>(fs: &FileSet<'a>, id: &str) -> Result<Vec<&'a [u8]>> {
    Ok(fs
        .segments(id)?
        .into_iter()
        .filter(|(n, _)| n != "__LINKEDIT")
        .map(|(_, b)| b)
        .collect())
}

fn count_in(segments: &[&[u8]], needle: &[u8]) -> usize {
    segments.iter().map(|s| count(s, needle)).sum()
}

/// Find the guest's anchors in kernel collection `fs`, checking every count the patches rely
/// on.
///
/// # Errors
///
/// A kext is missing, the boot policy kext holds no chain or more than one, or a count differs
/// from what the patches expect.
pub fn guest(fs: &FileSet) -> Result<Guest> {
    let consts = fs.section(BOOT_POLICY, "__TEXT", "__const")?;
    let certs = certs_in(consts);
    let mut chains = Vec::new();
    for leaf in &certs {
        for inter in certs
            .iter()
            .filter(|c| c.subject == leaf.issuer && c.raw != leaf.raw)
        {
            for root in certs
                .iter()
                .filter(|c| c.subject == inter.issuer && c.issuer == c.subject)
            {
                if leaf.extension(OID_IMG4).is_some() && inter.extension(OID_IMG4).is_some() {
                    chains.push((root, inter, leaf));
                }
            }
        }
    }
    ensure!(
        chains.len() == 1,
        "{BOOT_POLICY} holds {} LocalPolicy chains among its {} certificates, not one",
        chains.len(),
        certs.len()
    );
    let (root, inter, leaf) = chains[0];
    ensure!(
        leaf.point.len() == POINT_LEN && leaf.point[0] == 4,
        "the LocalPolicy leaf's key is not an uncompressed P-384 point"
    );
    let mut g = Guest {
        root: root.raw.to_vec(),
        intermediate: inter.raw.to_vec(),
        leaf: leaf.raw.to_vec(),
        point: leaf.point.to_vec(),
        intermediate_img4: inter.extension(OID_IMG4).unwrap().1.to_vec(),
        leaf_img4: leaf.extension(OID_IMG4).unwrap().1.to_vec(),
        root_kexts: Vec::new(),
    };

    // Each patch replaces exactly one occurrence: the root once in each entry that embeds it,
    // the intermediate and leaf once in the boot policy kext, and the key once more there on
    // its own, outside the leaf certificate the patch before it replaces.
    let bp = patched_segments(fs, BOOT_POLICY)?;
    for (what, n) in [
        ("intermediate", count_in(&bp, &g.intermediate)),
        ("leaf", count_in(&bp, &g.leaf)),
        ("root", count_in(&bp, &g.root)),
        (
            "leaf key outside the leaf",
            count_in(&bp, &g.point)
                .saturating_sub(count_in(&bp, &g.leaf) * count(&g.leaf, &g.point)),
        ),
    ] {
        ensure!(
            n == 1,
            "{BOOT_POLICY} holds the LocalPolicy {what} {n} times, not once"
        );
    }
    for id in fs.ids().filter(|id| *id != BOOT_POLICY) {
        let n = count_in(&patched_segments(fs, id)?, &g.root);
        ensure!(n <= 1, "{id} holds the img4 test root {n} times, not once");
        if n == 1 {
            g.root_kexts.push(id.to_owned());
        }
    }
    ensure!(
        g.root_kexts.iter().any(|id| id == APPLE_IMAGE4),
        "{APPLE_IMAGE4} does not hold the img4 test root"
    );
    Ok(g)
}

/// The project's keys: its root's, its intermediate's and its leaf's, which signs the
/// LocalPolicies.
pub struct Keys {
    pub root: SigningKey,
    pub intermediate: SigningKey,
    pub leaf: SigningKey,
}

impl Keys {
    /// The project's keys, built in: one set for every VM, since whoever can write a VM's disk
    /// can write its NVRAM and its bundle too, so a key per VM, or a secret one, would protect
    /// nothing (docs/design.md, "The LocalPolicy"). `keys/make.sh` made them.
    #[must_use]
    pub fn project() -> Self {
        Self::from_pem(
            include_str!("../keys/project-root.key"),
            include_str!("../keys/project-intermediate.key"),
            include_str!("../keys/project-leaf.key"),
        )
        .expect("the built-in LocalPolicy keys are P-384 private keys in PEM")
    }

    /// The keys of the root, the intermediate and the leaf, each a P-384 private key in PEM
    /// (PKCS#8 or SEC1).
    ///
    /// # Errors
    ///
    /// One is not such a key.
    pub fn from_pem(root: &str, intermediate: &str, leaf: &str) -> Result<Self> {
        let key = |what: &str, pem: &str| -> Result<SigningKey> {
            use p384::pkcs8::DecodePrivateKey as _;
            let k = p384::SecretKey::from_pkcs8_pem(pem)
                .or_else(|_| p384::SecretKey::from_sec1_pem(pem))
                .with_context(|| format!("the {what} key is not a P-384 private key in PEM"))?;
            Ok(SigningKey::from(k))
        };
        Ok(Self {
            root: key("root", root)?,
            intermediate: key("intermediate", intermediate)?,
            leaf: key("leaf", leaf)?,
        })
    }
}

/// The project's chain, each certificate as long as the guest's it stands in for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Chain {
    pub root: Vec<u8>,
    pub intermediate: Vec<u8>,
    pub leaf: Vec<u8>,
    /// The leaf's public key, an uncompressed point.
    pub point: Vec<u8>,
}

fn oid(o: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    put(&mut out, UNIVERSAL, OBJECT_IDENTIFIER, o);
    out
}

fn octets(v: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    put(&mut out, UNIVERSAL, OCTET_STRING, v);
    out
}

fn bits(unused: u8, v: &[u8]) -> Vec<u8> {
    let mut b = vec![unused];
    b.extend_from_slice(v);
    let mut out = Vec::new();
    put(&mut out, UNIVERSAL, BIT_STRING, &b);
    out
}

/// A name of organization `reims-restore` and common name `cn`.
fn name(cn: &str) -> Vec<u8> {
    let rdn = |o: &[u8], v: &str| {
        let mut s = Vec::new();
        put(&mut s, UNIVERSAL, UTF8_STRING, v.as_bytes());
        let mut set = Vec::new();
        put(&mut set, UNIVERSAL_CONSTRUCTED, SET, &seq(&[&oid(o), &s]));
        set
    };
    seq(&[&rdn(OID_O, "reims-restore"), &rdn(OID_CN, cn)])
}

fn extension(o: &[u8], critical: bool, value: &[u8]) -> Vec<u8> {
    let mut crit = Vec::new();
    if critical {
        put(&mut crit, UNIVERSAL, BOOLEAN, &[0xff]);
    }
    seq(&[&oid(o), &crit, &octets(value)])
}

fn point_of(k: &SigningKey) -> Vec<u8> {
    k.verifying_key().to_sec1_point(false).as_bytes().to_vec()
}

/// What a certificate of the chain is.
struct Spec<'a> {
    subject: &'a str,
    issuer: &'a str,
    key: &'a SigningKey,
    signer: &'a SigningKey,
    /// The root's extensions, or a certificate under it carrying this img4 extension value.
    img4: Option<&'a [u8]>,
    len: usize,
}

/// A certificate of `spec` exactly `spec.len` bytes long: the filler extension sized to make
/// up the difference, and the serial drawn again until the signature's DER length lands it.
fn certificate(spec: &Spec) -> Result<Vec<u8>> {
    let point = point_of(spec.key);
    let sha = sha2::Sha256::digest(&point);
    let key_id = &sha[..20];
    let alg = seq(&[&oid(OID_ECDSA_SHA384)]);
    let spki = seq(&[&seq(&[&oid(OID_EC_KEY), &oid(OID_P384)]), &bits(0, &point)]);
    let utc = |s: &str| {
        let mut out = Vec::new();
        put(&mut out, UNIVERSAL, UTC_TIME, s.as_bytes());
        out
    };
    let validity = seq(&[&utc("260101000000Z"), &utc("491231235959Z")]);
    let mut fixed: Vec<Vec<u8>> = Vec::new();
    match spec.img4 {
        None => {
            let mut ca = Vec::new();
            put(&mut ca, UNIVERSAL, BOOLEAN, &[0xff]);
            fixed.push(extension(OID_BASIC_CONSTRAINTS, true, &seq(&[&ca])));
            // digitalSignature, keyCertSign, cRLSign.
            fixed.push(extension(OID_KEY_USAGE, true, &bits(1, &[0x86])));
            fixed.push(extension(OID_SKI, false, &octets(key_id)));
            let mut aki = Vec::new();
            put(&mut aki, 0x80, 0, key_id);
            fixed.push(extension(OID_AKI, false, &seq(&[&aki])));
        }
        Some(img4) => {
            fixed.push(extension(OID_BASIC_CONSTRAINTS, true, &seq(&[])));
            // digitalSignature, nonRepudiation, keyEncipherment, dataEncipherment.
            fixed.push(extension(OID_KEY_USAGE, true, &bits(4, &[0xf0])));
            fixed.push(extension(OID_IMG4, true, img4));
        }
    }
    let mut filler = 0usize;
    for attempt in 0u64..4096 {
        // An eight-byte serial, its top bit clear and its top byte not zero, so its encoding's
        // length never changes.
        let mut s = [0u8; 8];
        getrandom::fill(&mut s)
            .map_err(|e| anyhow::anyhow!("no random bytes for a serial: {e}"))?;
        s[0] = (s[0] & 0x7f).max(1);
        let mut serial = Vec::new();
        put(&mut serial, UNIVERSAL, INTEGER, &s);
        let mut exts = fixed.clone();
        exts.push(extension(OID_FILLER, false, &octets(&vec![0u8; filler])));
        let mut ext_list = Vec::new();
        put(
            &mut ext_list,
            CONTEXT_CONSTRUCTED,
            3,
            &seq(&exts.iter().map(Vec::as_slice).collect::<Vec<_>>()),
        );
        let mut version = Vec::new();
        put(&mut version, CONTEXT_CONSTRUCTED, 0, &[0x02, 0x01, 0x02]);
        let tbs = seq(&[
            &version,
            &serial,
            &alg,
            &name(spec.issuer),
            &validity,
            &name(spec.subject),
            &spki,
            &ext_list,
        ]);
        let sig: p384::ecdsa::Signature = spec.signer.sign(&tbs);
        let cert = seq(&[&tbs, &alg, &bits(0, sig.to_der().as_bytes())]);
        if cert.len() == spec.len {
            return Ok(cert);
        }
        // Grow or shrink the filler by the difference, but keep drawing once it is near: the
        // signature's length varies by a byte or two from one serial to the next.
        let want = (filler as i64 + spec.len as i64 - cert.len() as i64).max(0) as usize;
        if want != filler && attempt % 8 == 0 {
            filler = want;
        }
    }
    bail!("no certificate of {} bytes for {}", spec.len, spec.subject)
}

/// The project's chain for `guest`, from `keys`.
///
/// # Errors
///
/// No certificate of a guest's length can be made (one shorter than the fixed parts).
pub fn chain(guest: &Guest, keys: &Keys) -> Result<Chain> {
    let (root_n, inter_n, leaf_n) = (
        "reims-restore secp384r1 Root",
        "reims-restore secp384r1 Intermediate",
        "reims-restore secp384r1 Leaf",
    );
    let root = certificate(&Spec {
        subject: root_n,
        issuer: root_n,
        key: &keys.root,
        signer: &keys.root,
        img4: None,
        len: guest.root.len(),
    })?;
    let intermediate = certificate(&Spec {
        subject: inter_n,
        issuer: root_n,
        key: &keys.intermediate,
        signer: &keys.root,
        img4: Some(&guest.intermediate_img4),
        len: guest.intermediate.len(),
    })?;
    let leaf = certificate(&Spec {
        subject: leaf_n,
        issuer: inter_n,
        key: &keys.leaf,
        signer: &keys.intermediate,
        img4: Some(&guest.leaf_img4),
        len: guest.leaf.len(),
    })?;
    Ok(Chain {
        root,
        intermediate,
        leaf,
        point: point_of(&keys.leaf),
    })
}

/// Ambrosia's `Kernel > Patch` entries that swap `guest`'s anchors for `chain`'s, in the order
/// they apply: the boot policy kext's intermediate, leaf, leaf key (the copy outside the leaf,
/// the one left once the leaf is replaced) and root, then the root in each other kext. Each
/// expects one replacement, so a count that differs stops the boot.
#[must_use]
pub fn patches(guest: &Guest, chain: &Chain) -> plist::Value {
    let entry = |id: &str, find: &[u8], replace: &[u8]| {
        let mut d = plist::Dictionary::new();
        d.insert("Identifier".into(), plist::Value::String(id.into()));
        d.insert("Find".into(), plist::Value::Data(find.to_vec()));
        d.insert("Replace".into(), plist::Value::Data(replace.to_vec()));
        d.insert("Expect".into(), plist::Value::Integer(1.into()));
        plist::Value::Dictionary(d)
    };
    let mut list = vec![
        entry(BOOT_POLICY, &guest.intermediate, &chain.intermediate),
        entry(BOOT_POLICY, &guest.leaf, &chain.leaf),
        entry(BOOT_POLICY, &guest.point, &chain.point),
        entry(BOOT_POLICY, &guest.root, &chain.root),
    ];
    for id in &guest.root_kexts {
        list.push(entry(id, &guest.root, &chain.root));
    }
    plist::Value::Array(list)
}

/// Ambrosia's configuration bundle of `config`: a `newc` cpio archive (magic `070701`, the
/// form Ambrosia reads) of one file, `config.plist`, in XML.
///
/// # Errors
///
/// The plist does not serialise.
pub fn config_bundle(config: &plist::Dictionary) -> Result<Vec<u8>> {
    let mut xml = Vec::new();
    plist::Value::Dictionary(config.clone())
        .to_writer_xml(&mut xml)
        .context("write the configuration's plist")?;
    let mut out = Vec::new();
    newc_entry(&mut out, 1, 0o100644, "config.plist", &xml);
    newc_entry(&mut out, 0, 0, "TRAILER!!!", &[]);
    Ok(out)
}

/// One `newc` entry: its header's thirteen 8-digit hex fields, its name with its NUL, and its
/// data, each padded to four bytes.
fn newc_entry(out: &mut Vec<u8>, ino: u32, mode: u32, name: &str, data: &[u8]) {
    let fields = [
        ino,
        mode,
        0,
        0,
        u32::from(mode != 0),
        0,
        data.len() as u32,
        0,
        0,
        0,
        0,
        name.len() as u32 + 1,
        0,
    ];
    out.extend_from_slice(b"070701");
    for f in fields {
        out.extend_from_slice(format!("{f:08X}").as_bytes());
    }
    out.extend_from_slice(name.as_bytes());
    out.push(0);
    while out.len() % 4 != 0 {
        out.push(0);
    }
    out.extend_from_slice(data);
    while out.len() % 4 != 0 {
        out.push(0);
    }
}

/// The configuration of a VM's anchors: `Kernel > Patch` of [`patches`].
#[must_use]
pub fn config(guest: &Guest, chain: &Chain) -> plist::Dictionary {
    let mut kernel = plist::Dictionary::new();
    kernel.insert("Patch".into(), patches(guest, chain));
    let mut d = plist::Dictionary::new();
    d.insert("Kernel".into(), plist::Value::Dictionary(kernel));
    d
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernelcache::tests::fileset;

    const IMAGE4: &str = "com.apple.security.Image4";

    fn key(n: u8) -> SigningKey {
        SigningKey::from(p384::SecretKey::from_slice(&[n; 48]).unwrap())
    }

    // A guest's chain made here, of its own keys and an img4 extension of made-up bytes, at
    // lengths of its own: the root self-signed, the two under it with the extension.
    fn guest_chain() -> (Chain, Vec<u8>, Vec<u8>) {
        let keys = Keys {
            root: key(1),
            intermediate: key(2),
            leaf: key(3),
        };
        let img4_i = seq(&[&octets(b"an intermediate's made-up img4 properties")]);
        let img4_l = seq(&[&octets(b"a leaf's made-up img4 properties")]);
        let g = Guest {
            root: vec![0; 600],
            intermediate: vec![0; 820],
            leaf: vec![0; 810],
            point: Vec::new(),
            intermediate_img4: img4_i.clone(),
            leaf_img4: img4_l.clone(),
            root_kexts: Vec::new(),
        };
        (chain(&g, &keys).unwrap(), img4_i, img4_l)
    }

    #[test]
    fn a_chain_is_built_at_the_guests_lengths_and_verifies() {
        let (c, img4_i, _) = guest_chain();
        assert_eq!(
            (c.root.len(), c.intermediate.len(), c.leaf.len()),
            (600, 820, 810)
        );
        let root = parse_cert(&c.root).unwrap();
        let inter = parse_cert(&c.intermediate).unwrap();
        let leaf = parse_cert(&c.leaf).unwrap();
        assert_eq!(root.issuer, root.subject);
        assert_eq!(inter.issuer, root.subject);
        assert_eq!(leaf.issuer, inter.subject);
        assert_eq!(leaf.point, c.point);
        assert_eq!(inter.extension(OID_IMG4), Some((true, &img4_i[..])));
        assert!(root.extension(OID_IMG4).is_none());
        // Each signature verifies under its issuer's key.
        use p384::ecdsa::signature::Verifier as _;
        for (cert, issuer) in [
            (&c.root, &root),
            (&c.intermediate, &root),
            (&c.leaf, &inter),
        ] {
            let top = read_whole(cert).unwrap();
            let mut ch = Children::new(top.value);
            let tbs = ch.next().unwrap().unwrap();
            ch.next().unwrap();
            let sig = ch.next().unwrap().unwrap();
            let vk = p384::ecdsa::VerifyingKey::from_sec1_bytes(issuer.point).unwrap();
            let s = p384::ecdsa::Signature::from_der(&sig.value[1..]).unwrap();
            vk.verify(tbs.raw, &s).unwrap();
        }
    }

    #[test]
    fn the_project_keys_are_built_in_and_make_a_chain() {
        let k = Keys::project();
        let points = [&k.root, &k.intermediate, &k.leaf].map(point_of);
        assert!(points.iter().all(|p| p.len() == 97 && p[0] == 4));
        assert!(points[0] != points[1] && points[1] != points[2] && points[0] != points[2]);
        let (g, _, _) = guest_chain();
        let guest = Guest {
            root: g.root,
            intermediate: g.intermediate,
            leaf: g.leaf,
            point: g.point,
            intermediate_img4: seq(&[&octets(b"img4")]),
            leaf_img4: seq(&[&octets(b"img4")]),
            root_kexts: Vec::new(),
        };
        let c = chain(&guest, &k).unwrap();
        assert_eq!(c.leaf.len(), guest.leaf.len());
        assert_eq!(c.point, points[2]);
        assert!(Keys::from_pem("not a key", "", "").is_err());
    }

    /// A boot policy kext's constants as the guest's hold them, made here: other data, the
    /// chain, the leaf's key on its own, and an unrelated self-signed certificate.
    fn kernel(g: &Chain, other: &[u8]) -> Vec<u8> {
        let mut bp = b"some constants before".to_vec();
        bp.extend_from_slice(&g.root);
        bp.extend_from_slice(&g.intermediate);
        bp.extend_from_slice(&g.leaf);
        bp.extend_from_slice(&[0x5a; 84]);
        bp.extend_from_slice(&g.point);
        bp.extend_from_slice(other);
        let mut i4 = b"image4's constants".to_vec();
        i4.extend_from_slice(&g.root);
        let mut a4 = other.to_vec();
        a4.extend_from_slice(&g.root);
        fileset(&[
            ("com.apple.kernel", b"the kernel"),
            (BOOT_POLICY, &bp),
            (APPLE_IMAGE4, &a4),
            (IMAGE4, &i4),
        ])
    }

    #[test]
    fn the_guests_anchors_are_found_by_their_shape() {
        let (g, img4_i, img4_l) = guest_chain();
        // A self-signed certificate that anchors nothing here, as the production roots don't.
        let other = certificate(&Spec {
            subject: "another root",
            issuer: "another root",
            key: &key(9),
            signer: &key(9),
            img4: None,
            len: 640,
        })
        .unwrap();
        let kc = kernel(&g, &other);
        let fs = FileSet::parse(&kc).unwrap();
        let found = guest(&fs).unwrap();
        assert_eq!(found.root, g.root);
        assert_eq!(found.intermediate, g.intermediate);
        assert_eq!(found.leaf, g.leaf);
        assert_eq!(found.point, g.point);
        assert_eq!(
            (&found.intermediate_img4, &found.leaf_img4),
            (&img4_i, &img4_l)
        );

        // The project's chain at those lengths, and the six patches in order.
        let ours = chain(
            &found,
            &Keys {
                root: key(4),
                intermediate: key(5),
                leaf: key(6),
            },
        )
        .unwrap();
        let p = patches(&found, &ours);
        let list = p.as_array().unwrap();
        assert_eq!(list.len(), 6);
        let field = |i: usize, k: &str| list[i].as_dictionary().unwrap()[k].clone();
        assert_eq!(field(2, "Find").as_data().unwrap(), &g.point[..]);
        assert_eq!(field(2, "Replace").as_data().unwrap(), &ours.point[..]);
        assert_eq!(found.root_kexts, [APPLE_IMAGE4, IMAGE4]);
        assert_eq!(field(4, "Identifier").as_string(), Some(APPLE_IMAGE4));
        assert_eq!(field(5, "Identifier").as_string(), Some(IMAGE4));
        for i in 0..6 {
            assert_eq!(
                field(i, "Find").as_data().unwrap().len(),
                field(i, "Replace").as_data().unwrap().len()
            );
            assert_eq!(field(i, "Expect").as_unsigned_integer(), Some(1));
        }
    }

    #[test]
    fn a_count_the_patches_dont_expect_fails() {
        let (g, _, _) = guest_chain();
        // The root twice in Image4: its patch would replace two.
        let mut i4 = g.root.clone();
        i4.extend_from_slice(&g.root);
        let mut bp = g.root.clone();
        bp.extend_from_slice(&g.intermediate);
        bp.extend_from_slice(&g.leaf);
        bp.extend_from_slice(&g.point);
        let kc = fileset(&[(BOOT_POLICY, &bp), (APPLE_IMAGE4, &g.root), (IMAGE4, &i4)]);
        let e = guest(&FileSet::parse(&kc).unwrap()).unwrap_err();
        assert!(format!("{e}").contains("2 times"), "{e}");
        // No standalone copy of the leaf's key.
        let mut bp = g.root.clone();
        bp.extend_from_slice(&g.intermediate);
        bp.extend_from_slice(&g.leaf);
        let kc = fileset(&[
            (BOOT_POLICY, &bp),
            (APPLE_IMAGE4, &g.root),
            (IMAGE4, &g.root),
        ]);
        assert!(guest(&FileSet::parse(&kc).unwrap()).is_err());
    }

    #[test]
    fn the_root_is_swapped_in_each_kext_that_holds_it() {
        let (g, _, _) = guest_chain();
        let mut bp = g.root.clone();
        bp.extend_from_slice(&g.intermediate);
        bp.extend_from_slice(&g.leaf);
        bp.extend_from_slice(&g.point);
        // AppleImage4 alone, with no Image4 in the collection: five patches.
        let kc = fileset(&[(BOOT_POLICY, &bp), (APPLE_IMAGE4, &g.root)]);
        let found = guest(&FileSet::parse(&kc).unwrap()).unwrap();
        assert_eq!(found.root_kexts, [APPLE_IMAGE4]);
        let ours = chain(
            &found,
            &Keys {
                root: key(4),
                intermediate: key(5),
                leaf: key(6),
            },
        )
        .unwrap();
        assert_eq!(patches(&found, &ours).as_array().unwrap().len(), 5);
        // A root in another kext, and in no AppleImage4: the kext that verifies is unpatched.
        let kc = fileset(&[
            (BOOT_POLICY, &bp),
            (APPLE_IMAGE4, b"no root"),
            (IMAGE4, &g.root),
        ]);
        let e = guest(&FileSet::parse(&kc).unwrap()).unwrap_err();
        assert!(format!("{e}").contains("does not hold"), "{e}");
    }

    #[test]
    fn the_bundle_is_a_newc_archive_of_the_config() {
        let mut d = plist::Dictionary::new();
        d.insert(
            "Kernel".into(),
            plist::Value::Dictionary(plist::Dictionary::new()),
        );
        let b = config_bundle(&d).unwrap();
        assert_eq!(&b[..6], b"070701");
        assert_eq!(b.len() % 4, 0);
        let name_at = 110;
        assert_eq!(&b[name_at..name_at + 13], b"config.plist\0");
        let size = usize::from_str_radix(std::str::from_utf8(&b[54..62]).unwrap(), 16).unwrap();
        let data_at = (name_at + 13).div_ceil(4) * 4;
        let back = plist::Value::from_reader_xml(&b[data_at..data_at + size]).unwrap();
        assert_eq!(back, plist::Value::Dictionary(d));
        assert!(b.windows(10).any(|w| w == b"TRAILER!!!"));
    }
}
