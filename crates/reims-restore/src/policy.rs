//! The macOS volume group's LocalPolicies, which the restore writes and signs itself
//! (docs/design.md, "The LocalPolicy"): iSCPreboot's `/<G>/LocalPolicy/<LPNH>.img4`, which the
//! installed macOS boots with, and `<LPNH>.recovery.img4` beside it, its recoveryOS's. Each is
//! the LocalPolicy IM4P ([`img4::LPOL_IM4P`]) stitched to a manifest whose properties bind it to
//! the VM, its boot manifest, its Cryptex1 ticket and the policy nonces in NVRAM.
//!
//! The properties are those of a fresh restore by Virtualization.framework of the same build,
//! whose policies the restore OS signs with Apple's built-in "Hacktivation" key **[vz]**. Ours
//! are signed by the project's own chain, which Ambrosia swaps in for the kernel's anchors at
//! boot ([`crate::anchors`]), so the kernel checks the signature and every binding as it would
//! Apple's.

use std::collections::BTreeMap;

use anyhow::{Context, Result, ensure};
use p384::ecdsa::signature::Signer as _;
use uuid::Uuid;

use crate::img4::{self, Prop};

/// What binds a LocalPolicy to one VM and one install.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Binding {
    /// The VM's ECID.
    pub ecid: u64,
    /// The chip, board and security domain, as the boot manifest carries them.
    pub chip: u64,
    pub board: u64,
    pub security_domain: u64,
    /// The OS's version as the boot manifest's `love` carries it, such as `26.2.131.0.0,0`.
    pub love: Vec<u8>,
    /// The boot manifest (Preboot's `/<G>/boot/<H>/System/Library/Caches/apticket.der`): its
    /// SHA-384 is `nsih`.
    pub boot_manifest_hash: [u8; 48],
    /// The Cryptex1 ticket: its SHA-384 is `spih`.
    pub cryptex1_ticket_hash: [u8; 48],
    /// The policy nonces whose SHA-384 are `lpnh` and `rpnh` (NVRAM's `boop-storage-nonces`).
    pub local_nonce: [u8; 16],
    pub remote_nonce: [u8; 16],
    /// The volume group: `vuid`.
    pub group: Uuid,
}

/// Which of the two policies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// `<LPNH>.img4`, the installed macOS's.
    MacOs,
    /// `<LPNH>.recovery.img4`, its recoveryOS's.
    Recovery,
}

impl Binding {
    /// `lpnh`: the policy's name, in upper-case hex.
    pub fn lpnh(&self) -> [u8; 48] {
        img4::sha384(&self.local_nonce)
    }

    /// The policy's file name in `/<G>/LocalPolicy/`.
    pub fn file_name(&self, kind: Kind) -> String {
        let hex: String = self.lpnh().iter().map(|b| format!("{b:02X}")).collect();
        match kind {
            Kind::MacOs => format!("{hex}.img4"),
            Kind::Recovery => format!("{hex}.recovery.img4"),
        }
    }

    /// The manifest's own properties (MANP) **[vz]**.
    pub fn props(&self, kind: Kind) -> BTreeMap<String, Prop> {
        let mut p = BTreeMap::new();
        let mut put = |k: &str, v: Prop| {
            p.insert(k.to_owned(), v);
        };
        put("BORD", Prop::Int(self.board));
        put("CEPO", Prop::Int(1));
        put("CHIP", Prop::Int(self.chip));
        put("CPRO", Prop::Bool(true));
        put("CSEC", Prop::Bool(true));
        put("ECID", Prop::Int(self.ecid));
        put("SDOM", Prop::Int(self.security_domain));
        put("lobo", Prop::Bool(true));
        put("love", Prop::Data(self.love.clone()));
        put("lpnh", Prop::Data(self.lpnh().to_vec()));
        put("nsih", Prop::Data(self.boot_manifest_hash.to_vec()));
        put(
            "rpnh",
            Prop::Data(img4::sha384(&self.remote_nonce).to_vec()),
        );
        put("vuid", Prop::Data(self.group.as_bytes().to_vec()));
        match kind {
            Kind::MacOs => {
                put("hrlp", Prop::Bool(true));
                put("spih", Prop::Data(self.cryptex1_ticket_hash.to_vec()));
                put("stng", Prop::Int(1));
            }
            Kind::Recovery => put("rolp", Prop::Bool(true)),
        }
        p
    }

    /// The signed MANB `SET`: the properties and the `lpol` object, the LocalPolicy IM4P's
    /// digest with `EKEY`.
    pub fn manb(&self, kind: Kind) -> Vec<u8> {
        let mut lpol = BTreeMap::new();
        lpol.insert(
            "DGST".to_owned(),
            Prop::Data(img4::sha384(&img4::LPOL_IM4P).to_vec()),
        );
        lpol.insert("EKEY".to_owned(), Prop::Bool(true));
        let mut objects = BTreeMap::new();
        objects.insert("lpol".to_owned(), lpol);
        img4::encode_manb(&self.props(kind), &objects)
    }

    /// The LocalPolicy, an IMG4, signed by `signer`.
    pub fn sign(&self, kind: Kind, signer: &Signer) -> Vec<u8> {
        let manb = self.manb(kind);
        let im4m = img4::encode_im4m(&manb, &signer.sign(&manb), &signer.chain);
        img4::stitch_img4(&img4::LPOL_IM4P, &im4m)
    }
}

/// The key LocalPolicies are signed with, and the certificate chain the manifest carries.
pub struct Signer {
    key: p384::ecdsa::SigningKey,
    /// `SEQUENCE { intermediate, leaf }`, as encoded.
    chain: Vec<u8>,
}

impl Signer {
    /// A signer of the leaf's key (PKCS#8 or SEC1 PEM) and the chain's certificates (DER), the
    /// intermediate first, as a manifest stores them.
    pub fn new(key_pem: &str, certs: &[&[u8]]) -> Result<Self> {
        use p384::pkcs8::DecodePrivateKey as _;
        let key = p384::SecretKey::from_pkcs8_pem(key_pem)
            .or_else(|_| p384::SecretKey::from_sec1_pem(key_pem))
            .context("the LocalPolicy key is not a P-384 private key in PEM")?;
        Self::with_key(p384::ecdsa::SigningKey::from(key), certs)
    }

    /// A signer of the leaf's `key` and the chain's certificates (DER), the intermediate first.
    pub fn with_key(key: p384::ecdsa::SigningKey, certs: &[&[u8]]) -> Result<Self> {
        ensure!(
            !certs.is_empty(),
            "the LocalPolicy signer has no certificates"
        );
        let body: Vec<u8> = certs.concat();
        let mut chain = vec![0x30];
        chain.extend(der_len(body.len()));
        chain.extend(body);
        Ok(Self { key, chain })
    }

    /// ECDSA P-384 with SHA-384 over `msg`, DER-encoded, as a manifest's signature is.
    fn sign(&self, msg: &[u8]) -> Vec<u8> {
        let s: p384::ecdsa::Signature = self.key.sign(msg);
        s.to_der().as_bytes().to_vec()
    }
}

fn der_len(n: usize) -> Vec<u8> {
    if n < 0x80 {
        return vec![n as u8];
    }
    let b = n.to_be_bytes();
    let skip = b.iter().take_while(|x| **x == 0).count();
    let mut out = vec![0x80 | (b.len() - skip) as u8];
    out.extend(&b[skip..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use p384::ecdsa::signature::Verifier as _;

    fn binding() -> Binding {
        Binding {
            ecid: 0x0123_4567_89ab_cdef,
            chip: 0xfe00,
            board: 0x20,
            security_domain: 1,
            love: b"26.2.131.0.0,0".to_vec(),
            boot_manifest_hash: [1; 48],
            cryptex1_ticket_hash: [2; 48],
            local_nonce: [3; 16],
            remote_nonce: [4; 16],
            group: Uuid::from_u128(0xbf30_4653),
        }
    }

    fn signer() -> (Signer, p384::ecdsa::VerifyingKey) {
        let key = p384::SecretKey::from_slice(&[7u8; 48]).unwrap();
        let pem = {
            use p384::pkcs8::EncodePrivateKey as _;
            key.to_pkcs8_pem(Default::default()).unwrap().to_string()
        };
        let s = Signer::new(&pem, &[b"\x30\x03\x02\x01\x01", b"\x30\x03\x02\x01\x02"]).unwrap();
        let v = p384::ecdsa::VerifyingKey::from(&p384::ecdsa::SigningKey::from(key));
        (s, v)
    }

    #[test]
    fn the_macos_policy_carries_vzs_properties() {
        let b = binding();
        let names: Vec<String> = b.props(Kind::MacOs).into_keys().collect();
        assert_eq!(
            names,
            [
                "BORD", "CEPO", "CHIP", "CPRO", "CSEC", "ECID", "SDOM", "hrlp", "lobo", "love",
                "lpnh", "nsih", "rpnh", "spih", "stng", "vuid"
            ]
        );
        let names: Vec<String> = b.props(Kind::Recovery).into_keys().collect();
        assert_eq!(
            names,
            [
                "BORD", "CEPO", "CHIP", "CPRO", "CSEC", "ECID", "SDOM", "lobo", "love", "lpnh",
                "nsih", "rolp", "rpnh", "vuid"
            ]
        );
        assert_eq!(
            b.props(Kind::MacOs)["vuid"],
            Prop::Data(b.group.as_bytes().to_vec())
        );
    }

    #[test]
    fn a_signed_policy_parses_and_verifies() {
        let b = binding();
        let (s, v) = signer();
        let lp = b.sign(Kind::MacOs, &s);
        let (im4p, im4m) = img4::split_img4(&lp).unwrap();
        assert_eq!(im4p, img4::LPOL_IM4P);
        let m = img4::parse_im4m(im4m.expect("the policy carries its manifest")).unwrap();
        assert_eq!(m.props, b.props(Kind::MacOs));
        assert_eq!(m.signed, b.manb(Kind::MacOs));
        assert_eq!(
            m.objects["lpol"]["DGST"],
            Prop::Data(img4::sha384(&img4::LPOL_IM4P).to_vec())
        );
        assert_eq!(
            m.certs,
            [
                b"\x30\x03\x02\x01\x01".to_vec(),
                b"\x30\x03\x02\x01\x02".to_vec()
            ]
        );
        let sig = p384::ecdsa::Signature::from_der(&m.signature).unwrap();
        v.verify(&m.signed, &sig).unwrap();
    }

    #[test]
    fn the_policy_is_named_for_its_nonces_hash() {
        let b = binding();
        let n = b.file_name(Kind::MacOs);
        assert_eq!(n.len(), 96 + 5);
        assert!(n.starts_with(&format!("{:02X}", img4::sha384(&[3; 16])[0])));
        assert_eq!(
            b.file_name(Kind::Recovery),
            n.replace(".img4", ".recovery.img4")
        );
    }

    #[test]
    fn a_key_that_is_not_p384_is_refused() {
        let p256 = {
            use p256::pkcs8::EncodePrivateKey as _;
            p256::SecretKey::from_slice(&[7u8; 32])
                .unwrap()
                .to_pkcs8_pem(Default::default())
                .unwrap()
                .to_string()
        };
        assert!(Signer::new(&p256, &[b"\x30\x00"]).is_err());
        assert!(Signer::new("not a key", &[b"\x30\x00"]).is_err());
    }
}
