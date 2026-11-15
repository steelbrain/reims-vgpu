//! Apple Encrypted Archives (`.aea`), as IPSWs from macOS 15 on carry their larger images: the
//! key fetched from Apple's key service and unwrapped, then the archive decrypted at random
//! offsets by `aea-tools`.
//!
//! An IPSW's archive is profile 1: a symmetric key, with no signature. Its auth data names where
//! the key is: `com.apple.wkms.fcs-key-url`, an HTTPS URL on `wkms-public.apple.com` that answers a
//! P-256 private key in PEM, and `com.apple.wkms.fcs-response`, JSON whose `enc-request` (an
//! HPKE encapsulated key) and `wrapped-key` (the sealed archive key) that private key opens with
//! HPKE's base mode: DHKEM(P-256, HKDF-SHA256), HKDF-SHA256 and AES-256-GCM, with empty info
//! and associated data.

use std::collections::VecDeque;
use std::io;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aea_tools::reader::AeaReader;
use anyhow::{Context, Result, bail, ensure};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as B64;
use hpke::Deserializable as _;

use crate::source::{Cursor, ReadAt};

/// The host every archive key URL must be on.
pub const KEY_HOST: &str = "wkms-public.apple.com";

/// Whether `head` starts like an Apple Encrypted Archive.
pub fn is_aea(head: &[u8]) -> bool {
    head.starts_with(b"AEA1")
}

/// An archive's prologue: its profile and auth data.
#[derive(Debug)]
pub struct Prologue {
    /// The profile; IPSWs use 1.
    pub profile: u32,
    /// The auth data's fields, in order.
    pub fields: Vec<(String, Vec<u8>)>,
}

impl Prologue {
    /// Reads the prologue at the start of `src`.
    pub fn read(src: &dyn ReadAt) -> Result<Self> {
        let mut head = [0u8; 12];
        src.read_exact_at(0, &mut head)
            .context("read the archive's prologue")?;
        ensure!(is_aea(&head), "not an Apple Encrypted Archive");
        let profile = u32::from_le_bytes([head[4], head[5], head[6], 0]);
        let len = u32::from_le_bytes(head[8..12].try_into().unwrap()) as usize;
        ensure!(len <= 1 << 20, "the archive's auth data is {len} bytes");
        let mut ad = vec![0u8; len];
        src.read_exact_at(12, &mut ad)
            .context("read the archive's auth data")?;
        Ok(Self {
            profile,
            fields: parse_auth_data(&ad)?,
        })
    }

    /// The field called `key`.
    pub fn field(&self, key: &str) -> Option<&[u8]> {
        self.fields
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_slice())
    }
}

/// Splits auth data into its `key\0value` fields, each after a 32-bit little-endian size that
/// counts itself.
fn parse_auth_data(mut ad: &[u8]) -> Result<Vec<(String, Vec<u8>)>> {
    let mut out = Vec::new();
    while !ad.is_empty() {
        ensure!(
            ad.len() >= 4,
            "the archive's auth data ends inside a field's size"
        );
        let n = u32::from_le_bytes(ad[..4].try_into().unwrap()) as usize;
        ensure!(
            n >= 4 && n <= ad.len(),
            "the archive's auth data has a field of {n} bytes"
        );
        let e = &ad[4..n];
        let nul = e
            .iter()
            .position(|&b| b == 0)
            .context("an auth data field has no key")?;
        let key = std::str::from_utf8(&e[..nul]).context("an auth data key is not UTF-8")?;
        out.push((key.to_owned(), e[nul + 1..].to_vec()));
        ad = &ad[n..];
    }
    Ok(out)
}

/// Where an archive's key is: the key service URL, and the sealed key.
#[derive(Debug, Clone)]
pub struct KeyRequest {
    /// The URL that answers the private key, on [`KEY_HOST`].
    pub url: String,
    /// The HPKE encapsulated key.
    pub enc_request: Vec<u8>,
    /// The sealed archive key: 32 bytes and a 16-byte tag.
    pub wrapped_key: Vec<u8>,
}

impl KeyRequest {
    /// The key request an IPSW archive's prologue carries.
    pub fn from_prologue(p: &Prologue) -> Result<Self> {
        ensure!(
            p.profile == 1,
            "the archive is of profile {}, not an IPSW's 1",
            p.profile
        );
        let url = p
            .field("com.apple.wkms.fcs-key-url")
            .context("the archive names no key URL (com.apple.wkms.fcs-key-url)")?;
        let url = std::str::from_utf8(url)
            .context("the archive's key URL is not UTF-8")?
            .to_owned();
        check_key_url(&url)?;
        let resp = p
            .field("com.apple.wkms.fcs-response")
            .context("the archive carries no sealed key (com.apple.wkms.fcs-response)")?;
        let v: serde_json::Value =
            serde_json::from_slice(resp).context("parse the archive's fcs-response")?;
        let b64 = |k: &str| -> Result<Vec<u8>> {
            let s = v
                .get(k)
                .and_then(|x| x.as_str())
                .with_context(|| format!("the fcs-response has no {k}"))?;
            B64.decode(s)
                .with_context(|| format!("the fcs-response's {k} is not base64"))
        };
        Ok(Self {
            url,
            enc_request: b64("enc-request")?,
            wrapped_key: b64("wrapped-key")?,
        })
    }
}

/// Refuses a key URL that isn't HTTPS on Apple's key service: the URL comes from the IPSW.
fn check_key_url(url: &str) -> Result<()> {
    let rest = url
        .strip_prefix("https://")
        .with_context(|| format!("the archive's key URL {url:?} is not HTTPS"))?;
    let host = rest.split(['/', '?', '#']).next().unwrap_or("");
    ensure!(
        host == KEY_HOST,
        "the archive's key URL is on {host:?}, not Apple's key service {KEY_HOST}"
    );
    Ok(())
}

/// Something that answers a key URL with the private key's PEM.
pub trait KeyService {
    /// Fetches `url`.
    fn fetch_pem(&self, url: &str) -> Result<String>;
}

/// Apple's key service over HTTPS.
pub struct AppleKeyService {
    agent: ureq::Agent,
}

impl Default for AppleKeyService {
    fn default() -> Self {
        let agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(60)))
            .user_agent(crate::USER_AGENT)
            .build()
            .into();
        Self { agent }
    }
}

impl KeyService for AppleKeyService {
    fn fetch_pem(&self, url: &str) -> Result<String> {
        check_key_url(url)?;
        let mut last = None;
        for attempt in 0..3 {
            if attempt > 0 {
                std::thread::sleep(Duration::from_secs(2 << attempt));
            }
            match self.agent.get(url).call() {
                Ok(mut r) => {
                    return r
                        .body_mut()
                        .with_config()
                        .limit(64 << 10)
                        .read_to_string()
                        .context("read the key service's answer");
                }
                Err(ureq::Error::StatusCode(c)) if (400..500).contains(&c) => {
                    bail!(
                        "Apple's key service refused the archive's key ({c}): the IPSW may be too new or withdrawn"
                    )
                }
                Err(e) => last = Some(e),
            }
        }
        Err(last.unwrap())
            .context("reach Apple's key service (wkms-public.apple.com) for the archive's key")
    }
}

/// The archive key: `req`'s sealed key opened with the private key in `pem`.
pub fn unwrap_key(req: &KeyRequest, pem: &str) -> Result<[u8; 32]> {
    use p256::pkcs8::DecodePrivateKey as _;
    type Kem = hpke::kem::DhP256HkdfSha256;
    let sk = p256::SecretKey::from_pkcs8_pem(pem.trim())
        .context("the key service's answer is not a P-256 private key")?;
    let sk = <Kem as hpke::Kem>::PrivateKey::from_bytes(&sk.to_bytes())
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let enc = <Kem as hpke::Kem>::EncappedKey::from_bytes(&req.enc_request)
        .map_err(|e| anyhow::anyhow!("the archive's enc-request is not a P-256 point: {e:?}"))?;
    let key = hpke::single_shot_open::<hpke::aead::AesGcm256, hpke::kdf::HkdfSha256, Kem>(
        &hpke::OpModeR::Base,
        &sk,
        &enc,
        b"",
        &req.wrapped_key,
        b"",
    )
    .map_err(|_| anyhow::anyhow!("the key service's key doesn't open the archive's key"))?;
    key.as_slice()
        .try_into()
        .context("the archive's key is not 32 bytes")
}

/// Fetches and unwraps the key of the archive `src` holds.
pub fn archive_key(src: &dyn ReadAt, service: &dyn KeyService) -> Result<[u8; 32]> {
    let req = KeyRequest::from_prologue(&Prologue::read(src)?)?;
    let pem = service.fetch_pem(&req.url)?;
    unwrap_key(&req, &pem)
}

/// An archive's plaintext, read at random offsets.
///
/// `aea-tools` decrypts and checks each segment (its HMAC, its size and its SHA-256); the index
/// of segments is kept here, built as reads reach further into the archive, with the clusters
/// counted from the plaintext's size. (`aea-tools`' own stream counts them from the archive's
/// compressed size, so it never indexes a compressed archive's tail.)
pub struct Aea<S: ReadAt + Unpin> {
    state: Mutex<State<S>>,
    len: u64,
}

/// A decrypted segment, by its cluster and index in it.
type Segment = ((u32, u32), Arc<Vec<u8>>);

struct State<S: ReadAt + Unpin> {
    reader: AeaReader<Cursor<S>>,
    /// Each segment's cluster, index in it, first plaintext byte and size, in order.
    segments: Vec<(u32, u32, u64, u64)>,
    /// The next cluster to index, and where its plaintext starts.
    next_cluster: u32,
    indexed: u64,
    clusters: u32,
    cache: VecDeque<Segment>,
}

/// How many decrypted segments are kept.
const CACHE_SEGMENTS: usize = 64;

impl<S: ReadAt + Unpin> Aea<S> {
    /// Opens the archive `src` holds with its `key`; checks the key against the archive's root
    /// header.
    pub fn open(src: S, key: &[u8; 32]) -> Result<Self> {
        let mut reader = AeaReader::new(key, Cursor::new(src)).context("read the archive")?;
        let rh = reader
            .get_root_header()
            .context("decrypt the archive's root header: wrong key, or a damaged archive")?;
        let len = u64::from_le_bytes(rh.raw_size);
        let seg = u64::from(u32::from_le_bytes(rh.segment_size));
        let per = u64::from(u32::from_le_bytes(rh.segments_per_cluster));
        ensure!(
            seg > 0 && per > 0 && seg <= 64 << 20,
            "the archive's root header is damaged"
        );
        let clusters =
            u32::try_from(len.div_ceil(seg * per)).context("the archive has too many clusters")?;
        let state = State {
            reader,
            segments: Vec::new(),
            next_cluster: 0,
            indexed: 0,
            clusters,
            cache: VecDeque::new(),
        };
        Ok(Self {
            state: Mutex::new(state),
            len,
        })
    }
}

fn damaged(e: impl std::fmt::Display) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("the archive is damaged: {e}"),
    )
}

impl<S: ReadAt + Unpin> State<S> {
    /// Indexes clusters until the segment holding `offset` is known.
    fn index_to(&mut self, offset: u64) -> io::Result<()> {
        while self.indexed <= offset {
            if self.next_cluster >= self.clusters {
                return Err(damaged(format!(
                    "its segments end at {} bytes",
                    self.indexed
                )));
            }
            let c = self.next_cluster;
            let h = self.reader.get_cluster_header(c).map_err(damaged)?;
            for (i, info) in h.segment_info.iter().enumerate() {
                let size = u64::from(info.decompressed_size);
                if size > 0 {
                    self.segments.push((c, i as u32, self.indexed, size));
                    self.indexed += size;
                }
            }
            self.next_cluster += 1;
        }
        Ok(())
    }

    fn segment(&mut self, c: u32, s: u32) -> io::Result<Arc<Vec<u8>>> {
        if let Some((_, d)) = self.cache.iter().find(|(k, _)| *k == (c, s)) {
            return Ok(d.clone());
        }
        let d = Arc::new(self.reader.get_segment(c, s).map_err(damaged)?);
        if self.cache.len() >= CACHE_SEGMENTS {
            self.cache.pop_front();
        }
        self.cache.push_back(((c, s), d.clone()));
        Ok(d)
    }
}

impl<S: ReadAt + Unpin> ReadAt for Aea<S> {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        if offset >= self.len || buf.is_empty() {
            return Ok(0);
        }
        let mut st = self.state.lock().unwrap();
        st.index_to(offset)?;
        let i = st
            .segments
            .partition_point(|&(_, _, start, _)| start <= offset)
            - 1;
        let (c, s, start, size) = st.segments[i];
        let seg = st.segment(c, s)?;
        let d: &[u8] = &seg;
        if d.len() as u64 != size {
            return Err(damaged("a segment's size disagrees with its header"));
        }
        let within = (offset - start) as usize;
        let n = buf.len().min(d.len() - within);
        buf[..n].copy_from_slice(&d[within..within + n]);
        Ok(n)
    }

    fn len(&self) -> u64 {
        self.len
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p256::pkcs8::EncodePrivateKey as _;

    #[test]
    fn the_key_service_is_sent_only_the_pinned_headers() {
        let svc = AppleKeyService::default();
        let h = crate::sent_headers::of(|url| {
            svc.agent.get(url).call().unwrap();
        });
        crate::sent_headers::assert_pinned(&h, &[]);
    }

    fn ad(fields: &[(&str, &[u8])]) -> Vec<u8> {
        let mut out = Vec::new();
        for (k, v) in fields {
            let n = 4 + k.len() + 1 + v.len();
            out.extend_from_slice(&(n as u32).to_le_bytes());
            out.extend_from_slice(k.as_bytes());
            out.push(0);
            out.extend_from_slice(v);
        }
        out
    }

    fn prologue(profile: u32, fields: &[(&str, &[u8])]) -> Vec<u8> {
        let a = ad(fields);
        let mut p = b"AEA1".to_vec();
        p.extend_from_slice(&profile.to_le_bytes()[..3]);
        p.push(0);
        p.extend_from_slice(&(a.len() as u32).to_le_bytes());
        p.extend_from_slice(&a);
        p
    }

    /// Seals `key` to a fresh P-256 key as Apple's key service would; returns the PEM and the
    /// fcs-response JSON.
    fn sealed(key: &[u8; 32]) -> (String, String) {
        type Kem = hpke::kem::DhP256HkdfSha256;
        let mut seed = [0u8; 32];
        getrandom::fill(&mut seed).unwrap();
        let sk = p256::SecretKey::from_slice(&{
            let mut s = seed;
            s[0] &= 0x7f;
            s[31] |= 1;
            s
        })
        .unwrap();
        let pem = sk.to_pkcs8_pem(Default::default()).unwrap().to_string();
        let pk =
            <Kem as hpke::Kem>::PublicKey::from_bytes(&sk.public_key().to_sec1_bytes()).unwrap();
        let mut rng = rand_core_rng();
        let (enc, ct) = hpke::single_shot_seal_with_rng::<
            hpke::aead::AesGcm256,
            hpke::kdf::HkdfSha256,
            Kem,
        >(&hpke::OpModeS::Base, &pk, b"", key, b"", &mut rng)
        .unwrap();
        use hpke::Serializable as _;
        let json = format!(
            "{{\"enc-request\": \"{}\", \"wrapped-key\": \"{}\"}}",
            B64.encode(enc.to_bytes()),
            B64.encode(ct)
        );
        (pem, json)
    }

    fn rand_core_rng() -> impl hpke::rand_core::CryptoRng {
        struct R;
        impl hpke::rand_core::TryRng for R {
            type Error = core::convert::Infallible;
            fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
                let mut b = [0u8; 4];
                getrandom::fill(&mut b).unwrap();
                Ok(u32::from_le_bytes(b))
            }
            fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
                let mut b = [0u8; 8];
                getrandom::fill(&mut b).unwrap();
                Ok(u64::from_le_bytes(b))
            }
            fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
                getrandom::fill(dst).unwrap();
                Ok(())
            }
        }
        impl hpke::rand_core::TryCryptoRng for R {}
        R
    }

    #[test]
    fn the_key_request_comes_from_the_auth_data_and_unwraps() {
        let key = [0x5au8; 32];
        let (pem, json) = sealed(&key);
        let url = "https://wkms-public.apple.com/fcs-keys/abc=";
        let p = prologue(
            1,
            &[
                ("com.apple.wkms.url", b"x"),
                ("com.apple.wkms.fcs-response", json.as_bytes()),
                ("com.apple.wkms.fcs-key-url", url.as_bytes()),
            ],
        );
        let pr = Prologue::read(&p).unwrap();
        assert_eq!(pr.profile, 1);
        let req = KeyRequest::from_prologue(&pr).unwrap();
        assert_eq!(req.url, url);
        assert_eq!(unwrap_key(&req, &pem).unwrap(), key);
        // Another private key doesn't open it.
        let (other, _) = sealed(&key);
        assert!(unwrap_key(&req, &other).is_err());
    }

    #[test]
    fn a_key_url_off_apples_key_service_is_refused() {
        for url in [
            "http://wkms-public.apple.com/fcs-keys/a",
            "https://example.com/fcs-keys/a",
            "https://wkms-public.apple.com.example.com/a",
        ] {
            let p = prologue(
                1,
                &[
                    ("com.apple.wkms.fcs-response", b"{}"),
                    ("com.apple.wkms.fcs-key-url", url.as_bytes()),
                ],
            );
            assert!(
                KeyRequest::from_prologue(&Prologue::read(&p).unwrap()).is_err(),
                "{url}"
            );
        }
    }

    #[test]
    fn damaged_auth_data_is_refused() {
        let mut p = prologue(1, &[("k", b"v")]);
        p[12] = 200;
        assert!(Prologue::read(&p).is_err());
        assert!(Prologue::read(&b"NOPE\0\0\0\0\0\0\0\0".to_vec()).is_err());
    }
}
