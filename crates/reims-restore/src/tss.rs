//! Apple's TSS (`gs.apple.com`), which signs the tickets a virtual Mac's macOS boots with, bound
//! to the VM's ECID. A restore needs these:
//! - the boot manifest, from the install identity (`macOS Customer`): what Preboot's
//!   `/<G>/boot/<H>/System/Library/Caches/apticket.der` holds, and the paired recoveryOS's;
//! - the SFR manifest, from the erase identity: iSCPreboot's `/SFR/current/apticket.der`, the
//!   ticket the restore OS itself boots with;
//! - the Cryptex1 ticket, for the OS, app and Rosetta cryptexes: Preboot's
//!   `/<G>/cryptex1/current/apticket.vma2macosap.<ECID>.im4m`, whose signature and UDID the
//!   guest's kernel checks at every boot. Its unpersonalized sibling beside it,
//!   `apticket.vma2macosap.im4m`, is not TSS's: it is the IPSW's
//!   `Firmware/Manifests/restore/cryptex1/macOS Customer/apticket.vma2macosap.im4m` as it is
//!   (the same bytes on 27.2.1);
//! - when Apple's restore OS runs the restore, what it asks the host to have signed: the paired
//!   recoveryOS's LocalPolicy manifest, iSCPreboot's
//!   `/3D3287DE-280D-4619-AAAB-D97469CA9C71/LocalPolicy/<hash>.recovery.img4`
//!   ([`TssClient::local_policy_ticket`]), and the requests it generates itself, Cryptex1's
//!   among them ([`TssClient::device_request`]). The macOS volume group's own LocalPolicy
//!   (`/<G>/LocalPolicy/<hash>.img4` and `.recovery.img4`) is not TSS's: its chain is another
//!   root's (`.img4 test secp384r1 Root Certificate Authority`, on 14.6.1).
//!
//! A request is an XML plist built from the build identity, as Virtualization.framework's restore
//! builds it: the identity's chip, board and security domain, its `Ap,*` (or `Cryptex1,*`)
//! values, the properties its `Info` asks the manifest to carry (`RequestManifestProperties`),
//! and one entry per component with its digest. What each request holds is set so that the
//! tickets TSS returns carry the same properties and objects as those of a fresh install by
//! Virtualization.framework of the same build (checked on 14.6.1 and 27.2.1):
//! - the boot manifest covers the install identity's trusted components but `BaseSystem` and the
//!   `Restore*` ones;
//! - the SFR manifest covers the erase identity's trusted components but `BaseSystem`;
//! - the Cryptex1 ticket covers the `Cryptex1,*` components, by digest alone;
//! - the recoveryOS's LocalPolicy manifest covers the LocalPolicy IM4P, and carries what the
//!   restore OS's arguments give (`nsih`, `ronh`, `vuid`).
//!
//! Every ticket is checked before it is returned: it parses, it is bound to the ECID (or UDID)
//! and chip and board asked for, it carries the nonces sent, and it holds each component asked
//! for with the digest sent.

use std::fmt;
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use plist::{Dictionary, Value};

use crate::img4::{self, Prop};
use crate::manifest::{Identity, parse_int};

/// TSS's endpoint for personalization requests.
pub const TSS_URL: &str = "https://gs.apple.com/TSS/controller?action=2";

/// The client version the request names, as the community's TSS clients send it.
const VERSION_INFO: &str = "libauthinstall-1104.0.9";

/// The SEP nonce of Virtualization.framework's tickets (their `snon`): 20 bytes of 0xAA.
pub const VZ_SEP_NONCE: [u8; 20] = [0xaa; 20];

/// The nonces a ticket is bound to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ApNonces {
    /// The AP nonce: the manifest's `BNCH`, and the Cryptex1 ticket's `cnch` where the same
    /// nonce is sent for both, as Virtualization.framework's restore does.
    pub ap_nonce: [u8; 32],
    /// The SEP nonce: the manifest's `snon`.
    pub sep_nonce: [u8; 20],
}

impl ApNonces {
    /// A random AP nonce and Virtualization.framework's SEP nonce.
    pub fn generate() -> Result<Self> {
        let mut ap_nonce = [0; 32];
        getrandom::fill(&mut ap_nonce)
            .map_err(|e| anyhow::anyhow!("no random bytes for the AP nonce: {e}"))?;
        Ok(Self {
            ap_nonce,
            sep_nonce: VZ_SEP_NONCE,
        })
    }
}

/// A random ECID, never 0.
pub fn random_ecid() -> Result<u64> {
    loop {
        let mut b = [0; 8];
        getrandom::fill(&mut b)
            .map_err(|e| anyhow::anyhow!("no random bytes for the ECID: {e}"))?;
        let ecid = u64::from_be_bytes(b);
        if ecid != 0 {
            return Ok(ecid);
        }
    }
}

/// The UDID a Cryptex1 ticket is bound to: the chip ID and the ECID, 8 bytes each, big-endian.
/// It is the byte-reversed form of the device tree's `unique-device-id` (ECID, then the chip ID
/// in 4 bytes, then 4 zero bytes, little-endian).
pub fn cryptex1_udid(chip_id: u64, ecid: u64) -> [u8; 16] {
    let mut udid = [0; 16];
    udid[..8].copy_from_slice(&chip_id.to_be_bytes());
    udid[8..].copy_from_slice(&ecid.to_be_bytes());
    udid
}

/// Which AP manifest to ask for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApTicket {
    /// The boot manifest, from the install identity (`manifest::VARIANT_INSTALL`).
    Boot,
    /// The SFR manifest, from the erase identity (`manifest::VARIANT_ERASE`).
    Sfr,
}

impl ApTicket {
    fn what(self) -> &'static str {
        match self {
            ApTicket::Boot => "boot",
            ApTicket::Sfr => "SFR",
        }
    }
}

/// Why a ticket couldn't be had. The errors [`TssClient`]'s calls return carry one of these, which
/// `anyhow::Error::downcast_ref` finds.
#[derive(Debug)]
pub enum TssError {
    /// TSS couldn't be reached, or answered without a status, on every attempt.
    Network {
        /// The endpoint.
        url: String,
        /// How many attempts were made.
        attempts: u32,
        /// The last attempt's failure.
        error: String,
    },
    /// Apple doesn't sign this build for the virtual Mac any more (TSS's status 94 or 69).
    NotSigned {
        /// The product type, `VirtualMac2,1`.
        product: String,
        /// The build and version, such as `26B131 (27.2.1)`.
        build: String,
        /// TSS's status.
        status: i64,
        /// TSS's message.
        message: String,
    },
    /// TSS refused the request with another status.
    Refused {
        /// TSS's status.
        status: i64,
        /// TSS's message.
        message: String,
    },
    /// TSS answered success with something that holds no ticket.
    BadResponse(String),
    /// The ticket TSS returned doesn't match what was asked for.
    BadTicket(String),
}

impl fmt::Display for TssError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TssError::Network {
                url,
                attempts,
                error,
            } => {
                write!(
                    f,
                    "couldn't reach Apple's TSS ({url}) in {attempts} attempts: {error}"
                )
            }
            TssError::NotSigned {
                product,
                build,
                status,
                message,
            } => write!(
                f,
                "Apple no longer signs build {build} for {product}, so it can't be installed: TSS answered status \
                 {status}, {message:?}"
            ),
            TssError::Refused { status, message } => {
                write!(
                    f,
                    "Apple's TSS refused the request: status {status}, {message:?}"
                )
            }
            TssError::BadResponse(what) => {
                write!(f, "Apple's TSS answered without a ticket: {what}")
            }
            TssError::BadTicket(what) => write!(
                f,
                "the ticket from Apple's TSS doesn't match the request: {what}"
            ),
        }
    }
}

impl std::error::Error for TssError {}

/// A client of Apple's TSS.
pub struct TssClient {
    url: String,
    agent: ureq::Agent,
    attempts: u32,
    backoff: Duration,
}

impl Default for TssClient {
    fn default() -> Self {
        Self::new()
    }
}

impl TssClient {
    /// A client of [`TSS_URL`]: 20 s to connect, 120 s for a whole exchange, and 4 attempts, 2 s
    /// apart and doubling, where TSS can't be reached or answers without a status.
    pub fn new() -> Self {
        Self::with_url(TSS_URL)
    }

    /// A client of another endpoint.
    pub fn with_url(url: impl Into<String>) -> Self {
        let roots = ureq::tls::RootCerts::new_with_certs(&[apple_root_ca()]);
        let config = ureq::Agent::config_builder()
            .tls_config(ureq::tls::TlsConfig::builder().root_certs(roots).build())
            .user_agent(crate::USER_AGENT)
            .timeout_connect(Some(Duration::from_secs(20)))
            .timeout_global(Some(Duration::from_secs(120)))
            .http_status_as_error(false)
            .build();
        Self {
            url: url.into(),
            agent: config.into(),
            attempts: 4,
            backoff: Duration::from_secs(2),
        }
    }

    /// Sets how many attempts a request gets, and the wait before the second (doubling after).
    pub fn retries(mut self, attempts: u32, backoff: Duration) -> Self {
        self.attempts = attempts.max(1);
        self.backoff = backoff;
        self
    }

    /// The boot or SFR manifest for `ecid`, from `id` (the install identity for
    /// [`ApTicket::Boot`], the erase identity for [`ApTicket::Sfr`]), as IM4M DER.
    pub fn ap_ticket(
        &self,
        id: &Identity,
        kind: ApTicket,
        ecid: u64,
        n: &ApNonces,
    ) -> Result<Vec<u8>> {
        let req = ap_request(id, kind, ecid, n)?;
        self.ticket(id, &req)
            .with_context(|| format!("the {} manifest", kind.what()))
    }

    /// The Cryptex1 ticket for `ecid`, from the install identity `id`, as IM4M DER.
    pub fn cryptex1_ticket(&self, id: &Identity, ecid: u64, nonce: &[u8; 32]) -> Result<Vec<u8>> {
        let req = cryptex1_request(id, ecid, nonce)?;
        self.ticket(id, &req).context("the Cryptex1 ticket")
    }

    /// A LocalPolicy's manifest for `ecid`, as IM4M DER, for the restore OS's
    /// `RecoveryOSLocalPolicy` request: `args` are the request's `Arguments`, whose
    /// `Ap,NextStageIM4MHash`, `Ap,RecoveryOSPolicyNonceHash` and `Ap,VolumeUUID` (and
    /// `Ap,LocalBoot`, `ApNonce`, where present) go into the request. The LocalPolicy itself is
    /// `img4::stitch_img4(&img4::LPOL_IM4P, &ticket)`.
    pub fn local_policy_ticket(
        &self,
        id: &Identity,
        args: &Dictionary,
        ecid: u64,
    ) -> Result<Vec<u8>> {
        let req = local_policy_request(id, args, ecid)?;
        self.ticket(id, &req).context("the LocalPolicy's manifest")
    }

    /// TSS's answer to a request the restore OS generated itself (a `FirmwareUpdaterData`
    /// request's `Arguments`, as `Cryptex1` and `Cryptex1LocalPolicy` send them), completed from
    /// `id` as Apple's restore host completes it. The ticket under the request's response tag is
    /// checked, then the whole answer returned.
    pub fn device_request(&self, id: &Identity, arguments: &Dictionary) -> Result<Dictionary> {
        let req = device_request(id, arguments)?;
        let what = arguments
            .get("MessageArgUpdaterName")
            .and_then(Value::as_string)
            .unwrap_or("a device-generated request");
        let answer = self
            .answer(id, &req)
            .with_context(|| format!("TSS's answer to {what}"))?;
        Ok(answer)
    }

    fn ticket(&self, id: &Identity, req: &Request) -> Result<Vec<u8>> {
        let answer = self.answer(id, req)?;
        Ok(answer
            .get(req.ticket_key.as_str())
            .and_then(Value::as_data)
            .unwrap_or_default()
            .to_vec())
    }

    /// Sends `req` and checks the ticket in TSS's answer.
    fn answer(&self, id: &Identity, req: &Request) -> Result<Dictionary> {
        let answer = match self.send(&req.body) {
            Err(TssError::Refused {
                status: status @ (69 | 94),
                message,
            }) => {
                return Err(TssError::NotSigned {
                    product: id
                        .dict()
                        .get("Ap,ProductType")
                        .and_then(Value::as_string)
                        .unwrap_or("VirtualMac2,1")
                        .to_owned(),
                    build: build_name(id),
                    status,
                    message,
                }
                .into());
            }
            r => r?,
        };
        let ticket = answer
            .get(req.ticket_key.as_str())
            .and_then(Value::as_data)
            .ok_or_else(|| TssError::BadResponse(format!("no {} in its answer", req.ticket_key)))?;
        if let Some(expected) = &req.expected {
            expected.check(ticket)?;
        }
        Ok(answer)
    }

    /// Sends a request and returns TSS's answer, the plist after `REQUEST_STRING=`.
    pub fn send(&self, body: &Dictionary) -> Result<Dictionary, TssError> {
        let mut xml = Vec::new();
        Value::Dictionary(body.clone())
            .to_writer_xml(&mut xml)
            .map_err(|e| TssError::BadResponse(format!("the request didn't encode: {e}")))?;
        let mut wait = self.backoff;
        let mut last = String::new();
        for attempt in 1..=self.attempts {
            if attempt > 1 {
                std::thread::sleep(wait);
                wait = wait.saturating_mul(2);
            }
            match self.post(&xml) {
                Ok(text) => match parse_response(&text)? {
                    Some(answer) => return Ok(answer),
                    None => {
                        last = format!(
                            "an answer with no status: {:?}",
                            text.chars().take(200).collect::<String>()
                        )
                    }
                },
                Err(e) => last = e,
            }
        }
        Err(TssError::Network {
            url: self.url.clone(),
            attempts: self.attempts,
            error: last,
        })
    }

    /// One POST; the body's text, or why there is none.
    fn post(&self, xml: &[u8]) -> Result<String, String> {
        let mut response = self
            .agent
            .post(&self.url)
            .header("Cache-Control", "no-cache")
            .header("Content-Type", "text/xml; charset=\"utf-8\"")
            .send(xml)
            .map_err(|e| e.to_string())?;
        let status = response.status();
        let text = response
            .body_mut()
            .read_to_string()
            .map_err(|e| format!("reading the answer: {e}"))?;
        if status.is_success() {
            Ok(text)
        } else {
            Err(format!(
                "HTTP {status}: {:?}",
                text.chars().take(200).collect::<String>()
            ))
        }
    }
}

/// The PEM of Apple Root CA, the root of `gs.apple.com`'s certificate chain (under it, Apple
/// Server Authentication CA). Mozilla's roots, which TLS clients bring, don't hold it, so TSS's
/// TLS is verified against it alone.
const APPLE_ROOT_CA_PEM: &str = include_str!("apple-root-ca.pem");

fn apple_root_ca() -> ureq::tls::Certificate<'static> {
    let pem = &APPLE_ROOT_CA_PEM[APPLE_ROOT_CA_PEM.find("-----BEGIN").unwrap_or(0)..];
    ureq::tls::Certificate::from_pem(pem.as_bytes()).expect("apple-root-ca.pem holds a certificate")
}

/// A TSS answer's plist: `STATUS=0&MESSAGE=SUCCESS&REQUEST_STRING=<plist>`. `None` where the text
/// has no status (a proxy's page, a cut-off answer), which is worth another attempt.
fn parse_response(text: &str) -> Result<Option<Dictionary>, TssError> {
    let Some(at) = text.find("STATUS=") else {
        return Ok(None);
    };
    let digits: String = text[at + 7..]
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '-')
        .collect();
    let Ok(status) = digits.parse::<i64>() else {
        return Ok(None);
    };
    let message = text
        .find("MESSAGE=")
        .map(|m| text[m + 8..].split('&').next().unwrap_or("").to_owned())
        .unwrap_or_default();
    if status != 0 {
        return Err(TssError::Refused { status, message });
    }
    let xml = text
        .find("REQUEST_STRING=")
        .map(|r| &text[r + 15..])
        .or_else(|| text.find("<?xml").map(|x| &text[x..]))
        .ok_or_else(|| TssError::BadResponse("status 0 with no REQUEST_STRING".into()))?;
    let value = Value::from_reader_xml(xml.as_bytes())
        .map_err(|e| TssError::BadResponse(format!("its REQUEST_STRING is not a plist: {e}")))?;
    value
        .into_dictionary()
        .map(Some)
        .ok_or_else(|| TssError::BadResponse("its REQUEST_STRING is not a dictionary".into()))
}

/// A request and what its ticket must hold.
pub struct Request {
    /// The plist sent.
    pub body: Dictionary,
    /// The answer's key that holds the ticket.
    pub ticket_key: String,
    /// What the ticket must hold, for an Image4 ticket.
    pub expected: Option<Expected>,
}

/// What a ticket must hold.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Expected {
    /// MANP properties and their values.
    pub props: Vec<(&'static str, Prop)>,
    /// Each component asked for: its name in the request and its digest.
    pub objects: Vec<(String, Vec<u8>)>,
}

impl Expected {
    /// What a request's ticket must hold: the chip, board and ECID it names (the UDID, for a
    /// Cryptex1 ticket), the nonces it sends, and an object with each digest it sends.
    pub fn of(body: &Dictionary) -> Self {
        let int = |k: &str| body.get(k).and_then(Value::as_unsigned_integer);
        let data = |k: &str| body.get(k).and_then(Value::as_data).map(<[u8]>::to_vec);
        let mut props = Vec::new();
        for (key, tag) in [("ApChipID", "CHIP"), ("ApBoardID", "BORD")] {
            if let Some(v) = int(key) {
                props.push((tag, Prop::Int(v)));
            }
        }
        // A Cryptex1 ticket names the device by its UDID alone.
        if !body.contains_key("@Cryptex1,Ticket")
            && let Some(v) = int("ApECID")
        {
            props.push(("ECID", Prop::Int(v)));
        }
        for (key, tag) in [
            ("ApNonce", "BNCH"),
            ("SepNonce", "snon"),
            ("Cryptex1,UDID", "UDID"),
            ("Cryptex1,Nonce", "cnch"),
        ] {
            if let Some(v) = data(key) {
                props.push((tag, Prop::Data(v)));
            }
        }
        let objects = body
            .iter()
            .filter_map(|(k, v)| {
                Some((
                    k.clone(),
                    v.as_dictionary()?.get("Digest")?.as_data()?.to_vec(),
                ))
            })
            .collect();
        Self { props, objects }
    }

    /// Checks an IM4M against what was asked for.
    pub fn check(&self, ticket: &[u8]) -> Result<(), TssError> {
        let m = img4::parse_im4m(ticket)
            .map_err(|e| TssError::BadTicket(format!("it doesn't parse: {e:#}")))?;
        for (key, want) in &self.props {
            match m.props.get(*key) {
                Some(got) if got == want => {}
                Some(got) => {
                    return Err(TssError::BadTicket(format!(
                        "its {key} is {got:?}, not {want:?}"
                    )));
                }
                None => return Err(TssError::BadTicket(format!("it has no {key}"))),
            }
        }
        for (name, digest) in &self.objects {
            let found = match img4::component_tag(name) {
                Some(tag) => m.objects.get(tag).map(|o| (tag.to_owned(), o)),
                None => m
                    .objects
                    .iter()
                    .find(|(_, o)| o.get("DGST").and_then(Prop::as_data) == Some(digest))
                    .map(|(t, o)| (t.clone(), o)),
            };
            let Some((tag, object)) = found else {
                return Err(TssError::BadTicket(format!("it has no object for {name}")));
            };
            if !digest.is_empty() && object.get("DGST").and_then(Prop::as_data) != Some(digest) {
                return Err(TssError::BadTicket(format!(
                    "its {tag} ({name}) has another digest than the one asked for"
                )));
            }
        }
        Ok(())
    }
}

/// The request for an AP manifest.
pub fn ap_request(id: &Identity, kind: ApTicket, ecid: u64, n: &ApNonces) -> Result<Request> {
    let mut body = client_tags();
    common_tags(&mut body, id, ecid)?;
    body.insert("@ApImg4Ticket".into(), Value::Boolean(true));
    body.insert("ApNonce".into(), Value::Data(n.ap_nonce.to_vec()));
    body.insert("SepNonce".into(), Value::Data(n.sep_nonce.to_vec()));
    for (k, v) in id.dict() {
        if k.starts_with("Ap,") {
            body.insert(k.clone(), tss_value(v));
        }
    }
    // What the identity asks the manifest to carry, such as Ap,Timestamp and Ap,VM: the identity's
    // own value, else its Info's (Ap,Timestamp's 0 has TSS stamp the ticket, its `tstp`).
    if let Some(Value::Array(keys)) = id.info("RequestManifestProperties") {
        for key in keys.iter().filter_map(Value::as_string) {
            let v = id.dict().get(key).or_else(|| id.info(key)).with_context(|| {
                format!("the identity asks the manifest to carry {key}, which neither it nor its Info has")
            })?;
            body.insert(key.to_owned(), tss_value(v));
        }
    }
    let mut any = false;
    for (name, comp) in id.components() {
        let wanted = comp.trusted()
            && !name.starts_with("Cryptex1,")
            && name != "BaseSystem"
            && (kind == ApTicket::Sfr || !name.starts_with("Restore"));
        if !wanted {
            continue;
        }
        let mut entry = comp.dict().clone();
        entry.remove("Info");
        if let Some(Value::Array(rules)) = comp.info().and_then(|i| i.get("RestoreRequestRules")) {
            apply_rules(&mut entry, rules);
        }
        if comp.digest().is_none() {
            entry.insert("Digest".into(), Value::Data(Vec::new()));
        }
        body.insert(name.to_owned(), Value::Dictionary(entry));
        any = true;
    }
    ensure!(any, "the identity has no trusted components to personalize");
    let expected = Some(Expected::of(&body));
    Ok(Request {
        body,
        ticket_key: "ApImg4Ticket".into(),
        expected,
    })
}

/// The request for a Cryptex1 ticket, as the restore OS generates it. TSS answers a Cryptex1
/// request without `Cryptex1,ProductionMode` with status 168, "An internal error occurred.", and
/// one with `ApSecurityMode` false or with its components marked `Trusted` with status 94
/// (measured on 14.6.1 and 27.2.1). `Cryptex1,UseProductClass` would add `upcl`, which
/// Virtualization.framework's tickets don't carry, so it isn't sent.
pub fn cryptex1_request(id: &Identity, ecid: u64, nonce: &[u8; 32]) -> Result<Request> {
    let mut body = client_tags();
    common_tags(&mut body, id, ecid)?;
    body.insert("@Cryptex1,Ticket".into(), Value::Boolean(true));
    body.insert("Cryptex1,ProductionMode".into(), Value::Boolean(true));
    for (k, v) in id.dict() {
        if k.starts_with("Cryptex1,") {
            body.insert(k.clone(), tss_value(v));
        }
    }
    let udid = cryptex1_udid(id.int("ApChipID")?, ecid);
    body.insert("Cryptex1,Nonce".into(), Value::Data(nonce.to_vec()));
    body.insert("Cryptex1,UDID".into(), Value::Data(udid.to_vec()));
    let mut any = false;
    for (name, comp) in id.components() {
        if !name.starts_with("Cryptex1,") {
            continue;
        }
        let digest = comp
            .digest()
            .with_context(|| format!("{name} has no digest"))?;
        let mut entry = Dictionary::new();
        entry.insert("Digest".into(), Value::Data(digest.to_vec()));
        body.insert(name.to_owned(), Value::Dictionary(entry));
        any = true;
    }
    ensure!(
        any,
        "the identity has no Cryptex1 components: is it a macOS 13 or later IPSW?"
    );
    let expected = Some(Expected::of(&body));
    Ok(Request {
        body,
        ticket_key: "Cryptex1,Ticket".into(),
        expected,
    })
}

/// The request for a LocalPolicy's manifest (the restore OS's `RecoveryOSLocalPolicy`), from the
/// request's `Arguments`.
pub fn local_policy_request(id: &Identity, args: &Dictionary, ecid: u64) -> Result<Request> {
    let mut params = Dictionary::new();
    params.insert("ApECID".into(), Value::Integer(ecid.into()));
    for key in ["ApChipID", "ApBoardID", "ApSecurityDomain"] {
        params.insert(key.into(), Value::Integer(id.int(key)?.into()));
    }
    params.insert(
        "Ap,LocalBoot".into(),
        args.get("Ap,LocalBoot")
            .cloned()
            .unwrap_or(Value::Boolean(true)),
    );
    params.insert("Ap,LocalPolicy".into(), local_policy_entry());
    for key in [
        "Ap,NextStageIM4MHash",
        "Ap,RecoveryOSPolicyNonceHash",
        "ApNonce",
    ] {
        if let Some(v) = args.get(key) {
            params.insert(key.into(), v.clone());
        }
    }
    if let Some(v) = args.get("Ap,VolumeUUID") {
        params.insert(
            "Ap,VolumeUUID".into(),
            Value::Data(volume_uuid(v)?.to_vec()),
        );
    }
    let mut body = client_tags();
    local_policy_tags(&mut body, &params)?;
    let expected = Some(Expected::of(&body));
    Ok(Request {
        body,
        ticket_key: "ApImg4Ticket".into(),
        expected,
    })
}

/// The request for a `FirmwareUpdaterData` request's `Arguments`: for `Cryptex1` and
/// `Cryptex1LocalPolicy`, the request the restore OS generated (`DeviceGeneratedRequest`) over
/// what it reports (`MessageArgInfo`) and the identity's values it names
/// (`DeviceGeneratedTags.BuildIdentityTags`); for another updater, the generated request as it is.
/// The answer's ticket is under the first of `DeviceGeneratedTags.ResponseTags`.
pub fn device_request(id: &Identity, arguments: &Dictionary) -> Result<Request> {
    let updater = arguments
        .get("MessageArgUpdaterName")
        .and_then(Value::as_string)
        .unwrap_or_default();
    let tags = arguments
        .get("DeviceGeneratedTags")
        .and_then(Value::as_dictionary);
    let response_tag = tags
        .and_then(|t| t.get("ResponseTags"))
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .and_then(Value::as_string);
    let generated = arguments
        .get("DeviceGeneratedRequest")
        .and_then(Value::as_dictionary)
        .context("the restore OS's request has no DeviceGeneratedRequest")?;
    let mut body = client_tags();
    if !matches!(updater, "Cryptex1" | "Cryptex1LocalPolicy") {
        for (k, v) in generated {
            body.insert(k.clone(), v.clone());
        }
        let ticket_key = response_tag
            .with_context(|| format!("the {updater:?} request names no response tag"))?;
        return Ok(Request {
            body,
            ticket_key: ticket_key.to_owned(),
            expected: None,
        });
    }

    let mut params = arguments
        .get("MessageArgInfo")
        .and_then(Value::as_dictionary)
        .cloned()
        .unwrap_or_default();
    for key in tags
        .and_then(|t| t.get("BuildIdentityTags"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let Some(key) = key.as_string() else { continue };
        if let Some(v) = id.dict().get(key) {
            params.insert(key.to_owned(), tss_value(v));
        }
    }
    for key in ["ApProductionMode", "ApSecurityMode"] {
        if !params.contains_key(key) {
            params.insert(key.into(), Value::Boolean(true));
        }
    }
    for key in ["ApChipID", "ApBoardID"] {
        if !params.contains_key(key) {
            params.insert(key.into(), Value::Integer(id.int(key)?.into()));
        }
    }
    for (k, v) in generated {
        params.insert(k.clone(), v.clone());
    }
    for key in [
        "ApECID",
        "UniqueBuildID",
        "ApChipID",
        "ApBoardID",
        "ApSecurityDomain",
    ] {
        if let Some(v) = params.get(key) {
            body.insert(key.into(), tss_value(v));
        }
    }
    let default_tag = if params.contains_key("Ap,LocalPolicy") {
        local_policy_tags(&mut body, &params)?;
        if let Some(v) = params.get("Ap,NextStageCryptex1IM4MHash") {
            body.insert("Ap,NextStageCryptex1IM4MHash".into(), v.clone());
        }
        "ApImg4Ticket"
    } else {
        body.insert("@Cryptex1,Ticket".into(), Value::Boolean(true));
        for key in ["ApSecurityMode", "ApProductionMode"] {
            body.insert(key.into(), params[key].clone());
        }
        // TSS refuses a Cryptex1 request without it (status 168); see [`cryptex1_request`].
        body.insert(
            "Cryptex1,ProductionMode".into(),
            params["ApProductionMode"].clone(),
        );
        for (k, v) in &params {
            if k.starts_with("Cryptex1") {
                body.insert(k.clone(), tss_value(v));
            }
        }
        "Cryptex1,Ticket"
    };
    let expected = Some(Expected::of(&body));
    Ok(Request {
        body,
        ticket_key: response_tag.unwrap_or(default_tag).to_owned(),
        expected,
    })
}

/// The tags that open every request: who asks, and the request's own UUID.
fn client_tags() -> Dictionary {
    let mut body = Dictionary::new();
    body.insert("@HostPlatformInfo".into(), Value::String("mac".into()));
    body.insert("@VersionInfo".into(), Value::String(VERSION_INFO.into()));
    body.insert(
        "@UUID".into(),
        Value::String(uuid::Uuid::new_v4().hyphenated().to_string().to_uppercase()),
    );
    body
}

/// The identity's chip, board and security domain, its build, the ECID, and production and
/// security mode on.
fn common_tags(body: &mut Dictionary, id: &Identity, ecid: u64) -> Result<()> {
    body.insert("ApECID".into(), Value::Integer(ecid.into()));
    let build = id
        .dict()
        .get("UniqueBuildID")
        .and_then(Value::as_data)
        .context("the identity has no UniqueBuildID")?;
    body.insert("UniqueBuildID".into(), Value::Data(build.to_vec()));
    for key in ["ApChipID", "ApBoardID", "ApSecurityDomain"] {
        body.insert(key.into(), Value::Integer(id.int(key)?.into()));
    }
    body.insert("ApProductionMode".into(), Value::Boolean(true));
    body.insert("ApSecurityMode".into(), Value::Boolean(true));
    Ok(())
}

/// A LocalPolicy's tags, from `params`.
fn local_policy_tags(body: &mut Dictionary, params: &Dictionary) -> Result<()> {
    body.insert("@ApImg4Ticket".into(), Value::Boolean(true));
    for key in ["Ap,LocalBoot", "Ap,LocalPolicy", "Ap,NextStageIM4MHash"] {
        let v = params
            .get(key)
            .with_context(|| format!("a LocalPolicy request needs {key}"))?;
        body.insert(key.into(), v.clone());
    }
    for key in ["Ap,RecoveryOSPolicyNonceHash", "Ap,VolumeUUID", "ApNonce"] {
        if let Some(v) = params.get(key) {
            body.insert(key.into(), v.clone());
        }
    }
    for key in ["ApECID", "ApChipID", "ApBoardID", "ApSecurityDomain"] {
        if let Some(v) = params.get(key) {
            body.insert(key.into(), tss_value(v));
        }
    }
    for key in ["ApSecurityMode", "ApProductionMode"] {
        body.insert(
            key.into(),
            params.get(key).cloned().unwrap_or(Value::Boolean(true)),
        );
    }
    Ok(())
}

/// The `Ap,LocalPolicy` entry: the LocalPolicy IM4P's digest, trusted.
fn local_policy_entry() -> Value {
    let mut lpol = Dictionary::new();
    lpol.insert(
        "Digest".into(),
        Value::Data(img4::sha384(&img4::LPOL_IM4P).to_vec()),
    );
    lpol.insert("Trusted".into(), Value::Boolean(true));
    Value::Dictionary(lpol)
}

/// A volume UUID as the restore OS gives it (text, or 16 bytes), as 16 bytes.
fn volume_uuid(v: &Value) -> Result<[u8; 16]> {
    match v {
        Value::String(s) => Ok(*uuid::Uuid::parse_str(s)
            .with_context(|| format!("Ap,VolumeUUID {s:?}"))?
            .as_bytes()),
        Value::Data(d) => d
            .as_slice()
            .try_into()
            .with_context(|| format!("Ap,VolumeUUID of {} bytes", d.len())),
        _ => bail!("Ap,VolumeUUID is neither text nor data"),
    }
}

/// An identity's value as TSS takes it: a hex string such as `0x01` becomes an integer.
fn tss_value(v: &Value) -> Value {
    match v {
        Value::String(s) if s.starts_with("0x") || s.starts_with("0X") => parse_int(v)
            .map(|i| Value::Integer(i.into()))
            .unwrap_or_else(|| v.clone()),
        _ => v.clone(),
    }
}

/// Applies a component's `RestoreRequestRules` to its entry, for a production device in full
/// security asking for an Image4 ticket: each rule whose conditions all hold sets its actions'
/// booleans (`EPRO`, `ESEC`). A condition on anything else (a demotion, DFU) doesn't hold.
fn apply_rules(entry: &mut Dictionary, rules: &[Value]) {
    for rule in rules.iter().filter_map(Value::as_dictionary) {
        let holds = rule
            .get("Conditions")
            .and_then(Value::as_dictionary)
            .is_some_and(|c| {
                c.iter().all(|(k, v)| match k.as_str() {
                    "ApRawProductionMode"
                    | "ApCurrentProductionMode"
                    | "ApRawSecurityMode"
                    | "ApRequiresImage4" => v.as_boolean() == Some(true),
                    _ => false,
                })
            });
        if !holds {
            continue;
        }
        for (k, v) in rule
            .get("Actions")
            .and_then(Value::as_dictionary)
            .into_iter()
            .flatten()
        {
            if let Some(b) = v.as_boolean() {
                entry.insert(k.clone(), Value::Boolean(b));
            }
        }
    }
}

/// The identity's build, such as `26B131 (27.2.1)`.
fn build_name(id: &Identity) -> String {
    let build = id.info_str("BuildNumber").unwrap_or("?");
    match id
        .dict()
        .get("Ap,ProductMarketingVersion")
        .or_else(|| id.dict().get("ProductMarketingVersion"))
        .and_then(Value::as_string)
    {
        Some(v) => format!("{build} ({v})"),
        None => build.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    use super::*;

    #[test]
    fn tss_is_sent_only_the_pinned_headers() {
        let h = crate::sent_headers::of(|url| {
            TssClient::with_url(url).post(b"<plist/>").unwrap();
        });
        crate::sent_headers::assert_pinned(
            &h,
            &["cache-control", "content-type", "content-length"],
        );
        let ct = h
            .iter()
            .find(|(k, _)| k == "content-type")
            .map(|(_, v)| v.as_str());
        assert_eq!(ct, Some("text/xml; charset=\"utf-8\""));
    }
    use crate::manifest::{BuildManifest, DEVICE_CLASS, VARIANT_ERASE, VARIANT_INSTALL};

    fn rules(current: bool) -> Value {
        let mode = if current {
            "ApCurrentProductionMode"
        } else {
            "ApRawProductionMode"
        };
        let rule = |cond: &[(&str, Value)], action: (&str, bool)| {
            let mut c = Dictionary::new();
            for (k, v) in cond {
                c.insert((*k).into(), v.clone());
            }
            let mut a = Dictionary::new();
            a.insert(action.0.into(), Value::Boolean(action.1));
            let mut r = Dictionary::new();
            r.insert("Conditions".into(), Value::Dictionary(c));
            r.insert("Actions".into(), Value::Dictionary(a));
            Value::Dictionary(r)
        };
        let t = Value::Boolean(true);
        let f = Value::Boolean(false);
        Value::Array(vec![
            rule(
                &[(mode, f.clone()), ("ApRequiresImage4", t.clone())],
                ("EPRO", false),
            ),
            rule(
                &[(mode, t.clone()), ("ApRequiresImage4", t.clone())],
                ("EPRO", true),
            ),
            rule(
                &[
                    (mode, t.clone()),
                    ("ApDemotionPolicyOverride", "Demote".into()),
                    ("ApRequiresImage4", t.clone()),
                ],
                ("EPRO", false),
            ),
            rule(
                &[("ApRawSecurityMode", f), ("ApRequiresImage4", t.clone())],
                ("ESEC", false),
            ),
            rule(
                &[("ApRawSecurityMode", t.clone()), ("ApRequiresImage4", t)],
                ("ESEC", true),
            ),
        ])
    }

    fn comp(path: &str, digest: u8, trusted: bool) -> Value {
        let mut info = Dictionary::new();
        info.insert("Path".into(), Value::String(path.into()));
        info.insert(
            "RestoreRequestRules".into(),
            rules(path.contains("Restore")),
        );
        let mut d = Dictionary::new();
        d.insert("Digest".into(), Value::Data(vec![digest; 48]));
        if trusted {
            d.insert("Trusted".into(), Value::Boolean(true));
        }
        d.insert("Info".into(), Value::Dictionary(info));
        Value::Dictionary(d)
    }

    /// A BuildManifest with a virtual Mac's install and erase identities, written from scratch.
    fn manifest() -> Vec<u8> {
        let ident = |variant: &str, erase: bool| {
            let mut info = Dictionary::new();
            info.insert("DeviceClass".into(), Value::String(DEVICE_CLASS.into()));
            info.insert("Variant".into(), Value::String(variant.into()));
            info.insert("BuildNumber".into(), Value::String("99A1".into()));
            info.insert("Ap,Timestamp".into(), Value::Integer(0.into()));
            info.insert("Ap,VM".into(), Value::String("0x01".into()));
            info.insert(
                "RequestManifestProperties".into(),
                Value::Array(vec![
                    Value::String("Ap,Timestamp".into()),
                    Value::String("Ap,VM".into()),
                ]),
            );
            let mut m = Dictionary::new();
            m.insert("KernelCache".into(), comp("kernelcache.test", 1, true));
            m.insert("DeviceTree".into(), comp("Firmware/dt.im4p", 2, true));
            m.insert("BaseSystem".into(), comp("base.dmg", 3, true));
            m.insert(
                "RestoreDeviceTree".into(),
                comp("Firmware/RestoreDT.im4p", 4, true),
            );
            m.insert("NewThing".into(), comp("Firmware/new.im4p", 5, true));
            m.insert("Cryptex1,SystemOS".into(), comp("os.dmg", 6, false));
            m.insert(
                "Cryptex1,SystemTrustCache".into(),
                comp("os.trustcache", 7, false),
            );
            if erase {
                m.insert("iBSS".into(), comp("Firmware/ibss.im4p", 8, true));
            }
            let mut d = Dictionary::new();
            d.insert("ApChipID".into(), Value::String("0xFE00".into()));
            d.insert("ApBoardID".into(), Value::String("0x20".into()));
            d.insert("ApSecurityDomain".into(), Value::String("0x01".into()));
            d.insert("UniqueBuildID".into(), Value::Data(vec![0x42; 20]));
            d.insert(
                "Ap,OSLongVersion".into(),
                Value::String("99.1.0.0.0,0".into()),
            );
            d.insert(
                "Ap,ProductType".into(),
                Value::String("VirtualMac2,1".into()),
            );
            d.insert(
                "Ap,ProductMarketingVersion".into(),
                Value::String("99.1".into()),
            );
            d.insert("Ap,VM".into(), Value::String("0x01".into()));
            d.insert("Cryptex1,ChipID".into(), Value::String("0xFF10".into()));
            d.insert("Cryptex1,NonceDomain".into(), Value::Integer(0.into()));
            d.insert(
                "Cryptex1,Version".into(),
                Value::String("99.1.0.0.0,0".into()),
            );
            d.insert("Info".into(), Value::Dictionary(info));
            d.insert("Manifest".into(), Value::Dictionary(m));
            Value::Dictionary(d)
        };
        let mut root = Dictionary::new();
        root.insert("ProductVersion".into(), Value::String("99.1".into()));
        root.insert("ProductBuildVersion".into(), Value::String("99A1".into()));
        root.insert(
            "BuildIdentities".into(),
            Value::Array(vec![
                ident(VARIANT_ERASE, true),
                ident(VARIANT_INSTALL, false),
            ]),
        );
        let mut out = Vec::new();
        Value::Dictionary(root).to_writer_xml(&mut out).unwrap();
        out
    }

    const ECID: u64 = 0xc208_f7ea_febd_c581;
    const NONCES: ApNonces = ApNonces {
        ap_nonce: [0x11; 32],
        sep_nonce: VZ_SEP_NONCE,
    };

    fn keys(d: &Dictionary) -> Vec<&str> {
        let mut k: Vec<&str> = d.keys().map(String::as_str).collect();
        k.sort_unstable();
        k
    }

    fn names(e: &Expected) -> Vec<&str> {
        let mut n: Vec<&str> = e.objects.iter().map(|(n, _)| n.as_str()).collect();
        n.sort_unstable();
        n
    }

    #[test]
    fn boot_requests_carry_the_installed_os_components() {
        let bm = BuildManifest::parse(&manifest()).unwrap();
        let id = bm.vma2_identity(VARIANT_INSTALL).unwrap();
        let req = ap_request(&id, ApTicket::Boot, ECID, &NONCES).unwrap();
        let b = &req.body;
        assert_eq!(b["ApECID"].as_unsigned_integer(), Some(ECID));
        assert_eq!(b["ApChipID"].as_unsigned_integer(), Some(0xfe00));
        assert_eq!(b["ApBoardID"].as_unsigned_integer(), Some(0x20));
        assert_eq!(b["ApSecurityDomain"].as_unsigned_integer(), Some(1));
        assert_eq!(b["ApNonce"].as_data(), Some(&[0x11; 32][..]));
        assert_eq!(b["SepNonce"].as_data(), Some(&[0xaa; 20][..]));
        assert_eq!(b["@ApImg4Ticket"].as_boolean(), Some(true));
        assert_eq!(b["ApProductionMode"].as_boolean(), Some(true));
        assert_eq!(b["ApSecurityMode"].as_boolean(), Some(true));
        assert_eq!(b["UniqueBuildID"].as_data(), Some(&[0x42; 20][..]));
        assert_eq!(b["Ap,OSLongVersion"].as_string(), Some("99.1.0.0.0,0"));
        assert_eq!(
            b["Ap,VM"].as_unsigned_integer(),
            Some(1),
            "a hex string goes as an integer"
        );
        assert_eq!(
            b["Ap,Timestamp"].as_unsigned_integer(),
            Some(0),
            "Info's RequestManifestProperties"
        );
        assert!(!b.contains_key("Cryptex1,ChipID"));
        // Trusted components, but BaseSystem, Restore* and Cryptex1,*.
        let comps: Vec<&str> = keys(b)
            .into_iter()
            .filter(|k| b[*k].as_dictionary().is_some())
            .collect();
        assert_eq!(comps, ["DeviceTree", "KernelCache", "NewThing"]);
        let expected = req.expected.as_ref().unwrap();
        let krnl = b["KernelCache"].as_dictionary().unwrap();
        assert_eq!(keys(krnl), ["Digest", "EPRO", "ESEC", "Trusted"]);
        assert_eq!(krnl["EPRO"].as_boolean(), Some(true));
        assert_eq!(krnl["ESEC"].as_boolean(), Some(true));
        assert_eq!(
            expected.props,
            [
                ("CHIP", Prop::Int(0xfe00)),
                ("BORD", Prop::Int(0x20)),
                ("ECID", Prop::Int(ECID)),
                ("BNCH", Prop::Data(vec![0x11; 32])),
                ("snon", Prop::Data(vec![0xaa; 20])),
            ]
        );
        assert_eq!(names(expected), ["DeviceTree", "KernelCache", "NewThing"]);
    }

    #[test]
    fn sfr_requests_carry_the_restore_components_too() {
        let bm = BuildManifest::parse(&manifest()).unwrap();
        let id = bm.vma2_identity(VARIANT_ERASE).unwrap();
        let req = ap_request(&id, ApTicket::Sfr, ECID, &NONCES).unwrap();
        assert_eq!(
            names(req.expected.as_ref().unwrap()),
            [
                "DeviceTree",
                "KernelCache",
                "NewThing",
                "RestoreDeviceTree",
                "iBSS"
            ]
        );
        // The restore components' rules key on the current production mode; they apply too.
        let rdtr = req.body["RestoreDeviceTree"].as_dictionary().unwrap();
        assert_eq!(rdtr["EPRO"].as_boolean(), Some(true));
    }

    #[test]
    fn cryptex1_requests_carry_the_cryptexes_by_digest() {
        let bm = BuildManifest::parse(&manifest()).unwrap();
        let id = bm.vma2_identity(VARIANT_INSTALL).unwrap();
        let req = cryptex1_request(&id, ECID, &[0x22; 32]).unwrap();
        let b = &req.body;
        assert_eq!(b["@Cryptex1,Ticket"].as_boolean(), Some(true));
        assert!(!b.contains_key("@ApImg4Ticket"));
        assert_eq!(b["Cryptex1,ProductionMode"].as_boolean(), Some(true));
        assert!(!b.contains_key("Cryptex1,UseProductClass"));
        assert_eq!(b["Cryptex1,ChipID"].as_unsigned_integer(), Some(0xff10));
        assert_eq!(b["Cryptex1,NonceDomain"].as_unsigned_integer(), Some(0));
        assert_eq!(b["Cryptex1,Version"].as_string(), Some("99.1.0.0.0,0"));
        assert_eq!(b["Cryptex1,Nonce"].as_data(), Some(&[0x22; 32][..]));
        let udid = [
            0, 0, 0, 0, 0, 0, 0xfe, 0, 0xc2, 0x08, 0xf7, 0xea, 0xfe, 0xbd, 0xc5, 0x81,
        ];
        assert_eq!(b["Cryptex1,UDID"].as_data(), Some(&udid[..]));
        assert!(!b.contains_key("Ap,OSLongVersion"));
        assert!(!b.contains_key("KernelCache"));
        let os = b["Cryptex1,SystemOS"].as_dictionary().unwrap();
        assert_eq!(keys(os), ["Digest"]);
        let expected = req.expected.as_ref().unwrap();
        assert!(
            expected
                .props
                .contains(&("UDID", Prop::Data(udid.to_vec())))
        );
        assert!(
            expected
                .props
                .contains(&("cnch", Prop::Data(vec![0x22; 32])))
        );
        assert!(!expected.props.iter().any(|(k, _)| *k == "ECID"));
        assert_eq!(
            names(expected),
            ["Cryptex1,SystemOS", "Cryptex1,SystemTrustCache"]
        );
    }

    #[test]
    fn tss_tls_is_verified_against_apple_root_ca() {
        use sha2::Digest;
        let der = apple_root_ca().der().to_vec();
        let fingerprint: [u8; 32] = sha2::Sha256::digest(&der).into();
        let hex: String = fingerprint.iter().map(|b| format!("{b:02X}")).collect();
        assert_eq!(
            hex,
            "B0B1730ECBC7FF4505142C49F1295E6EDA6BCAED7E2C68C5BE91B5A11001F024"
        );
    }

    #[test]
    fn the_udid_reverses_the_device_trees_unique_device_id() {
        let mut dt = Vec::new();
        dt.extend_from_slice(&ECID.to_le_bytes());
        dt.extend_from_slice(&0xfe00u32.to_le_bytes());
        dt.extend_from_slice(&[0; 4]);
        dt.reverse();
        assert_eq!(cryptex1_udid(0xfe00, ECID).to_vec(), dt);
    }

    /// A ticket TSS could have returned for `req`, with `tweak` applied to its properties.
    fn ticket_for(
        req: &Request,
        tweak: impl FnOnce(&mut BTreeMap<String, Prop>, &mut BTreeMap<String, BTreeMap<String, Prop>>),
    ) -> Vec<u8> {
        let expected = req.expected.as_ref().unwrap();
        let mut props: BTreeMap<String, Prop> = expected
            .props
            .iter()
            .map(|(k, v)| ((*k).to_owned(), v.clone()))
            .collect();
        props.insert("CEPO".into(), Prop::Int(1));
        let mut objects = BTreeMap::new();
        for (name, digest) in &expected.objects {
            let tag = img4::component_tag(name).unwrap_or("newt");
            let mut o = BTreeMap::new();
            o.insert("DGST".to_owned(), Prop::Data(digest.clone()));
            o.insert("EKEY".to_owned(), Prop::Bool(true));
            objects.insert(tag.to_owned(), o);
        }
        tweak(&mut props, &mut objects);
        img4::encode_im4m(
            &img4::encode_manb(&props, &objects),
            &[1; 96],
            &img4::seq(&[]),
        )
    }

    #[test]
    fn tickets_are_checked_against_the_request() {
        let bm = BuildManifest::parse(&manifest()).unwrap();
        let id = bm.vma2_identity(VARIANT_INSTALL).unwrap();
        let req = ap_request(&id, ApTicket::Boot, ECID, &NONCES).unwrap();
        let expected = req.expected.as_ref().unwrap();
        expected.check(&ticket_for(&req, |_, _| {})).unwrap();

        let bad = |t: Vec<u8>, what: &str| {
            let e = expected.check(&t).unwrap_err();
            assert!(matches!(e, TssError::BadTicket(_)), "{what}: {e}");
            assert!(e.to_string().contains(what), "{e} should say {what}");
        };
        bad(
            ticket_for(&req, |p, _| {
                drop(p.insert("ECID".into(), Prop::Int(ECID + 1)))
            }),
            "ECID",
        );
        bad(ticket_for(&req, |p, _| drop(p.remove("CHIP"))), "no CHIP");
        bad(
            ticket_for(&req, |p, _| drop(p.insert("BORD".into(), Prop::Int(0x22)))),
            "BORD",
        );
        bad(
            ticket_for(&req, |p, _| {
                drop(p.insert("BNCH".into(), Prop::Data(vec![0; 32])))
            }),
            "BNCH",
        );
        bad(
            ticket_for(&req, |_, o| drop(o.remove("krnl"))),
            "KernelCache",
        );
        bad(
            ticket_for(&req, |_, o| {
                drop(
                    o.get_mut("dtre")
                        .unwrap()
                        .insert("DGST".into(), Prop::Data(vec![9; 48])),
                )
            }),
            "dtre",
        );
        // A component no table names is found by its digest.
        bad(ticket_for(&req, |_, o| drop(o.remove("newt"))), "NewThing");
        bad(vec![0x30, 0x03, 0x02, 0x01, 0x00], "parse");
    }

    #[test]
    fn answers_parse_as_tss_writes_them() {
        let mut d = Dictionary::new();
        d.insert("ApImg4Ticket".into(), Value::Data(vec![1, 2, 3]));
        let mut xml = Vec::new();
        Value::Dictionary(d).to_writer_xml(&mut xml).unwrap();
        let ok = format!(
            "STATUS=0&MESSAGE=SUCCESS&REQUEST_STRING={}",
            String::from_utf8(xml).unwrap()
        );
        let answer = parse_response(&ok).unwrap().unwrap();
        assert_eq!(answer["ApImg4Ticket"].as_data(), Some(&[1u8, 2, 3][..]));

        let refused =
            parse_response("STATUS=94&MESSAGE=This device isn't eligible for the requested build.")
                .unwrap_err();
        assert!(
            matches!(refused, TssError::Refused { status: 94, ref message } if message.starts_with("This device"))
        );
        assert!(matches!(
            parse_response("STATUS=0&MESSAGE=SUCCESS"),
            Err(TssError::BadResponse(_))
        ));
        assert!(matches!(
            parse_response("STATUS=0&MESSAGE=SUCCESS&REQUEST_STRING=nope"),
            Err(TssError::BadResponse(_))
        ));
        assert!(
            parse_response("<html>a proxy's page</html>")
                .unwrap()
                .is_none()
        );
    }

    /// Serves `answers` on a local port, one connection each; returns the URL.
    fn serve(answers: Vec<(&'static str, String)>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!(
            "http://{}/TSS/controller?action=2",
            listener.local_addr().unwrap()
        );
        std::thread::spawn(move || {
            for (status, body) in answers {
                let (mut s, _) = listener.accept().unwrap();
                let mut req = Vec::new();
                let mut buf = [0; 4096];
                // Read the headers and the body Content-Length gives.
                loop {
                    let n = s.read(&mut buf).unwrap();
                    req.extend_from_slice(&buf[..n]);
                    let text = String::from_utf8_lossy(&req);
                    if let Some(end) = text.find("\r\n\r\n") {
                        let len = text
                            .lines()
                            .find_map(|l| {
                                l.to_ascii_lowercase()
                                    .strip_prefix("content-length:")
                                    .map(|v| v.trim().parse::<usize>().unwrap())
                            })
                            .unwrap_or(0);
                        if req.len() >= end + 4 + len {
                            break;
                        }
                    }
                    if n == 0 {
                        break;
                    }
                }
                let head = String::from_utf8_lossy(&req).to_ascii_lowercase();
                assert!(head.contains("user-agent: ineturl/1.0"), "{head}");
                assert!(head.contains("<key>@apimg4ticket</key>"), "{head}");
                let out = format!(
                    "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                s.write_all(out.as_bytes()).unwrap();
            }
        });
        url
    }

    #[test]
    fn a_refused_build_says_it_is_not_signed() {
        let bm = BuildManifest::parse(&manifest()).unwrap();
        let id = bm.vma2_identity(VARIANT_INSTALL).unwrap();
        let url = serve(vec![(
            "200 OK",
            "STATUS=94&MESSAGE=This device isn't eligible for the requested build.".into(),
        )]);
        let client = TssClient::with_url(url).retries(1, Duration::ZERO);
        let e = client
            .ap_ticket(&id, ApTicket::Boot, ECID, &NONCES)
            .unwrap_err();
        let tss = e.downcast_ref::<TssError>().unwrap();
        assert!(
            matches!(tss, TssError::NotSigned { status: 94, .. }),
            "{tss:?}"
        );
        let text = format!("{e:#}");
        assert!(
            text.contains("99A1 (99.1)") && text.contains("VirtualMac2,1"),
            "{text}"
        );
    }

    #[test]
    fn transient_failures_are_retried_and_tickets_returned() {
        let bm = BuildManifest::parse(&manifest()).unwrap();
        let id = bm.vma2_identity(VARIANT_INSTALL).unwrap();
        // The ticket a correct server returns for this request, whatever its UUID.
        let req = ap_request(&id, ApTicket::Boot, ECID, &NONCES).unwrap();
        let ticket = ticket_for(&req, |_, _| {});
        let mut d = Dictionary::new();
        d.insert("ApImg4Ticket".into(), Value::Data(ticket.clone()));
        let mut xml = Vec::new();
        Value::Dictionary(d).to_writer_xml(&mut xml).unwrap();
        let ok = format!(
            "STATUS=0&MESSAGE=SUCCESS&REQUEST_STRING={}",
            String::from_utf8(xml).unwrap()
        );
        let url = serve(vec![
            ("503 Service Unavailable", "busy".into()),
            ("200 OK", "<html>no status</html>".into()),
            ("200 OK", ok),
        ]);
        let client = TssClient::with_url(url).retries(3, Duration::ZERO);
        assert_eq!(
            client
                .ap_ticket(&id, ApTicket::Boot, ECID, &NONCES)
                .unwrap(),
            ticket
        );
    }

    #[test]
    fn an_unreachable_server_is_a_network_error() {
        let bm = BuildManifest::parse(&manifest()).unwrap();
        let id = bm.vma2_identity(VARIANT_INSTALL).unwrap();
        // A port nothing listens on: bind one, then drop it.
        let port = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let client =
            TssClient::with_url(format!("http://127.0.0.1:{port}/")).retries(2, Duration::ZERO);
        let e = client.cryptex1_ticket(&id, ECID, &[0; 32]).unwrap_err();
        assert!(
            matches!(
                e.downcast_ref::<TssError>(),
                Some(TssError::Network { attempts: 2, .. })
            ),
            "{e:#}"
        );
    }

    fn args(pairs: &[(&str, Value)]) -> Dictionary {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), v.clone()))
            .collect()
    }

    #[test]
    fn local_policy_requests_carry_the_restore_oss_arguments() {
        let bm = BuildManifest::parse(&manifest()).unwrap();
        let id = bm.vma2_identity(VARIANT_ERASE).unwrap();
        let a = args(&[
            ("Ap,NextStageIM4MHash", Value::Data(vec![1; 48])),
            ("Ap,RecoveryOSPolicyNonceHash", Value::Data(vec![2; 48])),
            (
                "Ap,VolumeUUID",
                Value::String("BF304653-4074-415B-8D52-3B779D769491".into()),
            ),
            ("SomethingElse", Value::Boolean(true)),
        ]);
        let req = local_policy_request(&id, &a, ECID).unwrap();
        let b = &req.body;
        assert_eq!(
            keys(b),
            [
                "@ApImg4Ticket",
                "@HostPlatformInfo",
                "@UUID",
                "@VersionInfo",
                "Ap,LocalBoot",
                "Ap,LocalPolicy",
                "Ap,NextStageIM4MHash",
                "Ap,RecoveryOSPolicyNonceHash",
                "Ap,VolumeUUID",
                "ApBoardID",
                "ApChipID",
                "ApECID",
                "ApProductionMode",
                "ApSecurityDomain",
                "ApSecurityMode",
            ]
        );
        assert_eq!(b["Ap,LocalBoot"].as_boolean(), Some(true));
        let lpol = b["Ap,LocalPolicy"].as_dictionary().unwrap();
        assert_eq!(
            lpol["Digest"].as_data(),
            Some(&img4::sha384(&img4::LPOL_IM4P)[..])
        );
        assert_eq!(lpol["Trusted"].as_boolean(), Some(true));
        let uuid = [
            0xbf, 0x30, 0x46, 0x53, 0x40, 0x74, 0x41, 0x5b, 0x8d, 0x52, 0x3b, 0x77, 0x9d, 0x76,
            0x94, 0x91,
        ];
        assert_eq!(b["Ap,VolumeUUID"].as_data(), Some(&uuid[..]));
        let expected = req.expected.as_ref().unwrap();
        assert!(expected.props.contains(&("ECID", Prop::Int(ECID))));
        assert_eq!(
            expected.objects,
            [(
                "Ap,LocalPolicy".to_owned(),
                img4::sha384(&img4::LPOL_IM4P).to_vec()
            )]
        );
        // The next stage's hash is required.
        assert!(local_policy_request(&id, &Dictionary::new(), ECID).is_err());
    }

    /// `FirmwareUpdaterData` arguments as the restore OS sends them for `updater`.
    fn updater_args(updater: &str, generated: Dictionary, response: &str) -> Dictionary {
        let mut tags = Dictionary::new();
        tags.insert(
            "BuildIdentityTags".into(),
            Value::Array(
                [
                    "Cryptex1,ChipID",
                    "Cryptex1,Version",
                    "UniqueBuildID",
                    "Nope",
                ]
                .map(|s| Value::String(s.into()))
                .to_vec(),
            ),
        );
        tags.insert(
            "ResponseTags".into(),
            Value::Array(vec![Value::String(response.into())]),
        );
        args(&[
            ("MessageArgUpdaterName", Value::String(updater.into())),
            (
                "MessageArgInfo",
                Value::Dictionary(args(&[("ApSecurityDomain", Value::Integer(1.into()))])),
            ),
            ("DeviceGeneratedTags", Value::Dictionary(tags)),
            ("DeviceGeneratedRequest", Value::Dictionary(generated)),
        ])
    }

    #[test]
    fn cryptex1_device_requests_merge_as_the_restore_host_does() {
        let bm = BuildManifest::parse(&manifest()).unwrap();
        let id = bm.vma2_identity(VARIANT_ERASE).unwrap();
        let mut os = Dictionary::new();
        os.insert("Digest".into(), Value::Data(vec![6; 48]));
        let generated = args(&[
            ("ApECID", Value::Integer(ECID.into())),
            ("Cryptex1,Nonce", Value::Data(vec![3; 32])),
            (
                "Cryptex1,UDID",
                Value::Data(cryptex1_udid(0xfe00, ECID).to_vec()),
            ),
            ("Cryptex1,SystemOS", Value::Dictionary(os)),
            ("ApProductionMode", Value::Boolean(false)),
        ]);
        let req =
            device_request(&id, &updater_args("Cryptex1", generated, "Cryptex1,Ticket")).unwrap();
        let b = &req.body;
        assert_eq!(req.ticket_key, "Cryptex1,Ticket");
        assert_eq!(b["@Cryptex1,Ticket"].as_boolean(), Some(true));
        assert_eq!(b["ApECID"].as_unsigned_integer(), Some(ECID));
        assert_eq!(
            b["ApChipID"].as_unsigned_integer(),
            Some(0xfe00),
            "filled in from the identity"
        );
        assert_eq!(
            b["ApSecurityDomain"].as_unsigned_integer(),
            Some(1),
            "from MessageArgInfo"
        );
        assert_eq!(
            b["UniqueBuildID"].as_data(),
            Some(&[0x42; 20][..]),
            "a build identity tag"
        );
        assert_eq!(
            b["Cryptex1,ChipID"].as_unsigned_integer(),
            Some(0xff10),
            "a build identity tag"
        );
        assert_eq!(
            b["ApProductionMode"].as_boolean(),
            Some(false),
            "the generated request wins"
        );
        assert_eq!(
            b["Cryptex1,ProductionMode"].as_boolean(),
            Some(false),
            "ApProductionMode's, where not generated"
        );
        assert!(b.contains_key("Cryptex1,SystemOS"));
        assert!(
            !b.contains_key("Cryptex1,NonceDomain"),
            "only the tags it names come from the identity"
        );
        assert!(!b.contains_key("Nope"));
        let expected = req.expected.as_ref().unwrap();
        assert!(expected.props.contains(&("cnch", Prop::Data(vec![3; 32]))));
        assert_eq!(names(expected), ["Cryptex1,SystemOS"]);
    }

    #[test]
    fn cryptex1_local_policy_device_requests_are_local_policy_requests() {
        let bm = BuildManifest::parse(&manifest()).unwrap();
        let id = bm.vma2_identity(VARIANT_ERASE).unwrap();
        let generated = args(&[
            ("ApECID", Value::Integer(ECID.into())),
            ("Ap,LocalBoot", Value::Boolean(false)),
            ("Ap,LocalPolicy", local_policy_entry()),
            ("Ap,NextStageIM4MHash", Value::Data(vec![4; 48])),
            ("Ap,NextStageCryptex1IM4MHash", Value::Data(vec![5; 48])),
        ]);
        let req = device_request(
            &id,
            &updater_args(
                "Cryptex1LocalPolicy",
                generated,
                "Cryptex1LocalPolicy,Ticket",
            ),
        )
        .unwrap();
        let b = &req.body;
        assert_eq!(req.ticket_key, "Cryptex1LocalPolicy,Ticket");
        assert_eq!(b["@ApImg4Ticket"].as_boolean(), Some(true));
        assert!(!b.contains_key("@Cryptex1,Ticket"));
        assert_eq!(b["Ap,LocalBoot"].as_boolean(), Some(false));
        assert_eq!(
            b["Ap,NextStageCryptex1IM4MHash"].as_data(),
            Some(&[5; 48][..])
        );
        assert!(
            !b.contains_key("Cryptex1,ChipID"),
            "a local policy request carries no Cryptex1 values"
        );
        assert_eq!(names(req.expected.as_ref().unwrap()), ["Ap,LocalPolicy"]);
    }

    #[test]
    fn other_device_requests_go_as_generated() {
        let bm = BuildManifest::parse(&manifest()).unwrap();
        let id = bm.vma2_identity(VARIANT_ERASE).unwrap();
        let generated = args(&[
            ("@Thing,Ticket", Value::Boolean(true)),
            ("Thing,Nonce", Value::Data(vec![1])),
        ]);
        let req = device_request(&id, &updater_args("Thing", generated, "Thing,Ticket")).unwrap();
        assert_eq!(
            keys(&req.body),
            [
                "@HostPlatformInfo",
                "@Thing,Ticket",
                "@UUID",
                "@VersionInfo",
                "Thing,Nonce"
            ]
        );
        assert_eq!(req.ticket_key, "Thing,Ticket");
        assert!(req.expected.is_none());
        let mut no_tag = updater_args("Thing", Dictionary::new(), "x");
        no_tag.remove("DeviceGeneratedTags");
        assert!(device_request(&id, &no_tag).is_err());
    }

    /// Live: asks Apple's TSS, for a random ECID, for the tickets of a build manifest's virtual
    /// Mac: the boot, SFR and Cryptex1 tickets; a LocalPolicy's manifest; and a Cryptex1 ticket
    /// through the restore OS's request path. Each is checked against its request; written to
    /// `REIMS_RESTORE_LIVE_OUT` where set; and, with a directory of a fresh install's tickets in
    /// `REIMS_RESTORE_LIVE_REF` (`boot.im4m`, `sfr.im4m`, `cryptex1.im4m` and `lp.im4m`, the
    /// paired recoveryOS's LocalPolicy, an IM4M or the IMG4 holding it), compared with them: the
    /// same MANP keys, the same objects with the same properties, and the same values but for
    /// those bound to the ECID, the nonces, the time and the restore OS's arguments.
    #[test]
    #[ignore = "asks Apple's TSS: REIMS_RESTORE_LIVE_MANIFEST=<BuildManifest.plist> or REIMS_RESTORE_LIVE_IPSW=<ipsw>, \
                [REIMS_RESTORE_LIVE_OUT=<dir>] [REIMS_RESTORE_LIVE_REF=<dir>]"]
    fn live_tickets() {
        let manifest = match std::env::var("REIMS_RESTORE_LIVE_MANIFEST") {
            Ok(path) => std::fs::read(path).unwrap(),
            Err(_) => crate::ipsw::Ipsw::open(
                std::env::var("REIMS_RESTORE_LIVE_IPSW")
                    .expect("REIMS_RESTORE_LIVE_IPSW")
                    .as_ref(),
            )
            .unwrap()
            .read("BuildManifest.plist")
            .unwrap(),
        };
        let bm = BuildManifest::parse(&manifest).unwrap();
        let install = bm.vma2_identity(VARIANT_INSTALL).unwrap();
        let erase = bm.vma2_identity(VARIANT_ERASE).unwrap();
        let ecid = random_ecid().unwrap();
        let n = ApNonces::generate().unwrap();
        let tss = TssClient::new();
        let mut failures = Vec::new();
        let mut tickets: Vec<(&str, Vec<u8>)> = Vec::new();
        let mut keep = |name: &'static str, r: Result<Vec<u8>>| match r {
            Ok(t) => tickets.push((name, t)),
            Err(e) => failures.push(format!("{name}: {e:#}")),
        };
        let boot = tss.ap_ticket(&install, ApTicket::Boot, ecid, &n);
        let next_stage = boot.as_ref().map(|b| img4::sha384(b)).unwrap_or([0; 48]);
        keep("boot.im4m", boot);
        keep("sfr.im4m", tss.ap_ticket(&erase, ApTicket::Sfr, ecid, &n));
        keep(
            "cryptex1.im4m",
            tss.cryptex1_ticket(&install, ecid, &n.ap_nonce),
        );

        let mut nonce_hash = [0; 16];
        getrandom::fill(&mut nonce_hash).unwrap();
        let lp_args = args(&[
            ("Ap,NextStageIM4MHash", Value::Data(next_stage.to_vec())),
            (
                "Ap,RecoveryOSPolicyNonceHash",
                Value::Data(img4::sha384(&nonce_hash).to_vec()),
            ),
            (
                "Ap,VolumeUUID",
                Value::String(uuid::Uuid::new_v4().hyphenated().to_string().to_uppercase()),
            ),
        ]);
        keep("lp.im4m", tss.local_policy_ticket(&erase, &lp_args, ecid));

        // The Cryptex1 request through the restore OS's path.
        let host_built = cryptex1_request(&install, ecid, &n.ap_nonce).unwrap().body;
        let generated: Dictionary = host_built
            .iter()
            .filter(|(k, _)| k.starts_with("Cryptex1,") || *k == "ApECID")
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let identity_tags: Vec<Value> = install
            .dict()
            .keys()
            .filter(|k| k.starts_with("Cryptex1,") || *k == "UniqueBuildID")
            .map(|k| Value::String(k.clone()))
            .collect();
        let mut tags = Dictionary::new();
        tags.insert("BuildIdentityTags".into(), Value::Array(identity_tags));
        tags.insert(
            "ResponseTags".into(),
            Value::Array(vec![Value::String("Cryptex1,Ticket".into())]),
        );
        let updater = args(&[
            ("MessageArgUpdaterName", Value::String("Cryptex1".into())),
            (
                "MessageArgInfo",
                Value::Dictionary(args(&[("ApSecurityDomain", Value::Integer(1.into()))])),
            ),
            ("DeviceGeneratedTags", Value::Dictionary(tags)),
            ("DeviceGeneratedRequest", Value::Dictionary(generated)),
        ]);
        keep(
            "cryptex1-device.im4m",
            tss.device_request(&install, &updater)
                .map(|a| a["Cryptex1,Ticket"].as_data().unwrap_or_default().to_vec()),
        );

        println!(
            "{} {}: ECID {ecid:#018x}",
            bm.product_version().unwrap(),
            bm.build_version().unwrap()
        );
        // Values bound to this request rather than to the build.
        let per_request = [
            "BNCH", "ECID", "UDID", "cnch", "srvn", "tstp", "nsih", "ronh", "vuid",
        ];
        let mut mismatches = Vec::new();
        for (name, der) in &tickets {
            let ours = img4::parse_im4m(der).unwrap();
            println!(
                "{name}: MANP {:?}",
                ours.props
                    .iter()
                    .map(|(k, v)| format!("{k}={v:?}"))
                    .collect::<Vec<_>>()
            );
            println!(
                "{name}: objects {:?}",
                ours.objects
                    .iter()
                    .map(|(k, o)| (k, o.keys().collect::<Vec<_>>()))
                    .collect::<Vec<_>>()
            );
            if let Ok(dir) = std::env::var("REIMS_RESTORE_LIVE_OUT") {
                std::fs::create_dir_all(&dir).unwrap();
                std::fs::write(std::path::Path::new(&dir).join(name), der).unwrap();
            }
            let Ok(dir) = std::env::var("REIMS_RESTORE_LIVE_REF") else {
                continue;
            };
            let Ok(reference) = std::fs::read(std::path::Path::new(&dir).join(name)) else {
                continue;
            };
            let reference = match img4::split_img4(&reference) {
                Ok((_, Some(im4m))) => im4m.to_vec(),
                _ => reference,
            };
            let theirs = img4::parse_im4m(&reference).unwrap();
            let ours_keys: Vec<_> = ours.props.keys().collect();
            let their_keys: Vec<_> = theirs.props.keys().collect();
            if ours_keys != their_keys {
                mismatches.push(format!(
                    "{name}: MANP keys {ours_keys:?}, the reference's {their_keys:?}"
                ));
            }
            for (k, v) in &theirs.props {
                if !per_request.contains(&k.as_str()) && ours.props.get(k) != Some(v) {
                    mismatches.push(format!(
                        "{name}: MANP {k} is {:?}, the reference's {v:?}",
                        ours.props.get(k)
                    ));
                }
            }
            if ours.objects != theirs.objects {
                mismatches.push(format!("{name}: objects differ from the reference's"));
            }
        }
        for f in &failures {
            println!("FAILED {f}");
        }
        for m in &mismatches {
            println!("MISMATCH {m}");
        }
        assert!(
            failures.is_empty() && mismatches.is_empty(),
            "{} requests failed, {} mismatches",
            failures.len(),
            mismatches.len()
        );
    }
}
