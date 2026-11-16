//! Prints the plan of a restore of an IPSW as a listing, one line per entry, for comparing with
//! a listing of a disk Virtualization.framework restored (docs/design.md, "The disk"). Every
//! entry made from the IPSW is listed with its size and, up to 64 MiB, the first 16 hex digits of
//! its SHA-256; the VM's own values in paths are masked: the volume group as `<G>`, the recoveryOS
//! group as `<R>`, the ECID as `<ECID>` and each 96-digit hash as `<H96>`.
//!
//! Tickets and policies are placeholders unless `--tickets DIR` gives the restored disk's
//! `boot.der` (its boot manifest), `sfr.der` (its SFR manifest) and `cx1.der` (its Cryptex1
//! ticket): then the boot objects are stitched to them and compared byte for byte. The policies
//! stay placeholders, since a restore's own are signed for its own nonces.
//!
//! With `--tickets`, the ECID is the boot manifest's, and the volume group the UUID in `DIR/group`
//! where that file exists, so neither need be named on a command line.
//!
//! `--hash-all` hashes every image whole, however big. The boot directories' root hash IM4P is made
//! with Apple's root hash, where a restore makes it with its own seal's.
//!
//! Usage: cargo run --release --example plan_listing -- IPSW [--tickets DIR] [--group UUID]
//! [--ecid HEX] [--hash-all]

use std::io::Read as _;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use reims_restore::aea::AppleKeyService;
use reims_restore::image;
use reims_restore::img4;
use reims_restore::ipsw::Ipsw;
use reims_restore::manifest::{BuildManifest, VARIANT_ERASE, VARIANT_INSTALL};
use reims_restore::plan::{self, Content, Data, Node, RECOVERY_GROUP};
use sha2::Digest as _;
use uuid::Uuid;

const HASHED: u64 = 64 << 20;

struct Args {
    ipsw: PathBuf,
    tickets: Option<PathBuf>,
    group: Uuid,
    ecid: u64,
    /// Hash every image whole, however big.
    hash_all: bool,
}

fn args() -> Result<Args> {
    let mut a = std::env::args().skip(1);
    let mut out = Args {
        ipsw: PathBuf::new(),
        tickets: None,
        group: Uuid::from_u128(0x6767_6767),
        ecid: 0x1111_2222_3333_4444,
        hash_all: false,
    };
    while let Some(x) = a.next() {
        let mut val = || a.next().with_context(|| format!("{x} needs a value"));
        match x.as_str() {
            "--tickets" => out.tickets = Some(val()?.into()),
            "--group" => out.group = val()?.parse()?,
            "--ecid" => out.ecid = u64::from_str_radix(&val()?, 16)?,
            "--hash-all" => out.hash_all = true,
            _ if out.ipsw.as_os_str().is_empty() => out.ipsw = x.into(),
            _ => bail!("usage: plan_listing IPSW [--tickets DIR] [--group UUID] [--ecid HEX]"),
        }
    }
    if out.ipsw.as_os_str().is_empty() {
        bail!("usage: plan_listing IPSW [--tickets DIR] [--group UUID] [--ecid HEX]");
    }
    if let Some(d) = &out.tickets {
        let boot = img4::parse_im4m(&std::fs::read(d.join("boot.der"))?)?;
        if let Some(e) = boot.props.get("ECID").and_then(img4::Prop::as_int) {
            out.ecid = e;
        }
        if let Ok(g) = std::fs::read_to_string(d.join("group")) {
            out.group = g.trim().parse().context("the group file")?;
        }
    }
    Ok(out)
}

fn main() -> Result<()> {
    let args = args()?;
    let ipsw = Ipsw::open(&args.ipsw)?;
    let bm = BuildManifest::parse(&ipsw.read("BuildManifest.plist")?)?;
    let install = bm.vma2_identity(VARIANT_INSTALL)?;
    let erase = bm.vma2_identity(VARIANT_ERASE)?;
    let keys = AppleKeyService::default();
    // A stand-in for the seal's root hash: Apple's, so that a listing with a restored disk's
    // tickets matches that disk's root_hash.img4 byte for byte.
    let seal = img4::RootHash::parse(
        &img4::parse_im4p(
            &ipsw.read(
                install
                    .component("SystemVolume")?
                    .path()
                    .context("SystemVolume")?,
            )?,
        )?
        .payload,
    )?;
    let b = |s: &str| -> Arc<[u8]> { s.as_bytes().into() };
    let ticket = |name: &str, placeholder: &str| -> Result<Arc<[u8]>> {
        Ok(match &args.tickets {
            Some(d) => std::fs::read(d.join(name))
                .with_context(|| format!("read {}", d.join(name).display()))?
                .into(),
            None => b(placeholder),
        })
    };
    let names: Vec<String> = ipsw.names().map(str::to_owned).collect();
    let has = |n: &str| ipsw.has(n);
    let p = plan::plan(&plan::Inputs {
        manifest: &bm,
        has: &has,
        names: &names,
        install,
        erase,
        ecid: args.ecid,
        group: args.group,
        boot_manifest: ticket("boot.der", "boot manifest")?,
        sfr_manifest: ticket("sfr.der", "sfr manifest")?,
        cryptex1_ticket: ticket("cx1.der", "cryptex1 ticket")?,
        policy: ("POLICY.img4".into(), b("policy")),
        policy_recovery: ("POLICY.recovery.img4".into(), b("policy recovery")),
        recovery_os_policy: b("recovery os policy"),
        ronh: "RONH".into(),
        eacs_store: Uuid::nil(),
    })?;
    let masks = [
        (args.group.to_string().to_uppercase(), "<G>"),
        (RECOVERY_GROUP.to_string().to_uppercase(), "<R>"),
        (format!("{:016X}", args.ecid), "<ECID>"),
    ];
    let mask = |s: &str| -> String {
        let mut s = s.to_owned();
        for (v, n) in &masks {
            s = s.replace(v.as_str(), n);
        }
        mask_h96(&s)
    };
    for (container, vols) in [
        ("isc", &p.isc),
        ("main", &p.main),
        ("recovery", &p.recovery),
    ] {
        for v in vols {
            println!("== {container} {} {:?}", v.name, v.role);
            let entries = match &v.content {
                Content::Entries(e) => e,
                other => {
                    println!("   ({other:?})");
                    continue;
                }
            };
            let mut rows = Vec::new();
            for e in entries {
                let (kind, size, extra) = match &e.node {
                    Node::Dir => ("d", "-".to_owned(), String::new()),
                    Node::Symlink(t) => ("l", t.len().to_string(), format!("-> {}", mask(t))),
                    Node::Clone(of) => ("f", "?".into(), format!("clone of {}", mask(of))),
                    Node::File(d) => {
                        let (size, hash) = describe(&ipsw, &keys, d, args.hash_all, &seal)?;
                        ("f", size.to_string(), hash)
                    }
                };
                rows.push((
                    mask(&e.path),
                    format!(
                        "{kind} {:04o} {}:{} size={size} {} {extra}",
                        e.mode,
                        e.uid,
                        e.gid,
                        mask(&e.path)
                    ),
                ));
            }
            rows.sort();
            for (_, r) in rows {
                println!("{r}");
            }
        }
    }
    Ok(())
}

/// Each run of 96 upper-case hex digits as `<H96>`.
fn mask_h96(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::new();
    let mut at = 0;
    while at < b.len() {
        let run = b[at..]
            .iter()
            .take_while(|c| c.is_ascii_digit() || (b'A'..=b'F').contains(c))
            .count();
        if run >= 96 {
            out.push_str(&"<H96>".repeat(run / 96));
            out.push_str(&s[at + run / 96 * 96..at + run]);
            at += run;
        } else if run > 0 {
            out.push_str(&s[at..at + run]);
            at += run;
        } else {
            let c = s[at..].chars().next().unwrap();
            out.push(c);
            at += c.len_utf8();
        }
    }
    out
}

fn hex16(b: &[u8]) -> String {
    sha2::Sha256::digest(b)
        .iter()
        .take(8)
        .map(|x| format!("{x:02x}"))
        .collect()
}

fn describe(
    ipsw: &Ipsw,
    keys: &AppleKeyService,
    d: &Data,
    hash_all: bool,
    seal: &img4::RootHash,
) -> Result<(u64, String)> {
    Ok(match d {
        Data::Bytes(b) => (b.len() as u64, hex16(b)),
        Data::Member(m) => {
            let size = ipsw.member(m)?.size;
            let h = if size <= HASHED {
                hex16(&ipsw.read(m)?)
            } else {
                "big".into()
            };
            (size, h)
        }
        Data::Plain(m) => {
            let (len, mut r) = image::plain_stream(ipsw, m, keys)?;
            if len > HASHED && !hash_all {
                (len, "big".into())
            } else {
                let mut h = sha2::Sha256::new();
                let mut buf = vec![0u8; 1 << 20];
                let mut n = 0u64;
                loop {
                    let k = r.read(&mut buf)?;
                    if k == 0 {
                        break;
                    }
                    h.update(&buf[..k]);
                    n += k as u64;
                }
                anyhow::ensure!(n == len, "{m}: {n} bytes read of {len}");
                let d = h.finalize();
                (len, d.iter().take(8).map(|x| format!("{x:02x}")).collect())
            }
        }
        Data::Stitched {
            component,
            member,
            manifest,
        } => match img4::personalize(component, &ipsw.read(member)?, manifest) {
            Ok(out) => (out.len() as u64, hex16(&out)),
            Err(e) => (
                ipsw.member(member)?.size,
                format!("stitched {component} from {member}: {e:#}"),
            ),
        },
        Data::SealedRootHash { manifest } => {
            let out = img4::stitch_img4(&seal.system_im4p(), manifest);
            (out.len() as u64, hex16(&out))
        }
    })
}
