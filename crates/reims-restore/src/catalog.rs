//! Where IPSWs come from: Apple's catalog of macOS IPSWs, and the IPSWs already on disk
//! (docs/design.md, "Downloading IPSWs").
//!
//! Apple's catalog ([`URL`]) is the one Virtualization.framework's
//! `VZMacOSRestoreImage.fetchLatestSupported` reads (measured: on macOS 27.0.1 it answered the
//! catalog's VirtualMac2,1 URL). It lists one release, the current one, for each Mac and for the
//! virtual Mac (VirtualMac2,1), and each release replaces the last. VirtualMac2,1's entry gives
//! the IPSW's version, build and URL but no digest; the physical Macs' entries for the same URL
//! give its SHA-1, which [`parse`] takes from them. Another release is restored from an IPSW the
//! user has: one on disk is known by Apple's file name for it,
//! `UniversalMac_<version>_<build>_Restore.ipsw` ([`on_disk`]).

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use plist::Value;

use crate::http::Client;

/// Apple's catalog of macOS IPSWs.
pub const URL: &str =
    "https://mesu.apple.com/assets/macos/com_apple_macOSIPSW/com_apple_macOSIPSW.xml";

/// The virtual Mac's model in Apple's catalog.
pub const DEVICE: &str = "VirtualMac2,1";

/// An IPSW Apple's catalog lists for the virtual Mac.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    /// macOS's version, such as 27.2.1.
    pub version: String,
    /// Its build, such as 26B131.
    pub build: String,
    pub url: String,
    /// The IPSW's SHA-1, in lowercase hex, where the catalog gives one.
    pub sha1: Option<String>,
}

impl Entry {
    /// The IPSW's file name: the URL's last part, Apple's own name for it.
    pub fn file_name(&self) -> &str {
        self.url.rsplit('/').next().unwrap_or_default()
    }
}

/// The virtual Mac's entries in the catalog `xml`.
pub fn parse(xml: &[u8]) -> Result<Vec<Entry>> {
    let root = Value::from_reader_xml(xml).context("Apple's IPSW catalog is not a plist")?;
    let mut digests = BTreeMap::new();
    collect_digests(&root, &mut digests);
    let by_version = root
        .as_dictionary()
        .and_then(|d| d.get("MobileDeviceSoftwareVersionsByVersion"))
        .and_then(Value::as_dictionary)
        .context("Apple's IPSW catalog has no MobileDeviceSoftwareVersionsByVersion")?;
    let mut out = Vec::new();
    for v in by_version.values() {
        let Some(builds) = v
            .as_dictionary()
            .and_then(|d| d.get("MobileDeviceSoftwareVersions"))
            .and_then(Value::as_dictionary)
            .and_then(|d| d.get(DEVICE))
            .and_then(Value::as_dictionary)
        else {
            continue;
        };
        for b in builds.values() {
            let Some(r) = b
                .as_dictionary()
                .and_then(|d| d.get("Restore"))
                .and_then(Value::as_dictionary)
            else {
                continue;
            };
            let s = |k: &str| r.get(k).and_then(Value::as_string).map(str::to_owned);
            let (Some(version), Some(build), Some(url)) =
                (s("ProductVersion"), s("BuildVersion"), s("FirmwareURL"))
            else {
                continue;
            };
            let sha1 = s("FirmwareSHA1")
                .or_else(|| digests.get(&url).cloned())
                .map(|d| d.to_ascii_lowercase());
            let e = Entry {
                version,
                build,
                url,
                sha1,
            };
            if !out.contains(&e) {
                out.push(e);
            }
        }
    }
    sort(&mut out);
    Ok(out)
}

/// Every FirmwareURL in `v` with its FirmwareSHA1, by URL.
fn collect_digests(v: &Value, out: &mut BTreeMap<String, String>) {
    match v {
        Value::Dictionary(d) => {
            if let (Some(u), Some(s)) = (
                d.get("FirmwareURL").and_then(Value::as_string),
                d.get("FirmwareSHA1").and_then(Value::as_string),
            ) && !s.is_empty()
            {
                out.insert(u.to_owned(), s.to_owned());
            }
            d.values().for_each(|v| collect_digests(v, out));
        }
        Value::Array(a) => a.iter().for_each(|v| collect_digests(v, out)),
        _ => {}
    }
}

/// Fetches Apple's catalog and the virtual Mac's entries in it.
pub fn fetch(client: &Client, url: &str) -> Result<Vec<Entry>> {
    let (_, mut r) = client.get(url, None)?;
    if r.status() != 200 {
        bail!("Apple's IPSW catalog answered {}", r.status());
    }
    let body = r
        .body_mut()
        .with_config()
        .limit(16 << 20)
        .read_to_vec()
        .context("read Apple's IPSW catalog")?;
    let entries = parse(&body)?;
    if entries.is_empty() {
        bail!("Apple's IPSW catalog lists no IPSW for the virtual Mac ({DEVICE})");
    }
    Ok(entries)
}

// ------------------------------------------------------------------ IPSWs on disk

/// An IPSW on disk, known by Apple's file name for it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OnDisk {
    pub version: String,
    pub build: String,
    pub path: PathBuf,
    pub size: u64,
}

/// The version and build in Apple's name for an IPSW, `UniversalMac_<version>_<build>_Restore.ipsw`.
pub fn name_parts(name: &str) -> Option<(String, String)> {
    let mid = name
        .strip_prefix("UniversalMac_")?
        .strip_suffix("_Restore.ipsw")?;
    let (version, build) = mid.split_once('_')?;
    (is_version(version) && is_build(build)).then(|| (version.to_owned(), build.to_owned()))
}

/// The IPSWs in `dirs`, newest first; a directory that doesn't exist has none.
pub fn on_disk(dirs: &[PathBuf]) -> Vec<OnDisk> {
    let mut out: Vec<OnDisk> = Vec::new();
    for d in dirs {
        let Ok(rd) = std::fs::read_dir(d) else {
            continue;
        };
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            let Some((version, build)) = name_parts(&name) else {
                continue;
            };
            let Ok(m) = e.metadata() else { continue };
            if m.is_file() && !out.iter().any(|o| o.build == build) {
                out.push(OnDisk {
                    version,
                    build,
                    path: e.path(),
                    size: m.len(),
                });
            }
        }
    }
    out.sort_by(|a, b| newer(&b.version, &b.build, &a.version, &a.build));
    out
}

// ------------------------------------------------------------------ choosing one

/// What a name of a release ([`pick`]) found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Found {
    /// An IPSW on disk.
    OnDisk(OnDisk),
    /// One to download.
    Download(Entry),
}

/// The IPSW `want` names, from those on disk and those Apple's catalog lists:
/// `latest`, the newest of them all; a build (26B131); or a version: the one of that version
/// (27.2.1), else the newest of those it begins (27.2, 27). One on disk is taken over a download
/// of the same build.
pub fn pick(want: &str, on_disk: &[OnDisk], listed: &[Entry]) -> Result<Found> {
    let want = want.trim();
    let mut all: Vec<(&str, &str)> = on_disk
        .iter()
        .map(|o| (o.version.as_str(), o.build.as_str()))
        .chain(
            listed
                .iter()
                .map(|e| (e.version.as_str(), e.build.as_str())),
        )
        .collect();
    all.sort_by(|a, b| newer(b.0, b.1, a.0, a.1));
    let chosen = if want.eq_ignore_ascii_case("latest") {
        all.first().copied()
    } else if is_build(&build_case(want)) {
        all.iter()
            .copied()
            .find(|(_, b)| b.eq_ignore_ascii_case(want))
    } else if is_version(want) {
        all.iter()
            .copied()
            .find(|(v, _)| *v == want)
            .or_else(|| all.iter().copied().find(|(v, _)| begins(v, want)))
    } else {
        bail!("{want:?} is not a macOS version (27.2.1), a build (26B131) or latest");
    };
    let Some((_, build)) = chosen else {
        bail!("{}", missing(want, on_disk, listed));
    };
    if let Some(o) = on_disk.iter().find(|o| o.build == build) {
        return Ok(Found::OnDisk(o.clone()));
    }
    let e = listed.iter().find(|e| e.build == build).unwrap();
    Ok(Found::Download(e.clone()))
}

/// Says that no IPSW of `want` is known: Apple's catalog lists only the current release, so
/// another is given by its IPSW's path.
fn missing(want: &str, on_disk: &[OnDisk], listed: &[Entry]) -> String {
    let names = |v: Vec<String>| v.join(", ");
    if want.eq_ignore_ascii_case("latest") {
        return "Apple's catalog of IPSWs couldn't be read, and there's no IPSW here: try again \
                when it can be reached, or give an IPSW's path"
            .to_owned();
    }
    let mut s = if listed.is_empty() {
        format!("there's no macOS {want} IPSW here, and Apple's catalog couldn't be read")
    } else {
        format!(
            "Apple offers only the current macOS, {}, to download",
            names(
                listed
                    .iter()
                    .map(|e| format!("{} ({})", e.version, e.build))
                    .collect()
            )
        )
    };
    if !on_disk.is_empty() {
        s.push_str(&format!(
            ", and the IPSWs here are {}",
            names(
                on_disk
                    .iter()
                    .map(|o| format!("{} ({})", o.version, o.build))
                    .collect()
            )
        ));
    }
    s.push_str(&format!(": for macOS {want}, give its IPSW's path"));
    s
}

/// A macOS version: numbers and dots, such as 27, 27.2 or 27.2.1.
pub fn is_version(s: &str) -> bool {
    !s.is_empty()
        && !s.starts_with('.')
        && !s.ends_with('.')
        && !s.contains("..")
        && s.chars().all(|c| c.is_ascii_digit() || c == '.')
}

/// A macOS build: two digits, an uppercase letter, then digits and perhaps a lowercase letter
/// (26B131, 25D771280a).
pub fn is_build(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() >= 4
        && b[..2].iter().all(u8::is_ascii_digit)
        && b[2].is_ascii_uppercase()
        && b[3].is_ascii_digit()
        && b[3..]
            .iter()
            .enumerate()
            .all(|(i, c)| c.is_ascii_digit() || (i == b.len() - 4 && c.is_ascii_lowercase()))
}

/// `s` with its third character, a build's train letter, in upper case: 23g93 as 23G93.
fn build_case(s: &str) -> String {
    s.char_indices()
        .map(|(i, c)| if i == 2 { c.to_ascii_uppercase() } else { c })
        .collect()
}

/// Whether version `v` begins with `prefix`'s numbers (27.2.1 begins 27.2 and 27, not 27.20).
fn begins(v: &str, prefix: &str) -> bool {
    let (v, p) = (numbers(v), numbers(prefix));
    v.len() >= p.len() && v[..p.len()] == p[..]
}

fn numbers(v: &str) -> Vec<u32> {
    v.split('.').map(|n| n.parse().unwrap_or(0)).collect()
}

/// Orders (version, build) pairs, newer first, as `Ordering` of the second against the first.
fn newer(av: &str, ab: &str, bv: &str, bb: &str) -> std::cmp::Ordering {
    let (a, b) = (numbers(av), numbers(bv));
    a.cmp(&b).then_with(|| build_key(ab).cmp(&build_key(bb)))
}

/// A build as something to order: its train (26B), then its number.
fn build_key(b: &str) -> (String, u64) {
    let split = b
        .find(|c: char| c.is_ascii_uppercase())
        .map_or(0, |i| i + 1);
    let digits: String = b[split..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    (b[..split].to_owned(), digits.parse().unwrap_or(0))
}

fn sort(entries: &mut [Entry]) {
    entries.sort_by(|a, b| newer(&b.version, &b.build, &a.version, &a.build));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A catalog shaped as Apple's: a physical Mac's entry with the IPSW's SHA-1, the virtual
    /// Mac's without it, and a build for another Mac alone.
    const CATALOG: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>MobileDeviceProductTypes</key><dict/>
<key>MobileDeviceSoftwareVersionsByVersion</key><dict><key>1</key><dict>
<key>MobileDeviceSoftwareVersions</key><dict>
  <key>Mac99,1</key><dict>
    <key>30A100</key><dict><key>Restore</key><dict>
      <key>BuildVersion</key><string>30A100</string>
      <key>FirmwareSHA1</key><string>ABCDEF0123456789ABCDEF0123456789ABCDEF01</string>
      <key>FirmwareURL</key><string>https://updates.cdn-apple.com/x/1111/UniversalMac_31.0_30A100_Restore.ipsw</string>
      <key>ProductVersion</key><string>31.0</string>
    </dict></dict>
    <key>30A7100</key><dict><key>Restore</key><dict>
      <key>BuildVersion</key><string>30A7100</string>
      <key>FirmwareURL</key><string>https://updates.cdn-apple.com/x/2222/UniversalMac_31.0_30A7100_Restore.ipsw</string>
      <key>ProductVersion</key><string>31.0</string>
    </dict></dict>
  </dict>
  <key>VirtualMac2,1</key><dict>
    <key>30A100</key><dict><key>Restore</key><dict>
      <key>BuildVersion</key><string>30A100</string>
      <key>FirmwareURL</key><string>https://updates.cdn-apple.com/x/1111/UniversalMac_31.0_30A100_Restore.ipsw</string>
      <key>ProductVersion</key><string>31.0</string>
    </dict></dict>
  </dict>
</dict></dict></dict></dict></plist>"#;

    fn entry(v: &str, b: &str) -> Entry {
        Entry {
            version: v.into(),
            build: b.into(),
            url: format!("https://updates.cdn-apple.com/x/UniversalMac_{v}_{b}_Restore.ipsw"),
            sha1: None,
        }
    }

    fn disk(v: &str, b: &str) -> OnDisk {
        OnDisk {
            version: v.into(),
            build: b.into(),
            path: PathBuf::from(format!("/c/UniversalMac_{v}_{b}_Restore.ipsw")),
            size: 1,
        }
    }

    #[test]
    fn the_virtual_macs_entry_takes_its_digest_from_the_same_url() {
        let e = parse(CATALOG.as_bytes()).unwrap();
        assert_eq!(
            e,
            [Entry {
                version: "31.0".into(),
                build: "30A100".into(),
                url: "https://updates.cdn-apple.com/x/1111/UniversalMac_31.0_30A100_Restore.ipsw"
                    .into(),
                sha1: Some("abcdef0123456789abcdef0123456789abcdef01".into()),
            }]
        );
        assert_eq!(e[0].file_name(), "UniversalMac_31.0_30A100_Restore.ipsw");
    }

    #[test]
    fn the_catalog_is_fetched_with_the_pinned_headers() {
        use crate::test_server::Server;
        let s = Server::start(|_, _| (200, vec![], CATALOG.as_bytes().to_vec()));
        let e = fetch(&Client::local(), &s.url("/c.xml")).unwrap();
        assert_eq!(e[0].build, "30A100");
        let h = &s.requests()[0].headers;
        crate::sent_headers::assert_pinned(h, &["connection"]);
        assert_eq!(
            URL,
            "https://mesu.apple.com/assets/macos/com_apple_macOSIPSW/com_apple_macOSIPSW.xml"
        );
        crate::http::check_url(URL).unwrap();
    }

    #[test]
    fn ipsws_on_disk_are_known_by_apples_names() {
        assert_eq!(
            name_parts("UniversalMac_14.6.1_23G93_Restore.ipsw"),
            Some(("14.6.1".into(), "23G93".into()))
        );
        assert_eq!(
            name_parts("UniversalMac_14.6.1_23G93_Restore.ipsw.part"),
            None
        );
        assert_eq!(name_parts("macOS-Tahoe-online-installer.dmg"), None);
        let dir = std::env::temp_dir().join(format!("reims-restore-disk-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("a")).unwrap();
        std::fs::create_dir_all(dir.join("b")).unwrap();
        std::fs::write(dir.join("a/UniversalMac_14.6.1_23G93_Restore.ipsw"), b"x").unwrap();
        std::fs::write(dir.join("b/UniversalMac_27.2.1_26B131_Restore.ipsw"), b"yy").unwrap();
        std::fs::write(dir.join("b/UniversalMac_14.6.1_23G93_Restore.ipsw"), b"x").unwrap();
        std::fs::write(
            dir.join("b/UniversalMac_27.2.1_26B131_Restore.ipsw.part"),
            b"",
        )
        .unwrap();
        let d = on_disk(&[dir.join("a"), dir.join("b"), dir.join("none")]);
        let names: Vec<_> = d.iter().map(|o| (o.build.as_str(), o.size)).collect();
        assert_eq!(names, [("26B131", 2), ("23G93", 1)]);
        assert_eq!(
            d[1].path,
            dir.join("a/UniversalMac_14.6.1_23G93_Restore.ipsw")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_release_is_named_by_version_build_or_latest() {
        let listed = [entry("27.2.1", "26B131"), entry("27.2", "26B123")];
        let have = [disk("14.6.1", "23G93"), disk("27.2", "26B123")];
        let dl = |e: &Entry| Found::Download(e.clone());
        assert_eq!(pick("latest", &have, &listed).unwrap(), dl(&listed[0]));
        assert_eq!(pick("27.2.1", &have, &listed).unwrap(), dl(&listed[0]));
        // an exact version first, then the newest it begins; one on disk over a download
        assert_eq!(
            pick("27.2", &have, &listed).unwrap(),
            Found::OnDisk(have[1].clone())
        );
        assert_eq!(pick("27", &have, &listed).unwrap(), dl(&listed[0]));
        assert_eq!(
            pick("14", &have, &listed).unwrap(),
            Found::OnDisk(have[0].clone())
        );
        assert_eq!(
            pick("23g93", &have, &listed).unwrap(),
            Found::OnDisk(have[0].clone())
        );
        assert_eq!(pick("26B131", &have, &listed).unwrap(), dl(&listed[0]));
        let e = pick("15.6.1", &have, &listed[..1]).unwrap_err().to_string();
        assert_eq!(
            e,
            "Apple offers only the current macOS, 27.2.1 (26B131), to download, and the IPSWs \
             here are 14.6.1 (23G93), 27.2 (26B123): for macOS 15.6.1, give its IPSW's path"
        );
        let e = pick("15.6.1", &[], &[]).unwrap_err().to_string();
        assert_eq!(
            e,
            "there's no macOS 15.6.1 IPSW here, and Apple's catalog couldn't be read: for \
             macOS 15.6.1, give its IPSW's path"
        );
        assert!(pick("27.20", &have, &listed).is_err());
        assert!(pick("sonoma", &have, &listed).is_err());
        assert!(pick("latest", &[], &[]).is_err());
    }

    #[test]
    fn builds_and_versions_parse() {
        for b in ["26B131", "23G93", "25D771280a", "20A5303"] {
            assert!(is_build(b), "{b}");
        }
        for b in ["26b131", "2B13", "26B", "26B131aa", "27.2"] {
            assert!(!is_build(b), "{b}");
        }
        assert!(is_version("27") && is_version("27.2.1"));
        assert!(!is_version("27.") && !is_version(".2") && !is_version("27..1"));
        assert_eq!(
            newer("27.2.1", "26B131", "27.2", "26B123"),
            std::cmp::Ordering::Greater
        );
    }
}
