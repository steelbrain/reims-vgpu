//! Downloading the IPSW Apple's catalog lists (catalog.rs) into a cache directory: from Apple's
//! CDN through the crate's one client (http.rs), into `<name>.part`, which a later download
//! continues by a ranged request, then checked against the SHA-1 Apple's catalog gives and
//! renamed to Apple's own name for the file, so the file is never at that name before it is
//! whole and checked. An IPSW already there is taken as it is: it was checked when it was
//! renamed into place.
//!
//! A download only ever appends to its `.part`: one it can't continue (longer than the file, or
//! where the CDN answers the whole file rather than the rest) is unlinked and a new one made, so
//! no file is truncated or rewritten in place, and another name for the same file (a hard link)
//! keeps what it holds. It learns the file's length first: a `.part` as long as the file is
//! only checked, never fetched over.
//!
//! One download of a file runs at a time: each holds a lock file beside it ([`lock_path`]), and
//! looks for the finished file again once it holds the lock, which the download that finished it
//! removes after the rename.
//!
//! A download reports its step and its progress, stops when its cancel flag is set (keeping
//! what it has, which the next continues), and before it fetches checks that the file system has
//! the space the rest of the file takes: the length Apple's CDN answers, less what is there. One
//! download of a file runs at a time: each holds a lock on its `.<name>.lock` ([`lock_path`]).

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use sha1::{Digest as _, Sha1};

use crate::catalog::{self, Entry};
use crate::http::{Client, content_range};
use crate::pipeline::Event;

/// Tries of a request that fails or a body that stops, in a row, before the download gives up.
const TRIES: u32 = 5;

/// Room left beyond the file, on top of its length.
const MARGIN: u64 = 64 << 20;

/// Downloads `e`'s IPSW into `dir`, or finds it there; its path.
pub fn download(
    client: &Client,
    e: &Entry,
    dir: &Path,
    events: &mut dyn FnMut(Event),
    cancel: &AtomicBool,
) -> Result<PathBuf> {
    download_with(client, e, dir, events, cancel, &free_space)
}

/// [`download`], with the space free where `dir` is from `free`.
fn download_with(
    client: &Client,
    e: &Entry,
    dir: &Path,
    events: &mut dyn FnMut(Event),
    cancel: &AtomicBool,
    free: &dyn Fn(&Path) -> Result<u64>,
) -> Result<PathBuf> {
    let name = e.file_name();
    // Apple's name, which is also a safe one: no directory in it
    ensure!(
        catalog::name_parts(name).is_some(),
        "Apple's catalog names the IPSW {name:?}, not UniversalMac_<version>_<build>_Restore.ipsw"
    );
    std::fs::create_dir_all(dir).with_context(|| format!("make {}", dir.display()))?;
    let done = dir.join(name);
    if done.is_file() {
        return Ok(done);
    }
    let _lock = lock(dir, name)?;
    // Finished by the download that held the lock before.
    if done.is_file() {
        return Ok(done);
    }
    let part_path = dir.join(format!("{name}.part"));
    let what = format!("Downloading macOS {} ({})", e.version, e.build);
    events(Event::Step(what.clone()));

    let total = length(client, &e.url)?;
    let mut part = match std::fs::metadata(&part_path) {
        Ok(m) if m.len() <= total => OpenOptions::new()
            .append(true)
            .open(&part_path)
            .with_context(|| format!("open {}", part_path.display()))?,
        Ok(_) => {
            // longer than the file: start over
            std::fs::remove_file(&part_path)
                .with_context(|| format!("remove {}", part_path.display()))?;
            new_part(&part_path)?
        }
        Err(_) => new_part(&part_path)?,
    };
    let mut have = part.metadata()?.len();
    let mut hash = Sha1::new();
    if have > 0 {
        hash_prefix(&part_path, have, &mut hash, cancel)?;
    }
    let need = total - have + MARGIN;
    let free = free(dir)?;
    if free < need {
        bail!(
            "downloading {name} needs {} free in {}, and there are {}",
            gib(need),
            dir.display(),
            gib(free)
        );
    }
    let mut buf = vec![0u8; 1 << 20];
    let mut tries = 0;
    while have < total {
        ensure!(!cancel.load(Ordering::SeqCst), "the download was stopped");
        let answer = client.get(&e.url, Some(&format!("bytes={have}-")));
        let mut r = match answer {
            Ok((_, r)) if r.status().is_server_error() => {
                retry(
                    &mut tries,
                    anyhow::anyhow!("Apple's CDN answered {}", r.status()),
                )?;
                continue;
            }
            Ok((_, r)) => r,
            Err(err) if is_network(&err) => {
                retry(&mut tries, err)?;
                continue;
            }
            Err(err) => return Err(err),
        };
        match r.status().as_u16() {
            206 => {
                let cr = header(&r, "content-range").and_then(content_range);
                if cr != Some((have, total - 1, total)) {
                    // another range, or the file changed: start over
                    part = start_over(&part_path, &mut have, &mut hash)?;
                    continue;
                }
            }
            200 => {
                // the whole file, not the rest: start over, from this answer
                let len: Option<u64> = header(&r, "content-length").and_then(|v| v.parse().ok());
                ensure!(
                    len == Some(total),
                    "Apple's CDN answered the IPSW as {len:?} bytes, then as {total}"
                );
                if have > 0 {
                    part = start_over(&part_path, &mut have, &mut hash)?;
                }
            }
            s => bail!("Apple's CDN answered {s} for {}", e.url),
        }
        // ureq ends a body read at its limit with an error, not its end (ureq 3.5.1): the file
        // is whole when `have` reaches `total`, whichever the read answered.
        let mut body = r.body_mut().with_config().limit(total - have).reader();
        let read = loop {
            if cancel.load(Ordering::SeqCst) {
                part.flush()?;
                bail!("the download was stopped");
            }
            let n = match body.read(&mut buf) {
                Ok(0) => break Ok(()),
                Ok(n) => n,
                Err(err) => break Err(err),
            };
            part.write_all(&buf[..n])
                .with_context(|| format!("write {}", part_path.display()))?;
            hash.update(&buf[..n]);
            have += n as u64;
            tries = 0;
            events(Event::Progress {
                what: what.clone(),
                done: have,
                total,
            });
        };
        if have == total {
            break;
        }
        // the body stopped short: continue from where it stopped
        let err = match read {
            Err(err) => anyhow::Error::from(err),
            Ok(()) => anyhow::anyhow!("Apple's CDN ended the IPSW at {have} of {total} bytes"),
        };
        retry(&mut tries, err)?;
    }
    part.sync_all()?;
    drop(part);

    let got: String = hash.finalize().iter().map(|b| format!("{b:02x}")).collect();
    match &e.sha1 {
        Some(want) if *want != got => {
            let _ = std::fs::remove_file(&part_path);
            bail!(
                "the downloaded {name} isn't the file Apple's catalog lists: its SHA-1 is {got}, \
                 and Apple's is {want}. It was removed: download it again"
            );
        }
        Some(_) => {}
        None => events(Event::Note(format!(
            "Apple's catalog gives no SHA-1 for {name}: it is kept unchecked"
        ))),
    }
    std::fs::rename(&part_path, &done)
        .with_context(|| format!("rename {} to {name}", part_path.display()))?;
    // Held still, so a download that takes the lock after this finds the file.
    let _ = std::fs::remove_file(lock_path(dir, name));
    Ok(done)
}

/// The length of the file at `url`, from a request for its first byte.
fn length(client: &Client, url: &str) -> Result<u64> {
    let mut tries = 0;
    loop {
        let (_, r) = match client.get(url, Some("bytes=0-0")) {
            Ok(a) => a,
            Err(err) if is_network(&err) => {
                retry(&mut tries, err)?;
                continue;
            }
            Err(err) => return Err(err),
        };
        let total = match r.status().as_u16() {
            206 => header(&r, "content-range")
                .and_then(content_range)
                .map(|c| c.2),
            200 => header(&r, "content-length").and_then(|v| v.parse().ok()),
            s if s >= 500 => {
                retry(&mut tries, anyhow::anyhow!("Apple's CDN answered {s}"))?;
                continue;
            }
            s => bail!("Apple's CDN answered {s} for {url}"),
        };
        return total.context("Apple's CDN answered the IPSW without its length");
    }
}

/// A new, empty `.part`, made where none is.
fn new_part(path: &Path) -> Result<File> {
    OpenOptions::new()
        .create_new(true)
        .append(true)
        .open(path)
        .with_context(|| format!("make {}", path.display()))
}

/// Unlinks the `.part` at `path` and makes a new one in its place, to download the file from
/// its start: the old file is never truncated, so another name for it keeps what it holds.
fn start_over(path: &Path, have: &mut u64, hash: &mut Sha1) -> Result<File> {
    std::fs::remove_file(path).with_context(|| format!("remove {}", path.display()))?;
    *have = 0;
    *hash = Sha1::new();
    new_part(path)
}

/// What downloading `e`'s IPSW into `dir` still takes: the file's length on Apple's CDN, and how
/// much of it `dir` holds (all of it where the file is there).
pub fn remaining(client: &Client, e: &Entry, dir: &Path) -> Result<(u64, u64)> {
    let name = e.file_name();
    if let Ok(m) = std::fs::metadata(dir.join(name)) {
        return Ok((m.len(), m.len()));
    }
    let have = std::fs::metadata(dir.join(format!("{name}.part"))).map_or(0, |m| m.len());
    let total = length(client, &e.url)?;
    // a part longer than the file is started over
    Ok((total, if have > total { 0 } else { have }))
}

/// The lock file a download of `name` holds in its directory: hidden, so a listing of the
/// directory's files leaves it out.
pub fn lock_path(dir: &Path, name: &str) -> PathBuf {
    dir.join(format!(".{name}.lock"))
}

/// Takes the lock of a download of `name` into `dir`, held until the file returned closes, or
/// says another download has it.
fn lock(dir: &Path, name: &str) -> Result<File> {
    let path = lock_path(dir, name);
    let f = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&path)
        .with_context(|| format!("open {}", path.display()))?;
    // flock(LOCK_EX | LOCK_NB) on Unix, LockFileEx on Windows; the lock goes when the file is
    // closed.
    if f.try_lock().is_err() {
        bail!("another download of {name} is running");
    }
    Ok(f)
}

/// Hashes the first `have` bytes of the file at `path` into `hash`, opening it to read alone.
fn hash_prefix(path: &Path, have: u64, hash: &mut Sha1, cancel: &AtomicBool) -> Result<()> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut r = f.take(have);
    let mut buf = vec![0u8; 1 << 20];
    loop {
        ensure!(!cancel.load(Ordering::SeqCst), "the download was stopped");
        let n = r.read(&mut buf)?;
        if n == 0 {
            return Ok(());
        }
        hash.update(&buf[..n]);
    }
}

/// Counts a failed try, waiting before the next, or ends the download with `err` after the last.
fn retry(tries: &mut u32, err: anyhow::Error) -> Result<()> {
    *tries += 1;
    if *tries >= TRIES {
        return Err(err.context(format!("the download failed {TRIES} times in a row")));
    }
    std::thread::sleep(Duration::from_millis(250 << *tries));
    Ok(())
}

/// Whether `err` is the network's, which another try may not meet, rather than a refusal.
fn is_network(err: &anyhow::Error) -> bool {
    err.chain()
        .any(|c| c.downcast_ref::<ureq::Error>().is_some())
}

fn header<'a>(r: &'a ureq::http::Response<ureq::Body>, name: &str) -> Option<&'a str> {
    r.headers().get(name).and_then(|v| v.to_str().ok())
}

/// The space free to this user on the file system `dir` is on.
pub fn free_space(dir: &Path) -> Result<u64> {
    crate::sys::free_space(dir)
}

fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / (1u64 << 30) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_server::{Reply, Server};
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;

    const NAME: &str = "UniversalMac_31.0_30A100_Restore.ipsw";

    fn data() -> Vec<u8> {
        (0..3_000_000u32)
            .map(|i| (i.wrapping_mul(2654435761) >> 24) as u8)
            .collect()
    }

    fn sha1_hex(b: &[u8]) -> String {
        Sha1::digest(b).iter().map(|b| format!("{b:02x}")).collect()
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("reims-restore-dl-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    /// A CDN serving `data` at /NAME by ranges, or whole where `ignore_range`; its first answer
    /// to a request for more than the first byte (the length's) is cut off after `cut` bytes.
    fn cdn(data: Vec<u8>, cut: Option<usize>, ignore_range: bool) -> Server {
        let answers = AtomicUsize::new(0);
        Server::start(move |path, req| {
            if path != format!("/x/{NAME}") {
                return Reply::from((404, vec![], vec![]));
            }
            let len = data.len();
            let range = req.header("range").and_then(|r| r.strip_prefix("bytes="));
            if range == Some("0-0") && !ignore_range {
                return Reply::from((
                    206,
                    vec![("Content-Range".into(), format!("bytes 0-0/{len}"))],
                    data[..1].to_vec(),
                ));
            }
            let a: usize = range
                .and_then(|r| r.strip_suffix('-'))
                .map_or(0, |a| a.parse().unwrap());
            let first = range != Some("0-0") && answers.fetch_add(1, Ordering::SeqCst) == 0;
            let mut r = if ignore_range || range.is_none() {
                Reply::from((200, vec![], data.clone()))
            } else if a >= len {
                Reply::from((
                    416,
                    vec![("Content-Range".into(), format!("bytes */{len}"))],
                    vec![],
                ))
            } else {
                Reply::from((
                    206,
                    vec![(
                        "Content-Range".into(),
                        format!("bytes {a}-{}/{len}", len - 1),
                    )],
                    data[a..].to_vec(),
                ))
            };
            if first {
                r.cut = cut;
            }
            r
        })
    }

    fn entry(s: &Server, sha1: Option<String>) -> Entry {
        Entry {
            version: "31.0".into(),
            build: "30A100".into(),
            url: s.url(&format!("/x/{NAME}")),
            sha1,
        }
    }

    fn plenty(_: &Path) -> Result<u64> {
        Ok(u64::MAX / 2)
    }

    fn run(e: &Entry, dir: &Path, cancel: &AtomicBool) -> (Result<PathBuf>, Vec<Event>) {
        let mut seen = Vec::new();
        let r = download_with(
            &Client::local(),
            e,
            dir,
            &mut |ev| seen.push(ev),
            cancel,
            &plenty,
        );
        (r, seen)
    }

    /// A body that stops part way is continued by a ranged request from where it stopped, and
    /// the file, checked, takes Apple's name.
    #[test]
    fn a_download_continues_where_its_answer_stopped() {
        let d = data();
        let s = cdn(d.clone(), Some(1_000_000), false);
        let dir = temp_dir("resume");
        let (r, ev) = run(
            &entry(&s, Some(sha1_hex(&d))),
            &dir,
            &AtomicBool::new(false),
        );
        assert_eq!(r.unwrap(), dir.join(NAME));
        assert_eq!(std::fs::read(dir.join(NAME)).unwrap(), d);
        assert!(!dir.join(format!("{NAME}.part")).exists());
        // and its lock file is gone with it
        assert!(!lock_path(&dir, NAME).exists());
        let ranges: Vec<_> = s
            .requests()
            .iter()
            .map(|q| q.header("range").unwrap().to_owned())
            .collect();
        assert_eq!(ranges, ["bytes=0-0", "bytes=0-", "bytes=1000000-"]);
        assert_eq!(ev[0], Event::Step("Downloading macOS 31.0 (30A100)".into()));
        assert!(ev.contains(&Event::Progress {
            what: "Downloading macOS 31.0 (30A100)".into(),
            done: 3_000_000,
            total: 3_000_000
        }));
        // a second download finds it, and asks for nothing
        let (r, _) = run(
            &entry(&s, Some(sha1_hex(&d))),
            &dir,
            &AtomicBool::new(false),
        );
        assert_eq!(r.unwrap(), dir.join(NAME));
        assert_eq!(s.requests().len(), 3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A `.part` left by an earlier run is continued; where the CDN answers the whole file
    /// instead of the range, it starts over; a `.part` as long as the file is only checked,
    /// with nothing fetched past the file's length.
    #[test]
    fn a_partial_file_is_continued_or_started_over() {
        let d = data();
        for (ignore_range, have, ranges) in [
            (false, 1_234_567, &["bytes=0-0", "bytes=1234567-"][..]),
            (true, 1_234_567, &["bytes=0-0", "bytes=1234567-"][..]),
            (false, 3_000_000, &["bytes=0-0"][..]),
        ] {
            let s = cdn(d.clone(), None, ignore_range);
            let dir = temp_dir("partial");
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join(format!("{NAME}.part")), &d[..have]).unwrap();
            let (r, _) = run(
                &entry(&s, Some(sha1_hex(&d))),
                &dir,
                &AtomicBool::new(false),
            );
            assert_eq!(r.unwrap(), dir.join(NAME), "{ignore_range} {have}");
            assert_eq!(std::fs::read(dir.join(NAME)).unwrap(), d);
            let asked: Vec<_> = s
                .requests()
                .iter()
                .map(|q| q.header("range").unwrap().to_owned())
                .collect();
            assert_eq!(asked, ranges);
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    /// A `.part` that is another name for a file (a hard link) keeps that file as it was when
    /// the download starts over: the part is unlinked and made anew, never truncated or
    /// rewritten in place. So does a `.part` longer than the file.
    #[test]
    fn starting_over_leaves_another_name_for_the_part_as_it_was() {
        let d = data();
        for (ignore_range, kept) in [(true, vec![9u8; 1000]), (false, vec![9u8; 3_000_001])] {
            let s = cdn(d.clone(), None, ignore_range);
            let dir = temp_dir("links");
            std::fs::create_dir_all(&dir).unwrap();
            let other = dir.join("other.bin");
            std::fs::write(&other, &kept).unwrap();
            std::fs::hard_link(&other, dir.join(format!("{NAME}.part"))).unwrap();
            let (r, _) = run(
                &entry(&s, Some(sha1_hex(&d))),
                &dir,
                &AtomicBool::new(false),
            );
            assert_eq!(r.unwrap(), dir.join(NAME), "{ignore_range}");
            assert_eq!(std::fs::read(dir.join(NAME)).unwrap(), d);
            assert_eq!(std::fs::read(&other).unwrap(), kept, "{ignore_range}");
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    /// A file whose SHA-1 isn't Apple's is removed, and the error says so.
    #[test]
    fn a_download_that_isnt_apples_file_is_removed() {
        let d = data();
        let s = cdn(d, None, false);
        let dir = temp_dir("digest");
        let wrong = "0".repeat(40);
        let (r, _) = run(
            &entry(&s, Some(wrong.clone())),
            &dir,
            &AtomicBool::new(false),
        );
        let e = r.unwrap_err().to_string();
        assert!(
            e.contains(&format!("and Apple's is {wrong}. It was removed")),
            "{e}"
        );
        assert!(!dir.join(NAME).exists());
        assert!(!dir.join(format!("{NAME}.part")).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A redirect off the allowed hosts is refused before anything is fetched there.
    #[test]
    fn a_redirect_off_apples_hosts_is_refused() {
        let s = Server::start(|_, _| {
            (
                302,
                vec![("Location".into(), format!("https://example.com/{NAME}"))],
                vec![],
            )
        });
        let dir = temp_dir("redirect");
        let (r, _) = run(&entry(&s, None), &dir, &AtomicBool::new(false));
        let e = format!("{:#}", r.unwrap_err());
        assert!(
            e.contains(&format!("https://example.com/{NAME} is not local")),
            "{e}"
        );
        assert_eq!(s.requests().len(), 1);
        assert!(!dir.join(NAME).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A stopped download keeps what it has, and the next continues it.
    #[test]
    fn a_stopped_download_keeps_its_part_for_the_next() {
        let d = data();
        let s = cdn(d.clone(), None, false);
        let dir = temp_dir("cancel");
        let cancel = Arc::new(AtomicBool::new(false));
        let mut events = 0;
        let stop = cancel.clone();
        let r = download_with(
            &Client::local(),
            &entry(&s, Some(sha1_hex(&d))),
            &dir,
            &mut |e| {
                if matches!(e, Event::Progress { .. }) {
                    events += 1;
                    stop.store(true, Ordering::SeqCst);
                }
            },
            &cancel,
            &plenty,
        );
        assert_eq!(r.unwrap_err().to_string(), "the download was stopped");
        assert_eq!(events, 1);
        let kept = std::fs::metadata(dir.join(format!("{NAME}.part")))
            .unwrap()
            .len();
        assert!(kept > 0 && kept < d.len() as u64, "{kept}");
        assert!(!dir.join(NAME).exists());
        let (r, _) = run(
            &entry(&s, Some(sha1_hex(&d))),
            &dir,
            &AtomicBool::new(false),
        );
        assert_eq!(r.unwrap(), dir.join(NAME));
        assert_eq!(std::fs::read(dir.join(NAME)).unwrap(), d);
        assert_eq!(
            s.requests().last().unwrap().header("range").unwrap(),
            format!("bytes={kept}-")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The space the rest of the file takes is checked against what is free, from the length
    /// the CDN answers.
    #[test]
    fn a_download_checks_the_space_the_rest_of_the_file_takes() {
        let d = data();
        let s = cdn(d.clone(), None, false);
        let dir = temp_dir("space");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(format!("{NAME}.part")), &d[..1_000_000]).unwrap();
        let r = download_with(
            &Client::local(),
            &entry(&s, None),
            &dir,
            &mut |_| {},
            &AtomicBool::new(false),
            &|_| Ok(MARGIN + 1_999_999),
        );
        let e = r.unwrap_err().to_string();
        assert!(
            e.starts_with(&format!("downloading {NAME} needs 0.1 GiB free in")),
            "{e}"
        );
        let r = download_with(
            &Client::local(),
            &entry(&s, None),
            &dir,
            &mut |_| {},
            &AtomicBool::new(false),
            &|_| Ok(MARGIN + 2_000_000),
        );
        assert_eq!(r.unwrap(), dir.join(NAME));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The download's requests carry the pinned headers and the range alone.
    #[test]
    fn a_download_sends_only_the_pinned_headers() {
        let d = data();
        let s = cdn(d.clone(), None, false);
        let dir = temp_dir("headers");
        let (r, _) = run(
            &entry(&s, Some(sha1_hex(&d))),
            &dir,
            &AtomicBool::new(false),
        );
        r.unwrap();
        for q in s.requests() {
            crate::sent_headers::assert_pinned(&q.headers, &["range"]);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// One download of a file at a time.
    #[test]
    fn only_one_download_of_a_file_runs_at_a_time() {
        let s = cdn(data(), None, false);
        let dir = temp_dir("lock");
        std::fs::create_dir_all(&dir).unwrap();
        let held = lock(&dir, NAME).unwrap();
        let (r, _) = run(&entry(&s, None), &dir, &AtomicBool::new(false));
        assert_eq!(
            r.unwrap_err().to_string(),
            format!("another download of {NAME} is running")
        );
        assert!(s.requests().is_empty());
        drop(held);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// What a download still takes: the CDN's length, less what is here.
    #[test]
    fn what_a_download_still_takes() {
        let d = data();
        let s = cdn(d.clone(), None, false);
        let dir = temp_dir("remaining");
        let e = entry(&s, None);
        let c = Client::local();
        assert_eq!(remaining(&c, &e, &dir).unwrap(), (3_000_000, 0));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(format!("{NAME}.part")), &d[..1000]).unwrap();
        assert_eq!(remaining(&c, &e, &dir).unwrap(), (3_000_000, 1000));
        assert_eq!(s.requests()[0].header("range"), Some("bytes=0-0"));
        std::fs::write(dir.join(NAME), &d).unwrap();
        assert_eq!(remaining(&c, &e, &dir).unwrap(), (3_000_000, 3_000_000));
        assert_eq!(s.requests().len(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn only_apples_file_names_are_written() {
        let s = cdn(data(), None, false);
        let mut e = entry(&s, None);
        e.url = s.url("/x/../../etc/passwd");
        let dir = temp_dir("name");
        let (r, _) = run(&e, &dir, &AtomicBool::new(false));
        assert!(r.unwrap_err().to_string().contains("not UniversalMac_"));
        assert!(s.requests().is_empty());
    }
}
