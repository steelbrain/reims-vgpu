//! The measured pick's verdict, kept for the host in one file, `device-checks/gpu-choice` under
//! the library's cache directory (`crate::platform::cache_directory`): which of the timed GPUs
//! the workload found fastest, under a key of every timed GPU's UUID and driver version, so only
//! the first open after a GPU or driver change times them ([`super::measured`]).
//!
//! The file is text, one field a line:
//!
//! ```text
//! reims-vgpu gpu choice 1
//! key 00000000040000000000000000000000:109060102,00000000c50000000000000000000000:109060102
//! pick 00000000c50000000000000000000000
//! time 00000000c50000000000000000000000 21544
//! time 00000000040000000000000000000000 40711
//! ```
//!
//! The key lists each GPU as its UUID in hex and its driver version in decimal, in the order of
//! the UUIDs; `time` lines hold each GPU's workload in microseconds, for the record. A file that
//! is missing, unreadable, of another header or another key, or whose pick is not among the key's
//! GPUs answers nothing ([`read`]), and the choice times the GPUs again and writes the file anew
//! ([`write`]), through a file beside it renamed over it, so a reader never meets half a file.

use std::fmt::Write as _;
use std::path::Path;
use std::time::Duration;

use super::Seen;

/// The file's first line: its format.
const HEADER: &str = "reims-vgpu gpu choice 1";

/// The key of the GPUs `timed`: each one's UUID and driver version, in the order of the UUIDs;
/// `None` where one has no UUID.
pub(super) fn key(timed: &[&Seen]) -> Option<String> {
    let mut parts: Vec<([u8; 16], u32)> = timed
        .iter()
        .map(|s| Some((s.uuid?, s.driver_version)))
        .collect::<Option<_>>()?;
    parts.sort_unstable();
    let parts: Vec<String> = parts
        .iter()
        .map(|(uuid, driver)| format!("{}:{driver}", hex(uuid)))
        .collect();
    Some(parts.join(","))
}

/// The UUID of the GPU the verdict at `path` picked under `key`: `None` where the file is
/// missing, unreadable, of another format or another key, or picks a GPU the key does not list.
pub(super) fn read(path: &Path, key: &str) -> Option<[u8; 16]> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut lines = text.lines();
    if lines.next()? != HEADER || lines.next()?.strip_prefix("key ")? != key {
        return None;
    }
    let pick = lines.next()?.strip_prefix("pick ")?;
    let listed = key
        .split(',')
        .any(|part| part.split(':').next() == Some(pick));
    listed.then(|| uuid(pick)).flatten()
}

/// Writes the verdict that `pick` is the fastest of the GPUs under `key`, with each one's time,
/// to `path`, making its directory where there is none: through a file beside it, renamed over
/// it.
pub(super) fn write(
    path: &Path,
    key: &str,
    pick: &[u8; 16],
    times: &[([u8; 16], Duration)],
) -> std::io::Result<()> {
    let mut text = format!("{HEADER}\nkey {key}\npick {}\n", hex(pick));
    for (uuid, time) in times {
        let _ = writeln!(text, "time {} {}", hex(uuid), time.as_micros());
    }
    if let Some(directory) = path.parent() {
        std::fs::create_dir_all(directory)?;
    }
    let beside = path.with_extension(format!("{}.new", std::process::id()));
    std::fs::write(&beside, text)?;
    std::fs::rename(&beside, path).inspect_err(|_| {
        let _ = std::fs::remove_file(&beside);
    })
}

/// `uuid` in lowercase hex, 32 digits.
fn hex(uuid: &[u8; 16]) -> String {
    uuid.iter().fold(String::with_capacity(32), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

/// The UUID of 32 hex digits `text`; `None` where it is anything else.
fn uuid(text: &str) -> Option<[u8; 16]> {
    if text.len() != 32 || !text.is_ascii() {
        return None;
    }
    let mut uuid = [0u8; 16];
    for (at, byte) in uuid.iter_mut().enumerate() {
        *byte = u8::from_str_radix(text.get(2 * at..2 * at + 2)?, 16).ok()?;
    }
    Some(uuid)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::tests::{discrete, integrated};
    use super::*;

    /// A fresh directory of this test's own under the system's temporary directory, removed
    /// where the test ends.
    pub(crate) struct Scratch(pub(crate) std::path::PathBuf);

    impl Scratch {
        pub(crate) fn new(name: &str) -> Scratch {
            let path = std::env::temp_dir().join(format!(
                "rvg-gpu-choice-{name}-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            let _ = std::fs::remove_dir_all(&path);
            Scratch(path)
        }

        pub(crate) fn file(&self) -> std::path::PathBuf {
            self.0.join("device-checks").join("gpu-choice")
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// The key lists every GPU's UUID and driver version, in the UUIDs' order whatever the
    /// loader's, and none where a GPU has no UUID
    #[test]
    fn the_key_is_every_gpus_uuid_and_driver() {
        let (a, b) = (integrated(11, 0xc5), discrete(4, 0x04));
        let key = key(&[&a, &b]).unwrap();
        assert_eq!(
            key,
            "04000000000000000000000000000000:1,c5000000000000000000000000000000:1"
        );
        assert_eq!(super::key(&[&b, &a]).unwrap(), key);
        let newer = Seen {
            driver_version: 2,
            ..b.clone()
        };
        assert_ne!(super::key(&[&a, &newer]).unwrap(), key);
        let no_uuid = Seen { uuid: None, ..b };
        assert_eq!(super::key(&[&a, &no_uuid]), None);
    }

    /// A verdict written is read back under its key, and under no other: another driver version
    /// or another set of GPUs reads nothing
    #[test]
    fn a_verdict_is_read_under_its_key_alone() {
        let scratch = Scratch::new("key");
        let (a, b) = (integrated(11, 0xc5), discrete(4, 0x04));
        let key = key(&[&a, &b]).unwrap();
        let (ua, ub) = (a.uuid.unwrap(), b.uuid.unwrap());
        let times = [
            (ua, Duration::from_micros(21_544)),
            (ub, Duration::from_micros(40_711)),
        ];
        write(&scratch.file(), &key, &ua, &times).unwrap();
        assert_eq!(read(&scratch.file(), &key), Some(ua));
        let newer = Seen {
            driver_version: 2,
            ..b.clone()
        };
        assert_eq!(
            read(&scratch.file(), &super::key(&[&a, &newer]).unwrap()),
            None
        );
        assert_eq!(read(&scratch.file(), &super::key(&[&a]).unwrap()), None);
        let text = std::fs::read_to_string(scratch.file()).unwrap();
        assert!(text.contains(&format!("time {} 40711", hex(&ub))), "{text}");
    }

    /// A missing, empty, truncated or garbled file, one of another format, and one whose pick the
    /// key does not list read nothing
    #[test]
    fn a_corrupt_verdict_reads_nothing() {
        let scratch = Scratch::new("corrupt");
        let (a, b) = (integrated(11, 0xc5), discrete(4, 0x04));
        let key = key(&[&a, &b]).unwrap();
        assert_eq!(read(&scratch.file(), &key), None);
        std::fs::create_dir_all(scratch.file().parent().unwrap()).unwrap();
        let other = hex(&[0x77; 16]);
        let ua = hex(&a.uuid.unwrap());
        for text in [
            String::new(),
            format!("{HEADER}\n"),
            format!("{HEADER}\nkey {key}\n"),
            format!("{HEADER}\nkey {key}\npick {}\n", &ua[..31]),
            format!("{HEADER}\nkey {key}\npick {other}\n"),
            format!("reims-vgpu gpu choice 2\nkey {key}\npick {ua}\n"),
            "\u{0}\u{ff}garbage".to_owned(),
        ] {
            std::fs::write(scratch.file(), &text).unwrap();
            assert_eq!(read(&scratch.file(), &key), None, "{text:?}");
        }
        std::fs::write(scratch.file(), [0xff, 0xfe, 0x00]).unwrap();
        assert_eq!(read(&scratch.file(), &key), None);
    }

    /// UUIDs go to hex and back; anything but 32 hex digits is none
    #[test]
    fn a_uuid_reads_as_it_is_written() {
        let u: [u8; 16] = std::array::from_fn(|i| (i * 17) as u8);
        assert_eq!(uuid(&hex(&u)), Some(u));
        assert_eq!(uuid("00"), None);
        assert_eq!(uuid(&"g".repeat(32)), None);
        assert_eq!(uuid(&"é".repeat(16)), None);
    }
}
