//! The runtime's pipeline records, kept across processes, and the warm pass each device runs over
//! them: what stands in for Metal's own pipeline caches across a host's processes, as the shader
//! cache ([`crate::platform`]'s `SHADER_CACHE`) stands in for its compiled functions.
//!
//! - **Recorded.** Each pipeline state a device creates hands the runtime's record of it
//!   (`DeviceOptions::pipeline_records`) to [`hook`], on the creating thread; the hook only sends
//!   it to the writer thread (`reims-vgpu-records`), which appends each record the file does not
//!   hold yet to the records file ([`RECORDS`], beside the shader cache's directory, whose
//!   libraries the records name), once, in the order first made. The file is appended to and
//!   never rewritten; past [`CAPACITY`] bytes nothing more is appended.
//! - **Warmed.** Each device the process adopts hands the file's records to the runtime's
//!   `Device::warm` ([`warm`]), with an explicit budget ([`WARM_BUDGET`]), from a thread of its
//!   own (`reims-vgpu-warm`), which reads the file, so the device's open waits for no read. The
//!   runtime's pass makes their translations and driver records on a thread of the device's, its
//!   makings below every other making of the process, and changes no answer: no state,
//!   identifier or resource ID the guest sees. Once the pass's counts reach the records handed
//!   (`Device::warm_statistics`), one info line says what became of them ([`summary`]). So a
//!   guest's first creations of the states an earlier process made find them made, after a host
//!   restart, and after an update of the runtime, whose translations of the earlier build's the
//!   shader cache keeps apart (a record holds a descriptor, never a translation).
//! - **Left out once it stood.** A record whose warm making stood [`RECORD_BOUND`] without ending
//!   (a driver compile that never ends, or runs for minutes) is left out of every later pass: the
//!   warm thread appends a tombstone of it ([`leave_out`]) as the bound passes, so a process
//!   killed meanwhile leaves it too, and [`read`] hands no pass a record the file holds a
//!   tombstone of. A host then pays for such a making in the start whose pass stands on it the
//!   bound, not at every start: its driver memory, the records behind it the pass does not reach,
//!   and the exit's wait for it. Only the pass leaves it out: the guest's own creation of its
//!   state is made as ever, and the hook keeps its record as before. The record in making is read
//!   from the pass's counts ([`Watch`]), and its time from this thread's polls.
//! - **The file's format**: frames, each [`MAGIC`], the payload's length and the FNV-1a hash of
//!   its bytes (each a little-endian `u32`), then its bytes: a record's, or a tombstone's,
//!   [`UNFINISHED`] followed by the bytes of the record it leaves out. A reader skips a frame
//!   whose length runs past the file or whose hash does not match (a process that died
//!   mid-write, or two writing at once), and goes on at the next [`MAGIC`] after its start; a
//!   record of another format the runtime reads as none (`PipelineRecord::from_bytes`) and the
//!   pass drops, and an earlier build, which knows no tombstone, hands the runtime a tombstone's
//!   bytes, which it reads so.
//!
//! A test build's devices keep and warm no records ([`KEPT`]): this crate's unit tests', and every
//! crate's tests', which turn on its `testkit` feature (only dev-dependencies do, `cargo xtask
//! testkit`). So a test of what a device compiles runs as it would on a host that never ran a
//! guest, no test's pipelines join the records a host's next process warms, and no test process
//! makes the host's guests' pipelines beside the run's other tests.

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};

use crate::core::platform::os::clock::Clock as _;
use crate::core::sync::Mutex;
use metal2vulkan_runtime as rt;

/// Whether this build's devices keep records and warm from them: all but a test build's (the
/// module docs), whose `device_options` sets no hook and whose `gpu::device::adopt` starts no warm
/// pass.
pub(crate) const KEPT: bool = !cfg!(any(test, feature = "testkit"));

/// The records file under `HOME`, beside the shader cache's directory.
pub(crate) const RECORDS: &str = ".cache/reims-vgpu-vulkan-pipelines";

/// The most bytes the records file holds: past them, no record is appended.
pub(crate) const CAPACITY: u64 = 64 << 20;

/// The driver's pipeline cache data one warm pass may add, in bytes (`Device::warm`'s budget): a
/// backstop, half the shader cache's cap ([`SHADER_CACHE_CAPACITY`], 25 GiB on a host). What
/// bounds the pass is the records it is handed, every one the file holds, within [`CAPACITY`]: it
/// stops before its next record only once its makings added this much, counting the records left
/// as not run, as past it its writes would push out of the cache, the least recently used first,
/// what the guest's boots read from it. A making whose data the shader cache holds already adds
/// nothing, so the budget can bind only where the records' data is made anew, as after a change of
/// the translator or the driver, and then only where a driver's data for each record is large; no
/// driver's size is assumed. **Measured** 2026-11-12 on homelab-um790-xtx-1's RTX 2080 (NVIDIA
/// 620.70.06), a macOS 26 guest's walk after a translator change (job m3-max-23661), the budget
/// 4 GiB then: of 4,565 records, 2,573 were made, 130 translated only and 130 failed, 1 was unread,
/// and 1,731 were not run, the budget spent by 4.30 GB of driver data within 50 s. That driver's
/// records were about 1.85 MB each, nearly all of it one section the same in every record (as
/// metal2vulkan read them), so the 4,565 add about 8.5 GB, within the half.
///
/// The bytes counted are what each making writes to its record, before the shader cache's store
/// keeps a large record's chunks once (metal2vulkan's dedup, `RUNTIME_VERSION` 2187), so on such
/// a driver they can be many times what the disk takes. **Measured** 2026-11-12 on the same host
/// and guest, the first walk after the dedup landed (job m3-max-24343): of 4,604 records, 4,123
/// were made, 474 translated only and 6 failed, 1 was unread and none was not run, 5.64 GB
/// counted within 112 s, while the shader cache's directory grew from 10,498 to 10,858 MiB; a
/// 4 GiB budget would have stopped that pass with about a quarter of its records not run. The
/// next walk's pass (job m3-max-24344, a macOS 15.8.2 guest) added nothing, within 15.75 s.
///
/// [`SHADER_CACHE_CAPACITY`]: crate::platform::SHADER_CACHE_CAPACITY
pub(crate) const WARM_BUDGET: u64 = crate::platform::SHADER_CACHE_CAPACITY / 2;

/// How often the warm thread looks whether the runtime's pass has ended.
const WARM_POLL: std::time::Duration = std::time::Duration::from_millis(250);

/// How long the warm pass may stand on one record before later passes leave it out (the module
/// docs, [`Watch`]): the bound the runtime's exit gives a pipeline making in the driver (its exit
/// waits at most 60 s for one). **Measured** 2026-11-12 on homelab-ms01-1's Iris Xe (ANV, Xe-LP),
/// macOS 26.7.2 guests' boots over 2,983 records: the making of one, a compute kernel whose
/// driver compile does not end, held the pass at every start, the process's glibc heap in use going
/// from 22 MB to 4.57 GB once the pass reached it and the boot's peak from 9.6 to 18-20 GiB, and
/// each exit waited the bound for it; the log of such a boot (job m3-max-25012) names 1,520
/// makings, and each of the 1,519 others took under 10 ms.
///
/// The time is this thread's, from the poll where the pass's counts last moved: it counts the
/// pass's waits behind the process's other makings (the runtime ranks the pass's makings below
/// all of them), so a record the guest's makings starve that long is left out as well, and only
/// warmed no more: the guest's creation of its state makes it as ever.
pub(crate) const RECORD_BOUND: std::time::Duration = std::time::Duration::from_secs(60);

/// The start of each frame of the records file.
pub(crate) const MAGIC: u32 = u32::from_le_bytes(*b"RVGR");

/// The start of a tombstone's payload, the bytes of the record it leaves out following (the
/// module docs): never a record's start, which is the runtime's format name (`M2VPREC`).
pub(crate) const UNFINISHED: &[u8] = b"reims-vgpu: unfinished warm making\0";

/// The bytes of a frame before its record's.
const FRAME_HEAD: usize = 12;

/// The records file's path under `home`.
pub(crate) fn path_in(home: &Path) -> PathBuf {
    home.join(RECORDS)
}

/// FNV-1a over `bytes`, 32 bits: each frame's check.
fn fnv1a(bytes: &[u8]) -> u32 {
    bytes.iter().fold(0x811c_9dc5u32, |h, &b| {
        (h ^ u32::from(b)).wrapping_mul(0x0100_0193)
    })
}

/// The frame of `record`'s bytes.
pub(crate) fn frame(record: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(FRAME_HEAD + record.len());
    out.extend_from_slice(&MAGIC.to_le_bytes());
    out.extend_from_slice(&(record.len() as u32).to_le_bytes());
    out.extend_from_slice(&fnv1a(record).to_le_bytes());
    out.extend_from_slice(record);
    out
}

/// The records `file` holds, in its order: each whole frame's bytes (the module docs).
pub(crate) fn split(file: &[u8]) -> Vec<&[u8]> {
    let word = |at: usize| {
        file.get(at..at + 4)
            .and_then(|b| <[u8; 4]>::try_from(b).ok())
            .map(u32::from_le_bytes)
    };
    let mut out = Vec::new();
    let mut at = 0;
    while at + FRAME_HEAD <= file.len() {
        if word(at) != Some(MAGIC) {
            at += 1;
            continue;
        }
        let len = word(at + 4).unwrap_or_default() as usize;
        let body = at + FRAME_HEAD;
        match file.get(body..body.saturating_add(len)) {
            Some(record) if Some(fnv1a(record)) == word(at + 8) => {
                out.push(record);
                at = body + len;
            }
            _ => at += 1,
        }
    }
    out
}

/// The key a record's bytes are kept once by.
fn key(record: &[u8]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    record.hash(&mut h);
    h.finish()
}

/// The records file at `path` as the writer keeps it: open for appending, the keys of the records
/// it holds, and its bytes.
struct Keeper {
    path: PathBuf,
    file: Option<File>,
    held: HashSet<u64>,
    bytes: u64,
}

impl Keeper {
    /// The file at `path` as it stands: its records' keys read, opened to append (none where it
    /// cannot be).
    fn open(path: PathBuf) -> Keeper {
        let read = std::fs::read(&path).unwrap_or_default();
        let held = split(&read).into_iter().map(key).collect();
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let file = OpenOptions::new().create(true).append(true).open(&path);
        if let Err(e) = &file {
            log::warn!(
                target: "reims_vgpu_vulkan::records",
                "the pipeline records file {} did not open: {e}; no record is kept",
                path.display()
            );
        }
        Keeper {
            path,
            file: file.ok(),
            held,
            bytes: read.len() as u64,
        }
    }

    /// Appends `record` where the file does not hold it and has room: one write of its frame, so
    /// a process appending at once lands before or after it.
    fn keep(&mut self, record: &[u8]) {
        let Some(file) = self.file.as_mut() else {
            return;
        };
        let framed = frame(record);
        if self.bytes + framed.len() as u64 > CAPACITY || !self.held.insert(key(record)) {
            return;
        }
        match file.write_all(&framed) {
            Ok(()) => self.bytes += framed.len() as u64,
            Err(e) => {
                log::warn!(
                    target: "reims_vgpu_vulkan::records", kind:% = e;
                    "a pipeline record was not appended to {}: {e}; no more are kept",
                    self.path.display()
                );
                self.file = None;
            }
        }
    }
}

/// The writer's end of the records file at `path`: a thread that appends each record sent
/// ([`Keeper::keep`]), reading the file first, until every sender is gone.
fn writer(path: PathBuf) -> Option<Sender<rt::PipelineRecord>> {
    let (send, receive): (Sender<rt::PipelineRecord>, Receiver<rt::PipelineRecord>) = channel();
    let started = std::thread::Builder::new()
        .name("reims-vgpu-records".into())
        .spawn(move || {
            let mut keeper = Keeper::open(path);
            for record in receive {
                keeper.keep(&record.to_bytes());
            }
        });
    started.ok().map(|_| send)
}

/// The hook `DeviceOptions::pipeline_records` takes, keeping each record in the file at `path`
/// (the module docs): it sends the record to the writer thread and returns. None where the
/// thread did not start.
pub(crate) fn hook(path: PathBuf) -> Option<Arc<dyn Fn(rt::PipelineRecord) + Send + Sync>> {
    let send = Mutex::new(writer(path)?);
    Some(Arc::new(move |record| {
        let _ = send.lock().send(record);
    }))
}

/// The records of `frames`, a records file's ([`split`]), that a warm pass is handed, in their
/// order: every one but those a tombstone leaves out (the module docs), and how many those are.
fn warmable<'a>(frames: &[&'a [u8]]) -> (Vec<&'a [u8]>, usize) {
    let unfinished: HashSet<&[u8]> = frames
        .iter()
        .filter_map(|frame| frame.strip_prefix(UNFINISHED))
        .collect();
    let records = frames.iter().filter(|frame| !frame.starts_with(UNFINISHED));
    let (left_out, handed): (Vec<&[u8]>, Vec<&[u8]>) =
        records.partition(|record| unfinished.contains(*record));
    (handed, left_out.len())
}

/// The records of the file at `path` a warm pass is handed, in its order ([`warmable`]), and how
/// many its tombstones leave out.
pub(crate) fn read(path: &Path) -> (Vec<rt::PipelineRecord>, usize) {
    let bytes = std::fs::read(path).unwrap_or_default();
    let (records, left_out) = warmable(&split(&bytes));
    let records = records
        .into_iter()
        .filter_map(rt::PipelineRecord::from_bytes)
        .collect();
    (records, left_out)
}

/// Appends to the file at `path` the tombstone of `record` (the module docs): one write of its
/// frame, as the keeper's ([`Keeper::keep`]), past [`CAPACITY`] too, since no later pass makes a
/// record with a tombstone, so tombstones stay as few as the records whose makings stood.
pub(crate) fn leave_out(path: &Path, record: &[u8]) -> std::io::Result<()> {
    let mut tombstone = UNFINISHED.to_vec();
    tombstone.extend_from_slice(record);
    OpenOptions::new()
        .append(true)
        .open(path)?
        .write_all(&frame(&tombstone))
}

/// The warm thread's watch over the record the pass makes (the module docs): the pass makes its
/// records one at a time, in the order handed, and counts each as it ends, so while it runs the
/// sum of its record counts is the place of the record it makes, among the records handed.
struct Watch {
    /// The records handed.
    handed: u64,
    /// The sum of the counts at the last poll that moved it.
    at: u64,
    /// When that poll was, on the monotonic clock in nanoseconds.
    since: u64,
    /// Whether the record at `at` overran already.
    overran: bool,
}

impl Watch {
    /// The watch over a pass of `handed` records, as it starts at `now`.
    fn new(handed: u64, now: u64) -> Watch {
        Watch {
            handed,
            at: 0,
            since: now,
            overran: false,
        }
    }

    /// The place of the record the pass has stood on for [`RECORD_BOUND`] by `now`, by its counts
    /// then (`statistics`), once a record; `None` while none has.
    fn overran(&mut self, statistics: &rt::WarmStatistics, now: u64) -> Option<u64> {
        let at = counted(statistics);
        if at != self.at {
            (self.at, self.since, self.overran) = (at, now, false);
            return None;
        }
        let stood = now.saturating_sub(self.since) >= RECORD_BOUND.as_nanos() as u64;
        (stood && at < self.handed && !std::mem::replace(&mut self.overran, true)).then_some(at)
    }
}

/// The records a warm pass's counts say ended, each warmed, translated only, failed, unread, not
/// warmable or not run.
fn counted(s: &rt::WarmStatistics) -> u64 {
    s.records_warmed
        + s.records_translated
        + s.records_failed
        + s.records_unread
        + s.records_not_warmable
        + s.records_not_run
}

/// Hands `device` the records of the file at `path`, from a thread of its own that reads the
/// file, so the caller waits for nothing; the runtime's pass then runs on the device's own thread
/// (the module docs). The thread ends at `stop`, as the device goes ([`crate::let_go`]).
pub(crate) fn warm(
    device: &rt::Device,
    path: PathBuf,
    stop: &Arc<crate::let_go::Stop>,
) -> Option<std::thread::JoinHandle<()>> {
    let (device, stop) = (device.clone(), Arc::clone(stop));
    let spawned = std::thread::Builder::new()
        .name("reims-vgpu-warm".into())
        .spawn(move || {
            let (records, left_out) = read(&path);
            if records.is_empty() || stop.stopped() {
                return;
            }
            log::info!(
                target: "reims_vgpu_vulkan::records",
                "warming {} pipeline records from {} ({left_out} left out: their warm making \
                 stood {} s in an earlier pass)",
                records.len(),
                path.display(),
                RECORD_BOUND.as_secs()
            );
            let handed = records.len() as u64;
            // A recovery starts the counts of the new generation's passes from none, and ends this
            // one: past it, the counts name no record of this pass.
            let recoveries = device.work_counts().device_recoveries;
            device.warm(records.clone(), WARM_BUDGET);
            // The pass runs on the runtime's thread and says nothing when it ends: its counts
            // reach the records handed then. The device's end stops the polls, which hold its
            // runtime device.
            let clock = crate::os::OsClock::new();
            let mut watch = Watch::new(handed, clock.now_mono().as_nanos());
            let mut polls = 0u64;
            loop {
                let statistics = device.warm_statistics();
                if let Some(line) = summary(&statistics, handed) {
                    log::info!(
                        target: "reims_vgpu_vulkan::records",
                        "{line}, within {:.2} s",
                        (polls * WARM_POLL.as_millis() as u64) as f64 / 1000.0
                    );
                    return;
                }
                if let Some(at) = watch.overran(&statistics, clock.now_mono().as_nanos())
                    && device.work_counts().device_recoveries == recoveries
                    && let Some(record) = records.get(at as usize)
                {
                    let left = leave_out(&path, &record.to_bytes());
                    log::warn!(
                        target: "reims_vgpu_vulkan::records",
                        "the warm pass's making of pipeline record {at} of {handed} stood {} s \
                         without ending: {}; the guest's own creation of its state still makes it",
                        RECORD_BOUND.as_secs(),
                        match left {
                            Ok(()) => "later passes leave it out".to_owned(),
                            Err(e) => format!("its tombstone was not appended ({e})"),
                        }
                    );
                }
                polls += 1;
                if stop.wait_until(clock.now_mono().saturating_add(WARM_POLL)) {
                    return;
                }
            }
        });
    spawned.ok()
}

/// What a warm pass of `handed` records made of them, once its counts add up to them (each record
/// warmed, translated only, failed, unread, not warmable or not run, by `statistics`); `None`
/// while the pass runs.
pub(crate) fn summary(statistics: &rt::WarmStatistics, handed: u64) -> Option<String> {
    let s = statistics;
    (counted(s) >= handed).then(|| {
        format!(
            "warmed {handed} pipeline records: {} made, {} translated only, {} failed, {} unread, \
             {} not warmable, {} not run (the budget spent or the process ending); {} bytes of \
             driver pipeline data added",
            s.records_warmed,
            s.records_translated,
            s.records_failed,
            s.records_unread,
            s.records_not_warmable,
            s.records_not_run,
            s.pipeline_bytes
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unique scratch path under the system's temporary directory, removed at the end.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str) -> Scratch {
            let dir = std::env::temp_dir().join(format!(
                "rvg-records-{name}-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Scratch(dir)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// The pass's line comes once every record handed is counted, and names each count
    #[test]
    fn the_pass_is_summed_up_once_every_record_is_counted() {
        let mut s = rt::WarmStatistics::default();
        s.records_warmed = 300;
        s.records_translated = 2;
        s.records_unread = 1;
        assert_eq!(summary(&s, 312), None);
        s.records_failed = 4;
        s.records_not_warmable = 1;
        s.records_not_run = 4;
        s.pipeline_bytes = 1 << 20;
        let line = summary(&s, 312).unwrap();
        assert_eq!(
            line,
            "warmed 312 pipeline records: 300 made, 2 translated only, 4 failed, 1 unread, 1 not \
             warmable, 4 not run (the budget spent or the process ending); 1048576 bytes of \
             driver pipeline data added"
        );
    }

    /// The warm pass's budget is a backstop, half the shader cache's cap, and on a host's cap it
    /// holds the RTX 2080 walk's 4,565 records at that driver's 1.85 MB a record (the budget's
    /// doc)
    #[test]
    fn the_warm_budget_is_half_the_cache_and_holds_a_walk_s_records() {
        assert_eq!(WARM_BUDGET, crate::platform::SHADER_CACHE_CAPACITY / 2);
        const { assert!(4_565u64 * 1_853_000 <= rt::DEFAULT_CAPACITY / 2) };
    }

    /// The records file lies beside the hosts' shader cache directory, under `HOME`, not in it:
    /// the runtime's store keeps that directory to its capacity
    #[test]
    fn the_file_lies_beside_the_shader_cache() {
        let path = path_in(Path::new("/home/u"));
        assert_eq!(
            path,
            Path::new("/home/u/.cache/reims-vgpu-vulkan-pipelines")
        );
        assert_eq!(path.parent(), Some(Path::new("/home/u/.cache")));
    }

    /// Frames read back as the records written, in order; a damaged or cut frame is skipped and
    /// the next whole one read, wherever it starts
    #[test]
    fn frames_read_back_skipping_damage() {
        let (a, b, c) = (b"first".to_vec(), vec![7u8; 300], b"third".to_vec());
        let mut file = frame(&a);
        let mut damaged = frame(&b);
        damaged[FRAME_HEAD + 10] ^= 0xff;
        file.extend_from_slice(&damaged);
        file.extend_from_slice(&frame(&c));
        file.extend_from_slice(&frame(&b)[..20]);
        assert_eq!(split(&file), [a.as_slice(), c.as_slice()]);
        assert!(split(&[]).is_empty());
        assert!(split(&frame(&a)[..FRAME_HEAD]).is_empty());
    }

    /// The time [`RECORD_BOUND`] past `start`, in nanoseconds.
    fn bound_past(start: u64) -> u64 {
        start + RECORD_BOUND.as_nanos() as u64
    }

    /// The counts of a pass whose first `ended` records ended, each made.
    fn ended(ended: u64) -> rt::WarmStatistics {
        let mut s = rt::WarmStatistics::default();
        s.records_warmed = ended;
        s
    }

    /// A record whose warm making stands the bound is named once, its tombstone appended, and the
    /// next pass is handed every record but it
    #[test]
    fn an_unfinished_record_is_not_handed_to_the_next_pass() {
        let dir = Scratch::new("unfinished");
        let path = dir.0.join("records");
        let records = [b"one".as_slice(), b"two", b"three"];
        let mut keeper = Keeper::open(path.clone());
        for record in records {
            keeper.keep(record);
        }
        let mut watch = Watch::new(3, 1_000);
        assert_eq!(watch.overran(&ended(0), 2_000), None);
        assert_eq!(watch.overran(&ended(1), 3_000), None);
        assert_eq!(watch.overran(&ended(1), bound_past(3_000) - 1), None);
        assert_eq!(watch.overran(&ended(1), bound_past(3_000)), Some(1));
        assert_eq!(
            watch.overran(&ended(1), bound_past(9_000)),
            None,
            "named once"
        );
        leave_out(&path, records[1]).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(warmable(&split(&bytes)), (vec![records[0], records[2]], 1));
        // The keeper, the guest's creations' end of the file, still holds it, and keeps it once.
        let mut keeper = Keeper::open(path.clone());
        keeper.keep(records[1]);
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
    }

    /// A record whose making ends within the bound is named by no poll, and the next pass is
    /// handed it; the next record's time runs from the poll that saw its making start, and the
    /// counts' end (the records handed) names none
    #[test]
    fn a_finished_record_is_still_handed_to_the_next_pass() {
        let dir = Scratch::new("finished");
        let path = dir.0.join("records");
        let mut keeper = Keeper::open(path.clone());
        keeper.keep(b"one");
        keeper.keep(b"two");
        let mut watch = Watch::new(2, 0);
        assert_eq!(watch.overran(&ended(0), bound_past(0) - 1), None);
        assert_eq!(watch.overran(&ended(1), bound_past(0)), None);
        assert_eq!(
            watch.overran(&ended(1), bound_past(bound_past(0)) - 1),
            None
        );
        assert_eq!(watch.overran(&ended(2), bound_past(bound_past(0))), None);
        assert_eq!(
            watch.overran(&ended(2), bound_past(bound_past(bound_past(0)))),
            None
        );
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(
            warmable(&split(&bytes)),
            (vec![b"one".as_slice(), b"two"], 0)
        );
    }

    /// A file of records alone, as every earlier build wrote it, hands a pass all of them in its
    /// order; a tombstone of a record the file does not hold leaves out none; and a tombstone's
    /// bytes, which an earlier build hands the runtime, are no record to it
    #[test]
    fn the_file_s_existing_frames_still_read() {
        let (a, b) = (b"first".to_vec(), vec![7u8; 300]);
        let mut file = frame(&a);
        file.extend_from_slice(&frame(&b));
        assert_eq!(
            warmable(&split(&file)),
            (vec![a.as_slice(), b.as_slice()], 0)
        );
        let mut tombstone = UNFINISHED.to_vec();
        tombstone.extend_from_slice(b"absent");
        file.extend_from_slice(&frame(&tombstone));
        assert_eq!(split(&file).len(), 3);
        assert_eq!(
            warmable(&split(&file)),
            (vec![a.as_slice(), b.as_slice()], 0)
        );
        assert!(rt::PipelineRecord::from_bytes(&tombstone).is_none());
    }

    /// The keeper appends each record once, across reopenings of the file, and nothing past its
    /// capacity
    #[test]
    fn a_record_is_kept_once() {
        let dir = Scratch::new("once");
        let path = dir.0.join("sub").join("records");
        let mut keeper = Keeper::open(path.clone());
        keeper.keep(b"one");
        keeper.keep(b"two");
        keeper.keep(b"one");
        drop(keeper);
        let mut keeper = Keeper::open(path.clone());
        keeper.keep(b"two");
        keeper.keep(b"three");
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(
            split(&bytes),
            [b"one".as_slice(), b"two".as_slice(), b"three".as_slice()]
        );
        keeper.bytes = CAPACITY - 4;
        keeper.keep(b"four");
        assert_eq!(std::fs::read(&path).unwrap(), bytes, "past its capacity");
    }

    /// A state a device with the hook creates is recorded in the file, and a device over the same
    /// shader cache directory warms it: the pass makes it, unasked
    #[test]
    fn a_created_state_is_recorded_and_warmed() {
        let dir = Scratch::new("warm");
        let path = dir.0.join("records");
        let options = || {
            let mut options = rt::DeviceOptions::default();
            options.shader_cache = rt::ShaderCacheOptions::in_directory(dir.0.join("cache"));
            options
        };
        {
            let mut recording = options();
            recording.pipeline_records = hook(path.clone());
            let device = rt::Device::system_default_with(&recording).unwrap();
            let library = device
                .new_library_with_data(crate::gpu::spin::SPIN)
                .unwrap();
            let function = library.new_function_with_name("spin").unwrap();
            let state = device
                .new_compute_pipeline_state_with_function(
                    &function,
                    &rt::PipelineOptions::default(),
                )
                .unwrap();
            drop(state);
            assert_eq!(device.warm_statistics().records_made, 1);
        }
        // The writer appends on its own thread, within 30 s.
        for _ in 0..3000 {
            if !read(&path).0.is_empty() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(read(&path).0.len(), 1);
        let device = rt::Device::system_default_with(&options()).unwrap();
        let stop = Arc::new(crate::let_go::Stop::default());
        let _polls = warm(&device, path, &stop);
        // The pass ends within 120 s.
        let mut s = device.warm_statistics();
        for _ in 0..12000 {
            if s.records_warmed + s.records_translated + s.records_failed + s.records_unread > 0 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
            s = device.warm_statistics();
        }
        assert_eq!(s.records_warmed, 1, "{s:?}");
    }
}
