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
//! - **The file's format**: frames, each [`MAGIC`], the record's length and the FNV-1a hash of its
//!   bytes (each a little-endian `u32`), then its bytes. A reader skips a frame whose length runs
//!   past the file or whose hash does not match (a process that died mid-write, or two writing at
//!   once), and goes on at the next [`MAGIC`] after its start; a record of another format the
//!   runtime reads as none (`PipelineRecord::from_bytes`) and the pass drops.
//!
//! The unit tests' devices keep and warm no records (`device_options`): a test of what a device
//! compiles runs as it would on a host that never ran a guest.

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};

use crate::core::sync::Mutex;
use metal2vulkan_runtime as rt;

/// The records file under `HOME`, beside the shader cache's directory.
pub(crate) const RECORDS: &str = ".cache/reims-vgpu-vulkan-pipelines";

/// The most bytes the records file holds: past them, no record is appended.
pub(crate) const CAPACITY: u64 = 64 << 20;

/// The driver's pipeline cache data one warm pass may add, in bytes (`Device::warm`'s budget):
/// more than a macOS guest's login and built-in apps make, so the pass reaches every record.
pub(crate) const WARM_BUDGET: u64 = 4 << 30;

/// How often the warm thread looks whether the runtime's pass has ended.
const WARM_POLL: std::time::Duration = std::time::Duration::from_millis(250);

/// The start of each frame of the records file.
pub(crate) const MAGIC: u32 = u32::from_le_bytes(*b"RVGR");

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

/// The records of the file at `path`, in its order.
pub(crate) fn read(path: &Path) -> Vec<rt::PipelineRecord> {
    let bytes = std::fs::read(path).unwrap_or_default();
    split(&bytes)
        .into_iter()
        .filter_map(rt::PipelineRecord::from_bytes)
        .collect()
}

/// Hands `device` the records of the file at `path`, from a thread of its own that reads the
/// file, so the caller waits for nothing; the runtime's pass then runs on the device's own thread
/// (the module docs).
pub(crate) fn warm(device: &rt::Device, path: PathBuf) {
    let device = device.clone();
    let _ = std::thread::Builder::new()
        .name("reims-vgpu-warm".into())
        .spawn(move || {
            let records = read(&path);
            if records.is_empty() {
                return;
            }
            log::info!(
                target: "reims_vgpu_vulkan::records",
                "warming {} pipeline records from {}",
                records.len(),
                path.display()
            );
            let handed = records.len() as u64;
            device.warm(records, WARM_BUDGET);
            // The pass runs on the runtime's thread and says nothing when it ends: its counts
            // reach the records handed then. The device lives as long as the process
            // (`gpu::device::adopt` keeps it), so holding it here holds nothing longer.
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
                polls += 1;
                std::thread::sleep(WARM_POLL);
            }
        });
}

/// What a warm pass of `handed` records made of them, once its counts add up to them (each record
/// warmed, translated only, failed, unread, not warmable or not run, by `statistics`); `None`
/// while the pass runs.
pub(crate) fn summary(statistics: &rt::WarmStatistics, handed: u64) -> Option<String> {
    let s = statistics;
    let ended = s.records_warmed
        + s.records_translated
        + s.records_failed
        + s.records_unread
        + s.records_not_warmable
        + s.records_not_run;
    (ended >= handed).then(|| {
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
            if !read(&path).is_empty() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(read(&path).len(), 1);
        let device = rt::Device::system_default_with(&options()).unwrap();
        warm(&device, path);
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
