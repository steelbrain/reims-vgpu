//! The answers the device gave keyed pipeline creates, kept from one run of the host to the next
//! (`RGFX_FEATURE_PIPELINE_ANSWERS`), as Metal's own shader cache keeps a process's compiled
//! pipelines warm after a reboot: the kext asks for them at its start (`PIPELINE_ANSWERS`) and
//! takes them into the table it answers equal creates of any process from, so a guest's first
//! process after a boot does not wait for the creates an earlier run of the host made.
//!
//! - **What is kept.** A made pipeline's reply to a create that asked for its key and could be
//!   keyed ([`RGFX_PIPELINE_REFLECT_KEYED`]), with what the device gives each create alone
//!   (`gpu_resource_id`, `unique_identifier`, `resource_index`) and its `obj_id` and `seqno` 0, as
//!   the kext keeps an answer. A refused create is never kept, and a kept answer whose create the
//!   device refuses, or answers with other values, is dropped: the second is a degradation
//!   ([`PipelineAnswers::observe`]).
//! - **Under what.** Each file is the answers of one identity: the platform's account of what
//!   decides its answers to a pipeline create (on the Vulkan platform the runtime's translation
//!   profile, its semantics, artifact and runtime versions and the driver's version), the build of
//!   this library that runs ([`library_build`]: how its decode makes a create's descriptor, its
//!   platform's queries and its replies are that build's), and [`ANSWERS_FORMAT`], the file's
//!   layout. Answers of another identity are never read, so an upgrade of the library starts
//!   from none, as a driver's does.
//! - **The file** (`~/.cache/reims-vgpu-pipeline-answers/<identity>`) is a header of
//!   [`HEADER_BYTES`] (its magic, [`ANSWERS_FORMAT`], the record size, the identity and a
//!   checksum of them) and then records of [`RECORD_BYTES`] each (kept or dropped, the key, the
//!   answer and a checksum of them with the identity), appended in the order learnt and never
//!   rewritten in place. Any record whose checksum or kind does not read, bytes past the last whole
//!   record, or a header that is not this identity's, and the whole file reads as none and is
//!   started again: the kext takes what the device hands it, so no damaged or foreign answer
//!   reaches it. At most [`CAPACITY`] records are appended; past half of them the file is written
//!   again at its next open with the newest answers kept, a quarter of the capacity at most (the
//!   2,048 answers the kext's table holds). The newest
//!   [`KEEP_FILES`] files of the directory are kept, and the rest removed.
//! - A test build keeps no file ([`KEPT`]): its answers are kept in memory for the device's life
//!   alone, so no test's pipelines reach a host's later guests.

use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use reims_vgpu_graphics::wire::{
    RGFX_EVT_PIPELINE_REFLECTION, RGFX_PIPELINE_KIND_MESH, RGFX_PIPELINE_OK,
    RGFX_PIPELINE_REFLECT_KEYED, RGFX_PIPELINE_REFLECTION_KEY_OFFSET, RgfxPipelineReflection,
};
use reims_vgpu_vulkan::core::num::ToUsize;
use reims_vgpu_vulkan::core::sync::Mutex;
use sha2::{Digest, Sha256};
use zerocopy::{FromBytes, IntoBytes};

#[cfg(unix)]
mod unix;
#[cfg(windows)]
mod windows;

// A file's build ([`library_build`]), each OS's own.
#[cfg(unix)]
use unix::file_build;
#[cfg(windows)]
use windows::file_build;

/// A pipeline create's key ([`RGFX_PIPELINE_REFLECT_KEYED`]).
pub(crate) type AnswerKey = [u8; 32];

/// The file's layout, which the identity of every file names: bump it whenever the layout
/// changes. What the device answers a create is the platform's identity's and the library
/// build's ([`library_build`]), which no bump need follow.
pub(crate) const ANSWERS_FORMAT: u32 = 1;

/// Whether this build keeps the answers in a file: all but a test build's (the module docs).
pub(crate) const KEPT: bool = !cfg!(any(test, feature = "testkit"));

/// The directory of the files, under `HOME`.
const DIR: &str = ".cache/reims-vgpu-pipeline-answers";

/// The most records a file holds: past them, no record is appended (about 1 MiB).
pub(crate) const CAPACITY: usize = 8192;

/// How many files of the directory are kept, the newest: one for each identity a host ran lately.
const KEEP_FILES: usize = 8;

/// The bytes a file starts with.
const MAGIC: [u8; 16] = *b"rgfx-answers-v1\0";

/// The header's bytes: [`MAGIC`], [`ANSWERS_FORMAT`] and [`RECORD_BYTES`] (each a little-endian
/// `u32`), the identity, and the FNV-1a hash of those 56 bytes (a little-endian `u64`).
pub(crate) const HEADER_BYTES: usize = 64;

/// A record's bytes: its kind ([`KEPT_RECORD`] or [`DROPPED_RECORD`]) and 4 bytes of 0 (each a
/// little-endian `u32`), the key, the answer ([`RgfxPipelineReflection`]), and the FNV-1a hash of
/// the identity and those 120 bytes (a little-endian `u64`).
pub(crate) const RECORD_BYTES: usize = 128;

/// A record of an answer kept.
const KEPT_RECORD: u32 = 1;

/// A record of the answer of a key dropped.
const DROPPED_RECORD: u32 = 2;

/// Where a record's answer lies.
const ANSWER_AT: usize = 40;

/// The bytes of a keyed reply: the answer, then its key.
pub(crate) const KEYED_REPLY_BYTES: usize = RGFX_PIPELINE_REFLECTION_KEY_OFFSET as usize + 32;

/// The device's kept answers: those of earlier runs read at its start, and those it learns.
pub(crate) struct PipelineAnswers {
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    /// The identity, where answers are kept in a file.
    identity: Option<[u8; 32]>,
    /// The answers, each with when it was learnt (the newer the larger).
    kept: HashMap<AnswerKey, (RgfxPipelineReflection, u64)>,
    /// The keys whose answers were dropped for answering otherwise, which this run learns no more.
    unsteady: HashSet<AnswerKey>,
    /// The next answer's stamp.
    next: u64,
    /// The file, where it is written, and the records it holds.
    file: Option<File>,
    records: usize,
}

/// How a create of a key the device keeps an answer of was answered now ([`PipelineAnswers::observe`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Observed {
    /// None is kept: the answer is now, where it was made.
    New,
    /// As the kept answer.
    Same,
    /// Made with other values than the kept answer, which is dropped.
    Changed,
    /// Refused, and the kept answer dropped.
    Refused,
    /// Neither kept nor made.
    Nothing,
}

impl PipelineAnswers {
    /// The answers of `identity`, the platform's, of this library's build ([`answers_identity`],
    /// [`library_build`]): read from its file under `HOME` where this build keeps them
    /// ([`KEPT`]) and the platform and the build each give one; else kept in memory alone.
    pub(crate) fn open(identity: Option<&[u8]>) -> PipelineAnswers {
        let identity = identity
            .zip(library_build())
            .map(|(platform, build)| answers_identity(platform, &build));
        let dir = std::env::var_os("HOME").map(|home| PathBuf::from(home).join(DIR));
        match (identity, dir) {
            (Some(identity), Some(dir)) if KEPT => PipelineAnswers::open_in(&dir, identity),
            _ => PipelineAnswers::in_memory(),
        }
    }

    /// Answers kept in memory alone.
    pub(crate) fn in_memory() -> PipelineAnswers {
        PipelineAnswers {
            state: Mutex::new(State::default()),
        }
    }

    /// The answers of `identity`, whose file lies in `dir`.
    pub(crate) fn open_in(dir: &Path, identity: [u8; 32]) -> PipelineAnswers {
        let path = dir.join(hex(&identity));
        let mut state = State {
            identity: Some(identity),
            ..State::default()
        };
        let read = std::fs::read(&path)
            .ok()
            .and_then(|bytes| read(&bytes, &identity));
        let unread = read.is_none();
        let records = read.as_ref().map_or(0, |records| records.len());
        for (kind, key, answer) in read.into_iter().flatten() {
            state.next += 1;
            if kind == KEPT_RECORD {
                state.kept.insert(key, (answer, state.next));
            } else {
                state.kept.remove(&key);
            }
        }
        // A missing, damaged or foreign file is started again, and one past half its capacity
        // written again with its newest answers: as many as the kext's table holds.
        let rewrite = unread || records > CAPACITY / 2;
        state.file = if rewrite {
            state.records = 0;
            let newest = newest(&state.kept, CAPACITY / 4);
            let keys: HashSet<AnswerKey> = newest.iter().map(|(key, _)| *key).collect();
            state.kept.retain(|key, _| keys.contains(key));
            rewrite_file(dir, &path, &identity, &newest).inspect(|_| state.records = newest.len())
        } else {
            state.records = records;
            OpenOptions::new().append(true).open(&path).ok()
        };
        prune(dir, &path);
        log::info!(
            target: "reims_vgpu::graphics::answers",
            "kept pipeline answers: {} of earlier runs read from {}{}",
            state.kept.len(),
            path.display(),
            if state.file.is_none() { " (none is kept in it this run)" } else { "" }
        );
        PipelineAnswers {
            state: Mutex::new(state),
        }
    }

    /// Takes the device's answer to a create of `key` now: `made`, its reply's record, or `None`
    /// where it was refused. A refusal drops the kept answer. A made answer is kept where none is;
    /// one whose values differ from the kept answer drops it (the platform answers otherwise
    /// under one identity: a degradation, its line once per boot), and the key is kept no more in
    /// this run.
    pub(crate) fn observe(
        &self,
        key: &AnswerKey,
        made: Option<&RgfxPipelineReflection>,
    ) -> Observed {
        let mut state = self.state.lock();
        let answer = made.map(kept_form);
        let kept = state.kept.get(key).map(|(kept, _)| *kept);
        match (answer, kept) {
            (None, Some(_)) => {
                state.drop_answer(key);
                Observed::Refused
            }
            (None, None) => Observed::Nothing,
            (Some(answer), Some(kept)) if answer == kept => Observed::Same,
            (Some(_), Some(_)) => {
                state.drop_answer(key);
                state.unsteady.insert(*key);
                log::warn!(
                    target: "reims_vgpu::graphics::answers",
                    "a pipeline create of a key the device kept an answer of was answered with \
                     other values: the kept answer is dropped (the platform's answers changed \
                     under one identity)"
                );
                Observed::Changed
            }
            (Some(answer), None) => {
                if state.unsteady.contains(key) {
                    return Observed::Nothing;
                }
                state.next += 1;
                let stamp = state.next;
                state.kept.insert(*key, (answer, stamp));
                state.append(KEPT_RECORD, key, &answer);
                Observed::New
            }
        }
    }

    /// The keyed replies of the kept answers, newest first, from the `first`th, at most `most` of
    /// them, each stamped with `seqno` ([`KEYED_REPLY_BYTES`] each).
    pub(crate) fn replies(&self, first: u32, most: u32, seqno: u64) -> Vec<Vec<u8>> {
        let state = self.state.lock();
        let mut all: Vec<(&AnswerKey, &(RgfxPipelineReflection, u64))> =
            state.kept.iter().collect();
        all.sort_by_key(|(_, (_, stamp))| Reverse(*stamp));
        all.into_iter()
            .skip(first.to_usize())
            .take(most.to_usize())
            .map(|(key, (answer, _))| {
                let mut reply = *answer;
                reply.header.seqno = seqno;
                let mut bytes = reply.as_bytes().to_vec();
                bytes.extend_from_slice(key);
                bytes
            })
            .collect()
    }

    /// How many answers are kept.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.state.lock().kept.len()
    }
}

impl State {
    /// Drops the answer of `key`, in memory and in the file.
    fn drop_answer(&mut self, key: &AnswerKey) {
        if let Some((answer, _)) = self.kept.remove(key) {
            self.append(DROPPED_RECORD, key, &answer);
        }
    }

    /// Appends a record of `kind` to the file, where one is written and has room.
    fn append(&mut self, kind: u32, key: &AnswerKey, answer: &RgfxPipelineReflection) {
        let (Some(identity), Some(file)) = (self.identity, self.file.as_mut()) else {
            return;
        };
        if self.records >= CAPACITY {
            return;
        }
        if file
            .write_all(&record(&identity, kind, key, answer))
            .is_ok()
        {
            self.records += 1;
        } else {
            log::warn!(
                target: "reims_vgpu::graphics::answers",
                "the pipeline answers' file took no more records: this run's answers are kept \
                 in memory alone"
            );
            self.file = None;
        }
    }
}

/// The identity of a file: SHA-256 of the crate's own words, [`ANSWERS_FORMAT`], the platform's
/// identity bytes and the library's build ([`library_build`]), each of the two after its length.
pub(crate) fn answers_identity(platform: &[u8], build: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"rgfx-pipeline-answers");
    hasher.update(ANSWERS_FORMAT.to_le_bytes());
    for part in [platform, build] {
        hasher.update(u64::try_from(part.len()).unwrap_or(u64::MAX).to_le_bytes());
        hasher.update(part);
    }
    hasher.finalize().into()
}

/// The build of this library that runs: the file its code runs from, the host's executable, which
/// links the library statically (QEMU's `reims-graphics-pci`), as [`file_build`] tells it. A new
/// build of the library, or of its host, is another identity's, so no answer a build gave reaches
/// a guest of another, whatever its decode, platform or replies changed. `None` where the file
/// cannot be told: then nothing is kept past the device's life.
pub(crate) fn library_build() -> Option<Vec<u8>> {
    file_build(&std::env::current_exe().ok()?)
}

/// `reflection` as it is kept: a keyed reply's header (its size counting the key, its seqno 0),
/// and 0 where the device gives each create its own.
fn kept_form(reflection: &RgfxPipelineReflection) -> RgfxPipelineReflection {
    let mut kept = *reflection;
    kept.header.r#type = RGFX_EVT_PIPELINE_REFLECTION;
    kept.header.size = KEYED_REPLY_BYTES as u32;
    kept.header.seqno = 0;
    kept.obj_id = 0;
    kept.gpu_resource_id = 0;
    kept.unique_identifier = 0;
    kept.resource_index = 0;
    kept.flags |= RGFX_PIPELINE_REFLECT_KEYED;
    kept
}

/// Whether `answer` is one a file may hold: a made pipeline's keyed answer, as [`kept_form`] gives
/// it, of no mesh pipeline.
fn keepable(answer: &RgfxPipelineReflection) -> bool {
    answer.header.r#type == RGFX_EVT_PIPELINE_REFLECTION
        && answer.header.size as usize == KEYED_REPLY_BYTES
        && answer.header.seqno == 0
        && answer.status == RGFX_PIPELINE_OK
        && answer.kind != RGFX_PIPELINE_KIND_MESH
        && answer.flags & RGFX_PIPELINE_REFLECT_KEYED != 0
        && answer.obj_id == 0
        && answer.gpu_resource_id == 0
        && answer.unique_identifier == 0
        && answer.resource_index == 0
}

/// The header of a file of `identity`.
fn header(identity: &[u8; 32]) -> [u8; HEADER_BYTES] {
    let mut out = [0; HEADER_BYTES];
    out[..16].copy_from_slice(&MAGIC);
    out[16..20].copy_from_slice(&ANSWERS_FORMAT.to_le_bytes());
    out[20..24].copy_from_slice(&(RECORD_BYTES as u32).to_le_bytes());
    out[24..56].copy_from_slice(identity);
    let sum = fnv1a(&[&out[..56]]);
    out[56..].copy_from_slice(&sum.to_le_bytes());
    out
}

/// A record of `kind` of `key`'s `answer`, in a file of `identity`.
fn record(
    identity: &[u8; 32],
    kind: u32,
    key: &AnswerKey,
    answer: &RgfxPipelineReflection,
) -> [u8; RECORD_BYTES] {
    let mut out = [0; RECORD_BYTES];
    out[..4].copy_from_slice(&kind.to_le_bytes());
    out[8..40].copy_from_slice(key);
    out[ANSWER_AT..ANSWER_AT + size_of::<RgfxPipelineReflection>()]
        .copy_from_slice(answer.as_bytes());
    let sum = fnv1a(&[identity, &out[..120]]);
    out[120..].copy_from_slice(&sum.to_le_bytes());
    out
}

/// The records of the file `bytes`, of `identity`, in their order: `None` where any part does
/// not read (the module docs).
fn read(
    bytes: &[u8],
    identity: &[u8; 32],
) -> Option<Vec<(u32, AnswerKey, RgfxPipelineReflection)>> {
    let (head, body) = bytes.split_at_checked(HEADER_BYTES)?;
    if head != header(identity) || !body.len().is_multiple_of(RECORD_BYTES) {
        return None;
    }
    body.as_chunks::<RECORD_BYTES>()
        .0
        .iter()
        .map(|r| {
            let kind = u32::from_le_bytes(r[..4].try_into().ok()?);
            let sum = u64::from_le_bytes(r[120..].try_into().ok()?);
            let key: AnswerKey = r[8..40].try_into().ok()?;
            let (answer, _) = RgfxPipelineReflection::read_from_prefix(&r[ANSWER_AT..]).ok()?;
            let well_formed = (kind == KEPT_RECORD || kind == DROPPED_RECORD)
                && r[4..8] == [0; 4]
                && r[ANSWER_AT + size_of::<RgfxPipelineReflection>()..120]
                    .iter()
                    .all(|&b| b == 0)
                && sum == fnv1a(&[identity, &r[..120]])
                && keepable(&answer);
            well_formed.then_some((kind, key, answer))
        })
        .collect()
}

/// The `most` newest of `kept`, oldest first.
fn newest(
    kept: &HashMap<AnswerKey, (RgfxPipelineReflection, u64)>,
    most: usize,
) -> Vec<(AnswerKey, RgfxPipelineReflection)> {
    let mut all: Vec<_> = kept.iter().map(|(k, (a, s))| (*s, *k, *a)).collect();
    all.sort_by_key(|(stamp, _, _)| Reverse(*stamp));
    all.truncate(most);
    all.reverse();
    all.into_iter().map(|(_, k, a)| (k, a)).collect()
}

/// Writes the file at `path` again, in `dir`: its header and a record of each of `answers`, through
/// a file beside it renamed over it; the file opened to append, or `None` where it could not be.
fn rewrite_file(
    dir: &Path,
    path: &Path,
    identity: &[u8; 32],
    answers: &[(AnswerKey, RgfxPipelineReflection)],
) -> Option<File> {
    std::fs::create_dir_all(dir).ok()?;
    let mut bytes = header(identity).to_vec();
    for (key, answer) in answers {
        bytes.extend_from_slice(&record(identity, KEPT_RECORD, key, answer));
    }
    let staged = path.with_extension(format!("{}.new", std::process::id()));
    std::fs::write(&staged, &bytes).ok()?;
    if std::fs::rename(&staged, path).is_err() {
        let _ = std::fs::remove_file(&staged);
        return None;
    }
    OpenOptions::new().append(true).open(path).ok()
}

/// Removes the files of `dir` but the [`KEEP_FILES`] last written and `keep`.
fn prune(dir: &Path, keep: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut files: Vec<(std::time::SystemTime, PathBuf)> = entries
        .filter_map(Result::ok)
        .filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.path())))
        .filter(|(_, path)| path != keep && path.is_file())
        .collect();
    files.sort_by_key(|(modified, _)| Reverse(*modified));
    for (_, path) in files.into_iter().skip(KEEP_FILES.saturating_sub(1)) {
        let _ = std::fs::remove_file(path);
    }
}

/// `bytes` in lowercase hex.
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The 64-bit FNV-1a hash of `parts`, one after another.
fn fnv1a(parts: &[&[u8]]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for part in parts {
        for &b in *part {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_graphics::wire::{RGFX_PIPELINE_KIND_COMPUTE, RgfxMsgHeader};

    /// A unique scratch directory under the system's temporary directory, removed at the end.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str) -> Scratch {
            let dir = std::env::temp_dir().join(format!(
                "rvg-answers-{name}-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Scratch(dir)
        }

        fn file(&self, identity: &[u8; 32]) -> PathBuf {
            self.0.join(hex(identity))
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    const IDENTITY: [u8; 32] = [7; 32];

    /// A made compute pipeline's reply to create `seqno` of object `obj_id`, of `threads` threads.
    fn made(seqno: u64, obj_id: u64, threads: u64) -> RgfxPipelineReflection {
        RgfxPipelineReflection {
            header: RgfxMsgHeader {
                r#type: RGFX_EVT_PIPELINE_REFLECTION,
                size: size_of::<RgfxPipelineReflection>() as u32,
                seqno,
            },
            obj_id,
            gpu_resource_id: obj_id + 100,
            unique_identifier: obj_id + 200,
            max_total_threads_per_threadgroup: threads,
            static_threadgroup_memory_length: 64,
            thread_execution_width: 32,
            imageblock_sample_length: 0,
            resource_index: 3,
            status: RGFX_PIPELINE_OK,
            kind: RGFX_PIPELINE_KIND_COMPUTE,
            flags: 0,
        }
    }

    fn key(n: u8) -> AnswerKey {
        [n; 32]
    }

    /// A kept answer is read back by the next run of the same identity, as the kext takes it: a
    /// keyed reply with the create's own values 0 and the asking message's seqno.
    #[test]
    fn a_kept_answer_reaches_the_next_run_as_a_keyed_reply() {
        let dir = Scratch::new("next-run");
        let answers = PipelineAnswers::open_in(&dir.0, IDENTITY);
        assert_eq!(
            answers.observe(&key(1), Some(&made(5, 0x1_0000_0001, 1024))),
            Observed::New
        );
        drop(answers);
        let answers = PipelineAnswers::open_in(&dir.0, IDENTITY);
        let replies = answers.replies(0, 16, 77);
        assert_eq!(replies.len(), 1);
        assert_eq!(replies[0].len(), KEYED_REPLY_BYTES);
        let (reply, key_bytes) = RgfxPipelineReflection::read_from_prefix(&replies[0]).unwrap();
        assert_eq!(key_bytes, key(1));
        assert_eq!(reply.header.seqno, 77);
        assert_eq!(reply.header.size as usize, KEYED_REPLY_BYTES);
        assert_eq!(
            (
                reply.obj_id,
                reply.gpu_resource_id,
                reply.unique_identifier,
                reply.resource_index
            ),
            (0, 0, 0, 0)
        );
        assert_eq!(reply.max_total_threads_per_threadgroup, 1024);
        assert_eq!(reply.static_threadgroup_memory_length, 64);
        assert_ne!(reply.flags & RGFX_PIPELINE_REFLECT_KEYED, 0);
    }

    /// The answers go newest first, from the asked one, at most as many as asked.
    #[test]
    fn replies_go_newest_first_from_the_asked_one() {
        let answers = PipelineAnswers::in_memory();
        for n in 1..=5u8 {
            answers.observe(&key(n), Some(&made(1, 1, u64::from(n))));
        }
        let threads = |replies: Vec<Vec<u8>>| -> Vec<u64> {
            replies
                .iter()
                .map(|r| {
                    RgfxPipelineReflection::read_from_prefix(r)
                        .unwrap()
                        .0
                        .max_total_threads_per_threadgroup
                })
                .collect()
        };
        assert_eq!(threads(answers.replies(0, 2, 1)), [5, 4]);
        assert_eq!(threads(answers.replies(2, 2, 1)), [3, 2]);
        assert_eq!(threads(answers.replies(4, 2, 1)), [1]);
        assert!(answers.replies(5, 2, 1).is_empty());
    }

    /// A refused create is never kept, and drops the kept answer of its key, in the file too.
    #[test]
    fn a_refusal_is_never_kept_and_drops_the_kept_answer() {
        let dir = Scratch::new("refusal");
        let answers = PipelineAnswers::open_in(&dir.0, IDENTITY);
        assert_eq!(answers.observe(&key(1), None), Observed::Nothing);
        assert_eq!(answers.len(), 0);
        answers.observe(&key(2), Some(&made(1, 1, 256)));
        assert_eq!(answers.observe(&key(2), None), Observed::Refused);
        assert_eq!(answers.len(), 0);
        drop(answers);
        assert_eq!(PipelineAnswers::open_in(&dir.0, IDENTITY).len(), 0);
    }

    /// An answer made with other values than the kept one drops it, in the file too, and its key
    /// is kept no more in the run; one made as the kept one changes nothing.
    #[test]
    fn an_answer_made_otherwise_drops_the_kept_one() {
        let dir = Scratch::new("changed");
        let answers = PipelineAnswers::open_in(&dir.0, IDENTITY);
        answers.observe(&key(1), Some(&made(1, 1, 1024)));
        drop(answers);
        let answers = PipelineAnswers::open_in(&dir.0, IDENTITY);
        // Another create of the key: its own seqno, object and resource id change nothing.
        assert_eq!(
            answers.observe(&key(1), Some(&made(9, 0x2_0000_0005, 1024))),
            Observed::Same
        );
        assert_eq!(
            answers.observe(&key(1), Some(&made(10, 0x2_0000_0006, 512))),
            Observed::Changed
        );
        assert_eq!(answers.len(), 0);
        assert_eq!(
            answers.observe(&key(1), Some(&made(11, 0x2_0000_0007, 512))),
            Observed::Nothing
        );
        assert_eq!(answers.len(), 0);
        drop(answers);
        assert_eq!(PipelineAnswers::open_in(&dir.0, IDENTITY).len(), 0);
    }

    /// Another identity's file is never read: the answers of one platform's runtime, profile or
    /// driver reach no other's.
    #[test]
    fn another_identitys_answers_are_never_read() {
        let dir = Scratch::new("identity");
        let answers = PipelineAnswers::open_in(&dir.0, IDENTITY);
        answers.observe(&key(1), Some(&made(1, 1, 1024)));
        drop(answers);
        assert_eq!(PipelineAnswers::open_in(&dir.0, [8; 32]).len(), 0);
        // A file of another identity copied over this one's reads as none.
        std::fs::copy(dir.file(&IDENTITY), dir.file(&[9; 32])).unwrap();
        assert_eq!(PipelineAnswers::open_in(&dir.0, [9; 32]).len(), 0);
        assert_eq!(PipelineAnswers::open_in(&dir.0, IDENTITY).len(), 1);
    }

    /// The identity names the platform's bytes and the library's build each apart: another of
    /// either is another identity, so another build's answers are never read. The build that runs
    /// is told, and the same at each ask.
    #[test]
    fn another_build_or_platform_is_another_identity() {
        let identity = answers_identity(b"platform", b"build 1");
        assert_ne!(identity, answers_identity(b"platform", b"build 2"));
        assert_ne!(identity, answers_identity(b"platform 2", b"build 1"));
        assert_ne!(
            answers_identity(b"ab", b"c"),
            answers_identity(b"a", b"bc"),
            "the parts' bounds are named"
        );
        let build = library_build().expect("the test binary's build is told");
        assert_eq!(library_build(), Some(build.clone()));
        let exe = std::fs::canonicalize(std::env::current_exe().unwrap()).unwrap();
        assert!(build.starts_with(exe.as_os_str().as_encoded_bytes()));
        assert_eq!(build.len(), exe.as_os_str().len() + 3 * 8 + 4 * 8);
    }

    /// Another file of the same size at the same path, renamed over it as a link or an install
    /// writes one, is another build; the file untouched, or named through a link, is the same.
    #[test]
    fn a_file_renamed_over_is_another_build() {
        let dir = Scratch::new("build");
        let path = dir.0.join("qemu");
        std::fs::write(&path, [1; 64]).unwrap();
        let first = file_build(&path).unwrap();
        assert_eq!(
            file_build(&path),
            Some(first.clone()),
            "untouched, the same"
        );
        let link = dir.0.join("current");
        std::os::unix::fs::symlink(&path, &link).unwrap();
        assert_eq!(
            file_build(&link),
            Some(first.clone()),
            "through a link, the file's"
        );
        let next = dir.0.join("qemu.new");
        std::fs::write(&next, [2; 64]).unwrap();
        std::fs::rename(&next, &path).unwrap();
        assert_ne!(
            file_build(&path),
            Some(first),
            "a new file of the same size"
        );
        assert_eq!(file_build(&dir.0.join("none")), None);
    }

    /// Any damage reads the whole file as none, and it is started again: a flipped byte of a
    /// record, of the header, or bytes past the last whole record.
    #[test]
    fn a_damaged_file_reads_as_none() {
        for damage in [HEADER_BYTES + 50, 20, usize::MAX] {
            let dir = Scratch::new("damaged");
            let answers = PipelineAnswers::open_in(&dir.0, IDENTITY);
            answers.observe(&key(1), Some(&made(1, 1, 1024)));
            answers.observe(&key(2), Some(&made(1, 1, 512)));
            drop(answers);
            let path = dir.file(&IDENTITY);
            let mut bytes = std::fs::read(&path).unwrap();
            if damage == usize::MAX {
                bytes.extend_from_slice(&[1, 2, 3]);
            } else {
                bytes[damage] ^= 0x40;
            }
            std::fs::write(&path, &bytes).unwrap();
            let answers = PipelineAnswers::open_in(&dir.0, IDENTITY);
            assert_eq!(answers.len(), 0, "damage at {damage}");
            // Started again: what it learns now is read by the next run.
            answers.observe(&key(3), Some(&made(1, 1, 64)));
            drop(answers);
            assert_eq!(
                PipelineAnswers::open_in(&dir.0, IDENTITY).len(),
                1,
                "damage at {damage}"
            );
        }
    }

    /// A record that reads but holds no answer a file may hold (a refused one, one with its
    /// create's own values) reads the file as none, whatever its checksum.
    #[test]
    fn a_record_of_no_keepable_answer_reads_the_file_as_none() {
        let dir = Scratch::new("unkeepable");
        let mut refused = kept_form(&made(1, 1, 64));
        refused.status = 1;
        let mut own = kept_form(&made(1, 1, 64));
        own.gpu_resource_id = 5;
        for answer in [refused, own] {
            let mut bytes = header(&IDENTITY).to_vec();
            bytes.extend_from_slice(&record(&IDENTITY, KEPT_RECORD, &key(1), &answer));
            std::fs::write(dir.file(&IDENTITY), &bytes).unwrap();
            assert_eq!(PipelineAnswers::open_in(&dir.0, IDENTITY).len(), 0);
        }
    }

    /// A file holds at most its capacity of records; one past half of it is written again at its
    /// next open with its newest answers, a quarter of the capacity.
    #[test]
    fn the_file_is_capped_and_written_again_with_its_newest_answers() {
        let dir = Scratch::new("cap");
        let answers = PipelineAnswers::open_in(&dir.0, IDENTITY);
        for n in 0..CAPACITY + 10 {
            let mut k = [0u8; 32];
            k[..8].copy_from_slice(&(n as u64).to_le_bytes());
            answers.observe(&k, Some(&made(1, 1, n as u64)));
        }
        drop(answers);
        let len = std::fs::metadata(dir.file(&IDENTITY)).unwrap().len() as usize;
        assert_eq!(len, HEADER_BYTES + CAPACITY * RECORD_BYTES);
        let answers = PipelineAnswers::open_in(&dir.0, IDENTITY);
        assert_eq!(answers.len(), CAPACITY / 4);
        let newest = answers.replies(0, 1, 0);
        let (reply, _) = RgfxPipelineReflection::read_from_prefix(&newest[0]).unwrap();
        assert_eq!(
            reply.max_total_threads_per_threadgroup,
            (CAPACITY - 1) as u64
        );
        drop(answers);
        let len = std::fs::metadata(dir.file(&IDENTITY)).unwrap().len() as usize;
        assert_eq!(len, HEADER_BYTES + CAPACITY / 4 * RECORD_BYTES);
    }

    /// Answers kept in memory alone, as a test build's device keeps them, write no file.
    #[test]
    fn answers_in_memory_write_no_file() {
        let answers = PipelineAnswers::in_memory();
        assert_eq!(
            answers.observe(&key(1), Some(&made(1, 1, 64))),
            Observed::New
        );
        assert_eq!(answers.replies(0, 4, 1).len(), 1);
        const { assert!(!KEPT) };
    }
}
