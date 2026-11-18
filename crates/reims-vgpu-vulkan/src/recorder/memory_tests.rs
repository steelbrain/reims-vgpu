//! The memory a recording holds, on the Vulkan platform: what reaches the stream as `m2v.write`
//! records around commits ([`super::before_commit`]). The recorder is the process's, started
//! once, so each test records in a child copy of the test binary of its own
//! ([`in_recording_child`]).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use super::format::{self, Record, Value};
use super::{ENV_VAR, id_at, on, start_from_env, state};
use crate::core::host::HostPtr;
use crate::core::platform::Handle as _;
use crate::core::platform::blit::{BlitCmd, BlitEncoder as _};
use crate::core::platform::command::{
    CommandBuffer as _, CommandQueue as _, CompletedHandler, PartCut,
};
use crate::core::platform::device::GpuDevice as _;
use crate::core::platform::resource::{Buffer as _, Heap as _};
use crate::core::platform::types::NsRange;
use crate::core::raise::OnFault;
use crate::gpu::command::{VulkanCommandBuffer, VulkanCommandQueue};
use crate::gpu::resource::{VulkanBuffer, VulkanHeap};
use crate::platform::VulkanPlatform;

/// The marker that makes a run of a recording test its child, which records: the test's name.
const RECORD_CHILD: &str = "REIMS_VGPU_RECORDER_CHILD";

/// The bytes of the block the tests' buffer fills.
const BLOCK: u64 = 0x1_0000;

/// Runs `body` in a child copy of this test binary that records into a scratch directory of its
/// own, `body` given the directory its stream goes into; in the child, runs it.
fn in_recording_child(test: &str, body: impl FnOnce(&Path)) {
    if std::env::var(RECORD_CHILD).as_deref() == Ok(test) {
        let streams = PathBuf::from(std::env::var_os(ENV_VAR).expect("the child records"));
        start_from_env();
        assert!(on(), "the child does not record");
        body(&streams);
        return;
    }
    let name = test.rsplit("::").next().unwrap_or(test);
    let dir = std::env::temp_dir().join(format!("rvg-recorder-{}-{name}", std::process::id()));
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test, "--nocapture", "--test-threads=1"])
        .env(RECORD_CHILD, test)
        .env(ENV_VAR, dir.join("streams"))
        .output()
        .unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(out.status.success(), "{out:?}");
    let ran = String::from_utf8_lossy(&out.stdout).contains("1 passed");
    assert!(ran, "the child ran no test: {out:?}");
}

/// The records of the stream in `streams` once `done` holds of them, read again every 50 ms for
/// at most 10 s; what it last read where it never does.
fn records_once(streams: &Path, done: impl Fn(&[Record]) -> bool) -> Vec<Record> {
    let segment = streams.join(format!("{}.0000.m2vr", std::process::id()));
    let mut records = Vec::new();
    for _ in 0..200 {
        let read = std::fs::read(&segment)
            .ok()
            .and_then(|b| format::read(&b, &mut Vec::new()).ok());
        if let Some(read) = read {
            records = read.records;
            if done(&records) {
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    records
}

/// The place in `records` of the `n`th call record of `selector`, from 0.
fn place(records: &[Record], selector: &str, n: usize) -> Option<usize> {
    let calls = records.iter().enumerate();
    calls
        .filter(|(_, r)| matches!(r, Record::Call(c) if c.selector == selector))
        .nth(n)
        .map(|(i, _)| i)
}

/// The bytes the `m2v.write` records of `target` among `records[range]` write, by the byte they
/// land at, a blob's read from its record.
fn writes_of(
    records: &[Record],
    target: u64,
    range: std::ops::Range<usize>,
) -> BTreeMap<u64, Vec<u8>> {
    let blob = |hash: &format::Hash| {
        records.iter().find_map(|r| match r {
            Record::Blob(h, data) if h == hash => Some(data.clone()),
            _ => None,
        })
    };
    let mut out = BTreeMap::new();
    for record in records.get(range).unwrap_or_default() {
        let Record::Call(c) = record else { continue };
        if c.selector != "m2v.write" || c.receiver != target {
            continue;
        }
        for pair in c.args.chunks(2) {
            let [Value::Uint(at), data] = pair else {
                panic!("a write's arguments: {:?}", c.args);
            };
            let bytes = match data {
                Value::Bytes(b) => b.clone(),
                Value::Blob(h) => blob(h).expect("a written blob's record"),
                other => panic!("a write's bytes: {other:?}"),
            };
            out.insert(*at, bytes);
        }
    }
    out
}

/// Whether the recorder holds `cb` committed: its encoding over, among the command buffers not
/// yet seen complete.
fn held_committed(cb: &VulkanCommandBuffer) -> bool {
    let state = state().expect("the child records");
    state.drain(true);
    let id = id_at(cb.id().0).expect("the command buffer is recorded");
    let sync = state.sync.lock();
    !sync.cbs.contains_key(&id) && sync.pending.iter().any(|p| p.id == id)
}

/// A recording test's objects: a queue; a buffer of the [`BLOCK`]'s bytes at the start of a
/// `Shared` heap over host memory, a block, whose bytes the stream writes as the heap's; and a
/// `Shared` buffer of 4 KiB.
struct Rig {
    q: VulkanCommandQueue,
    heap: VulkanHeap,
    y: VulkanBuffer,
    z: VulkanBuffer,
}

impl Rig {
    fn new() -> Self {
        use crate::gpu::resource::tests::{guest_pages_of, heap_over, shared};
        let d = crate::gpu::device::tests::dev();
        let heap = heap_over::<VulkanPlatform>(&d, guest_pages_of(BLOCK), BLOCK);
        let y = heap.new_buffer_of_length(BLOCK, 0, 0).unwrap().unwrap();
        Self {
            q: d.new_command_queue().unwrap(),
            heap,
            y,
            z: shared(4096),
        }
    }

    /// The block's stream id, the heap's.
    fn block(&self) -> u64 {
        id_at(self.heap.id().0).expect("the heap is recorded")
    }

    /// A command buffer of the queue.
    fn command_buffer(&self) -> VulkanCommandBuffer {
        self.q.command_buffer(OnFault::Return).unwrap().unwrap()
    }

    /// Encodes `cmds` in a blit encoder of `cb` of their own.
    fn blit(&self, cb: &VulkanCommandBuffer, cmds: Vec<BlitCmd<'_, VulkanPlatform>>) {
        let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
        for cmd in cmds {
            e.encode(OnFault::Return, cmd).unwrap();
        }
        e.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
    }

    /// Encodes in `cb` a copy of the buffer's first 16 bytes into the `Shared` one: a command
    /// buffer that reaches the block.
    fn read_back(&self, cb: &VulkanCommandBuffer) {
        let copy = BlitCmd::CopyBufferToBuffer {
            source_buffer: Some(&self.y),
            source_offset: 0,
            destination_buffer: Some(&self.z),
            destination_offset: 0,
            size: 16,
        };
        self.blit(cb, vec![copy]);
    }

    /// Encodes in `cb` a fill of `bytes` of the buffer with `value`.
    fn fill(&self, cb: &VulkanCommandBuffer, bytes: std::ops::Range<u64>, value: u8) {
        let range = NsRange {
            location: bytes.start,
            length: bytes.end - bytes.start,
        };
        let fill = BlitCmd::FillBuffer {
            buffer: &self.y,
            range,
            value,
        };
        self.blit(cb, vec![fill]);
    }

    /// Encodes in `cb` a copy of `source`'s first bytes into `bytes` of the buffer.
    fn copy_in(
        &self,
        cb: &VulkanCommandBuffer,
        source: &VulkanBuffer,
        bytes: std::ops::Range<u64>,
    ) {
        let copy = BlitCmd::CopyBufferToBuffer {
            source_buffer: Some(source),
            source_offset: 0,
            destination_buffer: Some(&self.y),
            destination_offset: bytes.start,
            size: bytes.end - bytes.start,
        };
        self.blit(cb, vec![copy]);
    }

    /// Writes `value` into `bytes` of the buffer, as the CPU.
    fn write(&self, bytes: std::ops::Range<u64>, value: u8) {
        write_at(self.y.contents().as_ptr::<u8>().addr(), bytes, value);
    }
}

/// Writes `value` into `bytes` from the host address `base`, as the CPU: the tests' buffer's.
fn write_at(base: usize, bytes: std::ops::Range<u64>, value: u8) {
    assert!(bytes.end <= BLOCK);
    // SAFETY: `bytes` lies in the tests' buffer, `Shared`, whose memory the test's pages hold.
    unsafe {
        let at = HostPtr::from_addr(base + bytes.start as usize).as_ptr::<u8>();
        std::ptr::write_bytes(at, value, (bytes.end - bytes.start) as usize);
    }
}

/// `n` bytes of `value`.
fn bytes(value: u8, n: usize) -> Vec<u8> {
    vec![value; n]
}

/// A chunk's bytes: each run of `value`, `n` long, in order, then zeros to 4 KiB.
fn chunk(runs: &[(u8, usize)]) -> Vec<u8> {
    let mut out: Vec<u8> = runs.iter().flat_map(|&(v, n)| bytes(v, n)).collect();
    out.resize(4096, 0);
    out
}

/// Commits `cb` and waits for it.
fn run(cb: &VulkanCommandBuffer) {
    cb.commit(OnFault::Return).unwrap();
    cb.wait_until_completed(OnFault::Return).unwrap();
}

/// The block's writes between each two `commit` records of the stream, once it holds `n` of
/// them: the `i`th map those after commit `i`, up to the next.
fn writes_between_commits(streams: &Path, block: u64, n: usize) -> Vec<BTreeMap<u64, Vec<u8>>> {
    let records = records_once(streams, |rs| place(rs, "commit", n - 1).is_some());
    let commits: Vec<usize> = (0..n)
        .map(|i| place(&records, "commit", i).expect("the commit's record"))
        .collect();
    (commits.windows(2))
        .map(|w| writes_of(&records, block, w[0]..w[1]))
        .collect()
}

/// A command buffer committed inside `m2v.split` or `m2v.commitPart` writes the memory it
/// reaches that changed before that call's record, as a `commit` does, and is held committed:
/// a CPU write into a block buffer a blit reads is in the stream ahead of the split and of the
/// cut part that commit it.
#[test]
fn on_vulkan_a_split_and_a_cut_part_write_what_they_reach() {
    let test = "recorder::memory_tests::on_vulkan_a_split_and_a_cut_part_write_what_they_reach";
    in_recording_child(test, |streams| {
        let r = Rig::new();
        let first = r.command_buffer();
        r.read_back(&first);
        r.write(0..4096, 0x5a);
        let second = first.split(OnFault::Return).unwrap();
        let second = second.expect("an ended blit's buffer splits");
        assert!(held_committed(&first), "the split committed the first part");
        r.read_back(&second);
        r.write(4096..8192, 0x6b);
        assert!(matches!(
            second.commit_part(OnFault::Return),
            Ok(PartCut::Cut)
        ));
        assert!(held_committed(&second), "the cut committed the second part");
        second.wait_until_completed(OnFault::Return).unwrap();
        let records = records_once(streams, |rs| place(rs, "m2v.commitPart", 0).is_some());
        let split = place(&records, "m2v.split", 0).expect("the split's record");
        let cut = place(&records, "m2v.commitPart", 0).expect("the cut's record");
        let before_split = writes_of(&records, r.block(), 0..split);
        assert_eq!(before_split.get(&0), Some(&bytes(0x5a, 4096)));
        let before_cut = writes_of(&records, r.block(), split..cut);
        assert_eq!(before_cut.get(&4096), Some(&bytes(0x6b, 4096)));
    });
}

/// A fill writes only its range: CPU writes after the fill's command buffer completed, into a
/// chunk it never touched and into the bytes it filled, are in the stream before the next commit
/// that reaches them; the chunk the fill wrote part of, written again whole.
#[test]
fn on_vulkan_cpu_writes_after_a_fill_are_written() {
    let test = "recorder::memory_tests::on_vulkan_cpu_writes_after_a_fill_are_written";
    in_recording_child(test, |streams| {
        let r = Rig::new();
        let filled = r.command_buffer();
        r.fill(&filled, 0..256, 0x01);
        run(&filled);
        r.write(4096..8192, 0x11);
        r.write(0..256, 0x22);
        let next = r.command_buffer();
        r.read_back(&next);
        run(&next);
        let w = writes_between_commits(streams, r.block(), 2);
        assert_eq!(w[0].get(&0), Some(&chunk(&[(0x22, 256)])));
        assert_eq!(w[0].get(&4096), Some(&bytes(0x11, 4096)));
    });
}

/// A CPU write made while a fill runs, outside its range, is in the stream before the next commit
/// that reaches it.
#[test]
fn on_vulkan_a_cpu_write_beside_a_running_fill_is_written() {
    let test = "recorder::memory_tests::on_vulkan_a_cpu_write_beside_a_running_fill_is_written";
    in_recording_child(test, |streams| {
        let r = Rig::new();
        let filled = r.command_buffer();
        r.fill(&filled, 0..256, 0x01);
        filled.commit(OnFault::Return).unwrap();
        r.write(8192..12288, 0x33);
        filled.wait_until_completed(OnFault::Return).unwrap();
        let next = r.command_buffer();
        r.read_back(&next);
        run(&next);
        let w = writes_between_commits(streams, r.block(), 2);
        assert_eq!(w[0].get(&8192), Some(&bytes(0x33, 4096)));
    });
}

/// The chunks a copy writes whole are the stream's at its command buffer's completion, written
/// no more; a CPU write after it into one of them is in the stream before the next commit that
/// reaches it, and one into the other before the commit after.
#[test]
fn on_vulkan_chunks_a_copy_writes_whole_are_taken_at_completion() {
    let test =
        "recorder::memory_tests::on_vulkan_chunks_a_copy_writes_whole_are_taken_at_completion";
    in_recording_child(test, |streams| {
        let r = Rig::new();
        let source = crate::gpu::resource::tests::shared(8192);
        write_at(source.contents().as_ptr::<u8>().addr(), 0..8192, 0x44);
        let copied = r.command_buffer();
        r.copy_in(&copied, &source, 0..8192);
        run(&copied);
        r.write(0..16, 0x55);
        let second = r.command_buffer();
        r.read_back(&second);
        run(&second);
        r.write(4096..4112, 0x66);
        let third = r.command_buffer();
        r.read_back(&third);
        run(&third);
        let w = writes_between_commits(streams, r.block(), 3);
        assert_eq!(w[0].get(&0), Some(&chunk(&[(0x55, 16), (0x44, 4080)])));
        assert_eq!(
            w[0].get(&4096),
            None,
            "the copy's own bytes are the stream's"
        );
        assert_eq!(w[1].get(&4096), Some(&chunk(&[(0x66, 16), (0x44, 4080)])));
    });
}

/// The chunk a fill wrote only part of is written before the next commit that reaches it, the
/// GPU's bytes with the rest of it.
#[test]
fn on_vulkan_a_chunk_a_fill_writes_part_of_is_written() {
    let test = "recorder::memory_tests::on_vulkan_a_chunk_a_fill_writes_part_of_is_written";
    in_recording_child(test, |streams| {
        let r = Rig::new();
        let filled = r.command_buffer();
        r.fill(&filled, 0..256, 0x01);
        run(&filled);
        let next = r.command_buffer();
        r.read_back(&next);
        run(&next);
        let w = writes_between_commits(streams, r.block(), 2);
        assert_eq!(w[0].get(&0), Some(&chunk(&[(0x01, 256)])));
    });
}

/// What a command buffer's GPU work wrote is taken before its handlers run: a CPU write that a
/// completion-seen handler of core's, or a completed handler, makes into a chunk a copy wrote
/// whole is in the stream before the next commit that reaches it.
#[test]
fn on_vulkan_writes_its_handlers_make_are_written() {
    let test = "recorder::memory_tests::on_vulkan_writes_its_handlers_make_are_written";
    in_recording_child(test, |streams| {
        let r = Rig::new();
        let source = crate::gpu::resource::tests::shared(8192);
        write_at(source.contents().as_ptr::<u8>().addr(), 0..8192, 0x44);
        let copied = r.command_buffer();
        r.copy_in(&copied, &source, 0..8192);
        let base = r.y.contents().as_ptr::<u8>().addr();
        let seen: CompletedHandler = Box::new(move |_| write_at(base, 0..16, 0x77));
        copied
            .add_completion_seen_handler(OnFault::Return, seen)
            .unwrap();
        let completed: CompletedHandler = Box::new(move |_| write_at(base, 4096..4112, 0x78));
        copied
            .add_completed_handler(OnFault::Return, completed)
            .unwrap();
        run(&copied);
        let next = r.command_buffer();
        r.read_back(&next);
        run(&next);
        let w = writes_between_commits(streams, r.block(), 2);
        assert_eq!(w[0].get(&0), Some(&chunk(&[(0x77, 16), (0x44, 4080)])));
        assert_eq!(w[0].get(&4096), Some(&chunk(&[(0x78, 16), (0x44, 4080)])));
    });
}
