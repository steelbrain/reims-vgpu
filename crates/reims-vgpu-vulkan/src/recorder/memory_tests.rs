//! The memory a recording holds, on the Vulkan platform: what reaches the stream as `m2v.write`
//! records around commits ([`super::before_commit`]). The recorder is the process's, started
//! once, so each test records in a child copy of the test binary of its own
//! ([`in_recording_child`]).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use super::format::{self, Record, Value};
use super::{ENV_VAR, id_at, on, start_from_env, state};
use crate::core::platform::Handle as _;
use crate::core::platform::blit::{BlitCmd, BlitEncoder as _};
use crate::core::platform::command::{CommandBuffer as _, CommandQueue as _, PartCut};
use crate::core::platform::device::GpuDevice as _;
use crate::core::platform::resource::{Buffer as _, Heap as _};
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
    !sync.cbs.contains_key(&id) && !sync.pending.is_empty()
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

    /// Writes `value` into `bytes` of the buffer, as the CPU.
    fn write(&self, bytes: std::ops::Range<u64>, value: u8) {
        assert!(bytes.end <= BLOCK);
        // SAFETY: `bytes` lies in the buffer, `Shared`, whose memory the test's pages hold.
        unsafe {
            let at = self.y.contents().as_ptr::<u8>().add(bytes.start as usize);
            std::ptr::write_bytes(at, value, (bytes.end - bytes.start) as usize);
        }
    }
}

/// `n` bytes of `value`.
fn bytes(value: u8, n: usize) -> Vec<u8> {
    vec![value; n]
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
        assert!(matches!(second.commit_part(OnFault::Return), Ok(PartCut::Cut)));
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
