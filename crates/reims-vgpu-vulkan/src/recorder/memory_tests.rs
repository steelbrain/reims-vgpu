//! The memory a recording holds, on the Vulkan platform: what reaches the stream as `m2v.write`
//! records around commits ([`super::before_commit`]), completions ([`super::completion_seen`])
//! and announces ([`super::before_modify`]).
//! The recorder is the process's, started once, so each test records in a child copy of the
//! test binary of its own ([`in_recording_child`]).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::Duration;

use super::format::{self, Record, Value};
use super::{ENV_VAR, id_at, on, start_from_env, state};
use crate::core::host::HostPtr;
use crate::core::platform::Handle as _;
use crate::core::platform::Platform as _;
use crate::core::platform::blit::{BlitCmd, BlitEncoder as _, ResourceRef};
use crate::core::platform::command::{
    CommandBuffer as _, CommandQueue as _, CompletedHandler, PartCut,
};
use crate::core::platform::compute::{ComputeCmd, ComputeEncoder as _};
use crate::core::platform::device::GpuDevice as _;
use crate::core::platform::event::{EventRef, Notification, SharedEvent};
use crate::core::platform::icb::{
    IcbComputeCmd, IndirectCommandBuffer as _, IndirectComputeCommand as _,
};
use crate::core::platform::resource::{Buffer as _, Heap as _, Library as _};
use crate::core::platform::types::{CommandBufferStatus, MtlSize, NsRange};
use crate::core::platform::values::IcbDesc;
use crate::core::raise::OnFault;
use crate::gpu::command::{VulkanCommandBuffer, VulkanCommandQueue};
use crate::gpu::icb::VulkanIcb;
use crate::gpu::resource::{VulkanBuffer, VulkanComputePipeline, VulkanHeap};
use crate::platform::VulkanPlatform;

/// The marker that makes a run of a recording test its child, which records: the test's name.
const RECORD_CHILD: &str = "REIMS_VGPU_RECORDER_CHILD";

/// The bytes of the block the tests' buffer fills.
const BLOCK: u64 = 0x1_0000;

/// `inc_k`: one thread adds 1 to `c[0]`, and marks `c[n]` with `n * 0x11` below 16.
const INC: &[u8] = include_bytes!("../../../../tests/shaders/compute_inc.metallib");

/// `MTLResourceUsageRead`.
const READ: u64 = 1;

/// `MTLResourceUsageRead | MTLResourceUsageWrite`.
const READ_WRITE: u64 = 3;

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

/// Whether the recorder holds the command buffer of stream id `id` committed once the guest let
/// go of it, as core lets go of each at its commit, while the GPU still runs it.
fn held_released(id: u64) -> bool {
    let state = state().expect("the child records");
    state.drain(true);
    let sync = state.sync.lock();
    (sync.pending.iter()).any(|p| p.id == id && p.weak.strong_count() == 0)
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

    /// An indirect command buffer of one compute command, which the CPU encoded binding the
    /// buffer where `bound` (and dispatching nothing), else none.
    fn icb(&self, bound: bool) -> VulkanIcb {
        let desc = IcbDesc {
            command_types: 0x60,
            max_kernel_buffer_bind_count: 1,
            ..IcbDesc::default()
        };
        let d = crate::gpu::device::tests::dev();
        let made = d.new_indirect_command_buffer(&desc, 1, 0);
        let icb = made.unwrap().unwrap();
        if bound {
            let command = icb.compute_command(0).unwrap().unwrap();
            let bind = IcbComputeCmd::SetKernelBuffer {
                buffer: Some(&self.y),
                offset: 0,
                index: 0,
            };
            command.encode(bind).unwrap();
        }
        icb
    }

    /// Encodes in `cb` a compute encoder of its own that runs one thread of `inc` ([`INC`]) on
    /// `c`, and, where `used` names one, makes that buffer resident with that usage first
    /// (`useResources:count:usage:`).
    fn count(
        &self,
        cb: &VulkanCommandBuffer,
        inc: &VulkanComputePipeline,
        c: &VulkanBuffer,
        used: Option<(&VulkanBuffer, u64)>,
    ) {
        let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
        let buffers = [Some(c)];
        let resources = used.map(|(b, usage)| ([Some(ResourceRef::Buffer(b))], usage));
        let mut cmds = Vec::new();
        if let Some((resources, usage)) = &resources {
            cmds.push(ComputeCmd::UseResources {
                resources,
                usage: *usage,
            });
        }
        cmds.extend([
            ComputeCmd::SetComputePipelineState { state: Some(inc) },
            ComputeCmd::SetBuffers {
                buffers: &buffers,
                offsets: &[0],
                range: NsRange::new(0, 1),
            },
            ComputeCmd::DispatchThreads {
                threads_per_grid: MtlSize::new(1, 1, 1),
                threads_per_threadgroup: MtlSize::new(1, 1, 1),
            },
            ComputeCmd::EndEncoding,
        ]);
        for cmd in cmds {
            e.encode(OnFault::Return, cmd).unwrap();
        }
    }

    /// Encodes in `cb` an execution of `icb`'s command, in a compute encoder of its own that
    /// names nothing else.
    fn execute(&self, cb: &VulkanCommandBuffer, icb: &VulkanIcb) {
        let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
        let execute = ComputeCmd::ExecuteCommandsInBuffer {
            indirect_command_buffer: icb,
            execution_range: NsRange::new(0, 1),
        };
        for cmd in [execute, ComputeCmd::EndEncoding] {
            e.encode(OnFault::Return, cmd).unwrap();
        }
    }
}

/// The pipeline of `inc_k` ([`INC`]).
fn inc_pipeline() -> VulkanComputePipeline {
    let d = crate::gpu::device::tests::dev();
    let library = d.new_library(INC).unwrap().unwrap().unwrap();
    let f = library.new_function(OnFault::Return, "inc_k").unwrap();
    d.new_compute_pipeline_state_for_function(f.as_ref())
        .unwrap()
        .unwrap()
}

/// The host address of `b`'s bytes, a `Shared` buffer's.
fn base_of(b: &VulkanBuffer) -> usize {
    b.contents().as_ptr::<u8>().addr()
}

/// What `inc_k` ([`INC`]) leaves in a chunk of zeros it ran on once: `c[0]` and `c[1]`, 1 and
/// `0x11`.
fn counted_once() -> Vec<u8> {
    chunk(&[(1, 1), (0, 3), (0x11, 1)])
}

/// The bytes each `m2v.write` record of `target` among `records[range]` writes at its byte `at`,
/// in the stream's order.
fn writes_at(
    records: &[Record],
    target: u64,
    range: std::ops::Range<usize>,
    at: u64,
) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    for i in range {
        if let Some(bytes) = writes_of(records, target, i..i + 1).remove(&at) {
            out.push(bytes);
        }
    }
    out
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

/// The chunk a fill wrote only part of is written at its command buffer's completion, before the
/// next commit, the GPU's bytes with the rest of it.
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

/// A command buffer reaches what the commands of an indirect command buffer it executes name: a
/// CPU write into a block buffer only such a command binds is in the stream before the commit of
/// the next command buffer that executes it.
#[test]
fn on_vulkan_a_write_an_indirect_command_binds_is_written() {
    let test = "recorder::memory_tests::on_vulkan_a_write_an_indirect_command_binds_is_written";
    in_recording_child(test, |streams| {
        let r = Rig::new();
        let icb = r.icb(true);
        let first = r.command_buffer();
        r.execute(&first, &icb);
        run(&first);
        r.write(0..16, 0x5c);
        let second = r.command_buffer();
        r.execute(&second, &icb);
        run(&second);
        let w = writes_between_commits(streams, r.block(), 2);
        assert_eq!(w[0].get(&0), Some(&chunk(&[(0x5c, 16)])));
    });
}

/// A copy of indirect commands copies what they name: a CPU write into a block buffer only a
/// command copied into another buffer binds is in the stream before the commit of a command
/// buffer that executes the copy.
#[test]
fn on_vulkan_a_write_a_copied_indirect_command_binds_is_written() {
    let test =
        "recorder::memory_tests::on_vulkan_a_write_a_copied_indirect_command_binds_is_written";
    in_recording_child(test, |streams| {
        let r = Rig::new();
        let (icb, copy) = (r.icb(true), r.icb(false));
        let copied = r.command_buffer();
        let into = BlitCmd::CopyIndirectCommandBuffer {
            source: &icb,
            source_range: NsRange::new(0, 1),
            destination: &copy,
            destination_index: 0,
        };
        r.blit(&copied, vec![into]);
        run(&copied);
        r.write(0..16, 0x5d);
        let executed = r.command_buffer();
        r.execute(&executed, &copy);
        run(&executed);
        let w = writes_between_commits(streams, r.block(), 2);
        assert_eq!(w[0].get(&0), Some(&chunk(&[(0x5d, 16)])));
    });
}

/// A `Managed` buffer of two chunks.
fn managed() -> VulkanBuffer {
    let d = crate::gpu::device::tests::dev();
    let made = d.new_buffer_of_length(OnFault::Return, 8192, 0x10);
    made.unwrap().expect("a Managed buffer")
}

/// Writes `value` into `bytes` of `m`, as the CPU.
fn write_into(m: &VulkanBuffer, bytes: std::ops::Range<u64>, value: u8) {
    write_at(m.contents().as_ptr::<u8>().addr(), bytes, value);
}

/// `didModifyRange:` of all of `m`: an announce.
fn announce(m: &VulkanBuffer) {
    m.did_modify_range(OnFault::Return, NsRange::new(0, m.length()))
        .unwrap();
}

/// Runs, in a command buffer of `r`'s queue, a fill of `bytes` of `m` with `value` and then a
/// `synchronizeResource:` of `m`: the GPU's write read back into the CPU's copy, as a guest
/// reads a `Managed` buffer the GPU wrote (on a device of one copy, the synchronize does
/// nothing).
fn fill_and_read_back(r: &Rig, m: &VulkanBuffer, bytes: std::ops::Range<u64>, value: u8) {
    let cb = r.command_buffer();
    let range = NsRange {
        location: bytes.start,
        length: bytes.end - bytes.start,
    };
    let fill = BlitCmd::FillBuffer {
        buffer: m,
        range,
        value,
    };
    let synchronize = BlitCmd::SynchronizeResource {
        resource: Some(ResourceRef::Buffer(m)),
    };
    r.blit(&cb, vec![fill, synchronize]);
    run(&cb);
}

/// `m`'s writes before each `didModifyRange:` record of the stream, once it holds `n` of them:
/// the `i`th map those after the one before announce `i` (from the stream's start for the
/// first), up to it.
fn writes_before_announces(
    streams: &Path,
    m: &VulkanBuffer,
    n: usize,
) -> Vec<BTreeMap<u64, Vec<u8>>> {
    let target = id_at(m.id().0).expect("the buffer is recorded");
    let records = records_once(streams, |rs| place(rs, "didModifyRange:", n - 1).is_some());
    let mut ends = vec![0];
    for i in 0..n {
        ends.push(place(&records, "didModifyRange:", i).expect("the announce's record"));
    }
    (ends.windows(2))
        .map(|w| writes_of(&records, target, w[0]..w[1]))
        .collect()
}

/// What the GPU wrote and the guest read back is in the stream at its announce: a `Managed`
/// buffer whose two chunks a fill wrote whole and a `synchronizeResource:` brought back,
/// announced with no CPU write since, is written with the GPU's bytes before that
/// `didModifyRange:`, though the stream took them at the completion without a write. An announce
/// after it writes nothing.
#[test]
fn on_vulkan_an_announce_writes_what_the_gpu_wrote() {
    let test = "recorder::memory_tests::on_vulkan_an_announce_writes_what_the_gpu_wrote";
    in_recording_child(test, |streams| {
        let r = Rig::new();
        let m = managed();
        write_into(&m, 0..8192, 0xa1);
        announce(&m);
        fill_and_read_back(&r, &m, 0..8192, 0x4e);
        announce(&m);
        announce(&m);
        let w = writes_before_announces(streams, &m, 3);
        let all = |value| BTreeMap::from([(0, bytes(value, 4096)), (4096, bytes(value, 4096))]);
        assert_eq!(w[0], all(0xa1), "the CPU's write");
        assert_eq!(w[1], all(0x4e), "the GPU's write, read back");
        assert_eq!(w[2], BTreeMap::new(), "nothing changed since");
    });
}

/// A `synchronizeResource:` takes all of its buffer as the GPU's write: of a `Managed` buffer a
/// fill wrote only part of a chunk of, read back and announced with no CPU write since, that
/// chunk is written before the `didModifyRange:`, the GPU's bytes with the CPU's beside them,
/// and the chunk nothing wrote is not.
#[test]
fn on_vulkan_an_announce_writes_what_a_synchronize_took() {
    let test = "recorder::memory_tests::on_vulkan_an_announce_writes_what_a_synchronize_took";
    in_recording_child(test, |streams| {
        let r = Rig::new();
        let m = managed();
        write_into(&m, 0..8192, 0xa2);
        announce(&m);
        fill_and_read_back(&r, &m, 0..16, 0x4f);
        announce(&m);
        let w = writes_before_announces(streams, &m, 2);
        let part = chunk(&[(0x4f, 16), (0xa2, 4080)]);
        assert_eq!(w[1], BTreeMap::from([(0, part)]));
    });
}

/// A CPU write over the GPU's is what an announce writes: of a `Managed` buffer the GPU wrote
/// and the guest read back, then wrote again as the CPU, the CPU's bytes are written before the
/// `didModifyRange:`.
#[test]
fn on_vulkan_an_announce_writes_a_cpu_write_over_the_gpus() {
    let test = "recorder::memory_tests::on_vulkan_an_announce_writes_a_cpu_write_over_the_gpus";
    in_recording_child(test, |streams| {
        let r = Rig::new();
        let m = managed();
        write_into(&m, 0..8192, 0xa3);
        announce(&m);
        fill_and_read_back(&r, &m, 0..8192, 0x50);
        write_into(&m, 0..8192, 0xa3);
        announce(&m);
        let w = writes_before_announces(streams, &m, 2);
        let all = BTreeMap::from([(0, bytes(0xa3, 4096)), (4096, bytes(0xa3, 4096))]);
        assert_eq!(w[1], all);
    });
}

/// An announce of bytes the stream's last write of them recorded writes nothing.
#[test]
fn on_vulkan_an_announce_of_recorded_bytes_writes_nothing() {
    let test = "recorder::memory_tests::on_vulkan_an_announce_of_recorded_bytes_writes_nothing";
    in_recording_child(test, |streams| {
        let m = managed();
        write_into(&m, 0..8192, 0xa4);
        announce(&m);
        announce(&m);
        let w = writes_before_announces(streams, &m, 2);
        assert_eq!(w[0].len(), 2, "the CPU's write");
        assert_eq!(w[1], BTreeMap::new());
    });
}

/// A CPU write that puts back what a kernel changed is written: a counter a kernel added to,
/// zeroed again by the CPU, is written with the kernel's bytes at its command buffer's
/// completion, after its commit, and with the zeros before the next commit that reaches it.
#[test]
fn on_vulkan_a_cpu_write_that_restores_what_a_kernel_changed_is_written() {
    let test = "recorder::memory_tests::on_vulkan_a_cpu_write_that_restores_what_a_kernel_changed_is_written";
    in_recording_child(test, |streams| {
        let (r, inc) = (Rig::new(), inc_pipeline());
        write_at(base_of(&r.z), 0..4096, 0);
        let first = r.command_buffer();
        r.count(&first, &inc, &r.z, None);
        run(&first);
        write_at(base_of(&r.z), 0..8, 0);
        let second = r.command_buffer();
        r.count(&second, &inc, &r.z, None);
        run(&second);
        let z = id_at(r.z.id().0).expect("the buffer is recorded");
        let records = records_once(streams, |rs| place(rs, "commit", 1).is_some());
        let commits = [0, 1].map(|n| place(&records, "commit", n).expect("a commit's record"));
        let between = writes_at(&records, z, commits[0]..commits[1], 0);
        assert_eq!(between, vec![counted_once(), bytes(0, 4096)]);
        assert_eq!(
            writes_at(&records, z, 0..commits[0], 0),
            vec![bytes(0, 4096)]
        );
    });
}

/// A CPU write made while a kernel's command buffer runs, into a chunk the kernel stores to, is
/// written with the kernel's bytes at the completion: the stream's bytes are not taken as the
/// GPU's without a write there.
#[test]
fn on_vulkan_a_cpu_write_while_a_kernel_runs_is_written() {
    let test = "recorder::memory_tests::on_vulkan_a_cpu_write_while_a_kernel_runs_is_written";
    in_recording_child(test, |streams| {
        let (r, inc) = (Rig::new(), inc_pipeline());
        write_at(base_of(&r.z), 0..4096, 0);
        let gate = crate::gpu::device::tests::dev().new_shared_event().unwrap();
        let held = r.command_buffer();
        held.encode_wait(Some(EventRef::Shared(&gate)), 1).unwrap();
        r.count(&held, &inc, &r.z, None);
        held.commit(OnFault::Return).unwrap();
        write_at(base_of(&r.z), 1024..1040, 0x5e);
        gate.set_signaled_value(1).unwrap();
        held.wait_until_completed(OnFault::Return).unwrap();
        let next = r.command_buffer();
        r.read_back(&next);
        run(&next);
        let z = id_at(r.z.id().0).expect("the buffer is recorded");
        let records = records_once(streams, |rs| place(rs, "commit", 1).is_some());
        let commits = [0, 1].map(|n| place(&records, "commit", n).expect("a commit's record"));
        let mut both = counted_once();
        both[1024..1040].fill(0x5e);
        assert_eq!(
            writes_at(&records, z, commits[0]..commits[1], 0),
            vec![both]
        );
    });
}

/// What a blit writes whole is still taken as the stream's without a write where a kernel runs
/// beside it in its command buffer, and the kernel's bytes are written at the completion.
#[test]
fn on_vulkan_a_blits_chunks_beside_a_kernel_are_taken_without_a_write() {
    let test = "recorder::memory_tests::on_vulkan_a_blits_chunks_beside_a_kernel_are_taken_without_a_write";
    in_recording_child(test, |streams| {
        let (r, inc) = (Rig::new(), inc_pipeline());
        write_at(base_of(&r.z), 0..4096, 0);
        let source = crate::gpu::resource::tests::shared(8192);
        write_at(base_of(&source), 0..8192, 0x44);
        let both = r.command_buffer();
        r.copy_in(&both, &source, 0..8192);
        r.count(&both, &inc, &r.z, None);
        run(&both);
        let next = r.command_buffer();
        r.read_back(&next);
        run(&next);
        let z = id_at(r.z.id().0).expect("the buffer is recorded");
        let records = records_once(streams, |rs| place(rs, "commit", 1).is_some());
        let commits = [0, 1].map(|n| place(&records, "commit", n).expect("a commit's record"));
        let block = writes_of(&records, r.block(), commits[0]..commits[1]);
        assert_eq!(block.get(&0), None, "the copy's own bytes are the stream's");
        assert_eq!(
            block.get(&4096),
            None,
            "the copy's own bytes are the stream's"
        );
        let kernel = writes_at(&records, z, commits[0]..commits[1], 0);
        assert_eq!(kernel, vec![counted_once()]);
    });
}

/// A completion hashes only what a kernel could store to: of a command buffer whose kernel binds
/// one chunk and that makes a block buffer of 16 resident for reading, the one chunk; with the
/// block buffer resident for writing too, all 17.
#[test]
fn on_vulkan_a_completion_hashes_only_what_a_kernel_could_store_to() {
    let test =
        "recorder::memory_tests::on_vulkan_a_completion_hashes_only_what_a_kernel_could_store_to";
    in_recording_child(test, |_| {
        let (r, inc) = (Rig::new(), inc_pipeline());
        let hashed = || {
            let stats = &state().expect("the child records").stats;
            stats.completion_chunks_hashed.load(Ordering::Relaxed)
        };
        let read = r.command_buffer();
        r.count(&read, &inc, &r.z, Some((&r.y, READ)));
        let was = hashed();
        run(&read);
        assert_eq!(hashed() - was, 1, "the bound chunk alone");
        let written = r.command_buffer();
        r.count(&written, &inc, &r.z, Some((&r.y, READ_WRITE)));
        let was = hashed();
        run(&written);
        assert_eq!(
            hashed() - was,
            1 + BLOCK / 4096,
            "the bound chunk and the block's"
        );
    });
}

/// A CPU write that puts back what a fill wrote in part of a chunk is written: the chunk is
/// written with the fill's bytes at the completion, and with the CPU's before the next commit
/// that reaches it.
#[test]
fn on_vulkan_a_cpu_write_that_restores_what_a_fill_wrote_part_of_is_written() {
    let test = "recorder::memory_tests::on_vulkan_a_cpu_write_that_restores_what_a_fill_wrote_part_of_is_written";
    in_recording_child(test, |streams| {
        let r = Rig::new();
        r.write(0..4096, 0);
        let filled = r.command_buffer();
        r.fill(&filled, 0..256, 0x01);
        run(&filled);
        r.write(0..256, 0);
        let next = r.command_buffer();
        r.read_back(&next);
        run(&next);
        let records = records_once(streams, |rs| place(rs, "commit", 1).is_some());
        let commits = [0, 1].map(|n| place(&records, "commit", n).expect("a commit's record"));
        let between = writes_at(&records, r.block(), commits[0]..commits[1], 0);
        assert_eq!(between, vec![chunk(&[(0x01, 256)]), bytes(0, 4096)]);
    });
}

/// The completion of a command buffer the guest let go of once it committed it, as core does,
/// is taken though a commit reaps the command buffer before it completes: a counter its kernel
/// added to while the commit after it ran, zeroed again by the CPU, is written with the kernel's
/// bytes at the completion and with the zeros before the next commit that reaches it.
#[test]
fn on_vulkan_a_released_command_buffers_completion_is_written() {
    let test = "recorder::memory_tests::on_vulkan_a_released_command_buffers_completion_is_written";
    in_recording_child(test, |streams| {
        let (r, inc) = (Rig::new(), inc_pipeline());
        write_at(base_of(&r.z), 0..4096, 0);
        let gate = crate::gpu::device::tests::dev().new_shared_event().unwrap();
        let released = r.command_buffer();
        released
            .encode_wait(Some(EventRef::Shared(&gate)), 1)
            .unwrap();
        r.count(&released, &inc, &r.z, None);
        released.commit(OnFault::Return).unwrap();
        let id = id_at(released.id().0).expect("the command buffer is recorded");
        let raw = released.raw.clone();
        drop(released);
        assert!(held_released(id), "held committed, and let go of");
        let reaps = r.command_buffer();
        reaps.commit(OnFault::Return).unwrap();
        gate.set_signaled_value(1).unwrap();
        raw.wait_until_completed().unwrap();
        write_at(base_of(&r.z), 0..8, 0);
        let next = r.command_buffer();
        r.count(&next, &inc, &r.z, None);
        run(&next);
        let z = id_at(r.z.id().0).expect("the buffer is recorded");
        let records = records_once(streams, |rs| place(rs, "commit", 2).is_some());
        let commits = [0, 2].map(|n| place(&records, "commit", n).expect("a commit's record"));
        let between = writes_at(&records, z, commits[0]..commits[1], 0);
        assert_eq!(between, vec![counted_once(), bytes(0, 4096)]);
    });
}

/// The command buffer `cb`, made of `r`'s queue, adds to the `Shared` buffer's counter with
/// `inc`, then signals `signal` to 1, then waits for `gate` to reach 1 before a fill of the
/// block buffer: its completion waits for the gate, its signal does not.
fn count_signal_then_wait(
    r: &Rig,
    inc: &VulkanComputePipeline,
    signal: &crate::gpu::event::VulkanSharedEvent,
    gate: &crate::gpu::event::VulkanSharedEvent,
) -> VulkanCommandBuffer {
    let cb = r.command_buffer();
    r.count(&cb, inc, &r.z, None);
    cb.encode_signal(Some(EventRef::Shared(signal)), 1).unwrap();
    cb.encode_wait(Some(EventRef::Shared(gate)), 1).unwrap();
    r.fill(&cb, 0..16, 0x01);
    cb
}

/// A CPU write the guest makes once an event a command buffer signals reached the signal's
/// value, before the command buffer completes, is written: a counter its kernel added to
/// before the signal, zeroed again by a listener's notification of it while the command buffer
/// waits for a gate, is written with the kernel's bytes before the notification runs and with
/// the zeros at the completion.
#[test]
fn on_vulkan_a_cpu_write_once_a_signal_is_reached_is_written() {
    let test = "recorder::memory_tests::on_vulkan_a_cpu_write_once_a_signal_is_reached_is_written";
    in_recording_child(test, |streams| {
        let (r, inc) = (Rig::new(), inc_pipeline());
        write_at(base_of(&r.z), 0..4096, 0);
        let d = crate::gpu::device::tests::dev();
        let (signal, gate) = (d.new_shared_event().unwrap(), d.new_shared_event().unwrap());
        let cb = count_signal_then_wait(&r, &inc, &signal, &gate);
        let listener = crate::platform::tests::platform().new_shared_event_listener();
        let (told, notified) = std::sync::mpsc::channel();
        let base = base_of(&r.z);
        let zero: Notification = Box::new(move || {
            write_at(base, 0..8, 0);
            told.send(()).unwrap();
        });
        SharedEvent::notify_listener(&signal, &listener, 1, zero).unwrap();
        cb.commit(OnFault::Return).unwrap();
        (notified.recv_timeout(Duration::from_secs(10))).expect("the notification");
        gate.set_signaled_value(1).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        let next = r.command_buffer();
        r.count(&next, &inc, &r.z, None);
        run(&next);
        let z = id_at(r.z.id().0).expect("the buffer is recorded");
        let records = records_once(streams, |rs| place(rs, "commit", 1).is_some());
        let commits = [0, 1].map(|n| place(&records, "commit", n).expect("a commit's record"));
        let between = writes_at(&records, z, commits[0]..commits[1], 0);
        assert_eq!(between, vec![counted_once(), bytes(0, 4096)]);
    });
}

/// A CPU write the guest makes once a command buffer that waited for another's signal
/// completed, before that other completes, is written: a counter the other's kernel added to
/// before the signal, zeroed again by the CPU while the other waits for a gate, is written with
/// the kernel's bytes as the waiter completes and with the zeros at the other's completion.
#[test]
fn on_vulkan_a_cpu_write_once_a_waiter_on_a_signal_completes_is_written() {
    let test = "recorder::memory_tests::on_vulkan_a_cpu_write_once_a_waiter_on_a_signal_completes_is_written";
    in_recording_child(test, |streams| {
        let (r, inc) = (Rig::new(), inc_pipeline());
        write_at(base_of(&r.z), 0..4096, 0);
        let d = crate::gpu::device::tests::dev();
        let (signal, gate) = (d.new_shared_event().unwrap(), d.new_shared_event().unwrap());
        let cb = count_signal_then_wait(&r, &inc, &signal, &gate);
        let other = d.new_command_queue().unwrap();
        let waiter = other.command_buffer(OnFault::Return).unwrap().unwrap();
        waiter
            .encode_wait(Some(EventRef::Shared(&signal)), 1)
            .unwrap();
        cb.commit(OnFault::Return).unwrap();
        run(&waiter);
        write_at(base_of(&r.z), 0..8, 0);
        gate.set_signaled_value(1).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        let next = r.command_buffer();
        r.count(&next, &inc, &r.z, None);
        run(&next);
        let z = id_at(r.z.id().0).expect("the buffer is recorded");
        let records = records_once(streams, |rs| place(rs, "commit", 2).is_some());
        let commits = [0, 2].map(|n| place(&records, "commit", n).expect("a commit's record"));
        let between = writes_at(&records, z, commits[0]..commits[1], 0);
        assert_eq!(between, vec![counted_once(), bytes(0, 4096)]);
    });
}

/// A CPU write the guest makes once it read a command buffer's status final, before the
/// runtime ran the recorder's hook, is written: a counter its kernel added to, zeroed again by
/// the CPU while a runtime handler added ahead of the hook holds it back, is written with the
/// kernel's bytes as the status reads final and with the zeros before the next commit.
#[test]
fn on_vulkan_a_cpu_write_once_a_status_reads_final_is_written() {
    let test = "recorder::memory_tests::on_vulkan_a_cpu_write_once_a_status_reads_final_is_written";
    in_recording_child(test, |streams| {
        let (r, inc) = (Rig::new(), inc_pipeline());
        write_at(base_of(&r.z), 0..4096, 0);
        let counted = r.command_buffer();
        r.count(&counted, &inc, &r.z, None);
        let (go, held) = std::sync::mpsc::channel::<()>();
        let hold: metal2vulkan_runtime::Handler =
            Box::new(move |_: &metal2vulkan_runtime::CommandBuffer| {
                let _ = held.recv_timeout(Duration::from_secs(10));
            });
        counted.raw.add_completion_seen_handler(hold).unwrap();
        counted.commit(OnFault::Return).unwrap();
        for polls in 0.. {
            if counted.status() == CommandBufferStatus::COMPLETED {
                break;
            }
            assert!(polls < 10_000, "the command buffer completes");
            std::thread::sleep(Duration::from_millis(1));
        }
        write_at(base_of(&r.z), 0..8, 0);
        go.send(()).unwrap();
        counted.wait_until_completed(OnFault::Return).unwrap();
        let next = r.command_buffer();
        r.count(&next, &inc, &r.z, None);
        run(&next);
        let z = id_at(r.z.id().0).expect("the buffer is recorded");
        let records = records_once(streams, |rs| place(rs, "commit", 1).is_some());
        let commits = [0, 1].map(|n| place(&records, "commit", n).expect("a commit's record"));
        let between = writes_at(&records, z, commits[0]..commits[1], 0);
        assert_eq!(between, vec![counted_once(), bytes(0, 4096)]);
    });
}

/// A no-copy buffer's escape ([`super::buffer_escaped`]) is one [`super::BUFFER_ESCAPED`]
/// record on the buffer, `[kind]`, where the note came in the stream's order: after the buffer's
/// create and after the `m2v.write` of its bytes that a commit made before the note wrote, and
/// before the one of a commit made after it, so a player meets the record before any write the
/// GPU may read the buffer's address from. An object that is no buffer of the Vulkan platform's
/// writes none, and the record is counted by its selector.
#[test]
fn on_vulkan_a_buffers_escape_lies_between_the_commits_around_its_note() {
    let test = "recorder::memory_tests::on_vulkan_a_buffers_escape_lies_between_the_commits_around_its_note";
    in_recording_child(test, |streams| {
        use crate::core::platform::HostRange;
        use crate::core::platform::values::{BufferDesc, OptionSet};
        use crate::gpu::resource::tests::{guest_pages_of, shared};
        let d = crate::gpu::device::tests::dev();
        let pages = guest_pages_of(BLOCK);
        // SAFETY: pages of the test's own, left mapped until the process ends.
        let over = unsafe { HostRange::new(HostPtr::from_addr(pages as usize), BLOCK) };
        let b = d
            .new_buffer(BufferDesc {
                length: BLOCK,
                options: vec![OptionSet::ResourceOptions(0)],
                pinned_gpu_address: None,
                contents: Some(over),
                no_copy: true,
                deallocator: None,
                zero_from: None,
            })
            .unwrap()
            .expect("a no-copy buffer over the pages");
        let (z, q) = (shared(4096), d.new_command_queue().unwrap());
        // A command buffer that reads the buffer's first 16 bytes, committed and waited for.
        let read_back = || {
            let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
            let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
            let copy = BlitCmd::CopyBufferToBuffer {
                source_buffer: Some(&b),
                source_offset: 0,
                destination_buffer: Some(&z),
                destination_offset: 0,
                size: 16,
            };
            for cmd in [copy, BlitCmd::EndEncoding] {
                e.encode(OnFault::Return, cmd).unwrap();
            }
            run(&cb);
        };
        let pages = usize::try_from(pages).unwrap();
        write_at(pages, 0..4096, 0x11);
        read_back();
        // The guest's note that its `gpuResourceID` was read (`RGFX_BUFFER_ESCAPED_RESOURCE_ID`).
        super::buffer_escaped(&b, 2);
        super::buffer_escaped(&(), 0);
        write_at(pages, 0..4096, 0x22);
        read_back();

        let id = id_at(b.id().0).expect("the buffer is recorded");
        let records = records_once(streams, |rs| place(rs, "commit", 1).is_some());
        let made = (records.iter())
            .position(|r| matches!(r, Record::Call(c) if c.result == id))
            .expect("the buffer's create");
        let escaped: Vec<(usize, &format::Call)> = (records.iter().enumerate())
            .filter_map(|(i, r)| match r {
                Record::Call(c) if c.selector == super::BUFFER_ESCAPED => Some((i, c)),
                _ => None,
            })
            .collect();
        let [(at, record)] = escaped[..] else {
            panic!("one escape record: {escaped:?}");
        };
        assert_eq!(
            (record.receiver, &record.args[..]),
            (id, &[Value::Uint(2)][..])
        );
        let commits = [0, 1].map(|n| place(&records, "commit", n).expect("a commit's record"));
        assert!(
            made < commits[0] && commits[0] < at && at < commits[1],
            "the create at {made}, the escape at {at}, the commits at {commits:?}"
        );
        let before = writes_of(&records, id, made..commits[0]);
        assert_eq!(before.get(&0), Some(&bytes(0x11, 4096)));
        assert!(writes_of(&records, id, commits[0]..at).is_empty());
        let after = writes_of(&records, id, at..commits[1]);
        assert_eq!(after.get(&0), Some(&bytes(0x22, 4096)));
        let counted = state().expect("the child records").selectors.lock();
        assert_eq!(counted.get(super::BUFFER_ESCAPED), Some(&1));
    });
}
