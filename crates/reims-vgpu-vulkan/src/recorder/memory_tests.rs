//! The memory a recording holds, on the rt platform: what reaches the stream as `m2v.write`
//! records around commits ([`super::before_commit`]), for the buffers of the guest's blocks.
//! The recorder is the process's, started once, so each test records in a child copy of the
//! test binary of its own ([`in_recording_child`]).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use super::format::{self, Record, Value};
use super::{ENV_VAR, id_at, on, start_from_env};
use crate::core::host::HostPtr;
use crate::core::platform::Handle as _;
use crate::core::platform::blit::{BlitCmd, BlitEncoder as _};
use crate::core::platform::command::{CommandBuffer as _, CommandQueue as _};
use crate::core::platform::compute::{ComputeCmd, ComputeEncoder as _};
use crate::core::platform::device::GpuDevice;
use crate::core::platform::resource::{Buffer as _, Heap as _, Library as _};
use crate::core::platform::types::{MtlSize, NsRange};
use crate::core::platform::values::{HeapDesc, OptionSet};
use crate::core::platform::{HostRange, Platform};
use crate::core::raise::OnFault;
use crate::rt_platform::RtPlatform;

/// The marker that makes a run of a recording test its child, which records: the test's name.
const RECORD_CHILD: &str = "REIMS_VGPU_RECORDER_CHILD";

/// The bytes of the block the tests' buffer fills.
const BLOCK: u64 = 0x1_0000;

/// `inc_k`: one thread adds 1 to `c[0]`, and marks `c[n]` with `n * 0x11` below 16.
const INC: &[u8] = include_bytes!("../../../../tests/shaders/compute_inc.metallib");

/// Runs `body` in a child copy of this test binary that records into a scratch directory of its
/// own, `body` given the directory its stream goes into; in the child, runs it. The child's
/// stderr lines that start `recorder-test: ` are passed on to the parent's.
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
    let told = String::from_utf8_lossy(&out.stderr);
    (told.lines())
        .filter(|l| l.starts_with("recorder-test: "))
        .for_each(|l| eprintln!("{l}"));
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

/// The bytes the block tests' buffers take in their block: 176 of them, as the guest's small
/// buffers lie in its blocks, from 256 bytes past a chunk's start, at the alignment Metal asks of
/// a buffer's offset in a heap.
const IN_BLOCK: std::ops::Range<u64> = 0x2100..0x21b0;

/// `MTLResourceStorageModeShared`, with no other option.
const SHARED: u64 = 0;

/// A block of [`BLOCK`] bytes on `d`, a `Shared` heap over pages of the test's, as the decoder
/// makes one over the guest's memory, and its stream id.
fn block_on<P: Platform>(d: &P::Device) -> (P::Heap, u64)
where
    P::Device: GpuDevice<P>,
{
    let pages = crate::os::allocate_aliasable(BLOCK).unwrap();
    // SAFETY: pages of the test's, left mapped until the process ends.
    let bytes = unsafe { HostRange::new(HostPtr::from_addr(pages.addr()), BLOCK) };
    let made = d.new_heap_over_host_memory(
        bytes,
        &HeapDesc {
            size: BLOCK,
            heap_type: 1,
            options: vec![OptionSet::StorageMode(0), OptionSet::HazardTrackingMode(0)],
            pinned_gpu_address: None,
        },
    );
    let heap = made.unwrap().heap.unwrap();
    let id = id_at(heap.id().0).expect("the block is recorded");
    (heap, id)
}

/// A `Shared` buffer of `heap`'s over [`IN_BLOCK`] (`newBufferWithLength:options:offset:`, the
/// stream's `m2v.newBufferInBlock`), each of its bytes then written `value` by the CPU.
fn in_block_with<P: Platform>(heap: &P::Heap, value: u8) -> P::Buffer {
    let length = IN_BLOCK.end - IN_BLOCK.start;
    let made = heap.new_buffer_of_length(length, SHARED, IN_BLOCK.start);
    let b = made.unwrap().expect("a buffer in the block");
    write_at(b.contents().addr(), 0..length, value);
    b
}

/// The pipeline of `inc_k` ([`INC`]) on `d`.
fn inc_on<P: Platform>(d: &P::Device) -> P::ComputePipeline {
    let library = d.new_library(INC).unwrap().unwrap().unwrap();
    let f = library.new_function(OnFault::Return, "inc_k").unwrap();
    d.new_compute_pipeline_state_for_function(f.as_ref())
        .unwrap()
        .unwrap()
}

/// Commits on `q` a command buffer of one compute encoder that binds `c` at index 0 and runs one
/// thread of `inc` on it, and waits for it.
fn count_on<P: Platform>(q: &P::CommandQueue, inc: &P::ComputePipeline, c: &P::Buffer) {
    let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
    let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
    let buffers = [Some(c)];
    for cmd in [
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
    ] {
        e.encode(OnFault::Return, cmd).unwrap();
    }
    cb.commit(OnFault::Return).unwrap();
    cb.wait_until_completed(OnFault::Return).unwrap();
}

/// Commits on `q` a command buffer of one blit encoder that copies the bytes of `from`, a buffer
/// of [`IN_BLOCK`]'s length, into `to`, and waits for it.
fn read_on<P: Platform>(q: &P::CommandQueue, from: &P::Buffer, to: &P::Buffer) {
    let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
    let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
    let copy = BlitCmd::CopyBufferToBuffer {
        source_buffer: Some(from),
        source_offset: 0,
        destination_buffer: Some(to),
        destination_offset: 0,
        size: IN_BLOCK.end - IN_BLOCK.start,
    };
    for cmd in [copy, BlitCmd::EndEncoding] {
        e.encode(OnFault::Return, cmd).unwrap();
    }
    cb.commit(OnFault::Return).unwrap();
    cb.wait_until_completed(OnFault::Return).unwrap();
}

/// The place in `records` of the `n`th buffer made in a block (`m2v.newBufferInBlock`), from 0.
fn made_in_block(records: &[Record], n: usize) -> usize {
    place(records, "m2v.newBufferInBlock:offset:length:options:", n)
        .unwrap_or_else(|| panic!("buffer {n} made in a block: {records:?}"))
}

/// The bytes of the block of stream id `block` that a player holds once it played
/// `records[..end]`: the block's `m2v.write` records among them, in order, over the zeros a block
/// is made of.
fn played(records: &[Record], block: u64, end: usize) -> Vec<u8> {
    let mut held = vec![0u8; BLOCK as usize];
    for i in 0..end {
        for (at, data) in writes_of(records, block, i..i + 1) {
            let at = at as usize;
            held[at..at + data.len()].copy_from_slice(&data);
        }
    }
    held
}

/// [`IN_BLOCK`]'s bytes of `held`, a block's.
fn in_block(held: &[u8]) -> &[u8] {
    &held[IN_BLOCK.start as usize..IN_BLOCK.end as usize]
}

/// A CPU write into a `Shared` buffer made in a block, which a compute encoder binds, is in the
/// stream before the commit, as an `m2v.write` of the block between the buffer's make and the
/// commit: a player holds the CPU's bytes there as the kernel runs. On `d`, a device of `P`, the
/// stream in `streams`. The device's `hasUnifiedMemory` answer goes to stderr, a line the parent
/// passes on ([`in_recording_child`]).
fn kernel_reads_a_block_buffer_the_cpu_wrote<P: Platform>(d: &P::Device, streams: &Path) {
    let unified = d.identity_caps().flags.has_unified_memory;
    eprintln!("recorder-test: hasUnifiedMemory {unified:?}");
    let (heap, block) = block_on::<P>(d);
    let b = in_block_with::<P>(&heap, 0x5a);
    let q = d.new_command_queue().expect("a queue");
    count_on::<P>(&q, &inc_on::<P>(d), &b);

    let records = records_once(streams, |rs| place(rs, "commit", 0).is_some());
    let made = made_in_block(&records, 0);
    let commit = place(&records, "commit", 0).expect("the commit's record");
    let written = writes_of(&records, block, made..commit);
    assert!(
        !written.is_empty(),
        "no write of the block between the buffer's make at {made} and its commit at {commit}"
    );
    let length = (IN_BLOCK.end - IN_BLOCK.start) as usize;
    assert_eq!(
        in_block(&played(&records, block, commit)),
        &bytes(0x5a, length)[..],
        "the stream's bytes of the buffer at its commit"
    );
}

/// The rt platform's arm of [`kernel_reads_a_block_buffer_the_cpu_wrote`], on its default
/// device: its commits write what they reach.
#[test]
fn on_vulkan_rt_a_cpu_write_into_a_block_buffer_a_kernel_binds_is_written_before_its_commit() {
    let test = "recorder::memory_tests::\
                on_vulkan_rt_a_cpu_write_into_a_block_buffer_a_kernel_binds_is_written_before_its_commit";
    in_recording_child(test, |streams| {
        let d = rt_device();
        kernel_reads_a_block_buffer_the_cpu_wrote::<RtPlatform>(&d, streams);
    });
}

/// A buffer made in block space an earlier buffer left, which the CPU writes with the very bytes
/// the earlier one left there and a compute encoder then binds: no write of the block comes
/// between its make and its commit, as the bytes did not change, and a player that plays the
/// stream holds the CPU's bytes there at the commit, those the earlier buffer's write recorded.
/// So a buffer read before any write of its own, in a stream, can be a buffer whose bytes the
/// block's earlier writes hold. On `d`, a device of `P`, the stream in `streams`.
fn a_reused_block_space_holds_the_equal_bytes<P: Platform>(d: &P::Device, streams: &Path) {
    let (heap, block) = block_on::<P>(d);
    let q = d.new_command_queue().expect("a queue");
    let to = d.new_buffer_of_length(OnFault::Return, 4096, SHARED);
    let to = to.unwrap().expect("a Shared buffer");
    let earlier = in_block_with::<P>(&heap, 0x5a);
    read_on::<P>(&q, &earlier, &to);
    drop(earlier);
    let later = in_block_with::<P>(&heap, 0x5a);
    count_on::<P>(&q, &inc_on::<P>(d), &later);

    let records = records_once(streams, |rs| place(rs, "commit", 1).is_some());
    let made = [0, 1].map(|n| made_in_block(&records, n));
    let commits = [0, 1].map(|n| place(&records, "commit", n).expect("a commit's record"));
    assert!(
        made[0] < commits[0] && commits[0] < made[1] && made[1] < commits[1],
        "the makes at {made:?}, the commits at {commits:?}"
    );
    let earlier_writes = writes_of(&records, block, made[0]..commits[0]);
    assert!(
        !earlier_writes.is_empty(),
        "no write of the block before the earlier buffer's commit"
    );
    let later_writes = writes_of(&records, block, made[1]..commits[1]);
    assert!(
        later_writes.is_empty(),
        "a write of the block between the later buffer's make and its commit: {:?}",
        later_writes.keys().collect::<Vec<_>>()
    );
    let length = (IN_BLOCK.end - IN_BLOCK.start) as usize;
    assert_eq!(
        in_block(&played(&records, block, commits[1])),
        &bytes(0x5a, length)[..],
        "the stream's bytes of the later buffer at its commit"
    );
}

/// The rt platform's arm of [`a_reused_block_space_holds_the_equal_bytes`], on its default
/// device.
#[test]
fn on_vulkan_rt_a_buffer_in_reused_block_space_holds_the_equal_bytes_an_earlier_write_recorded() {
    let test = "recorder::memory_tests::\
                on_vulkan_rt_a_buffer_in_reused_block_space_holds_the_equal_bytes_an_earlier_write_recorded";
    in_recording_child(test, |streams| {
        let d = rt_device();
        a_reused_block_space_holds_the_equal_bytes::<RtPlatform>(&d, streams);
    });
}

/// The rt platform's default device, on the GPU the test holds.
fn rt_device() -> <RtPlatform as Platform>::Device {
    let platform = RtPlatform::new();
    let registry_id = platform
        .system_default_device()
        .expect("an rt device on the GPU the test holds")
        .facts()
        .registry_id;
    platform
        .open_device(registry_id)
        .expect("the default device")
}
