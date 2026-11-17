//! Private buffers around dispatches: the bytes a dispatch of a compute pipeline made with linked
//! functions found and left in each Private buffer range it binds, which no CPU can read and
//! [`super::before_commit`]'s writes therefore never hold, so that two recordings of the same
//! dispatch (one form against another, or one host against another) can be compared where its
//! bytes before match.
//!
//! - **Which dispatches.** Each recorded dispatch (`dispatchThreadgroups:…`, `dispatchThreads:…`
//!   and their indirect forms) whose pipeline state, as the guest set it, was made with linked
//!   functions ([`crate::gpu::resource::ComputePipelineObject::links_functions`]), outside a block
//!   of command predication ([`predication::bound_buffers`]). Only those: a snapshot splits the
//!   compute encoder twice and copies every bound Private byte twice, which taken for every
//!   dispatch that binds a Private buffer would cost a recording more than the stream itself.
//! - **Which bytes.** Each buffer index the guest bound a Private buffer to, from its offset to
//!   the buffer's end, each buffer and offset once; at most [`MOST`] bytes a dispatch, else the
//!   dispatch is not snapshotted. A buffer the stream never made is left out.
//! - **How.** Before the dispatch, the compute encoder's runtime encoder ends, a blit encoder of
//!   the command buffer's copies each range into the first half of a Shared buffer of the
//!   recorder's own, and a fresh compute encoder takes the guest's bindings and residency again
//!   ([`predication::blit_between`]); after the dispatch, the same into the second half. At the
//!   command buffer's completion, a handler copies the Shared buffer out and queues it, and the
//!   writer writes each range's two byte runs into the `blobs` directory and the record.
//! - **The record**, after the command buffer's completion, so later in the stream than the
//!   dispatch: `m2v.privateBytes`, its receiver the buffer (its stream id, which a `release`
//!   may already have ended), its arguments `[dispatch, {location, length}, before, after]`:
//!   the dispatch's position in the stream (how many call records came before it, from 0, as a
//!   player counts the calls it played), the range of the buffer's bytes, and the bytes before
//!   and after the dispatch as blobs (tag 11) by the SHA-256 of their bytes, which lie in the
//!   blob directory alone, as a library's do. A command buffer that failed writes none.
//!
//! Nothing here runs where the process does not record: the encoder's hook reads its recorded
//! call, which is `None` there.

use std::sync::atomic::Ordering;

use metal2vulkan_runtime as rt;

use super::{Active, Call, Item, State, Value, id_at, state, typed};
use crate::core::platform::Handle as _;
use crate::core::platform::compute::ComputeCmd;
use crate::core::platform::types::{CommandBufferStatus, NsRange};
use crate::gpu::encode_compute::VulkanComputeEncoder;
use crate::gpu::predication;
use crate::gpu::resource::VulkanBuffer;
use crate::platform::VulkanPlatform;

/// The record's selector.
pub const SELECTOR: &str = "m2v.privateBytes";

/// The most bytes of Private ranges one dispatch's snapshot copies each way.
pub const MOST: u64 = 256 << 20;

/// One Private buffer range of a snapshot: its buffer's stream id, its offset and length in the
/// buffer, and where its bytes lie in each half of the staging buffer.
#[derive(Debug, Clone, Copy)]
pub struct Range {
    buffer: u64,
    offset: u64,
    length: u64,
    at: u64,
}

/// A dispatch's snapshot between the blit before it and the blit after it.
pub struct Snapshot {
    sel: &'static str,
    /// The buffers, held until the blit after the dispatch is encoded.
    buffers: Vec<VulkanBuffer>,
    ranges: Vec<Range>,
    /// The recorder's Shared buffer: the bytes before, then the bytes after.
    staging: rt::Buffer,
    /// The bytes of each half.
    total: u64,
}

/// A snapshot's bytes at its command buffer's completion, for the writer; `None` where the
/// command buffer failed.
pub struct Read {
    snap: u64,
    ranges: Vec<Range>,
    total: u64,
    bytes: Option<Vec<u8>>,
}

/// Whether `cmd` dispatches a kernel.
fn is_dispatch(cmd: &ComputeCmd<'_, VulkanPlatform>) -> bool {
    matches!(
        cmd,
        ComputeCmd::DispatchThreadgroups { .. }
            | ComputeCmd::DispatchThreadgroupsIndirect { .. }
            | ComputeCmd::DispatchThreads { .. }
            | ComputeCmd::DispatchThreadsIndirect { .. }
    )
}

/// Before `cmd` of `enc`, the call `active` records: where it is a dispatch to snapshot, the blit
/// of its Private ranges' bytes before it, and the snapshot.
#[inline]
pub fn before(
    active: &Option<Active>,
    enc: &VulkanComputeEncoder,
    cmd: &ComputeCmd<'_, VulkanPlatform>,
) -> Option<Snapshot> {
    let a = active.as_ref()?;
    if !a.recorded() || !is_dispatch(cmd) {
        return None;
    }
    snapshot(enc, cmd.selector())
}

fn snapshot(enc: &VulkanComputeEncoder, sel: &'static str) -> Option<Snapshot> {
    let state = state()?;
    let bound = predication::bound_buffers(
        enc,
        |p| p.links_functions,
        |b| matches!(b.raw.storage_mode(), rt::StorageMode::Private),
    )?;
    let (mut buffers, mut ranges, mut total) = (Vec::new(), Vec::<Range>::new(), 0);
    for (b, offset) in bound {
        let length = b.raw.length().saturating_sub(offset);
        if length == 0 {
            continue;
        }
        let Some(buffer) = id_at(b.id().0) else {
            skipped(state);
            continue;
        };
        if ranges
            .iter()
            .any(|r| r.buffer == buffer && r.offset == offset)
        {
            continue;
        }
        ranges.push(Range {
            buffer,
            offset,
            length,
            at: total,
        });
        total += length;
        buffers.push(b);
    }
    let device = &buffers.first()?.device.raw;
    if total > MOST {
        skipped(state);
        return None;
    }
    let options = crate::gpu::resource_options(sel, 0);
    let Ok(staging) = crate::gpu::own_buffer(device, 2 * total, options, rt::BufferContents::Zero)
    else {
        skipped(state);
        return None;
    };
    copy(enc, sel, &buffers, &ranges, &staging, 0);
    Some(Snapshot {
        sel,
        buffers,
        ranges,
        staging,
        total,
    })
}

/// Encodes the copies of `ranges`, of `buffers`, into `staging` from `half`, between `enc`'s
/// commands.
fn copy(
    enc: &VulkanComputeEncoder,
    sel: &'static str,
    buffers: &[VulkanBuffer],
    ranges: &[Range],
    staging: &rt::Buffer,
    half: u64,
) {
    predication::blit_between(enc, sel, |blit| {
        for (b, r) in buffers.iter().zip(ranges) {
            blit.copy_from_buffer_to_buffer(&b.raw, r.offset, staging, half + r.at, r.length)?;
        }
        Ok(())
    });
}

/// After the dispatch `snapshot` was taken before, which the encoder `sent` (else it went
/// nowhere): the blit of its ranges' bytes after it, and the handler that reads both at the
/// command buffer's completion; the call `active` records names the snapshot, for its records.
#[inline]
pub fn after(
    active: &mut Option<Active>,
    enc: &VulkanComputeEncoder,
    snapshot: Option<Snapshot>,
    sent: bool,
) {
    let Some(snapshot) = snapshot else { return };
    if let Some(a) = active.as_mut()
        && sent
    {
        a.snap = taken(enc, snapshot);
    }
}

fn taken(enc: &VulkanComputeEncoder, snapshot: Snapshot) -> Option<u64> {
    let state = state()?;
    let Snapshot {
        sel,
        buffers,
        ranges,
        staging,
        total,
    } = snapshot;
    copy(enc, sel, &buffers, &ranges, &staging, total);
    drop(buffers);
    let Some(cb) = enc.command_buffer() else {
        skipped(state);
        return None;
    };
    let snap = state.snaps.fetch_add(1, Ordering::Relaxed) + 1;
    let handler: rt::Handler = Box::new(move |done: &rt::CommandBuffer| {
        let completed = done.status().raw() == CommandBufferStatus::COMPLETED.raw();
        let bytes = completed.then(|| read(&staging, 2 * total));
        drop(staging);
        state.push(Item::Private(Read {
            snap,
            ranges,
            total,
            bytes,
        }));
    });
    if cb.raw.add_completed_handler(handler).is_err() {
        skipped(state);
        return None;
    }
    Some(snap)
}

/// The first `n` bytes of `staging`, a Shared buffer the GPU's work completed.
fn read(staging: &rt::Buffer, n: u64) -> Vec<u8> {
    let Some(base) = staging.contents() else {
        return Vec::new();
    };
    let n = usize::try_from(n.min(staging.length())).unwrap_or(0);
    let mut out = vec![0u8; n];
    // SAFETY: `base` is the CPU address of the buffer's bytes, live while it lives, and `n` is at
    // most its length; the command buffer that wrote them completed, and nothing else writes them.
    unsafe { std::ptr::copy_nonoverlapping(base.as_ptr(), out.as_mut_ptr(), n) };
    out
}

fn skipped(state: &State) {
    state.stats.private_skipped.fetch_add(1, Ordering::Relaxed);
}

impl State {
    /// The `m2v.privateBytes` records of a snapshot read at its command buffer's completion:
    /// their call records' length, and how many.
    pub(super) fn private_records(&self, read: Read) -> (usize, u64) {
        let Read {
            snap,
            ranges,
            total,
            bytes,
        } = read;
        let dispatch = self.dispatched.lock().remove(&snap);
        let (Some(dispatch), Some(bytes)) = (dispatch, bytes) else {
            skipped(self);
            return (0, 0);
        };
        let (mut written, mut records) = (0, 0);
        for r in ranges {
            let part = |from: u64| {
                let start = usize::try_from(from + r.at).ok()?;
                let end = start.checked_add(usize::try_from(r.length).ok()?)?;
                bytes.get(start..end)
            };
            let (Some(before), Some(after)) = (part(0), part(total)) else {
                skipped(self);
                continue;
            };
            let range = typed::ns_range(NsRange {
                location: r.offset,
                length: r.length,
            });
            let (before, after) = (self.blob_file(before), self.blob_file(after));
            self.count_selector(SELECTOR);
            written += self.emit(Call {
                selector: SELECTOR.to_owned(),
                receiver: r.buffer,
                args: vec![
                    Value::Uint(dispatch),
                    range,
                    Value::Blob(before),
                    Value::Blob(after),
                ],
                ..Call::default()
            });
            records += 1;
            let stats = &self.stats;
            stats.private_ranges.fetch_add(1, Ordering::Relaxed);
            (stats.private_bytes).fetch_add(2 * r.length, Ordering::Relaxed);
        }
        (written, records)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::Duration;

    use super::*;
    use crate::core::platform::blit::{BlitCmd, BlitEncoder as _};
    use crate::core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use crate::core::platform::compute::ComputeEncoder as _;
    use crate::core::platform::device::GpuDevice as _;
    use crate::core::platform::resource::{Buffer as _, Library as _};
    use crate::core::platform::types::MtlSize;
    use crate::core::platform::values::{
        DynamicLibrariesDesc, FullComputePipelineDesc, LinkedFunctionsDesc,
    };
    use crate::core::raise::OnFault;
    use crate::gpu::command::VulkanCommandBuffer;
    use crate::gpu::device::tests::dev;
    use crate::gpu::resource::tests::shared;
    use crate::gpu::resource::{VulkanComputePipeline, VulkanLibrary};
    use crate::recorder::format;

    /// The marker that makes a run of [`a_recorded_dispatch_snapshots_its_private_ranges`] the
    /// child, which records.
    const CHILD: &str = "REIMS_VGPU_VULKAN_PRIVATE_CHILD";

    /// The words the Private buffer holds before the dispatch.
    const WORDS: [u32; 4] = [1, 2, 3, 4];

    /// The byte the dispatch binds the Private buffer from.
    const OFFSET: u64 = 4;

    /// `linked_caller.metal`'s kernel, which adds one to the word at its buffer 0 through
    /// `linked_callee.metal`'s `linked_add_one`, as a pipeline made with that function linked.
    fn linked_pipeline() -> VulkanComputePipeline {
        let d = dev();
        let library =
            |bytes: &[u8]| -> VulkanLibrary { d.new_library(bytes).unwrap().unwrap().unwrap() };
        let caller = library(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/linked_caller.metallib"
        )));
        let callee = library(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/linked_callee.metallib"
        )));
        let kernel = (caller.new_function(OnFault::Return, "linked_kernel"))
            .unwrap()
            .unwrap();
        let add_one = (callee.new_function(OnFault::Return, "linked_add_one"))
            .unwrap()
            .unwrap();
        let desc = FullComputePipelineDesc {
            label: None,
            function: &kernel,
            threadgroup_size_is_multiple_of_execution_width: false,
            max_total_threads_per_threadgroup: 0,
            max_call_stack_depth: 1,
            support_indirect_command_buffers: false,
            stage_input: None,
            linked: LinkedFunctionsDesc {
                functions: vec![add_one],
                ..LinkedFunctionsDesc::default()
            },
            libraries: DynamicLibrariesDesc::default(),
            open_cl_mode: false,
        };
        let pipeline = d.new_compute_pipeline_state_full(&desc).unwrap().unwrap();
        assert!(pipeline.links_functions);
        pipeline
    }

    /// Runs `cmds` in a blit encoder of a command buffer of its own, to completion.
    fn blit(cmds: Vec<BlitCmd<'_, VulkanPlatform>>) {
        let q = dev().new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
        let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
        for cmd in cmds {
            e.encode(OnFault::Return, cmd).unwrap();
        }
        e.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        assert_eq!(cb.error(), None);
    }

    /// A Private buffer holding [`WORDS`], copied in from a Shared one.
    fn private_holding_words() -> VulkanBuffer {
        let source = shared(16);
        // SAFETY: the Shared buffer's 16 bytes, written before any command uses them.
        unsafe {
            std::ptr::copy_nonoverlapping(WORDS.as_ptr(), source.contents().as_ptr::<u32>(), 4);
        }
        let private = (dev().new_buffer_of_length(OnFault::Return, 16, 0x20))
            .unwrap()
            .unwrap();
        blit(vec![BlitCmd::CopyBufferToBuffer {
            source_buffer: Some(&source),
            source_offset: 0,
            destination_buffer: Some(&private),
            destination_offset: 0,
            size: 16,
        }]);
        private
    }

    /// The words of `private`, a Private buffer of 16 bytes, copied out into a Shared one.
    fn words_of(private: &VulkanBuffer) -> [u32; 4] {
        let read = shared(16);
        blit(vec![BlitCmd::CopyBufferToBuffer {
            source_buffer: Some(private),
            source_offset: 0,
            destination_buffer: Some(&read),
            destination_offset: 0,
            size: 16,
        }]);
        let mut out = [0u32; 4];
        // SAFETY: the Shared buffer's 16 bytes, read after the copy into it completed.
        unsafe {
            std::ptr::copy_nonoverlapping(read.contents().as_ptr::<u32>(), out.as_mut_ptr(), 4)
        };
        out
    }

    /// One thread of `pipeline` over `private` bound from [`OFFSET`], in a compute encoder of a
    /// command buffer of its own, run to completion: the command buffer.
    fn dispatch(pipeline: &VulkanComputePipeline, private: &VulkanBuffer) -> VulkanCommandBuffer {
        let q = dev().new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
        let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
        let one = MtlSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        for cmd in [
            ComputeCmd::SetComputePipelineState {
                state: Some(pipeline),
            },
            ComputeCmd::SetBuffers {
                buffers: &[Some(private)],
                offsets: &[OFFSET],
                range: NsRange::new(0, 1),
            },
            ComputeCmd::DispatchThreads {
                threads_per_grid: one,
                threads_per_threadgroup: one,
            },
            ComputeCmd::EndEncoding,
        ] {
            e.encode(OnFault::Return, cmd).unwrap();
        }
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        assert_eq!(cb.error(), None);
        cb
    }

    /// The little-endian bytes of `words`.
    fn bytes(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// With the process not recording, a dispatch of a pipeline made with linked functions over a
    /// Private buffer is encoded in its compute encoder's one runtime encoder: no blit around it.
    #[test]
    fn an_unrecorded_dispatch_encodes_no_blit() {
        assert!(!super::super::on());
        let private = private_holding_words();
        let cb = dispatch(&linked_pipeline(), &private);
        assert_eq!(cb.encoders_made(), 1);
        assert_eq!(words_of(&private), [1, 3, 3, 4]);
    }

    /// A recorded dispatch of a pipeline made with linked functions over a Private buffer bound
    /// from byte 4 splits its compute encoder around two blits, and once its command buffer
    /// completed the stream holds one `m2v.privateBytes` record of the buffer: the dispatch's
    /// position, the bound range, and the bytes the test put there and the bytes the kernel left,
    /// as blobs in the blob directory. The recorder is the process's, so the test runs in a
    /// child of its own, recording into a scratch directory.
    #[test]
    fn a_recorded_dispatch_snapshots_its_private_ranges() {
        let Some(streams) = std::env::var_os(CHILD).map(|_| {
            PathBuf::from(std::env::var_os(super::super::ENV_VAR).expect("the child records"))
        }) else {
            let dir =
                std::env::temp_dir().join(format!("rvg-recorder-private-{}", std::process::id()));
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "recorder::private::tests::a_recorded_dispatch_snapshots_its_private_ranges",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(CHILD, "1")
                .env(super::super::ENV_VAR, dir.join("streams"))
                .output()
                .unwrap();
            let _ = std::fs::remove_dir_all(&dir);
            assert!(out.status.success(), "{out:?}");
            let ran = String::from_utf8_lossy(&out.stdout).contains("1 passed");
            assert!(ran, "the child ran no test: {out:?}");
            return;
        };
        super::super::start_from_env();
        assert!(super::super::on());
        let private = private_holding_words();
        let cb = dispatch(&linked_pipeline(), &private);
        // The compute encoder, the blit before and the encoder after it, the blit after the
        // dispatch and the encoder after that.
        assert_eq!(cb.encoders_made(), 5);
        assert_eq!(words_of(&private), [1, 3, 3, 4]);
        let buffer = id_at(private.id().0).expect("the buffer has its stream id");
        let segment = streams.join(format!("{}.0000.m2vr", std::process::id()));
        let mut found = None;
        for _ in 0..200 {
            let read = std::fs::read(&segment)
                .ok()
                .and_then(|b| format::read(&b, &mut Vec::new()).ok());
            let calls: Vec<Call> = read
                .into_iter()
                .flat_map(|r| r.records)
                .filter_map(|r| match r {
                    format::Record::Call(c) => Some(c),
                    _ => None,
                })
                .collect();
            if calls.iter().any(|c| c.selector == SELECTOR) {
                found = Some(calls);
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let calls = found.expect("the stream holds the snapshot's record within 10 s");
        let records: Vec<&Call> = calls.iter().filter(|c| c.selector == SELECTOR).collect();
        assert_eq!(records.len(), 1, "{records:?}");
        let record = records[0];
        assert_eq!(record.receiver, buffer);
        let Some(Value::Uint(at)) = record.args.first() else {
            panic!("no dispatch position: {record:?}");
        };
        let dispatched = &calls[usize::try_from(*at).unwrap()];
        assert_eq!(
            dispatched.selector,
            "dispatchThreads:threadsPerThreadgroup:"
        );
        assert_eq!(
            record.args.get(1),
            Some(&typed::ns_range(NsRange::new(OFFSET, 12)))
        );
        let blobs = streams.parent().unwrap().join("blobs");
        let blob = |v: Option<&Value>| match v {
            Some(Value::Blob(hash)) => std::fs::read(blobs.join(super::super::hex(hash))).unwrap(),
            other => panic!("not a blob: {other:?}"),
        };
        assert_eq!(blob(record.args.get(2)), bytes(&[2, 3, 4]));
        assert_eq!(blob(record.args.get(3)), bytes(&[3, 3, 4]));
    }
}
