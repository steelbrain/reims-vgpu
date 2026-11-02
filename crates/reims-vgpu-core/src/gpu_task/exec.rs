//! The execs and infos ([`GpuTask::exec`], [`GpuTask::info`]), their completions, and the
//! queries that read the task's objects and write a reply into guest memory (compute info,
//! heap texture size) or reset one (rate maps).
//!
//! Each runs as a job. Its catch sites log and reply 2; any other raise ends the task.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use super::paging::CompleteSync;
use super::service::{Access, GpuTask};
use super::{DeathKind, RaiseKind, error_text, exception, raise_caught, unraised};
use crate::fatal;
use crate::fifo::wire::Records;
use crate::guest::GuestPtr;
use crate::guest::records::{APV_PAGE_MASK, CmdExecIndirectCmdBuffer, CmdExecIndirectResource3};
use crate::memory::cursor::Cursor;
use crate::num::{ToU64, ToUsize, Truncate};
use crate::platform::command::{CommandBuffer, CommandQueue, CompletedHandler, CompletedInfo};
use crate::platform::event::{EventRef, SharedEvent};
use crate::platform::resource::ComputePipeline;
use crate::platform::types::CommandBufferStatus;
use crate::platform::{Handle, Platform};
use crate::raise::{OnFault, RaiseResult};
use crate::resources::manager::states::ComputePipelineStates;
use crate::resources::manager::states::RasterizationRateMaps;
use crate::rt::{Held, Hold};
use crate::tasks::wire::Reply;

/// How long an info waits for the previous exec to be scheduled.
const INFO_WAIT_MS: u64 = 5000;
/// The object type a rate-map reset reads back from the object list.
const RATE_MAP_OBJECT_TYPE: u8 = 7;
/// The compute-info keys.
const INFO_KEY_MAX_TOTAL_THREADS: u32 = 1;
const INFO_KEY_THREAD_EXECUTION_WIDTH: u32 = 3;
const INFO_KEY_STATIC_THREADGROUP_MEMORY: u32 = 4;
/// One compute-info entry: a `u32` key, then a `u32` value.
const COMPUTE_INFO_ENTRY_SIZE: u64 = 8;
/// The heap texture reply's bounds: 16 bytes to a guest page less one.
const HEAP_REPLY_MIN: u64 = 16;
const HEAP_REPLY_MAX: u64 = APV_PAGE_MASK;
/// The record sizes of an exec's payload.
const RESOURCE3_SIZE: u64 = crate::num::const_u64(size_of::<CmdExecIndirectResource3>());
const CMD_BUFFER_SIZE: u64 = crate::num::const_u64(size_of::<CmdExecIndirectCmdBuffer>());

/// A FIFO's event and the value last signalled on it.
pub(super) struct FifoInfo<P: Platform> {
    /// The event the device made from the host's handle, taken unchecked: `None` for none.
    pub(super) event: Option<P::SharedEvent>,
    /// The value the last exec will signal; `u64::MAX` before any.
    pub(super) signaled: Arc<AtomicU64>,
}

impl<P: Platform> FifoInfo<P> {
    /// A FIFO with `event` and nothing signalled yet.
    pub(super) fn new(event: Option<P::SharedEvent>) -> FifoInfo<P> {
        FifoInfo {
            event,
            signaled: Arc::new(AtomicU64::new(u64::MAX)),
        }
    }
}

/// Stores one compute-info entry `{key, value}` at `at` with one paired store, so a vCPU
/// reading the entry with one 64-bit load sees the old pair or the new one, never the new key
/// beside the old value. The pair is one single-copy-atomic access inside a 16-byte granule
/// (FEAT_LSE2), and one per register across one.
fn store_compute_info_pair(entries: GuestPtr, at: u64, key: u32, value: u32) {
    entries.write_u32_pair(at, key, value, DeathKind::NullReply);
}

impl<P: Platform> GpuTask<P> {
    /// FIFO `stamp_index`'s event (another reference) and value cell, if it is set.
    fn fifo(&self, stamp_index: u32) -> Option<(Option<P::SharedEvent>, Arc<AtomicU64>)> {
        let fifos = self.fifos.lock();
        let f = fifos.get(&stamp_index)?;
        Some((
            f.event.as_ref().map(Handle::clone_ref),
            Arc::clone(&f.signaled),
        ))
    }

    /// Whether an exec's records need more bytes than `data` has.
    fn command_too_big(data: &[u8], resource_count: u32, cmd_buffer_count: u32) -> bool {
        u64::from(resource_count)
            .wrapping_mul(RESOURCE3_SIZE)
            .wrapping_add(u64::from(cmd_buffer_count).wrapping_mul(CMD_BUFFER_SIZE))
            > data.len().to_u64()
    }

    /// An exec's resource records, then its command-buffer records, which follow them. The size
    /// check before has made both fit.
    fn exec_records(
        data: &[u8],
        resource_count: u32,
        cmd_buffer_count: u32,
    ) -> (
        Records<'_, CmdExecIndirectResource3>,
        Records<'_, CmdExecIndirectCmdBuffer>,
    ) {
        let resources = super::service::records(data, resource_count);
        let skip = u64::from(resource_count)
            .wrapping_mul(RESOURCE3_SIZE)
            .to_usize();
        let rest = data.get(skip..).unwrap_or_default();
        (resources, super::service::records(rest, cmd_buffer_count))
    }

    /// One cursor per command buffer, in order.
    fn cursors(
        &self,
        hold: Hold<'_>,
        cmd_buffers: Records<'_, CmdExecIndirectCmdBuffer>,
    ) -> RaiseResult<Vec<Option<Cursor>>> {
        let mut cursors = Vec::new();
        for cmd_buffer in cmd_buffers.iter() {
            cursors.push(self.cursor(hold, cmd_buffer.offset, cmd_buffer.length, false)?);
        }
        Ok(cursors)
    }

    /// Runs the command buffers `data` names on the GPU, as a job: on the calling thread when
    /// `here` ([`GpuTask::job_on`]); replies `(status, shared-texture backing ids, their count)`.
    #[expect(clippy::too_many_arguments, reason = "the exec message's fields")]
    pub(super) fn exec(
        &self,
        here: bool,
        data: Vec<u8>,
        cmd_buffer_count: u32,
        resource_count: u32,
        stamp_index: u32,
        fifo_event_value_in: u64,
        gpu_completion: u64,
        reply: Reply<(u32, Option<Vec<u8>>, u32)>,
    ) {
        self.job_on(here, move |this, hold| {
            this.exec_job(
                hold,
                &data,
                cmd_buffer_count,
                resource_count,
                stamp_index,
                fifo_event_value_in,
                gpu_completion,
                &reply,
            )
        });
    }

    /// The exec, under the shared task lock: the object lists mapped (2 when they cannot be),
    /// the payload's size and the FIFO checked (2), then, under a handler that replies 2: the
    /// resources paged in, one command buffer that waits on the paging event (when a paging
    /// buffer was committed) or on the FIFO's event (when `fifo_event_value_in` is neither 0 nor
    /// the last signalled value), the decode, the completion, the FIFO's next value stored, the
    /// scheduled signal and the commit, then paging-out behind it. Replies 1 with the backing
    /// ids.
    ///
    /// The FIFO is signalled when the buffer is scheduled, not completed: the signal is encoded
    /// with `encodeSignalEventScheduled:value:`, after the completed handler is added and before
    /// the commit (`tests::exec_commits_then_completes`). A raise in paging-out comes after the
    /// commit; the late completion then finds its token gone on the host.
    #[expect(clippy::too_many_arguments, reason = "the exec message's fields")]
    fn exec_job(
        self: &Arc<Self>,
        hold: Hold<'_>,
        data: &[u8],
        cmd_buffer_count: u32,
        resource_count: u32,
        stamp_index: u32,
        fifo_event_value_in: u64,
        gpu_completion: u64,
        reply: &Reply<(u32, Option<Vec<u8>>, u32)>,
    ) {
        let _lock = self.task_lock(Access::Shared);
        if let Err(t) = self.ensure_object_list_mapped(hold) {
            log::warn!("task {}: exec refused: {}", self.id(), exception(&t));
            reply.call((2, None, 0));
            hold.land(t);
            return;
        }
        if Self::command_too_big(data, resource_count, cmd_buffer_count) {
            log::warn!(
                "task {}: exec refused: {resource_count} resources and {cmd_buffer_count} \
                 command buffers do not fit its {} bytes",
                self.id(),
                data.len()
            );
            reply.call((2, None, 0));
            return;
        }
        let (resources, cmd_buffers) = Self::exec_records(data, resource_count, cmd_buffer_count);
        let Some((event, signaled)) = self.fifo(stamp_index) else {
            log::warn!(
                "task {}: exec refused: FIFO {stamp_index} has no event",
                self.id()
            );
            reply.call((2, None, 0));
            return;
        };
        let wait_value = fifo_event_value_in;
        let signaled_value = signaled.load(Ordering::SeqCst);
        let mut shared_texture_backing_ids: Vec<u32> = Vec::new();
        let executed = (|| -> RaiseResult<_> {
            let prepare_event = if wait_value == 0 || wait_value == signaled_value {
                None
            } else {
                event.as_ref()
            };
            let sync = self.prepare_resources(
                hold,
                resources,
                prepare_event,
                fifo_event_value_in,
                &mut shared_texture_backing_ids,
            )?;
            let command_queue = self.command_queue.lock().as_ref().map(Handle::clone_ref);
            let command_buffer = match &command_queue {
                Some(q) => q.command_buffer(OnFault::Return)?,
                None => None,
            };
            if sync.needs_sync {
                let paging_event = &self.core().paging_event;
                if paging_event.is_none() {
                    fatal!(DeathKind::SyncWithoutPagingEvent)
                }
                if let Some(b) = &command_buffer {
                    b.encode_wait(paging_event.as_ref().map(EventRef::Event), sync.value)?;
                }
            } else if wait_value != 0
                && wait_value != signaled_value
                && let Some(b) = &command_buffer
            {
                b.encode_wait(event.as_ref().map(EventRef::Shared), fifo_event_value_in)?;
            }
            // Live cursors over guest RAM
            let mut cursors = self.cursors(hold, cmd_buffers)?;
            let (deserializer, _, _) = self.parts();
            // Held past the commit, to the end of the exec (`Decoders`).
            let _decoders =
                deserializer.decode_segments(hold, &mut cursors, command_buffer.as_ref())?;
            let this = Arc::clone(self);
            let needs_complete = sync.needs_complete;
            // The completion runs as the task's code, and not at all once the task has ended.
            let completed: CompletedHandler = self.carry(move |buffer: CompletedInfo| {
                this.exec_completed(buffer, needs_complete, gpu_completion);
            });
            if let Some(b) = &command_buffer {
                b.add_completed_handler(OnFault::Return, completed)?;
            }
            let complete_event = event.as_ref().map(Handle::clone_ref);
            let complete_value = fifo_event_value_in.wrapping_add(1);
            signaled.store(complete_value, Ordering::SeqCst);
            if let Some(b) = &command_buffer {
                b.encode_signal_when_scheduled(complete_event.as_ref(), complete_value)?;
            }
            if let Some(b) = &command_buffer {
                b.commit(OnFault::Return)?;
            }
            let complete = CompleteSync {
                needs_complete: sync.needs_complete,
                event: complete_event,
                value: complete_value,
            };
            if complete.needs_complete {
                let this = Arc::clone(self);
                let exec_buffer = command_buffer.as_ref().map(Handle::clone_ref);
                let synced: CompletedHandler = self.carry(move |sync_buffer: CompletedInfo| {
                    this.sync_completed(sync_buffer, exec_buffer, gpu_completion);
                });
                self.complete_resources(hold, resources, complete, synced)?;
            }
            Ok(())
        })();
        if let Err(t) = executed {
            log::warn!("task {}: exec faulted: {}", self.id(), exception(&t));
            reply.call((2, None, 0));
            hold.land(t);
            return;
        }
        let count = shared_texture_backing_ids.len().low_u32();
        let backing_ids = (count != 0).then(|| {
            shared_texture_backing_ids
                .iter()
                .flat_map(|id| id.to_le_bytes())
                .collect::<Vec<u8>>()
        });
        reply.call((1, backing_ids, count));
    }

    /// The exec buffer's completion, on a Metal thread: an error is 3; without paging-out
    /// behind it, it completes the host's token.
    fn exec_completed(&self, buffer: CompletedInfo, needs_complete: bool, gpu_completion: u64) {
        let mut status = 1;
        if buffer.status == CommandBufferStatus::ERROR {
            log::warn!(
                "task {}: an exec's command buffer failed: {}",
                self.id(),
                error_text(buffer.error.as_ref())
            );
            status = 3;
        }
        if !needs_complete {
            self.complete_token(gpu_completion, status, buffer.error);
        }
    }

    /// The paging-out buffer's completion, on a Metal thread: it waits for the exec buffer,
    /// then completes the host's token with 1 when both completed, else 3 with the exec's error
    /// (when it failed) or the paging-out's. A completion handler has no handler and no holder
    /// ([`Hold::NONE`]): a raise in the wait ends the task there.
    pub(super) fn sync_completed(
        &self,
        sync_buffer: CompletedInfo,
        exec_buffer: Option<P::CommandBuffer>,
        gpu_completion: u64,
    ) {
        if let Some(b) = &exec_buffer {
            unraised(b.wait_until_completed(OnFault::Die));
        }
        let exec_status = exec_buffer
            .as_ref()
            .map_or(CommandBufferStatus::default(), |b| b.status());
        if sync_buffer.status == CommandBufferStatus::COMPLETED
            && exec_status == CommandBufferStatus::COMPLETED
        {
            self.complete_token(gpu_completion, 1, None);
        } else if exec_status == CommandBufferStatus::ERROR {
            // Metal is asked for the exec's error on this path only, whether or not the line is
            // written, then again for the token
            let error = exec_buffer.as_ref().and_then(|b| b.error());
            log::warn!(
                "task {}: the exec behind a paging-out failed: {}",
                self.id(),
                error_text(error.as_ref())
            );
            self.complete_token(
                gpu_completion,
                3,
                exec_buffer.as_ref().and_then(|b| b.error()),
            );
        } else {
            self.complete_token(gpu_completion, 3, sync_buffer.error);
        }
    }

    /// Decodes the command buffers `data` names without committing them, as a job, and signals
    /// the FIFO on the CPU. Replies 1, 2 (no FIFO, the object lists, the size, a raise) or 3
    /// (the previous exec was not scheduled in time).
    ///
    /// It waits up to 5 s for the previous exec to be scheduled, holding the shared task lock
    /// and the service queue. Decoded, never committed. The paging-in's backing ids are dropped.
    pub(super) fn info(
        &self,
        data: Vec<u8>,
        cmd_buffer_count: u32,
        resource_count: u32,
        stamp_index: u32,
        fifo_event_value_in: u64,
        reply: Reply<u32>,
    ) {
        self.job(move |this, hold| {
            let _lock = this.task_lock(Access::Shared);
            let Some((event, signaled)) = this.fifo(stamp_index) else {
                log::warn!(
                    "task {}: info refused: FIFO {stamp_index} has no event",
                    this.id()
                );
                reply.call(2);
                return;
            };
            // No event answers no
            let waited = match &event {
                Some(e) => e.wait_until_signaled_value(fifo_event_value_in, INFO_WAIT_MS),
                None => false,
            };
            if !waited {
                reply.call(3);
                return;
            }
            if let Err(t) = this.ensure_object_list_mapped(hold) {
                log::warn!("task {}: info refused: {}", this.id(), exception(&t));
                reply.call(2);
                hold.land(t);
                return;
            }
            if Self::command_too_big(&data, resource_count, cmd_buffer_count) {
                log::warn!(
                    "task {}: info refused: {resource_count} resources and {cmd_buffer_count} \
                     command buffers do not fit its {} bytes",
                    this.id(),
                    data.len()
                );
                reply.call(2);
                return;
            }
            let (resources, cmd_buffers) =
                Self::exec_records(&data, resource_count, cmd_buffer_count);
            let mut shared_texture_backing_ids: Vec<u32> = Vec::new();
            let ran = (|| -> RaiseResult<_> {
                let _ = this.prepare_resources(
                    hold,
                    resources,
                    None,
                    0,
                    &mut shared_texture_backing_ids,
                )?;
                let command_queue = this.command_queue.lock().as_ref().map(Handle::clone_ref);
                let command_buffer = match &command_queue {
                    Some(q) => q.command_buffer(OnFault::Return)?,
                    None => None,
                };
                let mut cursors = this.cursors(hold, cmd_buffers)?;
                let (deserializer, _, _) = this.parts();
                let _decoders =
                    deserializer.decode_segments(hold, &mut cursors, command_buffer.as_ref())?;
                signaled.store(fifo_event_value_in.wrapping_add(1), Ordering::SeqCst);
                if let Some(e) = &event {
                    e.set_signaled_value(signaled.load(Ordering::SeqCst))?;
                }
                Ok(())
            })();
            if let Err(t) = ran {
                log::warn!("task {}: info faulted: {}", this.id(), exception(&t));
                reply.call(2);
                hold.land(t);
                return;
            }
            reply.call(1);
        });
    }

    /// Writes the compute pipeline's info into the guest's reply at `offset`: up to `count`
    /// entries for the keys below `max_key`, then a `{0, 0}` end when there is room. No task
    /// lock. Replies 1, or 2 (logged) when the object lists cannot be mapped, the manager
    /// refuses the reference, or a raise.
    ///
    /// Reference 0 names no pipeline, and ends the task.
    pub(super) fn get_compute_info(
        &self,
        pipeline_state_reference: u32,
        max_key: u32,
        count: u32,
        offset: u64,
        reply: Reply<u32>,
    ) {
        self.job(move |this, hold| {
            if let Err(t) = this.ensure_object_list_mapped(hold) {
                log::warn!(
                    "task {}: compute info refused: {}",
                    this.id(),
                    exception(&t)
                );
                reply.call(2);
                hold.land(t);
                return;
            }
            // The entries' mapping, held until after the reply.
            let mut held = None;
            let wrote = (|| -> RaiseResult<_> {
                let (_, resource_manager, _) = this.parts();
                let pipeline = match resource_manager
                    .get_object::<ComputePipelineStates>(hold, pipeline_state_reference)?
                {
                    Err(e) => {
                        log::warn!(
                            "task {}: compute info refused: pipeline {pipeline_state_reference} \
                             not found: {e}",
                            this.id()
                        );
                        reply.call(2);
                        return Ok(false);
                    }
                    Ok(pipeline) => pipeline,
                };
                let Some(pipeline) = pipeline else {
                    fatal!(DeathKind::NoComputePipeline)
                };
                let length = u64::from(count).wrapping_mul(COMPUTE_INFO_ENTRY_SIZE);
                let mapped = this.mapped_address_for_offset(hold, offset, length, true)?;
                let entries = GuestPtr::new(mapped.address);
                held = mapped.mapping;
                let mut index: u32 = 0;
                let put = |index: &mut u32, key: u32, value: u32| {
                    let at = u64::from(*index).wrapping_mul(COMPUTE_INFO_ENTRY_SIZE);
                    store_compute_info_pair(entries, at, key, value);
                    *index = index.wrapping_add(1);
                };
                if INFO_KEY_MAX_TOTAL_THREADS < max_key && index < count {
                    let v = pipeline.max_total_threads_per_threadgroup().low_u32();
                    put(&mut index, INFO_KEY_MAX_TOTAL_THREADS, v);
                }
                if INFO_KEY_THREAD_EXECUTION_WIDTH < max_key && index < count {
                    let v = pipeline.thread_execution_width().low_u32();
                    put(&mut index, INFO_KEY_THREAD_EXECUTION_WIDTH, v);
                }
                if INFO_KEY_STATIC_THREADGROUP_MEMORY < max_key && index < count {
                    let v = pipeline.static_threadgroup_memory_length().low_u32();
                    put(&mut index, INFO_KEY_STATIC_THREADGROUP_MEMORY, v);
                }
                // The `{0, 0}` end is one 64-bit store at every alignment
                if index < count {
                    let at = u64::from(index).wrapping_mul(COMPUTE_INFO_ENTRY_SIZE);
                    entries.write_u64(at, 0, DeathKind::NullReply);
                }
                Ok(true)
            })();
            match wrote {
                Ok(true) => reply.call(1),
                // Replied 2 already
                Ok(false) => {}
                Err(t) => {
                    log::warn!(
                        "task {}: compute info faulted: {}",
                        this.id(),
                        exception(&t)
                    );
                    reply.call(2);
                    hold.land(t);
                }
            }
            drop(held);
        });
    }

    /// Writes the heap texture's size and alignment (two `u64`s) into the guest's reply at
    /// `reply_virtual_offset`, under the unique task lock. Replies 1, or 2 (logged) for a reply
    /// length outside 16 to a guest page less one, the manager's refusal, or a raise.
    ///
    /// The reply's mapping is held by a holder of its own ([`Held::raising`]), a landing frame
    /// whose hold holds there ([`Hold::of`]), released before the reply of 1 or after the
    /// refusal's reply of 2; a raise carries it on to the job's holder, so it goes after the
    /// handler's reply.
    pub(super) fn heap_texture_size_and_align(
        &self,
        data: Vec<u8>,
        reply_virtual_offset: u64,
        reply_length: u64,
        reply: Reply<u32>,
    ) {
        self.job(move |this, hold| {
            let _lock = this.task_lock(Access::Unique);
            let status = (|| -> RaiseResult<_> {
                let wrote = Held::raising(|held| {
                    let hold = Hold::of(held);
                    if !(HEAP_REPLY_MIN..=HEAP_REPLY_MAX).contains(&reply_length) {
                        return raise_caught(
                            RaiseKind::ReplyLength,
                            format_args!(
                                "reply of {reply_length:#x} bytes is not {HEAP_REPLY_MIN:#x} to \
                                 {HEAP_REPLY_MAX:#x}"
                            ),
                        );
                    }
                    let mapped = this.mapped_address_for_offset(
                        hold,
                        reply_virtual_offset,
                        reply_length,
                        true,
                    )?;
                    let address = mapped.address;
                    held.keep(mapped.mapping);
                    let (_, resource_manager, _) = this.parts();
                    match resource_manager.heap_texture_size_and_align(&data)? {
                        Err(e) => {
                            log::warn!("task {}: heap texture size refused: {e}", this.id());
                            reply.call(2);
                            Ok(false)
                        }
                        Ok(value) => {
                            let at = GuestPtr::new(address);
                            at.write_u64(0, value.size, DeathKind::NullReply);
                            at.write_u64(8, value.align, DeathKind::NullReply);
                            Ok(true)
                        }
                    }
                })?;
                Ok(wrote.then_some(1))
            })();
            match status {
                // Replied 2 already
                Ok(None) => {}
                Ok(Some(s)) => reply.call(s),
                Err(t) => {
                    log::warn!(
                        "task {}: heap texture size faulted: {}",
                        this.id(),
                        exception(&t)
                    );
                    reply.call(2);
                    hold.land(t);
                }
            }
        });
    }

    /// Resets rate map `map` from its serialized data in the task, under the shared task lock.
    /// Replies 1, or 2 when the object lists cannot be mapped (logged), for reference 0
    /// (logged), or when the object list has no entry for it.
    ///
    /// Outside any handler but the object lists' mapping, so a raise in the reset ends the
    /// task; a map never created succeeds.
    pub(super) fn reset_rasterization_rate_map(&self, map: u32, reply: Reply<u32>) {
        self.job(move |this, hold| {
            let _lock = this.task_lock(Access::Shared);
            if let Err(t) = this.ensure_object_list_mapped(hold) {
                log::warn!(
                    "task {}: rate map reset refused: {}",
                    this.id(),
                    exception(&t)
                );
                reply.call(2);
                hold.land(t);
                return;
            }
            if map == 0 {
                log::warn!("task {}: rate map reset refused: reference 0", this.id());
                reply.call(2);
                return;
            }
            let (deserializer, resource_manager, lists) = this.parts();
            let Some(rate_map) = resource_manager.get_existing_object::<RasterizationRateMaps>(map)
            else {
                reply.call(1);
                return;
            };
            let Ok(entry) = lists.get_object_entry(map) else {
                reply.call(2);
                return;
            };
            if entry.object_type() == RATE_MAP_OBJECT_TYPE {
                let data = unraised(
                    // The job has no handler here.
                    this.data_from_virtual_offset(
                        hold,
                        OnFault::Die,
                        entry.offset,
                        u64::from(entry.length()),
                    ),
                )
                .unwrap_or_default();
                unraised(
                    deserializer.reset_rasterization_rate_map_with_serialized_data(
                        &data,
                        rate_map.as_ref(),
                    ),
                );
            }
            reply.call(1);
        });
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;

    use parking_lot::Mutex;

    use super::super::testing::*;
    use super::*;
    use crate::platform::device::GpuDevice;
    use crate::platform::types::NsErrorText;
    use crate::tasks::isolation::TaskLife;
    use crate::tasks::launcher::ServiceSide;
    use crate::tasks::wire::ToService;
    use crate::testkit::guest_ram::RawRegion;
    use crate::testkit::objects::kind;
    use crate::testkit::platform::FAKE_REGISTRY_ID;
    use crate::testkit::script::Step;

    /// An entry `{key, value}` as one 64-bit load of it sees it.
    fn word(key: u32, value: u32) -> u64 {
        u64::from(value) << 32 | u64::from(key)
    }

    fn ns_error(description: &str) -> NsErrorText {
        NsErrorText {
            domain: "MTLCommandBufferErrorDomain".into(),
            code: 1,
            description: description.into(),
        }
    }

    fn exec_message(
        stamp_index: u32,
        fifo_event_value_in: u64,
        gpu_completion: u64,
        reply: Reply<(u32, Option<Vec<u8>>, u32)>,
    ) -> ToService<F> {
        ToService::ExecIndirect3 {
            data: Vec::new(),
            cmd_buffer_count: 0,
            resource_count: 0,
            stamp_index,
            fifo_event_value_in,
            gpu_completion,
            reply,
        }
    }

    fn info_message(stamp_index: u32, reply: Reply<u32>) -> ToService<F> {
        ToService::InfoIndirect {
            data: Vec::new(),
            cmd_buffer_count: 0,
            resource_count: 0,
            stamp_index,
            fifo_event_value_in: 0,
            reply,
        }
    }

    /// A missing FIFO and an oversized count reply 2, each logged
    #[test]
    fn exec_early_replies() {
        let tx = ready(&rw_pages(4));
        let exec = |count: u32, index: u32| {
            let (got, reply) = sink();
            tx.send(ToService::ExecIndirect3 {
                data: vec![0; 8],
                cmd_buffer_count: count,
                resource_count: 0,
                stamp_index: index,
                fifo_event_value_in: 0,
                gpu_completion: 0x77,
                reply,
            });
            let v: Vec<u32> = got.lock().iter().map(|r| r.0).collect();
            v
        };
        let lines = Lines::capture();
        assert_eq!(exec(1, 0), vec![2]);
        assert_eq!(exec(0, 3), vec![2]);
        assert_eq!(
            lines.take(),
            [
                "task 7: exec refused: 0 resources and 1 command buffers do not fit its 8 bytes",
                "task 7: exec refused: FIFO 3 has no event"
            ]
        );
    }

    /// The exec adds its completion, signals when scheduled, then commits; the completion
    /// completes the token
    #[test]
    fn exec_commits_then_completes() {
        let tx = ready(&rw_pages(4));
        with_fifo(&tx, 1);
        // Metal completes the buffer after the commit, on its own thread
        tx.p.world()
            .set_auto_complete(Some(crate::testkit::CompleteOn::Thread));
        let mark = tx.p.call_log().mark();
        let (got, reply) = sink();
        tx.send(exec_message(1, 0, 0x77, reply));
        assert_eq!(got.lock().len(), 1);
        assert_eq!(got.lock()[0].0, 1);
        assert!(got.lock()[0].1.is_none());
        let sels = tx.p.call_log().sels_since(mark);
        let pos = |s: &str| sels.iter().position(|x| *x == s).unwrap();
        assert!(pos("addCompletedHandler:") < pos("encodeSignalEventScheduled:value:"));
        assert!(pos("encodeSignalEventScheduled:value:") < pos("commit"));
        assert!(!sels.contains(&"encodeWaitForEvent:value:"));
        assert_eq!(tx.task.fifos.lock()[&1].signaled.load(Ordering::SeqCst), 1);
        assert_eq!(*tx.host.sent.lock(), vec!["complete 119 1".to_owned()]);
    }

    /// An exec taken here (`on_message_here`, inside the delivery's run of the task's code, as
    /// the connection delivers one sent with `send_here`) runs its job on this thread, after the
    /// jobs queued before it, and commits and replies as on the worker
    #[test]
    fn an_exec_taken_here_runs_its_job_on_this_thread_after_the_jobs_before_it() {
        let tx = ready(&rw_pages(4));
        with_fifo(&tx, 1);
        tx.p.world()
            .set_auto_complete(Some(crate::testkit::CompleteOn::Thread));
        let order = Arc::new(Mutex::new(Vec::new()));
        let o = order.clone();
        tx.task.job(move |_, _| {
            std::thread::sleep(std::time::Duration::from_millis(20));
            o.lock().push(("earlier", std::thread::current().id()));
        });
        let o = order.clone();
        let reply = Reply::new(move |v: (u32, Option<Vec<u8>>, u32)| {
            assert_eq!(v.0, 1);
            o.lock().push(("exec", std::thread::current().id()));
        });
        let mark = tx.p.call_log().mark();
        let ran = tx
            .life
            .run(|| tx.task.on_message_here(exec_message(1, 0, 0x77, reply)));
        assert_eq!(ran, Some(()));
        let order = order.lock().clone();
        let here = std::thread::current().id();
        assert_eq!(order.len(), 2);
        assert_eq!(order[0].0, "earlier");
        assert_ne!(order[0].1, here);
        assert_eq!(order[1], ("exec", here));
        let sels = tx.p.call_log().sels_since(mark);
        let pos = |s: &str| sels.iter().position(|x| *x == s).unwrap();
        assert!(pos("addCompletedHandler:") < pos("encodeSignalEventScheduled:value:"));
        assert!(pos("encodeSignalEventScheduled:value:") < pos("commit"));
        assert_eq!(tx.task.fifos.lock()[&1].signaled.load(Ordering::SeqCst), 1);
        tx.settle();
    }

    /// A death in a job taken here unwinds to the delivery's run, which ends the task; the
    /// service queue is given up on the way, and the thread leaves the task's code
    #[test]
    fn a_death_in_a_job_taken_here_ends_the_task_and_gives_the_queue_up() {
        let tx = ready(&rw_pages(4));
        let after = AtomicBool::new(false);
        let ran = tx.life.run(|| {
            tx.task.job_on(true, |_, _| {
                crate::fatal!(DeathKind::SyncWithoutPagingEvent)
            });
            after.store(true, Ordering::SeqCst);
        });
        assert_eq!(ran, None);
        // The death unwound past the rest of the delivery.
        assert!(!after.load(Ordering::SeqCst));
        assert!(tx.life.death().is_some());
        assert!(TaskLife::current().is_none());
        // The queue was given up: a sync takes it at once.
        tx.task.queue.sync(|| ());
        // Later jobs of the ended task run nothing.
        let ran = Arc::new(AtomicBool::new(false));
        let r = ran.clone();
        tx.task.job(move |_, _| r.store(true, Ordering::SeqCst));
        tx.settle();
        assert!(!ran.load(Ordering::SeqCst));
    }

    /// A wait value other than the last signalled one makes the exec wait on the FIFO
    #[test]
    fn exec_waits_for_the_fifo() {
        let tx = ready(&rw_pages(4));
        with_fifo(&tx, 1);
        let mark = tx.p.call_log().mark();
        tx.send(exec_message(1, 5, 1, Reply::new(|_| {})));
        let waits = tx.p.call_log().since(mark);
        assert_eq!(
            waits
                .iter()
                .filter(|c| c.sel == "encodeWaitForEvent:value:")
                .count(),
            1
        );
    }

    /// No FIFO event → 2; a wait that times out (no event) → 3
    #[test]
    fn info_early_replies() {
        let tx = ready(&rw_pages(4));
        let (got, reply) = sink();
        tx.send(info_message(1, reply));
        assert_eq!(*got.lock(), vec![2]);
        tx.task.on_message(ToService::SetFifoEvent {
            event: None,
            index: 1,
        });
        let (got, reply) = sink();
        tx.send(info_message(1, reply));
        assert_eq!(*got.lock(), vec![3]);
    }

    /// An info decodes and signals on the CPU, never commits
    #[test]
    fn info_signals_without_a_commit() {
        let tx = ready(&rw_pages(4));
        with_fifo(&tx, 1);
        let mark = tx.p.call_log().mark();
        let (got, reply) = sink();
        tx.send(info_message(1, reply));
        assert_eq!(*got.lock(), vec![1]);
        let sels = tx.p.call_log().sels_since(mark);
        assert!(sels.contains(&"setSignaledValue:"));
        assert!(!sels.contains(&"commit"));
        assert_eq!(tx.task.fifos.lock()[&1].signaled.load(Ordering::SeqCst), 1);
    }

    /// An object list that cannot map: each ensuring message catches it and replies 2, one line
    /// each
    #[test]
    fn ensure_failure_replies_2_per_message() {
        // no object list in the task root: nothing to map
        let tx = fixture(1, &rw_pages(4));
        tx.ram
            .write_u32(page_pa(ROOT_PAGE) + ROOT_OBJECT_LIST_LENGTH, 0);
        assert_eq!(
            handshake_with(&tx, ROOT_PAGE, TASK_LEN, FAKE_REGISTRY_ID, true),
            vec![(1, pid())]
        );
        with_fifo(&tx, 1);
        let lines = Lines::capture();
        let (got, reply) = sink();
        tx.task.on_message(ToService::GetComputeInfo {
            pipeline_state_reference: 1,
            max_key: 8,
            count: 1,
            offset: 0,
            reply: reply.clone(),
        });
        tx.task.on_message(info_message(1, reply.clone()));
        tx.task
            .on_message(ToService::ResetRasterizationRateMap { map: 1, reply });
        let (exec_got, exec_reply) = sink();
        tx.send(exec_message(1, 0, 1, exec_reply));
        assert_eq!(*got.lock(), vec![2, 2, 2]);
        assert_eq!(exec_got.lock()[0].0, 2);
        let refused: Vec<String> = lines
            .take()
            .into_iter()
            .filter(|l| l.contains(" refused: "))
            .collect();
        assert_eq!(
            refused,
            [
                "task 7: compute info refused: the task's object lists could not be mapped",
                "task 7: info refused: the task's object lists could not be mapped",
                "task 7: rate map reset refused: the task's object lists could not be mapped",
                "task 7: exec refused: the task's object lists could not be mapped"
            ]
        );
    }

    /// An exec's and an info's raises log the raise, and reply 2
    #[test]
    fn exec_and_info_raises_log_the_raise() {
        let tx = ready(&rw_pages(4));
        with_fifo(&tx, 1);
        // the command queue raises at the first send inside each handler
        tx.p.script().push_n(
            "commandBuffer",
            2,
            Step::raise("NSGenericException", "boom"),
        );
        let lines = Lines::capture();
        let (exec_got, exec_reply) = sink();
        tx.task.on_message(exec_message(1, 0, 1, exec_reply));
        let (got, reply) = sink();
        tx.send(info_message(1, reply));
        assert_eq!(exec_got.lock()[0].0, 2);
        assert_eq!(*got.lock(), vec![2]);
        assert_eq!(
            lines.take(),
            ["task 7: exec faulted: boom", "task 7: info faulted: boom"]
        );
    }

    /// A failed exec buffer's completion logs its error, or that Metal gave none; the token
    /// completes with 3
    #[test]
    fn exec_completion_logs_the_error() {
        let tx = ready(&rw_pages(4));
        with_fifo(&tx, 1);
        for error in [Some(ns_error("GPU fault")), None] {
            tx.send(exec_message(1, 0, 5, Reply::new(|_| {})));
            let ids = tx.p.world().committed();
            assert_eq!(ids.len(), 1);
            assert!(tx.p.world().complete(
                ids[0],
                CommandBufferStatus::ERROR,
                error,
                crate::testkit::CompleteOn::Caller
            ));
            // the next exec waits on the value this one signalled
            tx.task.fifos.lock()[&1].signaled.store(0, Ordering::SeqCst);
        }
        assert_eq!(
            *tx.host.sent.lock(),
            vec!["complete 5 3".to_owned(), "complete 5 3".to_owned()]
        );
    }

    /// A paging-out's completion behind a failed exec logs the exec buffer's error, or that
    /// Metal gave none
    #[test]
    fn sync_completion_logs_the_exec_error() {
        let tx = ready(&rw_pages(4));
        let lines = Lines::capture();
        let queue = tx.p.device().new_command_queue().unwrap();
        for error in [Some(ns_error("exec fault")), None] {
            let exec = queue.command_buffer(OnFault::Die).unwrap().unwrap();
            exec.commit(OnFault::Die).unwrap();
            assert!(tx.p.world().complete(
                tx.p.world().committed()[0],
                CommandBufferStatus::ERROR,
                error,
                crate::testkit::CompleteOn::Caller
            ));
            tx.task.sync_completed(
                CompletedInfo {
                    status: CommandBufferStatus::COMPLETED,
                    error: None,
                    gpu_times_ns: None,
                },
                Some(exec),
                9,
            );
        }
        assert_eq!(
            lines.take(),
            [
                "task 7: the exec behind a paging-out failed: exec fault",
                "task 7: the exec behind a paging-out failed: Metal gave no error"
            ]
        );
        assert_eq!(
            *tx.host.sent.lock(),
            vec!["complete 9 3".to_owned(), "complete 9 3".to_owned()]
        );
    }

    /// A synchronize's failed paging buffer logs Metal's error, or that it gave none; the
    /// completion replies 3
    #[test]
    fn synchronize_completion_logs_the_error() {
        let tx = ready(&rw_pages(4));
        for error in [Some(ns_error("paging fault")), None] {
            let (got, reply) = sink();
            tx.send(ToService::SynchronizeResources {
                resources: Vec::new(),
                count: 0,
                discard: false,
                reply,
            });
            let ids = tx.p.world().committed();
            assert_eq!(ids.len(), 1);
            assert!(tx.p.world().complete(
                ids[0],
                CommandBufferStatus::ERROR,
                error.clone(),
                crate::testkit::CompleteOn::Caller
            ));
            assert_eq!(*got.lock(), vec![3]);
        }
    }

    /// Compute info for reference 0 ends the task
    #[test]
    fn compute_info_reference_zero_dies() {
        let tx = ready(&rw_pages(4));
        tx.send(ToService::GetComputeInfo {
            pipeline_state_reference: 0,
            max_key: 8,
            count: 4,
            offset: PAGE,
            reply: Reply::new(|_| {}),
        });
        assert_eq!(
            tx.life.death().map(crate::raise::death::DeathId::site),
            Some(DeathKind::NoComputePipeline.site())
        );
    }

    /// Each entry is stored whole and the end as one 64-bit zero: the keys follow `max_key` and
    /// `count`, the values are truncated to 32 bits, and entries past them keep the guest's
    /// bytes
    #[test]
    fn compute_info_writes_whole_entries() {
        let tx = ready(&rw_pages(4));
        let pipeline =
            tx.p.world()
                .new_obj::<kind::ComputePipeline>(Default::default());
        pipeline.set_static_threadgroup_memory_length(0x1_0000_0100);
        assert!(
            tx.task
                .parts()
                .1
                .register_object::<ComputePipelineStates>(1, pipeline)
        );
        let reply_pa = page_pa(DATA_PAGE + 1);
        let run = |max_key: u32, count: u32, offset: u64| {
            tx.ram.write(reply_pa, &[0xff; 0x40]);
            tx.p.fake_vm().sync();
            let (got, reply) = sink();
            tx.send(ToService::GetComputeInfo {
                pipeline_state_reference: 1,
                max_key,
                count,
                offset,
                reply,
            });
            tx.p.fake_vm().sync();
            assert_eq!(*got.lock(), vec![1]);
            let at = reply_pa + (offset & PAGE_MASK);
            (0..6)
                .map(|i| tx.ram.read_u64(at + 8 * i).unwrap())
                .collect::<Vec<u64>>()
        };
        let all = [word(1, 1024), word(3, 32), word(4, 0x100), 0];
        assert_eq!(run(5, 5, PAGE), [&all[..], &[u64::MAX; 2]].concat());
        // count 3: no room for the end
        assert_eq!(run(5, 3, PAGE), [&all[..3], &[u64::MAX; 3]].concat());
        // max_key 4 stops before key 4; max_key 2 before key 3; count 1 stops after key 1
        assert_eq!(run(4, 5, PAGE), [&all[..2], &[0], &[u64::MAX; 3]].concat());
        assert_eq!(run(2, 5, PAGE), [&all[..1], &[0], &[u64::MAX; 4]].concat());
        assert_eq!(run(5, 1, PAGE), [&all[..1], &[u64::MAX; 5]].concat());
        // A reply 4 mod 8: entries at 4 and 20 sit in a granule, 12 and 28 cross one; the
        // bytes are the same
        assert_eq!(run(5, 5, PAGE + 4), [&all[..], &[u64::MAX; 2]].concat());
        assert_eq!(tx.ram.read_u32(reply_pa).unwrap(), u32::MAX);
    }

    /// A pair inside a 16-byte granule is one single-copy-atomic store: a vCPU polling the
    /// entry with one 64-bit load sees the old pair or the new one, never the new key beside
    /// the old value
    #[test]
    fn compute_info_pairs_are_never_torn_in_a_granule() {
        let region = RawRegion::new(PAGE).unwrap();
        let entries = GuestPtr::new(region.addr());
        let old = word(u32::MAX, u32::MAX);
        let new = word(1, 1024);
        for at in [0u64, 4, 8] {
            entries.write_u64(at, old, DeathKind::NullReply);
            let polling = AtomicBool::new(false);
            let done = AtomicBool::new(false);
            let torn = std::thread::scope(|s| {
                let poller = s.spawn(|| {
                    let mut torn = Vec::new();
                    while !done.load(Ordering::Acquire) {
                        let w = entries.read_u64(at, DeathKind::NullReply);
                        polling.store(true, Ordering::Release);
                        if w != old && w != new && torn.len() < 8 {
                            torn.push(w);
                        }
                    }
                    torn
                });
                // The stores start once the poller reads, so the two overlap
                while !polling.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }
                for _ in 0..200_000 {
                    store_compute_info_pair(entries, at, 1, 1024);
                    store_compute_info_pair(entries, at, u32::MAX, u32::MAX);
                }
                done.store(true, Ordering::Release);
                poller.join().unwrap()
            });
            // {1, 0xffffffff} is the tear two stores show: the new key beside the old value
            assert!(torn.is_empty(), "entry at {at}: torn pairs {torn:#x?}");
        }
    }

    /// The pair lands at every alignment: the key at +0, the value at +4, nothing else
    /// touched; a NULL reply dies
    #[test]
    fn compute_info_pairs_land_at_every_alignment() {
        let region = RawRegion::new(PAGE).unwrap();
        let entries = GuestPtr::new(region.addr());
        for at in 0..16u64 {
            region.zero(0, 0x40);
            store_compute_info_pair(entries, at, 0x0403_0201, 0x0807_0605);
            let mut want = vec![0u8; 0x40];
            want[at as usize..at as usize + 8].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
            assert_eq!(region.read_vec(0, 0x40).unwrap(), want, "entry at {at}");
        }
        for at in [0u64, 12] {
            assert_eq!(
                crate::testkit::capture::catch_death(|| {
                    store_compute_info_pair(GuestPtr::default(), at, 1, 2)
                })
                .unwrap_err(),
                DeathKind::NullReply.site()
            );
        }
    }

    /// A reply length outside 16 to a page less one raises, caught: 2, logged with the reason
    #[test]
    fn heap_texture_reply_length_checked() {
        let tx = ready(&rw_pages(4));
        for len in [15, PAGE] {
            let (got, reply) = sink();
            tx.send(ToService::HeapTextureSizeAndAlign {
                data: Vec::new(),
                reply_virtual_offset: 0,
                reply_length: len,
                reply,
            });
            assert_eq!(*got.lock(), vec![2]);
        }
    }

    /// A raise inside the reply's holder carries its mapping on: the reply page's mapping is
    /// still held when the handler replies 2, and the job's holder releases it after
    #[test]
    fn heap_texture_raise_keeps_the_mapping_until_after_the_reply() {
        let tx = ready(&rw_pages(4));
        tx.p.script().push(
            "heapTextureSizeAndAlignWithDescriptor:",
            Step::raise("NSInvalidArgumentException", "bad descriptor"),
        );
        // A size-22 command of 0x28 bytes
        let mut data = vec![0u8; 0x28];
        data[0] = 22;
        data[8] = 2; // a 2D texture type
        let p = tx.p.clone();
        let seen: Arc<Mutex<Vec<(usize, u32)>>> = Arc::new(Mutex::new(Vec::new()));
        let s2 = seen.clone();
        tx.p.call_log().clear();
        tx.send(ToService::HeapTextureSizeAndAlign {
            data,
            reply_virtual_offset: 0,
            reply_length: 16,
            reply: Reply::new(move |v: u32| {
                let freed = p.call_log().calls_to("mach_vm_deallocate").len();
                s2.lock().push((freed, v));
            }),
        });
        let calls = tx.p.call_log().sels();
        assert!(calls.contains(&"heapTextureSizeAndAlignWithDescriptor:"));
        // At the reply the mapping is not yet deallocated
        assert_eq!(*seen.lock(), vec![(0, 2)]);
        // The job's holder released it after the reply
        assert_eq!(tx.p.call_log().calls_to("mach_vm_deallocate").len(), 1);
    }

    /// The manager's refusals reply 2
    #[test]
    fn manager_refusals_reply_2() {
        let tx = ready(&rw_pages(4));
        let (got, reply) = sink();
        // a command shorter than a header
        tx.task.on_message(ToService::HeapTextureSizeAndAlign {
            data: vec![0; 4],
            reply_virtual_offset: 0,
            reply_length: 16,
            reply: reply.clone(),
        });
        // a reference past the object list
        tx.send(ToService::GetComputeInfo {
            pipeline_state_reference: 1,
            max_key: 8,
            count: 1,
            offset: 0,
            reply,
        });
        assert_eq!(*got.lock(), vec![2, 2]);
    }

    /// Reference 0 is refused (2)
    #[test]
    fn reset_rate_map_zero_is_refused() {
        let tx = ready(&rw_pages(4));
        let (got, reply) = sink();
        tx.send(ToService::ResetRasterizationRateMap { map: 0, reply });
        assert_eq!(*got.lock(), vec![2]);
    }

    /// A map never created succeeds
    #[test]
    fn reset_rate_map_never_created_succeeds() {
        let tx = ready(&rw_pages(4));
        let (got, reply) = sink();
        tx.send(ToService::ResetRasterizationRateMap { map: 5, reply });
        assert_eq!(*got.lock(), vec![1]);
    }
}
