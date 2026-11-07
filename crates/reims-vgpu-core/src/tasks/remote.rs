//! Remote tasks: a user task under process isolation, run by its own service
//! (`tasks::isolation`).
//!
//! [`RemoteTask`] is the host's proxy of one: each task command becomes one [`ToService`]
//! message on the task's link ([`ServiceLink`], `tasks::launcher`), and completes through a
//! [`Token`] of the task's [`Endpoint`] when the service replies or the task dies. The
//! endpoint is what the host exports on the connection: it answers the service's messages
//! ([`completions`] for its completions, [`device_link`] for the IOSurfaces it keeps alive and
//! the kernel task's shared textures).
//!
//! Ordering contracts (each named test pins it):
//! - **The memory map** must be whole pages at non-NULL, page-aligned host addresses; any other
//!   map makes no task, and nothing is launched (`a_map_that_is_not_whole_pages_makes_no_task`).
//! - **Construction** launches a fresh service, sends the handshake with the device's registry
//!   id, its argument-buffer support, a copy of its features and its memory map (read in that
//!   order), and blocks the FIFO thread until the handshake's token completes; any status but 1
//!   is no task (`the_handshake_carries_the_devices_configuration`,
//!   `a_reply_other_than_one_is_nil_and_the_task_goes`).
//! - **A reply holds the task**: every reply the link holds keeps the task alive, so the task
//!   drops when the link releases its last reply (`a_pending_reply_keeps_the_task_alive`). A
//!   reply's captures drop tokens first, the task last: a token's release releases the FIFO's
//!   completion (and the FIFO it holds) while the task still lives
//!   (`the_token_goes_before_the_tasks_drop`,
//!   `the_reply_drops_its_token_then_the_task_then_the_endpoint`,
//!   `the_exec_reply_drops_gpu_token_then_token_then_the_task`).
//! - **Drop drains before it invalidates**: the task's drop waits, with no timeout, until every
//!   outstanding token is completed or crashed, then invalidates the link (the service ends),
//!   then releases the FIFO events in index order
//!   (`drop_waits_for_the_gpu_completion_then_invalidates`).
//!   A last reference dropped inside a reply with a token outstanding hangs the link's queue
//!   (`the_last_reference_in_a_reply_with_a_token_outstanding_hangs`).
//! - **FIFO events**: a FIFO's event crosses once per stamp index (`setFifoEvent:forIndex:`)
//!   before the command that needs it, its handle carried by the message. A FIFO's deletion
//!   clears its index after the message (`forwards_then_clears_the_stamp_bit`).
//! - **Exec** blocks the FIFO thread until the service's reply (or the death), and the service
//!   runs the exec's job on that thread, in its turn after every message sent before it
//!   (`an_exec_is_taken_on_the_fifo_thread_after_the_messages_before_it`); only a reply of 1
//!   advances the FIFO's event value, a reply of neither 1 nor 4 completes at once, and a 1
//!   completes later through the service's `completeAsyncTokenID:`
//!   (`success_advances_the_counter_and_completes_through_the_service`).
//! - **After a death** every command completes with 4 inside the call and sends nothing
//!   (`crashed_forward`, `after_a_crash_both_tokens_are_born_four`). Three FIFO commands are not
//!   faulted by a dead task: MapMemory2 (57), whose map is a no-op success that never asks the
//!   service, and a child FIFO's define and delete (48, 49): the define asks no task, and the
//!   delete waits for each task's completion whatever its status (`DeviceCore::delete_fifo`).

pub mod completions;
pub mod device_link;
pub mod endpoint;

use std::fmt;
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use parking_lot::Mutex;

use crate::fifo::wire::Records;
use crate::guest::records::{
    APV_MAX_FIFOS, APV_PAGE_SIZE, CmdExecIndirectCmdBuffer, CmdExecIndirectResource,
    CmdExecIndirectResource3,
};
use crate::memory::map::MemoryMap;
use crate::num::{self, ToU64, ToUsize, Truncate};
use crate::platform::event::SharedEvent;
use crate::platform::{Handle, Platform};
use crate::raise::death::DeathSite;
use crate::rt::Hold;
use crate::seams::links::TaskDeviceLink;
use crate::tasks::launcher::{ServiceLink, TaskLauncher};
use crate::tasks::wire::{Reply, TaskConfiguration, ToService};
use crate::tasks::{Completion, ExecArgs, ExecStatus, ResourceInvalidation, ResourceReference};

pub use completions::{Completions, Token, TokenHandler};
pub use endpoint::Endpoint;

/// The remote tasks' deaths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeathKind {
    /// The handshake reads the features of a device that is gone.
    DeviceGone,
    /// A FIFO event shared at a stamp index past the FIFOs'.
    StampIndex,
    /// A completion token signalled a second time.
    TokenSignalledTwice,
}

impl DeathKind {
    /// The death's dotted id.
    pub const fn id(self) -> &'static str {
        match self {
            DeathKind::DeviceGone => "tasks.remote.device_gone",
            DeathKind::StampIndex => "tasks.remote.stamp_index",
            DeathKind::TokenSignalledTwice => "tasks.remote.token_signalled_twice",
        }
    }

    /// The site a death of this kind records: the kind of fault, and the id.
    pub const fn site(self) -> DeathSite {
        use crate::raise::death::DeathKind as Fault;
        let kind = match self {
            DeathKind::DeviceGone => Fault::NullDeref,
            DeathKind::StampIndex | DeathKind::TokenSignalledTwice => Fault::Assert,
        };
        DeathSite {
            kind,
            file: self.id(),
            line: 0,
        }
    }
}

impl fmt::Display for DeathKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

impl From<DeathKind> for crate::raise::DeathId {
    fn from(k: DeathKind) -> crate::raise::DeathId {
        crate::raise::DeathId::Tasks(crate::tasks::DeathKind::Remote(k))
    }
}

/// The host's proxy of an isolated user task: the task commands as messages to its service.
///
/// Fields drop in declaration order after [`Drop::drop`]: the link, the (emptied) FIFO events,
/// then the endpoint.
pub struct RemoteTask<P: Platform> {
    link: Arc<dyn ServiceLink<P>>,
    /// The event last shared for each stamp index, one slot per FIFO. A slot is touched only by
    /// the FIFO thread whose index it is; its lock guards the slot's memory and never serialises
    /// two FIFOs.
    fifo_events: Box<[Mutex<Option<P::SharedEvent>>]>,
    endpoint: Arc<Endpoint<P>>,
    task_id: u32,
    /// One bit per stamp index whose event the service has been sent.
    shared_fifo_events: AtomicU64,
    /// The pid the handshake's reply carries (nothing reads it).
    pid: AtomicI32,
    /// This task, for the replies that hold it.
    me: Weak<RemoteTask<P>>,
}

/// The data an exec's reply fills: the ids of the kernel task's shared-texture backings the
/// command used, and their count.
#[derive(Default)]
struct Backings {
    data: Vec<u8>,
    count: u32,
}

// The replies, as the link releases them.
//
// Each reply's captures are one struct, its fields in the order they drop (declaration order), and
// the reply's closure only calls the struct's `invoke`, so the closure captures the struct whole
// (the variables a closure captures by move drop in an unspecified order). The order is fixed: the
// tokens first, whose release releases the FIFO's completion (and the FIFO it holds), the task
// last, whose drop may block in the drain and then invalidates the link.

/// A forwarding command's reply: completes its token with the service's status.
struct ForwardReply<P: Platform> {
    /// Dropped first.
    token: Arc<Token>,
    /// Dropped before the task, so while the task lives this is never the endpoint's last
    /// reference.
    endpoint: Arc<Endpoint<P>>,
    /// Held for its release: dropped last.
    _task: Option<Arc<RemoteTask<P>>>,
}

impl<P: Platform> ForwardReply<P> {
    fn invoke(&self, status: u32) {
        self.endpoint
            .completions()
            .complete(&self.token, ExecStatus(status));
    }
}

/// An exec's reply: stores the backings the service reports, and completes the GPU wait with the
/// service's status.
struct ExecReply<P: Platform> {
    /// The FIFO thread's wait: dropped first.
    gpu_token: Arc<Token>,
    /// The completion token, whose id crossed the wire: held for its release, after the GPU
    /// wait's.
    _token: Arc<Token>,
    backings: Arc<Mutex<Backings>>,
    /// As in [`ForwardReply`].
    endpoint: Arc<Endpoint<P>>,
    /// Held for its release: dropped last.
    _task: Option<Arc<RemoteTask<P>>>,
}

impl<P: Platform> ExecReply<P> {
    fn invoke(&self, (status, data, count): (u32, Option<Vec<u8>>, u32)) {
        {
            let mut b = self.backings.lock();
            b.count = count;
            // A nil `NSData` appends nothing.
            if let Some(data) = data {
                b.data.extend_from_slice(&data);
            }
        }
        self.endpoint
            .completions()
            .complete(&self.gpu_token, ExecStatus(status));
    }
}

/// The handshake's reply: completes the construction's wait and stores the pid it carries.
struct HandshakeReply<P: Platform> {
    /// Dropped first.
    token: Arc<Token>,
    task: Arc<RemoteTask<P>>,
    /// The endpoint the construction made, held apart from the task: dropped last.
    endpoint: Arc<Endpoint<P>>,
}

impl<P: Platform> HandshakeReply<P> {
    fn invoke(&self, (status, pid): (u32, i32)) {
        self.endpoint
            .completions()
            .complete(&self.token, ExecStatus(status));
        // Stored after the completion, racing the waiter; nothing reads it.
        self.task.pid.store(pid, Ordering::Relaxed);
    }
}

/// A token handler that runs the FIFO's completion.
fn completing(done: Completion) -> TokenHandler {
    Box::new(move |status, error| done(status, error))
}

impl<P: Platform> RemoteTask<P> {
    /// Launches the task's service through `launcher`, sends the handshake and blocks the calling
    /// (FIFO) thread until it completes: the task, or `None` for any status but 1 (the task then
    /// drops when the link releases the handshake's reply). A memory map an isolated task cannot
    /// use ([`whole_pages`]) is `None` before anything is launched.
    pub fn launch(
        platform: &Arc<P>,
        device: TaskDeviceLink<P>,
        launcher: &dyn TaskLauncher<P>,
        task_root: u32,
        length: u64,
        task_id: u32,
    ) -> Option<Arc<Self>> {
        let map = device.upgrade().and_then(|d| d.memory_map());
        if map.is_some_and(|m| !whole_pages(&m)) {
            return None;
        }
        // The endpoint is made before the connection opens.
        let endpoint = Arc::new(Endpoint::for_task(device.clone(), task_id));
        let link = launcher.launch(platform, endpoint.clone());
        let task = Arc::new_cyclic(|me| RemoteTask {
            link,
            fifo_events: (0..APV_MAX_FIFOS).map(|_| Mutex::new(None)).collect(),
            endpoint,
            task_id,
            shared_fifo_events: AtomicU64::new(0),
            pid: AtomicI32::new(0),
            me: me.clone(),
        });
        let device = device.upgrade();
        // The registry id, then the argument-buffer support; zero and false for a device that is
        // gone.
        let configuration = TaskConfiguration {
            registry_id: device.as_ref().map_or(0, |d| {
                crate::platform::device::GpuDevice::caps(&d.mtl_device()).registry_id
            }),
            supports_argument_buffers: device
                .as_ref()
                .is_some_and(|d| d.supports_argument_buffers()),
        };
        let token = task.endpoint.completions().issue();
        // Reading a gone device's features is a NULL dereference (the device's own calls never
        // reach here without it).
        let Some(device) = device else {
            crate::fatal!(DeathKind::DeviceGone)
        };
        let features = device.features().load();
        let memory_map = device.memory_map();
        drop(device);
        let reply = {
            let reply = HandshakeReply {
                token: token.clone(),
                task: task.clone(),
                endpoint: task.endpoint.clone(),
            };
            Reply::new(move |v: (u32, i32)| reply.invoke(v))
        };
        task.link.send(ToService::Handshake {
            task_root,
            task_size: length,
            task_id,
            configuration,
            features,
            memory_map,
            reply,
        });
        // The FIFO thread blocks until the reply or a crash.
        if token.wait() != ExecStatus::OK {
            // The reply may still hold the task, which then goes when the link releases it.
            return None;
        }
        Some(task)
    }

    /// The guest's id of this task.
    pub fn task_id(&self) -> u32 {
        self.task_id
    }

    /// The pid the handshake reported; 0 before its reply.
    pub fn pid(&self) -> i32 {
        self.pid.load(Ordering::Relaxed)
    }

    /// Sends the service the FIFO's event for `stamp_index`, unless it was sent since the FIFO's
    /// creation or its last deletion. `event` is `None` for the FIFO's nil event, whose nil handle
    /// crosses.
    ///
    /// The slot keeps its own reference to the event; the message carries the event's new
    /// handle. No path reaches the handle's creation under a handler: a raise from it dies at
    /// its send, with the event kept in the slot, nothing sent and the index left unmarked.
    fn share_fifo_event(&self, event: Option<&P::SharedEvent>, stamp_index: u32) {
        // The bit is formed before the index is checked (the FIFO's index is always below 64).
        let bit = num::bit64(stamp_index);
        if self.shared_fifo_events.load(Ordering::SeqCst) & bit != 0 {
            return;
        }
        crate::fatal_unless!(stamp_index < APV_MAX_FIFOS, DeathKind::StampIndex);
        let handle = {
            let Some(slot) = self.fifo_events.get(stamp_index.to_usize()) else {
                crate::fatal!(DeathKind::StampIndex)
            };
            let mut slot = slot.lock();
            // The old event goes before the new one is kept.
            drop(slot.take());
            *slot = event.map(Handle::clone_ref);
            match slot.as_ref() {
                Some(e) => e.new_shared_event_handle(),
                None => None,
            }
        };
        self.link.send(ToService::SetFifoEvent {
            event: handle,
            index: stamp_index,
        });
        self.shared_fifo_events.fetch_or(bit, Ordering::SeqCst);
    }

    /// The reply of a forwarding command: completes `token`, and holds the task.
    fn forward_reply(&self, token: Arc<Token>) -> Reply<u32> {
        let reply = ForwardReply {
            token,
            endpoint: self.endpoint.clone(),
            _task: self.me.upgrade(),
        };
        Reply::new(move |status: u32| reply.invoke(status))
    }

    /// A token whose handler is the FIFO's completion, then one message whose reply completes
    /// it: every forwarding command.
    fn forward(&self, done: Completion, message: impl FnOnce(Reply<u32>) -> ToService<P>) {
        let token = self.endpoint.completions().issue_with(completing(done));
        let reply = self.forward_reply(token);
        self.link.send(message(reply));
    }

    /// Runs one exec: shares the FIFO's event, sends the command and blocks the FIFO thread until
    /// the service's reply (or the crash). The service runs the exec's job on the FIFO thread
    /// itself, and its reply too when the link's queue is idle ([`ServiceLink::send_here`]): with
    /// the connection queue, the service queue and the link's queue idle, as they are between a
    /// FIFO's execs, an exec wakes none of the four threads a message sent with `send` waits for
    /// in turn (those queues' workers, then the FIFO thread). A reply of 1 advances the FIFO's
    /// event value; the kernel task's shared-texture backings the reply lists are invalidated for
    /// the guest; the completion then comes from the service (1), came from the crash (4), or is
    /// made here.
    #[allow(clippy::too_many_arguments)] // the exec's parts, as the service's message takes them
    fn exec(
        &self,
        hold: Hold<'_>,
        cmd_buffer_count: u32,
        command_data: Vec<u8>,
        stamp_index: u32,
        fifo_event: Option<&P::SharedEvent>,
        fifo_event_value: &Arc<AtomicU64>,
        resource_count: u32,
        done: Completion,
    ) {
        self.share_fifo_event(fifo_event, stamp_index);
        // The FIFO's event value advances only on the service's 1.
        let gpu_token = {
            let value = fifo_event_value.clone();
            self.endpoint
                .completions()
                .issue_with(Box::new(move |status, _error| {
                    if status == ExecStatus::OK {
                        value.fetch_add(1, Ordering::Relaxed);
                    }
                }))
        };
        let backings = Arc::new(Mutex::new(Backings::default()));
        // The completion token: its id crosses the wire.
        let token = self.endpoint.completions().issue_with(completing(done));
        let reply = {
            let reply = ExecReply {
                gpu_token: gpu_token.clone(),
                _token: token.clone(),
                backings: backings.clone(),
                endpoint: self.endpoint.clone(),
                _task: self.me.upgrade(),
            };
            Reply::new(move |v: (u32, Option<Vec<u8>>, u32)| reply.invoke(v))
        };
        self.link.send_here(ToService::ExecIndirect3 {
            data: command_data,
            cmd_buffer_count,
            resource_count,
            stamp_index,
            // Read at the send.
            fifo_event_value_in: fifo_event_value.load(Ordering::Relaxed),
            gpu_completion: token.id(),
            reply,
        });
        let status = gpu_token.wait();
        // The reply's count of ids, bounded by the ids it sent (the service always sends as
        // many as it counts).
        let ids: Vec<u32> = {
            let b = backings.lock();
            let all: Vec<u32> = b
                .data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| u32::from_ne_bytes(*w))
                .collect();
            let n = b.count.to_usize().min(all.len());
            all.get(..n).map(<[u32]>::to_vec).unwrap_or_default()
        };
        self.endpoint
            .device_link()
            .invalidate_shared_texture_backings(hold, &ids);
        if status != ExecStatus::OK && status != ExecStatus::CRASHED {
            self.endpoint.completions().complete(&token, status);
        }
    }
}

impl<P: Platform> Drop for RemoteTask<P> {
    /// Drains, invalidates the link, then releases the FIFO events in index order.
    fn drop(&mut self) {
        // Blocks until every outstanding token is completed or crashed.
        self.endpoint.completions().drain();
        // The service ends; the endpoint the link still holds then sees the invalidation.
        self.link.invalidate();
        for slot in self.fifo_events.iter_mut() {
            drop(slot.get_mut().take());
        }
    }
}

/// Whether an isolated task can use `map`: every range a whole number of pages, at a host
/// address that is page-aligned and not NULL. The service reads guest RAM through these same host
/// ranges.
fn whole_pages<P: Platform>(map: &MemoryMap<P>) -> bool {
    const IN_PAGE: u64 = APV_PAGE_SIZE - 1;
    map.ranges().iter().all(|r| {
        let host = r.virtual_address.addr().to_u64();
        host != 0 && r.physical_length != 0 && (host | r.physical_length) & IN_PAGE == 0
    })
}

/// The resources' records, then the command buffers': the bytes of an exec's or an info's
/// message. The FIFO's payload holds them contiguously, and this is that span.
fn command_data<R>(
    resources: &Records<'_, R>,
    cmd_buffers: &Records<'_, CmdExecIndirectCmdBuffer>,
) -> Vec<u8>
where
    R: zerocopy::FromBytes + zerocopy::IntoBytes + zerocopy::Immutable + zerocopy::KnownLayout,
{
    let mut data = Vec::with_capacity(
        resources
            .as_bytes()
            .len()
            .saturating_add(cmd_buffers.as_bytes().len()),
    );
    data.extend_from_slice(resources.as_bytes());
    data.extend_from_slice(cmd_buffers.as_bytes());
    data
}

/// The task commands ([`Task`](crate::tasks::Task)'s, where each is documented): each becomes
/// one message to the service.
impl<P: Platform> RemoteTask<P> {
    /// Tears the task down: nothing, the service runs until the task drops.
    pub fn teardown(&self) {
        // The service runs until the task drops.
    }

    /// Deletes the object a serialized delete names.
    pub fn delete_object_with_serialized_data(&self, data: &[u8], done: Completion) {
        let data = data.to_vec();
        self.forward(done, |reply| ToService::DeleteObject { data, reply });
    }

    /// Deletes a resource.
    pub fn delete_resource(&self, resource: u32, done: Completion) {
        self.forward(done, |reply| ToService::DeleteResource { resource, reply });
    }

    /// Discards resources.
    pub fn discard_resources(&self, resources: Records<'_, ResourceReference>, done: Completion) {
        // The records' bytes only, whatever follows them in the payload.
        let data = resources.as_bytes().to_vec();
        let count = resources.count();
        self.forward(done, |reply| ToService::DiscardResources {
            resources: data,
            count,
            reply,
        });
    }

    /// A FIFO was deleted: forwarded, then its event's index is cleared.
    pub fn fifo_deleted(&self, stamp_index: u32, done: Completion) {
        self.forward(done, |reply| ToService::FifoDeleted { stamp_index, reply });
        // After the send.
        self.shared_fifo_events
            .fetch_and(!num::bit64(stamp_index), Ordering::SeqCst);
    }

    /// Flushes the resources into guest memory, blocking until the service's flush.
    pub fn flush_resources(&self) {
        let token = self.endpoint.completions().issue();
        let reply = self.forward_reply(token.clone());
        self.link.send(ToService::FlushResources { reply });
        // Blocks until the service's flush; its status is ignored.
        let _ = token.wait();
    }

    /// Applies validity flags to resources.
    pub fn invalidate_resources(
        &self,
        resources: Records<'_, ResourceInvalidation>,
        done: Completion,
    ) {
        let data = resources.as_bytes().to_vec();
        let count = resources.count();
        self.forward(done, |reply| ToService::InvalidateResources {
            resources: data,
            count,
            reply,
        });
    }

    /// Maps task memory: a no-op success.
    pub fn map_memory_at_offset(&self, offset: u64, length: u64, done: Completion) {
        let _ = (offset, length);
        // A no-op success, no token and no message (a dead task too).
        done(ExecStatus::OK, None);
    }

    /// Discards the resource over replaced physical memory.
    pub fn replace_physical(&self, physical: u32, done: Completion) {
        self.forward(done, |reply| ToService::ReplacePhysical { physical, reply });
    }

    /// Sets the object and placement lists.
    pub fn set_object_list_offset(
        &self,
        object_list_offset: u64,
        object_list_length: u32,
        placement_list_offset: u64,
        placement_list_length: u32,
        done: Completion,
    ) {
        self.forward(done, |reply| ToService::SetObjectList {
            object_list_offset,
            object_list_length,
            placement_list_offset,
            placement_list_length,
            reply,
        });
    }

    /// Synchronizes resources into guest memory, and with `discard` discards them.
    pub fn synchronize_resources(
        &self,
        resources: Records<'_, ResourceReference>,
        discard: bool,
        done: Completion,
    ) {
        let data = resources.as_bytes().to_vec();
        let count = resources.count();
        self.forward(done, |reply| ToService::SynchronizeResources {
            resources: data,
            count,
            discard,
            reply,
        });
    }

    /// Unmaps task memory.
    pub fn unmap_memory_at_offset(&self, offset: u64, length: u64, done: Completion) {
        // No range check on the host's side.
        self.forward(done, |reply| ToService::UnmapMemory {
            offset,
            length,
            reply,
        });
    }

    /// Executes command buffers with 24-byte resource records, sent as 8-byte ones.
    pub fn exec_indirect2(
        &self,
        hold: Hold<'_>,
        args: ExecArgs<'_, P, CmdExecIndirectResource>,
        done: Completion,
    ) {
        let resource_count = args.resources.count();
        let cmd_buffer_count = args.cmd_buffers.count();
        // Each 24-byte record becomes 8 bytes: its reference, one flag byte, three zero bytes.
        let mut data = Vec::new();
        // One flag byte carried from record to record, so bits 3-7 carry over (the first
        // record's are 0; the service reads only bits 0-2).
        let mut flag_byte: u8 = 0;
        for resource in args.resources.iter() {
            // Each flag byte is cut to its low bit.
            let bits = u32::from(resource.invalidate_host) & 1
                | (u32::from(resource.invalidate_guest) & 1).wrapping_shl(1)
                | (u32::from(resource.synchronize) & 1).wrapping_shl(2);
            flag_byte = (flag_byte & !0b111) | bits.low_u8();
            data.extend_from_slice(&resource.reference.to_ne_bytes());
            data.push(flag_byte);
            data.extend_from_slice(&[0, 0, 0]);
        }
        data.extend_from_slice(args.cmd_buffers.as_bytes());
        self.exec(
            hold,
            cmd_buffer_count,
            data,
            args.stamp_index,
            args.fifo_event,
            args.fifo_event_value,
            resource_count,
            done,
        )
    }

    /// Executes command buffers with 8-byte resource records.
    pub fn exec_indirect3(
        &self,
        hold: Hold<'_>,
        args: ExecArgs<'_, P, CmdExecIndirectResource3>,
        done: Completion,
    ) {
        let data = command_data(&args.resources, &args.cmd_buffers);
        self.exec(
            hold,
            args.cmd_buffers.count(),
            data,
            args.stamp_index,
            args.fifo_event,
            args.fifo_event_value,
            args.resources.count(),
            done,
        )
    }

    /// Writes a compute pipeline's info into task memory.
    pub fn get_compute_info(
        &self,
        pipeline_state_reference: u32,
        max_key: u32,
        count: u32,
        offset: u64,
        done: Completion,
    ) {
        self.forward(done, |reply| ToService::GetComputeInfo {
            pipeline_state_reference,
            max_key,
            count,
            offset,
            reply,
        });
    }

    /// Writes a heap texture's size and alignment into task memory.
    pub fn heap_texture_size_and_align(
        &self,
        serializer_payload: &[u8],
        reply_virtual_offset: u64,
        reply_length: u64,
        done: Completion,
    ) {
        let data = serializer_payload.to_vec();
        self.forward(done, |reply| ToService::HeapTextureSizeAndAlign {
            data,
            reply_virtual_offset,
            reply_length,
            reply,
        });
    }

    /// Writes the info of command buffers into task memory.
    pub fn info_indirect(&self, args: ExecArgs<'_, P, CmdExecIndirectResource3>, done: Completion) {
        self.share_fifo_event(args.fifo_event, args.stamp_index);
        let data = command_data(&args.resources, &args.cmd_buffers);
        // The host never advances the FIFO's event value for an info. The reply carries only a
        // status, so no kernel-task invalidation.
        let cmd_buffer_count = args.cmd_buffers.count();
        let resource_count = args.resources.count();
        let stamp_index = args.stamp_index;
        let fifo_event_value_in = args.fifo_event_value.load(Ordering::Relaxed);
        self.forward(done, |reply| ToService::InfoIndirect {
            data,
            cmd_buffer_count,
            resource_count,
            stamp_index,
            fifo_event_value_in,
            reply,
        });
    }

    /// Resets a rasterization rate map.
    pub fn reset_rasterization_rate_map(&self, map: u32, done: Completion) {
        self.forward(done, |reply| ToService::ResetRasterizationRateMap {
            map,
            reply,
        });
    }
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;
    use crate::device::test_support::{Opts, TestDevice, device};
    use crate::host::HostPtr;
    use crate::memory::descriptor::{GuestPhysicalRange, MemoryMapDescriptor};
    use crate::memory::map::MemoryMap;
    use crate::num::ToU64;
    use crate::platform::device::GpuDevice;
    use crate::raise::death::DeathKind as Fault;
    use crate::rt::Held;
    use crate::tasks::launcher::{HostLink, ServiceSide};
    use crate::tasks::wire::ToHost;
    use crate::testkit::capture::catch_death;
    use crate::testkit::capture::expect_death;
    use crate::testkit::capture::expect_hang;
    use crate::testkit::launcher::ThreadLauncher;
    use crate::testkit::platform::FAKE_REGISTRY_ID;
    use crate::testkit::{FakePlatform, Step};
    use std::time::{Duration, Instant};
    use zerocopy::IntoBytes;

    type P = FakePlatform;

    /// Runs `f` as a FIFO command does: with a holder of its own, released at the
    /// command's end (the testkit has no real pool to push).
    fn in_command<T>(f: impl for<'u> FnOnce(Hold<'u>) -> T) -> T {
        let held = Held::new();
        let r = f(Hold::of(&held));
        drop(held);
        r
    }

    /// One message as the scripted service saw it: its selector, its scalar arguments in
    /// order, and its data.
    #[derive(Debug, Clone, PartialEq)]
    struct Msg {
        sel: &'static str,
        nums: Vec<u64>,
        data: Vec<u8>,
    }

    const NIL: u64 = u64::MAX;

    fn summarize(m: &ToService<P>) -> Msg {
        let u = |x: u32| u64::from(x);
        let (nums, data): (Vec<u64>, Vec<u8>) = match m {
            ToService::Handshake {
                task_root,
                task_size,
                task_id,
                configuration,
                features,
                memory_map,
                ..
            } => (
                vec![
                    u(*task_root),
                    *task_size,
                    u(*task_id),
                    configuration.registry_id,
                    u64::from(configuration.supports_argument_buffers),
                    memory_map
                        .as_ref()
                        .map_or(NIL, |m| m.ranges().len().to_u64()),
                ],
                format!("{features:?}").into_bytes(),
            ),
            ToService::FlushResources { .. } => (vec![], vec![]),
            ToService::UnmapMemory { offset, length, .. } => (vec![*offset, *length], vec![]),
            ToService::SetObjectList {
                object_list_offset,
                object_list_length,
                placement_list_offset,
                placement_list_length,
                ..
            } => (
                vec![
                    *object_list_offset,
                    u(*object_list_length),
                    *placement_list_offset,
                    u(*placement_list_length),
                ],
                vec![],
            ),
            ToService::GetComputeInfo {
                pipeline_state_reference,
                max_key,
                count,
                offset,
                ..
            } => (
                vec![
                    u(*pipeline_state_reference),
                    u(*max_key),
                    u(*count),
                    *offset,
                ],
                vec![],
            ),
            ToService::ReplacePhysical { physical, .. } => (vec![u(*physical)], vec![]),
            ToService::InvalidateResources {
                resources, count, ..
            } => (vec![u(*count)], resources.clone()),
            ToService::SynchronizeResources {
                resources,
                count,
                discard,
                ..
            } => (vec![u(*count), u64::from(*discard)], resources.clone()),
            ToService::DiscardResources {
                resources, count, ..
            } => (vec![u(*count)], resources.clone()),
            ToService::HeapTextureSizeAndAlign {
                data,
                reply_virtual_offset,
                reply_length,
                ..
            } => (vec![*reply_virtual_offset, *reply_length], data.clone()),
            ToService::DeleteObject { data, .. } => (vec![], data.clone()),
            ToService::DeleteResource { resource, .. } => (vec![u(*resource)], vec![]),
            ToService::ExecIndirect3 {
                data,
                cmd_buffer_count,
                resource_count,
                stamp_index,
                fifo_event_value_in,
                gpu_completion,
                ..
            } => (
                vec![
                    u(*cmd_buffer_count),
                    u(*resource_count),
                    u(*stamp_index),
                    *fifo_event_value_in,
                    *gpu_completion,
                ],
                data.clone(),
            ),
            ToService::InfoIndirect {
                data,
                cmd_buffer_count,
                resource_count,
                stamp_index,
                fifo_event_value_in,
                ..
            } => (
                vec![
                    u(*cmd_buffer_count),
                    u(*resource_count),
                    u(*stamp_index),
                    *fifo_event_value_in,
                ],
                data.clone(),
            ),
            ToService::ResetRasterizationRateMap { map, .. } => (vec![u(*map)], vec![]),
            ToService::SetFifoEvent { event, index } => (
                vec![u(*index), event.as_ref().map_or(NIL, |e| Handle::id(e).0)],
                vec![],
            ),
            ToService::FifoDeleted { stamp_index, .. } => (vec![u(*stamp_index)], vec![]),
        };
        Msg {
            sel: m.selector(),
            nums,
            data,
        }
    }

    /// The `Reply<u32>` of a message that has one.
    fn reply_u32(m: ToService<P>) -> Option<Reply<u32>> {
        match m {
            ToService::FlushResources { reply }
            | ToService::UnmapMemory { reply, .. }
            | ToService::SetObjectList { reply, .. }
            | ToService::GetComputeInfo { reply, .. }
            | ToService::ReplacePhysical { reply, .. }
            | ToService::InvalidateResources { reply, .. }
            | ToService::SynchronizeResources { reply, .. }
            | ToService::DiscardResources { reply, .. }
            | ToService::HeapTextureSizeAndAlign { reply, .. }
            | ToService::DeleteObject { reply, .. }
            | ToService::DeleteResource { reply, .. }
            | ToService::InfoIndirect { reply, .. }
            | ToService::ResetRasterizationRateMap { reply, .. }
            | ToService::FifoDeleted { reply, .. } => Some(reply),
            _ => None,
        }
    }

    /// What the service can do: send to the host.
    struct Ctx {
        host: Arc<dyn HostLink<P>>,
    }

    impl Ctx {
        fn complete(&self, token_id: u64, success: u32) {
            self.host.send(ToHost::CompleteAsyncToken {
                token_id,
                success,
                error: None,
            });
        }
    }

    /// The service in its plainest form: the handshake replies (1, 4242), an exec replies
    /// (1, nil, 0) and then completes its GPU token with 1, every other reply is 1.
    fn standard(ctx: &Ctx, m: ToService<P>) {
        match m {
            ToService::Handshake { reply, .. } => reply.call((1, 4242)),
            ToService::ExecIndirect3 {
                gpu_completion,
                reply,
                ..
            } => {
                reply.call((1, None, 0));
                ctx.complete(gpu_completion, 1);
            }
            ToService::SetFifoEvent { .. } => {}
            other => {
                if let Some(r) = reply_u32(other) {
                    r.call(1)
                }
            }
        }
    }

    type Behave = Arc<dyn Fn(&Ctx, ToService<P>) + Send + Sync>;

    struct Svc {
        ctx: Ctx,
        seen: Arc<Mutex<Vec<Msg>>>,
        behave: Behave,
    }

    impl ServiceSide<P> for Svc {
        fn on_message(&self, m: ToService<P>) {
            self.seen.lock().push(summarize(&m));
            (self.behave)(&self.ctx, m)
        }
    }

    struct H {
        platform: Arc<P>,
        /// The device the task is launched for.
        td: TestDevice,
        dev: Arc<crate::device::DeviceCore<P>>,
        launcher: ThreadLauncher,
        seen: Arc<Mutex<Vec<Msg>>>,
    }

    fn harness(behave: impl Fn(&Ctx, ToService<P>) + Send + Sync + 'static) -> H {
        harness_on(device(Opts::default()), behave)
    }

    /// A harness whose device has a kernel task (task 0, local, over guest page 0x55).
    fn kernel_harness(behave: impl Fn(&Ctx, ToService<P>) + Send + Sync + 'static) -> H {
        let td = device(Opts::default()).with_root_page();
        td.task_root_node(0x55, 0x4000);
        {
            {
                let hold = Hold::NONE;
                td.device
                    .create_task_id(hold, 0, 0x55, 0x4000, false)
                    .unwrap();
            }
        }
        harness_on(td, behave)
    }

    fn harness_on(
        td: TestDevice,
        behave: impl Fn(&Ctx, ToService<P>) + Send + Sync + 'static,
    ) -> H {
        let platform = td.platform.clone();
        let dev: Arc<crate::device::DeviceCore<P>> = Arc::clone(&td.device);
        let seen = Arc::new(Mutex::new(Vec::new()));
        let behave: Behave = Arc::new(behave);
        let (s2, b2) = (seen.clone(), behave);
        let launcher = ThreadLauncher::with_factory(Arc::new(move |_platform, host| {
            let svc: Arc<dyn ServiceSide<P>> = Arc::new(Svc {
                ctx: Ctx { host },
                seen: s2.clone(),
                behave: b2.clone(),
            });
            svc
        }));
        H {
            platform,
            td,
            dev,
            launcher,
            seen,
        }
    }

    impl H {
        fn try_task(&self) -> Option<Arc<RemoteTask<P>>> {
            RemoteTask::launch(
                &self.platform,
                Arc::downgrade(&self.dev),
                &self.launcher,
                7,
                0x10000,
                3,
            )
        }

        fn task(&self) -> Arc<RemoteTask<P>> {
            self.try_task().expect("handshake")
        }

        fn msgs(&self) -> Vec<Msg> {
            self.seen.lock().clone()
        }

        fn last(&self) -> Msg {
            self.msgs().last().cloned().unwrap()
        }

        fn wait_for(&self, n: usize) {
            wait_until(|| self.seen.lock().len() >= n);
        }

        fn event(&self) -> <P as Platform>::SharedEvent {
            self.platform.device().new_shared_event().unwrap()
        }
    }

    fn wait_until(f: impl Fn() -> bool) {
        let start = Instant::now();
        while !f() {
            assert!(start.elapsed() < Duration::from_secs(10), "timed out");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    type Statuses = Arc<Mutex<Vec<u32>>>;

    fn done() -> (Statuses, Completion) {
        let s: Statuses = Arc::new(Mutex::new(Vec::new()));
        let s2 = s.clone();
        (
            s,
            Arc::new(move |st: ExecStatus, _| s2.lock().push(st.raw())),
        )
    }

    /// Runs its closure when dropped: a probe for when a capture is released.
    struct OnDrop<F: FnMut()>(F);

    impl<F: FnMut()> Drop for OnDrop<F> {
        fn drop(&mut self) {
            (self.0)()
        }
    }

    /// [`done`] whose completion runs `on_release` when it is released: the FIFO's completion,
    /// which the token's release releases (and with it the FIFO it holds).
    fn probed_done(on_release: impl FnMut() + Send + Sync + 'static) -> (Statuses, Completion) {
        let s: Statuses = Arc::new(Mutex::new(Vec::new()));
        let s2 = s.clone();
        let probe = OnDrop(on_release);
        (
            s,
            Arc::new(move |st: ExecStatus, _| {
                let _held = &probe;
                s2.lock().push(st.raw());
            }),
        )
    }

    /// A gate the scripted service blocks on until the test opens it.
    #[derive(Clone, Default)]
    struct Gate(Arc<(Mutex<bool>, parking_lot::Condvar)>);

    impl Gate {
        fn open(&self) {
            *self.0.0.lock() = true;
            self.0.1.notify_all();
        }
        fn wait(&self) {
            let mut g = self.0.0.lock();
            while !*g {
                self.0.1.wait(&mut g);
            }
        }
    }

    /// A death in the service's own code.
    const SERVICE_DEATH: DeathSite = DeathSite {
        kind: Fault::Abort,
        file: "service",
        line: 0,
    };

    /// Kills the task on `sel` (a death in the service's own code).
    fn dies_on(sel: &'static str) -> impl Fn(&Ctx, ToService<P>) + Send + Sync + 'static {
        move |ctx, m| {
            if m.selector() == sel {
                crate::fatal!(SERVICE_DEATH)
            }
            standard(ctx, m)
        }
    }

    /// A crashed task: the service died on an unmap, whose completion saw 4.
    fn crashed(h: &H) -> Arc<RemoteTask<P>> {
        let t = h.task();
        let (s, d) = done();
        t.unmap_memory_at_offset(0, 0x4000, d);
        wait_until(|| *s.lock() == vec![4]);
        t
    }

    /// Gives the device a map of `(physical address, length, host address)` ranges.
    fn with_map(h: &H, ranges: &[(u64, u64, usize)]) {
        let mut d = MemoryMapDescriptor::new();
        for &(pa, len, va) in ranges {
            d.add_range(GuestPhysicalRange {
                physical_address: pa,
                physical_length: len,
                host_address: HostPtr::from_addr(va),
            });
        }
        h.td.set_memory_map(MemoryMap::new(h.platform.clone(), &d));
    }

    /// Resources then command buffers, contiguous, as the FIFO's payload holds them.
    fn payload<R: IntoBytes + zerocopy::Immutable>(
        resources: &[R],
        cmd: &[CmdExecIndirectCmdBuffer],
    ) -> (Vec<u8>, usize) {
        let mut v = resources.as_bytes().to_vec();
        let split = v.len();
        v.extend_from_slice(cmd.as_bytes());
        (v, split)
    }

    fn cmd() -> Vec<CmdExecIndirectCmdBuffer> {
        vec![CmdExecIndirectCmdBuffer {
            offset: 0x4000,
            length: 0x40,
        }]
    }

    fn res3() -> Vec<CmdExecIndirectResource3> {
        vec![
            CmdExecIndirectResource3 {
                reference: 9,
                flags: 5,
            },
            CmdExecIndirectResource3 {
                reference: 10,
                flags: 2,
            },
        ]
    }

    /// Runs an exec3 with `res3()` and `cmd()` at `stamp` with `event`.
    fn exec3(
        hold: Hold<'_>,
        t: &RemoteTask<P>,
        event: Option<&<P as Platform>::SharedEvent>,
        stamp: u32,
        value: &Arc<AtomicU64>,
        d: Completion,
    ) {
        let (bytes, split) = payload(&res3(), &cmd());
        let args = ExecArgs {
            cmd_buffers: Records::new(&bytes[split..], 1).unwrap(),
            resources: Records::new(&bytes[..split], 2).unwrap(),
            stamp_index: stamp,
            fifo_event: event,
            fifo_event_value: value,
        };
        t.exec_indirect3(hold, args, d)
    }

    mod launch {
        use super::*;

        /// The handshake carries the device's registry id, argument-buffer support, features and
        /// memory map, read in that order
        #[test]
        fn the_handshake_carries_the_devices_configuration() {
            let h = harness(standard);
            let t = h.task();
            assert_eq!(t.task_id(), 3);
            let m = &h.msgs()[0];
            assert_eq!(
                m.sel,
                "handshakeWithTaskRoot:taskSize:taskID:configuration:features:memoryMap:reply:"
            );
            let argument_buffers = u64::from(h.dev.supports_argument_buffers());
            assert_eq!(
                m.nums,
                vec![7, 0x10000, 3, FAKE_REGISTRY_ID, argument_buffers, NIL]
            );
            let features = h.dev.features().load();
            assert_eq!(m.data, format!("{features:?}").into_bytes());
            // The pid is stored after the completion, on the link's queue.
            wait_until(|| t.pid() == 4242);
            assert_eq!(t.shared_fifo_events.load(Ordering::SeqCst), 0);
            assert_eq!(t.fifo_events.len(), 64);
            assert_eq!(h.launcher.launched().len(), 1);
        }

        /// Released holding the last references, the handshake's reply drops its token while the
        /// task lives, then the task (whose drop invalidates the link), then the endpoint it holds
        #[test]
        fn the_reply_drops_its_token_then_the_task_then_the_endpoint() {
            let h = harness(standard);
            let t = h.task();
            wait_until(|| Arc::strong_count(&t) == 1);
            // The reply's own endpoint, which nothing else holds here, so its drop shows.
            let endpoint = Arc::new(Endpoint::new(Arc::downgrade(&h.dev)));
            // (what, the task's strong count) at each release.
            type Seen = Arc<Mutex<Vec<(&'static str, usize)>>>;
            let seen: Seen = Arc::new(Mutex::new(Vec::new()));
            let task = Arc::downgrade(&t);
            let probe = |what: &'static str| {
                let (seen, task) = (seen.clone(), task.clone());
                OnDrop(move || seen.lock().push((what, task.strong_count())))
            };
            // The handshake's token has no handler; this one has a probe, so its release shows.
            let token = {
                let p = probe("token");
                endpoint.completions().issue_with(Box::new(move |_, _| {
                    let _held = &p;
                }))
            };
            // An outstanding token the endpoint's table alone holds: released with the endpoint.
            {
                let p = probe("endpoint");
                drop(endpoint.completions().issue_with(Box::new(move |_, _| {
                    let _held = &p;
                })));
            }
            let reply = HandshakeReply {
                token: token.clone(),
                task: t,
                endpoint,
            };
            let reply = Reply::new(move |v: (u32, i32)| reply.invoke(v));
            // The link runs the reply once: the token completes and the pid is stored.
            reply.call((1, 77));
            assert_eq!(token.wait(), ExecStatus::OK);
            assert_eq!(task.upgrade().map(|t| t.pid()), Some(77));
            drop(token);
            assert!(seen.lock().is_empty());
            // Then releases it, holding the last references.
            drop(reply);
            assert_eq!(
                *seen.lock(),
                vec![("token", 1), ("endpoint", 0)],
                "the token while the task lives, then the task, then the endpoint"
            );
            // The task's drop invalidated the connection: the task ends.
            wait_until(|| h.launcher.launched()[0].ended());
        }

        /// The handshake takes token 1
        #[test]
        fn the_handshake_token_is_the_first() {
            let h = harness(standard);
            let t = h.task();
            let tok = t.endpoint.completions().issue();
            assert_eq!(tok.id(), 2);
            // Completed, or the task's drop would wait for it (Q33).
            t.endpoint.completions().complete(&tok, ExecStatus::OK);
        }

        /// The configuration carries the device's argument-buffer support
        #[test]
        fn configuration_reads_supports_argument_buffers() {
            let h = harness(standard);
            h.td.set_supports_argument_buffers(true);
            let _t = h.task();
            assert_eq!(h.msgs()[0].nums[4], 1);
            let h = harness(standard);
            h.td.set_supports_argument_buffers(false);
            let _t = h.task();
            assert_eq!(h.msgs()[0].nums[4], 0);
        }

        /// A handshake reply other than 1 is no task, and the task goes with the reply
        #[test]
        fn a_reply_other_than_one_is_nil_and_the_task_goes() {
            let h = harness(|ctx, m| match m {
                ToService::Handshake { reply, .. } => reply.call((2, 1)),
                other => standard(ctx, other),
            });
            assert!(h.try_task().is_none());
            // The task's drop invalidated the connection: the task ends.
            wait_until(|| h.launcher.launched()[0].ended());
            assert!(crate::testkit::launcher::death_of(&h.launcher.launched()[0]).is_none());
        }

        /// A death in the service during the handshake is no task
        #[test]
        fn a_service_death_during_the_handshake_is_nil() {
            let h = harness(dies_on(
                "handshakeWithTaskRoot:taskSize:taskID:configuration:features:memoryMap:reply:",
            ));
            assert!(h.try_task().is_none());
            let site = crate::testkit::launcher::death_of(&h.launcher.launched()[0]);
            assert_eq!(site, Some(SERVICE_DEATH));
        }

        /// The memory map crosses the handshake: the device's own map
        #[test]
        fn the_memory_map_crosses_the_handshake() {
            let h = harness(standard);
            with_map(&h, &[(0, 0x4000, 0x10_0000), (0x10000, 0x8000, 0x20_0000)]);
            let _t = h.task();
            assert_eq!(h.msgs()[0].nums[5], 2);
        }

        /// A map with a range that is not whole pages (its host address or its length), a NULL
        /// host address or a zero length makes no task: nothing is launched and the define fails
        #[test]
        fn a_map_that_is_not_whole_pages_makes_no_task() {
            // A guest page and half of one: 0x4000 and 0x2000 on a 16 KiB guest page.
            let (page, half) = (APV_PAGE_SIZE, APV_PAGE_SIZE / 2);
            for bad in [
                (0, page + half, 0x10_0000),
                (0, page, 0x10_0000 + half as usize),
                (0, 0, 0x10_0000),
                (0, page, 0),
            ] {
                let h = harness(standard);
                with_map(&h, &[(0x10_0000, 0x4000, 0x30_0000), bad]);
                assert!(h.try_task().is_none(), "{bad:x?}");
                assert!(h.msgs().is_empty());
                assert!(h.launcher.launched().is_empty());
            }
        }

        /// A device that is gone dies at its features
        #[test]
        fn a_device_that_is_gone_is_a_null_dereference_at_the_features() {
            expect_death(DeathKind::DeviceGone.site(), || {
                let h = harness(standard);
                let weak = {
                    let other = device(Opts::default());
                    let dev: Arc<crate::device::DeviceCore<P>> = Arc::clone(&other.device);
                    Arc::downgrade(&dev)
                };
                let _ = RemoteTask::launch(&h.platform, weak, &h.launcher, 1, 1, 1);
            });
        }

        /// The gone device's death names itself
        #[test]
        fn the_device_gone_death_is_named() {
            assert_eq!(
                DeathKind::DeviceGone.to_string(),
                "tasks.remote.device_gone"
            );
        }
    }

    mod drop_order {
        use super::*;

        /// The task's drop waits for the exec's GPU completion, then invalidates the link and
        /// releases its events
        #[test]
        fn drop_waits_for_the_gpu_completion_then_invalidates() {
            let gate = Gate::default();
            let g2 = gate.clone();
            let h = harness(move |ctx, m| match m {
                ToService::ExecIndirect3 {
                    gpu_completion,
                    reply,
                    ..
                } => {
                    reply.call((1, None, 0));
                    // The GPU completes later, on a thread of its own, as a Metal completion
                    // handler does: the exec's handler runs on the FIFO thread, which waits
                    // for it.
                    let (g, ctx) = (
                        g2.clone(),
                        Ctx {
                            host: Arc::clone(&ctx.host),
                        },
                    );
                    std::thread::spawn(move || {
                        g.wait();
                        ctx.complete(gpu_completion, 1);
                    });
                }
                other => standard(ctx, other),
            });
            let t = h.task();
            let ev = h.event();
            let v = Arc::new(AtomicU64::new(0));
            let (s, d) = done();
            // One FIFO command.
            in_command(|hold| exec3(hold, &t, Some(&ev), 0, &v, d));
            wait_until(|| ev.retain_count() == 2);
            // Only the test holds the task once the link has released the exec's reply.
            wait_until(|| Arc::strong_count(&t) == 1);
            let finished = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let f2 = finished.clone();
            let j = std::thread::spawn(move || {
                drop(t);
                f2.store(true, Ordering::SeqCst);
            });
            std::thread::sleep(Duration::from_millis(50));
            assert!(!finished.load(Ordering::SeqCst));
            assert!(!h.launcher.launched()[0].ended());
            gate.open();
            j.join().unwrap();
            assert_eq!(*s.lock(), vec![1]);
            // The events are released and the task ends on the invalidation.
            assert_eq!(ev.retain_count(), 1);
            wait_until(|| h.launcher.launched()[0].ended());
        }

        /// With nothing outstanding the drop invalidates at once
        #[test]
        fn drop_with_nothing_outstanding_invalidates_at_once() {
            let h = harness(standard);
            let t = h.task();
            wait_until(|| Arc::strong_count(&t) == 1);
            drop(t);
            wait_until(|| h.launcher.launched()[0].ended());
        }

        /// A pending reply holds the task.
        #[test]
        fn a_pending_reply_keeps_the_task_alive() {
            let held: Arc<Mutex<Vec<Reply<u32>>>> = Arc::new(Mutex::new(Vec::new()));
            let h2 = held.clone();
            let h = harness(move |ctx, m| match m {
                ToService::GetComputeInfo { reply, .. } => h2.lock().push(reply),
                other => standard(ctx, other),
            });
            let t = h.task();
            wait_until(|| Arc::strong_count(&t) == 1);
            let (s, d) = done();
            t.get_compute_info(1, 2, 3, 4, d);
            assert_eq!(Arc::strong_count(&t), 2);
            h.wait_for(2);
            held.lock()[0].call(1);
            wait_until(|| *s.lock() == vec![1]);
            wait_until(|| Arc::strong_count(&t) == 1);
        }

        /// The last reference dropped inside a reply while a token is outstanding: the drain
        /// waits on the link's queue for a `completeAsyncTokenID:` queued behind it.
        #[test]
        fn the_last_reference_in_a_reply_with_a_token_outstanding_hangs() {
            let clock = crate::testkit::VirtualClock::default();
            expect_hang(&clock, Duration::from_secs(60), || {
                let gate = Gate::default();
                let g2 = gate.clone();
                let stamp_token = Arc::new(Mutex::new(0u64));
                let st2 = stamp_token.clone();
                let h = harness(move |ctx, m| match m {
                    ToService::ExecIndirect3 {
                        gpu_completion,
                        reply,
                        ..
                    } => {
                        *st2.lock() = gpu_completion;
                        reply.call((1, None, 0));
                    }
                    ToService::DeleteResource { reply, .. } => {
                        g2.wait();
                        reply.call(1);
                        ctx.complete(*st2.lock(), 1);
                    }
                    other => standard(ctx, other),
                });
                let t = h.task();
                let v = Arc::new(AtomicU64::new(0));
                let (exec_done, d) = done();
                exec3(Hold::NONE, &t, None, 0, &v, d);
                wait_until(|| Arc::strong_count(&t) == 1);
                let (_s, d) = done();
                t.delete_resource(5, d);
                drop(t);
                gate.open();
                // The exec's completion never comes: the link's queue is stuck in the drain.
                wait_until(|| !exec_done.lock().is_empty());
            });
        }
    }

    /// A forwarding command's reply drops its token before the task.
    mod forward_reply_release {
        use super::*;

        /// The FIFO has let go of the task (its command ended) and the task was deleted,
        /// so the delete's reply holds the task's last reference and the token's: the link's
        /// release of the reply releases the token (the FIFO's completion goes) while the task
        /// still holds its events, and only then drops the task.
        #[test]
        fn the_token_goes_before_the_tasks_drop() {
            let gate = Gate::default();
            let g2 = gate.clone();
            let h = harness(move |ctx, m| match m {
                ToService::DeleteResource { reply, .. } => {
                    g2.wait();
                    reply.call(1);
                }
                other => standard(ctx, other),
            });
            let t = h.task();
            let ev = Arc::new(h.event());
            // One FIFO command.
            in_command(|_| t.share_fifo_event(Some(&ev), 0));
            // The handshake's reply is released, and so is the handle the service was sent.
            wait_until(|| Arc::strong_count(&t) == 1 && ev.retain_count() == 2);
            let seen: Arc<Mutex<Vec<(usize, i64)>>> = Arc::new(Mutex::new(Vec::new()));
            let (task, ev2, seen2) = (Arc::downgrade(&t), ev.clone(), seen.clone());
            let (s, d) =
                probed_done(move || seen2.lock().push((task.strong_count(), ev2.retain_count())));
            t.delete_resource(5, d);
            h.wait_for(3);
            drop(t);
            gate.open();
            wait_until(|| !seen.lock().is_empty());
            assert_eq!(*s.lock(), vec![1]);
            // The completion went while the reply still held the task, whose drop had not yet
            // released the event.
            assert_eq!(*seen.lock(), vec![(1, 2)]);
            // Then the task's drop: the event goes and the task ends on the invalidation.
            wait_until(|| ev.retain_count() == 1);
            wait_until(|| h.launcher.launched()[0].ended());
        }

        /// With the exec's GPU completion outstanding, the task's drop blocks in the drain on the
        /// link's queue for good; the delete's completion (and the FIFO it holds) was released
        /// before.
        #[test]
        fn the_completion_goes_before_the_drop_blocks_in_drain() {
            let released: Arc<Mutex<Option<usize>>> = Arc::new(Mutex::new(None));
            let r2 = released.clone();
            let clock = crate::testkit::VirtualClock::default();
            expect_hang(&clock, Duration::from_secs(60), move || {
                let gate = Gate::default();
                let g2 = gate.clone();
                let stamp_token = Arc::new(Mutex::new(0u64));
                let st2 = stamp_token.clone();
                let h = harness(move |ctx, m| match m {
                    ToService::ExecIndirect3 {
                        gpu_completion,
                        reply,
                        ..
                    } => {
                        *st2.lock() = gpu_completion;
                        reply.call((1, None, 0));
                    }
                    ToService::DeleteResource { reply, .. } => {
                        g2.wait();
                        reply.call(1);
                        ctx.complete(*st2.lock(), 1);
                    }
                    other => standard(ctx, other),
                });
                let t = h.task();
                let v = Arc::new(AtomicU64::new(0));
                let (exec_done, d) = done();
                exec3(Hold::NONE, &t, None, 0, &v, d);
                wait_until(|| Arc::strong_count(&t) == 1);
                let task = Arc::downgrade(&t);
                let r3 = r2.clone();
                let (_s, d) = probed_done(move || *r3.lock() = Some(task.strong_count()));
                t.delete_resource(5, d);
                drop(t);
                gate.open();
                wait_until(|| r2.lock().is_some());
                // The exec's completion never comes: the link's queue is stuck in the drain.
                wait_until(|| !exec_done.lock().is_empty());
            });
            // Released while the reply still held the task, before its drop ran.
            assert_eq!(*released.lock(), Some(1));
        }
    }

    mod teardown {
        use super::*;

        /// Teardown sends nothing and leaves the task running
        #[test]
        fn is_a_no_op() {
            let h = harness(standard);
            let t = h.task();
            t.teardown();
            std::thread::sleep(Duration::from_millis(10));
            assert_eq!(h.msgs().len(), 1);
            assert!(!h.launcher.launched()[0].ended());
        }
    }

    mod flush_resources {
        use super::*;

        /// The flush blocks until the service's reply, whose status is ignored
        #[test]
        fn blocks_until_the_service_replies() {
            let gate = Gate::default();
            let g2 = gate.clone();
            let h = harness(move |ctx, m| match m {
                ToService::FlushResources { reply } => {
                    g2.wait();
                    reply.call(2);
                }
                other => standard(ctx, other),
            });
            let t = h.task();
            let finished = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let (t2, f2) = (t.clone(), finished.clone());
            let j = std::thread::spawn(move || {
                t2.flush_resources();
                f2.store(true, Ordering::SeqCst);
            });
            h.wait_for(2);
            assert_eq!(h.last().sel, "flushResourcesWithReply:");
            std::thread::sleep(Duration::from_millis(30));
            assert!(!finished.load(Ordering::SeqCst));
            gate.open();
            // The result (2 here) is ignored.
            j.join().unwrap();
        }

        /// After a crash the flush returns at once
        #[test]
        fn returns_at_once_after_a_crash() {
            let h = harness(dies_on("unmapMemoryAtOffset:length:reply:"));
            let t = crashed(&h);
            t.flush_resources();
        }
    }

    mod map_memory {
        use super::*;

        /// Map (57) is a no-op success with a memory map and for remote tasks
        #[test]
        fn succeeds_at_once_without_a_message_even_when_dead() {
            let h = harness(dies_on("unmapMemoryAtOffset:length:reply:"));
            let t = h.task();
            let (s, d) = done();
            t.map_memory_at_offset(0, 0x4000, d.clone());
            assert_eq!(*s.lock(), vec![1]);
            let t = {
                drop(t);
                crashed(&h)
            };
            let n = h.msgs().len();
            let (s, d) = done();
            t.map_memory_at_offset(0, 0x4000, d);
            assert_eq!(*s.lock(), vec![1]);
            assert_eq!(h.msgs().len(), n);
        }
    }

    /// Calls `f` on a task, waits for its completion and returns the message and the statuses.
    fn forwarded(
        behave: impl Fn(&Ctx, ToService<P>) + Send + Sync + 'static,
        f: impl FnOnce(&RemoteTask<P>, Completion),
    ) -> (Msg, Vec<u32>) {
        let h = harness(behave);
        let t = h.task();
        let (s, d) = done();
        f(&t, d);
        wait_until(|| !s.lock().is_empty());
        let got = s.lock().clone();
        (h.last(), got)
    }

    /// Every forwarding method, after a crash: its completion runs with 4 inside the call and
    /// nothing reaches the service.
    fn crashed_forward(f: impl FnOnce(&RemoteTask<P>, Completion)) {
        let h = harness(dies_on("unmapMemoryAtOffset:length:reply:"));
        let t = crashed(&h);
        let n = h.msgs().len();
        let (s, d) = done();
        f(&t, d);
        assert_eq!(*s.lock(), vec![4]);
        std::thread::sleep(Duration::from_millis(5));
        assert_eq!(h.msgs().len(), n);
    }

    /// Replies with the message's first scalar, to show the status is forwarded unvalidated.
    fn echo_first(ctx: &Ctx, m: ToService<P>) {
        let first = summarize(&m).nums.first().copied().unwrap_or(1).low_u32();
        match reply_u32(m) {
            Some(r) => r.call(first),
            None => {
                let _ = ctx;
            }
        }
    }

    fn echo(ctx: &Ctx, m: ToService<P>) {
        match m {
            ToService::Handshake { reply, .. } => reply.call((1, 1)),
            other => echo_first(ctx, other),
        }
    }

    /// Unmap forwards its range and completes with the service's status
    #[test]
    fn unmap_memory() {
        let (m, s) = forwarded(echo, |t, d| t.unmap_memory_at_offset(9, 0x4000, d));
        assert_eq!(m.sel, "unmapMemoryAtOffset:length:reply:");
        assert_eq!(m.nums, vec![9, 0x4000]);
        assert_eq!(s, vec![9]);
        crashed_forward(|t, d| t.unmap_memory_at_offset(0, 1, d));
    }

    /// Setting the object lists forwards the four values
    #[test]
    fn set_object_list_offset() {
        let (m, s) = forwarded(echo, |t, d| t.set_object_list_offset(2, 24, 0x8000, 32, d));
        assert_eq!(
            m.sel,
            "setObjectListOffset:objectListLength:placementListOffset:placementListLength:reply:"
        );
        assert_eq!(m.nums, vec![2, 24, 0x8000, 32]);
        assert_eq!(s, vec![2]);
        crashed_forward(|t, d| t.set_object_list_offset(0, 0, 0, 0, d));
    }

    /// Invalidation forwards its records
    #[test]
    fn invalidate_resources() {
        let recs = [
            ResourceInvalidation {
                reference: 4,
                clear_host_valid: 1,
                set_host_valid: 0,
                clear_guest_valid: 0,
                set_guest_valid: 1,
            },
            ResourceInvalidation::default(),
        ];
        let bytes = recs.as_bytes().to_vec();
        let b2 = bytes.clone();
        let (m, s) = forwarded(echo, move |t, d| {
            t.invalidate_resources(Records::new(&b2, 2).unwrap(), d)
        });
        assert_eq!(m.sel, "invalidateResources:count:reply:");
        assert_eq!(m.nums, vec![2]);
        assert_eq!(m.data, bytes);
        assert_eq!(s, vec![2]);
        crashed_forward(|t, d| t.invalidate_resources(Records::empty(), d));
    }

    /// Synchronization forwards its records and the discard flag
    #[test]
    fn synchronize_resources() {
        // Only `count * 4` bytes cross, whatever follows them in the payload.
        let bytes = vec![1u8, 0, 0, 0, 2, 0, 0, 0, 0xff];
        let b2 = bytes.clone();
        let (m, s) = forwarded(standard, move |t, d| {
            t.synchronize_resources(Records::<ResourceReference>::new(&b2, 2).unwrap(), true, d)
        });
        assert_eq!(m.sel, "synchronizeResources:count:discard:reply:");
        assert_eq!(m.nums, vec![2, 1]);
        assert_eq!(m.data, bytes[..8].to_vec());
        assert_eq!(s, vec![1]);
        let (m, _) = forwarded(standard, |t, d| {
            t.synchronize_resources(Records::empty(), false, d)
        });
        assert_eq!(m.nums, vec![0, 0]);
        crashed_forward(|t, d| t.synchronize_resources(Records::empty(), false, d));
    }

    /// Discard forwards its records
    #[test]
    fn discard_resources() {
        let bytes = vec![3u8, 0, 0, 0];
        let b2 = bytes.clone();
        let (m, s) = forwarded(echo, move |t, d| {
            t.discard_resources(Records::new(&b2, 1).unwrap(), d)
        });
        assert_eq!(m.sel, "discardResources:count:reply:");
        assert_eq!(m.nums, vec![1]);
        assert_eq!(m.data, bytes);
        assert_eq!(s, vec![1]);
        crashed_forward(|t, d| t.discard_resources(Records::empty(), d));
    }

    /// Compute info forwards its four values
    #[test]
    fn get_compute_info() {
        let (m, s) = forwarded(echo, |t, d| t.get_compute_info(6, 7, 8, 0x9000, d));
        assert_eq!(m.sel, "getComputeInfo:maxKey:count:offset:reply:");
        assert_eq!(m.nums, vec![6, 7, 8, 0x9000]);
        assert_eq!(s, vec![6]);
        crashed_forward(|t, d| t.get_compute_info(0, 0, 0, 0, d));
    }

    /// Replace-physical forwards its reference; the status is unvalidated
    #[test]
    fn replace_physical() {
        let (m, s) = forwarded(echo, |t, d| t.replace_physical(0xdead, d));
        assert_eq!(m.sel, "replacePhysical:reply:");
        assert_eq!(m.nums, vec![0xdead]);
        // Any value the service sends is the completion's status, unvalidated.
        assert_eq!(s, vec![0xdead]);
        crashed_forward(|t, d| t.replace_physical(1, d));
    }

    /// Delete forwards its reference
    #[test]
    fn delete_resource() {
        let (m, s) = forwarded(echo, |t, d| t.delete_resource(3, d));
        assert_eq!(m.sel, "deleteResource:reply:");
        assert_eq!(m.nums, vec![3]);
        assert_eq!(s, vec![3]);
        crashed_forward(|t, d| t.delete_resource(1, d));
    }

    /// Heap texture size and align forwards its payload and reply range
    #[test]
    fn heap_texture_size_and_align() {
        let (m, s) = forwarded(standard, |t, d| {
            t.heap_texture_size_and_align(&[1, 2, 3], 0x8000, 16, d)
        });
        assert_eq!(
            m.sel,
            "heapTextureSizeAndAlignWithSerializedData:replyVirtualOffset:replyLength:reply:"
        );
        assert_eq!(m.nums, vec![0x8000, 16]);
        assert_eq!(m.data, vec![1, 2, 3]);
        assert_eq!(s, vec![1]);
        crashed_forward(|t, d| t.heap_texture_size_and_align(&[], 0, 0, d));
    }

    /// Object deletion forwards its payload
    #[test]
    fn delete_object_with_serialized_data() {
        let (m, s) = forwarded(standard, |t, d| {
            t.delete_object_with_serialized_data(&[9, 8], d)
        });
        assert_eq!(m.sel, "deleteObjectWithSerializedData:reply:");
        assert_eq!(m.data, vec![9, 8]);
        assert_eq!(s, vec![1]);
        crashed_forward(|t, d| t.delete_object_with_serialized_data(&[], d));
    }

    /// Rate-map reset forwards its reference
    #[test]
    fn reset_rasterization_rate_map() {
        let (m, s) = forwarded(echo, |t, d| t.reset_rasterization_rate_map(12, d));
        assert_eq!(m.sel, "resetRasterizationRateMap:reply:");
        assert_eq!(m.nums, vec![12]);
        assert_eq!(s, vec![12]);
        crashed_forward(|t, d| t.reset_rasterization_rate_map(1, d));
    }

    mod fifo_deleted {
        use super::*;

        /// A FIFO's deletion forwards, then unmarks its index: its next command shares its
        /// event again
        #[test]
        fn forwards_then_clears_the_stamp_bit() {
            let h = harness(echo);
            let t = h.task();
            let ev = h.event();
            t.share_fifo_event(Some(&ev), 2);
            assert_eq!(t.shared_fifo_events.load(Ordering::SeqCst), 0b100);
            let (s, d) = done();
            t.fifo_deleted(2, d);
            assert_eq!(t.shared_fifo_events.load(Ordering::SeqCst), 0);
            wait_until(|| *s.lock() == vec![2]);
            h.wait_for(3);
            assert_eq!(h.msgs()[2].sel, "fifoDeleted:reply:");
            assert_eq!(h.msgs()[2].nums, vec![2]);
            // The next exec of that FIFO sends its event again.
            t.share_fifo_event(Some(&ev), 2);
            h.wait_for(4);
            assert_eq!(h.msgs()[3].sel, "setFifoEvent:forIndex:");
        }

        /// A deletion unmarks only its own index, on a dead task too
        #[test]
        fn clears_only_its_bit_and_even_when_dead() {
            let h = harness(dies_on("unmapMemoryAtOffset:length:reply:"));
            let t = h.task();
            t.share_fifo_event(None, 0);
            t.share_fifo_event(None, 1);
            let t = {
                let (s, d) = done();
                t.unmap_memory_at_offset(0, 1, d);
                wait_until(|| *s.lock() == vec![4]);
                t
            };
            let (s, d) = done();
            t.fifo_deleted(1, d);
            assert_eq!(*s.lock(), vec![4]);
            assert_eq!(t.shared_fifo_events.load(Ordering::SeqCst), 0b1);
        }
    }

    mod share_fifo_event {
        use super::*;

        /// The event's handle crosses once per index
        #[test]
        fn sends_the_events_handle_once_per_index() {
            let h = harness(standard);
            let t = h.task();
            let ev = h.event();
            // One FIFO command.
            in_command(|_| {
                t.share_fifo_event(Some(&ev), 5);
                t.share_fifo_event(Some(&ev), 5);
            });
            h.wait_for(2);
            std::thread::sleep(Duration::from_millis(5));
            let msgs = h.msgs();
            assert_eq!(msgs.len(), 2);
            assert_eq!(msgs[1].sel, "setFifoEvent:forIndex:");
            assert_eq!(msgs[1].nums[0], 5);
            assert_ne!(msgs[1].nums[1], NIL);
            assert_eq!(h.platform.call_log().count("newSharedEventHandle"), 1);
            wait_until(|| ev.retain_count() == 2);
            assert_eq!(t.shared_fifo_events.load(Ordering::SeqCst), 1 << 5);
        }

        /// A nil event crosses as a nil handle
        #[test]
        fn a_nil_event_sends_a_nil_handle() {
            let h = harness(standard);
            let t = h.task();
            t.share_fifo_event(None, 0);
            h.wait_for(2);
            assert_eq!(h.msgs()[1].nums, vec![0, NIL]);
            assert_eq!(h.platform.call_log().count("newSharedEventHandle"), 0);
            assert_eq!(t.shared_fifo_events.load(Ordering::SeqCst), 1);
        }

        /// A new event releases the one the slot kept
        #[test]
        fn a_new_event_releases_the_old_one() {
            let h = harness(standard);
            let t = h.task();
            let (a, b) = (h.event(), h.event());
            // One FIFO command each.
            in_command(|_| t.share_fifo_event(Some(&a), 1));
            t.fifo_deleted(1, done().1);
            in_command(|_| t.share_fifo_event(Some(&b), 1));
            // The handles in flight hold their events until the service drops the messages.
            wait_until(|| a.retain_count() == 1);
            wait_until(|| b.retain_count() == 2);
            assert!(t.fifo_events[1].lock().as_ref().map(Handle::id) == Some(b.id()));
        }

        /// An index of 64 dies
        #[test]
        fn an_index_of_64_dies() {
            let h = harness(standard);
            let t = h.task();
            let r = catch_death(|| t.share_fifo_event(None, 64));
            assert_eq!(r.err(), Some(DeathKind::StampIndex.site()));
        }

        /// The index's bit is formed before the check, its shift taken mod 64: with bit 0 set,
        /// index 64 is taken as shared and skips the check
        #[test]
        fn an_index_of_64_with_bit_0_set_is_skipped() {
            let h = harness(standard);
            let t = h.task();
            t.share_fifo_event(None, 0);
            h.wait_for(2);
            t.share_fifo_event(None, 64);
            std::thread::sleep(Duration::from_millis(5));
            assert_eq!(h.msgs().len(), 2);
        }

        /// A FIFO busy in its own slot (here, slot 0 held as if its handle's creation were
        /// running) never holds another FIFO's slot back
        #[test]
        fn one_slot_in_flight_does_not_serialise_another() {
            let h = harness(standard);
            let t = h.task();
            let ev = h.event();
            let held = t.fifo_events[0].lock();
            std::thread::scope(|s| {
                let j = s.spawn(|| t.share_fifo_event(Some(&ev), 1));
                wait_until(|| j.is_finished());
            });
            drop(held);
            assert_eq!(t.shared_fifo_events.load(Ordering::SeqCst), 0b10);
            h.wait_for(2);
            assert_eq!(h.msgs()[1].nums[0], 1);
        }

        /// A raise from the handle's creation dies at the send, the event kept before it and
        /// nothing sent: the index stays unmarked
        #[test]
        fn a_raise_from_new_shared_event_handle_dies_at_the_send() {
            let h = harness(standard);
            let t = h.task();
            let ev = h.event();
            h.platform.script().push(
                "newSharedEventHandle",
                Step::raise("NSGenericException", "x"),
            );
            let r = catch_death(|| t.share_fifo_event(Some(&ev), 3));
            assert_eq!(
                r.err().map(|s| (s.kind, s.file)),
                Some((Fault::UncaughtObjc, "newSharedEventHandle"))
            );
            assert!(t.fifo_events[3].lock().is_some());
            assert_eq!(t.shared_fifo_events.load(Ordering::SeqCst), 0);
        }
    }

    mod exec {
        use super::*;
        use crate::tasks::test_support::Lines;

        /// A reply of 1 advances the FIFO's event value before the exec returns; the completion
        /// comes from the service
        #[test]
        fn success_advances_the_counter_and_completes_through_the_service() {
            let h = harness(standard);
            let t = h.task();
            let ev = h.event();
            let v = Arc::new(AtomicU64::new(41));
            let (s, d) = done();
            exec3(Hold::NONE, &t, Some(&ev), 2, &v, d);
            // The counter moved when the reply came, before the exec returned.
            assert_eq!(v.load(Ordering::SeqCst), 42);
            wait_until(|| *s.lock() == vec![1]);
            let msgs = h.msgs();
            assert_eq!(msgs[1].sel, "setFifoEvent:forIndex:");
            assert_eq!(
                msgs[2].sel,
                "execIndirect3WithData:cmdBufferCount:resourceCount:stampIndex:fifoEventValueIn:gpuCompletion:reply:"
            );
            // Tokens: 1 the handshake, 2 the GPU wait, 3 the stamp (its id crosses).
            assert_eq!(msgs[2].nums, vec![1, 2, 2, 41, 3]);
            let (bytes, _) = payload(&res3(), &cmd());
            assert_eq!(msgs[2].data, bytes);
            // No backings: the kernel task is not asked (the device has none: asking raises).
        }

        /// A failing reply completes at once and does not advance
        #[test]
        fn a_failure_completes_at_once_without_advancing() {
            let h = harness(|ctx, m| match m {
                ToService::ExecIndirect3 { reply, .. } => reply.call((2, None, 0)),
                other => standard(ctx, other),
            });
            let t = h.task();
            let v = Arc::new(AtomicU64::new(0));
            let (s, d) = done();
            exec3(Hold::NONE, &t, None, 0, &v, d);
            assert_eq!(*s.lock(), vec![2]);
            assert_eq!(v.load(Ordering::SeqCst), 0);
        }

        /// A 4 in the reply is left to the crash: the stamp token is not completed by the exec.
        #[test]
        fn a_reply_of_four_does_not_complete() {
            let h = harness(|ctx, m| match m {
                ToService::ExecIndirect3 { reply, .. } => reply.call((4, None, 0)),
                other => standard(ctx, other),
            });
            let t = h.task();
            let v = Arc::new(AtomicU64::new(0));
            let (s, d) = done();
            exec3(Hold::NONE, &t, None, 0, &v, d);
            std::thread::sleep(Duration::from_millis(10));
            assert!(s.lock().is_empty());
            assert_eq!(v.load(Ordering::SeqCst), 0);
            // The stamp token is still outstanding: dropping the task would block in the drain
            // (Q33), so the test leaves it, as a FIFO that never completes would.
            std::mem::forget(t);
        }

        /// A crash during the exec completes it with 4. The crash signals the exec's GPU token,
        /// which wakes the FIFO thread, before the token whose handler is the completion, so the
        /// exec may return before its completion runs
        /// (`the_exec_reply_drops_gpu_token_then_token_then_the_task`): here the completion takes
        /// 50 ms, and the test waits for it
        #[test]
        fn a_crash_during_the_exec_completes_four() {
            let h = harness(dies_on(
                "execIndirect3WithData:cmdBufferCount:resourceCount:stampIndex:fifoEventValueIn:gpuCompletion:reply:",
            ));
            let t = h.task();
            let v = Arc::new(AtomicU64::new(0));
            let s: Statuses = Arc::new(Mutex::new(Vec::new()));
            let s2 = s.clone();
            let d: Completion = Arc::new(move |st: ExecStatus, _| {
                std::thread::sleep(Duration::from_millis(50));
                s2.lock().push(st.raw());
            });
            exec3(Hold::NONE, &t, None, 0, &v, d);
            wait_until(|| !s.lock().is_empty());
            assert_eq!(*s.lock(), vec![4]);
            assert_eq!(v.load(Ordering::SeqCst), 0);
            // The kernel task is not asked (the device has none: asking raises).
        }

        /// After a crash both tokens are born 4; the completion runs inside the call.
        #[test]
        fn after_a_crash_both_tokens_are_born_four() {
            let h = harness(dies_on("unmapMemoryAtOffset:length:reply:"));
            let t = crashed(&h);
            let n = h.msgs().len();
            let v = Arc::new(AtomicU64::new(0));
            let (s, d) = done();
            exec3(Hold::NONE, &t, None, 0, &v, d);
            assert_eq!(*s.lock(), vec![4]);
            assert_eq!(v.load(Ordering::SeqCst), 0);
            std::thread::sleep(Duration::from_millis(5));
            assert_eq!(h.msgs().len(), n);
        }

        /// The reply's backings go to the kernel task, up to its count
        #[test]
        fn the_replys_backings_go_to_the_kernel_task_up_to_its_count() {
            let h = kernel_harness(|ctx, m| match m {
                ToService::ExecIndirect3 {
                    gpu_completion,
                    reply,
                    ..
                } => {
                    reply.call((1, Some([7u32, 8, 9].as_bytes().to_vec()), 2));
                    ctx.complete(gpu_completion, 1);
                }
                other => standard(ctx, other),
            });
            let t = h.task();
            let lines = Lines::capture();
            let v = Arc::new(AtomicU64::new(0));
            let (s, d) = done();
            exec3(Hold::NONE, &t, None, 0, &v, d);
            wait_until(|| !s.lock().is_empty());
            // The kernel task looks each one up (it has none of them).
            assert_eq!(
                backing_misses(&lines),
                [
                    "shared texture backing 7 to invalidate not found",
                    "shared texture backing 8 to invalidate not found"
                ]
            );
        }

        /// A count above the reply's ids passes the ids there are (the service always sends a
        /// matching count).
        #[test]
        fn backing_count_larger_than_data() {
            let h = kernel_harness(|ctx, m| match m {
                ToService::ExecIndirect3 {
                    gpu_completion,
                    reply,
                    ..
                } => {
                    reply.call((1, Some(vec![5, 0, 0, 0]), 2));
                    ctx.complete(gpu_completion, 1);
                }
                other => standard(ctx, other),
            });
            let t = h.task();
            let lines = Lines::capture();
            let v = Arc::new(AtomicU64::new(0));
            let (s, d) = done();
            exec3(Hold::NONE, &t, None, 0, &v, d);
            wait_until(|| !s.lock().is_empty());
            assert_eq!(
                backing_misses(&lines),
                ["shared texture backing 5 to invalidate not found"]
            );
        }

        /// The kernel task's lines for backings it does not have, in order.
        fn backing_misses(lines: &Lines) -> Vec<String> {
            lines
                .take()
                .into_iter()
                .filter(|l| l.starts_with("shared texture backing "))
                .collect()
        }

        /// The completion token completed with 3 by the service before the reply of 2 keeps its
        /// 3.
        #[test]
        fn a_completion_before_a_failing_reply_wins() {
            let h = harness(|ctx, m| match m {
                ToService::ExecIndirect3 {
                    gpu_completion,
                    reply,
                    ..
                } => {
                    ctx.complete(gpu_completion, 3);
                    reply.call((2, None, 0));
                }
                other => standard(ctx, other),
            });
            let t = h.task();
            let v = Arc::new(AtomicU64::new(0));
            let (s, d) = done();
            exec3(Hold::NONE, &t, None, 0, &v, d);
            assert_eq!(*s.lock(), vec![3]);
            assert_eq!(v.load(Ordering::SeqCst), 0);
        }

        /// A reply of 0 (NotComplete) never ends the wait.
        #[test]
        fn an_exec_reply_of_zero_hangs_the_fifo() {
            let clock = crate::testkit::VirtualClock::default();
            expect_hang(&clock, Duration::from_secs(60), || {
                let h = harness(|ctx, m| match m {
                    ToService::ExecIndirect3 { reply, .. } => reply.call((0, None, 0)),
                    other => standard(ctx, other),
                });
                let t = h.task();
                let v = Arc::new(AtomicU64::new(0));
                let (_s, d) = done();
                exec3(Hold::NONE, &t, None, 0, &v, d);
            });
        }

        /// The GPU completion may arrive before the exec's own reply.
        #[test]
        fn the_completion_may_precede_the_reply() {
            let h = harness(|ctx, m| match m {
                ToService::ExecIndirect3 {
                    gpu_completion,
                    reply,
                    ..
                } => {
                    ctx.complete(gpu_completion, 1);
                    reply.call((1, None, 0));
                }
                other => standard(ctx, other),
            });
            let t = h.task();
            let v = Arc::new(AtomicU64::new(0));
            let (s, d) = done();
            exec3(Hold::NONE, &t, None, 0, &v, d);
            assert_eq!(*s.lock(), vec![1]);
            assert_eq!(v.load(Ordering::SeqCst), 1);
        }

        /// The exec's reply drops the GPU wait's token, then the completion token, and the task
        /// last. The service dies on the exec: the interruption signals both tokens 4 and the
        /// link then releases the reply unrun, holding the last references to both tokens (the
        /// FIFO thread's exec has returned) and to the task.
        #[test]
        fn the_exec_reply_drops_gpu_token_then_token_then_the_task() {
            let h = harness(dies_on(
                "execIndirect3WithData:cmdBufferCount:resourceCount:stampIndex:fifoEventValueIn:gpuCompletion:reply:",
            ));
            let t = h.task();
            wait_until(|| Arc::strong_count(&t) == 1);
            let v = Arc::new(AtomicU64::new(0));
            // At the FIFO completion's release: the counter's holders (the test's, and the GPU
            // wait's handler's while it lives), and the task's.
            let seen: Arc<Mutex<Vec<(usize, usize)>>> = Arc::new(Mutex::new(Vec::new()));
            let (value, task, seen2) = (Arc::downgrade(&v), Arc::downgrade(&t), seen.clone());
            let (s, probed) = probed_done(move || {
                seen2
                    .lock()
                    .push((value.strong_count(), task.strong_count()))
            });
            // The completion runs inside the crash on the link's queue; it holds that queue until
            // this (FIFO) thread's exec has returned, so the reply is released after it.
            let fifo_returned = Gate::default();
            let d: Completion = {
                let returned = fifo_returned.clone();
                Arc::new(move |st, e| {
                    probed(st, e);
                    returned.wait();
                })
            };
            exec3(Hold::NONE, &t, None, 0, &v, d);
            let task = Arc::downgrade(&t);
            drop(t);
            fifo_returned.open();
            wait_until(|| !seen.lock().is_empty());
            assert_eq!(*s.lock(), vec![4]);
            assert_eq!(v.load(Ordering::SeqCst), 0);
            // The GPU wait's handler was gone (only the test holds the counter) and the task was
            // still held when the completion token's handler, the FIFO's completion, went.
            assert_eq!(*seen.lock(), vec![(1, 1)]);
            // Then the task goes.
            wait_until(|| task.strong_count() == 0);
        }

        /// The FIFO's event value is read at the send, and advances wrapping
        #[test]
        fn the_counter_is_read_at_the_send() {
            let h = harness(standard);
            let t = h.task();
            let v = Arc::new(AtomicU64::new(u64::MAX));
            exec3(Hold::NONE, &t, None, 0, &v, done().1);
            // The advance wraps.
            assert_eq!(v.load(Ordering::SeqCst), 0);
            assert_eq!(h.msgs()[2].nums[3], u64::MAX);
        }
    }

    mod exec_indirect2 {
        use super::*;

        /// The remote exec2 carries flag bits 3-7 from record to record; each flag byte of the
        /// 24-byte record is cut to its low bit
        #[test]
        fn converts_each_record_to_eight_bytes() {
            let h = harness(standard);
            let t = h.task();
            let recs = vec![
                CmdExecIndirectResource {
                    reference: 9,
                    invalidate_host: 3,
                    invalidate_guest: 2,
                    synchronize: 1,
                    field7: 0xff,
                    field8: u64::MAX,
                    field10: u64::MAX,
                },
                CmdExecIndirectResource {
                    reference: 10,
                    invalidate_host: 0,
                    invalidate_guest: 1,
                    synchronize: 0,
                    ..CmdExecIndirectResource::default()
                },
            ];
            let (bytes, split) = payload(&recs, &cmd());
            let v = Arc::new(AtomicU64::new(0));
            let (s, d) = done();
            t.exec_indirect2(
                Hold::NONE,
                ExecArgs {
                    cmd_buffers: Records::new(&bytes[split..], 1).unwrap(),
                    resources: Records::new(&bytes[..split], 2).unwrap(),
                    stamp_index: 1,
                    fifo_event: None,
                    fifo_event_value: &v,
                },
                d,
            );
            wait_until(|| *s.lock() == vec![1]);
            let m = &h.msgs()[2];
            assert_eq!(m.nums[..4], [1, 2, 1, 0]);
            let mut want = vec![9, 0, 0, 0, 0b101, 0, 0, 0, 10, 0, 0, 0, 0b010, 0, 0, 0];
            want.extend_from_slice(cmd().as_bytes());
            assert_eq!(m.data, want);
            assert_eq!(v.load(Ordering::SeqCst), 1);
        }

        /// With no resources only the command buffers cross
        #[test]
        fn no_resources_sends_only_the_command_buffers() {
            let h = harness(standard);
            let t = h.task();
            let c = cmd();
            let v = Arc::new(AtomicU64::new(0));
            t.exec_indirect2(
                Hold::NONE,
                ExecArgs {
                    cmd_buffers: Records::new(c.as_bytes(), 1).unwrap(),
                    resources: Records::empty(),
                    stamp_index: 0,
                    fifo_event: None,
                    fifo_event_value: &v,
                },
                done().1,
            );
            assert_eq!(h.msgs()[2].data, c.as_bytes().to_vec());
        }
    }

    mod info_indirect {
        use super::*;

        /// The remote info does not advance the FIFO's event value and invalidates no
        /// kernel-task backing
        #[test]
        fn forwards_without_advancing_or_invalidating() {
            let h = harness(|ctx, m| match m {
                ToService::InfoIndirect { reply, .. } => reply.call(3),
                other => standard(ctx, other),
            });
            let t = h.task();
            let ev = h.event();
            let (bytes, split) = payload(&res3(), &cmd());
            let v = Arc::new(AtomicU64::new(5));
            let (s, d) = done();
            t.info_indirect(
                ExecArgs {
                    cmd_buffers: Records::new(&bytes[split..], 1).unwrap(),
                    resources: Records::new(&bytes[..split], 2).unwrap(),
                    stamp_index: 4,
                    fifo_event: Some(&ev),
                    fifo_event_value: &v,
                },
                d,
            );
            wait_until(|| *s.lock() == vec![3]);
            let msgs = h.msgs();
            assert_eq!(msgs[1].sel, "setFifoEvent:forIndex:");
            assert_eq!(msgs[1].nums[0], 4);
            assert_eq!(
                msgs[2].sel,
                "infoIndirectWithData:cmdBufferCount:resourceCount:stampIndex:fifoEventValueIn:reply:"
            );
            assert_eq!(msgs[2].nums, vec![1, 2, 4, 5]);
            assert_eq!(msgs[2].data, bytes);
            assert_eq!(v.load(Ordering::SeqCst), 5);
            // The kernel task is not asked (the device has none: asking raises).
        }

        /// After a crash the info completes with 4 inside the call
        #[test]
        fn after_a_crash_completes_four_inside_the_call() {
            let h = harness(dies_on("unmapMemoryAtOffset:length:reply:"));
            let t = crashed(&h);
            let c = cmd();
            let v = Arc::new(AtomicU64::new(0));
            let (s, d) = done();
            t.info_indirect(
                ExecArgs {
                    cmd_buffers: Records::new(c.as_bytes(), 1).unwrap(),
                    resources: Records::empty(),
                    stamp_index: 0,
                    fifo_event: None,
                    fifo_event_value: &v,
                },
                d,
            );
            assert_eq!(*s.lock(), vec![4]);
        }
    }

    /// A death ends the task on every one of its threads, so nothing the service does after a
    /// death on another of its threads reaches the host.
    mod task_death {
        use super::*;
        use crate::tasks::isolation::TaskLife;

        /// Dies as the task's code on a thread of its own, as a Metal completion of the task's
        /// does.
        fn die_on_another_thread() {
            let life = TaskLife::current().expect("the task's code");
            let j = std::thread::spawn(move || life.run(|| crate::fatal!(SERVICE_DEATH)));
            assert!(j.join().expect("the thread ends").is_none());
        }

        /// A reply sent after a death on another thread of the task is not delivered
        #[test]
        fn a_reply_sent_after_a_death_on_another_service_thread_is_not_delivered() {
            let h = harness(|ctx, m| match m {
                ToService::DiscardResources { reply, .. } => {
                    die_on_another_thread();
                    // The task is dead by now: this reply never reaches the host.
                    reply.call(1);
                }
                other => standard(ctx, other),
            });
            let t = h.task();
            let (s, d) = done();
            t.discard_resources(Records::empty(), d);
            wait_until(|| !s.lock().is_empty());
            assert_eq!(*s.lock(), vec![4]);
        }

        /// A `completeAsyncTokenID:` sent after the death does not complete the exec's token
        /// with 1 (nor advance the FIFO's counter): the crash completes it with 4.
        #[test]
        fn a_completion_sent_after_a_death_is_not_delivered() {
            let h = harness(|ctx, m| match m {
                ToService::ExecIndirect3 {
                    gpu_completion,
                    reply,
                    ..
                } => {
                    die_on_another_thread();
                    ctx.complete(gpu_completion, 1);
                    reply.call((1, None, 0));
                }
                other => standard(ctx, other),
            });
            let t = h.task();
            let v = Arc::new(AtomicU64::new(0));
            let (s, d) = done();
            exec3(Hold::NONE, &t, None, 0, &v, d);
            wait_until(|| !s.lock().is_empty());
            assert_eq!(*s.lock(), vec![4]);
            assert_eq!(v.load(Ordering::SeqCst), 0);
        }
    }

    /// An exec is delivered on the FIFO thread (`send_here`): the service takes it there, after
    /// every message sent before it, and a death there ends only the task.
    mod here {
        use super::*;
        use std::sync::atomic::AtomicBool;
        use std::thread::ThreadId;

        const EXEC: &str = "execIndirect3WithData:cmdBufferCount:resourceCount:stampIndex:fifoEventValueIn:gpuCompletion:reply:";

        /// The threads the scripted service took each message on, by selector.
        type Threads = Arc<Mutex<Vec<(&'static str, ThreadId)>>>;

        /// The standard service, recording the thread of each message; a discard takes 20 ms,
        /// so an exec sent right after it finds the connection's queues busy.
        fn recording(threads: &Threads) -> impl Fn(&Ctx, ToService<P>) + Send + Sync + 'static {
            let threads = threads.clone();
            move |ctx, m| {
                threads
                    .lock()
                    .push((m.selector(), std::thread::current().id()));
                if matches!(m, ToService::DiscardResources { .. }) {
                    std::thread::sleep(Duration::from_millis(20));
                }
                standard(ctx, m)
            }
        }

        /// The exec is taken on the FIFO thread, after the discard sent before it, which the
        /// connection queue's worker delivered
        #[test]
        fn an_exec_is_taken_on_the_fifo_thread_after_the_messages_before_it() {
            let threads: Threads = Arc::default();
            let h = harness(recording(&threads));
            let t = h.task();
            let (discarded, d) = done();
            t.discard_resources(Records::empty(), d);
            let v = Arc::new(AtomicU64::new(0));
            let (s, d) = done();
            exec3(Hold::NONE, &t, None, 0, &v, d);
            wait_until(|| !s.lock().is_empty());
            assert_eq!(*s.lock(), vec![1]);
            assert_eq!(*discarded.lock(), vec![1]);
            let here = std::thread::current().id();
            let seen = threads.lock().clone();
            let at = |sel: &str| seen.iter().position(|(s, _)| *s == sel).unwrap();
            let discard = at("discardResources:count:reply:");
            let exec = at(EXEC);
            assert!(discard < exec);
            assert_ne!(seen[discard].1, here);
            assert_eq!(seen[exec].1, here);
        }

        /// The exec's reply, made on the FIFO thread while the link's queue is idle, runs there:
        /// the FIFO's event value has advanced when the service's `reply` returns
        #[test]
        fn the_reply_runs_on_the_fifo_thread_when_the_link_queue_is_idle() {
            let v = Arc::new(AtomicU64::new(0));
            let at_reply: Arc<Mutex<Option<u64>>> = Arc::default();
            let (v2, at2) = (v.clone(), at_reply.clone());
            let h = harness(move |ctx, m| match m {
                ToService::ExecIndirect3 {
                    gpu_completion,
                    reply,
                    ..
                } => {
                    reply.call((1, None, 0));
                    *at2.lock() = Some(v2.load(Ordering::SeqCst));
                    ctx.complete(gpu_completion, 1);
                }
                other => standard(ctx, other),
            });
            let t = h.task();
            // The link's queue has run everything the handshake sent it.
            std::thread::sleep(Duration::from_millis(20));
            let (s, d) = done();
            exec3(Hold::NONE, &t, None, 0, &v, d);
            wait_until(|| !s.lock().is_empty());
            assert_eq!(*s.lock(), vec![1]);
            assert_eq!(*at_reply.lock(), Some(1));
        }

        /// With the link's queue busy (a completion the host is still running), the reply goes
        /// behind it on the queue's worker, in its order: the value has not advanced at the reply
        #[test]
        fn the_reply_goes_to_a_busy_link_queue_in_its_order() {
            let gate = Gate::default();
            let in_completion = Arc::new(AtomicBool::new(false));
            let v = Arc::new(AtomicU64::new(0));
            let at_reply: Arc<Mutex<Option<u64>>> = Arc::default();
            let (v2, at2, g2, busy) = (
                v.clone(),
                at_reply.clone(),
                gate.clone(),
                in_completion.clone(),
            );
            let h = harness(move |ctx, m| match m {
                ToService::ExecIndirect3 {
                    gpu_completion,
                    reply,
                    ..
                } => {
                    wait_until(|| busy.load(Ordering::SeqCst));
                    reply.call((1, None, 0));
                    *at2.lock() = Some(v2.load(Ordering::SeqCst));
                    g2.open();
                    ctx.complete(gpu_completion, 1);
                }
                other => standard(ctx, other),
            });
            let t = h.task();
            // The discard's completion runs on the link's queue and holds it until the exec's
            // reply has been made.
            let order = Arc::new(Mutex::new(Vec::new()));
            let (o2, g3, b2) = (order.clone(), gate.clone(), in_completion.clone());
            let discarded: Completion = Arc::new(move |st, _| {
                b2.store(true, Ordering::SeqCst);
                g3.wait();
                o2.lock().push(("discard", st.raw()));
            });
            t.discard_resources(Records::empty(), discarded);
            let (s, d) = done();
            exec3(Hold::NONE, &t, None, 0, &v, d);
            wait_until(|| !s.lock().is_empty());
            assert_eq!(*s.lock(), vec![1]);
            assert_eq!(*at_reply.lock(), Some(0));
            assert_eq!(*order.lock(), vec![("discard", 1)]);
            assert_eq!(v.load(Ordering::SeqCst), 1);
        }

        /// A death in the service's code, taken on the FIFO thread, ends only that task: the
        /// FIFO sees 4 and goes on, the dead task completes later commands with 4 inside the
        /// call, and another task of the same device still runs its execs
        #[test]
        fn a_death_in_an_exec_taken_here_ends_only_its_task() {
            let die = Arc::new(AtomicBool::new(true));
            let died_on: Arc<Mutex<Option<ThreadId>>> = Arc::default();
            let (die2, died2) = (die.clone(), died_on.clone());
            let h = harness(move |ctx, m| {
                if m.selector() == EXEC && die2.load(Ordering::SeqCst) {
                    *died2.lock() = Some(std::thread::current().id());
                    crate::fatal!(SERVICE_DEATH)
                }
                standard(ctx, m)
            });
            let dead = h.task();
            let v = Arc::new(AtomicU64::new(0));
            let (s, d) = done();
            exec3(Hold::NONE, &dead, None, 0, &v, d);
            assert_eq!(*s.lock(), vec![4]);
            assert_eq!(v.load(Ordering::SeqCst), 0);
            assert_eq!(*died_on.lock(), Some(std::thread::current().id()));
            // This thread left the task's code with the death.
            assert!(crate::tasks::isolation::TaskLife::current().is_none());
            // The dead task answers 4 inside the call.
            let (s, d) = done();
            exec3(Hold::NONE, &dead, None, 0, &v, d);
            assert_eq!(*s.lock(), vec![4]);
            // Another task goes on.
            die.store(false, Ordering::SeqCst);
            let alive = h.task();
            let w = Arc::new(AtomicU64::new(0));
            let (s, d) = done();
            exec3(Hold::NONE, &alive, None, 0, &w, d);
            wait_until(|| !s.lock().is_empty());
            assert_eq!(*s.lock(), vec![1]);
            assert_eq!(w.load(Ordering::SeqCst), 1);
        }
    }
}
