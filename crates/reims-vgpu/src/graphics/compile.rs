//! Pipeline compiles off the control thread: the [`CompilePool`] that runs them and the
//! [`Pending`] pipelines a command buffer waits for.
//!
//! A pipeline create on ring 0 is prepared on the control thread, which then goes on with the
//! ring (the display's commands, other processes' creates), and is compiled on one of the pool's
//! threads, `reims-vgpu.graphics-compile-*`; its reply goes on ring 0's completion ring when the
//! compile ends, keyed to the create by its seqno. Its process waits for that reply, as Metal's
//! synchronous creators wait for their compile; nothing else does.
//!
//! **Order.** From its dispatch until it is stored, the pipeline is pending. A command buffer
//! that names a pending pipeline (`SET_RENDER_PIPELINE_STATE`, `SET_COMPUTE_PIPELINE_STATE`: the
//! only ops that name one) waits for it before it is decoded; one that names none is not held.
//! An `OBJECT_DESTROY` of a pending pipeline, or its context's close, cancels it: the compile's
//! pipeline is then not stored, so the destroy keeps its place in ring order. The decoder's lock is
//! taken before the pending set's wherever both are held, and a waiter holds only the pending
//! set's.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, Sender};

use reims_graphics::ops::compute::{
    RGFX_OP_COMPUTE_SET_COMPUTE_PIPELINE_STATE, RgfxComputeOpSetComputePipelineState,
};
use reims_graphics::ops::render::{
    RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE, RgfxOpSetRenderPipelineState,
};
use reims_graphics::ops::{
    RGFX_COMMAND_BUFFER_OPS_OFFSET, RGFX_OP_BODY_OFFSET, RgfxCommandBuffer, RgfxOpHeader,
};
use reims_graphics::wire::RGFX_CONTEXT_OBJ_ID_SHIFT;
use reims_vgpu_core::sync::{Condvar, Mutex};
use zerocopy::FromBytes;

/// The most compile threads a device runs.
const MAX_THREADS: usize = 4;

/// A compile, run on one of the pool's threads.
pub(crate) type CompileJob = Box<dyn FnOnce() + Send>;

/// The device's compile threads, fed in the order compiles are handed over.
pub(crate) struct CompilePool {
    /// The threads' queue; `None` where no thread could be started.
    sender: Mutex<Option<Sender<CompileJob>>>,
    /// The threads that started.
    threads: usize,
    /// Holds the threads before each compile while shut (tests only).
    #[cfg(test)]
    gate: Arc<Gate>,
}

impl CompilePool {
    /// A pool of one thread per core the host offers, two at the least and [`MAX_THREADS`] at
    /// the most. A thread the host refuses is not started; with none, compiles run where they
    /// are handed over.
    pub(crate) fn new() -> CompilePool {
        let threads = std::thread::available_parallelism()
            .map_or(2, |n| n.get())
            .clamp(2, MAX_THREADS);
        let (sender, receiver) = mpsc::channel::<CompileJob>();
        let receiver = Arc::new(Mutex::new(receiver));
        #[cfg(test)]
        let gate = Arc::new(Gate::default());
        let mut started = 0;
        for i in 0..threads {
            let receiver = Arc::clone(&receiver);
            #[cfg(test)]
            let gate = Arc::clone(&gate);
            let spawned = std::thread::Builder::new()
                .name(format!("reims-vgpu.graphics-compile-{i}"))
                .spawn(move || {
                    while let Some(job) = next_job(&receiver) {
                        #[cfg(test)]
                        gate.pass();
                        job();
                    }
                });
            if spawned.is_ok() {
                started += 1;
            }
        }
        if started == 0 {
            log::warn!(
                target: "reims_vgpu::graphics",
                "no compile thread could be started: pipelines compile on the control thread"
            );
        }
        CompilePool {
            sender: Mutex::new((started > 0).then_some(sender)),
            threads: started,
            #[cfg(test)]
            gate,
        }
    }

    /// The compiles the pool runs at once: its threads, or 1 where none started and each
    /// compile runs where it is handed over.
    pub(crate) fn threads(&self) -> usize {
        self.threads.max(1)
    }

    /// Hands `job` to the threads, or gives it back where the pool has none.
    pub(crate) fn submit(&self, job: CompileJob) -> Result<(), CompileJob> {
        match &*self.sender.lock() {
            Some(sender) => sender.send(job).map_err(|e| e.0),
            None => Err(job),
        }
    }

    /// Shuts the gate: the threads hold before their next compile (tests only).
    #[cfg(test)]
    pub(crate) fn hold(&self) {
        self.gate.set(false);
    }

    /// Opens the gate (tests only).
    #[cfg(test)]
    pub(crate) fn release(&self) {
        self.gate.set(true);
    }
}

/// The next compile from the shared queue; `None` once the pool is gone.
fn next_job(receiver: &Mutex<Receiver<CompileJob>>) -> Option<CompileJob> {
    receiver.lock().recv().ok()
}

/// A gate the tests shut to hold every compile.
#[cfg(test)]
#[derive(Default)]
struct Gate {
    shut: Mutex<bool>,
    opened: Condvar,
}

#[cfg(test)]
impl Gate {
    fn set(&self, open: bool) {
        *self.shut.lock() = !open;
        self.opened.notify_all();
    }

    fn pass(&self) {
        let mut shut = self.shut.lock();
        while *shut {
            self.opened.wait(&mut shut);
        }
    }
}

/// The pipelines being compiled, by id: whether a destroy or a close cancelled each.
#[derive(Default)]
pub(crate) struct Pending {
    /// Pending ids, each with whether it was cancelled.
    ids: Mutex<HashMap<u64, bool>>,
    /// Notified as each pending pipeline is done.
    done: Condvar,
}

impl Pending {
    /// `obj_id` is being compiled.
    pub(crate) fn begin(&self, obj_id: u64) {
        self.ids.lock().insert(obj_id, false);
    }

    /// An `OBJECT_DESTROY` of `obj_id`: a pending pipeline is not stored when its compile ends.
    pub(crate) fn cancel(&self, obj_id: u64) {
        if let Some(cancelled) = self.ids.lock().get_mut(&obj_id) {
            *cancelled = true;
        }
    }

    /// Context `context`'s close: none of its pending pipelines is stored.
    pub(crate) fn cancel_context(&self, context: u32) {
        for (id, cancelled) in self.ids.lock().iter_mut() {
            if id >> RGFX_CONTEXT_OBJ_ID_SHIFT == u64::from(context) {
                *cancelled = true;
            }
        }
    }

    /// Whether `obj_id`'s compiled pipeline is to be stored: it is pending and not cancelled.
    pub(crate) fn live(&self, obj_id: u64) -> bool {
        self.ids.lock().get(&obj_id) == Some(&false)
    }

    /// `obj_id`'s compile is done (and its pipeline stored, where it is live): its waiters go on.
    pub(crate) fn finish(&self, obj_id: u64) {
        self.ids.lock().remove(&obj_id);
        self.done.notify_all();
    }

    /// Waits until no pipeline the command buffer `message` names is pending.
    pub(crate) fn wait_for_command_buffer(&self, message: &[u8]) {
        let mut ids = self.ids.lock();
        if ids.is_empty() {
            return;
        }
        let named = named_pipelines(message);
        while named.iter().any(|id| ids.contains_key(id)) {
            self.done.wait(&mut ids);
        }
    }
}

/// The pipelines a `COMMAND_BUFFER` message's ops name: each `SET_RENDER_PIPELINE_STATE` and
/// `SET_COMPUTE_PIPELINE_STATE`'s, nil left out. The walk stops at an op it cannot step past; the
/// decoder refuses such a stream itself.
pub(crate) fn named_pipelines(message: &[u8]) -> Vec<u64> {
    let mut named = Vec::new();
    let Ok((cb, _)) = RgfxCommandBuffer::read_from_prefix(message) else {
        return named;
    };
    let start = RGFX_COMMAND_BUFFER_OPS_OFFSET as usize;
    let end = usize::try_from(cb.op_bytes)
        .ok()
        .and_then(|n| start.checked_add(n))
        .map_or(message.len(), |end| end.min(message.len()));
    let mut at = start;
    while let Some(op) = message.get(at..end) {
        let Ok((header, _)) = RgfxOpHeader::read_from_prefix(op) else {
            break;
        };
        let len = header.len as usize;
        if len < RGFX_OP_BODY_OFFSET as usize || !len.is_multiple_of(8) || len > op.len() {
            break;
        }
        let body = &op[RGFX_OP_BODY_OFFSET as usize..len];
        let pipeline = match header.tag {
            RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE => {
                RgfxOpSetRenderPipelineState::read_from_prefix(body)
                    .ok()
                    .map(|(m, _)| m.pipeline_obj_id)
            }
            RGFX_OP_COMPUTE_SET_COMPUTE_PIPELINE_STATE => {
                RgfxComputeOpSetComputePipelineState::read_from_prefix(body)
                    .ok()
                    .map(|(m, _)| m.pipeline_obj_id)
            }
            _ => None,
        };
        if let Some(id) = pipeline.filter(|&id| id != 0) {
            named.push(id);
        }
        at += len;
    }
    named
}
