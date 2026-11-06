//! Pipeline compiles off ring 0's threads: the [`CompilePool`] that runs them and the
//! [`Pending`] pipelines a command buffer waits for.
//!
//! A pipeline create on ring 0 is prepared on its context's lane (`reims-graphics`' device, on one
//! of ring 0's threads, `reims-vgpu.graphics-ring0-*`), which then goes on with the context's later
//! messages, and is compiled on one of the pool's threads, `reims-vgpu.graphics-compile-*`; its
//! reply goes on ring 0's completion ring when the compile ends, keyed to the create by its seqno.
//! Its process waits for that reply, as Metal's synchronous creators wait for their compile;
//! nothing else does. A library create is made there too.
//!
//! **Turns.** Metal compiles each process's pipelines in that process, so one process's burst of
//! creates delays no other's. The pool's threads serve the processes (contexts) in turns: the
//! next compile is of the context that owns the scanout where it has one waiting (the context of
//! the surface the display last presented, WindowServer's), else of the context with the fewest
//! compiles running, the one longest without a turn among equals; each context's own compiles go
//! in the order it handed them over. So a context's single create waits for at most the first
//! running compile to end, whatever another context has queued, and the scanout owner's waits
//! behind no other context's. Library makes go before every pipeline compile: a pipeline that
//! names a library another context is still making waits for it on its thread, and that library
//! has then always been taken by a thread already. Each compile's wait and run time is logged at
//! debug (`reims_vgpu::graphics::compile`).
//!
//! **Order.** From its dispatch until it is stored, the pipeline is pending. A command buffer
//! that names a pending pipeline (`SET_RENDER_PIPELINE_STATE`, `SET_COMPUTE_PIPELINE_STATE`: the
//! only ops that name one) waits for it before it is decoded; one that names none is not held.
//! An `OBJECT_DESTROY` of a pending pipeline, or its context's close, cancels it: the compile's
//! pipeline is then not stored, so the destroy keeps its place in ring order: the compile's store
//! asks the pending set under the lock of the table the pipeline goes in, and the destroy cancels
//! it before it removes from that table. A table's lock is taken before the pending set's wherever
//! both are held, and a waiter holds only the pending set's.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use reims_graphics::icb::{
    RGFX_ICB_OP_COMPUTE_PIPELINE, RGFX_ICB_OP_RENDER_PIPELINE, RGFX_ICB_SLOT_PIPELINE,
    RgfxIcbObject, RgfxIcbRecord, slot_bytes,
};
use reims_graphics::ops::compute::{
    RGFX_OP_COMPUTE_SET_COMPUTE_PIPELINE_STATE, RgfxComputeOpSetComputePipelineState,
};
use reims_graphics::ops::render::{
    RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE, RgfxOpSetRenderPipelineState,
};
use reims_graphics::ops::{
    RGFX_COMMAND_BUFFER_OPS_OFFSET, RGFX_OP_BODY_OFFSET, RgfxCommandBuffer, RgfxOpHeader,
};
use reims_graphics::region::Region;
use reims_graphics::wire::{RGFX_CONTEXT_OBJ_ID_SHIFT, RgfxIcbSlotRun, RgfxIcbSlots, RgfxIcbWrite};
use reims_vgpu_core::platform::os::clock::MonoTime;
use reims_vgpu_core::sync::{Condvar, Mutex};
use zerocopy::FromBytes;

/// The most compile threads a device runs.
const MAX_THREADS: usize = 4;

/// The target of each compile's line.
const LOG_TARGET: &str = "reims_vgpu::graphics::compile";

/// A compile, run on one of the pool's threads.
pub(crate) type CompileJob = Box<dyn FnOnce() + Send>;

/// The platform's monotonic clock, which times each compile's wait and run.
pub(crate) type Now = Box<dyn Fn() -> MonoTime + Send + Sync>;

/// What a compile makes: a library, or a pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CompileKind {
    /// A `LIBRARY_CREATE` or a library's last `LIBRARY_FROM_REGION` chunk.
    Library,
    /// A `RENDER_PIPELINE_CREATE` or a `COMPUTE_PIPELINE_CREATE`.
    Pipeline,
}

/// A compile's object: its kind, its id, and the context that created it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CompileOf {
    /// A library or a pipeline.
    pub(crate) kind: CompileKind,
    /// The object's guest-picked id, its context in its top bits.
    pub(crate) obj_id: u64,
}

impl CompileOf {
    /// A compile of `kind` for object `obj_id`.
    pub(crate) fn new(kind: CompileKind, obj_id: u64) -> CompileOf {
        CompileOf { kind, obj_id }
    }

    /// The context that created the object.
    fn context(self) -> u32 {
        (self.obj_id >> RGFX_CONTEXT_OBJ_ID_SHIFT) as u32
    }
}

/// A compile waiting for a thread.
struct Queued {
    of: CompileOf,
    job: CompileJob,
    /// When it was handed over, and how many compiles were waiting then.
    handed: MonoTime,
    ahead: usize,
}

/// A context's waiting compiles, each kind in the order handed over.
#[derive(Default)]
struct ContextQueue {
    libraries: VecDeque<Queued>,
    pipelines: VecDeque<Queued>,
}

impl ContextQueue {
    fn of(&mut self, kind: CompileKind) -> &mut VecDeque<Queued> {
        match kind {
            CompileKind::Library => &mut self.libraries,
            CompileKind::Pipeline => &mut self.pipelines,
        }
    }

    fn holds(&self, kind: CompileKind) -> bool {
        match kind {
            CompileKind::Library => !self.libraries.is_empty(),
            CompileKind::Pipeline => !self.pipelines.is_empty(),
        }
    }

    fn is_empty(&self) -> bool {
        self.libraries.is_empty() && self.pipelines.is_empty()
    }
}

/// What the pool's threads share.
#[derive(Default)]
struct PoolState {
    /// Each context's waiting compiles.
    queues: HashMap<u32, ContextQueue>,
    /// The contexts with compiles waiting, in the order they take turns: a context goes to the
    /// back as a compile of it is taken.
    turns: VecDeque<u32>,
    /// The compiles each context has running.
    running: HashMap<u32, usize>,
    /// The context that owns the scanout, whose compiles go first.
    favoured: Option<u32>,
    /// The pool is gone (or stopped): the threads end once nothing waits.
    closed: bool,
}

impl PoolState {
    /// The compiles waiting.
    fn waiting(&self) -> usize {
        self.queues
            .values()
            .map(|q| q.libraries.len() + q.pipelines.len())
            .sum()
    }

    /// Takes the next compile (the module doc, "Turns"), and counts it running.
    fn take(&mut self) -> Option<Queued> {
        let kind = if self.queues.values().any(|q| q.holds(CompileKind::Library)) {
            CompileKind::Library
        } else {
            CompileKind::Pipeline
        };
        let holds = |c: &u32| self.queues.get(c).is_some_and(|q| q.holds(kind));
        let context = self.favoured.filter(holds).or_else(|| {
            self.turns
                .iter()
                .copied()
                .filter(holds)
                .min_by_key(|c| self.running.get(c).copied().unwrap_or(0))
        })?;
        let queue = self.queues.get_mut(&context)?;
        let queued = queue.of(kind).pop_front()?;
        let emptied = queue.is_empty();
        self.turns.retain(|&c| c != context);
        if emptied {
            self.queues.remove(&context);
        } else {
            self.turns.push_back(context);
        }
        *self.running.entry(context).or_default() += 1;
        Some(queued)
    }

    /// A compile of `context` ended.
    fn ended(&mut self, context: u32) {
        if let Some(n) = self.running.get_mut(&context) {
            *n -= 1;
            if *n == 0 {
                self.running.remove(&context);
            }
        }
    }
}

/// The pool's state, the condition its threads wait on for a compile, and its clock.
struct Shared {
    state: Mutex<PoolState>,
    waiting: Condvar,
    now: Now,
}

/// The device's compile threads, which serve the contexts in turns (the module doc, "Turns").
pub(crate) struct CompilePool {
    shared: Arc<Shared>,
    /// The threads that started.
    threads: usize,
    /// Holds the threads before each compile while shut (tests only).
    #[cfg(test)]
    gate: Arc<Gate>,
}

impl CompilePool {
    /// A pool of one thread per core the host offers, two at the least and [`MAX_THREADS`] at
    /// the most. A thread the host refuses is not started; with none, compiles run where they
    /// are handed over. `now` reads the platform's monotonic clock.
    pub(crate) fn new(now: Now) -> CompilePool {
        let threads = std::thread::available_parallelism()
            .map_or(2, |n| n.get())
            .clamp(2, MAX_THREADS);
        CompilePool::with_threads(threads, now)
    }

    /// The platform's monotonic clock, as the pool reads it.
    pub(crate) fn now(&self) -> MonoTime {
        (self.shared.now)()
    }

    /// A pool of `threads` threads, those the host starts.
    fn with_threads(threads: usize, now: Now) -> CompilePool {
        let shared = Arc::new(Shared {
            state: Mutex::new(PoolState::default()),
            waiting: Condvar::new(),
            now,
        });
        #[cfg(test)]
        let gate = Arc::new(Gate::default());
        let mut started = 0;
        for i in 0..threads {
            let shared = Arc::clone(&shared);
            #[cfg(test)]
            let gate = Arc::clone(&gate);
            let spawned = std::thread::Builder::new()
                .name(format!("reims-vgpu.graphics-compile-{i}"))
                .spawn(move || {
                    while let Some(queued) = next_compile(&shared) {
                        #[cfg(test)]
                        gate.pass();
                        run(&shared, queued);
                    }
                });
            if spawned.is_ok() {
                started += 1;
            }
        }
        if started == 0 {
            log::warn!(
                target: "reims_vgpu::graphics",
                "no compile thread could be started: pipelines compile where they are prepared"
            );
            shared.state.lock().closed = true;
        }
        CompilePool {
            shared,
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

    /// Hands `job`, the compile of `of`, to the threads, or gives it back where the pool has
    /// none.
    pub(crate) fn submit(&self, of: CompileOf, job: CompileJob) -> Result<(), CompileJob> {
        let mut state = self.shared.state.lock();
        if state.closed {
            return Err(job);
        }
        let context = of.context();
        let ahead = state.waiting();
        let handed = (self.shared.now)();
        let queue = state.queues.entry(context).or_default();
        let joins = queue.is_empty();
        queue.of(of.kind).push_back(Queued {
            of,
            job,
            handed,
            ahead,
        });
        if joins {
            state.turns.push_back(context);
        }
        drop(state);
        self.shared.waiting.notify_one();
        Ok(())
    }

    /// Context `context` owns the scanout: its compiles go first. A change is logged.
    pub(crate) fn favour(&self, context: u32) {
        let mut state = self.shared.state.lock();
        if state.favoured != Some(context) {
            state.favoured = Some(context);
            log::debug!(
                target: LOG_TARGET,
                "context {context} owns the scanout: its compiles go first"
            );
        }
    }

    /// Context `context` closed: it owns the scanout no longer.
    pub(crate) fn context_closed(&self, context: u32) {
        let mut state = self.shared.state.lock();
        if state.favoured == Some(context) {
            state.favoured = None;
        }
    }

    /// The context whose compiles go first (tests only).
    #[cfg(test)]
    pub(crate) fn favoured(&self) -> Option<u32> {
        self.shared.state.lock().favoured
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

    /// Lets the threads go: every compile from then on runs where it is handed over, as with a
    /// pool no thread started for (tests only).
    #[cfg(test)]
    pub(crate) fn stop(&self) {
        self.close();
    }

    /// Ends the threads once nothing waits; later compiles run where they are handed over.
    fn close(&self) {
        self.shared.state.lock().closed = true;
        self.shared.waiting.notify_all();
    }
}

impl Drop for CompilePool {
    fn drop(&mut self) {
        self.close();
    }
}

/// The next compile for a thread, waiting for one; `None` once the pool is closed and nothing
/// waits.
fn next_compile(shared: &Shared) -> Option<Queued> {
    let mut state = shared.state.lock();
    loop {
        if let Some(queued) = state.take() {
            return Some(queued);
        }
        if state.closed {
            return None;
        }
        shared.waiting.wait(&mut state);
    }
}

/// Runs a compile the pool took, and logs its wait and its run time.
fn run(shared: &Shared, queued: Queued) {
    let Queued {
        of,
        job,
        handed,
        ahead,
    } = queued;
    let started = (shared.now)();
    job();
    let ended = (shared.now)();
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    log::debug!(
        target: LOG_TARGET,
        "context {}'s {} {:#x}: waited {:.1} ms, {ahead} compiles waiting when handed over; \
         compiled in {:.1} ms",
        of.context(),
        match of.kind {
            CompileKind::Library => "library",
            CompileKind::Pipeline => "pipeline",
        },
        of.obj_id,
        ms(started.saturating_duration_since(handed)),
        ms(ended.saturating_duration_since(started)),
    );
    shared.state.lock().ended(of.context());
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

/// The pipelines being compiled, by id: whether a destroy or a close cancelled each, and the
/// device's resets, which forget them all.
#[derive(Default)]
pub(crate) struct Pending {
    /// Pending ids, each with whether it was cancelled and the reset count its compile began at.
    ids: Mutex<HashMap<u64, (bool, u64)>>,
    /// The device's resets so far ([`Self::reset`]).
    resets: AtomicU64,
    /// Notified as each pending pipeline is done, and at a reset.
    done: Condvar,
}

impl Pending {
    /// `obj_id` is being compiled: the reset count its compile begins at, which its
    /// [`Self::live`] and [`Self::finish`] name.
    pub(crate) fn begin(&self, obj_id: u64) -> u64 {
        let mut ids = self.ids.lock();
        let began = self.resets.load(Ordering::Acquire);
        ids.insert(obj_id, (false, began));
        began
    }

    /// An `OBJECT_DESTROY` of `obj_id`: a pending pipeline is not stored when its compile ends.
    pub(crate) fn cancel(&self, obj_id: u64) {
        if let Some((cancelled, _)) = self.ids.lock().get_mut(&obj_id) {
            *cancelled = true;
        }
    }

    /// Context `context`'s close: none of its pending pipelines is stored.
    pub(crate) fn cancel_context(&self, context: u32) {
        for (id, (cancelled, _)) in self.ids.lock().iter_mut() {
            if id >> RGFX_CONTEXT_OBJ_ID_SHIFT == u64::from(context) {
                *cancelled = true;
            }
        }
    }

    /// The device's reset: no pipeline pending is stored, a command buffer waiting on one goes on,
    /// and a compile begun before it neither stores its pipeline nor ends the wait of one begun
    /// after it under the same id.
    pub(crate) fn reset(&self) {
        let mut ids = self.ids.lock();
        self.resets.fetch_add(1, Ordering::AcqRel);
        ids.clear();
        drop(ids);
        self.done.notify_all();
    }

    /// Whether `obj_id`'s compiled pipeline, begun at reset count `began`, is to be stored: it is
    /// pending since then and not cancelled.
    pub(crate) fn live(&self, obj_id: u64, began: u64) -> bool {
        self.ids.lock().get(&obj_id) == Some(&(false, began))
    }

    /// `obj_id`'s compile, begun at reset count `began`, is done (and its pipeline stored, where
    /// it is live): its waiters go on.
    pub(crate) fn finish(&self, obj_id: u64, began: u64) {
        let mut ids = self.ids.lock();
        if ids.get(&obj_id).is_some_and(|&(_, at)| at == began) {
            ids.remove(&obj_id);
        }
        drop(ids);
        self.done.notify_all();
    }

    /// Whether no pipeline the command buffer `message` names is pending, answered waiting for
    /// nothing and allocating nothing: false where another thread holds the table.
    pub(crate) fn none_named_by(&self, message: &[u8]) -> bool {
        let Some(ids) = self.ids.try_lock() else {
            return false;
        };
        ids.is_empty() || !pipelines_named(message).any(|id| ids.contains_key(&id))
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

impl Pending {
    /// Waits until no pipeline the `ICB_WRITE` `message`'s records set is pending.
    pub(crate) fn wait_for_icb_write(&self, message: &[u8]) {
        let mut ids = self.ids.lock();
        if ids.is_empty() {
            return;
        }
        let named = icb_write_pipelines(message);
        while named.iter().any(|id| ids.contains_key(id)) {
            self.done.wait(&mut ids);
        }
    }

    /// Waits until no pipeline the slots of the `ICB_SLOTS` `message` set in `region` is pending.
    /// The slots are read outside the lock: a run of 16384 commands is 2.4 MiB of them.
    pub(crate) fn wait_for_icb_slots(&self, region: &Region, message: &[u8]) {
        if self.ids.lock().is_empty() {
            return;
        }
        let named = icb_slots_pipelines(region, message);
        let mut ids = self.ids.lock();
        while named.iter().any(|id| ids.contains_key(id)) {
            self.done.wait(&mut ids);
        }
    }
}

/// The pipelines the slots an `ICB_SLOTS` message names set in `region` (their pipeline bit set),
/// nil left out, read as they stand: whom the message waits for. A pipeline the guest sets in a
/// slot after this read, racing its own post, and that is still being compiled as the decoder
/// takes the slot, is refused alone there. A run that does not lie in the region, or of a shape
/// the decoder does not take, names none.
pub(crate) fn icb_slots_pipelines(region: &Region, message: &[u8]) -> Vec<u64> {
    let mut named = Vec::new();
    let Ok((slots, mut rest)) = RgfxIcbSlots::read_from_prefix(message) else {
        return named;
    };
    for _ in 0..slots.run_count {
        let Ok((run, after)) = RgfxIcbSlotRun::read_from_prefix(rest) else {
            break;
        };
        rest = after;
        let stride = u64::from(run.slot_bytes);
        let Some(least) = slot_bytes(
            run.kind,
            u32::from(run.bindings_a),
            u32::from(run.bindings_b),
        ) else {
            continue;
        };
        let count = u64::from(run.command_count);
        if stride < least
            || !stride.is_multiple_of(8)
            || !run.region_offset.is_multiple_of(8)
            || stride
                .checked_mul(count)
                .is_none_or(|len| !region.contains(run.region_offset, len))
        {
            continue;
        }
        // Read a chunk of slots at a time, so the copy stays small.
        const CHUNK: u64 = 256;
        let mut start = 0;
        while start < count {
            let n = CHUNK.min(count - start);
            let at = run.region_offset + start * stride;
            let Some(bytes) = region.read(at, (n * stride) as usize) else {
                break;
            };
            for slot in bytes.chunks_exact(stride as usize) {
                // The mask and the pipeline, the slot's first and third words.
                let word = |k: usize| u64::read_from_prefix(&slot[k * 8..]).map_or(0, |(w, _)| w);
                let (mask, pipeline) = (word(0), word(2));
                if mask & RGFX_ICB_SLOT_PIPELINE != 0 && pipeline != 0 {
                    named.push(pipeline);
                }
            }
            start += n;
        }
    }
    named
}

/// The pipelines an `ICB_WRITE` message's records set, nil left out. The walk stops at a record
/// it cannot step past, as the decoder does.
pub(crate) fn icb_write_pipelines(message: &[u8]) -> Vec<u64> {
    let mut named = Vec::new();
    let Ok((write, _)) = RgfxIcbWrite::read_from_prefix(message) else {
        return named;
    };
    let end = (write.header.size as usize).min(message.len());
    let mut at = size_of::<RgfxIcbWrite>();
    while let Some(rest) = message.get(at..end) {
        let Ok((rec, body)) = RgfxIcbRecord::read_from_prefix(rest) else {
            break;
        };
        let len = usize::from(rec.size);
        if len < size_of::<RgfxIcbRecord>() || len % 8 != 0 || len > rest.len() {
            break;
        }
        if (rec.op == RGFX_ICB_OP_RENDER_PIPELINE || rec.op == RGFX_ICB_OP_COMPUTE_PIPELINE)
            && let Ok((pipeline, _)) = RgfxIcbObject::read_from_prefix(body)
            && pipeline.obj_id != 0
        {
            named.push(pipeline.obj_id);
        }
        at += len;
    }
    named
}

/// The pipelines a `COMMAND_BUFFER` message's ops name: each `SET_RENDER_PIPELINE_STATE` and
/// `SET_COMPUTE_PIPELINE_STATE`'s, nil left out. The walk stops at an op it cannot step past; the
/// decoder refuses such a stream itself.
pub(crate) fn named_pipelines(message: &[u8]) -> Vec<u64> {
    pipelines_named(message).collect()
}

/// [`named_pipelines`], read in place, allocating nothing.
fn pipelines_named(message: &[u8]) -> impl Iterator<Item = u64> + '_ {
    let end = RgfxCommandBuffer::read_from_prefix(message)
        .ok()
        .map(|(cb, _)| {
            let start = RGFX_COMMAND_BUFFER_OPS_OFFSET as usize;
            usize::try_from(cb.op_bytes)
                .ok()
                .and_then(|n| start.checked_add(n))
                .map_or(message.len(), |end| end.min(message.len()))
        });
    let mut at = RGFX_COMMAND_BUFFER_OPS_OFFSET as usize;
    std::iter::from_fn(move || {
        let end = end?;
        loop {
            let op = message.get(at..end)?;
            let (header, _) = RgfxOpHeader::read_from_prefix(op).ok()?;
            let len = header.len as usize;
            if len < RGFX_OP_BODY_OFFSET as usize || !len.is_multiple_of(8) || len > op.len() {
                return None;
            }
            let body = &op[RGFX_OP_BODY_OFFSET as usize..len];
            at += len;
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
                return Some(id);
            }
        }
    })
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // test scaffolding: real-time waits for the pool's threads
mod tests {
    use std::sync::LazyLock;
    use std::sync::mpsc;
    use std::time::Instant;

    use super::*;

    /// A pool of `threads` threads on the host's monotonic time.
    fn pool(threads: usize) -> CompilePool {
        CompilePool::with_threads(threads, Box::new(|| MonoTime::from_nanos(since_start())))
    }

    /// Nanoseconds since the tests' first reading.
    fn since_start() -> u64 {
        static START: LazyLock<Instant> = LazyLock::new(Instant::now);
        u64::try_from(START.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }

    /// The object `n` of context `context`.
    fn obj(context: u32, n: u64) -> u64 {
        (u64::from(context) << RGFX_CONTEXT_OBJ_ID_SHIFT) | n
    }

    /// The order compiles start in, by name.
    type Starts = Arc<Mutex<Vec<String>>>;

    /// A compile of `kind` for context `context` that notes its start under `name`, then runs
    /// for `takes`.
    fn compile(
        pool: &CompilePool,
        starts: &Starts,
        kind: CompileKind,
        context: u32,
        name: &str,
        takes: Duration,
    ) {
        let (starts, name) = (Arc::clone(starts), name.to_owned());
        let of = CompileOf::new(kind, obj(context, 1));
        let job: CompileJob = Box::new(move || {
            starts.lock().push(name);
            std::thread::sleep(takes);
        });
        assert!(pool.submit(of, job).is_ok());
    }

    /// Waits until `n` compiles have started.
    fn started(starts: &Starts, n: usize) -> Vec<String> {
        for _ in 0..5000 {
            let now = starts.lock().clone();
            if now.len() >= n {
                return now;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        panic!("{n} compiles did not start");
    }

    /// One context's backlog holds no other context's create: with four threads busy on context
    /// 1's twenty compiles, context 2's single create starts as the first of them ends and so
    /// finishes one compile time after that, where in the order compiles were handed over it
    /// would wait for five
    #[test]
    fn a_context_s_create_waits_behind_no_other_context_s_backlog() {
        const TAKES: Duration = Duration::from_millis(100);
        let pool = pool(4);
        let starts = Starts::default();
        pool.hold();
        for i in 0..20 {
            compile(
                &pool,
                &starts,
                CompileKind::Pipeline,
                1,
                &format!("1.{i}"),
                TAKES,
            );
        }
        let (finished, finish) = mpsc::channel::<Instant>();
        let create: CompileJob = {
            let starts = Arc::clone(&starts);
            Box::new(move || {
                starts.lock().push("2".to_owned());
                std::thread::sleep(TAKES);
                let _ = finished.send(Instant::now());
            })
        };
        assert!(
            pool.submit(CompileOf::new(CompileKind::Pipeline, obj(2, 1)), create)
                .is_ok()
        );
        let released = Instant::now();
        pool.release();
        let done = finish
            .recv_timeout(Duration::from_secs(10))
            .expect("context 2's create never ran");
        let order = starts.lock().clone();
        let at = order.iter().position(|n| n == "2");
        assert!(
            at.is_some_and(|at| at <= 4),
            "context 2's create started late: {order:?}"
        );
        // Handed over last, it would finish after six compile times; it waits for one.
        let took = done.saturating_duration_since(released);
        assert!(
            took < 3 * TAKES,
            "context 2's create finished {took:?} after the release, more than a compile's wait"
        );
    }

    /// The scanout owner's compile goes before every other context's waiting, and so before one
    /// handed over earlier by a context with as few running
    #[test]
    fn the_scanout_owner_s_compile_goes_first() {
        let pool = pool(1);
        let starts = Starts::default();
        pool.favour(3);
        assert_eq!(pool.favoured(), Some(3));
        let (unblock, blocked) = mpsc::channel::<()>();
        let blocker: CompileJob = Box::new(move || {
            let _ = blocked.recv();
        });
        assert!(
            pool.submit(CompileOf::new(CompileKind::Pipeline, obj(9, 1)), blocker)
                .is_ok()
        );
        // The one thread holds the blocker; then three contexts' compiles wait.
        std::thread::sleep(Duration::from_millis(20));
        for i in 0..3 {
            compile(
                &pool,
                &starts,
                CompileKind::Pipeline,
                1,
                &format!("1.{i}"),
                Duration::ZERO,
            );
        }
        compile(
            &pool,
            &starts,
            CompileKind::Pipeline,
            2,
            "2",
            Duration::ZERO,
        );
        compile(
            &pool,
            &starts,
            CompileKind::Pipeline,
            3,
            "3",
            Duration::ZERO,
        );
        unblock.send(()).unwrap();
        let order = started(&starts, 5);
        assert_eq!(order[0], "3", "{order:?}");
        // Then the contexts in turns: 1 and 2 had none running, 1 handed over first.
        assert_eq!(order[1..], ["1.0", "2", "1.1", "1.2"], "{order:?}");
        pool.context_closed(3);
        assert_eq!(pool.favoured(), None);
    }

    /// A library make goes before every pipeline compile, whatever its context, so a pipeline
    /// that waits on its thread for another context's library never waits for one no thread has
    /// taken
    #[test]
    fn a_library_make_goes_before_pipeline_compiles() {
        let pool = pool(1);
        let starts = Starts::default();
        pool.favour(1);
        let (unblock, blocked) = mpsc::channel::<()>();
        let blocker: CompileJob = Box::new(move || {
            let _ = blocked.recv();
        });
        assert!(
            pool.submit(CompileOf::new(CompileKind::Pipeline, obj(9, 1)), blocker)
                .is_ok()
        );
        std::thread::sleep(Duration::from_millis(20));
        compile(
            &pool,
            &starts,
            CompileKind::Pipeline,
            1,
            "pipeline 1",
            Duration::ZERO,
        );
        compile(
            &pool,
            &starts,
            CompileKind::Pipeline,
            2,
            "pipeline 2",
            Duration::ZERO,
        );
        compile(
            &pool,
            &starts,
            CompileKind::Library,
            2,
            "library 2",
            Duration::ZERO,
        );
        unblock.send(()).unwrap();
        assert_eq!(
            started(&starts, 3),
            ["library 2", "pipeline 1", "pipeline 2"]
        );
    }

    /// A stopped pool hands each compile back, to run where it was handed over
    #[test]
    fn a_stopped_pool_hands_compiles_back() {
        let pool = pool(2);
        pool.stop();
        let job: CompileJob = Box::new(|| {});
        assert!(
            pool.submit(CompileOf::new(CompileKind::Library, obj(1, 1)), job)
                .is_err()
        );
    }
}
