//! Pipeline compiles off ring 0's threads: the [`CompilePool`] that runs them and the
//! [`Pending`] pipelines a command buffer waits for.
//!
//! A pipeline create on ring 0 is prepared on its context's lane (`reims-vgpu-graphics`' device, on one
//! of ring 0's threads, `reims-vgpu.graphics-ring0-*`), which then goes on with the context's later
//! messages, and is compiled on one of the pool's threads, `reims-vgpu.graphics-compile-*`; its
//! reply goes on ring 0's completion ring once the compile made its answers, keyed to the create
//! by its seqno, and the pipeline is stored after it ("Order"). Its process waits for that reply,
//! as Metal's synchronous creators wait for their compile; nothing else does. A library create is
//! made there too.
//!
//! **Turns.** Metal compiles each process's pipelines in that process, so one process's burst of
//! creates delays no other's. The pool's threads serve the processes (contexts) in turns: the
//! next compile is of the context that owns the scanout where it has one waiting (the context of
//! the surface the display last presented, WindowServer's), else of the context with the fewest
//! compiles running, the one longest without a turn among equals: a context goes to the back of
//! the turns as a compile of it is taken and again as one ends, so a context whose compiles all
//! ended before a thread took the next goes behind one that waited through them. Each context's
//! own compiles go in the order it handed them over. So a context's single create waits for at
//! most the first running compile to end, whatever another context has queued, and the scanout
//! owner's waits
//! behind no other context's. Library makes go before every pipeline compile: a pipeline that
//! names a library another context is still making waits for it on its thread, and that library
//! has then always been taken by a thread already. A command buffer's dispatches made ahead of its
//! walk ([`super::decode::prepass`]) go after both, one make a compile, on every thread of the
//! pool's but one at once (one at the least), so one thread is left to the creates where the pool
//! has two or more; they do not count among a context's compiles running, and those no thread
//! took by the walk's first dispatch are made on the walk's thread (`decode::prepass`, "The walk
//! makes the rest"). Each compile's wait and run time is logged at debug
//! (`reims_vgpu::graphics::compile`), and, in the profiling trace
//! ([`reims_vgpu_vulkan::core::trace`]), its hand-over, start and end as `cmphand`, `cmpstart`
//! and `cmpend` events, each with the object's id.
//!
//! **The compile in flight.** As a thread starts a compile it logs a begin line at debug: the
//! context, the kind and the object id its end line names, and the description it was handed with
//! ([`CompilePool::submit_described`]): a pipeline's pixel formats, vertex layout and functions,
//! each function by a key, never by its name (`decode::PipelineJob::what`). A begin with no end
//! on a thread that died names the compile it died in. The thread also keeps the same text in a
//! record of its own while the compile runs ([`InFlight`]), which the Windows fault lines read
//! (`crate::faults`), so a crash's report names the compile at the warn level users run at.
//!
//! **Order.** A compile's thread settles it in two steps (`GraphicsControl::settle`). The
//! reply: the platform's make of the state (the device's pipeline creator) made every answer the
//! reply carries, and the reply is posted. The store, after it: the make's second step, the
//! state's finish (`ComputePipeline::finish`, `RenderPipeline::finish`), runs on the same thread,
//! where the platform makes what it left of the state (the driver's pipeline, where it defers
//! that create), so no draw or dispatch of the guest's makes it on its lane; then a keyed
//! create's answer is kept where the make finished, and the pipeline is stored. On the rt
//! platform the finish is the driver's create of the state's main pipelines (rt's
//! `finish_pipeline`), so the reply no longer waits for it; the store, and the command buffers
//! that name the pipeline, still do. A finish that fails is the create's late failure
//! (`decode::FINISH_FAILED`): counted, the first of the process logged at warn, no answer kept,
//! and the pipeline stored as made, its draws and dispatches making what the finish left again
//! at their encode and, where that make fails too, failing as an encode the platform refuses;
//! the VM goes on. So the guest meets that failure, if at all, in its command buffers, not as its
//! create's error, where a Mac's synchronous creator answers it.
//!
//! From its dispatch until it is stored, the pipeline is pending, the window between its reply
//! and its store included. A command buffer that names a pending pipeline
//! (`SET_RENDER_PIPELINE_STATE`, `SET_COMPUTE_PIPELINE_STATE`: the only ops that name one) is
//! begun on its queue's service thread, which goes on to the queue's next buffers, and waits for
//! it on its queue's lane before its walk ([`Pending::wait_on_lane`]), the lane walking its next
//! buffers meanwhile; its view of the table then names the pipeline stored since, as one taken
//! after the store would. One that names none is not held. An `ICB_WRITE`, `ICB_SLOTS` or
//! `ICB_SLOTS_IN_PAGES` that sets a pending pipeline in an indirect command buffer, and a
//! function handle or table create naming a pending compute state, wait on ring 0's thread for
//! its store, a cancelled one's too, or the device's reset ([`Pending::wait_for_icb_write`],
//! [`Pending::wait_for_icb_slots`], [`Pending::wait_for_ids`]). An `OBJECT_DESTROY` of a pending
//! pipeline, or its context's close, cancels it, between its reply and its store as before its
//! reply: its finish runs only where it began before the cancel, and its pipeline is then not
//! stored, so the destroy keeps its place in ring order: the compile's store asks the pending set
//! under the lock of the table the pipeline goes in, and the destroy cancels it before it removes
//! from that table; a buffer waits for no cancelled pipeline. A table's lock is taken before the
//! pending set's wherever both are held, and a waiter holds only the pending set's. The device's
//! reset ends every buffer's wait as it begins ([`Pending::resetting`]), and no compile begun
//! before it is stored, nor finished where its finish had not begun.
//!
//! Each create's begin takes a ticket ([`Ticket`]), a number no other create of the device's life
//! takes, and its store asks for that ticket: an id's pending entry is its newest create's. So a
//! compile whose pipeline was destroyed or whose context closed, and whose id a later create
//! took meanwhile (a process given the closed context's number again makes the same ids), stores
//! nothing under the id and ends no wait for the later create, however long its finish takes:
//! what names the id waits for the later create's store, and then runs with its pipeline. A
//! compile begun before the device's reset finds its ticket gone too. Neither a close nor a reset
//! waits for a compile's make or finish.

//! **A make held for its stage.** A compute pipeline whose make the platform refused for a stage
//! whose compile runs outside the process
//! ([`reims_vgpu_vulkan::core::platform::stage::making`]) stays pending, its compile's thread
//! let go, until the stage's fill ends: where the fill filled the store, the make is made again
//! on the pool, once, and its answer stands; after a breach or a failed fill, or the pipeline's
//! destroy or its context's close meanwhile, the first refusal stands, made no more; and after a
//! cancelled fill (the device's reset, the process's exit) the pipeline is let go of unstored, on
//! the thread that cancelled it, and its reply posted where the ring has room, waiting for
//! nothing. A command buffer that names it waits for it on its lane meanwhile, minutes where its
//! stage's compile takes them. Each such create answered refused after its fill breached or
//! failed, or whose make again refused after a fill that filled the store, is counted
//! ([`HELD_CREATE_REFUSED`]): the creates its stages' fills left refused for good. One let go of
//! at a cancelled fill, or answered its first refusal after its destroy or its context's close
//! at a fill that filled the store, is not.
//!
//! **At the exit.** Each compile a thread runs is a making of the exit gate's
//! ([`reims_vgpu_vulkan::exit_gate`]): the process's exit waits for those in progress, and a
//! thread that takes its next compile once the exit began parks for good without running it, so
//! no compile of the pool's is inside the driver's compiler as the driver's destructors run.
//!
//! **After a heavy make.** Each compile's wall time goes to the platform's OS layer as the
//! compile ends ([`reims_vgpu_vulkan::os::make_ended`]). On Linux with glibc, a make of 250 ms or
//! more asks a thread of the layer's own to hand back to Linux the heap the make's translation
//! left free; the compile's thread only asks, and nothing the guest sees changes.

use std::cell::Cell;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use reims_vgpu_graphics::icb::{
    RGFX_ICB_OP_COMPUTE_PIPELINE, RGFX_ICB_OP_RENDER_PIPELINE, RGFX_ICB_SLOT_PIPELINE,
    RgfxIcbObject, RgfxIcbRecord, slot_bytes,
};
use reims_vgpu_graphics::ops::compute::{
    RGFX_OP_COMPUTE_SET_COMPUTE_PIPELINE_STATE, RgfxComputeOpSetComputePipelineState,
};
use reims_vgpu_graphics::ops::render::{
    RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE, RgfxOpSetRenderPipelineState,
};
use reims_vgpu_graphics::ops::{
    RGFX_COMMAND_BUFFER_OPS_OFFSET, RGFX_OP_BODY_OFFSET, RgfxCommandBuffer, RgfxOpHeader,
};
use reims_vgpu_graphics::region::Region;
use reims_vgpu_graphics::wire::{
    RGFX_CONTEXT_OBJ_ID_SHIFT, RgfxIcbSlotRun, RgfxIcbSlots, RgfxIcbWrite,
};
use reims_vgpu_vulkan::core::platform::os::clock::MonoTime;
use reims_vgpu_vulkan::core::sync::{Condvar, Mutex};
use reims_vgpu_vulkan::core::trace;
use reims_vgpu_vulkan::exit_gate;
use zerocopy::FromBytes;

/// The most compile threads a device runs.
const MAX_THREADS: usize = 4;

/// The target of each compile's line.
const LOG_TARGET: &str = "reims_vgpu::graphics::compile";

/// The degradation code of a compute pipeline create held for its stage's fill that is answered
/// refused: after the fill breached or failed, its make's first refusal, or after a fill that
/// filled the store, its make's refusal again. Counted once a create, so a run's count is how
/// many creates its stages' fills left refused for good: the make's own codes count each create
/// refused for its stage at its first make (`stage-unfilled-make`, and rt's refusal's kind),
/// whatever its fill's end.
pub(crate) const HELD_CREATE_REFUSED: &str = "stage-create-refused";

/// A compile, run on one of the pool's threads.
pub(crate) type CompileJob = Box<dyn FnOnce() + Send>;

/// The platform's monotonic clock, which times each compile's wait and run.
pub(crate) type Now = Box<dyn Fn() -> MonoTime + Send + Sync>;

/// What a compile makes: a library, a pipeline, or what a command buffer's dispatch makes at its
/// first use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CompileKind {
    /// A `LIBRARY_CREATE` or a library's last `LIBRARY_FROM_REGION` chunk.
    Library,
    /// A `RENDER_PIPELINE_CREATE` or a `COMPUTE_PIPELINE_CREATE`.
    Pipeline,
    /// One dispatch of a `COMMAND_BUFFER`'s pre-pass, made ahead of its walk
    /// ([`super::decode::prepass`]): its object is the command buffer.
    Prepass,
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

/// The prefix of the pool's threads' names, each `reims-vgpu.graphics-compile-<n>`.
pub(crate) const THREAD_PREFIX: &str = "reims-vgpu.graphics-compile-";

/// The bytes of a compile's description the thread's record of it keeps ([`InFlight`]): a
/// longer one is cut at a character's boundary.
pub(crate) const IN_FLIGHT_BYTES: usize = 1024;

thread_local! {
    /// The compile in flight on this thread ([`InFlight`]): its text's length, 0 where none runs,
    /// and its bytes. Initialised by a constant and with no destructor, so once a thread has set
    /// it, its read takes no lock and allocates nothing, as a fault handler's read must.
    static IN_FLIGHT: (Cell<usize>, Cell<[u8; IN_FLIGHT_BYTES]>) =
        const { (Cell::new(0), Cell::new([0; IN_FLIGHT_BYTES])) };
}

/// The calling thread's record of the compile it runs (the module docs, "The compile in
/// flight"): set from the compile's text while held, cleared as it drops, a panic's unwind
/// included.
pub(crate) struct InFlight(());

impl InFlight {
    /// Records `text` as the compile in flight on the calling thread, cut to
    /// [`IN_FLIGHT_BYTES`] at a character's boundary.
    pub(crate) fn set(text: &str) -> InFlight {
        let mut take = text.len().min(IN_FLIGHT_BYTES);
        while !text.is_char_boundary(take) {
            take = take.saturating_sub(1);
        }
        let mut bytes = [0u8; IN_FLIGHT_BYTES];
        if let (Some(dst), Some(src)) = (bytes.get_mut(..take), text.as_bytes().get(..take)) {
            dst.copy_from_slice(src);
        }
        let _ = IN_FLIGHT.try_with(|(len, buf)| {
            len.set(0);
            buf.set(bytes);
            len.set(take);
        });
        InFlight(())
    }

    /// Copies the calling thread's compile in flight into `out`, cut to its length; how many
    /// bytes it copied, 0 where none runs. It takes no lock and allocates nothing where the thread
    /// has run a compile before, so a fault handler may call it on a compile thread
    /// (`crate::faults`); on another thread it may allocate the record.
    #[cfg_attr(not(all(windows, target_arch = "x86_64")), allow(dead_code))]
    pub(crate) fn read(out: &mut [u8]) -> usize {
        IN_FLIGHT
            .try_with(|(len, buf)| {
                let bytes = buf.get();
                let n = len.get().min(out.len()).min(IN_FLIGHT_BYTES);
                if let (Some(dst), Some(src)) = (out.get_mut(..n), bytes.get(..n)) {
                    dst.copy_from_slice(src);
                    n
                } else {
                    0
                }
            })
            .unwrap_or(0)
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        let _ = IN_FLIGHT.try_with(|(len, _)| len.set(0));
    }
}

/// A compile waiting for a thread.
struct Queued {
    of: CompileOf,
    /// What it compiles, as its begin line and its thread's record name it, past its object
    /// ([`CompilePool::submit_described`]); empty where it was handed over with none.
    what: String,
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
    prepasses: VecDeque<Queued>,
}

impl ContextQueue {
    fn of(&mut self, kind: CompileKind) -> &mut VecDeque<Queued> {
        match kind {
            CompileKind::Library => &mut self.libraries,
            CompileKind::Pipeline => &mut self.pipelines,
            CompileKind::Prepass => &mut self.prepasses,
        }
    }

    fn holds(&self, kind: CompileKind) -> bool {
        match kind {
            CompileKind::Library => !self.libraries.is_empty(),
            CompileKind::Pipeline => !self.pipelines.is_empty(),
            CompileKind::Prepass => !self.prepasses.is_empty(),
        }
    }

    fn len(&self) -> usize {
        self.libraries.len() + self.pipelines.len() + self.prepasses.len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// What the pool's threads share.
#[derive(Default)]
struct PoolState {
    /// Each context's waiting compiles.
    queues: HashMap<u32, ContextQueue>,
    /// The contexts with compiles waiting, in the order they take turns: a context goes to the
    /// back as a compile of it is taken and as one ends.
    turns: VecDeque<u32>,
    /// The compiles each context has running, its pre-passes' not among them.
    running: HashMap<u32, usize>,
    /// The pre-passes' makes running, and the most that run at once: one thread fewer than the
    /// pool's, one at the least.
    prepasses_running: usize,
    prepasses_most: usize,
    /// The context that owns the scanout, whose compiles go first.
    favoured: Option<u32>,
    /// The pool is gone (or stopped): the threads end once nothing waits.
    closed: bool,
    /// The context of each compile taken, in the order taken (tests only): a thread runs a
    /// compile only after it has taken it, and one put off between the two runs its compile
    /// after a later take's.
    #[cfg(test)]
    taken: Vec<u32>,
}

impl PoolState {
    /// The compiles waiting.
    fn waiting(&self) -> usize {
        self.queues.values().map(ContextQueue::len).sum()
    }

    /// Takes the next compile (the module doc, "Turns"), and counts it running.
    fn take(&mut self) -> Option<Queued> {
        let waits = |kind| self.queues.values().any(|q| q.holds(kind));
        let kind = if waits(CompileKind::Library) {
            CompileKind::Library
        } else if waits(CompileKind::Pipeline) {
            CompileKind::Pipeline
        } else if self.prepasses_running < self.prepasses_most {
            CompileKind::Prepass
        } else {
            return None;
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
        match kind {
            CompileKind::Prepass => self.prepasses_running += 1,
            _ => *self.running.entry(context).or_default() += 1,
        }
        #[cfg(test)]
        self.taken.push(context);
        Some(queued)
    }

    /// A compile of `kind` of `context` ended: the context goes to the back of the turns, behind
    /// those that waited while it ran.
    fn ended(&mut self, context: u32, kind: CompileKind) {
        if kind == CompileKind::Prepass {
            self.prepasses_running = self.prepasses_running.saturating_sub(1);
        } else if let Some(n) = self.running.get_mut(&context) {
            *n -= 1;
            if *n == 0 {
                self.running.remove(&context);
            }
        }
        if let Some(at) = self.turns.iter().position(|&c| c == context)
            && let Some(c) = self.turns.remove(at)
        {
            self.turns.push_back(c);
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

    /// A pool of `threads` threads, those the host starts, with the exit gate's hook registered
    /// first, the device's driver loaded: the process's exit waits for a compile in progress,
    /// and no thread starts one once it began (the module doc, "At the exit").
    fn with_threads(threads: usize, now: Now) -> CompilePool {
        exit_gate::arm();
        let shared = Arc::new(Shared {
            state: Mutex::new(PoolState::default()),
            waiting: Condvar::new(),
            now,
        });
        #[cfg(test)]
        let gate = Arc::new(Gate::default());
        let mut started: usize = 0;
        for i in 0..threads {
            let shared = Arc::clone(&shared);
            #[cfg(test)]
            let gate = Arc::clone(&gate);
            let spawned = std::thread::Builder::new()
                .name(format!("{THREAD_PREFIX}{i}"))
                .spawn(move || {
                    while let Some(queued) = next_compile(&shared) {
                        #[cfg(test)]
                        gate.pass();
                        // A making of the exit gate's: the exit waits for it, and one taken
                        // after the exit began parks the thread, its compile not run.
                        let Some(_making) = exit_gate::making() else {
                            exit_gate::stay()
                        };
                        run(&shared, queued);
                    }
                });
            if spawned.is_ok() {
                started += 1;
            }
        }
        let mut state = shared.state.lock();
        state.prepasses_most = started.saturating_sub(1).max(1);
        if started == 0 {
            log::warn!(
                target: "reims_vgpu::graphics",
                "no compile thread could be started: pipelines compile where they are prepared"
            );
            state.closed = true;
        }
        drop(state);
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
        self.submit_described(of, String::new(), job)
    }

    /// [`Self::submit`], with `what`, the compile's description, which its begin line and its
    /// thread's record of it name past its object (the module docs, "The compile in flight").
    pub(crate) fn submit_described(
        &self,
        of: CompileOf,
        what: String,
        job: CompileJob,
    ) -> Result<(), CompileJob> {
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
            what,
            job,
            handed,
            ahead,
        });
        if joins {
            state.turns.push_back(context);
        }
        drop(state);
        trace::event("cmphand", &[of.obj_id]);
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

    /// The context of each compile the threads took, in the order taken (tests only).
    #[cfg(test)]
    fn taken(&self) -> Vec<u32> {
        self.shared.state.lock().taken.clone()
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

    /// What opens the gate as [`Self::release`] does, once the pool is gone too (tests only).
    #[cfg(test)]
    pub(crate) fn opener(&self) -> impl Fn() + Send + Sync + 'static {
        let gate = Arc::clone(&self.gate);
        move || gate.set(true)
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

/// What a compile of `kind` is called in its lines.
fn kind_name(kind: CompileKind) -> &'static str {
    match kind {
        CompileKind::Library => "library",
        CompileKind::Pipeline => "pipeline",
        CompileKind::Prepass => "command buffer's dispatch made ahead",
    }
}

/// A compile's text: its context, kind and object, then `what` where it has one.
fn in_flight_text(of: CompileOf, what: &str) -> String {
    let mut text = format!(
        "context {}'s {} {:#x}",
        of.context(),
        kind_name(of.kind),
        of.obj_id
    );
    if !what.is_empty() {
        text.push_str(": ");
        text.push_str(what);
    }
    text
}

/// Runs a compile the pool took: logs its begin and keeps its record while it runs (the module
/// docs, "The compile in flight"), logs its wait and its run time, and hands its run time to the
/// OS layer (the module docs, "After a heavy make").
fn run(shared: &Shared, queued: Queued) {
    let Queued {
        of,
        what,
        job,
        handed,
        ahead,
    } = queued;
    let started = (shared.now)();
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    log::debug!(
        target: LOG_TARGET,
        "{}: begins after waiting {:.1} ms{}{what}",
        in_flight_text(of, ""),
        ms(started.saturating_duration_since(handed)),
        if what.is_empty() { "" } else { ": " },
    );
    let text = in_flight_text(of, &what);
    trace::event("cmpstart", &[of.obj_id]);
    let in_flight = InFlight::set(&text);
    job();
    drop(in_flight);
    trace::event("cmpend", &[of.obj_id]);
    let ended = (shared.now)();
    log::debug!(
        target: LOG_TARGET,
        "context {}'s {} {:#x}: waited {:.1} ms, {ahead} compiles waiting when handed over; \
         compiled in {:.1} ms",
        of.context(),
        kind_name(of.kind),
        of.obj_id,
        ms(started.saturating_duration_since(handed)),
        ms(ended.saturating_duration_since(started)),
    );
    shared.state.lock().ended(of.context(), of.kind);
    // A thread that found the pre-passes' makes at their most may take the next now.
    if of.kind == CompileKind::Prepass {
        shared.waiting.notify_one();
    }
    reims_vgpu_vulkan::os::make_ended(ended.saturating_duration_since(started));
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

/// A pipeline create's ticket ([`Pending::begin`]): its pipeline's id, and a number no other
/// create of the device's life takes, across destroys, closes, contexts given their number again
/// and resets. Its compile's [`Pending::live`] and [`Pending::finish`] name it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Ticket {
    /// The pipeline's id.
    obj_id: u64,
    /// The create's number, from 0 in the order the device's creates began.
    n: u64,
}

/// The pipelines being compiled, by id: each id's newest create, by its ticket ([`Ticket`]), and
/// whether a destroy or a close cancelled it; the device's reset forgets them all.
///
/// A create begun under an id whose older create is still compiling takes the id's entry: a
/// context given its number again after its close, whose process makes the same ids, or a
/// destroy and a create again. The older compile's ticket is then gone: its store stores
/// nothing ([`Self::live`]) and removes nothing ([`Self::finish`]), and what names the id waits
/// for the newest create's store, as the guest's order of the creates has it.
#[derive(Default)]
pub(crate) struct Pending {
    /// Each pending id's newest create: whether it was cancelled, and its ticket's number.
    ids: Mutex<HashMap<u64, (bool, u64)>>,
    /// The creates begun so far: the next create's ticket's number.
    tickets: AtomicU64,
    /// A reset begun ([`Self::resetting`]) and not yet carried out ([`Self::reset`]).
    resetting: AtomicBool,
    /// Notified as each pending pipeline is done or cancelled, and at a reset.
    done: Condvar,
}

impl Pending {
    /// `obj_id` is being compiled: its create's ticket, which its [`Self::live`] and
    /// [`Self::finish`] name. The create takes the id's entry from any older one still compiling.
    pub(crate) fn begin(&self, obj_id: u64) -> Ticket {
        let mut ids = self.ids.lock();
        let n = self.tickets.fetch_add(1, Ordering::Relaxed);
        ids.insert(obj_id, (false, n));
        Ticket { obj_id, n }
    }

    /// An `OBJECT_DESTROY` of `obj_id`: the id's newest create's pipeline is not stored when its
    /// compile ends, and no command buffer waits for it.
    pub(crate) fn cancel(&self, obj_id: u64) {
        if let Some((cancelled, _)) = self.ids.lock().get_mut(&obj_id) {
            *cancelled = true;
        }
        self.done.notify_all();
    }

    /// Context `context`'s close: none of its pending pipelines is stored, and no command buffer
    /// waits for one.
    pub(crate) fn cancel_context(&self, context: u32) {
        for (id, (cancelled, _)) in self.ids.lock().iter_mut() {
            if id >> RGFX_CONTEXT_OBJ_ID_SHIFT == u64::from(context) {
                *cancelled = true;
            }
        }
        self.done.notify_all();
    }

    /// The device's reset begins: every command buffer's wait on its lane ([`Self::wait_on_lane`])
    /// ends, and one that begins before the reset is carried out ([`Self::reset`]) ends at once,
    /// so the lanes' jobs let go of the reset's gate with no compile waited for.
    pub(crate) fn resetting(&self) {
        let ids = self.ids.lock();
        self.resetting.store(true, Ordering::Release);
        drop(ids);
        self.done.notify_all();
    }

    /// The device's reset: no pipeline pending is stored, a command buffer waiting on one goes on,
    /// and a compile begun before it, its ticket gone, neither stores its pipeline nor ends the
    /// wait of one begun after it under the same id.
    pub(crate) fn reset(&self) {
        let mut ids = self.ids.lock();
        ids.clear();
        self.resetting.store(false, Ordering::Release);
        drop(ids);
        self.done.notify_all();
    }

    /// Whether `obj_id` is pending, cancelled or not: its newest create's compile, begun since the
    /// device's last reset, has not been stored or let go of.
    #[cfg(test)]
    pub(crate) fn is_pending(&self, obj_id: u64) -> bool {
        self.ids.lock().contains_key(&obj_id)
    }

    /// Whether the compiled pipeline of the create `ticket` is to be stored: the create is its
    /// id's newest, pending, and not cancelled.
    pub(crate) fn live(&self, ticket: Ticket) -> bool {
        self.ids.lock().get(&ticket.obj_id) == Some(&(false, ticket.n))
    }

    /// The compile of the create `ticket` is done (and its pipeline stored, where it is live):
    /// its id's entry goes where the create is still its id's newest, and its waiters go on.
    pub(crate) fn finish(&self, ticket: Ticket) {
        let mut ids = self.ids.lock();
        if ids.get(&ticket.obj_id).is_some_and(|&(_, n)| n == ticket.n) {
            ids.remove(&ticket.obj_id);
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

    /// The pipelines the command buffer `message` names that are pending and not cancelled, each
    /// once: those its walk waits for ([`Self::wait_on_lane`]). Empty, allocating nothing, where
    /// none is pending.
    pub(crate) fn named_by(&self, message: &[u8]) -> Vec<u64> {
        let ids = self.ids.lock();
        if ids.is_empty() {
            return Vec::new();
        }
        let mut named: Vec<u64> = pipelines_named(message)
            .filter(|&id| uncancelled(&ids, id))
            .collect();
        named.sort_unstable();
        named.dedup();
        named
    }

    /// Waits, on a command buffer's lane, until none of the pipelines `named` is pending and not
    /// cancelled, or the device's reset begins ([`Self::resetting`]).
    pub(crate) fn wait_on_lane(&self, named: &[u64]) {
        let mut ids = self.ids.lock();
        while !self.resetting.load(Ordering::Acquire)
            && named.iter().any(|&id| uncancelled(&ids, id))
        {
            self.done.wait(&mut ids);
        }
    }
}

/// Whether `id` is pending in `ids`, a [`Pending`]'s, and not cancelled.
fn uncancelled(ids: &HashMap<u64, (bool, u64)>, id: u64) -> bool {
    ids.get(&id).is_some_and(|&(cancelled, _)| !cancelled)
}

impl Pending {
    /// Waits until none of the pipelines `named` is pending.
    pub(crate) fn wait_for_ids(&self, named: &[u64]) {
        let mut ids = self.ids.lock();
        while named.iter().any(|id| ids.contains_key(id)) {
            self.done.wait(&mut ids);
        }
    }

    /// Waits until no pipeline of those `named` answers is pending, where any is: `named` is
    /// asked only then, as it reads the guest's slots (an `ICB_SLOTS_IN_PAGES`'s, which the
    /// decoder finds in their pages blocks).
    pub(crate) fn wait_for_named(&self, named: impl FnOnce() -> Vec<u64>) {
        if self.ids.lock().is_empty() {
            return;
        }
        self.wait_for_ids(&named());
    }

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

/// The pipelines a `COMMAND_BUFFER` message's ops name, read in place, allocating nothing: each
/// `SET_RENDER_PIPELINE_STATE` and `SET_COMPUTE_PIPELINE_STATE`'s, nil left out. The walk stops at
/// an op it cannot step past; the decoder refuses such a stream itself.
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
        let mut now = Vec::new();
        let came = crate::graphics::tests::eventually(|| {
            now = starts.lock().clone();
            now.len() >= n
        });
        assert!(came, "{n} compiles did not start");
        now
    }

    /// Waits until `pool`'s threads have taken `n` compiles; the contexts of those taken, in
    /// order.
    fn took(pool: &CompilePool, n: usize) -> Vec<u32> {
        let mut now = Vec::new();
        let came = crate::graphics::tests::eventually(|| {
            now = pool.taken();
            now.len() >= n
        });
        assert!(came, "the threads did not take {n} compiles");
        now
    }

    /// The process's exit waits for a compile in progress, and the pool starts no compile once
    /// it began: a child exits 300 ms before its one thread's first compile ends, with a second
    /// queued behind it, which never runs (the module doc, "At the exit"). The second's thread
    /// parks, so the exit's status is the child's own
    #[test]
    fn an_exit_mid_compile_waits_for_it_and_starts_no_other() {
        const TEST: &str =
            "graphics::compile::tests::an_exit_mid_compile_waits_for_it_and_starts_no_other";
        let say = |line: &str| {
            use std::io::Write as _;
            let _ = writeln!(std::io::stderr(), "{line}");
        };
        if std::env::var("RVG_EXIT_CHILD").as_deref() == Ok(TEST) {
            let pool = pool(1);
            let (began, at_begin) = mpsc::channel();
            let first: CompileJob = Box::new(move || {
                began.send(()).unwrap();
                std::thread::sleep(Duration::from_millis(300));
                say("the first compile ended");
            });
            let second: CompileJob = Box::new(move || say("the second compile ran"));
            let of = |n| CompileOf::new(CompileKind::Pipeline, obj(1, n));
            assert!(pool.submit(of(1), first).is_ok());
            assert!(pool.submit(of(2), second).is_ok());
            at_begin.recv().unwrap();
            crate::graphics::tests::host_exit(7);
        }
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", TEST, "--nocapture", "--test-threads=1"])
            .env("RVG_EXIT_CHILD", TEST)
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(7), "{out:?}");
        assert!(stderr.contains("the first compile ended"), "{stderr}");
        assert!(!stderr.contains("the second compile ran"), "{stderr}");
    }

    /// One context's backlog holds no other context's create: with four threads busy on context
    /// 1's twenty compiles, context 2's single create is the fifth compile taken, as the first of
    /// the four ends, though all four end before a thread takes the next, where in the order
    /// compiles were handed over it would be taken after the twenty. The order is read as the
    /// pool took the compiles: a thread put off between its take and its compile's start lets a
    /// compile taken after its own start first. The order alone is checked, not when the create
    /// finished: one compile time after the release on an idle host, it took more than three
    /// under a loaded host's test run (0.46 s, with four sleeping threads)
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
        let (finished, finish) = mpsc::channel::<()>();
        let create: CompileJob = {
            let starts = Arc::clone(&starts);
            Box::new(move || {
                starts.lock().push("2".to_owned());
                std::thread::sleep(TAKES);
                let _ = finished.send(());
            })
        };
        assert!(
            pool.submit(CompileOf::new(CompileKind::Pipeline, obj(2, 1)), create)
                .is_ok()
        );
        pool.release();
        finish
            .recv_timeout(Duration::from_secs(10))
            .expect("context 2's create never ran");
        let taken = pool.taken();
        let at = taken.iter().position(|&context| context == 2);
        assert!(
            at.is_some_and(|at| at <= 4),
            "context 2's create was taken late: {taken:?}, started in {:?}",
            starts.lock().clone()
        );
    }

    /// A context's compile ending puts the context behind one that waited through it: with one
    /// thread, context 2's create handed over while context 1's compile runs is taken as that
    /// compile ends, before context 1's next, though neither then has a compile running and
    /// context 1 joined the turns first
    #[test]
    fn a_context_that_waited_through_another_s_compile_goes_next() {
        let pool = pool(1);
        let starts = Starts::default();
        pool.hold();
        compile(
            &pool,
            &starts,
            CompileKind::Pipeline,
            1,
            "1.0",
            Duration::ZERO,
        );
        // The thread takes 1.0 and holds at the gate; then 1 hands over two more, 2 its create.
        assert_eq!(took(&pool, 1), [1], "the thread took no compile");
        for name in ["1.1", "1.2"] {
            compile(
                &pool,
                &starts,
                CompileKind::Pipeline,
                1,
                name,
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
        pool.release();
        let order = started(&starts, 4);
        assert_eq!(pool.taken(), [1, 2, 1, 1], "started in {order:?}");
        assert_eq!(order, ["1.0", "2", "1.1", "1.2"]);
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
        assert_eq!(took(&pool, 1), [9], "the thread took no blocker");
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
        assert_eq!(took(&pool, 1), [9], "the thread took no blocker");
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

    /// A thread's record of its compile holds the text it was set from, cut at a character's
    /// boundary past [`IN_FLIGHT_BYTES`], and is empty once its holder drops
    #[test]
    fn the_compile_in_flight_is_held_while_its_record_lives() {
        let mut out = [0u8; IN_FLIGHT_BYTES];
        assert_eq!(InFlight::read(&mut out), 0, "nothing runs yet");
        let held = InFlight::set("context 1's pipeline 0x10000000004: render colors [0:65]");
        let n = InFlight::read(&mut out);
        assert_eq!(
            &out[..n],
            b"context 1's pipeline 0x10000000004: render colors [0:65]"
        );
        let mut short = [0u8; 7];
        assert_eq!(InFlight::read(&mut short), 7);
        assert_eq!(&short, b"context");
        drop(held);
        assert_eq!(InFlight::read(&mut out), 0, "cleared as its holder drops");
        // 'a' then two-byte characters: byte 1024 falls inside one, so the record keeps 1023.
        let long = format!("a{}", "é".repeat(IN_FLIGHT_BYTES));
        let held = InFlight::set(&long);
        let n = InFlight::read(&mut out);
        assert_eq!(n, IN_FLIGHT_BYTES - 1);
        assert_eq!(std::str::from_utf8(&out[..n]).unwrap(), &long[..n]);
        drop(held);
    }

    /// A compile reads, on its thread, its context, kind and object, and the description it was
    /// handed over with, after a colon, or none where it was handed over without one
    #[test]
    fn a_compile_s_thread_names_the_compile_it_runs() {
        let pool = pool(1);
        let (tx, seen) = mpsc::channel::<String>();
        let read = |tx: &mpsc::Sender<String>| {
            let tx = tx.clone();
            let job: CompileJob = Box::new(move || {
                let mut out = [0u8; IN_FLIGHT_BYTES];
                let n = InFlight::read(&mut out);
                let name = std::thread::current().name().unwrap_or("").to_owned();
                tx.send(format!("{name}|{}", String::from_utf8_lossy(&out[..n])))
                    .unwrap();
            });
            job
        };
        let described = CompileOf::new(CompileKind::Pipeline, obj(1, 4));
        assert!(
            pool.submit_described(
                described,
                "render colors [0:65] depth 0 stencil 0 samples 1".to_owned(),
                read(&tx)
            )
            .is_ok()
        );
        let wait = Duration::from_secs(10);
        assert_eq!(
            seen.recv_timeout(wait).unwrap(),
            format!(
                "{THREAD_PREFIX}0|context 1's pipeline {:#x}: render colors [0:65] depth 0 \
                 stencil 0 samples 1",
                obj(1, 4)
            )
        );
        // Handed over once the first has run: a library make queued beside a pipeline's goes
        // first.
        assert!(
            pool.submit(CompileOf::new(CompileKind::Library, obj(3, 0xf)), read(&tx))
                .is_ok()
        );
        assert_eq!(
            seen.recv_timeout(wait).unwrap(),
            format!("{THREAD_PREFIX}0|context 3's library {:#x}", obj(3, 0xf))
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
