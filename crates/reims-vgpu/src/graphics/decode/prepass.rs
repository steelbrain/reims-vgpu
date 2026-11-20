//! A command buffer's dispatch pre-pass: what its compute dispatches make at their first use,
//! made ahead of its walk on the device's compile threads.
//!
//! Where the platform makes a kernel's pipeline per dispatch shape (the rt platform: the state's
//! make lowers every variant, but a kernel's `VkPipeline` depends on the threadgroup size, the
//! threadgroup memory lengths and the grid's tiles, which come with each dispatch), the first
//! dispatch of each shape made it on the walk's thread, and the buffer's commit, and its
//! process's wait for it, waited the make out. The pre-pass reads a buffer's
//! dispatches as its queue's service thread begins it, and hands each pipeline not handed before
//! to the compile pool (`CompileKind::Prepass`, `graphics::compile`, "Turns"), whose threads make
//! it ([`ComputePipeline::prepare_dispatch`]) while the walk goes on; a dispatch then finds it
//! made, or waits for the one thread making it.
//!
//! - **What it reads.** Each compute encoder's `dispatchThreads`, `dispatchThreadgroups` and
//!   indirect `dispatchThreadgroups`, in stream order, with the compute state set before it and
//!   the threadgroup memory lengths its encoder set before it (each index once, in the order first
//!   set, at its last length). Not read: a dispatch with no state set, or of an id its view of the
//!   table holds no compute state for (refused, or never made: the walk drops or faults it), an
//!   indirect dispatch of no buffer (the walk faults it), the indirect `dispatchThreads`, which
//!   the platform's make ahead takes no grid of, and the dispatches an indirect command buffer
//!   holds, which its own writes encode.
//! - **Each pipeline once.** The platform names the pipeline a dispatch runs by a key
//!   ([`ComputePipeline::dispatch_variant`]): dispatches of equal keys, of one compute state or of
//!   several, run one pipeline. The device keeps the last [`MEMO`] keys handed, of every state
//!   ([`Handed`]): a dispatch whose key is there hands nothing, so neither does another shape of a
//!   pipeline handed (a grid of more whole threadgroups) nor another state's dispatch of it, and
//!   once a run's pipelines are made its buffers hand nothing and the pool is given nothing. A
//!   dispatch with no key, one the platform would make nothing for, hands nothing, counted
//!   (`dispatch-prepass-no-variant`). A buffer hands [`MOST`] at the most; the rest are counted
//!   (`dispatch-prepass-past-most`) and left to the walk.
//! - **One memo for the device.** The rt platform's key is of the kernel's compiled content,
//!   which every state of one kernel shares with its `VkPipeline`s, and names no other content's
//!   pipeline even once its own is gone. So one memo serves every state: the states the guest's
//!   processes each make of one kernel hand its pipelines once between them, not once a state,
//!   and the memo's bound holds its memory whatever the number of states. It forgets the oldest
//!   key first: a key of a pipeline gone stays until then, and a key forgotten and met again is
//!   handed once more, where the platform finds it made and returns.
//! - **The walk waits for none of it.** Each shape is one compile of the pool's, taken in the
//!   order handed by the first free thread; where no thread is free the walk makes the shape as
//!   before, and a thread that takes it later finds it made.
//! - **Closed at the walk's end.** Right after the buffer's walk and commit, before the message's
//!   views of the table close, the pre-pass is closed ([`Prepass::close`]): the shapes no thread
//!   took are dropped (`dispatch-prepass-dropped`), and the walk's thread waits for those being
//!   made. So nothing the pre-pass holds outlives the views: a compute state the guest destroyed
//!   during the walk is let go of as without the pre-pass, where its view closes, and the indices
//!   it gives back go back in their order. A shape still being made at the walk's end is one the
//!   walk did not dispatch (a binding the platform refused, or the walk refused or faulted before
//!   it), or did a moment before; each walk end that waited is counted
//!   (`dispatch-prepass-waited`, with its nanoseconds, `dispatch-prepass-wait-ns`, and the
//!   longest, `dispatch-prepass-wait-max-ns`), and its first of each kind, the walk's end, logged
//!   at info.
//! - **Nothing the guest sees changes.** A make ahead reports nothing: every result, error and
//!   value is its dispatch's own, decided in the walk as before; only how soon the walk has it
//!   changes.

use std::collections::{HashSet, VecDeque};
use std::fmt;
use std::hash::Hash;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use reims_vgpu_graphics::ops::compute::{
    RGFX_OP_COMPUTE_DISPATCH_THREADGROUPS, RGFX_OP_COMPUTE_DISPATCH_THREADGROUPS_INDIRECT,
    RGFX_OP_COMPUTE_DISPATCH_THREADS, RGFX_OP_COMPUTE_SET_COMPUTE_PIPELINE_STATE,
    RGFX_OP_COMPUTE_SET_THREADGROUP_MEMORY_LENGTH, RgfxComputeOpDispatchThreadgroups,
    RgfxComputeOpDispatchThreadgroupsIndirect, RgfxComputeOpDispatchThreads,
    RgfxComputeOpSetComputePipelineState, RgfxComputeOpSetThreadgroupMemoryLength,
};
use reims_vgpu_graphics::ops::{
    RGFX_OP_ENCODER_BEGIN_COMPUTE, RGFX_OP_ENCODER_END, RgfxCommandBuffer,
};
use reims_vgpu_vulkan::core::platform::compute::Dispatch;
use reims_vgpu_vulkan::core::platform::os::clock::MonoTime;
use reims_vgpu_vulkan::core::platform::resource::ComputePipeline;
use reims_vgpu_vulkan::core::platform::types::MtlSize;
use reims_vgpu_vulkan::core::platform::{Handle, Platform};
use reims_vgpu_vulkan::core::raise::RaiseResult;
use reims_vgpu_vulkan::core::sync::{Condvar, Mutex};

use super::cmdbuf::{op_bodies, ops_of};
use super::tables::Entry;
use super::{DecodeError, GraphicsDecoder, ObjHandle, read};

/// The most shapes one command buffer hands over.
pub(crate) const MOST: usize = 32;

/// The most keys of pipelines handed that the device remembers, of every compute state
/// ([`Handed`]). A pipeline whose key it forgot is handed again at its next dispatch, one make
/// ahead that finds it made.
pub(crate) const MEMO: usize = 1024;

/// The target of the pre-pass's lines.
const LOG_TARGET: &str = "reims_vgpu::graphics::prepass";

/// Codes counted ([`reims_vgpu_vulkan::answer::count_kind`]).
const HANDED: &str = "dispatch-prepass-handed";
const NO_VARIANT: &str = "dispatch-prepass-no-variant";
const PAST_MOST: &str = "dispatch-prepass-past-most";
const DROPPED: &str = "dispatch-prepass-dropped";
const WAITED: &str = "dispatch-prepass-waited";
const WAIT_NS: &str = "dispatch-prepass-wait-ns";
const WAIT_MAX_NS: &str = "dispatch-prepass-wait-max-ns";

/// The longest wait a walk's end counted, in nanoseconds.
static WAIT_MOST: AtomicU64 = AtomicU64::new(0);

/// A dispatch's shape: its arguments as the platform's make ahead takes them.
#[derive(Debug)]
struct Shape {
    dispatch: Dispatch,
    threads: MtlSize,
    lengths: Box<[(u64, u64)]>,
}

/// The keys of the pipelines the device's pre-passes handed, the latest [`MEMO`], of every
/// compute state (the module doc, "Each pipeline once" and "One memo for the device").
pub(crate) struct Handed<V>(Mutex<Memo<V>>);

/// [`Handed`]'s keys, and the order they were handed in.
struct Memo<V> {
    keys: HashSet<V>,
    order: VecDeque<V>,
}

impl<V> Default for Handed<V> {
    fn default() -> Self {
        Handed(Mutex::new(Memo {
            keys: HashSet::new(),
            order: VecDeque::new(),
        }))
    }
}

impl<V: Copy + Eq + Hash> Handed<V> {
    /// Whether `key` was not handed since the latest [`MEMO`]: where it was not, and `note`, it is
    /// noted as handed now, the oldest past [`MEMO`] forgotten.
    fn is_new(&self, key: V, note: bool) -> bool {
        let mut memo = self.0.lock();
        if memo.keys.contains(&key) {
            return false;
        }
        if note {
            if memo.order.len() == MEMO
                && let Some(oldest) = memo.order.pop_front()
            {
                memo.keys.remove(&oldest);
            }
            memo.order.push_back(key);
            memo.keys.insert(key);
        }
        true
    }
}

/// A shape handed over: the compute state it is of, by its handle and its guest id.
struct Item<P: Platform> {
    state: P::ComputePipeline,
    obj_id: u64,
    shape: Shape,
}

/// What a running make is of, for the line of a walk's end that waits for it.
#[derive(Clone, Copy, Debug)]
struct Running {
    obj_id: u64,
    dispatch: Dispatch,
    threads: MtlSize,
}

/// What a command buffer's pre-pass and the compile threads share.
struct Shared<P: Platform> {
    state: Mutex<State<P>>,
    /// Notified as the last make running ends once the pre-pass is closed.
    ended: Condvar,
}

/// The shapes not taken yet, in the order handed, those being made, and whether the walk's end
/// closed the pre-pass.
struct State<P: Platform> {
    items: VecDeque<Item<P>>,
    running: Vec<Running>,
    closed: bool,
}

/// How a command buffer's walk ended, as its pre-pass's close names it: the kind of the line of a
/// close that waited.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WalkEnd {
    /// It was walked and committed: a shape still being made is one a dispatch skipped (the
    /// platform refused a binding of its) or reached a moment before.
    Committed,
    /// It was not carried out ([`DecodeError`]), before the dispatches still being made.
    Refused,
    /// A raise ended it, before them.
    Faulted,
    /// It was never walked: its work was dropped unrun.
    Unwalked,
}

impl WalkEnd {
    /// How the walk whose result is `outcome` ended.
    pub(crate) fn of(outcome: &RaiseResult<Result<(), DecodeError>>) -> WalkEnd {
        match outcome {
            Ok(Ok(())) => WalkEnd::Committed,
            Ok(Err(_)) => WalkEnd::Refused,
            Err(_) => WalkEnd::Faulted,
        }
    }
}

impl fmt::Display for WalkEnd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            WalkEnd::Committed => "committed",
            WalkEnd::Refused => "refused",
            WalkEnd::Faulted => "faulted",
            WalkEnd::Unwalked => "unwalked",
        })
    }
}

/// A command buffer's pre-pass, held by its walk, which closes it at its end ([`Self::close`]);
/// dropped unclosed, it closes as a walk that never ran ([`WalkEnd::Unwalked`]).
pub(crate) struct Prepass<P: Platform> {
    shared: Arc<Shared<P>>,
    cb_obj_id: u64,
    handed: usize,
    closed: bool,
}

impl<P: Platform> Prepass<P> {
    /// The guest's id of its command buffer.
    pub(crate) fn cb_obj_id(&self) -> u64 {
        self.cb_obj_id
    }

    /// How many shapes it handed over: one compile of the pool's each.
    pub(crate) fn handed(&self) -> usize {
        self.handed
    }

    /// One compile of the pool's: the next shape no thread took yet, made on the calling thread,
    /// or nothing, where none is left or the pre-pass is closed.
    pub(crate) fn job(&self) -> Box<dyn FnOnce() + Send> {
        let shared = Arc::clone(&self.shared);
        Box::new(move || make_next(&shared))
    }

    /// The walk's end, `end`: the shapes no thread took are dropped, and the calling thread waits
    /// for those being made, the wait counted and its first line of each kind logged, timed with
    /// `now` (the module doc, "Closed at the walk's end").
    pub(crate) fn close(mut self, end: WalkEnd, now: impl Fn() -> MonoTime) {
        self.close_as(end, &now);
    }

    fn close_as(&mut self, end: WalkEnd, now: &dyn Fn() -> MonoTime) {
        if self.closed {
            return;
        }
        self.closed = true;
        let mut state = self.shared.state.lock();
        state.closed = true;
        let untaken = std::mem::take(&mut state.items);
        let waited = (!state.running.is_empty()).then(|| (now(), state.running.clone()));
        while !state.running.is_empty() {
            self.shared.ended.wait(&mut state);
        }
        drop(state);
        reims_vgpu_vulkan::answer::count_kind_by(DROPPED, untaken.len() as u64);
        drop(untaken);
        let Some((began, running)) = waited else {
            return;
        };
        let took = now().saturating_duration_since(began);
        let ns = u64::try_from(took.as_nanos()).unwrap_or(u64::MAX);
        reims_vgpu_vulkan::answer::count_kind(WAITED);
        reims_vgpu_vulkan::answer::count_kind_by(WAIT_NS, ns);
        let was = WAIT_MOST.fetch_max(ns, Ordering::Relaxed);
        reims_vgpu_vulkan::answer::count_kind_by(WAIT_MAX_NS, ns.saturating_sub(was));
        let first = running[0];
        log::info!(
            target: LOG_TARGET, kind:% = end;
            "command buffer {:#x}'s walk ended ({end}) with {} of its dispatches still being made \
             ahead, the first of compute state {:#x} ({:?}, {}x{}x{} threads a threadgroup): its \
             thread waited {:.1} µs for them",
            self.cb_obj_id,
            running.len(),
            first.obj_id,
            first.dispatch,
            first.threads.width,
            first.threads.height,
            first.threads.depth,
            took.as_secs_f64() * 1e6,
        );
    }
}

impl<P: Platform> Drop for Prepass<P> {
    fn drop(&mut self) {
        // Dropped with the work of a walk that never ran, where no clock is at hand: its wait
        // is counted, not timed.
        self.close_as(WalkEnd::Unwalked, &MonoTime::default);
    }
}

/// A compile of the pool's ([`Prepass::job`]): the next shape no thread took, made here, and let go
/// of before it counts as ended, so a close that waited for it holds none of it.
fn make_next<P: Platform>(shared: &Shared<P>) {
    let mut state = shared.state.lock();
    if state.closed {
        return;
    }
    let Some(item) = state.items.pop_front() else {
        return;
    };
    let running = Running {
        obj_id: item.obj_id,
        dispatch: item.shape.dispatch,
        threads: item.shape.threads,
    };
    state.running.push(running);
    drop(state);
    let shape = &item.shape;
    (item.state).prepare_dispatch(shape.dispatch, shape.threads, &shape.lengths);
    drop(item);
    let mut state = shared.state.lock();
    if let Some(at) = state
        .running
        .iter()
        .position(|r| r.obj_id == running.obj_id)
    {
        state.running.swap_remove(at);
    }
    if state.closed && state.running.is_empty() {
        shared.ended.notify_all();
    }
}

impl<P: Platform> GraphicsDecoder<P> {
    /// The pre-pass of the `COMMAND_BUFFER` `message` (the module doc): the shapes of its
    /// dispatches whose pipelines no buffer handed before, read through the message's views of the
    /// table; `None` where it hands none.
    pub(crate) fn prepass(&self, message: &[u8]) -> Option<Prepass<P>> {
        let m = read::<RgfxCommandBuffer>(message)?;
        let ops = ops_of(message, &m)?;
        let mut items = VecDeque::new();
        let mut state: Option<(u64, &Entry<P>)> = None;
        let mut lengths: Vec<(u64, u64)> = Vec::new();
        // The last dispatch's state, dispatch and threads, while the lengths stay as they were:
        // a run of the same dispatch looks its key up once.
        let mut last: Option<(u64, Dispatch, MtlSize)> = None;
        let mut past_most = 0u64;
        let mut no_variant = 0u64;
        for (tag, body) in op_bodies(ops) {
            let dispatch = match tag {
                RGFX_OP_ENCODER_BEGIN_COMPUTE | RGFX_OP_ENCODER_END => {
                    state = None;
                    lengths.clear();
                    last = None;
                    None
                }
                RGFX_OP_COMPUTE_SET_COMPUTE_PIPELINE_STATE => {
                    state = read::<RgfxComputeOpSetComputePipelineState>(body)
                        .map(|op| op.pipeline_obj_id)
                        .filter(|&id| id != 0)
                        .and_then(|id| Some(id).zip(self.entry(id)))
                        .filter(|(_, e)| matches!(e.handle, ObjHandle::ComputePipeline(_)));
                    None
                }
                RGFX_OP_COMPUTE_SET_THREADGROUP_MEMORY_LENGTH => {
                    if let Some(op) = read::<RgfxComputeOpSetThreadgroupMemoryLength>(body) {
                        match lengths.iter_mut().find(|(index, _)| *index == op.index) {
                            Some((_, length)) => *length = op.length,
                            None => lengths.push((op.index, op.length)),
                        }
                        last = None;
                    }
                    None
                }
                RGFX_OP_COMPUTE_DISPATCH_THREADS => {
                    read::<RgfxComputeOpDispatchThreads>(body).map(|op| {
                        let grid = MtlSize::new(op.grid_width, op.grid_height, op.grid_depth);
                        let threads =
                            MtlSize::new(op.threads_width, op.threads_height, op.threads_depth);
                        (Dispatch::Threads(grid), threads)
                    })
                }
                RGFX_OP_COMPUTE_DISPATCH_THREADGROUPS => {
                    read::<RgfxComputeOpDispatchThreadgroups>(body).map(|op| {
                        let groups = MtlSize::new(
                            op.threadgroups_width,
                            op.threadgroups_height,
                            op.threadgroups_depth,
                        );
                        let threads =
                            MtlSize::new(op.threads_width, op.threads_height, op.threads_depth);
                        (Dispatch::Threadgroups(groups), threads)
                    })
                }
                RGFX_OP_COMPUTE_DISPATCH_THREADGROUPS_INDIRECT => {
                    read::<RgfxComputeOpDispatchThreadgroupsIndirect>(body)
                        .filter(|op| op.indirect_buffer_obj_id != 0)
                        .map(|op| {
                            let threads =
                                MtlSize::new(op.threads_width, op.threads_height, op.threads_depth);
                            (Dispatch::ThreadgroupsIndirect, threads)
                        })
                }
                _ => None,
            };
            let (Some((dispatch, threads)), Some((obj_id, entry))) = (dispatch, state) else {
                continue;
            };
            if last == Some((obj_id, dispatch, threads)) {
                continue;
            }
            last = Some((obj_id, dispatch, threads));
            let ObjHandle::ComputePipeline(handle) = &entry.handle else {
                continue;
            };
            let Some(key) = handle.dispatch_variant(dispatch, threads, &lengths) else {
                no_variant += 1;
                continue;
            };
            let room = items.len() < MOST;
            if !self.shared.handed.is_new(key, room) {
                continue;
            }
            if !room {
                past_most += 1;
                continue;
            }
            let shape = Shape {
                dispatch,
                threads,
                lengths: lengths.as_slice().into(),
            };
            items.push_back(Item {
                state: handle.clone_ref(),
                obj_id,
                shape,
            });
        }
        reims_vgpu_vulkan::answer::count_kind_by(PAST_MOST, past_most);
        reims_vgpu_vulkan::answer::count_kind_by(NO_VARIANT, no_variant);
        if items.is_empty() {
            return None;
        }
        let handed = items.len();
        reims_vgpu_vulkan::answer::count_kind_by(HANDED, handed as u64);
        Some(Prepass {
            shared: Arc::new(Shared {
                state: Mutex::new(State {
                    items,
                    running: Vec::new(),
                    closed: false,
                }),
                ended: Condvar::new(),
            }),
            cb_obj_id: m.cb_obj_id,
            handed,
            closed: false,
        })
    }
}
