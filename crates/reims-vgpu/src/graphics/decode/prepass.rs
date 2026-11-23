//! A command buffer's dispatch pre-pass: what its compute dispatches make at their first use,
//! made ahead of its walk on the device's compile threads.
//!
//! Where the platform makes a kernel's pipeline per dispatch shape (the rt platform: the state's
//! make lowers every variant, but a kernel's `VkPipeline` depends on the threadgroup size, the
//! threadgroup memory lengths, the imageblock size and the grid's tiles, which come with each
//! dispatch), the first dispatch of each shape made it on the walk's thread, and the buffer's
//! commit, and its process's wait for it, waited the make out. The pre-pass reads a buffer's
//! dispatches as its queue's service thread begins it, and hands each pipeline not handed before
//! to the compile pool (`CompileKind::Prepass`, `graphics::compile`, "Turns"), whose threads make
//! it ([`ComputePipeline::prepare_dispatch`]) while the walk goes on; a dispatch then finds it
//! made, or waits for the one thread making it.
//!
//! - **What it reads.** Each compute encoder's `dispatchThreads`, `dispatchThreadgroups` and
//!   indirect `dispatchThreadgroups`, in stream order, with the compute state set before it, the
//!   threadgroup memory lengths its encoder set before it (each index once, in the order first
//!   set, at its last length) and the imageblock size it set last (`setImageblockWidth:height:`;
//!   1 by 1 where it set none, `IMAGEBLOCK_UNSET`). A size the encoder refuses leaves its size
//!   as it was, where the read takes the guest's: the rt platform answers no key of a dispatch at
//!   such a size of a kernel that reaches an imageblock, so it hands nothing (below), and the walk
//!   makes its pipeline. Not read: a dispatch with no state set, or of an id its view of the
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
//!   key the platform answers made (its pipeline made at its state's make, or by a make ahead
//!   or a dispatch before) enters the memo as handed and hands nothing: it takes no place of
//!   the buffer's [`MOST`] and gives the pool nothing. A key not made is handed as above. A
//!   dispatch with no key, one the platform would make nothing for, hands nothing, counted
//!   (`dispatch-prepass-no-variant`). A buffer hands [`MOST`] at the most, as many keys as the
//!   memo holds; the rest are counted (`dispatch-prepass-past-most`) and left to the walk.
//! - **One memo for the device.** The rt platform's key is of the kernel's compiled content,
//!   which every state of one kernel shares with its `VkPipeline`s, and names no other content's
//!   pipeline even once its own is gone. So one memo serves every state: the states the guest's
//!   processes each make of one kernel hand its pipelines once between them, not once a state,
//!   and the memo's bound holds its memory whatever the number of states. It forgets the oldest
//!   key first: a key of a pipeline gone stays until then, and a key forgotten and met again is
//!   handed once more, where the platform finds it made and returns.
//! - **The walk waits for none of it.** Each shape is one compile of the pool's, taken in the
//!   order handed by the first free thread. The walk waits for no thread to take one: it makes
//!   those no thread took by its first dispatch itself (below), and a thread that takes one later
//!   finds it made.
//! - **The walk makes the rest.** At its first dispatch, the walk's thread makes the shapes no
//!   compile thread took yet, one after another in the order handed ([`Untaken`]), beside the
//!   compile threads making those they took; then it dispatches, finding each shape made or
//!   waiting for the thread making it. Each shape handed is of one of the buffer's dispatches,
//!   whose commit waits for it in any case, so its makes run on one thread more, instead of the
//!   walk waiting at each dispatch whose shape is still queued. The makes it ran are counted
//!   (`dispatch-prepass-made-on-walk`), and a walk that ran any logs them at debug. Without it,
//!   a cold boot's dispatches waited 1044.5 ms in 31 waits for their pipelines before the
//!   desktop came up: 659.9 ms in 16 waits at makes running on a compile thread, which took the
//!   boot's 52 makes ahead three at a time, after 4265.6 ms of queueing in all, and 384.6 ms in
//!   15 at makes the walk ran itself, one at each dispatch, while the compile threads were idle,
//!   those of the 15 shapes past the 32 a buffer then handed at the most, of one buffer of 452
//!   dispatches with 47 new shapes (um790's RTX 2080, m3-max-65818; measured).
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
//! - **Whether every dispatch is made.** The read also says whether each dispatch it read runs a
//!   pipeline the platform answered made, and the buffer runs no compute command it does not read
//!   (an indirect `dispatchThreads`, or an indirect command buffer's commands): such a buffer's
//!   walk waits for no make, nor for a stage's fill (`reims_vgpu_vulkan::core::platform::stage`),
//!   which only a pipeline not made can be refused for. Only such a buffer is walked on its
//!   queue's service thread, which takes the queue's next messages once it is done
//!   (`graphics.rs`'s `carry_out`); any other goes to its queue's lane, whose thread steps out of
//!   the lane's walks while it waits (`decode::lane`).

use std::collections::{HashSet, VecDeque};
use std::fmt;
use std::hash::Hash;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use reims_vgpu_graphics::ops::compute::{
    RGFX_OP_COMPUTE_DISPATCH_THREADGROUPS, RGFX_OP_COMPUTE_DISPATCH_THREADGROUPS_INDIRECT,
    RGFX_OP_COMPUTE_DISPATCH_THREADS, RGFX_OP_COMPUTE_DISPATCH_THREADS_INDIRECT,
    RGFX_OP_COMPUTE_EXECUTE_COMMANDS_IN_BUFFER, RGFX_OP_COMPUTE_EXECUTE_COMMANDS_INDIRECT,
    RGFX_OP_COMPUTE_SET_COMPUTE_PIPELINE_STATE, RGFX_OP_COMPUTE_SET_IMAGEBLOCK_WIDTH,
    RGFX_OP_COMPUTE_SET_THREADGROUP_MEMORY_LENGTH, RgfxComputeOpDispatchThreadgroups,
    RgfxComputeOpDispatchThreadgroupsIndirect, RgfxComputeOpDispatchThreads,
    RgfxComputeOpSetComputePipelineState, RgfxComputeOpSetImageblockWidth,
    RgfxComputeOpSetThreadgroupMemoryLength,
};
use reims_vgpu_graphics::ops::{
    RGFX_OP_ENCODER_BEGIN_COMPUTE, RGFX_OP_ENCODER_END, RgfxCommandBuffer,
};
use reims_vgpu_vulkan::core::platform::compute::{Dispatch, IMAGEBLOCK_UNSET};
use reims_vgpu_vulkan::core::platform::os::clock::MonoTime;
use reims_vgpu_vulkan::core::platform::resource::ComputePipeline;
use reims_vgpu_vulkan::core::platform::types::MtlSize;
use reims_vgpu_vulkan::core::platform::{Handle, Platform};
use reims_vgpu_vulkan::core::raise::RaiseResult;
use reims_vgpu_vulkan::core::sync::{Condvar, Mutex};

use super::cmdbuf::{op_bodies, ops_of};
use super::tables::Entry;
use super::{DecodeError, GraphicsDecoder, ObjHandle, read};

/// The most shapes one command buffer hands over: as many keys as the device's memo holds
/// ([`MEMO`]), past which a buffer's later keys would push its first out.
pub(crate) const MOST: usize = MEMO;

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
const MADE_ON_WALK: &str = "dispatch-prepass-made-on-walk";
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
    imageblock: [u64; 2],
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
        Box::new(move || {
            make_next(&shared);
        })
    }

    /// The shapes no thread took yet, for the walk to make at its first dispatch (the module
    /// doc, "The walk makes the rest").
    pub(crate) fn untaken(&self) -> Untaken<P> {
        Untaken {
            shared: Arc::clone(&self.shared),
            cb_obj_id: self.cb_obj_id,
        }
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

/// What a command buffer's walk makes of its pre-pass at its first dispatch ([`Prepass::untaken`]):
/// the shapes no thread took yet. It holds the pre-pass's queue, whose shapes its close takes
/// from it, so nothing of them outlives the close however long it is kept.
pub(crate) struct Untaken<P: Platform> {
    shared: Arc<Shared<P>>,
    cb_obj_id: u64,
}

impl<P: Platform> Untaken<P> {
    /// Makes each shape no thread took yet on the calling thread, the walk's, in the order handed,
    /// until none is left or the pre-pass is closed; the makes are counted, and logged at debug
    /// with their time, read with `now`.
    pub(crate) fn make(self, now: impl Fn() -> MonoTime) {
        let began = now();
        let mut made = 0u64;
        while make_next(&self.shared) {
            made += 1;
        }
        if made == 0 {
            return;
        }
        reims_vgpu_vulkan::answer::count_kind_by(MADE_ON_WALK, made);
        log::debug!(
            target: LOG_TARGET,
            "command buffer {:#x}'s walk made {made} of its dispatches' pipelines no compile \
             thread took, in {:.1} ms",
            self.cb_obj_id,
            now().saturating_duration_since(began).as_secs_f64() * 1e3,
        );
    }
}

/// A compile of the pool's ([`Prepass::job`]), or one of the walk's ([`Untaken::make`]): the next
/// shape no thread took, made here, and let go of before it counts as ended, so a close that
/// waited for it holds none of it; whether there was one.
fn make_next<P: Platform>(shared: &Shared<P>) -> bool {
    let mut state = shared.state.lock();
    if state.closed {
        return false;
    }
    let Some(item) = state.items.pop_front() else {
        return false;
    };
    let running = Running {
        obj_id: item.obj_id,
        dispatch: item.shape.dispatch,
        threads: item.shape.threads,
    };
    state.running.push(running);
    drop(state);
    let shape = &item.shape;
    (item.state).prepare_dispatch(
        shape.dispatch,
        shape.threads,
        &shape.lengths,
        shape.imageblock,
    );
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
    true
}

/// A command buffer's dispatches as its pre-pass read them ([`GraphicsDecoder::prepass`]).
pub(crate) struct Ahead<P: Platform> {
    /// The shapes it hands over, where it hands any.
    pub(crate) prepass: Option<Prepass<P>>,
    /// Every dispatch it read runs a pipeline the platform answered made, and it runs no compute
    /// command the read skips (the module doc, "Whether every dispatch is made"): false for a
    /// message the read cannot read.
    pub(crate) made: bool,
}

impl<P: Platform> GraphicsDecoder<P> {
    /// The pre-pass of the `COMMAND_BUFFER` `message` (the module doc): the shapes of its
    /// dispatches whose pipelines no buffer handed before, read through the message's views of the
    /// table, and whether every dispatch it runs is made.
    pub(crate) fn prepass(&self, message: &[u8]) -> Ahead<P> {
        let unread = Ahead {
            prepass: None,
            made: false,
        };
        let Some(m) = read::<RgfxCommandBuffer>(message) else {
            return unread;
        };
        let Some(ops) = ops_of(message, &m) else {
            return unread;
        };
        let mut made = true;
        let mut items = VecDeque::new();
        let mut state: Option<(u64, &Entry<P>)> = None;
        let mut lengths: Vec<(u64, u64)> = Vec::new();
        let mut imageblock = IMAGEBLOCK_UNSET;
        // The last dispatch's state, dispatch and threads, while the lengths and the imageblock
        // size stay as they were: a run of the same dispatch looks its key up once.
        let mut last: Option<(u64, Dispatch, MtlSize)> = None;
        let mut past_most = 0u64;
        let mut no_variant = 0u64;
        for (tag, body) in op_bodies(ops) {
            let dispatch = match tag {
                RGFX_OP_ENCODER_BEGIN_COMPUTE | RGFX_OP_ENCODER_END => {
                    state = None;
                    lengths.clear();
                    imageblock = IMAGEBLOCK_UNSET;
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
                RGFX_OP_COMPUTE_SET_IMAGEBLOCK_WIDTH => {
                    if let Some(op) = read::<RgfxComputeOpSetImageblockWidth>(body) {
                        imageblock = [op.width, op.height];
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
                RGFX_OP_COMPUTE_DISPATCH_THREADS_INDIRECT
                | RGFX_OP_COMPUTE_EXECUTE_COMMANDS_IN_BUFFER
                | RGFX_OP_COMPUTE_EXECUTE_COMMANDS_INDIRECT => {
                    made = false;
                    None
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
            let variant = handle.dispatch_variant(dispatch, threads, &lengths, imageblock);
            let Some((key, is_made)) = variant else {
                made = false;
                no_variant += 1;
                continue;
            };
            made &= is_made;
            if is_made {
                // Made already: noted as handed, with no slot of the buffer's and no compile.
                self.shared.handed.is_new(key, true);
                continue;
            }
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
                imageblock,
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
            return Ahead {
                prepass: None,
                made,
            };
        }
        let handed = items.len();
        reims_vgpu_vulkan::answer::count_kind_by(HANDED, handed as u64);
        let prepass = Some(Prepass {
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
        });
        Ahead { prepass, made }
    }
}
