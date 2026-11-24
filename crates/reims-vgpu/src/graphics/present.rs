//! [`PresentQueue`]: the display pipe's presents (`PRESENT_SURFACE`), each shown once the command
//! buffers that write its surface's copy have finished, in the order the guest made them.
//!
//! **What a present waits for.** A compositor commits the command buffer that draws its frame
//! into an IOSurface, then submits the display pipe's transaction for that surface, without
//! waiting for the buffer to finish. The buffer rides its queue's own ring, the present ring 0,
//! and the host takes each on its own thread, so the present can reach the host before the
//! buffer has run, or before the host has even taken it. Each process that commits a buffer
//! writing a surface's copy names it in the copy's [`RgfxSurfaceWriters`] before its commit
//! returns (its queue and the buffer's seqno on the queue's ring), so a present reads there which
//! buffers it must see: it is shown once the host has posted each one's `CB_COMPLETE`, and only
//! those. A buffer of another queue, or one that writes another surface, holds nothing. A writer
//! names a queue the host holds from the queue's create on: the device tells the host of the
//! queue before it answers the create ([`PresentQueue::queue_made`]), and the guest names a
//! queue's buffers only after the answer, however late the queue's service thread has the
//! decoder make its handle. A writer naming a queue the host does not hold (never made, destroyed,
//! closed, or ended by the device's reset) holds nothing.
//!
//! A present whose writers have all finished, with none before it waiting, is shown at once; the
//! others wait in order, and each `CB_COMPLETE` shows those at the front that are ready, so a
//! present never overtakes one made before it. Each display's presents keep their own order: on
//! a device of several displays a present of one display never waits for another's. A process publishes a buffer on its queue's ring
//! before it names it a writer, so a writer whose seqno is past every seqno its queue's ring has
//! published when the present comes names no buffer, and holds nothing: a seqno the guest wrote
//! wrong never holds its display's presents. A failed buffer finishes as a completed one does
//! (its `CB_COMPLETE` carries the failure). A queue's completions, posted in its order, say how far
//! it has finished; a failure the host posts at once instead, ahead of buffers before it that may
//! still run (one it neither carried out nor stood in for: a buffer of a queue its process does
//! not hold, or one whose queue made no command buffer for it or its stand-in), finishes that
//! buffer alone. A queue's destroy, or its context's close, drops what the
//! presents wait for of it: no buffer of it will finish.
//!
//! A present of a surface whose device cache holds an older CPU generation than the present's
//! waits too for the fill the host commits for it on taking the present (`decode/surface_cache.rs`):
//! it is shown once that fill has finished, which tells the queue ([`PresentQueue::filled`]), so
//! the presents' lane that carries the present out (`reims-vgpu-graphics`' device) waits for nothing on
//! the GPU.
//!
//! **What a present holds.** A present holds its surface's region block from its taking
//! ([`DisplayHold`]): while it waits, while the display shows it (until the next present, or the
//! device's reset), and in each window draw of it until the draw has finished on the GPU, which
//! may come after the next present. A `BLOCK_RELEASE` of the block is answered only once no hold
//! of it is left, so the window never reads a block the kext has given out again, nor a surface's
//! pages its system has; the holds a present's show lets go of are let go of past the queue's
//! lock, so a release answered there holds up no present.
//!
//! [`PresentAnswers`]: when a shown present is answered. The guest's display pipe completes a
//! transaction on its answer, and holds at most two unanswered (the kext's `RgfxDisplayPipe`),
//! so the answer says the frame is on the display, as a display's flip completes at the vblank
//! that latches it: it is given at the first refresh of the device's display after its show (the
//! refresh thread, [`reims_vgpu_graphics::Host::refreshed`], after that refresh's `VBLANK`),
//! whether or not the host's window has drawn it yet: the device's display shows it from that
//! refresh on, whatever the window does, and the guest's compositor must not wait on the window.
//! A guest that presents once a vblank so never meets its limit of two: each frame is answered at
//! the vblank after its show, as a Mac's display answers a flip at the vblank that latches it.
//!
//! What an answer frees is the surface of every present before it, never its own: the guest
//! completes its transactions oldest first, and a transaction's completion tells its compositor
//! that the surface it showed is on the display, and that the surfaces shown before it are not,
//! so the compositor draws into those again, as it would on a Mac whose display scans the shown
//! surface out until the next flip. So the answers go in the order the presents were shown, and
//! none while a window draw of an earlier present, which may copy out of that present's surface,
//! could still run after the guest's next write of it: each draw holds the answers of the
//! presents after the one whose surface it may read until it lets go of them
//! ([`PresentAnswers::let_go`]). A draw of a surface whose every write is the platform's work (a
//! device cache's) lets go as it is scheduled: the guest commits its next write of the surface
//! after the answer, and the platform's one queue runs that write after the draw, however long
//! the draw then waits on the GPU (`DrawnPresents::after`, in `graphics.rs`, names the rule and
//! the test that pins it). A draw of a surface a write may reach around that queue (the guest's
//! CPU, the host's video) lets go once it has finished on the GPU. A present whose own draw
//! holds is answered, and holds the answers after it until its draw lets go. Answered out of
//! order, a present the window had not drawn freed the surface of a draw still running on the
//! GPU; the compositor's next frame then landed in it under the copy, and the window showed that
//! newer frame, then the older one it had presented between, then the newer again. Waiting for a
//! present's own draw instead held its answer a refresh where the draw ended past the first
//! refresh after its show (an RTX 5080 Laptop GPU at 30 frames a second: the show 6.0 ms after a
//! 120 Hz refresh, the draw done 5.2 ms later), which the guest saw as a frame a refresh late.
//! Waiting for each draw to finish held the answers behind the guest's own work queued on the GPU
//! ahead of the draw: an Iris Xe under a 3D benchmark's scene answered its presents 47.5 ms after
//! their show at the median, 993 of 1011 with their own draw still running, and the scene
//! presented 10.2 frames a second against its own 26.4 (**measured** 2026-11-21, job
//! m3-max-63530).
//!
//! **Where a present's time goes.** Each answer is logged at debug, on [`TIMING`]'s target:
//! how long after the device's last refresh its `PRESENT_SURFACE` was taken, how long it waited
//! for its writers and fill to be shown, how long after its show the window's first draw of it
//! let go of the answers it held, as it was scheduled or as it finished ([`LetGo`]), and at
//! which refresh it was answered. It tells a present the host answers a refresh late (its show
//! or its draw's let-go past the first refresh after its take) from one the guest's compositor
//! sent late in the refresh, which a guest's present latency alone does not. A display with no
//! window to draw it (a headless device, or a window with no sink) answers each present at its
//! show, and logs its line there, naming the last refresh before its show as the one that
//! answered it ([`PresentAnswers::answered_at_show`]); the host's summary counts those answers
//! too.
//!
//! A present shown late is logged once more, at its show, on the same target: one whose show came
//! [`LATE_TIMES`] times the median of the last [`LATE_WINDOW`] presents' or more after its take
//! (while fewer than [`LATE_LEAST`] were shown, one a writer held past its take). The line names
//! its writer that finished last after its take, its queue, context and seqno, with when the host
//! first saw that buffer (its doorbell's answer or its take from its ring, whichever came first)
//! and when its completion was posted, against the present's take and on the host's monotonic
//! clock, or says that none of its writers finished after its take. What it needs is kept only
//! where the line is on ([`Late`]).
//!
//! [`RgfxSurfaceWriters`]: reims_vgpu_graphics::wire::RgfxSurfaceWriters

use std::collections::{BTreeSet, HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;

use reims_vgpu_graphics::wire::{RGFX_CONTEXT_OBJ_ID_SHIFT, RGFX_STATUS_OK, RgfxSurfaceWriter};
use reims_vgpu_graphics::{CompletionPoster, PresentAnswer};
use reims_vgpu_vulkan::core::num::{ToU64 as _, ToUsize as _};
use reims_vgpu_vulkan::core::platform::Platform;
use reims_vgpu_vulkan::core::platform::os::clock::MonoTime;
use reims_vgpu_vulkan::core::sync::Mutex;

use super::color::ColorChange;
use super::compile::Now;
use super::decode::FillWait;
use super::{DisplayHold, Presented, ShownTexture, Shows};

/// The presents waiting for their surfaces' writers, and how far each queue has finished.
pub(crate) struct PresentQueue<P: Platform> {
    /// What each display shows, display `i`'s at index `i`, which a present of it changes when it
    /// is shown.
    presented: Vec<Arc<Presented<P>>>,
    /// The queues' progress and the presents waiting, under one lock, so a completion and a
    /// present never miss each other.
    state: Mutex<State<P>>,
}

/// [`PresentQueue`]'s state.
struct State<P: Platform> {
    /// By queue id: the highest seqno of a `CB_COMPLETE` the host has posted on its ring in the
    /// queue's order, each after those of the buffers before it ([`PresentQueue::finished`]),
    /// so every buffer at or below it has finished.
    finished: HashMap<u64, u64>,
    /// By queue id: the seqnos past `finished`'s whose failure the host posted at once, ahead of
    /// buffers before them that may still run ([`PresentQueue::failed_at_once`]): each is
    /// finished, and says nothing of the buffers before it. They go as `finished` passes them, or
    /// with their queue.
    apart: HashMap<u64, BTreeSet<u64>>,
    /// By display, display `i`'s at index `i`: its presents not yet shown, oldest first.
    waiting: Vec<VecDeque<Waiting<P>>>,
    /// By queue id: the poster onto its completion ring, through which what its submission ring
    /// has published is read.
    queues: HashMap<u64, CompletionPoster>,
    /// What the late-show lines keep, only where they are on.
    late: Late,
}

/// One present not yet shown.
struct Waiting<P: Platform> {
    /// What it shows: the IOSurface's texture, its block held, by its id (0 and `None` for the
    /// scanout), or what the display shows already.
    shows: Shows<P>,
    /// The colour stages it sets, as the display shows it (`color`).
    color: Option<ColorChange>,
    /// The present's answer, owed once it is shown.
    answer: PresentAnswer,
    /// The command buffers it waits for.
    writers: Vec<RgfxSurfaceWriter>,
    /// The fill of its surface's device cache it waits for, where it asked one.
    fill: Option<FillWait>,
    /// When it was taken.
    taken: Taken,
    /// Its writer that finished last after its take, for its late-show line, where that line is
    /// on.
    last: Option<LastWriter>,
}

impl<P: Platform> PresentQueue<P> {
    /// The presents of the displays `presented` shows, display `i` by `presented[i]`.
    pub(crate) fn new(presented: Vec<Arc<Presented<P>>>) -> Self {
        let waiting = presented.iter().map(|_| VecDeque::new()).collect();
        PresentQueue {
            presented,
            state: Mutex::new(State {
                finished: HashMap::new(),
                apart: HashMap::new(),
                waiting,
                queues: HashMap::new(),
                late: Late::default(),
            }),
        }
    }

    /// The platform's monotonic clock now, as the displays' answers read it.
    fn now(&self) -> Option<MonoTime> {
        self.presented.first().map(|presented| presented.answers.now())
    }

    /// The host sees the command buffer at `seqno` of queue `queue` now (its doorbell's answer or
    /// its take from its ring): where the late-show lines are on, the first sight of it is kept
    /// until it finishes, for the line of a present it is the last writer of (the module doc).
    pub(crate) fn buffer_seen(&self, queue: u64, seqno: u64) {
        if !late_lines() {
            return;
        }
        let Some(now) = self.now() else {
            return;
        };
        let mut state = self.state.lock();
        if state.finished.get(&queue).is_some_and(|&done| done >= seqno) {
            return;
        }
        let seen = state.late.seen.entry(queue).or_default();
        if seen.iter().rev().any(|&(s, _)| s == seqno) {
            return;
        }
        if seen.len() >= LATE_SEEN {
            seen.pop_front();
        }
        seen.push_back((seqno, now));
    }

    /// Queue `queue` was made, its completion ring posted onto through `poster`: what its
    /// submission ring has published bounds what a present waits for of it.
    pub(crate) fn queue_made(&self, queue: u64, poster: CompletionPoster) {
        self.state.lock().queues.insert(queue, poster);
    }

    /// Shows IOSurface `id`'s `texture` (`None`: the scanout) on display `display`, with the
    /// colour stages `color` sets, once each command buffer in `writers` has finished, and
    /// `fill`, its surface's device cache's fill, where it has one, and every present of the
    /// display made before it is shown; `answer` goes as [`Presented`] gives it. A writer of a
    /// queue the host does not hold, or past what its queue's ring has published, holds nothing
    /// (the module doc). Called with no lock of the decoder's held: what a ring has published is
    /// read with the ring held. `display` is one the queue shows.
    #[expect(
        clippy::too_many_arguments,
        reason = "a present's parts, as the device hands them"
    )]
    pub(crate) fn present(
        &self,
        display: usize,
        id: u32,
        texture: Option<ShownTexture<P>>,
        color: Option<ColorChange>,
        answer: PresentAnswer,
        writers: Vec<RgfxSurfaceWriter>,
        fill: Option<FillWait>,
    ) {
        let Some(presented) = self.presented.get(display) else {
            return;
        };
        let taken = presented.answers.taken();
        let writers = self.published(writers);
        self.push(
            display,
            Waiting {
                shows: Shows::Surface(id, texture),
                color,
                answer,
                writers,
                fill,
                taken,
                last: None,
            },
        );
    }

    /// A present that keeps what display `display` shows, with the colour stages `color` sets
    /// (`RGFX_PRESENT_KEEP`): shown once every present of the display made before it is.
    /// `display` is one the queue shows.
    pub(crate) fn present_kept(
        &self,
        display: usize,
        color: Option<ColorChange>,
        answer: PresentAnswer,
    ) {
        let Some(presented) = self.presented.get(display) else {
            return;
        };
        let taken = presented.answers.taken();
        self.push(
            display,
            Waiting {
                shows: Shows::Kept,
                color,
                answer,
                writers: Vec::new(),
                fill: None,
                taken,
                last: None,
            },
        );
    }

    /// `waiting` joins display `display`'s presents, which are shown as they are ready.
    fn push(&self, display: usize, waiting: Waiting<P>) {
        let mut state = self.state.lock();
        let Some(presents) = state.waiting.get_mut(display) else {
            return;
        };
        presents.push_back(waiting);
        let gone = self.show_ready(&mut state);
        drop(state);
        drop(gone);
    }

    /// A fill of a device cache finished: the presents at the front waiting for nothing more are
    /// shown.
    pub(crate) fn filled(&self) {
        let mut state = self.state.lock();
        if state.waits() {
            let gone = self.show_ready(&mut state);
            drop(state);
            drop(gone);
        }
    }

    /// The host posted the `CB_COMPLETE` of the command buffer at `seqno` on queue `queue`'s ring,
    /// through `poster`, in the queue's order: in its completion handler, the queue's buffers
    /// finishing in the order they were committed, so those before it have finished too (a
    /// failure posted out of that order is [`Self::failed_at_once`]'s). A buffer of a queue the
    /// device's reset ended counts for nothing: the rebooted guest's queue may take the same id,
    /// its seqnos from the start. The poster is looked at with the state held, which the reset
    /// takes once the queue has ended ([`Self::reset`]), so no buffer of the queue counts after
    /// the reset.
    pub(crate) fn finished(&self, queue: u64, seqno: u64, poster: &CompletionPoster) {
        let mut state = self.state.lock();
        if poster.stale() {
            return;
        }
        let done = state.finished.entry(queue).or_insert(0);
        let before = *done;
        *done = (*done).max(seqno);
        let done = *done;
        if late_lines() {
            state.late_finished(queue, |s| before < s && s <= done, self.now());
            if let Some(seen) = state.late.seen.get_mut(&queue) {
                seen.retain(|&(s, _)| s > done);
                if seen.is_empty() {
                    state.late.seen.remove(&queue);
                }
            }
        }
        if let Some(apart) = state.apart.get_mut(&queue) {
            apart.retain(|&s| s > done);
            if apart.is_empty() {
                state.apart.remove(&queue);
            }
        }
        if state.waits() {
            let gone = self.show_ready(&mut state);
            drop(state);
            drop(gone);
        }
    }

    /// The host posted the failure of the command buffer at `seqno` on queue `queue`'s ring,
    /// through `poster`, at once, out of the queue's order (a buffer it neither carried out nor
    /// stood in for): that buffer has finished, and the buffers before it may still run, so a
    /// present waiting for one of them still waits ([`Self::finished`] for the rest).
    pub(crate) fn failed_at_once(&self, queue: u64, seqno: u64, poster: &CompletionPoster) {
        let mut state = self.state.lock();
        if poster.stale() {
            return;
        }
        if state
            .finished
            .get(&queue)
            .is_some_and(|&done| done >= seqno)
        {
            return;
        }
        state.apart.entry(queue).or_default().insert(seqno);
        if late_lines() {
            state.late_finished(queue, |s| s == seqno, self.now());
        }
        if state.waits() {
            let gone = self.show_ready(&mut state);
            drop(state);
            drop(gone);
        }
    }

    /// `writers`, less each of a queue the host does not hold ([`Self::queue_made`] to
    /// [`Self::queue_gone`], [`Self::context_gone`] or [`Self::reset`]), and each whose seqno is
    /// past every seqno its queue's ring has published.
    fn published(&self, writers: Vec<RgfxSurfaceWriter>) -> Vec<RgfxSurfaceWriter> {
        if writers.is_empty() {
            return writers;
        }
        let posters: Vec<Option<CompletionPoster>> = {
            let state = self.state.lock();
            (writers.iter())
                .map(|w| state.queues.get(&w.queue_obj_id).cloned())
                .collect()
        };
        writers
            .into_iter()
            .zip(posters)
            .filter(|(w, poster)| {
                let Some(poster) = poster else {
                    return false;
                };
                let published = poster.published_seqno();
                let named = published.is_none_or(|published| w.seqno <= published);
                if !named {
                    log::warn!(
                        target: "reims_vgpu::graphics",
                        "a present's writer names seqno {} of queue {:#x}, past the {} its ring \
                         has published: it holds nothing",
                        w.seqno,
                        w.queue_obj_id,
                        published.unwrap_or_default()
                    );
                }
                named
            })
            .map(|(w, _)| w)
            .collect()
    }

    /// The poster onto queue `queue`'s completion ring, while the host holds the queue: where a
    /// shared event's other holders' values go for the process that named it its listen queue
    /// (`decode::shares`).
    pub(crate) fn queue_poster(&self, queue: u64) -> Option<CompletionPoster> {
        self.state.lock().queues.get(&queue).cloned()
    }

    /// Queue `queue` is gone: nothing more of it finishes.
    pub(crate) fn queue_gone(&self, queue: u64) {
        self.state.lock().queues.remove(&queue);
        self.forget(|q| q == queue);
    }

    /// Context `context` closed: none of its queues' command buffers finish.
    pub(crate) fn context_gone(&self, context: u32) {
        let gone = |q: u64| q >> RGFX_CONTEXT_OBJ_ID_SHIFT == u64::from(context);
        self.state.lock().queues.retain(|&q, _| !gone(q));
        self.forget(gone);
    }

    /// The device's reset: the presents waiting are dropped unshown, their answers owed to no
    /// one, and every queue's progress is forgotten. Their textures are let go of in a refusal
    /// scope.
    pub(crate) fn reset(&self) {
        let waiting = {
            let mut state = self.state.lock();
            state.finished.clear();
            state.apart.clear();
            state.queues.clear();
            state.late = Late::default();
            (state.waiting.iter_mut())
                .map(std::mem::take)
                .collect::<Vec<_>>()
        };
        if let Err(refusal) =
            reims_vgpu_vulkan::core::raise::refusals::refusing(move || drop(waiting))
        {
            log::warn!(
                target: "reims_vgpu::graphics", kind:% = refusal;
                "the presents waiting were dropped at the device's reset, and a texture's release \
                 was refused: {refusal}"
            );
        }
    }

    /// Drops the queues `gone` names from the progress and from what the presents wait for, then
    /// shows those now ready.
    fn forget(&self, gone: impl Fn(u64) -> bool) {
        let mut state = self.state.lock();
        state.finished.retain(|&q, _| !gone(q));
        state.apart.retain(|&q, _| !gone(q));
        state.late.seen.retain(|&q, _| !gone(q));
        for waiting in state.waiting.iter_mut().flatten() {
            waiting.writers.retain(|w| !gone(w.queue_obj_id));
        }
        let let_go = self.show_ready(&mut state);
        drop(state);
        drop(let_go);
    }

    /// Shows each display's presents at its front whose writers have finished, oldest first. The
    /// holds of the textures they replace are returned, for the caller to let go of once it has
    /// let go of the state: a release waiting for one is answered there, off the present's path.
    #[must_use]
    fn show_ready(&self, state: &mut State<P>) -> Vec<Arc<DisplayHold>> {
        let mut gone = Vec::new();
        let now = late_lines().then(|| self.now()).flatten();
        let State {
            finished,
            apart,
            waiting,
            late,
            ..
        } = state;
        for (presents, presented) in waiting.iter_mut().zip(&self.presented) {
            loop {
                let ready = presents.front().is_some_and(|front| {
                    front.fill.as_ref().is_none_or(FillWait::done)
                        && front.writers.iter().all(|w| {
                            let queue = &w.queue_obj_id;
                            (finished.get(queue)).is_some_and(|&done| done >= w.seqno)
                                || (apart.get(queue)).is_some_and(|a| a.contains(&w.seqno))
                        })
                });
                if !ready {
                    break;
                }
                let Some(shown) = presents.pop_front() else {
                    break;
                };
                if let Some(now) = now {
                    late.shown(&shown.taken, shown.last, now);
                }
                gone.extend(presented.show(shown.shows, shown.color, shown.answer, shown.taken));
            }
        }
        gone
    }

    /// The presents not yet shown, of every display (for the tests).
    #[cfg(test)]
    pub(crate) fn waiting(&self) -> usize {
        self.state.lock().waiting.iter().map(VecDeque::len).sum()
    }
}

impl<P: Platform> State<P> {
    /// Whether a present of any display waits.
    fn waits(&self) -> bool {
        self.waiting.iter().any(|presents| !presents.is_empty())
    }

    /// The command buffers of queue `queue` whose seqnos `newly` names finished at `now`: each
    /// is the last writer to finish so far of each present waiting for it (its late-show line).
    fn late_finished(&mut self, queue: u64, newly: impl Fn(u64) -> bool, now: Option<MonoTime>) {
        let Some(now) = now else {
            return;
        };
        let State { waiting, late, .. } = self;
        for present in waiting.iter_mut().flatten() {
            let finished = (present.writers.iter().rev())
                .find(|w| w.queue_obj_id == queue && newly(w.seqno))
                .copied();
            if let Some(writer) = finished {
                present.last = Some(LastWriter {
                    writer,
                    seen: late.seen(queue, writer.seqno),
                    done: now,
                });
            }
        }
    }
}

/// A present's answer, given once with [`RGFX_STATUS_OK`].
pub(crate) trait Answer: Send {
    /// Gives the answer.
    fn answer_ok(self);
}

impl Answer for PresentAnswer {
    fn answer_ok(self) {
        self.answer(RGFX_STATUS_OK);
    }
}

/// When shown presents are answered (the module doc).
pub(crate) struct PresentAnswers<A> {
    state: Mutex<AnswersState<A>>,
    /// The platform's monotonic clock, which the timing lines read.
    now: Now,
}

/// [`PresentAnswers`]' state.
struct AnswersState<A> {
    /// The device's refreshes so far.
    refreshes: u64,
    /// When the host was told of the last of them.
    last_refresh: Option<MonoTime>,
    /// The number the next present shown takes.
    next: u64,
    /// The presents shown and not yet answered, in the order shown, their numbers counting up
    /// from the front's.
    owed: VecDeque<Owed<A>>,
    /// The window's draws that hold answers, each by the number of the latest present shown when
    /// it began: it may read that present's surface, or a later one's, and no earlier one's.
    reading: Vec<u64>,
}

/// A present shown and not yet answered.
struct Owed<A> {
    answer: A,
    /// Whether a window draw has taken it ([`PresentAnswers::drawn`]), for its timing line.
    taken_by_draw: bool,
    /// When it was taken, shown, and the first draw that took it let go of what it held, and how.
    taken: Taken,
    shown: MonoTime,
    drawn: Option<(MonoTime, LetGo)>,
}

/// How a window draw let go of the answers it held ([`PresentAnswers::let_go`]), which its
/// presents' timing lines say.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LetGo {
    /// As it was scheduled on the platform's queue, ahead of every later write of the surface it
    /// may read.
    Scheduled,
    /// As it finished on the GPU, or once it would not come.
    Finished,
}

/// The target of each answer's timing line, at debug (the module doc).
pub(crate) const TIMING: &str = "reims_vgpu::graphics::present_timing";

/// How many presents' waits after their take a show is weighed against for its late-show line
/// (the module doc).
const LATE_WINDOW: usize = 64;

/// The fewest presents' waits the late-show line takes a median of; with fewer, a present one of
/// whose writers finished after its take is logged.
const LATE_LEAST: usize = 8;

/// How many times the median wait a present's wait after its take is, at least, for its late-show
/// line.
const LATE_TIMES: u32 = 2;

/// How many command buffers not yet finished a queue keeps the host's first sight of for the
/// late-show lines: past it the oldest go.
const LATE_SEEN: usize = 256;

/// Whether the late-show lines are on ([`TIMING`] at debug): what they need is kept only then.
fn late_lines() -> bool {
    log::log_enabled!(target: TIMING, log::Level::Debug)
}

/// What the late-show lines keep (the module doc), only where they are on.
#[derive(Debug, Default)]
struct Late {
    /// By queue id: when the host first saw each of its command buffers not yet finished, by
    /// seqno, oldest first.
    seen: HashMap<u64, VecDeque<(u64, MonoTime)>>,
    /// How long each of the last [`LATE_WINDOW`] presents shown waited after its take.
    waits: VecDeque<Duration>,
}

/// A present's writer that finished after its take, the last of them so far, for its late-show
/// line.
#[derive(Clone, Copy, Debug)]
struct LastWriter {
    writer: RgfxSurfaceWriter,
    /// When the host first saw the buffer, where that was kept.
    seen: Option<MonoTime>,
    /// When its completion was posted.
    done: MonoTime,
}

impl Late {
    /// When the host first saw the buffer at `seqno` of queue `queue`, where it was kept.
    fn seen(&self, queue: u64, seqno: u64) -> Option<MonoTime> {
        let seen = self.seen.get(&queue)?;
        seen.iter().find(|&&(s, _)| s == seqno).map(|&(_, at)| at)
    }

    /// A present taken at `taken` was shown `now`, `last` its writer that finished last after its
    /// take, where one did: its wait joins the window, and where it is late (the module doc) its
    /// line is logged.
    fn shown(&mut self, taken: &Taken, last: Option<LastWriter>, now: MonoTime) {
        let waited = now.saturating_duration_since(taken.at);
        let median = (self.waits.len() >= LATE_LEAST).then(|| {
            let mut waits: Vec<Duration> = self.waits.iter().copied().collect();
            waits.sort_unstable();
            waits.get(waits.len() / 2).copied().unwrap_or_default()
        });
        let window = self.waits.len();
        if window >= LATE_WINDOW {
            self.waits.pop_front();
        }
        self.waits.push_back(waited);
        let late = match median {
            Some(median) => waited > median && waited >= median.saturating_mul(LATE_TIMES),
            None => last.is_some(),
        };
        if !late {
            return;
        }
        let against = match median {
            Some(median) => format!(
                "{:.1} times the median {} us of the last {window}",
                waited.as_secs_f64() / median.as_secs_f64(),
                median.as_micros()
            ),
            None => format!("no median yet, {window} presents shown before it"),
        };
        let us = |t: MonoTime| t.as_nanos() / 1000;
        let Some(last) = last else {
            log::debug!(
                target: TIMING,
                "a present shown late, {} us after its take ({against}): none of its writers \
                 finished after its take; on the host's monotonic clock, in us: its take {}",
                waited.as_micros(),
                us(taken.at)
            );
            return;
        };
        let (queue, seqno) = (last.writer.queue_obj_id, last.writer.seqno);
        let seen = match last.seen {
            Some(seen) if seen <= taken.at => format!(
                "first seen by the host {} us before the present's take",
                taken.at.saturating_duration_since(seen).as_micros()
            ),
            Some(seen) => format!(
                "first seen by the host {} us after the present's take",
                seen.saturating_duration_since(taken.at).as_micros()
            ),
            None => "first seen by the host at a time not kept".to_owned(),
        };
        log::debug!(
            target: TIMING,
            "a present shown late, {} us after its take ({against}): its last writer to finish, \
             the command buffer seqno {seqno} of queue {queue:#x} of context {}, {seen}, its \
             completion posted {} us after the present's take; on the host's monotonic clock, in \
             us: the present's take {}, the buffer's first sight {}, its completion {}",
            waited.as_micros(),
            queue >> RGFX_CONTEXT_OBJ_ID_SHIFT,
            last.done.saturating_duration_since(taken.at).as_micros(),
            us(taken.at),
            last.seen.map_or_else(|| "unknown".to_owned(), |seen| us(seen).to_string()),
            us(last.done)
        );
    }
}

/// When a present was taken, against the device's refreshes ([`PresentAnswers::taken`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Taken {
    /// When.
    at: MonoTime,
    /// The device's refreshes before it.
    refreshes: u64,
    /// How long after the last of them, where there was one.
    since_refresh: Option<Duration>,
}

/// What an answer's timing line says ([`PresentAnswers::refreshed`],
/// [`PresentAnswers::answered_at_show`]).
struct Timing {
    taken: Taken,
    shown: MonoTime,
    /// Whether its display has a window to draw it; where it has, whether a window draw had taken
    /// it by its answer, and when and how that draw let go.
    window: bool,
    taken_by_draw: bool,
    drawn: Option<(MonoTime, LetGo)>,
    /// The refresh that answered it, and when: where its display has no window, the last refresh
    /// before its show, which it was answered at.
    refresh: u64,
    at: MonoTime,
}

impl Timing {
    /// Logs it, at debug on [`TIMING`]'s target.
    fn log(&self) {
        let us = |from: MonoTime, to: MonoTime| to.saturating_duration_since(from).as_micros();
        let taken = &self.taken;
        let after = match taken.since_refresh {
            Some(since) => format!("{} us after refresh {}", since.as_micros(), taken.refreshes),
            None => "before the first refresh".to_owned(),
        };
        let drawn = match (self.drawn, self.taken_by_draw) {
            _ if !self.window => "no window to draw it".to_owned(),
            (Some((drawn, how)), _) => {
                let how = match how {
                    LetGo::Scheduled => "scheduled",
                    LetGo::Finished => "done",
                };
                format!("its draw {how} {} us after its show", us(self.shown, drawn))
            }
            (None, true) => "its draw under way".to_owned(),
            (None, false) => "undrawn".to_owned(),
        };
        log::debug!(
            target: TIMING,
            "a present taken {after}: shown {} us after its take, {drawn}, answered at refresh {} \
             ({} after its take), {} us after its take",
            us(taken.at, self.shown),
            self.refresh,
            self.refresh - taken.refreshes,
            us(taken.at, self.at),
        );
    }
}

/// What a draw of the window holds in [`PresentAnswers`] ([`PresentAnswers::drawn`]), which it
/// hands back as it lets go ([`PresentAnswers::let_go`]): the presents it took, by their numbers,
/// for their timing lines, and the number of the latest present shown when it began, whose
/// surface or a later one's it may read, where any was.
#[derive(Debug, Default)]
pub(crate) struct Drawing {
    took: std::ops::Range<u64>,
    reads: Option<u64>,
}

impl Drawing {
    /// Whether the draw took no present and holds no answer: none was shown when it began.
    pub(crate) fn is_empty(&self) -> bool {
        self.took.is_empty() && self.reads.is_none()
    }
}

impl<A> PresentAnswers<A> {
    /// The clock the timing lines read, now.
    pub(crate) fn now(&self) -> MonoTime {
        (self.now)()
    }

    /// No present shown yet, the timing lines read on `now`.
    pub(crate) fn new(now: Now) -> Self {
        PresentAnswers {
            state: Mutex::new(AnswersState {
                refreshes: 0,
                last_refresh: None,
                next: 0,
                owed: VecDeque::new(),
                reading: Vec::new(),
            }),
            now,
        }
    }
}

/// For the tests: a clock that stays at 0.
#[cfg(test)]
impl<A> Default for PresentAnswers<A> {
    fn default() -> Self {
        PresentAnswers::new(Box::new(MonoTime::default))
    }
}

impl<A: Answer> PresentAnswers<A> {
    /// A present is taken now: when, against the device's refreshes, for its timing line.
    pub(crate) fn taken(&self) -> Taken {
        let at = (self.now)();
        let state = self.state.lock();
        Taken {
            at,
            refreshes: state.refreshes,
            since_refresh: state.last_refresh.map(|r| at.saturating_duration_since(r)),
        }
    }

    /// A present `taken` then is shown; the window will draw it.
    pub(crate) fn shown(&self, answer: A, taken: Taken) {
        let shown = (self.now)();
        let mut state = self.state.lock();
        state.next += 1;
        state.owed.push_back(Owed {
            answer,
            taken_by_draw: false,
            taken,
            shown,
            drawn: None,
        });
    }

    /// A present `taken` then is shown on a display with no window to draw it (a headless
    /// device, or a window with no sink), which answers it at once, at its show: its timing line
    /// is logged as a refresh's are, naming the last refresh before its show as the one that
    /// answered it, and the host's summary counts it.
    pub(crate) fn answered_at_show(&self, answer: A, taken: Taken) {
        let at = (self.now)();
        answer.answer_ok();
        if log::log_enabled!(target: TIMING, log::Level::Debug) {
            let refresh = self.state.lock().refreshes;
            let timing = Timing {
                taken,
                shown: at,
                window: false,
                taken_by_draw: false,
                drawn: None,
                refresh,
                at,
            };
            timing.log();
        }
        super::summary::answered(taken.at, at, at);
    }

    /// The window draws the latest present: it takes every present shown so far that no draw
    /// took, and, until it hands its [`Drawing`] back to [`Self::let_go`], holds the answer of
    /// every present shown after the latest one now, whose surface it may read.
    pub(crate) fn drawn(&self) -> Drawing {
        let mut state = self.state.lock();
        let end = state.next;
        let mut first = end;
        for (owed, number) in state.owed.iter_mut().rev().zip((0..end).rev()) {
            if owed.taken_by_draw {
                break;
            }
            owed.taken_by_draw = true;
            first = number;
        }
        let reads = end.checked_sub(1);
        if let Some(latest) = reads {
            state.reading.push(latest);
        }
        Drawing {
            took: first..end,
            reads,
        }
    }

    /// The draw `drawing` lets go, `how`: it was scheduled where the guest's next write of the
    /// surface it may read runs after it, or it has finished on the GPU, or will not come
    /// (`DrawnPresents::after`). The presents after the one it may read are no longer held by it.
    pub(crate) fn let_go(&self, drawing: Drawing, how: LetGo) {
        if drawing.is_empty() {
            return;
        }
        let drawn = ((self.now)(), how);
        let mut state = self.state.lock();
        if let Some(reads) = drawing.reads
            && let Some(i) = state.reading.iter().position(|&r| r == reads)
        {
            state.reading.swap_remove(i);
        }
        let front = state.next - state.owed.len().to_u64();
        for number in drawing.took {
            if let Some(owed) = number
                .checked_sub(front)
                .and_then(|i| state.owed.get_mut(i.to_usize()))
                && owed.drawn.is_none()
            {
                owed.drawn = Some(drawn);
            }
        }
    }

    /// A refresh of the device's display: the presents shown before it are answered, oldest
    /// first, but those after one whose surface a window draw that holds answers may read. The
    /// guest completes its transactions oldest first, and an answer frees the surfaces of the
    /// presents before it for the guest to draw into, never its own: none is given while a write
    /// of one of those surfaces could still run before a window draw that reads it.
    pub(crate) fn refreshed(&self) {
        let at = (self.now)();
        let timed = log::log_enabled!(target: TIMING, log::Level::Debug);
        let summed = super::summary::on();
        let mut timings = Vec::new();
        let mut summary = Vec::new();
        let answers: Vec<A> = {
            let mut state = self.state.lock();
            state.refreshes += 1;
            state.last_refresh = Some(at);
            let now = state.refreshes;
            // The latest present a draw that holds may read: those after it wait for the draw.
            let held_after = state.reading.iter().copied().min();
            let mut number = state.next - state.owed.len().to_u64();
            let mut answers = Vec::new();
            while !state.owed.is_empty() && held_after.is_none_or(|r| number <= r) {
                if let Some(owed) = state.owed.pop_front() {
                    if summed {
                        summary.push((owed.taken.at, owed.shown));
                    }
                    if timed {
                        timings.push(Timing {
                            taken: owed.taken,
                            shown: owed.shown,
                            window: true,
                            taken_by_draw: owed.taken_by_draw,
                            drawn: owed.drawn,
                            refresh: now,
                            at,
                        });
                    }
                    answers.push(owed.answer);
                }
                number += 1;
            }
            answers
        };
        for answer in answers {
            answer.answer_ok();
        }
        for timing in timings {
            timing.log();
        }
        for (taken, shown) in summary {
            super::summary::answered(taken, shown, at);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::Arc;

    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;

    use reims_vgpu_vulkan::core::platform::os::clock::MonoTime;
    use reims_vgpu_vulkan::core::sync::Mutex;

    use super::{Answer, Drawing, LetGo, PresentAnswers, TIMING};

    /// An answer that records the refresh it was given at.
    struct Recorded {
        frame: usize,
        log: Arc<Mutex<Vec<(usize, u64)>>>,
        refresh: Arc<Mutex<u64>>,
    }

    impl Answer for Recorded {
        fn answer_ok(self) {
            let at = *self.refresh.lock();
            self.log.lock().push((self.frame, at));
        }
    }

    /// How the window treats each frame in [`guest_at_the_limit`].
    #[derive(Clone, Copy)]
    enum Window {
        /// Draws it as it is shown, the draw letting go before the next refresh.
        Draws,
        /// Draws it as it is shown, the draw letting go only after the next refresh.
        DrawsLate,
        /// Draws nothing (hidden).
        Hidden,
    }

    /// A guest that presents once each refresh, after it, with at most two presents unanswered
    /// (the kext's display pipe), over `refreshes` refreshes: the refreshes at which it would have
    /// had to wait, and each frame's answer as (frame, refresh it was shown after, refresh it
    /// was answered at).
    fn guest_at_the_limit(window: Window, refreshes: u64) -> (Vec<u64>, Vec<(usize, u64, u64)>) {
        let answers = PresentAnswers::<Recorded>::default();
        let log = Arc::new(Mutex::new(Vec::new()));
        let refresh = Arc::new(Mutex::new(0u64));
        let mut shown_at = Vec::new();
        let mut waited = Vec::new();
        let mut late = Drawing::default();
        for r in 1..=refreshes {
            *refresh.lock() = r;
            answers.refreshed();
            // The draw of the frame before lets go only now, just after this refresh.
            answers.let_go(std::mem::take(&mut late), LetGo::Finished);
            let unanswered = shown_at.len() - log.lock().len();
            if unanswered >= 2 {
                waited.push(r);
                continue;
            }
            let frame = shown_at.len();
            shown_at.push(r);
            answers.shown(
                Recorded {
                    frame,
                    log: Arc::clone(&log),
                    refresh: Arc::clone(&refresh),
                },
                answers.taken(),
            );
            match window {
                Window::Draws => {
                    let drawn = answers.drawn();
                    answers.let_go(drawn, LetGo::Scheduled);
                }
                Window::DrawsLate => late = answers.drawn(),
                Window::Hidden => {}
            }
        }
        let answered = log
            .lock()
            .iter()
            .map(|&(frame, at)| (frame, shown_at[frame], at))
            .collect();
        (waited, answered)
    }

    /// A guest presenting once a vblank never waits on its limit of two: each frame is answered
    /// at the refresh after its show, whether the window's draw of it lets go before that
    /// refresh, only after it, or never comes (a hidden window), as a Mac's display completes a
    /// flip at the vblank that latches it
    #[test]
    fn a_present_is_answered_at_the_vblank_after_its_show() {
        for window in [Window::Draws, Window::DrawsLate, Window::Hidden] {
            let (waited, answered) = guest_at_the_limit(window, 12);
            assert!(
                waited.is_empty(),
                "the guest waited on its limit at {waited:?}"
            );
            assert_eq!(answered.len(), 11, "a frame went unanswered");
            for (frame, shown, at) in answered {
                assert_eq!(at, shown + 1, "frame {frame}'s answer");
            }
        }
    }

    /// The refreshes [`one_queue`] runs, and the ticks of the GPU's queue from one to the next.
    const REFRESHES: u64 = 80;
    const TICKS: u64 = 4;

    /// A window draw in [`one_queue`].
    struct Draw {
        /// The surface it reads, and the tick it began at.
        surface: usize,
        began: u64,
        /// What it holds, until it lets go.
        drawing: Option<Drawing>,
        /// The ticks it runs on the GPU, once it is scheduled.
        runs: Option<std::ops::Range<u64>>,
    }

    /// The GPU's one queue, which runs what is scheduled on it in that order, a refresh every
    /// [`TICKS`] of its ticks, under a compositor that draws its frames into [`SURFACES`] surfaces
    /// in turn, each frame's write a tick on the queue and its present shown once the write has
    /// run, with at most two presents unanswered; and a window that draws each present as it is
    /// shown, each draw `slow` ticks on the queue, scheduled there `late` ticks after it began,
    /// letting go of the answers it holds then, or as it begins where `at_begin`. Over
    /// [`REFRESHES`] refreshes: the frames the guest drew; how many times a write ran on the
    /// queue before a window draw begun before it that reads its surface had run; and each
    /// frame's answer, as (the tick its present was shown at, the refresh that answered it).
    fn one_queue(slow: u64, late: u64, at_begin: bool) -> (usize, usize, Vec<(u64, u64)>) {
        let answers = PresentAnswers::<Recorded>::default();
        let log = Arc::new(Mutex::new(Vec::new()));
        let refresh = Arc::new(Mutex::new(0u64));
        // The tick from which the queue is free.
        let mut queue = 0u64;
        let mut draws: Vec<Draw> = Vec::new();
        // The frames written and not yet shown: (frame, surface, the tick its write has run by).
        let mut writing: VecDeque<(usize, usize, u64)> = VecDeque::new();
        let mut shown_at = Vec::new();
        let mut frames = 0usize;
        let mut overtaken = 0usize;
        for tick in 0..=REFRESHES * TICKS {
            // The draws due are scheduled, behind what the queue holds, and let go.
            for draw in &mut draws {
                if draw.runs.is_some() || draw.began + late > tick {
                    continue;
                }
                let start = queue.max(tick);
                queue = start + slow;
                draw.runs = Some(start..queue);
                if let Some(drawing) = draw.drawing.take() {
                    answers.let_go(drawing, LetGo::Scheduled);
                }
            }
            // The frames whose writes have run are shown, and the window draws each.
            while writing.front().is_some_and(|&(_, _, end)| end <= tick) {
                let Some((frame, surface, _)) = writing.pop_front() else {
                    break;
                };
                shown_at.push(tick);
                answers.shown(
                    Recorded {
                        frame,
                        log: Arc::clone(&log),
                        refresh: Arc::clone(&refresh),
                    },
                    answers.taken(),
                );
                let mut drawing = Some(answers.drawn());
                if at_begin && let Some(drawing) = drawing.take() {
                    answers.let_go(drawing, LetGo::Scheduled);
                }
                draws.push(Draw {
                    surface,
                    began: tick,
                    drawing,
                    runs: None,
                });
            }
            if tick % TICKS == 0 {
                *refresh.lock() = tick / TICKS;
                answers.refreshed();
            }
            // Frame f draws into surface f % SURFACES, which frame f - SURFACES showed last: free
            // once a present after that one is answered, with at most two unanswered.
            let answered = log.lock().len();
            let freed = frames < SURFACES || frames - SURFACES + 2 <= answered;
            if !freed || frames - answered >= 2 {
                continue;
            }
            let surface = frames % SURFACES;
            let start = queue.max(tick);
            queue = start + 1;
            // Every draw begun by now that reads the surface must have run before the write.
            overtaken += (draws.iter())
                .filter(|d| d.surface == surface)
                .filter(|d| d.runs.as_ref().is_none_or(|runs| start < runs.end))
                .count();
            writing.push_back((frames, surface, queue));
            frames += 1;
        }
        let answered = (log.lock().iter())
            .map(|&(frame, at)| (shown_at[frame], at))
            .collect();
        (frames, overtaken, answered)
    }

    /// The compositor's surfaces in [`one_queue`], which its frames take in turn.
    const SURFACES: usize = 3;

    /// What an answer frees is the surface of every present before it, never its own, and the
    /// guest's work that writes a freed surface goes on the GPU's one queue with the window's
    /// draws, which runs them in the order they were scheduled: a guest that draws into a surface
    /// again only once the answer of a present after the one that last showed it has come never
    /// has that write run before a window draw that reads the surface, however long the draws
    /// run on the GPU (up to four refreshes) and however late they are scheduled (up to two
    /// refreshes and more), as no draw lets go of the answers it holds before it is scheduled. A
    /// draw scheduled as it begins, as the rt platform schedules one that presents, holds no
    /// answer past the refresh after its present's show, however long it then runs. Letting go
    /// as the draw begins instead lets a write run before a draw scheduled late
    #[test]
    fn the_guest_never_writes_a_surface_ahead_of_a_window_draw_that_reads_it() {
        for slow in [1u64, 4, 8, 16] {
            for late in [0u64, 3, 9] {
                let (frames, overtaken, answered) = one_queue(slow, late, false);
                assert_eq!(
                    overtaken, 0,
                    "slow {slow}, late {late}: a write ran before a window draw that reads its \
                     surface"
                );
                assert!(
                    frames >= (REFRESHES * TICKS / (slow + late + 2)) as usize,
                    "slow {slow}, late {late}: the guest drew only {frames} frames"
                );
                if late == 0 {
                    for (shown, at) in answered {
                        assert_eq!(
                            at,
                            shown.div_ceil(TICKS),
                            "slow {slow}: the answer of a frame shown at tick {shown}"
                        );
                    }
                }
            }
        }
        let (_, overtaken, _) = one_queue(4, 9, true);
        assert!(
            overtaken > 0,
            "draws that let go as they began let no write run before them"
        );
    }

    /// No present after one whose surface a window draw may read is answered until that draw
    /// lets go, as it is scheduled, however many refreshes that takes: the later present's answer
    /// would free that surface for the guest to write before the draw runs. Once the draw is
    /// scheduled, the next refresh answers it, however long the draw then runs on the GPU. The
    /// present whose surface the draw reads is answered at the refresh after its show
    #[test]
    fn a_later_answer_waits_for_the_scheduling_of_a_draw_of_an_earlier_surface() {
        let answers = PresentAnswers::<Recorded>::default();
        let log = Arc::new(Mutex::new(Vec::new()));
        let refresh = Arc::new(Mutex::new(0u64));
        let rec = |frame| Recorded {
            frame,
            log: Arc::clone(&log),
            refresh: Arc::clone(&refresh),
        };
        answers.shown(rec(0), answers.taken());
        // The window takes frame 0, and its draw is not yet scheduled.
        let drawing = answers.drawn();
        answers.shown(rec(1), answers.taken());
        for r in 1..=3 {
            *refresh.lock() = r;
            answers.refreshed();
        }
        assert_eq!(
            *log.lock(),
            [(0, 1)],
            "frame 1 answered before frame 0's draw was scheduled"
        );
        answers.let_go(drawing, LetGo::Scheduled);
        *refresh.lock() = 4;
        answers.refreshed();
        assert_eq!(*log.lock(), [(0, 1), (1, 4)]);
    }

    /// A later draw that lets go first (scheduled, finished, or dropped) does not let go of what
    /// an earlier one holds: the presents after the earlier one's surface wait for it
    #[test]
    fn a_later_draw_done_first_waits_for_the_earlier() {
        let answers = PresentAnswers::<Recorded>::default();
        let log = Arc::new(Mutex::new(Vec::new()));
        let refresh = Arc::new(Mutex::new(0u64));
        let rec = |frame| Recorded {
            frame,
            log: Arc::clone(&log),
            refresh: Arc::clone(&refresh),
        };
        answers.shown(rec(0), answers.taken());
        let first = answers.drawn();
        answers.shown(rec(1), answers.taken());
        let second = answers.drawn();
        answers.shown(rec(2), answers.taken());
        answers.let_go(second, LetGo::Scheduled);
        for r in 1..=3 {
            *refresh.lock() = r;
            answers.refreshed();
        }
        assert_eq!(
            *log.lock(),
            [(0, 1)],
            "answered past the earlier draw's surface"
        );
        answers.let_go(first, LetGo::Finished);
        *refresh.lock() = 4;
        answers.refreshed();
        assert_eq!(*log.lock(), [(0, 1), (1, 4), (2, 4)]);
    }

    /// A draw that began before any present was shown reads the scanout, and holds no answer
    #[test]
    fn a_draw_before_any_present_holds_none() {
        let answers = PresentAnswers::<Recorded>::default();
        let log = Arc::new(Mutex::new(Vec::new()));
        let refresh = Arc::new(Mutex::new(0u64));
        let drawing = answers.drawn();
        assert!(drawing.is_empty());
        answers.shown(
            Recorded {
                frame: 0,
                log: Arc::clone(&log),
                refresh: Arc::clone(&refresh),
            },
            answers.taken(),
        );
        *refresh.lock() = 1;
        answers.refreshed();
        assert_eq!(*log.lock(), [(0, 1)]);
        answers.let_go(drawing, LetGo::Finished);
    }

    /// Each answer logs at debug where its present's time went: the refresh it was taken after,
    /// where the window's draw of it was (scheduled or done, and when, or under way), and the
    /// refresh that answered it, counted from its take
    #[test]
    fn each_answer_logs_its_timing_at_debug() {
        let logs = reims_vgpu_vulkan::core::testkit::logs::capture();
        // A clock 100 us on at each read.
        let clock = AtomicU64::new(0);
        let answers = PresentAnswers::<Recorded>::new(Box::new(move || {
            MonoTime::from_nanos(clock.fetch_add(100_000, Ordering::Relaxed))
        }));
        let log = Arc::new(Mutex::new(Vec::new()));
        let refresh = Arc::new(Mutex::new(0u64));
        let rec = |frame| Recorded {
            frame,
            log: Arc::clone(&log),
            refresh: Arc::clone(&refresh),
        };
        // Taken before any refresh, its draw scheduled at once: answered at the first.
        answers.shown(rec(0), answers.taken());
        let drawing = answers.drawn();
        answers.let_go(drawing, LetGo::Scheduled);
        answers.refreshed();
        // Taken after refresh 1, never drawn: answered at the next refresh.
        answers.shown(rec(1), answers.taken());
        answers.refreshed();
        // Taken after refresh 2, its draw not yet let go at the next.
        answers.shown(rec(2), answers.taken());
        let drawing = answers.drawn();
        answers.refreshed();
        answers.let_go(drawing, LetGo::Scheduled);
        // Taken after refresh 3, its draw finished at once (a surface a write may reach around
        // the platform's queue).
        answers.shown(rec(3), answers.taken());
        let drawing = answers.drawn();
        answers.let_go(drawing, LetGo::Finished);
        answers.refreshed();
        let lines: Vec<String> = (logs.take().into_iter())
            .filter(|(level, target, _)| *level == log::Level::Debug && target == TIMING)
            .map(|(_, _, line)| line)
            .collect();
        assert_eq!(
            lines,
            [
                "a present taken before the first refresh: shown 100 us after its take, its draw \
                 scheduled 100 us after its show, answered at refresh 1 (1 after its take), 300 us \
                 after its take",
                "a present taken 100 us after refresh 1: shown 100 us after its take, undrawn, \
                 answered at refresh 2 (1 after its take), 200 us after its take",
                "a present taken 100 us after refresh 2: shown 100 us after its take, its draw \
                 under way, answered at refresh 3 (1 after its take), 200 us after its take",
                "a present taken 200 us after refresh 3: shown 100 us after its take, its draw \
                 done 100 us after its show, answered at refresh 4 (1 after its take), 300 us \
                 after its take",
            ]
        );
        drop(logs);
        assert_eq!(log.lock().len(), 4, "all answered");
    }

    /// A display with no window to draw its presents (a headless device, or a window with no
    /// sink) answers each at its show, logs its timing line there at debug, naming the last
    /// refresh before its show as the one that answered it, and counts it in the host's summary
    #[test]
    fn a_present_with_no_window_is_answered_timed_and_counted_at_its_show() {
        let logs = reims_vgpu_vulkan::core::testkit::logs::capture();
        super::super::summary::count_presents();
        drop(super::super::summary::take_presents());
        // A clock 100 us on at each read.
        let clock = AtomicU64::new(0);
        let answers = PresentAnswers::<Recorded>::new(Box::new(move || {
            MonoTime::from_nanos(clock.fetch_add(100_000, Ordering::Relaxed))
        }));
        let log = Arc::new(Mutex::new(Vec::new()));
        let refresh = Arc::new(Mutex::new(0u64));
        let rec = |frame| Recorded {
            frame,
            log: Arc::clone(&log),
            refresh: Arc::clone(&refresh),
        };
        // Taken before any refresh: answered at its show.
        answers.answered_at_show(rec(0), answers.taken());
        assert_eq!(*log.lock(), [(0, 0)]);
        // Taken after refresh 1, shown after refresh 2: answered at its show, no refresh later.
        *refresh.lock() = 1;
        answers.refreshed();
        let taken = answers.taken();
        *refresh.lock() = 2;
        answers.refreshed();
        answers.answered_at_show(rec(1), taken);
        assert_eq!(*log.lock(), [(0, 0), (1, 2)]);
        let lines: Vec<String> = (logs.take().into_iter())
            .filter(|(level, target, _)| *level == log::Level::Debug && target == TIMING)
            .map(|(_, _, line)| line)
            .collect();
        assert_eq!(
            lines,
            [
                "a present taken before the first refresh: shown 100 us after its take, no window \
                 to draw it, answered at refresh 0 (0 after its take), 100 us after its take",
                "a present taken 100 us after refresh 1: shown 200 us after its take, no window to \
                 draw it, answered at refresh 2 (1 after its take), 200 us after its take",
            ]
        );
        let us = Duration::from_micros;
        assert_eq!(
            super::super::summary::take_presents(),
            [[us(100), us(0), us(100)], [us(200), us(0), us(200)]]
        );
    }
}
