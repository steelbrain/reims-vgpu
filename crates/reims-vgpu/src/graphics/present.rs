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
//! present never overtakes one made before it. A process publishes a buffer on its queue's ring
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
//! none while the window's draw of an earlier present still runs, which may still be copying out
//! of that present's surface; a present whose own draw still runs is answered, and holds the
//! answers after it until its draw has finished. Answered out of order, a present the window had
//! not drawn freed the surface of a draw still running on the GPU; the compositor's next frame
//! then landed in it under the copy, and the window showed that newer frame, then the older one
//! it had presented between, then the newer again. Waiting for a present's own draw instead held
//! its answer a refresh where the draw ended past the first refresh after its show (an RTX 5080
//! Laptop GPU at 30 frames a second: the show 6.0 ms after a 120 Hz refresh, the draw done 5.2 ms
//! later), which the guest saw as a frame a refresh late.
//!
//! **Where a present's time goes.** Each answer is logged at debug, on [`TIMING`]'s target:
//! how long after the device's last refresh its `PRESENT_SURFACE` was taken, how long it waited
//! for its writers and fill to be shown, how long the window's draw of it took to finish, and at
//! which refresh it was answered. It tells a present the host answers a refresh late (its show
//! or its draw done past the first refresh after its take) from one the guest's compositor
//! sent late in the refresh, which a guest's present latency alone does not.
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
    /// What the display shows, which a present changes when it is shown.
    presented: Arc<Presented<P>>,
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
    /// The presents not yet shown, oldest first.
    waiting: VecDeque<Waiting<P>>,
    /// By queue id: the poster onto its completion ring, through which what its submission ring
    /// has published is read.
    queues: HashMap<u64, CompletionPoster>,
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
}

impl<P: Platform> PresentQueue<P> {
    pub(crate) fn new(presented: Arc<Presented<P>>) -> Self {
        PresentQueue {
            presented,
            state: Mutex::new(State {
                finished: HashMap::new(),
                apart: HashMap::new(),
                waiting: VecDeque::new(),
                queues: HashMap::new(),
            }),
        }
    }

    /// Queue `queue` was made, its completion ring posted onto through `poster`: what its
    /// submission ring has published bounds what a present waits for of it.
    pub(crate) fn queue_made(&self, queue: u64, poster: CompletionPoster) {
        self.state.lock().queues.insert(queue, poster);
    }

    /// Shows IOSurface `id`'s `texture` (`None`: the scanout), with the colour stages `color`
    /// sets, once each command buffer in `writers` has finished, and `fill`, its surface's
    /// device cache's fill, where it has one, and every present made before it is shown;
    /// `answer` goes as [`Presented`] gives it. A writer of a queue the host does not hold, or
    /// past what its queue's ring has published, holds nothing (the module doc). Called with no
    /// lock of the decoder's held: what a ring has published is read with the ring held.
    pub(crate) fn present(
        &self,
        id: u32,
        texture: Option<ShownTexture<P>>,
        color: Option<ColorChange>,
        answer: PresentAnswer,
        writers: Vec<RgfxSurfaceWriter>,
        fill: Option<FillWait>,
    ) {
        let taken = self.presented.answers.taken();
        let writers = self.published(writers);
        self.push(Waiting {
            shows: Shows::Surface(id, texture),
            color,
            answer,
            writers,
            fill,
            taken,
        });
    }

    /// A present that keeps what the display shows, with the colour stages `color` sets
    /// (`RGFX_PRESENT_KEEP`): shown once every present made before it is.
    pub(crate) fn present_kept(&self, color: Option<ColorChange>, answer: PresentAnswer) {
        self.push(Waiting {
            shows: Shows::Kept,
            color,
            answer,
            writers: Vec::new(),
            fill: None,
            taken: self.presented.answers.taken(),
        });
    }

    /// `waiting` joins the presents, which are shown as they are ready.
    fn push(&self, waiting: Waiting<P>) {
        let mut state = self.state.lock();
        state.waiting.push_back(waiting);
        let gone = self.show_ready(&mut state);
        drop(state);
        drop(gone);
    }

    /// A fill of a device cache finished: the presents at the front waiting for nothing more are
    /// shown.
    pub(crate) fn filled(&self) {
        let mut state = self.state.lock();
        if !state.waiting.is_empty() {
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
        *done = (*done).max(seqno);
        let done = *done;
        if let Some(apart) = state.apart.get_mut(&queue) {
            apart.retain(|&s| s > done);
            if apart.is_empty() {
                state.apart.remove(&queue);
            }
        }
        if !state.waiting.is_empty() {
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
        if !state.waiting.is_empty() {
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
            std::mem::take(&mut state.waiting)
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
        for waiting in &mut state.waiting {
            waiting.writers.retain(|w| !gone(w.queue_obj_id));
        }
        let let_go = self.show_ready(&mut state);
        drop(state);
        drop(let_go);
    }

    /// Shows the presents at the front whose writers have finished, oldest first. The holds of
    /// the textures they replace are returned, for the caller to let go of once it has let go of
    /// the state: a release waiting for one is answered there, off the present's path.
    #[must_use]
    fn show_ready(&self, state: &mut State<P>) -> Vec<Arc<DisplayHold>> {
        let mut gone = Vec::new();
        loop {
            let ready = state.waiting.front().is_some_and(|front| {
                front.fill.as_ref().is_none_or(FillWait::done)
                    && front.writers.iter().all(|w| {
                        let queue = &w.queue_obj_id;
                        (state.finished.get(queue)).is_some_and(|&done| done >= w.seqno)
                            || (state.apart.get(queue)).is_some_and(|a| a.contains(&w.seqno))
                    })
            });
            if !ready {
                return gone;
            }
            let Some(shown) = state.waiting.pop_front() else {
                return gone;
            };
            gone.extend((self.presented).show(shown.shows, shown.color, shown.answer, shown.taken));
        }
    }

    /// The presents not yet shown (for the tests).
    #[cfg(test)]
    pub(crate) fn waiting(&self) -> usize {
        self.state.lock().waiting.len()
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
    /// The window's draws that have not finished, each by the number of the latest present shown
    /// when it began: it may read that present's surface, or a later one's, and no earlier one's.
    reading: Vec<u64>,
}

/// A present shown and not yet answered.
struct Owed<A> {
    answer: A,
    /// Whether a window draw has taken it ([`PresentAnswers::drawn`]), for its timing line.
    taken_by_draw: bool,
    /// When it was taken, shown, and the first draw that took it finished (or was dropped).
    taken: Taken,
    shown: MonoTime,
    drawn: Option<MonoTime>,
}

/// The target of each answer's timing line, at debug (the module doc).
pub(crate) const TIMING: &str = "reims_vgpu::graphics::present_timing";

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

/// What an answer's timing line says ([`PresentAnswers::refreshed`]).
struct Timing {
    taken: Taken,
    shown: MonoTime,
    /// Whether a window draw had taken it by its answer, and when that draw finished.
    taken_by_draw: bool,
    drawn: Option<MonoTime>,
    /// The refresh that answered it, and when.
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
            (Some(drawn), _) => {
                format!("its draw done {} us after its show", us(self.shown, drawn))
            }
            (None, true) => "its draw running".to_owned(),
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
/// hands back once it has finished or will not ([`PresentAnswers::done`]): the presents it took,
/// by their numbers, for their timing lines, and the number of the latest present shown when it
/// began, whose surface or a later one's it may read, where any was.
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

    /// The window draws the latest present: it takes every present shown so far that no draw
    /// took, and, until it hands its [`Drawing`] back to [`Self::done`], holds the answer of every
    /// present shown after the latest one now, whose surface it may read.
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

    /// The draw `drawing` has finished on the GPU, or will not come: the presents after the one
    /// it may read are no longer held by it.
    pub(crate) fn done(&self, drawing: Drawing) {
        if drawing.is_empty() {
            return;
        }
        let drawn = (self.now)();
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
    /// first, but those after one whose surface a window draw still running may read. The guest
    /// completes its transactions oldest first, and an answer frees the surfaces of the presents
    /// before it for the guest to draw into, never its own: none is given while the window may
    /// still read one of those surfaces.
    pub(crate) fn refreshed(&self) {
        let at = (self.now)();
        let timed = log::log_enabled!(target: TIMING, log::Level::Debug);
        let mut timings = Vec::new();
        let answers: Vec<A> = {
            let mut state = self.state.lock();
            state.refreshes += 1;
            state.last_refresh = Some(at);
            let now = state.refreshes;
            // The latest present a running draw may read: those after it wait for the draw.
            let held_after = state.reading.iter().copied().min();
            let mut number = state.next - state.owed.len().to_u64();
            let mut answers = Vec::new();
            while !state.owed.is_empty() && held_after.is_none_or(|r| number <= r) {
                if let Some(owed) = state.owed.pop_front() {
                    if timed {
                        timings.push(Timing {
                            taken: owed.taken,
                            shown: owed.shown,
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
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use std::sync::atomic::{AtomicU64, Ordering};

    use reims_vgpu_vulkan::core::platform::os::clock::MonoTime;
    use reims_vgpu_vulkan::core::sync::Mutex;

    use super::{Answer, Drawing, PresentAnswers, TIMING};

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
        /// Draws it as it is shown, the draw finishing before the next refresh.
        Draws,
        /// Draws it as it is shown, the draw finishing only after the next refresh.
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
            // The draw of the frame before finishes only now, just after this refresh.
            answers.done(std::mem::take(&mut late));
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
                    answers.done(drawn);
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
    /// at the refresh after its show, whether the window's draw of it finishes before that
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

    /// What an answer frees is the surface of every present before it, never its own: a guest
    /// that draws into a surface again only once the answer of a present after the one that last
    /// showed it has come never draws into a surface a window draw still reads, however slow the
    /// window's draws are, each reading the latest surface shown and running up to four refreshes
    #[test]
    fn the_guest_never_draws_into_a_surface_a_window_draw_still_reads() {
        // The compositor's surfaces, which its frames take in turn.
        const SURFACES: usize = 3;
        for slow in [0u64, 1, 2, 4] {
            let answers = PresentAnswers::<Recorded>::default();
            let log = Arc::new(Mutex::new(Vec::new()));
            let refresh = Arc::new(Mutex::new(0u64));
            // The window's draws still running: the surface each reads, the refresh it finishes
            // just before, and what it holds.
            let mut running: Vec<(usize, u64, Drawing)> = Vec::new();
            let mut frames = 0usize;
            let mut writes = 0usize;
            for r in 1..=80u64 {
                let (finished, still): (Vec<_>, Vec<_>) =
                    running.into_iter().partition(|(_, end, _)| *end <= r);
                running = still;
                for (_, _, drawing) in finished {
                    answers.done(drawing);
                }
                *refresh.lock() = r;
                answers.refreshed();
                let answered = log.lock().len();
                // Frame f draws into surface f % SURFACES, which frame f - SURFACES showed last:
                // free once a present after that one is answered, with at most two unanswered.
                let freed = frames < SURFACES || frames - SURFACES + 2 <= answered;
                if !freed || frames - answered >= 2 {
                    continue;
                }
                let surface = frames % SURFACES;
                assert!(
                    running.iter().all(|(read, _, _)| *read != surface),
                    "slow {slow}: at refresh {r} frame {frames} draws into surface {surface}, \
                     which a window draw still reads"
                );
                writes += 1;
                answers.shown(
                    Recorded {
                        frame: frames,
                        log: Arc::clone(&log),
                        refresh: Arc::clone(&refresh),
                    },
                    answers.taken(),
                );
                frames += 1;
                // The window draws the frame just shown, finishing `slow` refreshes on.
                running.push((surface, r + 1 + slow, answers.drawn()));
            }
            assert!(
                writes >= (80 / (slow + 2)) as usize,
                "slow {slow}: the guest drew only {writes} frames in 80 refreshes"
            );
            if slow == 0 {
                for &(frame, at) in log.lock().iter() {
                    assert!(
                        at <= frame as u64 + 2,
                        "frame {frame} answered at refresh {at}"
                    );
                }
            }
        }
    }

    /// No present after one whose surface a window draw still reads is answered until that draw
    /// finishes, however many refreshes it takes: the later present's answer would free that
    /// surface for the guest to draw into under the copy. The present whose surface the draw
    /// reads is answered at the refresh after its show
    #[test]
    fn a_later_answer_waits_for_a_draw_of_an_earlier_surface() {
        let answers = PresentAnswers::<Recorded>::default();
        let log = Arc::new(Mutex::new(Vec::new()));
        let refresh = Arc::new(Mutex::new(0u64));
        let rec = |frame| Recorded {
            frame,
            log: Arc::clone(&log),
            refresh: Arc::clone(&refresh),
        };
        answers.shown(rec(0), answers.taken());
        // The window takes frame 0 and its draw runs on.
        let drawing = answers.drawn();
        answers.shown(rec(1), answers.taken());
        for r in 1..=3 {
            *refresh.lock() = r;
            answers.refreshed();
        }
        assert_eq!(
            *log.lock(),
            [(0, 1)],
            "frame 1 answered while frame 0's draw runs"
        );
        answers.done(drawing);
        *refresh.lock() = 4;
        answers.refreshed();
        assert_eq!(*log.lock(), [(0, 1), (1, 4)]);
    }

    /// A later draw that finishes, or is dropped, first does not let go of what an earlier one
    /// still running holds: the presents after the earlier one's surface wait for it
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
        answers.done(second);
        for r in 1..=3 {
            *refresh.lock() = r;
            answers.refreshed();
        }
        assert_eq!(
            *log.lock(),
            [(0, 1)],
            "answered past the earlier draw's surface"
        );
        answers.done(first);
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
        answers.done(drawing);
    }

    /// Each answer logs at debug where its present's time went: the refresh it was taken after,
    /// where the window's draw of it was, and the refresh that answered it, counted from its take
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
        // Taken before any refresh, drawn at once: answered at the first.
        answers.shown(rec(0), answers.taken());
        let drawing = answers.drawn();
        answers.done(drawing);
        answers.refreshed();
        // Taken after refresh 1, never drawn: answered at the next refresh.
        answers.shown(rec(1), answers.taken());
        answers.refreshed();
        // Taken after refresh 2, its draw still running at the next.
        answers.shown(rec(2), answers.taken());
        let drawing = answers.drawn();
        answers.refreshed();
        answers.done(drawing);
        let lines: Vec<String> = (logs.take().into_iter())
            .filter(|(level, target, _)| *level == log::Level::Debug && target == TIMING)
            .map(|(_, _, line)| line)
            .collect();
        assert_eq!(
            lines,
            [
                "a present taken before the first refresh: shown 100 us after its take, its draw \
                 done 100 us after its show, answered at refresh 1 (1 after its take), 300 us \
                 after its take",
                "a present taken 100 us after refresh 1: shown 100 us after its take, undrawn, \
                 answered at refresh 2 (1 after its take), 200 us after its take",
                "a present taken 100 us after refresh 2: shown 100 us after its take, its draw \
                 running, answered at refresh 3 (1 after its take), 200 us after its take",
            ]
        );
        drop(logs);
        assert_eq!(log.lock().len(), 3, "all answered");
    }
}
