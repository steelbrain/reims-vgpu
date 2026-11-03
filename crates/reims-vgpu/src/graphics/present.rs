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
//! those. A buffer of another queue, or one that writes another surface, holds nothing.
//!
//! A present whose writers have all finished, with none before it waiting, is shown at once; the
//! others wait in order, and each `CB_COMPLETE` shows those at the front that are ready, so a
//! present never overtakes one made before it. A process publishes a buffer on its queue's ring
//! before it names it a writer, so a writer whose seqno is past every seqno its queue's ring has
//! published when the present comes names no buffer, and holds nothing: a seqno the guest wrote
//! wrong never holds its display's presents. A failed buffer finishes as a completed one does
//! (its `CB_COMPLETE` carries the failure). A queue's destroy, or its context's close, drops what
//! the presents wait for of it: no buffer of it will finish.
//!
//! A present of a surface whose device cache holds an older CPU generation than the present's
//! waits too for the fill the host commits for it on taking the present (`decode/surface_cache.rs`):
//! it is shown once that fill has finished, which tells the queue ([`PresentQueue::filled`]), so
//! the presents' lane that carries the present out (`reims-graphics`' device) waits for nothing on
//! the GPU.
//!
//! [`PresentAnswers`]: when a shown present is answered. The guest's display pipe completes a
//! transaction on its answer, and holds at most two unanswered (the kext's `RgfxDisplayPipe`),
//! so the answer says the frame is on the display, as a display's flip completes at the vblank
//! that latches it: it is given at the first refresh of the device's display (the refresh thread,
//! [`reims_graphics::Host::refreshed`], after that refresh's `VBLANK`) once the window's draw of
//! it has finished on the host's GPU, the copy out of the surface done, so the guest may draw
//! into the surface again. A draw the window drops before its commit, or one that shows the
//! scanout in place of a surface it cannot draw, answers its presents at the next refresh too.
//! A present the window has not drawn by the second refresh after it was shown (the window is
//! hidden, or takes no drawable) is answered there: the device's display shows it whether or not
//! the host's window does, and the guest's compositor must not wait on the window. A guest that
//! presents once a vblank so never meets its limit of two: a frame drawn as it is made is
//! answered at the next vblank, and one the window does not draw at the second.
//!
//! The answers go in the order the presents were shown, none while the window's draw of an
//! earlier present still runs: the guest completes its transactions oldest first, so a later
//! present's answer completes every earlier one too and frees its surface for the guest's
//! compositor to draw into, while the window's draw may still be copying out of it. Answered
//! out of order, a present the window had not drawn by its second refresh freed the surface
//! of a draw still running on the GPU; the compositor's next frame then landed in it under the
//! copy, and the window showed that newer frame, then the older one it had presented between,
//! then the newer again.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use reims_graphics::wire::{RGFX_CONTEXT_OBJ_ID_SHIFT, RGFX_STATUS_OK, RgfxSurfaceWriter};
use reims_graphics::{CompletionPoster, PresentAnswer};
use reims_vgpu_core::num::{ToU64 as _, ToUsize as _};
use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::sync::Mutex;

use super::Presented;
use super::decode::FillWait;

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
    /// By queue id: the highest seqno of a `CB_COMPLETE` the host has posted on its ring.
    finished: HashMap<u64, u64>,
    /// The presents not yet shown, oldest first.
    waiting: VecDeque<Waiting<P>>,
    /// By queue id: the poster onto its completion ring, through which what its submission ring
    /// has published is read.
    queues: HashMap<u64, CompletionPoster>,
}

/// One present not yet shown.
struct Waiting<P: Platform> {
    /// The IOSurface's id; 0 for the scanout.
    id: u32,
    /// The texture over it; `None` for the scanout.
    texture: Option<P::Texture>,
    /// The present's answer, owed once it is shown.
    answer: PresentAnswer,
    /// The command buffers it waits for.
    writers: Vec<RgfxSurfaceWriter>,
    /// The fill of its surface's device cache it waits for, where it asked one.
    fill: Option<FillWait>,
}

impl<P: Platform> PresentQueue<P> {
    pub(crate) fn new(presented: Arc<Presented<P>>) -> Self {
        PresentQueue {
            presented,
            state: Mutex::new(State {
                finished: HashMap::new(),
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

    /// Shows IOSurface `id`'s `texture` (`None`: the scanout) once each command buffer in
    /// `writers` has finished, and `fill`, its surface's device cache's fill, where it has one,
    /// and every present made before it is shown; `answer` goes as [`Presented`] gives it. A
    /// writer past what its queue's ring has published holds nothing (the module doc). Called
    /// with no lock of the decoder's held: what a ring has published is read with the ring held.
    pub(crate) fn present(
        &self,
        id: u32,
        texture: Option<P::Texture>,
        answer: PresentAnswer,
        writers: Vec<RgfxSurfaceWriter>,
        fill: Option<FillWait>,
    ) {
        let writers = self.published(writers);
        let mut state = self.state.lock();
        state.waiting.push_back(Waiting {
            id,
            texture,
            answer,
            writers,
            fill,
        });
        self.show_ready(&mut state);
    }

    /// A fill of a device cache finished: the presents at the front waiting for nothing more are
    /// shown.
    pub(crate) fn filled(&self) {
        let mut state = self.state.lock();
        if !state.waiting.is_empty() {
            self.show_ready(&mut state);
        }
    }

    /// The host posted the `CB_COMPLETE` of the command buffer at `seqno` on queue `queue`'s ring,
    /// through `poster`. A buffer of a queue the device's reset ended counts for nothing: the
    /// rebooted guest's queue may take the same id, its seqnos from the start. The poster is
    /// looked at with the state held, which the reset takes once the queue has ended
    /// ([`Self::reset`]), so no buffer of the queue counts after the reset.
    pub(crate) fn finished(&self, queue: u64, seqno: u64, poster: &CompletionPoster) {
        let mut state = self.state.lock();
        if poster.stale() {
            return;
        }
        let done = state.finished.entry(queue).or_insert(0);
        *done = (*done).max(seqno);
        if !state.waiting.is_empty() {
            self.show_ready(&mut state);
        }
    }

    /// `writers`, less each whose seqno is past every seqno its queue's ring has published.
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
                let published = poster.as_ref().and_then(CompletionPoster::published_seqno);
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
            state.queues.clear();
            std::mem::take(&mut state.waiting)
        };
        if let Err(refusal) = reims_vgpu_core::raise::refusals::refusing(move || drop(waiting)) {
            log::warn!(
                target: "reims_vgpu::graphics",
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
        for waiting in &mut state.waiting {
            waiting.writers.retain(|w| !gone(w.queue_obj_id));
        }
        self.show_ready(&mut state);
    }

    /// Shows the presents at the front whose writers have finished, oldest first.
    fn show_ready(&self, state: &mut State<P>) {
        loop {
            let ready = state.waiting.front().is_some_and(|front| {
                front.fill.as_ref().is_none_or(FillWait::done)
                    && front.writers.iter().all(|w| {
                        state
                            .finished
                            .get(&w.queue_obj_id)
                            .is_some_and(|&done| done >= w.seqno)
                    })
            });
            if !ready {
                return;
            }
            let Some(shown) = state.waiting.pop_front() else {
                return;
            };
            self.presented.show(shown.id, shown.texture, shown.answer);
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
}

/// [`PresentAnswers`]' state.
struct AnswersState<A> {
    /// The device's refreshes so far.
    refreshes: u64,
    /// The number the next present shown takes.
    next: u64,
    /// The presents shown and not yet answered, in the order shown, their numbers counting up
    /// from the front's.
    owed: VecDeque<Owed<A>>,
}

/// A present shown and not yet answered.
struct Owed<A> {
    answer: A,
    /// The refresh count at its show.
    shown_at: u64,
    /// Where the window's draw of it is.
    draw: Draw,
}

/// Where the window's draw of a present is.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Draw {
    /// The window has not taken it.
    Not,
    /// A draw took it and has not finished.
    Running,
    /// Its draw finished on the GPU, or will not come.
    Done,
}

/// The presents a draw of the window took ([`PresentAnswers::drawn`]), by their numbers, which
/// it hands back once it has finished or will not ([`PresentAnswers::done`]).
#[derive(Debug, Default)]
pub(crate) struct Drawing(std::ops::Range<u64>);

impl Drawing {
    /// Whether the draw took no present.
    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl<A> Default for PresentAnswers<A> {
    fn default() -> Self {
        PresentAnswers {
            state: Mutex::new(AnswersState {
                refreshes: 0,
                next: 0,
                owed: VecDeque::new(),
            }),
        }
    }
}

impl<A: Answer> PresentAnswers<A> {
    /// A present is shown; the window will draw it.
    pub(crate) fn shown(&self, answer: A) {
        let mut state = self.state.lock();
        let shown_at = state.refreshes;
        state.next += 1;
        state.owed.push_back(Owed {
            answer,
            shown_at,
            draw: Draw::Not,
        });
    }

    /// The window draws the latest present: every present shown so far that no draw took, which
    /// the draw hands back to [`Self::done`] once it has finished or will not.
    pub(crate) fn drawn(&self) -> Drawing {
        let mut state = self.state.lock();
        let end = state.next;
        let mut first = end;
        for (owed, number) in state.owed.iter_mut().rev().zip((0..end).rev()) {
            if owed.draw != Draw::Not {
                break;
            }
            owed.draw = Draw::Running;
            first = number;
        }
        Drawing(first..end)
    }

    /// The draw that took `drawing`'s presents has finished on the GPU, or will not come: they
    /// are answered at the next refresh, once every present shown before them is.
    pub(crate) fn done(&self, drawing: Drawing) {
        if drawing.is_empty() {
            return;
        }
        let mut state = self.state.lock();
        let front = state.next - state.owed.len().to_u64();
        for number in drawing.0 {
            if let Some(owed) = number
                .checked_sub(front)
                .and_then(|i| state.owed.get_mut(i.to_usize()))
            {
                owed.draw = Draw::Done;
            }
        }
    }

    /// A refresh of the device's display: the presents whose draw has finished are answered, and
    /// those no draw took by the second refresh after their show, oldest first, up to the first
    /// whose draw still runs or that is not yet due. The guest completes its transactions oldest
    /// first, so an answer frees every earlier present's surface for the guest to draw into: none
    /// is given while the window's draw of an earlier one still reads its surface.
    pub(crate) fn refreshed(&self) {
        let answers: Vec<A> = {
            let mut state = self.state.lock();
            state.refreshes += 1;
            let now = state.refreshes;
            let mut answers = Vec::new();
            while let Some(front) = state.owed.front() {
                let due = match front.draw {
                    Draw::Done => true,
                    Draw::Not => front.shown_at + 2 <= now,
                    Draw::Running => false,
                };
                if !due {
                    break;
                }
                if let Some(owed) = state.owed.pop_front() {
                    answers.push(owed.answer);
                }
            }
            answers
        };
        for answer in answers {
            answer.answer_ok();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use reims_vgpu_core::sync::Mutex;

    use super::{Answer, Drawing, PresentAnswers};

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
            answers.shown(Recorded {
                frame,
                log: Arc::clone(&log),
                refresh: Arc::clone(&refresh),
            });
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

    /// A guest presenting once a vblank never waits on its limit of two: a frame the window draws
    /// as it is shown is answered at the next refresh, one whose draw finishes a refresh late at
    /// the one after, and one a hidden window never draws at the second
    #[test]
    fn a_present_is_answered_at_the_vblank_it_is_first_shown() {
        for (window, delay) in [
            (Window::Draws, 1u64),
            (Window::DrawsLate, 2),
            (Window::Hidden, 2),
        ] {
            let (waited, answered) = guest_at_the_limit(window, 12);
            assert!(
                waited.is_empty(),
                "the guest waited on its limit at {waited:?}"
            );
            assert_eq!(answered.len() as u64, 12 - delay, "a frame went unanswered");
            for (frame, shown, at) in answered {
                assert_eq!(at, shown + delay, "frame {frame}'s answer");
            }
        }
    }

    /// No present is answered while one shown before it waits for its draw to finish: the guest
    /// completes its transactions oldest first, so a later present's answer would free the
    /// earlier one's surface for the guest to draw into while the window's draw still reads it
    #[test]
    fn a_late_answer_never_overtakes_a_draw_still_running() {
        let answers = PresentAnswers::<Recorded>::default();
        let log = Arc::new(Mutex::new(Vec::new()));
        let refresh = Arc::new(Mutex::new(0u64));
        let rec = |frame| Recorded {
            frame,
            log: Arc::clone(&log),
            refresh: Arc::clone(&refresh),
        };
        answers.shown(rec(0));
        // The window takes frame 0 and its draw runs on.
        let drawing = answers.drawn();
        answers.shown(rec(1));
        for r in 1..=3 {
            *refresh.lock() = r;
            answers.refreshed();
        }
        assert!(
            log.lock().is_empty(),
            "frame 1 answered while frame 0's draw runs: {:?}",
            log.lock()
        );
        answers.done(drawing);
        *refresh.lock() = 4;
        answers.refreshed();
        assert_eq!(*log.lock(), [(0, 4), (1, 4)]);
    }

    /// A later draw that finishes, or is dropped, first waits for the earlier one still running:
    /// both are answered at the refresh after the earlier finishes, in order
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
        answers.shown(rec(0));
        let first = answers.drawn();
        answers.shown(rec(1));
        let second = answers.drawn();
        answers.done(second);
        for r in 1..=3 {
            *refresh.lock() = r;
            answers.refreshed();
        }
        assert!(log.lock().is_empty(), "answered: {:?}", log.lock());
        answers.done(first);
        *refresh.lock() = 4;
        answers.refreshed();
        assert_eq!(*log.lock(), [(0, 4), (1, 4)]);
    }

    /// A present is never answered before the window's draw of it has finished, while the
    /// window draws: a draw still running at the second refresh holds its answers
    #[test]
    fn a_drawn_present_waits_for_its_draw() {
        let answers = PresentAnswers::<Recorded>::default();
        let log = Arc::new(Mutex::new(Vec::new()));
        let refresh = Arc::new(Mutex::new(0u64));
        answers.shown(Recorded {
            frame: 0,
            log: Arc::clone(&log),
            refresh: Arc::clone(&refresh),
        });
        let drawing = answers.drawn();
        for r in 1..=3 {
            *refresh.lock() = r;
            answers.refreshed();
        }
        assert!(log.lock().is_empty());
        answers.done(drawing);
        assert!(log.lock().is_empty(), "answered before the refresh");
        *refresh.lock() = 4;
        answers.refreshed();
        assert_eq!(*log.lock(), [(0, 4)]);
    }
}
