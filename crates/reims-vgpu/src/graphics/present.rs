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
//! present never overtakes one made before it. A failed buffer finishes as a completed one does
//! (its `CB_COMPLETE` carries the failure). A queue's destroy, or its context's close, drops what
//! the presents wait for of it: no buffer of it will finish.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use reims_graphics::PresentAnswer;
use reims_graphics::wire::{RGFX_CONTEXT_OBJ_ID_SHIFT, RgfxSurfaceWriter};
use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::sync::Mutex;

use super::Presented;

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
}

impl<P: Platform> PresentQueue<P> {
    pub(crate) fn new(presented: Arc<Presented<P>>) -> Self {
        PresentQueue {
            presented,
            state: Mutex::new(State {
                finished: HashMap::new(),
                waiting: VecDeque::new(),
            }),
        }
    }

    /// Shows IOSurface `id`'s `texture` (`None`: the scanout) once each command buffer in
    /// `writers` has finished and every present made before it is shown; `answer` goes as
    /// [`Presented`] gives it.
    pub(crate) fn present(
        &self,
        id: u32,
        texture: Option<P::Texture>,
        answer: PresentAnswer,
        writers: Vec<RgfxSurfaceWriter>,
    ) {
        let mut state = self.state.lock();
        state.waiting.push_back(Waiting {
            id,
            texture,
            answer,
            writers,
        });
        self.show_ready(&mut state);
    }

    /// The host posted the `CB_COMPLETE` of the command buffer at `seqno` on queue `queue`'s ring.
    pub(crate) fn finished(&self, queue: u64, seqno: u64) {
        let mut state = self.state.lock();
        let done = state.finished.entry(queue).or_insert(0);
        *done = (*done).max(seqno);
        if !state.waiting.is_empty() {
            self.show_ready(&mut state);
        }
    }

    /// Queue `queue` is gone: nothing more of it finishes.
    pub(crate) fn queue_gone(&self, queue: u64) {
        self.forget(|q| q == queue);
    }

    /// Context `context` closed: none of its queues' command buffers finish.
    pub(crate) fn context_gone(&self, context: u32) {
        self.forget(|q| q >> RGFX_CONTEXT_OBJ_ID_SHIFT == u64::from(context));
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
        while let Some(front) = state.waiting.front() {
            let ready = front.writers.iter().all(|w| {
                state
                    .finished
                    .get(&w.queue_obj_id)
                    .is_some_and(|&done| done >= w.seqno)
            });
            if !ready {
                return;
            }
            let shown = state.waiting.pop_front().expect("the front present");
            self.presented.show(shown.id, shown.texture, shown.answer);
        }
    }

    /// The presents not yet shown (for the tests).
    #[cfg(test)]
    pub(crate) fn waiting(&self) -> usize {
        self.state.lock().waiting.len()
    }
}
