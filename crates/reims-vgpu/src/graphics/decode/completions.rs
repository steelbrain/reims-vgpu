//! A queue's order of `CB_COMPLETE` records, which the guest takes as its command buffers'
//! completion order on the queue, as Metal completes a queue's buffers in their commit order.
//!
//! A command buffer takes a turn in its queue's [`Completions`] in its commit order (in its turn
//! in its queue's lane, [`super::lane`]), and finishes it with the post of its record
//! ([`CompletionTurn::finish`]): a post runs once every turn before it has finished, in turn
//! order, one at a time, on the thread that finished the turn at the front or the turn that let it
//! run. The platform completes a queue's buffers in their commit order, so a buffer's turn is at
//! the front by its completion handler, and its post runs there; the order holds a post back only
//! where a buffer posts its completion ahead of its platform end, at its decode
//! ([`super::video::Landed`], "Completed at the decode" in its module docs), behind a buffer
//! committed before it whose own is still to come. A turn let go of unfinished (its buffer never
//! committed, or its handler dropped) finishes with no post.

use std::collections::BTreeMap;
use std::sync::Arc;

use reims_vgpu_vulkan::core::sync::Mutex;

/// A record's post, run in its turn.
pub(crate) type Post = Box<dyn FnOnce() + Send>;

/// A queue's turns of completion posts (the module docs).
#[derive(Default)]
pub(crate) struct Completions(Mutex<Turns>);

/// [`Completions`]' state.
#[derive(Default)]
struct Turns {
    /// The next turn to give.
    given: u64,
    /// The turn whose post runs next.
    front: u64,
    /// The turns finished past the front, with their posts.
    ahead: BTreeMap<u64, Option<Post>>,
    /// A thread runs the posts at the front.
    running: bool,
}

/// A command buffer's turn in its queue's [`Completions`], finished once: by
/// [`Self::finish`], or with no post as it is dropped.
pub(crate) struct CompletionTurn {
    line: Arc<Completions>,
    at: u64,
    finished: bool,
}

impl Completions {
    /// The next turn, after every turn given before it.
    pub(crate) fn turn(self: &Arc<Self>) -> CompletionTurn {
        let mut turns = self.0.lock();
        let at = turns.given;
        turns.given += 1;
        CompletionTurn {
            line: Arc::clone(self),
            at,
            finished: false,
        }
    }

    /// Turn `at` finished with `post`: runs it, and each finished turn after it, where it is at
    /// the front and no thread runs them; else leaves it to the thread that does.
    fn finish(&self, at: u64, post: Option<Post>) {
        let mut turns = self.0.lock();
        if turns.running || at != turns.front {
            turns.ahead.insert(at, post);
            return;
        }
        turns.running = true;
        turns.front += 1;
        drop(turns);
        let mut next = post;
        loop {
            if let Some(post) = next.take() {
                post();
            }
            let mut turns = self.0.lock();
            let front = turns.front;
            match turns.ahead.remove(&front) {
                Some(post) => {
                    turns.front += 1;
                    next = post;
                }
                None => {
                    turns.running = false;
                    return;
                }
            }
        }
    }

    /// How many turns finished past the front wait for it (for the tests).
    #[cfg(test)]
    pub(crate) fn waiting(&self) -> usize {
        self.0.lock().ahead.len()
    }
}

impl CompletionTurn {
    /// Finishes the turn with `post`, which runs once every turn before it has finished.
    pub(crate) fn finish(mut self, post: Option<Post>) {
        self.finished = true;
        self.line.finish(self.at, post);
    }
}

impl Drop for CompletionTurn {
    fn drop(&mut self) {
        if !self.finished {
            self.line.finish(self.at, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noting(log: &Arc<Mutex<Vec<u32>>>, n: u32) -> Option<Post> {
        let log = Arc::clone(log);
        Some(Box::new(move || log.lock().push(n)))
    }

    #[test]
    fn posts_run_in_turn_order_whatever_order_their_turns_finish_in() {
        let line = Arc::new(Completions::default());
        let log = Arc::new(Mutex::new(Vec::new()));
        let (a, b, c) = (line.turn(), line.turn(), line.turn());
        c.finish(noting(&log, 3));
        b.finish(noting(&log, 2));
        assert!(log.lock().is_empty(), "both wait for the first");
        assert_eq!(line.waiting(), 2);
        a.finish(noting(&log, 1));
        assert_eq!(*log.lock(), [1, 2, 3]);
        assert_eq!(line.waiting(), 0);
        line.turn().finish(noting(&log, 4));
        assert_eq!(*log.lock(), [1, 2, 3, 4], "a turn at the front posts at once");
    }

    #[test]
    fn a_turn_dropped_unfinished_lets_the_turns_after_it_post() {
        let line = Arc::new(Completions::default());
        let log = Arc::new(Mutex::new(Vec::new()));
        let (a, b) = (line.turn(), line.turn());
        b.finish(noting(&log, 2));
        drop(a);
        assert_eq!(*log.lock(), [2]);
    }

    #[test]
    fn a_post_that_finishes_a_later_turn_runs_it_after_itself() {
        let line = Arc::new(Completions::default());
        let log = Arc::new(Mutex::new(Vec::new()));
        let (a, b) = (line.turn(), line.turn());
        let inner = noting(&log, 2);
        let l = Arc::clone(&log);
        a.finish(Some(Box::new(move || {
            b.finish(inner);
            l.lock().push(1);
        })));
        assert_eq!(*log.lock(), [1, 2]);
    }
}
