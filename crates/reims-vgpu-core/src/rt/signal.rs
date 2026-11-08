//! `CoalescingSignal`: a dispatch source of type DATA_ADD or DATA_OR.
//!
//! The semantics kept (7.1, 8.6): `merge(v)` folds `v` into a pending word (add or or); a merge
//! of 0 does nothing; when the word goes nonzero and no handler invocation is scheduled or
//! running, one job is submitted to the target; the job takes the word (resetting it to 0) and
//! runs the handler with it. Merges while the handler runs schedule one more invocation after
//! it, never a concurrent one (a source's handler is serial, whatever its target). How many
//! invocations a burst of merges gives is timing, never asserted (decision 8).
//!
//! Creation takes no slot on the target: the first handler call is queued at the merge, behind
//! the blocks queued before it, and a cancel before any merge queues its callout at the cancel,
//! as libdispatch does on a queue not suspended at the source's creation (measured on
//! libdispatch: a source's first fire and its cancel on a busy target). Core's queues are never suspended, and a display's newFrame source
//! is this model on every queue the library gives a display.
//!
//! Cancelling, explicitly (`dispatch_source_cancel`) or by releasing the last reference to a source
//! nobody cancelled (libdispatch's `_dispatch_source_xref_dispose` marks it `DQF_RELEASED`, which
//! counts as cancelled), drops the pending data: the handler never starts again, and an invocation
//! already running finishes. Then the source's cancel callout frees the handler on the target: in
//! the slot of the invocation or activation already scheduled (right after a running handler
//! returns), or else in a job queued at the moment of the cancel. Measured against libdispatch on
//! macOS 26 (the IOSurface host device drops its doorbell signal without cancelling it): data
//! merged before the release, even while the handler ran and its next invocation waited behind
//! another block, is never delivered, and the handler block is freed on the target queue as
//! described.

use std::sync::Arc;

use parking_lot::Mutex;

use crate::rt::queue::Executor;

/// How merged data combines (`DISPATCH_SOURCE_TYPE_DATA_ADD` / `_OR`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Merge {
    /// DATA_ADD: pending += v (wrapping, as libdispatch's atomic add).
    Add,
    /// DATA_OR: pending |= v.
    Or,
}

#[derive(Debug, Default)]
struct State {
    pending: u64,
    scheduled: bool,
    cancelled: bool,
}

/// The source's event handler block.
type Handler = Arc<dyn Fn(u64) + Send + Sync>;

struct Inner {
    merge: Merge,
    target: Arc<dyn Executor>,
    /// The event handler; the cancel callout frees it (`None` from then on).
    handler: Mutex<Option<Handler>>,
    state: Mutex<State>,
}

/// The references `dispatch_retain` counts (libdispatch's xref count). Jobs on the target hold
/// the source itself (its internal references), so the release of the last of these, which
/// cancels the source, can come while an invocation is queued or running.
struct Retained {
    inner: Arc<Inner>,
}

impl Drop for Retained {
    /// `dispatch_release` of the last reference: `_dispatch_source_xref_dispose`. The source is
    /// cancelled (`DQF_RELEASED`) and its handler freed by the cancel callout on the target.
    fn drop(&mut self) {
        Inner::cancel(&self.inner);
    }
}

/// A resumed DATA_ADD/DATA_OR dispatch source with its event handler set. `Clone` is
/// `dispatch_retain`, and dropping the last clone is `dispatch_release` of the last reference,
/// which cancels the source.
#[derive(Clone)]
pub struct CoalescingSignal {
    retained: Arc<Retained>,
}

impl std::fmt::Debug for CoalescingSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoalescingSignal")
            .field("merge", &self.retained.inner.merge)
            .field("state", &*self.retained.inner.state.lock())
            .finish()
    }
}

impl CoalescingSignal {
    /// `dispatch_source_create(DISPATCH_SOURCE_TYPE_DATA_ADD, 0, 0, target)` + `dispatch_source_set_event_handler` + `dispatch_resume`.
    /// `handler` gets `dispatch_source_get_data`.
    pub fn new_add(
        target: Arc<dyn Executor>,
        handler: impl Fn(u64) + Send + Sync + 'static,
    ) -> Self {
        Self::new(Merge::Add, target, handler)
    }

    /// `dispatch_source_create(DISPATCH_SOURCE_TYPE_DATA_OR, 0, 0, target)` + handler + resume.
    pub fn new_or(
        target: Arc<dyn Executor>,
        handler: impl Fn(u64) + Send + Sync + 'static,
    ) -> Self {
        Self::new(Merge::Or, target, handler)
    }

    fn new(
        merge: Merge,
        target: Arc<dyn Executor>,
        handler: impl Fn(u64) + Send + Sync + 'static,
    ) -> Self {
        // `dispatch_resume` of the new source: on a queue that is not suspended its activation
        // takes no slot the handler waits for (see the module doc).
        let inner = Arc::new(Inner {
            merge,
            target,
            handler: Mutex::new(Some(Arc::new(handler))),
            state: Mutex::new(State::default()),
        });
        CoalescingSignal {
            retained: Arc::new(Retained { inner }),
        }
    }

    /// `dispatch_source_merge_data(source, v)`.
    pub fn merge(&self, v: u64) {
        let inner = &self.retained.inner;
        if v == 0 {
            return;
        }
        let submit = {
            let mut s = inner.state.lock();
            if s.cancelled {
                return;
            }
            s.pending = match inner.merge {
                Merge::Add => s.pending.wrapping_add(v),
                Merge::Or => s.pending | v,
            };
            if s.scheduled || s.pending == 0 {
                false
            } else {
                s.scheduled = true;
                true
            }
        };
        if submit {
            Inner::submit(inner);
        }
    }

    /// Cancels the source (`dispatch_source_cancel`).
    pub fn cancel(&self) {
        Inner::cancel(&self.retained.inner);
    }

    /// `dispatch_source_testcancel`.
    pub fn is_cancelled(&self) -> bool {
        self.retained.inner.state.lock().cancelled
    }
}

impl Inner {
    /// The source's own block on the target: a handler invocation or the cancel callout
    /// (`Executor::exec_source`: the host's dispatch queue runs it at its own QoS).
    fn submit(this: &Arc<Inner>) {
        let me = this.clone();
        this.target.exec_source(Box::new(move || Inner::fire(&me)));
    }

    /// `dispatch_source_cancel`, or the last release of a source not cancelled: the pending data
    /// is dropped, and the cancel callout is owed on the target. An invocation (or the
    /// activation) already scheduled runs it in its slot; otherwise a job queued now does.
    fn cancel(this: &Arc<Inner>) {
        let submit = {
            let mut s = this.state.lock();
            if s.cancelled {
                return;
            }
            s.cancelled = true;
            s.pending = 0;
            !std::mem::replace(&mut s.scheduled, true)
        };
        if submit {
            Inner::submit(this);
        }
    }

    /// The source's cancel callout: frees the handler (`_dispatch_source_cancel_callout`
    /// releases the event handler block), outside every lock, since what the handler captured
    /// may be released with it.
    fn cancel_callout(this: &Arc<Inner>) {
        let handler = this.handler.lock().take();
        this.state.lock().scheduled = false;
        drop(handler);
    }

    fn fire(this: &Arc<Inner>) {
        let data = {
            let mut s = this.state.lock();
            if s.cancelled {
                None
            } else {
                let data = std::mem::take(&mut s.pending);
                if data == 0 {
                    s.scheduled = false;
                    return;
                }
                Some(data)
            }
        };
        let Some(data) = data else {
            return Inner::cancel_callout(this);
        };
        // Only the one scheduled invocation reaches here, so the handler is still set.
        let handler = this.handler.lock().clone();
        if let Some(h) = &handler {
            h(data);
        }
        drop(handler);
        enum Next {
            Again,
            Callout,
            Idle,
        }
        let next = {
            let mut s = this.state.lock();
            if s.cancelled {
                Next::Callout
            } else if s.pending != 0 {
                Next::Again
            } else {
                s.scheduled = false;
                Next::Idle
            }
        };
        match next {
            Next::Again => Inner::submit(this),
            // A cancel (or the last release) while the handler ran: the callout follows it at
            // once, ahead of whatever was queued on the target meanwhile.
            Next::Callout => Inner::cancel_callout(this),
            Next::Idle => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rt::queue::{Job, SerialQueue};
    use crate::rt::thread::test_threads::StdThreads;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// A target that holds jobs until the test runs them: a busy queue (not suspended), as the
    /// library's own queues and a dispatch-less host queue are.
    #[derive(Default)]
    struct Manual(Mutex<Vec<Job>>);
    impl Executor for Manual {
        fn exec_async(&self, job: Job) {
            self.0.lock().push(job);
        }
    }
    impl Manual {
        fn run_all(&self) -> usize {
            let mut n = 0usize;
            loop {
                let jobs: Vec<Job> = std::mem::take(&mut *self.0.lock());
                if jobs.is_empty() {
                    return n;
                }
                for j in jobs {
                    j();
                    n = n.wrapping_add(1);
                }
            }
        }
        fn queued(&self) -> usize {
            self.0.lock().len()
        }
    }

    fn recorder() -> (Arc<Mutex<Vec<u64>>>, impl Fn(u64) + Send + Sync + 'static) {
        let log = Arc::new(Mutex::new(Vec::new()));
        let l = log.clone();
        (log, move |d| l.lock().push(d))
    }

    /// Several ADD merges before the handler give one call with the sum
    #[test]
    fn add_coalesces() {
        let t = Arc::new(Manual::default());
        let (log, h) = recorder();
        let s = CoalescingSignal::new_add(t.clone(), h);
        s.merge(1);
        s.merge(1);
        s.merge(3);
        assert_eq!(t.queued(), 1);
        t.run_all();
        assert_eq!(*log.lock(), vec![5]);
    }

    /// OR merges combine bits
    #[test]
    fn or_coalesces() {
        let t = Arc::new(Manual::default());
        let (log, h) = recorder();
        let s = CoalescingSignal::new_or(t.clone(), h);
        s.merge(1);
        s.merge(1);
        s.merge(4);
        t.run_all();
        assert_eq!(*log.lock(), vec![5]);
    }

    /// A merge of 0 does nothing
    #[test]
    fn zero_merge_is_noop() {
        let t = Arc::new(Manual::default());
        let (log, h) = recorder();
        let s = CoalescingSignal::new_add(t.clone(), h);
        // The creation takes no slot.
        assert_eq!(t.queued(), 0);
        s.merge(0);
        assert_eq!(t.queued(), 0);
        assert_eq!(t.run_all(), 0);
        assert!(log.lock().is_empty());
    }

    /// The creation takes no slot on a target that is not suspended (busy or idle): the first
    /// handler call is queued at the merge, behind the jobs queued before it, as every later one
    /// (libdispatch on a busy target: `P h1 after`, golden_coalescing.rs
    /// `first_fire_order_on_a_busy_target`)
    #[test]
    fn first_fire_is_queued_at_the_merge() {
        let t = Arc::new(Manual::default());
        let log = Arc::new(Mutex::new(Vec::<String>::new()));
        let l = log.clone();
        let s = CoalescingSignal::new_add(t.clone(), move |d| l.lock().push(format!("h{d}")));
        let l = log.clone();
        t.exec_async(Box::new(move || l.lock().push("P".into())));
        s.merge(1);
        assert_eq!(t.queued(), 2);
        let l = log.clone();
        t.exec_async(Box::new(move || l.lock().push("after".into())));
        t.run_all();
        s.merge(2);
        let l = log.clone();
        t.exec_async(Box::new(move || l.lock().push("last".into())));
        t.run_all();
        assert_eq!(*log.lock(), ["P", "h1", "after", "h2", "last"]);
    }

    /// An ADD that wraps to 0 is pending 0: the job runs and calls nothing
    #[test]
    fn add_wrapping_to_zero() {
        let t = Arc::new(Manual::default());
        let (log, h) = recorder();
        let s = CoalescingSignal::new_add(t.clone(), h);
        s.merge(u64::MAX);
        s.merge(1);
        t.run_all();
        assert!(log.lock().is_empty());
        s.merge(2);
        t.run_all();
        assert_eq!(*log.lock(), vec![2]);
    }

    /// A merge after the handler ran schedules a new call
    #[test]
    fn merge_after_fire_schedules_again() {
        let t = Arc::new(Manual::default());
        let (log, h) = recorder();
        let s = CoalescingSignal::new_add(t.clone(), h);
        s.merge(1);
        t.run_all();
        s.merge(2);
        t.run_all();
        assert_eq!(*log.lock(), vec![1, 2]);
    }

    /// Merges during the handler give exactly one more call, after it
    #[test]
    fn merge_during_handler_runs_after() {
        let t = Arc::new(Manual::default());
        let log = Arc::new(Mutex::new(Vec::new()));
        let slot: Arc<Mutex<Option<CoalescingSignal>>> = Arc::new(Mutex::new(None));
        let (l2, s2) = (log.clone(), slot.clone());
        let s = CoalescingSignal::new_add(t.clone(), move |d| {
            l2.lock().push(d);
            if d == 1
                && let Some(sig) = s2.lock().clone()
            {
                sig.merge(10);
                sig.merge(20);
            }
        });
        *slot.lock() = Some(s.clone());
        s.merge(1);
        assert_eq!(t.run_all(), 2);
        assert_eq!(*log.lock(), vec![1, 30]);
        *slot.lock() = None;
    }

    /// Cancel before the job runs drops the pending data; later merges do nothing
    #[test]
    fn cancel_stops_handler() {
        let t = Arc::new(Manual::default());
        let (log, h) = recorder();
        let s = CoalescingSignal::new_add(t.clone(), h);
        s.merge(1);
        s.cancel();
        assert!(s.is_cancelled());
        t.run_all();
        s.merge(1);
        t.run_all();
        assert!(log.lock().is_empty());
        assert!(format!("{s:?}").contains("cancelled: true"));
    }

    /// A handler that logs `h<data>` and, when it is freed, `~h`; and the log.
    fn probed(log: &Arc<Mutex<Vec<String>>>) -> impl Fn(u64) + Send + Sync + 'static {
        struct Probe(Arc<Mutex<Vec<String>>>);
        impl Drop for Probe {
            fn drop(&mut self) {
                self.0.lock().push("~h".into());
            }
        }
        let probe = Probe(log.clone());
        let l = log.clone();
        move |d| {
            let _ = &probe;
            l.lock().push(format!("h{d}"));
        }
    }

    fn mark(t: &Manual, log: &Arc<Mutex<Vec<String>>>, name: &'static str) {
        let l = log.clone();
        t.exec_async(Box::new(move || l.lock().push(name.into())));
    }

    /// `dispatch_release` of the last reference to a DATA_ADD/DATA_OR source nobody cancelled (the
    /// IOSurface host device's end drops its doorbell signal in a sync onto its hardware queue,
    /// with no cancel). libdispatch's `_dispatch_source_xref_dispose` marks the source released,
    /// which counts as cancelled. Measured on macOS 26 (a DATA_ADD source whose handler block
    /// captures an object with a logging destructor, released with the target suspended).
    mod drop_uncancelled {
        use super::*;

        /// Data merged before the release is never delivered; the cancel callout frees the handler
        /// in the slot of the invocation the merge queued (libdispatch: `~h A B`)
        #[test]
        fn pending_data_is_dropped() {
            let t = Arc::new(Manual::default());
            let log = Arc::new(Mutex::new(Vec::<String>::new()));
            let s = CoalescingSignal::new_add(t.clone(), probed(&log));
            assert_eq!(t.run_all(), 0);
            s.merge(7);
            mark(&t, &log, "A");
            drop(s);
            mark(&t, &log, "B");
            t.run_all();
            assert_eq!(*log.lock(), ["~h", "A", "B"]);
        }

        /// An idle source's release queues its cancel callout on the target then (libdispatch:
        /// `A ~h B`)
        #[test]
        fn idle_source_frees_its_handler_at_the_tail() {
            let t = Arc::new(Manual::default());
            let log = Arc::new(Mutex::new(Vec::<String>::new()));
            let s = CoalescingSignal::new_or(t.clone(), probed(&log));
            assert_eq!(t.run_all(), 0);
            mark(&t, &log, "A");
            drop(s);
            assert!(log.lock().is_empty());
            mark(&t, &log, "B");
            t.run_all();
            assert_eq!(*log.lock(), ["A", "~h", "B"]);
        }

        /// A release on a busy target before the target ran anything of the source's: with no merge
        /// the callout is queued at the release (libdispatch on a busy target: `A ~h B`,
        /// golden_coalescing.rs `cancel_on_a_busy_target`); after a merge it runs in the slot of
        /// the invocation the merge queued (`~h A B`)
        #[test]
        fn release_before_the_target_ran() {
            for pending in [false, true] {
                let t = Arc::new(Manual::default());
                let log = Arc::new(Mutex::new(Vec::<String>::new()));
                let s = CoalescingSignal::new_add(t.clone(), probed(&log));
                if pending {
                    s.merge(1);
                }
                mark(&t, &log, "A");
                drop(s);
                mark(&t, &log, "B");
                t.run_all();
                let want: &[&str] = if pending {
                    &["~h", "A", "B"]
                } else {
                    &["A", "~h", "B"]
                };
                assert_eq!(*log.lock(), want);
            }
        }

        /// Only the last reference cancels; a clone keeps the source delivering
        #[test]
        fn a_retained_clone_keeps_it_live() {
            let t = Arc::new(Manual::default());
            let log = Arc::new(Mutex::new(Vec::<String>::new()));
            let s = CoalescingSignal::new_add(t.clone(), probed(&log));
            let kept = s.clone();
            s.merge(2);
            drop(s);
            t.run_all();
            assert!(!kept.is_cancelled());
            kept.merge(3);
            t.run_all();
            assert_eq!(*log.lock(), ["h2", "h3"]);
            drop(kept);
            t.run_all();
            assert_eq!(*log.lock(), ["h2", "h3", "~h"]);
        }

        /// The IOSurface doorbell race: a merge while the handler runs queues the next invocation
        /// behind a block already waiting on the target (the teardown's sync), and that block
        /// releases the source. The handler is not called again, and the callout frees it in the
        /// queued invocation's slot (libdispatch: `handler call 1 / release / calls=1`)
        #[test]
        fn merge_during_handler_then_release_behind_it() {
            let t = Arc::new(Manual::default());
            let log = Arc::new(Mutex::new(Vec::<String>::new()));
            let slot: Arc<Mutex<Option<CoalescingSignal>>> = Arc::new(Mutex::new(None));
            let (t2, l2, s2) = (t.clone(), log.clone(), slot.clone());
            let inner = probed(&log);
            let s = CoalescingSignal::new_or(t.clone(), move |d| {
                inner(d);
                if d == 1 {
                    // The doorbell again (MMIO 0x1018), then the teardown's sync block queues.
                    if let Some(sig) = s2.lock().clone() {
                        sig.merge(1);
                    }
                    let (l3, s3) = (l2.clone(), s2.clone());
                    t2.exec_async(Box::new(move || {
                        l3.lock().push("release".into());
                        drop(s3.lock().take());
                    }));
                }
            });
            s.merge(1);
            *slot.lock() = Some(s);
            t.run_all();
            assert_eq!(*log.lock(), ["h1", "release", "~h"]);
        }

        /// The last release from inside the handler: the callout follows the handler at once, ahead
        /// of what was queued meanwhile (libdispatch: `handler call 1 / ~h A B`)
        #[test]
        fn release_while_the_handler_runs() {
            let t = Arc::new(Manual::default());
            let log = Arc::new(Mutex::new(Vec::<String>::new()));
            let slot: Arc<Mutex<Option<CoalescingSignal>>> = Arc::new(Mutex::new(None));
            let (t2, l2, s2) = (t.clone(), log.clone(), slot.clone());
            let inner = probed(&log);
            let s = CoalescingSignal::new_add(t.clone(), move |d| {
                inner(d);
                if let Some(sig) = s2.lock().clone() {
                    sig.merge(5);
                }
                mark(&t2, &l2, "A");
                drop(s2.lock().take());
                mark(&t2, &l2, "B");
            });
            s.merge(1);
            *slot.lock() = Some(s);
            t.run_all();
            assert_eq!(*log.lock(), ["h1", "~h", "A", "B"]);
        }
    }

    /// `dispatch_source_cancel`: measured on macOS 26 as `drop_uncancelled`: the cancel callout
    /// frees the handler on the target, and a later release adds nothing.
    mod cancel {
        use super::*;

        /// An idle source's callout is queued at the cancel (libdispatch: `A ~h B`); with data
        /// pending it runs in the queued invocation's slot (`~h A B`)
        #[test]
        fn frees_the_handler_on_the_target() {
            for pending in [false, true] {
                let t = Arc::new(Manual::default());
                let log = Arc::new(Mutex::new(Vec::<String>::new()));
                let s = CoalescingSignal::new_add(t.clone(), probed(&log));
                assert_eq!(t.run_all(), 0);
                if pending {
                    s.merge(1);
                }
                mark(&t, &log, "A");
                s.cancel();
                mark(&t, &log, "B");
                t.run_all();
                let want: &[&str] = if pending {
                    &["~h", "A", "B"]
                } else {
                    &["A", "~h", "B"]
                };
                assert_eq!(*log.lock(), want);
                drop(s);
                mark(&t, &log, "C");
                t.run_all();
                assert_eq!(log.lock().last().map(String::as_str), Some("C"));
                assert_eq!(log.lock().iter().filter(|l| *l == "~h").count(), 1);
            }
        }

        /// A cancel while the handler runs: it finishes, and the callout follows it at once, ahead
        /// of what was queued meanwhile (libdispatch: `handler call 1 / ~h A B`)
        #[test]
        fn during_the_handler() {
            let t = Arc::new(Manual::default());
            let log = Arc::new(Mutex::new(Vec::<String>::new()));
            let slot: Arc<Mutex<Option<CoalescingSignal>>> = Arc::new(Mutex::new(None));
            let (t2, l2, s2) = (t.clone(), log.clone(), slot.clone());
            let inner = probed(&log);
            let s = CoalescingSignal::new_add(t.clone(), move |d| {
                inner(d);
                let sig = s2.lock().clone();
                if let Some(sig) = sig {
                    sig.merge(5);
                    mark(&t2, &l2, "A");
                    sig.cancel();
                    mark(&t2, &l2, "B");
                }
            });
            *slot.lock() = Some(s.clone());
            s.merge(1);
            t.run_all();
            assert_eq!(*log.lock(), ["h1", "~h", "A", "B"]);
            *slot.lock() = None;
        }
    }

    /// On a real serial queue from many threads, every merged unit is delivered once
    #[test]
    fn serial_queue_target_conserves_sum() {
        let queue = SerialQueue::new(&StdThreads, "reims-vgpu.reporting");
        let total = Arc::new(AtomicU64::new(0));
        let t2 = total.clone();
        let s = CoalescingSignal::new_add(Arc::new(queue.clone()), move |d| {
            t2.fetch_add(d, Ordering::SeqCst);
        });
        std::thread::scope(|sc| {
            for _ in 0..4 {
                let s = s.clone();
                sc.spawn(move || {
                    for _ in 0..250 {
                        s.merge(1);
                    }
                });
            }
        });
        // Every submitted job ran once the queue drains twice (a fire may resubmit once).
        queue.sync(|| ());
        queue.sync(|| ());
        let mut n = 0;
        while total.load(Ordering::SeqCst) != 1000 && n < 1000 {
            queue.sync(|| ());
            n += 1;
        }
        assert_eq!(total.load(Ordering::SeqCst), 1000);
    }

    /// `dispatch_resume`, `dispatch_source_merge_data` and `dispatch_source_cancel`. libdispatch
    /// wakes a source on its target with no QoS of the thread that resumed, merged or cancelled:
    /// the handler runs at the target's QoS (DEF on the library's queues), not the merger's
    /// (measured against libdispatch).
    mod merge_qos {
        use super::*;

        /// A target that records how it was submitted to, and runs nothing until asked.
        #[derive(Default)]
        struct Recording {
            calls: Mutex<Vec<&'static str>>,
            jobs: Manual,
        }

        impl Executor for Recording {
            fn exec_async(&self, job: Job) {
                self.calls.lock().push("exec_async");
                self.jobs.exec_async(job)
            }
            fn exec_source(&self, job: Job) {
                self.calls.lock().push("exec_source");
                self.jobs.exec_async(job)
            }
        }

        /// Each handler invocation and the cancel callout are the source's own blocks
        /// (`exec_source`), never a `dispatch_async` of the merger's
        #[test]
        fn every_block_of_the_source_is_a_source_block() {
            let t = Arc::new(Recording::default());
            let (log, h) = recorder();
            let s = CoalescingSignal::new_or(t.clone(), h);
            assert_eq!(t.jobs.run_all(), 0);
            s.merge(1);
            t.jobs.run_all();
            s.merge(2);
            s.cancel();
            t.jobs.run_all();
            // An idle source's cancel queues its own callout.
            let idle = CoalescingSignal::new_or(t.clone(), |_| {});
            idle.cancel();
            t.jobs.run_all();
            assert_eq!(*log.lock(), vec![1]);
            assert_eq!(
                *t.calls.lock(),
                vec!["exec_source", "exec_source", "exec_source"]
            );
        }
    }
}
