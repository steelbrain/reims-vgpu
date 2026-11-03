//! The order the host commits a process's command buffers in across its queues
//! (`RGFX_FEATURE_COMMIT_ORDER`; docs/graphics-kext/design.md, "Commit order").
//!
//! Metal orders a tracked resource across a process's queues by the order the command buffers
//! were committed in (docs/graphics-kext/dgpu-oracle.md, Q6), and the platform orders it by the
//! order the host commits to it. A plain commit returns once the buffer is on its queue's ring, and
//! each queue's ring is taken on a thread of its own, so the bundle names the order: a buffer that
//! holds [`RGFX_OP_COMMIT_ORDER`] names the latest buffer each other queue of its process put on
//! its ring before it, and is committed only once those queues have committed through them.
//!
//! - **The marks.** Each queue's [`CommitMark`] (in its [`Marks`]) holds the highest seqno of its
//!   ring whose buffer was begun (its place in the commit order's tickets taken), and the highest
//!   whose buffer was committed, stood in for or failed. A buffer's handle raises the second as
//!   the message ends ([`super::GraphicsDecoder::end_message`], or the handle's drop), which in a
//!   queue's lane is before its turn passes, so the queue's buffers raise it in its ring's order.
//! - **Two waits** ([`wait`]). Before a buffer that names other queues takes its place among the
//!   tickets (its mark, at its begin on its queue's service thread), it waits until the buffers
//!   it names are begun ([`Until::Begun`]); an answer on a doorbell's thread, which waits for
//!   nothing, is left to the take where one is not. Then, after its turn in its queue and the
//!   tickets, it waits until they are committed ([`Until::Committed`]). The first keeps the
//!   tickets from crossing the order: a buffer it names never waits for a ticket issued after its
//!   own begin, so never for the buffer that names it, or for one of that buffer's queue taken
//!   after it. Each wait of a commit is then for a buffer begun before it, and the waits end.
//! - **No lock held.** A wait sleeps on the named queue's mark, which wakes it as the mark rises,
//!   in slices of [`WAIT_SLICE`], and ends where the named queue ends, its own queue ends (its
//!   destroy, or its context's close), or a reset of the device is under way ([`Cancel`]). So a
//!   process that names a cycle (each buffer named was on its ring before the one that names it,
//!   so the bundle never does) holds up only its own queues' threads, until it closes.
//! - **What names nothing** ([`super::surface_order::SurfaceOrder::commit_after`]): the buffer's
//!   own queue, whose order is its ring's; a queue of another context, which the guest's process
//!   cannot name; a queue the host does not hold, or holds ended; a seqno past every one the
//!   queue's ring has published; and one its mark has reached.
//! - **Cost.** A buffer that holds no op pays two atomic raises of its queue's marks, and a load of
//!   their waiters each. One that holds the op takes the order's lock once at its begin, and waits
//!   only where the queues' threads took the buffers in another order than the guest posted them.
//!
//! [`RGFX_OP_COMMIT_ORDER`]: reims_graphics::ops::RGFX_OP_COMMIT_ORDER

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use reims_vgpu_core::sync::{Condvar, Mutex};

use super::surface_order::Marks;

/// The longest a wait sleeps at once before it looks at its own queue and the device's reset
/// again; the named queue's mark wakes it at once as it rises.
pub(crate) const WAIT_SLICE: Duration = Duration::from_millis(10);

/// One queue's place in its process's commit order: how far its buffers have been begun and
/// committed.
#[derive(Default)]
pub(crate) struct CommitMark {
    /// The highest seqno of the queue's ring whose buffer took its place among the commit order's
    /// tickets.
    begun: AtomicU64,
    /// The highest seqno of the queue's ring whose buffer was committed, stood in for or failed.
    through: AtomicU64,
    /// The queue ended (its destroy, or its context's close): no wait for it waits more.
    ended: AtomicBool,
    /// How many threads wait on the mark; one that raises it wakes them only where any do.
    waiters: AtomicUsize,
    /// Held around a waiter's look at the mark and around the wake after it moves, so no wake is
    /// missed; never held while a waiter sleeps.
    lock: Mutex<()>,
    moved: Condvar,
}

/// What a [`wait`] waits for of each queue it names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Until {
    /// The buffer named took its place among the commit order's tickets.
    Begun,
    /// The buffer named was committed, stood in for or failed.
    Committed,
}

impl CommitMark {
    /// The queue's buffer at `seqno` took its place among the commit order's tickets, after every
    /// one before it on its ring.
    pub(crate) fn begin(&self, seqno: u64) {
        if self.begun.fetch_max(seqno, Ordering::SeqCst) < seqno {
            self.wake();
        }
    }

    /// The queue's buffer at `seqno` was committed, stood in for or failed, after every one before
    /// it on its ring.
    pub(crate) fn raise(&self, seqno: u64) {
        if self.through.fetch_max(seqno, Ordering::SeqCst) < seqno {
            self.wake();
        }
    }

    /// The queue ended: its waiters go on.
    pub(crate) fn end(&self) {
        self.ended.store(true, Ordering::SeqCst);
        self.wake();
    }

    /// Whether a wait for the queue's buffer at `seqno` is over: it got as far as `until`, or the
    /// queue ended. A buffer committed was begun.
    pub(crate) fn reached(&self, seqno: u64, until: Until) -> bool {
        (until == Until::Begun && self.begun.load(Ordering::SeqCst) >= seqno)
            || self.through.load(Ordering::SeqCst) >= seqno
            || self.ended.load(Ordering::SeqCst)
    }

    /// How far the queue's buffers have been begun and committed (for the tests).
    #[cfg(test)]
    pub(crate) fn state(&self) -> (u64, u64) {
        (
            self.begun.load(Ordering::SeqCst),
            self.through.load(Ordering::SeqCst),
        )
    }

    /// Wakes the mark's waiters, where any wait. A waiter counts itself before it looks at the
    /// mark under the lock, and the lock is taken here after the move: it either sees the move or
    /// sleeps before this wake.
    fn wake(&self) {
        if self.waiters.load(Ordering::SeqCst) == 0 {
            return;
        }
        drop(self.lock.lock());
        self.moved.notify_all();
    }
}

/// The device's resets, which end every wait so none holds a reset up, as the reset waits for the
/// work begun before it ([`super::Decoders::resetting`]): even while none is under way, odd while
/// one is.
#[derive(Default)]
pub(crate) struct Cancel(AtomicU64);

impl Cancel {
    /// The count's value now.
    fn now(&self) -> u64 {
        self.0.load(Ordering::SeqCst)
    }

    /// A reset is under way: every wait ends, and none waits until it is done.
    pub(crate) fn begin(&self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }

    /// The reset is done: waits wait again.
    pub(crate) fn end(&self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

/// What [`wait`] did (for the tests): whether it slept, and whether it ended without every named
/// queue's mark reached (its own queue ended, or a reset came).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Waited {
    pub(crate) slept: bool,
    pub(crate) cut_short: bool,
}

/// Waits, with no lock of the decoder's held, until each queue of `after` has got as far as
/// `until` through its seqno, where the buffer's own queue (`own`, its marks) has not ended and no
/// reset of `cancel`'s is or was under way since the wait began.
pub(crate) fn wait(
    after: &[(Arc<Marks>, u64)],
    own: Option<&Marks>,
    cancel: &Cancel,
    until: Until,
) -> Waited {
    let mut waited = Waited::default();
    let begun = cancel.now();
    let over = || {
        own.is_some_and(|own| own.commit.ended.load(Ordering::SeqCst))
            || !begun.is_multiple_of(2)
            || cancel.now() != begun
    };
    for (marks, seqno) in after {
        let mark = &marks.commit;
        if mark.reached(*seqno, until) {
            continue;
        }
        mark.waiters.fetch_add(1, Ordering::SeqCst);
        let mut guard = mark.lock.lock();
        loop {
            if mark.reached(*seqno, until) {
                break;
            }
            if over() {
                waited.cut_short = true;
                break;
            }
            waited.slept = true;
            let _ = mark.moved.wait_for(&mut guard, WAIT_SLICE);
        }
        drop(guard);
        mark.waiters.fetch_sub(1, Ordering::SeqCst);
        if waited.cut_short {
            break;
        }
    }
    waited
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // test scaffolding: real-time waits for the threads
mod tests {
    use std::sync::Arc;
    use std::sync::mpsc;
    use std::thread::JoinHandle;
    use std::time::{Duration, Instant};

    use reims_graphics::ops::{
        RGFX_COMMAND_BUFFER_OPS_OFFSET, RGFX_OP_COMMIT_ORDER, RGFX_OP_SUBMITTED, RgfxCommandBuffer,
        RgfxOpHeader, RgfxSurfaceOrderWait,
    };
    use reims_graphics::wire::{
        RGFX_CMD_COMMAND_BUFFER, RGFX_CMD_OBJECT_DESTROY, RGFX_CMD_QUEUE_CREATE,
        RGFX_CONTEXT_OBJ_ID_SHIFT, RgfxMsgHeader, RgfxObjectDestroy, RgfxQueueCreate,
    };
    use reims_vgpu_core::sync::Mutex;
    use reims_vgpu_core::testkit::FakePlatform;
    use zerocopy::IntoBytes;

    use super::super::{DecodeError, GraphicsDecoder};

    /// Queues 1 and 2 of process 1, and queue 1 of process 2.
    const A: u64 = 1 << RGFX_CONTEXT_OBJ_ID_SHIFT | 1;
    const A2: u64 = 1 << RGFX_CONTEXT_OBJ_ID_SHIFT | 2;
    const B: u64 = 2 << RGFX_CONTEXT_OBJ_ID_SHIFT | 1;

    /// How long a test waits for a thread it expects to go on.
    const PATIENCE: Duration = Duration::from_secs(10);
    /// How long a test watches a thread it expects to stay held.
    const WATCH: Duration = Duration::from_millis(100);

    struct Rig {
        p: Arc<FakePlatform>,
        dec: GraphicsDecoder<FakePlatform>,
        /// The buffers answered `CB_SUBMITTED`, in order.
        submitted: Arc<Mutex<Vec<u64>>>,
    }

    /// A decoder holding queues [`A`] and [`A2`] of process 1 and [`B`] of process 2.
    fn rig() -> Rig {
        let p = Arc::new(FakePlatform::new());
        let device = p.device();
        let mut dec = GraphicsDecoder::<FakePlatform>::new(
            Arc::clone(&p),
            Arc::new(reims_graphics::region::Region::new(0x10000).unwrap()),
        );
        let submitted = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&submitted);
        dec.set_submitted_sink(Arc::new(move |m| sink.lock().push(m.cb_obj_id)));
        for queue in [A, A2, B] {
            let create = RgfxQueueCreate {
                header: RgfxMsgHeader {
                    r#type: RGFX_CMD_QUEUE_CREATE,
                    size: size_of::<RgfxQueueCreate>() as u32,
                    seqno: 0,
                },
                obj_id: queue,
                ..RgfxQueueCreate::default()
            };
            dec.set_context(Some(context(queue)));
            assert_eq!(dec.decode(&device, create.as_bytes()).unwrap(), Ok(()));
        }
        Rig { p, dec, submitted }
    }

    fn context(queue: u64) -> u32 {
        (queue >> RGFX_CONTEXT_OBJ_ID_SHIFT) as u32
    }

    fn cb(queue: u64, seqno: u64) -> u64 {
        queue + 0x100 + seqno
    }

    /// The ops of a buffer that names `names` in its `COMMIT_ORDER` (none where empty), and asks
    /// for its `CB_SUBMITTED` where `asks`, as the bundle sends them.
    fn ops(names: &[(u64, u64)], asks: bool) -> Vec<u8> {
        let mut ops = Vec::new();
        if !names.is_empty() {
            let header = RgfxOpHeader {
                tag: RGFX_OP_COMMIT_ORDER,
                flags: 0,
                len: (8 + 16 * names.len()) as u32,
            };
            ops.extend_from_slice(header.as_bytes());
            for &(queue_obj_id, seqno) in names {
                let w = RgfxSurfaceOrderWait {
                    queue_obj_id,
                    seqno,
                };
                ops.extend_from_slice(w.as_bytes());
            }
        }
        if asks {
            let header = RgfxOpHeader {
                tag: RGFX_OP_SUBMITTED,
                flags: 0,
                len: 8,
            };
            ops.extend_from_slice(header.as_bytes());
        }
        ops
    }

    /// The `COMMAND_BUFFER` at `seqno` on `queue`, of `ops`.
    fn message(queue: u64, seqno: u64, ops: &[u8]) -> Vec<u8> {
        let m = RgfxCommandBuffer {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_COMMAND_BUFFER,
                size: (RGFX_COMMAND_BUFFER_OPS_OFFSET as usize + ops.len()) as u32,
                seqno,
            },
            queue_obj_id: queue,
            cb_obj_id: cb(queue, seqno),
            op_bytes: ops.len() as u64,
        };
        let mut message = m.as_bytes().to_vec();
        message.extend_from_slice(ops);
        message
    }

    impl Rig {
        /// A decoder for a message of `queue`'s, as the host makes one for each.
        fn sender(&self, queue: u64) -> GraphicsDecoder<FakePlatform> {
            (self.dec.decoders()).sender(Some(context(queue)), self.dec.sinks.clone())
        }

        /// Carries out queue `queue`'s buffer at `seqno` of `ops` on a thread named `name`, as a
        /// queue's service thread and its lane do: begun, walked and committed, and let go of.
        fn carry_out(
            &self,
            name: &str,
            queue: u64,
            seqno: u64,
            ops: &[u8],
        ) -> JoinHandle<Result<(), DecodeError>> {
            let mut dec = self.sender(queue);
            let (p, message) = (Arc::clone(&self.p), message(queue, seqno, ops));
            (std::thread::Builder::new().name(name.into()))
                .spawn(move || {
                    let device = p.device();
                    let begun = dec.begin_taken(&device, &message, Some(queue)).unwrap()?;
                    dec.finish_command_buffer(&device, &message, begun).unwrap()
                })
                .unwrap()
        }

        /// The threads that committed buffers, in commit order.
        fn commits(&self) -> Vec<String> {
            (self.p.call_log().calls_to("commit").into_iter())
                .map(|c| c.thread.unwrap_or_default())
                .collect()
        }

        /// How far queue `queue`'s buffers were begun and committed.
        fn state(&self, queue: u64) -> (u64, u64) {
            self.dec
                .shared
                .surface_order
                .lock()
                .marks(queue)
                .commit
                .state()
        }

        /// Destroys queue `queue`, as ring 0 does.
        fn destroy(&mut self, queue: u64) {
            let destroy = RgfxObjectDestroy {
                header: RgfxMsgHeader {
                    r#type: RGFX_CMD_OBJECT_DESTROY,
                    size: size_of::<RgfxObjectDestroy>() as u32,
                    seqno: 0,
                },
                obj_id: queue,
            };
            self.dec.set_context(Some(context(queue)));
            let device = self.p.device();
            assert_eq!(
                self.dec.decode(&device, destroy.as_bytes()).unwrap(),
                Ok(())
            );
        }
    }

    /// `thread`'s outcome, once it ends, which it must within [`PATIENCE`].
    fn joined<T>(thread: JoinHandle<T>, what: &str) -> T {
        let start = Instant::now();
        while !thread.is_finished() {
            assert!(start.elapsed() < PATIENCE, "{what} never went on");
            std::thread::sleep(Duration::from_millis(1));
        }
        thread.join().unwrap()
    }

    /// Waits until `done`, which must come within [`PATIENCE`].
    fn until(what: &str, done: impl Fn() -> bool) {
        let start = Instant::now();
        while !done() {
            assert!(start.elapsed() < PATIENCE, "{what} never came");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// A buffer that names a buffer of another queue of its process takes its place among the
    /// tickets only once that buffer is begun, and is committed only once it is: the host took
    /// it first, and commits it after.
    #[test]
    fn a_buffer_is_begun_and_committed_after_the_buffer_of_another_queue_it_names() {
        let r = rig();
        let x = r.carry_out("x", A2, 3, &ops(&[(A, 5)], false));
        std::thread::sleep(WATCH);
        assert_eq!(r.state(A2), (0, 0), "begun before the buffer it names");

        let device = r.p.device();
        let mut y = r.sender(A);
        let named = message(A, 5, &ops(&[], false));
        let begun = y.begin_taken(&device, &named, Some(A)).unwrap().unwrap();
        assert_eq!(r.state(A), (5, 0));
        until("its begin", || r.state(A2).0 == 3);
        std::thread::sleep(WATCH);
        assert!(
            r.commits().is_empty(),
            "committed before the buffer it names"
        );
        assert_eq!(
            y.finish_command_buffer(&device, &named, begun).unwrap(),
            Ok(())
        );
        drop(y);
        assert_eq!(r.state(A), (5, 5));
        assert_eq!(joined(x, "the buffer"), Ok(()));
        let commits = r.commits();
        assert_eq!(commits.len(), 2);
        assert_eq!(commits[1], "x", "{commits:?}");
        assert_eq!(r.state(A2), (3, 3));
    }

    /// A buffer named by a buffer of another queue does not wait for a ticket taken after its
    /// own begin: the queue that names it takes its later buffers, one asking for its
    /// `CB_SUBMITTED` among them, only once the buffer named is begun. Else the buffer named
    /// would wait for that ticket, the ticket's buffer for its turn behind the buffer that names
    /// it, and that one for the buffer named.
    #[test]
    fn a_buffer_named_never_waits_for_a_ticket_of_the_queue_that_names_it() {
        let r = rig();
        let (to_lane, lane) = mpsc::channel::<(GraphicsDecoder<FakePlatform>, Vec<u8>, _)>();
        let namer = r.sender(A2);
        let asker = r.sender(A2);
        let p = Arc::clone(&r.p);
        let service = (std::thread::Builder::new().name("x-service".into()))
            .spawn(move || {
                let device = p.device();
                for (mut dec, message) in [
                    (namer, message(A2, 3, &ops(&[(A, 5)], false))),
                    (asker, message(A2, 4, &ops(&[], true))),
                ] {
                    let begun = dec.begin_taken(&device, &message, Some(A2)).unwrap();
                    to_lane.send((dec, message, begun)).unwrap();
                }
            })
            .unwrap();
        let p = Arc::clone(&r.p);
        let lane = (std::thread::Builder::new().name("x-lane".into()))
            .spawn(move || {
                let device = p.device();
                let mut outcomes = Vec::new();
                for (dec, message, begun) in lane {
                    let begun = begun.unwrap();
                    outcomes.push(dec.finish_command_buffer(&device, &message, begun).unwrap());
                }
                outcomes
            })
            .unwrap();
        std::thread::sleep(WATCH);
        assert!(
            !r.submitted.lock().contains(&cb(A2, 4)),
            "its later buffer taken before the buffer named was begun"
        );

        let y = r.carry_out("y", A, 5, &ops(&[], false));
        assert_eq!(joined(y, "the buffer named"), Ok(()));
        joined(service, "the queue's service thread");
        assert_eq!(joined(lane, "the queue's lane"), [Ok(()), Ok(())]);
        assert_eq!(r.commits(), ["y", "x-lane", "x-lane"]);
        assert_eq!(*r.submitted.lock(), [cb(A2, 4)]);
    }

    /// A process that names a cycle (which the bundle never does) holds up only its own queues:
    /// another process's buffers are carried out meanwhile, and its context's close ends the
    /// waits, so its buffers' work goes on as any in flight at a close does.
    #[test]
    fn a_cycle_holds_only_its_own_queues_until_its_context_closes() {
        let mut r = rig();
        let (a, a2) = {
            let mut order = r.dec.shared.surface_order.lock();
            (order.marks(A), order.marks(A2))
        };
        let x = r.carry_out("x", A2, 3, &ops(&[(A, 5)], false));
        let y = r.carry_out("y", A, 5, &ops(&[(A2, 3)], false));
        std::thread::sleep(WATCH);
        assert!(!x.is_finished() && !y.is_finished());

        let device = r.p.device();
        r.dec.set_context(Some(2));
        for seqno in 1..=3 {
            let other = message(B, seqno, &ops(&[], false));
            assert_eq!(r.dec.decode(&device, &other).unwrap(), Ok(()));
        }
        assert_eq!(r.commits().len(), 3, "another process's buffers held");
        let create = RgfxQueueCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_QUEUE_CREATE,
                size: size_of::<RgfxQueueCreate>() as u32,
                seqno: 0,
            },
            obj_id: B + 1,
            ..RgfxQueueCreate::default()
        };
        assert_eq!(r.dec.decode(&device, create.as_bytes()).unwrap(), Ok(()));
        assert!(!x.is_finished() && !y.is_finished());

        r.dec.drop_context(1);
        let _ = joined(x, "a buffer of the cycle");
        let _ = joined(y, "a buffer of the cycle");
        // Each ended its message as any does: begun, and committed or failed.
        assert_eq!(a.commit.state(), (5, 5));
        assert_eq!(a2.commit.state(), (3, 3));
    }

    /// A buffer named whose begin fails (its op stream's length wrong, its fixed part cut short,
    /// or its command buffer refused) holds up the buffer that names it no more than one carried
    /// out: that one is begun once the failed one got as far as it will, and committed once its
    /// failure is posted.
    #[test]
    fn a_buffer_named_whose_begin_fails_lets_the_buffer_that_names_it_go_on() {
        use reims_vgpu_core::testkit::Step;
        use zerocopy::FromBytes;

        for how in ["op_bytes", "short", "refused"] {
            let r = rig();
            let x = r.carry_out("x", A2, 3, &ops(&[(A, 5)], false));
            std::thread::sleep(WATCH);
            assert_eq!(r.state(A2), (0, 0), "begun before the buffer it names");

            let device = r.p.device();
            let mut y = r.sender(A);
            let mut named = message(A, 5, &ops(&[], false));
            match how {
                "op_bytes" => {
                    let (mut m, _) = RgfxCommandBuffer::read_from_prefix(&named).unwrap();
                    m.op_bytes = 8;
                    named[..size_of::<RgfxCommandBuffer>()].copy_from_slice(m.as_bytes());
                }
                "short" => named.truncate(size_of::<RgfxMsgHeader>() + 4),
                _ => r.p.script().push("commandBuffer", Step::Nil),
            }
            let failed = y.begin_taken(&device, &named, Some(A)).unwrap();
            let expected = if how == "refused" {
                DecodeError::Refused
            } else {
                DecodeError::Malformed
            };
            assert_eq!(failed.err(), Some(expected), "{how}");
            until("its begin", || r.state(A2).0 == 3);
            std::thread::sleep(WATCH);
            assert!(r.commits().is_empty(), "committed before the failure");
            drop(y);
            assert_eq!(joined(x, "the buffer"), Ok(()));
            assert_eq!(r.commits(), ["x"]);
        }
    }

    /// A wait for another queue's commit ends where that queue ends, or the buffer's own queue
    /// ends, and the buffer is committed then.
    #[test]
    fn a_wait_for_commits_ends_where_either_queue_ends() {
        for gone in [A, A2] {
            let mut r = rig();
            let device = r.p.device();
            let mut y = r.sender(A);
            let named = message(A, 5, &ops(&[], false));
            let begun = y.begin_taken(&device, &named, Some(A)).unwrap().unwrap();
            let x = r.carry_out("x", A2, 3, &ops(&[(A, 5)], false));
            std::thread::sleep(WATCH);
            assert!(
                r.commits().is_empty(),
                "committed before the buffer it names"
            );
            r.destroy(gone);
            assert_eq!(joined(x, "the wait"), Ok(()));
            assert_eq!(r.commits(), ["x"]);
            drop(begun);
            drop(y);
        }
    }

    /// A reset of the device under way ends every wait, and none waits until it is done; then
    /// they wait again.
    #[test]
    fn a_wait_for_commits_ends_while_the_device_resets() {
        let r = rig();
        let device = r.p.device();
        let mut y = r.sender(A);
        let named = message(A, 5, &ops(&[], false));
        let begun = y.begin_taken(&device, &named, Some(A)).unwrap().unwrap();
        let x = r.carry_out("x", A2, 3, &ops(&[(A, 5)], false));
        std::thread::sleep(WATCH);
        assert!(
            r.commits().is_empty(),
            "committed before the buffer it names"
        );
        r.dec.decoders().resetting();
        assert_eq!(joined(x, "the wait"), Ok(()));
        let during = r.carry_out("during", A2, 4, &ops(&[(A, 5)], false));
        assert_eq!(joined(during, "a wait begun during the reset"), Ok(()));
        r.dec.decoders().reset_done();

        let after = r.carry_out("after", A2, 6, &ops(&[(A, 5)], false));
        std::thread::sleep(WATCH);
        assert!(!after.is_finished(), "a wait after the reset did not wait");
        assert_eq!(
            y.finish_command_buffer(&device, &named, begun).unwrap(),
            Ok(())
        );
        drop(y);
        assert_eq!(joined(after, "the wait after the reset"), Ok(()));
        let commits = r.commits();
        assert_eq!(commits[..2], ["x", "during"]);
        assert_eq!(commits[3], "after");
    }

    /// What a buffer's `COMMIT_ORDER` names that it is committed after: each queue once, at the
    /// highest seqno named, and nothing of its own queue, queue or seqno 0, another process's
    /// queue, a queue the host does not hold or holds ended, a seqno past what the queue's ring
    /// published, or one committed.
    #[test]
    fn commit_order_names_only_its_process_s_other_live_queues() {
        let mut r = rig();
        let (a, a2) = {
            let mut order = r.dec.shared.surface_order.lock();
            (order.marks(A), order.marks(A2))
        };
        r.dec.surface_order_queue_made(A2, Arc::new(|| Some(4)));
        let named = |r: &Rig, queue: u64, names: &[(u64, u64)]| {
            let order = r.dec.shared.surface_order.lock();
            let waits = names
                .iter()
                .map(|&(queue_obj_id, seqno)| RgfxSurfaceOrderWait {
                    queue_obj_id,
                    seqno,
                });
            (order.commit_after(queue, waits).into_iter())
                .map(|(marks, seqno)| {
                    let queue = if Arc::ptr_eq(&marks, &a) {
                        A
                    } else if Arc::ptr_eq(&marks, &a2) {
                        A2
                    } else {
                        0
                    };
                    (queue, seqno)
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(named(&r, A, &[(A, 5)]), [], "its own queue");
        assert_eq!(named(&r, A, &[(0, 5), (A2, 0)]), []);
        assert_eq!(named(&r, A, &[(B, 5)]), [], "another process's queue");
        assert_eq!(
            named(&r, A, &[(A | 9, 5)]),
            [],
            "a queue the host does not hold"
        );
        assert_eq!(named(&r, A, &[(A2, 5)]), [], "past what its ring published");
        assert_eq!(named(&r, A, &[(A2, 2), (A2, 4), (A2, 3)]), [(A2, 4)]);
        assert_eq!(named(&r, A2, &[(A, 7), (A, 9)]), [(A, 9)]);
        a2.commit.raise(4);
        assert_eq!(named(&r, A, &[(A2, 4)]), [], "committed");

        let begun = |r: &Rig, names: &[(u64, u64)]| {
            let order = r.dec.shared.surface_order.lock();
            let waits = names
                .iter()
                .map(|&(queue_obj_id, seqno)| RgfxSurfaceOrderWait {
                    queue_obj_id,
                    seqno,
                });
            order.commit_begun(A2, waits)
        };
        assert!(!begun(&r, &[(A, 9)]));
        a.commit.begin(9);
        assert!(begun(&r, &[(A, 9)]));
        assert!(!begun(&r, &[(A, 9), (A, 10)]));
        r.destroy(A);
        assert_eq!(named(&r, A2, &[(A, 10)]), [], "a queue ended");
        assert!(begun(&r, &[(A, 10)]));
    }

    /// An answer on a doorbell's thread takes a buffer's place among the tickets only where the
    /// buffers its `COMMIT_ORDER` names are begun, and notes it begun.
    #[test]
    fn an_answer_at_the_doorbell_waits_for_no_buffer_named() {
        let r = rig();
        let asking = message(A2, 3, &ops(&[(A, 5)], true));
        let answer = || (r.dec.decoders()).answer_at_doorbell(1, &asking, |_| true);
        assert!(!answer(), "answered before the buffer it names was begun");
        assert_eq!(r.state(A2), (0, 0));
        r.dec.shared.surface_order.lock().marks(A).commit.begin(5);
        assert!(answer());
        assert_eq!(r.state(A2), (3, 0));
    }
}
