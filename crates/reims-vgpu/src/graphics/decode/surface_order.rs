//! The order of command buffers of different queues over the IOSurfaces they share
//! (`RGFX_FEATURE_SURFACE_ORDER`; docs/graphics-kext/design.md, "Surface order").
//!
//! A command buffer that holds [`RGFX_OP_SURFACE_ORDER`] names, as the bundle committed it, the
//! buffers of other queues it runs after: each surface's writers of other queues, and its readers
//! too where the buffer writes it, as the surface's shared block named them when the commit read
//! it. The host makes the buffer wait for each on the GPU, on the named queue's surface timeline:
//! a shared event each buffer of the queue that holds the op signals to its own seqno as it ends.
//! A queue's buffers signal in their ring's order, each after the work before it on the queue, so
//! the timeline at a seqno means the queue's buffers up to that one have finished.
//!
//! - **No wait where none is needed.** A named buffer of the buffer's own queue is in its ring's
//!   order already. One the host has seen finish (its completed handler raised the queue's
//!   `done`) is a compare. Only the rest are GPU waits, which hold back none of the buffer's
//!   records, nor its queue: it gets its `CB_SUBMITTED` at its take and its `CB_SCHEDULED` at its
//!   commit, as a Mac's order of IOSurface accesses holds back none (`cmdbuf`'s `Line`). Two
//!   buffers that share no surface never wait for each other, since neither names the other.
//! - **A buffer that holds no op pays nothing** but two atomic operations on its own queue's
//!   [`Marks`], with no lock: it notes itself taken, and signals only where a buffer of another
//!   queue waits past what its queue signalled ([`Marks::quiet`]).
//! - **A buffer named before the host takes it.** Each queue's ring is taken on a thread of its
//!   own, so a buffer may be decoded before a buffer it names, committed earlier on another queue,
//!   is taken. Its wait is then for a value not yet signalled (Metal and the Vulkan platform wait
//!   for a value signalled later), and the named queue's buffer at that seqno signals it when the
//!   host takes it, whether or not it holds the op (`waited`). The waiter raises `waited` and then
//!   reads `taken`, and a buffer taken raises `taken` and then reads `waited`, each sequentially
//!   consistent, so at least one of the two sees the other.
//! - **Nothing waits for good.** A named buffer that does not run still signals: a stand-in
//!   signals at its seqno, and a buffer with none gets a command buffer of its own on its queue
//!   that only signals ([`SurfaceOrder::release`]). A queue's destroy and its context's close
//!   signal the highest value waited for of it ([`SurfaceOrder::ended`]). A queue the host does
//!   not hold, or a seqno past every one its ring has published, names no buffer and waits for
//!   nothing, so a slot a process left as it died, or one written wrong, holds nothing.
//! - **A signal never overtakes the queue's work.** Each value is signalled by a command buffer on
//!   the queue's own host queue, after the work committed there before it. A CPU set stands in only
//!   for a queue that makes no command buffer, and waits on the CPU first for the timeline to
//!   reach what the queue's committed buffers signal, so it releases no waiter before them either
//!   ([`Release::signal`]).
//!
//! - **Work the host starts off the command buffer waits too.** A video session's decode runs on
//!   a queue of its own as the host walks the buffer, so the buffer's GPU waits do not hold it:
//!   the host waits for the same buffers on the CPU first ([`OrderWait`]), so a decode's output
//!   lands after the reads of other queues committed before it, as any other write does.
//!
//! The order's state is the decoder's `surface_order` lock, a leaf: held while a buffer's waits
//! and signal are encoded, never across a platform call that may wait. A command buffer made only
//! to signal ([`Release`]) is made and committed after the lock is let go of.
//!
//! A shared event's value only rises (docs/graphics-kext/design.md, "Shared events"), so a second
//! signal of a value is harmless.
//!
//! [`RGFX_OP_SURFACE_ORDER`]: reims_graphics::ops::RGFX_OP_SURFACE_ORDER

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use reims_graphics::ops::RgfxSurfaceOrderWait;
use reims_graphics::wire::RGFX_CONTEXT_OBJ_ID_SHIFT;
use reims_vgpu_core::platform::command::{CommandBuffer, CommandQueue, CompletedInfo};
use reims_vgpu_core::platform::device::GpuDevice;
use reims_vgpu_core::platform::event::{EventRef, SharedEvent};
use reims_vgpu_core::platform::{Handle, Platform};
use reims_vgpu_core::raise::{OnFault, RaiseResult, refusals};

use super::commit_order;

/// The waits and signals run under a live raise handler (the ring loop's).
const ON: OnFault = OnFault::Return;

/// How many command buffers that hold the op pass between the logs of what the order did.
const LOGGED_EVERY: u64 = 4096;

/// What a queue's submission ring has published: the highest seqno of a message on it, taken or
/// not; `None` where the queue has ended at the device. Given by the host as the queue is made.
pub(crate) type Published = Arc<dyn Fn() -> Option<u64> + Send + Sync>;

/// One queue's marks, shared with its table entry (`tables::QueueCounts`), so a buffer that holds
/// no op reads and writes them with no lock ([`Self::quiet`]). Each only rises.
#[derive(Default)]
pub(crate) struct Marks {
    /// The highest seqno of a buffer the host took off the queue's ring.
    taken: AtomicU64,
    /// The highest seqno a buffer of another queue waits for.
    waited: AtomicU64,
    /// The highest seqno a command buffer on the queue signals; 0 for none. Raised under the
    /// order's lock, as the signal is encoded.
    signalled: AtomicU64,
    /// The highest seqno a command buffer committed on the queue signals: raised once it is
    /// committed, so a buffer that signals and is then not committed (its walk faulted) leaves it
    /// where it was.
    committed: AtomicU64,
    /// The highest seqno signalled whose command buffer has finished.
    done: AtomicU64,
    /// How far the queue's buffers have been committed, which a buffer of another queue that
    /// holds `COMMIT_ORDER` waits for ([`super::commit_order`]).
    pub(crate) commit: commit_order::CommitMark,
}

impl Marks {
    /// Notes the queue's buffer at `seqno` taken; whether it need signal nothing, for it holds no
    /// op: no buffer of another queue waits past what the queue signalled. Where not, the buffer
    /// signals under the order's lock ([`SurfaceOrder::signal`]).
    pub(crate) fn quiet(&self, seqno: u64) -> bool {
        self.taken.fetch_max(seqno, Ordering::SeqCst);
        self.waited.load(Ordering::SeqCst) <= self.signalled.load(Ordering::SeqCst)
    }
}

/// A signal a command buffer carries: its queue's `done` rises to `seqno` once it finishes.
pub(crate) struct Signalled {
    marks: Arc<Marks>,
    seqno: u64,
}

impl Signalled {
    /// The command buffer finished.
    pub(crate) fn finished(&self) {
        self.marks.done.fetch_max(self.seqno, Ordering::AcqRel);
    }

    /// What notes the command buffer committed, for after its commit ([`Committed::note`]).
    pub(crate) fn committed(&self) -> Committed {
        Committed {
            marks: Arc::clone(&self.marks),
            seqno: self.seqno,
        }
    }
}

/// A command buffer that signals its queue's timeline, to note once it is committed: a CPU set
/// that stands in for a later signal of the queue waits for its signal ([`Release::signal`]).
pub(crate) struct Committed {
    marks: Arc<Marks>,
    seqno: u64,
}

impl Committed {
    /// The command buffer is committed.
    pub(crate) fn note(&self) {
        self.marks.committed.fetch_max(self.seqno, Ordering::AcqRel);
    }
}

/// A signal of a queue's timeline that no buffer of the guest's carries, decided under the
/// order's lock and made after it ([`Self::signal`]).
#[must_use]
pub(crate) struct Release<P: Platform> {
    event: P::SharedEvent,
    marks: Arc<Marks>,
    value: u64,
    /// What the queue's command buffers committed when it was decided signal
    /// ([`Marks::committed`]).
    after: u64,
}

impl<P: Platform> Release<P> {
    /// Signals the timeline to its value with a command buffer of `host_queue` holding that alone,
    /// after the queue's work before it. Where there is no queue or it makes no command buffer (the
    /// platform out of memory), sets it on the CPU, once the timeline has reached what the queue's
    /// command buffers committed before the release was decided signal, as the command buffer's
    /// signal would have come after them: waiting on the CPU, with no lock held, until it has or a
    /// reset of the device comes (`cancel`), which ends every wait for another queue's buffers.
    pub(crate) fn signal(
        self,
        host_queue: Option<&P::CommandQueue>,
        cancel: &commit_order::Cancel,
    ) -> RaiseResult<()> {
        let Release {
            event,
            marks,
            value,
            after,
        } = self;
        let cb = match host_queue {
            Some(q) => q.command_buffer(ON)?,
            None => None,
        };
        match cb {
            Some(cb) => {
                cb.encode_signal(Some(EventRef::Shared(&event)), value)?;
                let done = Arc::clone(&marks);
                // Raised as the platform sees the signal's buffer done, with no handler thread
                // woken between.
                cb.add_completion_seen_handler(
                    ON,
                    Box::new(move |_: CompletedInfo| {
                        done.done.fetch_max(value, Ordering::AcqRel);
                    }),
                )?;
                cb.commit(ON)?;
                marks.committed.fetch_max(value, Ordering::AcqRel);
            }
            None => {
                let begun = cancel.now();
                while marks.done.load(Ordering::Acquire) < after && !cancel.cut_since(begun) {
                    if event.wait_until_signaled_value(after, CPU_WAIT_SLICE_MS) {
                        break;
                    }
                }
                event.set_signaled_value(value)?;
                marks.done.fetch_max(value, Ordering::AcqRel);
                marks.committed.fetch_max(value, Ordering::AcqRel);
            }
        }
        Ok(())
    }
}

/// A wait of a command buffer for a buffer of another queue ([`SurfaceOrder::before`]), kept for
/// the work the host starts for the command buffer off it: a video session's decode
/// ([`super::GraphicsDecoder::wait_for_order_on_cpu`]).
pub(crate) struct OrderWait<P: Platform> {
    event: P::SharedEvent,
    value: u64,
    marks: Arc<Marks>,
}

impl<P: Platform> OrderWait<P> {
    /// Waits on the CPU, at most `milliseconds`, for the buffer waited for to finish; whether it
    /// did.
    fn wait(&self, milliseconds: u64) -> bool {
        self.marks.done.load(Ordering::Acquire) >= self.value
            || self
                .event
                .wait_until_signaled_value(self.value, milliseconds)
    }
}

/// What the order did, for the log ([`SurfaceOrder::counts`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct OrderCounts {
    /// Command buffers that held the op.
    pub(crate) buffers: u64,
    /// Waits named by them of other queues.
    pub(crate) named: u64,
    /// Of those, the GPU waits made.
    pub(crate) waits: u64,
    /// Of those, of a buffer the host had not yet taken.
    pub(crate) ahead: u64,
    /// Waits named of a queue the host does not hold, or past what its ring has published.
    pub(crate) stale: u64,
    /// Signals for buffers that did not run: a command buffer of their own, or a CPU set.
    pub(crate) released: u64,
}

/// One queue the host holds, from its create to its end.
struct Queue<P: Platform> {
    /// Its marks, shared with its table entry.
    marks: Arc<Marks>,
    /// What its ring has published, as the host gave it; `None` where it gave nothing (the
    /// decoder's own tests).
    published: Option<Published>,
    /// Its surface timeline, made with the queue ([`SurfaceOrder::made`]), or at the first buffer
    /// that signals or waits on it where it was not.
    event: Option<P::SharedEvent>,
    /// The queue is gone (destroyed, or its context closed): nothing signals past `signalled`.
    ended: bool,
}

impl<P: Platform> Queue<P> {
    fn new() -> Self {
        Queue {
            marks: Arc::default(),
            published: None,
            event: None,
            ended: false,
        }
    }

    /// Its timeline, made where it has none; `None` where the platform makes no shared event.
    fn event(&mut self, device: &P::Device) -> Option<&P::SharedEvent> {
        if self.event.is_none() {
            self.event = device.new_shared_event();
        }
        self.event.as_ref()
    }
}

/// The queues' surface timelines, by queue id.
pub(crate) struct SurfaceOrder<P: Platform> {
    queues: HashMap<u64, Queue<P>>,
    counts: OrderCounts,
}

impl<P: Platform> SurfaceOrder<P> {
    pub(crate) fn new() -> Self {
        SurfaceOrder {
            queues: HashMap::new(),
            counts: OrderCounts::default(),
        }
    }

    /// What the order did so far.
    #[cfg(test)]
    pub(crate) fn counts(&self) -> OrderCounts {
        self.counts
    }

    /// The marks of queue `queue`, which the host holds from now where it did not (the tests'
    /// look at a queue's marks; a create takes them from [`Self::made`]).
    #[cfg(test)]
    pub(crate) fn marks(&mut self, queue: u64) -> Arc<Marks> {
        let q = self.queues.entry(queue).or_insert_with(Queue::new);
        Arc::clone(&q.marks)
    }

    /// Queue `queue` was made, with `timeline`, its surface timeline, made with no lock held: the
    /// marks for its table entry, and the timeline back where the queue has one already, for the
    /// caller to let go of with no lock held. A buffer's signal or wait then makes no event under
    /// the order's lock, which every queue's buffers that hold the op take: the platform's other
    /// threads' work on the device's memory and semaphores can stretch a creation to milliseconds
    /// (measured on the RTX 5080 Laptop GPU host: 0.6 to 9.4 ms a timeline at a queue's first
    /// buffer in a guest's boot, against 0.1 ms where no other thread makes or frees the device's
    /// memory or semaphores).
    pub(crate) fn made(
        &mut self,
        queue: u64,
        timeline: Option<P::SharedEvent>,
    ) -> (Arc<Marks>, Option<P::SharedEvent>) {
        let q = self.queues.entry(queue).or_insert_with(Queue::new);
        let spare = match q.event {
            None => {
                q.event = timeline;
                None
            }
            Some(_) => timeline,
        };
        (Arc::clone(&q.marks), spare)
    }

    /// Queue `queue` was made, its ring's published seqnos read through `published`: a wait for
    /// one of its buffers past them names none.
    pub(crate) fn queue_made(&mut self, queue: u64, published: Published) {
        let q = self.queues.entry(queue).or_insert_with(Queue::new);
        q.published = Some(published);
    }

    /// The queues of context `context` the host holds, not yet ended.
    pub(crate) fn queues_of(&self, context: u32) -> Vec<u64> {
        (self.queues.iter())
            .filter(|&(&id, q)| id >> RGFX_CONTEXT_OBJ_ID_SHIFT == u64::from(context) && !q.ended)
            .map(|(&id, _)| id)
            .collect()
    }

    /// The queues a buffer of queue `queue` that holds `COMMIT_ORDER` naming `waits` is committed
    /// after, each with the seqno to wait for ([`super::commit_order`]): each queue once, at its
    /// highest seqno named, of those [`Self::commit_named`] keeps.
    pub(crate) fn commit_after(
        &self,
        queue: u64,
        waits: impl Iterator<Item = RgfxSurfaceOrderWait>,
    ) -> Vec<(Arc<Marks>, u64)> {
        let mut after: Vec<(Arc<Marks>, u64)> = Vec::new();
        for (marks, s) in self.commit_named(queue, waits) {
            match after.iter_mut().find(|(m, _)| Arc::ptr_eq(m, marks)) {
                Some((_, v)) => *v = (*v).max(s),
                None => after.push((Arc::clone(marks), s)),
            }
        }
        after
    }

    /// Whether the buffers a buffer of queue `queue` that holds `COMMIT_ORDER` naming `waits` is
    /// committed after are all begun, so it may take its place among the commit order's tickets
    /// on a doorbell's thread, waiting for nothing ([`super::commit_order`]). Allocates nothing.
    pub(crate) fn commit_begun(
        &self,
        queue: u64,
        waits: impl Iterator<Item = RgfxSurfaceOrderWait>,
    ) -> bool {
        (self.commit_named(queue, waits))
            .all(|(marks, s)| marks.commit.reached(s, commit_order::Until::Begun))
    }

    /// The queues and seqnos of `waits` a buffer of queue `queue` that holds `COMMIT_ORDER` is
    /// committed after: all but the buffer's own queue, a queue of another context, a queue the
    /// host does not hold or holds ended, a seqno past every one the queue's ring has published,
    /// and one its mark has reached.
    fn commit_named(
        &self,
        queue: u64,
        waits: impl Iterator<Item = RgfxSurfaceOrderWait>,
    ) -> impl Iterator<Item = (&Arc<Marks>, u64)> {
        let context = queue >> RGFX_CONTEXT_OBJ_ID_SHIFT;
        waits.filter_map(move |w| {
            let (q, s) = (w.queue_obj_id, w.seqno);
            if q == queue || q == 0 || s == 0 || q >> RGFX_CONTEXT_OBJ_ID_SHIFT != context {
                return None;
            }
            let named = self.queues.get(&q)?;
            if named.ended
                || named
                    .marks
                    .commit
                    .reached(s, commit_order::Until::Committed)
            {
                return None;
            }
            if let Some(published) = named.published.as_ref()
                && published().is_none_or(|p| s > p)
            {
                return None;
            }
            Some((&named.marks, s))
        })
    }

    /// Whether a buffer of queue `queue` that names `waits` waits for none of them on the GPU, so
    /// it may be submitted as it is taken: each names its own queue, a queue the host does not
    /// hold, or a buffer the host has seen finish. Allocates nothing.
    pub(crate) fn ready(
        &self,
        queue: u64,
        mut waits: impl Iterator<Item = RgfxSurfaceOrderWait>,
    ) -> bool {
        waits.all(|w| {
            if w.queue_obj_id == queue || w.queue_obj_id == 0 || w.seqno == 0 {
                return true;
            }
            match self.queues.get(&w.queue_obj_id) {
                Some(q) => q.marks.done.load(Ordering::Acquire) >= w.seqno,
                None => true,
            }
        })
    }

    /// Makes `cb`, a command buffer of queue `queue` that holds the op, wait on the GPU for each
    /// buffer of `waits` not yet finished (the module doc); the waits it made, one a queue.
    pub(crate) fn before(
        &mut self,
        device: &P::Device,
        queue: u64,
        cb: &P::CommandBuffer,
        waits: &[RgfxSurfaceOrderWait],
    ) -> RaiseResult<Vec<OrderWait<P>>> {
        self.counts.buffers += 1;
        if self.counts.buffers.is_multiple_of(LOGGED_EVERY) {
            let c = self.counts;
            log::info!(
                target: "reims_vgpu::graphics",
                "surface order: {} command buffers named {} buffers of other queues: {} GPU \
                 waits ({} for buffers not yet taken), {} naming no buffer; {} signals for \
                 buffers that did not run",
                c.buffers, c.named, c.waits, c.ahead, c.stale, c.released
            );
        }
        // One wait a queue, for the highest value of it.
        let mut targets: Vec<(u64, u64)> = Vec::new();
        for w in waits {
            let (q, s) = (w.queue_obj_id, w.seqno);
            if q == queue || q == 0 || s == 0 {
                continue;
            }
            self.counts.named += 1;
            let Some(named) = self.queues.get_mut(&q) else {
                self.counts.stale += 1;
                continue;
            };
            let marks = Arc::clone(&named.marks);
            let done = marks.done.load(Ordering::Acquire);
            if done >= s || named.event(device).is_none() {
                continue;
            }
            let signalled = marks.signalled.load(Ordering::Acquire);
            let target = if named.ended {
                s.min(signalled)
            } else if s <= signalled {
                s
            } else {
                if let Some(published) = named.published.as_ref().and_then(|p| p())
                    && s > published
                {
                    self.counts.stale += 1;
                    continue;
                }
                // Not yet signalled. Raised before `taken` is read ([`Marks::quiet`]).
                marks.waited.fetch_max(s, Ordering::SeqCst);
                if marks.taken.load(Ordering::SeqCst) >= s {
                    // Taken without a signal: it held no op, so it used no surface the bundle
                    // named. What the queue signalled before it is what it can be ordered after.
                    signalled
                } else {
                    // Not yet taken: it signals once taken, if its ring holds it.
                    self.counts.ahead += 1;
                    s
                }
            };
            if target <= done {
                continue;
            }
            match targets.iter_mut().find(|(tq, _)| *tq == q) {
                Some((_, v)) => *v = (*v).max(target),
                None => targets.push((q, target)),
            }
        }
        let mut made = Vec::with_capacity(targets.len());
        for &(q, value) in &targets {
            let Some(named) = self.queues.get(&q) else {
                continue;
            };
            if let Some(event) = named.event.as_ref() {
                cb.encode_wait(Some(EventRef::Shared(event)), value)?;
                self.counts.waits += 1;
                made.push(OrderWait {
                    event: event.clone_ref(),
                    value,
                    marks: Arc::clone(&named.marks),
                });
            }
        }
        Ok(made)
    }

    /// Notes queue `queue`'s buffer at `seqno` taken, and has `cb`, its command buffer or its
    /// stand-in, signal the queue's timeline to `seqno` as it ends where it `names` the op or a
    /// buffer of another queue waits past what the queue has signalled; then what its completed
    /// handler calls ([`Signalled::finished`]). Called after the buffer's walk, before its commit.
    pub(crate) fn signal(
        &mut self,
        device: &P::Device,
        queue: u64,
        seqno: u64,
        names: bool,
        cb: &P::CommandBuffer,
    ) -> RaiseResult<Option<Signalled>> {
        let Some((event, marks)) = self.signal_due(device, queue, seqno, names) else {
            return Ok(None);
        };
        cb.encode_signal(Some(EventRef::Shared(&event)), seqno)?;
        Ok(Some(Signalled { marks, seqno }))
    }

    /// Queue `queue`'s buffer at `seqno`, which `names` the op or not, was taken and will not
    /// run, and has no stand-in: where a buffer of another queue may wait for it, the signal of the
    /// queue's timeline to `seqno` that stands in for its, to make once the order's lock is let go
    /// of ([`Release::signal`]).
    pub(crate) fn release(
        &mut self,
        device: &P::Device,
        queue: u64,
        seqno: u64,
        names: bool,
    ) -> Option<Release<P>> {
        let (event, marks) = self.signal_due(device, queue, seqno, names)?;
        self.counts.released += 1;
        let after = marks.committed.load(Ordering::Acquire);
        Some(Release {
            event,
            marks,
            value: seqno,
            after,
        })
    }

    /// Notes queue `queue`'s buffer at `seqno` taken; where it is to signal (it `names` the op, or
    /// a buffer of another queue waits past what the queue has signalled, and nothing after it
    /// signalled), the queue's timeline and marks, its signal counted in them.
    fn signal_due(
        &mut self,
        device: &P::Device,
        queue: u64,
        seqno: u64,
        names: bool,
    ) -> Option<(P::SharedEvent, Arc<Marks>)> {
        let q = self.queues.get_mut(&queue)?;
        let marks = Arc::clone(&q.marks);
        marks.taken.fetch_max(seqno, Ordering::SeqCst);
        let signalled = marks.signalled.load(Ordering::Acquire);
        if !(names || marks.waited.load(Ordering::SeqCst) > signalled) || seqno < signalled {
            return None;
        }
        let event = q.event(device)?.clone_ref();
        marks.signalled.fetch_max(seqno, Ordering::AcqRel);
        Some((event, marks))
    }

    /// Queue `queue` is gone (its destroy, or its context's close): nothing of it signals past
    /// what it has, so where a buffer of another queue waits past that, the signal of the highest
    /// value waited for, to make once the order's lock is let go of ([`Release::signal`]).
    /// Queues gone whose signals have all finished are let go of.
    pub(crate) fn ended(&mut self, queue: u64) -> Option<Release<P>> {
        let mut release = None;
        if let Some(q) = self.queues.get_mut(&queue) {
            q.ended = true;
            q.published = None;
            let marks = Arc::clone(&q.marks);
            // Nothing waits for its commits any more ([`super::commit_order`]).
            marks.commit.end();
            let waited = marks.waited.load(Ordering::SeqCst);
            if waited > marks.signalled.load(Ordering::Acquire)
                && let Some(event) = q.event.as_ref()
            {
                marks.signalled.fetch_max(waited, Ordering::AcqRel);
                self.counts.released += 1;
                let after = marks.committed.load(Ordering::Acquire);
                release = Some(Release {
                    event: event.clone_ref(),
                    marks,
                    value: waited,
                    after,
                });
            }
        }
        let gone: Vec<u64> = (self.queues.iter())
            .filter(|(_, q)| {
                q.ended
                    && q.marks.done.load(Ordering::Acquire)
                        >= q.marks.signalled.load(Ordering::Acquire)
            })
            .map(|(&id, _)| id)
            .collect();
        for id in gone {
            if let Some(q) = self.queues.remove(&id)
                && let Err(refusal) = refusals::refusing(move || drop(q.event))
            {
                log::warn!(
                    target: "reims_vgpu::graphics",
                    "queue {id:#x}'s surface timeline was let go of, and its release was \
                     refused: {refusal}"
                );
            }
        }
        release
    }

    /// The device's reset: every queue's timeline goes, each let go of in a refusal scope. The
    /// counts stay.
    pub(crate) fn reset(&mut self) {
        let queues = std::mem::take(&mut self.queues);
        if let Err(refusal) = refusals::refusing(move || drop(queues)) {
            log::warn!(
                target: "reims_vgpu::graphics",
                "the surface timelines were let go of at the device's reset, and a release was \
                 refused: {refusal}"
            );
        }
    }

    /// Queue `queue`'s timeline, where it has one: its event and (signalled, done, waited,
    /// taken), for the tests.
    #[cfg(test)]
    pub(crate) fn state(&self, queue: u64) -> Option<(P::SharedEvent, [u64; 4])> {
        let q = self.queues.get(&queue)?;
        let m = &q.marks;
        q.event.as_ref().map(|event| {
            (
                event.clone_ref(),
                [
                    m.signalled.load(Ordering::Acquire),
                    m.done.load(Ordering::Acquire),
                    m.waited.load(Ordering::Acquire),
                    m.taken.load(Ordering::Acquire),
                ],
            )
        })
    }
}

/// How long each slice of the host's CPU wait for a buffer of another queue lasts
/// ([`super::GraphicsDecoder::wait_for_order_on_cpu`]), between its looks at whether a reset of
/// the device is under way.
const CPU_WAIT_SLICE_MS: u64 = 100;

/// How long the host waits on the CPU for a buffer of another queue before the log names the
/// wait (diagnostics only): it waits on after.
const CPU_WAIT_LOGGED_MS: u64 = 2000;

impl<P: Platform> super::GraphicsDecoder<P> {
    /// Before the host starts work for the command buffer being walked off it, on a queue its
    /// GPU waits do not hold (a video session's decode): waits on the CPU for the buffers of other
    /// queues it waits for ([`OrderWait`]), once a command buffer, as its GPU waits would hold the
    /// work on a Mac's GPU: until each has finished, or a reset of the device comes, which ends
    /// every wait for another queue's buffers ([`super::commit_order::Cancel`]). Each named buffer
    /// finishes, or is signalled for, as the module doc says, so none waits for good. A wait past
    /// [`CPU_WAIT_LOGGED_MS`] is named in the log, and again once it ends.
    pub(super) fn wait_for_order_on_cpu(&self) {
        let waits = std::mem::take(&mut *self.order_waits.borrow_mut());
        let cancel = &self.shared.commit_cancel;
        let begun = cancel.now();
        for w in &waits {
            let mut waited = 0;
            while !w.wait(CPU_WAIT_SLICE_MS) {
                if cancel.cut_since(begun) {
                    log::info!(
                        target: "reims_vgpu::graphics",
                        "surface order: a reset of the device ended the CPU wait of work started                          off a command buffer for a buffer of another queue at {}",
                        w.value
                    );
                    return;
                }
                waited += CPU_WAIT_SLICE_MS;
                if waited == CPU_WAIT_LOGGED_MS {
                    log::warn!(
                        target: "reims_vgpu::graphics",
                        "surface order: work started off a command buffer has waited                          {CPU_WAIT_LOGGED_MS} ms on the CPU for a buffer of another queue at {},                          which has not finished; it waits on",
                        w.value
                    );
                }
            }
            if waited >= CPU_WAIT_LOGGED_MS {
                log::info!(
                    target: "reims_vgpu::graphics",
                    "surface order: the buffer of another queue at {} finished after about {waited}                      ms of a CPU wait",
                    w.value
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use std::sync::atomic::Ordering;

    use reims_graphics::ops::{
        RGFX_COMMAND_BUFFER_OPS_OFFSET, RGFX_OP_SUBMITTED, RGFX_OP_SURFACE_ORDER,
        RgfxCommandBuffer, RgfxOpHeader,
    };
    use reims_graphics::wire::{
        RGFX_CMD_COMMAND_BUFFER, RGFX_CMD_OBJECT_DESTROY, RGFX_CMD_QUEUE_CREATE,
        RGFX_CONTEXT_OBJ_ID_SHIFT, RGFX_EVT_CB_SUBMITTED, RgfxMsgHeader, RgfxObjectDestroy,
        RgfxQueueCreate,
    };
    use reims_vgpu_core::sync::Mutex;
    use reims_vgpu_core::testkit::calllog::{Arg, Call};
    use reims_vgpu_core::testkit::objects::CompleteOn;
    use reims_vgpu_core::testkit::{FakePlatform, Step};
    use zerocopy::IntoBytes;

    use super::super::{DecodeError, GraphicsDecoder, refusing};
    use super::*;

    const WAIT: &str = "encodeWaitForEvent:value:";
    const SIGNAL: &str = "encodeSignalEvent:value:";
    const SET: &str = "setSignaledValue:";

    /// Queue 1 of process 1, and of process 2.
    const A: u64 = 1 << RGFX_CONTEXT_OBJ_ID_SHIFT | 1;
    const B: u64 = 2 << RGFX_CONTEXT_OBJ_ID_SHIFT | 1;

    struct Rig {
        p: Arc<FakePlatform>,
        device: <FakePlatform as Platform>::Device,
        dec: GraphicsDecoder<FakePlatform>,
    }

    /// A decoder holding queue [`A`] of process 1 and queue [`B`] of process 2.
    fn rig() -> Rig {
        let p = Arc::new(FakePlatform::new());
        let device = p.device();
        let mut dec = GraphicsDecoder::<FakePlatform>::new(
            Arc::clone(&p),
            Arc::new(reims_graphics::region::Region::new(0x10000).unwrap()),
        );
        for queue in [A, B] {
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
        Rig { p, device, dec }
    }

    fn context(queue: u64) -> u32 {
        (queue >> RGFX_CONTEXT_OBJ_ID_SHIFT) as u32
    }

    /// The ops of a buffer that names `waits` in its `SURFACE_ORDER`, or holds none (`None`), and
    /// then `rest`.
    fn ops(waits: Option<&[(u64, u64)]>, rest: &[u8]) -> Vec<u8> {
        let mut ops = Vec::new();
        if let Some(waits) = waits {
            let header = RgfxOpHeader {
                tag: RGFX_OP_SURFACE_ORDER,
                flags: 0,
                len: (8 + 16 * waits.len()) as u32,
            };
            ops.extend_from_slice(header.as_bytes());
            for &(queue_obj_id, seqno) in waits {
                let w = RgfxSurfaceOrderWait {
                    queue_obj_id,
                    seqno,
                };
                ops.extend_from_slice(w.as_bytes());
            }
        }
        ops.extend_from_slice(rest);
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
        /// Carries out queue `queue`'s buffer at `seqno` of `ops`, as its process's ring hands it
        /// over, standing in for it where it is not carried out (as the host does); its outcome,
        /// and the calls made for it.
        fn commit(
            &mut self,
            queue: u64,
            seqno: u64,
            ops: &[u8],
        ) -> (Result<(), DecodeError>, Vec<Call>) {
            let message = message(queue, seqno, ops);
            let mark = self.p.call_log().mark();
            self.dec.set_context(Some(context(queue)));
            let outcome = refusing(|| self.dec.decode(&self.device, &message));
            let outcome = match outcome {
                Ok(Ok(())) => Ok(()),
                other => {
                    let _ = refusing(|| self.dec.stand_in_for_failed(&self.device, &message));
                    other.unwrap_or(Err(DecodeError::Refused))
                }
            };
            (outcome, self.p.call_log().since(mark))
        }

        /// The id of queue `queue`'s timeline's event.
        fn event(&self, queue: u64) -> u64 {
            let (event, _) = self.dec.shared.surface_order.lock().state(queue).unwrap();
            Arg::obj(&event).as_obj().unwrap().0
        }

        /// Queue `queue`'s timeline: (signalled, done, waited, taken).
        fn timeline(&self, queue: u64) -> [u64; 4] {
            self.dec.shared.surface_order.lock().state(queue).unwrap().1
        }
    }

    /// The (event, value) of each call to `sel` in `calls`.
    fn events(calls: &[Call], sel: &str) -> Vec<(u64, u64)> {
        (calls.iter())
            .filter(|c| c.sel == sel)
            .map(|c| {
                let event = c.args[0].as_obj().map_or(0, |o| o.0);
                (event, c.args[1].as_u64().unwrap_or(0))
            })
            .collect()
    }

    /// Work the host starts off a command buffer (a video session's decode) waits on the CPU for
    /// each buffer of another queue the command buffer runs after until it has finished, however
    /// long that takes, as the buffer's GPU wait would hold the work on a Mac's GPU: past the 2 s
    /// after which it once went on unordered. A reset of the device ends the wait.
    #[test]
    #[allow(clippy::disallowed_methods)] // test scaffolding: a thread waits, real time paces it
    fn work_off_a_buffer_waits_on_the_cpu_until_the_buffers_it_runs_after_finish() {
        use std::time::Duration;
        for reset in [false, true] {
            let mut r = rig();
            // B's buffer names A's at seqno 4, which the host has not taken.
            let (outcome, _) = r.commit(B, 1, &ops(Some(&[(A, 4)]), &[]));
            assert_eq!(outcome, Ok(()));
            assert_eq!(r.dec.order_waits.borrow().len(), 1);
            let (event, _) = r.dec.shared.surface_order.lock().state(A).unwrap();
            let shared = Arc::clone(&r.dec.shared);
            let Rig { p, dec, .. } = r;
            let waiting = std::thread::spawn(move || {
                dec.wait_for_order_on_cpu();
                dec
            });
            // Five seconds of the platform's clock, a slice at a time.
            for _ in 0..50 {
                p.virtual_clock()
                    .advance(Duration::from_millis(CPU_WAIT_SLICE_MS));
                std::thread::sleep(Duration::from_millis(1));
            }
            std::thread::sleep(Duration::from_millis(50));
            assert!(
                !waiting.is_finished(),
                "the wait went on before A's buffer finished"
            );
            if reset {
                shared.commit_cancel.begin();
            } else {
                event.set_signaled_value(4).unwrap();
            }
            let mut finished = false;
            for _ in 0..2000 {
                p.virtual_clock()
                    .advance(Duration::from_millis(CPU_WAIT_SLICE_MS));
                if waiting.is_finished() {
                    finished = true;
                    break;
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            assert!(finished, "the wait did not end (reset {reset})");
            let dec = waiting.join().unwrap();
            assert!(dec.order_waits.borrow().is_empty());
            if reset {
                shared.commit_cancel.end();
            }
        }
    }

    /// A buffer of process 2 that names process 1's buffer waits for it on the GPU, on queue
    /// A's timeline at its seqno, which that buffer signals as it ends; once it has finished, a
    /// buffer that names it waits for nothing, and so does one that names its own queue
    #[test]
    fn a_buffer_waits_on_the_gpu_for_a_named_buffer_of_another_process_until_it_finishes() {
        let mut r = rig();
        let (outcome, calls) = r.commit(A, 5, &ops(Some(&[]), &[]));
        assert_eq!(outcome, Ok(()));
        let a = r.event(A);
        assert_eq!(
            events(&calls, SIGNAL),
            [(a, 5)],
            "A's buffer signals its seqno"
        );
        assert!(events(&calls, WAIT).is_empty());

        let (outcome, calls) = r.commit(B, 3, &ops(Some(&[(A, 5)]), &[]));
        assert_eq!(outcome, Ok(()));
        let b = r.event(B);
        assert_eq!(events(&calls, WAIT), [(a, 5)], "B's waits for A's");
        assert_eq!(events(&calls, SIGNAL), [(b, 3)]);
        let c = r.dec.shared.surface_order.lock().counts();
        assert_eq!((c.named, c.waits, c.ahead), (1, 1, 0));

        r.p.world().complete_all(CompleteOn::Caller);
        assert_eq!(r.timeline(A)[1], 5, "A's done");
        let (_, calls) = r.commit(B, 4, &ops(Some(&[(A, 5), (B, 3)]), &[]));
        assert!(
            events(&calls, WAIT).is_empty(),
            "a finished buffer, and one of its own queue"
        );
        assert_eq!(events(&calls, SIGNAL), [(b, 4)]);
        assert_eq!(r.dec.shared.surface_order.lock().counts().waits, 1);
    }

    /// Process 2's buffer, committed after process 1's but taken first (each ring has its own
    /// thread), still waits for it: for a value not yet signalled, which process 1's buffer
    /// signals once it is taken, even one that holds no op; one that names a buffer its queue has
    /// signalled waits for that value
    #[test]
    fn a_buffer_taken_before_one_it_names_waits_for_it() {
        let mut r = rig();
        let (_, calls) = r.commit(B, 1, &ops(Some(&[(A, 7)]), &[]));
        let a = r.event(A);
        assert_eq!(events(&calls, WAIT), [(a, 7)], "waits before the signal");
        assert_eq!(r.timeline(A), [0, 0, 7, 0]);
        assert_eq!(r.dec.shared.surface_order.lock().counts().ahead, 1);

        // A's buffer 6 holds no op, so signals nothing anyone waits for: 6 is below 7, but it
        // signals all the same, as a buffer of a queue waited past what it signalled.
        let (_, calls) = r.commit(A, 6, &ops(None, &[]));
        assert_eq!(events(&calls, SIGNAL), [(a, 6)]);
        let (_, calls) = r.commit(A, 7, &ops(None, &[]));
        assert_eq!(
            events(&calls, SIGNAL),
            [(a, 7)],
            "the buffer waited for signals"
        );
        assert_eq!(r.timeline(A), [7, 0, 7, 7]);
        let (_, calls) = r.commit(A, 8, &ops(None, &[]));
        assert!(events(&calls, SIGNAL).is_empty(), "nothing waits past 7");
    }

    /// A buffer that holds no op waits for nothing, signals nothing and makes nothing; one that
    /// names a queue the host does not hold, or a seqno past every one the named queue's ring has
    /// published, waits for nothing
    #[test]
    fn buffers_that_share_no_named_buffer_wait_for_nothing() {
        let mut r = rig();
        let mark = r.p.call_log().mark();
        let (_, calls) = r.commit(A, 1, &ops(None, &[]));
        assert!(events(&calls, WAIT).is_empty() && events(&calls, SIGNAL).is_empty());
        assert!(
            !r.p.call_log().sels_since(mark).contains(&"newSharedEvent"),
            "nothing made"
        );

        let unknown = 3 << RGFX_CONTEXT_OBJ_ID_SHIFT | 1;
        r.dec.surface_order_queue_made(A, Arc::new(|| Some(10)));
        let (_, calls) = r.commit(B, 1, &ops(Some(&[(unknown, 4), (A, 11)]), &[]));
        assert!(events(&calls, WAIT).is_empty());
        assert_eq!(r.timeline(A)[2], 0, "nothing waits for A");
        let (_, calls) = r.commit(B, 2, &ops(Some(&[(A, 10)]), &[]));
        assert_eq!(
            events(&calls, WAIT),
            [(r.event(A), 10)],
            "within what A published"
        );
        let c = r.dec.shared.surface_order.lock().counts();
        assert_eq!((c.named, c.stale, c.waits), (3, 2, 1));
    }

    /// A queue's timeline is made with the queue, so neither its first buffer that holds the op,
    /// which signals it, nor a buffer of another queue that waits on it makes an event
    #[test]
    fn a_queues_timeline_is_made_with_it_and_no_buffer_makes_one() {
        let mut r = rig();
        let (a, b) = (r.event(A), r.event(B));
        let mark = r.p.call_log().mark();
        let (outcome, calls) = r.commit(A, 1, &ops(Some(&[]), &[]));
        assert_eq!(outcome, Ok(()));
        assert_eq!(
            events(&calls, SIGNAL),
            [(a, 1)],
            "A's first buffer signals its timeline"
        );
        let (outcome, calls) = r.commit(B, 1, &ops(Some(&[(A, 2)]), &[]));
        assert_eq!(outcome, Ok(()));
        assert_eq!(events(&calls, WAIT), [(a, 2)], "B's waits on A's timeline");
        assert_eq!(events(&calls, SIGNAL), [(b, 1)]);
        assert!(
            !r.p.call_log().sels_since(mark).contains(&"newSharedEvent"),
            "no event made at a buffer"
        );
    }

    /// A buffer that is not carried out still signals what may be waited for of it: its stand-in
    /// does; with none (its queue made no command buffer for it or its stand-in), a command
    /// buffer of its own on its queue does; and where its queue makes no command buffer, the
    /// timeline is set on the CPU, once the queue's earlier signals have come
    #[test]
    fn a_buffer_not_carried_out_still_signals() {
        let mut r = rig();
        let (_, calls) = r.commit(B, 1, &ops(Some(&[(A, 9)]), &[]));
        let a = r.event(A);
        assert_eq!(events(&calls, WAIT), [(a, 9)]);
        // An op the host does not know faults the buffer, and its stand-in signals.
        let unknown_op = RgfxOpHeader {
            tag: 0x0fff,
            flags: 0,
            len: 8,
        };
        let (outcome, calls) = r.commit(A, 9, &ops(Some(&[]), unknown_op.as_bytes()));
        assert!(outcome.is_err());
        assert_eq!(events(&calls, SIGNAL), [(a, 9)], "the stand-in signals");

        let (_, calls) = r.commit(B, 2, &ops(Some(&[(A, 10)]), &[]));
        assert_eq!(events(&calls, WAIT), [(a, 10)]);
        // Its queue makes no command buffer for the buffer, nor for its stand-in.
        r.p.script().push("commandBuffer", Step::Nil);
        r.p.script().push("commandBuffer", Step::Nil);
        let (outcome, calls) = r.commit(A, 10, &ops(Some(&[]), &[]));
        assert_eq!(outcome, Err(DecodeError::Refused));
        assert_eq!(
            events(&calls, SIGNAL),
            [(a, 10)],
            "a command buffer of its own signals"
        );
        assert!(calls.iter().any(|c| c.sel == "commit"));

        // The stand-in's and the signal's of its own have finished.
        r.p.world().complete_all(CompleteOn::Caller);
        for _ in 0..3 {
            r.p.script().push("commandBuffer", Step::Nil);
        }
        let (_, calls) = r.commit(A, 11, &ops(Some(&[]), &[]));
        assert!(events(&calls, SIGNAL).is_empty());
        let set: Vec<(u64, u64)> = (calls.iter())
            .filter(|c| c.sel == SET)
            .map(|c| (c.obj.0, c.args[0].as_u64().unwrap_or(0)))
            .collect();
        assert_eq!(set, [(a, 11)], "set on the CPU");
        assert_eq!(r.timeline(A)[0..2], [11, 11]);
        assert_eq!(r.dec.shared.surface_order.lock().counts().released, 2);
    }

    /// A CPU set that stands in for a buffer's signal, where its queue makes no command buffer,
    /// comes once the queue's command buffers committed before it have reached their signals, as
    /// a command buffer's signal would: it releases a buffer of another queue that waits for the
    /// buffer no earlier than the queue's work before it ends
    #[test]
    #[allow(clippy::disallowed_methods)] // test scaffolding: a thread carries the buffer out
    fn a_cpu_set_for_a_buffer_not_carried_out_waits_for_its_queue_s_earlier_work() {
        use std::time::Duration;
        let mut r = rig();
        // A's buffer 1 signals its timeline at 1, committed and not finished.
        let (outcome, calls) = r.commit(A, 1, &ops(Some(&[]), &[]));
        assert_eq!(outcome, Ok(()));
        let a = r.event(A);
        assert_eq!(events(&calls, SIGNAL), [(a, 1)]);
        // B's buffer waits for A's buffer 2.
        let (_, calls) = r.commit(B, 1, &ops(Some(&[(A, 2)]), &[]));
        assert_eq!(events(&calls, WAIT), [(a, 2)]);
        // A's queue makes no command buffer for its buffer 2, its stand-in, or a signal of its
        // own: the timeline is set on the CPU, on the thread that carries the buffer out.
        for _ in 0..3 {
            r.p.script().push("commandBuffer", Step::Nil);
        }
        let p = Arc::clone(&r.p);
        let begun = p.call_log().mark();
        let carrying = std::thread::spawn(move || {
            let committed = r.commit(A, 2, &ops(Some(&[]), &[]));
            (r, committed)
        });
        // The set, or a wait before it, comes: a set first came before A's buffer 1 finished.
        let mut seen = Vec::new();
        for _ in 0..3000 {
            seen = p.call_log().sels_since(begun);
            if seen.contains(&SET) || seen.contains(&"waitUntilSignaledValue:timeoutMS:") {
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(
            !seen.contains(&SET),
            "the timeline was set before A's buffer 1 finished: {seen:?}"
        );
        let finished = p.call_log().mark();
        p.world().complete_all(CompleteOn::Caller);
        let (r, (outcome, _)) = carrying.join().unwrap();
        assert_eq!(outcome, Err(DecodeError::Refused));
        let set: Vec<(u64, u64)> = (p.call_log().since(finished).iter())
            .filter(|c| c.sel == SET)
            .map(|c| (c.obj.0, c.args[0].as_u64().unwrap_or(0)))
            .collect();
        assert_eq!(set, [(a, 2)], "set on the CPU once A's buffer 1 finished");
        assert_eq!(r.timeline(A)[0..2], [2, 2]);
    }

    /// A queue's destroy, and its context's close, signal the highest value a buffer of another
    /// queue waits for of it, from a last command buffer on it; a queue nothing waits past
    /// signals nothing more
    #[test]
    fn a_queue_gone_signals_what_is_waited_for_of_it() {
        let mut r = rig();
        let _ = r.commit(B, 1, &ops(Some(&[(A, 12)]), &[]));
        let a = r.event(A);
        let destroy = RgfxObjectDestroy {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_OBJECT_DESTROY,
                size: size_of::<RgfxObjectDestroy>() as u32,
                seqno: 0,
            },
            obj_id: A,
        };
        r.dec.set_context(Some(1));
        let mark = r.p.call_log().mark();
        assert_eq!(r.dec.decode(&r.device, destroy.as_bytes()).unwrap(), Ok(()));
        let calls = r.p.call_log().since(mark);
        assert_eq!(events(&calls, SIGNAL), [(a, 12)]);
        assert!(calls.iter().any(|c| c.sel == "commit"));
        // Its timeline stays until that signal finishes: a buffer that names a later one of it
        // waits for no more than it signalled.
        let (_, calls) = r.commit(B, 2, &ops(Some(&[(A, 20)]), &[]));
        assert_eq!(events(&calls, WAIT), [(a, 12)]);
        r.p.world().complete_all(CompleteOn::Caller);
        let (_, calls) = r.commit(B, 3, &ops(Some(&[(A, 20)]), &[]));
        assert!(events(&calls, WAIT).is_empty(), "its last signal finished");

        // B's process closes with nothing waiting past what B signalled.
        let mark = r.p.call_log().mark();
        r.dec.drop_context(&r.device, 2);
        assert!(events(&r.p.call_log().since(mark), SIGNAL).is_empty());
    }

    /// A buffer a doorbell's thread would answer is submitted at its take only where it waits
    /// for no buffer of another queue: each it names is of its own queue, of no queue, or
    /// finished
    #[test]
    fn a_buffer_that_waits_is_not_submitted_as_it_is_taken() {
        let mut r = rig();
        let ready = |r: &Rig, waits: &[(u64, u64)]| {
            let waits: Vec<RgfxSurfaceOrderWait> = (waits.iter())
                .map(|&(queue_obj_id, seqno)| RgfxSurfaceOrderWait {
                    queue_obj_id,
                    seqno,
                })
                .collect();
            r.dec
                .shared
                .surface_order
                .lock()
                .ready(B, waits.iter().copied())
        };
        assert!(ready(
            &r,
            &[(B, 4), (3 << RGFX_CONTEXT_OBJ_ID_SHIFT | 1, 4)]
        ));
        assert!(!ready(&r, &[(A, 2)]), "A's buffer has not been taken");
        let _ = r.commit(A, 2, &ops(Some(&[]), &[]));
        assert!(!ready(&r, &[(A, 2)]), "nor finished");
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(ready(&r, &[(A, 2)]));
    }

    /// The records the rig's decoder posted, in order: each one's type and buffer.
    type Posted = Arc<Mutex<Vec<(u32, u64)>>>;

    /// Has the rig's decoder post its `CB_SUBMITTED`, `CB_SCHEDULED` and `CB_COMPLETE` records
    /// into one list, in the order posted.
    fn posting(r: &mut Rig) -> Posted {
        let posted: Posted = Arc::default();
        let sink = Arc::clone(&posted);
        r.dec.set_submitted_sink(Arc::new(move |m| {
            sink.lock().push((m.header.r#type, m.cb_obj_id))
        }));
        let sink = Arc::clone(&posted);
        r.dec.set_scheduled_sink(Arc::new(move |m| {
            sink.lock().push((m.header.r#type, m.cb_obj_id))
        }));
        let sink = Arc::clone(&posted);
        r.dec.set_completion_sink(Arc::new(move |m| {
            sink.lock().push((m.header.r#type, m.cb_obj_id))
        }));
        posted
    }

    /// A `SUBMITTED` op: the buffer asks for its `CB_SUBMITTED`.
    fn asks() -> Vec<u8> {
        let header = RgfxOpHeader {
            tag: RGFX_OP_SUBMITTED,
            flags: 0,
            len: 8,
        };
        header.as_bytes().to_vec()
    }

    /// The id the rig's [`message`] gives queue `queue`'s buffer at `seqno`.
    fn cb(queue: u64, seqno: u64) -> u64 {
        queue + 0x100 + seqno
    }

    /// Of `posted`, the records of the buffers `cbs`, in order.
    fn of(posted: &Posted, cbs: &[u64]) -> Vec<(u32, u64)> {
        (posted.lock().iter())
            .filter(|(_, c)| cbs.contains(c))
            .copied()
            .collect()
    }

    /// A buffer that waits on the GPU for a buffer of another queue holds back none of its
    /// records, nor its queue, as a Mac's order over IOSurfaces holds back none (Q16 of
    /// docs/graphics-kext/dgpu-oracle.md): with the platform holding every buffer (as the Vulkan
    /// platform holds a wait for a shared event's value), it gets its `CB_SUBMITTED` at its take
    /// and its `CB_SCHEDULED` at its commit, after those of the buffer committed before it on its
    /// queue, and so does the buffer after it; its GPU wait stays, and the completions keep their
    /// order. Once the platform takes them, nothing is posted twice, and a later buffer's records
    /// go at its scheduling again
    #[test]
    fn an_order_on_the_gpu_holds_back_no_record() {
        use reims_graphics::wire::{RGFX_EVT_CB_COMPLETE, RGFX_EVT_CB_SCHEDULED};
        const SUB: u32 = RGFX_EVT_CB_SUBMITTED;
        const SCHED: u32 = RGFX_EVT_CB_SCHEDULED;
        const DONE: u32 = RGFX_EVT_CB_COMPLETE;
        let mut r = rig();
        let posted = posting(&mut r);
        r.p.world().hold_submissions();
        let (p, x, o, c) = (cb(B, 1), cb(A, 5), cb(B, 2), cb(B, 3));

        let (outcome, _) = r.commit(B, 1, &ops(None, &asks()));
        assert_eq!(outcome, Ok(()));
        assert_eq!(
            of(&posted, &[p]),
            [(SUB, p)],
            "at its take; scheduled once taken"
        );
        let (_, _) = r.commit(A, 5, &ops(Some(&[]), &[]));
        assert!(of(&posted, &[x]).is_empty(), "A's: once taken");

        let (outcome, calls) = r.commit(B, 2, &ops(Some(&[(A, 5)]), &asks()));
        assert_eq!(outcome, Ok(()));
        assert_eq!(
            events(&calls, WAIT),
            [(r.event(A), 5)],
            "the GPU wait stays"
        );
        assert_eq!(
            of(&posted, &[p, o]),
            [(SUB, p), (SUB, o), (SCHED, p), (SCHED, o)],
            "its records at its take and its commit, its queue's earlier one's first"
        );
        let (held, line) = r
            .dec
            .queue_counts(B)
            .map(|c| (Arc::clone(&c.held), Arc::clone(&c.line)))
            .unwrap();
        assert_eq!(held.load(Ordering::Acquire), 0, "it holds no queue");
        let (_, _) = r.commit(B, 3, &ops(None, &asks()));
        assert_eq!(
            of(&posted, &[c]),
            [(SUB, c), (SCHED, c)],
            "the next one, behind it on the platform"
        );

        assert_eq!(r.p.world().release_submissions(), 4);
        assert_eq!(of(&posted, &[x]), [(SCHED, x)], "A's at its scheduling");
        r.p.world().complete_all(CompleteOn::Caller);
        assert_eq!(
            of(&posted, &[p, o, c]),
            [
                (SUB, p),
                (SUB, o),
                (SCHED, p),
                (SCHED, o),
                (SUB, c),
                (SCHED, c),
                (DONE, p),
                (DONE, o),
                (DONE, c)
            ],
            "posted once each, the completions in their order"
        );
        assert_eq!(line.ordered(), 0);

        // Nothing of B's waits for another queue now: a buffer's scheduling is the platform's.
        r.p.world().hold_submissions();
        let d = cb(B, 4);
        let (_, _) = r.commit(B, 4, &ops(Some(&[(A, 5)]), &asks()));
        assert_eq!(
            of(&posted, &[d]),
            [(SUB, d)],
            "A's finished: scheduled once taken"
        );
        r.p.world().release_submissions();
        assert_eq!(of(&posted, &[d]), [(SUB, d), (SCHED, d)]);
    }

    /// A buffer that waits for a guest's event still holds its queue as it did beside the surface
    /// order: it gets its `CB_SUBMITTED` and `CB_SCHEDULED` at its scheduling, and a buffer behind
    /// it that waits on the GPU for a buffer of another queue gets them after it, at its own
    /// scheduling, not at its commit
    #[test]
    fn an_event_wait_still_holds_its_queue_beside_an_order() {
        use reims_graphics::ops::{RGFX_OP_WAIT_EVENT, RgfxOpEvent};
        use reims_graphics::wire::{
            RGFX_CMD_SHARED_EVENT_CREATE, RGFX_EVT_CB_SCHEDULED, RgfxSharedEventCreate,
        };
        const SUB: u32 = RGFX_EVT_CB_SUBMITTED;
        const SCHED: u32 = RGFX_EVT_CB_SCHEDULED;
        let mut r = rig();
        let posted = posting(&mut r);
        let event = 2 << RGFX_CONTEXT_OBJ_ID_SHIFT | 7;
        let create = RgfxSharedEventCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_SHARED_EVENT_CREATE,
                size: size_of::<RgfxSharedEventCreate>() as u32,
                seqno: 0,
            },
            obj_id: event,
            initial_value: 0,
        };
        r.dec.set_context(Some(2));
        assert_eq!(r.dec.decode(&r.device, create.as_bytes()).unwrap(), Ok(()));
        r.p.world().hold_submissions();
        let (_, _) = r.commit(A, 5, &ops(Some(&[]), &[]));

        let wait = RgfxOpEvent {
            event_obj_id: event,
            value: 1,
        };
        let header = RgfxOpHeader {
            tag: RGFX_OP_WAIT_EVENT,
            flags: 0,
            len: (8 + size_of::<RgfxOpEvent>()) as u32,
        };
        let mut waits = header.as_bytes().to_vec();
        waits.extend_from_slice(wait.as_bytes());
        waits.extend(asks());
        let (h, o) = (cb(B, 1), cb(B, 2));
        let (outcome, _) = r.commit(B, 1, &ops(None, &waits));
        assert_eq!(outcome, Ok(()));
        let (held, line) = r
            .dec
            .queue_counts(B)
            .map(|c| (Arc::clone(&c.held), Arc::clone(&c.line)))
            .unwrap();
        assert_eq!(held.load(Ordering::Acquire), 1, "it holds its queue");
        let (outcome, calls) = r.commit(B, 2, &ops(Some(&[(A, 5)]), &asks()));
        assert_eq!(outcome, Ok(()));
        assert_eq!(events(&calls, WAIT), [(r.event(A), 5)]);
        assert!(
            of(&posted, &[h, o]).is_empty(),
            "both wait for the event's buffer"
        );

        r.p.world().release_submissions();
        assert_eq!(
            of(&posted, &[h, o]),
            [(SUB, h), (SCHED, h), (SUB, o), (SCHED, o)],
            "each at its scheduling, in their order"
        );
        assert_eq!(held.load(Ordering::Acquire), 0);
        assert_eq!(line.ordered(), 0);
    }

    /// A buffer answered on a doorbell's thread behind one that waits on the GPU for a buffer of
    /// another queue, answered between that buffer's take and its commit (as the queue's lane
    /// walks it), gets each record once and in its queue's order: its `CB_SUBMITTED` is the
    /// answer's, which no take or commit posts again, and its `CB_SCHEDULED` goes at its commit,
    /// after the ordered buffer's; once the platform takes both, nothing is posted again.
    #[test]
    fn a_doorbell_answer_behind_an_order_posts_each_record_once_in_order() {
        use reims_graphics::wire::{RGFX_EVT_CB_COMPLETE, RGFX_EVT_CB_SCHEDULED};
        const SUB: u32 = RGFX_EVT_CB_SUBMITTED;
        const SCHED: u32 = RGFX_EVT_CB_SCHEDULED;
        const DONE: u32 = RGFX_EVT_CB_COMPLETE;
        let mut r = rig();
        let posted = posting(&mut r);
        r.p.world().hold_submissions();
        let (_, _) = r.commit(A, 5, &ops(Some(&[]), &[]));
        let (o, d) = (cb(B, 2), cb(B, 3));

        // The ordered buffer is taken; its walk and commit are still to come.
        let ordered = message(B, 2, &ops(Some(&[(A, 5)]), &asks()));
        r.dec.set_context(Some(context(B)));
        let begun = r
            .dec
            .begin_taken(&r.device, &ordered, Some(B))
            .unwrap()
            .unwrap();
        assert_eq!(of(&posted, &[o]), [(SUB, o)], "at its take");

        // The next buffer, now at the head of the ring, ends in its ask: the doorbell answers it.
        let next = message(B, 3, &ops(None, &asks()));
        let sink = Arc::clone(&posted);
        let answered = r
            .dec
            .decoders()
            .answer_at_doorbell(context(B), &next, move |m| {
                sink.lock().push((m.header.r#type, m.cb_obj_id));
                true
            });
        assert!(answered, "no buffer before it holds its queue");
        assert_eq!(of(&posted, &[o, d]), [(SUB, o), (SUB, d)]);

        let finished = r
            .dec
            .finish_command_buffer(&r.device, &ordered, begun)
            .unwrap();
        assert_eq!(finished, Ok(()));
        assert_eq!(
            of(&posted, &[o, d]),
            [(SUB, o), (SUB, d), (SCHED, o)],
            "the ordered buffer's at its commit"
        );
        let begun = r
            .dec
            .begin_taken(&r.device, &next, Some(B))
            .unwrap()
            .unwrap();
        let finished = r
            .dec
            .finish_command_buffer(&r.device, &next, begun)
            .unwrap();
        assert_eq!(finished, Ok(()));
        assert_eq!(
            of(&posted, &[o, d]),
            [(SUB, o), (SUB, d), (SCHED, o), (SCHED, d)],
            "the answered one's at its commit, behind the ordered one's, its answer not again"
        );

        assert_eq!(r.p.world().release_submissions(), 3);
        r.p.world().complete_all(CompleteOn::Caller);
        assert_eq!(
            of(&posted, &[o, d]),
            [
                (SUB, o),
                (SUB, d),
                (SCHED, o),
                (SCHED, d),
                (DONE, o),
                (DONE, d)
            ],
            "posted once each"
        );
    }
}
