//! A command buffer committed in parts as it is walked, so the GPU starts on its first encoders
//! while the host walks the rest (docs/graphics-kext/design.md, "A command buffer in parts").
//!
//! Where the platform starts a command buffer's GPU work only at its commit (the Vulkan
//! platform's runtime records it and submits it then), a long buffer's walk and its GPU work run
//! in series: WindowServer's composite of a Liquid Glass desktop took a walk of 2.6 to 3.5 ms and
//! then 1.6 to 3.3 ms of GPU work a frame on an RTX 5080 Laptop GPU (measured),
//! against a refresh of 8.33 ms. So at an encoder's end, once the part walked so far took
//! [`SPLIT_AFTER`], the walk hands the platform that part (`CommandBuffer::split`), which commits
//! it on the buffer's queue and answers a command buffer of the queue that goes on from it, and
//! the walk goes on in that one. The platform keeps a buffer whole where a part would answer
//! otherwise than the whole (its docs list where).
//!
//! **What it gains** (measured on the M3 Max under macOS 27.0.1, the Vulkan platform on
//! MoltenVK, release, by `composite_cost`: a buffer of 60 render passes of 3 blended quads each,
//! frames one at a time, 200 a run, two runs). From the take of a frame's message to its
//! `CB_COMPLETE`, the median fell from 9.5-9.7 ms whole to 6.0 ms in parts over a 3840 x 2400
//! target, every 4th pass reading it; from 8.5-8.8 to 4.7 ms over 1920 x 1200; and from 10.5-10.6
//! to 7.0 ms over 1920 x 1200 with every pass clearing its target and reading it first. More
//! than the walk itself (1.3-1.5 ms) came off: the runtime's own work at a commit, its recording
//! and submission of the buffer's batches, runs beside the walk too. The walk took 1.5% more
//! instructions. Parts walked [`SPLIT_AFTER`] at least were the fastest of 100, 250 and 500 µs in
//! each shape and run: 100 µs cost the walk 5% more instructions and gained nothing, 500 µs left
//! 0.2-0.4 ms on the table.
//!
//! **What stays as it was.** The parts run in their order on the queue, as the encoders of one
//! buffer do, and each part is committed only where the whole buffer could be committed then,
//! waiting for nothing (`GraphicsDecoder::may_commit_now`): its turn on its queue is next, every
//! ticket and range message before its begin is done, and the buffers its `COMMIT_ORDER` names
//! are committed. Every wait of the last commit was met at the first part, so the walk never
//! waits between parts. Its handlers, records and tickets are the last part's, and come as they
//! came: its `CB_SUBMITTED`, `CB_SCHEDULED` and `CB_COMPLETE` once each, its ticket given back
//! at the last commit, its commit marks raised there. Its outcome is all the parts' ([`Parts`]):
//! the first failed part's error, else the last part's; and GPU times that start at the first
//! part's start and span the parts' GPU time summed, never ending after the last part's end
//! (below, "Its GPU times"). A signal of a shared event stays in the part it was encoded in, before the
//! cut, and its value goes to the guest as the event reaches it, as a whole buffer's does; one
//! of a shared event whose create the host refused goes with the buffer's completion, after its
//! last part. A buffer whose walk faults after a part has its failure at its stand-in's
//! completion, as any buffer whose walk faults; the parts committed before the fault ran, as the
//! encoders before a fault of Metal's GPU run.
//!
//! **Its GPU times.** Each part runs once the walk has handed it over, so the GPU waits between
//! parts for the walk of the next: from the first part's start to the last part's end the buffer
//! spans the host's walk of it, which a Mac, whose GPU is handed the whole buffer at its commit,
//! never waits for. The guest's compositor reads these times: macOS's SkyLight and QuartzCore,
//! which composite WindowServer's frames, reference `GPUStartTime` and `GPUEndTime` (read from
//! their selector references on macOS 27), and the bundle answers them with the record's span
//! (`rgfxNoteGPUTimes:`). On a Mac one buffer spans its encoders' GPU time summed: on both x86
//! Macs (macOS 26.7.1, an AMD Radeon Pro 5300M and an Intel UHD Graphics 630), a buffer of 8 or
//! 32 compute encoders or render passes spanned what the same encoders committed one buffer each
//! summed (AMD: 32 compute encoders 6.31 ms whole, 7.37 summed; 32 passes 11.04 and 10.89; Intel:
//! 13.05 and 12.41; 59.56 and 59.99), the CPU's time encoding it never counted, and its
//! `GPUStartTime` came 0.08 to 0.40 ms after its commit (measured, `oracle_cmdbuf DEVICE
//! gputimes`; docs/graphics-kext/dgpu-oracle.md, Q7). So the record starts at the first part's
//! start, which the GPU reached soonest after the commit, and spans the parts' GPU time summed,
//! each part's end less its start: the time the buffer's work kept the GPU, as a buffer committed
//! whole answers it. It never ends after the last part's end, where parts that overlap on the GPU
//! sum to more.
//!
//! **What another message sees** ([`SplitGate`]). While a buffer of a context has parts out, from
//! its first part's commit to its last, no other command buffer of the context is committed, and
//! the context's range messages that come are carried out after its last commit: each other
//! message comes wholly before the buffer's parts or wholly after them, as it would before a
//! whole buffer's commit or after it ("What a walk sees"). A buffer splits only while no other
//! commit or range message of its context is under way; the others' commits wait at their
//! commit, never the walk, and ring 0 never waits.
//!
//! **Which buffers.** One of a context, whose walk is not ordered on its queue (a video buffer's
//! is), that waits for no event and is behind no buffer of its queue that does ([`super::cmdbuf`]'s
//! `Held`), and whose queue holds room for another command buffer, so the platform's make of the
//! next part waits for none.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use reims_vgpu_core::platform::command::{CompletedHandler, CompletedInfo};
use reims_vgpu_core::platform::types::{CommandBufferStatus, NsErrorText};
use reims_vgpu_core::sync::{Condvar, Mutex};

use super::sequencer::RangeOp;

/// How long a part of a command buffer is walked, at least, before the walk commits it at an
/// encoder's end: each part costs its commit and the next part's make on the walk's thread, and
/// the GPU's wait for the part before it at their boundary. The fastest of those measured (the
/// module docs, "What it gains").
pub(crate) const SPLIT_AFTER: Duration = Duration::from_micros(250);

/// The contexts with a buffer whose parts are out, and the commits and range messages of each
/// context under way (the module docs).
#[derive(Default)]
pub(crate) struct SplitGate {
    contexts: Mutex<HashMap<u32, Gate>>,
    changed: Condvar,
}

/// One context's gate.
#[derive(Default)]
struct Gate {
    /// A buffer of the context has parts out.
    split: bool,
    /// The context's commits and range messages under way.
    under_way: usize,
    /// The range messages that came while `split`, carried out at its end in the order they
    /// came.
    deferred: Vec<RangeOp>,
    /// The commits waiting for `split`'s end.
    waiting: usize,
}

impl SplitGate {
    /// A command buffer of `context` is to be committed: waits, with no lock of the caller's
    /// held, while a buffer of the context has parts out, and holds a split of the context off
    /// until the answer drops, after the commit.
    pub(crate) fn commit(self: &Arc<Self>, context: u32) -> UnderWay {
        let mut contexts = self.contexts.lock();
        if let Some(g) = contexts.get_mut(&context).filter(|g| g.split) {
            g.waiting += 1;
            while contexts.get(&context).is_some_and(|g| g.split) {
                self.changed.wait(&mut contexts);
            }
            contexts.entry(context).or_default().waiting -= 1;
        }
        contexts.entry(context).or_default().under_way += 1;
        UnderWay {
            gate: Arc::clone(self),
            context,
        }
    }

    /// Carries out `op`, a range message of `context`'s: now, holding a split of the context off
    /// meanwhile, unless a buffer of the context has parts out, after whose last commit it is
    /// carried out then, on that thread. It waits for nothing.
    pub(crate) fn range(self: &Arc<Self>, context: u32, op: RangeOp) {
        {
            let mut contexts = self.contexts.lock();
            let gate = contexts.entry(context).or_default();
            if gate.split {
                gate.deferred.push(op);
                return;
            }
            gate.under_way += 1;
        }
        let under_way = UnderWay {
            gate: Arc::clone(self),
            context,
        };
        op();
        drop(under_way);
    }

    /// The gate of `context` for a buffer's parts, where no buffer of the context has parts out
    /// and no commit or range message of it is under way; `None` otherwise.
    pub(crate) fn try_split(self: &Arc<Self>, context: u32) -> Option<SplitHold> {
        let mut contexts = self.contexts.lock();
        let gate = contexts.entry(context).or_default();
        if gate.split || gate.under_way > 0 {
            return None;
        }
        gate.split = true;
        Some(SplitHold {
            gate: Arc::clone(self),
            context,
        })
    }

    /// The gate of `context` is free and holds nothing, so its entry goes.
    fn prune(contexts: &mut HashMap<u32, Gate>, context: u32) {
        if contexts.get(&context).is_some_and(|g| {
            !g.split && g.under_way == 0 && g.deferred.is_empty() && g.waiting == 0
        }) {
            contexts.remove(&context);
        }
    }

    /// How many commits of `context` wait for a buffer's parts (for the tests).
    #[cfg(test)]
    pub(crate) fn waiting(&self, context: u32) -> usize {
        self.contexts.lock().get(&context).map_or(0, |g| g.waiting)
    }
}

/// A commit or range message of a context under way ([`SplitGate::commit`],
/// [`SplitGate::range`]), which holds a split of the context off until it drops.
pub(crate) struct UnderWay {
    gate: Arc<SplitGate>,
    context: u32,
}

impl Drop for UnderWay {
    fn drop(&mut self) {
        let mut contexts = self.gate.contexts.lock();
        if let Some(g) = contexts.get_mut(&self.context) {
            g.under_way -= 1;
        }
        SplitGate::prune(&mut contexts, self.context);
    }
}

/// A buffer's hold of its context's gate while its parts are out ([`SplitGate::try_split`]):
/// dropped after its last commit, or where its walk stops, it carries out the range messages
/// that came meanwhile, in the order they came, and then lets the context's commits go.
pub(crate) struct SplitHold {
    gate: Arc<SplitGate>,
    context: u32,
}

impl Drop for SplitHold {
    fn drop(&mut self) {
        loop {
            let deferred = {
                let mut contexts = self.gate.contexts.lock();
                let Some(g) = contexts.get_mut(&self.context) else {
                    return;
                };
                if g.deferred.is_empty() {
                    g.split = false;
                    SplitGate::prune(&mut contexts, self.context);
                    self.gate.changed.notify_all();
                    return;
                }
                std::mem::take(&mut g.deferred)
            };
            for op in deferred {
                op();
            }
        }
    }
}

/// What the completions of a command buffer's parts before its last found, which the last
/// part's completion makes the buffer's ([`Parts::merged`]).
#[derive(Default)]
pub(crate) struct Parts {
    seen: Mutex<Seen>,
}

#[derive(Default)]
struct Seen {
    /// The GPU start of the first part that has GPU times.
    gpu_start: Option<u64>,
    /// The GPU time of the parts seen that have GPU times, summed: each one's end less its start.
    gpu_busy: u64,
    /// The error of the first part that failed.
    error: Option<NsErrorText>,
}

impl Parts {
    /// The completion handler of a part before the last: it notes what the part's completion
    /// found, then lets `keep` go (the part's place among its queue's command buffers).
    pub(crate) fn part_handler<K: Send + 'static>(self: &Arc<Self>, keep: K) -> CompletedHandler {
        let parts = Arc::clone(self);
        Box::new(move |info: CompletedInfo| {
            let mut seen = parts.seen.lock();
            if let Some((start, end)) = info.gpu_times_ns {
                seen.gpu_start.get_or_insert(start);
                seen.gpu_busy = seen.gpu_busy.saturating_add(end.saturating_sub(start));
            }
            if seen.error.is_none() {
                seen.error = info.error;
            }
            drop(seen);
            drop(keep);
        })
    }

    /// `handler`, the last part's, handed the whole buffer's outcome: the first failed part's
    /// error, and status, where a part before the last failed, else the last part's; and GPU
    /// times that start at the first part's start and span every part's GPU time summed, ending no
    /// later than the last part's end (the module docs, "Its GPU times"). The platform runs
    /// a queue's completion handlers in its commit order, so every part before the last has been
    /// seen.
    pub(crate) fn merged(self: Arc<Self>, handler: CompletedHandler) -> CompletedHandler {
        Box::new(move |mut info: CompletedInfo| {
            let seen = std::mem::take(&mut *self.seen.lock());
            if let Some(error) = seen.error {
                info.error = Some(error);
                info.status = CommandBufferStatus::ERROR;
            }
            if let (Some(first), Some((start, end))) = (seen.gpu_start, info.gpu_times_ns) {
                let busy = seen.gpu_busy.saturating_add(end.saturating_sub(start));
                let first = first.min(end);
                info.gpu_times_ns = Some((first, first.saturating_add(busy).min(end)));
            }
            handler(info);
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::*;

    /// A commit of a context waits while a buffer of it has parts out, and a commit of another
    /// context does not.
    #[test]
    fn a_commit_waits_for_its_context_s_parts_alone() {
        let gate = Arc::new(SplitGate::default());
        let hold = gate.try_split(1).expect("the gate is free");
        drop(gate.commit(2));
        let went = AtomicBool::new(false);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                let _commit = gate.commit(1);
                went.store(true, Ordering::Release);
            });
            std::thread::sleep(Duration::from_millis(50));
            assert!(!went.load(Ordering::Acquire), "committed between the parts");
            drop(hold);
        });
        assert!(went.load(Ordering::Acquire));
    }

    /// No buffer splits while another commit or range message of its context is under way, or
    /// another buffer of it has parts out.
    #[test]
    fn a_split_waits_for_nothing_and_takes_a_free_gate_alone() {
        let gate = Arc::new(SplitGate::default());
        let commit = gate.commit(1);
        assert!(gate.try_split(1).is_none(), "split beside a commit");
        drop(commit);
        let hold = gate.try_split(1).expect("the gate is free");
        assert!(
            gate.try_split(1).is_none(),
            "two buffers' parts out at once"
        );
        assert!(gate.try_split(2).is_some(), "another context's gate");
        drop(hold);
        assert!(gate.try_split(1).is_some());
        assert!(gate.contexts.lock().is_empty(), "a free gate's entry stays");
    }

    /// A range message that comes while a buffer's parts are out is carried out at the last
    /// part's commit, in the order they came, before the context's other commits go; one that
    /// comes with none out is carried out at once.
    #[test]
    fn a_range_message_comes_wholly_before_or_after_a_buffer_s_parts() {
        let gate = Arc::new(SplitGate::default());
        let done = Arc::new(Mutex::new(Vec::new()));
        let range = |n: u32| {
            let done = Arc::clone(&done);
            Box::new(move || done.lock().push(n)) as RangeOp
        };
        gate.range(1, range(1));
        assert_eq!(*done.lock(), [1]);
        let hold = gate.try_split(1).unwrap();
        gate.range(1, range(2));
        gate.range(1, range(3));
        gate.range(2, range(4));
        assert_eq!(*done.lock(), [1, 4], "carried out while the parts are out");
        std::thread::scope(|scope| {
            scope.spawn(|| {
                let _commit = gate.commit(1);
                done.lock().push(5);
            });
            std::thread::sleep(Duration::from_millis(50));
            drop(hold);
        });
        assert_eq!(*done.lock(), [1, 4, 2, 3, 5]);
    }

    fn error(code: i64) -> NsErrorText {
        NsErrorText {
            domain: "MTLCommandBufferErrorDomain".to_owned(),
            code,
            description: String::new(),
        }
    }

    /// The whole buffer's outcome: the first failed part's error, and GPU times that start at the
    /// first part's start and span the parts' GPU time summed, no later than the last part's end:
    /// the gaps between parts, where the GPU waited for the walk, are not the buffer's.
    #[test]
    fn a_buffer_s_outcome_is_all_its_parts() {
        let outcome = |parts: [CompletedInfo; 3]| {
            let p = Arc::new(Parts::default());
            let [first, second, last] = parts;
            p.part_handler(())(first);
            p.part_handler(())(second);
            let got = Arc::new(Mutex::new(None));
            let into = Arc::clone(&got);
            Arc::clone(&p).merged(Box::new(move |info| *into.lock() = Some(info)))(last);
            got.lock().take().unwrap()
        };
        let ok = |start, end| CompletedInfo {
            status: CommandBufferStatus::COMPLETED,
            error: None,
            gpu_times_ns: Some((start, end)),
        };
        let failed = |start, end, code| CompletedInfo {
            status: CommandBufferStatus::ERROR,
            error: Some(error(code)),
            gpu_times_ns: Some((start, end)),
        };
        assert_eq!(
            outcome([ok(10, 20), ok(25, 40), ok(45, 60)]),
            ok(10, 50),
            "10 + 15 + 15 of GPU time, from the first part's start"
        );
        assert_eq!(
            outcome([ok(10, 20), failed(25, 40, 8), failed(45, 60, 3)]),
            failed(10, 50, 8),
            "the first failed part's error"
        );
        assert_eq!(
            outcome([ok(10, 20), ok(25, 40), failed(45, 60, 3)]),
            failed(10, 50, 3)
        );
        assert_eq!(
            outcome([ok(10, 30), ok(20, 45), ok(40, 60)]),
            ok(10, 60),
            "parts that overlap end no later than the last part"
        );
        let untimed = CompletedInfo {
            status: CommandBufferStatus::COMPLETED,
            error: None,
            gpu_times_ns: None,
        };
        assert_eq!(
            outcome([untimed.clone(), ok(30, 40), ok(50, 60)]),
            ok(30, 50),
            "a part with no GPU times adds none"
        );
    }
}
