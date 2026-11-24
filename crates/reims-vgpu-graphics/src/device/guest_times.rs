//! The guest's times of each command buffer, where the guest's driver took
//! [`RGFX_FEATURE_GUEST_TIMES`](crate::wire::RGFX_FEATURE_GUEST_TIMES): a diagnostic. The host
//! sees a command buffer only once a queue's thread takes it off the queue's ring, so the time
//! before that shows as one span in which nothing is pending on the host. The bundle stamps each
//! buffer with the guest's clock as the buffer was made (`began`), as its `commit` was entered
//! (`committed`) and as its message went on its ring (`posted`), in an [`RGFX_OP_GUEST_TIMES`]
//! first in its op stream, and the kext keeps the guest's clock less the host's, with its bound,
//! in ring 0's header. The queue's thread adds the host's clock as it took the buffer (`taken`)
//! and as it came back to the ring after the buffer it took before it (`ring_free`), and splits
//! the span:
//! - `encode` = `committed - began`: the app's encode;
//! - `post_wait` = `posted - committed`: the commit's wait to post (the ring's room, the queue's
//!   buffers before it to be committed, the naming of its commit order);
//! - `queued` = `ring_free - posted`, where the host came back after the post: the buffer waited
//!   behind the queue's buffer before it, on the host;
//! - `transport` = `taken` less the later of `posted` and `ring_free`: the doorbell, the thread's
//!   wake and the ring.
//!
//! And the host's own `held`, from the take to the event: the buffer's wait at the gate for ring
//! 0's messages before it, and the read of a buffer that lies in a block, so a reader of the trace
//! places the take at the event's time less it.
//!
//! `posted` is mapped to the host's clock by the offset; while the guest holds no offset (its
//! bound 0) only `encode` and `post_wait` are given. The offset is off by at most its bound, half
//! a `HOST_TIME` read's round trip widened by the reading's age (microseconds), far under the
//! millisecond spans it splits, as the host's clock runs at the guest's clock's rate (`clock`):
//! the kext writes it only at a `VBLANK`'s reading, which may be seconds old. A span the clocks'
//! error would make negative is 0.
//!
//! Each buffer's spans go to the host's trace ([`Host::trace`](super::Host::trace)) as
//! `cbguest queue seqno encode post_wait queued transport bound held`, or
//! `cbguest queue seqno encode post_wait held` with no offset held, and a debug line of target
//! [`TARGET`] gives, at the first buffer a second or more after the last line, the buffers'
//! count, each span's sum and median and the longest transport. Neither changes anything the
//! device does.

use std::sync::atomic::Ordering;

use zerocopy::FromBytes;

use crate::ops::{
    RGFX_COMMAND_BUFFER_OPS_OFFSET, RGFX_OP_BODY_OFFSET, RGFX_OP_GUEST_TIMES, RgfxOpGuestTimes,
    RgfxOpHeader,
};
use crate::region::Region;
use crate::ring::{GUEST_CLOCK_BOUND, GUEST_CLOCK_OFFSET};
use crate::wire::RGFX_CONTROL_SUBMIT_OFFSET;

/// The log target of the once-a-second line.
pub(crate) const TARGET: &str = "reims_vgpu_graphics::guest_times";

/// How long the line's window is, in nanoseconds.
const WINDOW_NS: u64 = 1_000_000_000;

/// The guest's times a `COMMAND_BUFFER` message carries: its op stream's first op, where that is
/// an [`RGFX_OP_GUEST_TIMES`] of the body's size; `None` otherwise.
pub(super) fn of(message: &[u8]) -> Option<RgfxOpGuestTimes> {
    let ops = message.get(RGFX_COMMAND_BUFFER_OPS_OFFSET as usize..)?;
    let (header, rest) = RgfxOpHeader::read_from_prefix(ops).ok()?;
    let len = RGFX_OP_BODY_OFFSET as usize + size_of::<RgfxOpGuestTimes>();
    if header.tag != RGFX_OP_GUEST_TIMES || header.len as usize != len {
        return None;
    }
    RgfxOpGuestTimes::read_from_prefix(rest)
        .ok()
        .map(|(times, _)| times)
}

/// The guest's clock less the host's, and how far that may be off, as ring 0's header holds
/// them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Clock {
    /// The guest's clock less the host's, in nanoseconds.
    pub(crate) offset_ns: i64,
    /// How far `offset_ns` may be off, in nanoseconds; never 0.
    pub(crate) bound_ns: u32,
}

/// The offset ring 0's header holds: its bound read first, with acquire, as the kext stores it
/// after the offset with release; `None` while the bound is 0, the guest holding no reading.
pub(super) fn clock(region: &Region) -> Option<Clock> {
    let bound_ns = region
        .atomic_u32(RGFX_CONTROL_SUBMIT_OFFSET + GUEST_CLOCK_BOUND)
        .load(Ordering::Acquire);
    if bound_ns == 0 {
        return None;
    }
    let offset = region
        .atomic_u64(RGFX_CONTROL_SUBMIT_OFFSET + GUEST_CLOCK_OFFSET)
        .load(Ordering::Relaxed);
    Some(Clock {
        offset_ns: offset.cast_signed(),
        bound_ns,
    })
}

/// The guest's time `guest_ns` on the host's clock, by `clock`'s offset; 0 where that falls
/// before the host's clock's start.
pub(crate) fn on_host(guest_ns: u64, clock: Clock) -> u64 {
    let ns = i128::from(guest_ns) - i128::from(clock.offset_ns);
    u64::try_from(ns.max(0)).unwrap_or(u64::MAX)
}

/// A command buffer's spans before the host took it, in nanoseconds (the module doc).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Spans {
    /// `committed - began`.
    pub(crate) encode: u64,
    /// `posted - committed`.
    pub(crate) post_wait: u64,
    /// The spans on the host's clock, where the guest held an offset.
    pub(crate) host: Option<HostSpans>,
    /// From the take to the event, on the host's clock.
    pub(crate) held: u64,
}

/// The spans of [`Spans`] that map the guest's `posted` to the host's clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HostSpans {
    /// `ring_free - posted`, 0 where the host came back to the ring before the post.
    pub(crate) queued: u64,
    /// `taken` less the later of `posted` and `ring_free`.
    pub(crate) transport: u64,
    /// The offset's bound.
    pub(crate) bound: u32,
}

/// The spans of a buffer stamped `times`, taken at `taken_ns` (the host's clock) by a thread that
/// came back to the ring at `ring_free_ns` after its buffer before it (0 where it took none
/// before it in this pass), with the guest's offset `clock` where it holds one, its event at
/// `noted_ns`.
pub(crate) fn spans(
    times: &RgfxOpGuestTimes,
    clock: Option<Clock>,
    taken_ns: u64,
    ring_free_ns: u64,
    noted_ns: u64,
) -> Spans {
    let host = clock.map(|clock| {
        let posted = on_host(times.posted_ns, clock);
        HostSpans {
            queued: ring_free_ns.saturating_sub(posted),
            transport: taken_ns.saturating_sub(posted.max(ring_free_ns)),
            bound: clock.bound_ns,
        }
    });
    Spans {
        encode: times.committed_ns.saturating_sub(times.began_ns),
        post_wait: times.posted_ns.saturating_sub(times.committed_ns),
        host,
        held: noted_ns.saturating_sub(taken_ns),
    }
}

impl Spans {
    /// The values of the buffer's `cbguest` trace event: `queue seqno encode post_wait`, then
    /// `queued transport bound` where the guest held an offset, then `held`.
    pub(crate) fn event(&self, queue: u64, seqno: u64) -> Vec<u64> {
        let mut values = vec![queue, seqno, self.encode, self.post_wait];
        if let Some(host) = self.host {
            values.extend([host.queued, host.transport, u64::from(host.bound)]);
        }
        values.push(self.held);
        values
    }
}

/// The buffers' spans since the last line.
#[derive(Debug, Default)]
pub(super) struct Window {
    /// When the window opened, 0 before its first buffer.
    since_ns: u64,
    encode: Vec<u64>,
    post_wait: Vec<u64>,
    queued: Vec<u64>,
    transport: Vec<u64>,
    /// The largest offset bound among them.
    bound: u32,
}

impl Window {
    /// Adds a buffer's `spans`, at `now_ns`: the line, where the window opened a second or more
    /// before, which then starts afresh.
    pub(super) fn add(&mut self, spans: &Spans, now_ns: u64) -> Option<String> {
        if self.since_ns == 0 {
            self.since_ns = now_ns.max(1);
        }
        self.encode.push(spans.encode);
        self.post_wait.push(spans.post_wait);
        if let Some(host) = spans.host {
            self.queued.push(host.queued);
            self.transport.push(host.transport);
            self.bound = self.bound.max(host.bound);
        }
        let span = now_ns.saturating_sub(self.since_ns);
        if span < WINDOW_NS {
            return None;
        }
        let line = self.line(span);
        *self = Window {
            since_ns: now_ns.max(1),
            ..Window::default()
        };
        Some(line)
    }

    /// The line of the window, `span_ns` long.
    fn line(&mut self, span_ns: u64) -> String {
        let longest = self.transport.iter().copied().max().unwrap_or(0);
        let mut line = format!(
            "{} command buffers in {:.2} s: encode {}, post wait {}",
            self.encode.len(),
            span_ns as f64 / 1e9,
            sum_and_median(&mut self.encode),
            sum_and_median(&mut self.post_wait),
        );
        if self.transport.is_empty() {
            line.push_str("; the guest holds no clock offset: no queued or transport");
        } else {
            line.push_str(&format!(
                "; of {} with the clocks' offset (within {} us): queued {}, transport {}, the \
                 longest transport {} us",
                self.transport.len(),
                self.bound.div_ceil(1000),
                sum_and_median(&mut self.queued),
                sum_and_median(&mut self.transport),
                longest / 1000
            ));
        }
        line
    }
}

/// `values`' sum and median as the line gives them, in microseconds.
fn sum_and_median(values: &mut [u64]) -> String {
    let sum: u64 = values.iter().sum();
    let median = if values.is_empty() {
        0
    } else {
        let mid = values.len() / 2;
        *values.select_nth_unstable(mid).1
    };
    format!("{} us (median {} us)", sum / 1000, median / 1000)
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerocopy::IntoBytes;

    use crate::ops::{RgfxCommandBuffer, RgfxOpHeader};

    fn times() -> RgfxOpGuestTimes {
        RgfxOpGuestTimes {
            began_ns: 1_000_000,
            committed_ns: 3_000_000,
            posted_ns: 3_500_000,
        }
    }

    /// A buffer's spans: the guest's own from its stamps, and the host's from its stamps mapped
    /// by the offset, `queued` behind the queue's buffer before it where the thread came back to
    /// the ring after the post, and `transport` from the later of the two.
    #[test]
    fn a_buffer_s_spans_split_the_time_before_its_take() {
        // The guest's clock runs 1 ms ahead of the host's: posted at the host's 2.5 ms.
        let clock = Clock {
            offset_ns: 1_000_000,
            bound_ns: 4_000,
        };
        // Taken at 3 ms by a thread back at the ring before the post: all of it transport.
        let s = spans(&times(), Some(clock), 3_000_000, 2_000_000, 3_000_000);
        assert_eq!((s.encode, s.post_wait), (2_000_000, 500_000));
        assert_eq!(
            s.host,
            Some(HostSpans {
                queued: 0,
                transport: 500_000,
                bound: 4_000
            })
        );
        // Back at the ring at 2.8 ms, after the post: 0.3 ms behind the buffer before it.
        let s = spans(&times(), Some(clock), 3_000_000, 2_800_000, 3_100_000);
        assert_eq!(
            s.host,
            Some(HostSpans {
                queued: 300_000,
                transport: 200_000,
                bound: 4_000
            })
        );
        // Its event 0.1 ms after its take: held at the gate that long.
        assert_eq!(s.held, 100_000);
        // An offset behind the guest's clock maps forward.
        let s = spans(
            &times(),
            Some(Clock {
                offset_ns: -1_000_000,
                bound_ns: 1,
            }),
            5_000_000,
            0,
            5_000_000,
        );
        assert_eq!(s.host.map(|h| h.transport), Some(500_000));
        // The clocks' error past a span makes it 0, not negative.
        let s = spans(&times(), Some(clock), 2_400_000, 0, 2_400_000);
        assert_eq!(s.host.map(|h| h.transport), Some(0));
    }

    /// With no offset held the spans are the guest's alone, and the event carries no transport.
    #[test]
    fn with_no_offset_only_the_guest_s_spans_are_given() {
        let s = spans(&times(), None, 3_000_000, 0, 3_000_040);
        assert_eq!(s.host, None);
        assert_eq!(s.event(7, 9), vec![7, 9, 2_000_000, 500_000, 40]);
        let s = spans(
            &times(),
            Some(Clock {
                offset_ns: 0,
                bound_ns: 2_000,
            }),
            4_000_000,
            0,
            4_000_000,
        );
        assert_eq!(
            s.event(7, 9),
            vec![7, 9, 2_000_000, 500_000, 0, 500_000, 2_000, 0]
        );
    }

    /// The times are read only from an op stream whose first op is a whole `GUEST_TIMES`.
    #[test]
    fn the_times_are_the_stream_s_first_op_alone() {
        let message = |tag: u16, len: u32, after: &[u8]| {
            let mut m = RgfxCommandBuffer::default().as_bytes().to_vec();
            m.extend_from_slice(RgfxOpHeader { tag, flags: 0, len }.as_bytes());
            m.extend_from_slice(times().as_bytes());
            m.extend_from_slice(after);
            m
        };
        let end = RgfxOpHeader {
            tag: crate::ops::RGFX_OP_ENCODER_END,
            flags: 0,
            len: 8,
        };
        assert_eq!(
            of(&message(RGFX_OP_GUEST_TIMES, 32, end.as_bytes())),
            Some(times())
        );
        assert_eq!(of(&message(RGFX_OP_GUEST_TIMES, 24, &[])), None);
        assert_eq!(of(&message(crate::ops::RGFX_OP_SURFACE_USE, 32, &[])), None);
        let bare = RgfxCommandBuffer::default();
        assert_eq!(of(bare.as_bytes()), None);
        let short = message(RGFX_OP_GUEST_TIMES, 32, &[]);
        assert_eq!(of(&short[..short.len() - 8]), None);
        // The position the bundle stamps `posted_ns` at.
        let m = message(RGFX_OP_GUEST_TIMES, 32, &[]);
        let at = crate::ops::RGFX_GUEST_TIMES_POSTED_OFFSET as usize;
        assert_eq!(m[at..at + 8], times().posted_ns.to_le_bytes());
    }

    /// The window gives its line at the first buffer a second after it opened, with each span's
    /// sum and median, then starts afresh.
    #[test]
    fn the_line_comes_once_a_second() {
        let mut w = Window::default();
        let with = |transport: u64| Spans {
            encode: 2_000_000,
            post_wait: 1_000,
            host: Some(HostSpans {
                queued: 0,
                transport,
                bound: 1_500,
            }),
            held: 0,
        };
        assert_eq!(w.add(&with(10_000), 5), None);
        assert_eq!(w.add(&with(30_000), 500_000_000), None);
        let line = w.add(&with(20_000), 1_000_000_005).expect("a second on");
        assert_eq!(
            line,
            "3 command buffers in 1.00 s: encode 6000 us (median 2000 us), post wait 3 us \
             (median 1 us); of 3 with the clocks' offset (within 2 us): queued 0 us (median 0 \
             us), transport 60 us (median 20 us), the longest transport 30 us"
        );
        assert_eq!(w.add(&with(10_000), 1_500_000_000), None);
        let guest_only = Spans {
            host: None,
            ..with(0)
        };
        let line = w.add(&guest_only, 2_100_000_000).expect("a second on");
        assert!(line.starts_with("2 command buffers in 1.10 s"), "{line}");
    }
}
