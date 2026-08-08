//! How a phase census accumulates time, and the defect that made all three of
//! them unable to report a cost smaller than a microsecond.
//!
//! # The defect
//!
//! Every phase census here reports its columns in microseconds, and each of them
//! accumulated with `Duration::as_micros`. That **truncates**. A span taking
//! 700 ns contributes exactly `0`, so a part whose per-item cost is
//! sub-microsecond reads zero however many times it runs — and these censuses
//! divide their columns by populations in the tens of thousands per second,
//! which is precisely the regime where per-item costs are sub-microsecond and
//! the *totals* are not.
//!
//! The bias is not small and it is not random. It is a per-span floor of up to
//! 1 µs applied to every span, so the *cheap, frequent* parts vanish entirely
//! while the expensive ones are barely touched — which reads exactly like "the
//! expensive part is the whole cost", the conclusion a split exists to test.
//!
//! [`crate::runtime::bind_phase`] shows the shape directly. Its `attrs_us`
//! column read `0`, `3`, `23`, `10`, `0`, `0` on consecutive driven windows
//! against a `vertex_us` in the thousands: a column reporting the fraction of
//! attribute walks that happened to cross a microsecond boundary, not what the
//! walk costs.
//!
//! # The rule
//!
//! **Accumulate nanoseconds; divide to microseconds once, at the window
//! boundary.** A `u64` of nanoseconds is 584 years, so nothing here can
//! overflow, and one truncation per second-long window is a rounding error
//! rather than a per-item floor. The emitted field names stay in microseconds —
//! every reader of the fail log parses them that way.
//!
//! A conclusion taken with the truncating accumulator is not evidence. The
//! attribute walk reading a flat zero is the one this crate had recorded, and
//! it needs re-reading before it is cited again.

use std::time::Duration;

/// The nanoseconds to charge for one span.
///
/// Saturating rather than wrapping: a `Duration` longer than 584 years cannot
/// arise from a span here, and if one somehow did, a saturated maximum is a
/// visibly absurd number where a wrapped one is a plausible small one.
#[inline]
pub fn charge_ns(d: Duration) -> u64 {
    u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)
}

/// A nanosecond accumulator rendered as the microseconds the log field names.
#[inline]
pub fn to_us(ns: u64) -> u64 {
    ns / 1_000
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point: a sub-microsecond span is worth something. Under
    /// `as_micros` every one of these is zero.
    #[test]
    fn a_sub_microsecond_span_is_not_zero() {
        assert_eq!(charge_ns(Duration::from_nanos(700)), 700);
        assert_eq!(Duration::from_nanos(700).as_micros(), 0);
    }

    /// Many sub-microsecond spans sum to a microsecond count that is not zero,
    /// which is the reading a census divides by its population.
    #[test]
    fn many_sub_microsecond_spans_sum_to_a_real_figure() {
        let ns: u64 = (0..20_000)
            .map(|_| charge_ns(Duration::from_nanos(700)))
            .sum();
        assert_eq!(to_us(ns), 14_000);
    }

    /// The window boundary rounds down once, and only once.
    #[test]
    fn the_window_rounds_down_at_the_boundary_only() {
        assert_eq!(to_us(1_999), 1);
        assert_eq!(to_us(2_000), 2);
    }
}
