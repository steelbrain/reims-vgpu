//! Ordered lifetime points for native objects shared by recording encoders.
//!
//! A native handle removed from a shared registry can still be named by work
//! that began before the removal.  Recording therefore reserves a point before
//! it reads shared state.  Submitted work retires that point after its fence;
//! abandoned recording cancels it.  Since queue commit follows point order, a
//! handle captured at point N is safe when every point through N is terminal.

use parking_lot::Mutex;
use std::collections::BTreeSet;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct RecordingPoint(u64);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RecordingSequenceExhausted;

#[derive(Default)]
struct Order {
    latest: u64,
    retired_through: u64,
    terminal: BTreeSet<u64>,
}

impl Order {
    fn reserve(&mut self) -> Result<RecordingPoint, RecordingSequenceExhausted> {
        self.latest = self
            .latest
            .checked_add(1)
            .ok_or(RecordingSequenceExhausted)?;
        Ok(RecordingPoint(self.latest))
    }

    fn terminal(&mut self, point: RecordingPoint) {
        assert!(
            point.0 <= self.latest,
            "a recording point must be reserved before it becomes terminal"
        );
        assert!(
            point.0 > self.retired_through && self.terminal.insert(point.0),
            "a recording point can become terminal only once"
        );
        while let Some(next) = self.retired_through.checked_add(1) {
            if !self.terminal.remove(&next) {
                break;
            }
            self.retired_through = next;
        }
    }
}

/// Session-wide order shared by every encoder that records against one set of
/// native registries.
#[derive(Default)]
pub(crate) struct RetirementOrder {
    order: Mutex<Order>,
}

impl RetirementOrder {
    pub(crate) fn reserve(self: &Arc<Self>) -> Result<RecordingLease, RecordingSequenceExhausted> {
        let point = self.order.lock().reserve()?;
        Ok(RecordingLease {
            order: Arc::clone(self),
            point: Some(point),
        })
    }

    pub(crate) fn latest(&self) -> Option<RecordingPoint> {
        let latest = self.order.lock().latest;
        (latest != 0).then_some(RecordingPoint(latest))
    }

    pub(crate) fn retired(&self, point: RecordingPoint) -> bool {
        point.0 <= self.order.lock().retired_through
    }

    fn terminal(&self, point: RecordingPoint) {
        self.order.lock().terminal(point);
    }
}

/// A command buffer that may already name shared objects but has not reached
/// the queue.  Dropping it is cancellation, so an error return cannot strand a
/// hole in the retirement order.
pub(crate) struct RecordingLease {
    order: Arc<RetirementOrder>,
    point: Option<RecordingPoint>,
}

impl RecordingLease {
    pub(crate) fn submitted(mut self) -> SubmittedPoint {
        SubmittedPoint {
            order: Arc::clone(&self.order),
            point: self.point.take().expect("recording lease already consumed"),
        }
    }
}

impl Drop for RecordingLease {
    fn drop(&mut self) {
        if let Some(point) = self.point.take() {
            self.order.terminal(point);
        }
    }
}

/// A recording point owned by exactly one submitted command buffer.  Unlike a
/// recording lease, dropping this does not claim GPU completion; its fence must
/// explicitly retire it.
#[must_use = "a submitted recording point must retire with its fence"]
pub(crate) struct SubmittedPoint {
    order: Arc<RetirementOrder>,
    point: RecordingPoint,
}

impl SubmittedPoint {
    pub(crate) fn retire(self) {
        self.order.terminal(self.point);
    }

    #[cfg(test)]
    fn point(&self) -> RecordingPoint {
        self.point
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_advances_only_over_a_contiguous_prefix() {
        let order = Arc::new(RetirementOrder::default());
        let first = order.reserve().unwrap().submitted();
        let second = order.reserve().unwrap().submitted();
        let captured = order.latest().unwrap();

        second.retire();
        assert!(
            !order.retired(captured),
            "the earlier recording is still live"
        );
        first.retire();
        assert!(order.retired(captured));
    }

    #[test]
    fn abandoned_recording_cancels_its_point() {
        let order = Arc::new(RetirementOrder::default());
        let lease = order.reserve().unwrap();
        let point = order.latest().unwrap();

        drop(lease);
        assert!(order.retired(point));
    }

    #[test]
    fn later_recording_does_not_extend_an_earlier_capture() {
        let order = Arc::new(RetirementOrder::default());
        let first = order.reserve().unwrap().submitted();
        let captured = first.point();
        let later = order.reserve().unwrap().submitted();

        first.retire();
        assert!(order.retired(captured));
        assert!(!order.retired(later.point()));
        later.retire();
    }

    #[test]
    fn cancellation_closes_a_hole_before_a_later_completion() {
        let order = Arc::new(RetirementOrder::default());
        let abandoned = order.reserve().unwrap();
        let later = order.reserve().unwrap().submitted();
        let captured = later.point();

        later.retire();
        assert!(!order.retired(captured));
        drop(abandoned);
        assert!(order.retired(captured));
    }
}
