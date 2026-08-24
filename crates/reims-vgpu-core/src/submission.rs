//! Backend-independent submission envelopes retained by device state.

use reims_vgpu_protocol::{
    SegmentBoundary, SubmissionId, SubmissionIdentity, SubmissionResourceUse, TaskId,
};
use std::sync::Arc;

/// Protocol context shared by every operation in one submitted command stream.
///
/// Each value is an immutable snapshot. Executors may retain it without
/// observing later movement of the device-owned submission cursor or mutation
/// of the decoder and its resource-list accumulator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubmissionContext {
    pub identity: SubmissionIdentity,
    pub resources: Arc<[SubmissionResourceUse]>,
    /// Every admitted segment in command-buffer order.
    pub segments: Arc<[SegmentBoundary]>,
    /// Segment containing the operation currently submitted to the executor.
    pub segment: Option<SegmentBoundary>,
}

impl SubmissionContext {
    /// Context for direct test and tool operations outside a decoded EXEC packet.
    pub fn standalone(task_id: u32) -> Self {
        Self {
            identity: SubmissionIdentity {
                id: SubmissionId::new(0),
                task: TaskId::new(task_id),
            },
            resources: Arc::from([]),
            segments: Arc::from([]),
            segment: None,
        }
    }
}

/// Device-local ownership of the currently decoded submission envelope.
///
/// Callers can obtain immutable [`SubmissionContext`] snapshots, but cannot
/// mutate participation or segment position independently of submission
/// identity. Reset drops this owner and therefore its active envelope.
#[derive(Debug)]
pub struct SubmissionTracker {
    next_id: u64,
    active: Option<SubmissionContext>,
}

impl Default for SubmissionTracker {
    fn default() -> Self {
        Self {
            next_id: 1,
            active: None,
        }
    }
}

impl SubmissionTracker {
    /// Mint the next nonzero identity for `task`.
    pub fn next_identity(&mut self, task: TaskId) -> SubmissionIdentity {
        let identity = SubmissionIdentity {
            id: SubmissionId::new(self.next_id),
            task,
        };
        self.next_id = self.next_id.wrapping_add(1).max(1);
        identity
    }

    /// Install one complete participation envelope before its first segment.
    pub fn begin(
        &mut self,
        identity: SubmissionIdentity,
        resources: Arc<[SubmissionResourceUse]>,
        segments: Arc<[SegmentBoundary]>,
    ) {
        assert!(
            self.active.is_none(),
            "a submission cannot begin while another remains active"
        );
        self.active = Some(SubmissionContext {
            identity,
            resources,
            segments,
            segment: None,
        });
    }

    /// Select the active submission segment, when this operation belongs to an
    /// EXEC envelope. Direct tools and focused walkers intentionally have no
    /// active submission and continue to use standalone executor context.
    pub fn enter_segment_if_active(&mut self, segment: Option<SegmentBoundary>) {
        if let Some(active) = self.active.as_mut() {
            active.segment = segment;
        }
    }

    /// Immutable executor snapshot, or a standalone context for direct tools.
    pub fn context_or_standalone(&self, task_id: u32) -> SubmissionContext {
        self.active
            .clone()
            .unwrap_or_else(|| SubmissionContext::standalone(task_id))
    }

    /// Consume the active envelope at its single completion boundary.
    pub fn finish(&mut self) -> Option<SubmissionContext> {
        self.active.take()
    }
}

#[cfg(test)]
mod tests {
    use super::{SubmissionContext, SubmissionTracker};
    use reims_vgpu_protocol::{SegmentBoundary, SegmentKind, TaskId};

    #[test]
    fn standalone_context_has_no_invented_participation() {
        let context = SubmissionContext::standalone(7);
        assert_eq!(context.identity.task.get(), 7);
        assert_eq!(context.identity.id.get(), 0);
        assert!(context.resources.is_empty());
        assert!(context.segments.is_empty());
        assert_eq!(context.segment, None);
    }

    #[test]
    fn tracker_owns_identity_envelope_segment_and_completion_together() {
        let mut tracker = SubmissionTracker::default();
        let first = tracker.next_identity(TaskId::new(7));
        let second = tracker.next_identity(TaskId::new(7));
        assert_ne!(first.id, second.id);
        assert_ne!(first.id.get(), 0);

        let segment = SegmentBoundary {
            stream_index: 2,
            index: 3,
            kind: SegmentKind::Render,
            continues_previous: false,
            continues_next: true,
        };
        tracker.begin(
            first,
            std::sync::Arc::from([]),
            std::sync::Arc::from([segment]),
        );
        tracker.enter_segment_if_active(Some(segment));
        let snapshot = tracker.context_or_standalone(99);
        assert_eq!(snapshot.identity, first);
        assert_eq!(snapshot.segment, Some(segment));
        assert_eq!(snapshot.segments.as_ref(), &[segment]);

        let finished = tracker
            .finish()
            .expect("the active envelope completes once");
        assert_eq!(finished.identity, first);
        assert!(tracker.finish().is_none());
        assert_eq!(tracker.context_or_standalone(99).identity.id.get(), 0);
    }
}
