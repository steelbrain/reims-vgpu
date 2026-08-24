//! Semantic metadata carried by every backend submission.

use crate::{ContentVersion, ObjectTableRef, ResourceId, SubmissionId, TaskId};

/// Marker for a heterogeneous task resource-list reference.
pub enum ResourceObject {}

/// Marker for the serializer's heap-specific reference namespace.
///
/// Heap refs are resolved before heap-placed resources are constructed and do
/// not name slots in the task's heterogeneous object list. Keeping a separate
/// marker prevents an equal integer in those two namespaces from becoming an
/// accidental resource relation.
pub enum HeapObject {}

/// Marker for the indirect-command-buffer allocator's reference namespace.
///
/// These references are created and destroyed independently of task resource
/// list entries. An equal integer in the two namespaces does not identify the
/// same object.
#[derive(Debug)]
pub enum IndirectCommandBufferObject {}

/// The four validity transitions carried beside one submitted resource.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResourceValidity {
    pub clear_host: bool,
    pub set_host: bool,
    pub clear_guest: bool,
    pub set_guest: bool,
}

/// One resource participating in a guest submission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubmissionResourceUse {
    pub object: ObjectTableRef<ResourceObject>,
    /// Canonical identity when the object has already been constructed.
    /// Resource tables may also name declared residency entries which no
    /// command has resolved yet; those deliberately remain unresolved.
    pub resource: Option<ResourceId<ResourceObject>>,
    /// Content version observed after applying this record's pre-submission
    /// validity transition.
    pub expected_content: Option<ContentVersion>,
    pub validity: ResourceValidity,
}

/// Semantic encoder family selected by a segment header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentKind {
    Render,
    Compute,
    Blit,
    Event,
    Info,
}

/// The segment containing a backend operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegmentBoundary {
    /// Child command-buffer position within the submitted command-buffer list.
    pub stream_index: u32,
    /// Segment position within that child command buffer.
    pub index: u32,
    pub kind: SegmentKind,
    pub continues_previous: bool,
    pub continues_next: bool,
}

/// Stable identity shared by all operations decoded from one guest submission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubmissionIdentity {
    pub id: SubmissionId,
    pub task: TaskId,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resource_validity_is_not_a_wire_dword() {
        let use_ = SubmissionResourceUse {
            object: ObjectTableRef::new(7),
            resource: None,
            expected_content: None,
            validity: ResourceValidity {
                clear_host: true,
                set_guest: true,
                ..ResourceValidity::default()
            },
        };
        assert_eq!(use_.object.get(), 7);
        assert!(use_.validity.clear_host);
        assert!(use_.validity.set_guest);
    }
}
