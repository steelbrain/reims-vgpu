//! Device-scoped observation state, separate from lifecycle and content authority.

use std::collections::{BTreeMap, BTreeSet};

use crate::{MapIntervals, NodeWatch, ReleasedPages};

/// Instruments that observe contract transitions without selecting behavior.
#[derive(Debug, Default)]
pub struct DeviceObservations {
    pub map_audit: BTreeMap<u32, MapIntervals>,
    pub node_guard: BTreeMap<u32, NodeWatch>,
    pub released_pages: ReleasedPages,
    pub view_stale_reads: u64,
    /// Highest guest task identity observed. This measures namespace reach; it
    /// is not a capacity and must never become an admission input.
    max_task_id_seen: u32,
    /// Highest guest surface-mapping identity observed, under the same rule.
    max_mapping_id_seen: u32,
    /// Number of decoded map-family operations, for cadence observation only.
    map_family_events: u64,
    /// Overlong display-transaction shapes already emitted to the diagnostic
    /// channel. This deduplicates output only; every occurrence remains
    /// separately counted by the runtime instrument.
    display_txn_shapes: BTreeSet<(u16, usize)>,
}

impl DeviceObservations {
    pub fn observe_task_id(&mut self, task_id: u32) {
        self.max_task_id_seen = self.max_task_id_seen.max(task_id);
    }

    pub fn observe_mapping_id(&mut self, mapping_id: u32) {
        self.max_mapping_id_seen = self.max_mapping_id_seen.max(mapping_id);
    }

    pub fn max_task_id_seen(&self) -> u32 {
        self.max_task_id_seen
    }

    pub fn max_mapping_id_seen(&self) -> u32 {
        self.max_mapping_id_seen
    }

    pub fn note_map_family_event(&mut self) -> u64 {
        self.map_family_events = self.map_family_events.saturating_add(1);
        self.map_family_events
    }

    /// Return true only for the first sighting of an overlong wire shape.
    pub fn first_display_txn_shape(&mut self, opcode: u16, payload_len: usize) -> bool {
        self.display_txn_shapes.insert((opcode, payload_len))
    }

    #[cfg(feature = "test-fixtures")]
    pub fn display_txn_shape_count(&self) -> usize {
        self.display_txn_shapes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::DeviceObservations;

    #[test]
    fn namespace_reach_and_map_cadence_are_observation_owned() {
        let mut observations = DeviceObservations::default();
        observations.observe_task_id(12);
        observations.observe_task_id(4);
        observations.observe_mapping_id(31);
        assert_eq!(observations.max_task_id_seen(), 12);
        assert_eq!(observations.max_mapping_id_seen(), 31);
        assert_eq!(observations.note_map_family_event(), 1);
        assert_eq!(observations.note_map_family_event(), 2);
        assert!(observations.first_display_txn_shape(6, 64));
        assert!(!observations.first_display_txn_shape(6, 64));
        assert!(observations.first_display_txn_shape(7, 64));
    }
}
