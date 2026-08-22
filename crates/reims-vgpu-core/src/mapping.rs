//! Semantic mapping content currency and validity transitions.

use reims_vgpu_protocol::ResourceValidityOps;

/// Who owns a resource's authoritative bytes, as the guest last stated it and
/// as the device last produced them.
///
/// The booleans retain the four decoded validity statements. The sequence
/// numbers carry the behavior-selecting happens-before relation: a guest
/// `clear_host_valid` is a claim made at one moment, not a permanent veto over
/// every later device publication.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResourceValidity {
    pub host_valid: bool,
    pub guest_valid: bool,
    pub host_stated: bool,
    pub guest_stated: bool,
    /// Sequence at the guest's last `clear_host_valid`; zero means unstated.
    pub host_cleared_seq: u64,
    /// Sequence at the device's last publication of newer pixels.
    pub host_published_seq: u64,
}

/// Content currency and ownership statements for one mapping.
///
/// Mapping identity, page-table incarnation, and host materialization are not
/// members. A topology policy may choose where replicas live, but it cannot
/// give these currencies a second meaning or update only half of a guest-write
/// transition.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MappingContentState {
    /// Writes which reached the mapping's guest pages.
    pub guest_page_generation: u32,
    /// Changes to semantic surface content, wherever the newest bytes live.
    pub surface_epoch: u32,
    /// Decoded validity statements and publication ordering.
    pub validity: ResourceValidity,
}

impl MappingContentState {
    /// Record newer guest-owned bytes and return their page generation.
    pub fn guest_wrote(&mut self, sequence: u64) -> u32 {
        self.guest_page_generation = next_nonzero_u32(self.guest_page_generation);
        self.surface_epoch = next_nonzero_u32(self.surface_epoch);
        self.validity.host_cleared_seq = sequence;
        self.guest_page_generation
    }

    /// Record a device publication which changed semantic surface content but
    /// did not necessarily materialize it into guest pages.
    pub fn host_published(&mut self, sequence: u64) -> u32 {
        self.surface_epoch = next_nonzero_u32(self.surface_epoch);
        self.validity.host_published_seq = sequence;
        self.surface_epoch
    }

    /// Record a device write which also reached guest pages.
    pub fn host_wrote_guest_pages(&mut self, sequence: u64) -> u32 {
        self.guest_page_generation = next_nonzero_u32(self.guest_page_generation);
        self.host_published(sequence);
        self.guest_page_generation
    }

    /// Apply ordered validity bytes without changing content currency.
    pub fn apply_validity(&mut self, ops: ResourceValidityOps) {
        if ops.clear_host_valid != 0 {
            self.validity.host_valid = false;
            self.validity.host_stated = true;
        }
        if ops.set_host_valid != 0 {
            self.validity.host_valid = true;
            self.validity.host_stated = true;
        }
        if ops.clear_guest_valid != 0 {
            self.validity.guest_valid = false;
            self.validity.guest_stated = true;
        }
        if ops.set_guest_valid != 0 {
            self.validity.guest_valid = true;
            self.validity.guest_stated = true;
        }
    }
}

fn next_nonzero_u32(value: u32) -> u32 {
    match value.wrapping_add(1) {
        0 => 1,
        next => next,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_guest_write_advances_both_currencies_and_records_its_order() {
        let mut content = MappingContentState {
            guest_page_generation: u32::MAX,
            surface_epoch: u32::MAX,
            ..Default::default()
        };
        assert_eq!(content.guest_wrote(17), 1);
        assert_eq!(content.surface_epoch, 1);
        assert_eq!(content.validity.host_cleared_seq, 17);
    }

    #[test]
    fn a_host_only_publication_does_not_claim_new_guest_page_bytes() {
        let mut content = MappingContentState {
            guest_page_generation: 9,
            surface_epoch: 4,
            ..Default::default()
        };
        assert_eq!(content.host_published(23), 5);
        assert_eq!(content.guest_page_generation, 9);
        assert_eq!(content.validity.host_published_seq, 23);
    }

    #[test]
    fn validity_operations_apply_clear_before_set_without_moving_currency() {
        let mut content = MappingContentState {
            guest_page_generation: 7,
            surface_epoch: 11,
            ..Default::default()
        };
        content.apply_validity(ResourceValidityOps {
            clear_host_valid: 1,
            set_host_valid: 1,
            clear_guest_valid: 0,
            set_guest_valid: 0,
        });
        assert!(content.validity.host_valid);
        assert!(content.validity.host_stated);
        assert!(!content.validity.guest_stated);
        assert_eq!(content.guest_page_generation, 7);
        assert_eq!(content.surface_epoch, 11);
    }
}
