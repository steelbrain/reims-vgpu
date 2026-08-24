//! Completion-stamp values after packet-record decoding.

pub const STAMP_INDEX_MASK: u32 = 0xffff;
pub const STAMP_SLOT_LEN: u64 = core::mem::size_of::<u32>() as u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StampWait {
    pub index: u32,
    pub value: u32,
}

impl StampWait {
    pub const fn slot(self) -> u32 {
        self.index & STAMP_INDEX_MASK
    }

    pub fn satisfied_by(self, current: u32) -> bool {
        current.wrapping_sub(self.value) as i32 >= 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_index_bits_are_consumed_at_the_semantic_boundary() {
        assert_eq!(
            StampWait {
                index: 0xabcd_1234,
                value: 7
            }
            .slot(),
            0x1234
        );
    }

    #[test]
    fn satisfaction_survives_counter_wrap() {
        assert!(StampWait {
            index: 0,
            value: u32::MAX
        }
        .satisfied_by(0));
        assert!(!StampWait { index: 0, value: 1 }.satisfied_by(0));
    }
}
