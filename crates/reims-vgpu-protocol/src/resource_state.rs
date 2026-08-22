//! Decoded resource-state operations shared by command producers.

/// Ordered validity changes carried for one resource.
///
/// Each field is an operation byte, not a bit mask. Consumers apply clear
/// before set in field order when both operations are present.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResourceValidityOps {
    pub clear_host_valid: u8,
    pub set_host_valid: u8,
    pub clear_guest_valid: u8,
    pub set_guest_valid: u8,
}

impl ResourceValidityOps {
    /// Decode the wire dword into its four ordered operation bytes.
    pub const fn from_le_dword(flags: u32) -> Self {
        let b = flags.to_le_bytes();
        Self {
            clear_host_valid: b[0],
            set_host_valid: b[1],
            clear_guest_valid: b[2],
            set_guest_valid: b[3],
        }
    }

    pub const fn to_le_dword(self) -> u32 {
        u32::from_le_bytes([
            self.clear_host_valid,
            self.set_host_valid,
            self.clear_guest_valid,
            self.set_guest_valid,
        ])
    }

    /// Page-on transfers authority from host content to guest content.
    pub const PAGE_ON: Self = Self {
        clear_host_valid: 1,
        set_host_valid: 0,
        clear_guest_valid: 0,
        set_guest_valid: 1,
    };
}

#[cfg(test)]
mod tests {
    use super::ResourceValidityOps;

    #[test]
    fn validity_bytes_round_trip_without_becoming_a_mask() {
        let ops = ResourceValidityOps {
            clear_host_valid: 1,
            set_host_valid: 2,
            clear_guest_valid: 3,
            set_guest_valid: 4,
        };
        assert_eq!(ResourceValidityOps::from_le_dword(ops.to_le_dword()), ops);
        assert_eq!(ResourceValidityOps::PAGE_ON.to_le_dword(), 0x0100_0001);
    }
}
