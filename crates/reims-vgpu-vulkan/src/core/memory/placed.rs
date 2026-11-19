//! What of a block its holder ever handed out: [`Placed`], the one rule by which the device skips
//! zeroing bytes the block's source zeroed.
//!
//! A block's source may say its bytes are zeros as it hands it out (metal2vulkan-rt's
//! `BlockInfo::zeroed`: the driver zeroed the allocation, and no block of it held the range
//! before). The holder then zeroes only what it handed out since: each placement of bytes in the
//! block (a carve of the guest's pool, a buffer or heap pinned in a known Private block) raises
//! the block's mark past them, and the mark never falls, so every byte at or past it is still the
//! source's zero. A placement's bytes below the mark may hold what an earlier one wrote, and are
//! zeroed as before; those past it are not. A block whose source does not say so is zeroed
//! whole, as before.

/// A block's mark of the bytes ever handed out of it, from its take: every byte at or past it is
/// as the block's source handed it out. 0 for a block just taken.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Placed {
    to: u64,
}

impl Placed {
    /// Hands out the `length` bytes at `offset` of the block, whose source said its bytes were
    /// zeros where `zeroed`: raises the mark past them, and answers how many of them, from the
    /// first, may hold what was written before: all of them where the block is not `zeroed`, else
    /// those below the mark as it stood.
    pub fn place(&mut self, zeroed: bool, offset: u64, length: u64) -> u64 {
        let written = match zeroed {
            true => self.to.saturating_sub(offset).min(length),
            false => length,
        };
        self.to = self.to.max(offset.saturating_add(length));
        written
    }

    /// The mark: the end of the bytes ever handed out.
    pub fn to(self) -> u64 {
        self.to
    }
}

#[cfg(test)]
mod tests {
    use super::Placed;

    /// A zeroed block's placement is written only below the mark, which rises past it and never
    /// falls; a block not zeroed is written whole at every placement
    #[test]
    fn a_zeroed_block_is_written_only_below_its_mark() {
        let mut fresh = Placed::default();
        assert_eq!(fresh.place(true, 0, 0x1_0000), 0);
        assert_eq!(fresh.place(true, 0x2_0000, 0x1_0000), 0);
        assert_eq!(fresh.to(), 0x3_0000);
        // Below the mark, whole: the first placement's bytes again.
        assert_eq!(fresh.place(true, 0, 0x1_0000), 0x1_0000);
        // Partly below: its first bytes, up to the mark, the rest past it.
        assert_eq!(fresh.place(true, 0x2_8000, 0x1_0000), 0x8000);
        assert_eq!(fresh.to(), 0x3_8000);
        // In the gap below the mark that nothing was placed in: written, the mark not knowing.
        assert_eq!(fresh.place(true, 0x1_0000, 0x1_0000), 0x1_0000);
        assert_eq!(fresh.to(), 0x3_8000);

        let mut not = Placed::default();
        assert_eq!(not.place(false, 0, 0x1_0000), 0x1_0000);
        assert_eq!(not.place(false, 0x4_0000, 0x1_0000), 0x1_0000);
        assert_eq!(not.to(), 0x5_0000);
    }
}
