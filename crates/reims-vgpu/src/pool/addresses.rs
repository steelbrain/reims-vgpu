//! The GPU addresses the host gives the contexts' blocks of the Shared pool on a source whose
//! blocks have none (`HostBlocks`, the Vulkan platform's): `POOL_ADDRESSES`
//! (docs/graphics-kext/design.md, "The Shared pool"). A guest that took the feature places the
//! heap over a block at the address its take's answer names, and the platform pins it there, as
//! it pins every address the guest picks.
//!
//! The addresses lie in `[HOST_BASE, HOST_END)`, `[2^45, 2^46)`: below `RGFX_CONTEXT_VA_BASE`,
//! where the guest's own picks start, and far above where the Vulkan platform's model of Metal's
//! layout places the host's unpinned ranges (lowest first from 2^40, as far as its live ranges
//! reach), so a pin of one meets only another of these. A block's address holds Metal's span
//! of a plain range of its length ([`span`]): its pages, a guard page, rounded up to the 32 KiB
//! granule a pin starts on, as the guest's own picks do.
//!
//! The picks go next fit: each from the end of the last, wrapping to the start at the end. A
//! block's address is free again at its free, but a pick reaches it only once the picks wrap,
//! 32 TiB of spans later, so a platform that lets go of a pin a while after the heap over it
//! went never meets it.

/// The first address the host gives: 2^45.
pub(crate) const HOST_BASE: u64 = 1 << 45;

/// The end of the addresses the host gives: 2^46, `RGFX_CONTEXT_VA_BASE`, where the guest's own
/// picks start.
pub(crate) const HOST_END: u64 = 1 << 46;

/// Metal's page on Apple silicon: a range holds whole pages, and one more as its guard.
const PAGE: u64 = 0x4000;

/// The granule a pin's first page starts on, and the multiple its span is rounded up to.
const GRANULE: u64 = 0x8000;

/// The span of the space a block of `len` bytes holds: its pages and a guard page, rounded up to
/// [`GRANULE`]; `None` past the address space.
pub(crate) fn span(len: u64) -> Option<u64> {
    len.checked_next_multiple_of(PAGE)?
        .checked_add(PAGE)?
        .checked_next_multiple_of(GRANULE)
}

/// The addresses given and not yet freed, and where the next pick starts.
#[derive(Debug)]
pub(crate) struct HostAddresses {
    /// The spans held, by address: their ends.
    held: std::collections::BTreeMap<u64, u64>,
    /// Where the next pick looks first.
    next: u64,
}

impl Default for HostAddresses {
    fn default() -> Self {
        HostAddresses {
            held: std::collections::BTreeMap::new(),
            next: HOST_BASE,
        }
    }
}

impl HostAddresses {
    /// The address of a block of `len` bytes, its span now held: the first start from where the
    /// last pick ended whose span meets no held one, wrapping once to [`HOST_BASE`]; `None`
    /// where no room is left.
    pub(crate) fn pick(&mut self, len: u64) -> Option<u64> {
        let span = span(len)?;
        let at = self
            .fit(self.next, HOST_END, span)
            .or_else(|| self.fit(HOST_BASE, self.next, span))?;
        let end = at.checked_add(span)?;
        self.held.insert(at, end);
        self.next = if end >= HOST_END { HOST_BASE } else { end };
        Some(at)
    }

    /// Frees the span picked at `at`; nothing where none was.
    pub(crate) fn free(&mut self, at: u64) {
        self.held.remove(&at);
    }

    /// The lowest start from `from` whose `span` ends by `until` and meets no held span.
    fn fit(&self, from: u64, until: u64, span: u64) -> Option<u64> {
        let mut at = from;
        // A held span that starts below `from` may reach past it.
        if let Some((_, &end)) = self.held.range(..from).next_back() {
            at = at.max(end);
        }
        for (&start, &end) in self.held.range(from..) {
            if at.checked_add(span)? <= start {
                break;
            }
            at = at.max(end);
        }
        (at.checked_add(span)? <= until).then_some(at)
    }

    /// The spans held (for the tests).
    #[cfg(test)]
    pub(crate) fn held(&self) -> usize {
        self.held.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_block_s_span_holds_its_pages_and_a_guard_page_on_the_granule() {
        assert_eq!(span(0x1_0000), Some(0x1_8000));
        assert_eq!(span(0x10_0000), Some(0x10_8000));
        assert_eq!(span(0x4000), Some(0x8000));
        assert_eq!(span(0x8000), Some(0x1_0000));
        assert_eq!(span(u64::MAX), None);
    }

    #[test]
    fn picks_go_next_fit_from_the_base_on_the_granule() {
        let mut a = HostAddresses::default();
        assert_eq!(a.pick(0x1_0000), Some(HOST_BASE));
        assert_eq!(a.pick(0x2_0000), Some(HOST_BASE + 0x1_8000));
        assert_eq!(a.pick(0x1_0000), Some(HOST_BASE + 0x1_8000 + 0x2_8000));
        assert_eq!(a.held(), 3);
    }

    #[test]
    fn a_freed_address_is_picked_again_only_once_the_picks_wrap() {
        let mut a = HostAddresses::default();
        let first = a.pick(0x1_0000).unwrap();
        let second = a.pick(0x1_0000).unwrap();
        a.free(first);
        // The next pick goes on from the second's end, not back to the freed first.
        assert_eq!(a.pick(0x1_0000), Some(second + 0x1_8000));
        // Up to the end: a block that reaches it, then the picks wrap to the freed first.
        a.next = HOST_END - 0x1_8000;
        assert_eq!(a.pick(0x1_0000), Some(HOST_END - 0x1_8000));
        assert_eq!(a.pick(0x1_0000), Some(first));
        // Past the first, the second is still held: the next pick goes past both.
        assert_eq!(a.pick(0x1_0000), Some(second + 0x3_0000));
    }

    #[test]
    fn a_pick_never_meets_a_held_span_and_none_comes_where_no_room_is_left() {
        let mut a = HostAddresses::default();
        // Every span held but one gap of 0x1_8000 at the base: only a block of 64 KiB fits.
        a.held.insert(HOST_BASE + 0x1_8000, HOST_END);
        assert_eq!(a.pick(0x2_0000), None);
        assert_eq!(a.pick(0x1_0000), Some(HOST_BASE));
        assert_eq!(a.pick(0x1_0000), None);
        a.free(HOST_BASE);
        assert_eq!(a.pick(0x1_0000), Some(HOST_BASE));
        // A pick that would start inside a held span from below starts past it.
        let mut b = HostAddresses::default();
        b.held.insert(HOST_BASE, HOST_BASE + 0x10_0000);
        b.next = HOST_BASE + 0x8000;
        assert_eq!(b.pick(0x1_0000), Some(HOST_BASE + 0x10_0000));
    }
}
