//! The GPU addresses the host gives the contexts' blocks on a source whose blocks have none
//! (`HostBlocks`, the Vulkan platform's): the Shared pool's (`POOL_ADDRESSES`,
//! docs/graphics-kext/design.md, "The Shared pool"), the Private blocks
//! (`PRIVATE_BLOCKS`, "Private blocks") and the pages blocks' views (`IMPORT_ADDRESSES`,
//! "Import addresses"). A guest that took the feature places its objects at the address its
//! take's answer names, and the platform pins them there, as it pins every address the guest
//! picks.
//!
//! The pool's addresses lie in `[HOST_BASE, HOST_END)`, `[2^45, 2^46)`, the Private blocks' in
//! `[PRIVATE_BASE, PRIVATE_END)`, `[2^44, 2^45)`, and the pages blocks' views' in
//! `[IMPORT_BASE, IMPORT_END)`, `[2^43, 2^44)`, each picked by its own [`HostAddresses`]:
//! below `RGFX_CONTEXT_VA_BASE`, where the guest's own picks start, and far above where the
//! Vulkan platform's model of Metal's layout places the host's unpinned ranges (lowest first from
//! 2^40, as far as its live ranges reach), so a pin of one meets only another of these. A
//! block's address holds Metal's span of a plain range of its length ([`span`]): its pages, a
//! guard page, rounded up to the 32 KiB granule a pin starts on, as the guest's own picks do.
//!
//! The picks go next fit: each from the end of the last, wrapping to the start at the end. A
//! block's address is free again at its free, but a pick reaches it only once the picks wrap,
//! 32 TiB of spans later in the pool's range, 16 TiB in the Private one and 8 TiB in the pages
//! blocks' views', so a platform that lets go of a pin a while after the object over it went
//! never meets it.

/// The first address the host gives: 2^45.
pub(crate) const HOST_BASE: u64 = 1 << 45;

/// The end of the addresses the host gives: 2^46, `RGFX_CONTEXT_VA_BASE`, where the guest's own
/// picks start.
pub(crate) const HOST_END: u64 = 1 << 46;

/// The first address the host gives a Private block: 2^44.
pub(crate) const PRIVATE_BASE: u64 = 1 << 44;

/// The end of the addresses the host gives the Private blocks: 2^45, [`HOST_BASE`], where the
/// pool's start.
pub(crate) const PRIVATE_END: u64 = HOST_BASE;

/// The first address the host gives a pages block's view: 2^43.
pub(crate) const IMPORT_BASE: u64 = 1 << 43;

/// The end of the addresses the host gives the pages blocks' views: 2^44, [`PRIVATE_BASE`], where
/// the Private blocks' start.
pub(crate) const IMPORT_END: u64 = PRIVATE_BASE;

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

/// The addresses of one range given and not yet freed, and where the next pick starts.
#[derive(Debug)]
pub(crate) struct HostAddresses {
    /// The spans held, by address: their ends.
    held: std::collections::BTreeMap<u64, u64>,
    /// Where the next pick looks first.
    next: u64,
    /// The range's first address, a multiple of [`GRANULE`].
    base: u64,
    /// The range's end.
    end: u64,
}

impl Default for HostAddresses {
    /// The pool's: `[HOST_BASE, HOST_END)`.
    fn default() -> Self {
        HostAddresses::over(HOST_BASE, HOST_END)
    }
}

impl HostAddresses {
    /// The addresses `[base, end)`, none held; `base` a multiple of [`GRANULE`].
    pub(crate) fn over(base: u64, end: u64) -> HostAddresses {
        debug_assert!(base.is_multiple_of(GRANULE) && base < end);
        HostAddresses {
            held: std::collections::BTreeMap::new(),
            next: base,
            base,
            end,
        }
    }

    /// The Private blocks': `[PRIVATE_BASE, PRIVATE_END)`.
    pub(crate) fn private() -> HostAddresses {
        HostAddresses::over(PRIVATE_BASE, PRIVATE_END)
    }

    /// The pages blocks' views': `[IMPORT_BASE, IMPORT_END)`.
    pub(crate) fn imports() -> HostAddresses {
        HostAddresses::over(IMPORT_BASE, IMPORT_END)
    }

    /// The address of a block of `len` bytes, its span now held: the first start from where the
    /// last pick ended whose span meets no held one, wrapping once to the range's base; `None`
    /// where no room is left.
    pub(crate) fn pick(&mut self, len: u64) -> Option<u64> {
        let span = span(len)?;
        let at = self
            .fit(self.next, self.end, span)
            .or_else(|| self.fit(self.base, self.next, span))?;
        let end = at.checked_add(span)?;
        self.held.insert(at, end);
        self.next = if end >= self.end { self.base } else { end };
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

    /// The Private blocks' range is `[2^44, 2^45)`, below the pool's, and its picks stay in it:
    /// from its base, and wrapping to its base at its end, never into the pool's.
    #[test]
    fn the_private_range_lies_below_the_pool_s_and_its_picks_stay_in_it() {
        assert_eq!((PRIVATE_BASE, PRIVATE_END), (1 << 44, 1 << 45));
        let mut a = HostAddresses::private();
        assert_eq!(a.pick(0x10_0000), Some(PRIVATE_BASE));
        assert_eq!(a.pick(0x1_0000), Some(PRIVATE_BASE + 0x10_8000));
        a.free(PRIVATE_BASE);
        a.next = PRIVATE_END - 0x1_8000;
        assert_eq!(a.pick(0x1_0000), Some(PRIVATE_END - 0x1_8000));
        // No room for a second one before the end: it wraps to the freed base.
        assert_eq!(a.pick(0x1_0000), Some(PRIVATE_BASE));
        let mut full = HostAddresses::private();
        full.held.insert(PRIVATE_BASE, PRIVATE_END);
        assert_eq!(full.pick(0x1_0000), None);
    }

    /// The pages blocks' views' range is `[2^43, 2^44)`, below the Private blocks', and its picks
    /// stay in it.
    #[test]
    fn the_views_range_lies_below_the_private_one_and_its_picks_stay_in_it() {
        assert_eq!((IMPORT_BASE, IMPORT_END), (1 << 43, 1 << 44));
        let mut a = HostAddresses::imports();
        assert_eq!(a.pick(0x3000), Some(IMPORT_BASE));
        assert_eq!(a.pick(0x1_0000), Some(IMPORT_BASE + 0x8000));
        a.free(IMPORT_BASE);
        a.next = IMPORT_END - 0x1_8000;
        assert_eq!(a.pick(0x1_0000), Some(IMPORT_END - 0x1_8000));
        assert_eq!(a.pick(0x3000), Some(IMPORT_BASE));
        let mut full = HostAddresses::imports();
        full.held.insert(IMPORT_BASE, IMPORT_END);
        assert_eq!(full.pick(0x1000), None);
    }
}
