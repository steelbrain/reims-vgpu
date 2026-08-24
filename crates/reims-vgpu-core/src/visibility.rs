//! `MTLVisibilityResultMode`: which occlusion-query modes this device records.
//!
//! # Why the set lives here
//!
//! The decoder and executor translation both consume this Apple enum. One
//! shared set keeps admission and translation aligned.
//!
//! # The ordinals
//!
//! `Disabled = 0`, `Boolean = 1`, `Counting = 2`, from `MTLRenderCommandEncoder.h`.
//! `Disabled` is the ordinal a stream sends to *dis*arm, so it is not in the
//! recordable set: `runtime::decode::render` turns it into the absence of a
//! query rather than into a query with a mode, and a backend handed it anyway
//! is being asked to answer a question nobody posed.

/// Bit `n` set means `MTLVisibilityResultMode(n)` is a mode this device records.
pub const RECORDABLE_VISIBILITY_RESULT_MODES: u32 = 0b110;

/// Whether `mtl` is a visibility result mode this device records.
#[inline]
pub const fn visibility_result_mode_recordable(mtl: u32) -> bool {
    mtl < u32::BITS && (RECORDABLE_VISIBILITY_RESULT_MODES >> mtl) & 1 == 1
}

/// The ordinal that disarms an occlusion query, `MTLVisibilityResultModeDisabled`.
///
/// Named rather than written as a bare `0`, because `0` is also the first
/// *recordable* ordinal of four other Metal enums this crate decodes and the
/// reader cannot tell which one a literal means.
///
/// The Vulkan translator matches on it directly.
pub const VISIBILITY_RESULT_MODE_DISABLED: u32 = 0;

/// One past the highest ordinal the enum declares, for a sweep that wants to
/// prove a table refuses everything outside the set rather than sampling it.
///
/// Four past rather than one so a table that accepted an adjacent undeclared
/// ordinal would fail the sweep.
pub const VISIBILITY_RESULT_MODE_SWEEP_END: u32 = 7;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_disabling_ordinal_is_not_recordable() {
        assert!(!visibility_result_mode_recordable(
            VISIBILITY_RESULT_MODE_DISABLED
        ));
    }

    #[test]
    fn exactly_boolean_and_counting_are_recordable() {
        let recordable: Vec<u32> = (0..VISIBILITY_RESULT_MODE_SWEEP_END)
            .filter(|m| visibility_result_mode_recordable(*m))
            .collect();
        assert_eq!(recordable, vec![1, 2]);
    }

    #[test]
    fn no_ordinal_past_the_enum_is_recordable() {
        for mtl in 3..=u32::BITS + 4 {
            assert!(
                !visibility_result_mode_recordable(mtl),
                "ordinal {mtl} is not a declared MTLVisibilityResultMode"
            );
        }
    }
}
