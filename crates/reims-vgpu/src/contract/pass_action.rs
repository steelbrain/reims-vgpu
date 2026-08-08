//! The `MTLLoadAction` and `MTLStoreAction` ordinals a render-pass attachment
//! prefix carries, and the closed sets this device implements.
//!
//! Two adjacent 16-bit words of every colour, depth and stencil attachment. The
//! guest writes the Metal SDK's own ordinals into them, so the numbers here are
//! simultaneously a wire fact and an SDK fact — which is why the conversion on
//! the Metal encode path is an identity: nothing is mapped, only widened.
//!
//! # Why they live here rather than in the decoder or the backend
//!
//! They were declared twice. `runtime::decode::render` had them as `u16` under
//! `PASS_*`, `backend::metal::abi` as `u32` under `REIMS_VGPU_MTL_*`, five
//! ordinals each, and nothing in the toolchain compared the two — the only
//! thing that touched both was an identity `match` in [`crate::runtime::draw`]
//! that read as a translation table. A value that arrives on the wire and is
//! consumed by both backends belongs beside the other wire/SDK numbers, per this
//! module tree's own doc; `backend::metal::abi` keeps its spelling because that
//! file is a mirror of an archived C header and the mirror is its provenance,
//! with `const` assertions there pinning the two equal on every arm that
//! compiles it.
//!
//! # The widths differ, and that is the whole conversion
//!
//! The attachment prefix spells an action in 16 bits; the Metal C shim takes an
//! `MTLLoadAction`/`MTLStoreAction` as `uint32_t`. So the declaration that
//! crosses to C is `u32` and this one is `u16`, and everything between them is a
//! widening — see [`crate::runtime::draw`]'s `map_load_action`, which returns
//! DontCare for a value outside the set and widens every value inside it.

/// `MTLLoadActionDontCare` — the attachment's prior contents may be discarded.
pub const MTL_LOAD_ACTION_DONT_CARE: u16 = 0;
/// `MTLLoadActionLoad` — the pass composites onto the attachment's contents.
pub const MTL_LOAD_ACTION_LOAD: u16 = 1;
/// `MTLLoadActionClear` — the attachment starts at the record's clear value.
pub const MTL_LOAD_ACTION_CLEAR: u16 = 2;

/// `MTLStoreActionDontCare` — the pass's result for this attachment is dropped.
pub const MTL_STORE_ACTION_DONT_CARE: u16 = 0;
/// `MTLStoreActionStore` — the pass's result is written back to the attachment.
pub const MTL_STORE_ACTION_STORE: u16 = 1;

/// Whether `raw` is one of the three `MTLLoadAction` values.
///
/// The set is *closed* in the same sense as [`crate::contract::dispatch`]'s:
/// `MTLLoadAction` has exactly these three, so a fourth ordinal is a corrupt
/// record or a wrong wire offset rather than a guest feature this device has no
/// contract for yet.
#[must_use]
pub fn is_declared_load_action(raw: u16) -> bool {
    raw <= MTL_LOAD_ACTION_CLEAR
}

/// Whether `raw` is one of the two `MTLStoreAction` values this device implements.
///
/// Unlike the load set this one is *not* closed by the SDK — `MTLStoreAction`
/// also has `MultisampleResolve`, `StoreAndMultisampleResolve`, `Unknown` and
/// `CustomSampleDepthStore`. So a rejection here has two possible causes that
/// this predicate cannot tell apart: a misread field, or a guest asking for a
/// resolve this device does not implement. Either way the pass's result for that
/// attachment is dropped, which is why the callers that report say so rather
/// than naming a cause.
#[must_use]
pub fn is_declared_store_action(raw: u16) -> bool {
    raw <= MTL_STORE_ACTION_STORE
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The accepted load set is exactly the three declared ordinals.
    ///
    /// Swept past the top constant in both directions, because the predicate's
    /// job is to be closed: every value it rejects is substituted with DontCare
    /// by its callers, so an accidentally-accepted fourth ordinal would reach a
    /// Metal enum conversion that has no arm for it.
    #[test]
    fn only_the_three_declared_load_actions_are_accepted() {
        assert!(is_declared_load_action(MTL_LOAD_ACTION_DONT_CARE));
        assert!(is_declared_load_action(MTL_LOAD_ACTION_LOAD));
        assert!(is_declared_load_action(MTL_LOAD_ACTION_CLEAR));
        for raw in (MTL_LOAD_ACTION_CLEAR + 1)..=64u16 {
            assert!(
                !is_declared_load_action(raw),
                "{raw} is not a declared MTLLoadAction"
            );
        }
        assert!(!is_declared_load_action(u16::MAX));
    }

    /// The accepted store set is exactly the two this device implements.
    #[test]
    fn only_the_two_implemented_store_actions_are_accepted() {
        assert!(is_declared_store_action(MTL_STORE_ACTION_DONT_CARE));
        assert!(is_declared_store_action(MTL_STORE_ACTION_STORE));
        for raw in (MTL_STORE_ACTION_STORE + 1)..=64u16 {
            assert!(
                !is_declared_store_action(raw),
                "{raw} is not an implemented MTLStoreAction"
            );
        }
        assert!(!is_declared_store_action(u16::MAX));
    }

    /// Neither predicate can see the two adjacent words swapped.
    ///
    /// The store set is a strict subset of the load set — `{0, 1}` inside
    /// `{0, 1, 2}` — and the attachment prefix carries the two in adjacent
    /// words. So a decode that reads the store word as the load word produces a
    /// value both predicates accept, every time, and the only reading that can
    /// escape is a load of CLEAR landing where a store was expected. Pinned
    /// because it bounds what these two can be asked to prove: they narrow a
    /// value to its own contract, and no arrangement of them detects a field
    /// offset that is two bytes out.
    #[test]
    fn the_store_set_is_a_strict_subset_of_the_load_set() {
        let load: Vec<u16> = (0..=u16::MAX)
            .filter(|&r| is_declared_load_action(r))
            .collect();
        let store: Vec<u16> = (0..=u16::MAX)
            .filter(|&r| is_declared_store_action(r))
            .collect();
        assert!(
            store.iter().all(|r| load.contains(r)),
            "a store ordinal outside the load set would make a swap detectable"
        );
        assert_eq!(
            load.iter()
                .filter(|r| !store.contains(r))
                .collect::<Vec<_>>(),
            vec![&MTL_LOAD_ACTION_CLEAR]
        );
    }
}
